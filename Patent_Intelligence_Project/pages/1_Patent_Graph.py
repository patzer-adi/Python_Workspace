"""Patent Graph: a map of every patent in the PatentSBERTa index, plus query-driven investigations."""

from __future__ import annotations

import hashlib
import html
from pathlib import Path

import numpy as np
import streamlit as st

st.set_page_config(page_title="Patent Graph · Patent Intelligence", page_icon="🔎", layout="wide")

from src import investigations
from src.theme import apply_theme, render_nav, stretch

apply_theme()
render_nav("graph")

try:
    from src.graph_view import bar_chart, explore_figure, investigate_figure, picked_patent, similarity_heatmap
    from src.patent_graph import SECTION_NAMES, SimilarityIndex, cpc_section, cpc_subclass, pct, result_links
    from src.patent_map import build_map
    from src.patent_sbert_embeddings import (
        CHUNK_COLLECTION,
        DEFAULT_BATCH_SIZE,
        MODEL_NAME,
        PATENT_COLLECTION,
        GraphIndex,
        IndexMismatchError,
        MissingPatentTextError,
        ModelLoadError,
        _open_collection,
        build_index,
        get_graph_db_path,
        load_patent_sbert,
        load_records,
        normalize_id,
    )
except ImportError as exc:
    st.title("Patent Graph")
    st.error(f"A required package is missing: {exc}. Run `pip install plotly scikit-learn` in the project environment.")
    st.stop()

DB = str(get_graph_db_path())
ss = st.session_state
for key, default in dict(pg_qvec=None, pg_qtext="", pg_qsource=None, pg_center=None, pg_selected=None, pg_ver=0).items():
    ss.setdefault(key, default)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading the graph index…")
def load_graph(db: str, stamp: float):
    index = GraphIndex(db)
    try:
        return SimilarityIndex.from_index(index), index.all_metadata()
    finally:
        index.close()


@st.cache_resource(show_spinner="Computing the patent map (domains, layout, links). Cached after the first run…")
def load_map(db: str, stamp: float, n_clusters: int):
    sim_, records_ = load_graph(db, stamp)
    return build_map(sim_, records_, cache_dir=Path(db).parent, n_clusters=n_clusters or None)


@st.cache_resource(show_spinner="Loading PatentSBERTa on CPU (the first run downloads the model)…")
def get_patent_sbert():
    return load_patent_sbert()


@st.cache_data(ttl=600, show_spinner=False)
def source_counts() -> dict:
    out = {}
    for name in (PATENT_COLLECTION, CHUNK_COLLECTION):
        try:
            out[name] = int(_open_collection(name).count())
        except Exception:
            out[name] = None
    return out


def index_status() -> dict:
    index = GraphIndex(DB)
    try:
        return index.status()
    finally:
        index.close()


def _act(kind: str, value=None) -> None:
    """Callbacks only record what the user asked for; the script body applies it once the index is loaded."""
    ss["pg_action"] = (kind, value)


# --------------------------------------------------------------------------- #
# Header + index panel
# --------------------------------------------------------------------------- #

status = index_status()
src = source_counts()
st.title("Patent Graph")
coverage = f"<b>{status['count']:,}</b> patents embedded with {MODEL_NAME} in this graph index"
if src.get(PATENT_COLLECTION):
    coverage += f" · {src[PATENT_COLLECTION]:,} patents in the `{PATENT_COLLECTION}` collection"
st.markdown(f"<p class='pi-sub'>Coverage: {coverage}. Only embedded patents appear on the map.</p>", unsafe_allow_html=True)
if 0 < status["count"] < 200:
    st.warning(f"This is a small sample index ({status['count']} patents). Domains and the map are not representative of the "
               "full collection until it is built.")
elif src.get(PATENT_COLLECTION) and status["count"] < src[PATENT_COLLECTION]:
    st.info(f"{src[PATENT_COLLECTION] - status['count']:,} patents in `{PATENT_COLLECTION}` are not embedded yet and are missing from the graph.")

with st.expander("Graph index", expanded=status["count"] == 0):
    if ss.get("pg_build_msg"):
        st.success(ss.pop("pg_build_msg"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Patents indexed", f"{status['count']:,}")
    c2.metric("Model", (status["model"] or MODEL_NAME).split("/")[-1])
    c3.metric("Vector size", status["dim"] or "—")
    st.caption(f"Stored separately from ChromaDB at `{status['path']}`. Last updated: {status['updated'] or 'never'}.")
    source = st.selectbox(
        "Record source", ["auto", "patent_rag", "chunks"],
        help="auto / patent_rag: patent-level text from `patent_rag`; stops with an explanation if full text is missing. "
             "chunks: explicit choice to rebuild each patent's text from ALL its chunks in `patent_rag_chunks`.")
    manifest = st.text_input("Optional metadata CSV (patent_number, abstract, cpc_subclass …)", value="")
    cl1, cl2 = st.columns(2)
    batch = cl1.slider("Batch size", 2, 32, DEFAULT_BATCH_SIZE, help="Small batches keep memory low on an 8 GB machine.")
    limit = cl2.number_input("Sample limit (0 = all available)", 0, 100000, 0)
    b1, b2 = st.columns(2)
    run_update = b1.button("Build / update index", type="primary")
    confirm = b2.checkbox("I want to rebuild from scratch")
    run_rebuild = b2.button("Rebuild", disabled=not confirm)
    if run_update or run_rebuild:
        index = GraphIndex(DB)
        try:
            if run_rebuild:
                index.clear()
            with st.spinner("Reading patent records (read-only)…"):
                records_in, info = load_records(source, manifest_csv=manifest.strip() or None, limit=int(limit) or None)
            st.caption(f"Source: {info['source']} · {len(records_in):,} patents. " + " ".join(info["notes"]))
            model = get_patent_sbert()
            bar = st.progress(0.0, text="Starting…")

            def _progress(done: int, total: int, message: str) -> None:
                bar.progress(min(done / total, 1.0) if total else 1.0, text=message)

            summary = build_index(records_in, index, model, batch_size=batch, progress=_progress)
            ss["pg_build_msg"] = f"Encoded {summary['encoded']} patents; {summary['already_current']} were already current."
            index.close()
            st.rerun()
        except (IndexMismatchError, MissingPatentTextError, ModelLoadError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Indexing stopped: {exc}. Progress so far is saved; run it again to resume.")
        finally:
            try:
                index.close()
            except Exception:
                pass

if status["count"] == 0:
    st.info("The graph index is empty. Build it above (or with `python -m src.patent_sbert_embeddings build`).")
    st.stop()

stamp = Path(DB).stat().st_mtime
sim, records = load_graph(DB, stamp)

# --------------------------------------------------------------------------- #
# Query state transitions
# --------------------------------------------------------------------------- #

def _title(pid: str) -> str:
    return records.get(pid, {}).get("title") or "title not recorded"


def _embed_text(text: str):
    try:
        model = get_patent_sbert()
    except ModelLoadError as exc:
        st.error(str(exc))
        return None
    vec = np.asarray(model.encode([text[:6000]], convert_to_numpy=True, normalize_embeddings=True,
                                  show_progress_bar=False)[0], dtype=np.float32)
    if vec.shape[0] != sim.matrix.shape[1]:
        st.error("The encoder's vector size does not match this index. Rebuild the graph index.")
        return None
    return vec


def _set_text_query(text: str) -> None:
    text = " ".join(text.split())
    if not text:
        return
    with st.spinner("Embedding your query with PatentSBERTa…"):
        vec = _embed_text(text)
    if vec is not None:
        ss.update(pg_qvec=vec, pg_qtext=text, pg_qsource="text", pg_center=None, pg_selected=None, pg_ver=ss["pg_ver"] + 1)


def _set_patent_query(pid: str) -> None:
    if not sim.has(pid):
        st.warning(f"Patent {pid} is not in the graph index ({len(sim):,} patents), so it cannot be investigated here.")
        return
    ss.update(pg_qvec=sim.vector(pid), pg_qtext=_title(pid), pg_qsource="patent", pg_center=pid, pg_selected=None,
              pg_ver=ss["pg_ver"] + 1)


def _clear_query() -> None:
    ss.update(pg_qvec=None, pg_qtext="", pg_qsource=None, pg_center=None, pg_selected=None, pg_ver=ss["pg_ver"] + 1)


action = ss.pop("pg_action", None)
if action:
    kind, value = action
    if kind == "clear":
        _clear_query()
    elif kind == "select":
        ss["pg_selected"] = value
    elif kind == "patent":
        _set_patent_query(value)
    elif kind == "lastsearch":
        _set_text_query((ss.get("pi_query") or "")[:6000])
    elif kind == "restore":
        view = value
        for widget, name in (("pg_color", "color_by"), ("pg_edges", "edges"), ("pg_minedge", "min_edge"), ("pg_n", "n"),
                             ("pg_ncl", "domains")):
            ss[widget] = view[name]
        if view["source"] == "patent":
            _set_patent_query(view["center"])
        elif view["source"] == "text":
            _set_text_query(view["query"])
        else:
            _clear_query()
        if view.get("selected") and sim.has(view["selected"]):
            ss["pg_selected"] = view["selected"]

# --------------------------------------------------------------------------- #
# Sidebar: map settings, filters, investigations
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown("<div class='pi-label' style='margin:1.2rem 0 .3rem'>Map</div>", unsafe_allow_html=True)
    n_clusters = st.number_input("Number of domains (0 = automatic)", 0, 20, 0, key="pg_ncl",
                                 help="Domains are clusters of PatentSBERTa vectors. Automatic picks the best-separated count.")
m = load_map(DB, stamp, int(n_clusters))
domain_titles = {d.title: d for d in m.domains}
section_options = sorted({cpc_section(r) for r in records.values()}, key=lambda s: (s == "not recorded", s))
section_label = lambda s: f"{s} · {SECTION_NAMES[s]}" if s in SECTION_NAMES else "CPC not recorded"

with st.sidebar:
    st.markdown("<div class='pi-label' style='margin:1.2rem 0 .3rem'>View</div>", unsafe_allow_html=True)
    color_by = st.radio("Colour by", ["Semantic domain", "CPC section"], key="pg_color", horizontal=True)
    edges_per = st.slider("Links per patent", 0, 10, 4, key="pg_edges", help="Each patent's strongest neighbours only.")
    min_edge = st.slider("Minimum link cosine", 0.0, 0.95, 0.30, 0.01, key="pg_minedge")
    n_hits = st.slider("Results in an investigation", 5, 100, 25, key="pg_n")
    st.markdown("<div class='pi-label' style='margin:1rem 0 .3rem'>Filters</div>", unsafe_allow_html=True)
    chosen_domains = st.multiselect("Domains", list(domain_titles), default=list(domain_titles), key="pg_dom")
    chosen_sections = st.multiselect("CPC sections", section_options, default=section_options, key="pg_sec",
                                     format_func=section_label)

keep_domains = {domain_titles[t].index for t in chosen_domains}
active = np.isin(m.cluster, list(keep_domains)) & np.array([cpc_section(records.get(p)) in chosen_sections for p in m.ids])
filtered = int(active.sum()) < len(m.ids)

# --------------------------------------------------------------------------- #
# Query bar
# --------------------------------------------------------------------------- #

q1, q2 = st.columns([3.3, 2])
with q1:
    with st.form("pg_query_form", clear_on_submit=False):
        text = st.text_area("Investigate a technology", placeholder="Describe an invention or technical problem. The whole index is searched.",
                            height=84, label_visibility="collapsed")
        submitted = st.form_submit_button("Investigate", type="primary")
    if submitted and text.strip():
        _set_text_query(text)
    f1, f2, _ = st.columns([1.3, 1, 2.2])
    f1.button("Use my last search", on_click=_act, args=("lastsearch",), disabled=not ss.get("pi_query"),
              help="Investigate the text from the Search page.")
    f2.button("Back to map", on_click=_act, args=("clear",), disabled=ss["pg_qvec"] is None)
with q2:
    loc = st.text_input("Locate a patent", placeholder="Patent number or title words").strip().lower()
    if loc:
        options = [p for p, r in records.items() if loc in p.lower() or loc in (r.get("title") or "").lower()][:25]
        hint = "Matches"
    else:
        last = (ss.get("pi_response") or {}).get("results", [])
        options = [p for p in (normalize_id(x.get("patent_number")) for x in last) if p in records]
        hint = "From your last search"
    choice = st.selectbox(hint, options, format_func=lambda p: f"{p} — {_title(p)[:60]}", index=0 if options else None,
                          placeholder="No matching patents in the graph index")
    l1, l2 = st.columns(2)
    l1.button("Select on map", disabled=choice is None, on_click=_act, args=("select", choice))
    l2.button("Investigate from it", disabled=choice is None, on_click=_act, args=("patent", choice))

# --------------------------------------------------------------------------- #
# Mode, retrieval, selection
# --------------------------------------------------------------------------- #

investigating = ss["pg_qvec"] is not None
rag, rag_label, rag_missing = {}, "RAG score", 0
last_response = ss.get("pi_response") or {}
if last_response.get("results"):
    rag_label = "RRF score" if ss.get("pi_mode") == "document" else "RAG score (MiniLM chunks)"
    for p in last_response["results"]:
        pid = normalize_id(p.get("patent_number"))
        rag[pid] = (p.get("rank"), p.get("score"))
        rag_missing += 0 if sim.has(pid) else 1

hits: list[dict] = []
if investigating:
    exclude = sim.row(ss["pg_center"]) if ss["pg_qsource"] == "patent" and sim.has(ss["pg_center"]) else None
    hits = [{"idx": i, "pid": sim.ids[i], "cos": c}
            for i, c in sim.search(ss["pg_qvec"], n_hits, exclude=exclude, allowed=active if filtered else None)]

if ss["pg_selected"] is not None and not sim.has(ss["pg_selected"]):
    ss["pg_selected"] = None
if investigating and ss["pg_selected"] is None and hits:
    ss["pg_selected"] = hits[0]["pid"]
selected = ss["pg_selected"]


def _focus(pid: str | None):
    if not pid:
        return []
    return sim.search(sim.vector(pid), 8, exclude=sim.row(pid))


def _figure(sel: str | None):
    rev = f"{'inv' if investigating else 'map'}|{ss['pg_ver']}|{int(n_clusters)}"
    if investigating:
        if ss["pg_qsource"] == "patent" and sim.has(ss["pg_center"]):
            qxy = tuple(m.xy[sim.row(ss["pg_center"])])
        elif hits:
            w = np.array([max(h["cos"], 0.0) ** 2 for h in hits[:10]]) + 1e-9
            pts = np.array([m.xy[h["idx"]] for h in hits[:10]])
            qxy = tuple((pts * w[:, None]).sum(0) / w.sum())
        else:
            qxy = (0.0, 0.0)
        ids = [h["idx"] for h in hits]
        links = result_links(sim.matrix_for([h["pid"] for h in hits]), 3, min_edge) if len(hits) > 1 else []
        return investigate_figure(m, records, hits, query_xy=qxy, query_label=ss["pg_qtext"], links=links,
                                  color_by=color_by, selected=sel, uirevision=rev)
    return explore_figure(m, records, active=active, color_by=color_by, edges_per_node=edges_per, min_edge=min_edge,
                          selected=sel, focus=_focus(sel), uirevision=rev)


plot_key = "pg_plot_" + hashlib.md5(
    f"{investigating}|{ss['pg_ver']}|{n_hits}|{edges_per}|{min_edge}|{color_by}|{int(n_clusters)}|{sorted(keep_domains)}|{chosen_sections}".encode()
).hexdigest()[:12]
fig, lookup = _figure(selected)
clicked = picked_patent(ss.get(plot_key), lookup)
if clicked and ss.get("pg_click_seen") != (plot_key, clicked):
    ss["pg_click_seen"] = (plot_key, clicked)
    ss["pg_selected"] = selected = clicked
    fig, lookup = _figure(selected)

# --------------------------------------------------------------------------- #
# Graph + inspector
# --------------------------------------------------------------------------- #

left, right = st.columns([3.2, 1.25], gap="medium")
with left:
    if investigating:
        who = f"patent **{ss['pg_center']}**" if ss["pg_qsource"] == "patent" else "your query"
        scope = f"{int(active.sum()):,} filtered patents" if filtered else f"all {len(sim):,} indexed patents"
        st.markdown(f"<div class='pi-label'>Investigation · {who} · searched {scope}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='pi-label'>Map of {int(active.sum()):,} patents · {len(m.domains)} semantic domains</div>",
                    unsafe_allow_html=True)
    if not investigating and not active.any():
        st.info("The current filters hide every patent.")
    if investigating and not hits:
        st.info("No patents match the current filters.")
    st.plotly_chart(fig, key=plot_key, on_select="rerun", selection_mode="points", theme=None, **stretch(st.plotly_chart),
                    config={"displaylogo": False, "scrollZoom": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
    if investigating:
        st.caption("Amber star = your query, placed at the similarity-weighted centre of its best hits (a visual aid, not a "
                   "measurement). Dot size = cosine similarity to the query. Cyan lines link hits that are similar to each other.")
    else:
        st.caption(f"Layout: {m.layout} of PatentSBERTa vectors. Nearby patents are similar, but distances between far-apart "
                   "islands are not meaningful. Click a patent to inspect it; click legend entries to hide a domain; scroll to zoom.")

with right:
    st.markdown("<div class='pi-label'>Inspector</div>", unsafe_allow_html=True)
    if not selected:
        st.write("Click a patent on the map to inspect it.")
        st.markdown("<div class='pi-label' style='margin-top:.8rem'>Domains</div>", unsafe_allow_html=True)
        for d in m.domains:
            st.markdown(f"<span class='pi-dot' style='background:{d.color}'></span>{html.escape(d.title)} "
                        f"<span class='pi-why'>· {d.size:,}</span>", unsafe_allow_html=True)
    else:
        rec = records.get(selected, {})
        meta = rec.get("metadata", {})
        dom = m.domains[m.cluster[sim.row(selected)]]
        st.markdown(f"<div class='pi-id'>{html.escape(selected)}</div>", unsafe_allow_html=True)
        st.markdown(f"**{html.escape(_title(selected))}**")
        st.markdown(f"<span class='pi-dot' style='background:{dom.color}'></span>{html.escape(dom.title)}", unsafe_allow_html=True)
        i1, i2 = st.columns(2)
        i1.metric("CPC", cpc_subclass(rec))
        if investigating:
            cos_q = float(sim.matrix[sim.row(selected)] @ (ss["pg_qvec"] / np.linalg.norm(ss["pg_qvec"])))
            i2.metric("Cosine to query", f"{pct(cos_q):.1f}%")
        if selected in rag:
            r, s_ = rag[selected]
            st.caption(f"Search page: rank #{r} · {rag_label} {float(s_):.4f} (a different measure from cosine similarity)")
        shown = {k_: v for k_, v in meta.items() if k_ not in ("title", "abstract")}
        if shown:
            rows = "".join(f"<tr><td>{html.escape(str(k_))}</td><td>{html.escape(str(v))}</td></tr>" for k_, v in shown.items())
            st.markdown(f"<table class='pi-kv'>{rows}</table>", unsafe_allow_html=True)
        if not any("assignee" in key.lower() or "inventor" in key.lower() for key in meta):
            st.caption("Assignee, inventors and citations are not recorded in this index.")
        if meta.get("abstract"):
            st.markdown("<div class='pi-label' style='margin-top:.8rem'>Abstract</div>", unsafe_allow_html=True)
            st.write(meta["abstract"])
        else:
            with st.expander("Indexed text (what PatentSBERTa embedded)", expanded=True):
                st.write(rec.get("excerpt") or "Not available.")
        st.button("Investigate from this patent", key=f"inv_from_{selected}", on_click=_act, args=("patent", selected),
                  type="primary", **stretch(st.button))
        near = _focus(selected)
        if near:
            st.markdown("<div class='pi-label' style='margin:.8rem 0 .2rem'>Most similar in the index</div>", unsafe_allow_html=True)
            for j, c in near:
                other = sim.ids[j]
                st.button(f"{other} · {pct(c):.0f}% · {_title(other)[:34]}", key=f"nb_{selected}_{other}",
                          on_click=_act, args=("select", other), **stretch(st.button))

with st.sidebar:
    st.markdown("<div class='pi-label' style='margin:1.2rem 0 .3rem'>Investigations</div>", unsafe_allow_html=True)
    inv_name = st.text_input("Name", key="pg_inv_name", placeholder="Name this view", label_visibility="collapsed")
    if st.button("Save current view"):
        investigations.save(ss, inv_name, {
            "source": ss["pg_qsource"] or "explore", "query": ss["pg_qtext"] if ss["pg_qsource"] == "text" else "",
            "center": ss["pg_center"] or "", "selected": ss["pg_selected"] or "", "n": n_hits, "edges": edges_per,
            "min_edge": min_edge, "domains": int(n_clusters), "color_by": color_by})
    for i, item in enumerate(investigations.list_all(ss)[:12]):
        v = item["view"]
        what = v["center"] if v["source"] == "patent" else (v["query"][:24] + "…" if v["source"] == "text" else "map")
        st.button(f"{item['name']} · {what}", key=f"inv_{i}", on_click=_act, args=("restore", v), help=f"Saved {item['saved']}",
                  **stretch(st.button))
    if investigations.list_all(ss):
        st.download_button("Export investigations", investigations.to_json(ss), "investigations.json", "application/json")
    upload = st.file_uploader("Import investigations", type=["json"], key="pg_inv_upload", label_visibility="collapsed")
    if upload is not None and ss.get("pg_inv_sig") != (upload.name, upload.size):
        ss["pg_inv_sig"] = (upload.name, upload.size)
        try:
            st.caption(f"Imported {investigations.load_json(ss, upload.getvalue().decode('utf-8'))} investigation(s).")
        except (ValueError, UnicodeDecodeError):
            st.error("That file is not a valid investigations export.")

# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #

def _domain_table() -> list[dict]:
    return [{"domain": d.title, "patents": d.size, "share": f"{d.size / len(m.ids):.0%}", "CPC majority": d.cpc_hint} for d in m.domains]


if investigating:
    t_res, t_dom, t_mat = st.tabs(["Results", "Domains touched", "Similarity matrix"])
    with t_res:
        st.caption("Similarity % is the cosine between PatentSBERTa vectors. It is relative: unrelated patents often score 10–40%, so "
                   "compare results with each other rather than reading it as a probability.")
        rows = []
        for rank, h in enumerate(hits, start=1):
            pid, rec = h["pid"], records.get(h["pid"], {})
            row = {"rank": rank, "patent": pid, "title": _title(pid), "cosine similarity (%)": round(pct(h["cos"]), 1),
                   "domain": m.domains[m.cluster[h["idx"]]].title, "CPC": cpc_subclass(rec)}
            if rag:
                r, s_ = rag.get(pid, (None, None))
                row["search-page rank"] = r
                row[rag_label] = None if s_ is None else round(float(s_), 4)
            rows.append(row)
        if rows:
            st.dataframe(rows, hide_index=True, column_config={
                "cosine similarity (%)": st.column_config.ProgressColumn("cosine similarity (%)", min_value=0, max_value=100, format="%.1f")})
        if rag:
            st.caption(f"Search-page columns come from your last search on the Search page, which uses MiniLM + ChromaDB"
                       f"{' with RRF across document sections' if ss.get('pi_mode') == 'document' else ''}; they are not cosine "
                       f"similarities. {rag_missing} of its {len(rag)} patents are not in this graph index.")
    with t_dom:
        touched: dict[int, list[float]] = {}
        for h in hits:
            touched.setdefault(int(m.cluster[h["idx"]]), []).append(h["cos"])
        order = sorted(touched, key=lambda c: -len(touched[c]))
        st.plotly_chart(bar_chart([m.domains[c].title for c in order], [len(touched[c]) for c in order],
                                  [m.domains[c].color for c in order], "results in this domain"), theme=None, **stretch(st.plotly_chart))
        st.dataframe([{"domain": m.domains[c].title, "results": len(touched[c]), "best cosine (%)": round(pct(max(touched[c])), 1),
                       "mean cosine (%)": round(pct(float(np.mean(touched[c]))), 1), "CPC majority": m.domains[c].cpc_hint}
                      for c in order], hide_index=True)
    with t_mat:
        ids = [h["pid"] for h in hits][:40]
        if len(ids) > 2:
            st.plotly_chart(similarity_heatmap(ids, sim.matrix_for(ids), records), theme=None, **stretch(st.plotly_chart))
            st.caption("Pairwise cosine similarity between the top results" + (" (first 40 shown)." if len(hits) > 40 else "."))
        else:
            st.caption("Need at least three results.")
else:
    t_dom, t_cpc, t_list = st.tabs(["Domains", "CPC sections", "Patents"])
    with t_dom:
        st.caption("Domains are clusters of PatentSBERTa vectors, named by their most distinctive title/text terms. "
                   + (f"Cluster separation (silhouette): {m.silhouette:.2f}; values near 0 mean domains overlap." if m.silhouette is not None
                      else "Domain count was set manually."))
        st.dataframe(_domain_table(), hide_index=True)
    with t_cpc:
        counts: dict[str, int] = {}
        for p, on in zip(m.ids, active):
            if on:
                counts[cpc_section(records.get(p))] = counts.get(cpc_section(records.get(p)), 0) + 1
        keys = sorted(counts, key=lambda s: -counts[s])
        st.plotly_chart(bar_chart([section_label(k_) for k_ in keys], [counts[k_] for k_ in keys],
                                  ["#5E6B7A" if k_ == "not recorded" else "#4CC9E6" for k_ in keys], "patents"), theme=None, **stretch(st.plotly_chart))
        if set(counts) == {"not recorded"}:
            st.caption("CPC classes are not recorded in this index. Supply a metadata CSV with `cpc_subclass` and update the index to enable this view.")
    with t_list:
        listing = [{"patent": p, "title": _title(p), "domain": m.domains[m.cluster[i]].title, "CPC": cpc_subclass(records.get(p))}
                   for i, p in enumerate(m.ids) if active[i]][:500]
        st.dataframe(listing, hide_index=True)
        st.caption("First 500 shown." if int(active.sum()) > 500 else "")
