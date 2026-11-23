"""Plotly figures for the Patent Graph (graphite background; domain colours; amber = the query / focus)."""

from __future__ import annotations

import html
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go

from src.patent_graph import NOT_RECORDED, SECTION_NAMES, cpc_section, cpc_subclass, pct
from src.patent_map import MapData

BG, PANEL, LINE, TEXT, MUTED = "#0F1217", "#151A21", "#252C36", "#D9DEE5", "#8B95A3"
CYAN, AMBER, STEEL = "#4CC9E6", "#F0A93B", "#5E6B7A"
SECTION_COLORS = {"A": "#8FD18B", "B": "#FF8A65", "C": "#C792EA", "D": "#BCAAA4", "E": "#D4E157",
                  "F": "#6C8CFF", "G": "#4CC9E6", "H": "#F48FB1", "Y": "#4DB6AC", NOT_RECORDED: STEEL}
FONT = dict(family="IBM Plex Mono, monospace", color=TEXT, size=11)
HOVER = ("<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[2]}<br>CPC %{customdata[3]}"
         "%{customdata[4]}<extra></extra>")


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def _base(fig: go.Figure, height: int, uirevision: str) -> go.Figure:
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=FONT, height=height, uirevision=uirevision,
        margin=dict(l=0, r=0, t=6, b=0), dragmode="pan",
        legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=10),
                    itemsizing="constant"),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=LINE, font=FONT),
    )
    return fig


def _lines(x0, y0, x1, y1) -> tuple[np.ndarray, np.ndarray]:
    n = len(x0)
    xs, ys = np.full(3 * n, np.nan), np.full(3 * n, np.nan)
    xs[0::3], xs[1::3], ys[0::3], ys[1::3] = x0, x1, y0, y1
    return xs, ys


def _line_trace(x0, y0, x1, y1, color: str, width: float) -> go.Scattergl:
    xs, ys = _lines(x0, y0, x1, y1)
    return go.Scattergl(x=xs, y=ys, mode="lines", line=dict(color=color, width=width), hoverinfo="skip", showlegend=False)


def _grouping(m: MapData, records: dict, color_by: str):
    """Per-node group key, colour per group, legend text per group, ordered groups."""
    if color_by == "CPC section":
        names = np.array([cpc_section(records.get(p)) for p in m.ids], dtype=object)
        order = sorted(set(names), key=lambda s: (s == NOT_RECORDED, s))
        legend = {s: (f"{s} · {SECTION_NAMES[s]}" if s in SECTION_NAMES else "CPC not recorded") for s in order}
        colors = {s: SECTION_COLORS.get(s, STEEL) for s in order}
    else:
        names = np.array([m.domains[c].key for c in m.cluster], dtype=object)
        order = [d.key for d in m.domains]
        legend = {d.key: d.title for d in m.domains}
        colors = {d.key: d.color for d in m.domains}
    return names, colors, legend, order


def _custom(m: MapData, records: dict, idx, domain_names: dict, extra=None):
    out = []
    for n, i in enumerate(idx):
        pid = m.ids[i]
        rec = records.get(pid, {})
        out.append([pid, html.escape((rec.get("title") or "title not recorded")[:90]),
                    html.escape(domain_names[m.cluster[i]]), cpc_subclass(rec), extra[n] if extra else ""])
    return out


def _domain_labels(m: MapData, mask: np.ndarray, alpha: float) -> go.Scatter | None:
    xs, ys, texts, cols = [], [], [], []
    for d in m.domains:
        idx = np.where(mask & (m.cluster == d.index))[0]
        if not len(idx):
            continue
        xs.append(float(np.median(m.xy[idx, 0])))
        ys.append(float(np.median(m.xy[idx, 1])))
        texts.append(f"<b>{d.key}</b><br>" + d.label.replace(" · ", "<br>"))
        cols.append(_rgba(d.color, alpha))
    if not xs:
        return None
    return go.Scatter(x=xs, y=ys, mode="text", text=texts, textfont=dict(size=12, color=cols, family=FONT["family"]),
                      hoverinfo="skip", showlegend=False)


def _register(lookup: dict, curve: int, ids: list[str]) -> None:
    lookup["by_curve"][curve] = ids
    for p in ids:
        lookup["by_pid"][p] = p


def _new_lookup() -> dict:
    return {"by_pid": {}, "by_curve": {}}


def _selected_ring(fig, m, selected):
    if selected and selected in m.ids:
        i = m.ids.index(selected)
        fig.add_trace(go.Scatter(x=[m.xy[i, 0]], y=[m.xy[i, 1]], mode="markers", hoverinfo="skip", showlegend=False,
                                 marker=dict(size=24, color="rgba(0,0,0,0)", line=dict(color="#FFFFFF", width=2.5))))


# --------------------------------------------------------------------------- #
# Explore: the whole map
# --------------------------------------------------------------------------- #

def explore_figure(m: MapData, records: dict, *, active: np.ndarray, color_by: str = "Semantic domain",
                   edges_per_node: int = 4, min_edge: float = 0.3, selected: str | None = None,
                   focus: list[tuple[int, float]] | None = None, uirevision: str = "explore"):
    """Every indexed patent as a node, coloured by domain/CPC, with each patent's strongest links."""
    fig, lookup = go.Figure(), _new_lookup()
    xy = m.xy
    names, colors, legend, order = _grouping(m, records, color_by)
    domain_names = {d.index: d.title for d in m.domains}

    keep = ((m.edge_rank <= edges_per_node) & (m.edge_w >= min_edge) & active[m.edge_src] & active[m.edge_dst]
            if edges_per_node > 0 else np.zeros(len(m.edge_src), dtype=bool))
    s, t, w = m.edge_src[keep], m.edge_dst[keep], m.edge_w[keep]
    degree = np.bincount(np.concatenate([s, t]), minlength=len(m.ids)) if len(s) else np.zeros(len(m.ids), int)

    same = m.cluster[s] == m.cluster[t] if len(s) else np.zeros(0, bool)
    if len(s):
        cross = ~same
        if cross.any():
            fig.add_trace(_line_trace(xy[s[cross], 0], xy[s[cross], 1], xy[t[cross], 0], xy[t[cross], 1],
                                      "rgba(139,149,163,0.16)", 0.6))
        for d in m.domains:
            sel = same & (m.cluster[s] == d.index)
            if sel.any():
                fig.add_trace(_line_trace(xy[s[sel], 0], xy[s[sel], 1], xy[t[sel], 0], xy[t[sel], 1], _rgba(d.color, 0.30), 0.7))

    dim = np.where(~active)[0]
    if len(dim):
        fig.add_trace(go.Scattergl(x=xy[dim, 0], y=xy[dim, 1], mode="markers", hoverinfo="skip", showlegend=False,
                                   marker=dict(size=4, color="rgba(139,149,163,0.20)")))
    for g in order:
        idx = np.where(active & (names == g))[0]
        if not len(idx):
            continue
        fig.add_trace(go.Scattergl(
            x=xy[idx, 0], y=xy[idx, 1], mode="markers", name=legend[g], hovertemplate=HOVER,
            marker=dict(size=5 + np.minimum(degree[idx], 8) * 0.9, color=colors[g], opacity=0.92, line=dict(width=0)),
            customdata=_custom(m, records, idx, domain_names)))
        _register(lookup, len(fig.data) - 1, [m.ids[i] for i in idx])

    if color_by != "CPC section":
        lab = _domain_labels(m, active, 0.95)
        if lab is not None:
            fig.add_trace(lab)

    if selected and selected in m.ids:
        i0 = m.ids.index(selected)
        if focus:
            fx = [xy[j, 0] for j, _ in focus]
            fy = [xy[j, 1] for j, _ in focus]
            fig.add_trace(_line_trace(np.full(len(fx), xy[i0, 0]), np.full(len(fx), xy[i0, 1]), fx, fy, _rgba(AMBER, 0.55), 1.4))
            fig.add_trace(go.Scatter(x=fx, y=fy, mode="markers+text", text=[m.ids[j] for j, _ in focus], textposition="top center",
                                     textfont=dict(size=10, color=TEXT, family=FONT["family"]), hoverinfo="skip",
                                     showlegend=False, marker=dict(size=9, color="rgba(0,0,0,0)", line=dict(color=AMBER, width=1.5))))
        _selected_ring(fig, m, selected)
    _base(fig, 720, uirevision)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig, lookup


# --------------------------------------------------------------------------- #
# Investigate: a query against the whole index
# --------------------------------------------------------------------------- #

def investigate_figure(m: MapData, records: dict, hits: list[dict], *, query_xy: tuple[float, float],
                       query_label: str, links: list[tuple[int, int, float]], color_by: str = "Semantic domain",
                       selected: str | None = None, uirevision: str = "investigate"):
    """Dimmed map + the best hits sized by cosine, linked to the query node and to each other."""
    fig, lookup = go.Figure(), _new_lookup()
    xy = m.xy
    names, colors, legend, order = _grouping(m, records, color_by)
    domain_names = {d.index: d.title for d in m.domains}

    fig.add_trace(go.Scattergl(x=xy[:, 0], y=xy[:, 1], mode="markers", hoverinfo="skip", showlegend=False,
                               marker=dict(size=4, color=[_rgba(colors[g], 0.14) for g in names])))
    if color_by != "CPC section":
        lab = _domain_labels(m, np.ones(len(m.ids), bool), 0.45)
        if lab is not None:
            fig.add_trace(lab)

    qx, qy = query_xy
    hx = np.array([xy[h["idx"], 0] for h in hits])
    hy = np.array([xy[h["idx"], 1] for h in hits])
    cos = np.array([h["cos"] for h in hits])
    if len(hits):
        rank = np.empty(len(cos), dtype=int)
        rank[np.argsort(-cos)] = np.arange(len(cos))
        third = len(cos) / 3
        for sel, width, alpha in ((rank >= 2 * third, 0.9, 0.25), ((rank >= third) & (rank < 2 * third), 1.5, 0.40),
                                  (rank < third, 2.4, 0.65)):
            if sel.any():
                fig.add_trace(_line_trace(np.full(sel.sum(), qx), np.full(sel.sum(), qy), hx[sel], hy[sel], _rgba(AMBER, alpha), width))
    if links:
        a, b = np.array([l[0] for l in links]), np.array([l[1] for l in links])
        fig.add_trace(_line_trace(hx[a], hy[a], hx[b], hy[b], _rgba(CYAN, 0.35), 0.9))

    span = float(cos.max() - cos.min()) if len(cos) else 0.0
    sizes_all = 10 + 22 * (cos - cos.min()) / (span + 1e-9) if len(cos) else cos
    for g in order:
        pos = [n for n, h in enumerate(hits) if names[h["idx"]] == g]
        if not pos:
            continue
        idx = [hits[n]["idx"] for n in pos]
        fig.add_trace(go.Scattergl(
            x=xy[idx, 0], y=xy[idx, 1], mode="markers", name=legend[g], hovertemplate=HOVER,
            marker=dict(size=sizes_all[pos], color=colors[g], opacity=0.95, line=dict(color=BG, width=1)),
            customdata=_custom(m, records, idx, domain_names, [f"<br>cosine similarity {pct(cos[n]):.1f}%" for n in pos])))
        _register(lookup, len(fig.data) - 1, [m.ids[i] for i in idx])

    top = list(range(min(12, len(hits))))
    if selected and selected in m.ids:
        top += [n for n, h in enumerate(hits) if m.ids[h["idx"]] == selected and n not in top]
    if top:
        fig.add_trace(go.Scatter(x=hx[top], y=hy[top], mode="text", text=[m.ids[hits[n]["idx"]] for n in top],
                                 textposition="top center", textfont=dict(size=10, color=TEXT, family=FONT["family"]),
                                 hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[qx], y=[qy], mode="markers+text", text=["QUERY"], textposition="bottom center", name="Query",
                             textfont=dict(size=11, color=AMBER, family=FONT["family"]),
                             marker=dict(symbol="star", size=28, color=AMBER, line=dict(color="#FFFFFF", width=1.5)),
                             hovertemplate=f"<b>Query</b><br>{html.escape(query_label[:120])}<extra></extra>"))
    _selected_ring(fig, m, selected)
    _base(fig, 720, uirevision)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig, lookup


def picked_patent(state, lookup: dict) -> str | None:
    """Patent id from a Streamlit plotly selection state (dict-like or attribute-style), or None."""
    if not state:
        return None
    try:
        points = state["selection"]["points"]
    except (KeyError, TypeError):
        points = getattr(getattr(state, "selection", None), "points", None) or []
    for p in points:
        if not hasattr(p, "get"):
            continue
        cd = p.get("customdata")
        pid = cd[0] if isinstance(cd, (list, tuple)) and cd else cd
        if isinstance(pid, str) and pid in lookup["by_pid"]:
            return pid
        members = lookup["by_curve"].get(p.get("curve_number"))
        i = p.get("point_index")
        if members is not None and isinstance(i, int) and 0 <= i < len(members):
            return members[i]
    return None


# --------------------------------------------------------------------------- #
# Supporting charts
# --------------------------------------------------------------------------- #

def similarity_heatmap(ids: list[str], matrix, records: dict) -> go.Figure:
    z = np.asarray(matrix, dtype=float)
    fig = go.Figure(go.Heatmap(z=z * 100, x=ids, y=ids, zmin=0, zmax=100, colorscale=[[0, PANEL], [1, CYAN]],
                               hovertemplate="%{y} × %{x}<br>cosine %{z:.1f}%<extra></extra>",
                               colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(size=9), ticksuffix="%")))
    fig.update_xaxes(type="category", tickfont=dict(size=9), tickangle=-60, showgrid=False)
    fig.update_yaxes(type="category", tickfont=dict(size=9), autorange="reversed", showgrid=False)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=FONT,
                      height=max(420, 14 * len(ids) + 140), margin=dict(l=0, r=0, t=8, b=0))
    return fig


def bar_chart(labels: list[str], values: list[float], colors: list[str], xtitle: str) -> go.Figure:
    fig = go.Figure()
    if labels:
        fig.add_trace(go.Bar(x=values, y=labels, orientation="h", marker=dict(color=colors),
                             hovertemplate="%{y}: %{x}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
    fig.update_xaxes(title_text=xtitle, gridcolor=LINE, zeroline=False)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=FONT,
                      height=max(220, 28 * len(labels) + 90), margin=dict(l=0, r=0, t=8, b=0))
    return fig
