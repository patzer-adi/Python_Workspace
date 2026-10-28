
from pathlib import Path
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
GRAPH_DIR = PROJECT_DIR / "data" / "patent_graph_2016"

GRAPH_PATH = GRAPH_DIR / "patent_similarity_graph.graphml"
SAMPLE_PATH = (
    GRAPH_DIR / "samples" / "patents_2016_sample_2500.parquet"
)

st.set_page_config(
    page_title="Patent Graph Explorer",
    page_icon="🔬",
    layout="wide",
)

st.title("Patent Graph Explorer")
st.caption(
    "Explore relationships between patents using embedding similarity. "
    "Click any node to select it and highlight its five strongest connections."
)


# ============================================================
# LOAD EXISTING DATA
# ============================================================

@st.cache_data(show_spinner="Loading saved patent graph...")
def load_data():
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Graph not found: {GRAPH_PATH}")

    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Patent sample not found: {SAMPLE_PATH}")

    graph = nx.read_graphml(GRAPH_PATH)
    graph = nx.Graph(graph)

    # Standardise identifiers.
    graph = nx.relabel_nodes(graph, lambda n: str(n))

    patents = pd.read_parquet(SAMPLE_PATH)
    patents["patent_number"] = patents["patent_number"].astype(str)
    patents = patents.drop_duplicates("patent_number")
    patents = patents.set_index("patent_number", drop=False)

    # Use the sample as the authoritative source for patent details.
    missing = set(graph.nodes()) - set(patents.index)
    if missing:
        raise ValueError(
            f"{len(missing)} graph nodes are missing from the patent sample."
        )

    return graph, patents


@st.cache_data(show_spinner="Preparing graph layout and communities...")
def prepare_graph():
    graph, patents = load_data()

    # Build the union of the five strongest neighbours per patent.
    # This makes the display less cluttered than the full k=10 graph.
    reduced = nx.Graph()
    reduced.add_nodes_from(graph.nodes(data=True))

    for node in graph.nodes():
        strongest = sorted(
            graph[node].items(),
            key=lambda item: float(item[1].get("weight", 0)),
            reverse=True,
        )[:5]

        for neighbour, attrs in strongest:
            if node != neighbour:
                reduced.add_edge(
                    node,
                    neighbour,
                    weight=float(attrs.get("weight", 0)),
                )

    # Community detection.
    communities = nx.community.louvain_communities(
        reduced,
        weight="weight",
        resolution=1.0,
        seed=42,
    )

    cluster_of = {
        str(node): cid
        for cid, members in enumerate(communities)
        for node in members
    }

    # Compute a layout once and cache it.
    # Similarity-weighted edges influence attraction.
    positions = nx.spring_layout(
        reduced,
        seed=42,
        k=0.12,
        iterations=100,
        weight="weight",
    )

    positions = {
        str(node): (float(x), float(y))
        for node, (x, y) in positions.items()
    }

    return (
        reduced,
        patents,
        cluster_of,
        positions,
        len(communities),
    )


try:
    G, patents, cluster_of, positions, n_communities = prepare_graph()
except Exception as exc:
    st.error(f"Could not prepare the graph: {exc}")
    st.info(
        "Check that the graph and sample files exist at the paths "
        "configured at the top of this script."
    )
    st.stop()


# ============================================================
# SIDEBAR: FILTERS AND SEARCH
# ============================================================

st.sidebar.header("Explore patents")

query = st.sidebar.text_input(
    "Search by patent number or title",
    placeholder="e.g. 15006410 or wireless",
)

domain_values = (
    patents["ipc_label"].fillna("").astype(str)
    if "ipc_label" in patents.columns
    else pd.Series("", index=patents.index)
)

# Build broad domain labels from the first IPC section.
SECTION_DOMAINS = {
    "A": "Healthcare, Agriculture & Consumer Products",
    "B": "Transport, Manufacturing & Packaging",
    "C": "Chemistry, Pharmaceuticals & Materials",
    "D": "Textiles, Fabrics & Paper",
    "E": "Construction, Buildings & Mining",
    "F": "Mechanical Engineering, Engines & Heating",
    "G": "Computing, Optics & Measurement",
    "H": "Electronics, Electrical Systems & Communications",
}

patents = patents.copy()
patents["ipc_section"] = domain_values.str[:1]
patents["technology_domain"] = (
    patents["ipc_section"].map(SECTION_DOMAINS)
    .fillna("Other / Unclassified")
)

domains = ["All domains"] + sorted(
    patents["technology_domain"].dropna().unique().tolist()
)

selected_domain = st.sidebar.selectbox(
    "Technology domain",
    domains,
)

min_similarity = st.sidebar.slider(
    "Minimum displayed edge similarity",
    min_value=0.75,
    max_value=0.99,
    value=0.88,
    step=0.01,
    help="Changes which edges are displayed, not the saved graph.",
)

show_all_edges = st.sidebar.checkbox(
    "Show all graph edges",
    value=False,
    help="Showing all edges can make the graph visually dense.",
)

st.sidebar.caption(
    "Node colour = detected community. "
    "Edge colour = cosine-similarity band."
)


# ============================================================
# SEARCH RESULTS AND SELECTED PATENT
# ============================================================

if query.strip():
    q = query.strip().lower()

    search_mask = (
        patents["patent_number"].str.lower().str.contains(q, na=False)
    )

    if "title" in patents.columns:
        search_mask |= patents["title"].fillna("").astype(str).str.lower().str.contains(
            q, na=False
        )

    search_results = patents.loc[search_mask].head(100)
else:
    search_results = patents.iloc[0:0]

search_options = search_results["patent_number"].tolist()

selected_patent = None

if search_options:
    selected_patent = st.sidebar.selectbox(
        "Matching patents",
        search_options,
        format_func=lambda pid: (
            f"{pid} — "
            f"{str(patents.loc[pid].get('title', 'Untitled'))[:85]}"
        ),
        key="sidebar_patent_selection",
    )
elif query.strip():
    st.sidebar.warning("No matching patents found.")
else:
    # Keep the most recently clicked node selected across Streamlit reruns.
    selected_patent = st.session_state.get("clicked_patent")


# ============================================================
# DOMAIN FILTER
# ============================================================

visible_nodes = set(G.nodes())

if selected_domain != "All domains":
    visible_nodes = {
        str(pid)
        for pid, row in patents.iterrows()
        if row["technology_domain"] == selected_domain
    }

# Keep the selected patent available when a domain filter excludes it.
if selected_patent:
    visible_nodes.add(selected_patent)

# Build displayed edge list.
all_edges = []

for u, v, attrs in G.edges(data=True):
    u, v = str(u), str(v)
    similarity = float(attrs.get("weight", 0))

    if u not in visible_nodes or v not in visible_nodes:
        continue

    if not show_all_edges and similarity < min_similarity:
        continue

    all_edges.append((u, v, similarity))


# ============================================================
# GRAPH STATISTICS
# ============================================================

m1, m2, m3, m4 = st.columns(4)

m1.metric("Patents", f"{len(visible_nodes):,}")
m2.metric("Displayed edges", f"{len(all_edges):,}")
m3.metric("Communities", n_communities)
m4.metric(
    "Connected components",
    nx.number_connected_components(G.subgraph(visible_nodes))
    if visible_nodes
    else 0,
)


# ============================================================
# GRAPH RENDERING
# ============================================================

EDGE_BANDS = [
    (0.00, 0.80, "Similarity < 0.80", "#64748b"),
    (0.80, 0.85, "0.80–0.85", "#38bdf8"),
    (0.85, 0.90, "0.85–0.90", "#34d399"),
    (0.90, 0.95, "0.90–0.95", "#fbbf24"),
    (0.95, 1.01, "0.95+", "#f472b6"),
]

fig = go.Figure()

# When a patent is selected, highlight its five strongest graph connections.
highlight_nodes = set()
highlight_edges = set()
top_neighbours = []

if selected_patent and selected_patent in G:
    highlight_nodes.add(selected_patent)

    top_neighbours = sorted(
        (
            (str(neighbour), float(attrs.get("weight", 0.0)))
            for neighbour, attrs in G[selected_patent].items()
            if str(neighbour) in visible_nodes
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:5]

    for neighbour, _similarity in top_neighbours:
        highlight_nodes.add(neighbour)
        highlight_edges.add(tuple(sorted((selected_patent, neighbour))))


# Draw ordinary edges first. Selected connections are drawn separately in neon.
for low, high, label, color in EDGE_BANDS:
    xs, ys = [], []

    for u, v, similarity in all_edges:
        if not (low <= similarity < high):
            continue

        if selected_patent and tuple(sorted((u, v))) in highlight_edges:
            continue

        x0, y0 = positions[u]
        x1, y1 = positions[v]
        xs.extend([x0, x1, None])
        ys.extend([y0, y1, None])

    if xs:
        fig.add_trace(go.Scattergl(
            x=xs,
            y=ys,
            mode="lines",
            name=label,
            line=dict(color=color, width=0.7),
            opacity=0.12 if selected_patent else 0.45,
            hoverinfo="skip",
        ))

# Neon halo and bright core for the selected patent's top five connections.
if selected_patent and top_neighbours:
    halo_x, halo_y = [], []
    core_x, core_y = [], []

    for neighbour, _similarity in top_neighbours:
        x0, y0 = positions[selected_patent]
        x1, y1 = positions[neighbour]
        halo_x.extend([x0, x1, None])
        halo_y.extend([y0, y1, None])
        core_x.extend([x0, x1, None])
        core_y.extend([y0, y1, None])

    fig.add_trace(go.Scattergl(
        x=halo_x,
        y=halo_y,
        mode="lines",
        name="Strongest connections — glow",
        line=dict(color="#00ff9c", width=9),
        opacity=0.20,
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.add_trace(go.Scattergl(
        x=core_x,
        y=core_y,
        mode="lines",
        name="Top 5 strongest connections",
        line=dict(color="#39ffb6", width=3),
        opacity=1.0,
        hoverinfo="skip",
        showlegend=True,
    ))


# Draw nodes, using a dimmed style for unrelated patents when selected.
node_ids = sorted(visible_nodes)

node_x, node_y = [], []
node_colours, node_sizes, node_opacity = [], [], []
node_titles, node_custom = [], []

for pid in node_ids:
    x, y = positions[pid]
    row = patents.loc[pid]

    cid = cluster_of.get(pid, -1)
    base_color = f"hsl({(cid * 137.508) % 360:.1f}, 70%, 58%)"

    if selected_patent:
        if pid == selected_patent:
            color, size, opacity = "#ffffff", 17, 1.0
        elif pid in highlight_nodes:
            color, size, opacity = "#39ffb6", 12, 1.0
        else:
            color, size, opacity = "#374151", 4, 0.18
    else:
        color, size, opacity = base_color, 6, 0.90

    node_x.append(x)
    node_y.append(y)
    node_colours.append(color)
    node_sizes.append(size)
    node_opacity.append(opacity)

    title = str(row.get("title", "Title unavailable"))
    ipc = str(row.get("ipc_label", ""))

    node_titles.append(title)
    node_custom.append([pid, ipc, cid, G.degree(pid)])

fig.add_trace(go.Scattergl(
    x=node_x,
    y=node_y,
    mode="markers",
    name="Patents",
    marker=dict(
        size=node_sizes,
        color=node_colours,
        opacity=node_opacity,
        line=dict(width=0.5, color="#111827"),
    ),
    customdata=node_custom,
    text=node_titles,
    hovertemplate=(
        "<b>%{text}</b><br>"
        "Patent: %{customdata[0]}<br>"
        "IPC: %{customdata[1]}<br>"
        "Community: %{customdata[2]}<br>"
        "Graph degree: %{customdata[3]}"
        "<extra></extra>"
    ),
))

fig.update_layout(
    template="plotly_dark",
    title="Patent Similarity Network",
    height=780,
    margin=dict(l=5, r=5, t=55, b=5),
    hovermode="closest",
    legend=dict(orientation="h", y=1.02, x=0),
    xaxis=dict(visible=False),
    yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
)

clicked_points = plotly_events(
    fig,
    click_event=True,
    select_event=False,
    hover_event=False,
    override_height=780,
    override_width="100%",
    key="patent_graph_clicks",
)

# The patent nodes are the final Plotly trace, and node_ids matches their point order.
if clicked_points:
    clicked_point = clicked_points[0]
    point_index = clicked_point.get("pointIndex", clicked_point.get("pointNumber"))
    curve_number = clicked_point.get("curveNumber")

    if curve_number == len(fig.data) - 1 and point_index is not None:
        point_index = int(point_index)
        if 0 <= point_index < len(node_ids):
            clicked_patent = node_ids[point_index]
            if clicked_patent != st.session_state.get("clicked_patent"):
                st.session_state["clicked_patent"] = clicked_patent
                st.rerun()


# ============================================================
# SELECTED PATENT DETAILS
# ============================================================

if selected_patent and selected_patent in patents.index:
    row = patents.loc[selected_patent]

    st.divider()
    st.subheader(str(row.get("title", "Untitled patent")))
    st.caption(f"Patent number: {selected_patent}")

    detail1, detail2, detail3 = st.columns(3)

    detail1.write("**Technology domain**")
    detail1.write(row["technology_domain"])

    detail2.write("**IPC classification**")
    detail2.write(str(row.get("ipc_label", "Not available")))

    detail3.write("**CPC classification**")
    detail3.write(str(row.get("cpc_label", "Not available")))

    abstract = str(row.get("abstract", "")).strip()

    with st.expander("Read abstract", expanded=True):
        st.write(
            abstract
            if abstract and abstract.lower() != "nan"
            else "No abstract available."
        )

    st.subheader("Connected patents")

    neighbours = []
    for neighbour, attrs in G[selected_patent].items():
        neighbour = str(neighbour)
        similarity = float(attrs.get("weight", 0))

        if neighbour in patents.index:
            neighbours.append((neighbour, similarity))

    neighbours.sort(key=lambda item: item[1], reverse=True)

    if neighbours:
        neighbour_df = pd.DataFrame([
            {
                "Patent number": pid,
                "Title": str(patents.loc[pid].get("title", "Untitled")),
                "Cosine similarity": round(similarity, 4),
                "Technology domain": patents.loc[pid]["technology_domain"],
            }
            for pid, similarity in neighbours
        ])

        st.dataframe(
            neighbour_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption("The five strongest visible connections are highlighted in neon green on the graph.")
    else:
        st.info("No connected patents found.")
else:
    st.info(
        "Search for a patent in the sidebar to inspect its details "
        "and highlight its neighbours in the graph."
    )
