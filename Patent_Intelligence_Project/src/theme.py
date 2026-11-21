"""Shared visual system: graphite foundation, cyan = evidence/similarity, amber = focus."""

from __future__ import annotations

import inspect

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root { --bg:#0F1217; --panel:#151A21; --line:#252C36; --text:#D9DEE5; --muted:#8B95A3; --cyan:#4CC9E6; --amber:#F0A93B; }
html, body, .stApp, [data-testid="stAppViewContainer"] { background: var(--bg); color: var(--text);
  font-family: 'IBM Plex Sans', system-ui, sans-serif; }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1320px; padding-top: 1.4rem; padding-bottom: 3rem; }
h1 { font-size: 1.45rem; font-weight: 500; letter-spacing: -0.01em; margin-bottom: .15rem; }
h2, h3 { font-weight: 500; letter-spacing: -0.005em; }
h3 { font-size: 1.05rem; }
[data-testid="stSidebar"] { background: #0B0E12; border-right: 1px solid var(--line); }
[data-testid="stSidebarNav"] { display: none; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 2px; border-color: var(--line); }
.stButton > button, .stDownloadButton > button { border-radius: 2px; border: 1px solid var(--line); background: var(--panel);
  font-family: 'IBM Plex Mono', monospace; font-size: .78rem; letter-spacing: .02em; transition: border-color .15s, color .15s; }
.stButton > button:hover { border-color: var(--cyan); color: var(--cyan); }
.stButton > button[kind="primary"] { background: var(--amber); color: #11151A; border-color: var(--amber); }
.stButton > button[kind="primary"]:hover { color: #11151A; filter: brightness(1.08); }
[data-baseweb="tab-list"] { gap: 1.4rem; border-bottom: 1px solid var(--line); }
[data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace; font-size: .78rem; letter-spacing: .04em; text-transform: uppercase; }
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; font-size: 1.25rem; }
[data-testid="stMetricLabel"] { color: var(--muted); text-transform: uppercase; font-size: .68rem; letter-spacing: .08em; }
hr { border-color: var(--line); }
.pi-label { font-family: 'IBM Plex Mono', monospace; font-size: .68rem; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
.pi-brand { font-family: 'IBM Plex Mono', monospace; font-size: .8rem; letter-spacing: .16em; color: var(--text); }
.pi-brand b { color: var(--amber); font-weight: 500; }
.pi-sub { color: var(--muted); font-size: .98rem; margin-top: -.2rem; }
.pi-why { color: var(--muted); font-size: .85rem; }
.pi-kv { width: 100%; border-collapse: collapse; font-size: .82rem; }
.pi-kv td { padding: .28rem 0; border-bottom: 1px solid var(--line); vertical-align: top; }
.pi-kv td:first-child { color: var(--muted); font-family: 'IBM Plex Mono', monospace; font-size: .72rem; width: 38%; padding-right: .6rem; }
.pi-dot { display:inline-block; width:.7rem; height:.7rem; border-radius:50%; margin-right:.45rem; vertical-align:-1px; }
.pi-id { font-family: 'IBM Plex Mono', monospace; color: var(--amber); font-size: 1.05rem; }
</style>
"""


def apply_theme() -> None:
    """Inject the shared CSS. Call right after ``st.set_page_config``."""
    st.markdown(CSS, unsafe_allow_html=True)


def render_nav(active: str) -> None:
    """Compact left rail. ``active`` is 'search' or 'graph'."""
    with st.sidebar:
        st.markdown("<div class='pi-brand'>PATENT<b>/</b>INTELLIGENCE</div>", unsafe_allow_html=True)
        st.markdown("<div class='pi-label' style='margin:.9rem 0 .2rem'>Workspace</div>", unsafe_allow_html=True)
        try:
            st.page_link("streamlit_app.py", label="Search & evidence" + (" ●" if active == "search" else ""))
            st.page_link("pages/1_Patent_Graph.py", label="Patent graph" + (" ●" if active == "graph" else ""))
        except Exception:  # e.g. running a page outside the multipage context
            st.caption("Search & evidence · Patent graph")


def stretch(widget) -> dict:
    """Full-width kwargs that work on both old (use_container_width) and new (width='stretch') Streamlit."""
    try:
        if "width" in inspect.signature(widget).parameters:
            return {"width": "stretch"}
    except (TypeError, ValueError):
        pass
    return {"use_container_width": True}
