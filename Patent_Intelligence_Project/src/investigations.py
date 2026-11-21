"""Saved investigations: named Patent Graph views kept in the Streamlit session, exportable as JSON."""

from __future__ import annotations

import json
from datetime import datetime, timezone

KEY = "pi_investigations"
MAX_SAVED = 50
MAX_IMPORT_BYTES = 1_000_000
FORMAT = "patent-intelligence-investigations"
COLOR_OPTIONS = ("Semantic domain", "CPC section")
_LEGACY_COLORS = {"Hop distance": "Semantic domain", "CPC subclass": "CPC section"}


def _clamp(value, lo, hi, cast, default):
    try:
        return min(max(cast(value), lo), hi)
    except (TypeError, ValueError):
        return default


def _clean_view(view: dict) -> dict:
    """View = explore | text query | patent-as-query, plus display settings. Legacy centre-only views map to 'patent'."""
    if not isinstance(view, dict):
        raise ValueError("bad view")
    center = str(view.get("center") or "").strip()
    query = " ".join(str(view.get("query") or "").split())[:2000]
    source = view.get("source")
    if source not in ("explore", "text", "patent"):
        source = "patent" if center else "explore"
    if source == "patent" and not center:
        raise ValueError("view needs a centre patent")
    if source == "text" and not query:
        raise ValueError("view needs a query")
    color = _LEGACY_COLORS.get(view.get("color_by"), view.get("color_by"))
    return {
        "source": source, "query": query, "center": center,
        "selected": str(view.get("selected") or "").strip(),
        "n": _clamp(view.get("n"), 5, 100, int, 25),
        "edges": _clamp(view.get("edges", view.get("k")), 0, 10, int, 4),
        "min_edge": round(_clamp(view.get("min_edge", view.get("threshold")), 0.0, 0.95, float, 0.3), 2),
        "domains": _clamp(view.get("domains"), 0, 20, int, 0),
        "color_by": color if color in COLOR_OPTIONS else COLOR_OPTIONS[0],
    }


def list_all(ss) -> list[dict]:
    return list(ss.get(KEY, []))


def _put(ss, item: dict) -> None:
    items = [i for i in ss.get(KEY, []) if i["name"] != item["name"]]
    ss[KEY] = [item, *items][:MAX_SAVED]


def save(ss, name: str, view: dict) -> dict:
    items = ss.get(KEY, [])
    label = " ".join((name or "").split())[:60] or f"View {len(items) + 1}"
    item = {"name": label, "saved": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "view": _clean_view(view)}
    _put(ss, item)
    return item


def to_json(ss) -> str:
    return json.dumps({"format": FORMAT, "version": 2, "investigations": list_all(ss)}, indent=2)


def load_json(ss, text: str) -> int:
    """Merge an export into the session. Returns the number imported; raises ValueError if invalid."""
    if len(text.encode("utf-8")) > MAX_IMPORT_BYTES:
        raise ValueError("file too large")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("not JSON") from exc
    items = data.get("investigations") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("no investigations list")
    clean = []
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("bad entry")
        clean.append({
            "name": " ".join(str(raw.get("name") or "").split())[:60] or "Imported view",
            "saved": str(raw.get("saved") or "")[:40],
            "view": _clean_view(raw.get("view")),
        })
    for item in reversed(clean):
        _put(ss, item)
    return len(clean)
