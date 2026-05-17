import json

from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"
SOURCE_CATALOG_PATH = ROOT_DIR / "shl_product_catalog.json"


def _repair_json_text(text: str) -> str:
    out = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string and ch in "\r\n\t":
            out.append(" ")
            continue
        if ch == '"' and not escaped:
            in_string = not in_string
        out.append(ch)
        escaped = ch == "\\" and not escaped
        if ch != "\\":
            escaped = False
    return "".join(out)


def _load_raw_catalog(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(_repair_json_text(text))


def get_assessment(name: str) -> Optional[Dict[str, Any]]:
    """
    Fetch the full raw JSON entry for an assessment from catalog.json by name.

    Tries exact match first (case-insensitive), then partial/substring match.
    Returns None if not found.
    """
    path = CATALOG_PATH if CATALOG_PATH.exists() else SOURCE_CATALOG_PATH
    raw_items = _load_raw_catalog(path)
    query = name.strip().lower()

    for item in raw_items:
        if (item.get("name") or "").strip().lower() == query:
            return item

    for item in raw_items:
        item_name = (item.get("name") or "").strip().lower()
        if query in item_name or item_name in query:
            return item

    return None
