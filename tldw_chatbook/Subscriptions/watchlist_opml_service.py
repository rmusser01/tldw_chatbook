from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any


class WatchlistOpmlService:
    """Minimal OPML import/export for watchlist sources."""

    def parse(self, xml_text: str) -> list[dict[str, Any]]:
        """Parse OPML XML into source create payloads."""
        root = ET.fromstring(xml_text)
        items: list[dict[str, Any]] = []
        for outline in root.iter("outline"):
            url = outline.get("xmlUrl") or outline.get("htmlUrl")
            if not url:
                continue
            source_type = outline.get("type", "rss").lower()
            if source_type not in {"rss", "site", "forum"}:
                source_type = "rss"
            items.append({
                "name": outline.get("text") or outline.get("title") or "Untitled",
                "url": url,
                "source_type": source_type,
            })
        return items

    def export(self, sources: list[dict[str, Any]]) -> str:
        """Serialize source dicts to OPML 2.0 XML."""
        root = ET.Element("opml", {"version": "2.0"})
        body = ET.SubElement(root, "body")
        for source in sources:
            outline = ET.SubElement(body, "outline", {
                "text": str(source.get("name") or "Untitled"),
                "title": str(source.get("name") or "Untitled"),
                "type": str(source.get("source_type") or "rss"),
                "xmlUrl": str(source.get("url") or ""),
            })
        return ET.tostring(root, encoding="unicode")
