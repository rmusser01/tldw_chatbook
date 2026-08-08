from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any


class WatchlistOpmlService:
    """Minimal OPML import/export for watchlist sources."""

    def parse(self, xml_text: str) -> list[dict[str, Any]]:
        """Parse OPML XML into source create payloads, preserving folders.

        ADR-043. An outline with NO feed URL whose descendants include feeds
        is a FOLDER: feeds below it carry its name under `"folder"` (raw --
        case-insensitive reuse is the assignment layer's job, not the
        parser's). Nested folders flatten to the innermost name. An outline
        WITH a feed URL is a feed even when it has children; those children
        inherit its folder context. Top-level feeds carry `"folder": None`.
        A structural parser, not a sanitizer: names surface as literal
        strings (the C1-control test above pins that contract), and
        malformed XML raises `ET.ParseError` into the caller's handler.
        """
        root = ET.fromstring(xml_text)
        items: list[dict[str, Any]] = []

        def walk(element: ET.Element, folder: "str | None") -> None:
            for child in element:
                if child.tag != "outline":
                    # <body> and any other container: invisible to the
                    # mapping, but its descendants still count.
                    walk(child, folder)
                    continue
                url = child.get("xmlUrl") or child.get("htmlUrl")
                if url:
                    source_type = child.get("type", "rss").lower()
                    if source_type not in {"rss", "site", "forum"}:
                        source_type = "rss"
                    items.append({
                        "name": child.get("text") or child.get("title") or "Untitled",
                        "url": url,
                        "source_type": source_type,
                        "folder": folder,
                    })
                    # A feed's children inherit ITS context -- a feed is
                    # never a folder (ADR-043 rule 3).
                    walk(child, folder)
                else:
                    # Folder candidate: its name becomes the context for its
                    # descendants (the innermost name wins). A nameless one
                    # passes the current context through unchanged.
                    name = child.get("text") or child.get("title")
                    walk(child, name if name is not None else folder)

        walk(root, None)
        return items

    def export(self, sources: list[dict[str, Any]]) -> str:
        """Serialize source dicts to OPML 2.0 XML."""
        root = ET.Element("opml", {"version": "2.0"})
        body = ET.SubElement(root, "body")
        for source in sources:
            ET.SubElement(body, "outline", {
                "text": str(source.get("name") or "Untitled"),
                "title": str(source.get("name") or "Untitled"),
                "type": str(source.get("source_type") or "rss"),
                "xmlUrl": str(source.get("url") or ""),
            })
        return ET.tostring(root, encoding="unicode")
