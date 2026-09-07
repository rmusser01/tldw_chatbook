from __future__ import annotations

# Stdlib ElementTree is imported for document BUILDING only (`export`
# below): `Element`/`SubElement` have no defusedxml counterparts, and
# serializing a tree we constructed ourselves has no attacker-controlled
# input. Every PARSE of foreign text goes through `_safe_fromstring`.
import xml.etree.ElementTree as ET
from typing import Any

from loguru import logger

try:
    # TASK-19558: this module was the one XML parser in `Subscriptions/`
    # that never adopted defusedxml, while every sibling
    # (`security.py`, `monitoring_engine.py`, the three scrapers) uses this
    # exact try/except shape.
    #
    # NOT routed through `Utils/optional_deps.py` -- reviewed and declined
    # in review round 2 (Qodo #3). `defusedxml` is a CORE dependency
    # (`pyproject.toml` `[project] dependencies`, annotated "engine xml
    # security parsing (Q9: core)"), not an extra, and `optional_deps`
    # names it only inside the `ebook` extra's aggregate availability
    # check -- it publishes no import accessor for it. All six existing
    # defusedxml call sites in this app import it directly under exactly
    # this try/except; routing one of them through `get_safe_import`
    # (whose users are torch/transformers/aiohttp -- genuinely optional)
    # would make this the odd one out, not the consistent one. The
    # `except ImportError` arm is a belt-and-braces fallback for a broken
    # install, not an optional-feature gate.
    #
    # The concrete exposure measured for OPML is
    # ENTITY EXPANSION ("billion laughs"), not XXE: stdlib ElementTree
    # already ignores external entity references, but it happily expands
    # internal ones, so a ~1KB OPML file imported through Watchlists ▸
    # Import OPML expands to gigabytes inside the parser. defusedxml
    # raises `EntitiesForbidden` on the DTD instead.
    from defusedxml.ElementTree import fromstring as _safe_fromstring

    DEFUSEDXML_AVAILABLE = True
except ImportError:  # pragma: no cover - defusedxml is a core dependency
    from xml.etree.ElementTree import fromstring as _safe_fromstring

    DEFUSEDXML_AVAILABLE = False
    logger.warning(
        "defusedxml not available, using standard xml.etree for OPML import. "
        "Install defusedxml for better security."
    )


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

        Args:
            xml_text: Raw OPML XML text.

        Returns:
            Source-create payloads carrying ``name``, ``url``,
            ``source_type``, and the nearest ``folder`` name or ``None``.

        Raises:
            xml.etree.ElementTree.ParseError: If ``xml_text`` is malformed.
            defusedxml.common.EntitiesForbidden: If ``xml_text`` declares
                internal entities (the billion-laughs shape). Subclasses
                ``ValueError``; the caller's handler treats it the same way
                it treats a malformed document.
        """
        root = _safe_fromstring(xml_text)
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

    def export(
        self,
        watchlists: list[dict[str, Any]],
        unassigned: list[dict[str, Any]],
    ) -> str:
        """Serialize watchlist structure to OPML 2.0 XML (ADR-043 rule 5).

        One folder outline per watchlist (ordered case-insensitively by
        name) with its member feeds nested (same ordering); a source in
        several watchlists appears under EACH -- membership is many-to-many
        and the document says so faithfully. Unassigned feeds follow as
        flat top-level outlines, so a profile with no watchlists exports
        exactly the flat document the pre-mapping exporter produced.
        ElementTree escapes every attribute on serialize; parse() surfaces
        them as literal strings on the way back in.

        Args:
            watchlists: One dict per watchlist: ``name`` plus ``sources``,
                a list of feed dicts carrying ``name``/``url``/
                ``source_type``.
            unassigned: Feed dicts (same shape) belonging to no watchlist.

        Returns:
            The OPML document as a string.
        """
        root = ET.Element("opml", {"version": "2.0"})
        body = ET.SubElement(root, "body")

        def feed(parent: ET.Element, source: dict[str, Any]) -> None:
            ET.SubElement(parent, "outline", {
                "text": str(source.get("name") or "Untitled"),
                "title": str(source.get("name") or "Untitled"),
                "type": str(source.get("source_type") or "rss"),
                "xmlUrl": str(source.get("url") or ""),
            })

        def by_name(row: dict[str, Any]) -> str:
            return str(row.get("name") or "").lower()

        for watchlist in sorted(watchlists, key=by_name):
            name = str(watchlist.get("name") or "Untitled")
            folder = ET.SubElement(body, "outline", {"text": name, "title": name})
            for source in sorted(watchlist.get("sources") or [], key=by_name):
                feed(folder, source)
        for source in sorted(unassigned, key=by_name):
            feed(body, source)
        return ET.tostring(root, encoding="unicode")
