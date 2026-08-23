"""TASK-19558: the OPML importer parses with defusedxml, like every sibling.

`Subscriptions/watchlist_opml_service.py` was the one XML parser in the
`Subscriptions/` package that imported stdlib `xml.etree.ElementTree`
directly, while `security.py`, `monitoring_engine.py` and the three scrapers
all use the defusedxml-with-fallback shape. `defusedxml` is a core
dependency (`pyproject.toml`, "engine xml security parsing (Q9: core)"), so
this was a missed adoption rather than a dependency question.

**The exposure, measured rather than assumed.** It is ENTITY EXPANSION, not
XXE. Python's stdlib ElementTree has ignored external entity references for
years -- `test_stdlib_elementtree_still_ignores_external_entities` re-proves
that here rather than taking it on trust -- but it happily expands INTERNAL
ones, which is the billion-laughs shape: a small OPML file whose DTD nests
entities expands to gigabytes inside the parser before a single outline is
seen. Watchlists ▸ Import OPML feeds a user-chosen (or shared) file straight
into `parse()`, so the payload does not have to come from the person running
the app.

**The assertion is on the refusal, not on a timeout.** A test that measures
"this took too long" is a flake generator and proves nothing about the fix
(a faster machine passes it either way). The payload below is bounded
(6 levels, ~10^6 expansion -- large enough to be an unmistakable defect if
expanded, small enough to be harmless if it ever were), and the test asserts
that defusedxml REFUSES the document by raising `EntitiesForbidden` at the
DTD, before expansion is attempted at all.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tldw_chatbook.Subscriptions.watchlist_opml_service import (
    DEFUSEDXML_AVAILABLE,
    WatchlistOpmlService,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "tldw_chatbook"

#: Modules that parse XML with stdlib ElementTree and do NOT consult
#: defusedxml, recorded with what reaches them. This is a REGISTER OF OPEN
#: DEFECTS, not an allowlist of acceptable ones: every entry below is a live
#: entity-expansion exposure of the same shape as the OPML importer this
#: task fixed, and each is filed for its own change rather than fixed here
#: (TASK-19558 was scoped to the seams the holistic review named).
#:
#: The census is repo-wide precisely so that an EIGHTH such module cannot be
#: added silently -- it fails immediately, and the author must either harden
#: it or write down here what reaches it and why it is being left open.
#: `test_known_unhardened_entries_are_still_unhardened` deletes the register
#: as each one is fixed, so it cannot rot into folklore.
_KNOWN_UNHARDENED: dict[str, str] = {
    "Evals/eval_runner.py": (
        "Parses MODEL OUTPUT (`ET.fromstring` over a generated XML answer), "
        "so the payload is prompt-injection reachable -- the sharpest of "
        "the seven: a poisoned document in the corpus can choose the XML "
        "the model emits."
    ),
    "Local_Ingestion/XML_Ingestion.py": (
        "Parses user-supplied .xml files chosen for ingestion -- the same "
        "threat shape as the OPML importer fixed by TASK-19558 (a file the "
        "user did not write, imported on their behalf)."
    ),
    "Media/local_media_reading_service.py": (
        "`ET.iterparse` over stored media documents. iterparse streams "
        "elements but still expands internal entities, so a byte cap on the "
        "source does not bound the expansion."
    ),
    "Research_Interop/academic_providers.py": (
        "Parses a REMOTE Atom feed (arXiv) -- fully attacker-controlled "
        "bytes if the endpoint or the transport is."
    ),
    "Utils/file_extraction.py": (
        "Parses XML pulled out of user-supplied archives/documents during "
        "extraction."
    ),
    "Web_Scraping/Article_Extractor_Lib.py": (
        "Parses FETCHED sitemaps. A response size cap does not help here: "
        "amplification is the whole point of a billion-laughs payload -- a "
        "few hundred bytes on the wire become gigabytes in the parser."
    ),
    "Web_Scraping/Article_Scraper/crawler.py": (
        "Parses FETCHED sitemaps, same as `Article_Extractor_Lib` above."
    ),
}

#: Six nesting levels of a 10x entity: ~1,000,000 "lol" copies if expanded.
BILLION_LAUGHS_OPML = """<?xml version="1.0"?>
<!DOCTYPE opml [
  <!ENTITY lol "lol">
  <!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
  <!ENTITY lol2 "&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;">
  <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
  <!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;">
  <!ENTITY lol5 "&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;">
]>
<opml version="2.0">
  <body>
    <outline text="&lol5;" xmlUrl="https://example.invalid/feed.xml"/>
  </body>
</opml>
"""

BENIGN_OPML = """<?xml version="1.0"?>
<opml version="2.0">
  <body>
    <outline text="News">
      <outline text="Example" type="rss" xmlUrl="https://example.invalid/feed.xml"/>
    </outline>
  </body>
</opml>
"""


def test_defusedxml_is_the_parser_actually_in_use() -> None:
    """The fallback branch exists for a missing optional dep; it is not one."""
    assert DEFUSEDXML_AVAILABLE, (
        "defusedxml is a core dependency; the OPML importer fell back to "
        "stdlib ElementTree, which expands internal entities"
    )


def test_entity_expansion_opml_is_refused_at_the_dtd() -> None:
    from defusedxml.common import EntitiesForbidden

    with pytest.raises(EntitiesForbidden):
        WatchlistOpmlService().parse(BILLION_LAUGHS_OPML)


def test_the_refusal_subclasses_valueerror_so_import_handlers_catch_it() -> None:
    """Callers already handle a malformed document; this is the same shape."""
    from defusedxml.common import EntitiesForbidden

    assert issubclass(EntitiesForbidden, ValueError)


def test_stdlib_elementtree_would_have_expanded_the_same_payload() -> None:
    """The defect, demonstrated on the primitive the module used to import.

    Bounded on purpose: this parses the payload with stdlib ElementTree and
    shows the outline text is ~10^6 characters long -- i.e. the parser
    materialized the expansion. Nothing here is timed.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(BILLION_LAUGHS_OPML)
    outline = root.find("./body/outline")
    assert outline is not None
    expanded = outline.get("text") or ""
    assert len(expanded) >= 3 * 10**5, len(expanded)


def test_stdlib_elementtree_still_ignores_external_entities() -> None:
    """States the exposure honestly: this is NOT an XXE.

    A SYSTEM entity reference is left unresolved by stdlib ElementTree, so
    calling the pre-fix state an XXE would have been wrong.
    """
    import xml.etree.ElementTree as ET

    xxe = (
        '<?xml version="1.0"?>\n'
        '<!DOCTYPE opml [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>\n'
        '<opml version="2.0"><body>'
        '<outline text="&xxe;" xmlUrl="https://example.invalid/f.xml"/>'
        "</body></opml>"
    )
    try:
        root = ET.fromstring(xxe)
    except ET.ParseError:
        return  # refused outright: also not an XXE
    outline = root.find("./body/outline")
    assert outline is not None
    assert "root:" not in (outline.get("text") or "")


def test_ordinary_opml_still_parses_and_keeps_folder_structure() -> None:
    """The hardening did not change the parser's contract (ADR-043)."""
    items = WatchlistOpmlService().parse(BENIGN_OPML)
    assert items == [
        {
            "name": "Example",
            "url": "https://example.invalid/feed.xml",
            "source_type": "rss",
            "folder": "News",
        }
    ]


def test_malformed_opml_still_raises_parseerror() -> None:
    import xml.etree.ElementTree as ET

    with pytest.raises(ET.ParseError):
        WatchlistOpmlService().parse("<opml><body><outline")


# ---------------------------------------------------------------------------
# The guard: no NEW unhardened XML parser in `Subscriptions/`.
# ---------------------------------------------------------------------------

_PARSE_ENTRY_POINTS = frozenset(
    {"fromstring", "parse", "XML", "iterparse", "XMLParser"}
)


def unhardened_xml_parsers(source: str) -> list[tuple[int, str]]:
    """Calls to a stdlib-etree PARSE entry point in a module with no defusedxml.

    Deliberately AST-based (same reasoning as
    `Tests/Utils/test_egress_adoption_census.py`): a comment mentioning
    `ET.fromstring` produces no `Call` node, and a docstring mentioning
    `defusedxml` produces no `Import` node, so neither direction of the
    regex-over-text bypass is available.

    Document BUILDING (`Element`/`SubElement`/`tostring`) is not a parse and
    is not flagged -- `watchlist_opml_service.export()` legitimately keeps
    stdlib ElementTree for exactly that, because defusedxml has no
    counterpart for it and a tree we constructed ourselves has no
    attacker-controlled input.
    """
    tree = ast.parse(source)
    et_aliases: set[str] = set()
    parse_bindings: set[str] = set()
    has_defusedxml = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("defusedxml"):
                    has_defusedxml = True
                if alias.name in ("xml.etree.ElementTree", "xml.etree.cElementTree"):
                    et_aliases.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("defusedxml"):
                has_defusedxml = True
            if module in ("xml.etree.ElementTree", "xml.etree"):
                for alias in node.names:
                    if alias.name in _PARSE_ENTRY_POINTS:
                        parse_bindings.add(alias.asname or alias.name)
                    if alias.name == "ElementTree":
                        et_aliases.add(alias.asname or alias.name)
    if has_defusedxml:
        return []
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in et_aliases
            and func.attr in _PARSE_ENTRY_POINTS
        ):
            hits.append((node.lineno, f"{func.value.id}.{func.attr}"))
        elif isinstance(func, ast.Name) and func.id in parse_bindings:
            hits.append((node.lineno, func.id))
    return hits


def _unhardened_modules() -> dict[str, list[tuple[int, str]]]:
    """Every module under `tldw_chatbook/` that parses XML unhardened."""
    found: dict[str, list[tuple[int, str]]] = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        hits = unhardened_xml_parsers(path.read_text(encoding="utf-8", errors="replace"))
        if hits:
            found[path.relative_to(PACKAGE_ROOT).as_posix()] = hits
    return found


def test_no_module_parses_xml_without_defusedxml() -> None:
    """Repo-wide, so an EIGHTH unhardened parser cannot appear silently.

    The first version of this census was scoped to `Subscriptions/`, which
    made it green and inert: it could never red on anything, and the note
    saying the other seven were "deliberately not allowlisted so widening
    turns them red" described a widening that had not happened. The register
    above is the honest form -- the census covers the whole package, and the
    seven known-open modules are named with what reaches them.
    """
    offenders = {
        key: hits
        for key, hits in _unhardened_modules().items()
        if key not in _KNOWN_UNHARDENED
    }
    assert offenders == {}, (
        "These modules parse XML without consulting defusedxml: "
        f"{offenders} -- import `defusedxml.ElementTree` (a core dependency; "
        "see `watchlist_opml_service` for the shape), or add an entry to "
        "_KNOWN_UNHARDENED saying what reaches this parser and why it is "
        "being left open."
    )


def test_known_unhardened_entries_are_still_unhardened() -> None:
    """A register entry for a module that no longer needs it is stale.

    Without this, fixing one of the seven leaves a permanent hole in the
    census: the module would stay skipped, and a LATER regression in the
    same file would never red.
    """
    found = _unhardened_modules()
    for key in _KNOWN_UNHARDENED:
        path = PACKAGE_ROOT / key
        assert path.exists(), f"stale register entry for a deleted module: {key}"
        assert key in found, (
            f"stale register entry: {key} no longer parses XML unhardened -- "
            "drop the entry so the census covers it again"
        )


def test_the_register_matches_the_measured_population_exactly() -> None:
    """The register is a measurement, not a wish: no entry may be missing."""
    assert sorted(_unhardened_modules()) == sorted(_KNOWN_UNHARDENED)


def test_no_subscriptions_module_parses_xml_without_defusedxml() -> None:
    """The narrower claim TASK-19558 actually delivered, kept explicit.

    `Subscriptions/` is fully hardened -- the OPML importer was its last
    unhardened parser -- and no entry in the register lives there.
    """
    subscriptions = {
        key: hits
        for key, hits in _unhardened_modules().items()
        if key.startswith("Subscriptions/")
    }
    assert subscriptions == {}, (
        f"These Subscriptions modules parse XML without defusedxml: {subscriptions}"
    )


def test_the_xml_census_rediscovers_the_original_seam() -> None:
    """Bite-proof: the pre-fix module shape fails the census."""
    pre_fix = (
        "import xml.etree.ElementTree as ET\n"
        "\n"
        "def parse(xml_text):\n"
        "    return ET.fromstring(xml_text)\n"
    )
    assert unhardened_xml_parsers(pre_fix) == [(4, "ET.fromstring")]


def test_the_xml_census_ignores_document_building() -> None:
    """False-red direction: `export()`'s stdlib usage is not a parse."""
    builder = (
        "import xml.etree.ElementTree as ET\n"
        "\n"
        "def export():\n"
        "    root = ET.Element('opml')\n"
        "    ET.SubElement(root, 'body')\n"
        "    return ET.tostring(root, encoding='unicode')\n"
    )
    assert unhardened_xml_parsers(builder) == []


def test_the_xml_census_is_not_laundered_by_a_docstring() -> None:
    """False-green direction: prose naming defusedxml is not an import."""
    laundered = (
        '"""We should really use defusedxml here."""\n'
        "import xml.etree.ElementTree as ET\n"
        "\n"
        "def parse(xml_text):\n"
        "    # defusedxml would be better\n"
        "    return ET.fromstring(xml_text)\n"
    )
    assert unhardened_xml_parsers(laundered) == [(6, "ET.fromstring")]
