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
SUBSCRIPTIONS_ROOT = PACKAGE_ROOT / "Subscriptions"

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


def test_no_subscriptions_module_parses_xml_without_defusedxml() -> None:
    """Scoped to `Subscriptions/`, where the rule is fully satisfied today.

    Stated rather than overclaimed: a repo-wide version of this census
    currently reports seven other modules (`Evals/eval_runner.py`,
    `Local_Ingestion/XML_Ingestion.py`,
    `Media/local_media_reading_service.py`,
    `Research_Interop/academic_providers.py`, `Utils/file_extraction.py`,
    `Web_Scraping/Article_Extractor_Lib.py`,
    `Web_Scraping/Article_Scraper/crawler.py`) that parse XML with stdlib
    ElementTree and never import defusedxml. Those are a real, separate
    population -- outside this task's scope, and NOT allowlisted here,
    because an allowlist would quietly bless them. Widening the census to
    the package tree is what turns them red.
    """
    offenders: dict[str, list[tuple[int, str]]] = {}
    for path in sorted(SUBSCRIPTIONS_ROOT.rglob("*.py")):
        hits = unhardened_xml_parsers(path.read_text(encoding="utf-8"))
        if hits:
            offenders[str(path.relative_to(PACKAGE_ROOT.parent))] = hits
    assert offenders == {}, (
        "These Subscriptions modules parse XML without consulting "
        f"defusedxml: {offenders}"
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
