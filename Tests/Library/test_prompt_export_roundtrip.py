"""Round-trip test for the Library prompt Markdown exporter (Task 5).

``render_prompt_markdown`` must emit EXACTLY the custom ``### SECTION ###``
grammar ``Prompts_Interop.parse_markdown_prompts_from_content`` reads, so
export -> import reproduces a prompt's fields unchanged. The first test
below (adjusted only to use the parser's real output keys) is the brief's
stated acceptance criterion.

Two "known limitation" characterization tests at the bottom pin ACTUAL
(not idealized) parser behavior discovered while writing this exporter:
``parse_markdown_prompts_from_content``'s generic per-section regex (used
for AUTHOR/SYSTEM/USER/KEYWORDS, unlike the TITLE block's own separate
regex which handles ``details`` correctly) has two independent pre-existing
bugs --

1. Its trailing ``\\s*\\n`` (between a section's closing ``###`` and the
   captured value) is greedy and backtrack-swallows a blank value line's
   own newline whenever another section follows, so a BLANK
   AUTHOR/SYSTEM/USER value bleeds into capturing the literal text of the
   NEXT section's header instead of parsing back as ``None``.
2. Its capture uses a bare ``$`` (under ``re.MULTILINE``) as one of two
   "stop here" lookahead alternatives, and ``$`` matches before EVERY
   newline in multiline mode (not just end-of-string) -- so a multi-line
   SYSTEM/USER value is truncated after its first line, every time,
   regardless of what (if anything) follows it in the file.

Both are properties of ``Prompts_Interop.py`` (not modified by Task 5 --
see its file list), not of this renderer or its callers. They are pinned
here as characterization tests (not requirements) so a future fix to the
parser is validated against real, previously-observed behavior, and so
this discovery is not silently lost. Multi-line ``details`` is UNAFFECTED
(the TITLE block's own regex looks for a literal ``### AUTHOR ###``/``\\Z``
terminator, not a generic ``$``), so it round-trips correctly and is
covered as a real (non-characterization) test.
"""

from __future__ import annotations

from tldw_chatbook.Prompt_Management.prompt_markdown_export import render_prompt_markdown
from tldw_chatbook.Prompt_Management.Prompts_Interop import parse_markdown_prompts_from_content


def test_prompt_markdown_export_roundtrips():
    """The brief's stated acceptance criterion (key names adjusted to the
    parser's real output)."""
    detail = {
        "name": "Release note",
        "author": "me",
        "details": "d",
        "system_prompt": "sys text",
        "user_prompt": "user text",
        "keywords": ["release", "notes"],
    }
    text = render_prompt_markdown(detail)
    parsed = parse_markdown_prompts_from_content(text)
    assert len(parsed) == 1
    p = parsed[0]
    # The parser's real output keys (post ``_normalize_prompt_data``):
    # name/author/details/system_prompt/user_prompt/keywords (+
    # prompt_format/prompt_schema_version/prompt_definition, unused here).
    assert (p["name"], p["system_prompt"], p["user_prompt"]) == (
        "Release note",
        "sys text",
        "user text",
    )
    # Author/details/keywords round-trip too -- the parser DOES carry all
    # three (all single-line/non-blank here, which round-trips correctly;
    # see the module docstring above for the blank/multi-line exceptions).
    assert p["author"] == "me"
    assert p["details"] == "d"
    assert p["keywords"] == ["release", "notes"]


def test_prompt_markdown_export_roundtrips_multiline_details():
    """``details`` (derived from the TITLE block's own regex, which looks
    for a literal ``### AUTHOR ###``/end-of-string terminator rather than a
    generic ``$``) correctly round-trips multi-line content -- unlike
    SYSTEM/USER, see the "known limitation" test below."""
    detail = {
        "name": "Multi-line details prompt",
        "author": "Alice",
        "details": "Line one\nLine two\nLine three",
        "system_prompt": "sys text",
        "user_prompt": "user text",
        "keywords": ["a", "b"],
    }
    text = render_prompt_markdown(detail)
    parsed = parse_markdown_prompts_from_content(text)
    assert len(parsed) == 1
    p = parsed[0]
    assert p["name"] == "Multi-line details prompt"
    assert p["details"] == "Line one\nLine two\nLine three"
    assert p["system_prompt"] == "sys text"
    assert p["user_prompt"] == "user text"
    assert p["keywords"] == ["a", "b"]


def test_prompt_markdown_export_accepts_keywords_as_csv_string():
    """``keywords`` may already be a comma-separated string (the prompt
    editor's live ``#library-prompt-keywords`` Input value shape), not just
    a list."""
    detail = {
        "name": "CSV keywords prompt",
        "author": "Bob",
        "details": "",
        "system_prompt": "sys",
        "user_prompt": "usr",
        "keywords": "release, notes",
    }
    text = render_prompt_markdown(detail)
    parsed = parse_markdown_prompts_from_content(text)
    assert len(parsed) == 1
    assert parsed[0]["keywords"] == ["release", "notes"]


def test_prompt_markdown_export_documents_known_parser_limitation_blank_field():
    """CHARACTERIZATION TEST (not a requirement): pins the parser's ACTUAL
    behavior for a blank AUTHOR value followed by more sections -- it
    bleeds into the next section's header text instead of parsing back as
    ``None``. See the module docstring's bug (1) for the root cause. Other
    fields (details/system_prompt/user_prompt/keywords) are unaffected,
    since each section is extracted by an independent ``re.search`` over
    the whole content.
    """
    detail = {
        "name": "No author prompt",
        "author": "",
        "details": "some details",
        "system_prompt": "sys text",
        "user_prompt": "user text",
        "keywords": [],
    }
    text = render_prompt_markdown(detail)
    parsed = parse_markdown_prompts_from_content(text)
    assert len(parsed) == 1
    p = parsed[0]
    assert p["name"] == "No author prompt"
    assert p["details"] == "some details"
    assert p["system_prompt"] == "sys text"
    assert p["user_prompt"] == "user text"
    # BUG (pre-existing, in Prompts_Interop.py, not this renderer): a blank
    # AUTHOR value followed by another section parses back as that next
    # section's literal header text, not "" or None.
    assert p["author"] == "### SYSTEM ###"


def test_prompt_markdown_export_documents_known_parser_limitation_multiline_system():
    """CHARACTERIZATION TEST (not a requirement): pins the parser's ACTUAL
    behavior for a multi-line SYSTEM value -- it is truncated after the
    first line, regardless of what follows. See the module docstring's bug
    (2) for the root cause."""
    detail = {
        "name": "Multi-line system prompt",
        "author": "Alice",
        "details": "d",
        "system_prompt": "System line one\nSystem line two",
        "user_prompt": "User line one\nUser line two",
        "keywords": [],
    }
    text = render_prompt_markdown(detail)
    parsed = parse_markdown_prompts_from_content(text)
    assert len(parsed) == 1
    p = parsed[0]
    assert p["name"] == "Multi-line system prompt"
    # BUG (pre-existing, in Prompts_Interop.py, not this renderer):
    # truncated after the first line instead of keeping both.
    assert p["system_prompt"] == "System line one"
    assert p["user_prompt"] == "User line one"
