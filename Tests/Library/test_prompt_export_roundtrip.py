"""Round-trip test for the Library prompt Markdown exporter (Task 5).

``render_prompt_markdown`` must emit EXACTLY the custom ``### SECTION ###``
grammar ``Prompts_Interop.parse_markdown_prompts_from_content`` reads, so
export -> import reproduces a prompt's fields unchanged. The first test
below (adjusted only to use the parser's real output keys) is the brief's
stated acceptance criterion.

``parse_markdown_prompts_from_content``'s generic per-section regex (used
for AUTHOR/SYSTEM/USER/KEYWORDS, unlike the TITLE block's own separate
regex which handles ``details`` correctly) USED to have two independent
pre-existing bugs, both now fixed directly on the parser (see
``Prompts_Interop.py``'s per-section ``pattern`` construction, fix wave 1
of Task 5's review) because this spec's lossless round-trip acceptance
criterion cannot be met while they stand:

1. Its capture used a bare ``$`` (under ``re.MULTILINE``) as one of two
   "stop here" lookahead alternatives, and ``$`` matches before EVERY
   newline in multiline mode (not just end-of-string) -- so a multi-line
   SYSTEM/USER value was truncated after its first line, every time,
   regardless of what (if anything) followed it in the file. FIXED: the
   terminator is now "the next actual ``### WORD ###`` header line, or
   true end-of-string (``\\Z``)" -- a body line that merely *contains*
   ``###`` mid-line (not as a whole header line) no longer terminates a
   section, and interior blank lines are preserved.
2. A blank AUTHOR/SYSTEM/USER value, when another section followed it (the
   common case), used to bleed into capturing the literal text of the
   NEXT section's header instead of parsing back as ``None`` -- the old
   pattern's trailing ``\\s*\\n`` (between a section's closing ``###`` and
   the captured value) was greedy and could backtrack-swallow the blank
   value line's own newline. FIXED as a side effect of the same
   terminator fix: a blank value's zero-length capture now correctly
   matches the "next header line" lookahead at the blank line's own
   boundary.

The two tests below that used to pin these as "known limitation"
characterization tests now pin the CORRECT (fixed) behavior instead --
see ``test_prompt_markdown_export_roundtrips_blank_author_field`` and
``test_prompt_markdown_export_roundtrips_multiline_system_and_user``.
Multi-line ``details`` was always unaffected by either bug (the TITLE
block's own regex looks for a literal ``### AUTHOR ###``/``\\Z``
terminator, not a generic ``$``), so it round-trips correctly and is
covered as its own test.
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
    # see the module docstring above for the blank/multi-line coverage).
    assert p["author"] == "me"
    assert p["details"] == "d"
    assert p["keywords"] == ["release", "notes"]


def test_prompt_markdown_export_roundtrips_multiline_details():
    """``details`` (derived from the TITLE block's own regex, which looks
    for a literal ``### AUTHOR ###``/end-of-string terminator rather than a
    generic ``$``) correctly round-trips multi-line content -- exactly
    like the (now-fixed) SYSTEM/USER handling below."""
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


def test_prompt_markdown_export_roundtrips_blank_author_field():
    """FIX (formerly a "known limitation" characterization test): a blank
    AUTHOR value followed by more sections must parse back as ``None``,
    never as the next section's literal header text. See the module
    docstring's bug (2) for the (fixed) root cause. Other fields
    (details/system_prompt/user_prompt/keywords) are unaffected, since each
    section is extracted by an independent ``re.search`` over the whole
    content.
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
    # FIXED (was: bled into "### SYSTEM ###", the next section's literal
    # header text). A blank value now parses back as None (the parser's
    # own zero-length-capture default), never as a header string.
    assert p["author"] is None


def test_prompt_markdown_export_roundtrips_multiline_system_and_user():
    """FIX (formerly a "known limitation" characterization test): a
    multi-line SYSTEM/USER value must be preserved in full -- including
    interior blank lines -- not truncated after its first line. See the
    module docstring's bug (1) for the (fixed) root cause."""
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
    # FIXED (was: truncated after the first line).
    assert p["system_prompt"] == "System line one\nSystem line two"
    assert p["user_prompt"] == "User line one\nUser line two"


def test_prompt_markdown_export_roundtrips_multiline_with_blank_interior_lines():
    """A multi-line SYSTEM/USER body may itself contain blank interior
    lines -- those must not be mistaken for a section boundary and must
    survive the round-trip (modulo the parser's own ``.strip()`` of
    leading/trailing whitespace on the whole captured section)."""
    detail = {
        "name": "Blank interior lines prompt",
        "author": "Alice",
        "details": "d",
        "system_prompt": "System line one\n\nSystem line two\n\nSystem line three",
        "user_prompt": "User line one\n\n\nUser line two",
        "keywords": ["x"],
    }
    text = render_prompt_markdown(detail)
    parsed = parse_markdown_prompts_from_content(text)
    assert len(parsed) == 1
    p = parsed[0]
    assert p["system_prompt"] == "System line one\n\nSystem line two\n\nSystem line three"
    assert p["user_prompt"] == "User line one\n\n\nUser line two"
    assert p["keywords"] == ["x"]


def test_prompt_markdown_export_roundtrips_body_line_containing_hash_markers():
    """A body line that merely CONTAINS the literal text ``###`` mid-line
    (not as a whole ``### WORD ###`` header line) must not be mistaken for
    a section boundary -- only lines that ARE a header terminate a
    section's capture."""
    detail = {
        "name": "Hash markers in body prompt",
        "author": "Alice",
        "details": "d",
        "system_prompt": "Some system text with ### markers mid-line\nand a second line",
        "user_prompt": "A line with ### inline\nUser line two",
        "keywords": [],
    }
    text = render_prompt_markdown(detail)
    parsed = parse_markdown_prompts_from_content(text)
    assert len(parsed) == 1
    p = parsed[0]
    assert (
        p["system_prompt"]
        == "Some system text with ### markers mid-line\nand a second line"
    )
    assert p["user_prompt"] == "A line with ### inline\nUser line two"
