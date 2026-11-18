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

import json
import re

import pytest

from tldw_chatbook.Prompt_Management.prompt_markdown_export import (
    render_prompt_markdown,
)
from tldw_chatbook.Prompt_Management.Prompts_Interop import (
    parse_markdown_prompts_from_content,
)


def _structured_detail(*, artifact_type: str = "prompt") -> dict[str, object]:
    """Build a valid v2 artifact whose content exercises Markdown boundaries."""
    kind = "block_prompt" if artifact_type == "prompt" else "block_recipe"
    definition = {
        "kind": kind,
        "schema_version": 2,
        "lanes": [
            {
                "id": "system",
                "blocks": [
                    {
                        "id": "role",
                        "title": "Role",
                        "syntax": "freeform",
                        "content": "You are precise.",
                        "mapping_hint": "Keep the system role.",
                    },
                    {
                        "id": "constraints",
                        "title": "Constraints",
                        "syntax": "xml",
                        "xml_tag": "constraints",
                        "content": "Keep IDs stable.",
                    },
                ],
            },
            {
                "id": "user",
                "blocks": [
                    {
                        "id": "request",
                        "title": "Request",
                        "syntax": "markdown",
                        "content": "Include all details.",
                        "mapping_hint": "The user's goal.",
                    },
                    {
                        "id": "context",
                        "title": "Context",
                        "syntax": "freeform",
                        "content": "Section-like text: ### SYSTEM ###\nUnicode: Δ",
                    },
                ],
            },
        ],
    }
    return {
        "name": f"{artifact_type.title()} artifact",
        "author": "Compatibility tester",
        "details": "Preserve the complete artifact definition.",
        "system_prompt": "You are precise.\n\n<constraints>Keep IDs stable.</constraints>",
        "user_prompt": "# Request\n\nInclude all details.\n\nSection-like text: ### SYSTEM ###\nUnicode: Δ",
        "keywords": ["structured", artifact_type],
        "artifact_type": artifact_type,
        "prompt_format": "structured",
        "prompt_schema_version": 2,
        "prompt_definition": definition,
    }


def _legacy_reader_sections(content: str) -> dict[str, str]:
    """Characterize the prior reader: it knows only the classic sections."""
    values: dict[str, str] = {}
    for section in ("TITLE", "SYSTEM", "USER", "KEYWORDS"):
        match = re.search(
            rf"^[ \t]*###[ \t]*{section}[ \t]*###[ \t]*\r?\n"
            r"(.*?)(?=\r?\n[ \t]*###[ \t]*[A-Za-z][A-Za-z0-9_]*[ \t]*###[ \t]*(?:\r?\n|\Z)|\Z)",
            content,
            re.MULTILINE | re.DOTALL,
        )
        if match:
            values[section] = match.group(1).strip()
    return values


def _markdown_with_structure(
    structure: str, *, artifact_type: str = "prompt", name: str = "Foreign artifact"
) -> str:
    """Build a classic Markdown artifact with a deliberately supplied structure."""
    return (
        f"### TITLE ###\n{name}\n### AUTHOR ###\nImporter\n"
        "### SYSTEM ###\ncompiled system\n### USER ###\ncompiled user\n"
        "### KEYWORDS ###\ncompatibility\n"
        f"### ARTIFACT_TYPE ###\n{artifact_type}\n"
        f"### STRUCTURE ###\n```json\n{structure}\n```\n"
    )


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
    assert (
        p["system_prompt"] == "System line one\n\nSystem line two\n\nSystem line three"
    )
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


def test_prompt_markdown_export_keeps_legacy_output_byte_compatible():
    """Unstructured records retain the original human-readable bytes exactly."""
    detail = {
        "name": "Legacy export",
        "author": "Author",
        "details": "Details",
        "system_prompt": "System",
        "user_prompt": "User",
        "keywords": ["one", "two"],
    }

    assert render_prompt_markdown(detail) == (
        "### TITLE ###\nLegacy export\nDetails\n### AUTHOR ###\nAuthor\n"
        "### SYSTEM ###\nSystem\n### USER ###\nUser\n"
        "### KEYWORDS ###\none, two\n"
    )


@pytest.mark.parametrize("artifact_type", ["prompt", "recipe"])
def test_block_markdown_round_trip_preserves_definition(artifact_type: str):
    """Known Console v2 artifacts restore every canonical definition field."""
    detail = _structured_detail(artifact_type=artifact_type)

    markdown = render_prompt_markdown(detail)
    [imported] = parse_markdown_prompts_from_content(markdown)

    expected_structure = json.dumps(
        detail["prompt_definition"],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert markdown.endswith(
        f"\n### ARTIFACT_TYPE ###\n{artifact_type}\n"
        f"\n### STRUCTURE ###\n```json\n{expected_structure}\n```\n"
    )
    assert imported["artifact_type"] == artifact_type
    assert imported["prompt_format"] == "structured"
    assert imported["prompt_schema_version"] == 2
    assert imported["prompt_definition"] == detail["prompt_definition"]


def test_structured_markdown_keeps_section_like_json_text_inside_structure():
    """The next-section terminator does not split escaped JSON string content."""
    detail = _structured_detail()

    [imported] = parse_markdown_prompts_from_content(render_prompt_markdown(detail))

    context_block = imported["prompt_definition"]["lanes"][1]["blocks"][1]
    assert context_block["content"] == "Section-like text: ### SYSTEM ###\nUnicode: Δ"


def test_structured_export_remains_readable_to_the_prior_section_reader():
    """Older readers ignore appended sections while retaining classic fields."""
    detail = _structured_detail()

    sections = _legacy_reader_sections(render_prompt_markdown(detail))

    assert sections == {
        "TITLE": "Prompt artifact\nPreserve the complete artifact definition.",
        "SYSTEM": detail["system_prompt"],
        "USER": detail["user_prompt"],
        "KEYWORDS": "structured, prompt",
    }


@pytest.mark.parametrize(
    ("artifact_type", "structure"),
    [
        ("prompt", '{"kind":'),
        (
            "prompt",
            json.dumps(
                {
                    "kind": "block_recipe",
                    "schema_version": 2,
                    "lanes": [
                        {"id": "system", "blocks": []},
                        {"id": "user", "blocks": []},
                    ],
                }
            ),
        ),
        ("recipe", json.dumps({"schema_version": 1, "messages": []})),
        (
            "recipe",
            json.dumps({"definition_kind": "single_text_recipe", "schema_version": 2}),
        ),
        ("prompt", json.dumps({"kind": "future_prompt", "schema_version": 3})),
    ],
    ids=[
        "malformed-json",
        "discriminator-mismatch",
        "foreign-v1",
        "single-text-recipe",
        "future-version",
    ],
)
def test_non_console_structure_falls_back_to_a_legacy_prompt(
    artifact_type: str, structure: str
):
    """Foreign or invalid structure never leaks through a legacy import."""
    [imported] = parse_markdown_prompts_from_content(
        _markdown_with_structure(structure, artifact_type=artifact_type)
    )

    assert imported["artifact_type"] == "prompt"
    assert imported["prompt_format"] == "legacy"
    assert imported["prompt_schema_version"] is None
    assert imported["prompt_definition"] is None
    assert imported["system_prompt"] == "compiled system"
    assert imported["user_prompt"] == "compiled user"


def _mixin_detail():
    body = "\n{style}\n### TITLE ###\nsecret\n```\n"
    return {
        "name": "Literal\n### TITLE ###\nname",
        "author": None,
        "details": None,
        "system_prompt": "",
        "user_prompt": body,
        "keywords": [],
        "artifact_type": "mixin",
        "prompt_format": "structured",
        "prompt_schema_version": 3,
        "prompt_definition": json.dumps(
            {
                "kind": "text_mixin",
                "schema_version": 3,
                "body": body,
                "placement": "before",
            }
        ),
    }


def test_mixin_markdown_exact_roundtrip_and_frozen_legacy_reader_refusal():
    detail = _mixin_detail()
    markdown = render_prompt_markdown(detail)
    assert markdown.startswith("# Chatbook mix-in\n\n```json\n")
    assert len(markdown.splitlines()) == 5
    assert _legacy_reader_sections(markdown) == {}
    assert parse_markdown_prompts_from_content(markdown) == [detail]


@pytest.mark.parametrize(
    "mutation", ["bad-json", "future", "trailing-legacy", "missing-fence"]
)
def test_mixin_markdown_malformed_refuses_without_fallback_or_body_logs(mutation):
    from loguru import logger

    from tldw_chatbook.Prompt_Management.prompt_chatbook_record import (
        PromptChatbookRecordError,
    )

    markdown = render_prompt_markdown(_mixin_detail())
    if mutation == "bad-json":
        markdown = "# Chatbook mix-in\n\n```json\nsecret invalid\n```\n"
    elif mutation == "future":
        record = json.loads(markdown.splitlines()[3])
        record["record_version"] = 999
        markdown = "# Chatbook mix-in\n\n```json\n" + json.dumps(record) + "\n```\n"
    elif mutation == "trailing-legacy":
        markdown += "### TITLE ###\nsecret\n### USER ###\nsecret\n"
    else:
        markdown = markdown.rsplit("```", 1)[0]
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)))
    try:
        with pytest.raises(PromptChatbookRecordError) as error:
            parse_markdown_prompts_from_content(markdown)
    finally:
        logger.remove(sink)
    assert "secret" not in str(error.value)
    assert "secret" not in "".join(messages)


def test_mixin_markdown_old_section_metadata_cannot_flatten_to_prompt():
    from tldw_chatbook.Prompt_Management.prompt_chatbook_record import (
        PromptChatbookRecordError,
    )

    with pytest.raises(PromptChatbookRecordError):
        parse_markdown_prompts_from_content(
            _markdown_with_structure(
                '{"kind":"text_mixin","schema_version":3,"body":"secret","placement":"before"}',
                artifact_type="mixin",
            )
        )


def test_mixin_markdown_import_failure_does_not_log_body_or_exception(
    tmp_path, monkeypatch
):
    from loguru import logger

    import tldw_chatbook.Prompt_Management.Prompts_Interop as interop

    file = tmp_path / "mixin.md"
    file.write_text(render_prompt_markdown(_mixin_detail()))
    monkeypatch.setattr(interop, "is_initialized", lambda: True)

    def fail(**kwargs):
        raise ValueError("secret instruction body")

    monkeypatch.setattr(interop, "add_or_update_prompt_interop", fail)
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)))
    try:
        results = interop.import_prompts_from_files(file, base_directory=str(tmp_path))
    finally:
        logger.remove(sink)
    assert results[0]["status"] == "failure"
    assert "secret" not in results[0]["message"]
    assert "secret" not in "".join(messages)


def test_mixin_markdown_deeply_nested_json_refuses_with_bounded_error():
    from tldw_chatbook.Prompt_Management.prompt_chatbook_record import (
        PromptChatbookRecordError,
    )

    nested = '{"secret":' * 10000 + "0" + "}" * 10000
    with pytest.raises(PromptChatbookRecordError):
        parse_markdown_prompts_from_content(
            "# Chatbook mix-in\n\n```json\n" + nested + "\n```\n"
        )


def _bom_mixin_markdown(mutation):
    detail = _mixin_detail()
    body = "\ufeffbom_literal_body_must_not_leak\n{style}\n### TITLE ###\n"
    detail["name"] = "BOM literal"
    detail["user_prompt"] = body
    detail["prompt_definition"] = json.dumps(
        {"kind": "text_mixin", "schema_version": 3, "body": body, "placement": "before"}
    )
    markdown = render_prompt_markdown(detail)
    if mutation == "bad-json":
        markdown = "# Chatbook mix-in\n\n```json\nbom_literal_body_must_not_leak invalid\n```\n"
    elif mutation == "missing-fence":
        markdown = markdown.rsplit("```", 1)[0]
    elif mutation == "trailing-legacy":
        markdown += "### TITLE ###\nUnexpected legacy Prompt\n### USER ###\nbom_literal_body_must_not_leak\n"
    return "\ufeff" + markdown, detail


@pytest.mark.parametrize(
    "mutation", ["valid", "bad-json", "missing-fence", "trailing-legacy"]
)
def test_bom_mixin_markdown_never_falls_back_or_logs_literal_data(mutation):
    from loguru import logger

    from tldw_chatbook.Prompt_Management.prompt_chatbook_record import (
        PromptChatbookRecordError,
    )

    markdown, detail = _bom_mixin_markdown(mutation)
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)))
    try:
        if mutation == "valid":
            assert parse_markdown_prompts_from_content(markdown) == [detail]
            assert _legacy_reader_sections(markdown) == {}
        else:
            with pytest.raises(PromptChatbookRecordError) as error:
                parse_markdown_prompts_from_content(markdown)
            assert str(error.value) == "Invalid Chatbook Prompt record."
    finally:
        logger.remove(sink)
    assert "bom_literal_body_must_not_leak" not in "".join(messages)
    assert "Content snippet:" not in "".join(messages)
    assert "Parsed custom MD/TXT prompt:" not in "".join(messages)


@pytest.mark.parametrize(
    "mutation", ["valid", "bad-json", "missing-fence", "trailing-legacy"]
)
def test_bom_mixin_utf8_file_import_preserves_or_refuses_before_mutation(
    tmp_path, mutation
):
    from loguru import logger

    import tldw_chatbook.Prompt_Management.Prompts_Interop as interop

    markdown, detail = _bom_mixin_markdown(mutation)
    file = tmp_path / "bom-mixin.md"
    file.write_text(markdown, encoding="utf-8")
    assert file.read_bytes().startswith(b"\xef\xbb\xbf")
    interop.initialize_interop(tmp_path / "bom.db", client_id="bom-import")
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)))
    try:
        results = interop.import_prompts_from_files(file, base_directory=str(tmp_path))
        if mutation == "valid":
            assert len(results) == 1 and results[0]["status"] == "success"
            restored = interop.get_prompt_by_uuid(results[0]["prompt_uuid"])
            for field in ("artifact_type", "user_prompt", "prompt_definition"):
                assert restored[field] == detail[field]
        else:
            assert len(results) == 1 and results[0]["status"] == "failure"
            assert (
                results[0]["message"]
                == "Parsing error: Invalid Chatbook Prompt record."
            )
            assert interop.get_db_instance().get_all_active_prompt_ids() == []
    finally:
        logger.remove(sink)
        interop.shutdown_interop()
    assert "bom_literal_body_must_not_leak" not in "".join(messages)
    assert "Content snippet:" not in "".join(messages)
    assert "Parsed custom MD/TXT prompt:" not in "".join(messages)
