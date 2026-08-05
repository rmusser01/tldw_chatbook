from hashlib import sha256

from tldw_chatbook.Prompt_Management.prompt_block_compiler import compile_block_artifact
from tldw_chatbook.Prompt_Management.prompt_legacy_decomposer import decompose_legacy_lanes


def test_decomposer_recognizes_top_level_markdown_and_nested_xml_only() -> None:
    system = "# Role\n\nBe helpful.\n\n<context><item>α</item></context>"
    user = "# Goal\n\nAnswer succinctly."

    result = decompose_legacy_lanes(system, user)
    system_blocks = result.definition.lanes[0].blocks
    user_blocks = result.definition.lanes[1].blocks

    assert [(block.syntax, block.title, block.content) for block in system_blocks] == [
        ("markdown", "Role", "Be helpful.\n\n"),
        ("xml", "context", "<item>α</item>"),
    ]
    assert [(block.syntax, block.title, block.content) for block in user_blocks] == [
        ("markdown", "Goal", "Answer succinctly."),
    ]
    assert result.system_origin.text == system
    assert result.system_origin.fingerprint == sha256(system.encode()).hexdigest()
    assert result.user_origin.text == user
    assert compile_block_artifact(result.definition) != (system, user)


def test_decomposer_preserves_content_before_a_whitespace_closing_xml_tag() -> None:
    result = decompose_legacy_lanes("<context>body</context >", "")

    [block] = result.definition.lanes[0].blocks

    assert (block.syntax, block.xml_tag, block.content) == ("xml", "context", "body")


def test_decomposer_keeps_fenced_and_incomplete_or_ambiguous_content_freeform() -> None:
    system = "```markdown\n# inside a fence\n```\n\n<open>unfinished"
    user = "prefix <context>not top-level</context>\n# heading without blank"

    result = decompose_legacy_lanes(system, user)

    assert [block.syntax for block in result.definition.lanes[0].blocks] == ["freeform"]
    assert result.definition.lanes[0].blocks[0].content == system
    assert [block.syntax for block in result.definition.lanes[1].blocks] == ["freeform"]
    assert result.definition.lanes[1].blocks[0].content == user


def test_decomposer_ignores_headings_inside_tilde_fences() -> None:
    system = "~~~markdown\n# hidden\n\nsecret\n~~~\n\n# Visible\n\nShown"

    result = decompose_legacy_lanes(system, "")

    blocks = result.definition.lanes[0].blocks
    assert [(block.syntax, block.content) for block in blocks] == [
        ("freeform", "~~~markdown\n# hidden\n\nsecret\n~~~\n\n"),
        ("markdown", "Shown"),
    ]


def test_decomposer_does_not_close_a_fence_with_trailing_prose() -> None:
    system = "```markdown\n``` prose\n# still hidden\n\nsecret\n```"

    result = decompose_legacy_lanes(system, "")

    assert [block.syntax for block in result.definition.lanes[0].blocks] == ["freeform"]
    assert result.definition.lanes[0].blocks[0].content == system


def test_decomposer_does_not_treat_mixed_fence_markers_as_a_fence() -> None:
    system = "`~`~\n# Visible\n\nShown"

    result = decompose_legacy_lanes(system, "")

    assert [(block.syntax, block.content) for block in result.definition.lanes[0].blocks] == [
        ("freeform", "`~`~\n"),
        ("markdown", "Shown"),
    ]


def test_decomposer_records_empty_lanes_for_byte_preserving_legacy_origin() -> None:
    result = decompose_legacy_lanes("", "literal\n\n")

    assert result.definition.lanes[0].blocks == ()
    assert result.user_origin.text == "literal\n\n"
    assert result.user_origin.fingerprint == sha256(b"literal\n\n").hexdigest()
