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


def test_decomposer_keeps_fenced_and_incomplete_or_ambiguous_content_freeform() -> None:
    system = "```markdown\n# inside a fence\n```\n\n<open>unfinished"
    user = "prefix <context>not top-level</context>\n# heading without blank"

    result = decompose_legacy_lanes(system, user)

    assert [block.syntax for block in result.definition.lanes[0].blocks] == ["freeform"]
    assert result.definition.lanes[0].blocks[0].content == system
    assert [block.syntax for block in result.definition.lanes[1].blocks] == ["freeform"]
    assert result.definition.lanes[1].blocks[0].content == user


def test_decomposer_records_empty_lanes_for_byte_preserving_legacy_origin() -> None:
    result = decompose_legacy_lanes("", "literal\n\n")

    assert result.definition.lanes[0].blocks == ()
    assert result.user_origin.text == "literal\n\n"
    assert result.user_origin.fingerprint == sha256(b"literal\n\n").hexdigest()
