import json
from pathlib import Path

import pytest

from tldw_chatbook.Prompt_Management.prompt_artifact_codec import decode_console_v2
from tldw_chatbook.Prompt_Management.prompt_block_compiler import (
    compile_block_artifact,
    validate_xml_wrapper,
)


FIXTURES = (
    Path(__file__).parents[2] / "Docs" / "fixtures" / "console-block-prompts"
)


def _render_fixture_definition():
    fixture = json.loads((FIXTURES / "render-cases.json").read_text())
    decoded = decode_console_v2(
        {"system_prompt": "", "user_prompt": ""},
        artifact_type="prompt",
        raw=fixture["artifact"],
    )
    assert decoded.definition is not None
    return decoded.definition, fixture["expected"]


def test_compiler_renders_exact_canonical_text_from_shared_fixture() -> None:
    definition, expected = _render_fixture_definition()

    assert compile_block_artifact(definition) == (expected["system"], expected["user"])


@pytest.mark.parametrize("case", json.loads((FIXTURES / "error-cases.json").read_text())["xml"])
def test_invalid_xml_wrapper_preserves_the_original_issue_input(case: dict[str, str]) -> None:
    with pytest.raises(ValueError) as raised:
        validate_xml_wrapper(case["xml_tag"], case["content"])

    assert case["input"] in str(raised.value)
