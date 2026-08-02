import json
from pathlib import Path

import pytest

from tldw_chatbook.Prompt_Management.prompt_artifact_codec import (
    decode_prompt_artifact,
    deserialize_definition,
)
from tldw_chatbook.Prompt_Management.prompt_artifact_models import (
    BlockArtifactDefinition,
    PromptLane,
)


FIXTURES = (
    Path(__file__).parents[2] / "Docs" / "fixtures" / "console-block-prompts"
)


def _definition(kind: str = "block_prompt") -> dict[str, object]:
    return {
        "schema_version": 2,
        "kind": kind,
        "lanes": [{"id": "system", "blocks": []}, {"id": "user", "blocks": []}],
    }


def _structured_record(
    *,
    artifact_type: str = "prompt",
    version: int = 2,
    definition: object | None = None,
) -> dict[str, object]:
    return {
        "artifact_type": artifact_type,
        "prompt_format": "structured",
        "prompt_schema_version": version,
        "prompt_definition": _definition() if definition is None else definition,
        "system_prompt": "stored system",
        "user_prompt": "stored user",
    }


def test_legacy_record_keeps_compatibility_text_without_definition() -> None:
    decoded = decode_prompt_artifact(
        {"system_prompt": "System\n", "user_prompt": "User", "artifact_type": "prompt"}
    )

    assert decoded.state == "legacy"
    assert decoded.definition is None
    assert (decoded.compiled_system, decoded.compiled_user) == ("System\n", "User")
    assert not decoded.compatibility_stale


@pytest.mark.parametrize(
    ("artifact_type", "kind"), [("prompt", "block_prompt"), ("recipe", "block_recipe")]
)
def test_valid_console_v2_artifacts_decode_to_typed_definition(
    artifact_type: str, kind: str
) -> None:
    decoded = decode_prompt_artifact(
        _structured_record(artifact_type=artifact_type, definition=_definition(kind))
    )

    assert decoded.state == "supported_v2"
    assert decoded.artifact_type == artifact_type
    assert decoded.definition is not None
    assert decoded.definition.kind == kind
    assert decoded.compatibility_stale


@pytest.mark.parametrize("case", json.loads((FIXTURES / "error-cases.json").read_text())["structured"])
def test_rejected_structured_cases_have_explicit_states(case: dict[str, object]) -> None:
    decoded = decode_prompt_artifact(case["record"])

    assert decoded.state == case["state"]
    assert decoded.definition is None


def test_v1_and_future_structured_records_remain_foreign() -> None:
    v1 = decode_prompt_artifact(_structured_record(version=1, definition={"schema_version": 1}))
    future = decode_prompt_artifact(_structured_record(version=99, definition={"schema_version": 99}))

    assert v1.state == "foreign_v1"
    assert future.state == "unsupported"


def test_malformed_json_is_an_explicit_decoded_state() -> None:
    decoded = decode_prompt_artifact(_structured_record(definition="{not json"))

    assert decoded.state == "malformed"
    assert decoded.raw_definition is None


def test_invalid_xml_wrapper_is_malformed_not_an_accidental_decode_exception() -> None:
    definition = _definition()
    definition["lanes"] = [
        {
            "id": "system",
            "blocks": [
                {
                    "id": "context",
                    "title": "Context",
                    "syntax": "xml",
                    "xml_tag": "context",
                    "content": "<context>already wrapped</context>",
                }
            ],
        },
        {"id": "user", "blocks": []},
    ]

    decoded = decode_prompt_artifact(_structured_record(definition=definition))

    assert decoded.state == "malformed"
    assert decoded.definition is None


def test_single_text_recipe_v2_is_foreign_not_console_recipe() -> None:
    record = _structured_record(
        artifact_type="recipe",
        definition={
            "schema_version": 2,
            "definition_kind": "single_text_recipe",
            "blocks": [],
        },
    )

    decoded = decode_prompt_artifact(record)

    assert decoded.state == "unsupported"
    assert decoded.definition is None
    assert decoded.raw_definition == record["prompt_definition"]


def test_deserialize_definition_accepts_json_strings_and_mappings_only() -> None:
    assert deserialize_definition('{"schema_version": 2}') == {"schema_version": 2}
    assert deserialize_definition({"schema_version": 2}) == {"schema_version": 2}
    assert deserialize_definition("[]") is None
    assert deserialize_definition(None) is None


def test_models_reject_non_lane_members_with_a_validation_error() -> None:
    with pytest.raises(ValueError):
        BlockArtifactDefinition(
            kind="block_prompt",
            schema_version=2,
            lanes=(PromptLane(id="system", blocks=()), "not-a-lane"),  # type: ignore[arg-type]
        )
