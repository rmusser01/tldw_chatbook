"""Retained Prompt history normalization contracts for TASK-196."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

from tldw_chatbook.Prompt_Management import prompt_normalizers
from tldw_chatbook.Prompt_Management.prompt_source_capabilities import (
    PromptSourceCapabilities,
    local_prompt_capabilities,
)


PROMPT_UUID = "00000000-0000-4000-8000-000000000196"


def _definition(
    *, kind: str = "block_prompt", schema_version: int = 2
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "kind": kind,
        "lanes": [
            {
                "id": "system",
                "blocks": [
                    {
                        "id": "role",
                        "title": "Role",
                        "syntax": "freeform",
                        "content": "system-v2",
                    }
                ],
            },
            {
                "id": "user",
                "blocks": [
                    {
                        "id": "request",
                        "title": "Request",
                        "syntax": "freeform",
                        "content": "user-v2",
                    }
                ],
            },
        ],
    }


def _legacy_payload(*, version: int, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "uuid": "payload-must-not-win",
        "version": 999,
        "name": f"Prompt v{version}",
        "author": "  exact author  ",
        "details": "details\nwith [literal] markup",
        "system_prompt": "  exact system\n",
        "user_prompt": "exact user  ",
        "prompt_format": "legacy",
        "prompt_schema_version": None,
        "prompt_definition": None,
        "artifact_type": "prompt",
        "keywords": ["alpha", "beta gamma"],
    }
    payload.update(overrides)
    return payload


def _structured_payload(
    *, artifact_type: str = "prompt", definition: Any = None, **overrides: Any
) -> dict[str, Any]:
    if definition is None:
        definition = _definition(
            kind="block_recipe" if artifact_type == "recipe" else "block_prompt"
        )
    definition_text = (
        definition
        if isinstance(definition, str)
        else json.dumps(definition, separators=(",", ":"))
    )
    payload: dict[str, Any] = {
        "name": "Structured",
        "author": None,
        "details": "exact details",
        "system_prompt": "system-v2",
        "user_prompt": "user-v2",
        "prompt_format": "structured",
        "prompt_schema_version": 2,
        "prompt_definition": definition_text,
        "artifact_type": artifact_type,
        "keywords": [],
    }
    payload.update(overrides)
    return payload


def _row(
    *,
    change_id: int,
    version: int,
    payload: Any,
    payload_error: str | None = None,
    raw_payload: Any = None,
    **overrides: Any,
) -> dict[str, Any]:
    row = {
        "change_id": change_id,
        "entity": "Prompts",
        "entity_uuid": PROMPT_UUID,
        "operation": "create" if version == 1 else "update",
        "timestamp": f"2026-08-08T00:{version:02d}:00.000Z",
        "client_id": "history-test",
        "version": version,
        "payload": payload,
        "payload_error": payload_error,
        "raw_payload": raw_payload,
    }
    row.update(overrides)
    return row


def _page(
    items: list[dict[str, Any]],
    *,
    predecessor: dict[str, Any] | None = None,
    total_count: int | None = None,
    has_more: bool | None = None,
    next_before_change_id: int | None = None,
) -> dict[str, Any]:
    resolved_has_more = predecessor is not None if has_more is None else has_more
    if resolved_has_more and next_before_change_id is None and items:
        next_before_change_id = items[-1]["change_id"]
    return {
        "items": items,
        "predecessor": predecessor,
        "total_count": len(items) + (1 if predecessor is not None else 0)
        if total_count is None
        else total_count,
        "has_more": resolved_has_more,
        "next_before_change_id": next_before_change_id,
    }


def _normalize(
    payload: Any, *, capabilities: PromptSourceCapabilities | None = None
) -> dict[str, Any]:
    return prompt_normalizers.normalize_prompt_history_page(
        payload,
        backend="local",
        capabilities=capabilities,
    )


def test_history_page_preserves_envelope_and_uses_authoritative_sync_metadata() -> None:
    payload = _legacy_payload(version=2)
    result = _normalize(
        _page(
            [_row(change_id=22, version=2, payload=payload)],
            predecessor=_row(
                change_id=11, version=1, payload=_legacy_payload(version=1)
            ),
            total_count=7,
        )
    )

    assert set(result) == {
        "items",
        "total_count",
        "has_more",
        "next_before_change_id",
    }
    assert result["total_count"] == 7
    assert result["has_more"] is True
    assert result["next_before_change_id"] == 22
    assert len(result["items"]) == 1

    item = result["items"][0]
    assert item["backend"] == "local"
    assert item["change_id"] == 22
    assert item["version"] == 2
    assert item["operation"] == "update"
    assert item["timestamp"] == "2026-08-08T00:02:00.000Z"
    assert item["prompt_uuid"] == PROMPT_UUID
    assert item["name"] == "Prompt v2"
    assert item["author"] == "  exact author  "
    assert item["details"] == "details\nwith [literal] markup"
    assert item["system_prompt"] == "  exact system\n"
    assert item["user_prompt"] == "exact user  "
    assert item["prompt_definition"] is None
    assert item["keywords"] == ["alpha", "beta gamma"]
    assert item["keywords_captured"] is True
    assert item["definition_state"] == "legacy"
    assert item["compatibility_state"] == "compatible"
    assert item["compatibility_reason"] is None
    assert item["restore_eligible"] is True


@pytest.mark.parametrize(
    ("artifact_type", "eligible", "state", "reason"),
    [
        (
            "recipe",
            False,
            "legacy_recipe",
            "Legacy Recipe snapshots are preview-only.",
        ),
        (None, True, "compatible", None),
    ],
    ids=["explicit-recipe", "missing-type-defaults-to-prompt"],
)
def test_legacy_history_restore_eligibility_preserves_prompt_only_rule(
    artifact_type: str | None,
    eligible: bool,
    state: str,
    reason: str | None,
) -> None:
    payload = _legacy_payload(version=1)
    if artifact_type is None:
        del payload["artifact_type"]
    else:
        payload["artifact_type"] = artifact_type

    item = _normalize(
        _page([_row(change_id=1, version=1, payload=payload)], total_count=1),
        capabilities=local_prompt_capabilities(),
    )["items"][0]

    assert item["artifact_type"] == (artifact_type or "prompt")
    assert item["definition_state"] == "legacy"
    assert item["compatibility_state"] == state
    assert item["compatibility_reason"] == reason
    assert item["restore_eligible"] is eligible


@pytest.mark.parametrize(
    ("capability_overrides", "artifact_type"),
    [
        ({"structured_kinds": frozenset()}, "prompt"),
        ({"artifact_types": frozenset({"prompt"})}, "recipe"),
        ({"compiled_lane_limit": 4}, "prompt"),
        ({"definition_limit": 8}, "prompt"),
        ({"request_limit": 8}, "prompt"),
    ],
    ids=[
        "structured-kind",
        "artifact-type",
        "compiled-lane-limit",
        "definition-limit",
        "request-limit",
    ],
)
def test_structured_history_uses_current_source_capabilities_before_restore(
    capability_overrides: dict[str, Any], artifact_type: str
) -> None:
    capabilities = replace(local_prompt_capabilities(), **capability_overrides)
    payload = _structured_payload(artifact_type=artifact_type)

    item = _normalize(
        _page([_row(change_id=1, version=1, payload=payload)], total_count=1),
        capabilities=capabilities,
    )["items"][0]

    assert item["system_prompt"] == "system-v2"
    assert item["user_prompt"] == "user-v2"
    assert item["definition_state"] == "supported_v2"
    assert item["compatibility_state"] == "current_capability_unsupported"
    assert item["compatibility_reason"] == (
        "This retained version is not supported by current local Prompt capabilities."
    )
    assert item["restore_eligible"] is False


@pytest.mark.parametrize(
    ("artifact_type", "kind"),
    [("prompt", "block_prompt"), ("recipe", "block_recipe")],
)
def test_supported_structured_v2_prompt_and_recipe_are_restore_eligible(
    artifact_type: str, kind: str
) -> None:
    definition_text = json.dumps(_definition(kind=kind), separators=(",", ":"))

    result = _normalize(
        _page(
            [
                _row(
                    change_id=1,
                    version=1,
                    payload=_structured_payload(
                        artifact_type=artifact_type,
                        definition=json.loads(definition_text),
                    ),
                )
            ],
            total_count=1,
        )
    )["items"][0]

    assert result["artifact_type"] == artifact_type
    assert result["prompt_definition"] == definition_text
    assert result["system_prompt"] == "system-v2"
    assert result["user_prompt"] == "user-v2"
    assert result["definition_state"] == "supported_v2"
    assert result["compatibility_state"] == "compatible"
    assert result["compatibility_reason"] is None
    assert result["restore_eligible"] is True
    assert result["change_summary"] == "Created"


@pytest.mark.parametrize(
    ("row", "state", "reason", "preview"),
    [
        (
            _row(
                change_id=2,
                version=2,
                payload=None,
                payload_error="malformed_json",
                raw_payload="{not-json",
            ),
            "malformed_payload",
            "Retained snapshot JSON is malformed.",
            "{not-json",
        ),
        (
            _row(change_id=2, version=2, payload=["valid", "non-object"]),
            "non_object_payload",
            "Retained snapshot payload must be a JSON object.",
            ["valid", "non-object"],
        ),
    ],
)
def test_malformed_and_non_object_payloads_remain_visible_preview_only(
    row: dict[str, Any], state: str, reason: str, preview: Any
) -> None:
    item = _normalize(_page([row], total_count=1))["items"][0]

    assert item["change_id"] == 2
    assert item["version"] == 2
    assert item["prompt_uuid"] == PROMPT_UUID
    assert item["payload_preview"] == preview
    assert item["definition_state"] == "malformed"
    assert item["compatibility_state"] == state
    assert item["compatibility_reason"] == reason
    assert item["restore_eligible"] is False


@pytest.mark.parametrize(
    ("payload", "definition_state", "compatibility_state", "reason"),
    [
        (
            _legacy_payload(version=2, prompt_format="future-format"),
            "unsupported",
            "unknown_format",
            "Prompt format is unsupported.",
        ),
        (
            _structured_payload(definition="{not-definition-json"),
            "malformed",
            "malformed_definition",
            "Structured definition is malformed.",
        ),
        (
            _structured_payload(definition=_definition(schema_version=3)),
            "mismatched",
            "schema_mismatch",
            "Prompt schema version does not match the definition schema version.",
        ),
        (
            _structured_payload(system_prompt="stale compiled system"),
            "supported_v2",
            "compiled_text_mismatch",
            "Stored System/User text does not match the structured definition.",
        ),
        (
            _structured_payload(
                artifact_type="recipe", definition=_definition(kind="block_prompt")
            ),
            "mismatched",
            "artifact_kind_mismatch",
            "Artifact type does not match the structured definition kind.",
        ),
        (
            _structured_payload(prompt_schema_version=3),
            "unsupported",
            "unsupported_schema",
            "Prompt schema version is unsupported.",
        ),
        (
            _structured_payload(artifact_type="future-artifact"),
            "unsupported",
            "unsupported_artifact_type",
            "Artifact type is unsupported.",
        ),
        (
            _structured_payload(prompt_schema_version=1),
            "foreign_v1",
            "foreign_v1",
            "Structured-v1 artifacts are preview-only.",
        ),
    ],
)
def test_compatibility_invalid_snapshots_have_stable_preview_only_reasons(
    payload: dict[str, Any],
    definition_state: str,
    compatibility_state: str,
    reason: str,
) -> None:
    item = _normalize(
        _page([_row(change_id=2, version=2, payload=payload)], total_count=1)
    )["items"][0]

    assert item["definition_state"] == definition_state
    assert item["compatibility_state"] == compatibility_state
    assert item["compatibility_reason"] == reason
    assert item["restore_eligible"] is False
    assert item["system_prompt"] == payload["system_prompt"]
    assert item["user_prompt"] == payload["user_prompt"]
    assert item["prompt_definition"] == payload["prompt_definition"]


def test_foreign_v1_definition_is_preserved_without_calling_v2_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_definition = '{"schema_version":1,"roles":["system","user"]}'
    payload = _structured_payload(
        prompt_schema_version=1,
        definition=raw_definition,
        system_prompt="foreign [system]",
        user_prompt="foreign user",
    )

    def fail_decode(_record: Any) -> Any:
        raise AssertionError("foreign v1 must not enter the v2 artifact decoder")

    monkeypatch.setattr(prompt_normalizers, "decode_prompt_artifact", fail_decode)

    item = _normalize(
        _page([_row(change_id=1, version=1, payload=payload)], total_count=1)
    )["items"][0]

    assert item["prompt_definition"] == raw_definition
    assert item["system_prompt"] == "foreign [system]"
    assert item["user_prompt"] == "foreign user"
    assert item["compatibility_state"] == "foreign_v1"
    assert item["restore_eligible"] is False


def test_single_text_recipe_definition_kind_is_unsupported_not_malformed() -> None:
    definition = {
        "schema_version": 2,
        "definition_kind": "single_text_recipe",
        "blocks": [{"role": "system", "content": "foreign [literal] text"}],
    }
    definition_text = json.dumps(definition, separators=(",", ":"))
    payload = _structured_payload(
        artifact_type="recipe",
        definition=definition,
        system_prompt="foreign [system]",
        user_prompt="foreign user",
    )

    item = _normalize(
        _page([_row(change_id=2, version=2, payload=payload)], total_count=1)
    )["items"][0]

    assert item["definition_state"] == "unsupported"
    assert item["compatibility_state"] == "unsupported_definition_kind"
    assert item["compatibility_reason"] == (
        "Structured definition kind is unsupported."
    )
    assert item["restore_eligible"] is False
    assert item["prompt_definition"] == definition_text
    assert item["system_prompt"] == "foreign [system]"
    assert item["user_prompt"] == "foreign user"


def test_missing_historical_keywords_are_explicit_but_do_not_block_restore() -> None:
    payload = _legacy_payload(version=1)
    del payload["keywords"]

    item = _normalize(
        _page([_row(change_id=1, version=1, payload=payload)], total_count=1)
    )["items"][0]

    assert item["keywords"] == []
    assert item["keywords_captured"] is False
    assert item["restore_eligible"] is True


@pytest.mark.parametrize(
    "keywords",
    ["alpha,beta", ["Alpha"], ["beta", "alpha"], ["alpha", "alpha"], ["two  spaces"]],
)
def test_malformed_keyword_captures_are_preview_only(keywords: Any) -> None:
    item = _normalize(
        _page(
            [
                _row(
                    change_id=2,
                    version=2,
                    payload=_legacy_payload(version=2, keywords=keywords),
                )
            ],
            total_count=1,
        )
    )["items"][0]

    assert item["keywords_captured"] is False
    assert item["compatibility_state"] == "malformed_keywords"
    assert item["compatibility_reason"] == (
        "Captured keywords are not a canonical keyword list."
    )
    assert item["restore_eligible"] is False


def test_each_visible_row_compares_only_with_its_immediate_older_snapshot() -> None:
    version_1 = _legacy_payload(
        version=1,
        name="First",
        author="Same",
        system_prompt="system-one",
        keywords=["alpha"],
    )
    version_2 = _legacy_payload(
        version=2,
        name="Second",
        author="Same",
        system_prompt="system-one",
        keywords=["alpha"],
    )
    version_3 = _legacy_payload(
        version=3,
        name="Second",
        author="Same",
        system_prompt="system-three",
        keywords=["alpha", "beta"],
    )

    result = _normalize(
        _page(
            [
                _row(change_id=30, version=3, payload=version_3),
                _row(change_id=20, version=2, payload=version_2),
            ],
            predecessor=_row(change_id=10, version=1, payload=version_1),
            total_count=3,
        )
    )

    assert [item["version"] for item in result["items"]] == [3, 2]
    assert result["items"][0]["changed_fields"] == ["system_prompt", "keywords"]
    assert result["items"][0]["change_summary"] == "System, Keywords"
    assert result["items"][1]["changed_fields"] == ["name"]
    assert result["items"][1]["change_summary"] == "Name"


def test_version_one_is_created_even_without_a_predecessor() -> None:
    item = _normalize(
        _page(
            [_row(change_id=1, version=1, payload=_legacy_payload(version=1))],
            total_count=1,
        )
    )["items"][0]

    assert item["changed_fields"] == []
    assert item["change_summary"] == "Created"


@pytest.mark.parametrize("predecessor_kind", ["missing", "malformed", "gap"])
def test_unavailable_immediate_baselines_are_disclosed(predecessor_kind: str) -> None:
    current = _row(change_id=30, version=3, payload=_legacy_payload(version=3))
    if predecessor_kind == "missing":
        predecessor = None
    elif predecessor_kind == "malformed":
        predecessor = _row(
            change_id=20,
            version=2,
            payload=None,
            payload_error="malformed_json",
            raw_payload="{bad",
        )
    else:
        predecessor = _row(change_id=10, version=1, payload=_legacy_payload(version=1))

    page = _page(
        [current],
        predecessor=predecessor,
        total_count=3,
        has_more=predecessor is not None,
        next_before_change_id=30 if predecessor is not None else None,
    )
    item = _normalize(page)["items"][0]

    assert item["changed_fields"] == []
    assert item["change_summary"] == "Earlier baseline unavailable"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda page: [],
        lambda page: {**page, "items": "not-a-list"},
        lambda page: {**page, "predecessor": []},
        lambda page: {**page, "total_count": True},
        lambda page: {**page, "total_count": -1},
        lambda page: {**page, "total_count": 0},
        lambda page: {**page, "has_more": "yes"},
        lambda page: {**page, "has_more": True, "predecessor": None},
        lambda page: {**page, "has_more": True, "next_before_change_id": None},
        lambda page: {**page, "next_before_change_id": 999},
        lambda page: {**page, "has_more": False, "next_before_change_id": 2},
    ],
)
def test_invalid_page_envelopes_counts_and_cursors_fail_closed(mutate: Any) -> None:
    valid = _page(
        [_row(change_id=2, version=2, payload=_legacy_payload(version=2))],
        predecessor=_row(change_id=1, version=1, payload=_legacy_payload(version=1)),
        total_count=2,
    )

    with pytest.raises((TypeError, ValueError), match="retained history page"):
        _normalize(mutate(deepcopy(valid)))


@pytest.mark.parametrize(
    "row_override",
    [
        {"change_id": True},
        {"version": 0},
        {"operation": "delete"},
        {"timestamp": ""},
        {"entity_uuid": None},
    ],
)
def test_invalid_authoritative_sync_metadata_fails_closed(
    row_override: dict[str, Any],
) -> None:
    row = _row(change_id=1, version=1, payload=_legacy_payload(version=1))
    row.update(row_override)

    with pytest.raises((TypeError, ValueError), match="retained history row"):
        _normalize(_page([row], total_count=1))


def test_mixed_visible_prompt_uuids_fail_the_page_closed() -> None:
    other_uuid = "00000000-0000-4000-8000-000000000197"
    page = _page(
        [
            _row(change_id=2, version=2, payload=_legacy_payload(version=2)),
            _row(
                change_id=1,
                version=1,
                payload=_legacy_payload(version=1),
                entity_uuid=other_uuid,
            ),
        ],
        total_count=2,
    )

    with pytest.raises(ValueError, match="one Prompt UUID"):
        _normalize(page)


def test_cross_prompt_predecessor_fails_before_change_summary() -> None:
    other_uuid = "00000000-0000-4000-8000-000000000197"
    page = _page(
        [_row(change_id=2, version=2, payload=_legacy_payload(version=2))],
        predecessor=_row(
            change_id=1,
            version=1,
            payload=_legacy_payload(version=1),
            entity_uuid=other_uuid,
        ),
        total_count=2,
    )

    with pytest.raises(ValueError, match="one Prompt UUID"):
        _normalize(page)
