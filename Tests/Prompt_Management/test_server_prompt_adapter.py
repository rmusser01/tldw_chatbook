import json

import pytest

from tldw_chatbook.Prompt_Management.Prompts_Interop import (
    add_prompt,
    apply_server_prompt_version,
    export_prompt_to_server_payload,
    fetch_prompt_details,
    import_prompt_from_server_payload,
    initialize_interop,
    shutdown_interop,
)
from tldw_chatbook.Prompt_Management.server_prompt_adapter import (
    local_prompt_to_preview_payload,
    local_prompt_to_server_payload,
    server_prompt_to_local_update,
)


@pytest.fixture(autouse=True)
def interop_db():
    shutdown_interop()
    initialize_interop(":memory:", client_id="test-client")
    try:
        yield
    finally:
        shutdown_interop()


def test_server_prompt_to_local_update_preserves_structured_fields():
    payload = {
        "name": "Server Prompt",
        "author": "Server Author",
        "prompt_format": "structured",
        "prompt_schema_version": 1,
        "prompt_definition": {"schema_version": 1, "messages": [{"role": "user", "content": "hi"}]},
    }

    update = server_prompt_to_local_update(payload)

    assert update["prompt_format"] == "structured"
    assert update["prompt_schema_version"] == 1
    assert update["prompt_definition"]["schema_version"] == 1


def test_local_prompt_to_server_payload_keeps_legacy_snapshot():
    local_prompt = {
        "name": "Legacy Prompt",
        "prompt_format": "legacy",
        "system_prompt": "system",
        "user_prompt": "user",
    }

    payload = local_prompt_to_server_payload(local_prompt)

    assert payload["prompt_format"] == "legacy"
    assert payload["system_prompt"] == "system"
    assert payload["user_prompt"] == "user"


def test_local_prompt_to_preview_payload_parses_serialized_definition():
    local_prompt = {
        "name": "Structured Prompt",
        "prompt_format": "structured",
        "prompt_schema_version": 1,
        "prompt_definition": json.dumps(
            {"schema_version": 1, "messages": [{"role": "system", "content": "hello"}]}
        ),
    }

    payload = local_prompt_to_preview_payload(local_prompt)

    assert payload["prompt_format"] == "structured"
    assert payload["prompt_definition"]["messages"][0]["content"] == "hello"


def test_import_and_export_prompt_payload_round_trip_structured_fields():
    result = import_prompt_from_server_payload(
        {
            "name": "Round Trip Prompt",
            "author": "Parity Tester",
            "details": "Imported from server payload",
            "system_prompt": "legacy system",
            "user_prompt": "legacy user",
            "keywords": ["sync", "structured"],
            "prompt_format": "structured",
            "prompt_schema_version": 1,
            "prompt_definition": {
                "schema_version": 1,
                "messages": [{"role": "user", "content": "imported"}],
            },
        }
    )

    prompt = fetch_prompt_details(result["prompt_uuid"], include_deleted=True)
    assert prompt["prompt_format"] == "structured"
    assert json.loads(prompt["prompt_definition"])["schema_version"] == 1

    exported = export_prompt_to_server_payload(result["prompt_uuid"])
    assert exported["prompt_format"] == "structured"
    assert exported["prompt_definition"]["messages"][0]["content"] == "imported"
    assert exported["keywords"] == ["structured", "sync"]


def test_apply_server_prompt_version_updates_existing_prompt():
    _, prompt_uuid, _ = add_prompt(
        name="Versioned Prompt",
        author="Local Author",
        details="Local prompt",
        system_prompt="old system",
        user_prompt="old user",
    )

    apply_server_prompt_version(
        prompt_uuid,
        {
            "name": "Versioned Prompt",
            "prompt_format": "structured",
            "prompt_schema_version": 2,
            "prompt_definition": {
                "schema_version": 2,
                "messages": [{"role": "assistant", "content": "updated"}],
            },
        },
    )

    updated = fetch_prompt_details(prompt_uuid, include_deleted=True)
    assert updated["prompt_format"] == "structured"
    assert updated["prompt_schema_version"] == 2
    assert json.loads(updated["prompt_definition"])["schema_version"] == 2
