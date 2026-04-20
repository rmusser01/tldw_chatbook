"""
Tests for character/persona/chat-session endpoint wiring on the shared API client.
"""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api.client import TLDWAPIClient
from tldw_chatbook.tldw_api.character_persona_schemas import (
    PersonaExemplarCreate,
    PersonaProfileCreate,
    PresetCreate,
    PresetUpdate,
)


def _assert_request_call(call_args, expected_method, expected_endpoint, expected_kwargs):
    args, kwargs = call_args
    assert args[:2] == (expected_method, expected_endpoint)
    for key, value in expected_kwargs.items():
        assert kwargs[key] == value


@pytest.mark.asyncio
class TestCharacterPersonaClient:
    async def test_list_persona_profiles_hits_persona_endpoint(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"items": []})
        monkeypatch.setattr(client, "_request", mocked)

        await client.list_persona_profiles()

        _assert_request_call(
            mocked.await_args,
            "GET",
            "/api/v1/persona/profiles",
            {"params": {}},
        )

    async def test_list_character_exemplars_hits_character_endpoint(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"items": []})
        monkeypatch.setattr(client, "_request", mocked)

        await client.list_character_exemplars(12)

        _assert_request_call(
            mocked.await_args,
            "GET",
            "/api/v1/characters/12/exemplars",
            {},
        )

    async def test_greeting_and_preset_endpoints_are_wired(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"ok": True})
        monkeypatch.setattr(client, "_request", mocked)

        await client.list_greetings("chat-1")
        await client.select_greeting("chat-1", 2)
        await client.list_presets()
        await client.create_preset(PresetCreate(preset_id="custom", name="Custom", section_order=["system"], section_templates={"system": "hi"}))
        await client.update_preset("custom", PresetUpdate(name="Updated"))
        await client.delete_preset("custom")

        assert mocked.await_count == 6

    async def test_persona_request_models_can_be_serialized_for_client_use(self):
        profile = PersonaProfileCreate(id="persona-1", name="Guide")
        exemplar = PersonaExemplarCreate(persona_id="persona-1", kind="style", content="hello")

        assert profile.model_dump(exclude_none=True)["id"] == "persona-1"
        assert exemplar.model_dump(exclude_none=True)["persona_id"] == "persona-1"
