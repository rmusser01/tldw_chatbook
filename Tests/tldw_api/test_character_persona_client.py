"""
Tests for character/persona/chat-session endpoint wiring on the shared API client.
"""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api.client import TLDWAPIClient
from tldw_chatbook.tldw_api.character_persona_schemas import (
    CharacterCreateRequest,
    CharacterExemplarCreate,
    CharacterExemplarSearchRequest,
    CharacterExemplarSelectionDebugRequest,
    CharacterExemplarUpdate,
    CharacterQueryRequest,
    CharacterUpdateRequest,
    PersonaExemplarCreate,
    PersonaExemplarImportRequest,
    PersonaExemplarReviewRequest,
    PersonaExemplarUpdate,
    PersonaProfileCreate,
    PersonaProfileUpdate,
    PersonaSessionRequest,
    PersonaSetupState,
    PersonaVoiceDefaults,
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
    async def test_character_endpoint_wiring(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"ok": True})
        monkeypatch.setattr(client, "_request", mocked)

        await client.list_characters()
        await client.query_characters(CharacterQueryRequest(query="alpha"))
        await client.search_characters("alpha")
        await client.get_character(12)
        await client.create_character(CharacterCreateRequest(name="Ada"))
        await client.update_character(12, CharacterUpdateRequest(name="Ada v2"), 3)
        await client.delete_character(12, 4)
        await client.restore_character(12, 5)

        expected_calls = [
            ("GET", "/api/v1/characters/", {"params": {"limit": 100, "offset": 0}}),
            (
                "GET",
                "/api/v1/characters/query",
                {"params": CharacterQueryRequest(query="alpha").model_dump(exclude_none=True, mode="json")},
            ),
            ("GET", "/api/v1/characters/search/", {"params": {"query": "alpha", "limit": 10}}),
            ("GET", "/api/v1/characters/12", {}),
            ("POST", "/api/v1/characters/", {"json_data": {"name": "Ada"}}),
            (
                "PUT",
                "/api/v1/characters/12",
                {"json_data": {"name": "Ada v2"}, "params": {"expected_version": 3}},
            ),
            ("DELETE", "/api/v1/characters/12", {"params": {"expected_version": 4}}),
            ("POST", "/api/v1/characters/12/restore", {"params": {"expected_version": 5}}),
        ]
        assert len(mocked.await_args_list) == len(expected_calls)
        for call_args, expected in zip(mocked.await_args_list, expected_calls):
            _assert_request_call(call_args, *expected)

        query_payload = mocked.await_args_list[1][1]["params"]
        assert query_payload["query"] == "alpha"
        assert "page" in query_payload

    async def test_character_exemplar_endpoint_wiring(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"ok": True})
        monkeypatch.setattr(client, "_request", mocked)

        create_payload = CharacterExemplarCreate(text="hello")

        await client.list_character_exemplars(12)
        await client.get_character_exemplar(12, "ex-1")
        await client.create_character_exemplar(12, create_payload)
        await client.update_character_exemplar(12, "ex-1", CharacterExemplarUpdate(text="updated"))
        await client.delete_character_exemplar(12, "ex-1")
        await client.search_character_exemplars(12, CharacterExemplarSearchRequest(query="hello"))
        await client.select_character_exemplars_debug(
            12,
            CharacterExemplarSelectionDebugRequest(user_turn="why?"),
        )

        assert len(mocked.await_args_list) == 7
        _assert_request_call(mocked.await_args_list[0], "GET", "/api/v1/characters/12/exemplars", {})
        _assert_request_call(mocked.await_args_list[1], "GET", "/api/v1/characters/12/exemplars/ex-1", {})
        _assert_request_call(
            mocked.await_args_list[2],
            "POST",
            "/api/v1/characters/12/exemplars",
            {},
        )
        create_payload = mocked.await_args_list[2][1]["json_data"]
        assert create_payload["text"] == "hello"
        assert "character_id" not in create_payload
        _assert_request_call(
            mocked.await_args_list[3],
            "PUT",
            "/api/v1/characters/12/exemplars/ex-1",
            {"json_data": {"text": "updated"}},
        )
        _assert_request_call(mocked.await_args_list[4], "DELETE", "/api/v1/characters/12/exemplars/ex-1", {})
        _assert_request_call(
            mocked.await_args_list[5],
            "POST",
            "/api/v1/characters/12/exemplars/search",
            {"json_data": {"query": "hello", "filter": {"rhetorical": []}, "limit": 20, "offset": 0, "use_embedding_scores": False}},
        )
        _assert_request_call(
            mocked.await_args_list[6],
            "POST",
            "/api/v1/characters/12/exemplars/select/debug",
            {"json_data": {"user_turn": "why?", "selection_config": {"budget_tokens": 600, "max_exemplar_tokens": 120, "mmr_lambda": 0.7, "use_embedding_scores": False}}},
        )

    async def test_persona_profile_endpoint_wiring(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"ok": True})
        monkeypatch.setattr(client, "_request", mocked)

        await client.list_persona_profiles()
        await client.get_persona_profile("persona-1")
        await client.create_persona_profile(PersonaProfileCreate(id="persona-1", name="Guide"))
        await client.update_persona_profile("persona-1", PersonaProfileUpdate(name="Guide v2"), 7)
        await client.delete_persona_profile("persona-1", 8)
        await client.restore_persona_profile("persona-1", 9)

        expected_calls = [
            ("GET", "/api/v1/persona/profiles", {"params": {}}),
            ("GET", "/api/v1/persona/profiles/persona-1", {}),
            ("POST", "/api/v1/persona/profiles", {}),
            (
                "PATCH",
                "/api/v1/persona/profiles/persona-1",
                {"json_data": {"name": "Guide v2"}, "params": {"expected_version": 7}},
            ),
            ("DELETE", "/api/v1/persona/profiles/persona-1", {"params": {"expected_version": 8}}),
            ("POST", "/api/v1/persona/profiles/persona-1/restore", {"params": {"expected_version": 9}}),
        ]
        assert len(mocked.await_args_list) == len(expected_calls)
        for call_args, expected in zip(mocked.await_args_list, expected_calls):
            _assert_request_call(call_args, *expected)

        create_payload = mocked.await_args_list[2][1]["json_data"]
        assert create_payload["id"] == "persona-1"
        assert create_payload["name"] == "Guide"
        assert create_payload["voice_defaults"]["voice_chat_trigger_phrases"] == []
        assert create_payload["setup"]["status"] == "not_started"
        assert create_payload["setup"]["current_step"] == "persona"
        assert isinstance(PersonaVoiceDefaults.model_validate(create_payload["voice_defaults"]), PersonaVoiceDefaults)
        assert isinstance(PersonaSetupState.model_validate(create_payload["setup"]), PersonaSetupState)

    async def test_persona_exemplar_endpoint_wiring(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"ok": True})
        monkeypatch.setattr(client, "_request", mocked)

        await client.list_persona_exemplars("persona-1")
        await client.get_persona_exemplar("persona-1", "ex-1")
        await client.create_persona_exemplar("persona-1", PersonaExemplarCreate(content="hello"))
        await client.import_persona_exemplars("persona-1", PersonaExemplarImportRequest(transcript="hello world"))
        await client.update_persona_exemplar("persona-1", "ex-1", PersonaExemplarUpdate(content="updated"))
        await client.review_persona_exemplar("persona-1", "ex-1", PersonaExemplarReviewRequest(action="approve"))
        await client.delete_persona_exemplar("persona-1", "ex-1")

        expected_calls = [
            ("GET", "/api/v1/persona/profiles/persona-1/exemplars", {"params": {}}),
            ("GET", "/api/v1/persona/profiles/persona-1/exemplars/ex-1", {}),
            ("POST", "/api/v1/persona/profiles/persona-1/exemplars", {}),
            (
                "POST",
                "/api/v1/persona/profiles/persona-1/exemplars/import",
                {"json_data": {"transcript": "hello world", "max_candidates": 5}},
            ),
            ("PATCH", "/api/v1/persona/profiles/persona-1/exemplars/ex-1", {"json_data": {"content": "updated"}}),
            ("POST", "/api/v1/persona/profiles/persona-1/exemplars/ex-1/review", {"json_data": {"action": "approve"}}),
            ("DELETE", "/api/v1/persona/profiles/persona-1/exemplars/ex-1", {}),
        ]
        assert len(mocked.await_args_list) == len(expected_calls)
        for call_args, expected in zip(mocked.await_args_list, expected_calls):
            _assert_request_call(call_args, *expected)

        create_payload = mocked.await_args_list[2][1]["json_data"]
        assert create_payload["content"] == "hello"
        assert "persona_id" not in create_payload

    async def test_greetings_and_presets_endpoint_wiring(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"ok": True})
        monkeypatch.setattr(client, "_request", mocked)

        await client.list_greetings("chat-1")
        await client.select_greeting("chat-1", 2)
        await client.list_presets()
        await client.create_preset(
            PresetCreate(
                preset_id="custom",
                name="Custom",
                section_order=["system"],
                section_templates={"system": "hi"},
            )
        )
        await client.update_preset("custom", PresetUpdate(name="Updated"))
        await client.delete_preset("custom")

        expected_calls = [
            ("GET", "/api/v1/chats/chat-1/greetings", {}),
            ("PUT", "/api/v1/chats/chat-1/greetings/select", {"json_data": {"index": 2}}),
            ("GET", "/api/v1/chats/presets", {}),
            (
                "POST",
                "/api/v1/chats/presets",
                {"json_data": {"preset_id": "custom", "name": "Custom", "section_order": ["system"], "section_templates": {"system": "hi"}}},
            ),
            ("PUT", "/api/v1/chats/presets/custom", {"json_data": {"name": "Updated"}}),
            ("DELETE", "/api/v1/chats/presets/custom", {}),
        ]
        assert len(mocked.await_args_list) == len(expected_calls)
        for call_args, expected in zip(mocked.await_args_list, expected_calls):
            _assert_request_call(call_args, *expected)

    async def test_persona_session_request_model_serializes_cleanly(self):
        request = PersonaSessionRequest(persona_id="persona-1", project_id="project-1")

        assert request.model_dump(exclude_none=True) == {
            "persona_id": "persona-1",
            "project_id": "project-1",
        }
