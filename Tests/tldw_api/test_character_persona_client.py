"""
Tests for character/persona/chat-session endpoint wiring on the shared API client.
"""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api.client import TLDWAPIClient
from tldw_chatbook.tldw_api.character_persona_schemas import (
    ArchetypePreviewResponse,
    ArchetypeSummary,
    ArchetypeTemplate,
    CharacterChatSessionCreate,
    CharacterChatSessionUpdate,
    CharacterCreateRequest,
    CharacterMessageCreate,
    CharacterMessageListResponse,
    CharacterMessageResponse,
    CharacterMessageUpdate,
    CharacterExemplarCreate,
    CharacterExemplarSearchRequest,
    CharacterExemplarSelectionDebugRequest,
    CharacterExemplarUpdate,
    CharacterChatMessageCreate,
    CharacterChatMessageUpdate,
    CharacterChatSessionCreate,
    CharacterChatSessionUpdate,
    CharacterChatSettingsUpdate,
    CharacterMemoryArchiveRequest,
    CharacterMemoryCreate,
    CharacterMemoryExtractRequest,
    CharacterMemoryUpdate,
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
    ChatSettingsUpdate,
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

        await client.get_character_exemplar(12, "ex-1")
        await client.create_character_exemplar(12, create_payload)
        await client.update_character_exemplar(12, "ex-1", CharacterExemplarUpdate(text="updated"))
        await client.delete_character_exemplar(12, "ex-1")
        await client.search_character_exemplars(12, CharacterExemplarSearchRequest(query="hello"))
        await client.select_character_exemplars_debug(
            12,
            CharacterExemplarSelectionDebugRequest(user_turn="why?"),
        )

        assert len(mocked.await_args_list) == 6
        _assert_request_call(mocked.await_args_list[0], "GET", "/api/v1/characters/12/exemplars/ex-1", {})
        _assert_request_call(
            mocked.await_args_list[1],
            "POST",
            "/api/v1/characters/12/exemplars",
            {},
        )
        create_payload = mocked.await_args_list[1][1]["json_data"]
        assert create_payload["text"] == "hello"
        assert "character_id" not in create_payload
        _assert_request_call(
            mocked.await_args_list[2],
            "PUT",
            "/api/v1/characters/12/exemplars/ex-1",
            {"json_data": {"text": "updated"}},
        )
        _assert_request_call(mocked.await_args_list[3], "DELETE", "/api/v1/characters/12/exemplars/ex-1", {})
        _assert_request_call(
            mocked.await_args_list[4],
            "POST",
            "/api/v1/characters/12/exemplars/search",
            {"json_data": {"query": "hello", "filter": {"rhetorical": []}, "limit": 20, "offset": 0, "use_embedding_scores": False}},
        )
        _assert_request_call(
            mocked.await_args_list[5],
            "POST",
            "/api/v1/characters/12/exemplars/select/debug",
            {"json_data": {"user_turn": "why?", "selection_config": {"budget_tokens": 600, "max_exemplar_tokens": 120, "mmr_lambda": 0.7, "use_embedding_scores": False}}},
        )

    async def test_character_exemplar_batch_create_uses_list_payload(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"ok": True})
        monkeypatch.setattr(client, "_request", mocked)

        payload = [
            CharacterExemplarCreate(text="hello"),
            CharacterExemplarCreate(text="world"),
        ]

        await client.create_character_exemplar(12, payload)

        _assert_request_call(
            mocked.await_args,
            "POST",
            "/api/v1/characters/12/exemplars",
            {},
        )
        assert mocked.await_args.kwargs["json_data"] == [
            {"text": "hello", "source": {"type": "other"}, "novelty_hint": "unknown", "labels": {"emotion": "other", "scenario": "other", "rhetorical": []}, "safety": {"allowed": [], "blocked": []}, "rights": {"public_figure": True}},
            {"text": "world", "source": {"type": "other"}, "novelty_hint": "unknown", "labels": {"emotion": "other", "scenario": "other", "rhetorical": []}, "safety": {"allowed": [], "blocked": []}, "rights": {"public_figure": True}},
        ]

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
            ("GET", "/api/v1/persona/profiles", {"params": {"active_only": "false", "include_deleted": "false", "limit": 100, "offset": 0}}),
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

    async def test_persona_archetype_endpoint_wiring(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(
            side_effect=[
                [
                    {
                        "key": "researcher",
                        "label": "Researcher",
                        "tagline": "Investigates sources",
                        "icon": "search",
                    }
                ],
                {
                    "key": "researcher",
                    "label": "Researcher",
                    "tagline": "Investigates sources",
                    "icon": "search",
                    "persona": {
                        "name": "Researcher",
                        "system_prompt": "Investigate carefully.",
                        "personality_traits": ["careful"],
                    },
                    "mcp_modules": {"enabled": ["web"], "disabled": []},
                    "suggested_external_servers": ["semantic-scholar"],
                    "policy": {
                        "confirmation_mode": "destructive_only",
                        "tool_overrides": [{"tool": "web.search", "requires_confirmation": False}],
                    },
                    "voice_defaults": {"wake_phrase": "research"},
                    "scope_rules": [{"scope": "workspace"}],
                    "buddy": {"species": "owl", "palette": "sepia", "silhouette": "round"},
                    "starter_commands": [{"template_key": "summarize"}],
                },
                {
                    "name": "Researcher",
                    "system_prompt": "Investigate carefully.",
                    "archetype_key": "researcher",
                    "voice_defaults": {"wake_phrase": "research"},
                    "setup": {"status": "not_started", "current_step": "archetype"},
                },
            ]
        )
        monkeypatch.setattr(client, "_request", mocked)

        summaries = await client.list_persona_archetypes()
        template = await client.get_persona_archetype("researcher")
        preview = await client.preview_persona_archetype("researcher")

        assert isinstance(summaries[0], ArchetypeSummary)
        assert summaries[0].key == "researcher"
        assert isinstance(template, ArchetypeTemplate)
        assert template.persona.name == "Researcher"
        assert template.starter_commands[0].template_key == "summarize"
        assert isinstance(preview, ArchetypePreviewResponse)
        assert preview.archetype_key == "researcher"
        assert preview.setup.current_step == "archetype"
        assert [call.args[:2] for call in mocked.await_args_list] == [
            ("GET", "/api/v1/persona/archetypes"),
            ("GET", "/api/v1/persona/archetypes/researcher"),
            ("GET", "/api/v1/persona/archetypes/researcher/preview"),
        ]

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
            ("GET", "/api/v1/persona/profiles/persona-1/exemplars", {"params": {"include_disabled": "false", "include_deleted": "false", "include_deleted_personas": "false", "limit": 100, "offset": 0}}),
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

    async def test_character_message_endpoint_wiring(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        message_payload = {
            "id": "msg-1",
            "conversation_id": "chat-1",
            "sender": "user",
            "content": "Hello",
            "timestamp": "2026-04-25T00:00:00Z",
            "version": 2,
        }
        mocked = AsyncMock(
            side_effect=[
                message_payload,
                {"messages": [message_payload], "total": 1, "limit": 25, "offset": 5},
                message_payload,
                {**message_payload, "content": "Updated", "metadata_extra": {"pinned": True}},
                None,
                {"messages": [message_payload], "total": 1, "limit": 10, "offset": 0},
            ]
        )
        monkeypatch.setattr(client, "_request", mocked)

        created = await client.create_character_message(
            "chat-1",
            CharacterMessageCreate(role="user", content="Hello"),
            scope_type="workspace",
            workspace_id="ws-1",
        )
        listed = await client.list_character_messages(
            "chat-1",
            limit=25,
            offset=5,
            include_deleted=True,
            include_metadata=True,
            scope_type="workspace",
            workspace_id="ws-1",
        )
        detail = await client.get_character_message("msg-1", include_metadata=True)
        updated = await client.update_character_message(
            "msg-1",
            CharacterMessageUpdate(content="Updated", pinned=True),
            expected_version=2,
        )
        await client.delete_character_message("msg-1", expected_version=3)
        results = await client.search_character_messages("chat-1", query="hello", limit=10)

        expected_calls = [
            (
                "POST",
                "/api/v1/chats/chat-1/messages",
                {
                    "json_data": {"role": "user", "content": "Hello"},
                    "params": {"scope_type": "workspace", "workspace_id": "ws-1"},
                },
            ),
            (
                "GET",
                "/api/v1/chats/chat-1/messages",
                {
                    "params": {
                        "limit": 25,
                        "offset": 5,
                        "include_deleted": "true",
                        "include_character_context": "false",
                        "format_for_completions": "false",
                        "include_tool_calls": "false",
                        "include_metadata": "true",
                        "include_message_ids": "false",
                        "scope_type": "workspace",
                        "workspace_id": "ws-1",
                    }
                },
            ),
            (
                "GET",
                "/api/v1/messages/msg-1",
                {"params": {"include_tool_calls": "false", "include_metadata": "true"}},
            ),
            (
                "PUT",
                "/api/v1/messages/msg-1",
                {"json_data": {"content": "Updated", "pinned": True}, "params": {"expected_version": 2}},
            ),
            ("DELETE", "/api/v1/messages/msg-1", {"params": {"expected_version": 3}}),
            (
                "GET",
                "/api/v1/chats/chat-1/messages/search",
                {"params": {"query": "hello", "limit": 10}},
            ),
        ]
        assert len(mocked.await_args_list) == len(expected_calls)
        for call_args, expected in zip(mocked.await_args_list, expected_calls):
            _assert_request_call(call_args, *expected)

        assert isinstance(created, CharacterMessageResponse)
        assert isinstance(listed, CharacterMessageListResponse)
        assert isinstance(detail, CharacterMessageResponse)
        assert isinstance(updated, CharacterMessageResponse)
        assert updated.metadata_extra == {"pinned": True}
        assert isinstance(results, CharacterMessageListResponse)

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

    async def test_character_chat_session_crud_endpoint_wiring(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"ok": True})
        monkeypatch.setattr(client, "_request", mocked)

        await client.create_character_chat_session(
            CharacterChatSessionCreate(character_id=12, title="Evening Chat"),
            seed_first_message=True,
            greeting_strategy="alternate_index",
            alternate_index=1,
        )
        await client.list_character_chat_sessions(
            character_id=12,
            character_scope="character",
            include_deleted=True,
            deleted_only=True,
            include_settings=True,
            include_message_counts=False,
            limit=7,
            offset=3,
            scope_type="workspace",
            workspace_id="ws-1",
        )
        await client.get_character_chat_session(
            "chat-1",
            include_settings=True,
            scope_type="workspace",
            workspace_id="ws-1",
        )
        await client.update_character_chat_session(
            "chat-1",
            CharacterChatSessionUpdate(title="Evening Chat 2", state="Resolved"),
            expected_version=4,
            scope_type="workspace",
            workspace_id="ws-1",
        )
        await client.delete_character_chat_session(
            "chat-1",
            expected_version=5,
            hard_delete=True,
            scope_type="workspace",
            workspace_id="ws-1",
        )
        await client.restore_character_chat_session(
            "chat-1",
            expected_version=6,
            scope_type="workspace",
            workspace_id="ws-1",
        )

        expected_calls = [
            (
                "POST",
                "/api/v1/chats/",
                {
                    "json_data": {
                        "character_id": 12,
                        "assistant_kind": "character",
                        "assistant_id": "12",
                        "title": "Evening Chat",
                    },
                    "params": {
                        "seed_first_message": "true",
                        "greeting_strategy": "alternate_index",
                        "alternate_index": 1,
                    },
                },
            ),
            (
                "GET",
                "/api/v1/chats/",
                {
                    "params": {
                        "character_id": 12,
                        "character_scope": "character",
                        "include_deleted": "true",
                        "deleted_only": "true",
                        "include_settings": "true",
                        "include_message_counts": "false",
                        "limit": 7,
                        "offset": 3,
                        "scope_type": "workspace",
                        "workspace_id": "ws-1",
                    }
                },
            ),
            (
                "GET",
                "/api/v1/chats/chat-1",
                {
                    "params": {
                        "include_settings": "true",
                        "scope_type": "workspace",
                        "workspace_id": "ws-1",
                    }
                },
            ),
            (
                "PUT",
                "/api/v1/chats/chat-1",
                {
                    "json_data": {"title": "Evening Chat 2", "state": "resolved"},
                    "params": {"expected_version": 4, "scope_type": "workspace", "workspace_id": "ws-1"},
                },
            ),
            (
                "DELETE",
                "/api/v1/chats/chat-1",
                {
                    "params": {
                        "expected_version": 5,
                        "hard_delete": "true",
                        "scope_type": "workspace",
                        "workspace_id": "ws-1",
                    }
                },
            ),
            (
                "POST",
                "/api/v1/chats/chat-1/restore",
                {
                    "params": {
                        "expected_version": 6,
                        "scope_type": "workspace",
                        "workspace_id": "ws-1",
                    }
                },
            ),
        ]
        assert len(mocked.await_args_list) == len(expected_calls)
        for call_args, expected in zip(mocked.await_args_list, expected_calls):
            _assert_request_call(call_args, *expected)

    async def test_character_chat_settings_export_and_diagnostics_endpoint_wiring(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"ok": True})
        monkeypatch.setattr(client, "_request", mocked)

        await client.get_chat_settings("chat-1", scope_type="workspace", workspace_id="ws-1")
        await client.update_chat_settings(
            "chat-1",
            ChatSettingsUpdate(settings={"authorNote": "Stay concise."}),
            scope_type="workspace",
            workspace_id="ws-1",
        )
        await client.export_chat_history(
            "chat-1",
            format="markdown",
            include_metadata=False,
            include_character=False,
            page=2,
            page_size=25,
        )
        await client.get_author_note_info("chat-1")
        await client.export_lorebook_diagnostics("chat-1", page=2, size=10, order="desc")

        expected_calls = [
            (
                "GET",
                "/api/v1/chats/chat-1/settings",
                {"params": {"scope_type": "workspace", "workspace_id": "ws-1"}},
            ),
            (
                "PUT",
                "/api/v1/chats/chat-1/settings",
                {
                    "json_data": {"settings": {"authorNote": "Stay concise."}},
                    "params": {"scope_type": "workspace", "workspace_id": "ws-1"},
                },
            ),
            (
                "GET",
                "/api/v1/chats/chat-1/export",
                {
                    "params": {
                        "format": "markdown",
                        "include_metadata": "false",
                        "include_character": "false",
                        "page": 2,
                        "page_size": 25,
                    }
                },
            ),
            ("GET", "/api/v1/chats/chat-1/author-note/info", {}),
            (
                "GET",
                "/api/v1/chats/chat-1/diagnostics/lorebook",
                {"params": {"page": 2, "size": 10, "order": "desc"}},
            ),
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
