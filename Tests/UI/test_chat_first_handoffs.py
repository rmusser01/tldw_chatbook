from __future__ import annotations

from pathlib import Path
import re
import tomllib
from unittest.mock import AsyncMock, Mock, patch

import pytest
from textual.app import App, ComposeResult
from textual.css.query import NoMatches
from textual.widgets import Button, Static, TextArea

from tldw_chatbook.Chat.chat_models import ChatSessionData
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload, HANDOFF_BODY_CHAR_LIMIT
from tldw_chatbook.config import CONFIG_TOML_CONTENT
from tldw_chatbook.Constants import TAB_CHAT
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Chat_Widgets.chat_handoff_card import ChatHandoffCard
from tldw_chatbook.Widgets.Chat_Widgets.chat_session import ChatSession
from tldw_chatbook.UX_Interop import (
    build_server_parity_fixture_payloads,
    build_server_parity_handoff_packet,
)
from tldw_chatbook.runtime_policy.types import RuntimeSourceState


def test_chat_handoff_ui_smoke_consumes_server_parity_fixture_payloads():
    fixtures = build_server_parity_fixture_payloads()

    assert fixtures["local"]["active_source"] == "local"
    assert fixtures["server"]["active_server_profile_id"] == "srv-primary"
    assert fixtures["unavailable_server"]["server_reachability"] == "unreachable"
    assert fixtures["auth_failure"]["reason_code"] == "auth_required"
    assert fixtures["unsupported_action"]["unsupported_reason_code"] == "server_contract_missing"
    assert fixtures["workspace_isolation"]["workspace_scope_id"] == "workspace-a"
    assert fixtures["sync_dry_run_report"]["write_enabled"] is False


def test_chat_handoff_packet_exposes_sections_ui_needs_without_screen_inference():
    packet = build_server_parity_handoff_packet(
        RuntimeSourceState(
            active_source="server",
            active_server_id="srv-primary",
            server_configured=True,
            server_reachability="reachable",
            server_auth_state="authenticated",
            last_known_server_label="Primary Server",
        ),
        workspace_scope_ids=("workspace-a",),
    )

    sections = packet["sections"]
    assert sections["active_server"]["active_server_profile_id"] == "srv-primary"
    assert sections["source_selector"]["source_options"]
    assert sections["sync"]["dry_run_only"] is True
    assert sections["workspace_isolation"][0]["workspace_scope_id"] == "workspace-a"
    assert "server_unavailable" in sections["error_contracts"]


def test_chat_tabs_are_enabled_by_default_for_handoff_capable_chat():
    generated_config = tomllib.loads(CONFIG_TOML_CONTENT)
    repo_config_path = Path(__file__).resolve().parents[2] / "config.toml"
    repo_config = tomllib.loads(repo_config_path.read_text(encoding="utf-8"))

    assert generated_config["chat_defaults"]["enable_tabs"] is True
    assert repo_config["chat_defaults"]["enable_tabs"] is True


def test_open_chat_with_handoff_stores_payload_and_navigates():
    app = Mock()
    app.pending_chat_handoff = None
    app.post_message = Mock()
    app.notify = Mock()
    payload = ChatHandoffPayload(source="notes", item_type="note", title="Note", body="Body")

    from tldw_chatbook.app import TldwCli

    with patch("tldw_chatbook.app.get_cli_setting", return_value=True):
        TldwCli.open_chat_with_handoff(app, payload)

    assert app.pending_chat_handoff is payload
    message = app.post_message.call_args.args[0]
    assert isinstance(message, NavigateToScreen)
    assert message.screen_name == TAB_CHAT


def test_open_chat_with_handoff_refuses_when_tabs_disabled():
    app = Mock()
    app.pending_chat_handoff = None
    app.post_message = Mock()
    app.notify = Mock()
    payload = ChatHandoffPayload(source="notes", item_type="note", title="Note", body="Body")

    from tldw_chatbook.app import TldwCli

    with patch("tldw_chatbook.app.get_cli_setting", return_value=False):
        TldwCli.open_chat_with_handoff(app, payload)

    assert app.pending_chat_handoff is None
    app.post_message.assert_not_called()
    app.notify.assert_called_once()


def test_open_chat_with_handoff_blocked_message_defaults_to_use_in_chat():
    """Legacy callers (MediaWindow_v2, search_rag_window, SearchWindow, the
    standalone Notes tab, Study/Skills/Watchlists/Personas screens) don't
    pass ``action_label`` and must keep seeing today's exact wording -- the
    shared ``open_chat_with_handoff`` blocked-tabs gate is reachable from
    many destinations whose own button still reads "Use in Chat" (UX wave
    M2 investigation: this message is one shared string, not per-caller).
    """
    app = Mock()
    app.pending_chat_handoff = None
    app.post_message = Mock()
    app.notify = Mock()
    payload = ChatHandoffPayload(source="notes", item_type="note", title="Note", body="Body")

    from tldw_chatbook.app import TldwCli

    with patch("tldw_chatbook.app.get_cli_setting", return_value=False):
        TldwCli.open_chat_with_handoff(app, payload)

    app.notify.assert_called_once()
    assert app.notify.call_args.args[0] == "Use in Chat requires chat tabs to be enabled."


def test_open_chat_with_handoff_blocked_message_honors_caller_action_label():
    """The Library screen's four handoff call sites (notes/media/
    conversations/hub) pass ``action_label="Use in Console"`` since every
    Library "Use in"-style button already reads "Use in Console" -- the
    blocked-tabs notify must match the button the user actually pressed
    instead of always saying "Chat" (UX wave M2).
    """
    app = Mock()
    app.pending_chat_handoff = None
    app.post_message = Mock()
    app.notify = Mock()
    payload = ChatHandoffPayload(source="library", item_type="media", title="Media", body="Body")

    from tldw_chatbook.app import TldwCli

    with patch("tldw_chatbook.app.get_cli_setting", return_value=False):
        TldwCli.open_chat_with_handoff(app, payload, action_label="Use in Console")

    app.notify.assert_called_once()
    assert app.notify.call_args.args[0] == "Use in Console requires chat tabs to be enabled."


@pytest.mark.asyncio
async def test_chat_screen_consumes_pending_handoff_into_fresh_ephemeral_tab():
    payload = ChatHandoffPayload(
        source="workspace",
        item_type="workspace-source",
        title="Transcript",
        body="Body",
        source_id="source-1",
        runtime_backend="server",
        source_owner="workspace",
        source_selector_state="workspace",
        active_server_profile_id="srv-primary",
        discovery_owner="workspace",
        discovery_entity_id="source-1",
        scope_type="workspace",
        workspace_id="workspace-1",
        backend_contracts={"workspace_isolation": {"workspace_scope_id": "workspace-1"}},
    )
    app = Mock()
    app.pending_chat_handoff = payload
    app.notify = Mock()

    session = Mock()
    session.session_data = ChatSessionData(tab_id="tab-1")
    tab_container = Mock()
    tab_container.create_new_tab = AsyncMock(return_value="tab-1")
    tab_container.sessions = {"tab-1": session}
    tab_container.switch_to_tab_async = AsyncMock()

    screen = ChatScreen(app)
    screen.chat_window = Mock()
    screen._get_tab_container = Mock(return_value=tab_container)
    screen._apply_handoff_to_chat_session = AsyncMock()

    await screen._consume_pending_chat_handoff()

    session_data = tab_container.create_new_tab.await_args.kwargs["session_data"]
    assert session_data.conversation_id is None
    assert session_data.is_ephemeral is True
    assert session_data.runtime_backend == "server"
    assert session_data.handoff_payload.source_selector_state == "workspace"
    assert session_data.handoff_payload.active_server_profile_id == "srv-primary"
    assert session_data.scope_type == "workspace"
    assert session_data.workspace_id == "workspace-1"
    assert session_data.handoff_payload.title == "Transcript"
    assert app.pending_chat_handoff is None


def test_chat_screen_handoff_session_data_uses_unique_valid_tab_ids():
    payload = ChatHandoffPayload(
        source="notes",
        item_type="note",
        title="Plan",
        body="Body",
    )
    screen = ChatScreen(Mock())

    first = screen._session_data_for_handoff(payload)
    second = screen._session_data_for_handoff(payload)

    assert re.fullmatch(r"[a-f0-9]{8}", first.tab_id)
    assert re.fullmatch(r"[a-f0-9]{8}", second.tab_id)
    assert first.tab_id != second.tab_id


def test_chat_screen_start_chat_handoff_binds_character_session_identity():
    """Personas Start Chat should create character-bound Console sessions."""
    payload = ChatHandoffPayload(
        source="personas",
        item_type="character-card",
        title="Detective Sam (character)",
        body="Name: Detective Sam\nDescription: Methodical investigator",
        source_id="1",
        suggested_prompt="Respond as Detective Sam.",
        metadata={
            "intent": "start_chat",
            "selected_kind": "character",
            "selected_record_id": "1",
            "selected_name": "Detective Sam",
            "selected_target_id": "local:character:1",
        },
    )
    screen = ChatScreen(Mock())

    session_data = screen._session_data_for_handoff(payload)

    assert session_data.character_id == 1
    assert session_data.character_name == "Detective Sam"
    assert session_data.assistant_kind == "character"
    assert session_data.assistant_id == "1"
    assert session_data.discovery_owner == "ccp_character"
    assert session_data.discovery_entity_id == "1"


def test_chat_screen_start_chat_handoff_preserves_numeric_zero_character_id():
    """Character handoff metadata should not treat integer zero as missing."""
    payload = ChatHandoffPayload(
        source="personas",
        item_type="character-card",
        title="Zero (character)",
        body="Name: Zero",
        source_id="0",
        suggested_prompt="Respond as Zero.",
        metadata={
            "intent": "start_chat",
            "selected_kind": "character",
            "selected_record_id": 0,
            "selected_name": "Zero",
        },
    )
    screen = ChatScreen(Mock())

    session_data = screen._session_data_for_handoff(payload)

    assert session_data.character_id == 0
    assert session_data.character_name == "Zero"
    assert session_data.assistant_kind == "character"
    assert session_data.assistant_id == "0"
    assert session_data.discovery_owner == "ccp_character"


@pytest.mark.asyncio
async def test_chat_screen_pending_handoff_consumer_is_reentrant_safe():
    payload = ChatHandoffPayload(
        source="notes",
        item_type="note",
        title="Plan",
        body="Body",
    )
    app = Mock()
    app.pending_chat_handoff = payload
    app.notify = Mock()

    session = Mock()
    session.session_data = ChatSessionData(tab_id="tab-1")
    tab_container = Mock()
    tab_container.sessions = {"tab-1": session}
    tab_container.switch_to_tab_async = AsyncMock()

    screen = ChatScreen(app)
    screen.chat_window = Mock()
    screen._get_tab_container = Mock(return_value=tab_container)
    screen._apply_handoff_to_chat_session = AsyncMock()

    nested_called = False

    async def create_new_tab(*, session_data):
        nonlocal nested_called
        if not nested_called:
            nested_called = True
            await screen._consume_pending_chat_handoff()
        return "tab-1"

    tab_container.create_new_tab = AsyncMock(side_effect=create_new_tab)

    await screen._consume_pending_chat_handoff()

    assert tab_container.create_new_tab.await_count == 1
    assert app.pending_chat_handoff is None


def test_chat_handoff_payload_round_trip_preserves_runtime_scope_and_metadata():
    payload = ChatHandoffPayload(
        source="workspace",
        item_type="workspace-source",
        title="Transcript",
        body="source body",
        body_truncated=False,
        content_ref="workspace:workspace-1:source:source-1",
        source_id="source-1",
        display_summary="A workspace source",
        suggested_prompt="Use this source.",
        runtime_backend="server",
        source_owner="workspace",
        source_selector_state="workspace",
        active_server_profile_id="srv-primary",
        discovery_owner="workspace",
        discovery_entity_id="source-1",
        scope_type="workspace",
        workspace_id="workspace-1",
        backend_contracts={
            "workspace_isolation": {"workspace_scope_id": "workspace-1"},
            "active_server": {
                "active_server_profile_id": "srv-primary",
                "credential_source": "keyring:chatbook:server:srv-primary:access",
            },
        },
        unsupported_reports=[
            {
                "operation_id": "notes.graph.unsupported.workspace",
                "source": "workspace",
                "supported": False,
                "reason_code": "scope_not_supported",
                "user_message": "Workspace graph is not available.",
                "affected_action_ids": ["notes.graph.list.server"],
            }
        ],
        sync_dry_run_report={"dry_run": True, "write_enabled": False},
        metadata={"score": 0.87, "url": "https://example.com"},
    )

    restored = ChatHandoffPayload.from_dict(payload.to_dict())

    assert restored is not None
    assert restored.source == "workspace"
    assert restored.item_type == "workspace-source"
    assert restored.body_truncated is False
    assert restored.content_ref == "workspace:workspace-1:source:source-1"
    assert restored.runtime_backend == "server"
    assert restored.source_owner == "workspace"
    assert restored.source_selector_state == "workspace"
    assert restored.active_server_profile_id == "srv-primary"
    assert restored.scope_type == "workspace"
    assert restored.workspace_id == "workspace-1"
    assert restored.backend_contracts["workspace_isolation"]["workspace_scope_id"] == "workspace-1"
    assert "credential_source" not in restored.backend_contracts["active_server"]
    assert restored.unsupported_reports[0]["reason_code"] == "scope_not_supported"
    assert restored.sync_dry_run_report["write_enabled"] is False
    assert restored.metadata["score"] == 0.87


def test_chat_handoff_payload_persistence_redacts_secrets_and_normalizes_tuples():
    payload = ChatHandoffPayload(
        source="notes",
        item_type="note",
        title="Secret-adjacent",
        body="Body",
        backend_contracts={
            "active_server": {
                "active_server_profile_id": "srv-primary",
                "credential_source": "keyring:chatbook:server:srv-primary:access",
            },
            "source_selector": {"source_options": ({"source": "local"}, {"source": "server"})},
        },
        sync_dry_run_report={"conflict_ids": ("conflict-1",)},
    )

    data = payload.to_dict()

    assert "credential_source" not in data["backend_contracts"]["active_server"]
    assert data["backend_contracts"]["source_selector"]["source_options"] == [
        {"source": "local"},
        {"source": "server"},
    ]
    assert data["sync_dry_run_report"]["conflict_ids"] == ["conflict-1"]


def test_chat_handoff_payload_persistence_omits_nulls_and_redacts_model_metadata():
    payload = ChatHandoffPayload(
        source="notes",
        item_type="note",
        title="Metadata redaction",
        body="Body",
        metadata={"api_key": "should-not-persist", "safe": "ok"},
    )

    data = payload.to_dict()
    context = payload.model_context_block()

    assert "content_ref" not in data
    assert "api_key" not in data["metadata"]
    assert data["metadata"] == {"safe": "ok"}
    assert "api_key" not in context
    assert "should-not-persist" not in context
    assert "- safe: ok" in context


def test_chat_handoff_payload_model_context_formats_evidence_bundle_readably():
    payload = ChatHandoffPayload(
        source="search-rag",
        item_type="rag-result",
        title="Incident Review",
        body="Retrieved content",
        metadata={
            "evidence_bundle": {
                "bundle_id": "library-rag:incident",
                "query": "Why did the incident happen?",
                "status": "available",
                "references": [
                    {
                        "evidence_id": "S1",
                        "source_id": "note-42",
                        "source_type": "note",
                        "title": "Incident Review",
                        "snippet": "Expired credential caused the incident.",
                        "authority_label": "Source authority: local",
                        "status": "available",
                    }
                ],
            }
        },
    )

    context = payload.model_context_block()

    assert "[Staged evidence]" in context
    assert "Evidence bundle: library-rag:incident" in context
    assert "Evidence status: available" in context
    assert "[S1] Incident Review (note-42) - Source authority: local - available" in context
    assert "Snippet: Expired credential caused the incident." in context
    assert "evidence_bundle:" not in context


def test_chat_handoff_payload_from_source_content_caps_persisted_body():
    payload = ChatHandoffPayload.from_source_content(
        source="media",
        item_type="media",
        title="Long transcript",
        body="x" * (HANDOFF_BODY_CHAR_LIMIT + 1),
        content_ref="media:record-1",
    )

    assert len(payload.body) == HANDOFF_BODY_CHAR_LIMIT
    assert payload.body_truncated is True
    assert payload.content_ref == "media:record-1"


def test_chat_handoff_payload_from_source_content_preserves_upstream_truncation_flag():
    payload = ChatHandoffPayload.from_source_content(
        source="media",
        item_type="media",
        title="Already summarized transcript",
        body="short summary",
        body_truncated=True,
    )

    assert payload.body == "short summary"
    assert payload.body_truncated is True


def test_handoff_card_uses_status_source_title_and_metadata():
    from tldw_chatbook.Widgets.Chat_Widgets.chat_handoff_card import ChatHandoffCard

    payload = ChatHandoffPayload(
        source="search-web",
        item_type="web-result",
        title="Article",
        body="Article snippet",
        display_summary="Search result summary",
        runtime_backend="server",
        source_owner="server",
        source_selector_state="server",
        active_server_profile_id="srv-primary",
        sync_dry_run_report={"dry_run": True, "write_enabled": False},
        metadata={"url": "https://example.com", "score": 0.5},
    )

    card = ChatHandoffCard(payload)
    text = card.render_text()

    assert "Context staged" in text
    assert "Web Search" in text
    assert "Article" in text
    assert "Source: Server source" in text
    assert "Server: srv-primary" in text
    assert "Sync: dry-run only" in text
    assert "https://example.com" in text


@pytest.mark.asyncio
async def test_handoff_card_renders_user_controlled_text_as_plain_text():
    class HandoffCardHarness(App):
        def compose(self) -> ComposeResult:
            yield ChatHandoffCard(
                ChatHandoffPayload(
                    source="library",
                    item_type="library-source-snapshot",
                    title="[bold red]Injected title[/]",
                    body="[green]Injected body[/]",
                    display_summary="[blue]Injected summary[/]",
                    metadata={"note_titles": ["[reverse]Injected note[/]"]},
                )
            )

    async with HandoffCardHarness().run_test(size=(120, 20)) as pilot:
        card_body = pilot.app.query_one(".chat-handoff-card-body", Static)
        card = pilot.app.query_one(ChatHandoffCard)
        renderable = card_body.renderable

        assert getattr(renderable, "plain", "") == card.render_text()
        assert getattr(renderable, "spans", []) == []
        assert "[bold red]Injected title[/]" in renderable.plain
        assert "[blue]Injected summary[/]" in renderable.plain
        assert "[reverse]Injected note[/]" in renderable.plain


def test_handoff_card_surfaces_backend_contract_recovery_without_implying_write_sync():
    fixtures = build_server_parity_fixture_payloads()
    payload = ChatHandoffPayload(
        source="workspace",
        item_type="workspace-source",
        title="Workspace Transcript",
        body="Transcript body",
        runtime_backend="server",
        source_owner="workspace",
        source_selector_state="workspace",
        active_server_profile_id="srv-primary",
        workspace_id="workspace-a",
        unsupported_reports=[fixtures["unsupported_action"]],
        sync_dry_run_report=fixtures["sync_dry_run_report"],
    )

    card = ChatHandoffCard(payload)
    text = card.render_text()

    assert "Sync: dry-run only" in text
    assert "Resolve workspace conflicts before syncing." in text
    assert "Write sync is not enabled." in text
    assert "Unsupported actions: 1" in text
    assert "Server first-class conversation creation is not available." in text
    assert "sync enabled" not in text.lower()
    assert "will sync automatically" not in text.lower()


def test_handoff_card_ignores_malformed_sync_dry_run_report_copy():
    payload = {
        "source": "workspace",
        "item_type": "workspace-source",
        "title": "Workspace Transcript",
        "body": "Transcript body",
        "sync_dry_run_report": ["malformed-report"],
    }

    text = ChatHandoffCard(payload).render_text()

    assert "Sync: dry-run only" in text
    assert "Write sync is not enabled." not in text


class _HandoffSessionHarness(App):
    def __init__(self, session_data: ChatSessionData) -> None:
        super().__init__()
        self._session_data = session_data
        self.host = Mock()
        self.host.chat_enhanced_mode = False
        self.host.notify = Mock()

    def compose(self) -> ComposeResult:
        yield ChatSession(self.host, self._session_data)


@pytest.mark.asyncio
async def test_user_can_clear_staged_handoff_context_before_send():
    payload = ChatHandoffPayload(
        source="notes",
        item_type="note",
        title="Plan",
        body="Body",
        suggested_prompt="Use this note.",
    )
    session_data = ChatSessionData(tab_id="tab1", handoff_payload=payload)
    app = _HandoffSessionHarness(session_data)
    app.host._current_chat_handoff_payload = payload

    async with app.run_test() as pilot:
        session = pilot.app.query_one(ChatSession)
        await session.mount_handoff_card(payload)
        session.set_draft_text(payload.default_prompt())
        await pilot.pause(0.05)

        assert pilot.app.query_one(ChatHandoffCard) is not None
        clear_button = pilot.app.query_one("#clear-chat-handoff-context-tab1", Button)

        clear_button.press()
        await pilot.pause(0.05)

        with pytest.raises(NoMatches):
            pilot.app.query_one(ChatHandoffCard)
        assert session.session_data.handoff_payload is None
        assert app.host._current_chat_handoff_payload is None
        assert pilot.app.query_one("#chat-input-tab1", TextArea).text == ""


@pytest.mark.asyncio
async def test_clear_staged_handoff_context_preserves_unrelated_global_payload():
    payload = ChatHandoffPayload(
        source="notes",
        item_type="note",
        title="Plan",
        body="Body",
        suggested_prompt="Use this note.",
    )
    other_payload = ChatHandoffPayload(
        source="media",
        item_type="media",
        title="Other tab",
        body="Other body",
        suggested_prompt="Use this media.",
    )
    session_data = ChatSessionData(tab_id="tab1", handoff_payload=payload)
    app = _HandoffSessionHarness(session_data)
    app.host._current_chat_handoff_payload = other_payload

    async with app.run_test() as pilot:
        session = pilot.app.query_one(ChatSession)
        await session.mount_handoff_card(payload)
        session.set_draft_text(payload.default_prompt())
        await pilot.pause(0.05)

        pilot.app.query_one("#clear-chat-handoff-context-tab1", Button).press()
        await pilot.pause(0.05)

        assert app.host._current_chat_handoff_payload is other_payload


@pytest.mark.asyncio
async def test_clear_staged_handoff_context_keeps_sent_handoff_cards():
    staged_payload = ChatHandoffPayload(
        source="notes",
        item_type="note",
        title="Plan",
        body="Body",
        suggested_prompt="Use this note.",
    )
    sent_payload = ChatHandoffPayload(
        source="media",
        item_type="media",
        title="Already sent",
        body="Sent body",
        status="sent",
    )
    session_data = ChatSessionData(tab_id="tab1", handoff_payload=staged_payload)
    app = _HandoffSessionHarness(session_data)

    async with app.run_test() as pilot:
        session = pilot.app.query_one(ChatSession)
        await session.get_chat_log().mount(ChatHandoffCard(sent_payload))
        await session.mount_handoff_card(staged_payload)
        session.set_draft_text(staged_payload.default_prompt())
        await pilot.pause(0.05)

        pilot.app.query_one("#clear-chat-handoff-context-tab1", Button).press()
        await pilot.pause(0.05)

        cards = list(pilot.app.query(ChatHandoffCard))
        assert len(cards) == 1
        assert cards[0].payload.status == "sent"
        assert cards[0].payload.title == "Already sent"


@pytest.mark.asyncio
async def test_apply_handoff_mounts_card_and_prefills_tab_input():
    payload = ChatHandoffPayload(
        source="notes",
        item_type="note",
        title="Plan",
        body="Body",
        suggested_prompt="Use this note.",
    )
    session = Mock()
    session.mount_handoff_card = AsyncMock()
    session.set_draft_text = Mock()

    screen = ChatScreen(Mock())

    await screen._apply_handoff_to_chat_session(session, payload)

    session.mount_handoff_card.assert_awaited_once_with(payload)
    session.set_draft_text.assert_called_once_with("Use this note.")


def test_handoff_payload_formats_model_prompt_with_context_and_user_prompt():
    payload = ChatHandoffPayload(
        source="media",
        item_type="media",
        title="Lecture",
        body="Transcript body",
        metadata={"url": "https://example.com"},
    )

    prompt = payload.format_for_model("Summarize it.")

    assert "[Staged context]" in prompt
    assert "Transcript body" in prompt
    assert "[User prompt]" in prompt
    assert "Summarize it." in prompt


def test_apply_current_handoff_context_wraps_unsent_payload_only():
    from tldw_chatbook.Event_Handlers.Chat_Events.chat_events import apply_current_handoff_context

    payload = ChatHandoffPayload(
        source="notes",
        item_type="note",
        title="Plan",
        body="Body",
    )
    app = Mock()
    app._current_chat_handoff_payload = payload

    wrapped = apply_current_handoff_context(app, "Use this.")

    assert "[Staged context]" in wrapped
    assert "Body" in wrapped
    assert "[User prompt]\nUse this." in wrapped

    payload.status = "sent"
    assert apply_current_handoff_context(app, "Use this again.") == "Use this again."


def test_apply_current_handoff_context_ignores_mock_auto_attributes():
    from tldw_chatbook.Event_Handlers.Chat_Events.chat_events import apply_current_handoff_context

    app = AsyncMock()

    assert apply_current_handoff_context(app, "Use this.") == "Use this."
