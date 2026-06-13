# Use In Chat Handoffs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement current-dev `Use in Chat` handoffs so Notes, Workspaces, Media, RAG Search, and Web Search can stage one selected item into a fresh Chat session with visible context, source-honest backend parity state, and no auto-send.

**Architecture:** Use one app-owned pending handoff seam, one serializable `ChatHandoffPayload`, backend-owned `UX_Interop`/`runtime_policy` contract snapshots, and session-scoped Chat persistence. Source screens identify the selected item and visible content; backend parity contracts own active server status, source authority, unsupported actions, workspace isolation, and sync dry-run state. `TldwCli` owns navigation; `ChatScreen` consumes the pending payload after restore, creates a new tab, renders a handoff card, prefills the draft, and ensures the first user send includes staged context.

**Tech Stack:** Python 3.11+, Textual, dataclasses, existing `ChatSessionData` / `TabState` / `ChatTabContainer`, pytest

---

## Scope Check

This plan intentionally implements the single-item handoff slice from the rebaselined spec:

- Shared handoff payload and Chat persistence.
- App-owned pending handoff navigation.
- Chat destination UI and send-time context injection.
- Notes and Workspace source adapters.
- Media source adapter.
- RAG Search and Web Search source adapters.
- Backend-parity contract consumption for active server/source selectors, unsupported-action reports, workspace isolation, and sync dry-run diagnostics.
- Focused tests for the contracts and adapters.

This plan does not implement multi-select packaging, Chatbooks, Study/Flashcards/Quizzes handoffs, CCP/persona launch rewrites, automatic write sync, queued mutation replay, local CRUD for remote-only domains, or broader UI layout redesigns.

## Source Spec

- `Docs/superpowers/specs/2026-04-21-use-in-chat-handoffs-design.md`
- `Docs/superpowers/handoffs/2026-04-30-backend-parity-ux-handoff.md`

## File Map

- Create: `tldw_chatbook/Chat/chat_handoff_models.py`
  Responsibility: Own `ChatHandoffPayload`, serialization, prompt formatting, source labels, and sent/staged state.
- Modify: `tldw_chatbook/Chat/chat_models.py`
  Responsibility: Persist handoff state in live `ChatSessionData`.
- Modify: `tldw_chatbook/UI/Screens/chat_screen_state.py`
  Responsibility: Persist handoff state in saved `TabState`.
- Modify: `tldw_chatbook/Chat/tabs/tab_state_manager.py`
  Responsibility: Keep tab runtime state aware of handoff payloads for send-time operations.
- Modify: `config.toml` and generated defaults in `tldw_chatbook/config.py`
  Responsibility: Enable chat tabs by default for new/generated config while preserving explicit user opt-out.
- Create: `Tests/UI/test_chat_first_handoffs.py`
  Responsibility: Contract, app seam, Chat destination, card, and adapter-focused handoff tests.
- Modify: `Tests/UX_Interop/test_server_parity_contracts.py`
  Responsibility: Only change when the backend-owned contract shape changes; UI work should consume existing fixture payloads.
- Modify: `Tests/UI/test_chat_screen_state.py`
  Responsibility: Extend existing serialization tests for handoff payload preservation.
- Modify: `Tests/UI/test_chat_tab_container.py`
  Responsibility: Prove handoff sessions do not trigger persisted-conversation tab reuse.

- Modify: `tldw_chatbook/app.py`
  Responsibility: Add `pending_chat_handoff`, tab-availability guard, and `open_chat_with_handoff()`.
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
  Responsibility: Consume pending handoffs after normal restore, create fresh sessions, sync state, and mount handoff UI.
- Create: `tldw_chatbook/Widgets/Chat_Widgets/chat_handoff_card.py`
  Responsibility: Render staged/sent source context as a distinct non-message card.
- Modify: `tldw_chatbook/Widgets/Chat_Widgets/chat_session.py`
  Responsibility: Provide per-tab seams for mounting the handoff card and preloading draft text.
- Modify: `tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py`
  Responsibility: Apply staged handoff context to the outgoing model prompt.
- Modify: `tldw_chatbook/Event_Handlers/Chat_Events/chat_events_tabs.py`
  Responsibility: Bind the active session handoff to the send handler and mark it sent after successful first send.
- Modify: `Tests/Event_Handlers/Chat_Events/test_chat_events_tabs.py`
  Responsibility: Verify send-time staged context behavior.

- Modify: `tldw_chatbook/UI/Screens/notes_screen.py`
  Responsibility: Build Notes/Workspace payloads and handle `Use in Chat`.
- Modify: `tldw_chatbook/Widgets/Note_Widgets/notes_sidebar_right.py`
  Responsibility: Add note-context `Use in Chat` placement.
- Modify: `tldw_chatbook/Widgets/Note_Widgets/workspace_context_panel.py`
  Responsibility: Add workspace/source/artifact `Use in Chat` placements.
- Modify: `Tests/UI/test_notes_screen.py`
  Responsibility: Cover local, server, workspace note, workspace details, source, artifact, and dirty-state payloads.

- Modify: `tldw_chatbook/UI/MediaWindow_v2.py`
  Responsibility: Build Media payloads from hydrated detail and handle viewer action.
- Modify: `tldw_chatbook/Widgets/Media/media_viewer_panel.py`
  Responsibility: Add viewer-level `Use in Chat` action event.
- Modify: Media UI tests near existing coverage or create `Tests/UI/test_media_handoffs.py`
  Responsibility: Cover hydrated detail, sparse body, runtime backend, and disabled no-selection behavior.

- Modify: `tldw_chatbook/UI/Views/RAGSearch/search_result.py`
  Responsibility: Add per-result `Use in Chat` event.
- Modify: `tldw_chatbook/UI/Views/RAGSearch/search_rag_window.py`
  Responsibility: Normalize RAG and include-web card payloads.
- Modify: `tldw_chatbook/UI/SearchWindow.py`
  Responsibility: Cardify dedicated Web Search results and forward them through the same `Use in Chat` path.
- Create: `Tests/UI/test_search_handoffs.py`
  Responsibility: Cover RAG card events, Web result payloads, and dedicated Web Search behavior.

- Consume: `tldw_chatbook/UX_Interop/server_parity_contracts.py`
  Responsibility: Use `build_server_parity_handoff_packet()` and `build_server_parity_fixture_payloads()` as backend-owned source-authority smoke inputs.
- Consume: `tldw_chatbook/runtime_policy/domain_edge_contracts.py`
  Responsibility: Use domain authority, source selector states, workspace isolation requirements, and required unsupported reason codes.
- Consume: `tldw_chatbook/runtime_policy/unsupported_capabilities.py`
  Responsibility: Validate unsupported-action reports before using them for disabled states or inline explanations.
- Consume: `tldw_chatbook/Sync_Interop/*`
  Responsibility: Render sync reports as dry-run diagnostics only; do not imply write sync or local mirror completion.

## Task 0: Baseline Backend-Parity Handoff Contracts

**Files:**
- Create: `Tests/UI/test_chat_first_handoffs.py`
- Consume: `tldw_chatbook/UX_Interop/server_parity_contracts.py`
- Consume: `tldw_chatbook/runtime_policy/domain_edge_contracts.py`
- Consume: `tldw_chatbook/runtime_policy/unsupported_capabilities.py`

- [ ] **Step 1: Write UI smoke tests that consume backend-owned fixtures**

Add to `Tests/UI/test_chat_first_handoffs.py`:

```python
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
```

- [ ] **Step 2: Run the backend-parity fixture smoke tests**

Run: `pytest Tests/UX_Interop/test_server_parity_contracts.py Tests/UI/test_chat_first_handoffs.py -q`

Expected: PASS once the smoke tests are added. If `Tests/UX_Interop/test_server_parity_contracts.py` fails, fix or coordinate that backend-owned contract before adding UI-specific handoff wiring.

- [ ] **Step 3: Record the contract rules in UI implementation notes**

Before coding UI adapters, document these invariants in the test file or helper comments:

- UI consumes `UX_Interop` and `runtime_policy` contracts; it does not rebuild active server/auth/source state from raw config.
- Local, server, workspace, and remote-only states remain visually distinct.
- Unsupported reports drive disabled/hide/explanation states.
- Sync reports are dry-run diagnostics only; no write sync or mirror completion is implied.
- Workspace records require `workspace_id` and workspace isolation metadata when available.
- `source_selector_state="workspace"` comes from domain/workspace contracts; do not attempt to create `RuntimeSourceState(active_source="workspace")`.
- Persisted payload snapshots must be secret-redacted, JSON/TOML-safe, and size-bounded.

## Task 1: Add The Shared Handoff Contract And Persisted Session Fields

**Files:**
- Create: `tldw_chatbook/Chat/chat_handoff_models.py`
- Modify: `tldw_chatbook/Chat/chat_models.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen_state.py`
- Modify: `tldw_chatbook/Chat/tabs/tab_state_manager.py`
- Create: `Tests/UI/test_chat_first_handoffs.py`
- Modify: `Tests/UI/test_chat_screen_state.py`

- [ ] **Step 1: Write failing payload serialization tests**

Add to `Tests/UI/test_chat_first_handoffs.py`:

```python
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload, HANDOFF_BODY_CHAR_LIMIT


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
```

- [ ] **Step 2: Write failing session-state preservation tests**

Extend `Tests/UI/test_chat_screen_state.py`:

```python
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload


def test_chat_session_data_round_trip_preserves_handoff_payload():
    payload = ChatHandoffPayload(
        source="notes",
        item_type="note",
        title="Planning note",
        body="Plan content",
        source_id="note-1",
    )
    session = ChatSessionData(tab_id="tab-1", handoff_payload=payload)

    restored = ChatSessionData.from_dict(session.to_dict())

    assert restored.handoff_payload is not None
    assert restored.handoff_payload.title == "Planning note"


def test_tab_state_round_trip_preserves_handoff_payload():
    payload = ChatHandoffPayload(
        source="media",
        item_type="media",
        title="Video",
        body="Transcript",
        source_id="media-1",
    )
    tab_state = TabState(tab_id="tab-1", title="Media: Video", handoff_payload=payload)

    restored = TabState.from_dict(tab_state.to_dict())

    assert restored.handoff_payload is not None
    assert restored.handoff_payload.source == "media"
```

- [ ] **Step 3: Run focused tests to verify they fail**

Run: `pytest Tests/UI/test_chat_first_handoffs.py Tests/UI/test_chat_screen_state.py -q`

Expected: FAIL because `chat_handoff_models.py` and `handoff_payload` fields do not exist.

- [ ] **Step 4: Implement `ChatHandoffPayload`**

Create `tldw_chatbook/Chat/chat_handoff_models.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ChatHandoffPayload:
    source: str
    item_type: str
    title: str
    body: str
    body_truncated: bool = False
    content_ref: Optional[str] = None
    source_id: Optional[str] = None
    display_summary: str = ""
    suggested_prompt: str = ""
    runtime_backend: str = "local"
    source_owner: str = "local"
    source_selector_state: str = "local"
    active_server_profile_id: Optional[str] = None
    discovery_owner: str = "general_chat"
    discovery_entity_id: Optional[str] = None
    scope_type: Optional[str] = None
    workspace_id: Optional[str] = None
    backend_contracts: Dict[str, Any] = field(default_factory=dict)
    unsupported_reports: list[Dict[str, Any]] = field(default_factory=list)
    sync_dry_run_report: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "staged"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "item_type": self.item_type,
            "title": self.title,
            "body": self.body,
            "body_truncated": self.body_truncated,
            "content_ref": self.content_ref,
            "source_id": self.source_id,
            "display_summary": self.display_summary,
            "suggested_prompt": self.suggested_prompt,
            "runtime_backend": self.runtime_backend,
            "source_owner": self.source_owner,
            "source_selector_state": self.source_selector_state,
            "active_server_profile_id": self.active_server_profile_id,
            "discovery_owner": self.discovery_owner,
            "discovery_entity_id": self.discovery_entity_id,
            "scope_type": self.scope_type,
            "workspace_id": self.workspace_id,
            "backend_contracts": _json_safe_contract_snapshot(self.backend_contracts or {}),
            "unsupported_reports": _json_safe_contract_snapshot(self.unsupported_reports or []),
            "sync_dry_run_report": _json_safe_contract_snapshot(self.sync_dry_run_report) if self.sync_dry_run_report else None,
            "metadata": _json_safe_contract_snapshot(self.metadata or {}),
            "status": self.status,
        }

    @classmethod
    def from_source_content(cls, *, body: str, content_ref: Optional[str] = None, **kwargs: Any) -> "ChatHandoffPayload":
        body_text = str(body or "")
        body_truncated = len(body_text) > HANDOFF_BODY_CHAR_LIMIT
        if body_truncated:
            body_text = body_text[:HANDOFF_BODY_CHAR_LIMIT]
        return cls(
            body=body_text,
            body_truncated=body_truncated,
            content_ref=content_ref,
            **kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | "ChatHandoffPayload" | None) -> Optional["ChatHandoffPayload"]:
        if data is None:
            return None
        if isinstance(data, cls):
            return cls(**data.to_dict())
        return cls(
            source=str(data.get("source") or "unknown"),
            item_type=str(data.get("item_type") or "item"),
            title=str(data.get("title") or "Untitled"),
            body=str(data.get("body") or ""),
            body_truncated=bool(data.get("body_truncated", False)),
            content_ref=data.get("content_ref"),
            source_id=data.get("source_id"),
            display_summary=str(data.get("display_summary") or ""),
            suggested_prompt=str(data.get("suggested_prompt") or ""),
            runtime_backend=str(data.get("runtime_backend") or "local"),
            source_owner=str(data.get("source_owner") or "local"),
            source_selector_state=str(data.get("source_selector_state") or data.get("runtime_backend") or "local"),
            active_server_profile_id=data.get("active_server_profile_id"),
            discovery_owner=str(data.get("discovery_owner") or "general_chat"),
            discovery_entity_id=data.get("discovery_entity_id"),
            scope_type=data.get("scope_type"),
            workspace_id=data.get("workspace_id"),
            backend_contracts=_json_safe_contract_snapshot(data.get("backend_contracts") or {}),
            unsupported_reports=_json_safe_contract_snapshot(data.get("unsupported_reports") or []),
            sync_dry_run_report=_json_safe_contract_snapshot(data["sync_dry_run_report"]) if data.get("sync_dry_run_report") else None,
            metadata=_json_safe_contract_snapshot(data.get("metadata") or {}),
            status=str(data.get("status") or "staged"),
        )

    def default_prompt(self) -> str:
        return self.suggested_prompt.strip() or "Help me use this context."

    def model_context_block(self) -> str:
        metadata_lines = []
        for key, value in sorted((self.metadata or {}).items()):
            if value not in (None, ""):
                metadata_lines.append(f"- {key}: {value}")
        metadata = "\n".join(metadata_lines)
        return (
            "[Staged context]\n"
            f"Source: {self.source}\n"
            f"Item type: {self.item_type}\n"
            f"Title: {self.title}\n"
            f"Source ID: {self.source_id or 'unknown'}\n"
            f"Content ref: {self.content_ref or 'none'}\n"
            f"Body truncated: {self.body_truncated}\n"
            f"Source owner: {self.source_owner}\n"
            f"Source selector: {self.source_selector_state}\n"
            f"Active server: {self.active_server_profile_id or 'none'}\n"
            f"Workspace: {self.workspace_id or 'none'}\n"
            f"Sync dry-run only: {bool(self.sync_dry_run_report)}\n"
            f"Summary: {self.display_summary or 'none'}\n"
            f"Metadata:\n{metadata or '- none'}\n\n"
            f"Content:\n{self.body}"
        )

    def format_for_model(self, user_prompt: str) -> str:
        return f"{self.model_context_block()}\n\n[User prompt]\n{user_prompt.strip()}"


SECRET_CONTRACT_KEYS = frozenset({"credential_source", "token", "secret", "api_key", "password"})
HANDOFF_BODY_CHAR_LIMIT = 80_000


def _json_safe_contract_snapshot(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_safe_contract_snapshot(item)
            for key, item in value.items()
            if item is not None and not _is_secret_contract_key(str(key))
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_contract_snapshot(item) for item in value if item is not None]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_payload"):
        return _json_safe_contract_snapshot(value.to_payload())
    return str(value)


def _is_secret_contract_key(key: str) -> bool:
    normalized = key.lower()
    return any(secret_key in normalized for secret_key in SECRET_CONTRACT_KEYS)
```

Add a `ChatHandoffPayload.from_source_content(...)` helper or equivalent builder guard that caps body text to `HANDOFF_BODY_CHAR_LIMIT`, sets `body_truncated=True`, and records `content_ref` when the source can be rehydrated. Do not persist unbounded media transcripts or full backend packets into `ui_state.toml`.

- [ ] **Step 5: Add `handoff_payload` to session models**

In `tldw_chatbook/Chat/chat_models.py`, import `ChatHandoffPayload`, add `handoff_payload: Optional[ChatHandoffPayload] = None`, serialize via `handoff_payload.to_dict()`, and restore via `ChatHandoffPayload.from_dict(...)`.

In `tldw_chatbook/UI/Screens/chat_screen_state.py`, do the same for `TabState`.

In `tldw_chatbook/Chat/tabs/tab_state_manager.py`, add `handoff_payload: Optional[dict[str, Any]] = None` or `Optional[ChatHandoffPayload] = None`. Prefer `dict` if avoiding runtime import cycles.

- [ ] **Step 6: Run focused tests**

Run: `pytest Tests/UI/test_chat_first_handoffs.py Tests/UI/test_chat_screen_state.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Chat/chat_handoff_models.py tldw_chatbook/Chat/chat_models.py tldw_chatbook/UI/Screens/chat_screen_state.py tldw_chatbook/Chat/tabs/tab_state_manager.py Tests/UI/test_chat_first_handoffs.py Tests/UI/test_chat_screen_state.py
git commit -m "feat: add chat handoff payload contract"
```

## Task 2: Add App-Owned Handoff Navigation And Chat Destination Consumption

**Files:**
- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen_state.py`
- Modify: `config.toml`
- Modify: `tldw_chatbook/config.py`
- Modify: `Tests/UI/test_chat_first_handoffs.py`
- Modify: `Tests/UI/test_chat_tab_container.py`

- [ ] **Step 0: Enable chat tabs by default for new/generated config**

Add or update a config/defaults test near existing config coverage:

```python
def test_chat_tabs_are_enabled_by_default_for_handoff_capable_chat():
    config = load_default_config_for_test()

    assert config["chat_defaults"]["enable_tabs"] is True
```

Then set bundled/generated `[chat_defaults].enable_tabs = true`. `open_chat_with_handoff()` must still refuse when the resolved user config explicitly sets `enable_tabs = false`.

- [ ] **Step 1: Write failing app helper tests**

Add to `Tests/UI/test_chat_first_handoffs.py`:

```python
from unittest.mock import Mock, patch

from tldw_chatbook.Constants import TAB_CHAT
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen


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
```

- [ ] **Step 2: Write failing ChatScreen consumption tests**

Add to `Tests/UI/test_chat_first_handoffs.py`:

```python
import pytest
from unittest.mock import AsyncMock, Mock

from tldw_chatbook.Chat.chat_models import ChatSessionData
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


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
    app.get_current_screen_state = Mock(return_value={})

    tab_container = Mock()
    tab_container.create_new_tab = AsyncMock(return_value="tab-1")
    tab_container.sessions = {"tab-1": Mock()}
    tab_container.sessions["tab-1"].session_data = ChatSessionData(tab_id="tab-1")
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
```

- [ ] **Step 3: Write failing tab reuse guard test**

Extend `Tests/UI/test_chat_tab_container.py`:

```python
@pytest.mark.asyncio
async def test_handoff_session_with_no_conversation_id_does_not_reuse_existing_tab():
    app = Mock()
    app.notify = Mock()
    app.call_later = Mock()
    existing = _make_session(
        ChatSessionData(
            tab_id="aaaaaaaa",
            title="Existing",
            conversation_id="conv-1",
            runtime_backend="server",
        )
    )
    container = ChatTabContainer(app)
    container.sessions = {"aaaaaaaa": existing}
    container.max_tabs = 10
    mount_target = Mock()
    mount_target.mount = AsyncMock()
    container.query_one = Mock(return_value=mount_target)
    container.post_message = Mock()

    session_data = ChatSessionData(
        tab_id="handoff",
        title="Note: Plan",
        conversation_id=None,
        runtime_backend="server",
        handoff_payload=ChatHandoffPayload(
            source="notes",
            item_type="note",
            title="Plan",
            body="Body",
        ),
    )

    with patch("tldw_chatbook.Widgets.Chat_Widgets.chat_tab_container.ChatSession", return_value=_make_session(session_data)):
        tab_id = await container.create_new_tab(session_data=session_data)

    assert tab_id != "aaaaaaaa"
```

- [ ] **Step 4: Run focused tests to verify they fail**

Run: `pytest Tests/UI/test_chat_first_handoffs.py Tests/UI/test_chat_tab_container.py -q`

Expected: FAIL because `open_chat_with_handoff()` and `_consume_pending_chat_handoff()` do not exist.

- [ ] **Step 5: Implement app helper**

In `tldw_chatbook/app.py`, add:

```python
def open_chat_with_handoff(self, payload: ChatHandoffPayload) -> None:
    if not get_cli_setting("chat_defaults", "enable_tabs", True):
        self.notify(
            "Use in Chat requires chat tabs to be enabled.",
            severity="warning",
        )
        return
    self.pending_chat_handoff = payload
    self.post_message(NavigateToScreen(TAB_CHAT))
```

Import `ChatHandoffPayload` under normal imports. If import cycles appear, guard with `TYPE_CHECKING` for annotations and keep runtime import local inside the method.

- [ ] **Step 6: Implement ChatScreen handoff consumption**

In `tldw_chatbook/UI/Screens/chat_screen.py`, add a post-restore call at the end of `_perform_state_restoration()`:

```python
await self._consume_pending_chat_handoff()
```

Also schedule the same method after `on_mount()` for first-entry cases where no saved state restoration was scheduled:

```python
self.set_timer(0.15, self._consume_pending_chat_handoff)
```

Add helper methods:

```python
def _session_data_for_handoff(self, payload: ChatHandoffPayload) -> ChatSessionData:
    return ChatSessionData(
        tab_id="handoff",
        title=f"{payload.item_type.replace('-', ' ').title()}: {payload.title}",
        conversation_id=None,
        is_ephemeral=True,
        runtime_backend=payload.runtime_backend,
        discovery_owner=payload.discovery_owner,
        discovery_entity_id=payload.discovery_entity_id or payload.source_id,
        scope_type=payload.scope_type or "global",
        workspace_id=payload.workspace_id if payload.scope_type == "workspace" else None,
        handoff_payload=payload,
    )


async def _consume_pending_chat_handoff(self) -> None:
    payload = getattr(self.app_instance, "pending_chat_handoff", None)
    if payload is None:
        return
    payload = ChatHandoffPayload.from_dict(payload)
    tab_container = self._get_tab_container()
    if tab_container is None:
        self.app_instance.notify("Chat tabs are not available for Use in Chat.", severity="warning")
        return
    session_data = self._session_data_for_handoff(payload)
    tab_id = await tab_container.create_new_tab(session_data=session_data)
    if not tab_id:
        self.app_instance.notify("Could not create a chat session for this context.", severity="error")
        return
    await tab_container.switch_to_tab_async(tab_id)
    session = tab_container.sessions.get(tab_id)
    if session is not None:
        await self._apply_handoff_to_chat_session(session, payload)
    self.app_instance.pending_chat_handoff = None
```

- [ ] **Step 7: Run focused tests**

Run: `pytest Tests/UI/test_chat_first_handoffs.py Tests/UI/test_chat_tab_container.py -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add config.toml tldw_chatbook/config.py tldw_chatbook/app.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_chat_first_handoffs.py Tests/UI/test_chat_tab_container.py
git commit -m "feat: route handoffs into fresh chat tabs"
```

## Task 3: Render The Handoff Card And Prefill The Per-Tab Draft

**Files:**
- Create: `tldw_chatbook/Widgets/Chat_Widgets/chat_handoff_card.py`
- Modify: `tldw_chatbook/Widgets/Chat_Widgets/chat_session.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/UI/test_chat_first_handoffs.py`
- Modify: `Tests/UI/test_chat_window_enhanced.py`

- [ ] **Step 1: Write failing card widget tests**

Add to `Tests/UI/test_chat_first_handoffs.py`:

```python
from tldw_chatbook.Widgets.Chat_Widgets.chat_handoff_card import ChatHandoffCard


def test_handoff_card_uses_status_source_title_and_metadata():
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
```

- [ ] **Step 2: Write failing draft-prefill test**

Add to `Tests/UI/test_chat_first_handoffs.py`:

```python
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
```

- [ ] **Step 3: Run focused tests to verify they fail**

Run: `pytest Tests/UI/test_chat_first_handoffs.py -q`

Expected: FAIL because card and session seams do not exist.

- [ ] **Step 4: Implement `ChatHandoffCard`**

Create `tldw_chatbook/Widgets/Chat_Widgets/chat_handoff_card.py`:

```python
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static

from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload


SOURCE_LABELS = {
    "notes": "Notes",
    "workspace": "Workspace",
    "media": "Media",
    "search-rag": "RAG Search",
    "search-web": "Web Search",
}


class ChatHandoffCard(Container):
    DEFAULT_CSS = """
    ChatHandoffCard {
        width: 100%;
        padding: 1;
        margin-bottom: 1;
        border: round $primary;
        background: $boost;
    }
    """

    def __init__(self, payload: ChatHandoffPayload, **kwargs):
        super().__init__(**kwargs)
        self.payload = ChatHandoffPayload.from_dict(payload)

    def render_text(self) -> str:
        status = "Context sent" if self.payload.status == "sent" else "Context staged"
        source_label = SOURCE_LABELS.get(self.payload.source, self.payload.source.replace("-", " ").title())
        summary = self.payload.display_summary or self.payload.body[:240]
        metadata = " | ".join(
            f"{key}: {value}"
            for key, value in sorted((self.payload.metadata or {}).items())
            if value not in (None, "")
        )
        parts = [
            f"{status} from {source_label}",
            f"Title: {self.payload.title}",
            f"Type: {self.payload.item_type}",
            f"Summary: {summary or 'No preview available.'}",
        ]
        if self.payload.runtime_backend:
            parts.append(f"Backend: {self.payload.runtime_backend}")
        if self.payload.source_owner or self.payload.source_selector_state:
            parts.append(f"Source: {self._source_chip_label()}")
        if self.payload.active_server_profile_id:
            parts.append(f"Server: {self.payload.active_server_profile_id}")
        if self.payload.workspace_id:
            parts.append(f"Workspace: {self.payload.workspace_id}")
        if self.payload.sync_dry_run_report:
            parts.append("Sync: dry-run only")
        if self.payload.body_truncated:
            parts.append("Content: preview truncated")
        if self.payload.unsupported_reports:
            parts.append(f"Unsupported actions: {len(self.payload.unsupported_reports)}")
        if metadata:
            parts.append(metadata)
        parts.append("Review the draft below and send when ready.")
        return "\n".join(parts)

    def compose(self) -> ComposeResult:
        yield Static(self.render_text(), classes="chat-handoff-card-body")

    def _source_chip_label(self) -> str:
        state = self.payload.source_selector_state or self.payload.source_owner
        labels = {
            "local": "Local source",
            "server": "Server source",
            "workspace": "Workspace source",
            "shared": "Shared source",
        }
        return labels.get(str(state), str(state).replace("_", " ").title())
```

- [ ] **Step 5: Add `ChatSession` seams**

In `tldw_chatbook/Widgets/Chat_Widgets/chat_session.py`, add:

```python
async def mount_handoff_card(self, payload: ChatHandoffPayload) -> None:
    chat_log = self.query_one(f"#chat-log-{self.session_data.tab_id}")
    await chat_log.mount(ChatHandoffCard(payload))


def set_draft_text(self, text: str) -> None:
    input_widget = self.query_one(f"#chat-input-{self.session_data.tab_id}", TextArea)
    input_widget.load_text(text)
```

Import `ChatHandoffPayload` and `ChatHandoffCard`.

- [ ] **Step 6: Implement ChatScreen application helper**

In `tldw_chatbook/UI/Screens/chat_screen.py`:

```python
async def _apply_handoff_to_chat_session(self, session, payload: ChatHandoffPayload) -> None:
    if hasattr(session, "mount_handoff_card"):
        await session.mount_handoff_card(payload)
    if hasattr(session, "set_draft_text"):
        session.set_draft_text(payload.default_prompt())
```

- [ ] **Step 7: Run focused tests**

Run: `pytest Tests/UI/test_chat_first_handoffs.py Tests/UI/test_chat_window_enhanced.py -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Widgets/Chat_Widgets/chat_handoff_card.py tldw_chatbook/Widgets/Chat_Widgets/chat_session.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_chat_first_handoffs.py Tests/UI/test_chat_window_enhanced.py
git commit -m "feat: show staged chat handoff context"
```

## Task 4: Include Staged Context In The First Send

**Files:**
- Modify: `tldw_chatbook/Chat/chat_handoff_models.py`
- Modify: `tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py`
- Modify: `tldw_chatbook/Event_Handlers/Chat_Events/chat_events_tabs.py`
- Modify: `tldw_chatbook/Widgets/Chat_Widgets/chat_handoff_card.py`
- Modify: `Tests/Event_Handlers/Chat_Events/test_chat_events_tabs.py`
- Modify: `Tests/UI/test_chat_first_handoffs.py`

- [ ] **Step 1: Write failing prompt-format test**

Add to `Tests/UI/test_chat_first_handoffs.py`:

```python
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
```

- [ ] **Step 2: Write failing send-handler context tests**

Add to `Tests/Event_Handlers/Chat_Events/test_chat_events_tabs.py`:

```python
@pytest.mark.asyncio
async def test_tab_send_sets_current_handoff_payload_for_original_handler(monkeypatch):
    app = Mock()
    app.query_one = Mock()
    app.query = Mock()
    app.notify = Mock()
    app.set_current_chat_is_streaming = Mock()
    app.get_current_chat_is_streaming = Mock(return_value=False)
    app.current_chat_worker = None
    app.current_ai_message_widget = None

    payload = ChatHandoffPayload(
        source="notes",
        item_type="note",
        title="Plan",
        body="Body",
    )
    session_data = ChatSessionData(tab_id="tab-1", handoff_payload=payload)

    async def fake_original_handler(app_arg, event_arg):
        assert app_arg._current_chat_handoff_payload.title == "Plan"

    monkeypatch.setattr(chat_events_tabs.chat_events, "handle_chat_send_button_pressed", fake_original_handler)

    await chat_events_tabs.handle_chat_send_button_pressed_with_tabs(app, Mock(), session_data=session_data)

    assert session_data.handoff_payload.status == "sent"
    assert getattr(app, "_current_chat_handoff_payload", None) is None
```

- [ ] **Step 3: Run focused tests to verify they fail**

Run: `pytest Tests/UI/test_chat_first_handoffs.py Tests/Event_Handlers/Chat_Events/test_chat_events_tabs.py -q`

Expected: FAIL because send-time handoff context is not wired.

- [ ] **Step 4: Add prompt application helper**

In `tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py`, add a small helper near send handling:

```python
def apply_current_handoff_context(app: "TldwCli", message_text: str) -> str:
    payload = getattr(app, "_current_chat_handoff_payload", None)
    if payload is None:
        return message_text
    payload = ChatHandoffPayload.from_dict(payload)
    if payload is None or payload.status == "sent":
        return message_text
    return payload.format_for_model(message_text)
```

Use it after RAG context assembly and before world info dispatch:

```python
message_text_with_handoff = apply_current_handoff_context(app, message_text_with_rag)
message_text_with_world_info = message_text_with_handoff
```

Import `ChatHandoffPayload`.

- [ ] **Step 5: Bind active session payload in tab-aware send wrapper**

In `tldw_chatbook/Event_Handlers/Chat_Events/chat_events_tabs.py`, before calling the original handler:

```python
active_handoff = session_data.handoff_payload if session_data else None
if active_handoff is not None and getattr(active_handoff, "status", "staged") != "sent":
    app._current_chat_handoff_payload = active_handoff
else:
    app._current_chat_handoff_payload = None
```

In the `finally` block, clear `app._current_chat_handoff_payload`.

After the original handler completes without raising, mark the payload sent:

```python
if session_data and session_data.handoff_payload and session_data.handoff_payload.status != "sent":
    session_data.handoff_payload.status = "sent"
```

If card status refresh is easy from the active session, update the mounted card. If not, defer visual status refresh to a later small patch but keep the payload status correct.

- [ ] **Step 6: Run focused tests**

Run: `pytest Tests/UI/test_chat_first_handoffs.py Tests/Event_Handlers/Chat_Events/test_chat_events_tabs.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Chat/chat_handoff_models.py tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py tldw_chatbook/Event_Handlers/Chat_Events/chat_events_tabs.py tldw_chatbook/Widgets/Chat_Widgets/chat_handoff_card.py Tests/Event_Handlers/Chat_Events/test_chat_events_tabs.py Tests/UI/test_chat_first_handoffs.py
git commit -m "feat: send staged handoff context once"
```

## Task 5: Wire Notes And Workspace Source Adapters

**Files:**
- Modify: `tldw_chatbook/UI/Screens/notes_screen.py`
- Modify: `tldw_chatbook/Widgets/Note_Widgets/notes_sidebar_right.py`
- Modify: `tldw_chatbook/Widgets/Note_Widgets/workspace_context_panel.py`
- Modify: `Tests/UI/test_notes_screen.py`
- Modify: `Tests/UI/test_chat_first_handoffs.py`

- [ ] **Step 1: Write failing Notes payload tests**

Add to `Tests/UI/test_notes_screen.py`:

```python
def test_notes_screen_builds_local_note_handoff_from_visible_editor_text():
    app = Mock()
    app.notify = Mock()
    screen = NotesScreen(app)
    screen.state = NotesScreenState(
        scope_type=ScopeType.LOCAL_NOTE,
        selected_note_id=123,
        selected_note_version=4,
        selected_note_title="Draft",
        selected_note_content="Saved content",
    )
    editor = Mock()
    editor.text = "Visible unsaved content"
    screen.query_one = Mock(return_value=editor)

    payload = screen._build_current_chat_handoff_payload()

    assert payload.source == "notes"
    assert payload.item_type == "note"
    assert payload.runtime_backend == "local"
    assert payload.source_owner == "local"
    assert payload.source_selector_state == "local"
    assert payload.source_id == "123"
    assert payload.body == "Visible unsaved content"
```

- [ ] **Step 2: Write failing Workspace payload tests**

Add to `Tests/UI/test_notes_screen.py`:

```python
def test_notes_screen_builds_workspace_source_handoff_from_cached_payload():
    app = Mock()
    app.notify = Mock()
    screen = NotesScreen(app)
    screen.state = NotesScreenState(
        scope_type=ScopeType.WORKSPACE,
        workspace_subview=WorkspaceSubview.SOURCES,
        selected_workspace_id="workspace-1",
        selected_workspace_source_id="source-1",
    )
    screen._workspace_context_payload = {
        "workspace": {"id": "workspace-1", "name": "Research"},
        "notes": [],
        "sources": [{"id": "source-1", "title": "Transcript", "url": "https://example.com"}],
        "artifacts": [],
    }

    payload = screen._build_current_chat_handoff_payload()

    assert payload.source == "workspace"
    assert payload.item_type == "workspace-source"
    assert payload.runtime_backend == "server"
    assert payload.source_owner == "workspace"
    assert payload.source_selector_state == "workspace"
    assert payload.scope_type == "workspace"
    assert payload.workspace_id == "workspace-1"
    assert payload.title == "Transcript"
```

- [ ] **Step 3: Write failing button placement tests**

Add focused widget tests:

```python
def test_notes_sidebar_contains_use_in_chat_button():
    sidebar = NotesSidebarRight()
    ids = [child.id for child in sidebar.compose()]
    assert "notes-use-in-chat-button" in ids


def test_workspace_panel_contains_use_in_chat_buttons():
    panel = WorkspaceContextPanel()
    composed_ids = [getattr(child, "id", None) for child in panel.compose()]
    assert "workspace-use-in-chat-button" in composed_ids
    assert "workspace-source-use-in-chat-button" in composed_ids
    assert "workspace-artifact-use-in-chat-button" in composed_ids
```

If direct `compose()` inspection is awkward with nested containers, use a minimal Textual test app and query IDs after mount.

- [ ] **Step 4: Run focused tests to verify they fail**

Run: `pytest Tests/UI/test_notes_screen.py Tests/UI/test_chat_first_handoffs.py -q`

Expected: FAIL because builder methods, buttons, and handlers do not exist.

- [ ] **Step 5: Add Notes/Workspace `Use in Chat` buttons**

In `tldw_chatbook/Widgets/Note_Widgets/notes_sidebar_right.py`, add near `Save All Changes`:

```python
yield Button("Use in Chat", id="notes-use-in-chat-button", variant="primary")
```

In `tldw_chatbook/Widgets/Note_Widgets/workspace_context_panel.py`, add:

```python
yield Button("Use in Chat", id="workspace-use-in-chat-button", variant="primary")
yield Button("Use in Chat", id="workspace-source-use-in-chat-button", variant="primary")
yield Button("Use in Chat", id="workspace-artifact-use-in-chat-button", variant="primary")
```

- [ ] **Step 6: Implement NotesScreen payload builders**

In `tldw_chatbook/UI/Screens/notes_screen.py`, add `_build_current_chat_handoff_payload()` and small helpers for editor text and workspace records.

Key behavior:

```python
def _build_current_chat_handoff_payload(self) -> ChatHandoffPayload | None:
    if self.state.scope_type in (ScopeType.LOCAL_NOTE, ScopeType.SERVER_NOTE):
        return self._build_note_chat_handoff_payload()
    if self.state.scope_type == ScopeType.WORKSPACE:
        return self._build_workspace_chat_handoff_payload()
    return None
```

For local/server notes:

```python
runtime_backend = "server" if self.state.scope_type == ScopeType.SERVER_NOTE else "local"
return ChatHandoffPayload(
    source="notes",
    item_type="note",
    title=self.state.selected_note_title or "Untitled Note",
    body=self._read_editor_text() or self.state.selected_note_content,
    source_id=str(self.state.selected_note_id) if self.state.selected_note_id is not None else None,
    suggested_prompt="Use this note as context and help me work with it.",
    runtime_backend=runtime_backend,
    source_owner=runtime_backend,
    source_selector_state=runtime_backend,
    discovery_owner="notes",
    discovery_entity_id=str(self.state.selected_note_id) if self.state.selected_note_id is not None else None,
    scope_type="global",
    metadata={
        "note_version": self.state.selected_note_version,
        "keywords": list(self._selected_note_keywords),
        "unsaved_changes": self.state.has_unsaved_changes,
    },
)
```

For workspace records, map subview to item type and body source:

```python
metadata = {
    "workspace_subview": self.state.workspace_subview.value,
    "workspace_version": self.state.selected_workspace_version,
}
```

Workspace handoff payloads must set `source_owner="workspace"`, `source_selector_state="workspace"`, `scope_type="workspace"`, `workspace_id=<selected workspace>`, and include available workspace isolation metadata from backend-owned contracts. Do not treat workspace records as generic server-global notes just because their runtime backend is server-backed.

- [ ] **Step 7: Add NotesScreen button handlers**

In `notes_screen.py`:

```python
@on(Button.Pressed, "#notes-use-in-chat-button")
@on(Button.Pressed, "#workspace-use-in-chat-button")
@on(Button.Pressed, "#workspace-source-use-in-chat-button")
@on(Button.Pressed, "#workspace-artifact-use-in-chat-button")
def handle_use_in_chat_button(self, event: Button.Pressed) -> None:
    event.stop()
    payload = self._build_current_chat_handoff_payload()
    if payload is None:
        self._notify("Select an item before using it in Chat.", severity="warning")
        return
    open_chat = getattr(self.app_instance, "open_chat_with_handoff", None)
    if not callable(open_chat):
        self._notify("Use in Chat is not available.", severity="warning")
        return
    open_chat(payload)
```

- [ ] **Step 8: Run focused tests**

Run: `pytest Tests/UI/test_notes_screen.py Tests/UI/test_chat_first_handoffs.py -q`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add tldw_chatbook/UI/Screens/notes_screen.py tldw_chatbook/Widgets/Note_Widgets/notes_sidebar_right.py tldw_chatbook/Widgets/Note_Widgets/workspace_context_panel.py Tests/UI/test_notes_screen.py Tests/UI/test_chat_first_handoffs.py
git commit -m "feat: add notes workspace chat handoffs"
```

## Task 6: Wire Media Source Adapter

**Files:**
- Modify: `tldw_chatbook/UI/MediaWindow_v2.py`
- Modify: `tldw_chatbook/Widgets/Media/media_viewer_panel.py`
- Create or modify: `Tests/UI/test_media_handoffs.py`

- [ ] **Step 1: Write failing Media viewer event test**

Create `Tests/UI/test_media_handoffs.py`:

```python
from unittest.mock import Mock

from tldw_chatbook.Widgets.Media.media_viewer_panel import MediaViewerPanel


def test_media_viewer_emits_use_in_chat_event_for_loaded_media():
    panel = MediaViewerPanel(Mock())
    panel.media_data = {"id": "media-1", "title": "Lecture", "content": "Transcript"}

    event = panel._build_use_in_chat_event()

    assert event.media_data["title"] == "Lecture"
```

- [ ] **Step 2: Write failing MediaWindow payload test**

Add:

```python
def test_media_window_builds_handoff_from_hydrated_detail():
    app = Mock()
    app.media_runtime_state = MediaRuntimeState(runtime_backend="server")
    app.media_runtime_state.selected_record_id = "record-1"
    app.media_runtime_state.detail_by_record_id["record-1"] = {
        "id": "record-1",
        "title": "Lecture",
        "content": "Transcript",
        "url": "https://example.com",
        "media_type": "video",
    }
    window = MediaWindow(app)
    window.runtime_state = app.media_runtime_state
    window.viewer_panel = Mock()
    window.viewer_panel.media_data = {"id": "record-1", "title": "Fallback"}

    payload = window._build_current_media_chat_handoff_payload()

    assert payload.source == "media"
    assert payload.item_type == "media"
    assert payload.runtime_backend == "server"
    assert payload.discovery_entity_id == "record-1"
    assert payload.body == "Transcript"
```

- [ ] **Step 3: Run focused tests to verify they fail**

Run: `pytest Tests/UI/test_media_handoffs.py -q`

Expected: FAIL because event and builder do not exist.

- [ ] **Step 4: Add Media viewer button and event**

In `tldw_chatbook/Widgets/Media/media_viewer_panel.py`, add a message:

```python
class UseInChatRequested(Message):
    def __init__(self, media_data: dict[str, Any]) -> None:
        super().__init__()
        self.media_data = dict(media_data)
```

Add the button inside metadata `Actions`:

```python
yield Button("Use in Chat", id="media-use-in-chat-button", variant="primary", disabled=True)
```

Enable/disable it in `watch_media_data()`.

Handle press:

```python
@on(Button.Pressed, "#media-use-in-chat-button")
def handle_use_in_chat(self, event: Button.Pressed) -> None:
    event.stop()
    if self.media_data:
        self.post_message(self.UseInChatRequested(dict(self.media_data)))
```

- [ ] **Step 5: Add MediaWindow payload builder and handler**

In `tldw_chatbook/UI/MediaWindow_v2.py`:

```python
def _build_current_media_chat_handoff_payload(self) -> ChatHandoffPayload | None:
    record_id = getattr(self.runtime_state, "selected_record_id", None) if self.runtime_state else None
    detail = {}
    if record_id and self.runtime_state:
        detail.update(self.runtime_state.detail_by_record_id.get(record_id) or {})
    if not detail and isinstance(getattr(self.viewer_panel, "media_data", None), dict):
        detail.update(self.viewer_panel.media_data)
    if not detail:
        return None
    resolved_id = str(detail.get("id") or record_id or detail.get("source_id") or "")
    return ChatHandoffPayload.from_source_content(
        source="media",
        item_type="media",
        title=str(detail.get("title") or "Untitled Media"),
        body=str(detail.get("content") or detail.get("summary") or detail.get("analysis") or ""),
        content_ref=f"media:{resolved_id}" if resolved_id else None,
        source_id=resolved_id or None,
        display_summary=str(detail.get("summary") or ""),
        suggested_prompt="Use this media item as context and help me analyze or summarize it.",
        runtime_backend=self._runtime_backend(),
        source_owner=self._runtime_backend(),
        source_selector_state=self._runtime_backend(),
        discovery_owner="media",
        discovery_entity_id=resolved_id or None,
        scope_type="global",
        metadata={
            "url": detail.get("url"),
            "author": detail.get("author"),
            "media_type": detail.get("media_type"),
            "reading_progress": detail.get("reading_progress"),
            "content_available": bool(detail.get("content")),
        },
    )
```

Add handler:

```python
@on(MediaViewerPanel.UseInChatRequested)
def handle_media_use_in_chat(self, event: MediaViewerPanel.UseInChatRequested) -> None:
    payload = self._build_current_media_chat_handoff_payload()
    if payload is None:
        self.app_instance.notify("Select a media item before using it in Chat.", severity="warning")
        return
    self.app_instance.open_chat_with_handoff(payload)
```

- [ ] **Step 6: Run focused tests**

Run: `pytest Tests/UI/test_media_handoffs.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/UI/MediaWindow_v2.py tldw_chatbook/Widgets/Media/media_viewer_panel.py Tests/UI/test_media_handoffs.py
git commit -m "feat: add media chat handoffs"
```

## Task 7: Wire RAG Search And Web Search Source Adapters

**Files:**
- Modify: `tldw_chatbook/UI/Views/RAGSearch/search_result.py`
- Modify: `tldw_chatbook/UI/Views/RAGSearch/search_rag_window.py`
- Modify: `tldw_chatbook/UI/SearchWindow.py`
- Create: `Tests/UI/test_search_handoffs.py`

- [ ] **Step 1: Write failing SearchResult event test**

Create `Tests/UI/test_search_handoffs.py`:

```python
from tldw_chatbook.UI.Views.RAGSearch.search_result import SearchResult


def test_search_result_builds_use_in_chat_event_with_result_data():
    result = {"title": "Doc", "content": "Snippet", "source": "notes", "score": 0.8}
    card = SearchResult(result, 0)

    event = card._build_use_in_chat_event()

    assert event.index == 0
    assert event.result["title"] == "Doc"
```

- [ ] **Step 2: Write failing RAG payload normalization test**

Add:

```python
def test_search_window_normalizes_rag_result_payload():
    app = Mock()
    window = SearchRAGWindow(app_instance=app)
    result = {
        "title": "Chunk",
        "content": "Retrieved text",
        "source": "notes",
        "score": 0.91,
        "metadata": {"document_id": "doc-1"},
    }

    payload = window._build_search_chat_handoff_payload(result)

    assert payload.source == "search-rag"
    assert payload.item_type == "rag-result"
    assert payload.discovery_owner == "rag_search"
    assert payload.source_selector_state in {"local", "server"}
    assert payload.body == "Retrieved text"
    assert payload.metadata["score"] == 0.91
```

- [ ] **Step 3: Write failing Web payload normalization test**

Add:

```python
def test_search_window_normalizes_web_result_payload():
    app = Mock()
    window = SearchRAGWindow(app_instance=app)
    result = {
        "title": "Article",
        "content": "Snippet",
        "source": "web",
        "metadata": {"url": "https://example.com", "displayUrl": "example.com"},
    }

    payload = window._build_search_chat_handoff_payload(result)

    assert payload.source == "search-web"
    assert payload.item_type == "web-result"
    assert payload.discovery_owner == "web_search"
    assert payload.source_owner == "server"
    assert payload.metadata["url"] == "https://example.com"
```

- [ ] **Step 4: Run focused tests to verify they fail**

Run: `pytest Tests/UI/test_search_handoffs.py -q`

Expected: FAIL because SearchResult events and normalization helpers do not exist.

- [ ] **Step 5: Add SearchResult button and event**

In `tldw_chatbook/UI/Views/RAGSearch/search_result.py`:

```python
class UseInChatRequested(Message):
    def __init__(self, index: int, result: dict[str, Any]) -> None:
        super().__init__()
        self.index = index
        self.result = dict(result)
```

Add the button to the action row:

```python
yield Button("Use in Chat", id=f"use-in-chat-{self.index}", classes="result-button")
```

Handle press:

```python
@on(Button.Pressed)
def handle_button_pressed(self, event: Button.Pressed) -> None:
    if event.button.id == f"use-in-chat-{self.index}":
        event.stop()
        self.post_message(self.UseInChatRequested(self.index, self.result))
```

- [ ] **Step 6: Add SearchRAGWindow payload builder and handler**

In `tldw_chatbook/UI/Views/RAGSearch/search_rag_window.py`:

```python
def _build_search_chat_handoff_payload(self, result: dict[str, Any]) -> ChatHandoffPayload:
    source_kind = str(result.get("source") or "unknown").lower()
    is_web = source_kind == "web"
    metadata = dict(result.get("metadata") or {})
    if "score" in result:
        metadata["score"] = result.get("score")
    if result.get("citations"):
        metadata["citations"] = result.get("citations")
    get_source = getattr(self.app_instance, "get_authoritative_runtime_source", None)
    runtime_backend = str(get_source() if callable(get_source) else "local")
    if runtime_backend not in {"local", "server"}:
        runtime_backend = "local"
    source_owner = "server" if is_web else runtime_backend
    source_selector_state = "server" if is_web else runtime_backend
    return ChatHandoffPayload(
        source="search-web" if is_web else "search-rag",
        item_type="web-result" if is_web else "rag-result",
        title=str(result.get("title") or "Search Result"),
        body=str(result.get("content") or result.get("snippet") or ""),
        source_id=str(metadata.get("document_id") or metadata.get("url") or ""),
        display_summary=str(result.get("content") or "")[:240],
        suggested_prompt=(
            "Use this web result as source context and preserve attribution in your answer."
            if is_web else
            "Use this retrieved result as context and answer or reason from it carefully."
        ),
        runtime_backend=runtime_backend,
        source_owner=source_owner,
        source_selector_state=source_selector_state,
        discovery_owner="web_search" if is_web else "rag_search",
        discovery_entity_id=str(metadata.get("document_id") or metadata.get("url") or "") or None,
        scope_type="global",
        metadata=metadata,
    )
```

Add handler:

```python
@on(SearchResult.UseInChatRequested)
def handle_search_result_use_in_chat(self, event: SearchResult.UseInChatRequested) -> None:
    payload = self._build_search_chat_handoff_payload(event.result)
    self.app_instance.open_chat_with_handoff(payload)
```

Before forwarding server-backed or remote-only results, check the relevant runtime policy/source contract or unsupported report. If the result source is unavailable, auth-blocked, or capability-missing, disable the action or show the report message rather than building a handoff from stale screen data.

- [ ] **Step 7: Cardify dedicated Web Search results**

In `tldw_chatbook/UI/SearchWindow.py`, replace the dedicated Web Search Markdown-only output with reusable result cards.

Add a `VerticalScroll` result container:

```python
yield VerticalScroll(id="web-search-results-list")
```

Add a `web_search_results: list[dict[str, Any]]` field on `SearchWindow`.

Implement a button handler:

```python
@on(Button.Pressed, "#web-search-button")
async def handle_web_search_button_pressed(self, event: Button.Pressed) -> None:
    event.stop()
    query = self.query_one("#web-search-input", Input).value.strip()
    if not query:
        self.app_instance.notify("Enter a web search query.", severity="warning")
        return
    raw_results = await search_web_bing(query)
    parsed_results = parse_bing_results(raw_results)
    self.web_search_results = [
        {
            "title": result.get("name", "Web Result"),
            "content": result.get("snippet", ""),
            "source": "web",
            "score": 0.5,
            "metadata": {
                "url": result.get("url", ""),
                "displayUrl": result.get("displayUrl", ""),
                "query": query,
            },
        }
        for result in parsed_results
    ]
    await self._render_web_search_result_cards()
```

Render cards using `SearchResult`:

```python
async def _render_web_search_result_cards(self) -> None:
    results_list = self.query_one("#web-search-results-list")
    await results_list.remove_children()
    for index, result in enumerate(self.web_search_results):
        await results_list.mount(SearchResult(result, index))
```

Handle `SearchResult.UseInChatRequested` at the `SearchWindow` level for dedicated Web Search cards and call the same payload builder used by `SearchRAGWindow`, or factor a tiny shared helper if importing from `SearchRAGWindow` would create coupling.

- [ ] **Step 8: Run focused tests**

Run: `pytest Tests/UI/test_search_handoffs.py -q`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add tldw_chatbook/UI/Views/RAGSearch/search_result.py tldw_chatbook/UI/Views/RAGSearch/search_rag_window.py tldw_chatbook/UI/SearchWindow.py Tests/UI/test_search_handoffs.py
git commit -m "feat: add search chat handoffs"
```

## Task 8: End-To-End Regression Sweep And Documentation Notes

**Files:**
- Modify as needed from previous tasks.
- Optional Modify: `Docs/Development/chat-first-shell-migration.md`
- Optional Modify: `Docs/Development/navigation-architecture-analysis.md`
- Modify: `Docs/superpowers/specs/2026-04-21-use-in-chat-handoffs-design.md` only if implementation deliberately deviates from spec.

- [x] **Step 1: Run focused handoff and adjacent suites**

Run:

```bash
pytest Tests/UX_Interop/test_server_parity_contracts.py Tests/UI/test_chat_first_handoffs.py Tests/UI/test_chat_screen_state.py Tests/UI/test_chat_tab_container.py Tests/UI/test_notes_screen.py Tests/UI/test_media_handoffs.py Tests/UI/test_search_handoffs.py Tests/Event_Handlers/Chat_Events/test_chat_events_tabs.py -q
```

Expected: PASS.

Progress note: Passed. Combined handoff/search run included the listed files plus `Tests/UI/test_search_rag_window.py`: `128 passed, 16 skipped`.

- [x] **Step 2: Run broader Chat/UI smoke tests**

Run:

```bash
pytest Tests/UI/test_chat_window_enhanced.py Tests/UI/test_chat_shell_bar.py Tests/Event_Handlers/Chat_Events/test_chat_events_tabs.py -q
```

Expected: PASS.

Progress note: Passed with `55 passed, 16 skipped`.

- [x] **Step 3: Run formatting/static checks available in this repo**

Run:

```bash
pytest -q
```

Expected: PASS or document any unrelated existing failures with exact test names and failure summaries.

Progress note: Full `pytest -q` completed in 52:16 with `4847 passed, 219 skipped, 49 failed, 7 errors`. Handoff-focused files passed before the full sweep. A `pytest --lf -q` rerun reduced the reproducible set to `25 failed, 28 passed, 106 deselected`:

- `Tests/RuntimePolicy/test_boundary_guards.py::test_raw_server_client_construction_is_confined_to_runtime_policy_boundaries` - raw `ServerChatbookService` construction still appears outside the runtime-policy allowlist.
- Media ingest source panel failures all hit `InvalidSelectValueError: Illegal select value 'local_directory'` from `tldw_chatbook/Widgets/Media/media_ingestion_source_panel.py:118`: `Tests/UI/test_ingest_window.py::test_media_ingest_screen_mounts_rebuilt_window`, `Tests/UI/test_ingest_window.py::test_media_ingest_screen_passes_runtime_state_to_rebuilt_window`, `Tests/UI/test_ingestion_integration_comprehensive.py::test_rebuilt_ingest_window_mounts_current_panels`, `Tests/UI/test_ingestion_integration_comprehensive.py::test_processing_messages_update_window_state`, `Tests/UI/test_ingestion_integration_comprehensive.py::test_local_processing_handles_multiple_selected_files`, `Tests/UI/test_ingestion_ui_redesigned.py::TestMediaIngestWindowRebuilt::test_media_ingest_window_mounts_current_panels`, `Tests/UI/test_ingestion_ui_redesigned.py::TestMediaIngestWindowRebuilt::test_local_panel_exposes_current_file_ingest_controls`, `Tests/UI/test_ingestion_ui_redesigned.py::TestMediaIngestWindowRebuilt::test_source_panel_defaults_to_local_mode_message`, `Tests/UI/test_ingestion_ui_redesigned.py::TestMediaIngestWindowRebuilt::test_processing_selected_files_runs_ingest_and_resets_ui`, `Tests/UI/test_media_ingestion_source_panel.py::test_ingestion_source_panel_is_disabled_in_local_mode`, `Tests/UI/test_media_ingestion_source_panel.py::test_ingestion_source_panel_lists_sources_in_server_mode`, `Tests/UI/test_media_ingestion_source_panel.py::test_ingestion_source_panel_create_is_disabled_in_local_mode`, `Tests/UI/test_media_ingestion_source_panel.py::test_ingestion_source_panel_creates_allowed_server_source_and_refreshes_selection`, `Tests/UI/test_media_ingestion_source_panel.py::test_ingestion_source_panel_does_not_dispatch_create_when_runtime_state_switched_to_local`, `Tests/UI/test_media_ingestion_tab_integration.py::test_media_ingest_screen_exposes_current_window`, `Tests/UI/test_media_ingestion_tab_integration.py::test_media_ingest_screen_keeps_rebuilt_window_visible_in_server_mode`, `Tests/UI/test_new_ingest_integration.py::test_local_processing_uses_form_metadata`, `Tests/UI/test_new_ingest_integration.py::test_local_processing_resets_button_after_completion`, `Tests/UI/test_new_ingest_window.py::test_rebuilt_ingest_window_mounts_tabbed_shell`, `Tests/UI/test_new_ingest_window.py::test_rebuilt_ingest_window_tracks_active_tab`, `Tests/UI/test_new_ingest_window_integration.py::test_server_mode_loads_source_detail_on_mount`, `Tests/UI/test_new_ingest_window_integration.py::test_server_mode_save_sync_and_upload_use_scope_service`, and `Tests/UI/test_tab_links_navigation.py::TestTabLinksNavigation::test_all_tab_links_clickable_and_navigate`.
- `Tests/tldw_api/test_research_runs_client.py::test_research_runs_client_routes_lifecycle_and_artifact_calls` - client now sends `{"limit": 10, "offset": 0}` where the test expects only `{"limit": 10}`.

- [ ] **Step 4: Manual Textual smoke path**

Run:

```bash
python3 -m tldw_chatbook.app
```

Expected:

- Chat tabs enabled path can create a new tab from `Use in Chat`.
- Notes local note opens a fresh Chat tab with a staged context card and draft.
- Workspace source opens a fresh Chat tab with workspace scope visible in the shell bar.
- Media selected item opens a fresh Chat tab with media metadata visible.
- RAG result opens a fresh Chat tab with search result card metadata.
- Dedicated Web Search result opens a fresh Chat tab with URL attribution metadata.
- Server-unavailable/auth-required/capability-missing fixture states disable or explain server-backed handoff controls without breaking local handoffs.
- Sync dry-run fixture states render as diagnostics only and never say sync is enabled or complete.
- Workspace fixture states keep workspace IDs visible and do not render workspace notes as global notes.
- Nothing is auto-sent before pressing Send.

Manual smoke note: Deferred. Automated Textual handoff and shell tests passed, but the broad suite currently fails when navigating into the ingest screen because the media ingestion source panel cannot compose its create-source select. No spec deviation was found in the shipped handoff behavior.

- [x] **Step 5: Update docs only if behavior changed**

If implementation deliberately deviates from the spec, update the spec with the exact shipped behavior before marking work complete.

- [x] **Step 6: Commit final docs/test adjustments**

```bash
git add Docs/Development/chat-first-shell-migration.md Docs/Development/navigation-architecture-analysis.md Docs/superpowers/specs/2026-04-21-use-in-chat-handoffs-design.md
git commit -m "docs: record chat handoff implementation notes"
```

Skip this commit if no docs changed.

PR #145 rebase/review note (2026-05-01): Rebasing onto latest `origin/dev` showed the first seven handoff implementation commits were already present in `dev` via the merged `codex/use-in-chat-handoffs` branch, so the PR branch dropped those duplicate commits and now carries only review-closeout deltas. Gemini inline review comments were verified against the rebased code: Unified MCP context save failures already log `OSError` on `origin/dev`; RAG web search now runs synchronous Bing IO through `asyncio.to_thread`; handoff session data now uses unique 8-character hex tab IDs instead of a hardcoded handoff placeholder; and `MediaViewerPanel.clear_display()` no longer uses a bare `except:`. Focused review regressions passed with `3 passed`; adjacent blocker suites passed with `48 passed`.

## Implementation Notes For Workers

- Do not inject handoff context into legacy single-session Chat. The feature is intentionally tab-only for this slice.
- Do not query or mutate Chat directly from Notes, Media, or Search. Source screens must call `app_instance.open_chat_with_handoff(payload)`.
- Keep the handoff card visually distinct from user, assistant, and system messages.
- Keep the staged payload session-scoped. App-level `pending_chat_handoff` is only the navigation transfer slot.
- Do not make source content disappear after first send. Mark it as sent, but keep the card visible.
- Preserve `runtime_backend`, `scope_type`, and `workspace_id`; do not silently overwrite them with the current global backend.
- Preserve `source_owner`, `source_selector_state`, `active_server_profile_id`, unsupported reports, and sync dry-run reports; do not flatten them into generic metadata-only labels.
- Do not infer source authority from screen layout, raw config, or direct server clients. Consume `UX_Interop` and `runtime_policy` contracts.
- Do not persist raw backend packets or credential-bearing contract fields in chat/session state. Store only the sanitized subset needed for UI state, routing, and auditability.
- Do not persist unbounded source bodies into saved screen state. Cap body text, set `body_truncated=True`, and keep/refetch full content through explicit source references when available.
- Treat `server_unavailable`, `auth_required`, `permission_denied`, `capability_missing`, and `not_implemented_locally` as first-class disabled/explained states.
- Keep sync dry-run reports diagnostic-only. Do not show write sync, queued replay, or mirror completion.
- Keep workspace-scoped records isolated; workspace handoffs require `workspace_id` and must not render as global notes.
- If max tabs are reached, leave `pending_chat_handoff` uncleared and notify the user.
- Treat unsaved Notes and Workspace editor text as the visible source of truth where it can be read safely.

## Completion Criteria

- UI handoff tests consume backend-owned server parity fixtures and handoff packet sections without importing current UI screens from the contract layer.
- New/generated config enables chat tabs by default; explicit user opt-out still makes `Use in Chat` unavailable with a clear message.
- Handoff payload persistence is JSON/TOML-safe, secret-redacted, and body-size bounded.
- All new and modified focused tests pass.
- `Use in Chat` is visible or explicitly unavailable in each target surface.
- Server-backed handoffs handle unavailable server, auth required, permission denied, capability missing, and local-not-implemented states.
- Local, server, workspace, and remote-only records remain visually distinct in the handoff card and destination session metadata.
- Sync/report UI never implies automatic write sync or completed local mirroring from dry-run reports.
- Every handoff opens a fresh Chat tab with `conversation_id=None`.
- The destination Chat tab shows a staged handoff card and draft prompt.
- The first Send includes the staged context in the model prompt.
- The pending app handoff clears after successful consumption and does not replay.
- Handoff state survives tab switch and screen restore.
- Any deviation from the spec is reflected in the spec before completion.
