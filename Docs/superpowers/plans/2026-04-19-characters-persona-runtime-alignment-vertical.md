# Characters, Persona Profiles, And Runtime Alignment Vertical Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align `tldw_chatbook` character, persona profile, exemplar, and CCP-to-main-chat runtime behavior with the current `tldw_server` contracts while preserving standalone local mode and keeping chat UI changes minimal.

**Architecture:** Treat the server entity model as the reference shape, but add a mirrored local schema and local services so characters, persona profiles, and exemplars remain usable offline. Use one mode-aware character/persona facade for CCP and main chat, persist explicit runtime/discovery metadata in chat state, and enforce history isolation through data contracts and query helpers rather than UI heuristics.

**Tech Stack:** Python, Textual, Pydantic, `httpx`, SQLite-backed `CharactersRAGDB`, pytest

---

## File Map

- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
  Responsibility: Add persona-profile and exemplar tables, runtime/discovery metadata columns or bridge defaults, canonical-ID backfill helpers, and query helpers for CCP-scoped discovery.
- Modify: `tldw_chatbook/tldw_api/chat_conversation_schemas.py`
  Responsibility: Move chat-facing contracts toward string-first assistant identity and runtime/discovery metadata support.
- Create: `tldw_chatbook/tldw_api/character_persona_schemas.py`
  Responsibility: Pydantic request/response models for characters, personas, sessions, exemplars, greetings, and presets.
- Modify: `tldw_chatbook/tldw_api/client.py`
  Responsibility: Add server endpoint methods for characters, personas, exemplars, greetings, and presets.
- Modify: `tldw_chatbook/tldw_api/__init__.py`
  Responsibility: Export the new character/persona API models consistently.
- Create: `tests/tldw_api/test_character_persona_schemas.py`
  Responsibility: Verify validation for string IDs, ownership fields, greeting/preset payloads, and persona exemplar shapes.
- Create: `tests/tldw_api/test_character_persona_client.py`
  Responsibility: Verify endpoint wiring and parameter serialization for all new server methods.

- Create: `tldw_chatbook/Character_Chat/server_character_persona_service.py`
  Responsibility: Thin server-backed resource adapter modeled after `server_notes_workspace_service.py`.
- Create: `tldw_chatbook/Character_Chat/character_persona_scope_service.py`
  Responsibility: Mode-aware facade that exposes one local/server API for CCP and main chat.
- Modify: `tldw_chatbook/app.py`
  Responsibility: Instantiate and expose the new character/persona services next to existing notes/chatbook services.
- Create: `tests/Character_Chat/test_character_persona_scope_service.py`
  Responsibility: Verify local/server routing, mode checks, and normalization behavior.

- Modify: `tldw_chatbook/Chat/chat_models.py`
  Responsibility: Extend `ChatSessionData` with string-first canonical assistant identity plus runtime/discovery metadata.
- Modify: `tldw_chatbook/UI/Screens/chat_screen_state.py`
  Responsibility: Serialize and restore the expanded launched-session contract.
- Modify: `tldw_chatbook/Chat/tabs/tab_state_manager.py`
  Responsibility: Carry runtime/discovery metadata through tab creation and update.
- Modify: `tldw_chatbook/Widgets/Chat_Widgets/chat_tab_container.py`
  Responsibility: Reuse tabs by backend plus conversation ID and keep titles display-driven.
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
  Responsibility: Persist/restore the expanded session contract and preserve runtime backend on restore.
- Modify: `tldw_chatbook/Chat/chat_conversation_service.py`
  Responsibility: Normalize local conversations onto canonical string identity and explicit discovery ownership.
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py`
  Responsibility: Create conversations with canonical IDs and runtime/discovery defaults.
- Modify: `tldw_chatbook/Chat/Chat_Functions.py`
  Responsibility: Stop writing character names into `assistant_id` and route through the normalized persistence seam.
- Modify: `tests/UI/test_chat_screen_state.py`
  Responsibility: Verify round-trip restore of runtime backend and discovery metadata.
- Modify: `tests/Widgets/test_chat_tab_container.py`
  Responsibility: Verify duplicate-tab focus behavior and display-title derivation.
- Modify: `tests/Chat/test_chat_conversation_service.py`
  Responsibility: Verify normalized conversation payloads, history filtering fields, and canonical-ID shaping.
- Modify: `tests/Chat/test_chat_persistence_service.py`
  Responsibility: Verify create/update flows store canonical IDs and default runtime/discovery metadata.
- Modify: `tests/Chat/test_chat_functions.py`
  Responsibility: Verify legacy save paths stop encoding character names as `assistant_id`.

- Create: `tldw_chatbook/UI/CCP_Modules/ccp_persona_handler.py`
  Responsibility: Persona profile CRUD, exemplar CRUD hooks, and persona-scoped session discovery.
- Modify: `tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py`
  Responsibility: Move character actions onto the new scope service and keep character-specific exemplar/session flows separate from persona flows.
- Modify: `tldw_chatbook/UI/CCP_Modules/ccp_conversation_handler.py`
  Responsibility: Convert CCP session handling to string IDs and backend-aware discovery calls.
- Modify: `tldw_chatbook/UI/CCP_Modules/ccp_message_manager.py`
  Responsibility: Load conversation/session messages through normalized service contracts.
- Modify: `tldw_chatbook/UI/CCP_Modules/ccp_messages.py`
  Responsibility: Carry string-first IDs and persona-specific selection/load events.
- Modify: `tldw_chatbook/UI/Screens/ccp_screen.py`
  Responsibility: Add persona-first management state, mode-switch invalidation, and backend-aware session launch.
- Modify: `tldw_chatbook/Widgets/CCP_Widgets/ccp_sidebar_widget.py`
  Responsibility: Add a first-class persona navigation area without introducing a mixed-mode list.
- Create: `tldw_chatbook/Widgets/CCP_Widgets/ccp_persona_card_widget.py`
  Responsibility: Render persona profile details and persona-scoped session actions.
- Create: `tldw_chatbook/Widgets/CCP_Widgets/ccp_persona_editor_widget.py`
  Responsibility: Persona CRUD editor surface aligned with existing character editor patterns.
- Modify: `tldw_chatbook/Widgets/CCP_Widgets/__init__.py`
  Responsibility: Export persona widgets/messages.
- Modify: `tests/UI/test_ccp_screen.py`
  Responsibility: Verify state, mode-switch invalidation, persona selection, and session scoping.
- Modify: `tests/UI/test_ccp_handlers.py`
  Responsibility: Verify handler behavior for persona CRUD, string IDs, and backend switching.
- Modify: `tests/Widgets/test_ccp_widgets.py`
  Responsibility: Verify sidebar and persona widgets render and emit the right messages.

- Modify: `tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py`
  Responsibility: Replace character-only history heuristics with explicit discovery-owner filtering for general chat history.
- Modify: `tests/Event_Handlers/Chat_Events/test_chat_events.py`
  Responsibility: Verify general history excludes CCP-owned character and persona sessions.
- Modify: `Docs/Parity/2026-04-19-data-compatibility-map.md`
  Responsibility: Record the canonical-ID/runtime-discovery contract once implemented.
- Modify: `Docs/Parity/2026-04-19-rollout-backlog.md`
  Responsibility: Mark this vertical status and note deferred sync/reconciliation work.

## Task 1: Lock Canonical Identity And Local Runtime Metadata Foundations

**Files:**
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Modify: `tldw_chatbook/tldw_api/chat_conversation_schemas.py`
- Modify: `tldw_chatbook/Chat/chat_conversation_service.py`
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py`
- Create: `tests/ChaChaNotesDB/test_character_persona_runtime_parity.py`
- Modify: `tests/Chat/test_chat_conversation_service.py`
- Modify: `tests/Chat/test_chat_persistence_service.py`

- [ ] **Step 1: Write the failing DB and service tests**

```python
def test_character_conversation_stores_canonical_assistant_id(db_instance):
    conversation_id = db_instance.add_conversation(
        {
            "title": "Character Session",
            "assistant_kind": "character",
            "assistant_id": "char.local.alice",
            "character_id": 7,
            "runtime_backend": "local",
            "discovery_owner": "ccp_character",
            "discovery_entity_id": "char.local.alice",
        }
    )

    conversation = db_instance.get_conversation_by_id(conversation_id)
    assert conversation["assistant_id"] == "char.local.alice"
    assert conversation["runtime_backend"] == "local"
    assert conversation["discovery_owner"] == "ccp_character"
    assert conversation["discovery_entity_id"] == "char.local.alice"


def test_persistence_service_never_uses_display_name_as_assistant_id(db_instance):
    service = ChatPersistenceService(db_instance)
    conversation_id = service.create_conversation(
        character_id=7,
        character_name="Alice",
        assistant_kind="character",
        assistant_id="char.local.alice",
        runtime_backend="local",
        discovery_owner="ccp_character",
        discovery_entity_id="char.local.alice",
    )
    conversation = db_instance.get_conversation_by_id(conversation_id)
    assert conversation["assistant_id"] == "char.local.alice"
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python3 -m pytest tests/ChaChaNotesDB/test_character_persona_runtime_parity.py tests/Chat/test_chat_conversation_service.py tests/Chat/test_chat_persistence_service.py -q`
Expected: FAIL with missing runtime/discovery fields, missing canonical-ID assertions, and legacy assumptions that still use display names or integer-only identities.

- [ ] **Step 3: Implement the minimal DB and service changes**

```python
DEFAULT_RUNTIME_BACKEND = "local"
DEFAULT_DISCOVERY_OWNER = "general_chat"

def _normalize_assistant_identity(assistant_kind, assistant_id, character_id):
    normalized_kind = _normalize_assistant_kind(assistant_kind)
    normalized_id = _clean_text(assistant_id)
    if normalized_kind == "character" and not normalized_id and character_id is not None:
        normalized_id = str(character_id)
    return normalized_kind, normalized_id

def _normalize_runtime_visibility(runtime_backend, discovery_owner, discovery_entity_id):
    return (
        _clean_enum(runtime_backend, {"local", "server"}, DEFAULT_RUNTIME_BACKEND),
        _clean_enum(discovery_owner, {"general_chat", "ccp_character", "ccp_persona"}, DEFAULT_DISCOVERY_OWNER),
        _clean_text(discovery_entity_id),
    )
```

- [ ] **Step 4: Re-run the focused tests**

Run: `python3 -m pytest tests/ChaChaNotesDB/test_character_persona_runtime_parity.py tests/Chat/test_chat_conversation_service.py tests/Chat/test_chat_persistence_service.py -q`
Expected: PASS, including legacy-row backfill/default behavior.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/DB/ChaChaNotes_DB.py tldw_chatbook/tldw_api/chat_conversation_schemas.py tldw_chatbook/Chat/chat_conversation_service.py tldw_chatbook/Chat/chat_persistence_service.py tests/ChaChaNotesDB/test_character_persona_runtime_parity.py tests/Chat/test_chat_conversation_service.py tests/Chat/test_chat_persistence_service.py
git commit -m "feat: add canonical assistant runtime metadata"
```

## Task 2: Add Server Character, Persona, Exemplar, Greeting, And Preset API Coverage

**Files:**
- Create: `tldw_chatbook/tldw_api/character_persona_schemas.py`
- Modify: `tldw_chatbook/tldw_api/client.py`
- Modify: `tldw_chatbook/tldw_api/__init__.py`
- Create: `tests/tldw_api/test_character_persona_schemas.py`
- Create: `tests/tldw_api/test_character_persona_client.py`

- [ ] **Step 1: Write the failing schema and client tests**

```python
@pytest.mark.asyncio
async def test_list_persona_profiles_hits_persona_endpoint(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"items": []})
    monkeypatch.setattr(client, "_request", mocked)

    await client.list_persona_profiles()

    args, kwargs = mocked.await_args
    assert args[:2] == ("GET", "/api/v1/persona/profiles")
    assert kwargs["params"] == {}


def test_persona_exemplar_requires_string_persona_id():
    with pytest.raises(ValidationError):
        PersonaExemplarCreate(persona_id=123, kind="example", content="hello")
```

- [ ] **Step 2: Run the API tests to verify they fail**

Run: `python3 -m pytest tests/tldw_api/test_character_persona_schemas.py tests/tldw_api/test_character_persona_client.py -q`
Expected: FAIL with missing schema module exports and missing client methods.

- [ ] **Step 3: Implement the minimal API contract layer**

```python
class PersonaProfileResponse(BaseModel):
    id: str
    name: str
    instructions: str | None = None
    version: int = 1


async def list_persona_profiles(self) -> Dict[str, Any]:
    return await self._request("GET", "/api/v1/persona/profiles", params={})


async def list_character_exemplars(self, character_id: str) -> Dict[str, Any]:
    return await self._request("GET", f"/api/v1/characters/{character_id}/exemplars")
```

- [ ] **Step 4: Re-run the API tests**

Run: `python3 -m pytest tests/tldw_api/test_character_persona_schemas.py tests/tldw_api/test_character_persona_client.py -q`
Expected: PASS with validated string IDs and endpoint wiring.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/tldw_api/character_persona_schemas.py tldw_chatbook/tldw_api/client.py tldw_chatbook/tldw_api/__init__.py tests/tldw_api/test_character_persona_schemas.py tests/tldw_api/test_character_persona_client.py
git commit -m "feat: add character persona api coverage"
```

## Task 3: Add Mode-Aware Character And Persona Services

**Files:**
- Create: `tldw_chatbook/Character_Chat/server_character_persona_service.py`
- Create: `tldw_chatbook/Character_Chat/character_persona_scope_service.py`
- Modify: `tldw_chatbook/app.py`
- Create: `tests/Character_Chat/test_character_persona_scope_service.py`

- [ ] **Step 1: Write the failing service tests**

```python
def test_scope_service_routes_to_server_backend_when_mode_is_server():
    local_service = Mock()
    server_service = Mock()
    scope_service = CharacterPersonaScopeService(local_service=local_service, server_service=server_service)

    scope_service.list_persona_profiles(mode="server")

    server_service.list_persona_profiles.assert_called_once_with()
    local_service.list_persona_profiles.assert_not_called()
```

- [ ] **Step 2: Run the service tests to verify they fail**

Run: `python3 -m pytest tests/Character_Chat/test_character_persona_scope_service.py -q`
Expected: FAIL with missing service classes and missing app wiring.

- [ ] **Step 3: Implement the minimal local/server facade**

```python
class CharacterPersonaScopeService:
    def __init__(self, *, local_service, server_service):
        self.local_service = local_service
        self.server_service = server_service

    def _backend(self, mode: str):
        if mode == "server":
            return self.server_service
        return self.local_service
```

- [ ] **Step 4: Re-run the service tests**

Run: `python3 -m pytest tests/Character_Chat/test_character_persona_scope_service.py -q`
Expected: PASS, including app wiring assertions or constructor-level smoke coverage.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Character_Chat/server_character_persona_service.py tldw_chatbook/Character_Chat/character_persona_scope_service.py tldw_chatbook/app.py tests/Character_Chat/test_character_persona_scope_service.py
git commit -m "feat: add character persona scope services"
```

## Task 4: Extend Main Chat Runtime, Restore, And Legacy Save Paths

**Files:**
- Modify: `tldw_chatbook/Chat/chat_models.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen_state.py`
- Modify: `tldw_chatbook/Chat/tabs/tab_state_manager.py`
- Modify: `tldw_chatbook/Widgets/Chat_Widgets/chat_tab_container.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/Chat/Chat_Functions.py`
- Modify: `tests/UI/test_chat_screen_state.py`
- Modify: `tests/Widgets/test_chat_tab_container.py`
- Modify: `tests/Chat/test_chat_functions.py`

- [ ] **Step 1: Write the failing runtime and restore tests**

```python
def test_tab_state_round_trips_runtime_backend_and_discovery_fields():
    state = TabState(
        tab_id="tab-1",
        title="Persona Session",
        conversation_id="conv-1",
        assistant_kind="persona",
        assistant_id="persona.alpha",
        runtime_backend="server",
        discovery_owner="ccp_persona",
        discovery_entity_id="persona.alpha",
    )
    restored = TabState.from_dict(state.to_dict())
    assert restored.runtime_backend == "server"
    assert restored.discovery_owner == "ccp_persona"


def test_chat_functions_use_canonical_character_id_not_character_name(mocker):
    persistence_service = mocker.Mock()
    persistence_service.create_conversation.return_value = "conv-1"
    # Assert create_conversation receives assistant_id="char.local.alice", not "Alice".
```

- [ ] **Step 2: Run the runtime tests to verify they fail**

Run: `python3 -m pytest tests/UI/test_chat_screen_state.py tests/Widgets/test_chat_tab_container.py tests/Chat/test_chat_functions.py -q`
Expected: FAIL with missing runtime/discovery fields and duplicate-tab behavior not implemented.

- [ ] **Step 3: Implement the minimal runtime contract**

```python
@dataclass
class ChatSessionData:
    runtime_backend: str = "local"
    discovery_owner: str = "general_chat"
    discovery_entity_id: Optional[str] = None


def _session_reuse_key(session_data: ChatSessionData) -> tuple[str, str | None]:
    return (session_data.runtime_backend, session_data.conversation_id)
```

- [ ] **Step 4: Re-run the runtime tests**

Run: `python3 -m pytest tests/UI/test_chat_screen_state.py tests/Widgets/test_chat_tab_container.py tests/Chat/test_chat_functions.py -q`
Expected: PASS with stable restore semantics and canonical save behavior.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/chat_models.py tldw_chatbook/UI/Screens/chat_screen_state.py tldw_chatbook/Chat/tabs/tab_state_manager.py tldw_chatbook/Widgets/Chat_Widgets/chat_tab_container.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Chat/Chat_Functions.py tests/UI/test_chat_screen_state.py tests/Widgets/test_chat_tab_container.py tests/Chat/test_chat_functions.py
git commit -m "feat: add runtime metadata to chat sessions"
```

## Task 5: Refactor CCP Into Dual-Entity Character And Persona Management

**Files:**
- Create: `tldw_chatbook/UI/CCP_Modules/ccp_persona_handler.py`
- Modify: `tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py`
- Modify: `tldw_chatbook/UI/CCP_Modules/ccp_conversation_handler.py`
- Modify: `tldw_chatbook/UI/CCP_Modules/ccp_message_manager.py`
- Modify: `tldw_chatbook/UI/CCP_Modules/ccp_messages.py`
- Modify: `tldw_chatbook/UI/Screens/ccp_screen.py`
- Modify: `tldw_chatbook/Widgets/CCP_Widgets/ccp_sidebar_widget.py`
- Create: `tldw_chatbook/Widgets/CCP_Widgets/ccp_persona_card_widget.py`
- Create: `tldw_chatbook/Widgets/CCP_Widgets/ccp_persona_editor_widget.py`
- Modify: `tldw_chatbook/Widgets/CCP_Widgets/__init__.py`
- Modify: `tests/UI/test_ccp_screen.py`
- Modify: `tests/UI/test_ccp_handlers.py`
- Modify: `tests/Widgets/test_ccp_widgets.py`

- [ ] **Step 1: Write the failing CCP tests**

```python
def test_mode_switch_clears_selected_entity_and_session():
    state = CCPScreenState(
        selected_character_id="char.local.alice",
        selected_conversation_id="conv-1",
        conversation_search_results=[{"id": "conv-1"}],
    )
    state.reset_for_backend_change()
    assert state.selected_character_id is None
    assert state.selected_conversation_id is None
    assert state.conversation_search_results == []


def test_persona_selection_is_first_class_in_ccp_screen(mock_app_instance):
    screen = CCPScreen()
    screen.state.active_view = "persona_profiles"
    assert screen.state.active_view == "persona_profiles"
```

- [ ] **Step 2: Run the CCP tests to verify they fail**

Run: `python3 -m pytest tests/UI/test_ccp_screen.py tests/UI/test_ccp_handlers.py tests/Widgets/test_ccp_widgets.py -q`
Expected: FAIL with missing persona handler, missing persona widgets, and `int`-typed CCP state assumptions.

- [ ] **Step 3: Implement the minimal dual-entity CCP structure**

```python
@dataclass
class CCPScreenState:
    selected_character_id: Optional[str] = None
    selected_persona_id: Optional[str] = None
    selected_conversation_id: Optional[str] = None

    def reset_for_backend_change(self) -> None:
        self.selected_character_id = None
        self.selected_persona_id = None
        self.selected_conversation_id = None
        self.conversation_search_results = []
```

- [ ] **Step 4: Re-run the CCP tests**

Run: `python3 -m pytest tests/UI/test_ccp_screen.py tests/UI/test_ccp_handlers.py tests/Widgets/test_ccp_widgets.py -q`
Expected: PASS with persona CRUD and mode-switch invalidation behavior covered.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/CCP_Modules/ccp_persona_handler.py tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py tldw_chatbook/UI/CCP_Modules/ccp_conversation_handler.py tldw_chatbook/UI/CCP_Modules/ccp_message_manager.py tldw_chatbook/UI/CCP_Modules/ccp_messages.py tldw_chatbook/UI/Screens/ccp_screen.py tldw_chatbook/Widgets/CCP_Widgets/ccp_sidebar_widget.py tldw_chatbook/Widgets/CCP_Widgets/ccp_persona_card_widget.py tldw_chatbook/Widgets/CCP_Widgets/ccp_persona_editor_widget.py tldw_chatbook/Widgets/CCP_Widgets/__init__.py tests/UI/test_ccp_screen.py tests/UI/test_ccp_handlers.py tests/Widgets/test_ccp_widgets.py
git commit -m "feat: add persona management to ccp"
```

## Task 6: Add CCP Discovery, Main-Chat Launch, And General History Isolation

**Files:**
- Modify: `tldw_chatbook/UI/CCP_Modules/ccp_conversation_handler.py`
- Modify: `tldw_chatbook/UI/CCP_Modules/ccp_message_manager.py`
- Modify: `tldw_chatbook/UI/Screens/ccp_screen.py`
- Modify: `tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py`
- Modify: `tests/Event_Handlers/Chat_Events/test_chat_events.py`
- Modify: `tests/UI/test_ccp_screen.py`
- Modify: `tests/Widgets/test_chat_tab_container.py`

- [ ] **Step 1: Write the failing discovery and history tests**

```python
def test_general_history_excludes_ccp_owned_sessions(mocker):
    conversations = [
        {"id": "conv-general", "discovery_owner": "general_chat"},
        {"id": "conv-char", "discovery_owner": "ccp_character"},
        {"id": "conv-persona", "discovery_owner": "ccp_persona"},
    ]
    assert [row["id"] for row in filter_general_history(conversations)] == ["conv-general"]


def test_launching_same_session_focuses_existing_tab():
    assert reuse_session_tab("server", "conv-1") == ("server", "conv-1")
```

- [ ] **Step 2: Run the discovery/history tests to verify they fail**

Run: `python3 -m pytest tests/Event_Handlers/Chat_Events/test_chat_events.py tests/UI/test_ccp_screen.py tests/Widgets/test_chat_tab_container.py -q`
Expected: FAIL with legacy character-chat heuristics and missing launch/focus behavior.

- [ ] **Step 3: Implement the minimal launch and filtering behavior**

```python
def is_general_history_conversation(row: Mapping[str, Any]) -> bool:
    return (row.get("discovery_owner") or "general_chat") == "general_chat"


def launch_session_from_ccp(contract: Mapping[str, Any]) -> None:
    reuse_key = (contract["runtime_backend"], contract["conversation_id"])
    # Focus existing tab when present, otherwise create a new one from the contract.
```

- [ ] **Step 4: Re-run the discovery/history tests**

Run: `python3 -m pytest tests/Event_Handlers/Chat_Events/test_chat_events.py tests/UI/test_ccp_screen.py tests/Widgets/test_chat_tab_container.py -q`
Expected: PASS with general history isolation and selected-entity CCP discovery behavior.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/CCP_Modules/ccp_conversation_handler.py tldw_chatbook/UI/CCP_Modules/ccp_message_manager.py tldw_chatbook/UI/Screens/ccp_screen.py tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py tests/Event_Handlers/Chat_Events/test_chat_events.py tests/UI/test_ccp_screen.py tests/Widgets/test_chat_tab_container.py
git commit -m "feat: add ccp launch and history isolation"
```

## Task 7: Add Minimal Greeting/Preset Execution Support And Finish Docs

**Files:**
- Modify: `tldw_chatbook/Character_Chat/server_character_persona_service.py`
- Modify: `tldw_chatbook/Character_Chat/character_persona_scope_service.py`
- Modify: `tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py`
- Modify: `tldw_chatbook/UI/CCP_Modules/ccp_persona_handler.py`
- Modify: `Docs/Parity/2026-04-19-data-compatibility-map.md`
- Modify: `Docs/Parity/2026-04-19-rollout-backlog.md`
- Modify: `tests/Character_Chat/test_character_persona_scope_service.py`
- Modify: `tests/UI/test_ccp_handlers.py`

- [ ] **Step 1: Write the failing greeting/preset and docs tests/checks**

```python
def test_server_scope_service_lists_character_greetings():
    server = ServerCharacterPersonaService(client=Mock())
    server.client.list_character_greetings.return_value = {"items": [{"id": "greet-1"}]}
    payload = server.list_character_greetings("char.server.alice")
    assert payload["items"][0]["id"] == "greet-1"
```

- [ ] **Step 2: Run the focused tests/checks to verify they fail**

Run: `python3 -m pytest tests/Character_Chat/test_character_persona_scope_service.py tests/UI/test_ccp_handlers.py -q`
Expected: FAIL with missing greeting/preset methods or missing handler wiring.

- [ ] **Step 3: Implement minimal execution-compatible support and update docs**

```python
class ServerCharacterPersonaService:
    def list_character_greetings(self, character_id: str) -> dict[str, Any]:
        return self._require_client().list_character_greetings(character_id)

    def list_character_presets(self, character_id: str) -> dict[str, Any]:
        return self._require_client().list_character_presets(character_id)
```

- [ ] **Step 4: Re-run the focused tests plus a final sweep**

Run: `python3 -m pytest tests/ChaChaNotesDB/test_character_persona_runtime_parity.py tests/tldw_api/test_character_persona_schemas.py tests/tldw_api/test_character_persona_client.py tests/Character_Chat/test_character_persona_scope_service.py tests/UI/test_chat_screen_state.py tests/UI/test_ccp_screen.py tests/UI/test_ccp_handlers.py tests/Widgets/test_ccp_widgets.py tests/Widgets/test_chat_tab_container.py tests/Event_Handlers/Chat_Events/test_chat_events.py tests/Chat/test_chat_conversation_service.py tests/Chat/test_chat_persistence_service.py tests/Chat/test_chat_functions.py -q`
Expected: PASS, with docs updated separately in the same changeset.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Character_Chat/server_character_persona_service.py tldw_chatbook/Character_Chat/character_persona_scope_service.py tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py tldw_chatbook/UI/CCP_Modules/ccp_persona_handler.py Docs/Parity/2026-04-19-data-compatibility-map.md Docs/Parity/2026-04-19-rollout-backlog.md tests/Character_Chat/test_character_persona_scope_service.py tests/UI/test_ccp_handlers.py
git commit -m "feat: finish character persona runtime parity vertical"
```
