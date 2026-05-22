# Console Native Chat Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a shared native chat core and wire Console to it first so Console send, streaming, stop, recovery, transcript selection, and message actions no longer depend on hidden legacy chat widgets as the source of truth.

**Architecture:** Introduce pure Chat-layer state, provider, store, controller, and action services before replacing visible Console transcript behavior. Console widgets render service state and emit user intent; legacy chat screens remain in place until they can migrate to the same core later.

**Tech Stack:** Python 3.11+, Textual, httpx, pytest, existing `ChatPersistenceService`, existing Textual Console widgets, TCSS source files under `tldw_chatbook/css/components/`.

---

## Source Of Truth

- Spec: `Docs/superpowers/specs/2026-05-21-console-native-chat-core-design.md`
- Existing Console shell: `tldw_chatbook/UI/Screens/chat_screen.py`
- Current native composer: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Current Console session host: `tldw_chatbook/Widgets/Console/console_session_surface.py`
- Provider readiness baseline: `tldw_chatbook/Chat/provider_readiness.py`
- Persistence baseline: `tldw_chatbook/Chat/chat_persistence_service.py`
- Existing Gate 1.5 tests: `Tests/UI/test_console_internals_decomposition.py`

## Constraints

- Use TDD for each behavior slice.
- Do not edit generated CSS directly. Edit source TCSS and regenerate with `python tldw_chatbook/css/build_css.py` only when CSS changes.
- Use relative commands such as `python -m pytest ...`; assume the virtualenv is active.
- Do not use `timeout`; it is unavailable in this environment.
- Do not require a live llama.cpp server in normal CI. Use fake transports for automated tests and keep the live `127.0.0.1:9099` verification as a local/CDP QA gate.
- Do not rewrite legacy chat screens in this plan. Keep legacy screens compatible and migrate later.
- Unavailable action paths must visibly return `WIP` or placeholder reasons.
- Actual rendered CDP/textual-web screenshots are required before claiming visible Console state approval.

## File Structure

### Create

- `tldw_chatbook/Chat/console_chat_models.py`
  - Pure dataclasses/enums for native Console messages, sessions, variants, provider resolution, run state, and action availability.
- `tldw_chatbook/Chat/console_provider_gateway.py`
  - Native provider resolution and streaming gateway. First concrete implementation: local llama.cpp OpenAI-compatible streaming.
- `tldw_chatbook/Chat/console_chat_store.py`
  - Session/message store facade around in-memory state and `ChatPersistenceService`.
- `tldw_chatbook/Chat/console_chat_controller.py`
  - Orchestration service for send, stop, retry, selection, and run-state updates.
- `tldw_chatbook/Chat/console_message_actions.py`
  - Message action availability and dispatch service for Copy, Edit, Save as..., Regenerate, Continue, feedback, delete, and variants.
- `tldw_chatbook/Widgets/Console/console_transcript.py`
  - Native transcript widget, message view, separator rendering, keyboard selection, and selected-message action row.
- `tldw_chatbook/Widgets/Console/console_save_as_modal.py`
  - Save-as destination modal with available and explicit `WIP` entries.
- `Tests/Chat/test_console_chat_models.py`
- `Tests/Chat/test_console_provider_gateway.py`
- `Tests/Chat/test_console_chat_store.py`
- `Tests/Chat/test_console_chat_controller.py`
- `Tests/Chat/test_console_message_actions.py`
- `Tests/UI/test_console_native_transcript.py`
- `Tests/UI/test_console_native_chat_flow.py`
- `Docs/superpowers/qa/product-maturity/console-native-chat-core/README.md`
- `Docs/superpowers/qa/product-maturity/console-native-chat-core/2026-05-21-console-native-chat-core.md`

### Modify

- `tldw_chatbook/UI/Screens/chat_screen.py`
  - Instantiate the native controller/store/provider gateway.
  - Route native composer buttons to controller methods.
  - Sync controller state into inspector, transcript, composer, and status surfaces.
- `tldw_chatbook/Widgets/Console/__init__.py`
  - Export new Console transcript and modal widgets.
- `tldw_chatbook/Widgets/Console/console_composer_bar.py`
  - Keep it as visible draft source of truth and add controller-friendly send/stop state hooks.
- `tldw_chatbook/Widgets/Console/console_session_surface.py`
  - Replace `ChatTabContainer` dependency with native transcript in the final migration task.
- `tldw_chatbook/Widgets/Console/console_run_inspector.py`
  - Reflect native run state where needed.
- `tldw_chatbook/css/components/_agentic_terminal.tcss`
  - Add native transcript/message/action-row styles.
- `tldw_chatbook/css/core/_variables.tcss`
  - Add sizing tokens only if repeated hardcoded sizes are needed.
- `tldw_chatbook/css/tldw_cli_modular.tcss`
  - Regenerate only after source TCSS changes.
- `Tests/UI/test_console_internals_decomposition.py`
  - Update or add guardrails proving Console no longer mounts legacy tab/session widgets as behavioral source of truth.

## Task 0: Backlog And Baseline Verification

**Files:**
- Read: `AGENTS.md`
- Read: `Docs/superpowers/specs/2026-05-21-console-native-chat-core-design.md`
- Read: `Tests/UI/test_console_internals_decomposition.py`
- Modify: new or existing `backlog/tasks/task-* - Console-Native-Chat-Core.md`

- [ ] **Step 1: Create or identify the Backlog task**

Run:

```bash
backlog task list --plain
```

If no task exists for this work, create one:

```bash
backlog task create "Console native chat core" -s "In Progress" -l console,chat,ux --priority high --ac "Console uses native chat core for send and streaming,Local llama.cpp streaming path is verified,Transcript message selection and actions work,Unavailable action paths show WIP reasons,Actual CDP screenshots are captured and approved"
```

Expected: a task ID to reference in future implementation notes.

- [ ] **Step 2: Add an implementation plan to the Backlog task**

Run:

```bash
backlog task edit <task-id> --plan "1. Add native chat contracts\n2. Add provider gateway\n3. Add stores and controller\n4. Wire Console send/streaming\n5. Replace transcript rendering\n6. Add selected-message actions\n7. Capture CDP QA evidence"
```

Expected: task remains `In Progress` and includes the plan.

- [ ] **Step 3: Run current Console baseline**

Run:

```bash
python -m pytest -q Tests/UI/test_console_internals_decomposition.py Tests/Chat/test_console_display_state.py --tb=short
```

Expected: pass on current `origin/dev`. If this fails, stop and fix or document the baseline before starting the native core.

- [ ] **Step 4: Commit only if Backlog files changed**

Run:

```bash
git add backlog/tasks
git commit -m "Track Console native chat core task"
```

Expected: commit created only if the Backlog task file changed.

## Task 1: Native Console Chat Models

**Files:**
- Create: `tldw_chatbook/Chat/console_chat_models.py`
- Create: `Tests/Chat/test_console_chat_models.py`

- [ ] **Step 1: Write failing model tests**

Add:

```python
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleProviderSelection,
    ConsoleRunState,
    ConsoleRunStatus,
    ConsoleStagedSource,
    ConsoleVariantSet,
    ConsoleWorkspaceContext,
)


def test_run_state_blocks_with_visible_recovery_copy():
    state = ConsoleRunState.blocked("Provider blocked: select a model")

    assert state.status is ConsoleRunStatus.BLOCKED
    assert state.visible_copy == "Provider blocked: select a model"
    assert state.is_send_allowed is False
    assert state.is_stop_allowed is False


def test_run_state_retrying_is_visible_and_not_sendable():
    state = ConsoleRunState.retrying("Retrying failed response")

    assert state.status is ConsoleRunStatus.RETRYING
    assert state.visible_copy == "Retrying failed response"
    assert state.is_send_allowed is False
    assert state.is_stop_allowed is False


def test_variant_set_selects_current_variant_for_continue():
    variants = ConsoleVariantSet.from_contents(
        turn_id="turn-1",
        contents=["first", "second"],
        selected_index=1,
    )

    assert variants.current.content == "second"
    assert variants.can_go_previous is True
    assert variants.can_go_next is False


def test_workspace_context_blocks_cross_workspace_sources():
    context = ConsoleWorkspaceContext(
        active_workspace_id="workspace-a",
        staged_sources=(
            ConsoleStagedSource(
                source_id="note-1",
                label="Other workspace note",
                source_type="note",
                workspace_id="workspace-b",
            ),
        ),
    )

    assert context.has_policy_blocks is True
    assert context.allowed_sources == []
    assert "Other workspace note" in context.recovery_copy
    assert "workspace-a" in context.recovery_copy


def test_provider_selection_carries_workspace_context():
    context = ConsoleWorkspaceContext(active_workspace_id="workspace-a")
    selection = ConsoleProviderSelection(
        provider="llama_cpp",
        base_url="http://127.0.0.1:9099",
        explicit_model="local-model",
        workspace_context=context,
    )

    assert selection.provider == "llama_cpp"
    assert selection.workspace_context.active_workspace_id == "workspace-a"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest -q Tests/Chat/test_console_chat_models.py --tb=short
```

Expected: import failure for `tldw_chatbook.Chat.console_chat_models`.

- [ ] **Step 3: Implement minimal pure models**

Create:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal
from uuid import uuid4


class ConsoleMessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ConsoleRunStatus(str, Enum):
    IDLE = "idle"
    VALIDATING = "validating"
    STREAMING = "streaming"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    STOPPED = "stopped"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass(frozen=True)
class ConsoleStagedSource:
    source_id: str
    label: str
    source_type: str
    workspace_id: str | None = None


@dataclass(frozen=True)
class ConsoleWorkspaceContext:
    active_workspace_id: str = "global"
    staged_sources: tuple[ConsoleStagedSource, ...] = ()
    active_run_id: str | None = None
    handoff_id: str | None = None

    @property
    def blocked_sources(self) -> list[ConsoleStagedSource]:
        return [
            source
            for source in self.staged_sources
            if source.workspace_id not in (None, self.active_workspace_id)
        ]

    @property
    def allowed_sources(self) -> list[ConsoleStagedSource]:
        blocked = {source.source_id for source in self.blocked_sources}
        return [source for source in self.staged_sources if source.source_id not in blocked]

    @property
    def has_policy_blocks(self) -> bool:
        return bool(self.blocked_sources)

    @property
    def recovery_copy(self) -> str:
        labels = ", ".join(source.label for source in self.blocked_sources)
        return f"Workspace policy blocked sources outside {self.active_workspace_id}: {labels}"


@dataclass(frozen=True)
class ConsoleProviderSelection:
    provider: str
    base_url: str | None = None
    explicit_model: str | None = None
    configured_model: str | None = None
    workspace_context: ConsoleWorkspaceContext = field(default_factory=ConsoleWorkspaceContext)


@dataclass(frozen=True)
class ConsoleRunState:
    status: ConsoleRunStatus = ConsoleRunStatus.IDLE
    visible_copy: str = ""

    @classmethod
    def blocked(cls, visible_copy: str) -> "ConsoleRunState":
        return cls(ConsoleRunStatus.BLOCKED, visible_copy)

    @classmethod
    def retrying(cls, visible_copy: str = "Retrying failed response") -> "ConsoleRunState":
        return cls(ConsoleRunStatus.RETRYING, visible_copy)

    @property
    def is_send_allowed(self) -> bool:
        return self.status in {ConsoleRunStatus.IDLE, ConsoleRunStatus.COMPLETED, ConsoleRunStatus.FAILED, ConsoleRunStatus.STOPPED}

    @property
    def is_stop_allowed(self) -> bool:
        return self.status is ConsoleRunStatus.STREAMING


@dataclass
class ConsoleChatMessage:
    role: ConsoleMessageRole
    content: str
    id: str = field(default_factory=lambda: str(uuid4()))
    turn_id: str | None = None
    status: Literal["complete", "pending", "streaming", "stopped", "failed"] = "complete"


@dataclass(frozen=True)
class ConsoleVariant:
    content: str
    id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class ConsoleVariantSet:
    turn_id: str
    variants: list[ConsoleVariant]
    selected_index: int = 0

    @classmethod
    def from_contents(cls, *, turn_id: str, contents: list[str], selected_index: int = 0) -> "ConsoleVariantSet":
        return cls(turn_id=turn_id, variants=[ConsoleVariant(content) for content in contents], selected_index=selected_index)

    @property
    def current(self) -> ConsoleVariant:
        return self.variants[self.selected_index]

    @property
    def can_go_previous(self) -> bool:
        return self.selected_index > 0

    @property
    def can_go_next(self) -> bool:
        return self.selected_index < len(self.variants) - 1
```

Keep this first implementation intentionally small. Add only fields needed by failing tests and immediately upcoming tasks.

- [ ] **Step 4: Run model tests**

Run:

```bash
python -m pytest -q Tests/Chat/test_console_chat_models.py --tb=short
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add tldw_chatbook/Chat/console_chat_models.py Tests/Chat/test_console_chat_models.py
git commit -m "Add native Console chat state models"
```

## Task 2: llama.cpp Provider Gateway

**Files:**
- Create: `tldw_chatbook/Chat/console_provider_gateway.py`
- Create: `Tests/Chat/test_console_provider_gateway.py`
- Modify: `tldw_chatbook/Chat/console_chat_models.py`

- [ ] **Step 1: Write failing provider resolution tests**

Add tests using `httpx.MockTransport`:

```python
import httpx
import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    LlamaCppProviderConfig,
)


@pytest.mark.asyncio
async def test_llamacpp_prefers_explicit_model_but_still_probes_reachability():
    seen_paths = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok"})

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:9099")
    )

    resolved = await gateway.resolve_llamacpp(
        LlamaCppProviderConfig(base_url="http://127.0.0.1:9099", explicit_model="explicit-model")
    )

    assert resolved.ready is True
    assert resolved.model == "explicit-model"
    assert seen_paths == ["/health"]


@pytest.mark.asyncio
async def test_llamacpp_prefers_configured_model_but_still_probes_reachability():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(404, text="no health route, but server is reachable")

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:9099")
    )

    resolved = await gateway.resolve_llamacpp(
        LlamaCppProviderConfig(base_url="http://127.0.0.1:9099", configured_model="configured-model")
    )

    assert resolved.ready is True
    assert resolved.model == "configured-model"


@pytest.mark.asyncio
async def test_llamacpp_explicit_model_blocks_when_reachability_probe_cannot_connect():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        raise httpx.ConnectError("connection refused", request=request)

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:9099")
    )

    resolved = await gateway.resolve_llamacpp(
        LlamaCppProviderConfig(base_url="http://127.0.0.1:9099", explicit_model="explicit-model")
    )

    assert resolved.ready is False
    assert resolved.model == "explicit-model"
    assert "not reachable" in resolved.visible_copy


@pytest.mark.asyncio
async def test_llamacpp_uses_first_models_endpoint_result_when_no_configured_model():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "server-model"}]})

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:9099")
    )

    resolved = await gateway.resolve_llamacpp(LlamaCppProviderConfig(base_url="http://127.0.0.1:9099"))

    assert resolved.ready is True
    assert resolved.model == "server-model"


@pytest.mark.asyncio
async def test_llamacpp_unreachable_server_returns_blocked_recovery_copy():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:9099")
    )

    resolved = await gateway.resolve_llamacpp(LlamaCppProviderConfig(base_url="http://127.0.0.1:9099"))

    assert resolved.ready is False
    assert resolved.model is None
    assert "Provider blocked" in resolved.visible_copy
    assert "127.0.0.1:9099" in resolved.visible_copy


@pytest.mark.asyncio
async def test_llamacpp_empty_models_without_configured_model_returns_blocked_recovery_copy():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:9099")
    )

    resolved = await gateway.resolve_llamacpp(LlamaCppProviderConfig(base_url="http://127.0.0.1:9099"))

    assert resolved.ready is False
    assert resolved.model is None
    assert resolved.visible_copy == "Provider blocked: select or configure a llama.cpp model."


@pytest.mark.asyncio
async def test_resolve_for_send_dispatches_llamacpp_selection():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "server-model"}]})

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:9099")
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="llama_cpp", base_url="http://127.0.0.1:9099")
    )

    assert resolved.ready is True
    assert resolved.provider == "llama_cpp"
    assert resolved.model == "server-model"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest -q Tests/Chat/test_console_provider_gateway.py --tb=short
```

Expected: import failure for `console_provider_gateway`.

- [ ] **Step 3: Implement resolution contracts**

Implement:

- `LlamaCppProviderConfig(base_url, explicit_model=None, configured_model=None)`
- `ConsoleProviderResolution(provider, base_url, model, ready, visible_copy)`
- `ConsoleProviderGateway.resolve_llamacpp()`
- `ConsoleProviderGateway.resolve_for_send(selection: ConsoleProviderSelection)`

Resolution order must match the spec:

1. explicit Console/provider model selection
2. configured `api_settings.llama_cpp` model/default
3. first compatible `/v1/models` result
4. blocked state

Do not call `/v1/models` when an explicit or configured model is already available. A local llama.cpp server may not expose a usable models endpoint even though an explicit model can stream successfully.

However, explicit/configured model readiness still must prove the server is reachable before accepting Send. Implement a lightweight reachability probe:

- first request `GET /health`
- any HTTP response from `/health` counts as reachable, including `404`, because it proves the server accepted a connection
- network exceptions, DNS errors, connection refusal, and timeouts are blocked provider states
- only use `/v1/models` for model discovery when no explicit/configured model is available

Blocked states must be specific enough for Console recovery:

- unreachable server: `Provider blocked: llama.cpp server is not reachable at http://127.0.0.1:9099. Start llama.cpp or update Console provider settings.`
- no resolvable model: `Provider blocked: select or configure a llama.cpp model.`
- unsupported provider: `WIP: Console native provider '<provider>' is not wired yet. Select llama.cpp for this slice.`

`resolve_for_send()` is the controller-facing interface. It must dispatch `provider in {"llama_cpp", "local_llamacpp"}` to `resolve_llamacpp()` and return a blocked resolution with explicit `WIP` copy for unsupported providers.

- [ ] **Step 4: Add streaming contract tests**

Add:

```python
@pytest.mark.asyncio
async def test_llamacpp_stream_chat_yields_content_chunks():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.method == "POST"
        body = (
            b"data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\n"
            b"data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n"
            b"data: [DONE]\n\n"
        )
        return httpx.Response(200, content=body)

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:9099")
    )

    chunks = [
        chunk
        async for chunk in gateway.stream_llamacpp_chat(
            base_url="http://127.0.0.1:9099",
            model="test-model",
            messages=[{"role": "user", "content": "say hello"}],
        )
    ]

    assert chunks == ["hel", "lo"]


@pytest.mark.asyncio
async def test_stream_chat_dispatches_llamacpp_resolution():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n",
        )

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:9099")
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="llama_cpp",
            base_url="http://127.0.0.1:9099",
            explicit_model="test-model",
        )
    )

    chunks = [chunk async for chunk in gateway.stream_chat(resolution, [{"role": "user", "content": "hello"}])]

    assert chunks == ["ok"]
```

- [ ] **Step 5: Implement minimal SSE parsing**

Implement:

- `stream_llamacpp_chat()` with `httpx.AsyncClient.stream()` and OpenAI-compatible `data:` line parsing
- `stream_chat(resolution, messages)` as the controller-facing dispatcher to llama.cpp streaming

Ignore `[DONE]`; yield only content chunks.

- [ ] **Step 6: Run provider tests**

Run:

```bash
python -m pytest -q Tests/Chat/test_console_provider_gateway.py --tb=short
```

Expected: pass.

- [ ] **Step 7: Commit**

Run:

```bash
git add tldw_chatbook/Chat/console_provider_gateway.py tldw_chatbook/Chat/console_chat_models.py Tests/Chat/test_console_provider_gateway.py
git commit -m "Add Console llama.cpp provider gateway"
```

## Task 3: Native Store And Persistence Facade

**Files:**
- Create: `tldw_chatbook/Chat/console_chat_store.py`
- Create: `Tests/Chat/test_console_chat_store.py`
- Modify: `tldw_chatbook/Chat/console_chat_models.py`

- [ ] **Step 1: Write failing in-memory store tests**

Add:

```python
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole, ConsoleWorkspaceContext
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


def test_store_creates_session_and_appends_messages():
    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1", workspace_id="global")

    user_message = store.append_message(session.id, role=ConsoleMessageRole.USER, content="hello")
    assistant_message = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="")

    assert store.active_session_id == session.id
    assert user_message.content == "hello"
    assert assistant_message.status == "pending"
    assert [message.role for message in store.messages_for_session(session.id)] == [
        ConsoleMessageRole.USER,
        ConsoleMessageRole.ASSISTANT,
    ]


def test_store_updates_streaming_message_and_marks_stopped():
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="")

    store.append_stream_chunk(message.id, "hel")
    store.append_stream_chunk(message.id, "lo")
    store.mark_message_stopped(message.id)

    updated = store.get_message(message.id)
    assert updated.content == "hello"
    assert updated.status == "stopped"


def test_store_tracks_active_workspace_context():
    context = ConsoleWorkspaceContext(active_workspace_id="workspace-a")
    store = ConsoleChatStore(workspace_context=context)

    assert store.workspace_context.active_workspace_id == "workspace-a"

    store.set_workspace_context(ConsoleWorkspaceContext(active_workspace_id="workspace-b"))

    assert store.workspace_context.active_workspace_id == "workspace-b"


def test_store_creates_and_switches_sessions():
    store = ConsoleChatStore()
    first = store.ensure_session(title="Chat 1")
    store.append_message(first.id, role=ConsoleMessageRole.USER, content="first")
    second = store.create_session(title="Chat 2")

    assert store.active_session_id == second.id

    store.switch_session(first.id)

    assert store.active_session_id == first.id
    assert store.messages_for_session(first.id)[0].content == "first"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest -q Tests/Chat/test_console_chat_store.py --tb=short
```

Expected: import failure for `console_chat_store`.

- [ ] **Step 3: Implement in-memory store**

Implement enough to pass:

- session creation
- active session ID
- multiple sessions, `create_session(...)`, and `switch_session(session_id)`
- `workspace_context` storage and `set_workspace_context(...)`
- message append
- message lookup
- chunk append
- mark stopped/failed/completed

Do not wire DB persistence yet.

- [ ] **Step 4: Add persistence adapter tests**

Use a fake persistence object first:

```python
class FakePersistence:
    def __init__(self):
        self.created_conversations = []
        self.created_messages = []

    def create_conversation(self, **kwargs):
        self.created_conversations.append(kwargs)
        return "conv-1"

    def create_message(self, **kwargs):
        self.created_messages.append(kwargs)
        return f"msg-{len(self.created_messages)}"


def test_store_can_persist_user_and_assistant_messages_through_adapter():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(title="Chat 1")

    store.persist_session_if_needed(session.id)
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hello", persist=True)

    assert persistence.created_conversations[0]["conversation_title"] == "Chat 1"
    assert persistence.created_messages[0]["sender"] == "user"
    assert persistence.created_messages[0]["content"] == "hello"
```

- [ ] **Step 5: Implement persistence hooks**

Use `ChatPersistenceService` when provided. Keep the store usable without persistence for mounted tests.

- [ ] **Step 6: Run store tests**

Run:

```bash
python -m pytest -q Tests/Chat/test_console_chat_store.py --tb=short
```

Expected: pass.

- [ ] **Step 7: Commit**

Run:

```bash
git add tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/console_chat_models.py Tests/Chat/test_console_chat_store.py
git commit -m "Add native Console chat store"
```

## Task 4: Console Chat Controller Send/Stop Flow

**Files:**
- Create: `tldw_chatbook/Chat/console_chat_controller.py`
- Create: `Tests/Chat/test_console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_chat_models.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`

- [ ] **Step 1: Write failing blocked-send test**

Add:

```python
import pytest

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus, ConsoleStagedSource, ConsoleWorkspaceContext


class BlockedGateway:
    async def resolve_for_send(self, selection):
        return type("Resolution", (), {
            "ready": False,
            "visible_copy": "Provider blocked: select a model",
        })()


@pytest.mark.asyncio
async def test_blocked_send_preserves_draft_and_adds_recovery_message():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=BlockedGateway())

    result = await controller.submit_draft("hello")

    assert result.accepted is False
    assert result.should_clear_draft is False
    assert controller.run_state.status is ConsoleRunStatus.BLOCKED
    assert "Provider blocked" in controller.run_state.visible_copy
    assert store.messages_for_session(store.active_session_id)[-1].role.value == "system"


@pytest.mark.asyncio
async def test_blocked_workspace_source_preserves_draft_and_skips_provider_call():
    class RecordingGateway(BlockedGateway):
        calls = 0

        async def resolve_for_send(self, selection):
            self.calls += 1
            return await super().resolve_for_send(selection)

    context = ConsoleWorkspaceContext(
        active_workspace_id="workspace-a",
        staged_sources=(
            ConsoleStagedSource(
                source_id="note-1",
                label="Workspace B note",
                source_type="note",
                workspace_id="workspace-b",
            ),
        ),
    )
    gateway = RecordingGateway()
    store = ConsoleChatStore(workspace_context=context)
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    result = await controller.submit_draft("hello")

    assert result.accepted is False
    assert result.should_clear_draft is False
    assert gateway.calls == 0
    assert controller.run_state.status is ConsoleRunStatus.BLOCKED
    assert "Workspace B note" in controller.run_state.visible_copy
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest -q Tests/Chat/test_console_chat_controller.py --tb=short
```

Expected: import failure for `console_chat_controller`.

- [ ] **Step 3: Implement blocked-send controller path**

Implement:

- `ConsoleChatController.submit_draft(draft: str)`
- `ConsoleSubmitResult(accepted, should_clear_draft, visible_copy)`
- empty draft validation
- workspace/source policy validation before provider resolution
- controller construction of `ConsoleProviderSelection` carrying `store.workspace_context`
- provider blocked recovery message

- [ ] **Step 4: Add streaming success test**

Add:

```python
class StreamingGateway:
    async def resolve_for_send(self, selection):
        return type("Resolution", (), {
            "ready": True,
            "provider": "llama_cpp",
            "model": "test-model",
            "base_url": "http://127.0.0.1:9099",
            "visible_copy": "",
        })()

    async def stream_chat(self, resolution, messages):
        for chunk in ("hel", "lo"):
            yield chunk


@pytest.mark.asyncio
async def test_submit_draft_streams_assistant_message_to_completion():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())

    result = await controller.submit_draft("hello")

    messages = store.messages_for_session(store.active_session_id)
    assert result.accepted is True
    assert result.should_clear_draft is True
    assert messages[-2].content == "hello"
    assert messages[-1].content == "hello"
    assert messages[-1].status == "complete"
    assert controller.run_state.status is ConsoleRunStatus.COMPLETED
```

- [ ] **Step 5: Implement streaming path**

Implement the minimal loop:

- build `ConsoleProviderSelection` from the current Console provider/model/base URL plus `store.workspace_context`
- create user message
- create pending assistant message
- set run state streaming
- append chunks to pending assistant message
- mark complete

- [ ] **Step 6: Add stop test**

Add a fake gateway that waits on an injected event. Verify `controller.stop_active_run()` marks the assistant message stopped and run state stopped.

- [ ] **Step 7: Add streaming failure and retry tests**

Add two tests before implementing recovery:

```python
class FailingStreamingGateway(StreamingGateway):
    async def stream_chat(self, resolution, messages):
        yield "partial"
        raise RuntimeError("llama.cpp stream failed")


@pytest.mark.asyncio
async def test_submit_draft_marks_assistant_failed_when_stream_errors():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=FailingStreamingGateway())

    result = await controller.submit_draft("hello")

    messages = store.messages_for_session(store.active_session_id)
    assert result.accepted is True
    assert result.should_clear_draft is True
    assert messages[-1].content == "partial"
    assert messages[-1].status == "failed"
    assert controller.run_state.status is ConsoleRunStatus.FAILED
    assert "stream failed" in controller.run_state.visible_copy


@pytest.mark.asyncio
async def test_retry_failed_message_streams_replacement_from_original_turn():
    store = ConsoleChatStore()
    failing = FailingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=failing)
    await controller.submit_draft("hello")
    failed_id = store.messages_for_session(store.active_session_id)[-1].id

    controller.provider_gateway = StreamingGateway()
    result = await controller.retry_message(failed_id)

    assert result.accepted is True
    assert store.get_message(failed_id).status == "complete"
    assert store.get_message(failed_id).content == "hello"


@pytest.mark.asyncio
async def test_retry_failed_message_records_retrying_then_streaming_transition():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=FailingStreamingGateway())
    await controller.submit_draft("hello")
    failed_id = store.messages_for_session(store.active_session_id)[-1].id

    observed = []

    class ObservingGateway(StreamingGateway):
        async def stream_chat(self, resolution, messages):
            observed.append(controller.run_state.status)
            yield "recovered"

    controller.provider_gateway = ObservingGateway()
    result = await controller.retry_message(failed_id)

    assert result.accepted is True
    assert ConsoleRunStatus.RETRYING in controller.run_state_history
    assert observed == [ConsoleRunStatus.STREAMING]
    assert controller.run_state.status is ConsoleRunStatus.COMPLETED
```

- [ ] **Step 8: Implement stop, failure, and retry**

Use an `asyncio.Event` or cancel handle owned by the controller. Do not rely on Textual widget worker state.

For streaming errors:

- mark the assistant message `failed`
- set run state `FAILED`
- expose visible recovery copy in `run_state.visible_copy`
- preserve enough turn context for `retry_message(message_id)` to stream a replacement without duplicating the user message
- expose `run_state_history` or an equivalent event hook in tests so `FAILED -> RETRYING -> STREAMING -> COMPLETED` can be verified

For retry:

- set run state `RETRYING` immediately with visible copy before resolving the provider
- set run state `STREAMING` once the replacement stream starts
- keep the failed message visible while retrying
- mark the same failed message `complete` if retry succeeds

Fallback from a failed primary provider to a secondary provider is not implemented in this slice. If exposed in UI copy, label it `WIP: fallback provider routing is not implemented yet`.

- [ ] **Step 9: Run controller tests**

Run:

```bash
python -m pytest -q Tests/Chat/test_console_chat_controller.py --tb=short
```

Expected: pass.

- [ ] **Step 10: Commit**

Run:

```bash
git add tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_chat_models.py tldw_chatbook/Chat/console_chat_store.py Tests/Chat/test_console_chat_controller.py
git commit -m "Add native Console chat controller"
```

## Task 5: Wire Console Composer To Native Controller

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Create: `Tests/UI/test_console_native_chat_flow.py`

- [ ] **Step 1: Write failing mounted blocked-send test**

Add:

```python
import pytest

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import ConsoleHarness, _visible_text
from tldw_chatbook.Widgets.Console import ConsoleComposerBar


async def _wait_for_text(screen, pilot, expected: str, *, attempts: int = 20) -> None:
    for _ in range(attempts):
        if expected in _visible_text(screen):
            return
        await pilot.pause(0.05)
    raise AssertionError(f"Text not found: {expected!r}")


@pytest.mark.asyncio
async def test_console_native_blocked_send_preserves_composer_text_and_shows_recovery():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "", "model": ""}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("blocked draft")

        await pilot.click("#console-send-message")
        await pilot.pause(0.2)

        assert composer.draft_text() == "blocked draft"
        assert "Provider blocked" in _visible_text(console)
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
python -m pytest -q Tests/UI/test_console_native_chat_flow.py::test_console_native_blocked_send_preserves_composer_text_and_shows_recovery --tb=short
```

Expected: fails because Send still routes through legacy behavior or does not render native recovery.

- [ ] **Step 3: Instantiate native controller in `ChatScreen`**

Add private helpers:

- `_ensure_console_chat_store()`
- `_ensure_console_provider_gateway()`
- `_ensure_console_chat_controller()`
- `_current_console_workspace_context()`
- `_sync_console_chat_core_state()`

Use `ChatPersistenceService` only when `app_instance.chachanotes_db` is available.

`_current_console_workspace_context()` must collect the active workspace ID and staged sources known to Console. If the current app state has no workspace model yet, use `ConsoleWorkspaceContext(active_workspace_id="global")` and keep this fallback explicit in code. Every send must update `store.workspace_context` before calling the controller so workspace/source policy checks use current UI state.

- [ ] **Step 4: Route composer send and stop buttons**

Add handlers:

- `@on(Button.Pressed, "#console-send-message")`
- `@on(Button.Pressed, "#console-stop-generation")`

Send handler:

1. Read `composer.draft_text()`.
2. Schedule `controller.submit_draft(...)` as a Textual worker or app-owned async task; do not block the UI loop.
3. Clear composer only when `result.should_clear_draft`.
4. Refresh transcript/inspector/status.

When the composer contains collapsed paste chips, the send handler must use `composer.draft_text()` as the exact expanded payload. It must never read the visible collapsed label such as `Pasted Text: 120 Characters` as message content.

Stop handler:

1. Call `controller.stop_active_run()`.
2. Refresh transcript/inspector/status.

- [ ] **Step 5: Run focused UI test**

Run:

```bash
python -m pytest -q Tests/UI/test_console_native_chat_flow.py::test_console_native_blocked_send_preserves_composer_text_and_shows_recovery --tb=short
```

Expected: pass.

- [ ] **Step 6: Add mounted streaming stop test**

Use a fake gateway that yields one chunk and then waits until stopped. Verify the visible controls and state transition:

```python
class WaitingGateway:
    def __init__(self):
        self.release = asyncio.Event()

    async def resolve_for_send(self, selection):
        return type("Resolution", (), {
            "ready": True,
            "provider": "llama_cpp",
            "model": "test-model",
            "base_url": "http://127.0.0.1:9099",
            "visible_copy": "",
        })()

    async def stream_chat(self, resolution, messages):
        yield "partial"
        await self.release.wait()


@pytest.mark.asyncio
async def test_console_stop_interrupts_stream_and_keeps_partial_message_visible():
    gateway = WaitingGateway()
    app = _build_test_app()
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        await pilot.click("#console-send-message")
        await _wait_for_text(console, pilot, "partial")
        assert "Stop" in _visible_text(console)
        assert "streaming" in _visible_text(console).lower()

        await pilot.click("#console-stop-generation")
        await _wait_for_text(console, pilot, "stopped")

        store = console._ensure_console_chat_store()
        messages = store.messages_for_session(store.active_session_id)
        assert messages[-1].content == "partial"
        assert messages[-1].status == "stopped"
```

The visible button label/state must make it clear that an active stream can be stopped. After stop, the inspector/status surfaces must show stopped state and the partial assistant response must remain in the transcript.

- [ ] **Step 7: Add mounted collapsed-paste send payload test**

Inject a capturing fake gateway and assert the controller receives expanded pasted text, not the collapsed composer display label:

```python
class CapturingGateway:
    def __init__(self):
        self.sent_messages = []

    async def resolve_for_send(self, selection):
        return type("Resolution", (), {
            "ready": True,
            "provider": "llama_cpp",
            "model": "test-model",
            "base_url": "http://127.0.0.1:9099",
            "visible_copy": "",
        })()

    async def stream_chat(self, resolution, messages):
        self.sent_messages.append(messages)
        yield "accepted"


@pytest.mark.asyncio
async def test_console_native_send_uses_expanded_paste_payload_not_collapsed_label():
    long_text = "x" * 80
    gateway = CapturingGateway()
    app = _build_test_app()
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_pasted_text(long_text)

        assert "Pasted Text: 80 Characters" in _visible_text(console)

        await pilot.click("#console-send-message")
        await pilot.pause(0.4)

        assert gateway.sent_messages[-1][-1]["content"] == long_text
        assert "Pasted Text: 80 Characters" not in gateway.sent_messages[-1][-1]["content"]
```

- [ ] **Step 8: Add mounted accepted-send test with fake gateway**

Inject a fake gateway through the app or screen seam so the test does not need network. Define the fake gateway in this test module instead of importing from another test file:

```python
class StreamingGateway:
    async def resolve_for_send(self, selection):
        return type("Resolution", (), {
            "ready": True,
            "provider": "llama_cpp",
            "model": "test-model",
            "base_url": "http://127.0.0.1:9099",
            "visible_copy": "",
        })()

    async def stream_chat(self, resolution, messages):
        for chunk in ("hel", "lo"):
            yield chunk


@pytest.mark.asyncio
async def test_console_native_send_clears_composer_after_acceptance_and_updates_store():
    app = _build_test_app()
    app.console_provider_gateway_factory = lambda: StreamingGateway()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        await pilot.click("#console-send-message")
        await pilot.pause(0.4)

        assert composer.draft_text() == ""
        store = console._ensure_console_chat_store()
        messages = store.messages_for_session(store.active_session_id)
        assert messages[-2].content == "hello"
        assert messages[-1].content == "hello"
```

- [ ] **Step 9: Implement test injection seam**

Prefer a narrowly named app attribute such as `console_provider_gateway_factory` only for tests and future dependency injection.

- [ ] **Step 10: Run Console flow tests**

Run:

```bash
python -m pytest -q Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_internals_decomposition.py --tb=short
```

Expected: pass.

- [ ] **Step 11: Commit**

Run:

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Console/console_composer_bar.py Tests/UI/test_console_native_chat_flow.py
git commit -m "Wire Console composer to native chat controller"
```

## Task 6: Native Transcript Widget

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_transcript.py`
- Modify: `tldw_chatbook/Widgets/Console/__init__.py`
- Modify: `tldw_chatbook/Widgets/Console/console_session_surface.py`
- Create: `Tests/UI/test_console_native_transcript.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`

- [ ] **Step 1: Write failing transcript rendering test**

Add:

```python
from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage, ConsoleMessageRole
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


def test_console_transcript_renderable_uses_full_width_rules():
    transcript = ConsoleTranscript()
    transcript.set_messages([
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hello"),
        ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="world"),
    ])

    plain = transcript.to_plain_text(width=40)

    assert "─" * 40 in plain
    assert "User" in plain
    assert "Assistant" in plain
    assert "hello" in plain
    assert "world" in plain
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
python -m pytest -q Tests/UI/test_console_native_transcript.py --tb=short
```

Expected: import failure for `console_transcript`.

- [ ] **Step 3: Implement minimal transcript widget**

Implement:

- `ConsoleTranscript.set_messages(messages)`
- compact separator rendering
- `to_plain_text(width)`
- no action row when no selection exists

- [ ] **Step 4: Add selection/action-row test**

Add:

```python
def test_console_transcript_selected_message_shows_action_row():
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer", id="m1")
    transcript = ConsoleTranscript()
    transcript.set_messages([message])
    transcript.select_message("m1")

    plain = transcript.to_plain_text(width=80)

    assert "Copy | Edit | Save as..." in plain
    assert "--->" in plain
```

- [ ] **Step 5: Implement selection model**

Implement:

- `selected_message_id`
- click/select API
- selected row classes
- action row rendering

Do not implement action dispatch in this task.

- [ ] **Step 6: Add keyboard navigation mounted test**

Use a mounted Console or transcript-only app:

```python
from textual.app import App, ComposeResult


class TranscriptHarness(App):
    def compose(self) -> ComposeResult:
        transcript = ConsoleTranscript(id="console-native-transcript")
        transcript.set_messages([
            ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hello", id="m1"),
            ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer", id="m2"),
        ])
        yield transcript


@pytest.mark.asyncio
async def test_console_transcript_keyboard_selects_messages_and_enter_shows_actions():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.press("tab")
        await pilot.press("down")
        await pilot.press("enter")
        text = _visible_text(app)

    assert "Copy | Edit | Save as..." in text
```

- [ ] **Step 7: Add mouse click selection test**

```python
@pytest.mark.asyncio
async def test_console_transcript_click_selects_message_and_shows_actions():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.click("#console-message-m2")
        text = _visible_text(app)

    assert "Copy | Edit | Save as..." in text
```

- [ ] **Step 8: Add mounted Tab reachability smoke test**

Make sure keyboard users can move between the main Console areas after the native transcript exists:

```python
@pytest.mark.asyncio
async def test_console_tab_reaches_composer_and_transcript_regions():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        seen_focus_ids = set()
        for _ in range(12):
            focused = getattr(console.app, "focused", None)
            if focused is not None and getattr(focused, "id", None):
                seen_focus_ids.add(focused.id)
            await pilot.press("tab")

    assert "console-native-composer" in seen_focus_ids
    assert "console-native-transcript" in seen_focus_ids
```

- [ ] **Step 9: Implement focus and key bindings**

Implement transcript-local handling for:

- `up`
- `down`
- `j`
- `k`
- `enter`
- `escape`
- pointer click on a message row/card selects that message

Keep handling active only while transcript has focus.

- [ ] **Step 10: Mount transcript through a migration seam**

Change `ConsoleSessionSurface` to accept and render `ConsoleTranscript` state through a narrow seam without removing `ChatTabContainer` yet.

The intent of this task is to prove native transcript rendering and selection while preserving the legacy chat tab dependency until Task 9.

Mount `ConsoleTranscript(id="console-native-transcript")` in `ConsoleSessionSurface` in this task. `ChatTabContainer` may remain mounted only as an inactive hidden fallback during the migration, but the native transcript must be the visible/focusable surface used by the tests added in this task.

Do not add the “no `ChatTabContainer` mounted” guardrail in this task. That guardrail belongs to Task 9 after the native transcript/action path is complete.

- [ ] **Step 11: Update CSS source and regenerate CSS**

Edit `tldw_chatbook/css/components/_agentic_terminal.tcss`.

Run:

```bash
python tldw_chatbook/css/build_css.py
```

Expected: regenerated modular CSS changes only from source TCSS.

- [ ] **Step 12: Run transcript tests**

Run:

```bash
python -m pytest -q Tests/UI/test_console_native_transcript.py Tests/UI/test_console_native_chat_flow.py --tb=short
```

Expected: pass.

- [ ] **Step 13: Commit**

Run:

```bash
git add tldw_chatbook/Widgets/Console/console_transcript.py tldw_chatbook/Widgets/Console/__init__.py tldw_chatbook/Widgets/Console/console_session_surface.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_console_native_transcript.py Tests/UI/test_console_native_chat_flow.py
git commit -m "Render Console transcript natively"
```

## Task 7: Message Actions And Save As Modal

**Files:**
- Create: `tldw_chatbook/Chat/console_message_actions.py`
- Create: `tldw_chatbook/Widgets/Console/console_save_as_modal.py`
- Create: `Tests/Chat/test_console_message_actions.py`
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/UI/test_console_native_transcript.py`
- Modify: `Tests/UI/test_console_native_chat_flow.py`

- [ ] **Step 1: Write failing action availability tests**

Add:

```python
from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage, ConsoleMessageRole
from tldw_chatbook.Chat.console_message_actions import ConsoleMessageActionService


def test_assistant_message_actions_include_required_order():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer")

    actions = service.available_actions(message)

    assert [action.label for action in actions] == [
        "Copy",
        "Edit",
        "Save as...",
        "♻",
        "--->",
        "👍/👎",
        "🗑",
    ]


def test_streaming_assistant_message_only_exposes_stop_safe_actions():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="partial",
        status="streaming",
    )

    actions = service.available_actions(message)

    assert "Copy" not in [action.label for action in actions]
    assert "Save as..." not in [action.label for action in actions]
    assert "♻" not in [action.label for action in actions]
    assert all(action.disabled_reason for action in actions if not action.enabled)


def test_pending_assistant_message_does_not_expose_completed_message_actions():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        status="pending",
    )

    assert service.available_actions(message) == []


def test_unavailable_save_destinations_are_explicit_wip():
    service = ConsoleMessageActionService(available_save_destinations={"Chatbook"})
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer")

    destinations = service.save_as_destinations(message)

    note = next(destination for destination in destinations if destination.label == "Note")
    assert note.available is False
    assert "WIP" in note.reason
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
python -m pytest -q Tests/Chat/test_console_message_actions.py --tb=short
```

Expected: import failure.

- [ ] **Step 3: Implement action availability**

Implement pure action definitions first. Include textual fallbacks if emoji rendering is unavailable later, but keep canonical order and labels in the pure service.

Action availability must be gated by message/run status:

- completed user/assistant messages may expose completed-message actions
- failed assistant messages may expose retry plus safe recovery actions
- streaming/pending assistant messages must not expose Copy, Save as..., Regenerate, Continue, feedback, or Delete as active completed-message actions
- disabled actions must carry visible disabled/WIP reasons when rendered

- [ ] **Step 4: Add action dispatch tests for safe operations**

Test:

- Copy returns `ConsoleActionResult(status="completed", clipboard_text=<message content>)` so `ChatScreen` can write that content to the clipboard
- Continue returns selected content/turn target
- Variant previous/next changes selected variant
- Unimplemented actions return `status="wip"` with visible reason

- [ ] **Step 5: Implement safe action dispatch**

Do not wire destructive DB delete until confirmation behavior is designed. For the first pass, delete may return explicit `WIP: delete confirmation not implemented`.

`ChatScreen` must write `ConsoleActionResult.clipboard_text` to the app clipboard path used elsewhere in the project when Copy succeeds. If the project has no shared clipboard helper, keep the service pure and perform the Textual/app clipboard operation in the screen handler.

- [ ] **Step 6: Write Save-as modal mounted test**

Add a test that opens the modal and asserts available plus WIP entries:

```python
from textual.app import App

from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import _visible_text
from tldw_chatbook.Chat.console_message_actions import ConsoleSaveDestination
from tldw_chatbook.Widgets.Console.console_save_as_modal import ConsoleSaveAsModal


class SaveAsModalHarness(App):
    async def on_mount(self) -> None:
        await self.push_screen(
            ConsoleSaveAsModal(
                destinations=[
                    ConsoleSaveDestination(label="Chatbook", available=True, reason=""),
                    ConsoleSaveDestination(label="Note", available=False, reason="WIP: save as Note is not wired yet."),
                ]
            )
        )


@pytest.mark.asyncio
async def test_save_as_modal_lists_available_and_wip_destinations():
    app = SaveAsModalHarness()

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.1)
        text = _visible_text(app)

    assert "Chatbook" in text
    assert "Note" in text
    assert "WIP: save as Note is not wired yet." in text
```

- [ ] **Step 7: Implement `ConsoleSaveAsModal`**

Render destinations with:

- label
- availability
- disabled state
- reason for WIP/unavailable entries

- [ ] **Step 8: Wire transcript action events to `ChatScreen`**

Use typed message events or button IDs from `ConsoleTranscript`.

`ChatScreen` should call `ConsoleMessageActionService` and then:

- notify or update status for Copy/WIP
- open `ConsoleSaveAsModal` for `Save as...`
- call `controller.retry_message(message_id)` when a failed assistant message exposes the retry action
- call controller continuation/regeneration methods for `--->` and `♻` where implemented

Use stable transcript action button IDs so mounted tests and CDP can target actions reliably:

- `console-message-action-copy-<message_id>`
- `console-message-action-edit-<message_id>`
- `console-message-action-save-as-<message_id>`
- `console-message-action-retry-<message_id>`
- `console-message-action-regenerate-<message_id>`
- `console-message-action-continue-<message_id>`
- `console-message-action-feedback-up-<message_id>`
- `console-message-action-feedback-down-<message_id>`
- `console-message-action-delete-<message_id>`

Keyboard activation must use the same dispatch path as pointer activation for every action. Implement action-row focus movement and Enter activation so Copy, Edit, Save as..., Retry, Regenerate, Continue, feedback, and Delete all emit the same typed event/button handler used by click.

- [ ] **Step 9: Add mounted failed-stream retry recovery test**

Use a gateway that fails on the first call and streams successfully on retry:

```python
class FailThenRecoverGateway:
    def __init__(self):
        self.calls = 0

    async def resolve_for_send(self, selection):
        return type("Resolution", (), {
            "ready": True,
            "provider": "llama_cpp",
            "model": "test-model",
            "base_url": "http://127.0.0.1:9099",
            "visible_copy": "",
        })()

    async def stream_chat(self, resolution, messages):
        self.calls += 1
        if self.calls == 1:
            yield "partial"
            raise RuntimeError("llama.cpp stream failed")
        yield "recovered"


@pytest.mark.asyncio
async def test_console_failed_stream_renders_inline_retry_and_recovers():
    gateway = FailThenRecoverGateway()
    app = _build_test_app()
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        await pilot.click("#console-send-message")
        await _wait_for_text(console, pilot, "llama.cpp stream failed")
        assert "Retry" in _visible_text(console)

        failed_id = console._ensure_console_chat_store().messages_for_session(
            console._ensure_console_chat_store().active_session_id
        )[-1].id
        await pilot.click(f"#console-message-action-retry-{failed_id}")
        await _wait_for_text(console, pilot, "recovered")

        assert console._ensure_console_chat_store().get_message(failed_id).status == "complete"
```

The failed transcript row must include inline visible error copy plus a retry action. A user must not need to infer recovery from logs or hidden state.

- [ ] **Step 10: Add keyboard action activation parity test**

Mount a transcript with a selected failed assistant message, focus each action button by ID, activate with Enter, and assert the same event/action result as clicking the button:

```python
@pytest.mark.parametrize(
    "action_id,button_prefix",
    [
        ("copy", "console-message-action-copy"),
        ("save-as", "console-message-action-save-as"),
        ("retry", "console-message-action-retry"),
        ("regenerate", "console-message-action-regenerate"),
        ("continue", "console-message-action-continue"),
        ("feedback-up", "console-message-action-feedback-up"),
        ("feedback-down", "console-message-action-feedback-down"),
        ("delete", "console-message-action-delete"),
    ],
)
@pytest.mark.asyncio
async def test_console_action_row_enter_matches_click_for_all_actions(action_id, button_prefix):
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        failed = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="partial",
            status="failed",
        )
        console._sync_console_chat_core_state()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(failed.id)
        console._sync_console_chat_core_state()

        button_id = f"#{button_prefix}-{failed.id}"
        await pilot.click(button_id)
        clicked_action = console._last_console_action.action_id

        console._last_console_action = None
        transcript.focus_action(failed.id, action_id)
        await pilot.press("enter")
        keyboard_action = console._last_console_action.action_id

        assert keyboard_action == clicked_action == action_id
```

If the final implementation tracks action results with a different test seam, assert that Enter and click both call the same `ChatScreen` handler/action ID rather than only checking text. `ConsoleTranscript.focus_action(message_id, action_id)` may be a test-visible helper; the production behavior still needs normal Tab/arrow focus movement.

- [ ] **Step 11: Add mounted Escape collapse test**

```python
@pytest.mark.asyncio
async def test_console_transcript_escape_collapses_selected_action_row():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="answer")
        console._sync_console_chat_core_state()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        console._sync_console_chat_core_state()
        assert "Save as..." in _visible_text(console)

        await pilot.press("escape")

        assert "Save as..." not in _visible_text(console)
        assert transcript.selected_message_id is None
```

- [ ] **Step 12: Run action tests**

Run:

```bash
python -m pytest -q Tests/Chat/test_console_message_actions.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_native_chat_flow.py --tb=short
```

Expected: pass.

- [ ] **Step 13: Commit**

Run:

```bash
git add tldw_chatbook/Chat/console_message_actions.py tldw_chatbook/Widgets/Console/console_save_as_modal.py tldw_chatbook/Widgets/Console/console_transcript.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Chat/test_console_message_actions.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_native_chat_flow.py
git commit -m "Add Console selected-message actions"
```

## Task 8: Regeneration, Variant Navigation, And Continue Flow

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Modify: `tldw_chatbook/Chat/console_message_actions.py`
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/Chat/test_console_chat_controller.py`
- Modify: `Tests/Chat/test_console_chat_store.py`
- Modify: `Tests/Chat/test_console_message_actions.py`
- Modify: `Tests/UI/test_console_native_chat_flow.py`
- Modify: `Tests/UI/test_console_native_transcript.py`

- [ ] **Step 1: Write failing variant store test**

Add:

```python
def test_store_adds_regenerated_variant_and_selects_it():
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="first")

    store.add_variant(message.id, "second")

    updated = store.get_message(message.id)
    assert updated.variants.current.content == "second"
    assert updated.variants.can_go_previous is True
```

- [ ] **Step 2: Implement store variants**

Keep variants in native state first. Add persistence only if existing DB variant methods are straightforward to call safely.

- [ ] **Step 3: Write failing controller regenerate test**

Use fake gateway chunks to create second variant for the selected assistant message.

- [ ] **Step 4: Implement controller regenerate**

Regeneration should:

1. Resolve selected assistant turn.
2. Create a pending variant.
3. Stream chunks into that variant.
4. Select the new variant on completion.

- [ ] **Step 5: Add transcript `<` / `>` rendering test**

Assert action row includes `<` and `>` when selected message has variants, and add a display test proving variant navigation changes the rendered transcript content:

```python
def test_console_transcript_variant_navigation_changes_displayed_content():
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="first", id="m1")
    message.variants = ConsoleVariantSet.from_contents(turn_id="turn-1", contents=["first", "second"])
    transcript = ConsoleTranscript()
    transcript.set_messages([message])
    transcript.select_message("m1")

    assert "first" in transcript.to_plain_text(width=80)

    transcript.select_next_variant("m1")

    rendered = transcript.to_plain_text(width=80)
    assert "second" in rendered
    assert "first" not in rendered
```

- [ ] **Step 6: Implement variant action row**

Show:

```text
Copy | Edit | Save as... | < | > | ♻ | ---> | 👍/👎          🗑
```

Disable `<` or `>` when no previous/next variant exists.

- [ ] **Step 7: Add continue action service and controller tests**

Add action-service coverage proving `--->` resolves the selected message or selected variant as the continuation target:

```python
def test_continue_action_targets_selected_variant_content():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="first", id="m1")
    message.variants = ConsoleVariantSet.from_contents(turn_id="turn-1", contents=["first", "second"], selected_index=1)

    result = service.dispatch("continue", message)

    assert result.status == "continue_requested"
    assert result.target_message_id == "m1"
    assert result.target_content == "second"
```

Add controller coverage proving continuation streams a new assistant turn from the selected target without overwriting the original message:

```python
@pytest.mark.asyncio
async def test_continue_from_message_streams_new_assistant_turn_after_selected_message():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    source = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="seed")

    result = await controller.continue_from_message(source.id)

    messages = store.messages_for_session(session.id)
    assert result.accepted is True
    assert messages[-1].role is ConsoleMessageRole.ASSISTANT
    assert messages[-1].content == "hello"
    assert messages[-1].id != source.id
```

- [ ] **Step 8: Implement continue flow**

Implement:

- `ConsoleMessageActionService.dispatch("continue", message)` returning a typed continue request
- `ConsoleChatController.continue_from_message(message_id)` that builds provider messages from session history through the selected message/variant and appends a new assistant response
- visible UI dispatch from the transcript `--->` action to the controller
- keyboard Enter activation for the selected `--->` action where the transcript action row has focus

The action label remains `--->` and means “continue/extend this message’s thread of thought”.

- [ ] **Step 9: Add mounted UI continue test**

Use a fake streaming gateway and mounted Console:

```python
@pytest.mark.asyncio
async def test_console_continue_action_streams_new_message_from_selected_turn():
    app = _build_test_app()
    app.console_provider_gateway_factory = lambda: StreamingGateway()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session(title="Chat 1")
        source = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="seed")
        console._sync_console_chat_core_state()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(source.id)
        console._sync_console_chat_core_state()

        await pilot.click(f"#console-message-action-continue-{source.id}")
        await pilot.pause(0.4)

        messages = store.messages_for_session(session.id)
        assert messages[-1].role is ConsoleMessageRole.ASSISTANT
        assert messages[-1].content == "hello"
        assert messages[-1].id != source.id
```

- [ ] **Step 10: Run variant and continue tests**

Run:

```bash
python -m pytest -q Tests/Chat/test_console_chat_store.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_message_actions.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_native_chat_flow.py --tb=short
```

Expected: pass.

- [ ] **Step 11: Commit**

Run:

```bash
git add tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/console_message_actions.py tldw_chatbook/Widgets/Console/console_transcript.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_chat_store.py Tests/Chat/test_console_message_actions.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_native_chat_flow.py
git commit -m "Add Console response variants and continue flow"
```

## Task 9: Replace Legacy Console Transcript Dependency

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Modify: `tldw_chatbook/Widgets/Console/console_session_surface.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/Chat/test_console_chat_controller.py`
- Modify: `Tests/Chat/test_console_chat_store.py`
- Modify: `Tests/UI/test_console_internals_decomposition.py`
- Modify: `Tests/UI/test_console_native_chat_flow.py`

- [ ] **Step 1: Add guardrail proving `ChatTabContainer` is not mounted in Console**

Add to existing Gate 1.5 tests:

```python
assert len(console.query("#console-chat-tabs")) == 0
assert len(console.query("ChatTabContainer")) == 0
```

If querying by class name is unreliable, query for durable legacy IDs/classes produced by `ChatTabContainer` and `ChatSession`.

- [ ] **Step 2: Run guardrail and verify failure**

Run:

```bash
python -m pytest -q Tests/UI/test_console_internals_decomposition.py::test_console_gate15_does_not_mount_full_legacy_chat_window_chrome --tb=short
```

Expected: failure while `ConsoleSessionSurface` still mounts `ChatTabContainer`.

- [ ] **Step 3: Remove `ChatTabContainer` from `ConsoleSessionSurface`**

`ConsoleSessionSurface` should own:

- title row
- native tab strip header
- `ConsoleTranscript`

It should not create `ChatTabContainer`.

The native tab strip is not optional in the final Console migration. It must replace the existing Console `New tab`/session switching behavior with controller-owned state:

- `#console-session-tab-<session_id>` buttons for existing sessions
- `#console-new-chat-tab` button for creating a new session
- active tab styling based on `ConsoleChatStore.active_session_id`
- `ConsoleChatController.new_session()` delegates to `store.create_session(...)`
- `ConsoleChatController.switch_session(session_id)` delegates to `store.switch_session(...)`

- [ ] **Step 4: Add native tab/session switching tests**

Add store/controller tests:

```python
def test_controller_creates_and_switches_sessions():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    first = store.ensure_session(title="Chat 1")
    second = controller.new_session(title="Chat 2")

    assert store.active_session_id == second.id

    controller.switch_session(first.id)

    assert store.active_session_id == first.id
```

Add mounted UI coverage:

```python
@pytest.mark.asyncio
async def test_console_native_tab_strip_creates_and_switches_sessions():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        console._sync_console_chat_core_state()

        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        assert second != first.id
        await _wait_for_selector(console, pilot, f"#console-session-tab-{second}")
        assert "Chat 2" in _visible_text(console)

        await pilot.click(f"#console-session-tab-{first.id}")

        assert store.active_session_id == first.id
```

- [ ] **Step 5: Adapt `ChatScreen` state save/restore seams**

Where `ChatScreen` currently queries `_get_tab_container()` for Console, add native-store equivalents. Preserve legacy behavior only for direct legacy chat screens.

- [ ] **Step 6: Run focused regressions**

Run:

```bash
python -m pytest -q Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_native_transcript.py --tb=short
```

Expected: pass.

- [ ] **Step 7: Commit**

Run:

```bash
git add tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Widgets/Console/console_session_surface.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_chat_store.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_native_chat_flow.py
git commit -m "Remove legacy chat tab dependency from Console"
```

## Task 10: Live llama.cpp And CDP QA Evidence

**Files:**
- Create: `Docs/superpowers/qa/product-maturity/console-native-chat-core/README.md`
- Create: `Docs/superpowers/qa/product-maturity/console-native-chat-core/2026-05-21-console-native-chat-core.md`
- Modify: `backlog/tasks/<task-id>.md`

- [ ] **Step 1: Run focused automated verification**

Run:

```bash
python -m pytest -q Tests/Chat/test_console_chat_models.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_console_chat_store.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_message_actions.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_internals_decomposition.py --tb=short
```

Expected: pass.

- [ ] **Step 2: Run diff checks**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 3: Verify local llama.cpp health manually**

Run:

```bash
curl -sf http://127.0.0.1:9099/health || curl -sf http://127.0.0.1:9099/v1/models
```

Expected: either command receives an HTTP response from the local llama.cpp server. `/health` may return a non-2xx response on some builds; if that happens, verify reachability with the app/provider gateway or `curl -i http://127.0.0.1:9099/health` and record the response. `/v1/models` does not need to return a non-empty list when an explicit/configured model is used.

Then run one live Console send through the app against the explicit/configured llama.cpp model and verify a streamed assistant response appears. If the server is unreachable or no explicit/configured model can stream, stop the closeout and document the task/PR as blocked. Do not mark the Backlog task Done, do not claim live llama.cpp approval, and do not open/mark the PR ready for review until this gate passes or the user explicitly waives live llama.cpp verification for this slice.

- [ ] **Step 4: Launch textual-web/CDP QA**

Use the project’s documented CDP/textual-web process:

```text
Docs/superpowers/qa/product-maturity/screen-qa/textual-web-cdp-debugging.md
```

Capture actual rendered screenshots for:

- idle Console
- typed composer
- llama.cpp streaming
- stopped llama.cpp stream with partial assistant message visible
- completed response
- selected message with action row
- failed stream with inline retry action
- retried failed stream after recovery
- regenerated message with `<` / `>` controls
- blocked provider recovery
- WIP/unavailable action state

Do not use SVG/code mockups as approval artifacts.
If any required CDP screenshot cannot be captured, stop closeout and document the missing evidence as a blocker. Screenshot approval is a signoff gate, not an optional appendix.

- [ ] **Step 5: Write QA evidence**

In `Docs/superpowers/qa/product-maturity/console-native-chat-core/2026-05-21-console-native-chat-core.md`, include:

- commands run
- automated test results
- live llama.cpp result
- CDP screenshot filenames/paths
- user approval status for each screenshot
- explicit pass/fail status for the live llama.cpp gate
- remaining known limitations

- [ ] **Step 6: Update QA README**

Add an index entry in `Docs/superpowers/qa/product-maturity/console-native-chat-core/README.md`.

- [ ] **Step 7: Update Backlog task**

Mark acceptance criteria complete only after tests, live llama.cpp verification, and screenshot approvals are complete.

Run:

```bash
backlog task edit <task-id> -s Done --notes "Implemented Console native chat core with native send/stream/stop/recovery, transcript selection/actions, explicit WIP paths, and CDP-approved Console states. Verification recorded in Docs/superpowers/qa/product-maturity/console-native-chat-core/2026-05-21-console-native-chat-core.md."
```

- [ ] **Step 8: Commit QA closeout**

Run:

```bash
git add Docs/superpowers/qa/product-maturity/console-native-chat-core backlog/tasks
git commit -m "Record Console native chat core QA"
```

## Final Verification Before PR

Run focused verification:

```bash
python -m pytest -q Tests/Chat/test_console_chat_models.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_console_chat_store.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_message_actions.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_internals_decomposition.py --tb=short
git diff --check
```

If the Backlog task is being moved to `Done`, also run the broad repo verification gate or document any inherited baseline failures separately from this slice:

```bash
python -m pytest -q --tb=short
```

Expected:

- focused tests pass
- `git diff --check` has no output
- full-suite status is recorded before marking the Backlog task Done
- live llama.cpp health and streaming evidence are documented
- required CDP screenshots are captured and user-approved before visual claims are made
- if live llama.cpp or required CDP evidence is unavailable, the work remains blocked/in progress unless the user explicitly waives that gate

## PR Guidance

Prefer multiple PRs if the implementation becomes large:

1. Core models/store/provider gateway.
2. Controller and Console composer wiring.
3. Native transcript and selected-message actions.
4. Variant navigation, legacy transcript removal, and QA closeout.

Each PR must keep Console usable and include focused tests. Do not batch UI screenshot approval claims across screens.
