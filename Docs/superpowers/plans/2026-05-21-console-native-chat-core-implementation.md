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
    ConsoleRunState,
    ConsoleRunStatus,
    ConsoleVariantSet,
)


def test_run_state_blocks_with_visible_recovery_copy():
    state = ConsoleRunState.blocked("Provider blocked: select a model")

    assert state.status is ConsoleRunStatus.BLOCKED
    assert state.visible_copy == "Provider blocked: select a model"
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


@dataclass(frozen=True)
class ConsoleRunState:
    status: ConsoleRunStatus = ConsoleRunStatus.IDLE
    visible_copy: str = ""

    @classmethod
    def blocked(cls, visible_copy: str) -> "ConsoleRunState":
        return cls(ConsoleRunStatus.BLOCKED, visible_copy)

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

from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    LlamaCppProviderConfig,
)


@pytest.mark.asyncio
async def test_llamacpp_prefers_explicit_model_over_models_endpoint():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "server-model"}]})

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:9099")
    )

    resolved = await gateway.resolve_llamacpp(
        LlamaCppProviderConfig(base_url="http://127.0.0.1:9099", explicit_model="explicit-model")
    )

    assert resolved.ready is True
    assert resolved.model == "explicit-model"


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

Resolution order must match the spec:

1. explicit Console/provider model selection
2. configured `api_settings.llama_cpp` model/default
3. first compatible `/v1/models` result
4. blocked state

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
```

- [ ] **Step 5: Implement minimal SSE parsing**

Implement `stream_llamacpp_chat()` with `httpx.AsyncClient.stream()` and OpenAI-compatible `data:` line parsing. Ignore `[DONE]`; yield only content chunks.

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
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
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
from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus


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

- create user message
- create pending assistant message
- set run state streaming
- append chunks to pending assistant message
- mark complete

- [ ] **Step 6: Add stop test**

Add a fake gateway that waits on an injected event. Verify `controller.stop_active_run()` marks the assistant message stopped and run state stopped.

- [ ] **Step 7: Implement stop**

Use an `asyncio.Event` or cancel handle owned by the controller. Do not rely on Textual widget worker state.

- [ ] **Step 8: Run controller tests**

Run:

```bash
python -m pytest -q Tests/Chat/test_console_chat_controller.py --tb=short
```

Expected: pass.

- [ ] **Step 9: Commit**

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
- `_sync_console_chat_core_state()`

Use `ChatPersistenceService` only when `app_instance.chachanotes_db` is available.

- [ ] **Step 4: Route composer send and stop buttons**

Add handlers:

- `@on(Button.Pressed, "#console-send-message")`
- `@on(Button.Pressed, "#console-stop-generation")`

Send handler:

1. Read `composer.draft_text()`.
2. Schedule `controller.submit_draft(...)` as a Textual worker or app-owned async task; do not block the UI loop.
3. Clear composer only when `result.should_clear_draft`.
4. Refresh transcript/inspector/status.

Stop handler:

1. Call `controller.stop_active_run()`.
2. Refresh transcript/inspector/status.

- [ ] **Step 5: Run focused UI test**

Run:

```bash
python -m pytest -q Tests/UI/test_console_native_chat_flow.py::test_console_native_blocked_send_preserves_composer_text_and_shows_recovery --tb=short
```

Expected: pass.

- [ ] **Step 6: Add mounted accepted-send test with fake gateway**

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
async def test_console_native_send_clears_composer_after_acceptance_and_renders_messages():
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
        text = _visible_text(console)
        assert "hello" in text
```

- [ ] **Step 7: Implement test injection seam**

Prefer a narrowly named app attribute such as `console_provider_gateway_factory` only for tests and future dependency injection.

- [ ] **Step 8: Run Console flow tests**

Run:

```bash
python -m pytest -q Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_internals_decomposition.py --tb=short
```

Expected: pass.

- [ ] **Step 9: Commit**

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
@pytest.mark.asyncio
async def test_console_transcript_keyboard_selects_messages_and_enter_shows_actions():
    ...
    await pilot.press("tab")
    await pilot.press("down")
    await pilot.press("enter")
    assert "Copy | Edit | Save as..." in _visible_text(app)
```

- [ ] **Step 7: Implement focus and key bindings**

Implement transcript-local handling for:

- `up`
- `down`
- `j`
- `k`
- `enter`
- `escape`

Keep handling active only while transcript has focus.

- [ ] **Step 8: Replace `ConsoleSessionSurface` internals**

Change `ConsoleSessionSurface` to render `ConsoleTranscript` instead of `ChatTabContainer`.

Keep the title and tab header affordance if needed, but remove native Console dependency on `ChatTabContainer` for transcript messages.

- [ ] **Step 9: Update CSS source and regenerate CSS**

Edit `tldw_chatbook/css/components/_agentic_terminal.tcss`.

Run:

```bash
python tldw_chatbook/css/build_css.py
```

Expected: regenerated modular CSS changes only from source TCSS.

- [ ] **Step 10: Run transcript tests**

Run:

```bash
python -m pytest -q Tests/UI/test_console_native_transcript.py Tests/UI/test_console_native_chat_flow.py --tb=short
```

Expected: pass.

- [ ] **Step 11: Commit**

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

- [ ] **Step 4: Add action dispatch tests for safe operations**

Test:

- Copy returns `ConsoleActionResult(status="completed")`
- Continue returns selected content/turn target
- Variant previous/next changes selected variant
- Unimplemented actions return `status="wip"` with visible reason

- [ ] **Step 5: Implement safe action dispatch**

Do not wire destructive DB delete until confirmation behavior is designed. For the first pass, delete may return explicit `WIP: delete confirmation not implemented`.

- [ ] **Step 6: Write Save-as modal mounted test**

Add a test that opens the modal and asserts available plus WIP entries:

```python
@pytest.mark.asyncio
async def test_save_as_modal_lists_available_and_wip_destinations():
    ...
    assert "Chatbook" in _visible_text(app)
    assert "Note" in _visible_text(app)
    assert "WIP" in _visible_text(app)
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
- call controller continuation/regeneration methods for `--->` and `♻` where implemented

- [ ] **Step 9: Run action tests**

Run:

```bash
python -m pytest -q Tests/Chat/test_console_message_actions.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_native_chat_flow.py --tb=short
```

Expected: pass.

- [ ] **Step 10: Commit**

Run:

```bash
git add tldw_chatbook/Chat/console_message_actions.py tldw_chatbook/Widgets/Console/console_save_as_modal.py tldw_chatbook/Widgets/Console/console_transcript.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Chat/test_console_message_actions.py Tests/UI/test_console_native_transcript.py
git commit -m "Add Console selected-message actions"
```

## Task 8: Regeneration And Variant Navigation

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Modify: `tldw_chatbook/Chat/console_message_actions.py`
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py`
- Modify: `Tests/Chat/test_console_chat_controller.py`
- Modify: `Tests/Chat/test_console_chat_store.py`
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

Assert action row includes `<` and `>` when selected message has variants.

- [ ] **Step 6: Implement variant action row**

Show:

```text
Copy | Edit | Save as... | < | > | ♻ | ---> | 👍/👎          🗑
```

Disable `<` or `>` when no previous/next variant exists.

- [ ] **Step 7: Run variant tests**

Run:

```bash
python -m pytest -q Tests/Chat/test_console_chat_store.py Tests/Chat/test_console_chat_controller.py Tests/UI/test_console_native_transcript.py --tb=short
```

Expected: pass.

- [ ] **Step 8: Commit**

Run:

```bash
git add tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/console_message_actions.py tldw_chatbook/Widgets/Console/console_transcript.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_chat_store.py Tests/UI/test_console_native_transcript.py
git commit -m "Add Console response variants"
```

## Task 9: Replace Legacy Console Transcript Dependency

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_session_surface.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
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
- optional native tab strip header
- `ConsoleTranscript`

It should not create `ChatTabContainer`.

- [ ] **Step 4: Adapt `ChatScreen` state save/restore seams**

Where `ChatScreen` currently queries `_get_tab_container()` for Console, add native-store equivalents. Preserve legacy behavior only for direct legacy chat screens.

- [ ] **Step 5: Run focused regressions**

Run:

```bash
python -m pytest -q Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_native_transcript.py --tb=short
```

Expected: pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add tldw_chatbook/Widgets/Console/console_session_surface.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_native_chat_flow.py
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
curl -sf http://127.0.0.1:9099/v1/models
```

Expected: JSON response with a model list. If unavailable, document as blocked and do not claim live llama.cpp approval.

- [ ] **Step 4: Launch textual-web/CDP QA**

Use the project’s documented CDP/textual-web process. Capture actual rendered screenshots for:

- idle Console
- typed composer
- llama.cpp streaming
- completed response
- selected message with action row
- regenerated message with `<` / `>` controls
- blocked provider recovery
- WIP/unavailable action state

Do not use SVG/code mockups as approval artifacts.

- [ ] **Step 5: Write QA evidence**

In `Docs/superpowers/qa/product-maturity/console-native-chat-core/2026-05-21-console-native-chat-core.md`, include:

- commands run
- automated test results
- live llama.cpp result
- CDP screenshot filenames/paths
- user approval status for each screenshot
- remaining known limitations

- [ ] **Step 6: Update QA README**

Add an index entry in `Docs/superpowers/qa/product-maturity/console-native-chat-core/README.md`.

- [ ] **Step 7: Update Backlog task**

Mark acceptance criteria complete only after tests and screenshot approvals are complete.

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

Run:

```bash
python -m pytest -q Tests/Chat/test_console_chat_models.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_console_chat_store.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_message_actions.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_internals_decomposition.py --tb=short
git diff --check
```

Expected:

- focused tests pass
- `git diff --check` has no output
- live llama.cpp/CDP evidence is documented if the local server is available
- screenshots are user-approved before visual claims are made

## PR Guidance

Prefer multiple PRs if the implementation becomes large:

1. Core models/store/provider gateway.
2. Controller and Console composer wiring.
3. Native transcript and selected-message actions.
4. Variant navigation, legacy transcript removal, and QA closeout.

Each PR must keep Console usable and include focused tests. Do not batch UI screenshot approval claims across screens.
