# Console Session Settings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compact Console Settings summary and modal that let users view and edit provider/model/sampling settings for the active native Console tab, then apply those settings to the next native Console send.

**Architecture:** Add a pure Console session-settings contract under `tldw_chatbook/Chat`, store a settings snapshot on each native `ConsoleChatSession`, render the left-rail summary and modal as dedicated `Widgets/Console` modules, and keep `ChatScreen` as the coordinator that creates defaults, opens the modal, applies saves, and syncs the controller. Extend the existing native provider gateway only for supported `llama_cpp`/`local_llamacpp` payloads; unsupported providers remain selectable but WIP/blocked at send time.

**Tech Stack:** Python 3.11+, Textual, pytest/pytest-asyncio, httpx `MockTransport`, existing Console native chat modules, existing `Utils/token_counter.py`, existing modular TCSS build script.

---

## Source Documents

- Spec: `Docs/superpowers/specs/2026-05-25-console-session-settings-design.md`
- Prior Console rail plan: `Docs/superpowers/plans/2026-05-24-console-persistent-rails-implementation.md`
- Native Console chat core plan: `Docs/superpowers/plans/2026-05-21-console-native-chat-core-implementation.md`

## Preflight

- The branch is currently `codex/console-screen-next`.
- Before implementation, sync with current `origin/dev` to avoid building on stale Console code:

```bash
git fetch origin
git rebase origin/dev
```

Expected: rebase completes cleanly, or conflicts are resolved without reverting existing user/Codex work.

## File Structure

Create:

- `tldw_chatbook/Chat/console_session_settings.py`
  - Pure dataclasses and functions for settings defaults, validation, provider/model options, readiness labels, summary rows, and context token estimates.
- `tldw_chatbook/Widgets/Console/console_settings_summary.py`
  - Compact left-rail widget with four rows and a Settings button.
- `tldw_chatbook/Widgets/Console/console_settings_modal.py`
  - Modal editor for the current tab settings draft.
- `Tests/Chat/test_console_session_settings.py`
  - Pure tests for defaults, validation, readiness, option helpers, and context summary.
- `Tests/UI/test_console_session_settings.py`
  - Mounted Console tests for summary placement, modal behavior, save/cancel, active-run save lock, and per-tab isolation.

Modify:

- `tldw_chatbook/Chat/console_chat_models.py`
  - Extend `ConsoleProviderSelection` with optional sampling fields and a `streaming` flag.
- `tldw_chatbook/Chat/console_chat_store.py`
  - Store `ConsoleSessionSettings` on each `ConsoleChatSession`, support settings replacement, and create per-session settings snapshots.
- `tldw_chatbook/Chat/console_chat_controller.py`
  - Carry sampling fields into `_provider_selection()` and expose a small sync/update method for `ChatScreen`.
- `tldw_chatbook/Chat/console_provider_gateway.py`
  - Add a shared llama.cpp chat payload builder and use it for streaming and non-streaming calls.
- `tldw_chatbook/UI/Screens/chat_screen.py`
  - Initialize active session settings, mount/sync the summary, open/apply the modal, and build native provider selection from session settings.
- `tldw_chatbook/Widgets/Console/__init__.py`
  - Export the new widgets.
- `tldw_chatbook/css/components/_agentic_terminal.tcss`
  - Add Console Settings summary/modal styling.
- `tldw_chatbook/css/tldw_cli_modular.tcss`
  - Regenerate with `python3 tldw_chatbook/css/build_css.py`.
- Existing tests:
  - `Tests/Chat/test_console_chat_store.py`
  - `Tests/Chat/test_console_chat_controller.py`
  - `Tests/Chat/test_console_provider_gateway.py`
  - `Tests/Chat/test_console_chat_models.py`
  - `Tests/UI/test_console_native_chat_flow.py`
  - `Tests/UI/test_console_workspace_context_rail.py`
  - `Tests/UI/test_console_persistent_rails.py`

## Implementation Tasks

### Task 1: Add Pure Session Settings Contract

**Files:**
- Create: `tldw_chatbook/Chat/console_session_settings.py`
- Test: `Tests/Chat/test_console_session_settings.py`

- [ ] **Step 1: Write failing tests for defaults and option helpers**

Add tests for:

```python
def test_default_settings_prefers_chat_defaults_and_provider_config() -> None:
    config = {
        "chat_defaults": {
            "provider": "llama_cpp",
            "model": "chat-default",
            "temperature": 0.2,
            "top_p": 0.8,
            "max_tokens": 2048,
        },
        "api_settings": {
            "llama_cpp": {
                "api_url": "127.0.0.1:9099/v1",
                "model": "configured-model",
                "top_k": 40,
                "min_p": 0.05,
            },
        },
    }

    settings = build_default_console_session_settings(
        app_config=config,
        provider="llama_cpp",
        model=None,
    )

    assert settings.provider == "llama_cpp"
    assert settings.model == "configured-model"
    assert settings.base_url == "http://127.0.0.1:9099"
    assert settings.temperature == 0.2
    assert settings.top_p == 0.8
    assert settings.min_p == 0.05
    assert settings.top_k == 40
    assert settings.max_tokens == 2048
```

Also add tests that provider/model option helpers:

```python
def test_model_options_include_current_model_missing_from_registry() -> None:
    options = build_console_model_options(
        provider="llama_cpp",
        providers_models={"llama_cpp": ["listed-model"]},
        current_model="configured-model",
    )

    assert [option.value for option in options] == ["configured-model", "listed-model"]


def test_provider_options_include_all_configured_providers() -> None:
    options = build_console_provider_options(
        providers_models={
            "llama_cpp": ["local-model"],
            "openai": ["gpt-4.1"],
            "anthropic": ["claude-sonnet"],
        }
    )

    assert [option.value for option in options] == ["anthropic", "llama_cpp", "openai"]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python3 -m pytest Tests/Chat/test_console_session_settings.py -q
```

Expected: FAIL because `console_session_settings.py` and functions do not exist.

- [ ] **Step 3: Implement dataclasses and defaults**

Create `tldw_chatbook/Chat/console_session_settings.py` with these public contracts:

```python
@dataclass(frozen=True)
class ConsoleSessionSettings:
    provider: str
    model: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    top_p: float = 0.95
    min_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    streaming: bool = True
    persona_label: str = "General"
    character_label: str = ""


@dataclass(frozen=True)
class ConsoleSettingsOption:
    label: str
    value: str


@dataclass(frozen=True)
class ConsoleSettingsReadiness:
    label: str
    detail: str
    native_send_supported: bool


@dataclass(frozen=True)
class ConsoleSettingsContextEstimate:
    used_tokens: int | None
    token_limit: int | None
    label: str
    staged_source_count: int = 0
    staged_context_summary: str = ""
```

Implementation rules:

- Normalize providers with `provider_config_key()` from `tldw_chatbook.Chat.provider_readiness`.
- Normalize llama.cpp base URLs with `normalize_llamacpp_base_url()` from `tldw_chatbook.Chat.console_provider_gateway`.
- Implement `build_console_provider_options()` and `build_console_model_options()` as pure helpers over the configured provider/model registry. They must include all configured providers and never filter to Console-ready providers.
- Use provider+model profile defaults first for scalar sampling values, then `chat_defaults`, then provider-specific `api_settings.<provider>` where `chat_defaults` is absent.
- Prefer the explicit `model` argument, then `api_settings.<provider>.model`, `api_model`, `default_model`, then `chat_defaults.model`.
- Keep functions side-effect-free and network-free.

- [ ] **Step 4: Add validation and readiness tests**

Test these cases:

```python
def test_validation_rejects_out_of_range_temperature() -> None:
    settings = ConsoleSessionSettings(provider="llama_cpp", model="m", temperature=2.1)

    errors = validate_console_session_settings(settings, app_config={})

    assert "Temperature must be between 0 and 2." in errors


def test_readiness_wip_precedes_missing_key_for_openai() -> None:
    settings = ConsoleSessionSettings(provider="openai", model="gpt-4.1")

    readiness = build_console_settings_readiness(settings, app_config={"api_settings": {}})

    assert readiness.label == "WIP"
    assert "not wired" in readiness.detail


def test_invalid_url_precedes_wip_for_url_provider() -> None:
    settings = ConsoleSessionSettings(provider="vllm", model="m", base_url="file:///tmp/x")

    readiness = build_console_settings_readiness(settings, app_config={})

    assert readiness.label == "Invalid URL"


def test_readiness_labels_cover_missing_key_ready_and_unknown() -> None:
    missing = build_console_settings_readiness(
        ConsoleSessionSettings(provider="anthropic", model="claude-sonnet"),
        app_config={"api_settings": {"anthropic": {"api_key_env_var": "MISSING_KEY"}}},
        environ={},
        native_provider_keys={"llama_cpp", "local_llamacpp", "anthropic"},
    )
    ready = build_console_settings_readiness(
        ConsoleSessionSettings(provider="llama_cpp", model="m"),
        app_config={},
    )
    unknown = build_console_settings_readiness(
        ConsoleSessionSettings(provider="made_up_provider", model="m"),
        app_config={},
    )

    assert missing.label == "Missing key"
    assert ready.label == "Ready"
    assert unknown.label == "Unknown"


def test_readiness_unsupported_provider_missing_key_is_still_primary_wip() -> None:
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(provider="anthropic", model="claude-sonnet"),
        app_config={"api_settings": {"anthropic": {"api_key_env_var": "MISSING_KEY"}}},
        environ={},
    )

    assert readiness.label == "WIP"
    assert "missing API key" in readiness.detail
```

Run:

```bash
python3 -m pytest Tests/Chat/test_console_session_settings.py -q
```

Expected: FAIL for validation/readiness functions until implemented.

- [ ] **Step 5: Implement validation/readiness/context helpers**

Implement:

```python
def validate_console_session_settings(
    settings: ConsoleSessionSettings,
    *,
    app_config: Mapping[str, object],
) -> list[str]: ...

def build_console_settings_readiness(
    settings: ConsoleSessionSettings,
    *,
    app_config: Mapping[str, object],
    environ: Mapping[str, str] | None = None,
    native_provider_keys: set[str] | None = None,
) -> ConsoleSettingsReadiness: ...

def build_console_context_estimate(
    *,
    messages: Sequence[Mapping[str, str]],
    provider: str,
    model: str | None,
    staged_source_count: int = 0,
    staged_context_summary: str = "",
    max_tokens_response: int | None = None,
    system_prompt: str | None = None,
) -> ConsoleSettingsContextEstimate: ...
```

Validation details:

- `provider` required.
- `model` required except `llama_cpp`/`local_llamacpp`, where send-time model discovery can still happen.
- `base_url` invalid blocks only URL-based providers with a non-blank value.
- `temperature`: `0.0 <= value <= 2.0`.
- `top_p`: `0.0 <= value <= 1.0`.
- `min_p`: blank/`None` or `0.0 <= value <= 1.0`.
- `top_k`: blank/`None` or integer `>= 0`.
- `max_tokens`: blank/`None` or integer `>= 1`.
- Readiness defaults `native_provider_keys` to `{"llama_cpp", "local_llamacpp"}`. Tests may pass an override to cover future native providers, but production behavior must keep OpenAI/Anthropic/etc. primary `WIP` until native Console sending is wired.
- [ ] **Step 6: Run pure settings tests**

Run:

```bash
python3 -m pytest Tests/Chat/test_console_session_settings.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Chat/console_session_settings.py Tests/Chat/test_console_session_settings.py
git commit -m "feat: add Console session settings contract"
```

### Task 2: Store Settings Per Native Console Session

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Test: `Tests/Chat/test_console_chat_store.py`

- [ ] **Step 1: Write failing store tests**

Add tests:

```python
def test_console_sessions_store_independent_settings_snapshots() -> None:
    store = ConsoleChatStore()
    first_settings = ConsoleSessionSettings(provider="llama_cpp", model="a", temperature=0.1)
    second_settings = ConsoleSessionSettings(provider="openai", model="b", temperature=0.9)

    first = store.create_session(title="A", settings=first_settings)
    second = store.create_session(title="B", settings=second_settings)

    assert store.session_settings(first.id).model == "a"
    assert store.session_settings(second.id).model == "b"


def test_replacing_session_settings_does_not_mutate_other_sessions() -> None:
    store = ConsoleChatStore()
    first = store.create_session(settings=ConsoleSessionSettings(provider="llama_cpp", model="a"))
    second = store.create_session(settings=ConsoleSessionSettings(provider="llama_cpp", model="b"))

    store.replace_session_settings(
        first.id,
        ConsoleSessionSettings(provider="llama_cpp", model="changed"),
    )

    assert store.session_settings(first.id).model == "changed"
    assert store.session_settings(second.id).model == "b"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python3 -m pytest Tests/Chat/test_console_chat_store.py -q
```

Expected: FAIL because `settings` arguments/helpers do not exist.

- [ ] **Step 3: Implement store support**

Modify `ConsoleChatSession`:

```python
@dataclass
class ConsoleChatSession:
    title: str = "Chat 1"
    workspace_id: str = "global"
    id: str = field(default_factory=lambda: str(uuid4()))
    persisted_conversation_id: str | None = None
    settings: ConsoleSessionSettings | None = None
```

Modify `ensure_session()` and `create_session()` to accept `settings: ConsoleSessionSettings | None = None`.

Add:

```python
def session_settings(self, session_id: str) -> ConsoleSessionSettings | None:
    return self._session_or_raise(session_id).settings


def replace_session_settings(
    self,
    session_id: str,
    settings: ConsoleSessionSettings,
) -> ConsoleChatSession:
    session = self._session_or_raise(session_id)
    session.settings = settings
    return replace(session)
```

Do not persist settings to DB in this slice. Closing a session naturally removes settings because it removes the session object.

- [ ] **Step 4: Run store tests**

Run:

```bash
python3 -m pytest Tests/Chat/test_console_chat_store.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_store.py Tests/Chat/test_console_chat_store.py
git commit -m "feat: store Console settings per session"
```

### Task 3: Carry Settings Through Controller Selection

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_models.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Test: `Tests/Chat/test_console_chat_models.py`
- Test: `Tests/Chat/test_console_chat_controller.py`

- [ ] **Step 1: Write failing controller/model tests**

Add a model test:

```python
def test_console_provider_selection_carries_sampling_settings() -> None:
    selection = ConsoleProviderSelection(
        provider="llama_cpp",
        explicit_model="m",
        temperature=0.3,
        top_p=0.8,
        min_p=0.05,
        top_k=40,
        max_tokens=512,
        streaming=False,
    )

    assert selection.temperature == 0.3
    assert selection.streaming is False
```

Add a controller test with a capturing gateway:

```python
@pytest.mark.asyncio
async def test_controller_provider_selection_includes_sampling_settings() -> None:
    gateway = CapturingGateway()
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="m",
        temperature=0.4,
        top_p=0.7,
        min_p=0.03,
        top_k=20,
        max_tokens=300,
        streaming=False,
    )

    await controller.submit_draft("hello")

    assert gateway.selection.temperature == 0.4
    assert gateway.selection.top_p == 0.7
    assert gateway.selection.min_p == 0.03
    assert gateway.selection.top_k == 20
    assert gateway.selection.max_tokens == 300
    assert gateway.selection.streaming is False
```

Reuse or extend existing `StreamingGateway` test doubles in `Tests/Chat/test_console_chat_controller.py`.

- [ ] **Step 2: Run focused tests and verify failures**

Run:

```bash
python3 -m pytest Tests/Chat/test_console_chat_models.py Tests/Chat/test_console_chat_controller.py -q
```

Expected: FAIL for unknown `ConsoleProviderSelection`/controller fields.

- [ ] **Step 3: Extend selection/controller fields**

Extend `ConsoleProviderSelection` with defaults so existing tests keep working:

```python
temperature: float | None = None
top_p: float | None = None
min_p: float | None = None
top_k: int | None = None
max_tokens: int | None = None
streaming: bool = True
```

Extend `ConsoleChatController.__init__()` with matching fields, store them on `self`, and include them in `_provider_selection()`.

Add a controller sync helper:

```python
def update_provider_selection(self, selection: ConsoleProviderSelection) -> None:
    self.provider = selection.provider
    self.model = selection.explicit_model
    self.configured_model = selection.configured_model
    self.base_url = selection.base_url
    self.temperature = selection.temperature
    self.top_p = selection.top_p
    self.min_p = selection.min_p
    self.top_k = selection.top_k
    self.max_tokens = selection.max_tokens
    self.streaming = selection.streaming
```

- [ ] **Step 4: Run controller/model tests**

Run:

```bash
python3 -m pytest Tests/Chat/test_console_chat_models.py Tests/Chat/test_console_chat_controller.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_models.py tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_chat_models.py Tests/Chat/test_console_chat_controller.py
git commit -m "feat: pass Console sampling settings through controller"
```

### Task 4: Wire llama.cpp Payload Settings

**Files:**
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py`
- Test: `Tests/Chat/test_console_provider_gateway.py`

- [ ] **Step 1: Write failing payload-builder tests**

Add tests:

```python
def test_llamacpp_payload_includes_supported_sampling_params() -> None:
    payload = build_llamacpp_chat_payload(
        model="m",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        temperature=0.4,
        top_p=0.7,
        min_p=0.03,
        top_k=20,
        max_tokens=300,
    )

    assert payload == {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "temperature": 0.4,
        "top_p": 0.7,
        "min_p": 0.03,
        "top_k": 20,
        "max_tokens": 300,
    }


def test_llamacpp_payload_omits_blank_provider_defaults() -> None:
    payload = build_llamacpp_chat_payload(
        model="m",
        messages=[],
        stream=False,
        temperature=None,
        top_p=None,
        min_p=None,
        top_k=0,
        max_tokens=None,
    )

    assert payload == {"model": "m", "messages": [], "stream": False}
```

- [ ] **Step 2: Write failing gateway behavior tests**

Add/extend tests:

```python
@pytest.mark.asyncio
async def test_stream_chat_non_streaming_resolution_yields_completion_once() -> None:
    seen_payloads = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "done"}}]})

    gateway = ConsoleProviderGateway(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    resolution = ConsoleProviderResolution(
        provider="llama_cpp",
        base_url="http://127.0.0.1:9099",
        model="m",
        ready=True,
        streaming=False,
        temperature=0.2,
    )

    chunks = [chunk async for chunk in gateway.stream_chat(resolution, [{"role": "user", "content": "hi"}])]

    assert chunks == ["done"]
    assert seen_payloads == [{
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "temperature": 0.2,
    }]
```

- [ ] **Step 3: Run gateway tests and verify failures**

Run:

```bash
python3 -m pytest Tests/Chat/test_console_provider_gateway.py -q
```

Expected: FAIL for missing helper/resolution fields/non-streaming behavior.

- [ ] **Step 4: Implement gateway payload builder and streaming flag**

Modify `ConsoleProviderResolution` to carry:

```python
temperature: float | None = None
top_p: float | None = None
min_p: float | None = None
top_k: int | None = None
max_tokens: int | None = None
streaming: bool = True
```

Add:

```python
def build_llamacpp_chat_payload(
    *,
    model: str,
    messages: list[Mapping[str, Any]],
    stream: bool,
    temperature: float | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "stream": stream,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if min_p is not None:
        payload["min_p"] = min_p
    if top_k is not None and top_k > 0:
        payload["top_k"] = top_k
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return payload
```

Use the helper in both `stream_llamacpp_chat()` and `complete_llamacpp_chat()`. If `resolution.streaming is False`, `stream_chat()` must call `complete_llamacpp_chat()` directly and yield the completion once.

Keep existing streaming fallback behavior when `resolution.streaming is True`: attempt SSE streaming, then fallback to a non-streaming request if no content was emitted.

- [ ] **Step 5: Ensure resolve copies selection settings**

In `resolve_for_send()`, carry sampling fields from `ConsoleProviderSelection` into `LlamaCppProviderConfig` or directly into the returned `ConsoleProviderResolution`.

Unsupported providers should still return `ready=False` and existing WIP copy, but may also carry the selected sampling fields for future use.

- [ ] **Step 6: Run gateway tests**

Run:

```bash
python3 -m pytest Tests/Chat/test_console_provider_gateway.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Chat/console_provider_gateway.py Tests/Chat/test_console_provider_gateway.py
git commit -m "feat: apply Console settings to llama.cpp payloads"
```

### Task 5: Add Console Settings Summary Widget

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_settings_summary.py`
- Modify: `tldw_chatbook/Widgets/Console/__init__.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Test: `Tests/UI/test_console_session_settings.py`

- [ ] **Step 1: Write failing widget tests**

Add a small mounted widget test:

```python
@pytest.mark.asyncio
async def test_console_settings_summary_renders_four_rows_and_button() -> None:
    state = ConsoleSettingsSummaryState(
        model_row="Model: llama.cpp / model-a",
        context_row="Context: 12 / 4k",
        sampling_row="Sampling: T 0.70, P 0.95",
        identity_row="Persona: General",
        readiness_label="Ready",
    )

    app = SummaryHarness(state)
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()

        text = _visible_text(app)
        assert "Console Settings" in text
        assert "Model: llama.cpp / model-a" in text
        assert "Context: 12 / 4k" in text
        assert "Sampling: T 0.70, P 0.95" in text
        assert "Persona: General" in text
        assert app.query_one("#console-settings-open", Button).tooltip == "Open Console settings"
```

Use a local `SummaryHarness(App)` inside the test file.

- [ ] **Step 2: Run widget test and verify it fails**

Run:

```bash
python3 -m pytest Tests/UI/test_console_session_settings.py::test_console_settings_summary_renders_four_rows_and_button -q
```

Expected: FAIL because widget/state are missing.

- [ ] **Step 3: Implement summary state and widget**

Add `ConsoleSettingsSummaryState` to `console_session_settings.py`:

```python
@dataclass(frozen=True)
class ConsoleSettingsSummaryState:
    model_row: str
    context_row: str
    sampling_row: str
    identity_row: str
    readiness_label: str = ""
```

Add `build_console_settings_summary_state(...)` pure helper that:

- Builds model row as `Model: <provider> / <model>` with readiness suffix such as `(WIP)`.
- Builds context row from `ConsoleSettingsContextEstimate.label`.
- Builds sampling row as `Sampling: T 0.70, P 0.95` and appends `min_p`, `top_k`, and `max_tokens` only when set.
- Builds identity row as `Persona: General`, `Character: <name>`, or `Persona: <persona>`.

Create `ConsoleSettingsSummary(Vertical)` with:

- Static title `#console-settings-title`.
- Four `Static` rows with ids:
  - `#console-settings-model-row`
  - `#console-settings-context-row`
  - `#console-settings-sampling-row`
  - `#console-settings-identity-row`
- Button `#console-settings-open`.
- `sync_state(self, state: ConsoleSettingsSummaryState) -> None`.

Export `ConsoleSettingsSummary` from `tldw_chatbook/Widgets/Console/__init__.py`.

- [ ] **Step 4: Add summary styling**

Add TCSS under `tldw_chatbook/css/components/_agentic_terminal.tcss`:

```css
.console-settings-summary {
    height: auto;
    min-height: 7;
    padding: 0 1;
}

.console-settings-row {
    height: 1;
    min-height: 1;
    text-overflow: ellipsis;
}

#console-settings-open {
    width: 100%;
    height: 1;
    margin: 1 0 0 0;
}
```

Adjust exact properties if Textual rejects unsupported declarations. Do not use nested cards or large vertical margins.

- [ ] **Step 5: Run widget test**

Run:

```bash
python3 -m pytest Tests/UI/test_console_session_settings.py::test_console_settings_summary_renders_four_rows_and_button -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Chat/console_session_settings.py tldw_chatbook/Widgets/Console/console_settings_summary.py tldw_chatbook/Widgets/Console/__init__.py tldw_chatbook/css/components/_agentic_terminal.tcss Tests/UI/test_console_session_settings.py
git commit -m "feat: add Console settings summary widget"
```

### Task 6: Add Console Settings Modal

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_settings_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/__init__.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Test: `Tests/UI/test_console_session_settings.py`

- [ ] **Step 1: Write failing modal tests**

Add tests for draft/cancel/save behavior with a small modal harness:

```python
@pytest.mark.asyncio
async def test_console_settings_modal_cancel_discards_draft() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a", "model-b"]},
                context_estimate=ConsoleSettingsContextEstimate(used_tokens=10, token_limit=4096, label="10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()
        await pilot.click("#console-settings-cancel")

    assert app.saved_settings is None
```

Add save/validation tests:

```python
@pytest.mark.asyncio
async def test_console_settings_modal_save_returns_validated_settings() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleSettingsModal(
            settings=settings,
            app_config=app.app_config,
            providers_models={"llama_cpp": ["model-a", "model-b"]},
            context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
            can_save=True,
        )
        await app.push_screen(modal, callback=app.capture_saved_settings)
        await pilot.pause()
        app.query_one("#console-settings-temperature", Input).value = "0.42"
        app.query_one("#console-settings-top-p", Input).value = "0.88"
        await pilot.click("#console-settings-save")

    assert app.saved_settings is not None
    assert app.saved_settings.provider == "llama_cpp"
    assert app.saved_settings.model == "model-a"
    assert app.saved_settings.temperature == 0.42
    assert app.saved_settings.top_p == 0.88

@pytest.mark.asyncio
async def test_console_settings_modal_invalid_temperature_stays_open() -> None:
    app = ModalHarness()

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        app.query_one("#console-settings-temperature", Input).value = "2.5"
        await pilot.click("#console-settings-save")

        assert app.saved_settings is None
        assert len(app.query(ConsoleSettingsModal)) == 1
        assert "Temperature must be between 0 and 2." in _visible_text(app)

@pytest.mark.asyncio
async def test_console_settings_modal_disables_save_when_run_active() -> None:
    app = ModalHarness()

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=False,
            )
        )
        await pilot.pause()

        assert app.query_one("#console-settings-save", Button).disabled is True

@pytest.mark.asyncio
async def test_console_settings_modal_renders_context_sources_note_and_identity_rows() -> None:
    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        persona_label="Analyst",
        character_label="Vox",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(
                    used_tokens=10,
                    token_limit=4096,
                    label="10 / 4k",
                    staged_source_count=2,
                    staged_context_summary="2 staged",
                ),
                can_save=True,
            )
        )
        await pilot.pause()

        text = _visible_text(app)
        assert "Current         10 / 4k tokens" in text
        assert "Sources         2 staged" in text
        assert "Estimate only; no truncation changes in this version." in text
        assert "Current         Analyst / Vox" in text
        assert "Persona         Analyst [read-only]" in text
        assert "Character       Vox [read-only]" in text


@pytest.mark.asyncio
async def test_console_settings_modal_provider_select_lists_all_configured_providers() -> None:
    app = ModalHarness()

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
                app_config=app.app_config,
                providers_models={
                    "llama_cpp": ["model-a"],
                    "openai": ["gpt-4.1"],
                    "anthropic": ["claude-sonnet"],
                },
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()

        provider_select = app.query_one("#console-settings-provider", Select)
        option_values = {str(option.value) for option in provider_select._options}
        assert option_values >= {"llama_cpp", "openai", "anthropic"}
```

- [ ] **Step 2: Run modal tests and verify failures**

Run:

```bash
python3 -m pytest Tests/UI/test_console_session_settings.py -q
```

Expected: FAIL for missing modal.

- [ ] **Step 3: Implement modal structure**

Implement `ConsoleSettingsModal(ModalScreen[ConsoleSessionSettings | None])`.

Constructor inputs:

```python
def __init__(
    self,
    *,
    settings: ConsoleSessionSettings,
    app_config: Mapping[str, object],
    providers_models: Mapping[str, list[str]],
    context_estimate: ConsoleSettingsContextEstimate,
    can_save: bool,
) -> None: ...
```

Compose controls with stable ids:

- `#console-settings-modal`
- `#console-settings-provider`
- `#console-settings-readiness`
- `#console-settings-model-select`
- `#console-settings-model-input`
- `#console-settings-base-url`
- `#console-settings-temperature`
- `#console-settings-top-p`
- `#console-settings-min-p`
- `#console-settings-top-k`
- `#console-settings-max-tokens`
- `#console-settings-streaming`
- `#console-settings-context-current`
- `#console-settings-context-sources`
- `#console-settings-context-note`
- `#console-settings-identity-current`
- `#console-settings-persona-readonly`
- `#console-settings-character-readonly`
- `#console-settings-error`
- `#console-settings-cancel`
- `#console-settings-save`

Rules:

- Use `Select` when model options exist.
- Use `Input` when no model options exist.
- Do not perform live model discovery.
- `Save` disabled if `can_save` is false.
- `Escape` dismisses with `None`.
- `Cancel` dismisses with `None`.
- Context section must render `Current`, `Sources`, and `Note` as read-only rows. `Sources` comes from the active staged source count/summary passed through the context estimate object.
- Identity section must render `Current`, `Persona`, and `Character` as read-only rows. Persona and character fields are display-only in this slice and must not have editable controls.

- [ ] **Step 4: Implement modal draft parsing and validation**

On Save:

1. Build a `ConsoleSessionSettings` draft from current control values.
2. Run `validate_console_session_settings(draft, app_config=...)`.
3. If errors exist, update `#console-settings-error`, keep modal open.
4. If no errors, `dismiss(draft)`.

On provider change:

1. Rebuild readiness with `build_console_settings_readiness()`.
2. Update model select/input visibility.
3. Update base URL value/visibility according to URL-provider rules.
4. Do not mutate the original `settings`.

- [ ] **Step 5: Add modal styling**

Add TCSS:

```css
ConsoleSettingsModal {
    align: center middle;
}

#console-settings-modal {
    width: 92;
    max-width: 95%;
    height: auto;
    max-height: 95%;
    border: tall gray;
    background: black;
    padding: 1 2;
}

.console-settings-modal-section {
    height: auto;
    margin: 1 0 0 0;
}

.console-settings-modal-row {
    height: auto;
    min-height: 1;
}

.console-settings-error {
    height: auto;
    color: red;
}
```

Adjust to valid Textual CSS if needed.

- [ ] **Step 6: Run modal tests**

Run:

```bash
python3 -m pytest Tests/UI/test_console_session_settings.py -q
```

Expected: PASS for widget/modal tests.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_settings_modal.py tldw_chatbook/Widgets/Console/__init__.py tldw_chatbook/css/components/_agentic_terminal.tcss Tests/UI/test_console_session_settings.py
git commit -m "feat: add Console settings modal"
```

### Task 7: Wire Summary And Modal Into ChatScreen

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/UI/test_console_session_settings.py`
- Modify: `Tests/UI/test_console_native_chat_flow.py`
- Modify: `Tests/UI/test_console_workspace_context_rail.py`
- Modify: `Tests/UI/test_console_persistent_rails.py`

- [ ] **Step 1: Write failing mounted Console tests**

Add tests:

```python
@pytest.mark.asyncio
async def test_console_left_rail_renders_settings_below_staged_context() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        staged_context = console.query_one("#console-staged-context-tray")
        settings = console.query_one("#console-settings-summary")
        workspace_context = console.query_one("#console-workspace-context")

        assert staged_context.region.y < settings.region.y < workspace_context.region.y
        assert settings.region.width == staged_context.region.width
```

Add modal integration tests:

```python
@pytest.mark.asyncio
async def test_console_settings_save_updates_active_summary_only() -> None:
    app = _build_test_app()
    app.providers_models = {"llama_cpp": ["model-a", "model-b"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        console._replace_active_console_session_settings(
            ConsoleSessionSettings(provider="llama_cpp", model="model-a")
        )
        console._sync_console_settings_summary()

        await pilot.click("#console-settings-open")
        await _wait_for_selector(console, pilot, "#console-settings-modal")
        model_select = console.app.query_one("#console-settings-model-select", Select)
        model_select.value = "model-b"
        await pilot.click("#console-settings-save")
        await pilot.pause()

        assert "model-b" in _visible_text(console)

@pytest.mark.asyncio
async def test_console_settings_are_isolated_between_native_tabs() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        controller = console._ensure_console_chat_controller()
        first_id = controller.store.active_session_id
        console._replace_active_console_session_settings(
            ConsoleSessionSettings(provider="llama_cpp", model="first-model")
        )

        await pilot.click("#console-new-chat-tab")
        await pilot.pause()
        second_id = controller.store.active_session_id
        assert second_id != first_id
        console._replace_active_console_session_settings(
            ConsoleSessionSettings(provider="llama_cpp", model="second-model")
        )

        controller.switch_session(first_id)
        await console._sync_native_console_chat_ui()
        assert console._ensure_active_console_session_settings().model == "first-model"
        assert "first-model" in _visible_text(console)

        controller.switch_session(second_id)
        await console._sync_native_console_chat_ui()
        assert console._ensure_active_console_session_settings().model == "second-model"
        assert "second-model" in _visible_text(console)

@pytest.mark.asyncio
async def test_console_settings_cancel_keeps_original_summary() -> None:
    app = _build_test_app()
    app.providers_models = {"llama_cpp": ["model-a", "model-b"]}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        console._replace_active_console_session_settings(
            ConsoleSessionSettings(provider="llama_cpp", model="model-a")
        )
        console._sync_console_settings_summary()

        await pilot.click("#console-settings-open")
        await _wait_for_selector(console, pilot, "#console-settings-modal")
        model_select = console.app.query_one("#console-settings-model-select", Select)
        model_select.value = "model-b"
        await pilot.click("#console-settings-cancel")
        await pilot.pause()

        assert "model-a" in _visible_text(console)
        assert "model-b" not in _visible_text(console)

@pytest.mark.asyncio
async def test_console_settings_save_disabled_during_active_run() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        controller = console._ensure_console_chat_controller()
        controller.run_state = ConsoleRunState(ConsoleRunStatus.STREAMING, "Streaming response.")

        await pilot.click("#console-settings-open")
        await _wait_for_selector(console, pilot, "#console-settings-modal")

        assert console.app.query_one("#console-settings-save", Button).disabled is True

@pytest.mark.asyncio
async def test_console_send_blocker_uses_saved_session_provider() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-settings-summary")

        console._replace_active_console_session_settings(
            ConsoleSessionSettings(provider="openai", model="gpt-4.1")
        )

        assert "not wired" in console._console_send_blocked_reason()
```

For active-run lock, use an existing waiting gateway pattern from `Tests/UI/test_console_native_chat_flow.py`, start a send, open settings, assert `#console-settings-save.disabled is True`, stop/finish run, reopen or sync and assert save can be enabled.

- [ ] **Step 2: Run mounted tests and verify failures**

Run:

```bash
python3 -m pytest Tests/UI/test_console_session_settings.py Tests/UI/test_console_native_chat_flow.py::test_console_provider_selection_uses_configured_default_provider -q
```

Expected: FAIL for missing mounted summary/modal integration.

- [ ] **Step 3: Add ChatScreen settings helpers**

In `ChatScreen`, add helpers:

```python
def _providers_models(self) -> dict[str, list[str]]:
    return dict(getattr(self.app_instance, "providers_models", None) or get_cli_providers_and_models())


def _default_console_session_settings(self) -> ConsoleSessionSettings:
    provider, model = self._effective_console_provider_model()
    return build_default_console_session_settings(
        app_config=getattr(self.app_instance, "app_config", {}) or {},
        provider=str(provider or ""),
        model=str(model).strip() if _has_selected_text(model) else None,
    )


def _ensure_active_console_session_settings(self) -> ConsoleSessionSettings:
    store = self._ensure_console_chat_store()
    session = store.ensure_session(
        workspace_id=store.workspace_context.active_workspace_id,
        settings=self._default_console_session_settings(),
    )
    if session.settings is None:
        store.replace_session_settings(session.id, self._default_console_session_settings())
    return store.session_settings(session.id) or self._default_console_session_settings()
```

Add:

```python
def _replace_active_console_session_settings(self, settings: ConsoleSessionSettings) -> None: ...
def _active_console_settings_context_estimate(self) -> ConsoleSettingsContextEstimate: ...
def _build_console_settings_summary_state(self) -> ConsoleSettingsSummaryState: ...
def _sync_console_settings_summary(self) -> None: ...
```

Use active native transcript messages from `_native_console_messages()` and `count_tokens_chat_history()` through the pure helper.
Pass staged source count and staged-context summary from `_current_console_workspace_context()` / `_build_console_staged_context_state()` into `_active_console_settings_context_estimate()` so the modal can render `Sources` without querying the staged-context widget.

- [ ] **Step 4: Build provider selection from session settings**

Update `_build_console_provider_selection()`:

- Read active `ConsoleSessionSettings`.
- Normalize provider through `provider_config_key()`.
- Use `settings.model` as `explicit_model`.
- Still compute `configured_model` from `api_settings.<provider>` as fallback.
- Use `settings.base_url` for URL providers, with llama.cpp default when blank.
- Include `temperature`, `top_p`, `min_p`, `top_k`, `max_tokens`, and `streaming`.
- Include current workspace context.

Do not query hidden legacy sidebar widgets for Console settings.

Update `_console_send_blocked_reason()` in the same pass:

- Read provider/model/readiness from `_ensure_active_console_session_settings()` rather than `_effective_console_provider_model()`.
- Preserve the RAG/evidence blocking behavior.
- For unsupported native providers, rely on `build_console_settings_readiness()` and the existing WIP semantics so saved modal settings are reflected before the controller path.
- Keep the copy honest: if the saved provider is WIP, show WIP/native-not-wired copy; if the saved provider is missing a model, show `Console send blocked: Select a model before sending.`

- [ ] **Step 5: Sync controller through helper**

Update `_ensure_console_chat_controller()` and `_sync_console_chat_core_state()` to call:

```python
controller.update_provider_selection(selection)
```

Keep scalar assignments only as fallback if the helper is unavailable in tests.

- [ ] **Step 6: Mount summary in the left rail**

In `compose_content()`, mount after `ConsoleStagedContextTray` and before `ConsoleWorkspaceContextTray`:

```python
settings_summary = ConsoleSettingsSummary(
    self._build_console_settings_summary_state(),
    id="console-settings-summary",
    classes="console-left-rail-section console-settings-summary",
)
yield self._frame_console_region(settings_summary, variant="quiet")
```

Adjust heights so the left rail remains compact:

- `ConsoleStagedContextTray`: fixed/auto height for empty state, not `1fr`.
- `ConsoleSettingsSummary`: fixed/auto height.
- `ConsoleWorkspaceContextTray`: remaining height (`1fr` or larger).

Update existing workspace/rail geometry tests only where expectations changed because the new section sits between staged context and workspace context.

- [ ] **Step 7: Open/apply modal**

Add handler:

```python
@on(Button.Pressed, "#console-settings-open")
async def handle_console_settings_open(self, event: Button.Pressed) -> None:
    event.stop()
    controller = self._ensure_console_chat_controller()
    result = await self.app_instance.push_screen_wait(
        ConsoleSettingsModal(
            settings=self._ensure_active_console_session_settings(),
            app_config=getattr(self.app_instance, "app_config", {}) or {},
            providers_models=self._providers_models(),
            context_estimate=self._active_console_settings_context_estimate(),
            can_save=controller.run_state.is_send_allowed,
        )
    )
    if result is None:
        return
    self._replace_active_console_session_settings(result)
    await self._sync_native_console_chat_ui()
```

If `push_screen_wait` is not available in this app version, use Textual's callback form:

```python
self.app_instance.push_screen(modal, callback=self._apply_console_settings_modal_result)
```

and keep `_apply_console_settings_modal_result()` as the save path.

- [ ] **Step 8: Update session creation/switching**

Update native new-tab creation so each new session gets a snapshot:

```python
controller.new_session(settings=self._default_console_session_settings())
```

If `ConsoleChatController.new_session()` is extended, pass settings through to `store.create_session()`.

On session switch, call `_sync_console_settings_summary()` and `_sync_console_chat_core_state()`.

- [ ] **Step 9: Run focused mounted tests**

Run:

```bash
python3 -m pytest Tests/UI/test_console_session_settings.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_console_persistent_rails.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_console_persistent_rails.py
git commit -m "feat: wire Console settings into active sessions"
```

### Task 8: CSS Build And Visual QA

**Files:**
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Optional evidence: `Docs/superpowers/qa/console/2026-05-25-console-session-settings-qa.md` if screenshots/notes are captured.

- [ ] **Step 1: Regenerate modular CSS**

Run:

```bash
python3 tldw_chatbook/css/build_css.py
```

Expected: `tldw_chatbook/css/tldw_cli_modular.tcss` is regenerated and includes the new Console settings styles from `_agentic_terminal.tcss`.

- [ ] **Step 2: Run CSS/style smoke tests**

Run:

```bash
python3 -m pytest Tests/UI/test_console_session_settings.py Tests/UI/test_console_workspace_context_rail.py -q
```

Expected: PASS.

- [ ] **Step 3: Mounted screenshot QA**

Use the existing Textual harness style from current Console screenshot work. Exercise:

- Default size: `160x44`
- Compact size: `100x32`
- Long provider/model labels: provider `llama_cpp`, model `a-very-long-local-model-name-that-should-not-wrap-the-left-rail-into-the-transcript`
- Left rail open, right rail closed
- Left rail collapsed
- Modal open at default size
- Modal open at compact size

Expected:

- Summary is below staged context.
- Summary rows do not wrap into unreadable blocks.
- Settings button is visible and focusable.
- Transcript and composer are not cramped by the new section.
- Collapsed left rail does not leak settings detail.
- Modal fits within the viewport and shows validation text without color-only meaning.

- [ ] **Step 4: Record QA notes if screenshots expose any layout fixes**

If visual QA finds issues, fix them in `_agentic_terminal.tcss` and/or widget row truncation first, rerun:

```bash
python3 tldw_chatbook/css/build_css.py
python3 -m pytest Tests/UI/test_console_session_settings.py Tests/UI/test_console_workspace_context_rail.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Docs/superpowers/qa/console/2026-05-25-console-session-settings-qa.md
git commit -m "style: polish Console settings layout"
```

If no QA evidence file is created, omit it from `git add`.

### Task 9: End-To-End Verification And Cleanup

**Files:**
- Potentially all files touched above.

- [ ] **Step 1: Run the focused Console suite**

Run:

```bash
python3 -m pytest Tests/Chat/test_console_session_settings.py Tests/Chat/test_console_chat_models.py Tests/Chat/test_console_chat_store.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_provider_gateway.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_console_persistent_rails.py -q
```

Expected: PASS.

- [ ] **Step 2: Run broader UI smoke coverage**

Run:

```bash
python3 -m pytest Tests/UI/test_destination_shells.py Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py -q
```

Expected: PASS. If unrelated pre-existing failures occur, document exact failing tests and reason before continuing.

- [ ] **Step 3: Manual UAT path**

Run the app:

```bash
python3 -m tldw_chatbook.app
```

Exercise:

- Open Console.
- Confirm only the left rail is open on first start.
- Confirm `Console Settings` is visible under Context.
- Open Settings.
- Switch provider/model, change temperature/top-p/min-p/top-k/max tokens.
- Save.
- Confirm summary updates.
- Send a llama.cpp chat when a local server is available, or verify blocked recovery is honest when unavailable.
- Open a second Console tab and confirm it has its own settings snapshot.
- Return to first tab and confirm original settings are still there.
- Select unsupported provider and confirm WIP state is visible and send is blocked honestly.

- [ ] **Step 4: Inspect diff for scope creep**

Run:

```bash
git diff --stat origin/dev...HEAD
git diff origin/dev...HEAD -- tldw_chatbook/Chat tldw_chatbook/Widgets/Console tldw_chatbook/UI/Screens/chat_screen.py Tests/Chat Tests/UI
```

Expected: Changes are limited to Console session settings, gateway payloads, CSS, docs/tests. No global settings writes or persona editing are included.

- [ ] **Step 5: Final commit if verification caused fixes**

If verification required additional fixes:

```bash
git add <fixed-files>
git commit -m "fix: stabilize Console session settings"
```

Expected: clean final working tree except intentionally untracked screenshots/evidence files.

## Acceptance Checklist

- [ ] Left rail renders a compact `Console Settings` summary below Context.
- [ ] Settings modal opens from the summary button.
- [ ] Modal edits a draft and applies only on Save.
- [ ] Cancel discards draft changes.
- [ ] Save is disabled while a Console run is active.
- [ ] Settings are scoped to the active native Console tab/session.
- [ ] Switching tabs restores each tab's settings snapshot.
- [ ] `llama_cpp`/`local_llamacpp` payloads include supported sampling params.
- [ ] Non-streaming mode still yields one completion through the async iterator.
- [ ] Unsupported providers remain selectable but WIP/blocked for native sends.
- [ ] Context row is read-only and estimate-only.
- [ ] Persona/character rows are read-only in this slice.
- [ ] No normal Save writes global or workspace defaults.
- [ ] Focused tests and mounted UI tests pass.
- [ ] Default and compact visual QA do not show cramped transcript/composer layout.
