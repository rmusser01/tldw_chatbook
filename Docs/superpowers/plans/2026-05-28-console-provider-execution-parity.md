# Console Provider Execution Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make native Console send work through every provider already dispatched by `chat_api_call()`, while preserving the direct llama.cpp path and clear blocked-send recovery.

**Architecture:** Add one Console provider-support helper that resolves display/config, readiness, and execution identities. Feed that helper into Console Settings readiness and `ConsoleProviderGateway`, then add a generic `chat_api_call()` adapter path that bridges sync provider output into the async Console transcript without blocking Textual.

**Tech Stack:** Python 3.11+, Textual, pytest, pytest-asyncio, httpx `MockTransport`, existing `chat_api_call()` dispatcher.

---

## Source Spec

- `Docs/superpowers/specs/2026-05-28-console-provider-execution-parity-design.md`

## File Structure

- Create `tldw_chatbook/Chat/console_provider_support.py`
  - Owns Console provider identity resolution.
  - Imports `API_CALL_HANDLERS` lazily to avoid making settings helpers depend on gateway internals.
  - Provides direct-path vs generic-adapter classification.
- Modify `tldw_chatbook/Chat/console_session_settings.py`
  - Replaces the llama-only `NATIVE_CONSOLE_PROVIDER_KEYS` readiness gate with provider-support helper results.
  - Keeps pure, side-effect-free settings/readiness behavior.
- Modify `tldw_chatbook/Chat/provider_readiness.py`
  - Makes `[api_settings.*]` lookup use normalized provider keys instead of direct dictionary lookup only.
- Modify `tldw_chatbook/Chat/console_provider_gateway.py`
  - Accepts `config_provider`, `environ`, `chat_api_call_fn`, and safe-error dependencies.
  - Resolves generic providers through `get_provider_readiness()`.
  - Streams generic `chat_api_call()` results through a worker-thread-to-async-queue bridge.
- Modify `tldw_chatbook/UI/Screens/chat_screen.py`
  - Constructs the production gateway with a current config-provider callable so Settings changes do not become stale.
- Test `Tests/Chat/test_console_provider_support.py`
  - Provider alias, handler-key, and direct/generic classification contracts.
- Modify `Tests/Chat/test_console_session_settings.py`
  - Readiness no longer WIP-blocks supported generic providers.
  - URL override behavior is explicit and stable.
- Modify `Tests/Chat/test_console_provider_gateway.py`
  - Generic provider readiness, send kwargs, response normalization, redaction, streaming, cancellation, and llama preservation.
- Modify `Tests/UI/test_console_native_chat_flow.py`
  - Mounted Console generic-send and blocked-send composer preservation coverage.

## Task 1: Add Console Provider Identity Helper

**Files:**
- Create: `tldw_chatbook/Chat/console_provider_support.py`
- Test: `Tests/Chat/test_console_provider_support.py`

- [ ] **Step 1: Write failing alias and handler-key tests**

Add `Tests/Chat/test_console_provider_support.py`:

```python
from tldw_chatbook.Chat.console_provider_support import (
    DIRECT_CONSOLE_PROVIDER_KEYS,
    resolve_console_provider_identity,
    supported_console_provider_readiness_keys,
)


def test_aliases_resolve_to_readiness_and_execution_keys() -> None:
    cases = {
        "Custom": ("custom", "custom-openai-api"),
        "custom-openai-api": ("custom", "custom-openai-api"),
        "Custom-2": ("custom_2", "custom-openai-api-2"),
        "local_llm": ("local_llm", "local-llm"),
        "local-llm": ("local_llm", "local-llm"),
        "mlx_lm": ("local_mlx_lm", "local_mlx_lm"),
        "local_mlx_lm": ("local_mlx_lm", "local_mlx_lm"),
        "MistralAI": ("mistralai", "mistralai"),
    }
    for raw, expected in cases.items():
        identity = resolve_console_provider_identity(raw)
        assert (identity.readiness_key, identity.execution_key) == expected
        assert identity.is_supported is True


def test_direct_console_provider_keys_are_not_generic_adapter() -> None:
    for provider in DIRECT_CONSOLE_PROVIDER_KEYS:
        identity = resolve_console_provider_identity(provider)
        assert identity.uses_direct_llama_path is True
        assert identity.execution_key == provider


def test_all_chat_api_call_handlers_are_known_to_console_support() -> None:
    keys = supported_console_provider_readiness_keys()
    assert "openai" in keys
    assert "anthropic" in keys
    assert "local_vllm" in keys
    assert "custom" in keys
    assert "custom_2" in keys
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```bash
python -m pytest -q Tests/Chat/test_console_provider_support.py --tb=short
```

Expected: FAIL because `console_provider_support.py` does not exist.

- [ ] **Step 3: Implement the helper**

Create `tldw_chatbook/Chat/console_provider_support.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Collection

from tldw_chatbook.Chat.provider_readiness import provider_config_key

DIRECT_CONSOLE_PROVIDER_KEYS = frozenset({"llama_cpp", "local_llamacpp"})

READINESS_TO_EXECUTION_ALIASES = {
    "custom": "custom-openai-api",
    "custom_2": "custom-openai-api-2",
    "local_llm": "local-llm",
    "local_mlx_lm": "local_mlx_lm",
    "mistralai": "mistralai",
}

EXECUTION_TO_READINESS_ALIASES = {
    "custom-openai-api": "custom",
    "custom-openai-api-2": "custom_2",
    "local-llm": "local_llm",
    "mlx_lm": "local_mlx_lm",
}


@dataclass(frozen=True)
class ConsoleProviderIdentity:
    display_key: str
    readiness_key: str
    execution_key: str
    is_supported: bool
    uses_direct_llama_path: bool = False


def _handler_keys(handler_keys: Collection[str] | None = None) -> frozenset[str]:
    if handler_keys is not None:
        return frozenset(handler_keys)
    from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS

    return frozenset(API_CALL_HANDLERS)


def resolve_console_provider_identity(
    provider: str | None,
    *,
    handler_keys: Collection[str] | None = None,
) -> ConsoleProviderIdentity:
    display_key = provider_config_key(provider)
    exact = (provider or "").strip().lower()
    handlers = _handler_keys(handler_keys)

    if exact in DIRECT_CONSOLE_PROVIDER_KEYS or display_key in DIRECT_CONSOLE_PROVIDER_KEYS:
        key = exact if exact in DIRECT_CONSOLE_PROVIDER_KEYS else display_key
        return ConsoleProviderIdentity(key, key, key, True, True)

    readiness_key = EXECUTION_TO_READINESS_ALIASES.get(exact, display_key)
    execution_key = READINESS_TO_EXECUTION_ALIASES.get(readiness_key, exact if exact in handlers else readiness_key)
    is_supported = execution_key in handlers
    return ConsoleProviderIdentity(
        display_key=display_key,
        readiness_key=readiness_key,
        execution_key=execution_key,
        is_supported=is_supported,
        uses_direct_llama_path=False,
    )


def supported_console_provider_readiness_keys(handler_keys: Collection[str] | None = None) -> frozenset[str]:
    handlers = _handler_keys(handler_keys)
    return frozenset(
        resolve_console_provider_identity(key, handler_keys=handlers).readiness_key
        for key in handlers
    )
```

- [ ] **Step 4: Run helper tests**

Run:

```bash
python -m pytest -q Tests/Chat/test_console_provider_support.py --tb=short
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_provider_support.py Tests/Chat/test_console_provider_support.py
git commit -m "Add Console provider identity helper"
```

## Task 2: Replace Llama-Only Console Settings Readiness Gate

**Files:**
- Modify: `tldw_chatbook/Chat/console_session_settings.py`
- Modify: `tldw_chatbook/Chat/provider_readiness.py`
- Modify: `Tests/Chat/test_provider_readiness.py`
- Modify: `Tests/Chat/test_console_session_settings.py`

- [ ] **Step 1: Write failing readiness tests**

Update WIP-oriented tests in `Tests/Chat/test_console_session_settings.py`:

```python
def test_readiness_reports_missing_key_for_supported_openai_instead_of_wip() -> None:
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(provider="openai", model="gpt-4.1"),
        app_config={"api_settings": {"openai": {"api_key_env_var": "OPENAI_API_KEY"}}},
        environ={},
    )

    assert readiness.label == "Missing key"
    assert "OPENAI_API_KEY" in readiness.detail
    assert "not wired" not in readiness.detail


def test_readiness_reports_ready_for_keyless_supported_generic_provider() -> None:
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(provider="ollama", model="llama3", base_url="http://127.0.0.1:11434"),
        app_config={"api_settings": {"ollama": {"api_url": "http://127.0.0.1:11434"}}},
        environ={},
    )

    assert readiness.label == "Ready"
    assert readiness.native_send_supported is True
```

Add or update URL tests:

```python
def test_generic_provider_matching_configured_url_is_not_false_blocked() -> None:
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(provider="ollama", model="llama3", base_url="http://127.0.0.1:11434/"),
        app_config={"api_settings": {"ollama": {"api_url": "http://127.0.0.1:11434"}}},
        environ={},
    )

    assert readiness.label == "Ready"
```

Add normalized provider-settings lookup coverage in `Tests/Chat/test_provider_readiness.py`:

```python
def test_provider_readiness_matches_config_keys_by_normalized_provider_name():
    readiness = get_provider_readiness(
        "Custom-2",
        {"api_settings": {"Custom-2": {"api_key": "local-secret"}}},
        environ={},
    )

    assert readiness.ready is True
    assert readiness.api_key == "local-secret"
    assert readiness.api_key_source == "config:api_settings.custom_2.api_key"
```

- [ ] **Step 2: Run focused readiness tests to verify failure**

Run:

```bash
python -m pytest -q Tests/Chat/test_provider_readiness.py::test_provider_readiness_matches_config_keys_by_normalized_provider_name Tests/Chat/test_console_session_settings.py::test_readiness_reports_missing_key_for_supported_openai_instead_of_wip Tests/Chat/test_console_session_settings.py::test_readiness_reports_ready_for_keyless_supported_generic_provider --tb=short
```

Expected: FAIL with current WIP behavior.

- [ ] **Step 3: Normalize provider settings lookup in provider readiness**

Modify `tldw_chatbook/Chat/provider_readiness.py` so `get_provider_readiness()` loops through configured provider keys:

```python
def _provider_settings_for_key(app_config: Mapping[str, object], provider_key: str) -> Mapping[str, object]:
    api_settings = app_config.get("api_settings", {})
    if not isinstance(api_settings, Mapping):
        return {}
    for configured_provider, configured_settings in api_settings.items():
        if provider_config_key(str(configured_provider)) == provider_key and isinstance(configured_settings, Mapping):
            return configured_settings
    return {}
```

Then replace direct `api_settings.get(provider_key, {})` lookup with `_provider_settings_for_key(app_config, provider_key)`.

- [ ] **Step 4: Update Console settings readiness helper usage**

Modify `tldw_chatbook/Chat/console_session_settings.py`:

```python
from tldw_chatbook.Chat.console_provider_support import (
    DIRECT_CONSOLE_PROVIDER_KEYS,
    resolve_console_provider_identity,
    supported_console_provider_readiness_keys,
)
```

Keep `NATIVE_CONSOLE_PROVIDER_KEYS` as a compatibility alias only if existing tests import it:

```python
NATIVE_CONSOLE_PROVIDER_KEYS = DIRECT_CONSOLE_PROVIDER_KEYS
```

In `build_console_settings_readiness()`:

```python
identity = resolve_console_provider_identity(settings.provider)
native_keys = (
    {provider_config_key(provider) for provider in native_provider_keys}
    if native_provider_keys is not None
    else supported_console_provider_readiness_keys()
)
provider_key = identity.readiness_key
```

Replace the WIP branch with:

```python
if not identity.is_supported:
    return ConsoleSettingsReadiness(
        label="Unknown",
        detail=f"Provider blocked: `{settings.provider}` is not available in Console yet.",
        native_send_supported=False,
    )
```

Set ready support from the shared helper:

```python
native_send_supported = provider_key in native_keys and readiness.ready and identity.is_supported
```

- [ ] **Step 5: Preserve missing model and invalid URL behavior**

Ensure `validate_console_session_settings()` still requires a model for generic providers unless one is passed by the effective Console selection. Keep URL validation before readiness.

- [ ] **Step 6: Run readiness tests**

Run:

```bash
python -m pytest -q Tests/Chat/test_provider_readiness.py Tests/Chat/test_console_session_settings.py --tb=short
```

Expected: PASS after updating any old WIP assertions to the new supported-provider behavior.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Chat/provider_readiness.py tldw_chatbook/Chat/console_session_settings.py Tests/Chat/test_provider_readiness.py Tests/Chat/test_console_session_settings.py
git commit -m "Use shared Console provider readiness support"
```

## Task 3: Extend Gateway Resolution For Generic Providers

**Files:**
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py`
- Modify: `Tests/Chat/test_console_provider_gateway.py`

- [ ] **Step 1: Write failing generic resolution tests**

Add to `Tests/Chat/test_console_provider_gateway.py`:

```python
@pytest.mark.asyncio
async def test_resolve_for_send_openai_uses_env_key_and_execution_key() -> None:
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key_env_var": "OPENAI_API_KEY"}}},
        environ={"OPENAI_API_KEY": "sk-test-secret"},
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1", streaming=False)
    )

    assert resolved.ready is True
    assert resolved.provider == "openai"
    assert resolved.readiness_key == "openai"
    assert resolved.execution_key == "openai"
    assert resolved.api_key == "sk-test-secret"
    assert "sk-test-secret" not in resolved.visible_copy


@pytest.mark.asyncio
async def test_resolve_for_send_supported_provider_missing_key_blocks_without_wip() -> None:
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"anthropic": {"api_key_env_var": "ANTHROPIC_API_KEY"}}},
        environ={},
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="anthropic", explicit_model="claude-sonnet")
    )

    assert resolved.ready is False
    assert "Missing API key" in resolved.visible_copy
    assert "not wired" not in resolved.visible_copy
```

Add alias/base URL tests:

```python
@pytest.mark.asyncio
async def test_resolve_for_send_custom_alias_uses_custom_openai_execution_key() -> None:
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"custom": {"model": "m"}}},
        environ={},
    )

    resolved = await gateway.resolve_for_send(ConsoleProviderSelection(provider="Custom", configured_model="m"))

    assert resolved.ready is True
    assert resolved.readiness_key == "custom"
    assert resolved.execution_key == "custom-openai-api"


@pytest.mark.asyncio
async def test_resolve_for_send_blocks_generic_base_url_override_that_differs_from_config() -> None:
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"ollama": {"api_url": "http://127.0.0.1:11434"}}},
        environ={},
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="ollama", explicit_model="llama3", base_url="http://127.0.0.1:9999")
    )

    assert resolved.ready is False
    assert "save the endpoint in Settings" in resolved.visible_copy
```

- [ ] **Step 2: Run focused gateway resolution tests to verify failure**

Run:

```bash
python -m pytest -q Tests/Chat/test_console_provider_gateway.py::test_resolve_for_send_openai_uses_env_key_and_execution_key Tests/Chat/test_console_provider_gateway.py::test_resolve_for_send_supported_provider_missing_key_blocks_without_wip --tb=short
```

Expected: FAIL because `ConsoleProviderGateway.__init__()` lacks the new dependencies and generic providers still WIP-block.

- [ ] **Step 3: Extend `ConsoleProviderResolution`**

Add optional fields with safe defaults to preserve existing llama tests:

```python
@dataclass(frozen=True)
class ConsoleProviderResolution:
    provider: str
    base_url: str
    model: str | None
    ready: bool
    visible_copy: str = ""
    readiness_key: str = ""
    execution_key: str = ""
    api_key: str | None = None
    api_key_source: str | None = None
    ...
```

Set direct llama resolutions to `readiness_key=provider` and `execution_key=provider`.

- [ ] **Step 4: Add gateway dependencies**

Update `ConsoleProviderGateway.__init__()`:

```python
def __init__(
    self,
    *,
    http_client: httpx.AsyncClient | None = None,
    config_provider: Callable[[], Mapping[str, object]] | None = None,
    environ: Mapping[str, str] | None = None,
    chat_api_call_fn: Callable[..., Any] | None = None,
    safe_error_copy: Callable[[str, BaseException], str] | None = None,
) -> None:
    self._config_provider = config_provider or (lambda: {})
    self._environ = environ
    self._chat_api_call_fn = chat_api_call_fn or chat_api_call
    self._safe_error_copy = safe_error_copy or safe_provider_error_copy
```

Import `chat_api_call` lazily or at module level if tests show no cycle.

- [ ] **Step 5: Implement generic `resolve_for_send()`**

Use the shared identity helper:

```python
identity = resolve_console_provider_identity(selection.provider)
if not identity.is_supported:
    return blocked("Provider blocked: `<provider>` is not available in Console yet.")
if identity.uses_direct_llama_path:
    return await self.resolve_llamacpp(...)
```

Read current config at send time:

```python
app_config = self._config_provider() or {}
provider_settings = _provider_settings(app_config, identity.readiness_key)
readiness = get_provider_readiness(identity.readiness_key, app_config, environ=self._environ)
```

Use the same normalized provider-settings helper shape as `console_session_settings.py`, or import a shared pure helper if one is extracted during implementation. Do not rely on direct `api_settings[identity.readiness_key]` lookup.

Resolve model:

```python
model = selection.explicit_model or selection.configured_model or _first_string(
    provider_settings.get("model"),
    provider_settings.get("api_model"),
    provider_settings.get("default_model"),
)
if not model:
    return blocked("Select a model before sending.")
```

Reject nonmatching generic base URL overrides:

```python
if _has_different_generic_base_url(selection.base_url, provider_settings):
    return blocked("Provider blocked: save the endpoint in Settings before using it from Console.")
```

- [ ] **Step 6: Run gateway resolution tests**

Run:

```bash
python -m pytest -q Tests/Chat/test_console_provider_gateway.py -k "resolve_for_send" --tb=short
```

Expected: PASS, including existing llama resolution tests.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Chat/console_provider_gateway.py Tests/Chat/test_console_provider_gateway.py
git commit -m "Resolve generic Console providers"
```

## Task 4: Add Generic `chat_api_call()` Streaming Adapter

**Files:**
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py`
- Modify: `Tests/Chat/test_console_provider_gateway.py`

- [ ] **Step 1: Write failing non-streaming and streaming adapter tests**

Add tests:

```python
@pytest.mark.asyncio
async def test_stream_chat_generic_non_streaming_yields_completion_once() -> None:
    calls = []

    def fake_chat_api_call(**kwargs):
        calls.append(kwargs)
        return "generic done"

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1", streaming=False, temperature=0.2)
    )

    chunks = [chunk async for chunk in gateway.stream_chat(resolution, [{"role": "user", "content": "hi"}])]

    assert chunks == ["generic done"]
    assert calls[0]["api_endpoint"] == "openai"
    assert calls[0]["messages_payload"] == [{"role": "user", "content": "hi"}]
    assert calls[0]["model"] == "gpt-4.1"
    assert calls[0]["streaming"] is False
    assert calls[0]["temp"] == 0.2


@pytest.mark.asyncio
async def test_stream_chat_generic_sync_generator_yields_ordered_chunks() -> None:
    def fake_chat_api_call(**_kwargs):
        yield "hel"
        yield {"choices": [{"delta": {"content": "lo"}}]}

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1"))

    chunks = [chunk async for chunk in gateway.stream_chat(resolution, [{"role": "user", "content": "hi"}])]

    assert chunks == ["hel", "lo"]
```

- [ ] **Step 2: Run adapter tests to verify failure**

Run:

```bash
python -m pytest -q Tests/Chat/test_console_provider_gateway.py::test_stream_chat_generic_non_streaming_yields_completion_once Tests/Chat/test_console_provider_gateway.py::test_stream_chat_generic_sync_generator_yields_ordered_chunks --tb=short
```

Expected: FAIL because `stream_chat()` only dispatches llama providers.

- [ ] **Step 3: Implement worker-thread-to-async-queue bridge**

In `ConsoleProviderGateway.stream_chat()` add a generic branch:

```python
if resolution.execution_key:
    async for chunk in self._stream_generic_chat(resolution, messages):
        yield chunk
```

Implement `_stream_generic_chat()` so sync invocation and sync iteration both happen in a worker thread:

```python
async def _stream_generic_chat(self, resolution, messages):
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[_QueueItem] = asyncio.Queue()
    stop_event = threading.Event()

    def enqueue(item: _QueueItem) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, item)

    def worker() -> None:
        try:
            response = self._chat_api_call_fn(**self._chat_api_kwargs(resolution, messages))
            for text in self._iter_normalized_response(response):
                if stop_event.is_set():
                    break
                enqueue(_QueueItem.content(text))
        except BaseException as exc:
            enqueue(_QueueItem.error(self._safe_error_copy(resolution.provider, exc)))
        finally:
            enqueue(_QueueItem.done())

    task = asyncio.create_task(asyncio.to_thread(worker))
    try:
        while True:
            item = await queue.get()
            if item.kind == "done":
                break
            if item.kind == "error":
                yield item.text
                break
            if item.text:
                yield item.text
    finally:
        stop_event.set()
        task.cancel()
```

Adjust implementation details as needed; do not call `queue.put_nowait()` directly from the worker thread.

- [ ] **Step 4: Build generic kwargs exactly once**

Add `_chat_api_kwargs()`:

```python
kwargs = {
    "api_endpoint": resolution.execution_key,
    "messages_payload": list(messages),
    "api_key": resolution.api_key,
    "model": resolution.model,
    "streaming": resolution.streaming,
    "temp": resolution.temperature,
    "topp": resolution.top_p,
    "maxp": resolution.top_p,
    "topk": resolution.top_k,
    "minp": resolution.min_p,
    "max_tokens": resolution.max_tokens,
}
return {key: value for key, value in kwargs.items() if value is not None}
```

- [ ] **Step 5: Run adapter tests**

Run:

```bash
python -m pytest -q Tests/Chat/test_console_provider_gateway.py -k "generic or stream_chat" --tb=short
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Chat/console_provider_gateway.py Tests/Chat/test_console_provider_gateway.py
git commit -m "Stream generic Console provider responses"
```

## Task 5: Normalize Responses, Redact Errors, And Handle Cancel Safely

**Files:**
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py`
- Modify: `Tests/Chat/test_console_provider_gateway.py`

- [ ] **Step 1: Write failing response normalization tests**

Add tests for dict precedence and unsupported shapes:

```python
def test_normalize_generic_provider_response_shapes() -> None:
    assert list(ConsoleProviderGateway.normalize_provider_response({"content": "body"})) == ["body"]
    assert list(ConsoleProviderGateway.normalize_provider_response({"choices": [{"message": {"content": "choice"}}]})) == ["choice"]
    assert list(ConsoleProviderGateway.normalize_provider_response({"generated_text": "generated"})) == ["generated"]
    assert list(ConsoleProviderGateway.normalize_provider_response([{"content": "do not dump"}])) == [
        "Provider returned an unsupported response shape."
    ]
```

Add redaction/error tests:

```python
def test_safe_provider_error_copy_redacts_secret_like_values() -> None:
    copy = safe_provider_error_copy("openai", RuntimeError("Authorization: Bearer sk-1234567890abcdef"))

    assert "sk-1234567890abcdef" not in copy
    assert "openai" in copy
```

Add cancellation test using a generator that emits after cancellation:

```python
@pytest.mark.asyncio
async def test_stream_chat_generic_cancel_ignores_late_chunks() -> None:
    gate = threading.Event()

    def fake_chat_api_call(**_kwargs):
        yield "first"
        gate.wait(timeout=1)
        yield "late"

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(ConsoleProviderSelection(provider="openai", explicit_model="m"))
    stream = gateway.stream_chat(resolution, [{"role": "user", "content": "hi"}])

    assert await anext(stream) == "first"
    await stream.aclose()
    gate.set()
```

- [ ] **Step 2: Run response/error tests to verify failure**

Run:

```bash
python -m pytest -q Tests/Chat/test_console_provider_gateway.py -k "normalize or redacts or cancel" --tb=short
```

Expected: FAIL until helpers exist and cancellation semantics are implemented.

- [ ] **Step 3: Implement normalization helpers**

Keep helper methods small and testable:

```python
@staticmethod
def normalize_provider_response(response: Any) -> Iterator[str]:
    text = _content_from_provider_item(response)
    if text:
        yield text
        return
    if _is_iterable_response(response):
        emitted = False
        for item in response:
            text = _content_from_provider_item(item)
            if text:
                emitted = True
                yield text
            elif text == "":
                continue
            else:
                yield "Provider returned an unsupported response shape."
                emitted = True
        if not emitted:
            yield "Provider returned no assistant content."
        return
    yield "Provider returned an unsupported response shape."
```

Make dict extraction follow the spec precedence exactly. `_is_iterable_response()` must exclude `str`, `bytes`, mappings, lists, and tuples; lists/tuples are unsupported response shapes in this slice and must not be recursively dumped into the transcript.

- [ ] **Step 4: Implement safe error copy**

Use existing exception classes from `tldw_chatbook.Chat.Chat_Deps`:

```python
def safe_provider_error_copy(provider: str, exc: BaseException) -> str:
    category = "unexpected provider error"
    if isinstance(exc, ChatAuthenticationError):
        category = "authentication failed"
    elif isinstance(exc, ChatRateLimitError):
        category = "rate limit exceeded"
    elif isinstance(exc, ChatBadRequestError):
        category = "bad request"
    elif isinstance(exc, ChatConfigurationError):
        category = "configuration error"
    elif isinstance(exc, ChatProviderError):
        category = "provider unavailable"
    return f"Provider error from {provider}: {category}."
```

If any safe detail is included, redact API-key-like, bearer-token, password-like, and URL credential substrings first.

- [ ] **Step 5: Run gateway tests**

Run:

```bash
python -m pytest -q Tests/Chat/test_console_provider_gateway.py --tb=short
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Chat/console_provider_gateway.py Tests/Chat/test_console_provider_gateway.py
git commit -m "Harden generic Console provider output handling"
```

## Task 6: Wire Production Gateway With Current App Config

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/UI/test_console_native_chat_flow.py`

- [ ] **Step 1: Write failing mounted generic success test**

Add `ConsoleProviderGateway` to the imports in `Tests/UI/test_console_native_chat_flow.py`, then add:

```python
@pytest.mark.asyncio
async def test_console_native_generic_provider_send_renders_completed_message():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": {"api_key": "sk-test"}}

    def fake_chat_api_call(**_kwargs):
        return "generic provider response"

    app.console_provider_gateway_factory = lambda: ConsoleProviderGateway(
        config_provider=lambda: app.app_config,
        chat_api_call_fn=fake_chat_api_call,
    )

    async with app.run_test() as pilot:
        console = ChatScreen(app)
        await app.push_screen(console)
        await _wait_for_selector(console, "#console-composer-input", pilot)
        console.query_one("#console-composer-input", Input).value = "hello"
        await pilot.click("#console-send-button")
        await _wait_for_text(console, pilot, "generic provider response")
```

- [ ] **Step 2: Write failing blocked-send composer preservation test using real gateway**

Add:

```python
@pytest.mark.asyncio
async def test_console_native_missing_key_blocks_before_clearing_generic_draft():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": {"api_key_env_var": "MISSING_OPENAI_KEY"}}
    app.console_provider_gateway_factory = lambda: ConsoleProviderGateway(
        config_provider=lambda: app.app_config,
        environ={},
    )

    async with app.run_test() as pilot:
        console = ChatScreen(app)
        await app.push_screen(console)
        await _wait_for_selector(console, "#console-composer-input", pilot)
        composer = console.query_one("#console-composer-input", Input)
        composer.value = "preserve this"
        await pilot.click("#console-send-button")
        await _wait_for_text(console, pilot, "Missing API key")
        assert composer.value == "preserve this"
```

- [ ] **Step 3: Run mounted tests to verify failure**

Run:

```bash
python -m pytest -q Tests/UI/test_console_native_chat_flow.py::test_console_native_generic_provider_send_renders_completed_message Tests/UI/test_console_native_chat_flow.py::test_console_native_missing_key_blocks_before_clearing_generic_draft --tb=short
```

Expected: FAIL until generic gateway wiring and readiness are active in `ChatScreen`.

- [ ] **Step 4: Wire production gateway with `config_provider`**

Modify `_ensure_console_provider_gateway()` in `tldw_chatbook/UI/Screens/chat_screen.py`:

```python
self._console_provider_gateway = (
    factory()
    if callable(factory)
    else ConsoleProviderGateway(
        config_provider=lambda: getattr(self.app_instance, "app_config", {}) or {},
    )
)
```

Keep the factory path unchanged so tests can still inject custom gateways.

- [ ] **Step 5: Ensure Console settings readiness uses the same provider identity**

Verify `_active_console_settings_readiness()` still builds effective settings from `_build_console_provider_selection()` and calls `build_console_settings_readiness()` once. Do not introduce a separate provider/model computation path.

- [ ] **Step 6: Run mounted Console tests**

Run:

```bash
python -m pytest -q Tests/UI/test_console_native_chat_flow.py --tb=short
```

Expected: PASS or unrelated baseline failures documented separately.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_native_chat_flow.py
git commit -m "Wire Console generic provider sends"
```

## Task 7: Focused Verification And Screenshot Approval

**Files:**
- Modify only if verification exposes issues.

- [x] **Step 1: Run focused provider test suite**

Run:

```bash
python -m pytest -q Tests/Chat/test_console_provider_support.py Tests/Chat/test_console_session_settings.py Tests/Chat/test_console_provider_gateway.py Tests/UI/test_console_native_chat_flow.py --tb=short
```

Expected: PASS.

- [x] **Step 2: Run existing Console layout/session regressions**

Run:

```bash
python -m pytest -q Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_session_settings.py --tb=short
```

Expected: PASS or document unrelated baseline failures.

- [x] **Step 3: Run static diff check**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

- [x] **Step 4: Capture actual CDP/textual-web screenshots**

Use the project CDP workflow documented in:

- `Docs/superpowers/guides/2026-05-09-textual-web-cdp-debugging.md`

Capture and request approval for:

- Generic provider success: Console sends via fake/local generic provider and renders completed assistant message.
- Missing key blocked: draft text remains visible in composer and transcript/inspector show recovery.
- Base URL override blocked: recovery copy says to save endpoint in Settings.

- [x] **Step 5: Final commit if verification fixes were needed**

```bash
git status --short
git add <changed files>
git commit -m "Verify Console provider parity"
```

Only commit if verification required changes.

Evidence:

- `Docs/superpowers/qa/product-maturity/screen-qa/console/provider-parity/notes.md`
- `Docs/superpowers/qa/product-maturity/screen-qa/console/provider-parity/generic-success-2026-05-28.png`
- `Docs/superpowers/qa/product-maturity/screen-qa/console/provider-parity/missing-key-blocked-2026-05-28.png`
- `Docs/superpowers/qa/product-maturity/screen-qa/console/provider-parity/base-url-override-blocked-2026-05-28.png`
- User approval received in Codex thread after actual textual-web/CDP screenshot review.

## Review Checklist

- [x] No supported `API_CALL_HANDLERS` key receives the old llama-only WIP copy.
- [x] Missing key blocks before composer clear.
- [x] Direct llama.cpp tests still pass.
- [x] Generic non-streaming provider returns one completed assistant message.
- [x] Generic sync generator provider streams without blocking the event loop.
- [x] Stop/cancel ignores late sync-worker chunks.
- [x] Provider/model labels, Console Settings readiness, composer preflight, and Run Inspector derive from the same provider identity/readiness path.
- [x] No visible copy prints raw API keys, bearer tokens, URL credentials, or raw provider exception text.
- [x] Actual CDP/textual-web screenshots are captured before PR approval.
