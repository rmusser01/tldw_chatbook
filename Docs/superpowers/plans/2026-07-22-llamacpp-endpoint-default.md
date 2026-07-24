# TASK-433 llama.cpp endpoint default — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A user who accepts the prefilled llama.cpp endpoint gets working chat on both send paths, and existing users who saved a `/completion` (or `/v1`) endpoint also work — by fixing the config default to the server root and making the legacy `chat_with_llama` path normalize the base URL the way the Console path already does.

**Architecture:** Two independent changes — (1) config default + Settings placeholder + guidance now use the server root; (2) `chat_with_llama` normalizes its base URL via the existing public `normalize_llamacpp_base_url` before the shared caller appends `/v1/chat/completions`.

**Tech Stack:** Python 3.11+, pytest.

## Global Constraints

- Reuse the **existing public** `normalize_llamacpp_base_url` from `tldw_chatbook/Chat/console_provider_gateway.py` (do NOT rename it, do NOT reimplement it, do NOT touch the duplicate copy in `console_session_settings.py`).
- Import it with a **deferred (local) import** inside `chat_with_llama` — not at module top (keeps `console_provider_gateway`'s heavy deps out of app startup).
- Normalize only a **non-empty** `api_base_url`, **after** the existing empty-guard `raise ChatConfigurationError`.
- Change scoped to `chat_with_llama` and config/placeholder text only. Do NOT modify the shared `_chat_with_openai_compatible_local_server` or any other provider's behavior.
- In `config.py`, edit ONLY the `[api_settings.llama_cpp].api_url` line (~:2476, ending in `/completion`). Leave `llama_api_IP` (~:1072) and the `[providers]` `llama_cpp` (~:2270) alone.

---

### Task 1: Server-root default, placeholder, and guidance

**Files:**
- Modify: `tldw_chatbook/config.py` (the `[api_settings.llama_cpp].api_url` default, ~line 2476)
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py` (`PROVIDER_ENDPOINT_PLACEHOLDERS`, ~lines 341-342)
- Test: `Tests/Chat/test_chat_functions.py` (or an existing config-defaults test module — a plain `tomllib.loads(CONFIG_TOML_CONTENT)` assertion, no mounting)

**Interfaces:**
- Produces: the corrected default string `"http://localhost:8080"` under `[api_settings.llama_cpp].api_url`.

- [ ] **Step 1: Write the failing config-default test**

```python
# Tests/Chat/test_chat_functions.py  (add near the other module-level tests)
import tomllib
from tldw_chatbook.config import CONFIG_TOML_CONTENT


def test_llama_cpp_default_endpoint_is_server_root():
    cfg = tomllib.loads(CONFIG_TOML_CONTENT)
    api_url = cfg["api_settings"]["llama_cpp"]["api_url"]
    assert api_url == "http://localhost:8080"
    assert "/completion" not in api_url
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest Tests/Chat/test_chat_functions.py::test_llama_cpp_default_endpoint_is_server_root -q`
Expected: FAIL (`assert 'http://localhost:8080/completion' == 'http://localhost:8080'`).

- [ ] **Step 3: Fix the config default + comment** (`config.py`, ~line 2476)

Change:
```
    api_url = "http://localhost:8080/completion" # llama.cpp /completion endpoint
```
to:
```
    api_url = "http://localhost:8080" # llama.cpp server root; the OpenAI-compatible /v1/chat/completions path is appended automatically
```

- [ ] **Step 4: Align the Settings placeholders** (`settings_screen.py`, in `PROVIDER_ENDPOINT_PLACEHOLDERS`)

Change both the `llama_cpp` and `local_llamacpp` entries from `"http://127.0.0.1:9099/v1"` to the server-root form `"http://127.0.0.1:9099"` (drop the `/v1`; port unchanged — the 8080-vs-9099 difference is pre-existing and out of scope).

- [ ] **Step 5: Run the test to confirm it passes**

Run: `python -m pytest Tests/Chat/test_chat_functions.py::test_llama_cpp_default_endpoint_is_server_root -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/config.py tldw_chatbook/UI/Screens/settings_screen.py Tests/Chat/test_chat_functions.py
git commit -m "fix(config): llama.cpp default endpoint is the server root (task-433 AC#1/AC#2)"
```

---

### Task 2: Tolerate legacy endpoint suffixes on the `chat_with_llama` path

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py` (`chat_with_llama`, just after the `api_base_url` empty-guard, ~line 648)
- Test: `Tests/Chat/test_chat_functions.py` (URL-capture, mirrors the existing `captured["url"]` huggingface tests) and `Tests/Chat/test_console_provider_gateway.py` (direct normalizer-contract test if none exists)

**Interfaces:**
- Consumes: `normalize_llamacpp_base_url(api_url: str | None) -> str` from `tldw_chatbook/Chat/console_provider_gateway.py` (existing public function).

- [ ] **Step 1: Write the failing normalizer-contract test**

```python
# Tests/Chat/test_console_provider_gateway.py
from tldw_chatbook.Chat.console_provider_gateway import normalize_llamacpp_base_url


def test_normalize_llamacpp_base_url_strips_known_suffixes_to_root():
    root = "http://localhost:8080"
    assert normalize_llamacpp_base_url("http://localhost:8080/completion") == root
    assert normalize_llamacpp_base_url("http://localhost:8080/v1") == root
    assert normalize_llamacpp_base_url("http://localhost:8080/v1/chat/completions") == root
    assert normalize_llamacpp_base_url("http://localhost:8080") == root
    assert normalize_llamacpp_base_url("localhost:8080/completion") == root  # scheme-less
    # a reverse-proxy prefix is NOT an exact suffix -> left unchanged
    assert (
        normalize_llamacpp_base_url("http://host/proxy/v1/chat/completions")
        == "http://host/proxy/v1/chat/completions"
    )
```

- [ ] **Step 2: Run it (should PASS immediately — pins the reused contract)**

Run: `python -m pytest Tests/Chat/test_console_provider_gateway.py::test_normalize_llamacpp_base_url_strips_known_suffixes_to_root -q`
Expected: PASS (the function already behaves this way; this test guards the contract we now depend on). If any assertion fails, STOP and report — the design assumption is wrong.

- [ ] **Step 3: Write the failing URL-capture test for `chat_with_llama`**

Reuse the module's `_CapturedSession` (defined in `Tests/Chat/test_chat_functions.py`). The local caller calls `session.mount(...)` then `session.post(url, ...)`, so the stub needs a no-op `mount`; if `_CapturedSession` lacks it, add a no-op `mount(self, *a, **k): pass` to that shared stub (harmless to the huggingface tests, which don't call it). Patch `LLM_API_Calls_Local.requests.Session`.

```python
# Tests/Chat/test_chat_functions.py
import pytest


@pytest.mark.parametrize(
    "configured_url",
    [
        "http://localhost:8080/completion",
        "http://localhost:8080/v1",
        "http://localhost:8080",
    ],
)
def test_chat_with_llama_posts_to_v1_chat_completions_regardless_of_suffix(
    monkeypatch, configured_url
):
    from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local

    captured = {}
    response_data = {"choices": [{"message": {"content": "ok"}}]}
    # chat_with_llama reads the MODULE-LEVEL ``settings`` dict
    # (``settings.get("api_settings", {})``), not ``load_settings()`` — patch it
    # the way Tests/LLM_Management/test_mlx_lm.py does.
    monkeypatch.setattr(
        LLM_API_Calls_Local,
        "settings",
        {"api_settings": {"llama_cpp": {"api_url": configured_url, "model": "test-model"}}},
    )
    monkeypatch.setattr(
        LLM_API_Calls_Local.requests,
        "Session",
        lambda: _CapturedSession(captured, response_data),
    )

    LLM_API_Calls_Local.chat_with_llama(
        input_data=[{"role": "user", "content": "hi"}],
        api_key="",
        temp=0.7,
        streaming=False,
    )

    assert captured["url"] == "http://localhost:8080/v1/chat/completions"
```

Confirmed signature: `chat_with_llama(input_data, api_key=None, custom_prompt=None, temp=None, system_prompt=None, streaming=False, model=None, ...)` — `input_data` is the messages list; `model` comes from the mocked config. If the caller's response parsing needs more than `{"choices":[{"message":{"content":...}}]}` (e.g. a `status_code`/`.json()` on the stubbed response), extend `_CapturedSession`'s canned response accordingly (check `_CapturedSession.post`).

- [ ] **Step 4: Run it to confirm it fails**

Run: `python -m pytest Tests/Chat/test_chat_functions.py -k chat_with_llama_posts_to_v1 -q`
Expected: FAIL — for `/completion` the captured URL is `http://localhost:8080/completion/v1/chat/completions`; for `/v1` it is `.../v1/v1/chat/completions`.

- [ ] **Step 5: Add the deferred normalization** (`LLM_API_Calls_Local.py`, `chat_with_llama`)

Immediately after the empty-guard block:
```python
    api_base_url = llama_config.get("api_url")
    if not api_base_url:
        raise ChatConfigurationError(
            provider=llama_cpp_config_key_in_api_settings,
            message=f"{provider_display_name} API URL (api_url) is required and could not be determined from configuration.",
        )
    # task-433: tolerate legacy/partial endpoint forms (/completion, /v1, full
    # OpenAI path) by normalizing to the server root; the shared caller then
    # appends v1/chat/completions exactly once. Deferred import keeps
    # console_provider_gateway's deps out of app startup.
    from ..Chat.console_provider_gateway import normalize_llamacpp_base_url
    api_base_url = normalize_llamacpp_base_url(api_base_url)
```

- [ ] **Step 6: Run the URL-capture + normalizer tests to confirm they pass**

Run: `python -m pytest Tests/Chat/test_chat_functions.py -k chat_with_llama_posts_to_v1 Tests/Chat/test_console_provider_gateway.py::test_normalize_llamacpp_base_url_strips_known_suffixes_to_root -q`
Expected: PASS (all 3 parametrized URLs post to `http://localhost:8080/v1/chat/completions`).

- [ ] **Step 7: Regression — other providers unaffected + import sanity**

Run:
```bash
python -c "import tldw_chatbook.LLM_Calls.LLM_API_Calls_Local"   # deferred import doesn't break module load
python -m pytest Tests/Chat/test_chat_functions.py Tests/Chat/test_console_provider_gateway.py -q
```
Expected: PASS (kobold/ooba/vllm/hf URL tests unchanged; gateway tests green).

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py Tests/Chat/test_chat_functions.py Tests/Chat/test_console_provider_gateway.py
git commit -m "fix(llm): normalize llama.cpp endpoint so legacy /completion & /v1 work (task-433 AC#1)"
```

---

## Self-review notes

- **Spec coverage:** AC#1 (accepted default works) → Task 1 (server-root default) + Task 2 (caller tolerates it on the legacy path). AC#2 (guidance matches) → Task 1 (config comment + placeholder form). Both covered.
- **Placeholder scan:** the only prose-only step is confirming `chat_with_llama`'s exact parameter names / settings-read in Task 2 Step 3 — a necessary verification against the real signature, with the mirror source named (`test_mlx_lm.py`, existing `_CapturedSession` tests).
- **Type/name consistency:** `normalize_llamacpp_base_url` used identically in Task 2 and the spec; import path `..Chat.console_provider_gateway` from `LLM_Calls/`. The config default string `"http://localhost:8080"` matches between Task 1's edit and its test.
- **Deferred vs eager:** the import is deliberately inside `chat_with_llama` (Global Constraints), so Step 7's bare-`import` sanity check confirms no module-load regression.
