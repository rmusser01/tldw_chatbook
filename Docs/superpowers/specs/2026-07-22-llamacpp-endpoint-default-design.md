# TASK-433 — Fix llama.cpp prefilled endpoint default + tolerate legacy suffixes

- **Date:** 2026-07-22
- **Task:** TASK-433 (RP/character-card UX review). Settings ▸ Providers & Models llama.cpp endpoint.
- **Branch base:** origin/dev (tip `e35e27a08`).

## Problem

The config template's llama.cpp default is `[api_settings.llama_cpp].api_url = "http://localhost:8080/completion"` (`config.py:2476`) — llama.cpp's *native* completion endpoint. But the legacy chat caller `_chat_with_openai_compatible_local_server` (`LLM_API_Calls_Local.py:213-222`) appends `v1/chat/completions` unless the URL already ends with exactly that, so the prefilled default becomes `http://localhost:8080/completion/v1/chat/completions` → a user who keeps the suggested default gets failures after a green-looking setup.

The legacy caller is fragile for *any* partial path: the Settings placeholder form `http://127.0.0.1:9099/v1` (`settings_screen.py:341`) would likewise become `.../v1/v1/chat/completions`. Only a bare server root or the exact full `/v1/chat/completions` works.

Two send paths exist and they diverge:
- **Native Console / Personas preview** → `ConsoleProviderGateway._normalize_llamacpp_base_url` (`console_provider_gateway.py:95`) already **strips** known endpoint suffixes (`/completion`, `/completions`, `/v1`, `/chat/completions`, `/v1/chat/completions`, `/models`, ...) down to the server root — so it *tolerates* the bad default.
- **Legacy / main chat** (`Chat_Functions` `API_CALL_HANDLERS["llama_cpp"|"local_llamacpp"|"local_llamafile"] = chat_with_llama` → `_chat_with_openai_compatible_local_server`) — the fragile one the review cited.

## Decision (from brainstorming)

**Robust fix:** fix the default **and** make the legacy path tolerant, so new users, existing users who saved `/completion`, and any partial-path form all work and the two send paths behave identically. Prefilled default = **server root** `http://localhost:8080`.

## Design

### 1. Fix the config default + guidance (`config.py:2476`)
```
api_url = "http://localhost:8080/completion" # llama.cpp /completion endpoint
```
→
```
api_url = "http://localhost:8080" # llama.cpp server root; the OpenAI-compatible /v1/chat/completions path is appended automatically
```
Server root matches `console_provider_gateway`'s base-URL semantics and works on both send paths. AC#1 (new users). AC#2 (the misleading `/completion` guidance is gone).

**Target the right line.** `config.py` has other llama entries that are NOT this bug and must be left alone: `llama_api_IP = "http://127.0.0.1:8080/v1/chat/completions"` (~:1072, a different/legacy key, already correct-form) and `llama_cpp = "http://localhost:8080"` (~:2270, a `[providers]` address, already server root). Only the `[api_settings.llama_cpp].api_url` line (~:2476, ending in `/completion`) is the defect.

### 2. Reuse the existing public llama.cpp URL normalizer
`normalize_llamacpp_base_url` is **already public** at `console_provider_gateway.py:88` and is already imported by `chat_screen.py:137`, so **no rename is needed** (an earlier draft mistakenly said to publicize a `_`-prefixed name). It strips any known endpoint suffix (`/v1`, `/v1/models`, `/models`, `/v1/chat/completions`, `/chat/completions`, `/completion`, `/completions`) to the server root, prepends `http://` when scheme-less, and returns `DEFAULT_LLAMACPP_BASE_URL` (`http://127.0.0.1:9099`) for empty input.

*Duplication note (pre-existing, not fixed here):* a second, equivalent `normalize_llamacpp_base_url` lives in `console_session_settings.py:112`. Reuse the **`console_provider_gateway`** copy (the one the UI already uses). Deduping the two copies is a follow-up, not part of this task (non-goal).

### 3. Normalize the base URL on the legacy llama.cpp path (`LLM_API_Calls_Local.py`, `chat_with_llama`)
After `chat_with_llama` resolves `api_base_url = llama_config.get("api_url")` **and passes its existing empty-guard** (`if not api_base_url: raise ...`), normalize the non-empty value before handing it to `_chat_with_openai_compatible_local_server`. Use a **deferred (local) import** inside the function — this file already defers Chat/Character submodule imports, and a top-level import of `console_provider_gateway` (which pulls in `httpx`, `Chat_Deps`, etc.) would add app-startup weight and risk import cycles:
```python
def chat_with_llama(...):
    ...
    api_base_url = llama_config.get("api_url")
    if not api_base_url:
        raise ...            # unchanged empty-guard
    from ..Chat.console_provider_gateway import normalize_llamacpp_base_url
    api_base_url = normalize_llamacpp_base_url(api_base_url)
    ...
```
(Confirm the relative depth: `LLM_API_Calls_Local.py` is in `tldw_chatbook/LLM_Calls/`, so `from ..Chat.console_provider_gateway import ...`.)

`normalize_llamacpp_base_url` strips a known endpoint suffix to the server root; `_chat_with_openai_compatible_local_server` then appends `v1/chat/completions`. Result for every reasonable input:
- `http://localhost:8080/completion` → root `http://localhost:8080` → `http://localhost:8080/v1/chat/completions` ✅
- `http://localhost:8080/v1` → root → `.../v1/chat/completions` ✅ (fixes the placeholder form)
- `http://localhost:8080` → root → `.../v1/chat/completions` ✅
- `http://localhost:8080/v1/chat/completions` → root → `.../v1/chat/completions` ✅ (idempotent)
- `http://host/proxy/v1/chat/completions` (reverse-proxy prefix) → NOT an exact suffix match, returned unchanged → the caller's existing `endswith("v1/chat/completions")` branch uses it as-is ✅ (no regression — this is the one case the plain caller already handled).

Scoped to `chat_with_llama` (covers `llama_cpp`, `local_llamacpp`, `local_llamafile`; `local-llm` uses a different handler `chat_with_local_llm` and is out of scope); the shared `_chat_with_openai_compatible_local_server` and other providers (kobold/ooba/vllm/mlx) are untouched.

*Edge:* a non-empty but whitespace-only `api_url` normalizes to `DEFAULT_LLAMACPP_BASE_URL` (`:9099`) instead of the previous raw-string failure — an acceptable, marginally-better fallback for a garbage config value; noted, not specially handled.

### 4. Align the Settings placeholder form (`settings_screen.py:341-342`)
Change the `llama_cpp` and `local_llamacpp` placeholders from `http://127.0.0.1:9099/v1` to the server-root form `http://127.0.0.1:9099` so the hint matches the "server root" the default now uses (both forms work post-fix; this is form-consistency for AC#2). Port left at 9099 — the config-default-vs-gateway-default port difference (8080 vs 9099) is pre-existing and out of scope.

## Testing

- **Normalizer contract (`normalize_llamacpp_base_url`):** direct unit tests that it returns the server root for `/completion`, `/v1`, `/v1/chat/completions`, bare root, and `host:port` (no scheme), and leaves a reverse-proxy-prefixed `/proxy/v1/chat/completions` unchanged. (Add to `Tests/Chat/test_console_provider_gateway.py` if no direct test exists.)
- **Legacy caller build (`LLM_API_Calls_Local`):** mirror the established URL-capture pattern in `Tests/Chat/test_chat_functions.py` (the `captured["url"] == "..."` assertions) — with the HTTP post mocked, `chat_with_llama` given `api_url="http://localhost:8080/completion"` (and `/v1`, and bare root) posts to exactly `http://localhost:8080/v1/chat/completions`, never `.../completion/...` or `.../v1/v1/...`. (settings-dict patch pattern: see `Tests/LLM_Management/test_mlx_lm.py`.)
- **Config default:** the parsed `CONFIG_TOML_CONTENT` for `[api_settings.llama_cpp].api_url` equals `http://localhost:8080` and contains no `/completion`.
- **Regression:** existing `LLM_API_Calls_Local` / `test_console_provider_gateway` / config tests stay green; other providers' URL construction is unchanged (kobold/ooba/vllm/mlx callers unaffected — they don't go through `chat_with_llama`).

## Risks / mitigations

- **Import weight / cycles:** a deferred local import in `chat_with_llama` keeps `console_provider_gateway`'s heavy deps out of app-startup and avoids any cycle (verified: `console_provider_gateway` does not import `LLM_Calls`).
- **Behavior change scope:** normalization applied only in `chat_with_llama`, so non-llama providers on the shared caller are unaffected (kept green by their existing tests).
- **Existing saved values:** users who saved `http://localhost:8080/completion` (or `/v1`) now succeed (normalized), so the fix is retroactive without a migration.
- **Duplication:** reusing the `console_provider_gateway` copy adds a third consumer of a function duplicated in `console_session_settings`; deduping is a noted follow-up, not done here.

## Non-goals

- Unifying the 8080-vs-9099 port convention between the config default and the Console gateway/placeholder (pre-existing; out of scope).
- Supporting llama.cpp's *native* `/completion` protocol (the callers are OpenAI-compatible only).
- Refactoring the shared `_chat_with_openai_compatible_local_server` append logic for all providers (kept minimal; llama-scoped normalization is sufficient).
- Changing the Console/preview send path (already correct).
