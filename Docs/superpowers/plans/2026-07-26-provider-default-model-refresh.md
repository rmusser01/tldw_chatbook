# Provider Default Model Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make fresh installations default to DeepSeek V4 Flash, Claude Sonnet 5, and GPT-5.6 Terra while preserving compatible request payloads and existing user overrides.

**Architecture:** Update the two bundled catalog representations and all active fallback layers in `config.py`, then add narrowly scoped model-family request shaping in the existing OpenAI and Anthropic handlers. Keep endpoints, response normalization, user-config precedence, and provider boundaries unchanged. Extend both embedded and Python capability metadata for the new vision-capable defaults.

**Tech Stack:** Python 3.11+, requests, TOML/tomllib, pytest, Textual application configuration.

**Spec:** `Docs/superpowers/specs/2026-07-26-provider-default-model-refresh-design.md` (read first; it is the source of truth).

**Backlog:** `backlog/tasks/task-519 - Refresh-default-models-for-DeepSeek-Anthropic-and-OpenAI.md`

**ADR required:** no

**ADR path:** `backlog/decisions/020-automatic-model-catalog-refresh.md`

**Reason:** This is a bundled-default and request-compatibility refresh within existing provider boundaries. ADR-020 already governs catalog discovery and persistence; no storage, ownership, security, or service-boundary decision is added.

**Worktree:** `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-519-provider-default-models`

**Branch:** `codex/task-519-provider-default-models`

**Baseline:** The targeted suite passes in the worktree-local environment: `70 passed` for `Tests/test_config_model_catalog_defaults.py`, `Tests/Chat/test_chat_functions.py`, and `Tests/test_model_capabilities.py`. Use `.venv/bin/python` in the commands below; the shared repository environment contains optional `parakeet_mlx`, which aborts in a headless session without a Metal device. Ruff 0.16 is installed in this local environment for the explicit lint/format checks below.

---

### Task 0: Commit the reviewed planning artifacts

**Files:**

- Add: `Docs/superpowers/plans/2026-07-26-provider-default-model-refresh.md`
- Modify: `backlog/tasks/task-519 - Refresh-default-models-for-DeepSeek-Anthropic-and-OpenAI.md`

- [x] **Step 1: Verify the planning diff**

```bash
git diff --check
git diff -- Docs/superpowers/plans/2026-07-26-provider-default-model-refresh.md "backlog/tasks/task-519 - Refresh-default-models-for-DeepSeek-Anthropic-and-OpenAI.md"
```

Expected: no whitespace errors; TASK-519 is In Progress and links the approved
design, this plan, and ADR-020 with `ADR required: no`.

- [x] **Step 2: Commit the planning artifacts before implementation**

```bash
git add Docs/superpowers/plans/2026-07-26-provider-default-model-refresh.md "backlog/tasks/task-519 - Refresh-default-models-for-DeepSeek-Anthropic-and-OpenAI.md"
git commit -m "docs: plan provider default model refresh"
```

Expected: the plan and Backlog task are tracked before any production-code task
begins.

---

### Task 1: Refresh bundled catalogs and configuration defaults

**Files:**

- Modify: `tldw_chatbook/config.py`
- Modify: `Tests/test_config_model_catalog_defaults.py`
- Test: `Tests/test_config_model_catalog_defaults.py`

- [x] **Step 1: Add failing configuration assertions**

Extend `Tests/test_config_model_catalog_defaults.py` to import
`API_MODELS_BY_PROVIDER` alongside `CONFIG_TOML_CONTENT`, then add one focused
test:

```python
def test_recommended_provider_defaults_and_catalogs_are_current():
    parsed = tomllib.loads(CONFIG_TOML_CONTENT)

    assert parsed["api_settings"]["deepseek"]["model"] == "deepseek-v4-flash"
    assert parsed["api_settings"]["anthropic"]["model"] == "claude-sonnet-5"
    assert parsed["api_settings"]["openai"]["model"] == "gpt-5.6-terra"
    assert parsed["chat_defaults"]["provider"] == "OpenAI"
    assert parsed["chat_defaults"]["model"] == "gpt-5.6-terra"

    assert parsed["providers"]["DeepSeek"] == [
        "deepseek-v4-flash",
        "deepseek-v4-pro",
    ]
    assert parsed["providers"]["Anthropic"][:4] == [
        "claude-sonnet-5",
        "claude-opus-5",
        "claude-fable-5",
        "claude-haiku-4-5",
    ]
    assert parsed["providers"]["OpenAI"][:3] == [
        "gpt-5.6-terra",
        "gpt-5.6-sol",
        "gpt-5.6-luna",
    ]
    assert "deepseek-chat" not in parsed["providers"]["DeepSeek"]
    assert "deepseek-reasoner" not in parsed["providers"]["DeepSeek"]
    assert API_MODELS_BY_PROVIDER["DeepSeek"] == parsed["providers"]["DeepSeek"]
    assert API_MODELS_BY_PROVIDER["Anthropic"] == parsed["providers"]["Anthropic"]
    assert API_MODELS_BY_PROVIDER["OpenAI"] == parsed["providers"]["OpenAI"]
```

- [x] **Step 2: Run the test and verify it fails**

Run:

```bash
.venv/bin/python -m pytest Tests/test_config_model_catalog_defaults.py -x -q
```

Expected: FAIL because the embedded provider catalogs and defaults still contain
the previous model IDs.

- [x] **Step 3: Update active configuration defaults**

In `tldw_chatbook/config.py`:

- Lead both the Python catalog literal and embedded TOML `[providers]` with:
  - OpenAI: `gpt-5.6-terra`, `gpt-5.6-sol`, `gpt-5.6-luna`
  - Anthropic: `claude-sonnet-5`, `claude-opus-5`, `claude-fable-5`,
    `claude-haiku-4-5`
  - DeepSeek: only `deepseek-v4-flash`, `deepseek-v4-pro`
- Retain the existing supported OpenAI and Anthropic entries after the new
  entries, preserving their current order.
- Change embedded `[api_settings.openai].model` to `gpt-5.6-terra`,
  `[api_settings.anthropic].model` to `claude-sonnet-5`, and
  `[api_settings.deepseek].model` to `deepseek-v4-flash`.
- Change `[chat_defaults].model` to `gpt-5.6-terra`; leave its provider as
  `OpenAI`.
- Change the corresponding `load_settings()` legacy fallback strings.
- Change the minimal malformed-catalog fallback to
  `{"OpenAI": ["gpt-5.6-terra"]}`.
- Do not change `[character_defaults]`, `[analysis_defaults]`, or specialized
  model settings.

- [x] **Step 4: Run the configuration tests**

Run:

```bash
.venv/bin/python -m pytest Tests/test_config_model_catalog_defaults.py -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add tldw_chatbook/config.py Tests/test_config_model_catalog_defaults.py
git commit -m "config: refresh provider default models"
```

---

### Task 2: Apply the GPT-5.6 Chat Completions contract

**Files:**

- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py`
- Modify: `Tests/Chat/test_chat_functions.py`
- Test: `Tests/Chat/test_chat_functions.py`

- [x] **Step 1: Add failing GPT-5.6 payload tests**

In `TestProviderRequestPayloads`, add tests using `_CapturedSession`:

```python
def test_openai_gpt_5_6_default_uses_chat_completion_contract(self, monkeypatch):
    from tldw_chatbook.LLM_Calls import LLM_API_Calls

    captured = {}
    monkeypatch.setattr(
        LLM_API_Calls,
        "load_settings",
        lambda: {"openai_api": {"api_base_url": "https://api.openai.test/v1"}},
    )
    monkeypatch.setattr(
        LLM_API_Calls.requests,
        "Session",
        lambda: _CapturedSession(
            captured,
            {"id": "chat_test", "choices": [{"message": {"content": "OK"}}]},
        ),
    )

    LLM_API_Calls.chat_with_openai(
        input_data=[{"role": "user", "content": "test"}],
        api_key=DUMMY_OPENAI_API_KEY,
        model=None,
        streaming=False,
        max_tokens=512,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a value",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert captured["url"] == "https://api.openai.test/v1/chat/completions"
    assert captured["json"]["model"] == "gpt-5.6-terra"
    assert captured["json"]["reasoning_effort"] == "none"
    assert captured["json"]["max_completion_tokens"] == 512
    assert "max_tokens" not in captured["json"]
    assert "max_output_tokens" not in captured["json"]
    assert captured["json"]["tools"][0]["type"] == "function"


def test_openai_gpt_5_6_explicit_none_stays_on_chat_completions(self, monkeypatch):
    # Arrange the same captured session, call with reasoning_effort="none",
    # and assert /chat/completions plus reasoning_effort == "none".


def test_openai_gpt_5_6_explicit_reasoning_uses_responses_contract(self, monkeypatch):
    # Arrange a Responses-shaped result, call with reasoning_effort="high"
    # and max_tokens=512, then assert /responses, max_output_tokens == 512,
    # no max_completion_tokens, and reasoning == {"effort": "high"}.


@pytest.mark.parametrize(
    ("control_name", "control_value"),
    [("reasoning_summary", "auto"), ("verbosity", "medium")],
)
def test_openai_none_with_responses_only_control_uses_responses(
    self, monkeypatch, control_name, control_value
):
    # Arrange a Responses-shaped result, call gpt-5.6-terra with
    # reasoning_effort="none" and **{control_name: control_value}, then assert
    # /responses and max_output_tokens. This proves "none" stays on Chat only
    # when no Responses-only control is also present.
```

Retain the existing `o3` Responses and streaming tests as regression coverage.
Calling the default-contract test with `model=None` also verifies the OpenAI
handler fallback.

- [x] **Step 2: Run the new tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest Tests/Chat/test_chat_functions.py -k "gpt_5_6 or openai_reasoning" -x -q
```

Expected: FAIL because ordinary GPT-5.6 requests currently use `max_tokens`,
omit `reasoning_effort`, and explicit `"none"` selects Responses.

- [x] **Step 3: Implement narrowly scoped OpenAI model routing**

In `LLM_API_Calls.py`:

- Add a small predicate that matches OpenAI GPT-5.6 family IDs only (for example,
  `model_name == "gpt-5.6"` or `model_name.startswith("gpt-5.6-")`).
- Normalize the reasoning effort once.
- Change `_openai_use_responses_api()` so `reasoning_effort="none"` alone does
  not select Responses, while a non-`none` effort, `reasoning_summary`, or
  `verbosity` still does.
- On a GPT-5.6 Chat Completions request:
  - send top-level `reasoning_effort` using the explicit value or `"none"`;
  - send `max_completion_tokens`;
  - preserve existing messages, tools, tool choice, streaming, and normalized
    output behavior.
- On a Responses request, continue sending `max_output_tokens` and the existing
  `reasoning` / `text` objects.
- For older OpenAI models, retain the existing `max_tokens` Chat Completions
  contract.
- Change the handler fallback used when the OpenAI config has no model to
  `gpt-5.6-terra`.

- [x] **Step 4: Run focused OpenAI tests**

Run:

```bash
.venv/bin/python -m pytest Tests/Chat/test_chat_functions.py -k "openai" -q
```

Expected: PASS, including the existing Responses normalization tests.

- [x] **Step 5: Commit**

```bash
git add tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_chat_functions.py
git commit -m "fix: shape GPT-5.6 requests compatibly"
```

---

### Task 3: Apply the Claude Sonnet 5 thinking and sampling contract

**Files:**

- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py`
- Modify: `Tests/Chat/test_chat_functions.py`
- Test: `Tests/Chat/test_chat_functions.py`

- [x] **Step 1: Add failing Sonnet 5 and adaptive-effort tests**

Add four captured-payload tests:

```python
def test_anthropic_sonnet_5_default_omits_thinking_effort_and_sampling(
    self, monkeypatch
):
    # Omit the model argument so the test also verifies the handler fallback to
    # claude-sonnet-5. The mocked config may contain temperature/top_p/top_k
    # defaults to prove they are suppressed.
    assert captured["json"]["model"] == "claude-sonnet-5"
    assert "thinking" not in captured["json"]
    assert "output_config" not in captured["json"]
    assert "temperature" not in captured["json"]
    assert "top_p" not in captured["json"]
    assert "top_k" not in captured["json"]


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_anthropic_sonnet_5_effort_uses_output_config(
    self, monkeypatch, effort
):
    # Call claude-sonnet-5 with thinking_effort=effort.
    assert captured["json"]["output_config"] == {"effort": effort}
    assert "thinking" not in captured["json"]


def test_anthropic_sonnet_5_off_disables_thinking(self, monkeypatch):
    # Call claude-sonnet-5 with thinking_effort="off".
    assert captured["json"]["thinking"] == {"type": "disabled"}
    assert "output_config" not in captured["json"]


def test_anthropic_adaptive_model_effort_uses_output_config(self, monkeypatch):
    # Call an already-recognized adaptive model such as claude-opus-4-8.
    assert captured["json"]["thinking"] == {"type": "adaptive"}
    assert captured["json"]["output_config"] == {"effort": "high"}
    assert "effort" not in captured["json"]["thinking"]
```

In the Sonnet 5 effort case, also pass `thinking_budget_tokens=4096` and assert
that no `budget_tokens` field is emitted. In every Sonnet 5 test, assert sampling
fields are absent.

- [x] **Step 2: Run the new tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest Tests/Chat/test_chat_functions.py -k "sonnet_5 or adaptive_model_effort" -x -q
```

Expected: FAIL because Sonnet 5 is not recognized and adaptive effort is
currently nested inside `thinking`.

- [x] **Step 3: Separate Anthropic thinking mode from output effort**

In `LLM_API_Calls.py`:

- Add an exact Claude Sonnet 5 family predicate.
- Refactor `_anthropic_thinking_config()` into a helper contract that returns
  three values: `thinking_config`, `output_config`, and `max_tokens`.
- For Sonnet 5:
  - no setting: return neither `thinking` nor `output_config`;
  - supported effort: return `output_config={"effort": effort}` only;
  - `"off"`: return `thinking={"type": "disabled"}` only;
  - ignore fixed budgets and log a warning when one was supplied.
- For existing adaptive models:
  - keep `thinking={"type": "adaptive"}`;
  - put explicit effort in `output_config={"effort": effort}`;
  - never nest effort in `thinking`.
- Preserve legacy fixed-budget behavior for older non-adaptive models.
- In `chat_with_anthropic()`, attach `thinking` and `output_config` separately.
- Suppress `temperature`, `top_p`, and `top_k` for Sonnet 5 regardless of
  generic/config defaults or explicit arguments.
- Preserve the existing sampling behavior for older models.
- Change the handler fallback used when Anthropic config has no model to
  `claude-sonnet-5`.

- [x] **Step 4: Update old adaptive assertions**

Change the two existing adaptive-model tests from:

```python
assert captured["json"]["thinking"] == {"type": "adaptive", "effort": "high"}
```

to the documented split:

```python
assert captured["json"]["thinking"] == {"type": "adaptive"}
assert captured["json"]["output_config"] == {"effort": "high"}
```

Keep the legacy Sonnet 4 fixed-budget tests unchanged.

- [x] **Step 5: Run focused Anthropic tests**

Run:

```bash
.venv/bin/python -m pytest Tests/Chat/test_chat_functions.py -k "anthropic" -q
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_chat_functions.py
git commit -m "fix: shape Claude Sonnet 5 requests compatibly"
```

---

### Task 4: Refresh the DeepSeek handler fallback without changing its contract

**Files:**

- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py`
- Modify: `Tests/Chat/test_chat_functions.py`
- Test: `Tests/Chat/test_chat_functions.py`

- [x] **Step 1: Add a failing DeepSeek fallback request test**

Use `_CapturedSession`, monkeypatch the module-level `settings` to contain a
DeepSeek API key/base URL but no model, call `chat_with_deepseek(model=None)`,
and assert:

```python
assert captured["url"] == "https://api.deepseek.test/chat/completions"
assert captured["json"]["model"] == "deepseek-v4-flash"
assert captured["json"]["messages"] == [{"role": "user", "content": "test"}]
assert captured["json"]["max_tokens"] == 128
```

This test verifies that only the model choice changes; the existing endpoint and
Chat Completions payload stay intact.

- [x] **Step 2: Run the test and verify it fails**

Run:

```bash
.venv/bin/python -m pytest Tests/Chat/test_chat_functions.py -k "deepseek" -x -q
```

Expected: FAIL because the handler fallback remains `deepseek-chat`.

- [x] **Step 3: Change the DeepSeek handler fallback**

In `chat_with_deepseek()`, replace only its fallback model ID with
`deepseek-v4-flash`. Do not add new reasoning or endpoint behavior.

- [x] **Step 4: Run focused DeepSeek tests**

Run:

```bash
.venv/bin/python -m pytest Tests/Chat/test_chat_functions.py -k "deepseek" -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_chat_functions.py
git commit -m "fix: refresh DeepSeek handler fallback"
```

---

### Task 5: Recognize the new defaults as vision-capable

**Files:**

- Modify: `tldw_chatbook/config.py`
- Modify: `tldw_chatbook/model_capabilities.py`
- Modify: `Tests/test_config_model_catalog_defaults.py`
- Modify: `Tests/test_model_capabilities.py`
- Test: `Tests/test_config_model_catalog_defaults.py`
- Test: `Tests/test_model_capabilities.py`

- [x] **Step 1: Add failing capability tests**

In `Tests/test_model_capabilities.py`, extend `TestDefaultModels`:

```python
assert (
    model_capabilities_empty.is_vision_capable("OpenAI", "gpt-5.6-terra")
    is True
)
assert (
    model_capabilities_empty.is_vision_capable("Anthropic", "claude-sonnet-5")
    is True
)
```

In `Tests/test_config_model_catalog_defaults.py`, assert the embedded TOML direct
capability mappings:

```python
models = parsed["model_capabilities"]["models"]
assert models["gpt-5.6-terra"]["vision"] is True
assert models["claude-sonnet-5"]["vision"] is True
```

- [x] **Step 2: Run the capability tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest Tests/test_config_model_catalog_defaults.py Tests/test_model_capabilities.py -x -q
```

Expected: FAIL because neither new default is in the current direct maps or
patterns.

- [x] **Step 3: Add matching capability metadata**

In both the embedded `[model_capabilities.models]` table in `config.py` and
`DEFAULT_MODEL_CAPABILITIES` in `model_capabilities.py`, add:

```python
"gpt-5.6-terra": {"vision": True, "max_images": 10}
"claude-sonnet-5": {"vision": True, "max_images": 5}
```

Use TOML inline-table syntax in the embedded config. Direct mappings are
sufficient for the selected defaults; do not broaden family patterns or add
DeepSeek V4 vision metadata without separate vendor evidence.

- [x] **Step 4: Run capability and configuration tests**

Run:

```bash
.venv/bin/python -m pytest Tests/test_config_model_catalog_defaults.py Tests/test_model_capabilities.py -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add tldw_chatbook/config.py tldw_chatbook/model_capabilities.py Tests/test_config_model_catalog_defaults.py Tests/test_model_capabilities.py
git commit -m "config: recognize new default model capabilities"
```

---

### Task 6: Verify, document, and close TASK-519

**Files:**

- Modify: `backlog/tasks/task-519 - Refresh-default-models-for-DeepSeek-Anthropic-and-OpenAI.md` (via Backlog CLI)
- Verify: all files changed in Tasks 1–5

- [x] **Step 1: Run the complete targeted suite**

```bash
.venv/bin/python -m pytest Tests/test_config_model_catalog_defaults.py Tests/Chat/test_chat_functions.py Tests/test_model_capabilities.py -q
```

Expected: all tests pass.

- [x] **Step 2: Run lint, format, and static checks for changed Python files**

```bash
.venv/bin/ruff check --select E9,F63,F7,F82 tldw_chatbook/config.py tldw_chatbook/LLM_Calls/LLM_API_Calls.py tldw_chatbook/model_capabilities.py Tests/test_config_model_catalog_defaults.py Tests/Chat/test_chat_functions.py Tests/test_model_capabilities.py
.venv/bin/ruff format --check tldw_chatbook/LLM_Calls/LLM_API_Calls.py tldw_chatbook/model_capabilities.py Tests/test_config_model_catalog_defaults.py Tests/Chat/test_chat_functions.py Tests/test_model_capabilities.py
.venv/bin/python -m compileall -q tldw_chatbook/config.py tldw_chatbook/LLM_Calls/LLM_API_Calls.py tldw_chatbook/model_capabilities.py
git diff --check
```

Expected: all commands exit 0. `config.py` is included in Ruff's fatal
syntax/undefined-name lint set and `git diff --check`, but excluded from the
whole-file format check because the unchanged baseline file is not Ruff-formatted
and would require a large out-of-scope rewrite. The other five touched Python
files pass Ruff's formatter check.

- [x] **Step 3: Audit the diff against the approved scope**

Run:

```bash
git diff --stat 685a8e0d4...HEAD
git diff 685a8e0d4...HEAD -- tldw_chatbook/config.py tldw_chatbook/LLM_Calls/LLM_API_Calls.py tldw_chatbook/model_capabilities.py Tests/test_config_model_catalog_defaults.py Tests/Chat/test_chat_functions.py Tests/test_model_capabilities.py
```

Confirm:

- no existing user config migration was added;
- character, analysis, TTS, embedding, RAG, and historical fixture defaults are
  unchanged;
- retired DeepSeek aliases were removed only from active bundled catalogs and
  fallback defaults;
- Responses API normalization and older-model payload behavior remain covered.

- [x] **Step 4: Update TASK-519 through Backlog CLI**

Add concise implementation notes that list the changed files, model-aware
compatibility rules, ADR-020/design links, and exact verification results. Mark
all seven acceptance criteria complete only after the checks above pass, then
move the task to Done:

```bash
backlog task edit 519 --notes "<implementation summary and verification>"
backlog task edit 519 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --check-ac 6 --check-ac 7
backlog task edit 519 -s Done
```

- [x] **Step 5: Commit task closeout**

```bash
git add "backlog/tasks/task-519 - Refresh-default-models-for-DeepSeek-Anthropic-and-OpenAI.md"
git commit -m "docs: close provider default model refresh task"
```

- [x] **Step 6: Run final verification from the committed tree**

```bash
.venv/bin/python -m pytest Tests/test_config_model_catalog_defaults.py Tests/Chat/test_chat_functions.py Tests/test_model_capabilities.py -q
.venv/bin/ruff check --select E9,F63,F7,F82 tldw_chatbook/config.py tldw_chatbook/LLM_Calls/LLM_API_Calls.py tldw_chatbook/model_capabilities.py Tests/test_config_model_catalog_defaults.py Tests/Chat/test_chat_functions.py Tests/test_model_capabilities.py
.venv/bin/ruff format --check tldw_chatbook/LLM_Calls/LLM_API_Calls.py tldw_chatbook/model_capabilities.py Tests/test_config_model_catalog_defaults.py Tests/Chat/test_chat_functions.py Tests/test_model_capabilities.py
.venv/bin/python -m compileall -q tldw_chatbook/config.py tldw_chatbook/LLM_Calls/LLM_API_Calls.py tldw_chatbook/model_capabilities.py
git diff --check
git status --short
```

Expected: tests and static checks pass; `git status --short` is empty.

Verification scope note: the approved focused suite passed 89 tests. Fatal Ruff
rules passed on all six touched files; the formatter passed on the five
formatter-clean files, while `config.py` retains its pre-existing formatting
baseline. Broad default Ruff remains baseline-failing (452 existing findings).
A broader serial suite attempt was manually interrupted after 448 passed, 1
skipped, and 0 observed failures at 3% in 140.83s because its projected runtime
was impractical; xdist is unsuitable because repository tests share state. This
is not a full-suite pass. The approved scoped exception keeps TASK-519 Done
because its acceptance criteria require the focused suite.
