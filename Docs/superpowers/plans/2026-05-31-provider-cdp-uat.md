# Provider CDP UAT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a rendered-app CDP/Textual-web UAT sweep proving Chatbook Console can complete a two-turn conversation with every testable provider in the current provider execution surface.

**Architecture:** Keep the acceptance surface as the running Textual-web app. Add QA-only helpers for redacted provider inventory, isolated launch, CDP evidence capture, and report generation; make product-code fixes only when CDP evidence identifies a Chatbook defect. The provider inventory is extracted at runtime from `Chat_Functions.API_CALL_HANDLERS` and Console identity helpers so the sweep does not drift from the codebase.

**Tech Stack:** Python 3.11+, Textual/textual-serve, Playwright/Chromium over CDP, pytest, Backlog.md, existing Console provider gateway/readiness modules.

---

## Source Documents

- Spec: `Docs/superpowers/specs/2026-05-31-provider-cdp-uat-design.md`
- CDP runbook: `Docs/superpowers/qa/product-maturity/screen-qa/textual-web-cdp-debugging.md`
- Existing Console CDP example: `Docs/superpowers/qa/product-maturity/console-native-chat-core/capture_console_native_chat_core.mjs`
- Provider execution source: `tldw_chatbook/Chat/Chat_Functions.py`
- Provider identity source: `tldw_chatbook/Chat/console_provider_support.py`
- Provider readiness source: `tldw_chatbook/Chat/provider_readiness.py`

## Planning Requirements From Pre-Implementation Review

- The QA table must include both display/readiness key and execution key, because aliases such as `custom` and `custom-openai-api` differ.
- Runtime provider inventory extraction is authoritative. Static provider lists in the design are context only.
- Runtime provider inventory must include the model selected for UAT and the model source before any provider is attempted.
- Runtime provider inventory must include configured endpoint source and reachability for local/custom providers before CDP testing.
- External provider outcomes must be classified consistently:
  - `skip`: missing key, placeholder key, empty key, or unreachable local/custom endpoint.
  - `fail_external`: configured key exists but provider rejects auth, quota, billing, rate limit, or model availability.
  - `fail_chatbook`: request shape, response normalization, streaming fallback, or Console UI failure caused by Chatbook behavior.
- Do not save raw keys through UI or keyring. Save or confirm env var names only.
- Pass evidence must include rendered app evidence plus a redacted log/run-state signal that the selected provider execution path completed.
- Launch and inventory commands must isolate `HOME`, `XDG_CONFIG_HOME`, and `XDG_DATA_HOME` from the user's real profile.
- PNG screenshots must be visually inspected before commit because text secret scans cannot inspect image pixels.
- Before product code changes, create or identify a Backlog task and keep task hygiene current.

## File Structure

Create QA-only files:

- Create: `Docs/superpowers/qa/provider-cdp-uat/README.md`
  - Responsibility: Explain scope, safety rules, and how evidence is organized.
- Create: `Docs/superpowers/qa/provider-cdp-uat/provider_inventory.py`
  - Responsibility: Parse the adjacent `.env` safely, extract provider identities from the codebase, resolve model and endpoint sources, probe local/custom endpoint reachability, classify in-scope/skipped providers, and write redacted JSON/Markdown inventory.
- Create: `Docs/superpowers/qa/provider-cdp-uat/run_textual_web_with_env.py`
  - Responsibility: Launch `tldw-serve` with isolated `HOME`/XDG config/data dirs and `.env` values loaded into process environment without printing secrets.
- Create: `Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs`
  - Responsibility: Attach to Textual-web, disable textual-web overlays, provide screenshot/log-wait primitives, and assist the per-provider manual CDP flow.
- Create: `Docs/superpowers/qa/provider-cdp-uat/2026-05-31-provider-cdp-uat.md`
  - Responsibility: Durable QA report with provider table, evidence paths, failures, fixes, and residual risk.

Create tests for QA helpers:

- Create: `Tests/QA/test_provider_cdp_uat_helpers.py`
  - Responsibility: Verify `.env` parsing, placeholder filtering, secret redaction, provider alias inventory, model source selection, endpoint source extraction, launch isolation, and external outcome classification.

Conditional product files for Chatbook fixes:

- Modify only when CDP evidence shows a Chatbook defect:
  - `tldw_chatbook/Chat/console_provider_gateway.py`
  - `tldw_chatbook/Chat/console_provider_support.py`
  - `tldw_chatbook/Chat/provider_readiness.py`
  - `tldw_chatbook/Chat/Chat_Functions.py`
  - `tldw_chatbook/UI/Screens/chat_screen.py`
  - focused tests under `Tests/Chat/` or `Tests/UI/`

Backlog files:

- Modify: one new or existing task under `backlog/tasks/`
  - Responsibility: Track the UAT run, implementation plan, acceptance criteria, implementation notes, and Done hygiene.

## Task 1: Create Backlog Task And Start UAT Tracking

**Files:**
- Modify/Create: `backlog/tasks/task-<id> - Provider-CDP-UAT-sweep.md`

- [ ] **Step 1: Search for an existing task**

Run: `backlog task search "Provider CDP UAT" --plain`

Expected: Either an existing provider CDP UAT task is found, or no matching task is found.

- [ ] **Step 2: Create the task if needed**

Run:

```bash
backlog task create "Provider CDP UAT sweep" \
  -s "In Progress" \
  -l qa,providers,console,cdp \
  -d "Verify every testable Chatbook Console provider through rendered Textual-web/CDP using isolated config and redacted provider credentials." \
  --ac "Runtime provider inventory is extracted from Chatbook code,Inventory records model source and local endpoint reachability,Hosted providers with usable keys are tested through CDP,Local/custom providers are skipped unless endpoint is reachable,Each passed provider receives a second assistant reply in the same Console session,External failures are classified separately from Chatbook defects,Raw API keys do not appear in evidence,QA report and residual risks are recorded"
```

Expected: Backlog creates a task file under `backlog/tasks/` with status `In Progress`.

- [ ] **Step 3: Add the implementation plan to the task**

Run:

```bash
backlog task edit <task-id> --plan "1. Build redacted provider inventory and isolated CDP launch helpers.
2. Generate provider inventory and QA report skeleton.
3. Launch Chatbook through Textual-web/CDP with isolated HOME/XDG config/data.
4. Run a manual two-turn provider sweep through the rendered app.
5. Fix and rerun only Chatbook-caused provider defects.
6. Record evidence, residual risks, and task closeout."
```

Expected: The task has an `Implementation Plan` section before code changes begin.

- [ ] **Step 4: Commit the task start**

Run:

```bash
git add backlog/tasks
git commit -m "Track provider CDP UAT sweep"
```

Expected: Commit contains only the Backlog task change.

## Task 2: Add Redacted Inventory Helper Tests

**Files:**
- Create: `Tests/QA/test_provider_cdp_uat_helpers.py`
- Create: `Docs/superpowers/qa/provider-cdp-uat/provider_inventory.py`

- [ ] **Step 1: Write failing tests for env parsing and redaction**

Create `Tests/QA/test_provider_cdp_uat_helpers.py` with tests covering:

```python
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = ROOT / "Docs/superpowers/qa/provider-cdp-uat/provider_inventory.py"


def load_inventory_module():
    spec = importlib.util.spec_from_file_location("provider_inventory", INVENTORY_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_env_values_expands_simple_references_without_shelling_out(tmp_path: Path) -> None:
    inventory = load_inventory_module()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=sk-live\n"
        "CUSTOM_OPENAI_API_KEY=${OPENAI_API_KEY}\n"
        "PLACEHOLDER=<api_key>\n",
        encoding="utf-8",
    )

    values = inventory.load_env_values(env_file)

    assert values["OPENAI_API_KEY"] == "sk-live"
    assert values["CUSTOM_OPENAI_API_KEY"] == "sk-live"
    assert values["PLACEHOLDER"] == "<api_key>"


def test_should_use_key_value_rejects_empty_and_placeholder_values() -> None:
    inventory = load_inventory_module()

    assert inventory.should_use_key_value("sk-real") is True
    assert inventory.should_use_key_value("") is False
    assert inventory.should_use_key_value("<api_key>") is False
    assert inventory.should_use_key_value("${OPENAI_API_KEY}") is False


def test_mask_secret_never_returns_raw_secret() -> None:
    inventory = load_inventory_module()

    assert inventory.mask_secret("sk-abcdef123456") == "sk-a...3456"
    assert "abcdef" not in inventory.mask_secret("sk-abcdef123456")


def test_classify_external_outcome_is_stable() -> None:
    inventory = load_inventory_module()

    assert inventory.classify_external_outcome("missing_key") == "skip"
    assert inventory.classify_external_outcome("endpoint_unreachable") == "skip"
    assert inventory.classify_external_outcome("explicit_model_missing") == "skip"
    assert inventory.classify_external_outcome("auth") == "fail_external"
    assert inventory.classify_external_outcome("quota_or_rate_limit") == "fail_external"
    assert inventory.classify_external_outcome("request_shape") == "fail_chatbook"


def test_choose_uat_model_records_low_cost_override_source() -> None:
    inventory = load_inventory_module()

    selected = inventory.choose_uat_model(
        provider_key="openai",
        configured_models=["gpt-4.1"],
        provider_config={},
    )

    assert selected.model == "gpt-4o-mini-2024-07-18"
    assert selected.source == "override:openai"


def test_extract_endpoint_config_records_source_key() -> None:
    inventory = load_inventory_module()

    endpoint = inventory.extract_endpoint_config({"api_base": "http://127.0.0.1:8000"})

    assert endpoint.value == "http://127.0.0.1:8000"
    assert endpoint.source == "api_base"


def test_choose_uat_model_can_use_server_default_for_reachable_local_provider() -> None:
    inventory = load_inventory_module()

    selected = inventory.choose_uat_model(
        provider_key="custom-openai-api",
        configured_models=[],
        provider_config={},
        allow_server_default=True,
    )

    assert selected.model == ""
    assert selected.source == "server_default"
    assert selected.requires_explicit_selection is False
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `.venv/bin/python -m pytest -q Tests/QA/test_provider_cdp_uat_helpers.py --tb=short`

Expected: FAIL because helper module/functions do not exist.

## Task 3: Implement Provider Inventory And Classification Helper

**Files:**
- Create/Modify: `Docs/superpowers/qa/provider-cdp-uat/provider_inventory.py`
- Modify: `Tests/QA/test_provider_cdp_uat_helpers.py` if import path needs adjustment

- [ ] **Step 1: Implement `.env` parsing without shell sourcing**

Implement `load_env_values(path: Path) -> dict[str, str]`:

- Ignore blank lines and comments.
- Split only on the first `=`.
- Strip matching single or double quotes.
- Expand `${NAME}` references from values already parsed or from `os.environ`.
- Do not execute shell syntax.
- Preserve raw values only in memory.

- [ ] **Step 2: Implement key filtering and masking**

Implement:

```python
PLACEHOLDER_PREFIXES = ("<", "your_", "YOUR_")
PLACEHOLDER_VALUES = {"", "<API_KEY_HERE>", "CHANGE_ME_TO_SECURE_RANDOM_KEY_MIN_32_CHARS"}


def should_use_key_value(value: object) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if stripped in PLACEHOLDER_VALUES:
        return False
    if stripped.startswith("${") and stripped.endswith("}"):
        return False
    if stripped.startswith(PLACEHOLDER_PREFIXES):
        return False
    return bool(stripped)


def mask_secret(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"
```

- [ ] **Step 3: Implement provider inventory extraction**

The helper should import:

```python
from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS
from tldw_chatbook.Chat.console_provider_support import resolve_console_provider_identity
from tldw_chatbook.Chat.provider_readiness import PROVIDERS_REQUIRING_API_KEY_KEYS
```

For each handler key:

- Resolve `display_key`, `readiness_key`, and `execution_key`.
- Determine whether the provider requires a key.
- Map known env var names from config conventions:
  - `openai` -> `OPENAI_API_KEY`
  - `anthropic` -> `ANTHROPIC_API_KEY`
  - `cohere` -> `COHERE_API_KEY`
  - `deepseek` -> `DEEPSEEK_API_KEY`
  - `google` -> `GOOGLE_API_KEY`
  - `groq` -> `GROQ_API_KEY`
  - `huggingface` -> `HUGGINGFACE_API_KEY`
  - `mistral` and `mistralai` -> `MISTRAL_API_KEY`
  - `moonshot` -> `MOONSHOT_API_KEY`
  - `openrouter` -> `OPENROUTER_API_KEY`
  - `zai` -> `ZAI_API_KEY`, then `QWEN_API_KEY` only if the code/config supports it after inspection.
- Keyed QA row must include both `readiness_key` and `execution_key`.

- [ ] **Step 4: Resolve UAT model and model source**

Add model selection to each inventory row before CDP testing. The helper must record both `model` and `model_source`.

Model source precedence:

1. Explicit low-cost override for known hosted providers.
2. First configured model from `get_cli_providers_and_models()` or the current Console model list.
3. Provider config keys such as `[api_settings.<provider>].model`, `api_model`, or `default_model`.
4. Handler default documented in the provider implementation, such as `zai` using `glm-4.5-flash` when no model is supplied.
5. `server_default` for reachable local/custom endpoints whose Chatbook handler can omit or send a default model.
6. `explicit_model_missing` only when the provider requires an explicit model and no override, configured model, handler default, or server default applies.

Initial low-cost override table:

```python
LOW_COST_MODEL_OVERRIDES = {
    "openai": "gpt-4o-mini-2024-07-18",
    "anthropic": "claude-3-5-haiku-20241022",
    "cohere": "command-r-08-2024",
    "deepseek": "deepseek-chat",
    "google": "gemini-2.0-flash-lite",
    "groq": "llama-3.1-8b-instant",
    "mistral": "open-mistral-nemo",
    "mistralai": "open-mistral-nemo",
    "moonshot": "kimi-latest",
    "openrouter": "openai/gpt-4o-mini",
    "zai": "glm-4.5-flash",
}
```

Each model result should include whether the rendered UI must explicitly select/enter the model. `server_default` means the provider remains testable; the CDP flow should leave the model at the app/server default and record that source instead of skipping.

If a provider returns `model_unavailable`, classify that attempt as `fail_external`, update the QA row, and rerun only after updating the inventory to a new explicit model source. Do not silently switch models during the CDP sweep.

- [ ] **Step 5: Resolve local/custom endpoint source and reachability**

For local/custom providers, record:

- `endpoint`;
- `endpoint_source` such as `api_url`, `base_url`, `api_base`, `api_endpoint`, `endpoint`, or `config_missing`;
- `endpoint_reachable`;
- `endpoint_probe_url`;
- `endpoint_probe_status`.

Probe rules:

- Do not probe hosted provider endpoints.
- Normalize OpenAI-compatible local/custom base URLs and probe `/v1/models`.
- For `koboldcpp`, probe `/api/v1/model` when the configured endpoint points at `/api/v1/generate`; otherwise probe the configured root.
- Use a short timeout, no more than 3 seconds per endpoint.
- If no endpoint is configured or the endpoint is unreachable, initial status is `skip` with reason `endpoint_unreachable`.

- [ ] **Step 6: Implement external outcome classification**

Implement:

```python
def classify_external_outcome(reason: str) -> str:
    if reason in {"missing_key", "endpoint_unreachable", "explicit_model_missing"}:
        return "skip"
    if reason in {"auth", "quota_or_rate_limit", "model_unavailable"}:
        return "fail_external"
    if reason in {"request_shape", "response_shape", "streaming", "console_ui"}:
        return "fail_chatbook"
    return "unknown"
```

- [ ] **Step 7: Add CLI output**

The helper CLI should support:

```bash
QA_ROOT="${TMPDIR:-/tmp}/tldw-chatbook-provider-cdp-uat"
HOME="$QA_ROOT/home" \
XDG_CONFIG_HOME="$QA_ROOT/config" \
XDG_DATA_HOME="$QA_ROOT/data" \
.venv/bin/python Docs/superpowers/qa/provider-cdp-uat/provider_inventory.py \
  --env-file ../tldw_server2/tldw_Server_API/Config_Files/.env \
  --json Docs/superpowers/qa/provider-cdp-uat/provider-inventory.json \
  --markdown Docs/superpowers/qa/provider-cdp-uat/provider-inventory.md
```

Expected JSON/Markdown contains provider names, display/readiness/execution keys, model, model source, key source labels, masked key status, endpoint source/reachability for local/custom providers, and initial status. It must not contain raw key values.

- [ ] **Step 8: Run helper tests**

Run: `.venv/bin/python -m pytest -q Tests/QA/test_provider_cdp_uat_helpers.py --tb=short`

Expected: PASS.

- [ ] **Step 9: Commit helper and tests**

Run:

```bash
git add Docs/superpowers/qa/provider-cdp-uat/provider_inventory.py Tests/QA/test_provider_cdp_uat_helpers.py
git commit -m "Add provider CDP UAT inventory helper"
```

Expected: Commit contains only QA helper and tests.

## Task 4: Add Isolated Textual-Web Launch Helper

**Files:**
- Create: `Docs/superpowers/qa/provider-cdp-uat/run_textual_web_with_env.py`
- Modify: `Tests/QA/test_provider_cdp_uat_helpers.py`

- [ ] **Step 1: Add tests for launch environment construction**

Extend `Tests/QA/test_provider_cdp_uat_helpers.py` with tests for a pure function such as `build_launch_environment(...)`:

```python
LAUNCH_PATH = ROOT / "Docs/superpowers/qa/provider-cdp-uat/run_textual_web_with_env.py"


def load_launch_module():
    spec = importlib.util.spec_from_file_location("run_textual_web_with_env", LAUNCH_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_launch_environment_sets_isolated_home_and_redacted_env(tmp_path: Path) -> None:
    launch = load_launch_module()
    env_values = {"OPENAI_API_KEY": "sk-real", "PLACEHOLDER": "<api_key>"}

    launch_env = launch.build_launch_environment(
        worktree=tmp_path,
        qa_root=tmp_path / "qa-root",
        env_values=env_values,
    )

    assert launch_env["HOME"] == str(tmp_path / "qa-root" / "home")
    assert launch_env["XDG_CONFIG_HOME"] == str(tmp_path / "qa-root" / "config")
    assert launch_env["XDG_DATA_HOME"] == str(tmp_path / "qa-root" / "data")
    assert launch_env["PYTHONPATH"] == str(tmp_path)
    assert launch_env["OPENAI_API_KEY"] == "sk-real"
    assert "PLACEHOLDER" not in launch_env
```

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv/bin/python -m pytest -q Tests/QA/test_provider_cdp_uat_helpers.py --tb=short`

Expected: FAIL because launch helper does not exist.

- [ ] **Step 3: Implement launch helper**

The helper should:

- Load `.env` with `load_env_values()`.
- Filter placeholder/empty values with `should_use_key_value()`.
- Set:
  - `PYTHONPATH=<repo-root>`
  - `HOME=<qa-root>/home`
  - `XDG_CONFIG_HOME=<qa-root>/config`
  - `XDG_DATA_HOME=<qa-root>/data`
  - `TLDW_TEXTUAL_WEB_PORT=<port>`
- Create both `$HOME/.config/tldw_cli/config.toml` and `$XDG_CONFIG_HOME/tldw_cli/config.toml` before launch. This covers the current `config.py` HOME-based default while still honoring the plan requirement to isolate XDG config.
- Write a minimal isolated config:

```toml
[general]
default_tab = "chat"

[splash_screen]
enabled = false

[console]
collapse_large_pastes = true
paste_collapse_threshold = 50
```

- Execute `.venv/bin/tldw-serve --host 127.0.0.1 --port <port>`.
- Treat this as a long-running server process. Keep it open while CDP commands run from a second shell/session.
- Print only launch paths, port, and masked key-source summary.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest -q Tests/QA/test_provider_cdp_uat_helpers.py --tb=short`

Expected: PASS.

- [ ] **Step 5: Commit launch helper**

Run:

```bash
git add Docs/superpowers/qa/provider-cdp-uat/run_textual_web_with_env.py Tests/QA/test_provider_cdp_uat_helpers.py
git commit -m "Add isolated provider CDP launch helper"
```

Expected: Commit contains launch helper and tests only.

## Task 5: Add CDP Probe Helper And QA Report Template

**Files:**
- Create: `Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs`
- Create: `Docs/superpowers/qa/provider-cdp-uat/README.md`
- Create: `Docs/superpowers/qa/provider-cdp-uat/2026-05-31-provider-cdp-uat.md`

- [ ] **Step 1: Create README**

Include:

- Scope: manual CDP/Textual-web provider UAT.
- Safety: no raw keys in screenshots/logs/commits.
- Source of truth: runtime inventory JSON.
- Skip/fail classification rules.
- Evidence folder conventions.

- [ ] **Step 2: Create QA report template**

Create `2026-05-31-provider-cdp-uat.md` with sections:

```markdown
# Provider CDP UAT

Date: 2026-05-31
Branch:
Spec:
Backlog task:
Textual-web URL:
Isolated HOME:
Isolated XDG config:
Isolated data:
App log:

## Provider Inventory

| Display | Readiness key | Execution key | Model | Model source | Key source | Endpoint source/status | Status | Classification | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Run Notes

## Fixes And Reruns

## Residual Risks
```

- [ ] **Step 3: Create CDP helper**

Use the existing Console CDP script as the model. The helper must:

- Load Playwright from the bundled Node dependency path used by existing scripts.
- Use `process.env.TLDW_TEXTUAL_WEB_PORT`.
- Use `process.env.TLDW_QA_APP_LOG`.
- Disable `.intro-dialog`, `.closed-dialog`, and `.shade` pointer events.
- Provide commands for:
  - `screenshot(name)`
  - `focusTerminal()`
  - `typeText(text)`
  - `press(key)`
  - `readLogTail(offset)`
  - `waitForLog(pattern, timeout)`
- Avoid raw key output.

- [ ] **Step 4: Smoke-run CDP helper against no server**

Run: `node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs --help`

Expected: Prints usage/help without requiring a running server and without printing environment values.

- [ ] **Step 5: Commit CDP helper and templates**

Run:

```bash
git add Docs/superpowers/qa/provider-cdp-uat/README.md \
  Docs/superpowers/qa/provider-cdp-uat/2026-05-31-provider-cdp-uat.md \
  Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs
git commit -m "Add provider CDP UAT evidence harness"
```

Expected: Commit contains only QA evidence harness files.

## Task 6: Generate Runtime Provider Inventory

**Files:**
- Create: `Docs/superpowers/qa/provider-cdp-uat/provider-inventory.json`
- Create: `Docs/superpowers/qa/provider-cdp-uat/provider-inventory.md`
- Modify: `Docs/superpowers/qa/provider-cdp-uat/2026-05-31-provider-cdp-uat.md`

- [ ] **Step 1: Run provider inventory**

Run:

```bash
QA_ROOT="${TMPDIR:-/tmp}/tldw-chatbook-provider-cdp-uat"
HOME="$QA_ROOT/home" \
XDG_CONFIG_HOME="$QA_ROOT/config" \
XDG_DATA_HOME="$QA_ROOT/data" \
.venv/bin/python Docs/superpowers/qa/provider-cdp-uat/provider_inventory.py \
  --env-file ../tldw_server2/tldw_Server_API/Config_Files/.env \
  --json Docs/superpowers/qa/provider-cdp-uat/provider-inventory.json \
  --markdown Docs/superpowers/qa/provider-cdp-uat/provider-inventory.md
```

Expected: JSON and Markdown are generated. Hosted providers with non-placeholder keys and resolved models are marked initially testable. Local/custom providers include endpoint source and reachability before they are marked testable or skipped. `missing_key`, `explicit_model_missing`, and `endpoint_unreachable` providers are classified before CDP testing; reachable local/custom providers with `server_default` remain testable.

- [ ] **Step 2: Secret scan generated evidence**

Run:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
root = Path("Docs/superpowers/qa/provider-cdp-uat")
text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in root.glob("provider-inventory.*"))
for forbidden in ("sk-proj-", "sk-ant-", "AIza", "gsk_", "hf_", "sk-or-"):
    assert forbidden not in text, forbidden
print("redacted")
PY
```

Expected: Prints `redacted`.

- [ ] **Step 3: Update QA report inventory table**

Copy the redacted inventory rows into `2026-05-31-provider-cdp-uat.md`. Ensure each row includes display/readiness/execution keys, model, model source, key source, endpoint source/reachability where applicable, and initial classification.

- [ ] **Step 4: Commit inventory evidence**

Run:

```bash
git add Docs/superpowers/qa/provider-cdp-uat/provider-inventory.json \
  Docs/superpowers/qa/provider-cdp-uat/provider-inventory.md \
  Docs/superpowers/qa/provider-cdp-uat/2026-05-31-provider-cdp-uat.md
git commit -m "Record provider CDP UAT inventory"
```

Expected: Commit contains redacted inventory and report update only.

## Task 7: Launch Isolated Textual-Web App

**Files:**
- Generated outside repo: `${TMPDIR:-/tmp}/tldw-chatbook-provider-cdp-uat/...`
- Modify: `Docs/superpowers/qa/provider-cdp-uat/2026-05-31-provider-cdp-uat.md`

- [ ] **Step 1: Start server inside sandbox first**

Run:

```bash
.venv/bin/python Docs/superpowers/qa/provider-cdp-uat/run_textual_web_with_env.py \
  --env-file ../tldw_server2/tldw_Server_API/Config_Files/.env \
  --qa-root "${TMPDIR:-/tmp}/tldw-chatbook-provider-cdp-uat" \
  --port 8896
```

Expected: Server starts at `http://127.0.0.1:8896`, or fails with a sandbox/local-port permission error.
Keep the server process running while the remaining CDP steps run in another shell/session.

- [ ] **Step 2: If port binding fails, rerun with escalation**

Run the same command outside the sandbox with approval.

Expected: Server starts at `http://127.0.0.1:8896`.

- [ ] **Step 3: Record launch metadata**

Update the QA report with:

- exact command shape without raw env values;
- isolated HOME path;
- isolated XDG config path;
- isolated data path;
- app log path;
- Textual-web URL.

- [ ] **Step 4: Capture baseline screenshot**

Run:

```bash
QA_ROOT="${TMPDIR:-/tmp}/tldw-chatbook-provider-cdp-uat"
TLDW_TEXTUAL_WEB_PORT=8896 \
TLDW_QA_APP_LOG="$QA_ROOT/home/.local/share/tldw_cli/default_user/tldw_cli_app.log" \
node Docs/superpowers/qa/provider-cdp-uat/cdp_provider_probe.mjs screenshot baseline-console
```

Expected: Creates a PNG under `Docs/superpowers/qa/provider-cdp-uat/screenshots/` showing the Console/Chat screen, not a loader or splash.

- [ ] **Step 5: Verify screenshot file**

Run: `file Docs/superpowers/qa/provider-cdp-uat/screenshots/baseline-console.png`

Expected: Output contains `PNG image data`.

- [ ] **Step 6: Visually inspect baseline screenshot**

Open or view the screenshot and confirm:

- no raw API key is visible;
- the app is past loading/splash state;
- the target Console or Chat screen is usable.

Expected: Screenshot is safe to commit and useful as baseline evidence.

## Task 8: Run Manual CDP Provider Sweep

**Files:**
- Modify: `Docs/superpowers/qa/provider-cdp-uat/2026-05-31-provider-cdp-uat.md`
- Add screenshots under: `Docs/superpowers/qa/provider-cdp-uat/screenshots/`

- [ ] **Step 1: For each inventory row, decide attempt status**

Use `provider-inventory.json` as source of truth.

Rules:

- `missing_key` hosted provider: mark `skip`.
- `explicit_model_missing` provider with `requires_explicit_selection=true`: mark `skip` until inventory is updated with an explicit model source.
- local/custom provider with no configured endpoint or no reachable configured endpoint: mark `skip`.
- provider with usable key, resolved model or `server_default`, and reachable endpoint when applicable: attempt CDP UAT.

- [ ] **Step 2: Configure provider through rendered Settings**

For each attempted provider:

1. Open Settings in the rendered app.
2. Navigate to Providers and Models.
3. Select provider.
4. Confirm env var credential source, not raw key.
5. Select or enter the exact model recorded in `provider-inventory.json`, unless `model_source` is `server_default`.
6. Save if needed.
7. Return to Console.

Expected: Console/Settings shows provider and model with no raw key visible.
For `server_default`, leave the rendered app at its default/blank model state and record that the server or handler supplied the model.
If the server returns `model_unavailable`, record `fail_external`, update the inventory with a new explicit model source, and rerun that provider. Do not choose an ad hoc fallback in the rendered app without updating the inventory first.

- [ ] **Step 3: Configure active Console session**

Open Console settings in the rendered app and select the same provider/model. For `server_default`, select the provider and leave the model at the default/blank state.

Expected: Console settings summary shows selected provider/model.

- [ ] **Step 4: Send first turn**

Before sending, record app-log offset using the CDP helper.

Message:

```text
Reply with one short sentence: provider UAT turn one.
```

Expected: First assistant response appears and run completes.

- [ ] **Step 5: Send second turn**

Message:

```text
Reply with one short sentence: provider UAT turn two.
```

Expected: Second assistant response appears and run completes in the same session.

- [ ] **Step 6: Record provider execution evidence**

For each pass, record:

- screenshot path after second response;
- provider display/readiness/execution keys;
- model;
- model source;
- masked key source;
- endpoint source/reachability and probe result for local/custom providers;
- app log/run-state evidence after the pre-send offset proving selected provider path completed;
- status `pass`.

- [ ] **Step 7: Visually inspect provider screenshots**

Before committing screenshots, inspect each PNG and confirm:

- no raw key or secret-bearing settings value is visible;
- the selected provider/model and second assistant reply are visible enough for evidence;
- no unrelated private data is visible.

If a screenshot exposes a raw key, discard it, fix the capture flow, and recapture safe evidence.

- [ ] **Step 8: Classify failures immediately**

Use the classification rules:

- missing key, explicit model missing, or endpoint unreachable: `skip`;
- auth/quota/model issue from provider: `fail_external`;
- request/response/streaming/Console UI issue: `fail_chatbook`;
- insufficient evidence: `unknown`.

Expected: No provider remains with an unclassified failure.

## Task 9: Fix Chatbook-Caused Provider Defects

**Files:**
- Conditional modify: `tldw_chatbook/Chat/console_provider_gateway.py`
- Conditional modify: `tldw_chatbook/Chat/console_provider_support.py`
- Conditional modify: `tldw_chatbook/Chat/provider_readiness.py`
- Conditional modify: `tldw_chatbook/Chat/Chat_Functions.py`
- Conditional modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Conditional tests: `Tests/Chat/test_console_provider_gateway.py`, `Tests/Chat/test_console_provider_support.py`, `Tests/Chat/test_provider_readiness.py`, `Tests/Chat/test_chat_functions.py`, `Tests/UI/test_console_native_chat_flow.py`

Repeat this task for each `fail_chatbook` provider or shared failure class.

- [ ] **Step 1: Preserve failure evidence**

Update the QA report with:

- provider row;
- screenshot/log path;
- failure classification;
- suspected failing file/function;
- why it is Chatbook-caused.

- [ ] **Step 2: Write focused failing test**

Choose the narrowest test file:

- Gateway resolution/normalization: `Tests/Chat/test_console_provider_gateway.py`
- Provider alias/readiness identity: `Tests/Chat/test_console_provider_support.py`
- Credential readiness: `Tests/Chat/test_provider_readiness.py`
- Dispatcher parameter mapping: `Tests/Chat/test_chat_functions.py`
- Rendered Console behavior: `Tests/UI/test_console_native_chat_flow.py`

Expected: The test fails before the fix and does not call real provider APIs.

- [ ] **Step 3: Run failing test**

Run the focused test, for example:

```bash
.venv/bin/python -m pytest -q Tests/Chat/test_console_provider_gateway.py::test_name_here --tb=short
```

Expected: FAIL for the reproduced defect.

- [ ] **Step 4: Implement minimal fix**

Make the smallest scoped change. Do not refactor unrelated provider architecture. Do not log raw keys.

- [ ] **Step 5: Run focused tests**

Run the failing test plus nearby provider tests:

```bash
.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_provider_gateway.py \
  Tests/Chat/test_console_provider_support.py \
  Tests/Chat/test_provider_readiness.py \
  --tb=short
```

Expected: PASS.

- [ ] **Step 6: Rerun provider through CDP**

Repeat Task 8 for the fixed provider.

Expected: Provider either passes, is reclassified as external, or has new evidence for the next focused fix.

- [ ] **Step 7: Commit fix and evidence**

Run:

```bash
git add <changed-product-files> <changed-test-files> Docs/superpowers/qa/provider-cdp-uat
git commit -m "Fix Console provider UAT issue for <provider-or-class>"
```

Expected: Commit contains the focused fix, tests, and related evidence update.

## Task 10: Final Verification And Closeout

**Files:**
- Modify: `Docs/superpowers/qa/provider-cdp-uat/2026-05-31-provider-cdp-uat.md`
- Modify: `Docs/superpowers/qa/provider-cdp-uat/README.md` if index needs final links
- Modify: `backlog/tasks/task-<id> - Provider-CDP-UAT-sweep.md`

- [ ] **Step 1: Run helper tests**

Run: `.venv/bin/python -m pytest -q Tests/QA/test_provider_cdp_uat_helpers.py --tb=short`

Expected: PASS.

- [ ] **Step 2: Run focused provider/Console tests**

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_provider_gateway.py \
  Tests/Chat/test_console_provider_support.py \
  Tests/Chat/test_provider_readiness.py \
  Tests/UI/test_console_native_chat_flow.py \
  --tb=short
```

Expected: PASS.

- [ ] **Step 3: Run secret scan on QA evidence**

Run:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
root = Path("Docs/superpowers/qa/provider-cdp-uat")
for path in root.rglob("*"):
    if path.is_file() and path.suffix.lower() not in {".png"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for forbidden in ("sk-proj-", "sk-ant-", "AIza", "gsk_", "hf_", "sk-or-"):
            assert forbidden not in text, f"{forbidden} in {path}"
print("qa evidence redacted")
PY
```

Expected: Prints `qa evidence redacted`.

- [ ] **Step 4: Verify screenshots**

Run:

```bash
find Docs/superpowers/qa/provider-cdp-uat/screenshots -name "*.png" -print -exec file {} \;
```

Expected: Every screenshot reports `PNG image data`.

- [ ] **Step 5: Visually inspect screenshots before final commit**

Open the baseline and provider screenshots. Confirm no raw API keys, secret-bearing settings values, or unrelated private data are visible. This is required because the text secret scan excludes PNG files.

Expected: Every screenshot is safe to commit and corresponds to the provider row it supports.

- [ ] **Step 6: Complete QA report**

The report must include:

- runtime inventory source and timestamp;
- final provider table;
- pass/skip/fail_external/fail_chatbook/fixed-then-pass counts;
- second-turn evidence for each pass;
- model source and endpoint source/reachability for each relevant provider;
- code fixes and test commands;
- residual risks.

- [ ] **Step 7: Update Backlog task**

Only mark acceptance criteria complete when the evidence exists. Add implementation notes summarizing UAT execution, fixes, and residual risks.

Run:

```bash
backlog task edit <task-id> -s Done --notes "Completed Provider CDP UAT sweep with isolated Textual-web/CDP execution, redacted provider inventory, two-turn provider evidence, focused fixes where Chatbook defects were found, and residual risks recorded in Docs/superpowers/qa/provider-cdp-uat/2026-05-31-provider-cdp-uat.md."
```

Expected: Task status is Done only if all DoD items are satisfied.

- [ ] **Step 8: Final commit**

Run:

```bash
git add Docs/superpowers/qa/provider-cdp-uat backlog/tasks
git commit -m "Complete provider CDP UAT evidence"
```

Expected: Final commit contains QA report, screenshots, inventory, and task closeout.

- [ ] **Step 9: Final status**

Report:

- providers passed;
- providers skipped and why;
- external failures;
- Chatbook defects fixed;
- tests run;
- evidence report path;
- any blockers needing user action.
