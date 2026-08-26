# TASK-3796 Summarization Diagnostic Privacy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove private values from all 200 verified summarization diagnostics while preserving useful bounded metadata and every existing summarization return, error, retry, and streaming contract.

**Architecture:** Repair diagnostic arguments directly in the two existing summarization modules; do not add a production logging wrapper. A test-only AST ledger reconciles the 200 strict replacements/deletions and 323 frozen reviewed-safe calls against all 523 starting logger calls using stable module/function/event identities, while direct production-function sentinels prove private values do not reach stdlib logging or Loguru.

**Tech Stack:** Python 3.11+, pytest, stdlib `ast`/`logging`, Loguru capture, Requests transport seams, Ruff, the existing production diagnostic inventory checker.

**Design:** `Docs/superpowers/specs/2026-08-10-task-3796-summarization-diagnostic-privacy-design.md`

**Backlog:** `backlog/tasks/task-3796 - Remove-private-summarization-values-from-diagnostics.md`

**ADR required:** no

**ADR path:** `backlog/decisions/029-local-private-data-boundary.md`

**Reason:** ADR-029 already defines the persistent metadata boundary. This plan adds scoped source containment in two modules without changing global diagnostic admission or another architectural contract.

---

## File map and invariants

### Production files

- Modify: `tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py` — remove private diagnostic arguments from 100 inventoried sites; add only the existing `safe_metadata_token` import needed for bounded exception/type tokens.
- Modify: `tldw_chatbook/LLM_Calls/Summarization_General_Lib.py` — remove private diagnostic arguments from 100 inventoried sites; add only the existing `safe_metadata_token` import needed for bounded exception/type tokens.
- Modify: `Docs/security/production-diagnostic-inventory.json` — regenerate exactly the two summarization-owner entries after every privacy gate is green.

### Test and verification files

- Create: `Tests/LLM_Calls/summarization_diagnostic_guard.py` — test-only AST extraction, stable call identity, field extraction, exception-capture detection, and ledger reconciliation.
- Create: `Tests/fixtures/summarization_diagnostic_review.json` — reviewed 523-call starting ledger and final replacement/deletion outcomes; line numbers are optional navigation data and never identity.
- Create: `Tests/LLM_Calls/test_summarization_diagnostic_privacy.py` — guard unit tests, eight batch completion gates, twelve-category direct-function sentinel matrix, final all-call reconciliation, and manifest boundary assertions.
- Modify: `backlog/tasks/task-3796 - Remove-private-summarization-values-from-diagnostics.md` — check acceptance criteria and add final implementation notes only after verification.
- Modify: `Docs/superpowers/specs/2026-08-10-task-3796-summarization-diagnostic-privacy-design.md` — set final implemented status only at closeout.
- Modify: this plan — check completed steps and record any approved deviation.

All production line ranges below are pre-change navigation aids. Resolve the named functions again after earlier deletions shift the file; no test, ledger key, or formatter gate depends on those numbers.

### Non-negotiable implementation invariants

- The immutable starting arithmetic is exact: `200 private + 323 reviewed_safe = 523` calls, distributed `242 Local + 281 General`. Repair progress is a separate `outcome` field and never overwrites this provenance.
- Private-site batches are exact: Local `24 + 23 + 22 + 31 = 100`; General `36 + 24 + 20 + 20 = 100`.
- The 323 reviewed-safe calls retain their exact normalized expression structures unless a newly verified misclassification is first added to TASK-3796 and approved.
- A repaired message uses fixed event text and lazy metadata arguments. A dynamic string replacement must pass through `safe_metadata_token()` and accept its fixed `invalid` result.
- Never call `str(exc)`, render an exception object, log `response.text`, render a response/payload/decoded line, or enable traceback capture.
- Do not move returns, raises, yields, retries, or transport calls. Do not consume a generator in production merely to log metadata.
- Do not construct any test app, reduced app, or simplified application. Invoke the real functions and replace only transport/config/file/sleep seams.
- Run only tests related to these files/functions and the diagnostic inventory; do not run repository-wide pytest.

### Replacement patterns

Use the smallest event-specific form. Keep the existing level unless the entire redundant call is deleted.

```python
from tldw_chatbook.Utils.persistent_diagnostics import safe_metadata_token

# Exception/error detail: no message and no traceback.
logging.error(
    "Custom OpenAI API: request failed; exception_type=%s",
    safe_metadata_token(type(exc).__name__),
)

# Response body/output: status and size only when already available.
logging.warning(
    "Custom OpenAI API: non-success response; status_code=%s payload_length=%s",
    response.status_code,
    len(response.content),
)

# Prompt/input: a count, never a preview.
logging.debug("Custom OpenAI API: prompt prepared; character_count=%s", len(prompt))

# Credential/endpoint/path: fixed state only.
logging.debug("Custom OpenAI API: credential configured")
logging.debug("Custom OpenAI API: endpoint configured")
```

If an adjacent safe diagnostic already carries the operational state, delete the private duplicate instead of inventing a second event. Record `outcome: "deleted"` and a one-line reason in the ledger.

### General-module formatter protocol

The current General module has exactly one pre-existing Ruff-format delta: the multiline `_CHAT_DISPATCH_NAME_ALIASES.get(...)` assignment near current line 311. Hard-coded later ranges are unsafe because earlier diagnostic deletions shift line numbers. After each General batch:

1. run Ruff format on the whole General module and changed test file;
2. use `apply_patch` to restore only this exact current-dev baseline form:

```python
api_name_lower = _CHAT_DISPATCH_NAME_ALIASES.get(
    api_name_lower, api_name_lower
)
```

3. run full-file `ruff format --check --diff` on General and require its only output to be the same one-hunk collapse recorded on exact current `origin/dev`;
4. inspect the branch diff to prove the formatter introduced no unrelated change.

This formats every task-owned edit without either blessing or hiding the pre-existing baseline. If current dev's formatter baseline changes, rerun the identical command there and update this protocol before editing production.

---

### Task 1: Refresh current-dev ownership and reproduce the baseline

**Files:**
- Read: the task, approved design, ADR-029, testing/backlog lessons, two production modules, and focused tests
- No tracked edits

- [x] **Step 1: Confirm the worktree is clean and the task has no competing implementation**

Run:

```bash
git status --short --branch
gh pr list --state open --search "3796" --json number,title,headRefName,url
git for-each-ref --format='%(refname:short)' refs/remotes/ | rg -i '3796'
```

Expected: clean worktree; no unrelated open PR or remote branch claiming TASK-3796. Stop and reconcile if ownership changed.

- [x] **Step 2: Fetch and rebase onto the latest `origin/dev`**

Run:

```bash
git fetch origin dev
git rebase origin/dev
git merge-base --is-ancestor origin/dev HEAD
git rev-list --left-right --count origin/dev...HEAD
```

Expected: conflict-free rebase, ancestor exit 0, and `0 <ahead>`.

- [x] **Step 3: Recheck the approved source boundary after the rebase**

Run:

```bash
git diff --stat 59cf35d6e..origin/dev -- tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py tldw_chatbook/LLM_Calls/Summarization_General_Lib.py Tests/LLM_Calls/test_summarization_analyze.py Tests/Chat/test_cohere_summarize_v2.py Tests/Internal_Prompts/test_summarization_migration.py Tests/Internal_Prompts/test_summarization_prompt_parity.py
```

Expected: either no scoped upstream change, or a fully reviewed delta reconciled into the task/spec before continuing. Never carry old inventory counts across an upstream logger change or verified misclassification without a fresh all-call review.

- [x] **Step 4: Run only the focused behavioral baseline**

Run:

```bash
../../.venv/bin/python -B -m pytest -q Tests/LLM_Calls/test_summarization_analyze.py Tests/Chat/test_cohere_summarize_v2.py Tests/Internal_Prompts/test_summarization_migration.py Tests/Internal_Prompts/test_summarization_prompt_parity.py
```

Expected at the approved base: `37 passed`.

- [x] **Step 5: Run the cross-cutting inventory and static baselines**

Run:

```bash
../../.venv/bin/python -B -m pytest -q Tests/Architecture/test_persistent_diagnostic_inventory.py
../../.venv/bin/python -m ruff check tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py tldw_chatbook/LLM_Calls/Summarization_General_Lib.py
../../.venv/bin/python -m ruff format --check tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py
../../.venv/bin/python -m ruff format --check --diff tldw_chatbook/LLM_Calls/Summarization_General_Lib.py
```

Expected at the approved base: inventory `14 passed`, lint green, Local formatted, and General format red only for the pre-existing line-311 `_CHAT_DISPATCH_NAME_ALIASES` collapse. Record that exact baseline; do not reformat it unless a task-owned hunk reaches that statement.

---

### Task 2: Build the test-only 523-call reconciliation guard

**Files:**
- Create: `Tests/LLM_Calls/summarization_diagnostic_guard.py`
- Create: `Tests/fixtures/summarization_diagnostic_review.json`
- Create: `Tests/LLM_Calls/test_summarization_diagnostic_privacy.py`

- [x] **Step 1: Write synthetic failing tests for stable identity and failure modes**

Import the not-yet-created helper inside a small loader that converts `ModuleNotFoundError` into an intentional assertion failure; do not use a top-level import that aborts collection:

```python
def _load_guard_module():
    try:
        return importlib.import_module(
            "Tests.LLM_Calls.summarization_diagnostic_guard"
        )
    except ModuleNotFoundError:
        pytest.fail("summarization diagnostic guard is not implemented")
```

Use that loader in tests named:

```python
def test_guard_finds_stdlib_loguru_nested_and_bound_calls() -> None: ...
def test_guard_identity_ignores_line_movement() -> None: ...
def test_guard_rejects_changed_reviewed_safe_expression() -> None: ...
def test_guard_records_and_rejects_bare_name_message() -> None: ...
def test_guard_records_and_rejects_percent_formatted_message() -> None: ...
def test_guard_records_and_rejects_dot_format_message() -> None: ...
def test_guard_records_and_rejects_concatenated_message() -> None: ...
def test_guard_rejects_exception_and_traceback_capture() -> None: ...
```

The synthetic source must include stdlib logging, Loguru, `logger.opt(...)`, a nested generator, duplicate labels requiring an occurrence ordinal, bare-name first messages, `%` interpolation, `.format()`, string concatenation, and `exc_info=True`/`logger.exception` mutants. For each dynamic-message mutant, first prove its exact `message_shape`/expressions are recorded, then classify it as `metadata` and prove the strict validator rejects it.

- [x] **Step 2: Run the new guard tests and verify RED**

Run:

```bash
../../.venv/bin/python -B -m pytest -q Tests/LLM_Calls/test_summarization_diagnostic_privacy.py -k 'guard_' -vv
```

Expected: collected tests fail with the exact assertion `summarization diagnostic guard is not implemented`. A collection error, syntax error, or unrelated failure does not count as RED.

- [x] **Step 3: Implement the minimal test-only extractor**

Reuse `scripts.check_persistent_diagnostic_inventory._logger_symbols`, `_is_diagnostic_call`, and `_scope_names`; do not add a second production scanner. The helper's core record is:

```python
@dataclass(frozen=True)
class DiagnosticCall:
    module: str
    qualname: str
    method: str
    event: str
    occurrence: int
    message_shape: str
    expressions: tuple[str, ...]
    captures_exception: bool

    @property
    def identity(self) -> tuple[str, str, str, int]:
        return (self.module, self.qualname, self.event, self.occurrence)
```

`event` is the constant literal projection of the first argument; formatted values are excluded from the label. `message_shape` is the canonical `ast.dump(first_argument, include_attributes=False)` for every call, including constants, f-strings, bare names, `%`, `.format()`, and concatenation. `expressions` includes every dynamic subtree in the first argument plus positional arguments after the message, keyword fields, and bound fields. Ignore line numbers when comparing identity. A `metadata` outcome requires a constant string first argument; legacy dynamic first arguments may exist only as exact frozen or pending shapes.

- [x] **Step 4: Create and hand-review the starting ledger**

Emit the extractor snapshot to stdout, then add the reviewed JSON with `apply_patch`. Do not let a generator silently classify entries. Each entry retains an immutable starting record and a separately updated current outcome:

```json
{
  "site_id": "local-custom-openai-credential-0",
  "module": "tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py",
  "qualname": "summarize_with_custom_openai",
  "group": "local_custom",
  "starting_classification": "private",
  "category": "credential",
  "starting": {
    "method": "debug",
    "event": "Custom OpenAI API: Using API Key: ...",
    "occurrence": 0,
    "message_shape": "JoinedStr(values=[Constant(value='Custom OpenAI API: Using API Key: '), ...])",
    "expressions": ["custom_openai_api_key[:5]", "custom_openai_api_key[-5:]"],
    "captures_exception": false
  },
  "outcome": "pending",
  "current": {
    "method": "debug",
    "event": "Custom OpenAI API: Using API Key: ...",
    "occurrence": 0,
    "message_shape": "JoinedStr(values=[Constant(value='Custom OpenAI API: Using API Key: '), ...])",
    "expressions": ["custom_openai_api_key[:5]", "custom_openai_api_key[-5:]"],
    "captures_exception": false
  },
  "navigation_line": 1515
}
```

The 200 task-table entries use `starting_classification: "private"` and begin with `outcome: "pending"`. Every other discovered call uses `starting_classification: "reviewed_safe"`, `outcome: "frozen"`, exact normalized expressions, and a reason from the approved categories. `starting` never changes. `current` follows the repaired source and becomes `null` only for a reviewed deletion. Verify exact totals and groups:

```text
Local: 242 = 100 starting-private + 142 reviewed-safe
  local_core 24; local_adapters 23; local_vllm_ollama 22; local_custom 31
General: 281 = 100 starting-private + 181 reviewed-safe
  general_core 36; general_mid 24; general_streaming 20; general_tail 20
Overall: 523 = 200 starting-private + 323 reviewed-safe
```

- [x] **Step 5: Add the real-source reconciliation test**

```python
def test_ledger_retains_all_523_starting_sites() -> None:
    ledger = load_review_ledger()
    assert len(ledger) == 523
    assert Counter(item.starting_classification for item in ledger) == {
        "private": 200,
        "reviewed_safe": 323,
    }
    assert starting_projection_digest(ledger) == STARTING_PROJECTION_SHA256


def test_ledger_current_state_matches_sources() -> None:
    assert_ledger_matches_source(load_review_ledger(), scan_reviewed_modules())
```

Compute `STARTING_PROJECTION_SHA256` once from canonical JSON containing only site ID, module, qualname, group, starting classification/category, and the complete `starting` record; hard-code that digest in the test module. Outcome/current/navigation edits do not participate. The matcher compares each non-deleted `current` record to source and each deleted outcome to explicit absence. It must fail on additions, undeclared deletions, label/method/message-shape changes, expression changes, duplicate loss, unclassified calls, dynamic first arguments in a `metadata` record, or exception capture in a `frozen`/`metadata` record. A known-private `pending` record may temporarily match its inventoried starting traceback shape; it cannot survive a batch completion gate or the final zero-pending gate. The immutable projection digest preserves starting evidence after current source changes.

- [x] **Step 6: Run the guard foundation GREEN**

Run:

```bash
../../.venv/bin/python -B -m pytest -q Tests/LLM_Calls/test_summarization_diagnostic_privacy.py -k 'guard_ or ledger_' -vv
../../.venv/bin/python -m ruff format Tests/LLM_Calls/summarization_diagnostic_guard.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
../../.venv/bin/python -m ruff check Tests/LLM_Calls/summarization_diagnostic_guard.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
../../.venv/bin/python -m ruff format --check Tests/LLM_Calls/summarization_diagnostic_guard.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
```

Expected: all selected tests pass with exact `523/200/323` reconciliation.

- [x] **Step 7: Commit the test-only foundation**

```bash
git add Tests/LLM_Calls/summarization_diagnostic_guard.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py Tests/fixtures/summarization_diagnostic_review.json
git commit -m "test(security): inventory summarization diagnostic privacy"
```

---

### Task 3: Repair Local core, Llama, and Kobold diagnostics (24 sites)

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py:21-550`
- Modify: `Tests/LLM_Calls/test_summarization_diagnostic_privacy.py`
- Modify: `Tests/fixtures/summarization_diagnostic_review.json`

- [x] **Step 1: Add the failing 24-site batch gate and direct-function sentinels**

Add `test_no_pending_local_core_sites`, plus direct tests for `summarize_with_local_llm` success, malformed streamed JSON, and exception paths. Use canaries for input, response line, and exception message; fully consume the returned stream. Assert the pre-change return/yield/error contract separately from captured logging.

Build the required cross-module category matrix under one stable parameterized node name. Add cases incrementally as their production paths are repaired:

```python
@pytest.mark.parametrize(
    "case",
    RUNTIME_SENTINEL_CASES,
    ids=lambda case: f"{case.module}-{case.category}",
)
def test_runtime_sentinel_hides_private_value(case, monkeypatch, caplog) -> None:
    with capture_stdlib_and_loguru(caplog) as captured:
        result = case.invoke(monkeypatch)
        case.assert_contract(result)
    assert case.canary not in captured.text
```

Final IDs are exactly `local-input`, `local-prompt`, `local-credential`, `local-path`, `local-response`, `local-exception`, and the corresponding six `general-*` IDs. Extra provider/stream tests use separate descriptive names.

- [x] **Step 2: Run RED**

```bash
../../.venv/bin/python -B -m pytest -q Tests/LLM_Calls/test_summarization_diagnostic_privacy.py -k 'local_core or local_llm or runtime_sentinel' -vv
```

Expected: failures expose the owned canaries and report exactly 24 pending entries.

- [x] **Step 3: Apply the 24 direct repairs**

Repair `summarize_with_local_llm`, `summarize_with_llama`, and `summarize_with_kobold`. Delete redundant input/body previews when an adjacent type/status event survives; replace prompt, credential, endpoint, response, and exception details with the approved fixed/count/status/token shapes. Remove traceback capture.

Update only `outcome` to `metadata` or `deleted` and update `current` with final event/exact allowed fields or `null` plus a deletion reason. Do not modify `starting` or any `reviewed_safe` entry.

- [x] **Step 4: Run GREEN and static checks**

```bash
../../.venv/bin/python -B -m pytest -q Tests/LLM_Calls/test_summarization_diagnostic_privacy.py -k 'local_core or local_llm or runtime_sentinel' -vv
../../.venv/bin/python -m ruff format tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py Tests/LLM_Calls/summarization_diagnostic_guard.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
../../.venv/bin/python -m ruff check tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py Tests/LLM_Calls/summarization_diagnostic_guard.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
../../.venv/bin/python -m ruff format --check tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py Tests/LLM_Calls/summarization_diagnostic_guard.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
```

Expected: selected tests and static checks pass.

- [x] **Step 5: Commit the batch**

```bash
git add tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py Tests/fixtures/summarization_diagnostic_review.json
git commit -m "fix(security): redact local summarizer core diagnostics"
```

---

### Task 4: Repair Oobabooga and Tabby diagnostics (23 sites)

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py:551-994`
- Modify: `Tests/LLM_Calls/test_summarization_diagnostic_privacy.py`
- Modify: `Tests/fixtures/summarization_diagnostic_review.json`

- [x] **Step 1: Add `test_no_pending_local_adapters_sites` and real-function canaries**

Drive `summarize_with_oobabooga` through a signature-bound fake `requests.Session.post` result. Cover prompt, credential, endpoint, response, and exception canaries. Add a Tabby malformed-stream case and consume the generator.

- [x] **Step 2: Run RED**

```bash
../../.venv/bin/python -B -m pytest -q Tests/LLM_Calls/test_summarization_diagnostic_privacy.py -k 'local_adapters or oobabooga or tabby' -vv
```

Expected: owned canaries appear and the batch reports 23 pending entries.

- [x] **Step 3: Repair exactly the Oobabooga/Tabby entries and ledger records**

Use fixed credential/endpoint events, response status/length where available, line length for stream parse failures, and bounded exception class. Preserve all returned strings and retry/stream behavior.

- [x] **Step 4: Run GREEN, format/lint, and commit**

```bash
../../.venv/bin/python -B -m pytest -q Tests/LLM_Calls/test_summarization_diagnostic_privacy.py -k 'local_adapters or oobabooga or tabby' -vv
../../.venv/bin/python -m ruff format tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
../../.venv/bin/python -m ruff check tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
../../.venv/bin/python -m ruff format --check tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
git add tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py Tests/fixtures/summarization_diagnostic_review.json
git commit -m "fix(security): redact local adapter diagnostics"
```

---

### Task 5: Repair vLLM and Ollama diagnostics (22 sites)

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py:995-1491`
- Modify: `Tests/LLM_Calls/test_summarization_diagnostic_privacy.py`
- Modify: `Tests/fixtures/summarization_diagnostic_review.json`

- [x] **Step 1: Add `test_no_pending_local_vllm_ollama_sites` and failing function tests**

Use real `summarize_with_vllm`/`summarize_with_ollama` calls with signature-bound transport/config seams. Cover raw/processed/extracted input, prompt, credential, response, streamed-line, and exception-message canaries; assert the same summary/error/yield sequence.

- [x] **Step 2: Run RED, repair exactly 22 sites, and update the ledger**

```bash
../../.venv/bin/python -B -m pytest -q Tests/LLM_Calls/test_summarization_diagnostic_privacy.py -k 'local_vllm_ollama or vllm or ollama' -vv
```

Expected before repair: owned canary failures plus 22 pending entries. Replace previews with counts/fixed events, responses with status/length, and errors with bounded class tokens; do not change provider logic.

- [x] **Step 3: Run GREEN and commit**

```bash
../../.venv/bin/python -B -m pytest -q Tests/LLM_Calls/test_summarization_diagnostic_privacy.py -k 'local_vllm_ollama or vllm or ollama' -vv
../../.venv/bin/python -m ruff format tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
../../.venv/bin/python -m ruff check tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
../../.venv/bin/python -m ruff format --check tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
git add tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py Tests/fixtures/summarization_diagnostic_review.json
git commit -m "fix(security): redact vllm and ollama diagnostics"
```

---

### Task 6: Repair custom OpenAI and file-save diagnostics (31 sites)

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py:1492-2005`
- Modify: `Tests/LLM_Calls/test_summarization_diagnostic_privacy.py`
- Modify: `Tests/fixtures/summarization_diagnostic_review.json`

- [x] **Step 1: Add `test_no_pending_local_custom_sites` and complete the Local six-category matrix**

Directly exercise both custom OpenAI functions in success, non-success, raised-exception, and malformed-stream modes. Capture both logging systems; use distinct input, prompt, credential, endpoint, response, and exception canaries. Consume generators. Exercise `save_summary_to_file` with a private path canary while asserting the file-write contract.

- [x] **Step 2: Run RED, repair 31 entries, and run GREEN**

```bash
../../.venv/bin/python -B -m pytest -q Tests/LLM_Calls/test_summarization_diagnostic_privacy.py -k 'local_custom or custom_openai or save_summary' -vv
```

Expected RED: all category canaries prove their prior exposure and the group reports 31 pending sites. Apply direct fixed/count/status/token replacements or justified deletions, update only those ledger entries, then rerun the same command to green.

- [x] **Step 3: Add and pass the Local-module completion gate**

```python
def test_local_module_has_no_pending_private_sites() -> None:
    ledger = load_review_ledger()
    local = [item for item in ledger if item.module.endswith("Local_Summarization_Lib.py")]
    assert len(local) == 242
    assert sum(item.starting_classification == "reviewed_safe" for item in local) == 142
    assert sum(item.outcome == "frozen" for item in local) == 142
    assert not [item for item in local if item.outcome == "pending"]
    assert sum(item.outcome in {"metadata", "deleted"} for item in local) == 100
```

- [x] **Step 4: Run all Local privacy tests and commit**

```bash
../../.venv/bin/python -B -m pytest -q Tests/LLM_Calls/test_summarization_diagnostic_privacy.py -k 'no_pending_local or local_custom or (runtime_sentinel and local) or local_llm or local_core or oobabooga or tabby or vllm or ollama or local_save_summary' -vv
../../.venv/bin/python -m ruff format tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
../../.venv/bin/python -m ruff check tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
../../.venv/bin/python -m ruff format --check tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
git add tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py Tests/fixtures/summarization_diagnostic_review.json
git commit -m "fix(security): finish local summarizer diagnostic privacy"
```

---

### Task 7: Repair General core, OpenAI, and Anthropic diagnostics (36 sites)

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/Summarization_General_Lib.py:76-230`
- Modify: `tldw_chatbook/LLM_Calls/Summarization_General_Lib.py:490-1189`
- Modify: `Tests/LLM_Calls/test_summarization_diagnostic_privacy.py`
- Modify: `Tests/fixtures/summarization_diagnostic_review.json`

- [x] **Step 1: Add `test_no_pending_general_core_sites` and the General six-category matrix**

Call `extract_text_from_segments`, `extract_text_from_input`, `recursive_summarize_chunks`, `analyze`, `summarize_with_openai`, and `summarize_with_anthropic` directly. Use signature-bound transport/file/config seams only. Cover every category with distinct canaries, including the nested `analyze.consume_generator` and provider streaming generators, and assert unchanged return/error/yield contracts.

- [x] **Step 2: Run RED**

```bash
../../.venv/bin/python -B -m pytest -q Tests/LLM_Calls/test_summarization_diagnostic_privacy.py -k 'general_core or analyze or general_openai or anthropic or (runtime_sentinel and general)' -vv
```

Expected: category-specific canary failures and exactly 36 pending sites.

- [x] **Step 3: Repair the 36 owned calls without touching `_dispatch_to_api` line 311**

Apply the strict replacement patterns, update only the 36 ledger records, and preserve all in-band error strings. The pre-existing formatter delta at `_CHAT_DISPATCH_NAME_ALIASES` is outside the owned changed ranges.

- [x] **Step 4: Run GREEN and format the changed files**

```bash
../../.venv/bin/python -B -m pytest -q Tests/LLM_Calls/test_summarization_diagnostic_privacy.py -k 'general_core or analyze or general_openai or anthropic or (runtime_sentinel and general)' -vv
../../.venv/bin/python -B -m pytest -q Tests/LLM_Calls/test_summarization_analyze.py
../../.venv/bin/python -m ruff format tldw_chatbook/LLM_Calls/Summarization_General_Lib.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
```

Restore the one baseline statement with `apply_patch` as specified by the General-module formatter protocol, then run:

```bash
../../.venv/bin/python -m ruff check tldw_chatbook/LLM_Calls/Summarization_General_Lib.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
../../.venv/bin/python -m ruff format --check Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
../../.venv/bin/python -m ruff format --check --diff tldw_chatbook/LLM_Calls/Summarization_General_Lib.py
```

Expected: tests/lint/test formatting green; full-file General format remains baseline-red only at unchanged line 311 with output identical to Task 1.

- [x] **Step 5: Commit**

```bash
git add tldw_chatbook/LLM_Calls/Summarization_General_Lib.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py Tests/fixtures/summarization_diagnostic_review.json
git commit -m "fix(security): redact general summarizer core diagnostics"
```

---

### Task 8: Repair Cohere, Groq, and OpenRouter diagnostics (23 sites)

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/Summarization_General_Lib.py:1190-1862`
- Modify: `Tests/LLM_Calls/test_summarization_diagnostic_privacy.py`
- Modify: `Tests/fixtures/summarization_diagnostic_review.json`

- [x] **Step 1: Add `test_no_pending_general_mid_sites` and provider stream/error sentinels**

Directly invoke Cohere, Groq, and OpenRouter functions with signature-bound transports. Fully consume streams and assert response-line/body and exception canaries are absent without changing results.

- [x] **Step 2: Run RED, repair exactly 23 sites, and update the ledger**

```bash
../../.venv/bin/python -B -m pytest -q Tests/LLM_Calls/test_summarization_diagnostic_privacy.py -k 'general_mid or cohere or groq or openrouter' -vv
```

- [x] **Step 3: Run GREEN plus existing Cohere tests and commit**

```bash
../../.venv/bin/python -B -m pytest -q Tests/LLM_Calls/test_summarization_diagnostic_privacy.py -k 'general_mid or cohere or groq or openrouter' -vv
../../.venv/bin/python -B -m pytest -q Tests/Chat/test_cohere_summarize_v2.py
../../.venv/bin/python -m ruff format tldw_chatbook/LLM_Calls/Summarization_General_Lib.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
```

Restore the one baseline statement with `apply_patch`, then run:

```bash
../../.venv/bin/python -m ruff check tldw_chatbook/LLM_Calls/Summarization_General_Lib.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
../../.venv/bin/python -m ruff format --check Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
../../.venv/bin/python -m ruff format --check --diff tldw_chatbook/LLM_Calls/Summarization_General_Lib.py
```

Expected: the last command exits nonzero with exactly the recorded one-hunk baseline. After verifying that output, commit in a separate command block:

```bash
git add tldw_chatbook/LLM_Calls/Summarization_General_Lib.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py Tests/fixtures/summarization_diagnostic_review.json
git commit -m "fix(security): redact general provider diagnostics"
```

---

### Task 9: Repair HuggingFace, DeepSeek, and Mistral diagnostics (20 sites)

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/Summarization_General_Lib.py:1863-2437`
- Modify: `Tests/LLM_Calls/test_summarization_diagnostic_privacy.py`
- Modify: `Tests/fixtures/summarization_diagnostic_review.json`

- [x] **Step 1: Add `test_no_pending_general_streaming_sites` and malformed-stream sentinels**

Cover each provider's response/body and exception paths; consume HuggingFace, DeepSeek, and Mistral streams so decode/key-error diagnostics actually execute.

- [x] **Step 2: Run RED, repair 20 sites, then run GREEN**

```bash
../../.venv/bin/python -B -m pytest -q Tests/LLM_Calls/test_summarization_diagnostic_privacy.py -k 'general_streaming or huggingface or deepseek or mistral' -vv
```

Expected RED before repair: owned canaries plus 20 pending entries. After direct replacements/deletions and ledger updates, the same command passes.

- [x] **Step 3: Format, lint, and commit**

```bash
../../.venv/bin/python -m ruff format tldw_chatbook/LLM_Calls/Summarization_General_Lib.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
```

Restore the one baseline statement with `apply_patch`, then run:

```bash
../../.venv/bin/python -m ruff check tldw_chatbook/LLM_Calls/Summarization_General_Lib.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
../../.venv/bin/python -m ruff format --check Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
../../.venv/bin/python -m ruff format --check --diff tldw_chatbook/LLM_Calls/Summarization_General_Lib.py
```

Expected: the last command exits nonzero with only the known baseline hunk. Then commit separately:

```bash
git add tldw_chatbook/LLM_Calls/Summarization_General_Lib.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py Tests/fixtures/summarization_diagnostic_review.json
git commit -m "fix(security): redact streaming provider diagnostics"
```

---

### Task 10: Repair Google, mock, and chunk diagnostics (20 sites)

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/Summarization_General_Lib.py:2438-2761`
- Modify: `Tests/LLM_Calls/test_summarization_diagnostic_privacy.py`
- Modify: `Tests/fixtures/summarization_diagnostic_review.json`

- [x] **Step 1: Add `test_no_pending_general_tail_sites` and direct Google/mock/chunk tests**

Cover Google's input, prompt, credential, response/stream, and exception paths plus mock/chunk output/error diagnostics. Assert all public result strings/yields remain unchanged.

- [x] **Step 2: Run RED, repair 20 entries, and run GREEN**

```bash
../../.venv/bin/python -B -m pytest -q Tests/LLM_Calls/test_summarization_diagnostic_privacy.py -k 'general_tail or google or mock_llm or summarize_chunk' -vv
```

- [x] **Step 3: Add the General and whole-ledger completion gates**

```python
def test_general_module_has_no_pending_private_sites() -> None:
    general = general_ledger_entries()
    assert len(general) == 281
    assert sum(item.starting_classification == "reviewed_safe" for item in general) == 181
    assert sum(item.outcome == "frozen" for item in general) == 181
    assert not [item for item in general if item.outcome == "pending"]
    assert sum(item.outcome in {"metadata", "deleted"} for item in general) == 100


def test_complete_ledger_reconciles_without_private_sites() -> None:
    ledger = load_review_ledger()
    assert len(ledger) == 523
    assert sum(item.starting_classification == "reviewed_safe" for item in ledger) == 323
    assert sum(item.outcome == "frozen" for item in ledger) == 323
    assert sum(item.outcome in {"metadata", "deleted"} for item in ledger) == 200
    assert not [item for item in ledger if item.outcome == "pending"]
    assert_ledger_matches_source(ledger, scan_reviewed_modules())
```

For deleted records, `assert_ledger_matches_source` reconciles the retained starting identity against an explicit absence; current source-call count is `523 - deleted_count`.

- [x] **Step 4: Run all new tests, format, and commit**

```bash
../../.venv/bin/python -B -m pytest -q Tests/LLM_Calls/test_summarization_diagnostic_privacy.py -vv
../../.venv/bin/python -m ruff format tldw_chatbook/LLM_Calls/Summarization_General_Lib.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
```

Restore the one baseline statement with `apply_patch`, then run:

```bash
../../.venv/bin/python -m ruff check tldw_chatbook/LLM_Calls/Summarization_General_Lib.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
../../.venv/bin/python -m ruff format --check Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
../../.venv/bin/python -m ruff format --check --diff tldw_chatbook/LLM_Calls/Summarization_General_Lib.py
```

Expected: the last command exits nonzero with only the known baseline hunk. Then commit separately:

```bash
git add tldw_chatbook/LLM_Calls/Summarization_General_Lib.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py Tests/fixtures/summarization_diagnostic_review.json
git commit -m "fix(security): finish general summarizer diagnostic privacy"
```

---

### Task 11: Prove guard and runtime sensitivity with independent mutations

**Files:**
- Temporarily modify and restore the two production modules with `apply_patch`
- Modify: TASK-3796 implementation notes only after all evidence is complete

- [x] **Step 1: Record clean restoration hashes**

```bash
git hash-object tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py
git hash-object tldw_chatbook/LLM_Calls/Summarization_General_Lib.py
git status --short
```

Expected: clean status; record both hashes.

- [x] **Step 2: Collect and record the exact 12 owning node IDs**

Run:

```bash
../../.venv/bin/python -B -m pytest --collect-only -q Tests/LLM_Calls/test_summarization_diagnostic_privacy.py -k runtime_sentinel_hides_private_value
```

Expected: exactly these 12 selected nodes with no duplicate ID. Other tests in the file may be reported as deselected by the intentional `-k` filter:

```text
Tests/LLM_Calls/test_summarization_diagnostic_privacy.py::test_runtime_sentinel_hides_private_value[local-input]
Tests/LLM_Calls/test_summarization_diagnostic_privacy.py::test_runtime_sentinel_hides_private_value[local-prompt]
Tests/LLM_Calls/test_summarization_diagnostic_privacy.py::test_runtime_sentinel_hides_private_value[local-credential]
Tests/LLM_Calls/test_summarization_diagnostic_privacy.py::test_runtime_sentinel_hides_private_value[local-path]
Tests/LLM_Calls/test_summarization_diagnostic_privacy.py::test_runtime_sentinel_hides_private_value[local-response]
Tests/LLM_Calls/test_summarization_diagnostic_privacy.py::test_runtime_sentinel_hides_private_value[local-exception]
Tests/LLM_Calls/test_summarization_diagnostic_privacy.py::test_runtime_sentinel_hides_private_value[general-input]
Tests/LLM_Calls/test_summarization_diagnostic_privacy.py::test_runtime_sentinel_hides_private_value[general-prompt]
Tests/LLM_Calls/test_summarization_diagnostic_privacy.py::test_runtime_sentinel_hides_private_value[general-credential]
Tests/LLM_Calls/test_summarization_diagnostic_privacy.py::test_runtime_sentinel_hides_private_value[general-path]
Tests/LLM_Calls/test_summarization_diagnostic_privacy.py::test_runtime_sentinel_hides_private_value[general-response]
Tests/LLM_Calls/test_summarization_diagnostic_privacy.py::test_runtime_sentinel_hides_private_value[general-exception]
```

Record this collection output before mutation. Each following mutation command names one exact node from this list; never use a broad `-k` selector as mutation evidence.

- [x] **Step 3: Run 12 independent runtime mutations**

For each module/category pair (Local and General × input, prompt, credential, endpoint/path, response/output, exception/error detail):

1. use `apply_patch` to restore exactly one former private interpolation from the task inventory;
2. run only its exact cache-disabled node with `../../.venv/bin/python -B -m pytest -q <node-id> -vv`;
3. require RED on that category's distinctive canary, not setup or another assertion;
4. restore with `apply_patch`;
5. rerun the same node and require GREEN before the next mutation.

Do not combine mutations. For exception mutations, separately confirm traceback-capture mutants fail the structural guard.

- [x] **Step 4: Mutate the stable guard itself**

Temporarily add an unclassified logger call, change one frozen reviewed-safe expression, and enable one `exc_info=True`/Loguru exception capture. Each must fail its intended guard assertion. Restore between runs.

- [x] **Step 5: Prove exact restoration and final reconciliation**

```bash
git hash-object tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py
git hash-object tldw_chatbook/LLM_Calls/Summarization_General_Lib.py
git diff --check
git status --short
../../.venv/bin/python -B -m pytest -q Tests/LLM_Calls/test_summarization_diagnostic_privacy.py -vv
```

Expected: hashes match Step 1, clean status, all new tests green, 323 frozen reviewed-safe records unchanged, 200 repaired/deleted records reconciled, and zero pending/unclassified records.

---

### Task 12: Reconcile only the two diagnostic-inventory owners

**Files:**
- Modify: `Docs/security/production-diagnostic-inventory.json`
- Modify: `Tests/LLM_Calls/test_summarization_diagnostic_privacy.py`

- [x] **Step 1: Run the inventory checker and verify the expected RED boundary**

```bash
../../.venv/bin/python -B scripts/check_persistent_diagnostic_inventory.py
```

Expected: mismatch limited to Local and General summarization owner count/digest entries. Any other delta must be compared with the identical command at exact `origin/dev`; do not bless it.

- [x] **Step 2: Add a permanent two-owner manifest-boundary assertion**

Store the current-dev non-owned inventory fingerprint and the two owners' unchanged path/owner/reason fields in the review fixture. Assert that only their diagnostic count/digest may differ and that sink topology is identical.

- [x] **Step 3: Regenerate through the canonical checker and inspect the exact diff**

```bash
../../.venv/bin/python -B scripts/check_persistent_diagnostic_inventory.py --write
git diff -- Docs/security/production-diagnostic-inventory.json
```

Expected: exactly two owner entries change; call-count deltas equal the ledger's explicit deletion totals, reasons/owners are unchanged, and no sink topology or unrelated entry changes.

- [x] **Step 4: Run inventory/guard GREEN and commit separately**

```bash
../../.venv/bin/python -B -m pytest -q Tests/Architecture/test_persistent_diagnostic_inventory.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
git add Docs/security/production-diagnostic-inventory.json Tests/LLM_Calls/test_summarization_diagnostic_privacy.py Tests/fixtures/summarization_diagnostic_review.json
git commit -m "chore(security): reconcile summarization diagnostic inventory"
```

---

### Task 13: Run focused final verification, review, and close TASK-3796

#### Final-review audit correction

Tasks 2–12 were executed against the then-approved `199 private / 324 reviewed-safe` ledger. Final review identified `general-2efc909241862caf` as a misclassified provider-controlled Cohere response value. The approved correction changes the authoritative starting arithmetic to `200/323`, General to `100/181`, `general_mid` to 24, and response/output content to 72; final outcomes are `177 metadata + 23 deleted + 323 frozen = 523`. The canonical starting-projection digest consequently changes from historical `a4c9ba5f999199f02fd1c6186d1d88120f6d5f696071127ee192dff2c3503047` to corrected `85a5c6b74f0cd4eb15f8ca0f8abfa5e18ca7f26f749d97fc7b781090cabd7733`. The earlier commit history and test transcripts remain valid evidence of the pre-correction audit state, but all final gates and closeout notes use the corrected arithmetic and the added direct Cohere sentinel/mutation evidence.

**Files:**
- Modify: `backlog/tasks/task-3796 - Remove-private-summarization-values-from-diagnostics.md`
- Modify: `Docs/superpowers/specs/2026-08-10-task-3796-summarization-diagnostic-privacy-design.md`
- Modify: this plan
- Modify lesson docs only if a new incident—not an already documented rule—was discovered

- [x] **Step 1: Rebase onto the latest `origin/dev` and re-audit logger drift**

Repeat Task 1 Steps 1–3. Rerun the 523-call reconciliation against the new base. If upstream changed either module, reconcile every changed logger call before continuing.

- [x] **Step 2: Run the complete touched-functionality test set**

```bash
../../.venv/bin/python -B -m pytest -q Tests/LLM_Calls/test_summarization_diagnostic_privacy.py Tests/LLM_Calls/test_summarization_analyze.py Tests/Chat/test_cohere_summarize_v2.py Tests/Internal_Prompts/test_summarization_migration.py Tests/Internal_Prompts/test_summarization_prompt_parity.py Tests/Architecture/test_persistent_diagnostic_inventory.py
```

Expected: all selected tests pass. The sole permitted deviation is
`test_production_diagnostic_inventory_and_sink_topology_are_unchanged` failing with
the exact unrelated latest-dev baseline reproduced in a detached worktree at the
recorded Step 1 base (`6d72f15f8332b6469a5d644d409b80914634a8dd` for the
fresh-review run): these 17 owner paths only —
`Agents/agent_service.py`, `Chat/console_agent_bridge.py`,
`Chat/console_chat_controller.py`, `Chat/console_chat_store.py`,
`Chat/console_context_compaction.py`, `Chat/console_provider_gateway.py`,
`MCP/client.py`, `MCP/local_server_tools.py`, `MCP/prompts.py`, `MCP/server.py`,
`RAG_Search/fusion.py`, `RAG_Search/simplified/rag_service.py`,
`RAG_Search/simplified/search_service.py`, `UI/Console_Modules/session.py`,
`UI/Screens/chat_screen.py`, `UI/Screens/library_screen.py`, and `app.py` — with
the detached-base Git-patch manifest-diff fingerprint
`b77bd95ccc84d3bac066e0971a8bc24e20fdb58bef9b762d5ba77aa6399db4dd`
(`44` additions, `30` deletions, six persistent-sink files). That drift must have
separate backlog ownership and must not be written into TASK-3796's manifest. Any
different failed node, owner set, diff fingerprint, or sink topology is a real
failure. Do not run repository-wide pytest.

- [x] **Step 3: Run final static and formatting gates**

```bash
../../.venv/bin/python -m ruff check tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py tldw_chatbook/LLM_Calls/Summarization_General_Lib.py Tests/LLM_Calls/summarization_diagnostic_guard.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
../../.venv/bin/python -m ruff format tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py Tests/LLM_Calls/summarization_diagnostic_guard.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
../../.venv/bin/python -m ruff format --check tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py Tests/LLM_Calls/summarization_diagnostic_guard.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
../../.venv/bin/python -m ruff format tldw_chatbook/LLM_Calls/Summarization_General_Lib.py
```

Restore the one baseline statement with `apply_patch`, then run:

```bash
../../.venv/bin/python -m ruff format --check --diff tldw_chatbook/LLM_Calls/Summarization_General_Lib.py
```

Expected: nonzero with exactly the known baseline hunk. After comparing that output, run the remaining green gates separately:

```bash
../../.venv/bin/python -m py_compile tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py tldw_chatbook/LLM_Calls/Summarization_General_Lib.py Tests/LLM_Calls/summarization_diagnostic_guard.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py
git diff --check
```

Reconfirm the full-file diff is exactly that one unchanged baseline hunk and record the comparison; every other line has just passed the formatter.

- [x] **Step 4: Perform whole-branch scope and privacy self-review**

Inspect:

```bash
git diff --stat origin/dev...HEAD
git diff --name-status origin/dev...HEAD
git diff --check origin/dev...HEAD
git log --oneline origin/dev..HEAD
git status --short --branch
```

Review every production hunk for eager formatting, private response/path/key/exception values, traceback flags, accidental control-flow changes, and unrelated reviewed-safe rewrites.

- [x] **Step 5: Request an independent code review**

Use `@superpowers:requesting-code-review` with the exact `origin/dev...HEAD` range, approved spec/plan, inventory arithmetic, mutation evidence, focused test results, and known General formatter baseline. Fix and re-review every verified Critical/Important issue before closeout.

- [x] **Step 6: Close the task only after every gate meets its approved expectation**

Use the Backlog CLI to set Done and add concise implementation notes, then use `apply_patch` to preserve the detailed notes and check all four acceptance criteria. Notes must record:

- exact final `523 / 323 / 200 / 23 deleted` reconciliation;
- per-module/category and batch totals;
- direct sentinel and 12-mutation evidence;
- unchanged return/error/streaming contracts;
- exact focused test/static counts;
- two-owner-only manifest delta;
- ADR-029 and the no-new-ADR reason;
- any approved deviation and whether a genuine new lesson was added.

Set the design status to `implemented and verified` and check every completed plan step.

- [x] **Step 7: Commit exact closeout files and verify clean state**

```bash
git add "backlog/tasks/task-3796 - Remove-private-summarization-values-from-diagnostics.md" Docs/superpowers/specs/2026-08-10-task-3796-summarization-diagnostic-privacy-design.md Docs/superpowers/plans/2026-08-10-task-3796-summarization-diagnostic-privacy.md
git commit -m "docs(security): close TASK-3796 diagnostic privacy"
backlog task 3796 --plain
git diff --check origin/dev...HEAD
git status --short --branch
```

Expected: TASK-3796 Done with 4/4 criteria, implementation notes present, all plan steps checked, clean worktree, and current `origin/dev` an ancestor of HEAD.
