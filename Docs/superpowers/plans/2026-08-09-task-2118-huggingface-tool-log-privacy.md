# TASK-2118 HuggingFace Tool-Log Privacy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make HuggingFace request diagnostics metadata-only by logging allowlisted request fields and tool names without tool schemas or unrecognized payload values.

**Architecture:** Keep `chat_with_huggingface()` and the existing sensitive-request branch intact. Replace its two ordinary raw renderings with `safe_llm_request_payload_summary()` calls, then pin both call sites with one parameterized real-function Loguru test and two independent mutation checks. No new helper, abstraction, dependency, test application, or packaging boundary is needed.

**Tech Stack:** Python 3.11+, pytest, Loguru, Ruff, existing `safe_llm_request_payload_summary()` privacy helper, Backlog.md CLI.

---

## Context and decisions

- Approved spec: `Docs/superpowers/specs/2026-08-09-task-2118-huggingface-tool-log-privacy-design.md`
- Backlog task: `backlog/tasks/task-2118 - HuggingFace-tools-debug-log-dumps-full-tool-schemas.md`
- Governing decision: `backlog/decisions/029-local-private-data-boundary.md`
- Relevant testing lesson: `backlog/docs/lessons-testing-evidence.md` (Loguru requires a temporary sink; mutation evidence must prove the guard can fail)
- ADR required: no new ADR
- ADR path: `backlog/decisions/029-local-private-data-boundary.md`
- Reason: ADR-029 already excludes provider payloads and tool definitions from persistent logs while permitting bounded metadata such as tool names. This task applies that accepted contract.
- Latest-dev reconciliation: rebased onto `origin/dev` at `a33a6a6f8`; the two unsafe HuggingFace log calls and existing helper contract are unchanged. Fresh baselines are 68 passing privacy tests and 8 passing HuggingFace chat-function tests. Ruff lint and compilation are green. Both complete Python files have pre-existing formatter drift, while the three ranges this task will edit are formatter-clean; use range checks to avoid an unrelated whole-file rewrite.

## File map

- Modify `Tests/Chat/test_sensitive_llm_logging.py`: add two sentinels and one parameterized function-level regression test using the existing `_FakeSession` helper and `_captured_logs` context manager.
- Modify `tldw_chatbook/LLM_Calls/LLM_API_Calls.py`: route the two HuggingFace ordinary debug lines through the existing allowlist summary.
- Modify `backlog/tasks/task-2118 - HuggingFace-tools-debug-log-dumps-full-tool-schemas.md`: record the plan, checked acceptance criteria, sweep/mutation/test evidence, ADR decision, and final status.
- Keep `tldw_chatbook/Utils/sensitive_llm_logging.py` unchanged: its existing contract already produces the required output.

### Task 0: Reconcile the execution branch with current dev

**Files:**
- Verify: committed TASK-2118 spec, plan, and Backlog task record
- Re-inspect: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py:4341-4354`
- Re-run: focused baselines and the Task 2 logger inventory

- [ ] **Step 1: Require a clean, committed planning baseline**

Run:

```bash
git status --short
git log -1 --oneline
```

Expected: the working tree is clean and the checked-in plan/task record is at `HEAD`. Do not begin implementation with an untracked plan or modified Backlog record, because the later mutation-restoration check relies on a clean tree.

- [ ] **Step 2: Rebase onto the latest dev before touching tests or production**

Run:

```bash
git fetch origin dev
git rebase origin/dev
git merge-base --is-ancestor origin/dev HEAD
git status --short --branch
```

Expected: fetch/rebase succeeds, the ancestry command exits 0, and status shows only commits ahead of `origin/dev`. Resolve conflicts against the current privacy/helper contract; do not carry stale log code forward mechanically.

- [ ] **Step 3: Reconcile assumptions and refresh baselines**

Re-inspect the two HuggingFace log calls and `safe_llm_request_payload_summary()`, then run:

```bash
../../.venv/bin/python -B -m pytest Tests/Chat/test_sensitive_llm_logging.py -q
../../.venv/bin/python -B -m pytest Tests/Chat/test_chat_functions.py -q -k huggingface
../../.venv/bin/python -m ruff check tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_sensitive_llm_logging.py
../../.venv/bin/python -m ruff format --check --range=4341-4365 tldw_chatbook/LLM_Calls/LLM_API_Calls.py
../../.venv/bin/python -m ruff format --check --range=40-65 Tests/Chat/test_sensitive_llm_logging.py
../../.venv/bin/python -m ruff format --check --range=675-820 Tests/Chat/test_sensitive_llm_logging.py
```

Expected on the reviewed baseline: 68 privacy tests and 8 HuggingFace tests pass; lint and the three edited-range format checks pass. Run the Task 2 inventory as well and expect its reviewed 35-candidate classification. If current dev changes any defect, helper contract, count, range, or inventory category, stop and update/re-review the spec and plan before implementation.

### Task 1: Pin both HuggingFace logging leaks and apply the minimal call-site repair

**Files:**
- Modify: `Tests/Chat/test_sensitive_llm_logging.py:43-57,682-760`
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py:4341-4354`

- [ ] **Step 1: Add the failing real-function regression test**

Add `"TOOL-SCHEMA-ENUM-CANARY"` and `"HUGGINGFACE-USER-CANARY"` to `CANARIES`, then add this test beside the existing tool-summary and HuggingFace privacy tests:

```python
@pytest.mark.parametrize("sensitive", [False, True])
def test_huggingface_tool_logs_are_names_only(
    monkeypatch: pytest.MonkeyPatch,
    sensitive: bool,
) -> None:
    response_data = {
        "id": "hf-test",
        "choices": [{"message": {"content": "ok"}}],
    }
    session = _FakeSession(_FakeResponse(response_data))
    monkeypatch.setattr(
        cloud_adapters,
        "load_settings",
        lambda: {
            "huggingface_api": {
                "api_base_url": "https://hf.test/v1",
                "api_chat_path": "chat/completions",
                "api_retries": 0,
            }
        },
    )
    monkeypatch.setattr(cloud_adapters.requests, "Session", lambda: session)

    context = sensitive_llm_request() if sensitive else nullcontext()
    with _captured_logs() as logs, context:
        cloud_adapters.chat_with_huggingface(
            input_data=[{"role": "user", "content": "hello"}],
            api_key="key",
            model="org/model",
            streaming=False,
            user="HUGGINGFACE-USER-CANARY",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_hf_weather",
                        "description": "TOOL-SCHEMA-DESCRIPTION-CANARY",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "unit": {
                                    "type": "string",
                                    "enum": ["TOOL-SCHEMA-ENUM-CANARY"],
                                }
                            },
                        },
                    },
                }
            ],
        )

    _assert_canaries_absent(logs)
    tool_records = [entry for entry in logs if "HuggingFace Tools:" in entry]
    if sensitive:
        assert tool_records == []
    else:
        assert len(tool_records) == 1
        rendered_tools = tool_records[0].split("HuggingFace Tools: ", 1)[1].strip()
        assert rendered_tools == "{'tool_names': ['lookup_hf_weather']}"
```

This drives the production function directly; it does not compose a test or simplified application.

- [ ] **Step 2: Run the targeted test and verify RED**

Run:

```bash
../../.venv/bin/python -B -m pytest Tests/Chat/test_sensitive_llm_logging.py::test_huggingface_tool_logs_are_names_only -q
```

Expected: `1 failed, 1 passed`. The ordinary case must show the current raw tool-definition/user sentinels in captured logs; the sensitive case must pass because it already omits the tools line and raw payload.

- [ ] **Step 3: Replace only the two unsafe renderings**

Replace the ordinary HuggingFace logging block with:

```python
    else:
        logger.debug(
            "HuggingFace Final Payload (safe fields only): "
            f"{safe_llm_request_payload_summary(payload)}"
        )
    if "tools" in payload and not is_sensitive_llm_request():
        tools_summary = safe_llm_request_payload_summary(
            {"tools": payload["tools"]}, content_keys=()
        )
        logger.debug(f"HuggingFace Tools: {tools_summary}")
```

Do not change the sensitive metadata branch, helper implementation, request payload, transport, or response handling.

- [ ] **Step 4: Run the targeted test and verify GREEN**

Run the Step 2 command again.

Expected: `2 passed`.

- [ ] **Step 5: Run focused behavior and static gates**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Chat/test_sensitive_llm_logging.py -q
../../.venv/bin/python -m pytest Tests/Chat/test_chat_functions.py -q -k huggingface
../../.venv/bin/python -m ruff check tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_sensitive_llm_logging.py
../../.venv/bin/python -m ruff format --check --range=4341-4365 tldw_chatbook/LLM_Calls/LLM_API_Calls.py
../../.venv/bin/python -m ruff format --check --range=40-65 Tests/Chat/test_sensitive_llm_logging.py
../../.venv/bin/python -m ruff format --check --range=675-820 Tests/Chat/test_sensitive_llm_logging.py
../../.venv/bin/python -m py_compile tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_sensitive_llm_logging.py
git diff --check
```

Expected: 70 privacy tests pass after adding the two parameter cases; 8 HuggingFace tests pass; Ruff lint, all three edited-range format checks, compilation, and diff checks exit 0. Do not run Ruff formatting over either complete file because that would mix pre-existing formatter drift into this security repair.

- [ ] **Step 6: Commit the behavioral repair**

```bash
git add Tests/Chat/test_sensitive_llm_logging.py tldw_chatbook/LLM_Calls/LLM_API_Calls.py
git commit -m "fix(llm): redact HuggingFace tool logs"
```

### Task 2: Prove the sweep and both regression boundaries

**Files:**
- Inspect: `tldw_chatbook/LLM_Calls/**/*.py`
- Temporarily modify and restore: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py:4350-4360`
- Test: `Tests/Chat/test_sensitive_llm_logging.py::test_huggingface_tool_logs_are_names_only`

- [ ] **Step 1: Run the AST-assisted raw-payload/tool logger inventory**

Run this read-only inventory from the repository root:

```bash
../../.venv/bin/python - <<'PY'
import ast
from pathlib import Path

root = Path("tldw_chatbook/LLM_Calls")
methods = {"trace", "debug", "info", "warning", "error", "critical", "exception"}
owners = {"logger", "logging"}


def is_log_call(call: ast.Call) -> bool:
    func = call.func
    if not isinstance(func, ast.Attribute) or func.attr not in methods:
        return False
    value = func.value
    while isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
        value = value.func.value
    return isinstance(value, ast.Name) and value.id in owners


for path in sorted(root.rglob("*.py")):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not is_log_call(call):
            continue
        names = {
            node.id.lower() for node in ast.walk(call) if isinstance(node, ast.Name)
        }
        relevant = sorted(
            name
            for name in names
            if name == "data"
            or any(token in name for token in ("payload", "tool", "request_body"))
        )
        if relevant:
            segment = ast.get_source_segment(source, call) or ""
            print(
                f"{path}:{call.lineno}: names={','.join(relevant)} :: "
                f"{' '.join(segment.split())}"
            )
PY
```

Expected after the repair: 35 candidates across four modules, all inspected. Eleven request/tool candidates in `LLM_API_Calls.py` route through `safe_llm_request_payload_summary()` (the nine existing provider summaries plus the two corrected HuggingFace summaries). Fourteen are safe metadata-only calls: two HuggingFace model/stream/count/byte calls, one `LLM_API_Calls_Local.py` payload-**keys** call, and eleven `type(data)` calls. The remaining ten calls in `Local_Summarization_Lib.py` and `Summarization_General_Lib.py` log full or 500-character input data. They are confirmed privacy exposures but are individual content diagnostics, not AC 4 raw provider request-payload dictionaries/tool definitions; create a separate atomic Backlog task for them without adding it as a TASK-2118 dependency/reference, then record the classification (not a forward task ID) in TASK-2118 notes. There must be zero raw request-payload dictionary or raw tool-definition logs.

Corroborate label-based coverage with:

```bash
rg -n -i "logger\\.|logging\\." tldw_chatbook/LLM_Calls --glob '*.py' | rg -i "payload|tools?|loaded data|processed data"
```

Inspect every result in context. Record the classification in TASK-2118 Implementation Notes. If an additional AC 4 match exists, add a failing sentinel test and route it through the helper before continuing. Create the separately scoped Backlog task for the ten confirmed content diagnostics, but do not add its later task ID as a dependency/reference from TASK-2118; record the exact sites and classification here instead.

- [ ] **Step 2: Mutation-check the tool-definition guard**

Temporarily replace only the corrected `HuggingFace Tools` summary with the old line:

```python
        logger.debug(f"HuggingFace Tools: {payload['tools']}")
```

Run:

```bash
../../.venv/bin/python -B -m pytest Tests/Chat/test_sensitive_llm_logging.py::test_huggingface_tool_logs_are_names_only -vv
```

Expected: `1 failed, 1 passed`; inspect the traceback and require the ordinary (`sensitive=False`) case to fail on the planted description/schema or exact names-only assertion. Restore the helper-based implementation with `apply_patch` immediately and rerun the same cache-disabled command for `2 passed`.

- [ ] **Step 3: Mutation-check the Final Payload allowlist**

With the tools fix restored, temporarily replace only the corrected Final Payload summary with the prior denylist expression:

```python
        logger.debug(
            f"HuggingFace Final Payload (excluding messages, tools): {{ {', '.join(f'{k}: {v}' for k, v in payload.items() if k not in ['messages', 'tools'])} }}"
        )
```

Run the same `python -B ... -vv` targeted command.

Expected: `1 failed, 1 passed`; inspect the traceback and require the ordinary (`sensitive=False`) case to fail because `HUGGINGFACE-USER-CANARY` reappears. Restore the helper-based implementation with `apply_patch` immediately and rerun the same cache-disabled command for `2 passed`.

- [ ] **Step 4: Confirm no temporary mutation remains**

Run:

```bash
git diff --check
git diff HEAD -- tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_sensitive_llm_logging.py
git status --short
```

Expected: the production/test diff is empty and the working tree is clean; neither temporary mutation remains after the Task 1 commit.

### Task 3: Verify, document, and close TASK-2118

**Files:**
- Modify: `backlog/tasks/task-2118 - HuggingFace-tools-debug-log-dumps-full-tool-schemas.md`
- Verify: repository-wide tests and edited Python files

- [ ] **Step 1: Run the affected-module and full-project gates**

Run in the foreground and retain exact counts:

```bash
../../.venv/bin/python -m pytest Tests/Chat/test_sensitive_llm_logging.py -q
../../.venv/bin/python -m pytest Tests/Chat/test_chat_functions.py -q -k huggingface
../../.venv/bin/python -m pytest Tests/LLM_Calls/test_debug_log_fstring_hygiene.py -q
../../.venv/bin/python -m pytest -q
../../.venv/bin/python -m ruff check tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_sensitive_llm_logging.py
../../.venv/bin/python -m ruff format --check --range=4341-4365 tldw_chatbook/LLM_Calls/LLM_API_Calls.py
../../.venv/bin/python -m ruff format --check --range=40-65 Tests/Chat/test_sensitive_llm_logging.py
../../.venv/bin/python -m ruff format --check --range=675-820 Tests/Chat/test_sensitive_llm_logging.py
../../.venv/bin/python -m py_compile tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_sensitive_llm_logging.py
git diff --check
```

Expected: all gates exit 0. If the full suite is red, reproduce only its exact failing node IDs in a detached temporary worktree at current `origin/dev`; never check out `origin/dev` inside the active task worktree. Use this procedure after replacing the sample array entries with the exact node IDs from the branch run:

```bash
baseline_worktree="$(mktemp -d /tmp/task2118-origin-dev.XXXXXX)"
git worktree add --detach "$baseline_worktree" origin/dev
failing_nodes=(
  "Tests/exact_path.py::exact_test_name"
)
(
  cd "$baseline_worktree"
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest "${failing_nodes[@]}" -q
)
git worktree remove "$baseline_worktree"
```

Require identical node names and failure causes before classifying anything as baseline. If setup or collection fails instead of reproducing the branch assertion, that is not baseline evidence. Remove the temporary worktree through `git worktree remove` after the comparison.

- [ ] **Step 2: Complete Backlog evidence and status through the CLI**

Use `backlog task edit 2118` to:

- check acceptance criteria 1 through 4;
- add concise Implementation Notes covering the two helper-routed log sites, the 35-candidate sweep classification, both mutation failures, final test/static counts, no new dependency/helper, and the ADR-029 decision;
- state that the existing Loguru/mutation lesson applied and no new general lesson was discovered;
- set the task to `Done` only after every gate above is satisfied.

If the CLI canonicalizes the task filename, restore the original tracked filename with `apply_patch` so the closeout does not introduce an unrelated rename.

- [ ] **Step 3: Verify task hygiene and commit closeout**

Run:

```bash
backlog task 2118 --plain
git diff --check
git diff --stat
git status --short
```

Expected: status `Done`, 4/4 acceptance criteria checked, Implementation Plan and Implementation Notes present, and only TASK-2118 closeout documentation is uncommitted.

Commit:

```bash
git add 'backlog/tasks/task-2118 - HuggingFace-tools-debug-log-dumps-full-tool-schemas.md'
git commit -m "docs(security): close TASK-2118"
```

- [ ] **Step 4: Run fresh post-commit verification**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Chat/test_sensitive_llm_logging.py -q
../../.venv/bin/python -m pytest Tests/Chat/test_chat_functions.py -q -k huggingface
../../.venv/bin/python -m ruff check tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_sensitive_llm_logging.py
../../.venv/bin/python -m ruff format --check --range=4341-4365 tldw_chatbook/LLM_Calls/LLM_API_Calls.py
../../.venv/bin/python -m ruff format --check --range=40-65 Tests/Chat/test_sensitive_llm_logging.py
../../.venv/bin/python -m ruff format --check --range=675-820 Tests/Chat/test_sensitive_llm_logging.py
git diff --check
git status --short --branch
```

Expected: all focused/static gates pass, the worktree is clean, and `origin/dev` is an ancestor of `HEAD`.
