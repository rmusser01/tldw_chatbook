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
- Latest-dev reconciliation: the branch was rebased onto `origin/dev` at `f6911b37b`. No upstream commit changed the scoped production file, focused test files, or the two summarization modules since the prior reviewed base; only the testing-lessons document changed in scope. The pre-implementation baselines were 68 passing privacy tests and 8 passing HuggingFace chat-function tests. The identifier-filtered high-risk logger inventory remains 35 candidates across four modules; it is not the exhaustive AC 4 or summarization-content proof. The exhaustive source proof is the complete 763-call logger review plus the outbound-body/tool-structure correlation in Task 2. The final review additionally requires the complete 523-call summarization audit, the branch-owned diagnostic-manifest digest reconciliation, and exact latest-dev baseline evidence for unrelated manifest drift. Ruff lint and compilation are green. Both complete Python files have pre-existing formatter drift, while the three ranges this task edits (`4341-4365`, `40-70`, and `675-820`) are formatter-clean; use range checks to avoid an unrelated whole-file rewrite.

## File map

- Modify `Tests/Chat/test_sensitive_llm_logging.py`: add two sentinels and one parameterized function-level regression test using the existing `_FakeSession` helper and `_captured_logs` context manager.
- Modify `tldw_chatbook/LLM_Calls/LLM_API_Calls.py`: route the two HuggingFace ordinary debug lines through the existing allowlist summary.
- Modify `backlog/tasks/task-2118 - HuggingFace-tools-debug-log-dumps-full-tool-schemas.md`: record the plan, checked acceptance criteria, sweep/mutation/test evidence, ADR decision, and final status.
- Modify the separately generated summarization-privacy Backlog task: it owns the complete verified raw input, prompt, response/output, credential-fragment, private endpoint/path, and exception/error-detail boundary. Keep its backward TASK-2118 incident reference, but do not copy its later task ID into TASK-2118.
- Modify `Docs/security/production-diagnostic-inventory.json`: update only the reviewed `LLM_API_Calls.py` diagnostic digest while preserving its owner, reason, call count, every unrelated entry, and sink topology.
- Modify `backlog/docs/lessons-testing-evidence.md`: record why a heuristic candidate list must not be promoted into a complete remediation inventory.
- Keep `tldw_chatbook/Utils/sensitive_llm_logging.py` unchanged: its existing contract already produces the required output.

### Task 0: Reconcile the execution branch with current dev

**Files:**
- Verify: committed TASK-2118 spec, plan, and Backlog task record
- Re-inspect: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py:4341-4354`
- Re-run: focused baselines and the Task 2 logger inventory

- [x] **Step 1: Require a clean, committed planning baseline**

Run:

```bash
git status --short
git log -1 --oneline
```

Expected: the working tree is clean and the checked-in plan/task record is at `HEAD`. Do not begin implementation with an untracked plan or modified Backlog record, because the later mutation-restoration check relies on a clean tree.

- [x] **Step 2: Rebase onto the latest dev before touching tests or production**

Run:

```bash
git fetch origin dev
git rebase origin/dev
git merge-base --is-ancestor origin/dev HEAD
git status --short --branch
```

Expected: fetch/rebase succeeds, the ancestry command exits 0, and status shows only commits ahead of `origin/dev`. Resolve conflicts against the current privacy/helper contract; do not carry stale log code forward mechanically.

- [x] **Step 3: Reconcile assumptions and refresh baselines**

Re-inspect the two HuggingFace log calls and `safe_llm_request_payload_summary()`, then run:

```bash
../../.venv/bin/python -B -m pytest Tests/Chat/test_sensitive_llm_logging.py -q
../../.venv/bin/python -B -m pytest Tests/Chat/test_chat_functions.py -q -k huggingface
../../.venv/bin/python -m ruff check tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_sensitive_llm_logging.py
../../.venv/bin/python -m ruff format --check --range=4341-4365 tldw_chatbook/LLM_Calls/LLM_API_Calls.py
../../.venv/bin/python -m ruff format --check --range=40-70 Tests/Chat/test_sensitive_llm_logging.py
../../.venv/bin/python -m ruff format --check --range=675-820 Tests/Chat/test_sensitive_llm_logging.py
```

Expected on the reviewed baseline: 68 privacy tests and 8 HuggingFace tests pass; lint and the three edited-range format checks pass. Run the Task 2 inventory as well and expect its reviewed 35-candidate classification. If current dev changes any defect, helper contract, count, range, or inventory category, stop and update/re-review the spec and plan before implementation.

### Task 1: Pin both HuggingFace logging leaks and apply the minimal call-site repair

**Files:**
- Modify: `Tests/Chat/test_sensitive_llm_logging.py:43-57,682-760`
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py:4341-4354`

- [x] **Step 1: Add the failing real-function regression test**

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

- [x] **Step 2: Run the targeted test and verify RED**

Run:

```bash
../../.venv/bin/python -B -m pytest Tests/Chat/test_sensitive_llm_logging.py::test_huggingface_tool_logs_are_names_only -q
```

Expected: `1 failed, 1 passed`. The ordinary case must show the current raw tool-definition/user sentinels in captured logs; the sensitive case must pass because it already omits the tools line and raw payload.

- [x] **Step 3: Replace only the two unsafe renderings**

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

- [x] **Step 4: Run the targeted test and verify GREEN**

Run the Step 2 command again.

Expected: `2 passed`.

- [x] **Step 5: Run focused behavior and static gates**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Chat/test_sensitive_llm_logging.py -q
../../.venv/bin/python -m pytest Tests/Chat/test_chat_functions.py -q -k huggingface
../../.venv/bin/python -m ruff check tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_sensitive_llm_logging.py
../../.venv/bin/python -m ruff format --check --range=4341-4365 tldw_chatbook/LLM_Calls/LLM_API_Calls.py
../../.venv/bin/python -m ruff format --check --range=40-70 Tests/Chat/test_sensitive_llm_logging.py
../../.venv/bin/python -m ruff format --check --range=675-820 Tests/Chat/test_sensitive_llm_logging.py
../../.venv/bin/python -m py_compile tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_sensitive_llm_logging.py
git diff --check
```

Expected: 70 privacy tests pass after adding the two parameter cases; 8 HuggingFace tests pass; Ruff lint, all three edited-range format checks, compilation, and diff checks exit 0. Do not run Ruff formatting over either complete file because that would mix pre-existing formatter drift into this security repair.

- [x] **Step 6: Commit the behavioral repair**

```bash
git add Tests/Chat/test_sensitive_llm_logging.py tldw_chatbook/LLM_Calls/LLM_API_Calls.py
git commit -m "fix(llm): redact HuggingFace tool logs"
```

### Task 2: Prove the sweep and both regression boundaries

**Files:**
- Inspect: `tldw_chatbook/LLM_Calls/**/*.py`
- Temporarily modify and restore: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py:4350-4360`
- Test: `Tests/Chat/test_sensitive_llm_logging.py::test_huggingface_tool_logs_are_names_only`

- [x] **Step 1: Run the identifier-filtered high-risk candidate inventory**

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

Expected after the repair: exactly 35 high-risk candidates across four modules: `LLM_API_Calls.py` 13, `LLM_API_Calls_Local.py` 1, `Local_Summarization_Lib.py` 15, and `Summarization_General_Lib.py` 6. Eleven request/tool candidates in `LLM_API_Calls.py` route through `safe_llm_request_payload_summary()` (the nine existing provider summaries plus the two corrected HuggingFace summaries). Fourteen are safe metadata-only calls: two HuggingFace model/stream/count/byte calls, one `LLM_API_Calls_Local.py` payload-**keys** call, and eleven `type(data)` calls. The remaining candidate subset contains real content diagnostics, but it is not a complete summarization privacy inventory. A separate review of all 523 logger calls in both summarization modules owns that conclusion: 199 direct private diagnostics, split as 100 in `Local_Summarization_Lib.py` (13 input, 8 prompt, 8 credential-fragment, 6 private endpoint/path, 29 response/output, 36 exception/error-detail) and 99 in `Summarization_General_Lib.py` (8 input, 9 prompt, 13 credential-fragment, 5 private endpoint/path, 42 response/output, 22 exception/error-detail). These are outside AC 4's raw provider request-dictionary/tool-definition scope and are durably assigned to the separate follow-up without a forward task-ID reference.

This 35-site list is deliberately only a high-risk candidate inventory. Its identifier spelling filter can miss a body called `request_data`, `data2`, `retry_payload`, or an unrelated alias, so it must not be presented as exhaustive or authoritative AC 4 evidence. Inspect every result in context, retain its classification, and then complete Steps 2 and 3.

- [x] **Step 2: Enumerate and review every logger call in `LLM_Calls`**

Run this syntax-complete enumeration for calls rooted at the repository's `logger` or `logging` owners. `owner_root()` follows attribute and call chains, so `logger.opt(...).error(...)` and `logging.getLogger(...).debug(...)` are included:

```bash
../../.venv/bin/python -B - <<'PY'
import ast
from collections import Counter
from pathlib import Path

root = Path("tldw_chatbook/LLM_Calls")
methods = {"trace", "debug", "info", "warning", "error", "critical", "exception"}
owners = {"logger", "logging"}


def owner_root(expr: ast.AST) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return owner_root(expr.value)
    if isinstance(expr, ast.Call):
        return owner_root(expr.func)
    return None


rows: list[tuple[str, int, str, str]] = []
for path in sorted(root.rglob("*.py")):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr not in methods or owner_root(call.func.value) not in owners:
            continue
        segment = " ".join((ast.get_source_segment(source, call) or "").split())
        rows.append((str(path), call.lineno, call.func.attr, segment))

for path, line, method, segment in sorted(rows):
    print(f"{path}:{line}:{method}: {segment}")
counts = Counter(path for path, *_ in rows)
for path, count in sorted(counts.items()):
    print(f"COUNT {path}: {count}")
print(f"TOTAL: {len(rows)}")
PY
```

Expected reviewed inventory: 763 calls across eight modules:

- `LLM_API_Calls.py`: 171
- `LLM_API_Calls_Local.py`: 41
- `Local_Summarization_Lib.py`: 242
- `Summarization_General_Lib.py`: 281
- `huggingface_api.py`: 9
- `pricing_catalog.py`: 4
- `realtime/openai_session.py`: 13
- `realtime/transport.py`: 2

Review every printed source expression at its printed location, including the logger calls that do not contain conventional payload/tool identifiers. Classify whether it can render a provider request body, a tool/schema/definition structure, bounded metadata, response/error data, or unrelated diagnostics. The reviewed result after Task 1 is no additional raw provider request-body dictionary or raw tool/schema/definition structure. Existing raw summarization input/prompt/response diagnostics are not evidence that this narrower AC 4 boundary is clean; preserve their separate ownership and do not silently classify them as safe.

- [x] **Step 3: Correlate outbound bodies and tool/schema expressions independently of the 35-site filter**

Run this function-scoped correlation. It discovers outbound `json=`, `data=`, and `content=` expressions on request/post/put/patch/stream/send-like calls without filtering their identifier spellings. It prints every logger call in each request-containing function, marks exact body-expression name overlap, and follows only simple same-scope `alias = name` and `alias = name.copy()` relationships. That alias pass is deliberately flow-insensitive and conservative; manual review must resolve reassignments and it must not be described as general data-flow proof.

```bash
../../.venv/bin/python -B - <<'PY'
import ast
from collections import Counter, defaultdict
from pathlib import Path

root = Path("tldw_chatbook/LLM_Calls")
log_methods = {"trace", "debug", "info", "warning", "error", "critical", "exception"}
http_methods = {"request", "post", "put", "patch", "stream", "send"}
body_keys = {"json", "data", "content"}


def owner_root(expr: ast.AST) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return owner_root(expr.value)
    if isinstance(expr, ast.Call):
        return owner_root(expr.func)
    return None


def is_log_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in log_methods
        and owner_root(node.func.value) in {"logger", "logging"}
    )


def call_leaf(call: ast.Call) -> str:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return ""


def names(node: ast.AST) -> set[str]:
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


def target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Starred):
        return target_names(node.value)
    if isinstance(node, (ast.Tuple, ast.List)):
        return set().union(*(target_names(item) for item in node.elts))
    return set()


body_calls = request_scopes = logs_in_request_scopes = 0
keyword_counts: Counter[str] = Counter()
correlated: set[tuple[str, int]] = set()
exact: set[tuple[str, int]] = set()
alias_only: set[tuple[str, int]] = set()

for path in sorted(root.rglob("*.py")):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    def scope_of(node: ast.AST) -> str:
        parts: list[str] = []
        parent = parents.get(node)
        while parent is not None:
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parts.append(parent.name)
            parent = parents.get(parent)
        return ".".join(reversed(parts)) or "<module>"

    scopes: defaultdict[str, list[ast.AST]] = defaultdict(list)
    for node in ast.walk(tree):
        scopes[scope_of(node)].append(node)

    for scope, nodes in sorted(scopes.items()):
        bodies: list[tuple[ast.Call, list[ast.keyword]]] = []
        alias_edges: list[tuple[str, str]] = []
        for node in nodes:
            if isinstance(node, ast.Call) and call_leaf(node) in http_methods:
                keywords = [kw for kw in node.keywords if kw.arg in body_keys]
                if keywords:
                    bodies.append((node, keywords))
                    keyword_counts.update(kw.arg for kw in keywords if kw.arg)
            if isinstance(node, ast.Assign):
                targets = set().union(*(target_names(item) for item in node.targets))
                direct_name = isinstance(node.value, ast.Name)
                copied_name = (
                    isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr in {"copy", "deepcopy"}
                )
                if direct_name or copied_name:
                    alias_edges.extend((left, right) for left in targets for right in names(node.value))

        if not bodies:
            continue
        request_scopes += 1
        body_calls += len(bodies)
        body_names = set().union(
            *(names(keyword.value) for _, keywords in bodies for keyword in keywords)
        )
        aliases = set(body_names)
        changed = True
        while changed:
            before = len(aliases)
            for left, right in alias_edges:
                if left in aliases or right in aliases:
                    aliases.update((left, right))
            changed = len(aliases) != before

        print(f"SCOPE {path}::{scope}")
        for call, keywords in sorted(bodies, key=lambda row: row[0].lineno):
            rendered = ", ".join(
                f"{keyword.arg}={ast.unparse(keyword.value)}" for keyword in keywords
            )
            print(f"  BODY {call.lineno}: {ast.unparse(call.func)} {rendered}")
        print(f"  BODY_NAMES={sorted(body_names)} SIMPLE_ALIASES={sorted(aliases)}")
        for call in sorted((node for node in nodes if is_log_call(node)), key=lambda node: node.lineno):
            logs_in_request_scopes += 1
            overlap = names(call) & aliases
            exact_overlap = names(call) & body_names
            site = (str(path), call.lineno)
            if overlap:
                correlated.add(site)
                (exact if exact_overlap else alias_only).add(site)
            segment = " ".join((ast.get_source_segment(source, call) or "").split())
            print(
                f"  LOG {call.lineno}: exact={sorted(exact_overlap)} "
                f"alias={sorted(overlap - body_names)} :: {segment}"
            )

print(f"BODY_CALLS: {body_calls}")
print(f"REQUEST_SCOPES: {request_scopes}")
print(f"BODY_KEYWORDS: {dict(sorted(keyword_counts.items()))}")
print(f"LOGS_IN_REQUEST_SCOPES: {logs_in_request_scopes}")
print(f"CORRELATED_LOGS: {len(correlated)}")
print(f"EXACT_LOGS: {len(exact)}")
print(f"SIMPLE_ALIAS_ONLY_LOGS: {len(alias_only)}")
PY
```

Expected result: 57 outbound body-bearing calls across 33 function scopes, with 55 `json=` bodies, two `data=` bodies, and zero `content=` bodies. The script prints all 554 logger calls in those scopes to make the function/body correlation reproducible, but Step 2 already reviewed those expressions as part of the complete 763-call inventory; do not repeat that manual review. Instead, manually inspect all 57 body constructions and their assignment/reassignment flow plus every one of the 41 marked sites. Forty-one unique log sites correlate with an exact body-expression name or the limited simple-alias set: 27 exact and 14 alias-only, distributed as `LLM_API_Calls.py` 9, `LLM_API_Calls_Local.py` 1, `Local_Summarization_Lib.py` 20, and `Summarization_General_Lib.py` 11. The nine cloud request-body matches use the safe allowlist helper; the local adapter match logs keys only. The remaining marked summarization sites are type/model/path metadata or input/prompt previews emitted before the same generic variable name is reassigned to the outbound dictionary; none renders the constructed provider request dictionary.

Run a separate lexical tool/schema/definition scan over every logging expression:

```bash
../../.venv/bin/python -B - <<'PY'
import ast
from pathlib import Path

root = Path("tldw_chatbook/LLM_Calls")
methods = {"trace", "debug", "info", "warning", "error", "critical", "exception"}
tokens = {"tool", "schema", "definition"}


def owner_root(expr: ast.AST) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return owner_root(expr.value)
    if isinstance(expr, ast.Call):
        return owner_root(expr.func)
    return None


rows: list[tuple[str, int, list[str], str]] = []
for path in sorted(root.rglob("*.py")):
    source = path.read_text(encoding="utf-8")
    for call in ast.walk(ast.parse(source, filename=str(path))):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr not in methods or owner_root(call.func.value) not in {"logger", "logging"}:
            continue
        values: list[str] = []
        for node in ast.walk(call):
            if isinstance(node, ast.Name):
                values.append(node.id)
            elif isinstance(node, ast.Attribute):
                values.append(node.attr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                values.append(node.value)
        matched = sorted(token for token in tokens if any(token in value.lower() for value in values))
        if matched:
            segment = " ".join((ast.get_source_segment(source, call) or "").split())
            rows.append((str(path), call.lineno, matched, segment))

for path, line, matched, segment in sorted(rows):
    print(f"{path}:{line}: terms={matched} :: {segment}")
print(f"TOTAL: {len(rows)}")
PY
```

Expected: ten expressions, all in `LLM_API_Calls.py`: eight constant validation/event warnings, one Google diagnostic containing only a `tool_call_id`, and the corrected HuggingFace tool summary containing names only. There are zero lexical `schema` or `definition` matches and zero raw tool/schema/definition structures. This lexical scan can still miss a generically named runtime value, which is why it supplements rather than replaces the complete 763-call source review and request-body correlation.

These scripts prove the reviewed Python syntax under `LLM_Calls` for calls rooted at `logger`/`logging` and for the listed HTTP method/keyword shapes. They do not prove dynamic runtime values, custom logger aliases, positional/custom transport bodies, or generated code. Record this boundary exactly; the AC 4 conclusion is the combined result of the complete logger review, body/tool correlation, real-function sentinel, and independent mutations—not any single heuristic scan. If an additional AC 4 match exists, add a failing sentinel test and route it through the helper before continuing.

- [x] **Step 4: Verify the separate privacy follow-up is committed before mutation checks**

Run:

```bash
rg -l '^title: Remove raw summarization input data from debug logs$' backlog/tasks
git log -1 --oneline --all -- 'backlog/tasks/*Remove-raw-summarization-input-data-from-debug-logs.md'
git status --short
```

Expected: the title search returns exactly one generated task file, the log command shows the dedicated planning/follow-up commit, and the working tree is clean. The follow-up's description points backward to TASK-2118; TASK-2118 does not point forward to its later ID.

- [x] **Step 5: Mutation-check the tool-definition guard**

Temporarily replace only the corrected `HuggingFace Tools` summary with the old line:

```python
        logger.debug(f"HuggingFace Tools: {payload['tools']}")
```

Run:

```bash
../../.venv/bin/python -B -m pytest Tests/Chat/test_sensitive_llm_logging.py::test_huggingface_tool_logs_are_names_only -vv
```

Expected: `1 failed, 1 passed`; inspect the traceback and require the ordinary (`sensitive=False`) case to fail on the planted description/schema or exact names-only assertion. Restore the helper-based implementation with `apply_patch` immediately and rerun the same cache-disabled command for `2 passed`.

- [x] **Step 6: Mutation-check the Final Payload allowlist**

With the tools fix restored, temporarily replace only the corrected Final Payload summary with the prior denylist expression:

```python
        logger.debug(
            f"HuggingFace Final Payload (excluding messages, tools): {{ {', '.join(f'{k}: {v}' for k, v in payload.items() if k not in ['messages', 'tools'])} }}"
        )
```

Run the same `python -B ... -vv` targeted command.

Expected: `1 failed, 1 passed`; inspect the traceback and require the ordinary (`sensitive=False`) case to fail because `HUGGINGFACE-USER-CANARY` reappears. Restore the helper-based implementation with `apply_patch` immediately and rerun the same cache-disabled command for `2 passed`.

- [x] **Step 7: Confirm no temporary mutation remains**

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
- Verify: touched-file and affected-functionality tests plus edited Python static checks

- [x] **Step 1: Run the touched-file and affected-functionality gates**

Run in the foreground and retain exact counts:

```bash
../../.venv/bin/python -m pytest Tests/Chat/test_sensitive_llm_logging.py -q
../../.venv/bin/python -m pytest Tests/Chat/test_chat_functions.py -q -k huggingface
../../.venv/bin/python -m pytest Tests/LLM_Calls/test_debug_log_fstring_hygiene.py -q
../../.venv/bin/python -m pytest Tests/Architecture/test_persistent_diagnostic_inventory.py -q
../../.venv/bin/python -m ruff check tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_sensitive_llm_logging.py
../../.venv/bin/python -m ruff format --check --range=4341-4365 tldw_chatbook/LLM_Calls/LLM_API_Calls.py
../../.venv/bin/python -m ruff format --check --range=40-70 Tests/Chat/test_sensitive_llm_logging.py
../../.venv/bin/python -m ruff format --check --range=675-820 Tests/Chat/test_sensitive_llm_logging.py
../../.venv/bin/python -m py_compile tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_sensitive_llm_logging.py
git diff --check
```

Expected: the privacy, HuggingFace, hygiene, and branch-owned manifest gates exit 0. Per the user's explicit closeout scope, only tests related to the edited files and affected HuggingFace/logging functionality are completion gates. Do not run a repository-wide test suite, compose a test or simplified application, or perform detached-baseline replay for unrelated failures. The sole exception is the final-review requirement to reproduce an unchanged diagnostic-inventory failure on the exact current-dev commit when unrelated stored entries remain stale after the branch-owned digest is reconciled.

- [x] **Step 2: Complete Backlog evidence and status through the CLI**

Use `backlog task edit 2118` to:

- check acceptance criteria 1 through 4;
- add concise Implementation Notes covering the two helper-routed log sites and the complete combined AC 4 evidence: all 763 logger calls across eight modules reviewed; 57 body-bearing calls across 33 function scopes (`json=` 55, `data=` 2, `content=` 0); all 57 body construction/reassignment flows reviewed; 41 correlations (27 exact and 14 limited simple-alias) classified as nine safe helper summaries, one keys-only local diagnostic, and 31 summarization metadata/input-preview diagnostics that do not render the constructed request dictionary; ten lexical tool/schema/definition sites classified as eight constant warnings, one tool-call ID, and one HuggingFace names-only summary; and the static-analysis limitations for dynamic values, custom logger aliases, positional/custom transports, and generated code;
- retain the 35-candidate high-risk inventory and its 11 helper-routed / 14 metadata-only / remaining content-candidate classification, explicitly as a heuristic supplement rather than the sole AC 4 evidence;
- record the independent all-call summarization audit: all 523 logger calls reviewed and 199 direct private diagnostics assigned by module, enclosing function, diagnostic label, and category (100 local and 99 general; 21 input, 17 prompt, 21 credential-fragment, 11 private endpoint/path, 71 response/output, and 58 exception/error-detail);
- record the real-function sentinel plus both independent mutation failures and restored green counts, final test/static counts, no new dependency/helper, and the ADR-029 decision;
- state that the existing Loguru/mutation lesson applied and record the final-review incident in the testing-evidence lesson: a deliberately heuristic candidate list was incorrectly promoted into a complete follow-up inventory;
- set the task to `Done` only after every gate above is satisfied.

If the CLI canonicalizes the task filename, restore the original tracked filename with `apply_patch` so the closeout does not introduce an unrelated rename.

- [x] **Step 3: Verify task hygiene and commit closeout**

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

- [x] **Step 4: Run fresh post-commit verification**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Chat/test_sensitive_llm_logging.py -q
../../.venv/bin/python -m pytest Tests/Chat/test_chat_functions.py -q -k huggingface
../../.venv/bin/python -m pytest Tests/LLM_Calls/test_debug_log_fstring_hygiene.py -q
../../.venv/bin/python -m pytest Tests/Architecture/test_persistent_diagnostic_inventory.py -q
../../.venv/bin/python -m ruff check tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_sensitive_llm_logging.py
../../.venv/bin/python -m ruff format --check --range=4341-4365 tldw_chatbook/LLM_Calls/LLM_API_Calls.py
../../.venv/bin/python -m ruff format --check --range=40-70 Tests/Chat/test_sensitive_llm_logging.py
../../.venv/bin/python -m ruff format --check --range=675-820 Tests/Chat/test_sensitive_llm_logging.py
git diff --check
git status --short --branch
git merge-base --is-ancestor f6911b37b HEAD
```

Expected: all branch-owned focused/static gates pass, any diagnostic-inventory red is identical to the separately owned `f6911b37b` baseline, the worktree is clean, and `f6911b37b` is an ancestor of `HEAD`. A later PR integration pass must fetch/rebase current `origin/dev` again rather than treating this recorded commit as permanently latest.

### Task 4: Reconcile final whole-branch privacy and inventory review

**Files:**
- Modify: `Docs/security/production-diagnostic-inventory.json`
- Modify: `backlog/tasks/task-2118 - HuggingFace-tools-debug-log-dumps-full-tool-schemas.md`
- Modify: the separately filed summarization-privacy task
- Modify: `backlog/docs/lessons-testing-evidence.md`
- Verify: all 523 logger calls in the two summarization modules, the focused privacy tests, and the diagnostic-inventory architecture test

- [x] **Step 1: Reconcile the branch-owned diagnostic manifest entry**

Hand-review the `LLM_API_Calls.py` diagnostic change and update only its stored digest. Compare the complete generated inventory with the stored artifact; if unrelated latest-dev drift remains, reproduce the exact architecture failure on a detached worktree at `f6911b37b`, preserve every unrelated entry, and find or create an independently owned Backlog record.

- [x] **Step 2: Replace the heuristic follow-up with the complete summarization audit**

Review every logger call in `Local_Summarization_Lib.py` and `Summarization_General_Lib.py`. Record the stable module/function/label/category inventory in the separate To Do task, broaden its title/description/acceptance criteria without adding an implementation plan, and keep only the backward TASK-2118 incident reference.

- [x] **Step 3: Correct TASK-2118 evidence and the generalized testing lesson**

Remove the false implication that the identifier-filtered candidate subset was a complete follow-up inventory. Record the complete categorized ownership, the branch-only manifest digest, the exact latest-dev baseline drift, and the incident-based lesson about promoting heuristic evidence beyond its stated limits.

- [x] **Step 4: Run only the related completion gates and close the task**

Run the 70-test privacy module, 8-test HuggingFace subset, two-test log-f-string hygiene module, eight-test diagnostic-inventory architecture file, Ruff lint on the edited Python files, the three current edited-range format checks, `py_compile`, all Task 2 sweep/inventory scripts, and diff checks. Do not run the repository-wide suite, a test application, or a simplified application. Mark TASK-2118 Done only when branch-owned gates are green or an architecture failure is proven byte-for-byte identical to the owned latest-dev baseline drift.
