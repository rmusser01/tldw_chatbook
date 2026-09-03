# TASK-26944 Ruff Chat Citations Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove Ruff 0.15.22 formatter debt from the three `ruff-chat-citations` paths while proving Python semantics and comment attachment are unchanged.

**Architecture:** Reconcile the manifest-owned paths after a fresh `origin/dev` rebase, capture pre-format structure with a one-time standard-library guard in `/tmp`, and format only the reconciled owned set. Compare post-format structure, run direct citation tests and repository governance checks, then close the Backlog task with exact evidence.

**Tech Stack:** Python 3.12.11 standard library (`ast`, `tokenize`, `json`, `hashlib`), Ruff 0.15.22, pytest, Git, Backlog.md CLI

---

## File Map

- Modify: `Tests/Chat/test_citation_service_factory.py` — Ruff-generated layout only.
- Modify: `Tests/Chat/test_citation_trace_builder.py` — Ruff-generated layout only while retaining seven inline `# type: ignore[...]` directives.
- Modify: `tldw_chatbook/Chat/citation_trace_builder.py` — Ruff-generated production layout only.
- Modify: `backlog/tasks/task-26944 - Clean Ruff formatter debt for ruff-chat-citations.md` — plan, acceptance evidence, and closeout state.
- Create: `Docs/superpowers/plans/2026-09-02-task-26944-ruff-chat-citations.md` — executable implementation plan.
- Ephemeral only: `/tmp/task26944_paths.txt`, `/tmp/task26944_format_guard.py`, and `/tmp/task26944_before.json` — reconciled input manifest and uncommitted invariant capture.

No production helper, dependency, test module, schema, or ADR is added.

### Task 1: Rebase, Reconcile, and Capture the Baseline

**Files:**

- Inspect: `Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json`
- Inspect: the three assigned Python paths
- Create outside repository: `/tmp/task26944_paths.txt`
- Create outside repository: `/tmp/task26944_format_guard.py`
- Create outside repository: `/tmp/task26944_before.json`

- [ ] **Step 1: Verify the pinned toolchain**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c 'import sys; assert sys.version_info[:3] == (3, 12, 11), sys.version'
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c 'import subprocess; actual = subprocess.check_output(["/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python", "-m", "ruff", "--version"], text=True).strip(); assert actual == "ruff 0.15.22", actual'
```

Expected: both commands exit 0. Any mismatch blocks capture and formatting.

- [ ] **Step 2: Refresh and rebase the isolated branch**

Run:

```bash
git status --short --branch
git fetch origin dev
git rebase origin/dev
git rev-parse origin/dev
git merge-base --is-ancestor origin/dev HEAD
```

Expected: the worktree is clean before the fetch; rebase succeeds; the final
ancestry command exits 0. Record the fetched `origin/dev` SHA. Resolve only
documentation conflicts owned by this task; stop on any Python-path conflict.

- [ ] **Step 3: Reconcile the three manifest paths and freeze one input list**

Run:

```bash
git diff --name-status --find-renames e555df102c950c29beed5e7119f433d35eee1f3c HEAD
git ls-tree -r --name-only HEAD -- Tests/Chat/test_citation_service_factory.py Tests/Chat/test_citation_trace_builder.py tldw_chatbook/Chat/citation_trace_builder.py
```

The unrestricted rename-aware diff is required so a destination outside the three
original path names cannot be missed. For any rename candidate, confirm lineage
from the original blob/history and search all `TASK-26933`–`TASK-27015` task files
for competing destination ownership.

After reconciliation, create `/tmp/task26944_paths.txt` with `apply_patch`: one
sorted, present effective-owned path per line. On the approved baseline it must
contain exactly the original three paths. All later capture, Ruff, diff, and stage
commands consume this one list. If the list changes, amend this plan and the task
with the lineage and revised expected counts before continuing. If it is empty,
stop and amend the task as a reconciled no-op; never invoke Ruff with no path.

Run:

```bash
test -s /tmp/task26944_paths.txt
xargs /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check < /tmp/task26944_paths.txt
```

Expected on the approved baseline: the manifest has three lines and Ruff reports
that all three would be reformatted.

- [ ] **Step 4: Reconfirm the focused baseline**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_citation_service_factory.py Tests/Chat/test_citation_trace_builder.py -q
```

Expected on the approved no-drift baseline: 57 tests pass. These test paths are
the present test members of the reconciled list; amend the plan before proceeding
if reconciliation changes either test path. Record warnings separately from the
exit status.

- [ ] **Step 5: Build the ephemeral invariant guard**

Create `/tmp/task26944_format_guard.py` with no third-party imports and two CLI
operations: `capture OUTPUT PATH...` and `compare BASELINE PATH...`. The capture
must emit stable, sorted JSON containing, per path:

- SHA-256 of the source bytes;
- the `ast.dump(..., include_attributes=False)` after parsing with
  `type_comments=True` and recursively replacing only every
  `ast.TypeIgnore.lineno` with zero;
- all `tokenize.COMMENT` strings in source order;
- for inline `# noqa`, `# type: ignore`, and `# ruff:` directives, the deepest
  spanning AST node's field/index route plus the number of significant tokens
  preceding the comment in its logical statement;
- for standalone Ruff directives, the preceding and following `ast.stmt`
  field/index routes; and
- for each non-nested `# fmt: off` / `# fmt: on` pair, the ordered field/index
  routes of every enclosed AST node.

Treat `ENCODING`, `NL`, `NEWLINE`, `INDENT`, `DEDENT`, `COMMENT`, and `ENDMARKER`
as non-significant tokens. Fail on parse/tokenize errors, a missing or non-unique
deepest anchor, or unmatched/nested format ranges. `compare` must recapture the
current paths, ignore only source SHA-256, and exit nonzero with a per-field
diagnostic if any structural/comment value differs.

- [ ] **Step 6: Capture and sanity-check the baseline**

Run:

```bash
test -s /tmp/task26944_paths.txt
xargs /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python /tmp/task26944_format_guard.py capture /tmp/task26944_before.json < /tmp/task26944_paths.txt
xargs /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python /tmp/task26944_format_guard.py compare /tmp/task26944_before.json < /tmp/task26944_paths.txt
```

Expected: both commands exit 0; the second reports all three paths structurally
identical to the capture.

### Task 2: Apply Ruff and Prove the Change

**Files:**

- Modify: `Tests/Chat/test_citation_service_factory.py`
- Modify: `Tests/Chat/test_citation_trace_builder.py`
- Modify: `tldw_chatbook/Chat/citation_trace_builder.py`

- [ ] **Step 1: Format exactly the reconciled owned set**

Run:

```bash
test -s /tmp/task26944_paths.txt
xargs /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format < /tmp/task26944_paths.txt
```

Expected: Ruff reports three reformatted files. Do not hand-edit the result.

- [ ] **Step 2: Compare semantic and comment evidence**

Run:

```bash
test -s /tmp/task26944_paths.txt
xargs /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python /tmp/task26944_format_guard.py compare /tmp/task26944_before.json < /tmp/task26944_paths.txt
```

Expected: exit 0 with equal AST, ordered comments, directive anchors, standalone
directive placement, and format-range intervals for every path. On failure, stop;
restore only the listed paths with
`xargs git restore --source=HEAD -- < /tmp/task26944_paths.txt`, then investigate
without editing around the guard.

- [ ] **Step 3: Prove the Python diff is exactly owned**

Run:

```bash
diff -u /tmp/task26944_paths.txt <(git diff --name-only -- '*.py')
xargs git diff --stat -- < /tmp/task26944_paths.txt
```

Expected on the approved baseline: the sorted changed-Python output equals the
three-line reconciled manifest. If reconciliation found an already-clean path,
instead require the changed-Python output to be a subset of the manifest and
record why equality is not expected.

- [ ] **Step 4: Run Ruff verification**

Run:

```bash
test -s /tmp/task26944_paths.txt
xargs /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check < /tmp/task26944_paths.txt
xargs /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check < /tmp/task26944_paths.txt
```

Expected: lint passes and Ruff reports three files already formatted.

- [ ] **Step 5: Run focused and governance tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_citation_service_factory.py Tests/Chat/test_citation_trace_builder.py -q
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/CI/test_backlog_task_id_uniqueness.py -q
git diff --check
```

Expected: 57 focused tests pass; the Backlog uniqueness test passes; Git reports
no whitespace errors.

- [ ] **Step 6: Self-review the formatter-only diff**

Run:

```bash
xargs git diff --word-diff=porcelain -- < /tmp/task26944_paths.txt
git status --short
```

Expected: only Ruff-generated layout changes are present; no token, string,
comment, or behavior change appears.

- [ ] **Step 7: Commit the Python formatting**

Run:

```bash
xargs git add -- < /tmp/task26944_paths.txt
git diff --cached --check
git commit -m "style(chat): format citation helpers"
```

Expected: one commit containing only the three assigned Python paths.

### Task 3: Close the Backlog Record

**Files:**

- Modify: `backlog/tasks/task-26944 - Clean Ruff formatter debt for ruff-chat-citations.md`

- [ ] **Step 1: Record evidence and complete every acceptance criterion**

Use `apply_patch` to check all eight criteria and add concise Implementation Notes
containing the fetched/rebased SHA, reconciliation result, exact tool versions,
invariant result, focused-test rationale, exact verification commands/results,
Python changed-path proof, and the Python-formatting commit ID. State that no new
ADR or lesson was required and list the modified files.

- [ ] **Step 2: Mark the task Done through Backlog.md**

Run:

```bash
backlog task edit 26944 -s Done
backlog task 26944 --plain
```

Expected: the CLI reports `Done` and resolves the canonical task file. If the CLI
normalizes the filename, restore the tracked canonical filename with `apply_patch`
without changing task content, then rerun the view command.

- [ ] **Step 3: Rerun closeout governance**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/CI/test_backlog_task_id_uniqueness.py -q
git diff --check
git status --short
```

Expected: the uniqueness test passes; no whitespace error exists; only the task
record remains uncommitted.

- [ ] **Step 4: Commit the closeout**

Run:

```bash
git add 'backlog/tasks/task-26944 - Clean Ruff formatter debt for ruff-chat-citations.md'
git diff --cached --check
git commit -m "docs(task-26944): close Ruff citation cleanup"
git status --short --branch
```

Expected: the closeout commit contains only the task record and the final worktree
is clean.
