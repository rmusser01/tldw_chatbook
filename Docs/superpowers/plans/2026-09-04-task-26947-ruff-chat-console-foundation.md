# TASK-26947 Ruff Chat Console Foundation Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Apply Ruff 0.15.22's formatter to the exact 73 TASK-26000 Console foundation paths while proving Python semantics, comments, and formatter directives are unchanged.

**Architecture:** Treat TASK-26000's batch manifest as the ownership boundary. Reconcile every recorded path against current origin/dev, capture Python 3.12.11 structural evidence, run Ruff only on those 73 paths, compare the same evidence after formatting, and use the recorded 55-module focused suite as behavioral parity evidence.

**Tech Stack:** Python 3.12.11, Ruff 0.15.22, pytest, standard-library ast and tokenize, Git, Backlog.md.

**Spec:** Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md

## Global Constraints

- The 73 paths in TASK-26947's Assigned Paths JSON are the exact file allowlist; do not format or change any unassigned Python path.
- Normalize only ast.TypeIgnore.lineno when comparing AST dumps.
- Preserve ordered comment tokens, inline directive anchors, standalone Ruff directive adjacency, and fmt-off/fmt-on enclosed-node intervals.
- Do not make handwritten production behavior changes.
- Capture the untouched origin/dev focused-test baseline before formatting and require the post-format failure-key inventory to match it exactly.
- Use the exact 55 assigned test modules owned by this batch. Do not run the full suite without user opt-in.

---

### Task 1: Reconcile Ownership and Capture the Baseline

**Files:**

- Inspect: backlog/tasks/task-26947 - Clean Ruff formatter debt for ruff-chat-console-foundation.md
- Inspect: Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json
- Inspect: all 73 paths in TASK-26947's Assigned Paths JSON
- Create temporary evidence only: /tmp/task26947_before.json and /tmp/task26947_before.xml

**Interfaces:**

- Consumes: TASK-26000 batch label ruff-chat-console-foundation and pinned path digest c4150a472d5ef3d79bcc9e6795e0db669d8a27268b8cef71d5d9a71e5d86bf5a.
- Produces: reconciled 73-path allowlist, current-dev lineage notes, structural baseline, and focused-test failure keys.

- [ ] Verify HEAD and origin/dev are the same fetched commit and record that current commit plus TASK-26000's e555df102c950c29beed5e7119f433d35eee1f3c authority cut.
- [ ] Parse the task's Assigned Paths JSON, require exactly 55 Tests/Chat paths and 18 tldw_chatbook/Chat paths, require uniqueness, and recompute the recorded path digest.
- [ ] Confirm every assigned path exists at HEAD; inspect rename/deletion/modification lineage from the authority cut to HEAD and record every retained upstream-modified path.
- [ ] Run the exact 55 assigned test modules with Python 3.12.11, --tb=line, --disable-warnings, and JUnit XML at /tmp/task26947_before.xml; record exit code, counts, and normalized failure keys.
- [ ] Capture AST, ordered comment, directive-anchor, and fmt-range evidence for all 73 paths at /tmp/task26947_before.json.

### Task 2: Format Only the Assigned Paths

**Files:**

- Modify: only assigned paths changed by Ruff 0.15.22.

**Interfaces:**

- Consumes: the reconciled allowlist and /tmp/task26947_before.json.
- Produces: the deterministic Ruff output and a structural parity result.

- [ ] Invoke Ruff 0.15.22 format once with all 73 paths supplied explicitly.
- [ ] Compare the post-format AST/comment/directive/fmt-range evidence with /tmp/task26947_before.json and stop on any mismatch.
- [ ] Assert every changed Python path is in the 73-path allowlist and no assigned path was silently omitted.
- [ ] Review the formatter diff for handwritten or behavioral changes.
- [ ] Commit only the assigned Python paths changed by Ruff so Task 2's review package contains the formatter diff.

### Task 3: Run Focused and Governance Verification

**Files:**

- Verify: the exact 73 assigned paths.
- Verify: the exact 55 assigned Tests/Chat modules.
- Create temporary evidence only: /tmp/task26947_after.xml.

**Interfaces:**

- Consumes: the formatted allowlist and baseline failure inventory.
- Produces: Ruff, structural, focused-test, governance, and scope evidence suitable for task closeout.

- [ ] Run Ruff 0.15.22 check and format --check on all 73 assigned paths.
- [ ] Run the same 55-module focused pytest command with JUnit XML at /tmp/task26947_after.xml.
- [ ] Compare normalized before/after failure keys and require no additions or removals.
- [ ] Run Tests/CI/test_backlog_task_id_uniqueness.py and git diff --check.
- [ ] Reproduce every formatted assigned file by formatting its immutable branch-base blob through Ruff stdin and compare bytes.
- [ ] Run the persistent-diagnostic inventory check; if Ruff changes only a tracked statement's source-layout digest, regenerate only that derived artifact and record the semantic-equivalence evidence.
- [ ] If the diagnostic check requires a generated inventory refresh, commit only that derived artifact after its own verification.

### Task 4: Commit, Review, and Close TASK-26947

**Files:**

- Modify: backlog/tasks/task-26947 - Clean Ruff formatter debt for ruff-chat-console-foundation.md.
- Modify only if required by the diagnostic check: Docs/security/production-diagnostic-inventory.json.

**Interfaces:**

- Consumes: all Task 1-3 evidence.
- Produces: a formatter commit, reviewed task closeout, and merge-ready branch.

- [ ] Confirm Task 2's formatter commit contains only assigned Python paths and any Task 3 generated-artifact commit contains only the required inventory refresh.
- [ ] Request independent code review and address every Critical or Important finding.
- [ ] Add exact drift, structural, Ruff, focused-test, governance, and generated-artifact results to Implementation Notes.
- [ ] Check every acceptance criterion, set TASK-26947 to Done, and commit the task/plan closeout.
- [ ] Rebase onto latest origin/dev, rerun scope/reproduction/governance gates, publish a PR, address Qodo and CI findings, and merge only while strict-latest-base protection remains satisfied.

ADR required: no
ADR path: N/A
Reason: This is mechanical formatter cleanup under TASK-26000's accepted contract and introduces no architectural, persistence, security, dependency, or long-lived UX decision.
