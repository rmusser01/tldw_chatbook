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
- Compute each inline directive's significant-token position from its nearest logical owner: a uniquely validated same-line `except` clause for an `ExceptHandler` header, otherwise its nearest containing AST statement. Exclude only parenthesis pairs proven AST-neutral by an independent shadow parse/dump comparison.
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

- [x] Verify HEAD and origin/dev are the same fetched commit and record that current commit plus TASK-26000's e555df102c950c29beed5e7119f433d35eee1f3c authority cut.
- [x] Parse the task's Assigned Paths JSON, require exactly 55 Tests/Chat paths and 18 tldw_chatbook/Chat paths, require uniqueness, and recompute the recorded path digest.
- [x] Confirm every assigned path exists at HEAD; inspect rename/deletion/modification lineage from the authority cut to HEAD and record every retained upstream-modified path.
- [x] Run the exact 55 assigned test modules with Python 3.12.11, --tb=line, --disable-warnings, and JUnit XML at /tmp/task26947_before.xml; record exit code, counts, and normalized failure keys.
- [x] Capture AST, ordered comment, directive-anchor, and fmt-range evidence for all 73 paths at /tmp/task26947_before.json.

### Task 2: Format Only the Assigned Paths

**Files:**

- Modify: only assigned paths changed by Ruff 0.15.22.

**Interfaces:**

- Consumes: the reconciled allowlist and /tmp/task26947_before.json.
- Produces: the deterministic Ruff output and a structural parity result.

- [x] Invoke Ruff 0.15.22 format once with all 73 paths supplied explicitly.
- [x] Correct the ephemeral guard's directive-position metric, including the fail-closed `ExceptHandler` header boundary, restore the 73 assigned files to the immutable pre-format blobs, and recapture /tmp/task26947_before.json before rerunning Ruff.
- [x] Compare the post-format AST/comment/directive/fmt-range evidence with the corrected /tmp/task26947_before.json and stop on any mismatch.
- [x] Assert every changed Python path is in the 73-path allowlist and no assigned path was silently omitted.
- [x] Review the formatter diff for handwritten or behavioral changes.
- [x] Commit only the assigned Python paths changed by Ruff so Task 2's review package contains the formatter diff.

### Task 3: Run Focused and Governance Verification

**Files:**

- Verify: the exact 73 assigned paths.
- Verify: the exact 55 assigned Tests/Chat modules.
- Modify only if required by an immutable-base Ruff lint failure: assigned test imports proven unused and side-effect-free.
- Create temporary evidence only: /tmp/task26947_after.xml.

**Interfaces:**

- Consumes: the formatted allowlist and baseline failure inventory.
- Produces: Ruff, structural, focused-test, governance, and scope evidence suitable for task closeout.

- [x] Run Ruff 0.15.22 check and format --check on all 73 assigned paths.
- [x] If Ruff lint reproduces an unused-import failure on the immutable base, capture the owning test module result, use Ruff's safe fix on only the affected assigned test path, and require the same test result plus a clean lint/format check afterward; do not change production code or suppress the diagnostic.
- [x] Run the same 55-module focused pytest command with JUnit XML at /tmp/task26947_after.xml.
- [x] Compare normalized before/after failure keys and require no additions or removals.
- [x] Run Tests/CI/test_backlog_task_id_uniqueness.py and git diff --check.
- [x] Reproduce every formatted assigned file by formatting its immutable branch-base blob through Ruff stdin and compare bytes.
- [x] Keep the formatter-commit byte replay anchored to commit 44f5408e8d; separately prove any later assigned-test lint-fix commit with its immutable-base Ruff diagnostic, safe-fix diff, and targeted pre/post test result.
- [x] Run the persistent-diagnostic inventory check; if Ruff changes only a tracked statement's source-layout digest, regenerate only that derived artifact and record the semantic-equivalence evidence.
- [x] If the diagnostic check requires a generated inventory refresh, commit only that derived artifact after its own verification.

### Task 4: Commit, Review, and Close TASK-26947

**Files:**

- Modify: backlog/tasks/task-26947 - Clean Ruff formatter debt for ruff-chat-console-foundation.md.
- Modify only if required by the diagnostic check: Docs/security/production-diagnostic-inventory.json.
- Modify: Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md.
- Modify: backlog/docs/lessons-testing-evidence.md.

**Interfaces:**

- Consumes: all Task 1-3 evidence.
- Produces: a formatter commit, reviewed task closeout, and merge-ready branch.

- [x] Confirm Task 2's formatter commit contains only assigned Python paths and any Task 3 commits contain only the reviewed assigned-test lint cleanup and required generated-inventory refresh.
- [x] Request independent code review and address every Critical or Important finding.
- [x] Add exact drift, structural, Ruff, focused-test, governance, and generated-artifact results to Implementation Notes.
- [x] Clarify TASK-26000's directive-position definition and record the physical-line guard incident in the testing-evidence lessons so later formatter batches use the corrected metric.
- [x] Check every acceptance criterion, set TASK-26947 to Done, and commit the task/plan closeout.
- [x] Rebase onto pre-merge origin/dev `298a34557c2e02699d3505cf6f5c9880e12cda07`; its delta from the last paired-test base `1a1b5c19e0bb3243effb1ae9671158b6670ad6da` changed zero assigned paths. Retain the exact paired 55-module evidence from that unchanged assigned surface, rerun all-73 replay/v3, Ruff, governance, and inventory gates, and exercise upstream-changed dependent Console/theme/settings/shared-CSS surfaces against final HEAD and exact current dev; retain the one-file `console_agent_bridge.py` formatter follow-up that caught the earlier post-rebase blank-line omission.
- [ ] Root-owned integration after this documentation closeout: publish the PR, address Qodo and CI findings, enforce strict latest-base protection, and merge.

ADR required: no
ADR path: N/A
Reason: This is mechanical formatter cleanup under TASK-26000's accepted contract and introduces no architectural, persistence, security, dependency, or long-lived UX decision.
