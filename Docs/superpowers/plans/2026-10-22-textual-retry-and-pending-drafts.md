# Textual Retry and Pending Drafts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete TASK-36360 through TASK-36363 and deliver a reviewed PR against dev with current-head CI.
**Architecture:** Correct asynchronous acknowledgement in each current narrow owner, preserving newer accepted user input and existing persistence contracts.
**Tech Stack:** Python3.12, Textual8.x, SQLite, private TOML, pytest/Pilot.
**Spec:** The four linked Backlog tasks are the approved requirements; `Docs/superpowers/qa/2026-10-22-textual-book-after-pr3207.md` supplies qualified starting evidence.

## Global Constraints

- The four task Acceptance Criteria are binding; no unrelated findings or new global framework.
- Use this existing managed worktree/codex branch. Root owns metadata, shared UI census and integration docs; implementers own only their listed files and approved narrow same-owner helpers.
- Targeted local tests only. No full local suite. Hosted current-head CI is requested.
- Superpowers requesting-code-review replaces Qodo. Do not invoke Qodo. Do not merge or arm auto-merge.
- Preserve admission, privacy, validation, one-write admission, close/quit and literal-text guards; no schema/dependency/provider/idempotency-policy changes or raised budgets.
- Read design-language.md and component-patterns.md before UI edits; no visual changes/generated CSS edits planned.
- Preserve `.test-tmp/`; explicit UTF-8 reads/writes/subprocess capture; project Python3.12; unique private native test parents. No ACL/admission weakening.
- No shared source scans during temporary baseline reversal; review source-stable committed handoffs.
- ADR required: no new ADR. Existing ADR-011 and ADR-033 application-session-state apply; Settings additionally ADR-033 settings-commit, ADR-004 storage-restart, ADR-161; Scheduling ADR-018. Use exact links from task files. Proposed schedule ADR099/116 are context only.

## Review Focus

- Queued Changed delivery must not discard accepted raw edits (Tasks2/4).
- Returned-to-old-baseline, invalid and whitespace edits remain newer intent (Tasks2/3).
- Final-answer retry preserves earlier answers and uses the current final answer once (Task1).
- Retained create adopts saved identity and next explicit Save updates the same row (Task3).
- Unchanged success, failure/retry, Revert and obsolete-source notification remain truthful (Tasks2/3/4).

---
### Task 1: TASK-36360 — Quiz final-answer retry

**Requirements:** Read `backlog/tasks/task-36360 - Keep-quiz-final-answer-retry-from-duplicating-answers-and-inflating-scores.md` in full, including all ACs and exact ADR/owner links.
**Files:** Modify `tldw_chatbook/UI/Study_Modules/quizzes_handler.py`; create/modify `Tests/UI/test_study_quiz_attempt_retry.py`. Root owns task metadata/census. Notify root before a narrowly necessary same-owner helper.
**Interfaces:** Consume existing mounted controls/service entry points; produce unchanged service/return contracts. No later task consumes a new API.
**Approach:** Build the final payload from earlier completed answers plus current answer without mutating the pending-answer list before success. Preserve progression, service and grading contracts.
**Required tests:** Mounted actual Start/Submit plus real scope/local services and SQLite. Definite pre-write failure/retry sends and stores one answer/question with score bounded by total; editing final answer after failure uses its current value once and retains earlier answers; ordinary multi-question history/result remains correct. Independently assert payload, database and visible result.
**Audit starting point:** `.test-tmp/post-3207-audit/evals-study/report.md`. Use only qualified receipts; an observation test asserting the bug is not a repair test.

- [ ] Read current production flow, relevant architecture/lessons/ADRs and adjacent tests.
- [ ] Add behavior regressions proving every Required tests outcome above; state the production change each would catch. Run RED before production changes and retain exact native command/log/XML.
- [ ] Implement the smallest owner-local correction described above, following existing patterns.
- [ ] Run new plus adjacent owner tests GREEN and targeted Ruff/format checks. Qualify inherited legacy debt against baseline; no blanket suppressions or unrelated formatting.
- [ ] Self-review and commit only explicitly owned source/test files. Write full report with RED/GREEN receipts, commands/results, changed files and concerns. No subagents; root dispatches independent review.

### Task 2: TASK-36361 — Settings pending Save drafts

**Requirements:** Read `backlog/tasks/task-36361 - Preserve-newer-Appearance-Storage-and-Console-Behavior-drafts-after-pending-Saves.md` in full, including all ACs and exact ADR/owner links.
**Files:** Modify `tldw_chatbook/UI/Screens/settings_screen.py`; create/modify `Tests/UI/test_settings_pending_save_drafts.py`. Root owns task metadata/census. Notify root before a narrowly necessary same-owner helper.
**Interfaces:** Consume existing mounted controls/service entry points; produce unchanged service/return contracts. No later task consumes a new API.
**Approach:** Reuse existing media acknowledgement patterns. Advance actual saved baseline from submitted snapshot and retain newer raw Appearance/Storage/Console Behavior fields, even before queued Changed delivery; preserve widget editing context.
**Required tests:** For all three categories use actual Save and held real SettingsConfigAdapter.save_sections/private TOML writer. Cover same/different-field later edits, accepted but queued edits, return-to-old-baseline, next Save, unchanged success, real failure and confirmed Revert; preserve origin checks and Storage restart boundary.
**Audit starting point:** `.test-tmp/post-3207-audit/settings/report.md`. Use only qualified receipts; an observation test asserting the bug is not a repair test.

- [ ] Read current production flow, relevant architecture/lessons/ADRs and adjacent tests.
- [ ] Add behavior regressions proving every Required tests outcome above; state the production change each would catch. Run RED before production changes and retain exact native command/log/XML.
- [ ] Implement the smallest owner-local correction described above, following existing patterns.
- [ ] Run new plus adjacent owner tests GREEN and targeted Ruff/format checks. Qualify inherited legacy debt against baseline; no blanket suppressions or unrelated formatting.
- [ ] Self-review and commit only explicitly owned source/test files. Write full report with RED/GREEN receipts, commands/results, changed files and concerns. No subagents; root dispatches independent review.

### Task 3: TASK-36362 — Recurring Question pending Save drafts

**Requirements:** Read `backlog/tasks/task-36362 - Keep-newer-Recurring-Question-drafts-when-an-earlier-Save-completes.md` in full, including all ACs and exact ADR/owner links.
**Files:** Modify `tldw_chatbook/UI/Screens/scheduling/forms/automation_definition_form.py`; create/modify `Tests/UI/test_automation_definition_pending_save_drafts.py`. Root owns task metadata/census. Notify root before a narrowly necessary same-owner helper.
**Interfaces:** Consume existing mounted controls/service entry points; produce unchanged service/return contracts. No later task consumes a new API.
**Approach:** Compare submitted/current raw form across saved or queued acknowledgement. Retain newer form, adopt actual saved definition identity after create, and dismiss unchanged success. Keep existing one-write admission and owner semantics.
**Required tests:** Mounted actual keyboard with real SQLite create/edit, newer Name/Question/other fields including invalid/whitespace; unchanged close; next explicit Save updates same created row once; failure/refusal retry and existing close/quit guards. No mocked success persistence.
**Audit starting point:** `.test-tmp/post-3207-audit/scheduling/report.md`. Use only qualified receipts; an observation test asserting the bug is not a repair test.

- [ ] Read current production flow, relevant architecture/lessons/ADRs and adjacent tests.
- [ ] Add behavior regressions proving every Required tests outcome above; state the production change each would catch. Run RED before production changes and retain exact native command/log/XML.
- [ ] Implement the smallest owner-local correction described above, following existing patterns.
- [ ] Run new plus adjacent owner tests GREEN and targeted Ruff/format checks. Qualify inherited legacy debt against baseline; no blanket suppressions or unrelated formatting.
- [ ] Self-review and commit only explicitly owned source/test files. Write full report with RED/GREEN receipts, commands/results, changed files and concerns. No subagents; root dispatches independent review.

### Task 4: TASK-36363 — Audiobook pending file import drafts

**Requirements:** Read `backlog/tasks/task-36363 - Keep-newer-audiobook-source-edits-when-a-pending-file-import-completes.md` in full, including all ACs and exact ADR/owner links.
**Files:** Modify `tldw_chatbook/UI/STTS_Window.py`; create/modify `Tests/UI/test_audiobook_import_source_integrity.py`. Root owns task metadata/census. Notify root before a narrowly necessary same-owner helper.
**Interfaces:** Consume existing mounted controls/service entry points; produce unchanged service/return contracts. No later task consumes a new API.
**Approach:** Guard file-import completion against current live editor text and owning source; Changed can be queued behind awaiting parent. Discard obsolete import and its success announcement. Retain full-source/chapter guards.
**Required tests:** Mounted actual Import/picker and original UTF-8 off-loop file read with held completion; actual keyboard edit/replacement/clear. Editor, content_text and actual Generate payload agree on retained source. No-edit full import and chapters succeed, original file bytes unchanged. No synthesis/network.
**Audit starting point:** `.test-tmp/post-3207-audit/audiobook/report.md`. Use only qualified receipts; an observation test asserting the bug is not a repair test.

- [ ] Read current production flow, relevant architecture/lessons/ADRs and adjacent tests.
- [ ] Add behavior regressions proving every Required tests outcome above; state the production change each would catch. Run RED before production changes and retain exact native command/log/XML.
- [ ] Implement the smallest owner-local correction described above, following existing patterns.
- [ ] Run new plus adjacent owner tests GREEN and targeted Ruff/format checks. Qualify inherited legacy debt against baseline; no blanket suppressions or unrelated formatting.
- [ ] Self-review and commit only explicitly owned source/test files. Write full report with RED/GREEN receipts, commands/results, changed files and concerns. No subagents; root dispatches independent review.

## Integration and delivery

Root adds all new concrete UI nodes to the required census preserving existing prefix/slots/shards/caps, runs targeted integration/governance checks after stable handoffs, updates relevant docs and task notes/ACs, then marks Done via CLI after review/checks. Dispatch one final whole-branch review on the most capable reviewer using requesting-code-review, address findings, create and attach a PR against dev, and stay on current-head CI/review feedback. Rebase onto newer dev if needed; never merge dev into the branch. Do not merge the PR.
