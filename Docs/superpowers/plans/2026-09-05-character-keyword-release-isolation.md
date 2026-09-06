# Character Keyword Release Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox syntax; this is packaging and qualification of existing Tasks 1–5, not reimplementation or a new merged feature PR.

**Goal:** Prepare a navigation/Keyword-only delivery on current dev with preserved source history and honest release evidence.

**Architecture:** Reuse the existing local projection, FTS generation, typed navigation and incumbent UI. Exclude the separate Meaning subsystem by selecting the original Task-5 boundary, then replay only applicable later fixes.

**Tech Stack:** Python, Textual 8.x, SQLite/FTS5, pytest, Ruff and modular TCSS; existing environment only.

**Spec:** `Docs/superpowers/specs/2026-09-05-character-keyword-release-scope.md`, importing the navigation/Keyword requirements from `Docs/superpowers/specs/2026-09-03-character-conversation-navigation-design.md`.

## Global Constraints

- Character cards only; active local Data Profile only; selected visible user/assistant branch only.
- Blank Active-mode Enter in Ctrl+K keeps the incumbent MRU-other-tab behavior.
- Enter in Character surfaces activates the exact immutable highlighted conversation.
- Context and all character-search UI in this delivery remain Keyword-only.
- Work off the UI thread for operations that can exceed 100 ms; event-loop slices at most 50 ms and busy state within 100 ms.
- Preserve 52x20 reachability, identity/focus trust and production CSS.
- Do not ship Meaning controls/runtime/schema, implement TASK-31686, relax caps, change dependencies, or repeat ANN diagnostics.
- Preserve the full source branch and unrelated critique/main-checkout edits; no push, PR, merge or destructive cleanup.
- Targeted tests only. Native/platform/human evidence is not replaced with Pilot or mocks.

ADR required: no new ADR.

ADR path: backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md

Reason: existing independent Keyword boundary; reconcile allocation and delivery-status wording only.

### Task 1: Isolate and qualify the existing Keyword delivery — TASK-31245

**Files:**
- Replay original Tasks 1–5 from base `68f9d865fad623db6ec02e19632090c1140b3c89` through `c3d06dae49ec09360f9716b3cf414ae12b4b1c81`.
- Existing source families: `Character_Chat/character_conversation_navigation.py`, `DB/character_conversation_search.py`, `Chat/console_conversation_activation.py`, `UI/Navigation/character_conversation_navigation.py`, `UI/Console_Modules/character_context.py`, `Widgets/Console/console_character_context.py`, switcher/Roleplay/Library consumers and corresponding CSS.
- Governance: original Task31241–31245 records, original design/plan, applicable ADRs and guides; this isolation scope/plan.
- Test: original migration/projection/selected-branch FTS, activation, Roleplay browse, Library repair, Context/geometry, switcher/geometry/trust/activity/dismissal files and affected startup/static/resource checks.
- Report: `.superpowers/sdd/2026-09-05-character-keyword-release-isolation/task-1-report.md` plus a durable `Docs/QA/character-keyword-release.md` summary.

**Interfaces:** consume original Tasks1–5 contracts verbatim. Produce the same Keyword APIs on frozen dev `e990738b2812876c2593b91f62d0b2c5b2e3b69d`; no semantic API is an allowed dependency. Preserve separate task commit boundaries for future PR packaging.

- [ ] Read corresponding original task records via preserved branch before decisions. Keep Task31245 In Progress, and append its release-isolation plan/AC through CLI before new implementation changes. Earlier reviewed task code is not an invitation to repeat original implementation work.
- [ ] Verify the preserved source branch tip and frozen delivery base. Current branch is `codex/character-keyword-release`; controller has already committed deferred-task documentation on the source branch and left the unrelated critique untracked.
- [ ] Review allocations across remote/local refs and worktrees before replay; current dev schema is65. Preserve shipped migrations and ADRs; historical early commits still use provisional ADR116, which must not replace shipped Schedules116. Reconcile to the programme's valid allocation and include that correction in the governance boundary.
- [ ] Replay the exact existing23-commit prefix in order, without squashing adjacent task boundaries. Use `git rev-list --reverse BASE..TASK5_END` with the exact SHAs above to select commits. Resolve conflicts using both upstream intent and original contract; regenerate generated CSS, never hand-merge it. Record every conflict and rewritten task tip. No source-branch rewrite or broad checkout/reset is permitted.
- [ ] Audit `c3d06dae49..b09c7af2fc8f54a1073f27742d9b2b8e9c0dd1c2` production changes for later non-Meaning fixes. Preserve relevant Context containment, switcher painted focus/activation failure, native-open rollback and current-dev test setup fixes without copying semantic controls or dependencies. Controller independently inspects this seam. Record each accepted/excluded hunk family and reason.
- [ ] For any new behavior correction, first run a focused existing/new regression demonstrating the problem, make the smallest repair, then rerun that regression. Mechanical replay/allocation edits need diff/source equivalence, not invented RED claims.
- [ ] Verify absence of semantic imports, runtime, Settings controls and schema introduced by Tasks6–8. Current help/user guides describe Keyword-only availability. Keep the overall roadmap/spec's Meaning contract visibly deferred, not mislabeled as shipped.
- [ ] Run one serial targeted aggregate covering migration, projection/FTS, activation, browse/repair, Context and Ctrl+K trust/geometry/dismissal; record exact test list and raw output. Fix concrete owned failures with focused reruns. Do not run full repository or the original44-file semantic programme gate.
- [ ] Run the same three startup-budget files as the controller's frozen-dev baseline and compare exact results. Distinguish inherited failures from additions. Correct only safe feature-owned import regressions; no cap changes, hidden Context or broad import diet. If no supported bounded correction exists, report the precise unresolved gate.
- [ ] Verify source-owner teardown with real SQLite and quiescent fixture cleanup, and record aggregate FD/handle evidence on this Keyword subset. Do not infer the old full-programme growth is resolved from a smaller sample; do not modify unrelated production lifetimes.
- [ ] Check all new Python files with Ruff lint/format; compare changed legacy-path diagnostics against frozen dev without whole-file rewrites. Check CSS bundle sync and complete committed-range whitespace. Record remaining inherited/static qualification honestly.
- [ ] Produce bounded production-styled Pilot evidence for Keyboard, pointer, exact resume, unavailable recovery, cancellation and Context-to-Roleplay handoff at52x20 and120x50, with no more than one correction/confirmation capture batch. Native GUI/profile actions remain controller-owned and require separate authority.
- [ ] Self-review and commit exact scoped paths only. Do not mark unfinished criteria Done. Return DONE_WITH_CONCERNS if only unreachable or explicitly unresolved release evidence remains, with original gates clearly separated from new defects. Controller conducts independent task and final branch review before any integration handoff.
