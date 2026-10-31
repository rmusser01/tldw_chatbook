# Quiz Start and Reader Lifetimes Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for the parent implementation and superpowers:dispatching-parallel-agents for the independent Watchlists repair. Steps use checkbox syntax.

**Goal:** Complete TASK-36899 and TASK-36900 without changing service/storage contracts.

**Architecture:** Retain StudyQuizzesController ownership and shared Start admission; invalidate obsolete Start publication on context reset, including away/back. Return promptly from Watchlists ItemSelected by scheduling a screen-owned detail worker; publish one hydrated item only if its selection, snapshot, page, backend and mounted section still own the result.

**Tech Stack:** Python 3.12, Textual 8.x, existing SQLite services.

**Spec:** backlog/tasks/task-36899 - Keep-pending-quiz-Start-within-its-runtime-and-scope-lifetime.md and backlog/tasks/task-36900 - Keep-Watchlists-navigation-responsive-during-article-body-reads.md; bounded source/observations in Docs/superpowers/qa/2026-10-25-textual-book-after-pr3247.md.

## Global Constraints

- Targeted local tests only; no full sweep without user opt-in.
- No new dependencies, tokens, service APIs or persistence. Keep narrow list projection/shared rows and one hydrated reader publication.
- Superpowers requesting-code-review replaces Qodo. Do not merge this PR unless separately requested.
- Preserve unrelated work and private scratch; only parent edits Git, CI files, task status and global documentation.
- Existing Accepted ADR-011/033 apply to both repairs; ADR-042 also applies to Reader.

ADR required: no
ADR path: backlog/decisions/011-chatbook-workbench-ui-system.md; backlog/decisions/033-application-session-state-ownership.md; backlog/decisions/042-watchlists-reader-first-ia.md
Reason: direct bounded corrections under existing screen/runtime/snapshot ownership; no new architectural boundary.

## Review Focus

1. A completion after runtime/scope away/back cannot revive the old Start or clear a new draft.
2. A failed obsolete operation cannot overwrite current UI; already-created remote attempts need truthful feedback.
3. Cancellation cannot stop an already running DB thread; obsolete results/errors must remain powerless.
4. Section/page/query/backend transitions and actual unmount must retire Reader publication, including away/back.
5. Latest reader success still marks only the current item read and publishes its shared hydrated row once.

## Task 1: TASK-36899 (parent)

**Files:** modify UI/Study_Modules/quizzes_handler.py; extend Tests/UI/test_study_quiz_start_responsiveness.py; update Docs/Architecture/study.md.
**Interfaces:** existing reset_quiz_panel/handle_scope_changed/start_attempt; no new cross-module API.

- [x] Extend the existing mounted real services regression with runtime, workspace and away/back reset cases. Expected: old Start cannot adopt an attempt, clear a new draft or enable wrong-backend Submit.
- [x] Run the new cases against unchanged production and retain expected assertion failures.
- [x] Add a controller reset generation, captured at Start and checked before publishing; preserve single-flight/finally, removed-panel guard and truthful remote outcome.
- [x] Cover stale failure and ordinary retry/admission controls. Run complete affected Start file and targeted quiz service/scope regressions, scoped lint/format.
- [x] Document behavior/evidence; parent commits verified implementation for integration review.

## Task 2: TASK-36900 (delegated independent domain)

**Files:** modify UI/Screens/watchlists_collections_screen.py; create Tests/UI/test_watchlists_reader_responsiveness.py. Parent updates Docs/Architecture/subscriptions-watchlists.md.
**Interfaces:** existing ItemSelected -> controller.get_item_content; keep snapshot/page presentation-lock semantics and shared row identity.

- [x] Write mounted production-screen/private DB regressions: later item selection and enabled Sources navigation advance while B's body read remains held; ordinary control and superseded result/error protections.
- [x] Run against unchanged production and retain expected assertion failures.
- [x] Dispatch a dedicated explicitly named/exclusive detail-read worker from the handler. Capture monotonic selection authority and invalidate it on relevant lifecycle/context changes; guard error feedback and shared-row mutation at arrival.
- [x] Prove out-of-order/thread completion, page/query/backend/section away-back and unmount cannot publish or mark obsolete items read; retain one hydrated publication.
- [x] Run complete new file and relevant existing Reader/snapshot/detail tests, scoped lint/changed-range format. Return evidence and exact changed files; no Git/CI/task mutations.

## Integration

- [x] Register new/extended cases in the required UI lane using the existing selector/census conventions; preserve shard count and budgets.
- [x] Run targeted integration tests and governance/static checks, inspect private child XML independently.
- [x] Commit implementation; request one fresh whole-branch Superpowers review with task/plan/evidence and Review Focus.
- [x] Address Important/Critical findings with RED -> GREEN tests, revalidate targeted gates; mark task criteria/status Done via CLI only after definition of done is met.
- [ ] Push, create PR against dev, attach it to this chat, inspect current-head CI execution/review threads and fix actionable feedback until checks finish.

## Qualification

Fresh whole-branch Superpowers review found one Important pending-selection keyboard issue, no Critical/Minor issues. Three Pilot cases failed before the two-anchor repair, then passed in the complete final Reader18/18 run. Study61/61 and CI contracts152/152 passed; all45 private Study/Reader child reports independently clean. Legacy neighbor setup adaptations and matched baseline failures are explicitly qualified in Docs/superpowers/qa/2026-10-25-textual-quiz-and-reader-lifetimes.md. No full-suite or live-server claim.

Publication and hosted CI/review completion are recorded on the PR, avoiding an evidence-only commit that would invalidate the tested head. The remaining checkbox names that external follow-up.

CI follow-up: retained named captured Start mode/scope locals for the unchanged architecture keyword guard (8/8; adjacent architecture12/12; full Study61/61 requalified). Corrected existing TASK-34000.18 receipt-test layout readiness after measured stale click geometry. A wider highlight wait did not settle and was reverted; that original test remains unchanged. No Library product behavior or assertion/deadline relaxation. First hosted failures and all26newcase passes are retained separately from final PR qualification.
