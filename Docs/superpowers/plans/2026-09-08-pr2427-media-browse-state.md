# PR 2427 Media Browse state implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development or executing-plans; preserve every behavioral contract during the extraction.

**Goal:** Bring Media Browse under its unchanged 371-line limit through explicit UI-local state ownership.

**Architecture:** Compose one MediaBrowseState; move whole pure methods and initial data into it. Keep asynchronous orchestration and generation admission in the existing controller.

**Tech Stack:** Python 3.12, Textual 8, existing pytest and page types.

**Spec:** `backlog/decisions/128-library-media-browse-presentation-state.md`, implementing the user's approval after checkpoint 98a9aeeb85.

ADR required: yes.
ADR path: `backlog/decisions/128-library-media-browse-presentation-state.md`.
Reason: an explicit UI collaborator ownership boundary; ADR-067's domain/service contracts remain unchanged.

## Global Constraints

- Workspace: `.worktrees/pr2427-review-recovery` only; preserve unrelated dirty files and other workers' changes.
- No behavior change, generic forwarding, inheritance, DOM/CSS change, new dependencies, or suppressed assertions/warnings.
- Controller <=371 lines; every existing cap may only stay or fall. Pin the new state file at its measured size.
- Keep exact generation checks, worker groups/exclusivity, cancellation, clamp limit, callback ordering and constructor late binding.
- Root owns task/ADR/report edits, staging, commits, rebase, push and merge; worker freezes sources before root verification.

## Task 1: Separate presentation state and retarget exact consumers

Files: create `tldw_chatbook/UI/Library_Modules/library_media_browse_state.py`;
modify `library_media_browse_controller.py`, existing `library_media_controller.py`,
`UI/Screens/library_screen.py`, exact tracked test consumers, and
`Tests/Architecture/test_library_modules_size_ratchet.py`.

Interfaces: controller keeps its existing constructor and operational entry
points. `controller.state` is one per-controller MediaBrowseState instance.
Data fields retain names/initial values under state, except `_page_generation`
and `_facet_generation` stay on the controller.

- [ ] Capture the complete tracked consumer census (`git grep`, including ignored-but-tracked Tests/Live) and all original assertions before edits.
- [ ] Add a focused ownership regression requiring independent explicit state objects while two real controllers retain independent generation fencing; observe failure before extraction.
- [ ] Move whole pure members: `failure`, `applied_scope`, `mutation_refresh_scope`, `scope_for_page`, `pager`, `_failure_copy`, `retain_stale_items`, `note_analysis_state`, `reconcile_committed_mutation`, and `clear_fault_episode`. Move their initial data and existing `_load_failure`/`_raised_failure` helpers with needed constants/imports. Keep callable signatures and bodies unchanged except mechanical owner qualification.
- [ ] Replace initial data setup with `self.state = MediaBrowseState()`. Rewrite controller data reads/writes as `self.state.<field>`; keep both generations and `_current`, `begin`, `request`, `retry`, `_search`, `_load`, `_apply`, `begin_mutation`, `invalidate`, `request_facets`, `_load_facets`, `invalidate_facets`, `_run_worker` in the controller. Do not add terminal-transition methods unless a concrete blocker is reported to root.
- [ ] Retarget direct state/pure-method consumers and writable test fakes. Do not alter behavioral assertions. Preserve any intentionally patched module exports unless the exact test consumer is retargeted too.
- [ ] Run complete controller and pure-state regressions, retry/fault context, paging and direct consumer files. Verify moved-method AST equivalence after receiver normalization, exact runtime callback/generation checks, and no new dependency on Textual in the state module.
- [ ] Measure and lower the controller pin to actual lines; add an exact pin for state. Run complete Screen/module size guards and import/payload budgets. Never refresh an over-limit snapshot.
- [ ] Self-review and report exact files, RED/GREEN, assertion audit, measured counts and complete-file results. Root obtains independent spec/correctness review, then commits the attributable extraction.

## Task 2: Integrated Library qualification and publication

- [ ] After sources freeze, run complete `Tests/UI/test_library_shell.py` and the complete mounted Media/entry/reuse consumer files identified in Task 1. Keep warning evidence separate from failures.
- [ ] Run all six derived-artifact checks; inspect any moved diagnostic owner before regenerating a manifest.
- [ ] Rebase the committed checkpoint onto current dev, reconciling any changed contracts. Repeat affected checks if executable source changes.
- [ ] Publish, address current PR feedback, and merge normally only with all open local gates, required CI and final-head review satisfied.

## Preflight review

Task 1 produces the state interface consumed by Task 2's unchanged mounted tests;
there is no concurrent implementation in those files. Task 1 preserves state
semantics while changing receivers; Task 2 does not change those assertions to
accommodate a regression. Watchlists and Qodo fixes own separate files and do not
change this boundary. Exact numerical pins are measured after extraction, never
predicted values used as evidence.
