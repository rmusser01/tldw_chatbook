---
id: TASK-19022
title: Add Library starter rail and lifecycle-aware landing
status: Done
assignee: []
created_date: '2026-08-20 20:53'
labels:
  - library
  - ux
  - onboarding
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-20-library-lifecycle-progressive-disclosure-design.md
  - >-
    Docs/superpowers/plans/2026-08-20-library-starter-rail-landing.md
  - backlog/decisions/076-library-lifecycle-progressive-disclosure.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give new empty profiles a compact production-path Get started experience while preserving the full Library for existing users and power users.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A new empty profile sees Import, New note, and Explore all tools without a false empty claim while evidence is unresolved.
- [x] #2 Any authoritative eligible user content permanently graduates the profile; bundled, sample, trash-only, inaccessible, and failed-import records do not.
- [x] #3 Explore all tools persists independently of section collapse, and deep links or command-palette routes remain reachable.
- [x] #4 Legacy profiles without a lifecycle preference default to the expanded Library, while corrupt preferences fail safely to expanded.
- [x] #5 Starter and transition states are keyboard complete, announced in text, focus-safe, and usable at 100x30 and 170x48.
- [x] #6 Only modified/touched component and direct-owner tests are run; no repository-wide pytest claim is made.
<!-- AC:END -->

## Implementation Plan

1. Add the pure lifecycle/evidence contracts and commit ADR-076 with backward-compatible preference coercion.
2. Add source-owned provenance-aware tri-state evidence seams for Notes, Media, Conversations, Prompts, Skills, and Collections without returning records or private content.
3. Make LibraryScreen own one generation-fenced evidence aggregation, serialized lifecycle persistence, restart restoration, truthful loading/failure status, and unmount authority.
4. Render the compact Starter rail with production Import/New note routes, remembered Explore disclosure, and deep-link bypass.
5. Render the lifecycle-aware landing with truthful unresolved/recovery copy, settled graduation, semantic focus preservation, and both supported geometries.
6. Run only touched/direct-owner tests and static checks, complete production-hierarchy mounted UAT at 100x30 and 170x48, obtain independent reviews, update docs, and close by editing this five-digit task file directly per the backlog-hygiene lesson. Isolated-profile live UAT remains owned by Wave 1 closeout.

Detailed TDD steps, exact files, commands, inverses, and commit boundaries are in `Docs/superpowers/plans/2026-08-20-library-starter-rail-landing.md`.

ADR required: yes
ADR path: `backlog/decisions/076-library-lifecycle-progressive-disclosure.md`
Reason: The task adds a long-lived profile-local UX lifecycle, persistence contract, source-evidence boundary, and navigation-disclosure policy. ADR-067 remains authoritative for source paging and data ownership.

## Implementation Notes

- Added the ADR-076 lifecycle/evidence contracts and six source-owned,
  provenance-aware evidence seams without moving records or private content into
  the Library shell. ADR-067 continues to own paging and source data boundaries.
- Config bootstrap now retains one process-session fact when it creates the real
  profile config; `TldwCli` snapshots that fact before later settings reloads and
  Library reads it without consulting or writing wizard state. Existing configs
  without lifecycle state still open Expanded. LibraryScreen owns exclusive,
  generation-fenced aggregation, serialized profile-local lifecycle persistence,
  unmount cleanup, retry/warning state, permanent graduation, and focus-safe
  transitions.
- Added the compact production-path Get started rail and lifecycle-aware landing.
  Import, New note, Explore, Back, deep links, palette routes, collapsed-rail
  ownership, and the existing full power-user shell share the established router.
- Notes and Prompt local evidence counts run in worker threads under the shell's
  overall deadline. Prompt supports both synchronous leaves and the real thin
  async local adapter by running that adapter's complete coroutine on a worker
  loop, never the Textual loop. Local Skills evidence is pinned to its
  profile-owned managed store (blocked records excluded), and local Prompt
  evidence treats user-created names such as "Bundled" or "System" as content
  rather than invented provenance.
  Local Media evidence requests one `library_summary=True` row from the completed,
  non-trash, non-deleted population; summary-path diagnostics omit row material,
  exact totals, content, queries, IDs, and paths.
- Focused pure/source/direct-owner gate: 128 passed (60 + 5 + 30 + 3 + 7 + 9 +
  4 + 10). Mounted/direct-owner gate: 165 passed (22 rail + 13 landing + 129
  bounded shell + 1 palette); unrelated cases were deselected by the plan
  selectors. No repository-wide pytest command was run.
- Re-review Prompt compatibility used the exact node
  `Tests/Library/test_library_prompts_seam.py::test_prompts_user_content_evidence_runs_real_async_adapter_off_loop_and_is_cancellable`:
  RED showed the real async adapter never entered, emitted an unawaited-coroutine
  warning, and raised `TypeError` when the coroutine was converted to `int`;
  restored GREEN was `1 passed`. The final Prompt seam plus two onboarding nodes
  passed 11 cases.
- The bounded onboarding/production-stylesheet replay passed 29 cases, including
  two genuinely settled Starter geometry cases at 100x30 and 170x48. Both release
  all six evidence owners, assert compositor paint/containment for orientation and
  Import/New note/Explore, and traverse the complete six-action Library-local Tab
  order. Back from a mounted Search canvas returns to the landing and focuses
  Import; a replacement evidence generation cancels its predecessor worker
  immediately. An already-running bounded executor leaf may finish without apply
  authority.
- Exact inverse audit (each command was run with the stated mutant, then the same
  command returned `1 passed` after restoration):
  1. `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_library_shell.py::test_library_onboarding_cached_zero_snapshot_cannot_declare_starter` -- mutant trusted cached zeroes as lifecycle evidence; node failed because lifecycle became Starter instead of remaining Unknown.
  2. `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_library_shell.py::test_library_onboarding_late_generation_and_unmount_cannot_apply` -- mutant removed the generation/unmount apply guard; stale positive evidence changed the lifecycle instead of leaving the current Starter result authoritative.
  3. `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q 'Tests/UI/test_library_shell.py::test_library_onboarding_legacy_and_corrupt_preferences_open_expanded[None]'` -- mutant mapped absent legacy state to Unknown; node failed its Expanded assertion.
  4. `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_library_shell.py::test_library_starter_deep_link_opens_hidden_collection_or_note_route` -- mutant blocked hidden Starter routes at composition; node failed to select/mount the Collections destination.
  5. `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_library_shell.py::test_library_onboarding_restart_after_partial_failure_restores_unknown` -- mutant skipped initial Unknown persistence; the restarted screen did not restore Unknown after partial failure.
  6. `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_library_shell.py::test_library_onboarding_persistence_failure_keeps_session_and_warns` -- mutant reverted the session on a failed preference write; the node lost Graduated state instead of retaining it with the not-remembered warning.
  7. `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_library_shell.py::test_library_onboarding_positive_wins_while_another_owner_hangs` -- mutant waited for every owner; the positive result missed the one-second bound with `positive evidence waited for the hanging owner`.
- Against implementation base `6a2e7fa50`, the final worktree has 31 changed
  Python owners/tests: the reviewed 29-file inventory plus the required real
  config/app admission owners. Ruff check passed all 31. The six files omitted
  by the prior inventory are now present in the plan and gates; base comparison
  removed only proven formatting-only churn without changing their behavior.
  Ruff format check passed the plan's 11-file conforming allowlist; whole-file
  baseline format drift was not bulk-formatted. Base-range, staged, and worktree
  `git diff --check` passed.
- Review status: both final full-range reviews were addressed and independently
  re-reviewed with no remaining Critical or Important findings. The final
  implementation commit is `937dfa393`.
- No new lesson was added: the legacy-test admission mismatch was another instance
  of the existing synthetic-harness configuration-shape lesson, fixed by making
  harness profile admission explicit while preserving a real config-creation test.
- Per user direction, repository-wide pytest was not run; only modified/touched Library component and direct-owner gates are claimed.
