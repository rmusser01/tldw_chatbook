---
id: TASK-22988
title: Resume prior character chats from Roleplay
status: In Progress
assignee: []
created_date: '2026-08-26 15:14'
updated_date: '2026-08-27 07:09'
labels:
  - roleplay
  - console
  - ux
  - conversations
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-26-roleplay-resume-prior-character-chat-design.md
  - >-
    backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users who discover a saved local character conversation in Roleplay need to continue that authoritative chat in Console, rather than only copying a bounded transcript into draft context. Resuming must preserve the saved conversation and historical character behavior while keeping Roleplay as a read-only discovery surface.
<!-- SECTION:DESCRIPTION:END -->

## Renumbering provenance

Renumbered from TASK-22507 to TASK-22988 after the 2026-08-27 rebase exposed a
concurrent ID collision. The Full semantic-capture task was created at 14:34 and
keeps TASK-22507 under the older-arrival rule; this Roleplay Resume task was
created at 15:14 and therefore moves with all of its references.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Selecting a recent saved local character conversation opens its read-only preview, hides card-level actions, and Back to card restores them.
- [x] #2 The preview exposes Resume chat as the sole primary action in the approved three-row vertical hierarchy, with contained controls and usable transcript space at 80x24 and standard widths.
- [x] #3 Preview loading, empty, and failure states are distinct; Resume remains available from a valid row even when preview text cannot load.
- [x] #4 Resume passes only the validated local conversation ID to Console and never aliases the bounded transcript handoff or RAG scope.
- [x] #5 An already-live matching Console session is activated without duplication and preserves its live draft and settings; the active matching duplicate wins.
- [x] #6 A closed conversation is restored through the canonical Console path with its saved tree, active leaf, prompt, roleplay provenance, policies, speech preferences, and pinned prefill within existing safety limits and without a provider call.
- [x] #7 Earlier pending Console intents reach their existing terminal or transient-release outcome first, after which Resume becomes the final active-session target and the Console composer receives focus.
- [x] #8 Missing, failed, or cancelled resume preserves the prior active Console session and durable conversation and removes only a partial runtime session created by that attempt.
- [x] #9 Historical character behavior remains authoritative: version-2 metadata stores the character-name snapshot, version-1 conversations are not guessed or backfilled from the current card, and future versions remain fail-closed.
- [x] #10 Send transcript to Console draft remains a separate bounded 6000-character context action, and Open in Library remains unchanged.
- [x] #11 Targeted automated tests cover navigation, ordering, live-session reuse, hydration, failure atomicity, metadata compatibility, focus, contrast, and compact layout behavior.
- [x] #12 ADR-046 is amended before implementation to record the metadata-version and historical character-name authority decision.
- [x] #13 Preview ownership is bound to the exact selected character and conversation, and a stale async render cannot mutate the shared transcript after ownership changes.
- [x] #14 Changed public APIs document parameters and returns in Google style, and the resume conversation-ID limit uses a descriptive named constant.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Amend ADR-046 before code to define roleplay metadata v2, historical character-name authority, v1 no-guess behavior, and no schema migration.
2. Add failing metadata and persistence tests, then write v2 roleplay context and persist the saved character-name snapshot through existing merge-safe seams.
3. Add failing hydration tests, then restore the saved character identity through canonical hydration and remove current-card name lookup from the resume path.
4. Add failing rollback and opener tests, then share exact runtime cleanup and make the canonical Console opener active-match-first, tri-state, and cancellation-safe.
5. Add failing navigation-order tests, then capture an ID-only pre-mount context and settle earlier pending Console intents before Resume becomes the final active target.
6. Add failing Roleplay UI tests, then implement distinct preview states, hidden card actions, the three-row Resume hierarchy, single-flight navigation, focus behavior, compact layout, and generated CSS.
7. Run the documented targeted feature gate, regenerate consolidated CSS, self-review against all acceptance criteria, add implementation notes/evidence, and mark Done only when the repository Definition of Done is satisfied.

Detailed executable plan: Docs/superpowers/plans/2026-08-26-roleplay-resume-prior-character-chat-implementation.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Implemented ID-only Roleplay-to-Console resume through the canonical active-first Console opener. Metadata v2 preserves the historical character-name snapshot; v1 does not guess and future versions fail closed.
- Added exact-object atomic restore/rollback, tri-state outcomes, cancellation propagation, once-only ordered startup ownership, final projection/focus authority, and prior-session repaint on every failed open.
- Added the read-only Roleplay preview with retained card-action hiding, distinct loading/empty/error states, the explicit three-row action hierarchy, per-target attempt guards, and Back-owned load invalidation. The bounded transcript draft handoff and Library navigation remain separate.
- Core files span Console metadata/persistence/store/hydration/workspace/startup, Roleplay preview/controller/widgets, production CSS, focused Chat/UI tests, and ADR-046.
- Plan deviations: the joined gate exposed and corrected a historical hermetic-config fixture bug without changing production; whole-branch review added one consolidated regression-fix wave; the builder-owned app CSS bundle was included so source and production output remain synchronized.
- Verification: the exact nine-file targeted gate passed 1202/1202 with zero failures or skips; CSS build and bundle sync passed; Ruff passed across all 22 changed Python/test files; diff checks passed; final whole-branch re-review found all findings addressed and reported Ready to merge. The full repository suite was not run, per repository instruction.

Post-push Qodo follow-up: addressed all ten findings. Added the named resume-ID limit and Google-style public API documentation; bound preview ownership to the exact selected character; invalidated previews before character-detail awaits; and serialized token-guarded transcript replacement with two mounted race regressions. No new ADR is required because this hardens the existing Roleplay-read-only/Console-owner boundary without changing storage or cross-module authority.

Final rebase/required-check follow-up: rebased onto origin/dev 6bed8d6f. Generated stylesheet conflicts were resolved by rebuilding from source modules. Investigated the failed Derived Artifacts gate, reviewed all six changed diagnostic statements (+4 workspace, +2 chat_screen), removed the raw conversation ID from the saved-chat presentation warning, added a regression proving the identifier is absent from persistent logs, and regenerated Docs/security/production-diagnostic-inventory.json. The complete six-check local equivalent of the required job passed, as did the rebased nine-file feature gate (1204/1204). ADR check remains unchanged: ADR-046 governs this boundary; no new ADR is required.

Merge-candidate rebase follow-up: rebased onto origin/dev c6218918. The joined console_chat_store conflict preserves both exact Resume rollback and upstream semantic-capture policy hydration; generated CSS and the reviewed diagnostic inventory were regenerated from the combined tree. The repository's six derived-artifact checks, backlog ID guard, Ruff across all changed Python/test files, and diff checks pass. The focused gate recorded 1,201 passes; its only three failures are unchanged origin/dev tests that still import CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER from chat_screen after upstream moved that constant to Console_Modules.realtime. The two unaffected long UI shards passed 300/300 and 376/376. The task was renumbered from TASK-22507 to TASK-22988 under the older-arrival rule after the rebase introduced the already-landed semantic-capture TASK-22507. All ten Qodo findings remain addressed and their review threads resolved. ADR check remains unchanged: ADR-046 applies and no new ADR is required.

Final merge-window refresh: while the required run for c6218918 was queued, origin/dev advanced to 9f7b914a (citation tests and backlog documentation only). Rebased cleanly onto 9f7b914a. Post-rebase evidence: all six required derived-artifact commands passed; Resume startup navigation passed 17/17; the Roleplay conversation-focused workbench subset passed 37/37; metadata/hydration/resume/workspace paths passed 156/157, with the sole failure being the unchanged origin/dev test_console_resume_active_path import of CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER from its pre-refactor chat_screen location. The feature source paths had no conflict or upstream semantic change.

Final queue refresh: required run 33044126646 for the prior head was externally cancelled after 58 minutes with zero steps or logs. origin/dev then advanced through 40ba8fe7 and 37dda3ca. The final rebase onto 37dda3ca preserved the upstream Console review/selection-controller extraction; its sole conflict was the generated production diagnostic inventory, which was regenerated from the combined source tree. Post-rebase verification passed: Resume startup navigation 17/17; Roleplay conversation workbench 37/37; Console workspace tests; upstream controller-wiring and review-selection tests; all six required derived-artifact commands; Ruff across all 22 changed Python/test files; and git diff --check. All ten Qodo findings remain addressed and resolved. ADR check remains unchanged: ADR-046 applies and no new ADR is required.
<!-- SECTION:NOTES:END -->
