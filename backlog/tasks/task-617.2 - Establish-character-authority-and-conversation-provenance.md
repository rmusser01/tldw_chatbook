---
id: TASK-617.2
title: Establish character authority and conversation provenance
status: Done
assignee:
  - '@codex'
created_date: '2026-07-29 15:43'
updated_date: '2026-07-30 02:20'
labels:
  - roleplay
  - tts
  - identity
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-07-28-tts-character-identity-persona-separation-design.md
  - >-
    backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md
parent_task_id: TASK-617
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give local and server-backed character conversations durable, source-aware authority so later TTS assignment resolution can identify the exact character principal without using mutable routing, credentials, paths, or the currently active context. This increment implements approved Slice 3A.2 only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The character database exposes a durable local authority ID that remains stable across restart and is used to identify eligible local character conversations.
- [x] #2 Each configured server target owns a persisted canonical UUIDv4 authority scope; legacy targets are upgraded atomically before first authority use, and an unpersistable scope is unavailable rather than ephemeral.
- [x] #3 Server character authority is derived from the persisted target scope and the authenticated positive user ID using the approved versioned encoding, is fenced against target/auth-context changes, remains stable across mutable routing details and credential rotation, separates accounts, and fails closed without blocking ordinary text chat.
- [x] #4 The main conversation schema adds nullable assistant_authority_id; migration backfills only provable local character provenance, preserves unproven server provenance as null, and supports opaque server character IDs.
- [x] #5 Application-owned conversation CRUD and backup/restore preserve assistant_authority_id, while current Sync V2 and imports without proven provenance materialize it as null and never infer authority from the receiver's active context.
- [x] #6 Console character sessions carry source-aware assistant identity sufficient to produce an exact CharacterRef only when authority is proven; generic and Persona sessions remain authority-free.
- [x] #7 Focused deterministic tests cover authority stability and isolation, stale-context fencing, fail-closed identity errors, schema migration, CRUD, backup/restore, sync/import exclusion, and Console session identity without changing speech admission or TTS selection.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md
Reason: ADR-037 already governs the authority encoding, authenticated-context fencing, one-column schema migration, backup/restore behavior, privacy rules, and Sync V2 exclusion for Slice 3A.2.

1. Persist redacted canonical UUIDv4 authority scopes on configured targets and durably upgrade legacy missing scopes before authority use.
2. Add the exact expected-target-bound, runtime/auth-context-fenced server-user authority resolver and encoding.
3. Add schema v28, the DB-owned local authority accessor, nullable assistant_authority_id, local backfill, and joint conversation identity validation.
4. Carry provenance through local CRUD/backup while keeping unproven chatbook import and current Sync V2 authority-null.
5. Make native Console sessions persist and restore source-aware character identity and project CharacterRef only when complete.
6. Make Roleplay Start Chat handoffs source/target aware without changing speech or assignment behavior.
7. Run focused regressions/static checks, self-review, and document verification evidence.

Detailed plan: Docs/superpowers/plans/2026-07-29-character-authority-conversation-provenance.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Delivered approved Slice 3A.2. Configured server targets now own persisted,
redacted UUIDv4 authority scopes with atomic legacy upgrade and fail-closed
durability checks. The runtime context derives the exact versioned
`server-user-v1` authority from that scope and the authenticated positive user
ID, while revision, target, client, credential, and authentication-context
fences prevent stale or cross-account publication.

Schema v28 adds nullable `assistant_authority_id` and a DB-owned stable local
authority accessor. Migration backfills only provable local-character rows;
opaque server characters and unproven records remain authority-null.
Application-owned CRUD and SQLite backup preserve provenance. Portable
chatbook, legacy Tavern/SillyTavern, and current Sync V2 boundaries explicitly
remain authority-null, including when an imported character name or numeric ID
matches a receiving local card.

Console sessions persist and restore source-aware assistant identity and
produce `CharacterRef` only for complete, proven local or server character
principals. Roleplay handoffs preserve the selected source and exact server
target, fail closed to an unscoped but text-capable session when identity is
unavailable, and keep generic and Persona sessions authority-free. Local card,
avatar, dictionary, and lore projections remain local-only; review-driven
request-generation, ownership, cancellation, authentication, and stale-result
fences prevent cross-source or superseded UI publication.

Full validation at the then-rebased code revision `00bcdd925` over `origin/dev`
`f3ecd8672`:

- The final 20-file changed-test union produced 1292 passes and 5 inherited
  failures. The exact five nodes and failure shapes reproduce on
  `origin/dev` `f3ecd8672`: four stale module-level `settings` monkeypatches
  and one existing Personas MagicMock-await import-recovery fixture.
- The final legacy-import amendment gate passed 156 Character Chat, schema-v28,
  and Console identity tests. The external audio.cpp/profile regression gate
  passed 589 tests and did not change synthesis or profile selection.
- Changed production Python files pass Ruff. The complete changed-file sweep's
  10 E702 diagnostics are unchanged semicolon lines in
  `Tests/UI/test_console_character_avatar.py` and reproduce on the same
  `origin/dev`; the amendment files are clean. `git diff --check` passes.
- Mypy reports the same 128 errors in the same 7 changed production files on
  this branch and `origin/dev`; no production typing diagnostic was added.
  Added/expanded tests remain subject to existing repository typing debt.
- The required repository-wide gate is blocked during collection by the
  inherited `StreamDone` import error in
  `Tests/Event_Handlers/test_worker_events_contract.py`, reproduced on
  `origin/dev`. A supplemental run excluding that collector reached 654
  passes and 1 skip with no failures before it was intentionally stopped at
  2% because the projected runtime was multiple hours.

ADR-037 remains the governing decision. No dependency, speech-admission,
speech-snapshot, persistent TTS-assignment, managed audio.cpp, production TTS,
or production Sync V2 change entered this slice.

Two implementation-plan deviations were required by review: Roleplay handoff
work added narrow shared-pane/authentication race hardening needed to preserve
the source-provenance contract, and final whole-slice review found the legacy
Tavern/SillyTavern import boundary outside the originally listed chatbook
importer. Both were covered by deterministic regressions and stayed within
AC3, AC5, and AC6. Final independent spec and code review are APPROVED with no
unresolved Critical or Important finding.

The final rebase retained `dev`'s non-blocking Personas mount worker and moved
the source-aware server page reload into that deferred loader. The dedicated
mount-freeze, exact-server-handoff, and source-switch isolation regressions all
pass on the rebased history.

Immediately before merge, the branch was rebased without conflict onto the
newer `origin/dev` revision `13b7f6ee2`. The intervening base changes are
Watchlists-scoped. The non-blocking Personas mount, exact server handoff,
source-switch isolation, and legacy Tavern authority regressions passed again
(4 passed), and `git diff --check origin/dev..HEAD` remained clean.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 All acceptance criteria are checked and implementation notes record the delivered behavior and plan deviations.
- [x] #2 Deterministic authority, migration, CRUD, backup, import, Sync exclusion, Console, and Roleplay tests cover the new behavior.
- [x] #3 External audio.cpp and TTS profile regressions pass without selection or synthesis changes.
- [x] #4 Task-scoped lint, typing comparison, and diff checks pass or exact unchanged `origin/dev` baselines are documented.
- [x] #5 ADR-037, the approved Slice 3A design, and the detailed implementation plan remain current and linked.
- [x] #6 Independent specification and code review have no unresolved Critical or Important finding.
- [x] #7 No speech snapshot, TTS assignment, managed audio.cpp, dependency, production TTS, or production Sync V2 work enters this task.
<!-- DOD:END -->
