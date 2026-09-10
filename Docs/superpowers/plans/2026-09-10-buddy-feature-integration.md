# Integrate the approved Buddy character and Petdex work

Status: In progress
Creator: tldw-project

## Scope and authority

Finish the approved programme in the existing 2026-09-07 Buddy character specs,
preserving current dev's independent Buddy ownership and Console lifecycle. The
implementation exists on `codex/buddy-import-design` but was never shipped. This
plan integrates that implementation; historical verification is not evidence for
the combined code.

ADR required: no new architectural decision for restoration.
ADR paths: existing ADR-074, playback ADR-144 and Petdex ADR-145 on the source branch.
Reason: implement already approved contracts; reconcile the two colliding ADR IDs
against cached refs before publication. An independent-management entry point must
extend the existing ownership ADR explicitly before its implementation.

## Global Constraints

- Users can import a Petdex companion as a Buddy and create an independent,
  editable character from a Buddy. No live dependency on the original Buddy.
- Dynamic plays available animation with static fallback. Static changes expression
  while freezing its animation. Reduce motion and disabled animations take precedence.
- Retain current dev's independent ownership APIs, revision checks, owner bindings,
  snapshot fields and stronger path/error validation.
- Imported pets retain their actual creators and terms. Missing license remains
  unspecified. No downloaded executable instructions are run.
- Use existing guarded publication; preparation alone is not installation.
- Targeted tests only. Label headless application evidence accurately.
- Leave main checkouts and unrelated worktrees untouched. Root owns GitHub publication.

## Task 1: Integrate the existing approved implementation onto current dev

Work only in `/private/tmp/chatbook-buddy-qualification`, branch
`codex/buddy-petdex-qualification`, based on dev `16c72b5b1e`.

Source is the clean branch `codex/buddy-import-design` at
`b4e460f75142b37ff3df277fd712bdb1812007f9`, available at
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/buddy-import-design`.
Read that branch's tasks 32023, 32024, 32025, 32031 and their linked specs/ADRs,
plus applicable AGENTS and lessons-testing-evidence, lessons-live-verification,
lessons-backlog-hygiene and lessons-bounded-artwork-imports before edits.

1. Preserve the two design commits and integrate the four implementation commits
   in order: 3201becb23 playback, a552b8edad attribution, 451137c93e conversion,
   b4e460f751 Petdex. A merge preserving their history is acceptable when resolving
   the same final behavior. Do not replace dev files wholesale.
2. Reconcile duplicate native artwork validation with dev's `Persona_Visual/artwork.py`.
   Keep dev's snapshot metadata, independent Buddy ownership and newer errors.
   Preserve the missing Actor Pack checksummed attribution carrier and lineage.
3. Resolve task/ADR identity conflicts according to repository policy. Tasks 32023,
   32024, 32025, 32031 have no cached-ref collisions. ADR128 conflicts with a library
   presentation ADR; ADR134 conflicts with native-goal-runs fleet admission. Scan
   all cached refs for available numbers and fix new feature references consistently.
4. Regenerate CSS through the supported build command. Run targeted playback,
   attribution, conversion, Petdex, native snapshot/publication and Console Buddy
   regression tests; report exact commands/results. Use existing main `.venv` with
   this worktree and `packages/tldw_profile_core/src` on PYTHONPATH.
5. Update implementation records to distinguish historical tests from integration
   results. Use Backlog CLI for task state/notes. Commit all owned integration files.
6. Identify any remaining gap exposing Petdex import or character conversion through
   current independent Buddy management. Do not claim Persona-only paths fulfill
   independent-management support; report the exact seam for the next task.

Acceptance: all four missing feature paths present and verified on combined code;
existing independent Buddy import/management tests still pass; no duplicate ADR
identity; no third-party artifact binaries committed.

## Task 2: Finish independent Buddy installation and conversion journeys

After Task 1 review, amend the existing independent-ownership ADR/spec for the
smallest management entry points that reuse restored review/publication services.
Use the live prepared Homelander archive under `/private/tmp/petdex-live-install-20260910`
to qualify guarded app save, offline reload, exact attribution, independent character
creation and both motion modes in a disposable profile. Keep existing Persona
authoring routes working. Record headless versus native evidence honestly.

The implementation brief will be completed from Task 1's verified seams before
dispatch. Update corresponding tldw-stuff skill installation guidance after the
actual supported journey works. Review the full branch, create PR against dev,
address posted review and required checks, and merge under existing authorization.

## Task 1 integration closeout

Completed the history-preserving reconciliation onto dev 16c72b5b1e. Playback and
Petdex ADRs are now 144 and 145 after cached-ref collision checks. Fresh targeted
evidence, baseline limitations, preserved ownership, and exact Task 2 seams are
recorded in [the integration verification](../reviews/2026-09-10-buddy-feature-integration-verification.md).
Task 2 and live installation/server qualification remain pending with the root task.
