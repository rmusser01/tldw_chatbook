# Integrate the approved Buddy character and Petdex work

Status: Implementation and qualification complete; PR integration pending
Creator: tldw-project

## Scope and authority

Finish the approved programme in the existing 2026-09-07 Buddy character specs,
preserving current dev's independent Buddy ownership and Console lifecycle. The
implementation exists on `codex/buddy-import-design` but was never shipped. This
plan integrates that implementation; historical verification is not evidence for
the combined code.

ADR required: yes for the independent-management destination and partial-Apply
contract; no new decision for the restored playback/conversion/Petdex implementation.
ADR paths: existing ADR-074, ADR-139, playback ADR-144 and Petdex ADR-145, plus
supplemental ADR-146 for independent Buddy publication and character entry points.
Reason: implement the approved contracts and record the new destination without
rewriting Accepted ADR-139/145. ADR-146 partially supersedes only ADR-145's
saved-Persona-only restriction; saved Persona authoring and the source/HTTPS trust
decisions remain in force while the independent destination adds its recovery order.

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

Tracked by TASK-32238. Read that task and supplemental ADR146 before implementation.
ADR required: ADR146 supplements existing ADR139 and partially supersedes only
ADR145's saved-Persona-only destination restriction; retain saved Persona authoring,
ADR074 conversion, ADR144 playback and ADR145 source/HTTPS trust contracts.
No schema or runtime ownership change.

Work only in this worktree. Keep existing Persona authoring and archive character
routes working. Root owns live downloaded-pet qualification and tldw-stuff changes.

1. Add clear `Import from Petdex` and `Create character` entry points to the
   existing Console Buddy & Persona Management modal. The first uses the restored
   PetdexImportReviewDialog and native archive preparation without a saved Persona.
   The second operates on the selected installed independent Buddy.
2. Reuse BuddyLibrary.review_archive/publish_review for independent publication.
   Petdex review stages content for management Apply; cancelling the review or
   management form before Apply must not publish a Buddy or change preferences.
   Publication precedes settings persistence, so a partial Apply leaves the installed
   Buddy durable and the previous settings selected; retry the same form or reopen and
   verify the installed Buddy before importing again. Keep temporary bytes/source
   guards owned and released on all exit paths. Preserve notices, unspecified licenses
   and source mapping provenance. No executable source text.
3. Extend the saved snapshot boundary to read a real independent Buddy owner with
   its revision/version guard, preserving existing Persona snapshot callers. Reuse
   BuddyCharacterReviewDialog and existing character publication. Explicit Create
   commits one independent editable character; closing the management form later
   does not undo that explicit action. Creating a character must not apply staged
   Buddy follow-target or Persona preferences.
4. Keep source, profile and destination guards across asynchronous preparation,
   review and final publication. Reject stale source/selection/profile changes;
   preserve previous content/settings on errors. Run file/network work off the UI
   thread, drain owned work before cleaning staging. Show only actionable controls;
   connect the existing explicit Open in Console handoff if that action is shown.
5. Add discriminating unit and mounted workflow tests: Petdex review/cancel and
   management cancel leave counts/settings unchanged; Apply installs exactly one
   independent Buddy without creating a Persona; stale source/profile and failed
   publication remain safe; selected-Buddy conversion retains credits/expression
   bytes after source change and uses both motion modes. No fake publication or
   direct DB mutation in the end-to-end journey. Targeted tests only.
6. Update Buddy user guidance and current programme spec to name independent
   management as the normal route. Regenerate CSS with the supported build if
   defaults change. Run affected tests, formatting/lint and source-generation checks.
   Commit owned files, update task plan/notes with CLI, but leave live qualification
   AC and task completion to root until the real downloaded archive journey passes.

Existing seams: UI/Navigation/buddy_management.py owns application integration;
Widgets/Persona_Widgets/buddy_management_modal.py owns staged choices;
Petdex/review.py and petdex_import_review.py supply reviewed native archive bytes;
Persona_Visual/snapshot.py currently reads saved Persona owners only;
UI/Persona_Modules/buddy_conversion.py shows existing character review/publication
and explicit Console handoff. Avoid a second importer or conversion implementation.

The real prepared Homelander archive lives under
`/private/tmp/petdex-live-install-20260910`, with source/creator receipt at
`/private/tmp/petdex-live-preparation.json`. Root will qualify actual app save,
offline reload, attribution, independent character and rendering against your final
code in a disposable profile, without committing the third-party asset.

After Task2 review/qualification, root updates the collection installer guide,
performs final review, publishes against dev, resolves review/checks and merges.

## Task 1 integration closeout

Completed the history-preserving reconciliation onto dev 16c72b5b1e. Playback and
Petdex ADRs are now 144 and 145 after cached-ref collision checks. Fresh targeted
evidence, baseline limitations, preserved ownership, and exact Task 2 seams are
recorded in [the integration verification](../reviews/2026-09-10-buddy-feature-integration-verification.md).
Task 2 and the downloaded-source headless application qualification are complete;
see [the journey verification](../reviews/2026-09-10-independent-buddy-journey-verification.md)
for final review fixes, exact results, rulings and native/physical acceptance limits.
Server follow-ups landed in PRs 2940 and 2941. Root owns final Chatbook PR integration
and the dependent collection installer PR 19.
