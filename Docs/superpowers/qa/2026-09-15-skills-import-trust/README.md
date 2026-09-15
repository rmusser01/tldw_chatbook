# Skills import and trust — TASK-32655

Reviewed on `feat/component-pattern-library`, based on `53890aa2bd`.
This continues the Skills editor review through import, candidate selection,
trust setup, captured review and exact approval.

## Changes

- Trust setup and passphrase dialogs now use application design tokens for
  surfaces, spacing, sizing, foreground and readable errors. No token values changed.
- Sort and Import remain readable in narrow Items. Import's file/folder browsers
  and submission actions occupy separate rows. Import also opens beside a selected
  Skill, while an unsaved editor draft vetoes the transition.
- Import errors return to the path; candidate Cancel preserves the draft;
  successful import reaches Review. Deferred returns respect newer keyboard focus.
  Review remains visible when captured content expands, and approval returns to Trust.
- Disabled mount-time path events cannot erase a committed import receipt.
  A delayed presentation callback cannot restore status from an older operation.
- Retained Items refreshes after trust changes even while Work shows an editor.
  All three obsolete list-only refresh gates were removed. The editor refresh
  updates Items independently, preserving Work widgets and edits made during I/O.

ADR required: no. Existing ADRs [009](../../../../backlog/decisions/009-local-skill-trust-boundary.md),
[076](../../../../backlog/decisions/076-library-lifecycle-progressive-disclosure.md),
[086](../../../../backlog/decisions/086-library-adaptive-reader-shell.md),
[150](../../../../backlog/decisions/150-design-token-system-and-design-language.md)
and [161](../../../../backlog/decisions/161-component-pattern-library.md) govern
the unchanged trust boundary, ownership, layout and design language.

## Automated evidence

**355 distinct targeted checks pass.** [verification.json](verification.json)
records groups and overlap. The 298 Skills checks initially had two failures:
legacy backdrop tests loaded only consolidated class styles after the trust
dialogs moved to app CSS. All 13 modal contracts pass with the production app
stylesheets loaded. The other 296 checks passed in the combined run.

New coverage includes both themes at 170×48 and 80×24, rendered error contrast,
natural focus returns, candidate Cancel and selection, stale approval followed by
fresh approval, selected-Skill Import, dirty veto, newer keyboard focus, and a
forced delayed-input/presentation race. Prior editor tests also preserve newer
draft text while a write completes. Old callback fakes now execute the scheduled
callback; the real-package test uses the Library harness instead of assuming the
application's initial screen is Library.

The targeted command covered:

```text
Tests/UI/test_library_skill_import_trust_journeys.py
Tests/UI/test_library_skill_editor_journeys.py
Tests/Skills/test_skills_import.py
Tests/Skills/test_skill_import_choice_modal.py
Tests/Library/test_skill_trust_review_preview.py
Tests/Skills/test_skills_library_flow.py
Tests/Skills/test_skill_trust_service.py
Tests/UI/test_library_skills_canvas.py
Tests/UI/test_library_skills_reader.py
```

Wiring, design-token governance, bundle regeneration, CSS consolidation and the
Skills controller size checks also ran. Two inherited governance failures remain:

- CSS consolidation reports the same 22 pre-existing allowlist offenders;
  [baseline comparison](css-baseline.json) confirms zero additions.
- LibraryScreen is 35,202 lines against its 33,204 budget, one line smaller than
  HEAD. The Skills controller is 3,135 lines within its 3,142 budget. No budget changed.

New tests, the native runner and Work pane pass Ruff and formatter checks.
Modified existing files add no Ruff code/message diagnostics against HEAD
([comparison](lint-final.json)). CSS was rebuilt from sources and `git diff --check`
passes. No full repository suite was run. Independent production review found
no remaining issue in this slice.

## Native and persistence evidence

The final run is `run-005`, driven by [native_check.py](native_check.py) using
actual `TldwCli`, `LinuxDriver` and an owned tmux terminal. The
[isolated profile](isolation.json) uses synthetic files, private databases and a
file-backed generation marker with the null keyring backend. Production trust,
provider calls, script execution and grants were not exercised.

The baseline Skill is seeded through the local service. Package files, field
values and the post-capture disk edit are synthetic fixtures. Library routing
uses the existing screen action; visible controls are focused and activated
with Enter. Return-focus assertions do not set focus. Terminal sizes are changed
through tmux. Trust setup and approval use the real dialogs and services.

At 170×48 dark and 80×24 light, the [run result](result.json) verifies invalid-path
recovery, candidate Cancel, selecting only zeta, blocked initial import, literal
captured contents, rejection after a disk change, fresh review and successful
exact approval. Both final focus targets are the Trust tab. Bootstrap blank and
mismatch validation and approval's blank validation are exercised too.

All twelve SVGs were rendered through Quick Look and inspected. Representative
captures: [setup dark](bootstrap-170.svg), [setup light](bootstrap-80.svg),
[approval dialog](passphrase-80.svg), [compact import](import-error-80.svg),
[candidate chooser](choice-80.svg), [captured review](review-80.svg),
[approved wide](approved-170.svg), [approved compact](approved-80.svg).
An existing stale-approval notification remains over the lower pane in some
captures; the focused controls and selected candidates remain readable.

Normal Ctrl+Q was requested by the driver; a subsequent Ctrl+Q through the
terminal preceded observing `app.run` return and exit code 0. The owned shell
was then closed. No application error or unhandled exception appears in the log.
[Persistence checks](persistence.json) verify both exact SHA-256 values,
absence of unselected alpha packages, ten SQLite integrity results and zero
messages. A fresh trust-service instance starts locked and verifies both Skills
as trusted after unlock. This is service-reopen evidence, not a full app restart.

Earlier failed attempts are diagnostic only: they exposed focus and retained
Items defects, plus runner ordering, profile setup and stale attribute assumptions.
The final run above supplies native qualification.

Next: Skills Files and supporting-file interactions, then the remaining Library
destinations. Integration into `dev` remains pending.
