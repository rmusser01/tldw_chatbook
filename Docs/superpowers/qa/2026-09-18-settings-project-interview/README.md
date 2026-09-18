# Project-context interview — TASK-32775

The current question and typed answer stay visible together at compact size.
Focused answer and final-review fields use the existing token focus surface and
side accent without an outline covering their first text row. Failed or rejected
submissions keep the editable answer; successful acceptance clears it once.
Queued presses cannot submit a pending answer twice.

After finishing, Close restores a usable Review action. The screen retains the
known successful review transition instead of making another fallible draft read.
Reopening preserves applied edits rather than rebuilding from original answers.
Existing ADR-102 governs profile authority and encryption; ADR-150/161 govern
presentation. No new token, schema, service boundary or ADR was required.

## Targeted verification

**140 distinct cases pass across final and related runs:**

- [Ten new keyboard/state regressions](answer-final-tests.txt) cover dark/light
  × 80×24/170×48 paint, ValueError/RuntimeError recovery, pending duplicate
  presses, focused review values, Close/reopen and preserved applied edits.
- [84 existing interview, review, first-run and coordinator cases](related-tests.txt)
  and [14 workspace handoff cases](handoff-tests.txt) pass.
- [31 token, component and generated-CSS guards](governance-tests.txt) pass.

The original failures reproduce [missing question/value paint and lost rejected
answers](answer-red.txt), [hidden Review after Close](review-return-red.txt),
[covered focused review values](review-value-red.txt), and [the edited-review
return defect with a real coordinator](edited-review-return-red.txt). [Nine callback cases](final-callback-tests.txt) pass after the cached-state
correction; those cases are already included in the 84, not additional coverage.
Running the screen module alone first hit seven setup errors from the known
config-source selection guard. Collecting the same four related modules as the
84-case run preserves their existing import context; all nine selected cases pass.
The new ten cases use explicit private-profile child processes.

[Static checks](static.txt) show four unchanged legacy Ruff diagnostics, clean
new files, scoped formatting and diff hygiene. [Independent review](review.md)
found and then verified the edit-preservation correction; no introduced blocker
remains. No full suite or provider requests were run.

The CSS performance guard on commit `38e3cd5889` found one additional ancestor-scoped bare type rule (275 against the existing limit of 274). The correction gives both review Inputs a dedicated class and scopes their focus rule to it. The [unchanged ratchet passes](ci-selector-tests.txt); [bundle and scoped static checks](ci-static.txt) pass. The [combined focus/ratchet attempt](ci-focused-review-tests.txt) passed the focus case but hit a Console-store fixture startup error before the census; the isolated ratchet run passes. The focus case overlaps the ten cases above. The final native matrix was refreshed for this selector correction.

## Native review

The [runner](native_check.py) uses real TldwCli/LinuxDriver with TTY-backed output
and private HOME/config/data selected before imports. It initializes Tool Profiles
before creating workspaces. The real workspace registry creates automatic
Personas, and a real fixed local coordinator writes context through the encrypted
Personal Context repository with a private production passphrase protector.
Interview drafts use the disclosed memory-only fallback; durable draft resume
and external/adaptive interview execution are outside this native slice.

Each of four size/theme cells creates two workspaces through Settings. Cancelling
one interview discards its unsubmitted answer, preserves the workspace and writes
no canonical context. The second injects one failed answer write, verifies visible
retained input, accepts the retry and another answer, then edits the final review.
Closing and reopening preserves that applied edit even with resume unavailable.
Deselecting the second item and choosing Save only persists exactly the first
reviewed payload, leaves runtime disabled and preserves assistant defaults.
The encrypted repository is reopened and each selected record is checked again.
Controls receive focus before keyboard activation; these are action/paint checks,
not an exhaustive Tab traversal of every modal control.

| View | Dark compact | Light compact | Dark wide | Light wide |
| --- | --- | --- | --- | --- |
| Focused question and answer | ![textual-dark 80x24](textual-dark-80x24-answer.svg) | ![textual-light 80x24](textual-light-80x24-answer.svg) | ![textual-dark 170x48](textual-dark-170x48-answer.svg) | ![textual-light 170x48](textual-light-170x48-answer.svg) |
| Retained answer after failure | ![textual-dark 80x24](textual-dark-80x24-retry.svg) | ![textual-light 80x24](textual-light-80x24-retry.svg) | ![textual-dark 170x48](textual-dark-170x48-retry.svg) | ![textual-light 170x48](textual-light-170x48-retry.svg) |
| Review restored after Close | ![textual-dark 80x24](textual-dark-80x24-review-return.svg) | ![textual-light 80x24](textual-light-80x24-review-return.svg) | ![textual-dark 170x48](textual-dark-170x48-review-return.svg) | ![textual-light 170x48](textual-light-170x48-review-return.svg) |
| Save controls after selection | ![textual-dark 80x24](textual-dark-80x24-review.svg) | ![textual-light 80x24](textual-light-80x24-review.svg) | ![textual-dark 170x48](textual-dark-170x48-review.svg) | ![textual-light 170x48](textual-light-170x48-review.svg) |
| Discard disclosure | ![textual-dark 80x24](textual-dark-80x24-cancel.svg) | ![textual-light 80x24](textual-light-80x24-cancel.svg) | ![textual-dark 170x48](textual-dark-170x48-cancel.svg) | ![textual-light 170x48](textual-light-170x48-cancel.svg) |

All 20 final SVG captures were rendered and visually inspected. Run005 used the
exact final source and runner hashes in [native-result.json](native-result.json).
[capture-manifest.json](capture-manifest.json) records all 40 SVG/text hashes.
The [lifecycle receipt](lifecycle.json) verifies normal keyboard shutdown, exit 0,
PID 69704 absent before closing its owned terminal, twelve healthy SQLite
databases, instance-lock reacquisition, zero durable conversations/messages,
no error or faulthandler output, and unchanged default-profile fingerprints.

## Remaining boundaries

The first native attempt exposed a separate cold-start defect: automatic Persona
and profile creation can precede Tool Profile guard readiness, leaving assistant
defaults unset on an already-created workspace. [The failed run receipt](failed-run001-native-result.json)
and [lifecycle](failed-run001-lifecycle.json) preserve that finding. The interview
matrix explicitly initializes Tool Profiles; it does not qualify cold creation.

A [prior successful native run](pre-final-run002-native-result.json) used the
superseded resume-on-Close implementation. Its [lifecycle receipt](pre-final-run002-lifecycle.json)
verifies normal shutdown, but only the final run qualifies the corrected source.
Run003 stopped on a probe error: it inspected a goal payload using the preference field name `value` instead of `outcome`. The app returned normally with exit 1; [its receipt](failed-run003-lifecycle.json) records healthy private databases and unchanged default profiles. The final runner checks the correct goal field. All earlier app processes exited before their owned terminal sessions closed.

The [remaining Persona audit](../../reports/2026-09-18-workspace-persona-remaining-audit.md)
records valid-ID/control-choice collisions, incomplete catalog pagination and the
cold provisioning dependency. Broader Personal Context workflows and preexisting
worker/cancellation lifetime questions remain separate reviews.
