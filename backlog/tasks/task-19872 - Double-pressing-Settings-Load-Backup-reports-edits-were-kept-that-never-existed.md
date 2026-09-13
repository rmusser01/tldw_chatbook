---
id: TASK-19872
title: >-
  Double-pressing Settings Load Backup reports edits were kept that never
  existed
status: Done
assignee:
  - '@codex'
created_date: '2026-08-22'
updated_date: '2026-08-29 19:23'
labels:
  - ux
  - settings
  - concurrency
dependencies:
  - TASK-19559
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: surfaced by **TASK-19559**'s reviewer while checking the arrival-time
guards that task introduces.

**Historical filing context:** this did not reproduce at `3605bd52d` because
TASK-19559 was still in flight. At that commit `_advanced_load_backup_worker`
(`UI/Screens/settings_screen.py:8601`) was a plain
`@work(exclusive=True, thread=True)` with no arrival guard. TASK-19559 has since
merged in `f12bb21adf`, and current `dev` contains both the guard and the
double-completion sequence described below. The implementation must still
reproduce the interleaving before writing the fix; the original report was
reasoned from code rather than driven.

The shape: TASK-19559 replaces the worker's exclusivity with an arrival-time
guard — before applying a loaded backup, the callback compares the editor's
current text against what it expects, and declines to overwrite if the user has
typed something in the meantime. That is the right instinct. But the guard
compares against the editor's *live* content, and the first callback's own
write into the config `TextArea` is indistinguishable from a user edit.

So on a rapid double-press of "Load Backup":

1. two workers are dispatched
2. the first completes and writes the backup text into the editor
3. the second completes, compares, sees text it did not expect, and declines —
   reporting that the user's unsaved edits were preserved

The user had no unsaved edits. The application invented a reason for declining,
and told the user about work it protected that never existed.

**Reasoned from the code, not driven.** The sequence above follows from reading
the callback and the worker dispatch; it has not been reproduced in a running
app. Whoever picks this up should confirm the interleaving before fixing it.

No data is at risk — declining is the safe direction, and the backup text is
already in the editor from the first callback. The defect is that the message
is false, which is the same family as TASK-19550 / TASK-19861 / TASK-19869:
*the app describes an outcome it did not produce.*
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pressing "Load Backup" twice in quick succession does not report that
      unsaved edits were kept when the user made none
- [x] #2 The guard distinguishes a write the application itself made from a genuine
      user edit
- [x] #3 A real user edit made after the newest backup load is dispatched is
      still protected — that load declines to overwrite the edit and still says so
- [x] #4 A test drives both cases (double-press with no user edit; single press
      with an edit made while the read is in flight) and asserts the message, and is
      mutation-checked
- [x] #5 The interleaving is reproduced before the fix is written, and what was
      observed is recorded — this task was reasoned from code, not driven
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add deterministic, bounded-handshake mounted Textual tests that reproduce both overlapping
   completion orders, a stale error, normal serial repetition, and the existing
   post-dispatch typing protection; record the RED failure before production
   changes.
2. Add one monotonic backup-load request token to `SettingsScreen`, capture it
   at the button boundary, carry it through the existing worker, and return
   from stale callbacks before any UI or state side effect. Keep the existing
   dispatch-text guard for the newest request.
3. Mutation-check the token guard and dispatch-text guard independently, then
   update the Settings user guide and this task with the observed evidence.
4. Run the focused Settings gate, Ruff lint/format, diagnostic-inventory guard,
   pre-import payload ratchet, and both diff checks. Refresh only the diagnostic
   inventory if the scoped source edit causes reviewed line movement; the
   pre-import snapshot is diagnostic context and must not be refreshed.
5. Self-review and independently review the complete branch, then check the
   acceptance criteria and mark the task Done only after the final gate passes.

ADR required: no

ADR path: N/A

Reason: this is a localized concurrency bug fix that preserves the existing
Settings worker, UI, and state ownership boundaries; it introduces no durable
architecture or policy decision.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Reproduced the race before the fix with deterministic worker-start and
  callback-return handshakes: the exact focused run was `FFF.` (the serial
  repeat already passed). An old callback arriving first mutated the original
  editor and validation state to the old backup. When the newest callback
  arrived first, a stale old success overwrote the newest result with the false
  "unsaved edits were kept" refusal, while a stale old error overwrote the
  newest result with the old error. Added the minimal fix: one monotonic integer
  `_advanced_backup_load_token`.
- Each **Load Backup** press advances the token, the worker carries it to the
  callback, and a stale callback returns before changing the editor, result, or
  validation state. Pressing **Load Backup** still authorizes replacement of a
  pre-existing draft; the existing dispatch-text guard separately preserves
  typing made after the newest press. Serial successful repeats retain the
  ordinary loaded-preview success instead of manufacturing an unsaved-edit
  warning.
- Converted the existing genuine-typing characterization from sleeps to the
  same bounded event handshakes. The exact focused GREEN run was `7 passed`.
  Removing the token guard made all three overlapping-load parameters fail
  (`FFF`), and restoring it made them pass (`3 passed`). Removing the
  dispatch-text guard made the typing test fail because the backup overwrote
  the typed text; restoring it made the test pass.
- **Authoritative final integration run:** post-rebase verification on
  `origin/dev` `0d83188e14090068a37ae6523005e64e2be7998e` passed the focused
  Settings gate (`11 passed, 381 deselected`) and pre-import payload ratchet
  (`1 passed`; 491/500 modules, 9 headroom; 379551/380000 LOC, 449 headroom;
  142034/145000 `library` route LOC, 2966 headroom). The canonical diagnostic
  inventory passed without a rewrite at 540 owners, 1277 TASK-492 calls, 7337
  TASK-494 calls, and 8 sink files; scoped Ruff lint and both diff checks
  passed. Whole-file Ruff format remains a qualified current-`origin/dev`
  baseline exception rather than a pass: `--check` reports both scoped Python
  files would be reformatted, while `--diff` has only the identical baseline
  hunks (with line-offset differences) and no TASK-19872 hunk, so no file was
  formatted. The isolated worktree used the repository virtualenv at
  `../../.venv/bin/python`; pytest also emitted the known dependency and
  protected historical temp-cleanup warnings.
- Full-branch diff review found only the intended Settings implementation,
  focused tests, user guide, design, plan, and Backlog task filename
  normalization. It found no unrelated Settings refactor, button disabling,
  abstraction, dependency, or generated-artifact drift. No generalizable new
  incident arose, so no lessons entry was added.
- Commits and files: `14dfc5303b` added the deterministic tests in
  `Tests/UI/test_settings_configuration_hub.py`; `12bf836427`
  added the token guard in `tldw_chatbook/UI/Screens/settings_screen.py` and
  adjusted its worker-dispatch test. Commit `b3b0b2a088` updated
  `Docs/User_Guide/settings.md` and this task file; the closeout-only commit
  updates only this task's status and final evidence.
- ADR required: no. The localized concurrency fix preserves the existing
  worker, UI, and state ownership boundaries.
- PR review validated and addressed Qodo's two maintainability findings: the
  public backup-load handler now documents its event and newest-request
  ownership behavior, and the deterministic handshake tests share descriptive
  event-wait and worker-release timeout constants. The focused Settings gate
  remained `11 passed`, the pre-import ratchet remained `1 passed` at 379556
  LOC, and scoped Ruff lint, diagnostic inventory, formatter-diff inspection,
  and both diff checks passed with the existing whole-file formatter baseline.
<!-- SECTION:NOTES:END -->

## Notes

Low severity and deliberately so: the outcome is correct, only the explanation
is wrong. It is worth fixing because a spurious "we protected your edits"
teaches users to distrust the message on the occasion when it is true.
