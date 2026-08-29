---
id: TASK-19872
title: >-
  Double-pressing Settings Load Backup reports edits were kept that never
  existed
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-22'
updated_date: '2026-08-29 18:09'
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
- [ ] #1 Pressing "Load Backup" twice in quick succession does not report that
      unsaved edits were kept when the user made none
- [ ] #2 The guard distinguishes a write the application itself made from a genuine
      user edit
- [ ] #3 A real user edit made after the newest backup load is dispatched is
      still protected — that load declines to overwrite the edit and still says so
- [ ] #4 A test drives both cases (double-press with no user edit; single press
      with an edit made while the read is in flight) and asserts the message, and is
      mutation-checked
- [ ] #5 The interleaving is reproduced before the fix is written, and what was
      observed is recorded — this task was reasoned from code, not driven
<!-- AC:END -->



## Notes

Low severity and deliberately so: the outcome is correct, only the explanation
is wrong. It is worth fixing because a spurious "we protected your edits"
teaches users to distrust the message on the occasion when it is true.

## Implementation Plan

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
