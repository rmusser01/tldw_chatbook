# Textual book: follow-up analysis

Revisited the supplied 2025-07-25 edition after implementing the three scoped
Settings tasks in [PR #3176](https://github.com/rmusser01/tldw_chatbook/pull/3176),
on product source rebased onto dev `8136af6f6f`. The book is reference material;
its examples and exercises do not override repository contracts.

## First follow-up completed

**Medium priority: parallel-run programmatic events.** Chapter 7 pp. 144-145
applies beyond the now-owned paste region. A fresh mounted private-profile
probe set saved max_parallel_runs=1 and threshold=50, staged threshold=120,
changed the backing parallel-run value to 2 and called the real sync method.
After queued events settled, draft.values had gained `max_parallel_runs: 2`.
The equality assertion failed with that one extra key (44.09 s, one failed
diagnostic). This is a demonstrated draft-origin error under a controlled
refresh; no data loss or changed production save is claimed.

[TASK-35680](../../../backlog/tasks/task-35680%20-%20Keep-parallel-run-form-refreshes-out-of-Settings-drafts.md)
owns this bounded follow-up under ADR-033. The diagnostic-only probe was kept
out of production/tests committed by this PR; its recorded reproduction was
used to create the permanent red regression for implementation.

TASK-35680 is now implemented in the same PR. Permanent regressions reproduced
both refresh and mount/re-entry origin failures, then passed after native
prevention and stale/no-op handling. Compact/wide real-writer journeys and a
full private-profile keyboard Save/Revert check qualify the correction; see
[TASK-35680 qualification](./2026-10-19-task-35680-qualification.md).

## Candidates requiring evidence before more work

- `_syncing_console_agent_budget`, `_syncing_console_tool_result_display_chars`
  and `_syncing_console_sidechat` have the same temporary-flag shape. Qualify
  each against mounted value-changing refreshes and unrelated drafts before
  claiming a defect or applying a sweep. Mount-generated messages and queued
  user edits need separate handling; prevention is not a retroactive filter.
- The five agent-budget fields form a plausible next local widget cluster
  (chapters 7, 9 and 17), but decomposition alone is not a correctness fix or
  a performance result. Use the approved screen-decomposition design and
  [existing candidate inventory](../../../backlog/tasks/task-32809.3%20-%20Record-the-decomposition-candidates-without-opening-one.md)
  if that work is selected. Keep draft and save ownership in Settings and
  prove editing identity/focus before expanding this pilot.

## Existing owners to continue

- Blocking/message-pump work remains under
  [TASK-32804](../../../backlog/tasks/task-32804%20-%20Work-stream-synchronous-work-on-the-Textual-event-loop.md)
  and [TASK-2834](../../../backlog/tasks/task-2834%20-%20Guard-against-blocking-IO-reachable-from-a-Textual-message-handler.md).
  [TASK-32804.2](../../../backlog/tasks/task-32804.2%20-%20Stop-the-Library-and-RAG-settings-fields-re-loading-the-profile-on-every-keystroke.md)
  is already Done: the historical repeated profile load was cached and
  invalidated at its existing boundaries. Do not refile that old finding;
  any remaining repeated validation needs current measurement.
- Console polling/full reconciliation remains with
  [TASK-34563.33](../../../backlog/tasks/task-34563.33%20-%20Plan-Console-polling-and-full-state-reconciliation.md)
  and [TASK-34563.34](../../../backlog/tasks/task-34563.34%20-%20Verify-Console-polling-reconciliation-boundaries.md)
  under ADR-231. Worker advice from the book does not justify bypassing
  fresh authority, native-read retirement or existing lifecycle owners.
- Native attachment timing and the completed keyboard-responsiveness fix
  are already on [PR #3166](https://github.com/rmusser01/tldw_chatbook/pull/3166)
  (primary-checkout TASK-35614 / TASK-35645); no duplicate task was filed.

Chapter 8's dynamic actions were adapted to the existing custom footer and
guidance-only bindings. Blanket check_action gating would remove existing
explanations, so this review does not propose it. Chapter 9 likewise does not
justify recomposing editable regions on every value change: the new panel's
compact/wide tests demonstrate targeted updates retaining Input selection.
