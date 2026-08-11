---
id: TASK-15452
title: Equality-gate the Console per-keystroke DraftChanged workbench sync
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 12:05'
updated_date: '2026-08-11 20:27'
labels:
  - perf
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verified first-hand: the `ConsoleComposerBar.DraftChanged` handler (`chat_screen.py:18904`) routes through `_sync_console_workbench_actions_from_draft` (`:18143`), which rebuilds and pushes Workbench state with no comparison against `_last_console_workbench_state` — bypassing the guard the main sync path has at `:18039-18041`. Per printable keystroke this produces ~12 layout-invalidating `Static.update()` calls across DestinationHeader/ModeStrip/CommandStrip/RecoveryCallout plus two scheduled `sort_children` calls; Textual's `sort_children` bumps the DOM version up the ancestor chain even when the order is unchanged, invalidating the screen-wide `query_one` LRU cache so the next keystroke's ~15 screen-rooted queries become full DOM walks over the largest tree in the app. The same path also rebuilds provider/readiness state 7-8 times per keystroke.

Fix direction: route the draft path through the same state-equality guard as `:18039`; add `state == self.state -> return` early-outs inside the four Workbench widget sync methods (`UI/Workbench/workbench_widgets.py`); skip `sort_children` when the desired order already matches; build provider selection state once per keystroke and pass it down. Stability constraint: the slash-command popup open/close and first-run-guidance dismissal ride this handler — pin their behavior with tests before landing the gate. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No Workbench widget writes occur when the derived state is unchanged (evidence)
- [x] #2 sort_children is not invoked when child order already matches (evidence), and the query_one cache survives idle keystrokes
- [x] #3 Slash popup open/close and guidance dismissal behavior unchanged (tests pinned before the gate)
- [x] #4 Per-keystroke handler cost measured before/after and recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pin current behavior first: run the existing slash-popup / guidance / Workbench-readiness suites (Tests/UI/test_console_composer_draft_changed.py, Tests/UI/test_console_command_popup.py) green as the pre-change baseline, and add a new instrumented characterisation module that counts Workbench widget writes, sort_children calls and screen DOM-version bumps per keystroke (red before the gate, green after).
2. Extract the guarded push at chat_screen.py:18041-18060 into one _push_console_control_state_if_changed(control_state, workbench_state) helper and route BOTH _sync_console_control_bar and _sync_console_workbench_actions_from_draft through it, so the draft path can never leave _last_console_* ahead of the control bar / status chips. _sync_console_command_popup keeps running on every draft edit, outside the gate.
3. Add sentinel-guarded 'state unchanged -> return' early-outs to DestinationHeader.sync_state, ModeStrip.sync_modes, CommandStrip.sync_actions and RecoveryCallout.sync_state (a dedicated last-synced-state slot, NOT self.state, so the on_mount self-sync still runs).
4. Skip the scheduled sort_children when the children already sit in the desired order - checked both at schedule time and again inside the deferred callback.
5. Memoize the provider selection / readiness app-config for one derivation pass so the draft path builds provider state once instead of per leg.
6. Re-run the pinned suites plus the wider Console/Workbench suites; measure per-keystroke handler cost before/after in an isolated-HOME probe and record it in the task notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Routes the per-keystroke `ConsoleComposerBar.DraftChanged` sync through the same state-equality gate the coalesced control-bar sync already had, and makes the four Workbench widgets no-ops for a state they have already pushed.

## Approach

1. **One gate, one recorder.** Extracted the guarded push at `chat_screen.py:18039-18060` into `_push_console_control_state_if_changed(control_state, workbench_state) -> bool` -- now the only reader/writer of `_last_console_control_state` / `_last_console_workbench_state` -- and routed both `_sync_console_control_bar` and `_sync_console_workbench_actions_from_draft` through it. Deliberately the FULL push, not just the four Workbench widgets: the control bar consumes `workbench_state.actions` too, so a draft path that pushed only the Workbench and then recorded the new `_last_*` would make the next coalesced sync skip a control-bar refresh it still owed. That trap is pinned by `test_a_draft_edit_keeps_the_control_bar_in_step_with_last_state`.
2. **`_sync_console_command_popup` stays outside the gate** -- it filters on the draft text, which moves on every keystroke while the derived Workbench state does not. Pinned by `test_a_gated_keystroke_still_refilters_the_command_popup` ("/" then "p": zero Workbench writes, popup still narrows).
3. **Widget-level early-outs** in `DestinationHeader.sync_state`, `ModeStrip.sync_modes`, `CommandStrip.sync_actions`, `RecoveryCallout.sync_state`. Each compares against a dedicated `_UNSYNCED`-sentinel slot, NOT `self.state`: every one of these self-syncs its constructor state from `on_mount`, and comparing against `self.state` would turn that mount-time sync into a no-op and leave the status/density classes (which `compose` never sets) unapplied. `self.state` is still adopted before the early-out so identity semantics are unchanged.
4. **`sort_children` skipped when the order already matches**, checked at schedule time and again in the deferred callback. Textual's `NodeList._sort` calls `NodeList.updated`, which bumps the update counter on every ancestor up to the screen -- and that counter is part of the `query_one` LRU cache key, so two no-op sorts a keystroke evicted every cached `#id` lookup on the largest tree in the app. Python's sort is stable, so it is a no-op exactly when the child key sequence is non-decreasing; children queued for removal are still present at schedule time but can only ADD an inversion, never mask one.
5. **Per-pass derivation memo** (`_console_derivation_scope`) so the draft path derives provider selection / readiness config / provider-model display once instead of once per leg. Opt-in and scoped: outside the `with` block every lookup is live, and it is torn down in a `finally` so a raising leg cannot cache a stale selection.

## Measured (isolated-HOME pytest harness, ready native Console, 40 invocations)

| per keystroke | before | after |
|---|---|---|
| Workbench `Static.update` | 12 | 0 |
| `CommandStrip._sync_button` | 7 | 0 |
| `sort_children` | 2 | 0 |
| screen `_nodes._updates` bumps | 2 | 0 |
| `_build_console_provider_selection` | 7 | 1 |
| `_provider_readiness_app_config` | 63 | 13 (12 memo hits) |
| handler wall cost | 6.405 ms | 3.170-3.375 ms |

### Attribution of that wall-clock win -- corrected after review

The 6.405 -> ~3.2 ms is **almost entirely the derivation memo (change 5), not the
gate**. Review reproduced the timings and measured the decomposition: reverting the
gate + widget early-outs + sort skip while leaving the memo in place still measures
~3.37 ms -- statistically indistinguishable from HEAD. So ~95% of the measured
milliseconds are change 5.

That is not evidence the gate is worthless -- it is a limit of the probe. The probe
times one synchronous handler call. What changes 1-4 remove is `Static.update` /
`sort_children` / `refresh(layout=True)` calls whose real cost is paid **later**, in
Textual's render and layout pipeline, plus the screen-wide `query_one` LRU eviction
that makes the *next* keystroke's screen-rooted lookups full DOM walks. None of that
lands inside the timed call, so the probe cannot price it and does not claim to. The
honest split:

- **change 5 (memo)** -> the measured ~3.2 ms of handler CPU (the cProfile hot path
  was `build_console_settings_readiness` -> `resolve_console_provider_identity` ->
  ~1.0M `provider_config_key` calls per 60 invocations).
- **changes 1-4 (gate, widget early-outs, sort skip)** -> the render/layout and
  DOM-version churn: the 12 -> 0 / 7 -> 0 / 2 -> 0 / 2 -> 0 counter rows above, which
  are counted directly and are the real deliverable, unpriced in ms by this probe.

## Trade-offs

- The draft path now refreshes the control bar / status chips in the same turn on a state-CHANGING keystroke, where it previously waited for the next coalesced sync. Both widgets own equality guards, so this is free when nothing moved and is work the next tick would have done anyway -- and it is what keeps the recorded `_last_*` honest.
- Residual cost is now dominated by `_console_composer_or_none()`, which does an uncached `self.query("#console-native-composer")` full-DOM CSS walk twice per keystroke (61% of what is left). Out of scope here; worth its own task.

## Files

- `tldw_chatbook/UI/Screens/chat_screen.py` -- gate extraction, draft-path routing, derivation scope + three memoized legs
- `tldw_chatbook/UI/Workbench/workbench_widgets.py` -- `_UNSYNCED` sentinel, four sync early-outs, `_state_children_in_desired_order` sort skip
- `Tests/UI/test_console_draft_sync_equality_gate.py` (new) -- 9 mounted-Console tests incl. two mutation controls
- `Tests/UI/test_workbench_widgets.py` -- 11 new unit tests for the sort skip and the four early-outs

## Tests

`test_console_composer_draft_changed.py` (the slash-popup / guidance / readiness pin suite) 23 passed before AND after, untouched. New: 9 + 11. Wider: 444 Console composer/flow, 323 workbench consumers, 140 Console+workbench, 505 destination shells -- green apart from two failures that reproduce unchanged on dev c0c4753f8 (`test_console_registers_footer_workbench_shortcuts`, stale footer copy from `14cc326e4`; `test_destination_action_buttons_explain_their_outcome[mcp|tools_settings]`, tooltip-less `#mcp-tools-workspace-save` from `8b4b7de8e`).

### Known flake, recorded not omitted

`Tests/UI/test_console_control_bar_coalescing.py::test_requested_sync_still_executes`
flaked once in a batch run at HEAD ("three coalesced requests produced 4 runs
(expected exactly 1)" -- its spy counts `_sync_console_control_bar` runs and picks up
extra settling syncs under load). It passes in isolation and on repeat. Measured
head-to-head afterwards, 10 whole-file runs each: **2/10 at HEAD, 3/10 with the two
source files reverted to base c0c4753f8** -- so the flake is pre-existing on dev, and
this diff adds no new `_sync_console_control_bar` call site. Worth its own
deflaking task; not a gate on this one.

### Lint

`ruff check`: clean on all four changed/added files. `ruff format --check`: **not**
clean on `chat_screen.py` and `Tests/UI/test_workbench_widgets.py`. Every deviation is
pre-existing -- the `ruff format --diff` bodies for both files are byte-identical to
the same files at base c0c4753f8 (10 hunks in `chat_screen.py`, 1 in
`test_workbench_widgets.py`, line numbers merely shifted by this diff's insertions).
Nothing this diff added is unformatted, and the pre-existing hunks were deliberately
left alone rather than reformatted into an unrelated churn diff. The two files this
diff is mostly responsible for -- `workbench_widgets.py` and the new
`test_console_draft_sync_equality_gate.py` -- are format-clean.
<!-- SECTION:NOTES:END -->
