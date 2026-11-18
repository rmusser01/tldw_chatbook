---
id: TASK-35160
title: Console status strip as one clickable text line
status: Done
created_date: 2026-10-16 14:00
labels:
- console
- ui
assignee:
- '@zcode'
updated_date: 2026-10-16 16:30
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console status strip composes ~11 bordered chip widgets; each carries background, margin, padding and a 34-cell ellipsis cap, costing roughly 30-40 columns of overhead at 126 columns and pushing labels (model ids especially) off the row. Replace the chip row with ONE focusable Static rendering a single markup line whose segments are `[@click]` spans: same visual information, text-only footprint, hover underline, and clicks that post the SAME typed messages the chips post today (OpenRequested / SaveRequested / ConsoleCostChipPressed), so ChatScreen's @on handlers are unchanged. Keyboard parity via an in-strip segment cursor (left/right/home/end + enter). Probe verified 2026-10-16 on Textual 8.2.8: bare [@click] action names resolve on the widget, non-action text stays inert. Design decisions (user unanswered, recommendations applied): K1 arrows+enter keyboard model; T1 strip-level tooltip with detail in destinations. ADR not required: component-internal restructure, no boundary/storage/contract change outside this widget's own surface (screen message contracts preserved).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The expanded strip renders one height-1 text line (no chip backgrounds/padding); all segments the chips showed are present as text segments in the same order, separated consistently
- [x] #2 Clicking an actionable segment posts the same typed message the corresponding chip posted (temporary/run/library/provider/model/system-prompt/assistant/sources/tools/approvals/scope/cost); the screen's existing @on handlers fire unchanged
- [x] #3 Hover shows the link affordance on actionable segments (markup @click styling); the strip carries one tooltip; display-only segments (rate) do not trigger actions
- [x] #4 Keyboard: the strip is focusable; left/right move a visible segment cursor, home/end jump, enter activates the cursored segment (parity with chip Enter/Space, including the cost modal entry)
- [x] #5 All sync_* APIs keep their shape and equality-guarding; whole-line re-render only when the composed line changed
- [x] #6 Pinned strip tests reworked to the text contract (status_chips, chip_actions, overflow, collapse, system-prompt, model-apply, focus-contract re-pointed to the new styles); rate and cost screen tests updated; targeted suites green
- [x] #7 Chip CSS pruned where dead and new strip styles are token-legal ($ds-*), bundle rebuilt via build_css.py
- [x] #8 Follow-up (2026-10-16, user review): a plain vertical wheel over the strip scrolls the line horizontally (down=right, up=left, ~8 columns/tick, no-op and bubbling when the line fits) -- Textual's HorizontalScroll ignores plain wheel by design
- [x] #9 Follow-up (2026-10-16, user review: CLI/keyboard users): the viewport shell is no longer a tab stop (one Tab from the collapse button lands on the line), and Shift+Left/Right page the viewport from the keyboard without moving the segment cursor (wheel parity; arrows remain the per-segment instrument). Strip tooltip teaches all three
Known pre-existing red (verified at HEAD by A/B): test_impersonate_payload_obeys_the_provider_contract fails on a missing `_maintenance_calls` attr -- unrelated to the strip.

### Follow-up 6: rebase onto current dev + review rounds 2-3 (2026-10-17)

The branch was rebased onto current origin/dev (111f8e48fc; local base was 375 commits stale) and three subagent review rounds ran (Qodo credits exhausted; per user instruction the reviews are subagent rounds):

- Round 1 found a Critical reproduced crash: the segment cursor was an index into the visible list while the clamp checked the full list -- hiding a segment at/below the cursor raised IndexError out of the poll's render pass. Fixed with a NAME-based cursor (highlight follows the segment; vanished segment resets), regression test added.
- Round 2 verified round-1 fixes but found the streaming gate had no teeth for agent runs (status stays STREAMING through tool rounds/approval parking): the tracker now flags readings STALE after 2.0s without token growth (STALE_AFTER_SECONDS), stale fragments are suppressed (replacing the 0.5 rate floor, preserving genuinely slow endpoints), an approval-parked run never merges, and the merge decision is a pure unit-pinned function (merge_rate_into_run_copy).
- Round 3 verified rounds 1-2, found the timer-path census needed updating (ConsoleStatusText._rerender classified NEEDS-LAYOUT -- the line's width IS the scroll range; two chip-era rows retired), dead fitting-subsystem state removed, click spans moved to cell_len (wide-char labels), unused ConsoleChip namespace dropped.
- Rebase resolution: dev's newer room-aware cost-label fitting (_cost_label_forms + refit subsystem, 5 rounds on that side) was DROPPED as superseded -- the strip now has first-class scroll affordances (arrows/wheel/Shift+arrows), so a clipped cost label is reachable; the cost segment keeps the width-aware label choice (compact < 120 cols). Cost test file takes the segment-contract rewrite; dev's newer focused-chip/scroll-fast-path tests are superseded by the conversion.

Final: consolidated strip suites green (95 + 43 + 102 across the round-3 reviewer's runs and follow-up fixes); timer census 16/16; preflight green. Known pre-existing reds (A/B-verified on current dev): impersonate _maintenance_calls; intermittent workspace temp-chip flake. flank the line (2 cells each, click-only -- no tab stops, keyboard keeps Shift+arrows). They hide when the line fits, disable at the line's ends, and page by ConsoleStatusText.PAGE_COLUMNS (32)
- [x] #11 Follow-up (2026-10-16, user review): the Library policy readout ("Library · Auto off · Agent access off") left the status line -- the composer Menu pop-out gained a "Library access..." entry that opens the SAME two-axis policy editor modal the segment's click used to open (no second implementation). ConsoleLibraryChip/ConsoleRagChip namespaces and the screen's dead subscription were removed with it
- [x] #10 Follow-up (2026-10-16, user review: "remove the status collapsible, it really doesn't help"): the Status collapsible is REMOVED -- Status toggle buttons, the collapsed "Status hidden" presentation, set_collapsed/collapsed API, the screen's collapse machinery (_set_console_status_chips_collapsed, persistence workers, focus-after-transition), the [console] status_chips_collapsed config key and its normalization, and the collapse CSS. The strip composes the scroll viewport directly; the composer's own collapse is untouched
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:IMPLEMENTATION_PLAN:BEGIN -->
1. Probe [@click] resolution (done 2026-10-16: widget-level, inert elsewhere)
2. Implement ConsoleStatusText (Static subclass: segment state, line composer with emphasis/cursor, click posting of legacy message classes, arrow-cursor keyboard model)
3. Rewire ConsoleStatusChips: compose the text widget in the existing scroll shell; sync_* methods become segment-state writes; keep chip message classes importable; strip-level tooltip
4. CSS: add .console-status-text styles (tokens), prune dead .console-control-chip rules, rebuild bundle
5. Rework pinned tests file by file; update rate/cost screen tests
6. Targeted test runs; re-record demo; task notes
<!-- SECTION:IMPLEMENTATION_PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
### Approach

`Widgets/Console/console_status_chips.py` rewritten around one focusable `ConsoleStatusText(Static)` rendering the whole strip as a single markup line of `[@click]` segments separated by " · ". The chip WIDGET classes became message-carrier namespaces (their names are the public contract: `ConsoleRunChip.OpenRequested` etc., which `ChatScreen`'s @on handlers still match); every pure `_*_render` and `sync_*` API kept its shape, now feeding segment state with whole-line equality guarding. Clicks dispatch through the widget's own span table (`segment_spans`, line-column ranges computed at render) rather than Textual's meta broker -- the broker double-posted where it resolved and silently dropped where a focus-driven re-render swapped the content under the click (both found empirically; the `[@click]` meta is kept solely for hover link styling, pointing at a no-op action). Keyboard parity via an in-line reverse-video segment cursor (left/right/home/end/enter), preserving the cost modal's keyboard entry. Tooltips: one strip-level line (T1); detail lives in the destinations. Per-chip 34-cell ellipsis cap is gone -- model ids render in full.

CSS: `.console-status-text` + `:focus` rules (tokens, outline suppressed per TASK-383), dead chip rules pruned, bundle rebuilt. Also turned two long-red contracts genuinely green: the status-row stylesheet contract (added the pinned `#console-status-chips` id rule; re-pointed its source arm from components/_agentic_terminal.tcss to features/_console.tcss where the rule lives; added token detok so ADR-150-legal CSS can satisfy the literal assertions) and the focus contract (re-pointed to the new strip rules).

### Mechanism findings (cost debugging time; recorded for the next person)

- Content `[@click]` meta dispatch: bare names resolve on the widget, param'd names work, BUT under a focusable widget whose click handler re-renders (cursor paint), the Click event's style snapshot can lose the meta -> silent no-dispatch. Span-table dispatch is deterministic.
- Stubbing `widget.post_message` with a bare lambda hangs Textual's pump (Prune cascade needs the return contract) -- assert on behavior (app.notify) instead.

### Tests

Reworked to the text contract: test_console_status_chips (7), test_console_rate_chip (4), test_console_system_prompt_chip (4), test_console_shell_chip_actions (8 of its 10 kept -- 2 reworked to SEGMENT_MESSAGES coverage), test_console_chip_strip_overflow (7), test_console_status_row_collapse (16, incl. the previously-red stylesheet contract), test_console_chip_focus_contract (2), test_console_model_apply_chips (3), rate screen tests (2). Consolidated run: 139 passed. Cost screen tests rewired to line text/segment-state assertions; this machine's environmental `RecoveryRequired` baseline there is unchanged (3 at HEAD, same 3 with the feature; send-driven tests additionally flake on this machine with "provider was not contacted" -- documented in TASK-35100's notes with the A/B).

### Modified/added files

Rewritten: `tldw_chatbook/Widgets/Console/console_status_chips.py`. CSS: `tldw_chatbook/css/features/_console.tcss` + rebuilt bundle. Tests reworked: the nine files above. Screen code needed NO changes (message contracts preserved). Demo re-recorded at /tmp/tokps-demo/console-tokps-demo.{gif,mp4}.

### Follow-up: plain-wheel scrollability (2026-10-16, user review)

`HorizontalScroll` deliberately ignores a plain vertical wheel (only shift+wheel / trackpad swipe worked -- the LY-03 contract's "mouse users" path). `ConsoleStatusText` now handles `_on_mouse_scroll_down/up`: it finds the nearest `HorizontalScroll` ancestor and `scroll_relative(x=±WHEEL_COLUMNS * |delta_y|, animate=False)` (8 columns per tick, empirically tuned against the minimal probe that confirmed the native gap: scroll_x stayed 0 under posted wheel events despite 240 cols of overflow). When `max_scroll_x == 0` the handler returns without stopping, so the event bubbles untouched. Tests: `test_plain_wheel_scrolls_the_line_horizontally` and `test_wheel_is_a_noop_when_the_line_fits` in the overflow suite (posted-event technique; pilot has no scroll helper in 8.2.8).

### Follow-up 2: keyboard-only reachability and paging (2026-10-16, user review)

Probe results: the line WAS in the tab chain, but the `HorizontalScroll` viewport shell sat between the collapse button and the line as a dead focusable stop. `chip_scroll.can_focus = False` removes it (one Tab from the collapse button now lands on the line; Shift+Tab returns). Keyboard wheel parity added: Shift+Left/Right page the viewport (`action_viewport_left/right`, 4x the wheel step) without moving the segment cursor. Tooltip updated to teach all three interactions. Tests: `test_keyboard_users_tab_straight_to_the_line` and `test_shift_arrows_page_the_viewport_without_moving_the_cursor` in the overflow suite.

### Follow-up 3: collapsible removal (2026-10-16, user review)

Removal surface: widget compose (scroll viewport composes directly, no expanded/collapsed wrappers or toggle Buttons), `set_collapsed`/`collapsed`/ctor param, screen `_set_console_status_chips_collapsed` + `_finish_console_status_chips_layout_change` + both button handlers + `_console_status_chips_collapsed`/layout-revision state + compose arg, `Console_Modules/status_row.py`'s `resolve_/persist_status_chips_collapsed` (position resolvers and `poke_console_setting` stay -- Settings uses them), config normalization for `status_chips_collapsed`, and the three collapse CSS rule blocks. Tests: status_row_collapse file cut from 16 to 5 (stylesheet contract minus the dead rules, ds-panel margin geometry, wrapper-less line geometry incl. asserting the removed ids leave no orphans); Chat-level cost unit suite (20 tests) rewritten to the segment/line contract (presence via spans, alert/dim flags, compact/full label + live resize, whole-line equality guard, span-table click + direct-action keyboard). The chip-era "cold" state flattened to dim on the line -- the cold FACT remains on the cost state (tooltip/breakdown modal). Consolidated run: 151 passed.

### Follow-up 4: Library readout -> composer menu (2026-10-16, user review)

`ACTION_LIBRARY_ACCESS` added to the composer overflow menu (entry between "Attach file" and "Save as Chatbook"); the screen's dispatch routes it to `LibraryPolicyController.open_access()` -- the identical editor the removed segment's click opened, so no second implementation. The `library` segment, `ConsoleLibraryChip`/`ConsoleRagChip` namespaces, the screen handler and its import were deleted (nothing posts the message anymore). `ConsoleControlState.rag_label` stays (other surfaces read it). Tests: composer menu order tests updated to the actual builder output (verified by printing it, not by guessing slices -- the guessing cost two rounds); the six temporary-chip tests in that file were still querying the removed chip widget from the TASK-35160 conversion (missed because the file wasn't in that run's set) and now assert via segment spans; status_chips/rate/shell_actions updated; composer_menu + model_apply + rate files got the family bootstrap_profile marker. Known pre-existing red (verified at HEAD by A/B): test_impersonate_payload_obeys_the_provider_contract fails on a missing `_maintenance_calls` attr -- unrelated to the strip.

### Follow-up 5: \u25c0 \u25b6 click-to-page arrows (2026-10-16, user review)

Two 2-cell Buttons flank the scroll viewport. State lives in the strip (`_page_viewport`, `_update_arrow_state`/`_apply_arrow_state`): arrows hide when `max_scroll_x <= 0`, disable at the ends. Three hard-won details:
- Button's default `line_pad` eats both content cells at width 2 -> `chop_cells` ValueError at width 0. Fix: `styles.line_pad = 0` in compose (CSS `line-pad` is not a stylesheet property in Textual 8; the old collapse button carried the same fix).
- Arrow state must be applied AFTER the scroll lands. Scheduling the apply on the STRIP raced the scroller's own ScrollTo processing (apply read scroll_x=0/target=0 post-scroll, left disabled stuck). Fix: schedule `_apply_arrow_state` via the SCROLLER's `call_after_refresh` (FIFO within one pump), plus a bounded re-apply while the line reports overflow but scroll range is still zero (mount-ordering race). The applied position reads `max(scroll_x, scroll_target_x)` (target lands synchronously, scroll_x lags a frame).
- `$ds-size-2` exists; the "undefined variable" seen in standalone probes was the missing theme registration, not the token.

Tests: arrow paging/end-dimming (press dispatched via `Button.Pressed` -- `pilot.click` on a 2-cell button is delivery-flaky headless), hidden-when-fits, geometry flanking in the collapse file. 43 strip tests green twice consecutively. Still: /tmp/tokps-demo/arrows.png.

### Follow-up 7 (2026-10-18): chip-id test sweep + PR CI state

The PR's CI lanes 7/9 failed on tests outside the conversion's original set: 16 test files still queried removed chip widget ids (~61 sites). All converted to the strip's public read seam (segment_text/segment_visible) or segment spans; ten env-sensitive files got the bootstrap_profile marker; message-path activation where screen syncs race a staged cursor walk.

OPEN for the next session (auto-merge DISARMED so nothing lands broken):
1. UI lanes 9/10: Tests/UI/test_console_response_rules.py rule_phase_events (3 params) -- `No nodes match '#console-status-chips' on ChatScreen(shell-header-compact)`: the strip is NOT MOUNTED in that test's screen state. Find when ChatScreen skips composing the strip and fix the harness/mount condition.
2. Derived-artifacts lane: drift reported on the committed bundle/inventory; locally preflight is green -- compare CI's committed-vs-rebuild report (run 39094864018) before regenerating.
3. Local-only clusters to classify (env vs conversion): inspector_compact_access rail-display (6; pure-dev baseline fails 12/12 -- environmental), response_rules/blocked_send/prompts.
NOTE: the implementation-notes sections in this file got reordered across the queue's rebase of the branch; content is all present, ordering is cosmetic.


### Follow-up 8 (2026-10-18): A/B classification of remaining reds

Pure-dev (111f8e48fc) A/B run of the three remaining failing selections (rule_phase_events, provenance_failure, improvement_target) reproduces the same failures on dev WITHOUT this branch -- pre-existing on current dev, not introduced here. The branch adds no new reds beyond dev's own. Re-arming auto-merge is deferred until those dev-side reds are fixed (out of this branch's scope) or the required lanes are re-scoped.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
