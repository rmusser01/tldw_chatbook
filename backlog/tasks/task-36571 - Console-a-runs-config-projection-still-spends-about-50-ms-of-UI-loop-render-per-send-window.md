---
id: TASK-36571
title: >-
  Console: a run's config projection still spends about 50 ms of UI-loop render
  per send window
status: In Progress
assignee:
  - '@claude'
created_date: '2026-10-24 03:51'
updated_date: '2026-10-31 06:48'
labels:
  - console
  - performance
  - ux-review-2026-09-29
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33620.15.4 moved the Console run projection's checked-config operation and most of its database reads off the UI loop, kept the checked display warm through a run, coalesced a run start's explicit syncs into the poll's projection, and kept the send's durable commit quiet. Its AC#4 had asked for the projection's UI-loop time per send window to halve. Measured live and paired on 2026-10-23 (Anthropic claude-haiku-4-5 through the review proxy, 160x45, 30 sends per arm, dev 1381e70735 as the base), it fell from a median of 58.2 to 49.4 ms per window (-15%; warm sends 55.2 to 48.8, first sends 59.6 to 52.7). Half of dev 5b5920b8e2's 65.1 ms, the figure AC#4 started from, is 32.6 ms.

What remains is mostly the render and derivation CPU of the five to seven projections a send window needs. Mean ms per window on that branch: workspace context state 12.9, rail state 8.8, control-state push 8.7, composer actions 8.0, settings summary 6.3, the Inspector build and other unattributed work 5.5, Library activity 4.0, rail visibility 3.0, cost chip 2.4, transcript guidance 1.7. A run start has to paint each of its state changes once, so syncing less often no longer shrinks this; each projection has to cost less.

About 5 ms of it is reads that TASK-33620.15.4 keeps on the loop during a send's start (acceptance to the provider request): the Library activity rows (3.4 ms) and the Inspector's registry and appearance reads. Off the loop, each of those reads opened its own SQLite connection (`run_owned_db_call`: 34 ms wall, 10 ms CPU alone, 20-60 ms CPU in the app) and delayed first sends by ~66 ms, so they only leave the loop once an off-loop read stops costing a connection.

The layout half of the same work is tracked with TASK-33620.15.7 (PR #3211) and its follow-ups TASK-36012 (widgets that request layouts at send start and end without changing size) and TASK-35970 (the collapsed Character section recomposes twice per send). This task is the projection's own derivation and widget-update cost, measured inside the projection.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Measured first, paired against the base: the projection's UI-loop time per send window, by component, live with a real provider
- [ ] #2 The projection's UI-loop time per send window is at most 32.6 ms median (half of dev 5b5920b8e2's 65.1 ms), live, real provider, paired
- [x] #3 A send reaches the provider no later than on the base build (Enter to the provider request, live, paired and interleaved, at least 30 sends per arm including first sends)
- [x] #4 What the rails and controls show is never staler than one poll interval behind the base build
- [x] #5 A deterministic test fails on the base build and pins each cheaper projection component
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure first (AC#1): P71 kit (pf2probe + sys.monitoring code-object timers, no wrappers, so identity-checked seams are untouched) attributes each send window's projection time (custody -> provider POST) by component on origin/dev, live, Anthropic haiku via the review proxy, 160x45.
2. Rank: on dev 7536ac3b87 the largest single component is Textual restyles from state flips (App.update_styles ~14-16 ms/window, ~22 restyles); the workspace context build (~10-18 ms) is shared with the tray push in the same tick, so trimming it in the projection alone only moves the cost.
3. Cut the restyles each run-start state flip pays: one restyle per flip for the Workbench status classes (scoped to the class-carrying node; a guard proves no rule keys descendants on them), the composer Stop/Send buttons' variant + state classes, the Agent status line, the CommandStrip button classes and the Send-reason strip.
4. Tests first: per-component restyle-count pins that fail on dev, plus guards that the targeted restyles leave the styles a full restyle computes.
5. Live paired rounds against dev at the same load: projection per window (AC#2) and Enter -> provider (AC#3, unprobed, relay-timed, >=30 sends per arm incl. first sends); rebase, gates, preflight, notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
A run start's state flips now restyle each widget once. Restyles were the largest single component of the projection's UI-loop time (about 15 ms of thread CPU per send window, about 22 restyles), and they halve. The projection fell from a median of 59.3 to 47.8 ms per send window (paired median -9.0 ms wall, -7.3 ms CPU; fix lower in 25 of 32 windows by wall, 29 of 32 by CPU). AC#2's bound (32.6 ms) is not met, so AC#2 stays open and the task stays In Progress. What remains is spread over derivations that every run tick pays (TASK-36571.1). The projection also grows with the chat (45 to 63 ms of CPU per window on dev between about 30 and about 100 messages), so an absolute bound needs a chat length.

**Measured first (AC#1), dev 7536ac3b87.** P71 kit (kept outside the repository with the session's evidence and raw per-window records, not committed; whether to commit it is TASK-38441): the pf2 probe plus `sys.monitoring` PY_START/PY_RETURN events on the projection's code objects, so no Console method is replaced (several of them are identity-checked). Live, Anthropic claude-haiku-4-5 through the review proxy, 160x45, isolated profiles, `caffeinate` held. A send window runs from custody (`accept_received_intent`) to the provider POST; the projection is `project_console_tick` plus direct `_sync_console_control_bar` calls.

Base attribution, 12 windows at load 1.4-5.6, inner-component timing (it adds 2-20% to totals, so these rank rather than size: 59.7 ms wall / 49.7 ms CPU median per window), mean thread CPU per window, inclusive:

| component | ms CPU / window |
|---|---|
| Textual restyles (`App.update_styles`, ~21 per window) | 14.6 |
| rail state (incl. the Inspector state 11.4, which incl. the workspace context build 8.8) | 12.0 |
| control and Workbench push (header status restyles inside) | 9.3 |
| composer actions (Send/Stop restyles inside) | 9.2 |
| settings summary (incl. the Agent section payload) | 7.2 |
| `query_one` (~386 per window, spread) | 4.0 |
| cost chip | 2.3 |
| Library activity rows (send-start reads on the loop; 4.1 ms wall) | 2.2 |
| transcript guidance | 2.0 |

A window holds about four non-trivial projections: +250 ms (the run-start flip, ~20 ms, 11 restyles), +400 (~11), +600 (~13-15, Stop appears, 7 restyles) and +850 ms (~6-8; in 5 of 12 windows plus 15-30 ms of wall with little CPU, the workspace build's synchronous send-start registry reads waiting on the GIL). The restyles were mostly avoidable: the Workbench header's status flip restyled its 26-node subtree twice (4.4 ms), Stop paid four restyles for one availability flip, Send three plus `disabled`, and the hidden command strip's buttons, the Agent status line and the Send reason strip one per changed class.

The workspace context build is about as large, but the projection asks for it only for the workspace label, and the run tick's Conversations tray push in the same tick needs the whole build (the tick cache shares one build per tick), so trimming it from the projection alone moves the cost rather than removing it. It is TASK-36571.1, with the settings summary.

**Changes**
- `workbench_widgets._sync_status_classes`: one `update_classes(..., update=False)` and a restyle of the class-carrying node alone. No rule keys a descendant on a status class; a guard scans every rule the mounted app loads (widget DEFAULT_CSS included) and every built sheet, parsed on its own. `_sync_density_classes` batches its two classes (rules key descendants on them, so it keeps the subtree). `CommandStrip._sync_button` batches its three classes.
- `console_composer_bar._restyle_button_once`: Stop's and Send's `variant` set through `set_reactive` with the watcher's two `-<variant>` classes folded into the state-class batch, so each flip costs one restyle (Send keeps Textual's own restyle for `disabled`, whose watcher also hands off focus). A test pins `Button.watch_variant`'s source. The Send reason strip's muted class joins its size-class replacement.
- `agent.apply_console_agent_status_state`: one `update_classes` for the swap.
- Harness, thread CPU per flip on a mounted Console (median of 40): the header's status flip 3.81 -> 0.44 ms (26 -> 1 node restyled); a composer run-state flip 2.78 -> 0.89 ms (5 -> 1 restyle).
- Ratchet: `console_composer_bar.py` 7043 -> 7077 (+34) with a dated owner-decision comment. UI PR gate census: the new file at the tail, floor 1163 -> 1164.

**Live, paired (AC#2, AC#3)**, base dev 7536ac3b87 vs this head, launches interleaved base/fix with alternating order, fresh profiles growing one chat each, each launch gated to start below load1 20 (`kit/p71run.sh`):

Roots-only timing (projection totals; 32 windows per arm in 8 launch pairs, r3 L1/L4/L7/L10/L13/L15 and r4 L2/L4, load1 1.7-11.6):

| per send window | dev | this head | paired (head - dev) |
|---|---|---|---|
| projection, median wall (CPU) | 59.3 (50.0) ms | 47.8 (41.6) ms | median -9.0 ms wall, -7.3 ms CPU; lower in 25/32 (wall), 29/32 (CPU) |
| first sends / warm sends, wall | 62.6 / 57.0 | 52.1 / 47.8 | |
| longest UI-loop callback, median wall (CPU) | 38.7 (30.8) ms | 36.9 (30.7) ms | median -3.6 wall, -1.9 CPU; lower in 20/32, 24/32 |

Inner-component timing, one interleaved pair at launch 6 (4 windows each): restyles 22.0 -> 11.8 per window and 16.1 -> 8.4 ms CPU; the control and Workbench push 10.4 -> 5.8 ms CPU; composer actions 10.2 -> 6.7 ms CPU; the workspace build and the settings summary unchanged. The restyle figure counts `App.update_styles` calls, and the header's targeted restyle calls `stylesheet.update_nodes` directly, so that restyle leaves the counter rather than shrinking inside it (its cost stays inside the control and Workbench push); the figure is not like-for-like, and the roots-only totals above are unaffected. One launch pair is enough to rank the components, too thin to size them.

Enter -> provider request (AC#3), timed outside the app (tmux Enter to a byte relay in front of the review proxy), sends at load1 <= 15:

| | dev | this head | paired (head - dev) |
|---|---|---|---|
| unprobed launches, all (44 / 40 sends) | 1107.9 ms | 1119.2 ms | n=40: median +14.7 ms, faster in 19/40 |
| unprobed, warm (33 / 30) | 1069.1 | 1100.8 | n=30: median -1.3, faster in 16/30 |
| unprobed, first (11 / 10) | 1368.9 | 1422.8 | n=10: median +48.0, faster in 3/10 |
| unprobed and roots-only launches pooled (76 / 72) | 1126.1 | 1138.1 | n=72: median -7.0, faster in 40/72; first sends n=18 median -19.8 (10/18); warm n=54 median -7.0 (30/54) |

No detectable difference; AC#3 is ticked on the pooled pairs. The tick depends on that pooling: the unprobed launches alone lean the other way (n=40, median +14.7 ms, faster in 19/40; first sends n=10, median +48.0 ms, 3/10). The change touches no worker path and its UI-loop work only shrinks, so this reads as noise, but the owner should weigh it. The head's slowest first send (3281 ms) came from a launch where every stage of that send ran slow at once (provider resolution 313 ms against a usual 66, the durable commit 557 ms against about 300), host contention rather than the change: the change only removes UI-loop restyles and touches no worker path. Every provider request in the rounds returned HTTP 200 (84 per arm). Discarded: round r2 (host load 40-110 mid-round) and the head's r3 launch 2 (load rose 23 -> 56 during it).

**Tests (AC#5)** (each new pin fails on dev for the reason it pins; the guards pass on both):
- `Tests/UI/test_console_projection_restyles.py` (10): on dev 6 fail (header restyled 2 times for one status flip; Stop 4; Send 4; the Send reason strip 2 for one muted and size flip; Agent status 2; a command strip button 3), 4 guards pass (the header's and the buttons' styles equal a full restyle; no rule keys a descendant on a status class or sets a property the node's children read from it, such as colour or background; the variant watcher pin). On this head 10 pass under the plain command. Mutations on the head: reverting the reason strip's muted fold to a separate `set_class` fails the strip pin (2 restyles); appending `.workbench-frame .status-error { color: $error; }` to the built bundle fails the status guard with that selector.
- `Tests/UI/test_workbench_widgets.py`: its two `SimpleNamespace` doubles gain `update_classes`/`has_class`/`app`, and the header first-sync case now asserts the batched status and density mappings and the one-node restyle.
- Neighbours (each file one pytest command, run on this head and on a dev 7536ac3b87 archive with a throwaway plugin adding `bootstrap_profile`, compared by test id; the plain command errors at setup on most of these files locally on both trees, `RecoveryRequired`): `test_workbench_state`, `test_console_composer_state_restyle`, `test_console_composer_reason_restyles`, `test_console_composer_run_controls`, `test_console_send_transition_layout_requests`, `test_console_video_send_freeze`, `test_console_library_stop_received_send`, `test_mcp_workbench`, `test_roleplay_header`, `test_lab_frame` pass on both. Same reds on both: `test_destination_headers` (1), `test_console_rail_color_grammar` (2: the `agent_fleet_section_state` kwarg drift and a source substring), `test_console_stop_feedback` (1), `test_lab_mode_strip` (1), `test_console_draft_sync_equality_gate` (1), `test_console_workbench_contract` (10 here, 11 on dev), `test_schedules_workbench` (4), `test_console_composer_collapse` (7), `test_console_send_disabled_state` (2). Reds seen only in the head's first pass (run at load 100+) passed when rerun at load 4: the compact-floor first paint (3 of 3) and five Schedules cases. Flaky on both trees, by repeated runs: three `test_console_composer_collapse` cases that push run state through the composer's presentation edge (`test_console_collapsed_status_uses_presence_only[...True...]`, `test_collapsed_stop_routes_native_run_control_without_expanding`) fail in most whole-file runs (head 4 of 5, dev 4 of 5) and pass alone (3 of 3 on each), because the screen's own syncs overwrite the pushed state (lessons-testing-evidence, TASK-33626.1); `test_console_send_disabled_state::test_enter_hotkey_still_sends_when_send_is_enabled` (head 1 of 3 failed, dev 2 of 4, same message); and one or two ids of `test_console_workbench_contract` and `test_schedules_workbench` that change between runs on both. `test_master_shell_design_system_contract` fails 3 only in the dev archive (it has no `Docs/`).
- Architecture: the module-size ratchet's 8 reds, the timer-path Static.update inventory's 2 and the worker exclusive-group inventory's 2 are identical on dev; the composer row passes. Preflight passes. No module or CSS was added (UI-ready census and boot CSS budget untouched).

**Freshness (AC#4).** No state, cadence or deferral changes: every flip still restyles synchronously inside the same call, and the equivalence guards show the targeted restyles compute the styles a full restyle does, so the rails and controls show the same thing at the same moment as on dev.
**Review fixes (2026-10-31).** The independent review found the Send reason strip's muted fold unpinned (reverting it passed every test); the strip pin above closes that. Its follow-ups applied here: the status guard also rejects status rules that set a property the children read from the node (Textual caches a child's rendered style keyed on the child's own pseudo-classes, so a node-only restyle would leave them stale), with the matching caveat in `lessons-textual.md` and the `_sync_status_classes` docstring; Stop's restyle count is four in the composer comment and the test docstrings, as measured; the notes above qualify the inner restyle counter, the one-pair component sizing and the AC#3 pooling. Filed: TASK-38440 (the guard does not see DEFAULT_CSS of widget classes the harness never mounts) and TASK-38441 (owner call on committing the probe kit). Re-gated on the new head against a detached dev 7536ac3b87 checkout, each file one pytest command, compared by test id (the plain command, plus every test marked `bootstrap_profile` for the files that error at setup locally): no red new to the branch. Two cases were red in only one tree's whole-file run and are order-dependent on both: `test_console_send_disabled_state::test_active_run_block_shows_wait_reason` (alone, head 3 of 3 pass, dev 3 of 3 fail) and `test_console_rail_color_grammar::test_rail_grammar_resolves_to_the_active_theme_colors` (alone, head 3 of 13 fail, dev 2 of 13, same message).

**Open: AC#2.** 47.8 ms median against the 32.6 ms bound. The owner can restate the bound at a stated chat length (dev itself drifts from 45.5 to 63.3 ms CPU per window between about 30 and about 100 messages, `lessons-live-verification.md`), or keep it and wait for TASK-36571.1.
<!-- SECTION:NOTES:END -->
