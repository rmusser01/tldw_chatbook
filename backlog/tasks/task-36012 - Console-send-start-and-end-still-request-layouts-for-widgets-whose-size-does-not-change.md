---
id: TASK-36012
title: >-
  Console: send start and end still request layouts for widgets whose size does
  not change
status: Done
assignee:
  - '@claude'
created_date: '2026-10-21 07:45'
updated_date: '2026-10-26 15:47'
labels:
  - console
  - performance
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33620.15.7 stopped a Console run's 0.2 s poll ticks from requesting whole-screen layouts for unchanged widgets. In a headless count at 160x45 (one held send through the real Console, 2026-10-21), layout requests over six unchanged poll ticks fell from 128 on dev 906497d9b4 to 8, all eight from the Character section (TASK-35970). Its AC#2 ("no full layout when the widget's size is unchanged") is not yet shown for the send's own transitions, though. From Enter to the first streamed chunk, requests fell from 417 to 292, and from release to the end of the reply from 359 to 214. Most of the remaining requests leave the requesting widget's display, outer size, region size and virtual size unchanged at the next frame.

The largest same-size requesters at send start were the status line (6), the Inspector "More" control (6), the workspace context tray (6), the workspace details tray (4), the workspace status pair (4), the Inspector rail handle (4), the run inspector (4), the Send disabled-reason strip (3), and each of the Inspector rail's bounded sections and their viewports (3 each). At reply end, after the Character section (TASK-35970), they were the workspace context tray (6), the rail handle (4), the workspace details tray (4), the status pair (4) and the status line (3-4).

Each one needs checking: either its size can change in that transition (and the request is right), or it should repaint without a layout.

The same remainder keeps TASK-33620.15.7's AC#3 ("full reflows per send window drop by at least half") from holding reliably. Live and paired at 160x45 (36-turn chat, Anthropic claude-haiku-4-5, 16 sends per arm), the branch measured 13 reflows per send window in both rounds, while dev measured 27.5 in the first round and 20.5 in the round on that task's final head: -53%, then -37%, and -46% pooled.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each requester listed above is classified, with evidence, as one whose size can change in that transition or one that does not need a layout
- [x] #2 The ones that do not need a layout no longer request one during a send
- [x] #3 A deterministic test counts layout requests across a scripted send's start and end and fails on the base build
- [x] #4 Full reflows per send window do not rise (live, real provider, paired)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure first: headless probe of a held warm send and a cold first send at 160x45 (layout/recompose requests per requester, geometry before/after, stacks, full reflows per transition), dev vs branch.
2. Classify every listed requester (AC#1) with that evidence.
3. Test-first: a scripted warm send through the real Console counting requests across start and end (fails on dev), plus geometry A/Bs for every in-place repaint.
4. Fix the requesters whose size cannot change: Conversations tray in-place patch of geometry-neutral compose deltas; Details tray read-set guard; badge-only rail-state change syncs handles only; collapsed Inspector skips its outer-fold layout; Settings summary and send-authority rows repaint fixed boxes without layout; header re-writes only changed copy.
5. Live paired rounds (real Anthropic haiku via the review proxy, 160x45), base vs fix, full reflows per send window; visual check at 80x24, 120x40, 235x52.
6. Rebase onto newest origin/dev, re-run gates, preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
A send's start and end now repaint same-size widgets in place instead of asking Textual for whole-screen layouts.

**Measured first.** Headless probe (wrapping `Widget.refresh` and `Screen._refresh_layout`) of a held send through the real Console at 160x45 with a file-backed DB, Context rail open, Inspector collapsed, on dev 0d7e418db1. For each requester it recorded the stack, the state delta that triggered it, and display/outer/region/virtual size at the request and after the next reflow. A warm send (the second in a chat): 186-218 layout/recompose requests over 13-15 full reflows from Send to the first chunk, 115-123 over 4-5 from release to the run's end. A chat's first send: 286 over 18, then 174 over 5.

**AC#1 classification** (warm send unless noted):
- Conversations tray (6-10 per transition): recomposed every row because the active row gained/lost a "Running" attention fact. An off-DOM compose diff showed the only deltas were that row's action icon (`💬` <-> `⟳`, a fixed 4-cell slot), its `conversation-attention-running` colour class and two tooltips. The recompose also re-mounted the legacy alias button and triggered two bounded-section reconciles: three extra full reflows at the start, two at the end. No layout needed: fixed.
- Details tray (4-5), its status pairs (4) and handoff rows (2): recomposed on the same `conversation_browser` delta, a field it never renders (whole-state equality guard). Fixed.
- Collapsed Inspector rail itself (`#console-right-rail`): the run badge changes only `ConsoleRailState.right_badge`, yet `_sync_console_rail_visibility` re-ran the whole rail cascade and `refresh(layout=True)` on both rails; and its outer-fold reconcile laid out a `display: none` rail whose measure reads an empty viewport. Fixed (handles-only sync; skip that layout while hidden, since opening the rail resizes it and re-requests the reconcile).
- Inspector bounded sections, viewports and the eleven Settings rows: on a warm send the cascade above re-hid the Settings rows the Settings summary had just re-shown, and the summary re-wrote every row and re-sized its button (`set_styles(height=None)` lays out even with no rule). Fixed: no cascade on a badge-only change, unchanged rows are not re-written, changed rows repaint without layout (fixed one-row box, geometry A/B), the button re-sizes only when its label changes.
- Send-authority "Run" fact: its copy changes, but the row is a fixed one-row, full-width, nowrap box: repaints without layout (geometry A/B).
- Workbench header title: re-written with unchanged copy each transition: gated (subtitle and status chip change width and still lay out).
- Inspector rail handle (4): `DestinationRailHandle.sync_state` recomposes to add/remove the "running" badge Static. The handle's box is fixed, but a child appears/disappears and needs a layout to be placed: needs a layout (left as is).
- Status line (4-6): "Run: idle" <-> "Run: streaming"/cost copy changes its cell width and the chip strip's virtual width (202 -> 192 -> 180 cells at 160x45): size changes (same-width changes already repaint in place since TASK-33620.15.7). Its remaining same-size requests are Textual's own `virtual_size` echoes inside `_size_updated`.
- Send disabled-reason strip (3-7): shown ("Type to queue") when a run starts and hidden at its end: size changes.
- Run inspector (4-5), its "More" control (5-6) and every bounded section/viewport (3 each): a chat's first send only. Persisting the conversation turns the dictionary/world-book rows into "No ... in play" and enables their Attach actions, a structural change the inspector recomposes for inside the collapsed rail; every re-mounted child and the cascade that re-hides it asks for a layout. On warm sends it updates rows in place and its auto-height live-work row legitimately lays out. Not fixed here (deferring a collapsed rail's work is a design change): TASK-37000.
- Character section: TASK-35970, not touched.

AC#2 scope: every requester above that does not need a layout no longer requests one on a warm send. The one exception kept out is the first-send rebuild of the collapsed Inspector's run section: those widgets are new mounts whose real sizes are unknown until the rail opens, so removing their requests means deferring a collapsed rail's work, a design change filed as TASK-37000.

**Changes.**
- `Widgets/Console/compose_delta.py` (new): `plan_in_place_delta` diffs the off-DOM composes for the shown and next state and returns writes (tooltip, classes, public plain attributes, a Button label or Static content only in a box not sized by its content) or `None` to rebuild: different shape/type/id/inline styles/private state, a type outside `PATCHABLE_TYPES` (plus caller-listed ones), or a mounted tree that no longer shows the old state.
- `ConsoleWorkspaceContextTray._patch_in_place`: used before a recompose, only on a tray whose DOM is proved to show its state (same proof as `_can_skip_recompose`) and only when every changed field is one its compose reads (the screen's out-of-band alias reads `new_conversation_enabled`).
- `ConsoleWorkspaceDetailsTray`: records the state fields compose reads (`_ComposeReadView`, as TASK-26836) and rebuilds only when one changes.
- `ChatScreen._sync_console_rail_visibility(handles_only=...)` + `console_rail_state.console_rail_handle_copy[_only_changed]`; ChatScreen's size row lowered 24323 -> 24319 (exact measurement).
- `ConsoleInspectorRail._run_scheduled_outer_reconcile`: no layout while `display` is off.
- `ConsoleSettingsSummary.sync_state`, `ConsoleSendAuthoritySummary.sync_state`, `DestinationHeader.sync_state`: equality gates and fixed-box `layout=False` repaints.

**Tests** (`Tests/UI/test_console_send_transition_layout_requests.py`, added to the UI PR gate census, floor 905 -> 906): a scripted warm send through the real Console counting requests across start and end (tray, Details tray, collapsed rail, and unchanged widgets inside the collapsed rail must request none); a geometry/paint/classes/tooltip A/B of the tray patch against a recompose; positive controls (an approval that adds a row line still recomposes; an unrendered field change still recomposes; a rendered Details field still rebuilds); geometry A/Bs (`assert_geometry_is_content_independent`) for the settings and authority rows; the header gate; and `plan_in_place_delta`'s refusal rules (content-sized box, drift of label/class/tooltip, new shape/style, unlisted type, private-state change). All except the controls fail on dev 0d7e418db1; each fix was mutation-checked against them (13 mutations, each caught).

**Headless after** (same probe): warm send 85-96 requests over 9-13 reflows at the start (load-dependent), 29-33 over 2 at the end; first send 198 over 14-15, then 30 over 2.

**AC#4, live and paired** (2026-10-26; real app through `launch.sh`, isolated profile, null keyring, Anthropic claude-haiku-4-5 through the review proxy, 160x45; a sitecustomize probe counted full `Screen._refresh_layout` passes from Enter to the run poll stopping + 0.75 s; base = detached origin/dev 0d7e418db1). Two interleaved sets, 8 sends per launch, send 0 of each launch (a new chat) reported apart. Set 1 (B F F B B F, host load 15-32 while the suites ran): warm sends median 18 -> 15 full reflows per send window, layout requests 261 -> 113, layout time 371 -> 295 ms. Set 2 (F B B F, load 4-11): 16 -> 13, requests 251 -> 110, layout time 170 -> 141 ms. Pooled, 35 warm sends per arm: median 17 -> 14, mean 17.1 -> 14.3 (-16%); first sends 23 -> 18 (5 per arm). No fix launch's median exceeded the base launch beside it (one tie, 17 vs 17, at load 30). The remaining per-window requesters on the fix arm are the transcript's own mounts, the status line and reason strip (size changes), the rail handle's badge (child added/removed) and the Character section (TASK-35970). Evidence: `scratchpad/uxrev/evidence/bd5/LQ/` (`r1/`, `r2/`, `r1-summary.txt`, `r2-summary.txt`, `drive.py`, `probe/sitecustomize.py`, `lqan.py`).

**Visual check** (fix arm, live, a first and a warm send at each size; `LQ/visual/`): 80x24, 120x40 and 235x52 mid-run and after; Queue/Send, Dictate, Redirect and Stop paint whole at every size, the "Type to queue" reason strip shows mid-run, and the Conversations row shows the running icon during the run and the idle icon after (patched in place).

**Suite comparison**: 134 related files (every test file that names a touched module) on the branch and on a detached dev worktree, compared by test id. Branch-only reds: TASK-33620.15.7's Inspector-fact pin (expected: the fact now repaints without layout; updated); the rest re-ran green serially on the branch or fail identically on dev when re-run serially (`test_settings_active_compaction_close_anyway...`, `test_console_bootstrap_workspace_activation_and_details_use_target_default`, `test_conversation_row_renders_custom_icon_in_right_action_slot` flakes 2/6 on both). Governance: Python style inventory (ADR-161), component/design-token governance, screen and module size ratchets, UI-ready module census and timer-path inventory match dev (the last is red on dev for `ProjectIndexBindingCard._poll`, TASK-36191, identically).

**Review fix (2026-10-26).** The independent review found that every tray sync the patch refused (a conversation switch, a search, a rename, a first send, an approval that adds a row line) ran two throwaway off-DOM composes inside `sync_state` before the rebuild composed a third time. Verified first: a new pin counted 3 composes for each refused delta on the pre-fix head. Fix: `_browser_delta_may_patch` checks the state before composing. The two composes run only when `conversation_browser` is the sole changed field, the browsers differ only in run markers, attention facts, icons or colours, and every row's activity line is unchanged. The compose diff is still the authority on whether an admitted delta moves nothing.
- CPU per tray sync (headless, app stylesheets, `process_time` around sync + settle, median of 9; dev 0d7e418db1 / pre-fix / fix): conversation switch at 12 rows 81 / 142 / 82 ms and at 40 rows 203 / 382 / 203 ms (192 ms of the pre-fix figure blocked inside `sync_state`); search at 40 rows 203 / 381 / 203 ms; running marker (patched) at 40 rows 205 / 211 / 211 ms. A patched sync costs the same CPU as dev's rebuild, but the two composes now run inside `sync_state`. TASK-37030 is filed to drop the compose of the state already shown.
- Pins: `test_delta_the_patch_refuses_composes_once[conversation switch|search results|rename|approval]` assert one compose per refused sync. They pass on dev by design and failed on the pre-fix head (3 composes each). Three gate mutations were each caught: an empty patchable-field set (by the warm-send test and the tray A/B), no activity-line check (by the approval pin), and allowing other changed fields (by the unrendered-field control).

**Review follow-ups applied.**
- `plan_in_place_delta` rebuilds when a public attribute that is not a plain value differs. `test_delta_rebuilds_a_changed_public_object` failed before the change.
- The compose read view moved into `compose_delta.py` as `ComposeReadView`, shared by both trays.
- The dead first `MINIMUM_FILES` assignment was dropped from the census script, and its comment now gives the file's test count (20, about 22 s).
- The warm-send test's docstring says which assertions its first failure hides on the base build, and where each of those fixes is shown red.

**Live re-run with CPU columns** (real app via the launcher, isolated profile, null keyring, Anthropic claude-haiku-4-5 through the review proxy, 160x45, 8 sends per launch, launches B F F B, base = origin/dev 0d7e418db1, host load 6-11 with a test suite running). The probe now also samples process CPU and UI-thread CPU, and counts tray composes, per send window. Warm sends (14 per arm), median base -> fix:
- full reflows 16 -> 13 (mean 16.7 -> 13.4);
- layout requests 257 -> 111;
- layout time 171 -> 139 ms;
- process CPU 1302 -> 1205 ms;
- UI-thread CPU 554 -> 486 ms;
- tray composes 2 (15 ms) -> 4 (28 ms).

The patch's extra composes cost about 14 ms per send window, against about 68 ms of UI-thread CPU saved. First sends (2 per arm): reflows 22.5 -> 22.5 (19 and 26 vs 24 and 21); process CPU 2328 -> 2025 ms.

Visual check (fix, 120x40, live):
- switching conversations from a tray row rebuilds the tray, moving the selection highlight and the summary line;
- a warm send shows the running icon on its row mid-run and the idle icon after.

**Merge evidence (2026-10-26, base origin/dev 0d7e418db1, no rebase needed).**
- New file: 20 passed under the plain command on the branch. On origin/dev: 14 failed, 6 passed. The 6 that pass are 2 positive controls and the 4 refused-compose pins, which match dev by design and failed on the pre-fix head.
- 139 related test files (every file naming a touched module), run on the branch and on origin/dev and compared by test id. All 12 branch-only reds were resolved on re-run:
  - 6 passed serially on the branch;
  - 3 fail identically on dev serially (cost-chip active edit, switcher local probe, zero-result workspace search);
  - the two add-API-key recovery tests and the workspace persona bootstrap test flake on both arms with the same assertions (dev failed 5 of 6 and 3 of 6 runs).
- Governance: Python style inventory 37 passed; component pattern 18 passed; design tokens 8 passed; screen size ratchet 4 passed; UI-ready module census 4 passed (no module added, no CSS changed). The module size ratchet shows the same 8 reds as dev, none in touched files. The census check reports OK (907 entries, floor 906), and ./scripts/preflight.sh passes.
<!-- SECTION:NOTES:END -->
