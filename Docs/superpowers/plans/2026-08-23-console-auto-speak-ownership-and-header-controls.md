# Console Auto-Speak Ownership and Header Controls Implementation Plan

> Execute the two tasks as separate PRs. Stop after TASK-3070.10 is merged,
> create a fresh TASK-21201 worktree from the new `origin/dev`, then continue.

**Goal:** Put Console auto-speak policy behind `ConsoleHandsFreeController`, then move the Speak replies and Hands-free switches beside the Workbench status without changing speech behavior.

**Architecture:** `ChatScreen` keeps bounded Textual event delegates. `ConsoleHandsFreeController` owns policy and talks to the existing coordinator and widgets only through named late-bound callables wired in `UI/Console_Modules/wiring.py`. A later Console-specific speech widget mounts through one optional `DestinationHeader.before_status` seam; the control bar retains recovery actions and alone owns its dynamic height.

**Tech stack:** Python 3.11+, Textual 8.x, pytest/Pilot, Ruff, generated TCSS bundle.

**Design:** `Docs/superpowers/specs/2026-08-23-console-auto-speak-ownership-and-header-controls-design.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** The ownership change implements the approved Wave 6 boundary; the follow-up is routine layout polish that preserves existing contracts.

**Local-suite exception:** Do not run the full local suite. Run only the focused tests and gates listed below; GitHub Actions is the broad regression gate.

## Task A: Establish TASK-3070.10 red tests

**Modify:**

- `Tests/UI/test_console_controller_wiring.py`
- `Tests/Architecture/test_console_wave6_inventory.py` only if the existing contract lacks a no-DOM assertion

1. Add an unmounted wiring test that replaces the auto-speak coordinator after construction and proves enable, resume, retry, destination, auto-speak presentation, and Hands-free presentation edges are late-bound.
2. Assert the controller source contains no `query_one` call and the three decorated `ChatScreen` handlers remain bounded delegates.
3. Run the focused tests and confirm failure for the missing controller edges while recording the pre-existing Wave 6 inventory failure.

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_console_controller_wiring.py \
  Tests/Architecture/test_console_wave6_inventory.py
```

## Task B: Move the auto-speak policy edge

**Modify:**

- `tldw_chatbook/UI/Console_Modules/hands_free.py`
- `tldw_chatbook/UI/Console_Modules/wiring.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`

1. Add named constructor callables to `ConsoleHandsFreeController` for coordinator enable/resume/retry and for auto-speak and Hands-free presentation.
2. Move `_resolve_console_auto_speak_destination` and `_sync_console_auto_speak_controls` into the controller. Preserve their bodies except for the approved presentation callback substitution.
3. Replace `_sync_hands_free_switch`'s DOM query with the injected Hands-free presentation callback.
4. Add small controller request methods for enable, resume, and retry.
5. Wire every edge as a late-binding lambda. Coordinator construction may remain after the Hands-free controller because resolution occurs at call time.
6. Retarget coordinator destination and sync callbacks to `_hands_free`.
7. Reduce the three decorated screen handlers to `event.stop()` plus one controller delegation each.
8. Run the new red tests until green.

## Task C: Prove TASK-3070.10 behavior parity

**Targeted tests:**

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/test_probe_import_provenance.py \
  Tests/UI/test_console_controller_wiring.py \
  Tests/UI/test_console_auto_speak_wiring.py \
  Tests/UI/test_console_narrow_layout.py \
  Tests/UI/test_uat_first_time_character_chat.py \
  Tests/Architecture/test_console_wave6_inventory.py
```

**Static and derived gates:**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check \
  tldw_chatbook/UI/Console_Modules/hands_free.py \
  tldw_chatbook/UI/Console_Modules/wiring.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/UI/test_console_controller_wiring.py \
  Tests/Architecture/test_console_wave6_inventory.py

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/UI/Console_Modules/hands_free.py \
  tldw_chatbook/UI/Console_Modules/wiring.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/UI/test_console_controller_wiring.py \
  Tests/Architecture/test_console_wave6_inventory.py

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
  Scripts/check_persistent_diagnostic_inventory.py

git diff --check
```

If the diagnostic checker reports an expected source-derived change, inspect it, regenerate with `--write`, and rerun the checker. Do not accept unrelated inventory drift.

## Task D: Close TASK-3070.10 and merge independently

1. Self-review the diff against the Wave 6 inventory and design.
2. Check all TASK-3070.10 acceptance criteria and add concise Implementation Notes.
3. Set TASK-3070.10 to Done through Backlog CLI only after every DoD item is satisfied.
4. Commit, push, open the PR, and request Qodo review.
5. Address only technically valid findings, rerun focused gates, rebase on current `origin/dev`, and merge after required checks are green.
6. Do not begin TASK-21201 on this branch.

## Task E: Start TASK-21201 from merged dev

1. Fetch the merged `origin/dev` and create a fresh `codex/task-21201-console-speech-header` worktree.
2. Set TASK-21201 In Progress and attach this plan.
3. Immediately before editing UI code, read the Impeccable craft floor and run the layout detector against the header, control bar, and relevant tests.
4. Confirm no in-flight PR overlaps the header/control-bar files.

## Task F: Establish TASK-21201 geometry and interaction reds

**Modify:**

- `Tests/UI/test_console_workbench_contract.py`
- `Tests/UI/test_console_narrow_layout.py`
- the existing Console keyboard-navigation test file selected during implementation

Add production-CSS Pilot tests proving:

1. At 60, 90, 140, and 235 columns, title, both switches, and status share one row.
2. Status stays at the right padding edge for Ready, Running, and Blocked.
3. The subtitle region shrinks before any fixed child and ellipsizes without wrapping.
4. At 60x18 and 80x24 the header remains visible, both switches are reachable, and the normal transcript/composer geometry loses no row compared with the baseline.
5. Retry/Resume changes the control bar 1 -> 2 -> 1 rows with no blank row.
6. Tab cycles between header controls and composer/control-bar controls; F6 continues from the composer pane.
7. Programmatic sync remains silent while user gestures emit exactly one existing message.

Run these tests and confirm failure before production changes.

## Task G: Add the focused header speech widget

**Create:**

- `tldw_chatbook/Widgets/Console/console_speech_controls.py`

**Modify:**

- `tldw_chatbook/UI/Workbench/workbench_widgets.py`
- `tldw_chatbook/Widgets/Console/__init__.py`
- `tldw_chatbook/Widgets/Console/console_control_bar.py`
- `tldw_chatbook/UI/Console_Modules/wiring.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`

1. Move the two switches, labels, and their existing message types/handlers into `ConsoleSpeechControls`; preserve IDs, names, tooltips, and state-sync guards.
2. Add one optional `before_status: Widget | None` argument to `DestinationHeader` and yield it immediately before the existing status widget.
3. Compose `ConsoleSpeechControls` through that seam on Console only.
4. Keep Retry/Resume and their messages in the control bar.
5. Update the TASK-3070.10 presentation adapters in wiring to sync the header widget and recovery controls without changing policy.
6. Add `console-workbench-header` to the composer/control `CONSOLE_TAB_REGIONS` tuple and `CONSOLE_FOCUS_PANE_FOR_WIDGET` mapping.

## Task H: Make recovery height single-owned

**Modify:**

- `tldw_chatbook/Widgets/Console/console_control_bar.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`
- `tldw_chatbook/css/components/_agentic_terminal.tcss`

1. Make `ConsoleControlBar.sync_auto_speak` the sole place that sets one-row normal or two-row recovery height together with button visibility.
2. Remove the compose-time fixed height from `_compact_console_workbench_widget` for this bar.
3. Relax CSS to `height: auto`, `min-height: 1`, `max-height: 2`; keep the exact height authoritative in the widget sync.
4. Remove the compact-height rule that hides the Workbench header and retire the compact status stand-in plus its sync method.
5. Preserve the explicit focus-mode header hide.

## Task I: Style and regenerate production CSS

**Modify:**

- `tldw_chatbook/css/components/_agentic_terminal.tcss`
- `tldw_chatbook/css/tldw_cli_modular.tcss` (generated only)

1. Give the subtitle `min-width: 0`, `1fr`, nowrap, and ellipsis.
2. Give the speech group and status intrinsic non-shrinking widths.
3. Use existing semantic tokens and a two-cell group-to-status separation; add no new color vocabulary.
4. Regenerate and verify the bundle.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
  tldw_chatbook/css/build_css.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
  tldw_chatbook/css/check_bundle_sync.py
```

## Task J: Verify and close TASK-21201

Run only the focused geometry, navigation, speech, bundle, provenance, architecture, Ruff/format, diagnostic inventory, and `git diff --check` gates. Capture a live Textual render at 60x18, 80x24, and 140x42 and verify the painted output, not only widget regions.

Complete TASK-21201 acceptance criteria and Implementation Notes, set it Done only after DoD, then commit, push, open a separate PR, address Qodo feedback, rebase, and merge after required checks are green.
