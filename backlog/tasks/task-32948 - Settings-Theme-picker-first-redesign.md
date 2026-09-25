---
id: TASK-32948
title: Settings ▸ Theme picker-first redesign
status: In Progress
assignee: []
created_date: '2026-09-24 22:00'
labels:
  - settings
  - theme
  - ux
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A 2026-09-24 critique (18/40) found that Settings ▸ Theme is an editor that
happens to contain a theme list, not a way to switch themes: choosing a
theme takes three actions (select it in the tree, Apply it for the session,
then "Set as launch default" to keep it), themes appear as raw slugs with no
colour preview, and three surfaces — Appearance's 91-item dropdown, the
editor's tree, and the command palette's "Theme: Switch to…" — can disagree
about which theme is active or will load at the next launch. Most visits to
this category want to *use* a theme; few want to *edit* one.

This redesign makes Theme picker-first: a single filterable, grouped list of
every registered theme (yours, shipped, Textual's own) with a live preview
and one-key Use / Try / Revert, with the existing full editor demoted to an
explicit path behind Clone, New and (PR 2) Edit. Appearance stops duplicating
theme selection and instead shows a read-only summary that links into the
picker. Delivered as three PRs, each leaving a working screen (design spec
`Docs/superpowers/specs/2026-09-24-theme-picker-redesign-design.md`, §10).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 (PR 1) Theme opens on a picker: a filter box, a grouped list (yours / shipped / Textual) with a colour strip and word markers (`active`, `launch`, `overrides shipped`/`overrides textual`) per row, and a preview card that repaints on highlight without touching the running app
- [x] #2 (PR 1) Enter (or Use this theme) switches to the highlighted theme and saves it as the launch default in one action; `t`/Try switches for the session only and saves nothing; a Revert control appears after either and restores the state from before the first Try/Use in the chain
- [x] #3 (PR 1) `c`/Clone and `n`/New open the existing full editor in the same pane (a ContentSwitcher, not a modal), pre-loaded from the highlighted theme; Back returns to the picker and, with unsaved edits, asks Stay/Discard/Save
- [x] #4 (PR 1) Appearance shows a read-only theme row (launch default + active theme, each named) and an Open Theme button that lands on the picker; Appearance no longer has its own theme dropdown, and its Save/Preview no longer touch the theme
- [x] #5 (PR 1) At 80×24 and 190×55, under textual-dark and textual-light, every picker control is reachable and the list shows at least 5 rows at 80×24 (pinned by a geometry test); the picker's Use/Try chips and the highlighted row meet the existing contrast floor
- [x] #6 (PR 1) A config write failure on Use still switches the theme for the session and reports that the launch default was not saved, without crashing or leaving Revert unusable
- [ ] #7 (PR 2) The theme tree and its own New/Clone/Delete/Export row are removed from the editor now that the picker owns discovery; the editor gains Save as, Back to themes, Edit (opened from an existing saved theme's picker row) and Rename
- [ ] #8 (PR 2) Deleting the active or launch-default theme falls back to a working theme (not a crash or a dangling reference) and both the picker and Appearance reflect the fallback
- [ ] #9 (PR 3) Import brings an external theme file into the picker's YOUR THEMES group, through the same backup-recovery-gated file layer as Save/Delete/Export
- [ ] #10 (PR 3) The picker and editor state unreadable-theme-file and launch-default-missing conditions in plain language instead of silently dropping the row or crashing
- [ ] #11 (PR 3) Export's toast shows the full path and a Copy path chip
- [ ] #12 (PR 3) `Docs/User_Guide/settings.md` §Theme is rewritten to match the final (PR 3) picker + editor flow, with an updated Verified stamp
<!-- AC:END -->

## Implementation Plan

1. `theme_catalog` module: `ThemeEntry`/`build_catalog` (origins, markers, resolved colours) plus `use_theme`/`revert_theme`/`ThemeChange` as the one switching seam; route the command palette's `switch_theme` through it.
2. Extract `ThemePreview` from the editor so the picker can reuse it.
3. Build `ThemePicker` (filter, grouped `OptionList`, preview card, Use/Try/Clone/New/Revert, live markers via `app.theme_changed_signal`).
4. Wire `ThemePane` (a `ContentSwitcher`) into `settings_screen.py`: Theme opens on the picker; Clone/New swap in the unchanged editor behind a guarded Back.
5. Appearance: replace the theme `Select` with a read-only summary row + Open Theme button; stop writing `default_theme` from Appearance's own draft/save/preview path.
6. CSS: bound the picker's layout inside the scrolling detail pane at 80×24 and 190×55; extend the existing contrast test to the picker's chips and highlighted row.
7. Docs, task file, full test suites (targeted + hub-vs-base FAILED-set diff), preflight, and a live check in a scratch profile at both pinned sizes.

## Implementation Notes

**Status:** PR 1 shipped (this task file's ACs #1–#6). PR 2 (editor/file actions)
and PR 3 (import/edge states/final doc rewrite) are tracked by this same task
and are not started — their ACs (#7–#12) are left unchecked on purpose, per
the spec's 3-PR delivery plan (§10). This task stays `In Progress` until PR 3
closes it.

### What PR 1 shipped

- `tldw_chatbook/css/Themes/theme_catalog.py` (new): `ThemeEntry`, `build_catalog`,
  `display_name`, `is_catalog_theme`, `STRIP_KEYS`/`BASE_KEYS`, plus the
  switching seam `ThemeChange`, `use_theme`, `revert_theme`, `user_theme_names`,
  `current_launch_default`. `ThemeProvider.switch_theme` in `app.py` now routes
  through `use_theme`, replacing the old direct `self.app.theme = …` +
  `save_setting_to_cli_config(...)` pair.
- `tldw_chatbook/Widgets/theme_preview.py` (new): `ThemePreview`, extracted
  from the editor so both the editor and the picker share one paint path.
- `tldw_chatbook/Widgets/settings_theme_picker.py` (new): `ThemePicker`
  (filter, grouped `OptionList`, preview card, Use/Try/Clone/New/Revert,
  live markers) and `ThemePane` (the `ContentSwitcher` that swaps the picker
  for the editor on Clone/New, with a guarded Back).
- `tldw_chatbook/UI/Screens/settings_screen.py`: Theme's detail pane now
  mounts `ThemePane`; the pinned state banner is skipped for Theme; Appearance's
  theme `Select` is replaced by a read-only summary row (subscribed to
  `app.theme_changed_signal`) plus an Open Theme button; Appearance's draft,
  validation and save no longer touch `default_theme`.
- CSS (`css/components/_settings_splash_theme.tcss`, `css/features/_settings.tcss`,
  regenerated bundle): the picker's layout is bounded inside the scrolling
  detail pane at 80×24 / 190×55; the action-row chip stack is unconditional
  (not compact-only — it overflows at 190×55 too, see below); the highlighted
  row is bold via an ID-scoped selector.

### Retired or rewritten tests (with reasons)

- `test_settings_configuration_hub.py::test_settings_appearance_theme_options_*`
  (3 tests): pinned the deleted `_appearance_theme_options` dropdown source;
  the catalog module now owns theme enumeration/origin/dedupe, covered by
  `Tests/Utils/test_theme_catalog.py`.
- `test_settings_interface_keyboard_journeys.py`:
  `test_theme_launch_default_is_seen_by_appearance_without_losing_its_draft`,
  `test_partial_theme_save_updates_appearance_and_preserves_explicit_theme_draft`,
  `test_matching_launch_save_clears_appearance_sidebar_dirty_marker` — all
  three drove the removed Appearance theme `Select` / "Set as launch default"
  button; replaced by `test_try_then_use_then_revert_restores_original` and
  `test_filter_narrows_and_enter_uses` in the picker's own test file.
- `test_settings_theme_editor.py::test_settings_theme_editor_set_launch_default_requires_saved_theme`
  — covered the removed "Set as launch default" button; its write-failure
  sibling was rewritten as `test_settings_theme_editor_launch_default_write_failure_is_reported`
  (drives `_save_launch_default` directly, since Delete still calls it as a
  fallback).
- `Tests/UI/test_settings_theme_picker.py::test_compact_preview_has_two_rows` —
  covered `ThemePreview.__init__(compact=...)`, removed as YAGNI once the CSS
  alone hides non-rail/accent rows under the compact workbench class.
- Several `test_settings_configuration_hub.py` tests were **rewritten in
  place** rather than retired: `test_theme_category_opens_without_crashing`
  now asserts the picker is what Theme opens on; `test_state_banner_leads_with_persistence_badge`
  and `test_every_category_renders_the_state_banner` drop Theme's banner
  (Theme has none now); `test_settings_appearance_renders_guided_defaults_and_validates`
  and `test_settings_jk_never_steals_keys_from_select` were retargeted off the
  deleted theme `Select`.

### Deviations and rulings (from the SDD ledger, `progress.md`)

- **R1** — existing tests that drive the theme *editor* (not the picker) get
  a shared `Tests/UI/theme_editor_helpers.py::open_theme_editor` helper
  (focus the list, press `c`) instead of being retired, since the editor's
  own behaviour didn't change.
- **R2** — the preflight diagnostic-inventory row for the picker's
  `logger.warning(f"Theme revert failed: {exc}")` is closed (Task 7 ran
  `--write` after reading the flagged line; it doesn't interpolate a secret).
- **R3** — spec §6 says ThemePane routes "Back **and Save** to the picker";
  PR 1 wires **Back only** — Save stays in the editor for PR 1, and
  "Save returns to the picker" is deferred to PR 2's editor rework (spec §10
  lists Save-as/Edit under PR 2).
- **R4** — kept the single `#settings-theme-editor-view { height: auto }`
  rule (added early to prevent an editor-clipping regression) instead of
  duplicating it in the later layout pass.
- **R5** — the picker was re-added to the pre-existing Theme keyboard-journey
  walk, which an earlier task had narrowed to the editor only.
- **R6** — the picker's Clone/New buttons are `#settings-theme-picker-clone`
  / `#settings-theme-picker-new` (not `#settings-theme-clone`/`#settings-theme-new`,
  which the editor already owns and many existing tests pin) — duplicate ids
  would make `query_one`/`pilot.click` ambiguous.
- **R7** — the Scope Inspector's "Recovery" copy for Theme (two call sites)
  was rewritten to describe the picker's Revert/Clone flow instead of the old
  editor-only flow; no test pinned the old string.
- **R8** — `settings-theme-filter` was added to the search-index drift test's
  `NON_SETTING_CONTROLS` allowlist (it's a filter box, not a setting);
  `settings-theme-preset-target` (the drift test's other flagged id) is
  unverifiable locally on this machine on *either* branch (`RecoveryRequired`
  before any assertion runs, confirmed identical on a detached `origin/dev`
  worktree) — left for CI to decide, per the review's own instruction.
- **Colour contract fix at its source** (not a ruling, a bug fix): `theme_catalog._colours()`
  originally returned raw `Color.hex` values, which are 8-digit `#RRGGBBAA`
  for alpha<1 colours (e.g. `deep_dive_cyberspace`'s `error`) and an ANSI
  colour *name* (e.g. `"ansi_default"`) for Textual's `ansi-dark`/`ansi-light`
  builtins — both of which crashed `rich.style.Style(color=...)` the first
  time the picker painted a strip. Fixed by a `_colour_hex()` helper that
  always returns an uppercase `#RRGGBB` string, falling back to `#808080`
  for ANSI-typed colours (there is no resolvable RGB for those at this
  layer — they're only ever resolved against the real terminal's own
  palette). Pinned by `test_every_colour_of_every_entry_is_uppercase_rrggbb`.
- **Action-row chip stacking made unconditional, not compact-only**: the
  brief's CSS scoped the vertical chip stack to the compact workbench class,
  but the same five-chip row (Use/Try/Clone/New/Revert) overflows at
  190×55 too, because `#settings-theme-card-column` only ever gets 1fr
  (half) of the picker's width. Caught by the pre-existing, stricter
  keyboard-journey geometry test once the picker rejoined its walk (R5), not
  by the picker's own weaker "≥1 visible column" geometry test.
- **80×24 list-row floor**: growing/shrinking the list's own `min-height` has
  zero effect on the viewport overlap once the picker exceeds the visible
  area (the scroll position and the list's own end position shift together).
  Fixed by trimming the compact preview's border/margin instead, bringing
  visible list rows from 3 to 7 at 80×24 (floor is 5).
- **Incident (Task 2, now a standing HARD RULE in this branch's SDD context)**:
  an ad-hoc, unmocked `python -c` probe against the real `_apply_config_mutation`
  briefly wrote `general.default_theme = "nord"` to the user's live
  `~/.config/tldw_cli/config.toml`. Resolved with the user (set to
  `textual-dark`); no other config content was touched. Every subsequent task
  in this SDD chain ran only through pytest mocks or `TLDW_CONFIG_PATH`
  scratch profiles, verified by `~/.config/tldw_cli/config.toml`'s mtime
  staying unchanged across every session after the fix.

### Files (PR 1)

New: `tldw_chatbook/css/Themes/theme_catalog.py`, `tldw_chatbook/Widgets/theme_preview.py`,
`tldw_chatbook/Widgets/settings_theme_picker.py`, `Tests/Utils/test_theme_catalog.py`,
`Tests/UI/test_settings_theme_picker.py`, `Tests/UI/test_settings_theme_picker_screen.py`,
`Tests/UI/theme_editor_helpers.py`.
Modified: `tldw_chatbook/app.py`, `tldw_chatbook/UI/Screens/settings_screen.py`,
`tldw_chatbook/UI/Screens/settings_appearance_defaults.py`,
`tldw_chatbook/UI/Screens/settings_search_index.py`,
`tldw_chatbook/Widgets/settings_theme_editor.py`,
`tldw_chatbook/css/components/_settings_splash_theme.tcss`,
`tldw_chatbook/css/features/_settings.tcss`, `tldw_chatbook/css/tldw_cli_modular.tcss`,
`Docs/security/production-diagnostic-inventory.json`, plus the test files listed
above under "Retired or rewritten tests", `Tests/Architecture/test_no_blocking_io_on_message_pump.py`
(one stale, dev-inherited baseline entry removed), `Tests/UI/test_command_palette_providers.py`,
`Tests/UI/test_command_palette_basic.py`, `Tests/UI/test_settings_appearance_defaults.py`,
`Tests/UI/test_settings_save_commit_models.py`, `Tests/UI/test_css_build_integrity.py`,
and `Docs/User_Guide/settings.md` (Theme/Appearance sections rewritten for PR 1;
final rewrite deferred to PR 3 per AC #12).
<!-- Full task-by-task detail lives in .superpowers/sdd/2026-09-24-theme-picker-pr1/progress.md and task-1..7-report.md (gitignored, worktree-local). -->
