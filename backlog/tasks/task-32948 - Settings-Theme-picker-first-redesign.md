---
id: TASK-32948
title: Settings ▸ Theme picker-first redesign
status: Done
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
- [x] #7 (PR 2) The theme tree and its own New/Clone/Delete/Export row are removed from the editor now that the picker owns discovery; the editor gains Save as, Back to themes, Edit (opened from an existing saved theme's picker row) and Rename
- [x] #8 (PR 2) Deleting the active or launch-default theme falls back to a working theme (not a crash or a dangling reference) and both the picker and Appearance reflect the fallback
- [x] #9 (PR 3) Import brings an external theme file into the picker's YOUR THEMES group, through the same backup-recovery-gated file layer as Save/Delete/Export
- [x] #10 (PR 3) The picker and editor state unreadable-theme-file and launch-default-missing conditions in plain language instead of silently dropping the row or crashing
- [x] #11 (PR 3) Export's toast shows the full path; a Copy path chip appears in a result row on the picker card once the export completes (not on the toast itself — Textual's `notify()` renders plain text and can't host an interactive button, so the chip lives beside the path it copies instead)
- [x] #12 (PR 3) `Docs/User_Guide/settings.md` §Theme is rewritten to match the final (PR 3) picker + editor flow, with an updated Verified stamp
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

**Status:** DONE. All three PRs shipped (PR 1: ACs #1–#6; PR 2: ACs #7–#8;
PR 3: ACs #9–#12), per the spec's 3-PR delivery plan (§10). See "PR 3
(Tasks 1–5)" below for the final PR's Implementation Notes, rulings and
full-checks evidence.

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
### Final-review fixes

- **Click only highlights (spec D2).** `ThemeOptionList._on_click` sets the highlight and calls `prevent_default()` so Textual's `OptionList._on_click` (highlight + `action_select` = Use) never runs. A click on the row that is already highlighted does not Use either; Enter and the Use button are the only ways to Use.
- **Search never lands in the hidden editor.** `_land_search_focus_on_field` redirects a target inside `ThemePane` that is not the visible switcher child to `#settings-theme-list`. This is deliberately not a generic `display` check, because a collapsed Collapsible's contents are `display:none` too and TASK-23109 expands those.
- **Appearance can never write `default_theme` (spec §8).** `build_appearance_save_sections` drops `general.default_theme`. The config writer sets keys one at a time (`config._apply_literal_mutation_unlocked`), so the file's value survives. `_apply_appearance_save_result` merges each section into the in-memory config instead of replacing it, so memory keeps its launch default. There is a test for the `LaunchDefaultChanged`, then Appearance Save, path.
- **Theme help copy.** Theme gets its badge ("Applies immediately") and scope line back. The category description, runtime owner, and both "Open Theme picker" buttons/tooltips now describe the picker. The User Guide badge table is updated.
- **Revert lasts the whole session (R9).** The pending `ThemeChange` lives on `app.theme_revert_change`, so a recomposed picker brings back the "Revert to X" chip.
- **`revert_theme` returns bool.** It returns False when restoring the launch default fails. The picker then warns "Reverted the theme; the launch default was not restored".
- Tests cover Back ▸ Discard, Save and refused Save, plus the items above.

<!-- Full task-by-task detail lives in .superpowers/sdd/2026-09-24-theme-picker-pr1/progress.md and task-1..7-report.md (gitignored, worktree-local). -->

### PR 2 (Task 4): editor tree and library buttons removed — retired or rewritten tests

Retired (the tree is gone; each guarantee has a new owner):

- `test_settings_theme_editor.py::test_theme_tree_has_empty_state_guidance` → `test_settings_theme_picker.py::test_empty_your_themes_says_none_yet` (the empty state now lives in the picker list).
- `test_settings_theme_editor.py::test_settings_theme_editor_tree_lists_your_themes_first_and_expanded` → `Tests/Utils/test_theme_catalog.py::test_groups_and_order` (yours first) and `test_settings_theme_picker.py::test_filter_narrows_and_enter_uses` (the picker has no collapsed group; the filter replaces the shipped collapse).
- `test_settings_theme_editor.py::test_settings_theme_editor_tree_lists_every_textual_builtin` → `Tests/Utils/test_theme_catalog.py::test_groups_and_order` (asserts the TEXTUAL group equals `BUILTIN_THEMES`).
- `test_settings_theme_editor.py::test_settings_theme_editor_empty_your_themes_says_none_yet` → new `test_settings_theme_picker.py::test_empty_your_themes_says_none_yet` (disabled, so inert). The picker did not render "(none yet)" before this task (spec §9); it does now.

Rewritten (every file, confirmation and registration assertion kept):

- Delete tests (`…delete_blocks_builtin_themes`, `…delete_blocks_shipped_themes`, `…delete_removes_custom_theme`, `…delete_user_file_shadowing_shipped_name`, `…delete_missing_custom_theme_warns`, `…delete_keeps_app_theme`, `…delete_unregisters_and_restores_shadowed_shipped_theme`, `…delete_user_file_shadowing_builtin_restores_it`, `…name_box_drives_apply_save_reset_delete`) drive `request_delete(name)` (what the picker's Delete calls) instead of the removed `on_delete_theme`; tree-label checks became `list_user_theme_names()`.
- `…export_confirms_before_overwriting` drives `export_theme("ocean")` on a saved file (R14: the editor's Export is gone).
- `…cleared_name_blocks_actions_instead_of_using_stale_name` checks Try and Reset (Delete/Export now take an explicit name from the picker).
- `…survives_backup_recovery_pause` asserts `list_user_theme_names()` raises `RecoveryRequired` (the picker's pause row is `test_pause_row_disables_file_actions` / `test_picker_lists_via_editor_scope`).
- `…your_themes_placeholder_tracks_save_and_delete` → `…saved_theme_listing_tracks_save_and_delete` (the listing the picker renders gains/loses the theme).
- `Tests/Backup_Recovery/test_settings_file_participant_lifetimes.py::test_theme_save_failure_and_pause_preserve_draft_and_tree`: `_user_theme_labels` → `editor.list_user_theme_names()`; pause/failure assertions unchanged.
- `test_settings_configuration_hub.py::test_settings_jk_leaves_theme_editor_focus_alone`: the Tree selector became the Dark checkbox.
- `test_settings_theme_card_contrast.py`: the editor's button set is now Try/Save/Reset (filled) and Save as/Generate (plain).
- `test_settings_theme_picker_screen.py`: the R6 id test expects no editor Clone/New; the Back-prompt tests make a real edit (Clone now opens clean).
- `test_css_build_integrity.py`: the `#settings-theme-tree` pin became `#settings-theme-editor-header`.

### PR 2 complete (Tasks 1–6): editor file API, picker Edit/Rename/Delete/Export, backup-pause row, editor rework, PR 1 carry-over polish, geometry/contrast/docs

**What shipped**, on top of the retired/rewritten tests above:

- **Editor file API** (`settings_theme_editor.py`): `list_user_theme_names()`,
  `request_delete(name)`, `rename_user_theme(old, new) -> bool`,
  `export_theme(name)`, and a no-field `ThemesChanged` message posted after
  any successful delete/rename/save. Every file op resolves through
  `_user_theme_files() -> dict[name, Path]` (keyed by `[theme].name`, stem
  fallback), not `f"{name}.toml"` (R12), so a file whose declared name
  differs from its filename is still found by every operation, not just
  listed. `theme_from_file_data()` was extracted in `css/Themes/themes.py`
  so the startup loader and rename share one Theme-construction path.
- **Picker Edit/Rename/Delete/Export** (`settings_theme_picker.py`): a
  second `settings-action-row` with those four buttons, plus `e`/`r`/Delete-key
  list bindings, visible and armed only when the highlighted entry is one of
  **yours** (`_can_manage_files()`); Rename and Delete open confirmation
  UI, Export writes straight to `~/Downloads`. `ThemePicker` now takes an
  injectable `list_user_names` and is fed the editor's own
  `list_user_theme_names` (Task 3) instead of scanning the directory a
  second time.
- **Backup-pause row**: a disabled "Theme files unavailable while
  backup/recovery is in progress" row appears under YOUR THEMES during a
  pause; Edit, Rename, Delete and Export (buttons and keys alike, R17) are
  disabled with that string as a tooltip; Use and Try keep working. The
  editor's own Save and Save as get the same tooltip treatment (R20), fed
  from `picker.files_available` through `open_editor`.
- **Editor rework**: the theme tree, its hint, and the New/Clone/Delete/Export
  row are gone. A `#settings-theme-editor-header` Static reads "Editing
  \<name\> · copy of \<source\>" / "· new" / "· saved theme". Actions are
  Try (relabelled from Apply), Save, **Save as…** (new — prompts via
  `RagProfileNameModal`, always confirms an overwrite even of the loaded
  theme's own name, R21), Reset, Generate. Save and Save as share
  `_save_under`; on success both post `ThemesChanged` then `Saved(name)`,
  which `ThemePane._saved` turns into `show_picker()` +
  `refresh_catalog(highlight=name)` — Save always returns you to the picker.
  If the saved theme is the one currently running, `_reapply_if_active`
  repaints it live so it's never stale; this lives in the editor, not the
  pane's `Saved` handler, because a category-leave Save can fire against a
  pane that's about to be detached (R18).
- **Delete fallback**: deleting the active-and-launch-default theme resets
  both to Textual Dark and says so; deleting the merely-active one switches
  to your launch default; either way `ThemesChanged`/`LaunchDefaultChanged`
  keep the picker and Appearance in sync (AC #8; pinned by
  `test_appearance_summary_recomposes_after_launch_default_changes_elsewhere`).
- **PR 1 carry-over polish** (Task 5): the command palette's persisted-Use
  toast now matches the picker's exact wording via one shared
  `theme_catalog.use_theme_toast()` helper (fix round 1 folded the picker's
  own Use toast onto the same helper, closing a gap where only the palette
  warned on a failed cache reload); `ThemeChange` gained
  `caches_reloaded: bool = True` (`merge()` ANDs it, ORs `persisted`, R11);
  Appearance's Open Theme now highlights the **launch default**, not merely
  the active theme, queued via `SettingsScreen._after_category_panes` rather
  than `call_after_refresh` (the latter raced the category-swap worker and
  was overwritten by the picker's own `on_mount` highlight — a real bug the
  test caught, not a style choice); the Revert chip names the launch default
  too when a persisted change left it disagreeing with the active theme.
- **Geometry, contrast, docs** (Task 6): `test_every_picker_control_is_reachable`
  now writes a user theme, highlights it, and walks the four yours-only
  buttons too, at 80×24/190×55 across textual-dark/light (re-scrolling the
  list into view before measuring its row count — scrolling to the lower
  action row had pushed the list off-screen, a real regression the extended
  test caught: `3 >= 5` failed at 80×24 before that fix); a new
  `test_every_editor_control_is_reachable` walks Back/Name/Save/Save as/Try
  at both sizes. A new `test_picker_delete_chip_meets_contrast_at_rest_and_focus`
  (R19/R23) measures the picker's Delete chip across all 4 THEMES at rest
  and focused: focus was falling back to the generic neutral
  `.settings-action-row Button:focus`, because the colour-keeping
  `.theme-editor-action.-error:focus` rule only ever existed scoped to
  `#settings-theme-card` — the editor's own card, whose Delete Task 4
  removed — and was never re-added for the picker's Delete under
  `#settings-theme-card-column` (confirmed RED first: `focus shift 1.09:1`
  against the 3.0 floor). Fixed with the same rule re-scoped to
  `#settings-theme-card-column .settings-action-row .theme-editor-action.-error:focus`
  and a CSS bundle rebuild. Fix round 1 found the same defect on the picker's
  Use chip (variant primary) — R24 — via a sibling
  `test_picker_use_chip_meets_contrast_at_rest_and_focus`, also RED first
  (`focus shift 1.09:1`) then GREEN after adding
  `#settings-theme-card-column .settings-action-row .theme-editor-action.-primary:focus`
  and rebuilding the bundle again. The keyboard-journey Theme walk
  (`test_settings_interface_keyboard_journeys.py`) needed no code change —
  it discovers whatever's mounted under `#settings-theme-editor-view`
  generically (`Button, Input, Select, Checkbox, OptionList, Tree`) and was
  independently confirmed to already cover every new editor control (probed
  directly: back/name/dark-mode/apply/save/save-as/reset/generate/ten colour
  inputs/preset select), plus the full 4-parametrization run green
  (4 passed). `Docs/User_Guide/settings.md` §Theme rewritten for the
  picker's yours-only actions and their keys, the reworked editor
  actions/header, Save's return-to-picker behaviour, and the pause copy; the
  stray "Theme Library" name is gone. The production diagnostic inventory
  was regenerated after reading all 4 added / 1 removed logger rows in
  `settings_theme_editor.py` (all Task 1's file-op error logs: theme names
  and `_failure_reason()`-scrubbed reasons only, no paths — the removed one
  is the R16 fix that stopped logging the full path).

**Rulings (R10–R23; full detail in the gitignored, worktree-local
`.superpowers/sdd/2026-09-25-theme-picker-pr2/progress.md`)**:

- R10 — Task 1 kept `_delete_user_theme`'s tree-node sync until Task 4
  deleted the tree, so the two tasks' tests stayed green independently.
- R11 — `ThemeChange.caches_reloaded` follows Task 5's text, not the plan's
  stale "returns the new launch default" summary.
- R12 — list/delete/rename/export identity is `_user_theme_files()`
  (`[theme].name` → path), not `f"{name}.toml"`.
- R13 — renaming the theme currently loaded in the editor updates the
  editor's `current_theme_name` too, so a later Save can't resurrect the
  old file under the old name.
- R14 — the pre-Task-4 editor Export kept exporting the *working* (possibly
  unsaved) palette, not the saved file, until Task 4 removed the button.
- R15 — `RecoveryRequired` must escape the per-file scan handler (the
  picker's pause row depends on it, not a silent skip).
- R16 — no file-op notice or log interpolates a full path; theme names and
  `_failure_reason()`-scrubbed reasons only.
- R17 — the backup-pause gate covers Edit too, not just Rename/Delete/Export
  (the brief's own two statements disagreed; the more specific one won).
- R18 — Save's re-apply-if-active lives in the editor
  (`_reapply_if_active`), not the pane's `Saved` handler, since a
  category-leave Save can fire against a pane about to be detached.
- R19/R23 — Task 6 adds a picker-card contrast case for the Delete chip
  covering both rest and **focus** (PR 1's TASK-32947 found
  `.settings-action-row Button:focus` neutralises variant fills, and the
  colour-keeping override was never re-scoped from the editor's card to the
  picker's).
- R20 — the editor's Save/Save as are disabled with the pause tooltip, fed
  from `picker.files_available` via `ThemePane.open_editor`.
- R21 — Save as always confirms an existing target, including the source
  theme's own name (Save as always means "a new file").
- R22 — the editor-scoped `.theme-editor-action.-error:focus` rule was
  removed only after confirming it matched nothing (the editor's own Delete
  is gone; the unscoped `.theme-editor-action.-error:enabled` rest-state
  rule still matches the picker's Delete and was kept).
- R24 (fix round 1) — the same defect class as R19/R23, on the picker's Use
  chip (variant primary): `#settings-theme-card-column` needs its own
  `.theme-editor-action.-primary:focus` override too, or a focused Use chip
  falls back to the generic neutral fill. Only `-primary` and `-error` are
  fixed — the picker's other chips (Try/Clone/New/Revert/Edit/Rename/Export)
  carry no variant hue at rest to preserve, so the generic focus fill is
  correct for them and was left alone.

**Files (PR 2, cumulative, Tasks 1–6):** new —
`tldw_chatbook/Widgets/settings_theme_picker.py` (already existed from PR 1;
gains the file-action buttons/bindings), `Tests/UI/test_settings_theme_file_api.py`.
Modified — `tldw_chatbook/Widgets/settings_theme_editor.py`,
`tldw_chatbook/UI/Screens/settings_screen.py`, `tldw_chatbook/app.py`,
`tldw_chatbook/css/Themes/theme_catalog.py`, `tldw_chatbook/css/Themes/themes.py`,
`tldw_chatbook/css/components/_settings_splash_theme.tcss`,
`tldw_chatbook/css/features/_settings.tcss`, `tldw_chatbook/css/screen_agentic_settings.tcss`,
`tldw_chatbook/css/tldw_cli_modular.tcss`, `Docs/security/production-diagnostic-inventory.json`,
`Docs/User_Guide/settings.md`, plus the test files listed above under
"Retired or rewritten tests" and `Tests/Utils/test_theme_catalog.py`,
`Tests/UI/test_settings_theme_picker.py`, `Tests/UI/test_settings_theme_picker_screen.py`,
`Tests/UI/test_settings_theme_card_contrast.py`, `Tests/UI/theme_editor_helpers.py`,
`Tests/UI/test_command_palette_providers.py`, `Tests/UI/test_settings_configuration_hub.py`,
`Tests/UI/test_css_build_integrity.py`, `Tests/Backup_Recovery/test_settings_file_participant_lifetimes.py`.

**Full checks (Task 6, real output):** the 14 named suites together
(`Tests/Utils/test_theme_catalog.py`, `test_settings_theme_picker.py`,
`test_settings_theme_picker_screen.py`, `test_settings_theme_file_api.py`,
`test_settings_theme_editor.py`, `test_settings_theme_editor_render.py`,
`test_settings_theme_card_contrast.py`, `test_theme_contrast.py`,
`test_settings_interface_keyboard_journeys.py`, `test_css_build_integrity.py`,
`test_user_theme_loader.py`, `test_settings_file_participant_lifetimes.py`,
`test_no_blocking_io_on_message_pump.py`, `test_command_palette_providers.py`),
`-n 4`: **462 passed, 4 failed in 413.5s**. The 4 failures
(`TestTabNavigationProvider::test_palette_library_skills_command_opens_hidden_starter_route`,
`TestSettingsProvider::test_discover_shows_popular_settings`,
`TestLibraryIngestProvider::test_search_ingest_returns_exactly_one_hit`,
`TestLibraryIngestProvider::test_discover_includes_library_ingest`) are
stale hardcoded hit-counts unrelated to theme code (confirmed pre-existing
in Task 5's report by running them in isolation, untouched by this branch's
diff). Hub-vs-base comparison
(`test_settings_configuration_hub.py` + `test_command_palette_providers.py`,
`-k "theme or Theme or TabNavigationProvider or SettingsProvider or
LibraryIngestProvider"`) in a detached scratch worktree at
`origin/feat/theme-picker-pr1` (2c4585c1ea): base **10 failed, 42 passed**;
branch **10 failed, 43 passed** (branch's +1 pass is a new Task 5 test
matching the `-k` filter). The FAILED test names are identical on both —
same 6 hub tests without `@private_profile_test` (CI-only, `RecoveryRequired`)
plus the same 4 palette tests above. Scratch worktree removed afterward.
`ruff check` on every file this task touched: `All checks passed!` before
and after. `PYTHON=<venv> ./scripts/preflight.sh`: rc 1 → (after reading
and writing the diagnostic inventory rows above) rc 0.

**Live check (Task 6, real output).** Ran twice in a scratch `HOME`/
`TLDW_CONFIG_PATH`/XDG profile (`tmux`, splash disabled), at 190×55 and
80×24: Clone → edit Primary → Save (toast "Theme '\<name\>' saved"; returned
to the picker with the new theme highlighted, YOUR THEMES gained it) → `r`
Rename (toast "Renamed '\<old\>' to '\<new\>'") → `e` Edit (header "Editing
\<name\> · saved theme"; the earlier colour edit was still there — a real
reload of the saved file, not a fresh clone) → Save as… (prefilled
"\<name\>\_copy"; toast "Theme '\<name\>\_copy' saved") → Use (toast
"\<name\> is now your theme (was: Textual Dark)", matching Task 5's shared
toast helper) → Delete (confirmed; toast "Deleted '\<name\>'; launch default
and theme reset to Textual Dark" — AC #8 live) → Export (toast named the
scratch profile's own Downloads path). Every toast was captured in the same
command as its triggering click, per the brief's evidence-trap warning.
Both `<name>_theme.toml` exports were confirmed on disk under the scratch
`HOME/Downloads`, absent from the real `~/Downloads`. The real
`~/.config/tldw_cli/config.toml` mtime (`Sep 25 00:10:38`) and the real
`~/.config/tldw_cli/themes/` directory (empty) were checked before the first
launch and after both passes; neither changed. One trap hit along the way:
the OptionList's Delete-key binding only fires once the list itself has
keyboard focus, not merely a highlighted row from a prior click — a stray
click that leaves focus on a button (e.g. after "Use this theme") silently
no-ops the `delete` key; clicking back onto the list row first, or using the
Delete button directly, both work.

**R25 correction (fix round 1).** The original captures for the pass above
were saved to `/tmp/pr2_*.txt` (37 files), not under the run's scratch profile
(`.../scratchpad/live-pr2/`) as the brief's own convention asks — a review
found none there. Copied all 37 into
`.../scratchpad/live-pr2/captures_original/` (still `/tmp` too, but now also
durably in the expected location; full name list in task-6-report.md). Then
re-ran the two most load-bearing steps in the same scratch profile (it still
held `pr2-verify`/`pr2-verify80` from the earlier pass), capturing directly
under `.../scratchpad/live-pr2/captures/` this time:
`03_clone_toast.txt` (Clone), `04_save_returns_to_picker.txt` (Save — this
one capture shows both the "Theme 'textual-dark_copy' saved" toast *and* the
picker view it returned to, with the new theme highlighted under YOUR
THEMES), `05_use_toast.txt` (Use), `06_delete_confirm.txt` (the confirm
dialog), and `07_delete_fallback_toast.txt` ("Deleted 'textual-dark_copy';
launch default and theme reset to Textual Dark" — AC #8, live, again). Real
config mtime and the real `themes/` directory were re-checked before this
re-run and after: unchanged. See task-6-report.md's Fix Round 1 section for
the full path list.

### PR 2 final-review fixes (R12 completion, R26, R27)

- **I1 / R12 completion** — Edit/Clone/Save of a theme whose `[theme].name`
  differs from its file stem. `load_user_theme` and `_save_under` now resolve
  the file through `_user_theme_files()` (RecoveryRequired → the unavailable
  notice), so Edit on `a.toml` (name "b") loads b, Save writes back to
  `a.toml` (no second `b.toml`), and Clone clones b's palette. Deviation:
  no `_loaded_user_path` field — resolving at save time covers Save *and*
  Save as with no extra state.
- **I2** — renaming a non-active launch default only rewrites config
  (spec §7 step 4): new `theme_catalog.persist_launch_default()`; the
  running-theme switch stays for the active theme only.
- **I3** — the session's pending Revert follows a rename and is dropped on
  delete (`theme_catalog.retarget_pending_revert`); the picker re-syncs the
  Revert chip on every catalog refresh.
- **R26** — Esc with nothing focused in the editor view acts as "Back to
  themes" (leave prompt when dirty); the first Esc still releases field
  focus (task-1560). `SettingsScreen.on_key` + `_theme_editor_shown()`.
- **R27 (5)** — while paused, the picker keeps the last successfully listed
  user names, so your themes are not relabelled "shipped".
- **R27 (6)** — an `OSError` from the lister reads as "no user themes"
  (warning logged; `files_available` stays True).
- **R27 (7)** — the Rename prompt title escapes the theme name
  (`escape_markup`). The Save-as title ("Save theme as") contains no name,
  so it needed nothing; the name is only its Input's value (not markup).
- Tests: 2 in `Tests/UI/test_settings_theme_picker.py`, 8 in
  `Tests/UI/test_settings_theme_picker_screen.py`; User Guide updated
  (Esc, Revert/rename).

### Delete-of-launch-default fallback narrowed (user decision)

- User decision 2026-09-25: deleting the launch default changes only the
  setting unless that theme is on screen.

### PR 3 (Tasks 1–5): unreadable files, Import, launch-default-missing +
### Export Copy path, theme-name hardening, User Guide + close-out

**What shipped**, on top of PR 2:

- **Task 1 — unreadable files are listed, not hidden.** A saved file that
  fails to parse (bad TOML, missing/invalid `[colors].primary`, an
  unexpected `[colors]` key) is no longer silently skipped by the picker: it
  appears under YOUR THEMES as `<name> (unreadable)` with the reason as both
  a card label and every disabled button's tooltip. Use/Try/Clone/New/Edit/
  Rename/Export are disabled on that row (R29 extended this to New, which
  would otherwise silently start from the editor's last palette); Delete
  still works. `_scan_theme_files()` is the one pass that produces both the
  readable map and the unreadable map, reused by the startup loader's own
  check.
- **Task 2 — Import (AC #9).** `Import…` (button beside New, or `i`) opens
  a path prompt (`RagProfileNameModal`) for the user's own chosen source
  file — not something the app owns or has staged; the app reads whatever
  path is typed, pasted or dropped, wherever it lives on disk. That source
  is read through one non-blocking descriptor (R34, so a FIFO or a
  swapped-in special file can't stall the UI), capped at 64 KB, and refused
  with a specific reason for: not `.toml`, over the cap, not valid TOML,
  missing/invalid `[colors].primary`, an unexpected `[colors]` key (R32 —
  enforced once in the shared `theme_from_file_data`, so Import, the
  listing and the startup loader all refuse the same files), a name that
  isn't filename-safe or contains `[`, or the two Textual built-in names.
  Once accepted, the write itself goes through `_write_toml` /
  `_write_import`, the same backup-recovery-gated file layer Save, Rename
  and Delete already use (`RecoveryRequired` during a pause is caught and
  reported the same way, and nothing is written while paused) — Import adds
  no second write path of its own. A `[variables]` value
  that isn't a colour is dropped with a warning instead of refusing the
  whole file, per spec §7. Importing an existing name asks first
  (`Replace the saved theme '<name>'?`); Cancel leaves the file
  byte-identical. macOS drag-drop's backslash-escaped paths are unescaped
  (R35) without touching any other backslash.
- **Task 3 — launch-default-missing notice + Export Copy path.** The picker
  shows its own "Launch default missing: `<id>` — Use any theme to fix it"
  above the list (spec §9), matching Appearance's existing summary row;
  using any theme clears it. Export now posts `Exported(path)`, and the
  picker shows a result row — "Exported to `<full path>`" plus a **Copy
  path** button (`app.copy_to_clipboard`, "Path copied") — that clears on
  the next highlight.
- **Task 4 — theme names render literally everywhere (R28).** A full
  call-site audit of every place a theme name reaches the UI (notify,
  Button labels, the Revert chip, the command-palette's theme Hit/help,
  `ConfirmationDialog`) found and fixed the surfaces that parse markup but
  weren't escaping an untrusted (file-derived) name, including the palette's
  `ThemeProvider.search`, which crashed outright on a name containing `[`
  before this fix. `Reset` now resolves the theme file through the same
  `_user_theme_files()` identity every other file op uses (a name≠stem file
  used to be found by its stem, not its declared name). R31 fixed
  `dark = "false"` (a string) reading as `True` in three places that shared
  the bug (listing, editor load, Import) with one new `theme_file_dark()`
  helper.
- **Task 5 (this task) — User Guide rewrite, close-out, full checks, live
  check.** `Docs/User_Guide/settings.md` §Theme is rewritten in place (not
  patched) to state the launch-default-missing notice, the unreadable-file
  row and its disabled actions, and the full Import paragraph (path entry,
  the 64 KB/`.toml`/refusal-reason rules, the `[variables]` sanitisation,
  the Replace confirmation) as part of the section's one continuous
  narrative, alongside the pre-existing Use/Try/Revert, Edit/Rename/Delete/
  Export, editor Save/Save as/Reset/Back-and-leave-prompt, pause-row and
  Appearance-row coverage.

**Bug found and fixed in Task 5 (full checks).** The Export result row's
**Copy path** button was reachable by keyboard focus even while its row was
hidden: `_show()` and `show_export_result()` toggled `display` on the
wrapping `Horizontal#settings-theme-export-result`, but a widget's own
`.display` flag doesn't inherit a hidden ancestor's, so the button itself
still reported `display=True` at rest. `Tests/UI/
test_settings_interface_keyboard_journeys.py::test_interface_controls_are_keyboard_reachable_and_painted[Theme-*]`
caught it (all 4 parametrizations: `Failed: Keyboard focus never reached
#settings-theme-copy-path`) — reproduced in isolation, not a parallel-worker
flake. Every other conditionally-shown control in this widget (the Revert
chip, the four yours-only file-action buttons) already sets its own
`.display`, not just a container's; the Copy-path button gets the same
treatment now (`settings_theme_picker.py`, `_show()` and
`show_export_result()`). Green after the fix: the 4 keyboard-journey
parametrizations, plus a full rerun of `test_settings_theme_picker.py` +
`test_settings_theme_picker_screen.py` (77 passed) to confirm nothing else
regressed.

**Rulings R29–R38** (full detail in the gitignored, worktree-local
`.superpowers/sdd/2026-09-25-theme-picker-pr3/progress.md`):

- R29 — New and Clone are blocked on an unreadable row like the other
  actions (New silently used the editor's last palette otherwise).
- R30 — a CSS-injection `[variables]` entry is dropped (import succeeds),
  per spec §7, not refused.
- R31 — `dark = "false"` (a string) is coerced correctly in the loader, the
  editor's own load path, and Import.
- R32 — only the ten base colour keys are accepted in `[colors]`, enforced
  once in `theme_from_file_data` (Import, listing, startup loader).
- R33 — the Replace dialog and the Imported toast escape the name.
- R34 — the import source is opened `O_NONBLOCK`, `fstat`'d, then read
  (bounded) from the same fd — no FIFO stall.
- R35 — macOS drag-and-drop's backslash-escaped paths are unescaped.
- R36 — pinned regression tests: a FIFO, a symlink to a special file, a
  symlink to an oversized file, a NUL in the path, an uppercase `.TOML`
  suffix.
- R37 — the post-import `query_one(ThemePicker)` is gone; `ThemesChanged
  (highlight=name)` drives the single refresh, guarded against a missing
  pane.
- R38 — `_handle_theme_rename_result`'s `query_one(ThemePicker)` is
  guarded the same way (R37's sibling call site).

**2026-09-25 user decisions**, both confirmed live in Task 5's check:

- **Delete of the launch default while another theme is only Tried (not
  saved).** Deleting the configured launch default changes only the
  setting when that theme isn't the one on screen — the running theme is
  left alone. Live-verified: Used theme A (making it launch default and
  active), Tried theme B (screen shows B, A is still the launch default),
  deleted A — toast "Deleted 'a'; launch default reset to Textual Dark",
  and B stayed active/on screen afterward (confirmed via the picker's own
  `active` marker on B, unchanged by the delete).
- **Clone → rename in the Name box → Back discards the new name without a
  prompt.** This is existing behaviour (TASK-31251), reaffirmed for PR 3:
  the Name box drives `current_theme_name` for Try/Save/Reset, but typing in
  it alone does not set `is_modified` (only a colour/dark-mode edit does),
  so a Clone whose only change is the name is still "clean" and Back
  returns to the picker with no Stay/Discard/Save prompt, silently dropping
  the typed name. Confirmed by reading `on_theme_name_changed` (
  `settings_theme_editor.py`) and `on_clone_theme`'s `self.is_modified =
  False`.

**Full checks (Task 5, real output).**

- The 15 named suites together, `-n 4`: first pass **8 failed, 535 passed
  in 432.83s** — 4 of the 8 were the 4 `test_interface_controls_are_
  keyboard_reachable_and_painted[Theme-*]` failures fixed above, the other
  4 are the stale command-palette hit-count tests every prior PR 3 task
  report already pins as pre-existing and unrelated
  (`test_palette_library_skills_command_opens_hidden_starter_route`,
  `test_discover_shows_popular_settings`,
  `test_search_ingest_returns_exactly_one_hit`,
  `test_discover_includes_library_ingest`). After the fix: the 4 Theme
  keyboard-journey parametrizations pass in isolation (**4 passed**); a
  fresh full run of all 15 suites gives **4 failed, 539 passed in
  412.24s** — the remaining 4 are exactly the same pre-existing palette
  tests, zero Theme-related failures.
- **Hub-vs-base.** `Tests/UI/test_settings_configuration_hub.py` +
  `Tests/UI/test_command_palette_providers.py`, `-n 4`, in a detached
  scratch worktree at `origin/feat/theme-picker-pr2` (`bd321fa8ab`): base
  **300 failed, 181 passed**; branch (HEAD; neither of these two files was
  touched by Task 5's own geometry fix) **300 failed, 182 passed** — the
  branch's +1 pass is simply a hub test that this branch's palette/theme
  additions make collectible/passable that the base doesn't have. The
  FAILED test NAME sets are identical (compared as a Python set after
  stripping interleaved-log noise from the raw `FAILED` lines): 299
  distinct failures, every one the same RecoveryRequired/CI-only hub gate
  or the same 4 stale palette tests on both sides. Scratch worktree removed
  after.
- **Ruff.** Every file this task's 4 implementers touched, base vs HEAD,
  same per-file finding COUNT (`app.py` 481→481, `themes.py` 3→3,
  `settings_screen.py` 113→113, `settings_theme_editor.py` 13→13,
  `settings_theme_picker.py` 0→0 both before and after Task 5's own
  reachability fix, `theme_catalog.py` 0→0). No new findings anywhere.
- **Preflight.** `PYTHON=<venv> ./scripts/preflight.sh`: rc 0 (all 9
  derived-artifact checks pass — CSS bundle, Canvas Mermaid, profile-owned
  path census, diagnostic inventory, backlog task ids, chachanotes table
  allowlist, index plan pins, textual worker contract, timestamp writers,
  gated Tests/UI census).

**Live check (Task 5, real output).** Isolated `HOME`/`XDG_*`/
`TLDW_CONFIG_PATH` scratch profile under
`.../scratchpad/live-pr3/`, splash disabled, at 190×55 and 80×24; every
capture taken in the same command as its triggering keystroke, saved under
`.../scratchpad/live-pr3/captures/`. Real `~/.config/tldw_cli/config.toml`
mtime (`Sep 25 00:10:38 2026`) and the real (empty) `~/.config/tldw_cli/
themes/` directory were checked before the first launch and after the last
one: unchanged. The real `~/Downloads` tail was also unchanged throughout;
every Export in this session landed under the scratch profile's own
Downloads.

1. Imported a valid theme from a scratch path (typed, quoted) — toast
   "Imported 'pr3_imported'", the theme appeared under YOUR THEMES,
   highlighted. Imported a broken file (missing `[colors].primary`) —
   error toast "Missing [colors].primary", nothing written.
2. Dropped a garbage (`this is not valid toml [[[`) file directly into the
   scratch themes directory, reopened Theme (category switch forces a
   rescan) — it appeared as "Garbage (unreadable)" with "not valid TOML" on
   the card; Delete removed it from disk.
3. Used theme A as launch default, Tried theme B (screen shows B), then
   deleted A (the launch default, not on screen) — toast "Deleted
   'pr3_imported'; launch default reset to Textual Dark"; B stayed
   active/on screen, confirmed unchanged by a fresh filter+highlight after
   the delete.
4. Cloned a shipped theme, Saved it, Exported it — toast "Theme exported
   to: `<full path>`", the card showed "Exported to `<full path>`" with a
   Copy path button; clicking it toasted "Path copied", and the file was
   confirmed on disk under the scratch Downloads.
5. A theme named `x[/]`, dropped into the scratch themes directory before
   a fresh app launch (so the startup loader registers it — the realistic
   path for a `[`-containing name, since both Import and Rename refuse `[`
   or a literal `/` in a *typed* name; the startup loader and Save/Rename
   on an already-registered theme don't validate the name's contents,
   only its target *filename*), rendered literally everywhere it was
   driven live: the OptionList row ("X[/] · dark · yours"), the card title,
   and the Use toast ("X[/] is now your theme (was: Textual Dark)") — no
   `MarkupError`, no crash, at both 190×55 and 80×24 (confirmed the list
   floor holds with real YOUR THEMES + SHIPPED rows visible together at
   80×24 too).

One deviation from the brief's literal step 5 wording worth recording: a
raw `.toml` file dropped into the themes directory while the app is
*already running* is **not** picked up until the picker's underlying
`app.available_themes` registers it (readable "yours" entries in
`build_catalog` come from the app's live theme registry, not the disk scan
directly — only *unreadable* entries are synthesised straight from disk).
Import and Save/Rename both call `register_theme` explicitly; a bare drop
does not, and only takes effect after a restart (which runs the startup
loader). This is existing, correct, by-design behaviour, not a gap — it is
why Import exists — but it meant the live-check step for `x[/]` needed a
relaunch rather than a same-session drop, which the brief's wording didn't
spell out.

### Files (PR 3, cumulative, Tasks 1–5)

New: `Tests/UI/test_settings_theme_file_api.py`, `Tests/UI/
test_settings_theme_import.py`.
Modified: `tldw_chatbook/Widgets/settings_theme_editor.py`,
`tldw_chatbook/Widgets/settings_theme_picker.py`,
`tldw_chatbook/css/Themes/theme_catalog.py`,
`tldw_chatbook/css/Themes/themes.py`, `tldw_chatbook/app.py`,
`tldw_chatbook/UI/Screens/settings_screen.py`,
`Docs/User_Guide/settings.md`, plus `Tests/UI/test_settings_theme_picker.py`,
`Tests/UI/test_settings_theme_picker_screen.py`, `Tests/Utils/
test_theme_catalog.py`, `Tests/Utils/test_user_theme_loader.py`,
`Tests/UI/test_command_palette_providers.py`.

### PR 3 final-review fixes (R39, R40)

- **R39 (security, terminal escape injection):** TOML `\u001b` escapes in
  values (and raw ESC bytes, which the `toml` parser accepts, in quoted
  keys — it does not unescape keys) reached the terminal through import
  refusals, the unreadable-row card/tooltips and rename notices;
  `escape_markup` only escapes `[`. New `printable()` in
  `css/Themes/themes.py` (non-printable → `?`) is applied at the shared
  seams: `SettingsThemeEditor._failure_reason` (every file-error notice and
  the card error), `theme_from_file_data`'s key error, the Import refusal
  notice, the unreadable entry's display name, and the loader/scan log
  lines. A name that is not `isprintable()` is refused at Import ("control
  characters are not allowed") and by `theme_from_file_data` ("name has
  control characters"), so the startup loader skips it and the picker lists
  it as unreadable. Tests render the notice through `Content.from_markup` →
  `Strip` and assert every character is printable.
- **R40(a):** unreadable entries get id `unreadable:<stem>` (display
  unchanged), so a corrupted `nord.toml` is listed beside the built-in nord;
  `request_delete` resolves the prefixed id to the file (a readable theme
  of that exact name still wins) and names it by file name in the dialog
  and toast; `rename_user_theme` accepts either form.
- **R40(b):** `ThemePicker(list_themes=)` — one hook returning
  `(names, unreadable)`; `ThemePane` feeds it one `user_theme_listing()`
  scan per refresh. Per-file parse failures log at WARNING. Diagnostic
  inventory rows reviewed (file name + short path-free reason only) and
  rewritten.
- **R40(c):** Import accepts only what the editor's
  `_validate_color_input` accepts (`#RGB`/`#RRGGBB`): "`<key>: '<value>' is
  not #RRGGBB`".
- **R40(d):** a blocked Enter/t/c/n/e/r on an unreadable row notifies
  "This theme file can't be read: <error>" once per key.
- **R40(e):** unquoted POSIX drop paths unescape any `\X` → `X`.
- **R40(f):** Import prompt title "Import theme — full path to a .toml file".
- **R40(g):** `test_theme_file_dark_flag_coerces_strings` moved to
  `Tests/Utils/test_user_theme_loader.py`.
