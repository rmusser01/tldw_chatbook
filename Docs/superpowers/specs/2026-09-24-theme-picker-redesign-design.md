# Settings ▸ Theme: picker-first redesign — design

- **Task:** TASK-32948
- **Date:** 2026-09-24
- **Status:** Awaiting user review
- **Base:** origin/dev 6ddc582839, plus the fix wave on `fix/theme-ux-wave` (TASK-32940..32947). That wave must merge first.
- **Evidence:** `.impeccable/critique/2026-09-24T20-44-51Z__tldw-chatbook-widgets-settings-theme-editor-py.md` (18/40)

## 1. Why

Users say Settings ▸ Theme got harder to use over the past two months. The critique found two causes.

**Clipping, now fixed.** From 2026-07-19 to 2026-09-17 the editor was clipped inside the scrolling detail pane. The 2026-09-04 fix wave pushed the palette, the presets and the preview into the unreachable part. 65cc565d84 fixed this.

**The screen's model, still there.** Theme is an editor that happens to contain a theme list:

- Switching theme takes three actions: select loads it into the editor, Apply sets it for this session, and "Set as launch default" keeps it.
- Themes appear as raw slugs with no way to see their colours.
- Three surfaces disagree about which theme is in use: Appearance's 91-item dropdown, the editor tree, and the palette's "Theme: Switch to".

Most visits want to *use* a theme; few want to *edit* one. This redesign makes Theme a picker and moves the editor behind Clone, New and Edit.

## 2. Success criteria

- Switching to a theme and keeping it takes **one action**. From the filter that is the name, Enter to reach the list, Enter to use.
- Every theme can be judged before it is chosen: its row shows a colour strip, and the preview card shows the highlighted theme.
- The picker, Appearance and the palette always agree on which theme is **active** and which is the **launch default**. Both markers are shown as words.
- At 80×24 and 190×55, the filter, the list (at least 5 rows at 80×24), the preview and every action can be reached. A geometry test pins this.
- Nothing is lost:
  - Edits are guarded by the leave prompt from TASK-32941.
  - Shipped themes keep their extra `variables` (TASK-32940).
  - Every file operation goes through the backup-recovery layer.

## 3. Decisions (made with the user, 2026-09-24)

| # | Decision |
|---|---|
| D1 | Picker first. The editor is reached only through Clone, New and Edit. |
| D2 | Moving the highlight repaints a **preview card only**. The app keeps its theme until Use or Try, which preserves TASK-31255. |
| D3 | The editor **swaps into the same pane** (a `ContentSwitcher`), not a modal and not inline. |
| D4 | **Use** applies the theme and saves it as the launch default. **Try** applies it for this session only and shows a **Revert** chip. |
| D5 | Appearance shows a **read-only theme row** and an **Open Theme** button. The dropdown is removed. |
| D6 | Delivered as **3 PRs**, each leaving a working screen (§10). |

## 4. Architecture

Approach A. New units, each with one job:

| Unit | File | Job | Depends on |
|---|---|---|---|
| `theme_catalog` | `tldw_chatbook/css/Themes/theme_catalog.py` | Pure function `build_catalog(app_themes, user_dir, active, launch_default) -> list[ThemeEntry]`. `ThemeEntry` is a frozen dataclass with `id`, `display_name`, `origin` (`"yours"`, `"shipped"` or `"textual"`), `dark`, `strip` (7 hex colours), `is_active`, `is_launch_default`, `overrides_shipped`, and `error` (set for an unreadable file). Also provides `is_catalog_theme(name)`, replacing the scattered built-in checks in the editor. | `ALL_THEMES`, `textual.theme.BUILTIN_THEMES`, `load_user_themes` |
| `use_theme` | same module | `use_theme(app, name, *, persist: bool) -> ThemeChange`. Sets `app.theme`. When `persist` is set it writes `general.default_theme` to disk **and** updates the in-memory `app.app_config` dict, which the Settings screen's `_app_config_update_target()` also returns, which is what `handle_theme_launch_default_changed` (`settings_screen.py:24102`) does today. The palette's `switch_theme` currently writes only to disk. Returns a `ThemeChange(previous_active, previous_launch_default)` so Revert can restore **both**. Reverting a Use that followed a Try must not save the tried theme as the launch default. | config |
| `ThemePicker` | `tldw_chatbook/Widgets/settings_theme_picker.py` | Filter `Input`, grouped `OptionList`, preview card and action chips. Posts `EditRequested(name, mode)`. | catalog, `use_theme` |
| `ThemePane` | same file | `ContentSwitcher` holding `ThemePicker` and `SettingsThemeEditor`. Routes Clone, New and Edit to the editor, and Back and Save to the picker. | both |
| `SettingsThemeEditor` | existing file | Keeps the palette, presets, live preview and Generate. **Loses** the tree, the library buttons and "Set as launch default". **Gains** Save as and "Back to themes". | catalog |

Changes elsewhere:

- **Palette.** `ThemeProvider.switch_theme` (`app.py:1214`) calls `use_theme(persist=True)`, so the palette and the picker share one code path and one toast.
- **Settings screen.** Mounts `ThemePane` for the Theme category. Replaces the Appearance dropdown with the read-only row. Drops the pinned "State: Managed in editor" banner for Theme.

## 5. Picker

**Layout, wide** (detail pane at least 100 columns): two columns. On the left, the filter, the grouped list, then `[ New ]` and `[ Import… ]`. On the right, the preview card:

- a header with the display name and "light · shipped" (for example);
- a colour strip;
- a Console-shaped preview stub, the renderer extracted from the editor's preview and parameterised by colours;
- the action chips.

**Layout, narrow:** one column. The filter, then the list filling the remaining height with a fixed minimum and maximum, then a two-row preview (strip plus one sample line of text and button), then the chips.

**Rows.** Each row reads `› Apricot  ▮▮▮▮▮▮▮  active · launch`.

- The `›` cursor glyph and the marker words carry the state. The colour strip is decorative.
- The strip is drawn as styled segments inside the row's text, not one widget per swatch.
- The groups are YOUR THEMES, SHIPPED and TEXTUAL, as disabled `OptionList` headers. While filtering, each header shows a count.
- A file in your themes folder with the same name as a shipped or Textual theme is listed under Yours, marked "overrides shipped" or "overrides Textual". This matches startup, where user themes register after `ALL_THEMES` (`app.py:15873`). The theme it overrides is not listed separately.
- Display names are de-duplicated in the catalog, the single owner of the " · id" suffix from TASK-32945. The picker and Appearance both read that result.

**Filter.** Case-insensitive substring match on the display name and the id. The list starts with the active theme highlighted.

**Keys** (shown in the footer):

| Key | Where | Action |
|---|---|---|
| typing | filter | narrows the list |
| ↓ or Enter | filter | moves into the list |
| ↑ | first list row | moves back to the filter. `OptionList` wraps around at the ends (Textual 8.2.8 `find_next_enabled`; verified), so the picker handles ↑ on the first enabled row before the list does. |
| Enter | list | **Use** |
| `t` | list | **Try** |
| `c` | list | Clone |
| `e` | list, your themes only | Edit |
| `n` | list | New |
| `r` | list, your themes only | Rename |
| Delete | list, your themes only | Delete (existing confirmation) |
| F6 / Shift+F6 | anywhere | next / previous pane (TASK-32943) |

`/` is **not** rebound. It stays as Settings search (`settings_screen.py:31859`).

**Use.** Calls `use_theme(persist=True)`. The toast reads "Apricot is now your theme (was: Nord)". The picker shows a `[ Revert to Nord ]` chip for the rest of the session; Revert restores the previous active theme and the previous launch default from the `ThemeChange`.

**Try.** Calls `use_theme(persist=False)` and shows the same Revert chip. It does not change the launch default, and its toast says "for this session". The Revert chip exists only in the picker. If you leave Settings, the tried theme stays until you relaunch.

**Markers stay live.** The picker subscribes to `app.theme_changed_signal` (it exists in Textual 8.2.8), so a change from the palette, Appearance or anywhere else updates the markers while the picker is open.

## 6. Editor (behind Clone, New and Edit)

**Opening it.** The header reads "Editing warm_paper · copy of Apricot", or "· new", or "· saved theme".

**Actions:** `[ Save ] [ Save as… ] [ Try ] [ Back to themes ]`, then Generate from Primary with the palette tools.

**Save.** Writes the file through the backup layer, as today, and returns to the picker with the saved theme highlighted. It does **not** apply the theme, unless the theme being edited is the active one; then it re-applies so the app never shows a stale version of it.

**Save as.** Asks for a name (checked with `validate_filename`), writes a new file, and leaves the original untouched.

**Back to themes, Esc, or switching category with unsaved edits.** Shows the Stay / Discard / Save prompt from TASK-32941. A category switch rebuilds the detail pane (`settings_screen.py:3729`), so returning to Theme always opens the picker, never the editor.

**Carried colours.** TASK-32940 carries a shipped theme's extra `variables` into clones. The fix wave on `fix/theme-ux-wave` settles the rule: they are carried only while the 10 base colours and the dark flag still equal what was loaded. Once the palette diverges, none are carried and Textual derives them. This avoids contrast arithmetic, and it was measured: a light-converted Dracula clone kept `text-muted` at 1.80:1 under the old rule. The editor inherits this rule unchanged.

## 7. Actions on your own themes

All of these go through `raw._scope` and the backup participant, as Save and Delete already do.

**Rename.**

1. Ask for the new name and check it with `validate_filename`. If it is taken, show "Name taken" and change nothing.
2. Rename the file.
3. `app.unregister_theme(old)` (present in Textual 8.2.8), then register the new name.
4. If the theme was active, set `app.theme` to the new name. If it was the launch default, rewrite the config.
5. If the old name overrode a shipped theme, that shipped theme appears again.

**Delete of the active theme.** The theme is unregistered (`app.unregister_theme`), then the app switches to the launch default. If the deleted theme *was* the launch default, the app switches to `textual-dark` and the config is rewritten. The toast states what happened. This fixes the editor/app disagreement the critique found.

**Import** (PR 3).

- A dialog accepts a typed or pasted path. A terminal file-drop arrives as a pasted path (TASK-216).
- Checks: `path_validation`, `.toml` only, at most 64 KB, parsed with `toml`, hex colours validated. `[variables]` goes through the same validator `load_user_themes` uses since the fix wave: names must match `^[a-z0-9-]+$` and values must be colours, `auto NN%` or text-style keywords. Anything else is dropped with a warning. Nothing in the file is executed.
- On success the file is copied into your themes folder (asking before overwriting) and registered.
- On failure the message names the problem, for example "missing [colors].primary" or "background: 'blue' is not #RRGGBB".

**Export.** Unchanged, except the toast shows the full path and adds a `[ Copy path ]` chip (PR 3).

## 8. Appearance row

A read-only line: `Theme · Apricot (launch default) · active: Apricot` followed by `[ Open Theme ]`, which opens the Theme category with the launch default highlighted.

This removes `#settings-appearance-theme`, its entry in the Appearance draft (`default_theme` in the values, originals and dirty keys), and the draft-rebase half of `handle_theme_launch_default_changed`. `default_theme` is dropped from the Appearance draft keys and validation list (`settings_screen.py:9444`), so Appearance's Save can never write a stale theme. Its in-memory config update moves into `use_theme`, so every caller gets it.

Settings search keeps a "theme" entry, and it now lands on the picker.

## 9. Errors and edge states

| State | Behaviour |
|---|---|
| Backup or recovery pause | A "Theme files unavailable while backup/recovery is in progress" row (TASK-32942). Rename, Delete, Import, Save and Save as are disabled, with that reason as their tooltip. Use and Try still work on themes that are already registered. |
| Unreadable user file | Listed under Yours with "(unreadable)". The preview card shows the parse error. Use, Try and Edit are disabled; Delete is allowed. |
| Launch default missing | Appearance and the picker both say "launch default missing: foo". Use on any theme fixes it. |
| Empty "Your themes" | A "(none yet)" row (TASK-32945). |
| Filter matches nothing | "No themes match 'xyz'", with a `[ Clear filter ]` chip. |

## 10. Delivery

**PR 1: the picker.**

- `theme_catalog`, `use_theme`, and the palette's `switch_theme` routed through it.
- `ThemePicker` with the list, filter, preview card, Use, Try, Revert and live markers.
- `ThemePane` with the editor reached through Clone and New. The editor is unchanged except that "Set as launch default" is removed.
- The Appearance read-only row.
- The pinned banner dropped for Theme.
- The geometry test at 80×24 and 190×55.

**PR 2: editor and file actions.**

- The tree and library buttons removed from the editor.
- Save as, Back to themes, Edit, and Rename.
- The Delete fallback for the active or launch-default theme.

**PR 3: import and edge states.**

- Import.
- The unreadable-file and launch-default-missing states.
- The Export "Copy path" chip.
- A User Guide rewrite of `Docs/User_Guide/settings.md` §Theme.

Each PR updates the User Guide stamp.

## 11. Testing

**Unit tests** (no app needed): `build_catalog` covering the origins, the override marker, markers, strip colours and unreadable files; `is_catalog_theme`; `use_theme` with and without persist, including the `ThemeChange` return value; Rename and Import validation.

**Pilot tests** using `run_test` with production CSS and `@private_profile_test`:

- Use and Try change `app.theme` and the config as specified.
- Revert works.
- The markers update after a palette switch.
- Filter plus Enter plus Enter uses the theme.
- ↑ from the first row returns to the filter; `/` still reaches Settings search.
- Clone opens the editor and Back returns to the picker.
- The leave prompt appears.
- Deleting the active theme falls back correctly.
- Rename moves the launch default.

**Geometry.** At 80×24 and 190×55, under textual-dark and textual-light, every picker control has a non-empty visible region, and the list shows at least 5 rows at 80×24. This reuses the 2026-09-17 geometry harness.

**Contrast.** The TASK-32947 contrast test is extended to the picker's chips and the row cursor.

**Retired or rewritten tests** are listed in each PR's plan with the reason: the editor-tree and library-button tests, the Appearance dropdown and draft tests, and the `LaunchDefaultChanged` rebase tests. Each is replaced by a test of the new owner.

**Performance.** A catalog build for 91 themes costs about 17 ms (measured: 91 × `to_color_system().generate()`). It is built on mount and rebuilt after Save, Rename, Delete or Import, never on highlight.

## 12. Out of scope

- Whole-app live preview while browsing (D2).
- Recomputing carried variables for a recoloured clone. They are dropped and Textual derives them (§6).
- An app-wide fix for Textual's filled-button label contrast. TASK-32947 found this is app-wide and scoped its fix to the Theme card.
- Rebinding Settings' `/`.
