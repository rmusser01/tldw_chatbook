# Theme Picker (PR 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the editor side of the Settings ▸ Theme redesign. The editor loses its own theme tree and library buttons. Your themes get Edit, Rename, Delete and Export from the picker. The editor gains Save as. Save returns to the picker. Deleting the active or launch-default theme falls back cleanly. The backup-recovery pause is shown in the picker.

**Architecture:** The picker owns choosing a theme and the per-theme actions. The editor owns palette editing and **every theme-file operation**. The backup-recovery layer (`Backup_Recovery/settings_file_participants.py:49`) only recognises a `SettingsThemeEditor` instance as the source for `theme_directory`, `theme_file` and `theme_export` scopes. So the picker never touches files. It posts messages; `ThemePane` routes them to the editor's new public file API. Name prompts reuse `RagProfileNameModal`. That modal lives in the screen module, so it is pushed by `SettingsScreen` handlers; widgets must not import the screen.

**Tech Stack:** Python 3.12, Textual 8.2.8, pytest + pytest-asyncio.

**Spec:** `Docs/superpowers/specs/2026-09-24-theme-picker-redesign-design.md`, §6, §7, §9 (backup pause row) and §10 PR 2. Read it first. PR 1 is on `feat/theme-picker-pr1` (PR #2832). Its task file, `backlog/tasks/task-32948 - Settings-Theme-picker-first-redesign.md`, records rulings R1–R9. R3 ("Save returns to the picker in PR 2") and R9 bind here.

## Global Constraints

- **Branch and worktree:** `feat/theme-picker-pr2` off `feat/theme-picker-pr1`, in worktree `.claude/worktrees/theme-picker-pr2`. Prefix every shell command with `cd <worktree> &&`. Never `git stash`. Never push. Never open PRs.
- **Python:** `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` with `PYTHONPATH=<worktree>`. Assert that `tldw_chatbook.__file__` is under the worktree.
- **HARD RULE: never run code that writes the user's real config or themes.** That covers `use_theme(persist=True)`, `_apply_config_mutation`, `apply_settings_mutation_to_cli_config`, `save_setting_to_cli_config`, and any `TldwCli`/editor file operation outside `@private_profile_test`. It writes `~/.config/tldw_cli/`, and this happened once in PR 1. Ad-hoc probes set `HOME`, `TLDW_CONFIG_PATH` and the XDG dirs to a scratch directory first.
- **Tests:**
  - Pilot tests that build `TldwCli`/Settings, or touch theme files, use `@private_profile_test` and a `request` argument. Their `tmp_path` fixture in `Tests/UI/test_settings_theme_editor.py` resolves to the private profile's `themes/` directory.
  - Hub tests without that decorator fail locally with `RecoveryRequired`. They are CI-only; compare their FAILED sets branch vs base.
- **CSS:** ADR-161 tokens only. Never end a selector with a bare `Button`. Run `./build_css.sh`, then commit the regenerated bundle.
- **Commits:** a `Co-Authored-By: <the model you are> <noreply@anthropic.com>` trailer.
- **Copy:** use the words the spec uses. Marker and state text is always words, never colour alone.
- **Before the PR:** `PYTHON=<venv> ./scripts/preflight.sh` with the exit code checked directly. Run `ruff` on the touched files and confirm no new findings.

## Review Focus

1. **Delete the theme that is both active and the launch default.** The app moves to `textual-dark`, the launch default becomes `textual-dark`, the markers update, and the toast says so. Pinned in Task 1 (`test_delete_active_launch_default_falls_back_to_textual_dark`).
2. **Rename onto a name that is already taken.** "Name taken"; both files and the registrations are untouched. Pinned in Task 1 (`test_rename_to_taken_name_changes_nothing`).
3. **Rename or delete a user file that overrides a shipped or Textual theme.** The overridden catalog theme comes back registered. Pinned in Task 1 (`test_rename_override_restores_catalog_theme`).
4. **Backup/recovery pause.** No crash. The picker shows the unavailable row, and Rename, Delete, Export, Save and Save as are disabled with that reason as a tooltip. Use and Try still work. Pinned in Task 2 (`test_pause_row_disables_file_actions`) and Task 1 (`test_list_user_themes_raises_through_on_pause`).
5. **Save while editing the active theme.** The app re-applies the saved palette, so it is never stale, and you return to the picker with it highlighted. Pinned in Task 4 (`test_save_active_theme_reapplies_and_returns`).

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `tldw_chatbook/Widgets/settings_theme_editor.py` | modify | Public file API: `list_user_theme_names`, `request_delete`, `rename_user_theme`, `export_theme`, `save_as`. Remove the tree and library UI. Add the header and Save as. Save posts `Saved`. |
| `tldw_chatbook/Widgets/settings_theme_picker.py` | modify | Actions for your themes (Edit/Rename/Delete/Export). An injectable user-name lister. The pause row. `ThemePane` routes messages to the editor. |
| `tldw_chatbook/css/Themes/theme_catalog.py` | modify | `use_theme` returns the new launch default in `ThemeChange`. (Minor: see Task 5.) |
| `tldw_chatbook/UI/Screens/settings_screen.py` | modify | Handlers that push `RagProfileNameModal` for Rename and Save as. Open Theme highlights the launch default. |
| `tldw_chatbook/app.py` | modify | The palette toast matches the picker's. |
| `tldw_chatbook/css/components/_settings_splash_theme.tcss` | modify | Drop the tree rules. Style the header. |
| Tests | modify/create | See each task. |

---

### Task 1: Editor file API (no UI changes)

**Files:** modify `tldw_chatbook/Widgets/settings_theme_editor.py`; create `Tests/UI/test_settings_theme_file_api.py`.

**Interfaces produced** (later tasks use them verbatim):
- `SettingsThemeEditor.list_user_theme_names() -> set[str]`
  - Returns the `[theme].name` of each readable `*.toml` in the themes directory, falling back to the file stem.
  - It reads through `raw._scope(self, "theme_directory")`.
  - It **raises** `RecoveryRequired` when paused; the caller decides how to show that.
- `SettingsThemeEditor.request_delete(name: str) -> None`: shows the existing confirmation dialog, then `_delete_user_theme`.
- `SettingsThemeEditor.rename_user_theme(old: str, new: str) -> bool`.
- `SettingsThemeEditor.export_theme(name: str) -> None`: exports that saved theme's file data, not the editor's palette.
- `SettingsThemeEditor.ThemesChanged(Message)` has no fields. It is posted after any successful delete, rename, save or save-as, so the picker can rebuild.

- [ ] **Step 1: Write the failing tests** in `Tests/UI/test_settings_theme_file_api.py`.
  - Mount the editor in the isolated app, as `Tests/UI/test_settings_theme_editor.py`'s `_isolated_editor_app` and `_isolated_editor_app_with_real_screens` do. Copy the `tmp_path` fixture from that file so `tmp_path` is the private profile's `themes/` directory, and set `editor.custom_themes_path = tmp_path`.
  - Register `ALL_THEMES`.
  - Write theme files with `toml.dump({"theme": {"name": n, "dark": True}, "colors": {...10 base keys...}}, ...)`, then `app.register_theme(create_theme_from_dict(...))`.
  - Mock `tldw_chatbook.css.Themes.theme_catalog._apply_config_mutation` so it records calls and returns `SimpleNamespace(file_replaced=True, caches_reloaded=True)`. Mock `theme_catalog.current_launch_default` as well.

  Tests, each `@pytest.mark.asyncio @private_profile_test`:
  - `test_list_user_theme_names_reads_theme_name_not_stem`: a file `a.toml` whose `[theme] name = "b"` → `{"b"}`.
  - `test_list_user_themes_raises_through_on_pause`: monkeypatch `raw._scope` to raise `RecoveryRequired("x")` → `pytest.raises(RecoveryRequired)`.
  - `test_delete_active_launch_default_falls_back_to_textual_dark`: register and apply `mine`, launch default `mine`, then `request_delete("mine")` and confirm the dialog (click `Delete theme`). Assert:
    - the file is gone and `"mine" not in app.available_themes`;
    - `app.theme == "textual-dark"`;
    - the last config write is `{"general": {"default_theme": "textual-dark"}}`;
    - a `ThemesChanged` was posted (capture it with an `@on` handler on an app subclass).
  - `test_delete_active_non_default_switches_to_launch_default`: active `mine`, launch default `nord` → after the delete, `app.theme == "nord"` and no config write.
  - `test_rename_moves_file_registration_active_and_launch_default`: `mine` is active and the launch default; `rename_user_theme("mine", "ours")` returns True. Assert:
    - `ours.toml` exists, `mine.toml` doesn't, and the file's `[theme].name == "ours"`;
    - `"ours" in available_themes` and `"mine"` is not;
    - `app.theme == "ours"`;
    - the last config write is `{"general": {"default_theme": "ours"}}`.
  - `test_rename_to_taken_name_changes_nothing`: `mine` and `ours` both exist → returns False, a "Name taken" notify, both files intact.
  - `test_rename_rejects_invalid_name`: `"../evil"` → False, a notify, nothing changes.
  - `test_rename_override_restores_catalog_theme`: a user file named `apricot` (a shipped name) renamed to `apricot_mine` → `available_themes["apricot"]` is the shipped `ALL_THEMES` object again.
  - `test_export_theme_writes_saved_file_data`: export `mine` while the editor has a different palette loaded. The file in `Path.home()/"Downloads"` has `mine`'s colours. Monkeypatch `Path.home` to `tmp_path`.

- [ ] **Step 2: Run the tests.** Expected: FAIL, because the attributes are missing.

- [ ] **Step 3: Implement.** Refactor the existing internals; don't duplicate them:
  - `list_user_theme_names`: move the body of `_load_user_themes`' scan into it. Keep the per-file `try` that logs and skips unreadable files. Let `RecoveryRequired` escape. `_load_user_themes` calls it until Task 4 deletes the tree.
  - `request_delete(name)`: the body of `on_delete_theme`, parameterised by `name` instead of `self.current_theme_name`. Keep the built-in/shipped/no-file guards and the confirmation copy. `on_delete_theme` becomes `self.request_delete(self.current_theme_name)` after `_require_theme_name()`.
  - `_delete_user_theme(path, name)`. Keep the unlink and the catalog re-register. Replace the launch-default block and the `self.load_theme("textual-dark")` tail with this fallback, which uses the existing `theme_catalog` API:
    ```python
    from ..css.Themes.theme_catalog import current_launch_default, use_theme
    launch = current_launch_default()
    was_active = str(self.app.theme) in (name, f"custom_{name}")
    if launch == name:
        use_theme(self.app, "textual-dark", persist=True)
        self.post_message(self.LaunchDefaultChanged("textual-dark"))
        self.app.notify(f"Deleted '{name}'; launch default and theme reset to Textual Dark", severity="success")
    elif was_active:
        use_theme(self.app, launch if launch in self.app.available_themes else "textual-dark", persist=False)
        self.app.notify(f"Deleted '{name}'; switched to your launch default", severity="success")
    else:
        self.app.notify(f"Deleted theme '{name}'", severity="success")
    self.post_message(self.ThemesChanged())
    ```
    Keep `LaunchDefaultChanged`. The Settings handler still updates Appearance's in-memory config; see PR 1's Task 6 review. Delete the tree-node removal lines here; the picker rebuilds on `ThemesChanged`.
  - `rename_user_theme(old, new)`:
    1. `validate_filename(new)`; notify "Invalid theme name: …" and return False on `ValueError`.
    2. If `new == old`, return True.
    3. If `new` is a built-in or shipped name, or `(custom_themes_path / f"{new}.toml").exists()`, or `new in self.app.available_themes`, notify `f"Name taken: '{new}'"` and return False.
    4. Read `old.toml` through `raw._scope(self, "theme_file", selected_read=old_path)` and `raw._file(…, "r")`. Set `data["theme"]["name"] = new`, then write `new.toml` with the same temp-then-`raw._replace` pattern `_write_theme_file` uses, under `selected_read=new_path`.
    5. **Only after the new file exists**, unlink `old.toml` in its own `theme_file` scope. If the unlink fails, notify that both files now exist and return True; nothing is lost.
    6. Registrations: register `create_theme_from_dict(new, …)` built from the file data. Use the same helper the startup loader uses (`load_user_themes` in `css/Themes/themes.py`; read how it builds a `Theme` and call that code, don't copy it). Then restore the overridden catalog theme for `old`, or `unregister_theme(old)`, using the same expression `_delete_user_theme` uses.
    7. If `str(app.theme)` was `old` or `custom_{old}`, call `use_theme(app, new, persist=False)`. If `current_launch_default() == old`, call `use_theme(app, new, persist=True)` and post `LaunchDefaultChanged(new)`.
    8. Post `ThemesChanged` and notify `f"Renamed '{old}' to '{new}'"`. Return True.
    9. Wrap the whole method in `try/except RecoveryRequired`: notify "Theme files unavailable while backup/recovery is in progress" and return False.
  - `export_theme(name)`: the body of `on_export_theme`, parameterised. Its data is the saved file's contents, read through the `theme_file` scope, not `_theme_file_data(current palette)`. Keep the overwrite confirmation. `on_export_theme` calls `self.export_theme(self.current_theme_name)` for now.
  - `_write_theme_file`: post `ThemesChanged` after a successful write.

- [ ] **Step 4: Run** the new file and `Tests/UI/test_settings_theme_editor.py`, plus `Tests/Backup_Recovery/test_settings_file_participant_lifetimes.py`. All pass. Existing editor tests must stay green; the tree still exists in this task.
- [ ] **Step 5: Commit** with `feat(theme-editor): public file API — list, delete with fallback, rename, export by name (TASK-32948)`.

---

### Task 2: Picker actions for your themes, and the pause row

**Files:** modify `tldw_chatbook/Widgets/settings_theme_picker.py`, `Tests/UI/test_settings_theme_picker.py`.

**Interfaces:**
- **Consumes:** nothing from Task 1 directly. The picker never touches files.
- **Produces:**
  - `ThemePicker(list_user_names: Callable[[], set[str]] | None = None, **kwargs)`. When it is `None`, the picker uses `theme_catalog.user_theme_names(get_user_themes_dir())`, which is the PR 1 behaviour and is kept for isolated tests.
  - Messages with the fields shown:
    - `ThemePicker.RenameRequested(theme_id)`
    - `ThemePicker.DeleteRequested(theme_id)`
    - `ThemePicker.ExportRequested(theme_id)`
    - `ThemePicker.EditRequested(theme_id, mode)`, where `mode` now also accepts `"edit"`.
  - Buttons, which have `display` true only when the highlighted entry has `origin == "yours"`:
    - `#settings-theme-picker-edit` ("Edit")
    - `#settings-theme-picker-rename` ("Rename")
    - `#settings-theme-picker-delete` ("Delete", variant error)
    - `#settings-theme-picker-export` ("Export")
  - Keys on `ThemeOptionList`: `e` Edit, `r` Rename, `delete` Delete. Each is ignored unless the highlighted entry is yours; the three Export/Rename/Delete keys and buttons are also ignored while paused.
  - Pause state: `ThemePicker.files_available: bool`.
    - When `list_user_names` raises `RecoveryRequired`, `files_available` is False and a disabled option `THEMES_UNAVAILABLE_LABEL` appears under YOUR THEMES. Import it from the editor module (`settings_theme_editor.THEMES_UNAVAILABLE_LABEL`, which already exists). `ThemePane` imports the editor lazily inside methods, so check for an import cycle; if one appears, import the constant lazily as well. `RecoveryRequired` comes from `tldw_chatbook.Backup_Recovery.bootstrap`.
    - Rename, Delete, Export and Edit are disabled, with `tooltip = "Theme files unavailable while backup/recovery is in progress"`. Use and Try stay enabled.

- [ ] **Step 1: Failing tests.** Append them; imports go at the top of the file. They use the file's `_app`, `config_writes` and `_picker_app` helpers, adding a variant that passes `list_user_names`.
  - `test_yours_actions_only_show_for_your_themes`: `list_user_names=lambda: {"mine"}`, with `mine` registered. Highlight `mine` → the four buttons display. Highlight `nord` → they are hidden.
  - `test_rename_delete_export_edit_post_requests`: on `mine`, press `r`, `delete`, `e`, then click Export → the app captures `RenameRequested("mine")`, `DeleteRequested("mine")`, `EditRequested("mine","edit")`, `ExportRequested("mine")` (reuse PR 1's `_CaptureEditApp` pattern, extended).
  - `test_keys_ignored_for_catalog_themes`: on `nord`, `r`/`delete`/`e` post nothing.
  - `test_pause_row_disables_file_actions`: `list_user_names` raises `RecoveryRequired("x")` → the unavailable label is present, `files_available is False`, the file-action buttons are disabled with the tooltip, and pressing Enter on a shipped theme still Uses it.
- [ ] **Step 2: Run the tests.** They fail.
- [ ] **Step 3: Implement.**
  - `refresh_catalog` calls `self._list_user_names()` inside `try/except RecoveryRequired`. On a pause it uses an empty set and sets `files_available = False`.
  - `_render_list` inserts the disabled unavailable option under YOUR THEMES when `not files_available`.
  - `_show` sets the display and disabled state of the four buttons from the entry's origin and `files_available`.
  - Action methods post the messages.
- [ ] **Step 4: Run** the picker tests. They pass. Then run `Tests/Architecture/test_no_blocking_io_on_message_pump.py`.
- [ ] **Step 5: Commit** with `feat(theme-picker): Edit/Rename/Delete/Export for your themes; backup-pause row (TASK-32948)`.

---

### Task 3: Pane and screen wiring

**Files:** modify `settings_theme_picker.py` (`ThemePane`) and `settings_screen.py`; tests in `Tests/UI/test_settings_theme_picker_screen.py`.

**Interfaces:**
- **Consumes:** the Task 1 API and the Task 2 messages.
- **Produces:**
  - `ThemePane` constructs `ThemePicker(list_user_names=self._editor_names, id="settings-theme-picker")`, where `_editor_names` returns `self.query_one(SettingsThemeEditor).list_user_theme_names()`.
    - Composition order matters: the editor must exist before the picker's first `refresh_catalog`. Compose the editor first, keep `initial="settings-theme-picker"`, and make the picker's `on_mount` refresh deferred with `call_after_refresh` if needed. Verify it with a test.
  - `ThemePane` handlers:
    - `DeleteRequested` → `editor.request_delete(id)`.
    - `ExportRequested` → `editor.export_theme(id)`.
    - `EditRequested(mode="edit")` → `editor.load_user_theme(id)`, then show the editor. There is no clone.
    - `RenameRequested` is **not** handled in the pane; let it bubble to the screen.
    - `SettingsThemeEditor.ThemesChanged` → `picker.refresh_catalog()`.
  - `SettingsScreen` handles `ThemePicker.RenameRequested` by pushing `RagProfileNameModal(title=f"Rename theme '{id}'", initial=id, confirm_label="Rename")`. Its callback calls `editor.rename_user_theme(id, new)` when `new` is set and differs from `id`, then `picker.refresh_catalog(highlight=new)`.
- [ ] **Step 1: Failing tests.** Real screen, `_host()`, `@private_profile_test`, with the theme file written into the private profile `themes/` dir before mount.
  - `test_rename_from_picker_prompts_and_renames`: highlight `mine`, press `r`, type `ours` in `#settings-rag-profile-name-input`, click `#settings-rag-profile-name-confirm` → `ours` is highlighted in the list and `mine` is absent.
  - `test_delete_from_picker_confirms_and_rebuilds`: highlight `mine`, press `delete`, click "Delete theme" → `mine` is gone from the picker entries.
  - `test_edit_opens_saved_theme_in_editor`: highlight `mine`, press `e` → the pane shows the editor view and the editor's name box reads `mine`, with no `_copy` suffix.
  - `test_picker_lists_via_editor_scope`: monkeypatch the editor's `list_user_theme_names` to raise `RecoveryRequired` → the picker shows the unavailable row.
- [ ] **Step 2 to Step 4:** run the tests (red), implement, run them (green). Also run `Tests/UI/test_settings_theme_picker.py` and the Theme hub tests marked private-profile.
- [ ] **Step 5: Commit** with `feat(settings/theme): route picker file actions through the editor's backup-scoped API (TASK-32948)`.

---

### Task 4: Editor rework: Save as, Save back to the picker, header; tree removed

**Files:** modify `settings_theme_editor.py`, `settings_theme_picker.py` (`ThemePane`), `settings_screen.py` and `_settings_splash_theme.tcss`; tests in `Tests/UI/test_settings_theme_editor.py`, `Tests/UI/test_settings_theme_picker_screen.py`, `Tests/Backup_Recovery/test_settings_file_participant_lifetimes.py`, `Tests/UI/test_css_build_integrity.py` and the hub.

**Interfaces:**
- **Consumes:** Task 1 and Task 3.
- **Produces:**
  - `SettingsThemeEditor.Saved(theme_name: str)` message, posted after a successful Save or Save as.
  - `SettingsThemeEditor.SaveAsRequested(current_name: str)` message, posted by the new `#settings-theme-save-as` button ("Save as…").
  - `SettingsThemeEditor.save_as(name: str) -> None`. It writes a new file under `name`, with the overwrite confirmation if `name.toml` exists, and leaves the source file untouched.
  - `#settings-theme-editor-header` `Static`. Its text is `f"Editing {name} · copy of {display_name(source)}"`, `"· new"`, or `"· saved theme"`. It is set by `ThemePane.open_editor` through a new `editor.set_editing_context(source: str, mode)`.
  - `ThemePane` handles `Saved(name)` (guard: return early if the pane is not mounted or is being removed — the category-leave Save path in `SettingsScreen._confirm_theme_category_leave` calls `on_save_theme()` and then tears the pane down; a `QueryError` there must be swallowed, not crash) → `show_picker()`, then `picker.refresh_catalog(highlight=name)`. If `str(app.theme)` is `name` or `custom_{name}`, it first calls `use_theme(app, name, persist=False)` so the app shows the saved palette.
  - `SettingsScreen` handles `SaveAsRequested` → `RagProfileNameModal(title="Save theme as", initial=f"{current}_copy", confirm_label="Save")`. Its callback calls `editor.save_as(new)`.

- [ ] **Step 1: Failing tests.**
  - `test_category_leave_save_does_not_crash_on_saved_message` (screen): edit, switch category, choose Save in `ThemeLeaveModal` → the category switches, the file is saved, and there is no exception (the `Saved` handler runs while the pane is being torn down).
  - `test_save_active_theme_reapplies_and_returns` (screen): Edit `mine` while it is active, change the primary colour, Save → the pane shows the picker, `mine` is highlighted, and `app.available_themes[app.theme].primary` equals the new colour.
  - `test_save_as_keeps_original` (screen): Edit `mine`, then Save as `mine2` via the modal → both files exist, the picker highlights `mine2`, and `mine.toml` is unchanged byte for byte.
  - `test_editor_has_no_tree_or_library_buttons`: none of `#settings-theme-tree`, `#settings-theme-new`, `#settings-theme-clone`, `#settings-theme-delete` or `#settings-theme-export` exists.
  - `test_editor_header_names_the_source`: Clone `apricot` → the header reads `Editing apricot_copy · copy of Apricot`.
  - `test_clone_then_back_without_edits_does_not_prompt`: Clone, then Back immediately → no `ThemeLeaveModal`. Clone/New now start clean. Only the first real edit sets `is_modified`, so Back without edits doesn't prompt. This is a PR 1 final-review minor.
- [ ] **Step 2: Run the tests.** They fail.
- [ ] **Step 3: Implement.**
  - **Delete from the editor:** `_compose_library_section`'s tree, its hint, and the New/Clone/Delete/Export button row. Also delete `_load_user_themes`, `_sync_user_placeholder`, `on_theme_selected`, `_populate_theme_tree` and `on_show`, plus the `on_new_theme`/`on_clone_theme`/`on_delete_theme`/`on_export_theme` `@on(Button.Pressed…)` decorators. `on_new_theme` and `on_clone_theme` stay as plain methods, because `ThemePane.open_editor` calls them.
  - **Keep:** the Name input and the Dark checkbox, under a section titled "Theme", plus the palette, presets, actions and preview.
  - **Actions row:** Try (relabel the existing `#settings-theme-apply` to "Try"; the id is unchanged), Save, Save as…, Reset, Generate from Primary.
  - **After a successful write:** `on_save_theme` / `_write_theme_file` post `Saved(theme_name)`.
  - **Clone/New:** set `is_modified = False`. The first real edit marks it modified. Keep the existing `init=False` reactive and don't post on mount.
  - **CSS:** remove the `#settings-theme-tree*` rules and add a header style. `Tests/UI/test_css_build_integrity.py` pins `#settings-theme-tree`: replace that pin with `#settings-theme-editor-header`.
- [ ] **Step 4: Retire or rewrite the tests that pin the removed UI.** Run `grep -rln "settings-theme-tree\|_populate_theme_tree\|on_theme_selected\|settings-theme-new\"\|settings-theme-clone\"\|settings-theme-delete\|settings-theme-export\|_user_theme_labels\|none yet" Tests`.
  - Tests of the tree's contents are covered by the catalog and picker tests: the built-ins list, "(none yet)", the placeholder sync, and the shipped collapse. **Delete** them, and record each with its replacement in the task file.
  - Tests of Delete/Export/Clone *behaviour*: **rewrite** them to drive `request_delete` / `export_theme` / `open_editor`, keeping every assertion about files, confirmations and registrations.
  - `Tests/Backup_Recovery/test_settings_file_participant_lifetimes.py`'s `_user_theme_labels`: rewrite it to use `editor.list_user_theme_names()`. Keep the pause/no-mkdir assertions exactly.
  - Hub tests using the R1 helper `Tests/UI/theme_editor_helpers.py` still work, because Clone still exists in the picker.
- [ ] **Step 5: Run** every file touched plus the theme suites (the list in Task 6). Everything passes except the documented CI-only `RecoveryRequired` hub tests.
- [ ] **Step 6: Commit** with `feat(theme-editor): Save as, Save returns to the picker, header; tree and library buttons move to the picker (TASK-32948)`.

---

### Task 5: PR 1 carry-over polish

**Files:** `tldw_chatbook/app.py`, `tldw_chatbook/UI/Screens/settings_screen.py`, `tldw_chatbook/Widgets/settings_theme_picker.py`; tests next to the existing ones.

These three minors from PR 1's final review each need a test:
- **Palette toast parity.** `ThemeProvider.switch_theme` notifies `f"{display_name(theme)} is now your theme (was: {display_name(change.previous_active)})"`, the same as the picker. If the change persisted but the config cache refresh failed, show the warning `"…; configuration refresh failed — reopen Settings to refresh"`. That needs `_persist_launch_default` to report `caches_reloaded`: extend `ThemeChange` with `caches_reloaded: bool = True`, set it in `use_theme`, and keep every existing construction valid through the default. Test in `Tests/UI/test_command_palette_providers.py` next to the updated `test_switch_theme_success`.
- **Open Theme highlights the launch default** (spec §8). `handle_appearance_open_theme` selects the Theme category, then `picker.refresh_catalog(highlight=current_launch_default())` once mounted. Use `call_after_refresh`. Test in `test_settings_theme_picker_screen.py`: Try `nord` (active nord, launch textual-dark) → Appearance → Open Theme → the highlighted id is `textual-dark`.
- **Revert label.** When a pending Revert persisted, the chip reads `f"Revert to {prev_active} (launch: {prev_launch})"` if the two differ. Otherwise keep `"Revert to X"`. Test in the picker tests.

- [ ] Steps: tests first (red), implement, run the tests (green), commit with `fix(theme): palette toast parity, Open Theme highlights launch default, clearer Revert (TASK-32948)`.

---

### Task 6: Geometry, docs, task file, full checks, live check

- [ ] **Geometry.** Extend `test_every_picker_control_is_reachable` with the four yours-only buttons. Highlight a user theme first; write the file in the private profile. Add an editor-view geometry test at 80×24 and 190×55: `#settings-theme-back`, `#settings-theme-name`, `#settings-theme-save`, `#settings-theme-save-as` and `#settings-theme-apply` are reachable. Run `./build_css.sh` if the CSS changed.
- [ ] **Keyboard journey.** Update `Tests/UI/test_settings_interface_keyboard_journeys.py`'s Theme walk for the new editor controls.
- [ ] **User Guide.** Update `Docs/User_Guide/settings.md` §Theme:
  - the Edit/Rename/Delete/Export actions for your themes, with their keys;
  - Save returning to the picker, and Save as;
  - the Delete fallback, and the backup pause row;
  - that the editor no longer has its own theme list.
  - Update the Verified stamp.
- [ ] **Task file.** Append PR 2 to `backlog/tasks/task-32948 - …md`: tick the PR 2 ACs, add Implementation Notes (what shipped, retired and rewritten tests with their reasons, rulings, deviations), and keep the status In Progress, since PR 3 remains.
- [ ] **Full checks.**
  - Run `Tests/Utils/test_theme_catalog.py`, `Tests/UI/test_settings_theme_picker.py`, `Tests/UI/test_settings_theme_picker_screen.py`, `Tests/UI/test_settings_theme_file_api.py`, `Tests/UI/test_settings_theme_editor.py`, `Tests/UI/test_settings_theme_editor_render.py`, `Tests/UI/test_settings_theme_card_contrast.py`, `Tests/UI/test_theme_contrast.py`, `Tests/UI/test_settings_interface_keyboard_journeys.py`, `Tests/UI/test_css_build_integrity.py`, `Tests/Utils/test_user_theme_loader.py`, `Tests/Backup_Recovery/test_settings_file_participant_lifetimes.py`, `Tests/Architecture/test_no_blocking_io_on_message_pump.py` and `Tests/UI/test_command_palette_providers.py`, with `-n 4`.
  - Compare the hub FAILED set against the base, `origin/feat/theme-picker-pr1`, in a detached scratch worktree, then remove that worktree.
  - Run ruff with no new findings, and preflight with rc 0.
- [ ] **Live check. REQUIRED.**
  - Use the `verify` skill. Isolate `HOME`, `TLDW_CONFIG_PATH` and the XDG dirs in a scratch profile. Check the real config's mtime before and after.
  - At 80×24 and 190×55, drive this sequence: Clone → edit → Save (returns to the picker, highlighted) → Rename via `r` → Edit via `e` → Save as → Delete of the active launch-default theme (falls back to Textual Dark) → Export.
  - Save the captures **immediately** after each keystroke, in the same command, so the toasts are visible. This was PR 1's evidence trap.
- [ ] **Commit** with `docs(settings): Theme picker PR 2 — User Guide + task-32948`. Do not push.
