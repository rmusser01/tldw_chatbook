# Theme Picker (PR 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the Settings ▸ Theme redesign. PR 3 adds:
- **Import:** bring in a theme file from a pasted or dropped path.
- **Broken theme files are shown, not hidden:** listed as "(unreadable)", with the error in the preview card. They can be deleted.
- **Missing launch default:** a "launch default missing" notice in the picker.
- **Export:** shows the full path, with a **Copy path** button.
- **Theme-name hardening:** Ruling R28 from PR 2 (escape theme names wherever they are rendered as markup; Reset finds a file through the one theme identity).
- **Close-out:** the User Guide rewrite and closing TASK-32948.

**Architecture:**
- Same rule as PR 2: every theme-file operation lives on `SettingsThemeEditor`, because the backup layer only recognises the editor as the owner of theme files. The picker posts requests, and `ThemePane` or `SettingsScreen` route them.
- Import reads the user's chosen source file directly: it is not an app-owned file. It writes the copy into the themes directory through the editor's `theme_file` scope.
- The unreadable state is reported by the editor's listing and flows into the catalog as `ThemeEntry.error` (spec §4 already defines that field).

**Tech Stack:** Python 3.12, Textual 8.2.8 (`App.copy_to_clipboard` exists; verified), pytest.

**Spec:** `Docs/superpowers/specs/2026-09-24-theme-picker-redesign-design.md` §5 (Import button), §7 (Import, Export), §9 (unreadable, launch default missing), §10 PR 3. The task file `backlog/tasks/task-32948 - …md` records rulings R1–R28 and the 2026-09-25 user decisions.

## Global Constraints

- **Branch and worktree:** `feat/theme-picker-pr3` off `feat/theme-picker-pr2`, in worktree `.claude/worktrees/theme-picker-pr3`. Prefix every shell command with `cd <worktree> &&`. Never `git stash`, never push, never open PRs.
- **Python:** `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` with `PYTHONPATH=<worktree>`. Assert that `tldw_chatbook.__file__` is under the worktree.
- **HARD RULE:** never run code that writes the user's real config, themes or `~/Downloads`. Editor file operations, `use_theme(persist=True)`, `persist_launch_default` and `_apply_config_mutation` all write the live profile unless they run under `@private_profile_test` (with a `request` argument) or are mocked. Ad-hoc probes must set `HOME`, `TLDW_CONFIG_PATH` and the XDG dirs to a scratch directory first.
- **Tests:** pilot tests that touch theme files or build Settings use `@private_profile_test`. Their theme files go in `config._get_effective_config_path().parent / "themes"`. Hub tests without the decorator that fail with `RecoveryRequired` are CI-only.
- **CSS:** ADR-161 tokens only. Never end a selector with a bare `Button`. Run `./build_css.sh`, then commit the bundle.
- **Notices never contain full theme-directory paths (R16).** The one exception is Export's destination, which spec §7 says to show.
- **Theme names are untrusted text.** They come from file contents. Every place a name is rendered with markup parsing on (`notify`, `Static` with markup, `Button` labels, modal titles) must escape it with `tldw_chatbook.Utils.input_validation.escape_markup`, or use `markup=False` where the widget supports it.
- **Commits:** a Co-Authored-By trailer naming the model that wrote the commit.
- **Before the PR:** preflight rc 0 (checked directly), and ruff reports no new findings.

## Review Focus

1. **Import of a hostile or broken file:**
   - not `.toml`;
   - over 64 KB;
   - not valid TOML;
   - no `[colors].primary`;
   - a colour that is not a valid colour;
   - a `[theme].name` that is not a valid filename or contains `[` markup;
   - a `[variables]` value with CSS injection.

   Each is refused with a specific message; nothing is written and nothing raises. Pinned in Task 2.
2. **Import onto an existing name.** A confirmation appears; cancelling leaves the existing file byte-identical. Pinned in Task 2.
3. **An unreadable theme file.** It is listed under Yours as "(unreadable)"; Use, Try, Edit, Rename and Export are disabled; Delete works; nothing crashes at startup or on refresh. Pinned in Task 1.
4. **A theme name containing `x[/]`.** It can be Used, Renamed, Deleted and Reverted, and all toasts, labels and titles render it literally. Pinned in Task 4.
5. **A launch default that names a theme which no longer exists.** The picker and Appearance say "launch default missing: <id>". Use on any theme clears the notice. Pinned in Task 3.

---

### Task 1: Unreadable theme files are listed, not hidden

**Files:** `tldw_chatbook/Widgets/settings_theme_editor.py`, `tldw_chatbook/css/Themes/theme_catalog.py`, `tldw_chatbook/Widgets/settings_theme_picker.py`; tests in `Tests/UI/test_settings_theme_file_api.py`, `Tests/Utils/test_theme_catalog.py`, `Tests/UI/test_settings_theme_picker.py`.

**Interfaces produced:**
- `SettingsThemeEditor.user_theme_listing() -> tuple[dict[str, Path], dict[str, str]]` returns `(readable name → path, unreadable stem → short error)`.
  - A file counts as *unreadable* when TOML parsing fails, or when `theme_from_file_data` raises (for example, no primary colour).
  - The error text is short and contains no path, e.g. "not valid TOML", "missing [colors].primary", "invalid colour 'background'". Use `_failure_reason` style.
  - `_user_theme_files()` returns only the readable map, built from this listing, so every existing caller is unchanged.
  - `RecoveryRequired` still escapes (R15).
- `ThemeEntry.error: str | None = None` (spec §4).
- `build_catalog(..., unreadable: Mapping[str, str] | None = None)`. Each unreadable stem becomes an entry with `origin="yours"`, `error=<text>`, all ten colours `#808080`, `dark=True`, no markers, and `display_name(stem) + " (unreadable)"`.
- `ThemePicker(list_user_names=..., list_unreadable=...)`. Both callables are injectable. `ThemePane` passes `lambda: editor.user_theme_listing()[1]`. Alternatively, change the picker's listing hook to return both halves in one call, to avoid two directory scans; the implementer's choice, stated in the report.

**Picker behaviour for an entry with `error`:**
- The preview card title reads `<name> (unreadable)`. A line below it shows the error, rendered as plain text with `markup=False`.
- Use, Try, Clone, Edit, Rename and Export are disabled, with the tooltip "This theme file can't be read: <error>".
- **Delete stays enabled.** `DeleteRequested` carries the stem, and `request_delete` must resolve unreadable stems through the listing (not only through the readable map).
- Enter on the row does nothing, and the `t`, `c`, `e` and `r` keys are ignored.

**Tests (red first):**
- **Editor:** `a.toml` containing garbage and `b.toml` with no primary both land in unreadable, with the expected error texts; readable files are unaffected; a pause still raises.
- **Catalog:** an unreadable entry is built correctly, sorted under Yours, and has no active or launch marker.
- **Picker:**
  - the row shows "(unreadable)" and the card shows the error;
  - Use, Try, Clone, Edit, Rename and Export are disabled, and Enter/`t` do nothing;
  - Delete posts `DeleteRequested("a")`.
- **Screen (private profile):** Delete of an unreadable file removes it after the confirmation.
- **Startup:** `load_user_themes` still skips the file without raising. This test already exists; keep it green.

Commit: `feat(theme): unreadable theme files are listed with their error and can be deleted (TASK-32948)`.

---

### Task 2: Import

**Files:** the editor, the picker, `settings_screen.py`, and tests (a new `Tests/UI/test_settings_theme_import.py` plus picker-screen tests).

**Interfaces:**
- `SettingsThemeEditor.import_theme(source: str) -> None`.
  1. **Validate the path.** Strip whitespace and surrounding quotes, since a terminal drop can paste a quoted path. Expand `~`. Validate with `tldw_chatbook.Utils.path_validation.validate_browsing_path` (it must be absolute, contain no NUL, and be absolute after expansion). The suffix must be `.toml` (case-insensitive). The file must exist, be a regular file, and be at most **64 KB** (`stat().st_size`).
  2. **Read and check the contents.** Read it (plain read: a user file, not an app-owned one) and `toml.loads` it. The name is `[theme].name`, or the source stem; it must pass `validate_filename`. Every `[colors]` value must parse with `textual.color.Color.parse`. Build the theme with `theme_from_file_data(...)`, which applies `sanitize_theme_variables`; catch `TypeError`/`ValueError`/`AttributeError`. On any failure, `notify(<specific reason, no source path>, severity="error")` and return, having written nothing.

     Reason texts:
     - "Import needs a .toml file"
     - "Theme file is larger than 64 KB"
     - "File is not valid TOML"
     - "Missing [colors].primary"
     - "background: '<v>' is not a colour" (with `<v>` escaped)
     - "Invalid theme name: …"
  3. **Write.** The target is `custom_themes_path / f"{name}.toml"`. If a user theme already claims that name (per `_user_theme_files`), or the target file exists, confirm with `ConfirmationDialog` ("Replace the saved theme '<name>'?", "Replace" / "Keep existing"). Write the parsed data, normalised to `[theme] name/dark`, `[colors]` and the sanitised `[variables]`, through the same temp-then-replace code `_write_theme_file` uses (reuse it; don't copy it). Register the theme, post `ThemesChanged`, and notify "Imported '<name>'".
  4. **Pause.** On `RecoveryRequired`, show the unavailable notice and write nothing.
- **Picker:** an `[ Import… ]` button `#settings-theme-picker-import`, next to New, and the key `i` on the list. It posts `ThemePicker.ImportRequested()`. It is disabled while paused (R17 pattern).
- **Screen:** it handles `ImportRequested` by pushing `RagProfileNameModal(title="Import theme from file", initial="", confirm_label="Import")`. The callback calls `editor.import_theme(value)` when the value is non-empty, then `picker.refresh_catalog(highlight=<imported name>)`. Have `import_theme` return the imported name, or `None`, so the screen can highlight it; adjust the signature and record it.

**Tests (red first; source files written under `tmp_path`, a non-profile temp dir):**
- a valid import lands in the themes directory and is registered and highlighted;
- `[variables]` are sanitised;
- every hostile or broken case in Review Focus 1 is refused with its message, writes nothing and doesn't raise;
- the quoted-path paste form `'/tmp/x.toml'` works;
- a `~` path expands;
- a relative path is refused;
- importing onto an existing name confirms, and Cancel leaves the existing file byte-identical;
- during a pause, the Import button is disabled and `import_theme` writes nothing.

Commit: `feat(theme): Import a theme file from a pasted or dropped path (TASK-32948)`.

---

### Task 3: Launch default missing, and the Export Copy path

**Files:** the picker, the editor (the Export result), `settings_screen.py` if needed, and tests.

- **Launch default missing (spec §9).** When `current_launch_default()` is not in `app.available_themes`, the picker shows `#settings-theme-launch-missing`: "Launch default missing: <id> — Use any theme to fix it". It is a `Static` with `markup=False`, above the list. It updates on every `refresh_catalog` and is hidden otherwise. Appearance already renders the missing case (`_appearance_theme_summary`, `settings_screen.py` ~9540); add one screen test that pins both surfaces together.
  - Tests: the launch default is set to `deleted_one` (mocked `current_launch_default`) → the notice shows in both places; after a mocked Use of `nord`, the picker notice is gone.
- **Export Copy path (spec §7).** After a successful export, the editor posts `SettingsThemeEditor.Exported(path: Path)`. `ThemePane` shows a row in the picker card, `#settings-theme-export-result`, with a `Static` reading `Exported to <full path>` (`markup=False`) and a `Button("Copy path", id="settings-theme-copy-path")` that calls `self.app.copy_to_clipboard(str(path))` and notifies "Path copied". The row hides on the next highlight change. The Export toast keeps its full path.
  - Tests: export (with `Path.home` monkeypatched to `tmp_path`) → the row shows the path. Clicking Copy path calls `copy_to_clipboard` with that path; monkeypatch `app.copy_to_clipboard` to record it.

Commit: `feat(theme-picker): launch-default-missing notice; Export shows its path with Copy path (TASK-32948)`.

---

### Task 4: Theme-name hardening (Ruling R28)

**Files:** the editor, the picker, `settings_screen.py`, `app.py`, `theme_catalog.py` (`use_theme_toast`); tests.

- **Escape theme names wherever markup is parsed.** Audit every f-string that puts a theme name, display name or id into:
  - `app.notify(...)`
  - a `Button` label (the Revert chip, "Revert to …")
  - a `Static` without `markup=False`
  - a modal or dialog title or message (`ConfirmationDialog` messages for Delete, Overwrite and Replace)
  - `use_theme_toast`

  Escape each with `escape_markup`. Find them with `grep -n "notify(f\|label = f\|Static(f\|message=(\|title=f" …` across the four files. The OptionList row prompt is a Rich `Text`, which is literal, so it doesn't need escaping; check this.
  - Test: a user theme named `x[/]` (the file `x.toml` with `[theme] name = "x[/]"`). Use it (toast), then Revert it (chip label and toast), Rename it (modal title and toast), and Delete it (dialog and toast). Nothing raises, and the literal `x[/]` appears in each rendered string.
- **Reset through the one identity.** `_reset_theme` re-reads the loaded user theme by stem. Make it resolve through `_user_theme_files()`.
  - Test: Edit `a.toml` (named `b`), change a colour, Reset → the colours come back from `a.toml`.

Commit: `fix(theme): theme names render literally everywhere; Reset resolves the theme file by name (TASK-32948, R28)`.

---

### Task 5: User Guide rewrite, task close-out, full checks, live check

- [ ] **User Guide.** Rewrite `Docs/User_Guide/settings.md` §Theme as one coherent section for the finished design (spec §10: PR 3 owns "a User Guide rewrite"). It should cover:
  - the picker and its groups, markers and filter;
  - keys and actions;
  - Use, Try and Revert;
  - your-theme actions, and Import;
  - the editor (Save, Save as, Esc and Back, the leave prompt);
  - Export with Copy path;
  - the states: pause, unreadable and launch default missing;
  - the Appearance row.

  Remove the patchwork of per-PR sentences. Stamp it "Verified against".
- [ ] **Task file.** In `backlog/tasks/task-32948 - …md`, tick ACs #9 to #12, or whichever are the PR 3 ACs; confirm by reading them. Add PR 3 Implementation Notes (what shipped, rulings, tests). Set the status to **Done** only if every AC is ticked and the Definition of Done holds; otherwise leave it In Progress and say which AC is open.
- [ ] **Full checks.**
  - Run with `-n 4`: every theme suite (catalog, picker, picker screen, file API, import, editor, editor render, card contrast, theme contrast, keyboard journeys, css build integrity, user theme loader, backup participant, no-blocking-io, and palette TestThemeProvider).
  - Compare the hub FAILED set against the base `origin/feat/theme-picker-pr2`, using a detached scratch worktree that you then remove.
  - ruff reports no new findings; preflight rc 0 (read any diagnostic-inventory rows before running `--write`).
- [ ] **Live check. REQUIRED.** Use the `verify` skill with an isolated HOME, TLDW_CONFIG_PATH and XDG profile. Check the real config's mtime and the real themes directory before and after. Capture the screen in the same command as each keystroke. Sequence, at 80×24 and 190×55:
  1. Import a valid theme from a scratch path, then import a broken one (error toast).
  2. Drop a garbage `.toml` file into the scratch themes directory, reopen Theme, see "(unreadable)", and Delete it.
  3. Delete the launch default while trying another theme: the setting changes and the screen doesn't (user decision 2026-09-25). See the "launch default missing" notice only if applicable.
  4. Export, then Copy path.
  5. Use a theme named `x[/]`.
- [ ] Commit: `docs(settings): Theme User Guide rewrite + task-32948 close-out (PR 3)`. Do not push.
