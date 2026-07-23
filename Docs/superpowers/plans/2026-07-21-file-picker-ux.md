# File-Picker UX (TASK-430 + TASK-431) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the app's file picker usable — visible selection cursor, one uniform activation model (single-click selects, double-click/Enter/Go opens for files AND dirs), path-bar opens a typed file, Esc closes the topmost overlay first; and for card import, accept `.webp`, stop treating `.md` as a card, show a "N hidden by filter" hint, and keep the per-context start dir.

**Architecture:** All TASK-430 changes are overrides in `EnhancedFileDialog` / `SearchableDirectoryNavigation` (`Widgets/enhanced_file_picker.py`) or scoped app CSS — **no edits to the vendored `Third_Party/textual_fspicker/`**. TASK-431 AC#1 is local to `personas_screen.py`; AC#2 is additive in the picker; AC#3 already works (test only).

**Tech Stack:** Python 3.11+, Textual (OptionList-based picker), pytest + pytest-asyncio (Pilot-driven widget tests).

## Global Constraints

- **Run tests via the repo venv, from the worktree root:** `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest ...`
- **No edits to `tldw_chatbook/Third_Party/textual_fspicker/`** — override in the app subclasses only. Use the existing `_SUPPRESSED_BASE_HANDLERS`/`_get_dispatch_methods` (`enhanced_file_picker.py:919-931`) ONLY for `@on`-decorated base handlers; plain `_on_<message>` convention handlers are replaced by a normal subclass override (single most-derived method wins).
- **Filters must be callables**, never glob strings (`tests/UI/test_file_picker_filters_callable.py` polices this).
- **`Tests/` and `tests/` are byte-identical** — before editing, `diff -q Tests/UI/<f> tests/UI/<f>`; edit the one the venv actually collects (run the test from each path once to confirm) and keep them identical (copy after editing if both are collected).
- **Design source of truth:** `Docs/superpowers/specs/2026-07-21-file-picker-ux-design.md`.
- **Shared widget** — after each TASK-430 task, run the cross-consumer picker suites listed in Task 6.
- **Commit trailer:** `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

## Task 1: Visible selection cursor (430 AC#1)

**Files:**
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss` (near the generic `OptionList > .option-list--option-highlighted` rule ~1330)
- Test: `tests/UI/test_enhanced_file_dialog_mount.py`

**Interfaces:**
- Produces: the file list (`#file-list-pane` → `SearchableDirectoryNavigation`) renders its highlighted row with `$ds-focus-bg` instead of the near-invisible `$surface`.

- [ ] **Step 1: Write the failing test**

Add to `tests/UI/test_enhanced_file_dialog_mount.py` (reuse `_DialogHost`/`pilot`). Assert the highlighted-option component style resolves to the focus token, not `$surface`. Textual exposes component styles via `get_component_styles`:

```python
async def test_file_list_highlight_is_visible():
    app = _DialogHost()
    async with app.run_test() as pilot:
        await _open_dialog(pilot)  # mirror the file's existing open helper
        nav = pilot.app.screen.query_one(SearchableDirectoryNavigation)
        style = nav.get_component_styles("option-list--option-highlighted")
        # $surface (invisible) must NOT be the highlight bg; the focus token must.
        from tldw_chatbook.Widgets.enhanced_file_picker import SearchableDirectoryNavigation  # noqa
        assert style.background is not None
        # Resolve $ds-focus-bg (#51677e) — compare against the theme's surface.
        surface = pilot.app.screen.query_one(SearchableDirectoryNavigation).styles.background
        assert style.background != surface
```

If `get_component_styles` on the widget doesn't resolve the app-CSS rule in the test theme, fall back to asserting the rule exists and is id-scoped: parse `tldw_chatbook/css/tldw_cli_modular.tcss` for a `#file-list-pane .option-list--option-highlighted` block referencing `$ds-focus-bg`. Prefer the rendered-style assertion; use the source assertion only if the theme won't resolve in-test (note which in the report).

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest tests/UI/test_enhanced_file_dialog_mount.py -k highlight_is_visible -v`
Expected: FAIL — highlight still resolves to `$surface`.

- [ ] **Step 3: Add the scoped CSS rule**

In `tldw_chatbook/css/tldw_cli_modular.tcss`, immediately after the generic `OptionList > .option-list--option-highlighted { ... }` block (~1330-1340), add:

```css
/* TASK-430 AC#1: the file picker's list must show a clearly-visible selected
   row. The generic OptionList rule paints $surface (≈ dialog background),
   which is near-invisible; scope a higher-contrast focus token to the picker
   list only, using the sanctioned non-obscuring focus color (TASK-345). */
#file-list-pane .option-list--option-highlighted {
    background: $ds-focus-bg;
    color: $ds-focus-fg;
    text-style: bold;
}
```

Confirm `$ds-focus-bg`/`$ds-focus-fg` are defined (`css/core/_variables.tcss:17,24`) and that `#file-list-pane` is the pane id wrapping the `SearchableDirectoryNavigation` (`enhanced_file_picker.py:1074-1076`).

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest tests/UI/test_enhanced_file_dialog_mount.py -k highlight_is_visible -v`
Expected: PASS. Also run `tests/UI/test_non_obscuring_focus_contract.py` — must stay green.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/css/tldw_cli_modular.tcss tests/UI/test_enhanced_file_dialog_mount.py
git commit -m "fix(filepicker): visible selected-row cursor in the file list (task-430)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: Uniform activation — single-click selects, double-click/Enter/Go opens (430 AC#2)

**Files:**
- Modify: `tldw_chatbook/Widgets/enhanced_file_picker.py` (`SearchableDirectoryNavigation` ~492-655; the Go/Select button path `_confirm_single`/`_on_select_button` ~1270/1371)
- Test: `tests/UI/test_enhanced_file_dialog_mount.py`

**Interfaces:**
- Produces: single-click/select highlights without navigating (dirs) / fills filename (files); `action_open_highlighted` descends a dir or returns a file; Enter and double-click and the Go button all route through it.

**Read first:** the vendored `DirectoryNavigation._on_option_list_option_selected` (`Third_Party/textual_fspicker/parts/directory_navigation.py:466-481`) — it navigates dirs on select and posts `Selected` for files. Confirm it is a plain `_on_...` convention handler (no `@on` decorator) so a subclass override replaces it. Also read `_confirm_single` (`enhanced_file_picker.py:1270`) and `_on_select_button` (`:1371`) for the existing confirm path, and `DirectoryNavigation.BINDINGS`/how Enter maps to `action_select` (OptionList default).

- [ ] **Step 1: Write the failing tests**

Add to `tests/UI/test_enhanced_file_dialog_mount.py`. Use the file's existing pattern: set `nav.highlighted` to an index, drive select vs open, and poll the threaded loader (`for _ in range(20): await pilot.pause()`). You need a temp dir with at least one subdir and one file (mirror the file's existing tmp-dir fixture).

```python
async def test_single_select_on_dir_does_not_navigate():
    # highlight a subdirectory, "select" it (single-click semantics) -> location unchanged
    ...
    start = nav._location
    nav.highlighted = <index of the subdir>
    nav.action_select()          # OptionSelected path (single-click / current select)
    await _settle(pilot)
    assert nav._location == start          # did NOT descend

async def test_open_highlighted_descends_dir():
    nav.highlighted = <subdir index>
    nav.action_open_highlighted()
    await _settle(pilot)
    assert nav._location == <subdir path>  # descended

async def test_open_highlighted_returns_file():
    nav.highlighted = <file index>
    nav.action_open_highlighted()
    await _settle(pilot)
    # dialog dismissed with the file (via the host's captured result), or the
    # filename input is filled and confirm returns it — assert whichever the
    # confirm path produces.
    ...

async def test_single_select_on_file_fills_filename():
    nav.highlighted = <file index>
    nav.action_select()
    await _settle(pilot)
    assert host.query_one("#filename-input", Input).value == <file name>
```

Also UPDATE the existing tests that assumed select-navigates-a-dir (grep `action_select` in the file) to the new select-only semantics — preserve their real intent (e.g. a test that "selecting a dir then confirming enters it" becomes "opening a dir descends it"). Do not delete coverage; re-express it.

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest tests/UI/test_enhanced_file_dialog_mount.py -k "single_select or open_highlighted" -v`
Expected: FAIL — dir select still navigates; `action_open_highlighted` doesn't exist.

- [ ] **Step 3: Implement the select/open split**

In `SearchableDirectoryNavigation`:

```python
    def _on_option_list_option_selected(
        self, event: "OptionList.OptionSelected"
    ) -> None:
        """Select-only (task-430 AC#2): a single-click / OptionSelected
        highlights and, for a file, fills the filename input; it never
        auto-navigates a directory (opening is a separate action)."""
        event.stop()
        option = event.option
        if not isinstance(option, DirectoryEntry):
            return
        if not is_dir(option.location):
            # File: keep the existing fill-filename behavior.
            self.post_message(self.Selected(self, option.location))
        # Directory: do nothing here — highlight already moved; opening is
        # action_open_highlighted (Enter / double-click / Go).

    def action_open_highlighted(self) -> None:
        """Open the highlighted entry: descend a directory, or return a file."""
        if self.highlighted is None:
            return
        option = self.get_option_at_index(self.highlighted)
        if not isinstance(option, DirectoryEntry):
            return
        if is_dir(option.location):
            self._location = option.location.resolve()   # descend (vendored path)
        else:
            self.post_message(self.OpenFile(self, option.location))

    def on_click(self, event) -> None:
        """Double-click opens the highlighted entry (mouse ≈ Enter)."""
        if getattr(event, "chain", 1) >= 2:
            self.action_open_highlighted()
```

Add an `OpenFile` message class (mirror `Selected` at `directory_navigation.py`) on `SearchableDirectoryNavigation`, and rebind Enter to the new action by adding to `SearchableDirectoryNavigation.BINDINGS`: `Binding("enter", "open_highlighted", "Open", show=False)` (this overrides OptionList's default Enter→`action_select`). Verify `is_dir`, `DirectoryEntry`, `Binding`, `OptionList` are importable in this module (grep; add imports if needed).

In `EnhancedFileDialog`, handle the new `OpenFile` message (route a file open through the existing confirm path `_confirm_single`), and extend the **Go/Select button** (`_on_select_button`/`_confirm_single`, `:1270/1371`) so that when the highlighted entry is a directory and no filename is typed, it calls the nav's `action_open_highlighted` (descend) instead of confirming/no-op.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest tests/UI/test_enhanced_file_dialog_mount.py -v`
Expected: PASS (new + the updated existing tests).

**If Textual's event model fights you** (e.g. OptionSelected still fires on Enter after the rebind, or double-click `chain` isn't delivered): STOP and report DONE_WITH_CONCERNS or BLOCKED describing exactly what fired, rather than forcing a fragile hack. The controller can adjust the approach (e.g. gate on a click-timestamp, or drive "open" only via Enter+Go and defer double-click).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/enhanced_file_picker.py tests/UI/test_enhanced_file_dialog_mount.py
git commit -m "feat(filepicker): uniform activation — single-click selects, dbl-click/Enter/Go opens (task-430)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 3: Path bar opens a typed file + layered Esc (430 AC#3, AC#4)

**Files:**
- Modify: `tldw_chatbook/Widgets/enhanced_file_picker.py` (`EnhancedFileDialog`)
- Test: `tests/UI/test_enhanced_file_dialog_mount.py`

**Interfaces:**
- Produces: submitting a full existing-file path in the Ctrl+L bar returns/confirms the file; `action_smart_dismiss` closes the topmost overlay before dismissing the picker.

**Read first:** vendored `FileSystemPickerScreen._on_path_input_submit` (`base_dialog.py:495-540`) — note the `else`/file branch (`:527-531`) that navigates to `path.parent` and drops the file, and whether it is `@on`-decorated (if so, add to `_SUPPRESSED_BASE_HANDLERS`). And `FileSystemPickerScreen.BINDINGS` escape (`base_dialog.py:166`). And the overlay reactives: `show_recent` (`base_dialog.py:170`), `show_bookmarks` (`:912`), `search_active` (`base_dialog.py:173`), and the path bar container `#path-input-container` `styles.display`.

- [ ] **Step 1: Write the failing tests**

```python
async def test_path_bar_opens_a_file(tmp_path):
    # a real file exists under tmp_path
    ...
    dialog.action_focus_path_input()
    path_input = host.query_one("#path-input", Input)
    path_input.value = str(<the file>)
    await host.query_one("#go-to-path", Button).press()  # or Input.Submitted
    await _settle(pilot)
    assert host.result == <the file>          # dialog dismissed with the file

async def test_escape_closes_recent_overlay_first():
    dialog.action_toggle_recent()  # open Recent (mirror the real toggle action name)
    await pilot.pause()
    assert dialog.show_recent is True
    await pilot.press("escape")
    await pilot.pause()
    assert dialog.show_recent is False        # overlay closed
    assert host.result is _UNSET              # picker NOT dismissed
    await pilot.press("escape")
    await pilot.pause()
    assert host.result is None                # now dismissed
```

Match the real toggle action/attribute names (grep `show_recent`, `action_toggle_recent`/`action_focus_path_input`) and the host's result sentinel.

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest tests/UI/test_enhanced_file_dialog_mount.py -k "path_bar_opens or escape_closes" -v`
Expected: FAIL — path bar cd's to parent; Esc dismisses the whole picker.

- [ ] **Step 3: Override path submit + add smart_dismiss**

Override `_on_path_input_submit` in `EnhancedFileDialog` (copy the vendored `@on` decorators; add the base method to `_SUPPRESSED_BASE_HANDLERS` if it is `@on`-decorated): resolve the typed path; if `path.is_file()` → run through `_should_return(path)` and `self.dismiss(result=path)` (or fill `#filename-input` + confirm via `_confirm_single`), hiding the path bar; if `path.is_dir()` → keep the vendored cd behavior; else → keep the current gentle handling.

Add to `EnhancedFileDialog.BINDINGS`: `Binding("escape", "smart_dismiss", "Close", show=False)` and:

```python
    def action_smart_dismiss(self) -> None:
        """Close the topmost open overlay; dismiss the picker only when none open."""
        if self.query_one("#path-input-container").styles.display != "none":
            self.action_cancel_path_input()   # or set display none — match the real close
            return
        if getattr(self, "search_active", False):
            self.action_clear_search()         # match the real search-close action
            return
        if getattr(self, "show_recent", False):
            self.show_recent = False
            self._sync_sidebar()
            return
        if getattr(self, "show_bookmarks", False):
            self.show_bookmarks = False
            self._sync_sidebar()
            return
        self.dismiss(None)
```

Use the real close helpers/reactive names (grep `action_cancel_path_input`, `_sync_sidebar`, `action_clear_search`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest tests/UI/test_enhanced_file_dialog_mount.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/enhanced_file_picker.py tests/UI/test_enhanced_file_dialog_mount.py
git commit -m "feat(filepicker): path bar opens a typed file; Esc closes topmost overlay first (task-430)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 4: Card import filter — accept .webp, drop .md as a card (431 AC#1)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py` (~3953-3972); check `tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py:929`
- Test: `tests/UI/test_personas_workbench.py` (or a filter-focused test) + `tests/UI/test_file_picker_filters_callable.py` stays green

**Interfaces:**
- Produces: the "Character Cards" primary filter accepts `.png/.webp/.json` and NOT `.md`; a dedicated Markdown sub-filter remains; all filters are callables.

- [ ] **Step 1: Write the failing test**

```python
def test_character_cards_filter_accepts_webp_not_md():
    # Build the same Filters the import dialog uses (extract to a helper if needed),
    # then assert the primary "Character Cards" tester.
    from pathlib import Path
    tester = _character_cards_primary_filter()   # see Step 3
    assert tester(Path("x.webp")) is True
    assert tester(Path("x.png")) is True
    assert tester(Path("x.json")) is True
    assert tester(Path("README.md")) is False
```

Prefer extracting the filter construction into a small module-level helper (`_character_import_filters()`) so it is unit-testable without mounting the screen; the dialog code then calls the helper. If extraction is undesired, drive it through the screen the way the file's existing import tests do.

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest tests/UI/test_personas_workbench.py -k webp_not_md -v`
Expected: FAIL — `.webp` False, `.md` True today.

- [ ] **Step 3: Update the filter**

In `personas_screen.py` (~3957), change the primary tester and add a WEBP-inclusive images sub-filter:

```python
                filters=Filters(
                    (
                        "Character Cards",
                        lambda p: p.suffix.lower() in (".json", ".png", ".webp"),
                    ),
                    ("JSON Files", lambda p: p.suffix.lower() == ".json"),
                    (
                        "Card Images (PNG/WebP)",
                        lambda p: p.suffix.lower() in (".png", ".webp"),
                    ),
                    (
                        "Markdown Files",
                        lambda p: p.suffix.lower() in (".md", ".markdown"),
                    ),
                    ("All Files", lambda p: True),
                ),
```

Apply the equivalent change to the legacy CCP route (`ccp_character_handler.py:929`) if it constructs its own card filter. Keep every tester a `lambda`/callable (no glob strings).

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest tests/UI/test_personas_workbench.py -k webp_not_md tests/UI/test_file_picker_filters_callable.py -v`
Expected: PASS (and the callable-filter guard stays green).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py tests/UI/test_personas_workbench.py
git commit -m "fix(personas): card import filter accepts .webp, drops .md as a card (task-431)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 5: "N hidden by filter" hint (431 AC#2)

**Files:**
- Modify: `tldw_chatbook/Widgets/enhanced_file_picker.py` (`SearchableDirectoryNavigation._repopulate_display` ~605-654; a new `Static` + handler in `EnhancedFileDialog.compose`)
- Test: `tests/UI/test_enhanced_file_dialog_mount.py`

**Interfaces:**
- Produces: `SearchableDirectoryNavigation.FilterHiddenCountChanged(count)` message; a `Static#filter-hidden-notice` rendering "N hidden by filter" (blank when 0).

- [ ] **Step 1: Write the failing test**

```python
async def test_filter_hidden_count(tmp_path):
    # tmp_path has 2 .json and 3 .txt files; filter = only .json
    ...
    # apply a filter that excludes the .txt files, then read the notice
    notice = host.query_one("#filter-hidden-notice", Static)
    assert "3 hidden" in str(notice.renderable)
```

Drive the filter the way the file already changes filters (via the `Select#file-filter` or by setting `nav.file_filter`), then poll.

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest tests/UI/test_enhanced_file_dialog_mount.py -k filter_hidden_count -v`
Expected: FAIL — `#filter-hidden-notice` doesn't exist.

- [ ] **Step 3: Compute + emit the count**

In `_repopulate_display` (~605), compute the number of entries excluded **specifically by the active file_filter** (a real file that passes the show-hidden/dotfile check but fails `self.file_filter`), and post a new message after the existing `SearchCountChanged`:

```python
        filter_hidden = sum(
            1
            for entry in self._entries
            if not entry.location.is_dir()
            and not self._is_hidden(entry.location)   # match the real hidden check
            and self.file_filter is not None
            and not self.file_filter(entry.location)
        )
        self.post_message(self.FilterHiddenCountChanged(self, filter_hidden))
```

Add the `FilterHiddenCountChanged` message class (mirror `SearchCountChanged` ~509). Read how `self.hide()` splits the hidden-vs-filter decision (`directory_navigation.py:339-354`) and reuse its filter check rather than reinventing it (avoid double-counting dotfiles). In `EnhancedFileDialog.compose`, add `yield Static("", id="filter-hidden-notice")` near `#search-no-match` (`:1065`), and a handler `_on_filter_hidden_count_changed` that sets its text to `f"{count} hidden by filter"` (blank when 0). Style it like `#search-no-match`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest tests/UI/test_enhanced_file_dialog_mount.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/enhanced_file_picker.py tests/UI/test_enhanced_file_dialog_mount.py
git commit -m "feat(filepicker): show 'N hidden by filter' hint (task-431)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 6: Per-context start-dir test + cross-consumer regression + live-verify (431 AC#3)

**Files:**
- Test: `tests/UI/test_file_picker_start_dir.py` (new) or extend an existing picker test

- [ ] **Step 1: Add the per-context start-dir regression test**

The infra already exists (`_get_last_directory`/`_save_last_directory`, config key `filepicker.last_dir_{context}`). Add a test proving two contexts stay independent and a saved value is used:

```python
def test_last_dir_is_per_context(monkeypatch, tmp_path):
    # write filepicker.last_dir_character_import = <dirA> and last_dir_chat_images = <dirB>
    # construct EnhancedFileOpen(context="character_import") and assert its
    # resolved start _location is dirA, not dirB or a global default.
    ...
```

Match how config is read/written in the picker (grep `_get_last_directory`, the config accessor); monkeypatch the config store the way other picker tests do.

- [ ] **Step 2: Cross-consumer regression suites**

Run all picker + consumer suites (the shared-widget changes must not regress any):

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  tests/UI/test_enhanced_file_dialog_mount.py tests/test_enhanced_filepicker.py \
  tests/UI/test_file_picker_filters_callable.py tests/UI/test_file_picker_bookmarks_lazy.py \
  tests/UI/test_file_picker_action_tooltips.py tests/UI/test_legacy_attach_picker.py \
  tests/UI/test_eval_file_picker_dialog.py tests/UI/test_non_obscuring_focus_contract.py \
  tests/UI/test_mcp_workbench.py tests/UI/test_chat_image_attachment.py -q
```
Expected: all green. Investigate any failure (a consumer relying on old activation/Esc semantics) before proceeding.

- [ ] **Step 3: Live-verify in the real TUI**

Drive the Personas card import (`verify` recipe, scratch profile). Confirm: the selected row is clearly visible; single-click selects, double-click/Enter/Go opens (dir descends, file imports); Ctrl+L with a full file path opens it; Esc closes the Recent overlay first (picker stays), second Esc closes the picker; the card filter lists `.webp` and hides `.md`; a docs folder shows the "N hidden by filter" hint.

- [ ] **Step 4: Mark ACs + notes**

```bash
backlog task edit 430 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --notes "<summary>"
backlog task edit 431 --check-ac 1 --check-ac 2 --check-ac 3 --notes "<summary>"
```

---

## Self-Review

- **Spec coverage:** 430 AC#1→T1, AC#2→T2, AC#3+AC#4→T3; 431 AC#1→T4, AC#2→T5, AC#3→T6; cross-consumer regression→T6. All covered.
- **Placeholder scan:** test bodies say "match the real name/attribute" (grep instructions) and give concrete assertions — not TBDs. T2/T3 carry BLOCKED/adapt guidance for Textual event quirks, which is deliberate for a shared-widget interaction change.
- **Type/name consistency:** `action_open_highlighted`, `OpenFile`, `FilterHiddenCountChanged`, `#file-list-pane`, `#filter-hidden-notice`, `context="character_import"` used consistently across tasks.
- **Ordering:** 1 (CSS) → 2 (activation) → 3 (path-bar/Esc) → 4 (filter, local) → 5 (hidden-count) → 6 (start-dir test + regression + live). T2 is the riskiest; T6 gates the shared-widget blast radius.
