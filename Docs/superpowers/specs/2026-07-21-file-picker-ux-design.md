# TASK-430 + TASK-431 — File-picker UX (interaction model + card filter/start-location)

- **Date:** 2026-07-21
- **Tasks:** TASK-430 (interaction model) + TASK-431 (card filter + start-location), from the RP/character-card UX review. Done as one branch/PR (both are "file picker UX").
- **Branch base:** origin/dev
- **Widget:** `EnhancedFileOpen`/`EnhancedFileSave` → `EnhancedFileDialog(BaseFileDialog)` in `tldw_chatbook/Widgets/enhanced_file_picker.py` (1882 lines), on the vendored `tldw_chatbook/Third_Party/textual_fspicker/`. The list is a `SearchableDirectoryNavigation` (subclass of vendored `DirectoryNavigation`, itself a Textual `OptionList`).

## Problem

The file picker is high-friction (observed live during card import): the selected row is near-invisible; single-click navigates *directories* but *files* need select+Enter (inconsistent); a full file path typed in the Ctrl+L bar only navigates to the parent instead of opening the file; Esc inside the Recent overlay dismisses the whole picker; the card-import filter hides `.webp` (which the importer accepts) while including `.md` (so docs read as cards); and a filtered folder gives no hint that entries are hidden.

## Constraints & principles

- **No edits to the vendored `textual_fspicker`.** Every change is an override in `EnhancedFileDialog` / `SearchableDirectoryNavigation`, using the existing `_SUPPRESSED_BASE_HANDLERS` / `_get_dispatch_methods` machinery (`enhanced_file_picker.py:919-931`) to neutralize base `@on` handlers where needed.
- **Shared blast radius:** all TASK-430 changes affect ~15 consumers (Library ingest, chat images, chatbook, MCP, evals, LLM management, character/lore/dictionary import…). Cross-consumer regression is the main risk; the plan verifies the picker test suites and the consumer contracts (`test_legacy_attach_picker.py`, `test_eval_file_picker_dialog.py`, `test_file_picker_filters_callable.py`, `test_non_obscuring_focus_contract.py`).
- **No config migration**; the per-context last-dir infra already exists (`filepicker.last_dir_{context}`).
- **`Tests/` and `tests/` are byte-identical** — edit `tests/` (both are shipped; keep them in sync if the repo build copies one to the other — verify).

## Design

### TASK-430

#### AC#1 — Visible selection cursor
The list's highlighted-row color is currently governed by the **global** rule `OptionList > .option-list--option-highlighted { background: $surface; ... }` (`css/tldw_cli_modular.tcss:1330-1340`), which — by Textual origin priority — overrides the widget's own `DEFAULT_CSS` (`enhanced_file_picker.py:867-877`). `$surface` ≈ the dialog background, hence invisible.

Fix: add an **id-scoped** rule that beats the generic `OptionList` rule, painting the file list's highlighted row with a visibly-contrasting token:
```css
#file-list-pane .option-list--option-highlighted {
    background: $ds-focus-bg;   /* #51677e steel-blue, ~3:1 on dark surfaces (TASK-345) */
    color: $ds-focus-fg;
    text-style: bold;
}
```
Placed in `css/tldw_cli_modular.tcss` near the generic rule. Uses the existing `$ds-focus-bg`/`$ds-focus-fg` tokens (`css/core/_variables.tcss:17,24`) so it satisfies the non-obscuring-focus contract (`tests/UI/test_non_obscuring_focus_contract.py`). Scoped to `#file-list-pane`, so only the picker's list changes (other `OptionList`s keep their look).

#### AC#2 — Uniform activation (single-click selects, double-click/Enter/Go opens)
Today the divergence is in vendored `DirectoryNavigation._on_option_list_option_selected` (`directory_navigation.py:466-481`): on `OptionSelected` (fired by single-click AND Enter), a directory is navigated into immediately; a file only posts `Selected` (which fills `#filename-input`). The uniform model:
- **Single-click** → *select* (highlight; for a file, fill `#filename-input`; for a dir, just highlight — **no auto-navigate**).
- **Enter** → *open the highlighted* (descend dir / confirm file).
- **Double-click** → *open the highlighted* (mouse equivalent of Enter).
- **Go/Select button** → *open the highlighted* (descend dir OR confirm file) — today it only confirms files.

Implementation (all in `SearchableDirectoryNavigation` + `EnhancedFileDialog`, no vendored edit):
1. **Override `_on_option_list_option_selected`** in `SearchableDirectoryNavigation` so an `OptionSelected` (now only from a mouse single-click — see step 2) is *select-only*: for a dir, set the highlight/selection but do **not** assign `self._location`; for a file, post the existing `Selected` message (fills filename). Add this base handler to `_SUPPRESSED_BASE_HANDLERS` so the vendored navigate-on-select does not also run.
2. **Rebind Enter** on the list to `action_open_highlighted` (a new action) instead of the OptionList default `action_select` (which posts `OptionSelected`). This makes Enter *open*, and keeps `OptionSelected` firing only for mouse clicks.
3. **Add `on_click`** to the list: when `event.chain == 2` (double-click), call `action_open_highlighted`.
4. **`action_open_highlighted`**: resolve the highlighted `DirectoryEntry`; if it is a directory, navigate (`self._location = path`, the existing descend path); if a file, route through the dialog's confirm path (`_confirm_single` / fill+confirm) so it returns.
5. **Go/Select button** (`_on_select_button` → `_confirm_single`, `enhanced_file_picker.py:1270/1371`): extend so that when the highlighted entry is a directory (and no filename typed), it descends instead of no-op/confirm — mirroring `action_open_highlighted`.

Existing tests that call `dir_nav.action_select()` expecting immediate navigation (`test_enhanced_file_dialog_mount.py`) will be updated to the new semantics (select then open), with new assertions for single-click-selects / double-click-opens / Enter-opens for both files and dirs.

#### AC#3 — Path bar returns a file
Override the vendored `FileSystemPickerScreen._on_path_input_submit` (`base_dialog.py:495-540`) in `EnhancedFileDialog` (same `@on` decorators; add the base to `_SUPPRESSED_BASE_HANDLERS`). New behavior: resolve the typed path; if it is an existing **file**, run it through `_should_return` and `dismiss(result=path)` (Open) / confirm (Save) instead of only `cd`-ing to the parent; if a **directory**, `cd` as today; if nonexistent, keep the current gentle handling. Hides the path bar on success.

#### AC#4 — Layered Esc
The only Esc binding is vendored `FileSystemPickerScreen.BINDINGS: Binding("escape", "dismiss(None)")` (`base_dialog.py:166`). Add to `EnhancedFileDialog.BINDINGS`: `Binding("escape", "smart_dismiss", "Close", show=False)` and `action_smart_dismiss`: close the **topmost open overlay** in priority order — path bar (`#path-input-container` visible) → search (`search_active`) → recent (`show_recent`) → bookmarks (`show_bookmarks`) — and only `dismiss(None)` when none are open. Reuses the existing reactives/close helpers.

### TASK-431

#### AC#1 — Card filter accepts .webp; .md not a default card
**Local** to `personas_screen.py:3953-3972` (`_import_dialog_worker`). The "Character Cards" primary tester (`:3957-3961`) currently matches `.json/.md/.markdown/.png`. Change: add `.webp`; **remove `.md`/`.markdown` from the primary "Character Cards" tester** so a docs folder doesn't read as cards. Keep a dedicated **Markdown** sub-filter (`:3964`) for users who deliberately import a `.md` card, and add a **WEBP** sub-filter (or fold webp into the PNG/Images sub-filter). Filters must stay **callables** (not glob strings) to satisfy `tests/UI/test_file_picker_filters_callable.py`. Apply the same change to the legacy CCP route if it defines its own filter (`ccp_character_handler.py:929`).

#### AC#2 — "N hidden by filter" hint
**Additive** in `SearchableDirectoryNavigation._repopulate_display` (`enhanced_file_picker.py:605-654`): while iterating `self._entries`, count entries excluded **by the active `file_filter`** (distinct from search-hidden). Emit a new `FilterHiddenCountChanged` message (mirroring `SearchCountChanged`, `:509/654`), consumed in `EnhancedFileDialog` to render "N hidden by filter" into a small `Static` (a filter analog of `#search-no-match`, `:1065`). Zero → hidden. Low-risk; no behavior change to filtering itself.

#### AC#3 — Per-context start directory
Already implemented: `_get_last_directory` reads `filepicker.last_dir_{context}` (`:999`), `_save_last_directory` writes it on dismiss (`:1009/1757`), and character import passes `context="character_import"` (`personas_screen.py:3973`). Each context is isolated (no shared global last-dir). Work here is a **regression test** asserting that two different contexts keep independent last-dirs and that a saved `last_dir_character_import` is used on next open. If the review's "/Applications" observation reproduces, confirm the context string is distinct per surface (it is) — no code change expected beyond the test. Note the ordering caveat: a saved last-dir overrides a caller-passed `location=` (`:975-977`).

## Testing

Extend `tests/UI/test_enhanced_file_dialog_mount.py` (the established `pilot` + `query_one`/`.press()`/`action_*` harness; polls the threaded directory-load worker):
- **AC#1:** the file list's highlighted-option style resolves to `$ds-focus-bg` (assert the applied CSS/color, or a rendered-style probe).
- **AC#2:** single-click on a dir *highlights but does not navigate* (`self._location` unchanged); Enter / double-click / Go on a highlighted dir *descends*; single-click on a file fills `#filename-input`; Enter / double-click / Go on a file *returns* it. Update the existing `action_select()`-based tests to the new semantics.
- **AC#3 (path bar):** typing a full existing **file** path + Go dismisses with that file; a **dir** path cd's.
- **AC#4:** with the Recent overlay open, Esc closes the overlay and the picker stays; Esc again (nothing open) dismisses.
- **431 AC#1:** the Character Cards filter tester returns True for `.webp`, False for `.md`; filters remain callables (`test_file_picker_filters_callable.py` stays green).
- **431 AC#2:** a directory with N filter-excluded files reports "N hidden".
- **431 AC#3:** two contexts persist independent `filepicker.last_dir_*`; the saved value is used on next open.
- **Cross-consumer regression:** run `tests/UI/test_enhanced_file_dialog_mount.py`, `test_enhanced_filepicker.py`, `test_file_picker_filters_callable.py`, `test_file_picker_bookmarks_lazy.py`, `test_file_picker_action_tooltips.py`, `test_legacy_attach_picker.py`, `test_eval_file_picker_dialog.py`, `test_non_obscuring_focus_contract.py`, `test_mcp_workbench.py`, `test_chat_image_attachment.py`.
- **Live-verify:** in the real TUI, drive the Personas card import — visible cursor, single-click selects, double-click/Enter/Go opens (dir descends, file imports), Ctrl+L full file path opens, Esc closes the Recent overlay first; the card filter shows `.webp` and hides `.md`; a filtered folder shows the hidden count.

## Risks / mitigations

- **AC#2 app-wide activation change (highest risk):** no vendored edits; select/open split contained to the subclass; every picker test suite + the consumer-contract tests run; existing `action_select` tests updated deliberately (not deleted).
- **Cursor CSS clashing with the non-obscuring-focus contract:** use the sanctioned `$ds-focus-*` tokens and scope to `#file-list-pane`.
- **Path-bar returning a file in a Save dialog:** gate on `_should_return`; a Save returning an existing path is the intended "overwrite" affordance, but verify the Save flow's confirm path.
- **`Tests/` vs `tests/` duplication:** update whichever the build treats as source (verify) and keep them consistent.

## Non-goals

- Editing the vendored `textual_fspicker`.
- Redesigning Recent/Bookmarks/Search beyond the Esc layering.
- A separate "character import start dir" preference distinct from last-used (the per-context last-dir already covers the AC).
- Multi-select changes.
