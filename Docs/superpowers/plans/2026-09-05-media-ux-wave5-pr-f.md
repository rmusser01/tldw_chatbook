# Media UX fix wave 5 — PR F (select-mode reachability + focus visibility) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A keyboard-only user can enter Library ▸ Media select mode from anywhere, see where focus is, and build a selection with Down and Space immediately; a mouse user can toggle a row by clicking anywhere on it; the Reader pane's focus is visible in a plain-text capture, not only in colour.

**Architecture:** Select-mode entry already funnels through one seam, `_toggle_library_media_select_mode` (button and the `s` key), which ends with `_sync_library_canvas(self, "media", then=self._focus_library_media_items_pane)` — the pane, not a row, so Space's "focus must be on a media row" guard never holds. Task 1 retargets that `then` to `_focus_library_list_entry` (first still-checked row, else first row), keeps `Done` out of the `sort:` slot, and pins the keyboard path from the rail. Task 2 makes the whole row a select-mode toggle target and adds the Reader pane's glyph-level focus treatment (a heavy border on `:focus-within`, replacing PR B's colour-only tint) so `*:focus` outline behaviour is not the only carrier. Task 3 is the general root: after ANY Reader recompose, focus falls to the pane grip (task-31567) — restore it to the widget that held it.

**Tech Stack:** Python 3.12, Textual 8.x, pytest + pytest-asyncio; `Tests/UI/test_library_multiselect_media.py` (`_media_fake`, `_bind_media_mutation_seams`), `Tests/UI/test_library_media_render_fixes.py` (`_painted`, `_host`), `Tests/UI/test_library_media_reader_flow.py` (`_flow_app`), `Tests/UI/test_library_adaptive_reader_shell.py` (grip/pane geometry), `LibraryProductionCSSHarness` for painted focus captures.

**Spec:** `backlog/tasks/task-31631 - …select-mode-reachable-by-keyboard…md` (AC#1-#4), `backlog/tasks/task-31634 - …Reader-pane-focus-indication-must-not-be-colour-only.md` (AC#1-#2), `backlog/tasks/task-31567 - …restore-focus-after-any-Reader-recompose.md` (AC#1-#3); critique #5 P1 "Select mode is unreachable by keyboard" and the B §2/§3 focus measurements.

## Global Constraints

- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/media-wave5-f`, branch `fix/media-wave5-f` off dev. Every command: `cd <worktree> && PYTHONPATH=<worktree> /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest … -p no:cacheprovider`; absolute paths; UI test files in separate processes; every Bash call begins with the explicit `cd` and `git branch --show-current`.
- Compare failures against the base before claiming them (known: `test_library_ingest_canvas.py::test_progress_detail_paints_below_row…`, `test_library_ingest_retry_last` flake, the `test_library_shell.py` census in task-31249).
- No new `logger.*`. After any `BUNDLED_CSS` / TCSS edit: `python -m tldw_chatbook.css.build_css` then `python tldw_chatbook/css/check_bundle_sync.py` (exit 0); commit regenerated files. `screen_agentic_library.tcss` is the screen's own sheet (loaded by `LibraryScreen.CSS_PATH`), the bundle is separate — edit the source sheet, then rebuild.
- Focus rules that already cost time: widget-tier CSS loses to app-tier regardless of specificity; `*:focus { outline: solid }` in `core/_reset.tcss` PAINTS OVER content, so assert focus with painted-text probes (`_painted`), never region assertions; the Space binding is a priority binding gated to `.library-media-row` / `library-media-pane-grip` (PR B) — do not widen it.
- Five-key media summary contract frozen; review-set code and the Find focus token untouched; no new buttons on the media action toolbar; Analyze/Delete rows keep their positions (PR D's painted pins).
- Live verification: tmux (function `t() { tmux -L w5f "$@"; }` in every call, sleeps inside the call, `t kill-server` at the end), real config, ONE app instance only (stop if the "Another copy of tldw is already using this profile" toast appears); seeds via `MediaDatabase.add_media_with_keywords` with salted content, cleaned with `soft_delete_media`; capture plain AND `-e` (`capture-pane -pet`) for focus evidence.
- TDD per task; commit per task with the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; backlog task files are flipped by the controller.

---

### Task 1: Entering select mode focuses a row, and Done leaves the sort slot (task-31631 AC#1, AC#3, AC#4)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` — `_toggle_library_media_select_mode`: the entry branch's `then=` becomes `self._focus_library_list_entry` (read that helper at its definition: it prefers the first still-checked row when select mode holds a selection, else the first row); keep the exit branch as is. Verify `_focus_library_list_entry` focuses a `.library-media-row` Button (the marker/title button) and that `action_library_media_toggle_row_selection`'s "focused row" lookup finds it.
- Modify: `tldw_chatbook/Widgets/Library/library_media_canvas.py` — the select-mode toolbar: `Done` must not render in the cell range `sort:` occupies in browse mode. Read the two `_compose_*toolbar*` paths; put `Done` at the END of the summary row (`N selected  Select all N shown  Done`) or on its own cell range, whichever keeps the 36-cell floor (measure with a painted test).
- Test: `Tests/UI/test_library_multiselect_media.py` (fake: entering select mode calls the list-entry focus seam, not the pane seam), `Tests/UI/test_library_media_render_fixes.py` (painted, 235x52 and 100x30: from the rail press `s` → a row carries the focus outline in the painted capture → `Down` then `Space` yields `1 selected`; `Done`'s painted column range does not overlap `sort:`'s browse-mode range).

**Interfaces:**
- Produces: the invariant "select-mode entry focuses a row"; `Done` placement.

- [ ] Step 1: failing tests (fake seam ×1; painted keyboard path ×2 sizes; Done slot ×1).
- [ ] Step 2: run; confirm (pane focused, `0 selected` after Down+Space; Done overlaps sort:).
- [ ] Step 3: implement.
- [ ] Step 4: run `test_library_multiselect_media.py`, `test_library_media_render_fixes.py`, `test_library_media_toolbar_adapt.py`, `test_library_shell.py -k "select"` (compare to base).
- [ ] Step 5: live 235x52: from the rail, `s`, capture (plain + `-e`) → a row is outlined; `Down`, `Space` → `1 selected`; `s` → exits. Then 100x30 same.
- [ ] Step 6: commit `fix(library): select mode focuses a row on entry; Done leaves the sort slot (task-31631)`.

---

### Task 2: Whole-row toggle target, and a glyph-level Reader focus (task-31631 AC#2, task-31634 AC#1-#2)

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_media_canvas.py` — in select mode the row's title button (`open_viewer`, class `library-media-row`) must route through the same toggle as the marker: read `handle_library_media_row`'s select-mode branch (it toggles on ANY `.library-media-row` press) and find why a title click did not toggle live — most likely the title button is `can_focus`/`disabled` or is a `Static` in select mode; make the title a press target in select mode (same handler) without changing browse-mode open behaviour.
- Modify: `tldw_chatbook/css/screen_agentic_library.tcss` — `#library-media-viewer-content:focus-within` (PR B's `border: solid $accent` tint) becomes a glyph-level change: `border: heavy $accent` (the unfocused rule keeps its current style), so the plain capture differs between focused and unfocused states; then `python -m tldw_chatbook.css.build_css` + `check_bundle_sync`.
- Test: `Tests/UI/test_library_multiselect_media.py` (pressing the title button in select mode toggles the row; in browse mode it still opens the item), `Tests/UI/test_library_media_render_fixes.py` (painted: the Reader's top border row differs between unfocused and focused plain captures — assert on the border glyph characters, e.g. `┏`/`━` vs `┌`/`─` — at 235x52).

**Interfaces:**
- Consumes: Task 1's row-focus invariant.
- Produces: the Reader focus glyph contract (heavy border when focused).

- [ ] Step 1: failing tests (title toggle ×2; painted border glyph ×1).
- [ ] Step 2: run; confirm (title press opens or no-ops; border glyphs identical).
- [ ] Step 3: implement; rebuild the bundle; `check_bundle_sync` exit 0.
- [ ] Step 4: run `test_library_multiselect_media.py`, `test_library_media_render_fixes.py`, `test_library_media_reader_flow.py`, `test_library_media_reader_shell.py`, `test_library_shell.py -k "focus or select"` (compare to base).
- [ ] Step 5: live 235x52: F6 into the Reader → plain capture shows the heavy border; click a row title in select mode → `1 selected`.
- [ ] Step 6: commit `fix(library): whole-row select toggle; Reader focus is a heavy border, not a colour (task-31631, task-31634)`.

---

### Task 3: Restore focus after any Reader recompose (task-31567 AC#1-#3)

**Files:**
- Read: `tldw_chatbook/UI/Screens/library_screen.py` — `_sync_library_media_viewer_or_recompose`, `_arm_library_list_entry_focus`, `_focus_library_list_entry_if_current`, and every `refresh(recompose=True)` on the media path; `tldw_chatbook/Widgets/Library/library_media_viewer.py` / `library_media_content.py` — where the adaptive shell mounts the pane grips (`library-media-pane-grip`) and why they end up focused (Textual focuses the first focusable widget after a recompose when the previous focus target is gone).
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` — one seam `_restore_library_media_focus(previous: str | None)` called after each media-path recompose: records the focused widget's id (row id / `#library-media-viewer-content` / Find input / a toolbar button) BEFORE the recompose and re-focuses the same identity after it when it still exists, otherwise the list entry — never a grip. Wire it into `_sync_library_media_viewer_or_recompose` and the select-mode/receipt recomposes.
- Test: `Tests/UI/test_library_media_reader_flow.py` (focus on a row → open item → recompose → focus is on the Reader content, not the grip; focus in the Find input survives a mode switch; focus on a row survives a receipt repaint), `Tests/UI/test_library_adaptive_reader_shell.py` (Space on a focused row never collapses a pane — pin), painted at 235x52 and 100x30.

**Interfaces:**
- Consumes: Tasks 1-2.
- Produces: `_restore_library_media_focus` seam.

- [ ] Step 1: failing tests (three flows + the Space pin, both sizes).
- [ ] Step 2: run; confirm (focus on the grip after recompose).
- [ ] Step 3: implement the seam; wire every media-path recompose through it.
- [ ] Step 4: run `test_library_media_reader_flow.py`, `test_library_adaptive_reader_shell.py`, `test_library_media_render_fixes.py`, `test_library_multiselect_media.py`, `test_library_media_reader_shell.py`, `test_library_shell.py -k "focus or grip or recompose"` (compare to base).
- [ ] Step 5: live 235x52 and 100x30: open an item from a focused row → F6 ring → Find → Escape ladder; at no point does a grip carry focus (plain + `-e` captures).
- [ ] Step 6: commit `fix(library): restore focus after any Reader recompose instead of the pane grip (task-31567)`.
