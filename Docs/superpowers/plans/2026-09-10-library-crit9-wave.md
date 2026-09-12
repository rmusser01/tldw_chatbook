# Library critique-9 fix wave Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 26 in-scope findings of Library critique #9 (tasks 32210, 32212–32231, 32233–32237 — 32232 is already fixed on PR #2568 and 32211 is on the peer's PR #2557, both excluded) in seven independent branches that each ship as their own PR against `dev`.

**Architecture:** Each task below is one branch/worktree and one PR. Work stays inside the Library surface (`tldw_chatbook/UI/Screens/library_screen.py`, `tldw_chatbook/Widgets/Library/*.py`, `tldw_chatbook/Library/*.py`, `tldw_chatbook/UI/Library_Modules/*.py`, the split Library TCSS, `Docs/User_Guide/library*`), plus one vendored dialog (`tldw_chatbook/Third_Party/textual_fspicker/file_dialog.py`). Behaviour changes are test-first against the existing Library UI harnesses; every task ends with a live tmux check on an isolated scratch profile.

**Tech Stack:** Python 3.12, Textual 8.2.8, pytest, Backlog.md CLI, tmux.

**Spec:** `.impeccable/critique/2026-09-10T14-50-24Z__tldw-chatbook-ui-screens-library-screen-py.md` (the critique snapshot; the register rows are the binding requirements) and the task files `backlog/tasks/task-32210 … task-32237` (acceptance criteria).

**Baseline read for this plan:** `dev` @ `e6cb464239` (Merge PR #2549). The snapshot was taken at `02374bf66a`, BEFORE #2543 (Folder files as a mode), #2547 (editor-keys group) and #2549 landed — every Notes finding must be re-verified on the current tip before it is fixed.

## Global Constraints

- Work only inside your assigned worktree (`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/crit9-<group>`); every shell command starts with `cd <worktree> &&` because the shell cwd resets between calls. Never touch the main checkout or another group's worktree.
- Python: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` (3.12). Run pytest FROM the worktree cwd (`cd <worktree> && …/.venv/bin/python -m pytest <file or node id> -q -p no:cacheprovider`). Never run the whole suite (over an hour) and never use `-k` filtering as verification; run whole files or explicit node ids. `Tests/UI/test_library_shell.py` has ~226 pre-existing failures on dev: if you use it, compare failing NAME sets before and after your change (run it once on a clean `git stash` state or against `origin/dev`), and report only new names.
- Other sessions run pytest on this Mac; POSIX semaphores are exhausted, so `multiprocessing.Pool` fails with `[Errno 28]` and every local media Import fails in the live app. Do not try to fix or work around the host; if a test needs a process pool, mark it and move on.
- TDD: write the failing test, run it and show it failing, implement, run it passing. New tests go in the most specific existing `Tests/UI/test_library_*.py` file for the area, or a new `Tests/UI/test_library_crit9_<area>.py`; never into the 19k-line `test_library_shell.py`.
- CSS: edit the component source under `tldw_chatbook/css/components/` (Library rules live in `_agentic_terminal.tcss`, split into `screen_agentic_library.tcss` by the build), then run `cd <worktree> && …/.venv/bin/python -m tldw_chatbook.css.build_css` and commit the regenerated bundle files alongside. Widget `DEFAULT_CSS`/`BUNDLED_CSS` must parse standalone and never use ancestor-scoped bare-type subject rules (`Foo > Vertical`).
- Git: stage explicit paths only (never `git add -A`); commit after each green step; do NOT push and do NOT open or merge PRs, the controller does that after review. Commit trailer: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Backlog hygiene from the worktree root: `backlog task edit <id> -s "In Progress" --plan "<steps>"` before the first code change; at the end tick every AC (`- [x]`) you satisfied, add `--notes "<Implementation Notes>"`, and set `-s Done`. Leave an AC unticked with a note if you could not satisfy it.
- Docs: every user-visible change updates the matching `Docs/User_Guide/library*.md` page and appends a `*Verified against fix/library-crit9-<group> — 2026-09-10 (task-NNNNN: …)*` stamp in the page's existing stamp style.
- Live verification (required before reporting): `tmux -L crit9-<group> new-session -d -x 235 -y 52 "cd <worktree> && TLDW_CONFIG_PATH=<profile>/config.toml PYTHONPATH=<worktree> …/.venv/bin/python -m tldw_chatbook.app"`, `sleep 15`, drive with `send-keys`, observe with `capture-pane -p` (`-e` for colour). Profiles: `<SCRATCH>/crit9/wave/<group>/power/config.toml` (seeded: 11 media, 6 conversations, 7 notes incl. a 35 KB one, 5 prompts, 2 skills, inbox/ and file_notes/ folders) and `…/fresh/config.toml` (empty; first launch shows the setup wizard: Esc, Tab, Enter skips it). `SCRATCH` = `/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/92b26568-46c4-4abe-9da5-9676d5282276/scratchpad`. Ctrl+digit keys cannot be sent through tmux: reach Library with `C-p`, type `Switch to Library`, `Down`, `Enter`. Mouse clicks: `send-keys -l $'\x1b[<0;COL;ROWM'` then `…ROWm` (1-based, column by code points, not bytes). Quit with `C-q` then `tmux -L crit9-<group> kill-server`. One instance per profile at a time.
- Copy rules: blocked or disabled states carry a text reason and a next step on the same line; never colour-only meaning; no raw errno, UUID or ISO timestamp reaches the user.
- Scope: implement the acceptance criteria of your tasks and nothing else. If an AC needs a product decision you cannot make, implement the rest, leave that AC unticked, and say so in the report.

### Additional constraints for this wave

- **Make your profile first.** The seeded profiles are per-group; sharing one across two live branches corrupts its SQLite. Run exactly:
  ```
  SCRATCH=/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/92b26568-46c4-4abe-9da5-9676d5282276/scratchpad
  mkdir -p "$SCRATCH/crit9/wave"
  cp -R "$SCRATCH/crit9/A" "$SCRATCH/crit9/wave/<group>"
  sed -i '' "s#/crit9/A/#/crit9/wave/<group>/#g" "$SCRATCH/crit9/wave/<group>/power/config.toml" "$SCRATCH/crit9/wave/<group>/fresh/config.toml"
  ```
  If `$SCRATCH/crit9/A` is gone, reseed with `…/.venv/bin/python "$SCRATCH/seed_power_profile.py" "$SCRATCH/crit9/wave/<group>/power" crit8_power` (the `users_name` argument must stay `crit8_power`; the profile's `*_db_path` keys are absolute and already point at `data/crit8_power/`).
- **Never `git stash` bare.** Two crit8 branches lost work to a stash that a sibling worktree popped. If you must park work use `git stash push -m "crit9-<group>-<why>" -- <explicit paths>` and pop it by name in the same shell.
- **`_agentic_terminal.tcss` is shared, in three disjoint ranges.** Task 1 edits `.library-toolbar-count` (lines 130–132) and nothing else; Task 3 edits the rail search + heading block (lines ~2429–2455) and nothing else; Task 6 appends the density block at the end of the Library section and nothing else. Stay inside your range — never reflow, reorder or reformat a neighbouring rule. The regenerated `screen_agentic_library.tcss` / bundle files WILL conflict when the second PR lands: resolve by taking `dev`'s bundle, re-running `python -m tldw_chatbook.css.build_css`, and committing the regenerated result. Never hand-edit a bundle file.
- **`Docs/User_Guide/` pages are shared, source files are not.** More than one branch appends a stamp to `library.md` and to `library/media-and-conversations.md`. Append your stamp as a NEW line at the end of the page's existing stamp block; on a landing conflict keep BOTH stamps (`$SCRATCH/resolve_hunks.py` from the crit8 wave does this mechanically). Source-file ownership is exclusive: if a step would touch a `.py` outside your **Files** list, stop and report NEEDS_CONTEXT instead of editing it. Two single-line carve-outs are named explicitly in Tasks 4 and 5 — honour them exactly.
- **Grep for the pin before you fix.** This screen's live critiques mis-attribute the cause about 40% of the time, and several behaviours the snapshot calls bugs are prior design decisions with a pinning test. Before changing behaviour, `grep -rn "<the id or copy string>" Tests/` and read what the pin asserts. If a pin asserts the opposite of your fix, the plan says which way to go; if the plan does not, report NEEDS_CONTEXT rather than weakening the pin. Never delete or loosen an assertion to make a change pass.
- **Preflight before you report done:** `cd <worktree> && ./scripts/preflight.sh` (~35 s, installs nothing). It runs the same four checks as the required `Derived artifacts reproduce from their sources` CI job.

---

### Task 1: Media list — chooser cursor, empty toolbar, select strip (group `media-list`, tasks 32210, 32213, 32214, 32227)

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_choice_strip.py` (new `LibraryChoiceOptionList`)
- Modify: `tldw_chatbook/Widgets/Library/library_media_canvas.py` (type chooser `OptionList` 1228–1236, sort chooser `OptionList` 1254–1262, the `fresh_zero` early-return block 1022–1046, the select-strip count `Static` ~1311)
- Modify: `tldw_chatbook/Library/library_media_state.py` (`empty_copy` 1174–1186)
- Test: `Tests/UI/test_library_crit9_media_list.py` (new)
- Docs: `Docs/User_Guide/library/media-and-conversations.md` (the type/sort chooser rows in the control table at ~345, and the filtered-empty-page rule)

**Interfaces:**
- Produces: `LIBRARY_CHOICE_CURSOR = "█ "` and `class LibraryChoiceOptionList(OptionList)` in `library_choice_strip.py`. The subclass keeps a `█ ` prefix on exactly the highlighted option's prompt and strips it from all the others, on mount and on every `highlighted` change. `Option._set_prompt` mutates in place, so the `choice_value` attribute the pick handlers read survives.
- Consumes: `LIBRARY_CHOICE_ACTIVE_MARKER` (`✓`) from `tldw_chatbook/Library/library_shell_state.py:170` — unchanged; the highlighted active option reads `█ ✓ All types`.
- Consumes: the painted-cell harness idiom in `Tests/UI/test_library_row_focus_cue_t31983.py` (`app.screen._compositor.render_strips()`, `_THICK_LEFT_GLYPH = "█"`) and `Tests/UI/test_library_media_render_fixes.py::_painted`.

- [ ] **Step 1 (32210): failing test.** In `Tests/UI/test_library_crit9_media_list.py`, build the media host the way `Tests/UI/test_library_media_render_fixes.py` does (`_build_media_test_app` + `_seed_conversations` + `LibraryProductionCSSHarness`; import them from that module — it is green on dev, unlike `test_library_shell.py`), open the Media list, press `#library-media-type-filter`, wait for `#library-media-type-choices`, then:

```python
@pytest.mark.asyncio
async def test_the_type_chooser_marks_its_cursor_with_the_house_bar():
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        screen.query_one("#library-media-type-filter", Button).press()
        choices = await _wait_for_selector(screen, pilot, "#library-media-type-choices")
        choices.highlighted = 1
        await pilot.pause()
        prompts = [str(choices.get_option_at_index(i).prompt) for i in range(choices.option_count)]
        assert prompts[1].startswith("█ "), prompts
        assert [p for p in prompts if p.startswith("█ ")] == [prompts[1]], prompts
        painted = _painted(host, choices.region)
        assert "█" in painted, painted
```

- [ ] **Step 2: run it and show it failing** (`… -m pytest Tests/UI/test_library_crit9_media_list.py -q -p no:cacheprovider`). Expected FAIL: `AssertionError` on `prompts[1].startswith("█ ")` — today the only cue is the OptionList's `option-list--option-highlighted` background (measured live at 1.09:1, register row 6).

- [ ] **Step 3: implement.** Append to `tldw_chatbook/Widgets/Library/library_choice_strip.py`:

```python
from textual.widgets import OptionList

#: task-32210 (critique #9 row 6): the vertical choosers marked their cursor
#: with an OptionList background swap measured at 1.09:1 -- colour-only, and
#: invisible in a plain-text capture. This is the same `█` left-edge cue the
#: list rows carry (task-31983), rendered into the prompt because CSS cannot
#: target one option.
LIBRARY_CHOICE_CURSOR = "█ "


class LibraryChoiceOptionList(OptionList):
    """An OptionList whose highlighted option carries the house `█ ` cursor."""

    def on_mount(self) -> None:
        self._paint_choice_cursor()

    def watch_highlighted(self, highlighted: int | None) -> None:
        super().watch_highlighted(highlighted)
        self._paint_choice_cursor()

    def _paint_choice_cursor(self) -> None:
        for index in range(self.option_count):
            option = self.get_option_at_index(index)
            base = str(option.prompt).removeprefix(LIBRARY_CHOICE_CURSOR)
            wanted = (
                f"{LIBRARY_CHOICE_CURSOR}{base}"
                if index == self.highlighted
                else base
            )
            if str(option.prompt) != wanted:
                self.replace_option_prompt_at_index(index, wanted)
```

  In `library_media_canvas.py` replace `OptionList(` with `LibraryChoiceOptionList(` at both chooser sites (the `#library-media-type-choices` construction at 1228 and the `#library-media-sort-choices` construction at 1254) and import it (`from tldw_chatbook.Widgets.Library.library_choice_strip import LibraryChoiceOptionList`). Leave `library_media_trash_canvas.py:307` alone — it is not in this task's ACs and not in this branch's Files list. Run Step 1's test green, then run `Tests/UI/test_library_media_render_fixes.py::test_type_chooser_paints_every_option` and `Tests/UI/test_library_choice_strips.py` to prove the `✓` marker and the pick handlers still work.

- [ ] **Step 4 (32213): failing test.** Today `library_media_canvas.py` returns early inside the `fresh_zero` branch — `if self.canvas.query: return` at ~1032 — so a filter miss loses `type:`, `sort:`, `Export…`, `Trash`, `Select` and `Review these`. Write:

```python
@pytest.mark.asyncio
async def test_a_zero_result_filter_keeps_the_type_facet_and_names_it():
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _apply_media_type(screen, pilot, "pdf")      # press the chooser, pick "pdf"
        filter_box = screen.query_one("#library-media-filter", Input)
        filter_box.value = "zzzznomatch"
        await pilot.press("enter")
        await _wait_for_condition(pilot, lambda: not screen.query("#library-media-row-0"))
        assert screen.query("#library-media-type-filter"), "the type facet must survive a 0-result page"
        assert screen.query("#library-media-sort")
        assert screen.query("#library-media-trash-open")
        status = screen.query_one("#library-media-status", Static)
        assert str(status.content) == (
            'No media of type \'pdf\' matched “zzzznomatch” in titles, content or keywords.'
        ), str(status.content)
        assert screen.query("#library-media-empty-clear-type"), "a type filter needs its reset"
        assert not screen.query("#library-media-empty-import"), "a filter miss never suggests Import (task-31224)"
```

- [ ] **Step 5: run it failing, then implement.** Two changes:
  1. `tldw_chatbook/Library/library_media_state.py`, the `empty_copy` ladder at 1174–1186 — the query branch must name the active type when one is set:

```python
    empty_copy = ""
    if not rows:
        if result.scope.query and result.scope.media_type is not None:
            empty_copy = (
                f"No media of type '{result.scope.media_type}' matched "
                f"“{result.scope.query}” in titles, content or keywords."
            )
        elif result.scope.query:
            empty_copy = (
                f"No media matched “{result.scope.query}” "
                "in titles, content or keywords."
            )
        elif result.scope.media_type is not None:
            empty_copy = f"No media of type '{result.scope.media_type}'."
        else:
            empty_copy = LIBRARY_MEDIA_EMPTY_COPY
```

  2. `library_media_canvas.py`, the `fresh_zero` block: delete both `return` statements and let composition fall through to the toolbar. Keep the task-31224 rule intact by making the recovery button conditional rather than the whole block: render `Show all types` (`#library-media-empty-clear-type`) whenever `self.canvas.active_type is not None`, and `Import media` (`#library-media-empty-import`) only when `self.canvas.active_type is None and not self.canvas.query`. The `select_disabled = rendered_count == 0 and not select_mode` line below already disables `Select` correctly on a zero-row page, so no gate changes are needed.
  Re-run the new test plus `Tests/UI/test_library_media_toolbar_adapt.py` and `Tests/UI/test_library_multiselect_media.py` (both grep-hit `library-media-empty-clear-type`) and report their before/after result.

- [ ] **Step 6 (32227): failing test + fix.** The select strip paints `2 selected┃ Select all` — the count `Static` (`#library-media-selected-count`, `library_media_canvas.py:1310-1315`) and `#library-media-select-all`'s heavy focus border share a cell.

```python
@pytest.mark.asyncio
async def test_the_select_count_never_touches_the_first_action():
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _enter_select_mode(screen, pilot)
        count = screen.query_one("#library-media-selected-count", Static)
        first = screen.query_one("#library-media-select-all", Button)
        assert first.region.x - count.region.right >= 1, (count.region, first.region)
```

  **Fix in the shared class, not inline.** The comment above that `Static` (`library_media_canvas.py:1300-1309`) records a prior design decision: this counter is styled "as the general rule via the shared `library-toolbar-count` class … not a per-widget Python one-off, so every canvas's counter is covered by one declaration." Honour it — edit `tldw_chatbook/css/components/_agentic_terminal.tcss` lines 130–132 and nothing else in that file:

```
.library-toolbar-count {
    width: auto;
    /* task-32227 (critique #9 row 26): the count sat flush against the next
       action's focus border ("2 selected┃ Select all"). One cell, in the one
       declaration every canvas's counter already shares. */
    margin: 0 1 0 0;
}
```

  Then `python -m tldw_chatbook.css.build_css` and commit the regenerated bundle. **This is the only rule this branch may touch in `_agentic_terminal.tcss`** — Task 3 owns 2429–2455 and Task 6 owns the density blocks; the three ranges are disjoint. Re-run `Tests/UI/test_library_media_toolbar_adapt.py` and `Tests/UI/test_library_multiselect_conversations.py` (the class is shared with Conversations at rule 1877) and report failing-name sets before and after.

- [ ] **Step 7 (32214): SPIKE, decide before you fix.** `Tests/UI/test_library_shell.py:28194::test_library_media_list_focuses_first_row_and_arrow_keys_move_it` is a PIN from task-2856 AC1/AC5 that asserts the OPPOSITE of the report: entering the Media list must focus `#library-media-row-0`. Assessor B's repro had earlier keypresses in the session. Do this, in order:
  1. Launch the seeded power profile on socket `crit9-media-list`, `sleep 15`.
  2. Reach Library with `C-p`, `Switch to Library`, `Down`, `Enter`. Send NO other key.
  3. Click the rail's Media row with the mouse escape sequence only (`send-keys -l $'\x1b[<0;COL;ROWM'` then `…ROWm`).
  4. `capture-pane -e -p` immediately, then send `Down` once and capture again.
  5. **Decision rule.** If the heavy focus frame is on `type: All types` and `Down` moves nothing: it reproduces — fix by making the entry focus arm survive that path (grep `_focus_library_list_entry` and `entry-focus arm` in `library_screen.py`; note that `Tests/UI/test_library_crit8_keyboard.py:210` documents that a KEY press disarms the entry-focus arm, which is the most likely mechanism) and add a regression test to `Tests/UI/test_library_crit9_media_list.py`. Report the diff. If instead row 0 carries the `█` bar and `Down` moves to row 1: **close 32214 by evidence** — `backlog task edit 32214 -s Done --notes "Not reproducible from a clean entry at dev e6cb464239. Live capture on the seeded power profile (crit9-media-list): rail-click on Media, no prior keys, focus lands on #library-media-row-0 with the █ bar and Down moves to row 1. Assessor B's repro carried earlier keypresses, which disarm the entry-focus arm (see Tests/UI/test_library_crit8_keyboard.py:210). The pin test_library_shell.py::test_library_media_list_focuses_first_row_and_arrow_keys_move_it holds. Captures: <SCRATCH>/crit9/wave/media-list/caps/32214-*.txt"` and tick AC#1. Save both captures either way.

- [ ] **Step 8: live-verify** 32210 / 32213 / 32227 on the seeded profile at 235x52 and 100x30 (socket `crit9-media-list`), with `capture-pane -e` proving the `█` moves with the arrow keys in the type chooser. Captures to `<SCRATCH>/crit9/wave/media-list/caps/`.

- [ ] **Step 9: docs + backlog + commits.** In `Docs/User_Guide/library/media-and-conversations.md` update the `"type: All types"` and `"sort: Newest"` rows in the control table (~345) to say the highlighted row carries a leading `█` and the active value carries `✓`, and correct the filtered-empty-page paragraph to describe the toolbar staying put. Stamp. One commit per id: `fix(library-media): the type and sort choosers show their cursor (task-32210)`, `fix(library-media): a 0-result filter keeps its toolbar and names the type (task-32213)`, `fix(library-media): one cell between the select count and the first action (task-32227)`, and for 32214 either `fix(library-media): focus row 0 on every list entry (task-32214)` or a docs/backlog-only commit `chore(library-media): close task-32214 by evidence`.

---

### Task 2: Media reader — Markdown rule, More strip, Escape chip, Undo (group `media-reader`, tasks 32234, 32237, 32222, 32224)

**Files:**
- Modify: `tldw_chatbook/Library/library_media_viewer_state.py` (`_MARKDOWN_MEDIA_TYPES` 29–31, `_is_markdown_media` 247–255)
- Modify: `tldw_chatbook/Widgets/Library/library_media_viewer.py` (`RENDERED_VIEW_NOTE` 66, the More `ItemGrid` at ~439–443)
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` — **only** `_library_media_escape_label` (29780) and its docstring reference at 1201
- Modify: `tldw_chatbook/UI/Library_Modules/library_media_controller.py` (the delete-receipt Undo path; grep `undo` / `_library_media_delete_receipt_undo_failure` at 1880)
- Test: `Tests/UI/test_library_crit9_media_reader.py` (new); the two existing red pins in `Tests/UI/test_library_media_render_fixes.py` must go green untouched
- Docs: `Docs/User_Guide/library/media-and-conversations.md`

**Interfaces:**
- Consumes: `looks_like_markdown_content(content)` (`library_media_viewer_state.py`, capped at `MAX_MARKDOWN_SNIFF_CHARS`/`MAX_MARKDOWN_SNIFF_LINES`) — the sniff already exists and is already bounded; this task only removes the type gate in front of it.
- Consumes: `Tests/UI/test_library_media_render_fixes.py::_painted`, `_four_action_host`, `_open_reader_more`, `_reader_row_tops`.

- [ ] **Step 1 (32237): run the two red pins and record the failure.**
  ```
  cd <worktree> && …/.venv/bin/python -m pytest \
    "Tests/UI/test_library_media_render_fixes.py::test_more_opens_one_row_and_moves_the_reader_body_by_one" \
    "Tests/UI/test_library_media_render_fixes.py::test_more_stays_compact_at_the_narrow_reader_width" \
    -q -p no:cacheprovider
  ```
  Expected on dev (measured 2026-09-10): both FAIL with
  `assert 'Move to trash' in '  Edit metadata   Open original   Open manager        Move to …'`.
  **Root cause, already proven, do not re-diagnose:** `#library-media-reader-more-actions` is an `ItemGrid(min_column_width=15, max_column_width=16)`. `GridLayout.arrange` sets `container_width = min(len(children), width // max_column_width) * max_column_width`, then `columns = container_width // min_column_width` — so every column is exactly 16 cells. `#library-media-reader-more-actions > .library-media-action-danger` (`_agentic_terminal.tcss:3880`, task-31980) adds `margin: 0 0 0 2`, which comes out of the button's own box: `Move to trash` gets a 14-cell region where its auto width needs 15, and Textual wraps the label to `Move to` / `trash` with only line 1 visible at height 1. Measured: with `margin: 0` the same button gets width 15 and paints `Move to trash` in full.

- [ ] **Step 2 (32237): implement, keeping the task-31980 separation.** In `library_media_viewer.py` change the `ItemGrid` construction to:

```python
            with ItemGrid(
                id="library-media-reader-more-actions",
                classes="ds-toolbar",
                # task-32237: the column has to hold the longest label (13),
                # the Button's own two auto-width cells, and the danger
                # action's 2-cell separation (`.library-media-action-danger`,
                # task-31980) -- 16 cells cut "Move to trash" to "Move to"
                # because that margin is taken out of the button's box.
                min_column_width=17,
                max_column_width=17,
            ):
```

  Do NOT touch the TCSS margin rule and do not weaken either pin. Verified geometry with 17/17: 235x52 → one row, `  Edit metadata    Open original    Open manager       Move to trash`, body delta 1; 100x30 → two rows, all four labels whole, body delta 2; 60x24 → two rows, all four labels whole. Re-run both pins green, plus `Tests/UI/test_library_media_render_fixes.py::test_more_toggle_leaves_focus_on_the_more_button` and `::test_more_reads_as_an_open_disclosure_while_it_is_open`.

- [ ] **Step 3 (32237): add the 60x24 coverage AC#1 asks for.** In `Tests/UI/test_library_crit9_media_reader.py`:

```python
@pytest.mark.asyncio
async def test_the_more_strip_paints_every_label_at_the_narrowest_stage():
    host = _four_action_host()
    async with host.run_test(size=(60, 24)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        actions = await _open_reader_more(screen, pilot)
        painted = _painted(host, actions.region)
        for label in ("Edit metadata", "Open original", "Open manager", "Move to trash"):
            assert label in painted, painted
```

- [ ] **Step 4 (32234): failing test.** `_is_markdown_media` returns `False` for every type outside `_MARKDOWN_MEDIA_TYPES` before the sniff runs, so a `document` starting `# Roadmap sync` paints literal hashes and Info claims `No Markdown formatting to render — showing the stored text`.

```python
@pytest.mark.asyncio
async def test_a_document_with_real_markdown_renders_and_drops_the_false_note():
    host = _markdown_document_host()   # one item, "type": "document", content "# Roadmap sync\n\nBody.\n"
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        assert screen.query("#library-media-viewer-content-markdown"), list(
            screen.query_one("#library-media-viewer-content").children
        )
        painted = _painted(host, screen.query_one("#library-media-viewer-content").region)
        assert "Roadmap sync" in painted and "# Roadmap" not in painted, painted
        assert RENDERED_VIEW_NOTE not in _painted(host, screen.query_one("#library-media-viewer").region)
```

  plus a unit test in the same file: `assert _is_markdown_media("document", "# Roadmap sync\n\nBody.\n") is True` and `assert _is_markdown_media("document", "Plain prose with no markers.\n") is False`.

- [ ] **Step 5 (32234): implement — delete the allowlist gate.** In `library_media_viewer_state.py`:

```python
def _is_markdown_media(media_type: str, content: str) -> bool:
    """Whether the Reader defaults to the rendered view for this item.

    task-32234 (critique #9 row 3): the media-type allowlist used to run
    BEFORE the content sniff, so a `document` whose text starts `# Roadmap
    sync` painted its hashes literally while Info said "No Markdown
    formatting to render" -- a sentence the user could disprove by looking
    at the screen. The content is the only honest evidence, and the sniff
    is already bounded (MAX_MARKDOWN_SNIFF_CHARS/LINES), so it now decides
    alone for every type. `media_type` is kept in the signature: the
    "Type: markdown (stored as plaintext)" line in the caller still needs it.
    """
    return looks_like_markdown_content(content)
```

  Delete `_MARKDOWN_MEDIA_TYPES` and its comment block only if nothing else imports it — check with `grep -rn "_MARKDOWN_MEDIA_TYPES" tldw_chatbook Tests` first. `tldw_chatbook/Chunking/auto_selection.py:144` mentions it in a comment only; `Tests/UI/test_library_media_render_fixes.py:1623` and `Tests/UI/test_library_media_reader_scroller_resolution.py:199` reference it in COMMENTS and force types for reasons that still hold (their fixtures are plain prose / already `plaintext`), so both stay green — verify by running those two files and reporting the before/after failing-name sets. Update the two stale comments to say the sniff now decides alone. `RENDERED_VIEW_NOTE` keeps its exact wording — with the type gate gone it is now true whenever it is shown, which is the whole point.

- [ ] **Step 6 (32222): reconcile the Escape chip with the guide.** Read `_library_media_escape_label` (`library_screen.py:29780`) and write down what it returns in each viewer state; the pins in `Tests/UI/test_library_media_reader_flow.py:1440/1448/1676/1745` already fix three of them (`"close"`, `"focus Items"`). The live chip in the plain viewer reads `esc focus Library`. Then fix the DOCS to the code, not the code to the docs — the chip is pinned and the guide is not:
  - `Docs/User_Guide/library/media-and-conversations.md:577` and `:747` are the two contradicting claims; rewrite both to the single sentence **"Escape closes transient Reader state first — the Find bar, then the More strip — and then steps out of the Reader to the Items list; the footer chip always names the next step it will take."** and make the chip's own label the authority by quoting it.
  - Add a test that the guide and the code cannot drift again: assert the chip text for the plain viewer state equals the string the guide quotes, in `Tests/UI/test_library_crit9_media_reader.py`.
  If the live chip does NOT match what `_library_media_escape_label` returns for that state, that is a code bug — fix the code, keep the pins, and say so in the notes.

- [ ] **Step 7 (32224): Undo must not re-date the item.** First reproduce in a test:

```python
@pytest.mark.asyncio
async def test_undo_of_a_bulk_delete_keeps_the_stored_modified_time():
    # seed two items with distinct, OLD last_modified values through a real MediaDatabase
    # select both, delete, press the receipt's Undo, then assert the DB rows' last_modified
    # are byte-identical to what they were before the delete, and that the list order is unchanged.
```

  Trace the write: from the receipt's Undo handler in `library_media_controller.py` into the media DB restore. **Decision rule:** if the restore is a single `UPDATE … SET is_trash = 0, deleted = 0 …` that also stamps `last_modified`, drop `last_modified` from that statement's SET list (a restore is not an edit) and keep the version bump — that is the root-cause fix and satisfies AC#1's first clause. If the restore goes through a shared helper that stamps `last_modified` for every writer (grep its other callers before editing — a guard in the shared helper that only this path skips is a bigger diff than one explicit column list here), do NOT change the shared helper: instead take AC#1's second clause and change `Docs/User_Guide/library/media-and-conversations.md`'s "restore never rewrites the item" sentence to **"Restore brings the item back and marks it changed now, so it returns at the top of a Newest sort."**, tick AC#1, and record which branch you took and why in the Implementation Notes.

- [ ] **Step 8: live-verify** on the seeded profile at 235x52, 100x30 and 60x24 (socket `crit9-media-reader`): open the reader, press More, capture the strip at all three sizes; open a `document`-typed item with Markdown and capture the Read tab and the Info tab; delete two items and Undo, capturing the list order before and after. Captures to `<SCRATCH>/crit9/wave/media-reader/caps/`.

- [ ] **Step 9: docs + backlog + commits.** One commit per id: `fix(library-media): the More strip paints Move to trash in full (task-32237)`, `fix(library-media): the rendered-view rule sniffs content for every type (task-32234)`, `docs(library-media): one Escape sentence that matches the chip (task-32222)`, and the 32224 commit named for the branch you took.

---

### Task 3: Rail — search row, heading, carry-over, Details, guide tour (group `rail`, tasks 32212, 32220, 32226, 32230, 32219)

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_rail.py` (`compose` 735–815: heading row 743–759, search row 769–785; `_compose_details_body_children` 817–870)
- Modify: `tldw_chatbook/UI/Library_Modules/library_rag_search_controller.py` (`handle_library_search_changed` 743, `_patch_sibling_library_search_input` use at 864)
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` — **only** `_library_details_lines` (14268–14297), `_library_db_sizes_line` (14300–14360) and `_workspace_handoff_summary_label` (13143–13190)
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` — **only** the block at 2429–2455 (`#library-rail-search-row`, `#library-search-clear`, `#library-search-input`) plus one new `#library-rail-heading-label` rule immediately after it
- Test: `Tests/UI/test_library_crit9_rail.py` (new)
- Docs: `Docs/User_Guide/library.md`

**Interfaces:**
- Consumes: `Tests/UI/test_library_row_focus_cue_t31983.py`'s painted-frame idiom (`APP_STYLESHEETS`, `app.screen._compositor.render_strips()`) — the frame columns must be read through the PRODUCTION bundle, not a widget-only harness.
- Consumes: `_fit_title_no_mid_word_cut` (`library_rail.py:707`) — already the rail's no-mid-word-cut rule for row titles.
- Produces: `_library_details_lines` returns per-source DB-size values instead of one joined line (see Step 6).

- [ ] **Step 1 (32212): measure before you fix.** Write the painted-frame test FIRST and let it record reality:

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(235, 52), (100, 30), (60, 24)])
async def test_the_rail_search_row_never_pushes_the_canvas_frame(size):
    host = _library_host()
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        rail = screen.query_one("#library-rail")
        row = screen.query_one("#library-rail-search-row")
        box = screen.query_one("#library-search-input")
        clear = screen.query_one("#library-search-clear")
        assert row.region.right <= rail.region.right, (row.region, rail.region)
        assert clear.region.right <= rail.region.right, (clear.region, rail.region)
        assert box.region.right <= clear.region.x, (box.region, clear.region)
        canvas = screen.query_one("#library-canvas")
        assert canvas.region.x >= rail.region.right, (canvas.region, rail.region)
```

  Run it and paste the actual regions into your report. The critique's inference ("the clear button added by task-32069 widened the row past its pane") is INFERRED, not proven — the regions are the evidence.

- [ ] **Step 2 (32212): implement the smallest change the measurement justifies.** In the TCSS block at 2429–2455:

```
#library-rail-search-row {
    height: 3;
    width: 100%;
    /* task-32212 (critique #9 row 9): the row carries a fixed-width clear
       button beside a 1fr Input; without an explicit overflow rule a child
       that cannot shrink pushes the rail's own frame two cells right and
       clips the canvas border. */
    overflow-x: hidden;
}

#library-search-input {
    width: 1fr;
    /* task-32212: a 1fr child still refuses to go below its minimum, so the
       Input has to be allowed to give up cells to the clear button. */
    min-width: 0;
    height: 3;
    …
}
```

  Re-run Step 1's test at all three sizes. If the regions are still out of the rail, the cause is upstream of these rules: report the measured numbers, then try the second candidate — dropping `#library-search-clear`'s own `border: tall` (3 cells become 1 for the same `x` glyph) — and re-measure. Do NOT remove the clear button; task-32069 added it deliberately and `library_screen.py:31670` handles it.

- [ ] **Step 3 (32220): failing test + one-line fix.**

```python
@pytest.mark.asyncio
async def test_the_rail_heading_is_never_cut_mid_word():
    host = _library_host()
    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        painted = _painted(host, screen.query_one("#library-rail-heading").region)
        assert "Navigati" not in painted or "Navigation" in painted, painted
        assert "Navigat…" in painted or "Navigation" in painted, painted
```

  Fix: add, immediately after the search-row block in `_agentic_terminal.tcss`:

```
/* task-32220 (critique #9 row 17): at the 22-column compact rail the heading
   was cut mid-word to "Navigati". AC#1 accepts an ellipsis; the label keeps
   its full word wherever it fits. */
#library-rail-heading-label {
    text-overflow: ellipsis;
}
```

  Rebuild the bundle (`python -m tldw_chatbook.css.build_css`) and commit the regenerated files.

- [ ] **Step 4 (32226): the box is already emptied — the QUERY is what carries.** `_library_rail_search_value` (`library_screen.py`, in the `#library-search-input` handler block ~31584) already returns `""` off the Search/RAG row, so task-32069 did ship. What the reviewer saw is `_patch_sibling_library_search_input` + `handle_library_search_changed`: typing in the rail box on ANY canvas writes `_rag_search_state.query` on every `Input.Changed`, so the text reappears in the Search/RAG query box later. Failing test:

```python
@pytest.mark.asyncio
async def test_unsubmitted_rail_text_never_seeds_the_rag_query_box():
    host = _library_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-filter")
        box = screen.query_one("#library-search-input", Input)
        box.focus()
        await pilot.press(*"draft")          # typed, never submitted
        screen.query_one("#library-row-browse-search").press()
        query_box = await _wait_for_selector(screen, pilot, "#library-rag-query-input")
        assert query_box.value == "", query_box.value
        assert screen._rag_search_state.query == "", screen._rag_search_state.query
```

  Fix in `library_rag_search_controller.py::handle_library_search_changed`: only commit the typed value into `_rag_search_state.query` (and only mirror it to the sibling input) when the Search/RAG row is the selected row; on every other row keep the keystrokes local to the widget. Re-run `Tests/UI/test_library_rag_keystroke.py` and `Tests/UI/test_library_crit8_polish_shell.py` and report their failing-name sets before and after.

- [ ] **Step 5 (32230, AC#1): DB sizes, one source per line.** `_library_details_lines` returns `sizes_line` as one joined string that the rail renders through `library_dim_label_text("DB sizes", …)`, and it wraps mid-value at the rail's width. Change `_library_db_sizes_line` to return `tuple[str, ...]` (one `"<Source> <size>"` per DB, already-formatted), have `_library_details_lines` pass the tuple through as its third element, and in `_compose_details_body_children` replace the single `Static` with:

```python
        if len(details_lines) > 2 and details_lines[2]:
            sizes = details_lines[2]
            if isinstance(sizes, str):          # legacy single-line callers
                sizes = (sizes,)
            for index, size_line in enumerate(sizes):
                yield Static(
                    library_dim_label_text("DB sizes", size_line)
                    if index == 0
                    else f"{LIBRARY_DETAILS_CONTINUATION_PAD}{size_line}",
                    id="library-details-db-sizes"
                    if index == 0
                    else f"library-details-db-sizes-{index}",
                    classes="library-details-row",
                )
```

  Keep `#library-details-db-sizes` as the id of the FIRST row — `grep -rn "library-details-db-sizes" Tests/` first and keep every existing pin green. Test: at 100 columns each mounted size row's painted text contains no line break inside a value (assert every row's `region.height == 1`).

- [ ] **Step 6 (32230, AC#2): the Handoff line names what is blocked.** `_workspace_handoff_summary_label` (`library_screen.py:13143`) produces `Handoff · 0 eligible, ● 1 blocked`. Change it to name the blocker and its remedy in the house `reason · next step` grammar, using the eligibility reason the workspace state already carries (`tldw_chatbook/Workspaces/eligibility.py`, `reason_code`): `1 blocked · not in this workspace · Link it from the conversation's header`. Assert the exact string in a unit test that feeds a state with one blocked item; keep the 0-blocked case as `Handoff · 0 eligible` with no dot.

- [ ] **Step 7 (32219, AC#1): the guide's Layout tour.** `Docs/User_Guide/library.md:184-187` still says the Chunking Lab strip sits "directly under the header, on *every* Library canvas"; line 272 in the same page already says the truth (Details ▸ Actions). Delete the 184–187 bullet and replace it with a pointer to the control-table row, so the page has exactly one statement of the placement. Grep the whole page for `Chunking Lab` (lines 184, 185, 187, 272, 777, 790, 806) and make every occurrence agree; 777/790/806 are historical stamp text — leave those alone and say so.

- [ ] **Step 8 (32219, AC#2): the rail's fold.** At 52 rows the Details ▸ Actions group is below the fold with no cue. Take the cheaper of the two options the AC offers: the rail is already `overflow-y: auto` with a `scrollbar-color: $ds-text-muted` (`_agentic_terminal.tcss:1192`), so the missing piece is that the scrollbar reads as absent when the thumb fills most of the track. Add a one-row `Static` at the end of the Details body reading `▾ more below — scroll or press F6` that is `display`ed only when the rail's `max_scroll_y > 0`, computed in `LibraryRail.on_resize`/`on_mount`. Test: mount the rail at 52 rows with the Details section open, assert the cue is displayed; at 80 rows assert it is not. If measuring `max_scroll_y` from inside the rail proves unreliable in the harness, fall back to AC#2's second clause — move the Actions group above the Workspaces group in `_compose_details_body_children` so it is above the fold at 52 rows — and say which you took.

- [ ] **Step 9: live-verify** on both profiles at 235x52, 100x30 and 60x24 (socket `crit9-rail`): capture the rail frame columns with `capture-pane -e` (32212), the heading at 100 columns (32220), type into the rail box on Media then open Search/RAG (32226), open Details and capture the DB-sizes and Handoff rows (32230), and the fold cue at 52 rows (32219). Captures to `<SCRATCH>/crit9/wave/rail/caps/`.

- [ ] **Step 10: docs + backlog + commits.** Stamp `Docs/User_Guide/library.md`. One commit per id: `fix(library-rail): the search row fits its pane again (task-32212)`, `fix(library-rail): the heading ellipsises instead of cutting mid-word (task-32220)`, `fix(library-rail): unsubmitted rail text stays on its canvas (task-32226)`, `fix(library-rail): one DB size per line and an actionable Handoff row (task-32230)`, `docs(library): the Layout tour matches the shipped Chunking Lab placement (task-32219)`.

---

### Task 4: Import queue — focus after Show details, grouped failures (group `import`, tasks 32216, 32231)

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_ingest_canvas.py` (queue-row composition 596–700; the `Show details` Button at 662–673) — **carve-out: do NOT touch lines 1036–1038 (`_toggle_label`), which Task 5 owns**
- Modify: `tldw_chatbook/Library/library_ingest_state.py` (`build_ingest_queue_rows` / `IngestQueueRow`, the state ladder at 1750–1920) — **carve-out: do NOT touch line 325 (`_GLYPH_SKIPPED`), which Task 5 owns**
- Modify: `tldw_chatbook/UI/Library_Modules/library_ingest_controller.py` (the `Show details` and dismiss handlers)
- Test: `Tests/UI/test_library_crit9_import.py` (new), `Tests/Library/test_library_ingest_state.py` (extend for the grouping)
- Docs: `Docs/User_Guide/library/import-and-export.md`

**Interfaces:**
- Consumes: the synchronous-focus discipline `Tests/UI/test_library_ingest_clear_focus.py` documents — `Screen.set_focus(widget)` BEFORE the update, never `Widget.focus()` (which defers through `app.call_later` and loses the race with `_refresh_library_ingest_canvas_preserving_context`'s focus capture).
- Produces: `group_ingest_queue_rows(rows: Sequence[IngestQueueRow]) -> tuple[IngestQueueGroup, ...]` in `library_ingest_state.py`, where `IngestQueueGroup` carries `glyph`, `line`, `members: tuple[IngestQueueRow, ...]` and `expanded: bool`.

**Host note:** this Mac's POSIX semaphores are exhausted, so a real local Import always fails at parse-pool start with `[Errno 28]`. That is not a blocker for this task — it is the fixture. Every step below is exercisable from the harness (build the queue rows directly) and from the live app's failure rows; the success path is out of scope and must not be claimed as verified.

- [ ] **Step 1 (32216): failing test.** After pressing `Show details` on a failed row, focus lands on `#library-ingest-keywords` (the `Keywords (optional)` Input ~25 rows up) because the details toggle recomposes the canvas through the same preserving-context path `test_library_ingest_clear_focus.py` describes.

```python
@pytest.mark.asyncio
async def test_show_details_leaves_focus_on_the_button_it_toggled():
    host, screen, pilot_ctx = ...   # reuse the host in Tests/UI/test_library_ingest_canvas.py
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_import_canvas(host, pilot)
        await _seed_failed_job(screen, pilot, job_id="job-1")
        button_id = "library-ingest-details-job-1"
        screen.query_one(f"#{button_id}", Button).press()
        await _wait_for_selector(screen, pilot, "#library-ingest-detail-job-1-0")
        assert getattr(screen.focused, "id", None) == button_id, screen.focused
        screen.query_one(f"#{button_id}", Button).press()
        await _wait_for_condition(pilot, lambda: not screen.query("#library-ingest-detail-job-1-0"))
        assert getattr(screen.focused, "id", None) == button_id, screen.focused
```

- [ ] **Step 2 (32216): implement.** In the controller's `Show details` handler, before triggering the region update, call `self.screen.set_focus(event.button)` synchronously (the exact idiom in `library_screen.py`'s ingest Clear path — grep `set_focus` there and copy it), and after the recompose re-resolve the button by id and `set_focus` it again inside the same `call_after_refresh` the Reader's More toggle uses (`test_more_toggle_leaves_focus_on_the_more_button` is the working precedent). Run green.

- [ ] **Step 3 (32231): failing unit test for the grouping.** In `Tests/Library/test_library_ingest_state.py`:

```python
def test_identical_failures_group_into_one_row():
    rows = tuple(_failed_row(f"job-{n}", basename=f"note{n}.md", reason="Parse pool could not start") for n in range(4))
    groups = group_ingest_queue_rows(rows)
    assert len(groups) == 1
    assert groups[0].line == "✗ failed · 4 files · Parse pool could not start"
    assert groups[0].members == rows
    assert groups[0].expanded is False


def test_rows_with_different_reasons_never_group():
    rows = (_failed_row("a", reason="Parse pool could not start"), _failed_row("b", reason="Unsupported file type: .json."))
    assert len(group_ingest_queue_rows(rows)) == 2


def test_a_single_failure_keeps_its_own_filename_row():
    row = _failed_row("a", basename="one.md", reason="Parse pool could not start")
    groups = group_ingest_queue_rows((row,))
    assert groups[0].line == row.line          # unchanged, no "1 files"
```

  Group key = `(state, reason)`; only settled states (`failed`, `skipped`, `cancelled`) group; active states (`queued`, `parsing`, `writing`) never do, because their per-file progress is the point.

- [ ] **Step 4 (32231): implement the grouped row in the canvas.** A group with `len(members) > 1` renders one `Static` (`id=f"library-ingest-group-{key_hash}"`) plus one `Horizontal` of exactly three compact Buttons — `Show the 4 files` (`library-ingest-group-expand-…`), `Retry all` (`library-ingest-group-retry-…`, only when every member `can_retry`), `Dismiss all` (`library-ingest-group-dismiss-…`). Expanding renders the members' existing per-row widgets underneath, unchanged. A group of one renders exactly what it renders today. UI test:

```python
@pytest.mark.asyncio
async def test_four_identical_failures_paint_one_row_with_three_actions():
    ...
    assert len(screen.query(".library-ingest-row")) == 1
    painted = _painted(host, screen.query_one("#library-ingest-queue").region)
    assert "✗ failed · 4 files · Parse pool could not start" in painted, painted
    for label in ("Show the 4 files", "Retry all", "Dismiss all"):
        assert label in painted, painted
```

  `Dismiss all` reuses the existing single-row dismiss handler once per member; `Retry all` reuses the existing retry handler once per member. Do not invent a new registry call.

- [ ] **Step 5: run the existing ingest suites and compare failing-name sets** — `Tests/UI/test_library_ingest_canvas.py`, `Tests/UI/test_library_ingest_retry_last.py`, `Tests/UI/test_library_ingest_keyboard.py`, `Tests/Library/test_library_ingest_state.py`, `Tests/Library/test_library_ingest_jobs.py`. Report only new names.

- [ ] **Step 6: live-verify** (socket `crit9-import`) on the fresh profile: point Import at the seeded `inbox/` folder with four markdown files. Every job will fail at parse-pool start with the host's `[Errno 28]` — that is the fixture. Capture: the four failures collapsed into one grouped row; `Show the 4 files` expanding; `Show details` on a member leaving focus on itself (send `Tab` afterwards and prove the next focus is inside the queue, not the Keywords field); `Dismiss all` clearing the group. Captures to `<SCRATCH>/crit9/wave/import/caps/`. State in the report that the success path was NOT exercisable on this host.

- [ ] **Step 7: docs + backlog + commits.** In `Docs/User_Guide/library/import-and-export.md` describe the grouped failure row and its three actions, and the focus rule for `Show details`. Stamp. Commits: `fix(library-import): Show details keeps focus on the row it opened (task-32216)`, `fix(library-import): identical failures group into one row with Retry all and Dismiss all (task-32231)`.

---

### Task 5: Copy, glyphs and dialogs (group `grammar`, tasks 32235, 32236, 32221, 32229)

**Files:**
- Modify: `tldw_chatbook/Library/library_shell_state.py` (glyph constants around 108–170)
- Modify: `tldw_chatbook/Widgets/Library/library_search_rag_panel.py` (`marker = "✓" if option.selected else "○"` at 483; `library_rag_query_status_children` at 930–952)
- Modify: `tldw_chatbook/Library/library_rag_state.py` (the credential branch at 1296–1315; `_recovery_copy` at 797–815 stays, its RENDERING moves to the log)
- Modify: `tldw_chatbook/Widgets/Library/library_ingest_canvas.py` — **carve-out: lines 1036–1038 only** (`_toggle_label`'s `✓`/`○`)
- Modify: `tldw_chatbook/Library/library_ingest_state.py` — **carve-out: line 325 only** (`_GLYPH_SKIPPED`)
- Modify: `tldw_chatbook/Library/library_export_scope.py` (the summary builder at 197–225)
- Modify: `tldw_chatbook/Third_Party/textual_fspicker/file_dialog.py` (the `File name` Input at 141 and the label logic at 151–170)
- Test: `Tests/UI/test_library_crit9_grammar.py` (new), `Tests/Library/test_library_export_scope.py` (extend)
- Docs: `Docs/User_Guide/library.md` (the glyph sentences at 273, 646, 660–661), `Docs/User_Guide/library/search-and-rag.md`, `Docs/User_Guide/library/import-and-export.md`

**Interfaces:**
- Consumes: `NO_ANALYSIS_PROVIDER_REASON` (`"no analysis provider is configured"`) and `NO_ANALYSIS_PROVIDER_NEXT_STEP` (`"Set one in Settings ▸ Providers & Models"`) from `tldw_chatbook/Library/ingest_analysis.py:36-43` — these two constants are the "one place" AC#3 of 32236 asks for. Import them; do not restate the sentence.
- Produces, in `library_shell_state.py`:
  ```python
  #: task-32235: one meaning per glyph across every Library canvas.
  LIBRARY_GLYPH_SELECTED = "☑"      # a checkbox the user toggles, checked
  LIBRARY_GLYPH_UNSELECTED = "☐"    # the same checkbox, unchecked
  LIBRARY_GLYPH_OUTCOME_DONE = "✓"
  LIBRARY_GLYPH_OUTCOME_FAILED = "✗"
  LIBRARY_GLYPH_OUTCOME_SKIPPED = "–"
  ```

#### The glyph legend this plan adopts (32235)

**Decision.** One meaning per glyph, as follows — and this is a deliberate, documented deviation from the snapshot's proposed legend in exactly one place:

| Glyph | Meaning | Where |
|---|---|---|
| `█` (leading, CSS `border-left: thick`) | the keyboard cursor | focused list rows, focused evidence cards, the chooser cursor (task-32210) |
| `☐` / `☑` | selection the user toggles | Search/RAG Sources panel, Import type toggles |
| `✓` / `✗` / `–` | a settled outcome | Import queue rows, receipts |
| `▸` / `▾` | disclosure | section headers (trailing), tree nodes (leading) |
| `○` | **a blocked or disabled action** | `library_disabled_action_label` |
| `✓` (leading, in a chooser) | the active value of a chooser | `LIBRARY_CHOICE_ACTIVE_MARKER` |
| `▸ ` (leading, on a rail row) | the destination you are on | `library_rail.py:683` |

**The deviation and why.** The snapshot proposes that blocked actions "drop the glyph". The code shows that cannot hold as written: `LIBRARY_DISABLED_ACTION_MARKER = "○"` (`library_shell_state.py:108`) is the *only* non-colour cue on dozens of gated Library buttons — it exists precisely because task-4023 AC#1 (RC-07) found those buttons were distinguishable by dimming alone, and `library.md:646` documents it as the contrast remedy. Most of those controls carry their reason in a tooltip, not inline, so removing the glyph before every gate has an inline reason would re-introduce the colour-only meaning the house rules forbid. So `○` keeps exactly one meaning — blocked/disabled — and the other two meanings move off it (selection to `☐`/`☑`, the settled "skipped" outcome to `–`), which removes the three-way collision the critique measured with a three-literal diff. The leading-`▸`-on-a-rail-row versus leading-`▸`-on-a-tree-node overlap is resolved by context and documented in the legend (rail rows never expand; tree nodes are never rail rows); the Notes folder tree is peer-owned this wave and Task 7 confirms it against this legend.

- [ ] **Step 1 (32235): failing tests for the three literals.** In `Tests/UI/test_library_crit9_grammar.py`:

```python
def test_one_meaning_per_library_glyph():
    from tldw_chatbook.Library.library_shell_state import (
        LIBRARY_DISABLED_ACTION_MARKER, LIBRARY_GLYPH_SELECTED,
        LIBRARY_GLYPH_UNSELECTED, LIBRARY_GLYPH_OUTCOME_SKIPPED,
    )
    from tldw_chatbook.Library.library_ingest_state import _GLYPH_SKIPPED
    from tldw_chatbook.Widgets.Library.library_search_rag_panel import _source_option_label
    assert LIBRARY_DISABLED_ACTION_MARKER == "○"
    assert (LIBRARY_GLYPH_UNSELECTED, LIBRARY_GLYPH_SELECTED) == ("☐", "☑")
    assert _GLYPH_SKIPPED == LIBRARY_GLYPH_OUTCOME_SKIPPED == "–"
    assert LIBRARY_DISABLED_ACTION_MARKER not in {
        LIBRARY_GLYPH_SELECTED, LIBRARY_GLYPH_UNSELECTED, LIBRARY_GLYPH_OUTCOME_SKIPPED,
    }
```

  plus a painted test at 235x52 and 100x30 that opens the Search/RAG canvas and asserts `☐ Media (0)` (unchecked) and `☑` after a toggle, and a painted test on the Import queue that a skipped row reads `– skipped · weird.xyz` (AC#3 asks for captures at both sizes: save them alongside).

- [ ] **Step 2 (32235): implement.** Add the five constants above to `library_shell_state.py`. Then:
  - `library_search_rag_panel.py:483` → `marker = LIBRARY_GLYPH_SELECTED if option.selected else LIBRARY_GLYPH_UNSELECTED`
  - `library_ingest_canvas.py:1037` → the same two constants (its docstring at 1036 says `"``✓``/``○`` convention"` — update it)
  - `library_ingest_state.py:325` → `_GLYPH_SKIPPED = LIBRARY_GLYPH_OUTCOME_SKIPPED  # "–": a settled outcome, never a disabled control (task-32235)`
  `grep -rn "○" Tests/` first and fix every pin that asserts the OLD glyph in these three places — those pins are asserting the collision, so updating them is the point, but you must list each one you touched in the Implementation Notes.

- [ ] **Step 3 (32235): the guide.** Rewrite `Docs/User_Guide/library.md` lines 273, 646 and 660–661 to state the legend table above verbatim (as a markdown table), and delete any other sentence in the page that assigns a second meaning to `○` or to a leading `▸`. Stamp.

- [ ] **Step 4 (32236): failing test.** With no provider configured, the RAG Answer gate paints six lines through `_recovery_copy` — `Blocked.` / `Unavailable: …` / `Why: The configured provider has no usable API key. Set OPENAI_API_KEY or add api_key under [api_settings.openai].` / `Next: …` / `Recovery: <same sentence>` / `Owner: LLM provider credential.`

```python
@pytest.mark.asyncio
async def test_a_missing_provider_key_blocks_in_the_media_grammar():
    host = _library_host_without_provider()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_search_rag(host, pilot)
        screen.query_one("#library-rag-query-input", Input).value = "anything"
        await pilot.press("enter")
        await pilot.pause()
        callout = screen.query_one("#library-rag-query-blocked-callout", Static)
        assert str(callout.content) == (
            "No analysis provider is configured · Set one in Settings ▸ Providers & Models."
        ), str(callout.content)
        assert not screen.query("#library-rag-query-recovery")
        painted = _painted(host, screen.query_one("#library-rag-query").region)
        for banned in ("Owner:", "Recovery:", "Why:", "OPENAI_API_KEY", "[api_settings."):
            assert banned not in painted, painted
        assert screen.query_one("#library-rag-open-provider-settings", Button)
```

- [ ] **Step 5 (32236): implement.** Three edits:
  1. `library_rag_state.py`, the `credential_recovery` branch at ~1302: replace the interpolated env-var sentence with the shared constants —
     ```python
     from tldw_chatbook.Library.ingest_analysis import (
         NO_ANALYSIS_PROVIDER_NEXT_STEP,
         NO_ANALYSIS_PROVIDER_REASON,
     )
     ...
             if credential_recovery:
                 # task-32236 (critique #9 row 5): the Media reader's identical
                 # condition already says this sentence; one missing key must not
                 # produce two remedies, one of them TOML. The structured record
                 # (owner, the config-table remedy) still reaches the log below.
                 disabled_reason = (
                     f"{NO_ANALYSIS_PROVIDER_REASON.capitalize()} · "
                     f"{NO_ANALYSIS_PROVIDER_NEXT_STEP}."
                 )
                 owner = "LLM provider credential"
                 next_action = "Add the provider credential, then run again"
                 recovery_action = credential_recovery
     ```
  2. `library_search_rag_panel.py::library_rag_query_status_children` (~930–952): render the callout as `Static(reason, id="library-rag-query-blocked-callout", …)` — the bare reason, no `Blocked | ` prefix — drop the `#library-rag-query-recovery` `Static` entirely, and append `Button("Open Settings ▸ Providers", id="library-rag-open-provider-settings", classes="library-canvas-action", compact=True)` only when the blocker is the provider/credential one. Route the button through the screen's existing settings navigation (grep `Settings ▸ Providers` and the route id used by the Media reader's equivalent action; reuse it, do not add a new navigation path).
  3. Keep `_recovery_copy` and the `recovery_copy` field: log the structured record once at the gate (`logger.info`) so the diagnostic survives. Confirm with `grep -rn "recovery_copy" tldw_chatbook/` that no other Library surface renders it.
  Re-run `Tests/UI/test_library_content_hub.py`, `Tests/UI/test_product_maturity_gate16_library_search_rag.py` and `Tests/Library/test_library_rag_state.py`; the four existing `assert not screen.query("#library-rag-query-recovery")` pins stay green by construction, and any pin asserting the six-line block must be updated with a note.

- [ ] **Step 6 (32221): pluralisation.** `library_export_scope.py:197–225` builds `f"{counts.get('notes', 0)} notes · "`. Add one private helper and use it for all four nouns:

```python
def _count_phrase(count: int, singular: str) -> str:
    """Return "1 note" / "2 notes" -- every export noun pluralises the same way."""
    return f"{count} {singular}" if count == 1 else f"{count} {singular}s"
```

  **"media" is already plural**, so it never goes through `_count_phrase`: render it as `"1 media item"` / `"N media items"`. The three other nouns (`conversation`, `note`, `prompt`) take `_count_phrase`. Test in `Tests/Library/test_library_export_scope.py`:
  ```python
  @pytest.mark.parametrize(
      ("counts", "expected"),
      [
          ({"media": 1, "conversations": 1, "notes": 1, "prompts": 1},
           "Everything: 1 media item · 1 conversation · 1 note · 1 prompt"),
          ({"media": 0, "conversations": 2, "notes": 1, "prompts": 13},
           "Everything: 0 media items · 2 conversations · 1 note · 13 prompts"),
      ],
  )
  def test_export_scope_summary_pluralises_every_noun(counts, expected):
      assert library_export_scope_summary(LibraryExportScope(kind="everything"), counts) == expected
  ```
  Also fix the per-kind summaries below (`f"Notes · {counts.get('notes', 0)} items"` at 219–220 and its three siblings) to the same helper, and update the docstring example at `library_export_scope.py:197` to the new strings. Use the real function name you find at 197–225 in the assertion above.

- [ ] **Step 7 (32229): a path field in the file dialogs.** `tldw_chatbook/Third_Party/textual_fspicker/file_dialog.py:141` already mounts a `File name` Input whose label flips to `Folder path:` (task-32122). Extend that ONE field rather than adding a second: on `Input.Changed`, expand a leading `~` and, when the typed text resolves to an existing directory, navigate the tree to it. Add a dialog-scoped `Binding("ctrl+a", "select_all", show=False)` on that Input so Ctrl+A selects the text instead of Textual's default move-to-start. Tests in `Tests/UI/test_library_crit9_grammar.py`:

```python
@pytest.mark.asyncio
async def test_a_pasted_absolute_path_moves_the_dialog_tree(tmp_path):
    target = tmp_path / "deep" / "nested"
    target.mkdir(parents=True)
    ...  # open the export destination picker, type str(target), assert the tree's cwd is target

@pytest.mark.asyncio
async def test_a_tilde_path_expands(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    ...  # type "~/", assert the tree's cwd is tmp_path

@pytest.mark.asyncio
async def test_ctrl_a_selects_the_whole_file_name():
    ...  # type "report.zip", press ctrl+a, assert the Input's selection covers all 10 characters
```

  This file is shared by every file dialog in the app, which is exactly why the fix belongs here — run `Tests/UI/test_library_export_*.py` and `grep -rln "FileOpen\|FileSave" Tests/ | head` and execute what that names, reporting failing-name sets before and after.

- [ ] **Step 8: live-verify** (socket `crit9-grammar`) on the seeded profile: Search/RAG with no provider (capture the one-line block and the Settings button); the Sources panel checkboxes at 235x52 and 100x30; an Import that fails, showing `–  skipped` on an unsupported file; Export ▸ notes with exactly one note selected (capture `1 note`); the destination picker with a pasted absolute path. Captures to `<SCRATCH>/crit9/wave/grammar/caps/`.

- [ ] **Step 9: docs + backlog + commits.** Commits: `fix(library): one meaning per state glyph (task-32235)`, `fix(library-rag): the blocked answer speaks the Media grammar (task-32236)`, `fix(library-export): counts pluralise (task-32221)`, `fix(library): file dialogs accept a typed path (task-32229)`.

---

### Task 6: Shell — density rule, narrow Escape, Conversations footer, Skills trust (group `shell`, tasks 32217, 32225, 32228, 32223)

**Files:**
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` — **only** the canvas/pane blocks named in Step 2, plus one new density block appended at the end of the Library section
- Modify: `tldw_chatbook/Widgets/Library/library_prompts_canvas.py`, `library_skills_canvas.py`, `library_conversations_canvas.py`, `library_collections_capture_reader.py` (pane widths)
- Do NOT modify: `tldw_chatbook/Widgets/Library/library_media_viewer.py` — Task 2 owns it. The reader-content-box half of 32217 AC#2 is delivered by CSS only (`#library-media-viewer-content { height: 1fr; }`) inside the density block this branch owns; if it cannot be done without editing that Python, leave AC#2's reader clause unticked with the note `needs a change in library_media_viewer.py, owned by the crit9-media-reader branch this wave`.
- Modify: `tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py` and `tldw_chatbook/UI/Screens/library_screen.py` — **only** `_library_route_shortcuts_for_current_state` (3971–4060, the Conversations branch) and the `‹ Library` helpers at 6624–6650
- Test: `Tests/UI/test_library_crit9_shell.py` (new)
- Docs: `Docs/User_Guide/library/prompts.md`, `library/skills.md`, `library/media-and-conversations.md` (Conversations footer), `Docs/User_Guide/library.md` (narrow layouts — coordinate the stamp with Task 3)

#### The density rule this plan adopts (32217)

**Decision.** *A Library canvas pane that has nothing open gives its columns to its sibling, and a pane's primary content box takes the pane's remaining height (`height: 1fr`) with its action rows sitting directly beneath the content, never pinned to the pane floor.* Two sentences of reasoning: Media already proved the first half of this rule in `tldw_chatbook/Utils/adaptive_reader_state.py` (`list_grows`, resolved at 460), so applying it to the sibling canvases needs no new mechanism and cannot drift per canvas; the second half is the same rule turned vertically, and it is what turns "48-cell list beside a 145-cell *select something* pane" and "an 11-row Body in a 45-row pane" into readable canvases without inventing any new content. **AC#3 (the landing):** narrow the landing hub to a readable measure (`max-width: 96` on `#library-hub`, centred) rather than inventing recent-items/last-import content — a canvas that earns its space by growing new features is a product decision this wave does not own, and a 190-column-wide ten-line hub is fixed by measure alone.

- [ ] **Step 1 (32217): failing tests, measured.** In `Tests/UI/test_library_crit9_shell.py`, one test per canvas:

```python
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("row_id", "list_id", "detail_id"),
    [
        ("browse-prompts", "#library-prompts-list", "#library-prompt-work-pane"),
        ("browse-skills", "#library-skills-list", "#library-skill-work-pane"),
        ("browse-collections", "#library-collections-list", "#library-collections-detail"),
        ("browse-conversations", "#library-conversations-list", "#library-conversation-reader"),
    ],
)
async def test_a_canvas_with_nothing_open_gives_its_columns_to_the_list(row_id, list_id, detail_id):
    host = _library_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one(f"#library-row-{row_id}").press()
        listing = await _wait_for_selector(screen, pilot, list_id)
        canvas = screen.query_one("#library-canvas")
        assert listing.region.width >= canvas.region.width - 4, (listing.region, canvas.region)
```

  Run them and paste each measured pair into the report — those numbers are the AC#1 evidence. Use the real ids you find by grepping each canvas file; if a canvas's list has no id, give it one in the same commit.

- [ ] **Step 2 (32217): implement.** For each of the four canvases, mirror what Media does: when the detail/work pane holds nothing, set the pane's `display = False` (or width `0`) and the list's width to `1fr`, decided in the canvas's own compose from the state it already carries (`selected_id is None`, `presentation.selected is None` — grep each). Then append one density block to `_agentic_terminal.tcss`:

```
/* task-32217 (critique #9 row 14): the screen-wide density rule.
   A pane with nothing open gives its columns to its sibling (Media's rule,
   Utils/adaptive_reader_state.py::list_grows); a pane's primary content box
   takes the remaining height so its actions sit under the content instead of
   on the pane floor. */
#library-prompt-work-pane-body,
#library-skill-work-pane-body,
#library-note-work-pane-body,
#library-media-viewer-content {
    height: 1fr;
    min-height: 3;
}

#library-hub {
    max-width: 96;
}
```

  Rebuild the bundle. Add a height test: at 235x52 with an item open, `#library-media-viewer-content` (and each work-pane body) has `region.height >= pane.region.height - 8`, and the Analysis action row's `region.y` is less than 4 rows below the content's `region.bottom`.

- [ ] **Step 3 (32217): record the carry-over.** `library_notes_canvas.py`'s note-editor Body and `library_file_notes_workspace.py`'s File Notes canvas are owned by Task 7 this wave. Add their selectors to the density block **only if** they are not in those two files (check: the ids may be styled from the shared TCSS, in which case a selector-only addition is fine and does not touch the peer's Python). If the note editor Body needs a Python change, leave that clause of AC#2 unticked with the note `carried to the crit9-notes branch (peer-owned file) — selector <id> added to the density block, Python change deferred`.

- [ ] **Step 4 (32225): failing test.** Below 64 columns the Library collapses to one stage; Escape on a list is inert and the `‹ Library` return is never advertised.

```python
@pytest.mark.asyncio
async def test_escape_returns_to_the_rail_stage_below_64_columns():
    host = _library_host()
    async with host.run_test(size=(60, 24)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-0")
        assert ("esc", "back to Library") in screen._library_footer_shortcuts_for_current_state()
        await pilot.press("escape")
        await pilot.pause()
        assert screen.query("#library-rail"), "Escape must return to the rail stage"
```

  Implement by adding the single-stage case to `_library_route_shortcuts_for_current_state` (it currently has seven named contexts; this is the eighth) and by routing `escape` there to the same handler `‹ Library` presses (`library_screen.py:6624–6650` builds that control — reuse its callback, do not duplicate the navigation). The footer chip text is exactly `back to Library`, matching the control's own `‹ Library` label.

- [ ] **Step 5 (32228): failing test.** The Conversations branch of `_library_route_shortcuts_for_current_state` (around 4050) builds its own list and omits both `esc focus rail` and the `/` hint when the items pane is closed.

```python
@pytest.mark.asyncio
async def test_the_conversations_footer_advertises_the_same_list_keys_as_its_siblings():
    host = _library_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversations-filter")
        chips = screen._library_footer_shortcuts_for_current_state()
        assert ("/", "focus filter") in chips, chips
        assert ("esc", "focus rail") in chips, chips
        # and the keys work
        await pilot.press("slash")
        assert screen.focused is screen.query_one("#library-conversations-filter", Input)
        await pilot.press("escape")
        await pilot.pause()
        assert screen.focused is screen.query_one("#library-search-input", Input)
```

  Implement by making the Conversations branch return `self.LIBRARY_LIST_SHORTCUTS` plus its own hand-off chip, exactly as the Collections branch above it already does. Check `check_action` gates for `library_list_focus_rail` on this row before assuming the Escape hop works — if the gate excludes Conversations, that is the real fix.

- [ ] **Step 6 (32223): failing test + fix.** Skills list rows paint identically for a trust-approved and an unapproved skill.

```python
@pytest.mark.asyncio
async def test_the_skills_list_shows_each_row_s_trust_state():
    host = _skills_host()      # one "trusted" skill, one "quarantined_added"
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_skills(host, pilot)
        painted = _painted(host, screen.query_one("#library-skills-list").region)
        assert "· trusted" in painted, painted
        assert "· needs review" in painted, painted
```

  Implement in `library_skills_canvas.py` by appending the trust state to the row label from the map the editor already uses (`_SKILL_TRUST_LABELS` at 91–96 — reuse it, do not restate the strings): `trusted` → `trusted`; every `quarantined_*` and `trust_uninitialized` → `needs review`; `trust_locked` → `locked`. Text, not colour, and not a new glyph (the legend in Task 5 has no glyph for trust). The row already carries `button.skill_name` at 1296 — put the suffix on the same label.

- [ ] **Step 7: live-verify** (socket `crit9-shell`) on the seeded profile at 235x52, 100x30 and 60x24: Prompts, Skills, Collections and Conversations with nothing open (capture the list filling the canvas); the Media reader with an item open (Analysis actions under the content); the landing at 235 columns (measure); Escape on a list at 60 columns; the Conversations footer; the two Skills rows. Captures to `<SCRATCH>/crit9/wave/shell/caps/`.

- [ ] **Step 8: docs + backlog + commits.** Commits: `fix(library): a pane with nothing open gives its columns away (task-32217)`, `fix(library): Escape returns to the rail below 64 columns (task-32225)`, `fix(library-conversations): the footer advertises the list keys it honours (task-32228)`, `fix(library-skills): list rows carry their trust state (task-32223)`.

---

### Task 7: Notes — Escape, Sort/folder verbs, vocabulary (group `notes`, tasks 32233, 32215, 32218)

> **DISPATCH LAST — do not start this branch until the peer confirms PR #2565 is on `dev`.** A peer session owns `library_notes_controller.py`, `library_notes_canvas.py`, `canvas_sync.py` and `library_file_notes_workspace.py` for its wave-2 landing. No other branch in this wave may touch those four files. Before your first command, `git fetch -q origin && git log --oneline -1 origin/dev` and confirm #2565's merge commit is in the history; if it is not, stop and report BLOCKED.

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_notes_canvas.py` (filter row 855–880, `library-notes-browse-actions` toolbar 985–1120)
- Modify: `tldw_chatbook/UI/Library_Modules/library_notes_controller.py`
- Modify: `tldw_chatbook/UI/Library_Modules/canvas_sync.py`
- Modify: `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` — **only** `_library_list_focus_rail_target` (24987–25010) and the two Escape call sites at 23334 and 24979
- Test: `Tests/UI/test_library_crit8_keyboard.py` (32233 AC#3 names this file explicitly — add there, beside the Media pins), `Tests/UI/test_library_crit9_notes.py` (new, for 32215/32218)
- Docs: `Docs/User_Guide/library/notes.md`, `Docs/User_Guide/library/file-notes.md`

**Interfaces:**
- Consumes: `_library_list_focus_rail_target()` (`library_screen.py:24987`) — already returns `#library-search-input` for Notes; the hop does not take effect on that surface.
- Consumes: the Media pin `Tests/UI/test_library_crit8_keyboard.py:194::test_escape_from_a_list_filter_box_still_goes_where_the_footer_says` — it drives Media only, and its docstring records the design decision (`library_list_focus_rail` is declared AFTER the blur binding and must outrank it on a list canvas). Extend it; do not weaken it.

- [ ] **Step 1 (32233): re-verify live at the current tip BEFORE any fix.** The snapshot measured `02374bf66a`, before #2543 (Folder files became a mode) and #2547 (the editor-keys group: slash filter, Escape ladders, spoken vetoes, back cues) landed — and #2547's surface is exactly where 32233 sits. On the seeded power profile (socket `crit9-notes`) at `dev` >= `e6cb464239`:
  1. Open Library ▸ Notes. Press `/`, type `read`, press `escape`, then type `n`. Capture.
  2. On the plain Notes list (no filter focus), press `escape`, then type `abc`. Capture.
  3. Repeat both in the folder-tree layout.
  **Decision rule.** If Escape now blurs the filter and the next key does its canvas job, and Escape on the plain list lands in `#library-search-input`: **close 32233 by evidence** — `backlog task edit 32233 -s Done --notes "Not reproducible at dev <sha>. The snapshot measured 02374bf66a, before PR #2547 (editor-keys group) landed; captures at <SCRATCH>/crit9/wave/notes/caps/32233-*.txt show Escape blurring the filter and focusing #library-search-input from the plain list in both the database and folder-tree layouts."` — but STILL do Step 2, because AC#3 asks for the pins regardless. If either case still misbehaves, continue to Step 3.

- [ ] **Step 2 (32233 AC#3): pin both cases beside the Media pins.** In `Tests/UI/test_library_crit8_keyboard.py`, add:

```python
@pytest.mark.asyncio
async def test_escape_from_the_notes_filter_box_blurs_and_the_next_key_is_a_canvas_key():
    host = _build_library_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        filter_box = await _wait_for_selector(screen, pilot, "#library-notes-filter")
        await pilot.press("slash")
        await pilot.pause()
        assert screen.focused is filter_box
        await pilot.press("escape")
        await pilot.pause()
        assert screen.focused is not filter_box, screen.focused
        before = filter_box.value
        await pilot.press("n")
        await pilot.pause()
        assert screen.query_one("#library-notes-filter", Input).value == before


@pytest.mark.asyncio
@pytest.mark.parametrize("layout", ["database", "folder_tree"])
async def test_escape_on_the_plain_notes_list_focuses_the_rail_search_box(layout):
    ...
    await pilot.press("escape")
    await pilot.pause()
    assert screen.focused is screen.query_one("#library-search-input", Input)
```

  Run them. If Step 1 closed the task by evidence they pass immediately — that is the proof, keep them. If they fail, they are your TDD red.

- [ ] **Step 3 (32233): fix, if it still reproduces.** The gate is the escape-binding ladder in `library_screen.py`: `check_action("library_blur_text_field", …)` must be `False` on a Notes list canvas so `library_list_focus_rail` wins (the exact rule the Media pin's docstring states), and `_library_list_focus_rail_target()` must be reached from the Notes canvas's Escape path in BOTH layouts. Change the gate and the hop only — do not add a Notes-specific Escape handler; a per-canvas handler is how the two surfaces diverged in the first place.

- [ ] **Step 4 (32215): SPIKE first — half of this is probably already shipped.** `library_notes_canvas.py:985–1044` composes `#library-notes-browse-actions` with a `Sort:` chooser unconditionally, with a `Newest` default; the peer's PR #2558 (task-32172) landed that and #2565 re-pins its presence. Do this:
  1. `git log --oneline -20 -- tldw_chatbook/Widgets/Library/library_notes_canvas.py` and read #2558's diff.
  2. Live: open Notes on the seeded profile (7 notes) and capture the toolbar. If `Sort: Newest` is present on the populated list, AC#1 is already satisfied — tick it with the capture path and the PR number in the notes, and do NOT write a second Sort control.
  3. AC#2 (`New folder`, `Add to folder`, `Move note`, `Remove placement` are meaningless without a selection and carry no `○` or reason) almost certainly still stands. Failing test:

```python
@pytest.mark.asyncio
async def test_selection_scoped_folder_verbs_are_gated_with_their_reason():
    host = _notes_host_with_seven_notes()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_notes(host, pilot)
        for button_id in ("library-notes-add-to-folder", "library-notes-move", "library-notes-remove-placement"):
            button = screen.query_one(f"#{button_id}", Button)
            assert button.disabled, button_id
            assert str(button.label).startswith("○ "), (button_id, str(button.label))
            assert button.tooltip == "Select a note first.", (button_id, button.tooltip)
        await _check_row(screen, pilot, 0)
        for button_id in ("library-notes-add-to-folder", "library-notes-move", "library-notes-remove-placement"):
            assert not screen.query_one(f"#{button_id}", Button).disabled, button_id
```

  Implement with `library_disabled_action_label` (`tldw_chatbook/Library/library_shell_state.py:127`) — the same helper Media's `Select` uses — so the `○` marker stays the one blocked-action glyph Task 5's legend defines. Use the real ids you find in the file. `New folder` is NOT selection-scoped: leave it enabled.

- [ ] **Step 5 (32218): one noun per Notes source.** Inventory first — `grep -rn "Library notes\|Library database\|Folder files\|Folder Files\|back to Database" tldw_chatbook/ Docs/User_Guide/library/` — and paste the table of every site into your report. Then apply this vocabulary and change every site to it:

| Source | The one noun | Used in |
|---|---|---|
| notes stored in the app's database | **Library notes** | rail row, canvas title, strip, empty state, footer chip (`esc back to Library notes`) |
| notes stored as files in a folder | **Folder files** | rail row, canvas title, strip, empty state, footer chip (`esc back to Folder files`) |

  Delete `Library database`, `Folder Files` (the capitalised variant) and `Database` as user-visible words. Test: a single test that walks the mounted Notes canvas at 235x52 in both modes and asserts the painted text contains neither `Library database` nor `Folder Files` nor a bare `Database`, and that the mode's own noun appears in the title, the strip and the footer chip. Update `Docs/User_Guide/library/notes.md` and `library/file-notes.md` to the same two nouns (AC#2).

- [ ] **Step 6 (density carry-over from task-32217).** Task 6 applies the wave's density rule (a pane with nothing open gives its columns away; a primary content box takes `height: 1fr`) and records any Notes-side clause it could not deliver because these files are yours. Read Task 6's Implementation Notes on task-32217, apply the same rule to the note editor's Body box and the File Notes canvas, and say in YOUR notes that you did. Do not tick task-32217's ACs — that task belongs to Task 6.

- [ ] **Step 7: live-verify** (socket `crit9-notes`) on the seeded profile at 235x52 and 100x30, in both the database and folder-tree layouts: the Escape journeys from Step 1; the toolbar with and without a checked row; the full vocabulary walk (rail row → canvas title → strip → empty state → footer). Captures to `<SCRATCH>/crit9/wave/notes/caps/`.

- [ ] **Step 8: run the Notes suites and compare failing-name sets** — `Tests/UI/test_library_crit8_keyboard.py`, `Tests/UI/test_library_notes_wave_editor_keys.py`, `Tests/UI/test_library_notes_wave_list.py`, `Tests/UI/test_library_notes_wave_file_notes.py`, `Tests/UI/test_library_notes_characterization.py`, `Tests/UI/test_library_multiselect_notes.py`, `Tests/Library/test_library_notes_state.py`. Report only new names.

- [ ] **Step 9: docs + backlog + commits.** Commits: `fix(library-notes): Escape goes where the footer says on every Notes surface (task-32233)` (or a docs/backlog-only close-by-evidence commit plus the pins), `fix(library-notes): folder verbs are gated with their reason (task-32215)`, `fix(library-notes): one noun per Notes source (task-32218)`.

---

## Dispatch order

| Order | Task | Group | Backlog ids |
|---|---|---|---|
| 1 (parallel) | Task 1 | `media-list` | 32210, 32213, 32214, 32227 |
| 1 (parallel) | Task 2 | `media-reader` | 32222, 32224, 32234, 32237 |
| 1 (parallel) | Task 3 | `rail` | 32212, 32219, 32220, 32226, 32230 |
| 1 (parallel) | Task 4 | `import` | 32216, 32231 |
| 1 (parallel) | Task 5 | `grammar` | 32221, 32229, 32235, 32236 |
| 1 (parallel) | Task 6 | `shell` | 32217, 32223, 32225, 32228 |
| 2 (last) | Task 7 | `notes` | 32215, 32218, 32233 — **after PR #2565 is on `dev`** |

Landing order for the controller: Task 5 before Task 4 (Task 4's queue rows read the glyph constants Task 5 defines); Task 3 before Task 6 (both regenerate the CSS bundle — the second one merges `dev` and re-runs `build_css`); Task 6 before Task 7 (Task 7 reads Task 6's density notes). Tasks 1 and 2 are independent of everything.
