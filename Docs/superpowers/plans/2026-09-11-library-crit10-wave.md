# Library critique-10 fix wave Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close all 21 in-scope findings of Library critique #10 (tasks 32346–32366) plus the two product decisions the user has now made (task-32107 Conversations hand-off, task-32057 Collections) in eight independent branches that each ship as their own PR against `dev`.

**Architecture:** Each task below is one branch/worktree and one PR. Work stays inside the Library surface (`tldw_chatbook/UI/Screens/library_screen.py`, `tldw_chatbook/Widgets/Library/*.py`, `tldw_chatbook/Library/*.py`, `tldw_chatbook/UI/Library_Modules/*.py`, the split Library TCSS, `Docs/User_Guide/library*`), plus one workspace helper (`tldw_chatbook/Workspaces/eligibility.py` is read-only this wave) and one app-level stamp comment (`tldw_chatbook/app.py`, docstring only). Behaviour changes are test-first against the existing Library UI harnesses; every task ends with a live tmux check on an isolated scratch profile.

**Tech Stack:** Python 3.12, Textual 8.2.8, pytest, Backlog.md CLI, tmux.

**Spec:** `.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md` (the critique snapshot; the Priority Issues and Docs-vs-Live rows are the binding requirements) and the task files `backlog/tasks/task-32346 … task-32366`, `task-32107`, `task-32057` (acceptance criteria).

**Baseline read for this plan:** `origin/docs/library-critique-10` @ `98175acf6a` (= `dev` @ `1f3184655b` plus the 21 new task files). Every line number below was read at that commit.

## Global Constraints

- Work only inside your assigned worktree (`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/crit10-<group>`); every shell command starts with `cd <worktree> &&` because the shell cwd resets between calls. Never touch the main checkout or another group's worktree.
- Python: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` (3.12). Run pytest FROM the worktree cwd (`cd <worktree> && …/.venv/bin/python -m pytest <file or node id> -q -p no:cacheprovider`). Never run the whole suite (over an hour) and never use `-k` filtering as verification; run whole files or explicit node ids. **Read pytest output from a file, never from a pipeline's exit code**: `… -m pytest <node id> -q -p no:cacheprovider > /tmp/crit10-<group>-<step>.txt 2>&1; tail -40 /tmp/crit10-<group>-<step>.txt`. `Tests/UI/test_library_shell.py` has ~226 pre-existing failures on dev: if you use it, compare failing NAME sets before and after your change (run it once against `origin/dev`), and report only new names.
- Other sessions run pytest on this Mac; POSIX semaphores are exhausted, so `multiprocessing.Pool` fails with `[Errno 28]` and every local media Import fails in the live app. Do not try to fix or work around the host; if a test needs a process pool, mark it and move on.
- TDD: write the failing test, run it and show it failing, implement, run it passing. New tests go in the most specific existing `Tests/UI/test_library_*.py` file for the area, or a new `Tests/UI/test_library_crit10_<area>.py`; never into the 19k-line `test_library_shell.py`.
- CSS: edit the component source under `tldw_chatbook/css/components/` (Library rules live in `_agentic_terminal.tcss`, split into `screen_agentic_library.tcss` by the build), then run `cd <worktree> && …/.venv/bin/python -m tldw_chatbook.css.build_css` and commit the regenerated bundle files alongside. Widget `DEFAULT_CSS`/`BUNDLED_CSS` must parse standalone and never use ancestor-scoped bare-type subject rules (`Foo > Vertical`).
- Git: stage explicit paths only (never `git add -A`); commit after each green step; do NOT push and do NOT open or merge PRs, the controller does that after review. Commit trailer: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Backlog hygiene from the worktree root: `backlog task edit <id> -s "In Progress" --plan "<steps>"` before the first code change; at the end tick every AC (`- [x]`) you satisfied, add `--notes "<Implementation Notes>"`, and set `-s Done`. Leave an AC unticked with a note if you could not satisfy it.
- Docs: every user-visible change updates the matching `Docs/User_Guide/library*.md` page and appends a `*Verified against fix/library-crit10-<group> — 2026-09-11 (task-NNNNN: …)*` stamp in the page's existing stamp style.
- Live verification (required before reporting): `tmux -L crit10-<group> new-session -d -x 235 -y 52 "cd <worktree> && TLDW_CONFIG_PATH=<profile>/config.toml PYTHONPATH=<worktree> …/.venv/bin/python -m tldw_chatbook.app"`, `sleep 15`, drive with `send-keys`, observe with `capture-pane -p` (`-e` for colour). Profiles: `<SCRATCH>/crit10/wave/<group>/power/config.toml` (seeded: 11 media, 6 conversations, 7 notes incl. a 35 KB one, 5 prompts, 2 skills, inbox/ and file_notes/ folders) and `…/fresh/config.toml` (empty; first launch shows the setup wizard: Esc, Tab, Enter skips it). `SCRATCH` = `/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/92b26568-46c4-4abe-9da5-9676d5282276/scratchpad`. Ctrl+digit keys cannot be sent through tmux: reach Library with `C-p`, type `Switch to Library`, `Down`, `Enter`. Mouse clicks: `send-keys -l $'\x1b[<0;COL;ROWM'` then `…ROWm` (1-based, column by code points, not bytes). Quit with `C-q` then `tmux -L crit10-<group> kill-server`. One instance per profile at a time.
- Copy rules: blocked or disabled states carry a text reason and a next step on the same line; never colour-only meaning; no raw errno, UUID or ISO timestamp reaches the user.
- Scope: implement the acceptance criteria of your tasks and nothing else. If an AC needs a product decision you cannot make, implement the rest, leave that AC unticked, and say so in the report.

### Additional constraints for this wave

- **Make your profile first.** The seeded profiles are per-group; sharing one across two live branches corrupts its SQLite. Run exactly:
  ```
  SCRATCH=/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/92b26568-46c4-4abe-9da5-9676d5282276/scratchpad
  mkdir -p "$SCRATCH/crit10/wave"
  cp -R "$SCRATCH/crit10/A" "$SCRATCH/crit10/wave/<group>"
  sed -i '' "s#/crit10/A/#/crit10/wave/<group>/#g" "$SCRATCH/crit10/wave/<group>/power/config.toml" "$SCRATCH/crit10/wave/<group>/fresh/config.toml"
  ```
  The copy is complete: `crit10/A/fresh/inbox/` (`article.html`, `export.json`, `lecture-transcript.txt`, `reading-notes.md`, `weird.xyz`, `nested/`) and `crit10/A/fresh/file_notes/` were empty at prep time and were populated by assessor A before the review — copy them as they stand, do not re-seed. The `data/crit10_A_power/` directory name uses underscores and is deliberately NOT rewritten by that `sed`; the `*_db_path` keys point inside your copy once the `/crit10/A/` → `/crit10/wave/<group>/` prefix is replaced. `users_name` stays `crit10_A_power`.
- **Never `git stash` bare.** The stash is repo-global and a sibling worktree can pop yours (`backlog/docs/lessons-backlog-hygiene.md`, "The stash is repo-global"). If you must park work use `git stash push -m "crit10-<group>-<why>" -- <explicit paths>` and pop it by name in the same shell.
- **`library_screen.py` is shared, in disjoint methods.** This file is ~34k lines; never open it whole (`grep -n` then `sed -n '<a>,<b>p'`). Ownership this wave:
  | Range / symbol | Owner |
  |---|---|
  | `BINDINGS` (942–1139), `LIBRARY_*_SHORTCUTS` constants (1150–1298) except `LIBRARY_INGEST_SHORTCUTS` | Task 1 |
  | `_library_footer_shortcuts_for_current_state`, the `isinstance(focused, (Input, TextArea))` block **only** (4432–4442) | Task 1 |
  | `_library_route_shortcuts_for_current_state`, the media-Reader branch **only** (4152–4230) | Task 1 |
  | `check_action`, the `library_media_read_later/use_in_console/move_to_trash` branch **only** (24213–24243) | Task 1 |
  | `_library_media_find_unavailable_reason` + `handle_library_media_reader_find` (31490–31551) | Task 1 |
  | `LIBRARY_INGEST_SHORTCUTS` (1290–1297) | Task 2 |
  | `_library_landing_attention_action` (13131–13175); the evidence-settle branch of the onboarding apply (20150–20190); `_load_library_lifecycle_value` docstring (19887) | Task 3 |
  | `_link_selected_conversation_to_workspace` (12919–12971); `use_selected_conversation_as_source` / `link_selected_conversation_to_workspace` handlers (33806–33826) | Task 6 |
  | `_library_footer_shortcuts_for_current_state`, the `_library_narrow_stage_return_active()` block **only** (4450–4466) | Task 6 |
  | `_workspace_handoff_summary_label` (13608–13660); the Details widgets block (13770–13800); `_library_db_sizes_line` / the DB-sizes patcher (14934+) | Task 7 |
  Stay inside your range — never reflow, reorder or reformat a neighbouring method. If a step would touch a range outside your row, stop and report NEEDS_CONTEXT instead of editing it.
- **`_agentic_terminal.tcss` is shared, in two disjoint ranges.** Task 2 appends ONE new `.library-media-scope-line` rule at the end of the Library section and nothing else; Task 6 edits the rail-row block (1950–1995) and appends the reader-shell/conversations rules. The regenerated `screen_agentic_library.tcss` / bundle files WILL conflict when the second PR lands: resolve by taking `dev`'s bundle, re-running `python -m tldw_chatbook.css.build_css`, and committing the regenerated result. Never hand-edit a bundle file.
- **`library_media_viewer.py` is split between two tasks.** Task 1 owns `_compose_active_body` (481–577 and 983–1090). Task 5 owns `_compose_primary_toolbar` (384–403) — the Find Button block — **and nothing else in that file**. The two ranges are disjoint; neither may reflow the other.
- **`Docs/User_Guide/` pages are shared, source files are not.** More than one branch appends a stamp to `library.md` and to `library/media-and-conversations.md`. Append your stamp as a NEW line at the end of the page's existing stamp block; on a landing conflict keep BOTH stamps (`$SCRATCH/resolve_hunks.py` from the crit8 wave does this mechanically).
- **Grep for the pin before you fix.** This screen's live critiques mis-attribute the cause about 40% of the time, and several behaviours the snapshot calls bugs are prior design decisions with a pinning test. Before changing behaviour, `grep -rn "<the id or copy string>" Tests/` and read what the pin asserts. If a pin asserts the opposite of your fix, the plan says which way to go; if the plan does not, report NEEDS_CONTEXT rather than weakening the pin. Never delete or loosen an assertion to make a change pass.
- **Preflight before you report done:** `cd <worktree> && ./scripts/preflight.sh` (~35 s, installs nothing). It runs the same four checks as the required `Derived artifacts reproduce from their sources` CI job.

---

### Task 1: Reader — the footer under a focused Input, Find, rendered analysis (group `viewer`, tasks 32346, 32348, 32365)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` — **only** the ranges listed for Task 1 in the ownership table above
- Modify: `tldw_chatbook/Library/library_media_viewer_state.py` (`analysis_find_unavailable_reason` 521–549)
- Modify: `tldw_chatbook/Widgets/Library/library_media_viewer.py` — **only** `_compose_active_body` (481–577, 983–1090). **Do NOT touch `_compose_primary_toolbar` (384–403) — Task 5 owns it.**
- Test: `Tests/UI/test_library_crit10_viewer.py` (new)
- Docs: `Docs/User_Guide/library.md` (the footer rows at 259–261), `Docs/User_Guide/library/media-and-conversations.md` (the "Find" row at 529, the Analysis tab section)

**Interfaces:**
- Consumes: `self.check_action(<action>, ())` — the honest-footer idiom already used for the Reader's `l`/`c`/`t` chips at `library_screen.py:4207-4216`. Every chip this task adds is gated the same way.
- Consumes: `_library_slash_would_land()` (`library_screen.py`, used at 4470) — the existing dead-`/` predicate.
- Consumes: `LibraryMediaContentBody(content=…, is_markdown=…, mode=…, query=…, match_index=…)` (`tldw_chatbook/Widgets/Library/library_media_content.py:64`) — already the Read tab's raw/rendered switch; the Analysis tab already mounts it with `is_markdown=False, mode="raw"` (`library_media_viewer.py:1029-1040`).
- Consumes: `looks_like_markdown_content` / `_is_markdown_media` in `library_media_viewer_state.py` — task-32234 made the sniff decide alone for every type; the Analysis tab reuses the same sniff.

#### What the live review got wrong, and what it got right (read before Step 1)

The snapshot says `u`, `o` and `/` "still work" while an Input has focus. **They do not.** Every one of those is a non-priority `Binding`, and a focused `Input`/`TextArea` consumes a printable key before any Screen binding sees it — `library_screen.py` states this at five separate binding sites (1099, 1114, 1123, 1127, 1141). Only `shift+f6` and `_MEDIA_ROW_SELECT_KEY` are `priority=True`. So the footer's current suppression (task-31223, `library_screen.py:4426-4442`) is CORRECT and must not be reversed: re-listing `u use Library context in Console` as a live key while the caret is in the query box is exactly the dead-key lie task-31272 removed.

What IS wrong is that the suppression leaves no route out and no memory of what the keys were. The fix keeps every verb on screen and states the one gesture that re-arms them.

- [ ] **Step 1 (32346): failing test.** In `Tests/UI/test_library_crit10_viewer.py`, build the Library host the way `Tests/UI/test_library_crit9_shell.py` does (`_library_host`, `_active_library_screen`, `_wait_for_library_shell`, `_wait_for_selector` — import them from that module; run it first and confirm it is green on your worktree before you rely on it).

```python
import pytest
from textual.widgets import Input


@pytest.mark.asyncio
async def test_a_focused_search_box_keeps_the_canvas_verbs_behind_one_named_key():
    host = _library_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-search").press()
        query = await _wait_for_selector(screen, pilot, "#library-rag-query-input")
        query.focus()
        await pilot.pause()
        chips = screen._library_footer_shortcuts_for_current_state()
        labels = [label for _key, label in chips]
        assert labels[0] == "typing in field", chips
        assert ("esc", "leave field") in chips, chips
        joined = " ".join(labels)
        assert "after esc: u use Library context in Console · o open evidence" in joined, chips
        # AC#2: "F6 next pane" is LAST, so the responsive footer drops it
        # before any canvas verb.
        assert chips[-1] == ("F6", "next pane"), chips
        # AC#1's honesty half: no single printable key is advertised as live.
        assert not [key for key, _label in chips if len(key) == 1 and key.isprintable()], chips


@pytest.mark.asyncio
async def test_the_media_filter_box_keeps_the_list_verbs_the_same_way():
    host = _library_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-media").press()
        box = await _wait_for_selector(screen, pilot, "#library-media-filter")
        box.focus()
        await pilot.pause()
        chips = screen._library_footer_shortcuts_for_current_state()
        joined = " ".join(label for _key, label in chips)
        assert "after esc: " in joined and "s select" in joined, chips
        assert chips[-1] == ("F6", "next pane"), chips
```

- [ ] **Step 2: run it and show it failing.**
  ```
  cd <worktree> && …/.venv/bin/python -m pytest Tests/UI/test_library_crit10_viewer.py -q -p no:cacheprovider > /tmp/crit10-viewer-s2.txt 2>&1; tail -40 /tmp/crit10-viewer-s2.txt
  ```
  Expected FAIL: `AssertionError` on `("esc", "leave field") in chips` — today the block at 4432–4442 returns `(("", "typing in field"),) + <multi-char pairs>`, which on Search/RAG is exactly `typing in field | F6 next pane` (A caps 14/15).

- [ ] **Step 3 (32346): implement.** Replace the body of the `isinstance(focused, (Input, TextArea))` block in `_library_footer_shortcuts_for_current_state` (`library_screen.py:4432-4442`) with:

```python
        focused = self.focused
        if isinstance(focused, (Input, TextArea)):
            swallowed = tuple(
                pair
                for pair in shortcuts
                if len(pair[0]) == 1 and pair[0].isprintable()
            )
            kept = tuple(
                pair
                for pair in shortcuts
                if not (len(pair[0]) == 1 and pair[0].isprintable())
            )
            # task-32346 (critique #10 P1): the keys really are swallowed --
            # every one of them is a non-priority Binding and the Input eats
            # the keypress first (see the binding comments at 1099/1114/1123),
            # so re-advertising them as live would be the dead-key lie
            # task-31272 removed. What was missing is the way back: the field
            # state now names the gesture that re-arms the canvas, and the
            # verbs stay on screen behind it instead of vanishing.
            if self.is_mounted and not self._library_slash_would_land():
                swallowed = tuple(pair for pair in swallowed if pair[0] != "/")
            esc_pairs = tuple(pair for pair in kept if pair[0] == "esc")
            if not esc_pairs and self.check_action("library_blur_text_field", ()):
                esc_pairs = (("esc", "leave field"),)
            verbs = (
                (
                    (
                        "",
                        "after esc: "
                        + " · ".join(f"{key} {label}" for key, label in swallowed),
                    ),
                )
                if swallowed
                else ()
            )
            # AC#2: "F6 next pane" goes LAST so AppFooterStatus's
            # retain-the-prefix degradation drops it before any canvas verb.
            f6_pairs = tuple(pair for pair in kept if pair[0] == "F6")
            rest = tuple(
                pair for pair in kept if pair[0] not in ("F6", "esc")
            )
            shortcuts = (
                (("", "typing in field"),) + esc_pairs + verbs + rest + f6_pairs
            )
```

  `check_action("library_blur_text_field", ())` is the gate that decides whether `esc leave field` is honest on this surface; if it returns False the chip is simply absent and the kept `esc` (if any) speaks for itself. Run Step 1's tests green, then run the two footer suites and report before/after failing-name sets:
  ```
  cd <worktree> && …/.venv/bin/python -m pytest Tests/UI/test_library_crit8_keyboard.py Tests/UI/test_library_crit9_shell.py -q -p no:cacheprovider > /tmp/crit10-viewer-s3.txt 2>&1; tail -40 /tmp/crit10-viewer-s3.txt
  ```
  `grep -rn "typing in field" Tests/` returns nothing at this baseline — the behaviour is unpinned by name, exactly as task-32346 says. Your two new tests become the pin.

- [ ] **Step 4 (32348): prove how Find opens today, before adding anything.** Find is a Button-press handler only — `@on(Button.Pressed, "#library-media-reader-find")` at `library_screen.py:31516`. There is no key binding for it. The handler arms `self._media_state.find_open = True` and `find_focus_pending = True` and then calls `_after_library_media_viewer_sync(self._focus_library_media_content_search_input)`. The bar itself is mounted by `library_media_viewer.py::_compose_active_body`, and **only two of the four tabs mount one**: `read` (519–532) and `analysis` (1023–1032). `highlights` and `info` mount nothing. The gate in front of the handler, `analysis_find_unavailable_reason` (`library_media_viewer_state.py:541`), returns `""` for every `mode != "analysis"` — so on the Info and Highlights tabs Find is ENABLED, arms `find_open`, mounts nothing, and the focus call finds no input. B's capture `27-media-find.txt` shows the Reader on `Info (selected)` — that is the whole of D4. The subsequent `t` of "token" then reached the screen binding because focus had stayed outside any Input: D4a.

  Write the unit test first:

```python
def test_find_is_refused_on_the_tabs_that_have_no_search_bar():
    from tldw_chatbook.Library.library_media_viewer_state import (
        analysis_find_unavailable_reason,
    )

    for mode in ("info", "highlights"):
        assert analysis_find_unavailable_reason(
            mode=mode, analysis="anything", generating=False, editing=False
        ) == "This tab has no text to search · switch to Read or Analysis.", mode
    assert analysis_find_unavailable_reason(
        mode="read", analysis="", generating=False, editing=False
    ) == ""
    assert analysis_find_unavailable_reason(
        mode="analysis", analysis="", generating=False, editing=False
    ) == "No analysis to search yet."
```

- [ ] **Step 5 (32348): implement the refusal.** In `library_media_viewer_state.py::analysis_find_unavailable_reason`, insert before the `if mode != "analysis":` early return at 541:

```python
    # task-32348 (critique #10, B D4): Find mounts the bar that
    # ``_compose_active_body`` composes, and only the Read and Analysis
    # bodies compose one. On Info/Highlights the gate returned "" (the
    # function only ever considered the Analysis tab), so the button was
    # enabled, armed ``find_open``, mounted nothing, and left focus outside
    # any Input -- where the next typed character fired the screen's own
    # accelerators ("t" armed "Delete this media?", B D4a).
    if mode in ("info", "highlights"):
        return "This tab has no text to search · switch to Read or Analysis."
```

  Update the function's docstring first paragraph to say it answers for every tab, not only Analysis. Do NOT rename it (12 call sites and pins reference the name). Run:
  ```
  cd <worktree> && …/.venv/bin/python -m pytest Tests/UI/test_library_crit10_viewer.py Tests/UI/test_library_media_reader_flow.py Tests/UI/test_library_media_render_fixes.py -q -p no:cacheprovider > /tmp/crit10-viewer-s5.txt 2>&1; tail -40 /tmp/crit10-viewer-s5.txt
  ```
  Report before/after failing-name sets for the two existing files.

- [ ] **Step 6 (32348 AC#1): a keyboard route and a footer chip.** Three edits, all in `library_screen.py`:

  1. Extract the handler body so the key and the button share one implementation. Rename the current body of `handle_library_media_reader_find` (31516–31551) into a new private method and leave the `@on` handler as a two-line forwarder:

```python
    @on(Button.Pressed, "#library-media-reader-find")
    def handle_library_media_reader_find(self, event: Button.Pressed) -> None:
        """Open (or close) the Find bar for the tab being read.

        Args:
            event: The Find button press.
        """
        event.stop()
        self._toggle_library_media_find()

    def action_library_media_reader_find(self) -> None:
        """Open (or close) the Reader's Find bar from the keyboard (task-32348).

        The same gesture the Find button performs -- one implementation, so
        the key can never diverge from the control (the ``check_action``
        gate below is the same reason string the button's own disabled
        state reads).
        """
        self._toggle_library_media_find()

    def _toggle_library_media_find(self) -> None:
        """<the existing handler body, verbatim, minus ``event.stop()``>"""
```

  2. Add the binding immediately after the `("t", "library_media_move_to_trash", …)` entry at 1117:

```python
        # task-32348 (critique #10, B K20): Find had no key at all -- it was
        # a Button and nothing else, so a keyboard-only reader could not open
        # the search bar the guide promises. ``ctrl+f`` is free on this screen
        # (grep the BINDINGS above) and is not a printable key, so it works
        # from inside the Reader's own text controls too.
        Binding("ctrl+f", "library_media_reader_find", "Find", show=False),
```

  3. In `check_action`, extend the media-Reader branch (24213–24243). Add `"library_media_reader_find"` to the tuple at 24213 and, inside that branch, return `not self._library_media_find_unavailable_reason()` for it (placed after the view/substate/pending-detail fences, before the `use_in_console` early return):

```python
            if action == "library_media_reader_find":
                # task-32348: live exactly where the button is enabled, so
                # the footer chip below can never advertise a refusal.
                return not self._library_media_find_unavailable_reason()
```

  4. In `_library_route_shortcuts_for_current_state`'s Reader branch, add `("ctrl+f", "find")` to the gated loop at 4207–4216 (it is a multi-char key, so it survives the Input-focus transformation from Step 3 and stays useful while the Find field itself has focus):

```python
                for key, gated_action, label in (
                    ("ctrl+f", "library_media_reader_find", "find"),
                    ("l", "library_media_read_later", "read later"),
                    ("c", "library_media_use_in_console", "use in Console"),
                    ("t", "library_media_move_to_trash", "trash"),
```

  UI test:

```python
@pytest.mark.asyncio
async def test_ctrl_f_opens_the_reader_find_bar_and_the_footer_names_it():
    host = _library_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_first_media_reader(host, pilot)
        assert ("ctrl+f", "find") in screen._library_footer_shortcuts_for_current_state()
        await pilot.press("ctrl+f")
        await pilot.pause()
        assert screen.query("#library-media-content-search-controls")
        assert screen._media_state.find_open is True
```

- [ ] **Step 7 (32348 AC#2): no destructive single key while Find is pending.** In the same `check_action` branch, before the per-action returns:

```python
            if (
                action == "library_media_move_to_trash"
                and self._media_state.find_open
            ):
                # task-32348 AC#2 (B D4a): with the Find bar open the user is
                # typing a query. The bar takes focus on mount, so this is
                # belt-and-braces -- but the one path where it did NOT (a
                # tab with no bar, Step 5) armed "Delete this media?" from
                # the "t" of "token". The footer chip drops with the gate.
                return False
```

  Test:

```python
@pytest.mark.asyncio
async def test_t_never_arms_the_trash_while_find_is_open():
    host = _library_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_first_media_reader(host, pilot)
        await pilot.press("ctrl+f")
        await pilot.pause()
        assert screen.check_action("library_media_move_to_trash", ()) is False
        assert ("t", "trash") not in screen._library_footer_shortcuts_for_current_state()
        await pilot.press("escape")
        await pilot.pause()
        assert screen.check_action("library_media_move_to_trash", ()) is True
```

- [ ] **Step 8 (32365): failing test for the Analysis tab's raw Markdown.**

```python
@pytest.mark.asyncio
async def test_a_stored_analysis_renders_its_markdown_with_a_raw_toggle():
    host = _analysis_markdown_host()   # one item whose analysis starts "## Key contributions\n"
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_first_media_reader(host, pilot)
        screen.query_one("#library-media-reader-select-analysis", Button).press()
        body = await _wait_for_selector(screen, pilot, "#library-media-viewer-content")
        painted = _painted(host, body.region)
        assert "Key contributions" in painted, painted
        assert "## Key" not in painted, painted
        toggle = screen.query_one("#library-media-analysis-content-mode-raw", Button)
        toggle.press()
        await pilot.pause()
        assert "## Key contributions" in _painted(
            host, screen.query_one("#library-media-viewer-content").region
        )
```

  `_painted` is `Tests/UI/test_library_media_render_fixes.py::_painted` (`app.screen._compositor.render_strips()`); import it.

- [ ] **Step 9 (32365): implement.** In `library_media_viewer.py::_compose_analysis` (983–1090), the `LibraryMediaContentBody` construction at 1029–1040 hard-codes `is_markdown=False, mode="raw"` with the comment "Analysis is plain text -> raw mode, not Markdown". That comment is the prior decision this task reverses, so rewrite it rather than deleting it:

```python
            yield LibraryMediaContentBody(
                content=self.viewer.analysis,
                # task-32365 (critique #10): the Analysis body used to pin
                # `is_markdown=False, mode="raw"` because an analysis was
                # assumed to be plain text. Generated analyses are Markdown
                # in practice ("## Key contributions" painted as source, A
                # cap 43), and task-32234 already made the CONTENT sniff --
                # not a type guess -- the Read tab's authority. Same sniff,
                # same widget, same Raw toggle.
                is_markdown=looks_like_markdown_content(self.viewer.analysis),
                mode=self.analysis_content_mode,
                query=self.content_query,
                match_index=self.content_match_index,
                id="library-media-viewer-content",
            )
```

  Add `analysis_content_mode: str = "rendered"` to `LibraryMediaViewerPanel.__init__` beside `content_mode` (219 area) and thread it from the screen exactly as `content_mode` is threaded (grep `content_mode=` in `library_media_controller.py` and copy the one call site). Then reuse `_compose_content_mode_toggle` (578–657) for the Analysis tab: it already renders the Rendered/Raw pair; give it a `prefix` argument defaulting to `"library-media"` so the Analysis instance yields ids `library-media-analysis-content-mode-rendered` / `-raw`, and wire their presses through the same handler the Read toggle uses (grep `content-mode-raw` in `library_screen.py`). Compose the toggle only when `looks_like_markdown_content(self.viewer.analysis)` is True — a plain-text analysis has nothing to toggle. Import `looks_like_markdown_content` from `tldw_chatbook.Library.library_media_viewer_state`.

  Re-run `Tests/UI/test_library_media_render_fixes.py`, `Tests/UI/test_library_media_reader_scroller_resolution.py`, `Tests/UI/test_library_media_reader_match_nav_t22209.py` and report before/after failing-name sets — the last one drives Find over the analysis corpus and is the pin that the search still works in rendered mode.

- [ ] **Step 10: live-verify** (socket `crit10-viewer`) on the seeded profile at 235x52 and 100x30: open a media item; capture the footer with the filter box focused and with it blurred (32346); press `ctrl+f` and capture the bar plus the footer (32348); switch to Info, press `ctrl+f`, capture the refusal notification and the still-disabled state; type `token` on the Info tab and prove no delete confirmation appears (32348 AC#2); open the analysed item's Analysis tab and capture the rendered heading plus the Raw toggle (32365). Captures to `<SCRATCH>/crit10/wave/viewer/caps/`.

- [ ] **Step 11: docs + backlog + commits.** `Docs/User_Guide/library.md` 259–261: delete the `"enter select evidence"` claim (the chip is dynamic since task-32053) and add one sentence: **"While a text field has focus the footer leads with `typing in field`, names `esc leave field`, and keeps the canvas verbs after `after esc:` — they are not live until you leave the field."** `Docs/User_Guide/library/media-and-conversations.md` 529: append to the Find row **"`Ctrl+F` opens it from the keyboard. On Highlights and Info it is disabled and says so — those tabs have no text to search."**, and add a sentence to the Analysis section: **"A Markdown analysis renders like the Read tab, with the same Rendered/Raw toggle."** Stamp both. Commits: `fix(library): the footer keeps its canvas verbs behind one named key (task-32346)`, `fix(library-media): Find opens from the keyboard and refuses the tabs it cannot search (task-32348)`, `fix(library-media): a Markdown analysis renders (task-32365)`.

---

### Task 2: Media rows — labelled age, a scope line, copy polish (group `media-rows`, tasks 32347, 32350, 32364)

**Files:**
- Modify: `tldw_chatbook/Library/library_media_state.py` (`build_library_media_browse_state` 1062–1220, `LibraryMediaCanvasState` 829–882, `_secondary_text` 1270–1310, the two list call sites at 1124 and 1451)
- Modify: `tldw_chatbook/Widgets/Library/library_media_canvas.py` (`_media_row_label` 188–210, `compose`'s title/filter block 958–1035)
- Modify: `tldw_chatbook/UI/Library_Modules/library_media_controller.py` (the `build_library_media_browse_state(` call at 1887)
- Modify: `tldw_chatbook/Library/library_conversations_state.py` (the secondary builder at 224–225)
- Modify: `tldw_chatbook/Widgets/Library/library_prompts_canvas.py` (the row label's `Prompt · Local ·` prefix — grep `"Prompt · "` in that file)
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` — **only** `LIBRARY_INGEST_SHORTCUTS` (1290–1297)
- Modify: `tldw_chatbook/UI/Library_Modules/library_ingest_controller.py` — **only** `_library_ingest_shortcuts_for_current_state`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` — **only** one new `.library-media-scope-line` rule appended at the end of the Library section
- Test: `Tests/UI/test_library_crit10_media_rows.py` (new), `Tests/Library/test_library_media_state.py` (update the four pinned secondary strings), `Tests/Library/test_library_conversations_state.py` (update the four pinned separator strings)
- Docs: `Docs/User_Guide/library/media-and-conversations.md`, `Docs/User_Guide/library/prompts.md`, `Docs/User_Guide/library/import-and-export.md`

**Interfaces:**
- Consumes: `format_console_relative_age(value, now=…)` (`tldw_chatbook/Workspaces/conversation_browser_state.py:168`) — returns `"now"`, `"10m"`, `"1h"`, `"3d"`, `"2w"`, `"1y"` or `""`. **`"now"` is why the age cannot simply be wrapped in `f"added {age} ago"`.**
- Consumes: `media_trash_age_copy` — already prefixes `"trashed "`, which is why the label must go at the LIST call sites, never inside `_secondary_text` (the Trash secondary would read `pdf · added trashed 3m ago`).
- Consumes: `self._local_source_counts` (a plain dict on the screen, read at `library_screen.py:14788`) — the rail's unfiltered Media total.
- Produces: `media_added_age_copy(value, *, now) -> str` in `library_media_state.py`, mirroring `media_trash_age_copy`; `LibraryMediaCanvasState.scope_line: str`.

- [ ] **Step 1 (32347): name the pin, then write the new one.** `Tests/Library/test_library_media_state.py:590::test_media_secondary_fallback_when_no_type_no_age` is the pin task-32347 names, but read it first: it only asserts the **no-type** fallback (`secondary == "media"`) and does not constrain the `type · age` form at all. The real pins on the format are `test_library_media_state.py:305-306` (`"pdf · 3m"`, `"video · 2h"`) and `:529/:534`. Those four are the assertions this task updates (AC#2: updated, not loosened). `Tests/Library/test_library_media_trash_state.py:490-491` (`"pdf · trashed 3m"`) must stay **byte-identical** — the Trash line is a different age with its own label already.

  New test in `Tests/UI/test_library_crit10_media_rows.py`:

```python
from datetime import datetime, timezone

from tldw_chatbook.Library.library_media_state import media_added_age_copy


def test_the_age_label_says_what_the_age_is():
    now = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)
    assert media_added_age_copy("2026-09-11T11:50:00+00:00", now=now) == "added 10m ago"
    assert media_added_age_copy("2026-09-11T11:59:40+00:00", now=now) == "added just now"
    assert media_added_age_copy("", now=now) == ""
```

- [ ] **Step 2 (32347): implement.** Add to `library_media_state.py`, beside the other age helpers:

```python
def media_added_age_copy(value: str, *, now: datetime) -> str:
    """Return the Media row's age, labelled for what it is.

    task-32347 (critique #10 P1): the bare compact age ("10m") sat where a
    reader of an `audio` or `video` row reads a DURATION -- the same row
    read 11m and 14m later (A caps 36/39/47). The Trash list solved this
    long ago by labelling its own age ("trashed 3m"); this is the browse
    list's half of the same rule. ``format_console_relative_age`` returns
    the word "now" under a minute, which no "N ago" phrasing survives, so
    that case gets its own sentence.

    Args:
        value: The record's timestamp text.
        now: Reference time.

    Returns:
        "added 10m ago", "added just now", or "" when unparseable.
    """
    age = format_console_relative_age(value, now=now)
    if not age:
        return ""
    return "added just now" if age == "now" else f"added {age} ago"
```

  Replace the two browse call sites — `library_media_state.py:1124-1126` and `:1452-1454` — swapping `format_console_relative_age(...)` for `media_added_age_copy(...)` with the same arguments. **Do not touch line 1021** (the Trash row). Update the four pinned strings in `Tests/Library/test_library_media_state.py` to `"pdf · added 3m ago"`, `"video · added 2h ago"`, `"video · added 3m ago"`, `"pdf · added 2h ago"`, and add one line to each of those tests' docstrings naming task-32347. Run:
  ```
  cd <worktree> && …/.venv/bin/python -m pytest Tests/Library/test_library_media_state.py Tests/Library/test_library_media_trash_state.py Tests/UI/test_library_crit10_media_rows.py -q -p no:cacheprovider > /tmp/crit10-media-s2.txt 2>&1; tail -40 /tmp/crit10-media-s2.txt
  ```
  Then `grep -rln "· 3m\"\|· 2h\"\|type · age" Tests/` and fix every other pin that asserts the old form, listing each in your Implementation Notes.

- [ ] **Step 3 (32350): failing test for the scope line.**

```python
@pytest.mark.asyncio
async def test_the_media_header_states_the_applied_scope_not_the_typed_draft():
    host = _host()                 # the media host from test_library_media_render_fixes.py
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _apply_media_filter(screen, pilot, "attention")   # type + Enter
        box = screen.query_one("#library-media-filter", Input)
        box.value = "a different draft"                          # never submitted
        await pilot.pause()
        line = screen.query_one("#library-media-scope-line", Static)
        assert str(line.content).startswith("Media · 1 of "), str(line.content)
        assert 'filter “attention”' in str(line.content), str(line.content)
        assert "a different draft" not in str(line.content), str(line.content)
        assert screen.query("#library-media-scope-clear")


@pytest.mark.asyncio
async def test_clearing_the_applied_filter_also_clears_the_box():
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _apply_media_filter(screen, pilot, "attention")
        screen.query_one("#library-media-scope-clear", Button).press()
        await _wait_for_condition(
            pilot, lambda: not screen.query_one("#library-media-filter", Input).value
        )
        line = screen.query_one("#library-media-scope-line", Static)
        assert "filter" not in str(line.content), str(line.content)
        assert not screen.query("#library-media-scope-clear")
```

- [ ] **Step 4 (32350): implement.** The applied scope is already authoritative inside `build_library_media_browse_state` — `result.scope.query`, `result.scope.media_type`, `result.scope.sort_by`, `result.total`. Build the line there, so the canvas cannot render the Input buffer by accident.

  1. `LibraryMediaCanvasState` (829–882): add `scope_line: str = ""` and `scope_clearable: bool = False` at the end of the field list (both defaulted, so `build_library_media_state`'s callers are unaffected).
  2. `build_library_media_browse_state`: add `unfiltered_total: int | None = None` to the keyword-only signature, and build the line just above the `return LibraryMediaCanvasState(` at 1196:

```python
    # task-32350 (critique #10 P1): the filter box holds a DRAFT until it is
    # submitted (deliberate -- the debounce at
    # library_media_controller.py:2235 is what makes typing usable), so the
    # box and the list can legitimately disagree. Nothing said which one the
    # rows came from. This line is built from the APPLIED scope only, so it
    # cannot echo an unsubmitted draft.
    scope_parts = [
        f"Media · {result.total} of {unfiltered_total}"
        if unfiltered_total is not None and unfiltered_total >= result.total
        else f"Media · {result.total}"
    ]
    if result.scope.query:
        scope_parts.append(f"filter “{result.scope.query}”")
    scope_parts.append(
        f"type {result.scope.media_type}"
        if result.scope.media_type is not None
        else "all types"
    )
    scope_parts.append(f"sort {library_choice_label('sort', result.scope.sort_by)}")
    scope_line = " · ".join(scope_parts)
    scope_clearable = bool(result.scope.query) or result.scope.media_type is not None
```

  Import `library_choice_label` from `tldw_chatbook.Library.library_shell_state` (the same helper `#library-media-sort`'s button label uses at `library_media_canvas.py:1254` — grep it and use the identical call so the two can never disagree). Pass both new values into the constructor.
  3. `library_media_controller.py:1887`: pass
     ```python
             unfiltered_total=(
                 self._local_source_counts.get("media")
                 if self._library_loaded and not self._library_lookup_error
                 else None
             ),
     ```
  4. `library_media_canvas.py::compose`: immediately after the `title_row` block (ends at 1012) and before `filter_row`, yield

```python
        scope_row = Horizontal(id="library-media-scope-row")
        scope_row.styles.height = "auto"
        with scope_row:
            scope_static = Static(
                self.canvas.scope_line,
                id="library-media-scope-line",
                classes="library-media-scope-line",
                markup=False,
            )
            scope_static.styles.width = "auto"
            yield scope_static
            if self.canvas.scope_clearable:
                yield Button(
                    "Clear",
                    id="library-media-scope-clear",
                    classes="library-canvas-action",
                    compact=True,
                )
```

     Route `#library-media-scope-clear` to the existing clear path: add an `@on(Button.Pressed, "#library-media-scope-clear")` beside the one at 453 that calls the same `actions.handle_library_media_filter_clear(event)`, and extend the controller's `handle_library_media_filter_clear` (2249–2253) to ALSO reset the type facet and blank the Input widget:

```python
    @on(Button.Pressed, "#library-media-filter-clear")
    @on(Button.Pressed, "#library-media-scope-clear")
    def handle_library_media_filter_clear(self, event: Button.Pressed) -> None:
        event.stop()
        self._stop_library_media_filter_timer()
        # task-32350 AC#2: clearing the APPLIED filter clears the box too --
        # leaving a draft behind is exactly the split-brain this task closes.
        box = self.query("#library-media-filter")
        if box:
            box.first(Input).value = ""
        self._request_library_media_filter("")
```

  5. TCSS, appended once at the end of the Library section of `_agentic_terminal.tcss`:

```
/* task-32350 (critique #10 P1): the applied-scope line under the Media
   header. Quiet by design -- it is a statement of fact, not an action. */
.library-media-scope-line {
    color: $ds-text-muted;
    width: auto;
    height: 1;
    padding: 0 1 0 0;
}
```

  Rebuild the bundle (`python -m tldw_chatbook.css.build_css`) and commit the regenerated files. Run `Tests/UI/test_library_media_toolbar_adapt.py`, `Tests/UI/test_library_media_render_fixes.py`, `Tests/UI/test_library_multiselect_media.py` and report before/after failing-name sets.

- [ ] **Step 5 (32364 AC#1): a status word never prefixes a title.** `library_media_canvas.py:188-210` builds `" Loaded · <title> · <secondary>"` — the state is spliced into the title position. Move it to the secondary line, where every other row fact lives:

```python
    visible_title = _visible_row_title(title)
    # task-32364 AC#1 (critique #10): the state used to prefix the TITLE
    # ("▸ Loaded · Attention Is All You Need"), so the row's identity was
    # displaced by its status. task-30044's constraint still holds -- the
    # SHORT word, never the old prose -- it just belongs on the fact line.
    state = "Loading" if loading else "Loaded" if loaded else ""
    detail = f"{secondary} · {state.lower()}" if state else secondary
    if compact:
        return f" {visible_title} · {detail}"
    return f" {visible_title}\n    {detail}"
```

  `grep -rn "Loaded · " Tests/` and update every pin, listing them in the notes.

- [ ] **Step 6 (32364 AC#2): one separator.** `library_conversations_state.py:224` returns `f"{message_count} messages - {age}"`. Change the hyphen to `·`:

```python
        # task-32364 AC#2: every other Library list separates row facts with
        # "·"; Conversations was the one hyphen.
        return f"{message_count} messages · {age}"
```

  Update `Tests/Library/test_library_conversations_state.py:77,80,267` to the `·` form. In the same file, `"New Chat"` is the untitled-conversation display title (grep `New Chat` there) — replace it with `"Untitled conversation"`, matching `library_media_state.py`'s `"Untitled media"` (`_record_title`, 1232). In `library_prompts_canvas.py`, drop the leading `Prompt · ` from the row's secondary (the canvas is already titled Prompts) and replace `System + User` with `has system and user text` — grep both literals in that file and in `Tests/`.

- [ ] **Step 7 (32364 AC#3): the import footer names what Enter does.** `LIBRARY_INGEST_SHORTCUTS` (`library_screen.py:1290-1297`) says `("enter", "start")` for both steps, but the first Enter only validates ("1 will import") and the second runs (A caps 07/08). `_library_ingest_shortcuts_for_current_state` in `library_ingest_controller.py` already returns a per-state set — make the Enter label state-dependent there, using the same predicate the Start gate uses (grep `start_enabled` / `can_start` in that method):

```python
        # task-32364 AC#3: the first Enter validates the path and the second
        # runs the queue; one label for two different actions taught the
        # wrong thing at exactly the moment the user commits.
        enter_label = "start import" if <start gate open> else "check this path"
```

  Assert both strings in a unit test that drives the controller's state directly.

- [ ] **Step 8: live-verify** (socket `crit10-media-rows`) on the seeded profile at 235x52 and 100x30: the media list rows reading `audio · added 10m ago` (32347); the scope line with a filter applied and a different draft in the box, then Clear (32350); a selected row whose title is not prefixed (32364 AC#1); a conversation row's `·` (AC#2); the Import canvas footer before and after the first Enter (AC#3). Captures to `<SCRATCH>/crit10/wave/media-rows/caps/`.

- [ ] **Step 9: docs + backlog + commits.** `Docs/User_Guide/library/media-and-conversations.md`: update the row-anatomy paragraph to the new secondary form and document the scope line and its Clear; `Docs/User_Guide/library/prompts.md`: drop the `Prompt · Local ·` example; `Docs/User_Guide/library/import-and-export.md`: state the two-step Enter. Stamp all three. Commits: `fix(library-media): the row age says it is an age (task-32347)`, `fix(library-media): a scope line states the applied filter (task-32350)`, `fix(library): copy polish across the lists and the import footer (task-32364)`.

---

### Task 3: Onboarding and Import outcomes (group `onboarding-import`, tasks 32349, 32351)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` — **only** `_library_landing_attention_action` (13131–13175) and the all-empty branch of the onboarding evidence apply (20161–20175)
- Modify: `tldw_chatbook/Library/library_ingest_state.py` (`build_ingest_queue_groups`'s `_flush`, 2670–2700)
- Modify: `tldw_chatbook/Library/library_rail_state.py` — **docstring only** (`coerce_library_lifecycle`, 82–94)
- Test: `Tests/UI/test_library_crit10_onboarding.py` (new), `Tests/Library/test_library_ingest_state.py` (extend)
- Docs: `Docs/User_Guide/library.md` (the "Back to Get started" row at 312), `Docs/User_Guide/library/import-and-export.md`

**Interfaces:**
- Consumes: `aggregate_library_lifecycle(lifecycle, evidence)` (`library_rail_state.py:117`) — already promotes `UNKNOWN` + six `EMPTY` sources to `STARTER`, and `HAS_USER_CONTENT` to `GRADUATED`. **The six-source all-`EMPTY` evidence snapshot IS this wave's definition of "no content yet"; no new predicate is invented.**
- Consumes: `self._library_lifecycle_was_stored` (`library_screen.py:2208`, set from `_load_library_lifecycle_value` at 19887) — True only when `[library.rail_state] lifecycle` actually exists in config.
- Consumes: `IngestJobState` (`tldw_chatbook/Library/library_ingest_jobs.py:93`) — `QUEUED/PARSING/WRITING/DONE/FAILED/SKIPPED`.

#### How 32349 decides "no content yet" — and why the obvious fix is wrong

`coerce_library_lifecycle(raw=None, is_new_profile=False)` returns `EXPANDED` (`library_rail_state.py:91-94`). That is PROVEN as the cause: none of the four crit10 scratch profiles carries a `[library.rail_state]` section at all (`grep -n "rail_state" $SCRATCH/crit10/*/*/config.toml` → nothing), so a fresh profile whose config file pre-dates its first Library visit resolves EXPANDED, renders the full nine-row rail, and — because "Back to Get started" shows exactly when `lifecycle is EXPANDED and onboarding_all_empty` (`library_rail.py:902`) — offers a return to a view never seen. Assessor B's claim that the profile "carries `lifecycle = \"starter\"`" is wrong; there is no such key.

**Do not change the default to `UNKNOWN`.** That default exists for a reason: the evidence read is a worker, and the screen renders with `_library_onboarding_status = LOADING` before it lands. A returning user with content would then see the compact starter rail flash for the duration of the read. The `EXPANDED` default is the no-flash default and stays.

The bug is in the settle, not the default: `aggregate_library_lifecycle` refuses to demote `EXPANDED`, and it cannot tell an `EXPANDED` the user CHOSE (they pressed Explore, so `raw == "expanded"` is stored) from an `EXPANDED` nobody chose. The screen can: `_library_lifecycle_was_stored`. So an **unstored** `EXPANDED` is treated as provisional and, once the evidence settles all-EMPTY, falls back to `UNKNOWN` so the existing aggregate resolves it to `STARTER`. A returning user with content never reaches that branch at all — `HAS_USER_CONTENT` takes the `GRADUATED` branch above it.

- [ ] **Step 1 (32349): failing test.**

```python
@pytest.mark.asyncio
async def test_a_pre_written_config_with_no_content_lands_on_get_started():
    host = _empty_library_host()          # no [library.rail_state] in app_config
    async with host.run_test(size=(235, 52)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_onboarding_status is LibraryEvidenceStatus.SETTLED,
        )
        assert screen._library_lifecycle is LibraryLifecycle.STARTER, screen._library_lifecycle
        assert not screen.query("#library-rail-back-to-starter")


@pytest.mark.asyncio
async def test_a_returning_user_with_content_never_sees_the_starter_rail():
    host = _seeded_library_host()         # no [library.rail_state], one media item
    async with host.run_test(size=(235, 52)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        # before the evidence settles the full rail is already on screen
        assert screen._library_lifecycle is LibraryLifecycle.EXPANDED
        await _wait_for_condition(
            pilot,
            lambda: screen._library_onboarding_status is LibraryEvidenceStatus.SETTLED,
        )
        assert screen._library_lifecycle is LibraryLifecycle.GRADUATED


@pytest.mark.asyncio
async def test_an_explicit_explore_keeps_the_expanded_rail_when_empty():
    host = _empty_library_host(rail_state={"lifecycle": "expanded"})
    async with host.run_test(size=(235, 52)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_onboarding_status is LibraryEvidenceStatus.SETTLED,
        )
        assert screen._library_lifecycle is LibraryLifecycle.EXPANDED
        assert screen.query("#library-rail-back-to-starter")
```

- [ ] **Step 2 (32349): implement.** Replace the all-empty branch at `library_screen.py:20166-20172` with:

```python
        elif len(evidence) == 6 and all(
            item is LibraryContentEvidence.EMPTY for item in evidence
        ):
            self._library_onboarding_all_empty = True
            self._library_onboarding_status = LibraryEvidenceStatus.SETTLED
            lifecycle = self._library_lifecycle
            # task-32349 (critique #10, PROVEN in critique #8): an EXPANDED
            # nobody chose is a DEFAULT, not a decision --
            # ``coerce_library_lifecycle(raw=None, is_new_profile=False)``
            # returns EXPANDED so a returning user's full rail does not
            # flash a starter rail while this evidence loads. Once the
            # evidence settles all-EMPTY there is nothing to expand, so an
            # UNSTORED EXPANDED falls back to UNKNOWN and the aggregate
            # below resolves it to STARTER. A STORED "expanded" is a real
            # Explore press (``explore_library_lifecycle``) and is left
            # alone -- which is also what keeps "Back to Get started"
            # (``library_rail.py:902``: EXPANDED + all-empty) offered only
            # to someone who HAS seen Get started.
            if (
                lifecycle is LibraryLifecycle.EXPANDED
                and not self._library_lifecycle_was_stored
            ):
                lifecycle = LibraryLifecycle.UNKNOWN
            self._set_library_lifecycle(
                aggregate_library_lifecycle(lifecycle, evidence)
            )
```

  Add a cross-reference line to `coerce_library_lifecycle`'s docstring (`library_rail_state.py`, after line 89): `"An absent value defaults to EXPANDED so a returning profile does not flash the starter rail; the screen demotes an unstored EXPANDED once the six-source evidence settles all-EMPTY (task-32349)."` **No code change in that module.** Run:
  ```
  cd <worktree> && …/.venv/bin/python -m pytest Tests/UI/test_library_crit10_onboarding.py Tests/Library/test_library_rail_state.py Tests/Widgets/Library/test_library_rail.py Tests/UI/test_library_crit8_polish_shell.py -q -p no:cacheprovider > /tmp/crit10-onb-s2.txt 2>&1; tail -40 /tmp/crit10-onb-s2.txt
  ```
  Report before/after failing-name sets for the three existing files.

- [ ] **Step 3 (32351 AC#1): failing test for the batch name.**

```python
def test_a_recursive_folder_import_is_named_after_the_folder_the_user_chose():
    jobs = tuple(
        _queued_job(f"job-{n}", source_path=path, batch_id="b1")
        for n, path in enumerate(
            (
                "/tmp/inbox/reading-notes.md",
                "/tmp/inbox/article.html",
                "/tmp/inbox/lecture-transcript.txt",
                "/tmp/inbox/export.json",
                "/tmp/inbox/weird.xyz",
                "/tmp/inbox/nested/deep-note.md",
            )
        )
    )
    groups, _latest = build_ingest_queue_groups(jobs, now=NOW)
    assert len(groups) == 1
    assert groups[0].header_line.startswith("inbox — 6 files"), groups[0].header_line


def test_a_batch_whose_paths_share_no_root_falls_back_to_the_first_parent():
    jobs = (
        _queued_job("a", source_path="https://example.com/one", batch_id="b1"),
        _queued_job("b", source_path="https://example.com/two", batch_id="b1"),
    )
    groups, _latest = build_ingest_queue_groups(jobs, now=NOW)
    assert groups[0].header_line.startswith("example.com — 2 files")
```

  (Build `_queued_job` from the existing factory in `Tests/Library/test_library_ingest_state.py`; reuse its `NOW`.)

- [ ] **Step 4 (32351 AC#1): implement.** `library_ingest_state.py`'s `_flush` picks `source = PurePath(str(members[0].source_path)).parent.name or "batch"` — the parent of the FIRST member, which for a recursive scan is whichever subdirectory happened to be enumerated first ("nested", B D1). Replace that one line with:

```python
        # task-32351 AC#1 (critique #10, B D1): the first member's parent is
        # whichever subdirectory the recursive scan enumerated first, so a
        # six-file import of `inbox/` was labelled "nested" after its one
        # nested file. The folder the USER chose is the common root of every
        # member, which the members already carry -- no new field, no schema
        # change. ``commonpath`` raises on mixed absolute/relative or
        # non-path sources (URL imports), which keeps the old behaviour.
        parents = [str(PurePath(str(job.source_path)).parent) for job in members]
        try:
            source = PurePath(os.path.commonpath(parents)).name or "batch"
        except ValueError:
            source = PurePath(str(members[0].source_path)).parent.name or "batch"
```

  Add `import os` to the module's stdlib imports if absent. Run `Tests/Library/test_library_ingest_state.py` and `Tests/Library/test_library_ingest_jobs.py`, report before/after failing-name sets.

- [ ] **Step 5 (32351 AC#2): the landing card states the outcome.** `_library_landing_attention_action` (`library_screen.py:13131-13150`) returns the neutral `"An import needs review."` on the first live FAILED job. Replace that loop with a tally:

```python
        registry = self._library_ingest_registry()
        jobs_fn = getattr(registry, "jobs", None)
        if callable(jobs_fn):
            live = tuple(
                job
                for job in jobs_fn()
                if not job.permanent and not job.dismissed and not job.superseded
            )
            failed = sum(1 for job in live if job.state is IngestJobState.FAILED)
            skipped = sum(1 for job in live if job.state is IngestJobState.SKIPPED)
            if failed:
                # task-32351 AC#2 (critique #10, B D2): after 4 of 6 files
                # failed the landing said only "An import needs review." --
                # neutral where the queue itself was exact. The counts are
                # already in the registry snapshot this loop walks.
                noun = "file" if failed == 1 else "files"
                parts = [f"{failed} {noun} failed"]
                if skipped:
                    parts.append(f"{skipped} skipped")
                return LibraryLandingAttentionAction(
                    message=f"Last import: {', '.join(parts)}.",
                    action_label="Review",
                    action_kind="ingest-review",
                )
```

  Import `IngestJobState` at the top of `library_screen.py` if it is not already imported (grep first). Test:

```python
@pytest.mark.asyncio
async def test_the_landing_card_counts_the_failures_it_is_asking_about():
    host = _library_host_with_failed_import(failed=4, skipped=2)
    async with host.run_test(size=(235, 52)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        action = screen._library_landing_attention_action()
        assert action.message == "Last import: 4 files failed, 2 skipped."
        assert action.action_label == "Review"
```

  `grep -rn "An import needs review" Tests/` and update every pin.

- [ ] **Step 6: live-verify** (socket `crit10-onboarding-import`): on the **fresh** profile, launch, skip the wizard, open Library, and capture the Get started landing plus the compact rail with NO "Back to Get started" row (32349); press Explore and capture that the row appears; quit, relaunch and capture that the choice survived. On the **fresh** profile, point Import at `<profile>/inbox` — every job will fail at parse-pool start with the host's `[Errno 28]`, which is the fixture — and capture the queue group reading `inbox — 6 files` (32351 AC#1) and the landing's `Last import: 4 files failed, 2 skipped.` (AC#2). Captures to `<SCRATCH>/crit10/wave/onboarding-import/caps/`. State in the report that the import SUCCESS path was not exercisable on this host.

- [ ] **Step 7: docs + backlog + commits.** `Docs/User_Guide/library.md:312`: replace the "Back to Get started … never offered after graduation" sentence with **"It is offered only after you have explicitly expanded a still-empty Library from Get started, and never after graduation."** `Docs/User_Guide/library/import-and-export.md`: state that a folder batch is named after the folder you chose, and that the landing card names the failure counts. Stamp both. Commits: `fix(library): an empty profile lands on Get started whatever its config's age (task-32349)`, `fix(library-import): a batch is named after the folder you chose, and the landing counts the failures (task-32351)`.

---

### Task 4: Pagers, Collections and the Skills trust banner (group `pagers`, tasks 32352, 32354, 32363, 32057)

**Files:**
- Modify: `tldw_chatbook/Library/library_pager_state.py` (add `simple_library_pager_display`)
- Modify: `tldw_chatbook/Widgets/Library/library_skills_canvas.py` (`_compose_pager` 1320–1380)
- Modify: `tldw_chatbook/Widgets/Library/library_collections_capture_reader.py` (the canvas heading, the empty state at 421, the pager at 452–500)
- Modify: `tldw_chatbook/Widgets/Library/library_media_trash_canvas.py` (the pager block at 452–530)
- Modify: `tldw_chatbook/Library/library_skills_state.py` (`skill_trust_header_line`'s `needs_setup` branch, 811–815)
- Test: `Tests/UI/test_library_crit10_pagers.py` (new), `Tests/UI/test_library_skills_canvas.py` (update the pin), `Tests/Library/test_library_pager_state.py` (extend)
- Docs: `Docs/User_Guide/library/collections.md`, `Docs/User_Guide/library/skills.md`, `Docs/User_Guide/library.md` (the "empty Browse rows … pager mechanics" claim at 223 — the guide's own line number from the docs table)

**Interfaces:**
- Consumes: `library_pager_layout(pager, retry_visible=None) -> LibraryPagerLayout` (`library_pager_state.py:51`) — carries the task-28016/31237/32104 single-page rule: on `single_page` it keeps the range copy, drops `page_copy` and both boundary reasons, and hides the controls unless a Retry is visible.
- Consumes: `library_disabled_action_label(label, disabled)` (`library_shell_state.py:127`) — the `○` marker every other Library pager uses.
- Produces: `simple_library_pager_display(...)` in `library_pager_state.py`, for a source that pages itself and cannot satisfy `build_library_pager_display`'s row-count invariants.

#### The pinned decision this task reverses (32354 AC#2)

`Tests/UI/test_library_skills_canvas.py::test_skills_canvas_renders_exact_pager_and_source_wide_trust_count` pins the Skills canvas's four-line pager verbatim: the range copy, `Page 1 of 1`, the boundary reasons and the two `○ Previous ○ Next` forms. It is a real prior decision — the Skills canvas has composed its own `_compose_pager` (1320) since before `library_pager_layout` existed, and PR #2104 routed Media, Conversations and Prompts through the shared rule without touching it. **The reversal is deliberate and named**: at 60x24 that block costs 4 of 18 usable rows and pushes the second of two skills off the bottom (A caps 60/61), and `library.md` already documents the single-page suppression as the Library-wide rule. Update the pin to assert the SUPPRESSED shape for one page and the FULL shape for two; never delete it and never loosen it to a substring check.

- [ ] **Step 1 (32354): update the pin as a failing test.** Read `test_skills_canvas_renders_exact_pager_and_source_wide_trust_count` in full, then split it into two tests in the same file, keeping the trust-count assertions in the first:

Copy the existing test's host construction and its `await` preamble verbatim into both new tests (it is the only fixture that produces a mounted Skills canvas); the only thing that changes between them is how many skills the host seeds.

```python
@pytest.mark.asyncio
async def test_skills_canvas_suppresses_single_page_chrome_and_keeps_the_trust_count():
    """One page of skills renders the range and nothing else (task-32354)."""
    host = <the existing test's host, seeded with its two skills>
    async with host.run_test(size=(235, 52)) as pilot:
        screen = <the existing test's preamble, verbatim>
        assert str(screen.query_one("#library-skills-range", Static).content) == "1-2 of 2"
        assert not screen.query("#library-skills-page"), "Page 1 of 1 has nowhere to page to"
        assert not screen.query("#library-skills-pager-status")
        assert not screen.query(f"#{LIBRARY_SKILLS_PAGE_PREVIOUS_ID}"), "no dead Previous/Next"
        <the existing source-wide trust-count assertions, moved here unchanged>


@pytest.mark.asyncio
async def test_skills_canvas_renders_the_full_pager_when_a_second_page_exists():
    """A second page brings back every part of the pager (task-32354)."""
    host = <the same host, seeded with one more skill than the controller's page size>
    async with host.run_test(size=(235, 52)) as pilot:
        screen = <the same preamble, verbatim>
        status = str(screen.query_one("#library-skills-range", Static).content)
        assert status.endswith("· Page 1 of 2"), status
        assert screen.query(f"#{LIBRARY_SKILLS_PAGE_PREVIOUS_ID}")
        assert str(
            screen.query_one("#library-skills-pager-status", Static).content
        ) == "Already on the first page."
```

The page size is the Skills controller's own (`grep -n "page_size" tldw_chatbook/UI/Library_Modules/library_skills_controller.py`); seed one more than that, and paste the real number into your report.

- [ ] **Step 2 (32354): implement Skills.** `library_skills_canvas.py::_compose_pager` already receives a real `LibraryPagerDisplay` from its controller, so this is the Media shape verbatim. Replace the hand-rolled `reasons` block (1322–1331) and the body with:

```python
    def _compose_pager(self, pager: LibraryPagerDisplay) -> ComposeResult:
        """Render the controller-derived Skills pager through the shared rule.

        task-32354 (critique #10): this canvas composed the pager itself and
        so never got the single-page suppression task-28016/31237 gave every
        other Library list -- four lines of "Page 1 of 1 / Already on the
        first page. / ○ Previous ○ Next" for two skills, 4 of 18 usable rows
        at 60x24 (A caps 60/61). The rule is ``library_pager_layout``'s; this
        method only renders what it returns.
        """
        layout = library_pager_layout(pager)
        with Vertical(id="library-skills-pager", classes="library-source-pager"):
            yield Static(
                " · ".join(layout.status_parts),
                id="library-skills-range",
                classes="library-source-pager-status",
                markup=False,
            )
            status_copy = " · ".join(
                copy
                for copy in (pager.status_copy, *layout.boundary_reasons)
                if copy
            )
            if status_copy:
                yield Static(
                    status_copy,
                    id="library-skills-pager-status",
                    classes="library-source-pager-status",
                    markup=False,
                )
            if layout.controls_hidden:
                return
            <the existing Horizontal control block, unchanged>
```

  Note the id change: the range Static now carries the joined `status_parts` (range + page copy when a second page exists), so the separate `#library-skills-page` Static is gone on one page and present on two — which is what the two tests above assert. Import `library_pager_layout` alongside the existing `LibraryPagerDisplay` import. Run the Skills suites and report before/after failing-name sets:
  ```
  cd <worktree> && …/.venv/bin/python -m pytest Tests/UI/test_library_skills_canvas.py Tests/UI/test_library_crit10_pagers.py -q -p no:cacheprovider > /tmp/crit10-pagers-s2.txt 2>&1; tail -40 /tmp/crit10-pagers-s2.txt
  ```

- [ ] **Step 3 (32354 + 32352 AC#3): the shared factory, then Collections.** `library_collections_capture_reader.py` (452–500) builds `range_copy` by hand and yields Previous/Next with plain `disabled=` and no `○` marker at all — structurally identical to Trash's pager but visually different (B D6). It cannot use `build_library_pager_display`: that function raises `ValueError` when `row_count` disagrees with `applied_page`/`total`, and this canvas's page size and totals come from a service that can return a short page. Add a small factory to `library_pager_state.py` instead:

```python
def simple_library_pager_display(
    *,
    range_copy: str,
    page: int,
    total_pages: int,
    has_previous: bool,
    has_next: bool,
) -> LibraryPagerDisplay:
    """Build a pager display for a source that pages itself (task-32354).

    ``build_library_pager_display`` validates row counts against an exact
    total and raises when they disagree -- correct for the sources that own
    their paging end to end, fatal for one whose service may return a short
    page. This carries the same copy and the same boundary reasons so those
    sources can still go through ``library_pager_layout``.

    Args:
        range_copy: The already-formatted item range line.
        page: The applied page number.
        total_pages: Known page count, or 0 when it is unknown.
        has_previous: Whether a previous page can be loaded.
        has_next: Whether a next page can be loaded.

    Returns:
        A display whose ``single_page`` is True when neither direction can move.
    """
    return LibraryPagerDisplay(
        title_count=None,
        range_copy=range_copy,
        page_copy=f"Page {page} of {total_pages}" if total_pages else "",
        status_copy="",
        previous_disabled=not has_previous,
        next_disabled=not has_next,
        previous_reason="" if has_previous else _FIRST_PAGE_REASON,
        next_reason="" if has_next else _FINAL_PAGE_REASON,
        retry_visible=False,
        single_page=not has_previous and not has_next,
    )
```

  In `library_collections_capture_reader.py`, replace the block from `yield Static(range_copy, id="library-collections-page-range", …)` to the end of the `#library-collections-page-toolbar` `Horizontal` with:

```python
        total_pages = (
            0
            if state.exact_total is None
            else max(1, (state.exact_total + _CAPTURE_PAGE_SIZE - 1) // _CAPTURE_PAGE_SIZE)
        )
        pager = simple_library_pager_display(
            range_copy=range_copy,
            page=current_page,
            total_pages=total_pages,
            has_previous=has_previous,
            has_next=has_next,
        )
        layout = library_pager_layout(pager)
        yield Static(
            " · ".join(layout.status_parts),
            id="library-collections-page-range",
            markup=False,
        )
        if layout.boundary_reasons:
            yield Static(
                " · ".join(layout.boundary_reasons),
                id="library-collections-page-reason",
                markup=False,
            )
        if not layout.controls_hidden:
            with Horizontal(classes="ds-toolbar", id="library-collections-page-toolbar"):
                yield Button(
                    library_disabled_action_label("Previous", not has_previous),
                    id="library-collections-page-previous",
                    compact=True,
                    disabled=not has_previous,
                    tooltip=pager.previous_reason or "Load the previous page.",
                )
                yield Button(
                    library_disabled_action_label("Next", not has_next),
                    id="library-collections-page-next",
                    compact=True,
                    disabled=not has_next,
                    tooltip=pager.next_reason or "Load the next page.",
                )
```

  Introduce `_CAPTURE_PAGE_SIZE = 20` as a module constant and use it for the two existing `* 20` expressions above (they are the same magic number). Test, pinning both 32352 AC#3 and 32354 AC#1 for this canvas:

```python
@pytest.mark.asyncio
async def test_collections_shows_no_pager_chrome_at_zero_of_zero():
    host = _collections_host(total=0)
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_collections(host, pilot)
        assert not screen.query("#library-collections-page-toolbar")
        assert str(screen.query_one("#library-collections-page-range", Static).content) == "0–0 of 0"
```

- [ ] **Step 4 (32354): Trash.** `library_media_trash_canvas.py:452-530` renders range + page copy + controls unconditionally when `bounded`. Route it the same way: build `layout = library_pager_layout(self.pager, retry_visible=self.retry_visible)`, put `" · ".join(layout.status_parts)` in `#library-media-trash-range`, drop the separate `#library-media-trash-page` Static when `layout.status_parts` has one element, render `layout.boundary_reasons` when present, and `return` before the controls `Horizontal` when `layout.controls_hidden`. The pager `Vertical` pins `styles.height = 2` / `min_height = 2` at 454–455 — set both to `1` when `layout.controls_hidden`, otherwise leave them at 2. Add a test at `(60, 24)` asserting a single-page Trash shows no `#library-media-trash-pager-controls`, and re-run `Tests/UI/test_library_media_trash*.py` (`ls Tests/UI | grep trash`), reporting before/after failing-name sets.

- [ ] **Step 5 (32352 AC#1 + AC#2 + task-32057): one name, and a true empty state.** The user's decision for task-32057 is recorded below and implemented here.

  1. **AC#1 (one name).** The rail row says `Collections (N)`; the canvas heading says `Quick Capture`. Keep the rail row — rename the CANVAS. Grep the heading (`grep -rn "Quick Capture" tldw_chatbook/Widgets/Library/library_collections_capture_reader.py`): the canvas title becomes `Collections`, and the `#library-collections-quick-capture` **Button** keeps its own label `Quick Capture` (it is a distinct action — saving a URL — and renaming it would lose the verb). Assert in a test that the painted canvas contains `Collections` as its heading and that `Quick Capture` appears only on that button.
  2. **AC#2 (a true empty state).** `library_collections_capture_reader.py:421` reads `"No captures match this scope. Clear filters or save a URL with Quick Capture."` on a profile with zero captures and no filter set. Split the two cases; the filter case must be reachable only when a filter is set:

```python
        page = state.page
        if page is None or not page.items:
            # task-32352 AC#2 (critique #10): one sentence covered both an
            # empty collection and a filtered-to-nothing one, so a profile
            # that had never saved anything was told to clear filters it had
            # never set, and pointed at an action ("Quick Capture") as if it
            # were somewhere else on the screen.
            scope = state.requested_scope
            filtered = bool(
                scope is not None
                and (getattr(scope, "query", "") or getattr(scope, "tags", ()) )
            )
            empty_copy = (
                "No captures match these filters · clear them to see everything saved."
                if filtered
                else "No saved captures yet · press Quick Capture above to save a page by URL."
            )
            yield Static(
                empty_copy,
                id="library-collections-items-empty",
                classes="destination-purpose",
                markup=False,
            )
```

     Use the real filter-bearing attributes you find on `state.requested_scope` (grep the scope dataclass); if it carries none, the `filtered` branch is unreachable and you must say so in the notes rather than inventing a field.

- [ ] **Step 6 (32057): record the decision in the task file.** The user has decided AC#1 of task-32057. Append this exactly, under a new `## Decision` heading, to `backlog/tasks/task-32057 - Library-Collections-row-undocumented-captures-browser-legacy_read_only-service-and-rail-side-effects.md`, immediately before the existing `## Implementation Notes` section:

```markdown
## Decision

Decided by the user, 2026-09-11 (critique-10 fix wave, branch `fix/library-crit10-pagers`).

Neither Option A nor Option B. **The rail row stays, and the feature has one
name everywhere: "Collections".** The canvas heading "Quick Capture" is
retired in favour of "Collections"; "Quick Capture" survives only as the
label of the button that saves a URL, because that is a verb and not a
place. The canvas gets a true empty state that does not mention filters
unless one is set and does not point at an action that is not on screen.
The local service stays read-only — no schema migration, no membership
model, no "Add to collection" affordance this wave — and the guide says so.

Implemented on the crit10 `pagers` branch alongside task-32352.
```

  Then `backlog task edit 32057 --notes "<one-line pointer to the Decision section and this branch>"` and tick AC#1 (`- [x] #1`).

- [ ] **Step 7 (32363): the Skills trust banner states the precedence.** `library_skills_state.py:811-815` returns `"Skill trust isn't set up — set it up to review and use skills."` for the `needs_setup` posture, under which BOTH the trust-approved and the unapproved seeded skill render `⚠ needs review` (B D5). The list cannot distinguish them — with no trust store there is nothing to verify an approval against — so take AC#1's second branch and make the banner say so:

```python
    if posture == "needs_setup":
        # task-32363 (critique #10, B D5): with no trust store every skill
        # reads "needs review", including ones approved earlier -- the list
        # looked wrong rather than unverifiable. The banner now states the
        # precedence instead of leaving the reader to infer it.
        return (
            "Skill trust isn't set up, so every skill reads \"needs review\" — "
            "set it up to review and use skills.",
            "setup",
        )
```

  `grep -rn "Skill trust isn't set up" Tests/ tldw_chatbook/ Docs/` and update every pin and doc occurrence. Test the exact string in `Tests/Library/test_library_skills_state.py` (or the file that already covers `skill_trust_header_line` — grep for it).

- [ ] **Step 8: live-verify** (socket `crit10-pagers`) on the seeded profile at 235x52, 100x30 and 60x24: Skills with two skills (no pager chrome at any width; the second skill visible at 60x24 — this is the A cap 61 regression proof); Collections at `0–0 of 0` (no controls, the new empty sentence, the heading reading `Collections`); Trash empty and Trash with one item. Captures to `<SCRATCH>/crit10/wave/pagers/caps/`.

- [ ] **Step 9: docs + backlog + commits.** `Docs/User_Guide/library/collections.md`: the canvas is called Collections; describe the two empty states; restate that the local service is read-only. `Docs/User_Guide/library/skills.md`: the new banner sentence. `Docs/User_Guide/library.md`: the claim that empty Browse rows drop the "page 1 of 1" mechanics is now true for Skills, Collections and Trash too — say so. Stamp all three. Commits: `fix(library): Skills, Collections and Trash use the shared single-page pager rule (task-32354)`, `fix(library-collections): one name and a true empty state (task-32352, task-32057)`, `fix(library-skills): the trust banner states why every skill needs review (task-32363)`.

---

### Task 5: Export — full fidelity by default, a manifest, adjacent reasons (group `export`, tasks 32353, 32362)

**Files:**
- Modify: `tldw_chatbook/Library/library_export_state.py` (`DEFAULT_MEDIA_QUALITY` 61, the state dataclass 130–190, the builder 230–270)
- Modify: `tldw_chatbook/Widgets/Library/library_export_canvas.py` (`compose` 101–256)
- Modify: `tldw_chatbook/UI/Library_Modules/library_export_controller.py` (the counts worker — grep `counts` / `_apply_library_export_counts`)
- Modify: `tldw_chatbook/Widgets/Library/library_media_viewer.py` — **only** `_compose_primary_toolbar` (384–403). **Do NOT touch `_compose_active_body` — Task 1 owns it.**
- Test: `Tests/UI/test_library_crit10_export.py` (new), `Tests/Library/test_library_export_state.py` (extend)
- Docs: `Docs/User_Guide/library/import-and-export.md`, `Docs/User_Guide/library/media-and-conversations.md` (the Find row's disabled state)

**Interfaces:**
- Consumes: `media_quality_helper_copy(media_quality)` (`library_export_state.py:75`) — the per-option caption; unchanged.
- Consumes: `apply_library_export_submit_gate(button, state)` (`library_export_state.py`) — already sets the `○` label, the `disabled` flag and the tooltip in one place.
- Consumes: the inline-reason precedent in `library_media_viewer.py:1077-1085` (`#library-media-analysis-generate-reason`, task-31981) — a `Static` with class `library-media-action-reason` directly under the blocked control. Every reason this task adds uses that same class.

- [ ] **Step 1 (32353 AC#1): failing test, then a one-line default change.**

```python
def test_export_defaults_to_full_fidelity():
    from tldw_chatbook.Library.library_export_state import DEFAULT_MEDIA_QUALITY
    assert DEFAULT_MEDIA_QUALITY == "original"
```

  Implement:

```python
MEDIA_QUALITY_OPTIONS = ("thumbnail", "compressed", "original")
# task-32353 AC#1 (critique #10): the default used to be "thumbnail", which
# "keeps a small preview image instead of the full file" -- a silent data
# reduction chosen for someone whose reason for exporting is usually to keep
# the files. A lossy bundle is now something you ask for.
DEFAULT_MEDIA_QUALITY = "original"
```

  `grep -rn "thumbnail" Tests/ tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/UI/Wizards/` and update every pin that asserts the default. **Leave `tldw_chatbook/Chatbooks/chatbook_models.py:546` and `chatbook_creator.py:268` alone** — those are the Chatbooks service's own defaults, a different surface with its own callers, and `ChatbookCreationWizard.py:944` passes an explicit value. Say in the notes that you checked them.

- [ ] **Step 2 (32353 AC#2): failing test for the manifest line.**

```python
@pytest.mark.asyncio
async def test_the_export_canvas_says_what_the_bundle_will_contain():
    host = _export_host(media=2)      # two selected media items
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_export(host, pilot)
        await _wait_for_condition(
            pilot, lambda: not screen._export_state.counts_loading
        )
        line = screen.query_one("#library-export-consequence-line", Static)
        assert str(line.content) == "Bundle: 2 media items · full files · about 4 KB", str(line.content)
        contents = screen.query_one("#library-export-contents", Static)
        assert "Attention Is All You Need" in str(contents.content)
```

- [ ] **Step 3 (32353 AC#2): implement.** Three edits.
  1. `library_export_controller.py`: the counts worker already returns per-kind counts (grep `_apply_library_export_counts`). Extend its query with a byte total for the selected media — one `SELECT COALESCE(SUM(LENGTH(content)), 0) FROM Media WHERE id IN (…)` on the same connection, parameterised, in the same worker — and carry it back as `approx_bytes: int | None` (None when the scope is unbounded or the query fails; never raise out of the worker).
  2. `library_export_state.py`: add `consequence_line: str = ""` and `contents_lines: tuple[str, ...] = ()` to the state, and build them in the same builder that produces `scope_line` (230–270):

```python
    # task-32353 AC#2 (critique #10): the canvas asked for a destination and
    # a name and then wrote a bundle nobody had seen the contents of, at a
    # fidelity chosen by a control rendered at the same weight as "sort".
    # This states the consequence in one line, above the button.
    fidelity = {
        "original": "full files",
        "compressed": "compressed files",
        "thumbnail": "previews only",
    }[media_quality]
    size = (
        f" · about {format_byte_size(approx_bytes)}"
        if approx_bytes is not None
        else " · size known once it runs"
    )
    consequence_line = (
        f"Bundle: {_count_phrase(total, 'item')} · {fidelity}{size}"
        if not counts_loading
        else ""
    )
    contents_lines = tuple(titles[:20]) + (
        (f"+ {len(titles) - 20} more",) if len(titles) > 20 else ()
    )
```

     Reuse the existing `_count_phrase` helper added by task-32221 — it lives in `tldw_chatbook/Library/library_export_scope.py:220`, so import it from there rather than writing a second one ("media" pluralises as "media items" via that module's own special case at 248). For `format_byte_size`, reuse the formatter `_library_db_sizes_line` already uses (`grep -n "def .*byte\|MB\|_format_size" tldw_chatbook/UI/Screens/library_screen.py` — that helper produced "8.2MB"); if it lives on the screen, move it to `library_export_state.py`-adjacent shared code ONLY if it has no other caller, otherwise import it. Do not write a second formatter.
  3. `library_export_canvas.py::compose`: yield the consequence line and a contents disclosure immediately BEFORE `submit_button` (228), using the always-mounted display-toggle discipline the three quiet lines above already follow:

```python
        consequence_line = Static(
            state.consequence_line,
            id="library-export-consequence-line",
            classes="library-export-quiet-line",
            markup=False,
        )
        consequence_line.display = bool(state.consequence_line)
        yield consequence_line
        contents = Static(
            "\n".join(state.contents_lines),
            id="library-export-contents",
            classes="library-export-quiet-line",
            markup=False,
        )
        contents.display = bool(state.contents_lines)
        yield contents
```

- [ ] **Step 4 (32362): the Export button's reason sits beside it.** `apply_library_export_submit_gate` already computes the reason for the tooltip; the visible text `No destination chosen` is three rows up (B D12). Yield an inline reason directly under the submit button, the same shape task-31981 used in the Reader:

```python
        apply_library_export_submit_gate(submit_button, state)
        yield submit_button
        submit_reason = Static(
            state.submit_blocked_reason,
            id="library-export-submit-reason",
            classes="library-media-action-reason",
            markup=False,
        )
        submit_reason.display = bool(state.submit_blocked_reason)
        yield submit_reason
```

  Expose `submit_blocked_reason` on the export state from whatever `apply_library_export_submit_gate` already reads for its tooltip — **one source, not a second sentence**; if the gate computes the string locally, lift it into a module-level function and have both the tooltip and this Static call it. Test:

```python
@pytest.mark.asyncio
async def test_the_blocked_export_button_carries_its_reason_on_the_next_line():
    host = _export_host(media=2, destination="")
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_export(host, pilot)
        button = screen.query_one("#library-export-submit", Button)
        reason = screen.query_one("#library-export-submit-reason", Static)
        assert button.disabled
        assert str(button.label).startswith("○ ")
        assert str(reason.content) == button.tooltip
        assert reason.region.y == button.region.y + button.region.height
```

- [ ] **Step 5 (32362): the Reader's `○ Find` carries its reason too.** In `library_media_viewer.py::_compose_primary_toolbar` (384–403) the reason reaches a mouse tooltip only. Yield it inline, matching the Analysis-tab precedent in the same file:

```python
            if find_reason:
                find.disabled = True
                find.tooltip = find_reason
            yield find
            if find_reason:
                # task-32362 (critique #10, A cap 42): the "○" said blocked
                # and nothing said why unless you hovered -- the same gap
                # task-31981 closed for Generate, one toolbar over.
                yield Static(
                    find_reason,
                    id="library-media-reader-find-reason",
                    classes="library-media-action-reason",
                    markup=False,
                )
```

  **Dispatch note:** the reason string for the Info and Highlights tabs comes from Task 1's change to `analysis_find_unavailable_reason`. This branch works against whatever that function returns today; after Task 1 lands, re-run this branch's test and confirm the Info-tab case renders `This tab has no text to search · switch to Read or Analysis.`

- [ ] **Step 6: run the export suites and compare failing-name sets** — `ls Tests/UI | grep export` and `ls Tests/Library | grep export`, then run everything that names, plus `Tests/UI/test_library_media_reader_flow.py` (the primary toolbar's geometry pins). Report only new names.

- [ ] **Step 7: live-verify** (socket `crit10-export`) on the seeded profile at 235x52 and 100x30: the Export canvas with two media selected — capture the default `quality: original` with its helper line, the consequence line, the contents list, and the `○ Export bundle (.zip)` with its reason on the line below; choose a destination and capture the enabled button; open a media item's Info tab and capture the `○ Find` with its reason. Captures to `<SCRATCH>/crit10/wave/export/caps/`.

- [ ] **Step 8: docs + backlog + commits.** `Docs/User_Guide/library/import-and-export.md`: the default is full fidelity, the consequence line and the contents list are described, the blocked button's reason is inline. `Docs/User_Guide/library/media-and-conversations.md`: the Find row's disabled states carry their reason on the line below. Stamp both. Commits: `fix(library-export): full fidelity by default, with the bundle stated before it is written (task-32353)`, `fix(library): every blocked Export and Reader control carries its reason on the next line (task-32362)`.

---

### Task 6: Panes, grips and the Conversations hand-off (group `layout`, tasks 32355, 32359, 32360, 32361, 32107)

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py` (the grip label/geometry at 130–175 and the pane-open rule the guide describes)
- Modify: `tldw_chatbook/Widgets/Library/library_conversations_canvas.py` (the list/reader width split)
- Modify: `tldw_chatbook/Widgets/Library/library_conversation_reader.py` (`_blocked_reason_line` 60–86, `_workspace_link_offered` 130–145, the action block 236–290)
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` — **only** `_link_selected_conversation_to_workspace` (12919–12971), the two `@on` handlers at 33806–33826, and the `_library_narrow_stage_return_active()` block of `_library_footer_shortcuts_for_current_state` (4450–4466)
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` — the rail-row block (1950–1995) and new reader-shell/conversations rules appended after it
- Test: `Tests/UI/test_library_crit10_layout.py` (new)
- Docs: `Docs/User_Guide/library.md` (the Nav handle at 214/308, the rail-beside-Notes claim at 273), `Docs/User_Guide/library/media-and-conversations.md` (the Conversations split, the hand-off)

**Interfaces:**
- Consumes: `library_disabled_action_label` and the `library-media-action-reason` inline-reason class.
- Consumes: `registry.link_membership(workspace_id, item_type=…, item_id=…, title=…)` and `registry.unlink_membership(...)` (`tldw_chatbook/Workspaces/registry_service.py:1114` and `:1198`) — the link and its inverse both already exist, which is what makes one undoable step possible without new service code.
- Consumes: `loaded_metadata["_workspace_block"] / ["_workspace_block_detail"] / ["_workspace_block_linkable"]` — the established controller→reader injection seam (`library_conversation_reader.py:118-145`). The receipt rides the same seam.
- Consumes: `WorkspaceEligibility.reason_code` and `linkable_ineligibility_label` (`tldw_chatbook/Workspaces/eligibility.py:73-99`) — **read only this wave; do not edit that module.**

#### The 32107 decision the user has made

`eligibility.py:73-92` returns `active_context_eligible=False` with `reason_code` `not_in_active_workspace` or `cross_workspace` for anything outside the active workspace, and `_LINKABLE_REASON_LABELS` (99–102) already records that both of those are resolvable by a link. Task-32056 (PR #2523) turned that into a disabled `Use as source` plus a separate `Link to workspace` button — correct as far as it went, and the reason it produces is the best-written refusal on the screen. But every one of the six seeded conversations is blocked on a fresh profile (A cap 54), so the product's headline hand-off is off by default, gated on a noun Library never introduces.

**Decision (user, 2026-09-11): link-on-use.** Pressing `Use as source` on a conversation whose only block is a link-resolvable one links it to the active workspace and proceeds, in one step, with a receipt and an Undo. The separate `Link to workspace` button stays for the case where someone wants membership without a hand-off.

**What the gate still protects, and keeps protecting:** membership decides which items a Console turn may read, so a hand-off that silently widened the workspace would change what the model can see without anyone saying so. That is why the link is not silent — it is a visible, reversible act with its own receipt — and why the two non-linkable blocks (`no_active_workspace`, and the aggregate `LIBRARY_GENERIC_WORKSPACE_BLOCK` fallback for an item missing from the row model) keep refusing exactly as they do today.

- [ ] **Step 1 (32107): failing tests.**

```python
@pytest.mark.asyncio
async def test_use_as_source_links_an_unlinked_conversation_and_proceeds():
    host = _conversations_host(linked=False)
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_first_conversation(host, pilot)
        source = screen.query_one("#library-conversation-use-source", Button)
        assert not source.disabled, "a link-resolvable block no longer disables the hand-off"
        source.press()
        await pilot.pause()
        receipt = screen.query_one("#library-conversation-link-receipt", Static)
        assert str(receipt.content) == (
            "✓ linked · Local Default · this conversation can now be used in Console"
        ), str(receipt.content)
        assert screen.query_one("#library-conversation-link-undo", Button).display
        assert host.app.last_handoff_payload is not None      # the hand-off still ran


@pytest.mark.asyncio
async def test_undo_removes_the_membership_the_press_added():
    host = _conversations_host(linked=False)
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_first_conversation(host, pilot)
        screen.query_one("#library-conversation-use-source", Button).press()
        await pilot.pause()
        screen.query_one("#library-conversation-link-undo", Button).press()
        await pilot.pause()
        assert host.registry.unlinked == [("conversation", "conv-1")]
        assert not screen.query_one("#library-conversation-link-receipt", Static).display


@pytest.mark.asyncio
async def test_a_block_a_link_cannot_resolve_still_refuses():
    host = _conversations_host(linked=False, reason_code="no_active_workspace")
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_first_conversation(host, pilot)
        source = screen.query_one("#library-conversation-use-source", Button)
        assert source.disabled
        assert str(source.label).startswith("○ ")
```

- [ ] **Step 2 (32107): implement.** Four edits, all inside this branch's owned ranges.
  1. `library_conversation_reader.py:244` — `source.disabled = not self._actions_enabled()`. Keep the load fence, drop the link-resolvable workspace block from it:

```python
            # task-32107 (user decision, critique #10 P1): a block a link can
            # resolve no longer disables the hand-off -- pressing it links
            # the conversation into the active workspace and proceeds, in one
            # undoable step. The load fence and the blocks a link cannot
            # resolve still disable it; membership is what decides what a
            # Console turn may read, so the widening is visible and
            # reversible rather than silent.
            source.disabled = not self._actions_enabled() and not self._workspace_link_offered()
```

     Use the real predicate you find at `_actions_enabled`; if it folds the load fence and the workspace block together, split it so the two answers stay separable, and keep `_open_console_disabled_tooltip`'s load sentence winning when the fence is what blocks.
  2. `library_conversation_reader.py`, after the `link` Button (276–281): yield the receipt and its Undo.

```python
            receipt_workspace = str(
                self.loaded_metadata.get("_workspace_link_receipt") or ""
            ).strip()
            receipt = Static(
                f"✓ linked · {receipt_workspace} · this conversation can now be "
                "used in Console",
                id="library-conversation-link-receipt",
                classes="library-conversation-reader-block-reason",
                markup=False,
            )
            receipt.display = bool(receipt_workspace)
            yield receipt
            undo = Button(
                "Undo link",
                id="library-conversation-link-undo",
                classes="library-canvas-action",
                compact=True,
            )
            undo.display = bool(receipt_workspace)
            yield undo
```

  3. `library_screen.py:12919` — make `_link_selected_conversation_to_workspace` return the linked workspace's display name (`str` , `""` on failure) instead of `None`, and set `self._conversations_state.reader_loaded_metadata["_workspace_link_receipt"] = <name>` on success. Clear that key whenever a different conversation is loaded (find the single place `reader_loaded_metadata` is replaced and confirm the key does not survive — it is a fresh dict per load, so confirm by test rather than by adding a clear).
  4. `library_screen.py:33806` — in `use_selected_conversation_as_source`, link first when the block is link-resolvable, then proceed:

```python
    @on(Button.Pressed, "#library-conversation-use-source")
    def use_selected_conversation_as_source(self, event: Button.Pressed) -> None:
        """Stage the loaded transcript, linking it first when that is the block.

        Args:
            event: Source action press forwarded to the browse controller.
        """
        if self._library_conversation_link_would_unblock():
            # task-32107: one gesture, not two. A failed link leaves the
            # refusal exactly as it was and does not stage anything.
            if not self._link_selected_conversation_to_workspace():
                return
        return self._conversations_controller.use_selected_conversation_as_source(event)

    @on(Button.Pressed, "#library-conversation-link-undo")
    def undo_selected_conversation_workspace_link(self, event: Button.Pressed) -> None:
        """Remove the membership the last "Use as source" press added.

        Args:
            event: The Undo press, stopped here like its sibling handlers.
        """
        event.stop()
        self._undo_selected_conversation_workspace_link()
```

     `_library_conversation_link_would_unblock()` reads the same three metadata keys the reader's `_workspace_link_offered` reads — write it beside `_link_selected_conversation_to_workspace` and have the reader's predicate and this one call it, so the button's enabled state and the handler's decision cannot disagree. `_undo_selected_conversation_workspace_link` mirrors the link method exactly: same fence, same registry, `registry.unlink_membership(...)`, clear `_workspace_link_receipt`, `_invalidate_library_workspace_depth_state()`, `_sync_library_conversation_reader()`.

- [ ] **Step 3 (32107): record the decision in the task file.** Append this exactly, under a new `## Decision` heading at the end of `backlog/tasks/task-32107 - Library-hand-off-consistency-Conversations-Open-in-Console-is-disabled-when-ineligible-while-the-rails-Use-in-Console-stays-blocked-but-pressable-TASK-716.md`:

```markdown
## Decision

Decided by the user, 2026-09-11 (critique-10 fix wave, branch `fix/library-crit10-layout`).

**Link-on-use.** Pressing "Use as source" on a conversation whose only block
is one a link can resolve (`not_in_active_workspace`, `cross_workspace`)
links it into the active workspace and proceeds, in one step, with a
"✓ linked · <workspace>" receipt and an "Undo link" beside it. The separate
"Link to workspace" button stays for membership without a hand-off.

What the gate protects, and keeps protecting: workspace membership decides
which items a Console turn is allowed to read, so a hand-off that widened it
silently would change what the model can see without anyone saying so — which
is why the link is a visible, reversible act with its own receipt rather than
an implicit side effect, and why `no_active_workspace` and the aggregate
`LIBRARY_GENERIC_WORKSPACE_BLOCK` fallback still refuse exactly as they do
today.

The rail's `#library-use-in-console` keeps TASK-716's pressable-with-reason
grammar: it acts on a SET whose members may be blocked for different reasons,
so there is no single link that would unblock it.
```

  Tick both ACs and set the task Done with notes pointing at this branch.

- [ ] **Step 4 (32361): the Conversations panes.** B measured the list at ~140 columns and 80% empty while the reader wrapped at ~48 (B D11, caps 42/43). Failing test:

```python
@pytest.mark.asyncio
async def test_the_conversation_reader_takes_the_majority_of_a_wide_terminal():
    host = _conversations_host(linked=True)
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_first_conversation(host, pilot)
        reader = screen.query_one("#library-conversation-reader")
        items = screen.query_one("#library-conversations-list")
        assert reader.region.width > items.region.width, (reader.region, items.region)
        assert reader.region.width >= 100, reader.region
```

  Implement in `library_conversations_canvas.py`: with a conversation open, give the items pane a bounded width and the reader `1fr` — the same shape Media already uses (`grep -n "items_width\|list_grows" tldw_chatbook/Utils/adaptive_reader_state.py` and copy the resolved rule rather than inventing a ratio). Run the test at `(235, 52)`, `(100, 30)` and `(60, 24)` and paste the three measured pairs into your report.

- [ ] **Step 5 (32355 + 32360 AC#3): the grips.** `library_adaptive_reader_shell.py:151-156` paints `"<---"` / `"--->"` (or a one-cell guillemet below four cells) with the label carried only by `_name`/`tooltip` — invisible in a terminal. Two changes:
  1. **Label the handle** (32355 AC#1's second branch, which the guide already promises at `library.md:308`): when the grip is ≥ 5 cells wide, the collapsed label becomes `Nav ›` for the Library pane and `Items ›` for the Items pane (expanded: `‹ Nav` / `‹ Items`), built from `self.pane_label` so the two grips cannot diverge. Below 5 cells keep the guillemet. Assert the painted label at 235, 100 and 60 columns.
  2. **Stop overpainting** (32360 AC#3): B measured three literal grip runs inside the content area at 235x52 (`17-media-list.txt`, char columns 42, 42, 183) and a `<---` landing inside the "Unfiled" row at 60x24. `render()` (164+) paints arrows at computed rows of the grip's own `content_region` — the overpaint is the grip's region overlapping its neighbour, not the render. Measure first: assert in a test that `grip.region.right <= neighbour.region.x` for both grips at all three sizes, paste the measured regions into your report, and fix with the smallest CSS or width change the measurement justifies (the grip's own fixed `width`/`min_width`/`max_width` are set at 130–132). If the regions are already disjoint, the overpaint is a paint-over the region assertion cannot see (`backlog/docs/lessons-live-verification.md`, "Region assertions are blind to paint-over") — in that case prove it with `_painted` over the neighbour's region and fix the paint, not the geometry.
  3. **Keep the rail beside the editor** (32355 AC#1's first branch): the rail collapses to a grip while a note editor is open at 235 columns (A cap 11), which `library.md:273` says should not happen at ≥120 columns. Find the rule that closes the Library pane on an editor route (grep `library_open` in `library_adaptive_reader_shell.py` and in the `[library.*_reader]` config readers) and gate the auto-collapse on width < 120. **If the collapse turns out to be a persisted user preference rather than a rule** (`library_open = true/false` in the profile's `[library.reader]` section — the crit10 fresh profile has an EMPTY `[library.reader]` and the power profile has `library_open = true`), then the live observation is a preference, not a defect: keep the code, tick AC#1 via the labelled handle, and say so in the notes with the config evidence.

- [ ] **Step 6 (32360 AC#1 + AC#2): the single-stage return and the ellipsis.** At 60x24 Escape works but the footer at that width paints only the global cluster (B `59-60x24-escape-from-canvas.txt`). task-32225 already puts `("esc", "back to Library")` FIRST for exactly this reason, so the chip is registered and the responsive tier is dropping the whole context. In the `_library_narrow_stage_return_active()` block (`library_screen.py:4450-4466`), return a SHORT set at this width instead of prepending to a long one:

```python
        elif self._library_narrow_stage_return_active():
            # task-32360 AC#1 (critique #10, B D8): prepending the chip was
            # not enough -- at 60 columns AppFooterStatus drops the whole
            # screen context to "… F1 F6 Ctrl+P Ctrl+Q", chip and all. A
            # two-chip context fits inside that budget, so the return
            # survives the narrowest tier it exists for.
            shortcuts = (("esc", "back to Library"),) + tuple(
                pair for pair in shortcuts if pair[0] == "F6"
            )
```

  Test it by asserting the registered context at `(60, 24)` and by a painted assertion that `back to Library` appears in the footer row's paint. For AC#2 (copy clipping mid-word — "Create a note or add from", "New     Sort: Newest     Sele"), the Library rail rows already set `text-overflow: ellipsis` (`_agentic_terminal.tcss:1966`); add the same declaration to the canvas status and toolbar classes that clip (identify them from the capture, assert with `_painted` that the clipped string ends in `…`).

- [ ] **Step 7 (32359): the focused rail row differs from the active one by shape.** B measured both as bold+underline over backgrounds 3 RGB units apart (`16-rail-media-focused.ansi` vs `09-import-tab14.ansi`). `.library-rail-row-selected` (1971–1975) owns the ACTIVE treatment; focus currently borrows the Button default. This is a CSS-only fix — do not touch `library_rail.py`:

```
/* task-32359 (critique #10, B D9): the active destination and the focused
   row rendered identically (bold + underline, rgb(25,68,102) vs
   rgb(28,70,102)). Focus everywhere else on this screen is a SHAPE -- the
   house `█` left bar (task-31983) -- so the rail stops being the exception.
   Active keeps the background treatment above; focus adds the bar. */
.library-rail-row:focus {
    border-left: thick $ds-action-focus;
    padding: 0 1 0 0;
}
```

  Test with the painted-frame idiom from `Tests/UI/test_library_row_focus_cue_t31983.py` (`app.screen._compositor.render_strips()`, `_THICK_LEFT_GLYPH = "█"`): focus the Media rail row while Conversations is active, assert `█` paints on the focused row and NOT on the active one, then swap and assert the reverse. Rebuild the bundle and commit the regenerated files.

- [ ] **Step 8: run the suites and compare failing-name sets** — `Tests/UI/test_library_conversations*.py`, `Tests/UI/test_library_adaptive_reader*.py` (`ls Tests/UI | grep -i 'conversation\|adaptive\|reader_shell'`), `Tests/Widgets/Library/test_library_rail.py`, `Tests/UI/test_library_row_focus_cue_t31983.py`, `Tests/UI/test_library_crit9_shell.py`. Report only new names.

- [ ] **Step 9: live-verify** (socket `crit10-layout`) on the seeded profile at 235x52, 100x30 and 60x24: a conversation open with the reader wider than the list (32361); both grips reading `Nav ›` / `‹ Items` and not overpainting a row (32355, 32360 AC#3); a note editor open at 235 columns (32355 AC#1); the footer at 60x24 showing `esc back to Library` (32360 AC#1) and a clipped line ending in `…` (AC#2); `capture-pane -e` of the rail with focus on one row and the active marker on another (32359); `Use as source` on an unlinked conversation, capturing the receipt and then Undo (32107). Captures to `<SCRATCH>/crit10/wave/layout/caps/`.

- [ ] **Step 10: docs + backlog + commits.** `Docs/User_Guide/library.md`: the handle's real label (214, 308) and whatever the rail-beside-editor rule turns out to be (273). `Docs/User_Guide/library/media-and-conversations.md`: the Conversations split, and link-on-use with its receipt and Undo. Stamp both. Commits: `fix(library): pane grips carry their label (task-32355)`, `fix(library-rail): focus is a shape, not a second blue (task-32359)`, `fix(library): the single-stage return survives the narrowest footer (task-32360)`, `fix(library-conversations): the reader gets the width on a wide terminal (task-32361)`, `feat(library-conversations): Use as source links the conversation in one undoable step (task-32107)`.

---

### Task 7: Notes create/draft and the Details panel (group `notes-details`, tasks 32356, 32358, 32357)

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_notes_canvas.py` (the create-mode rows, 400–430 and 695–715)
- Modify: `tldw_chatbook/UI/Library_Modules/library_notes_controller.py` (`_library_note_status_line` 1408–1426, the Blank-note create path at 5595)
- Modify: `tldw_chatbook/Widgets/Library/library_rail.py` (`_compose_details_body_children`, 817–905)
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` — **only** `_workspace_handoff_summary_label` (13608–13660), the Details widgets block (13770–13800) and `_library_db_sizes_line` / its in-place patcher (14934+)
- Test: `Tests/UI/test_library_crit10_notes_details.py` (new)
- Docs: `Docs/User_Guide/library/notes.md`, `Docs/User_Guide/library.md` (the Details section)

**Interfaces:**
- Consumes: `library_dim_label_text(label, value)` — the Details rows' label/value grammar (grep it in `library_rail.py`).
- Consumes: `linkable_ineligibility_label(reason_code)` (`Workspaces/eligibility.py:105`) — already used by `_workspace_handoff_summary_label`. Read only.
- Consumes: `DestinationRailSectionHeader(…, open=…)` (`library_rail.py:880`) — the rail's own disclosure widget; the diagnostics disclosure reuses it rather than inventing a second collapsible.

- [ ] **Step 1 (32356): failing test.** `ctrl+n` opens a create canvas of Blank note plus eight dated templates (A cap 10), with Blank focused. The answer is almost always Blank.

```python
@pytest.mark.asyncio
async def test_ctrl_n_opens_a_blank_note_straight_away():
    host = _notes_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await pilot.press("ctrl+n")
        await _wait_for_selector(screen, pilot, "#library-note-body")
        assert not screen.query("#library-notes-create-blank"), "no chooser between ctrl+n and the editor"
        assert screen.focused is screen.query_one("#library-note-body")


@pytest.mark.asyncio
async def test_templates_are_one_row_inside_the_editor():
    host = _notes_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await pilot.press("ctrl+n")
        await _wait_for_selector(screen, pilot, "#library-note-body")
        opener = screen.query_one("#library-note-from-template", Button)
        assert str(opener.label) == "From a template…"
        opener.press()
        await pilot.pause()
        assert len(screen.query(".library-notes-template-row")) == 8
```

- [ ] **Step 2 (32356): implement.** Take the AC's second branch (one `From a template…` row) rather than moving eight templates into the editor — it is the smaller diff and keeps the create canvas as the one place templates live. In `library_notes_canvas.py`'s create mode: render the Blank-note action and a single `From a template…` Button (`#library-note-from-template`); the eight template rows render only when that opener has been pressed (a `templates_open: bool` on the canvas's create state, toggled by the opener, the same disclosure shape `quick_capture_open` uses in the Collections canvas). Then make `ctrl+n` skip the canvas entirely: `action_library_notes_new` currently routes to create mode — have it call the Blank-note create path directly (`library_notes_controller.py:5595`'s handler body; extract it the way Task 1 extracts the Find handler, so button and key share one implementation) and leave the create canvas reachable from the rail's `Create ▸ New note` row for the template path. Update the create-mode status line at 710 from `"Choose Blank note or a template."` to `"Next: Start typing, or choose a template."`.

- [ ] **Step 3 (32358): failing test, then the chip.**

```python
@pytest.mark.asyncio
async def test_the_draft_chip_and_the_list_agree_at_every_moment():
    host = _notes_host(notes=0)
    async with host.run_test(size=(235, 52)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await pilot.press("ctrl+n")
        await _wait_for_selector(screen, pilot, "#library-note-body")
        status = screen._library_note_status_line()
        listed = bool(screen.query(".library-notes-row"))
        assert (status == "Empty note — type to keep it") is listed, (status, listed)
```

  Implement in `library_notes_controller.py:1419`. The row IS committed to the DB the moment Blank note is pressed (LIB-14: `_library_note_pending_blank_gc_id` exists precisely because the empty row is real and gets garbage-collected if abandoned), so "Draft — not saved yet" contradicts the list and the count beside it (A cap 11). Say the true thing instead:

```python
        # task-32358 (critique #10): the row is already committed here --
        # that is what ``_library_note_pending_blank_gc_id`` exists to clean
        # up (LIB-14) -- so "Draft — not saved yet" contradicted the list and
        # the rail count on screen beside it. What is true is that an empty
        # note will not survive being abandoned.
        if self._library_note_is_pending_blank():
            return "Empty note — type to keep it"
```

  `grep -rn "Draft — not saved yet" Tests/ Docs/` and update every pin and doc line (task-32133 AC#2 is the prior decision this refines — cite it in the notes: the intent, "an untouched blank note must not claim Saved", is preserved).

- [ ] **Step 4 (32357 AC#1): the Details panel speaks outcomes.** Two strings.
  1. `library_screen.py:13787` — `"Server sync WIP · local only"`. `WIP` is engineering status in shipped copy (A cap 20). Replace with **`"Everything here is stored on this machine · syncing to a server isn't available yet."`**
  2. `_workspace_handoff_summary_label` (13608–13660) produces `Handoff · 0 eligible · 1 blocked · not in this workspace`. The count is right; the sentence is a status report, not a task. Keep the reason derivation exactly as task-32230 built it and change only the rendering, so the eligibility logic is untouched:

```python
        # task-32357 AC#1 (critique #10, A cap 20): the line was a correct
        # engineering summary of a concept the reader had never met. The
        # counts and the reason are unchanged; the sentence is now the task
        # the reader can act on.
        noun = "item" if blocked_count == 1 else "items"
        return f"{blocked_count} {noun} can't be used in Console yet · {reason} · {remedy}"
```

     Use the `reason` and `remedy` locals the method already computes. Keep the zero-blocked case exactly as it is. Assert both strings in a unit test that feeds a state with one blocked row and with none.

- [ ] **Step 5 (32357 AC#2): DB sizes behind a diagnostics disclosure.** `_compose_details_body_children` (`library_rail.py:817-905`) yields the DB-sizes rows inline (`#library-details-db-sizes`, plus the `-1`/`-2` continuation rows task-32230 added). Move them under a closed-by-default `DestinationRailSectionHeader("Diagnostics", section_id="library-details-diagnostics", open=False, id="library-rail-section-header-diagnostics")` with a `display`-toggled body, following the `details_body` construction two blocks above it verbatim. **Keep `#library-details-db-sizes` as the id of the first size row** — `grep -rn "library-details-db-sizes" Tests/` first and keep every pin green; a pin that queries the id while the section is closed still passes, because the rail's closed bodies keep their children mounted (see the TASK-23025 comment at 890–899). Test: the Diagnostics header is present and closed on mount; `#library-details-db-sizes` is queryable but not `display`ed; pressing the header displays it.

- [ ] **Step 6: run the suites and compare failing-name sets** — `Tests/UI/test_library_notes_wave_editor_keys.py`, `Tests/UI/test_library_notes_wave_list.py`, `Tests/UI/test_library_notes_characterization.py`, `Tests/Widgets/Library/test_library_rail.py`, `Tests/UI/test_destination_shells.py` (the seven tests that read the closed Details body — named in `library_rail.py:893-899`). Report only new names.

- [ ] **Step 7: live-verify** (socket `crit10-notes-details`) on the **fresh** profile at 235x52 and 100x30: `ctrl+n` landing directly in an editor with the chip reading `Empty note — type to keep it` while the list and the rail count already show the note (32356, 32358); the create canvas from the rail with one `From a template…` row that opens eight (32356); Details open, showing the new sync sentence, the handoff task sentence, and a closed `Diagnostics` header with the DB sizes inside it (32357). Captures to `<SCRATCH>/crit10/wave/notes-details/caps/`.

- [ ] **Step 8: docs + backlog + commits.** `Docs/User_Guide/library/notes.md`: `ctrl+n` opens a blank note directly; templates live behind one row on the create canvas; the editor's empty-note chip. `Docs/User_Guide/library.md`: the Details section's new sentences and the Diagnostics disclosure. Stamp both. Commits: `fix(library-notes): ctrl+n opens a note instead of a nine-option chooser (task-32356)`, `fix(library-notes): the empty-note chip agrees with the list (task-32358)`, `fix(library-rail): the Details panel speaks outcomes, not internals (task-32357)`.

---

### Task 8: The guide's remaining contradictions (group `docs`, task 32366)

> **DISPATCH LAST.** Every row this task owns is a claim no other branch's code change makes true. Start only after Tasks 1–7 have reported; re-read each page at that point, because seven branches are stamping the same two files.

**Files:**
- Modify: `Docs/User_Guide/library.md`
- Modify: `Docs/User_Guide/library/notes.md`, `Docs/User_Guide/library/search-and-rag.md`
- Test: none (docs only) — but every claim you keep must be checked live before you keep it
- Docs: the above

**How the 14 rows are distributed.** Each row is fixed by the branch that owns the code it describes; the rest are this task's. Do not duplicate a fix another branch made — verify it, then tick.

| # | Contradicted claim | Owner |
|---|---|---|
| 1 | landing footer `n new note` (live `ctrl+n`) | **this task** — `library.md:259`, delete the bare `n` and keep `ctrl+n new note` (task-32138 already made `ctrl+n` the advertised story; the guide kept both) |
| 2 | Search/RAG `enter select evidence` | Task 1 (32346), `library.md:261` |
| 3 | a `mode: ✓ Search ⇄ RAG Answer` toggle | **this task** — `library.md:296` and `:720`. No such control exists on the Search/RAG canvas (A caps 14/15). Delete both example strings and replace the glyph-table example with a control that ships (`type: ✓ All types`). |
| 4 | Chunking Lab "above" its gloss (renders below) | **this task** — `library.md:314`, change "above" to "below" |
| 5 | Notes editor `ctrl+s save note` chip | **this task** — `library.md` Notes editor row; the chip is `esc back to notes` only (A cap 56). Delete the `ctrl+s` half. |
| 6 | rail beside Notes at ≥120 columns | Task 6 (32355) |
| 7 | the `Nav` handle label | Task 6 (32355) |
| 8 | "Back to Get started … never offered after graduation" | Task 3 (32349), `library.md:312` |
| 9 | bulk-delete banner copy | **this task** — `library.md:517`; live copy is **"Delete 2 selected items? You can undo right away, or restore later from Trash."** (B `37`). Quote the live string. |
| 10 | footer `cancel delete` | **this task** — `library.md:519`; live chip is `esc cancel delete`. |
| 11 | a dirty-edit veto on Escape | **this task** — `library.md:526`; the Notes editor autosaves and Escape leaves immediately, DB-proven (B D7). Replace with **"A note edit is already saved when you leave it — Escape returns to the list and the chip says so."** Leave the Prompts half only if you verify it live; if you cannot, delete the claim rather than keep an unverified one. |
| 12 | the Find bar behaviour | Task 1 (32348) |
| 13 | glosses "shown consistently" (4 of 7 rows) | **this task** — `library.md:230`. Either add glosses for Conversations, Notes and Collections, or soften the claim. **Add them**: `Conversations — past chats`, `Notes — your writing`, `Collections — saved pages`. The gloss source is the rail row model (`grep -n "— your files\|short_title" tldw_chatbook/Library/library_shell_state.py`) — if adding a gloss needs a source change, that file is unowned this wave and you may edit it; say so in the notes. |
| 14 | empty Browse rows without pager mechanics | Task 4 (32354), `library.md:223` |

- [ ] **Step 1: verify before you edit.** Launch the seeded profile (socket `crit10-docs`) at 235x52 and capture, one per row you own: the landing footer; the Search/RAG canvas; the Details ▸ Actions Chunking Lab pair; the Notes editor footer; an armed bulk delete (its banner and its footer chip); Escape out of a dirty note; the rail's Browse rows. Captures to `<SCRATCH>/crit10/wave/docs/caps/`. A claim you cannot reach live is deleted from the guide, not kept.

- [ ] **Step 2: edit the pages.** Every claim you keep is quoted from a capture. Every claim another branch fixed gets its sentence rewritten to the shipped behaviour with the fixing task named in the stamp (32366 AC#1: "with the task that fixed the surface named in the stamp").

- [ ] **Step 3: stamp.** Append one line to each page's stamp block:
  `*Verified against fix/library-crit10-docs — 2026-09-11 (task-32366: 14 critique-10 claims reconciled; surface fixes in task-32346, 32348, 32349, 32354, 32355).*`
  On a conflict with a sibling branch's stamp, keep BOTH.

- [ ] **Step 4: backlog + commit.** Tick AC#1, add Implementation Notes listing all 14 rows and which branch closed each. Commit: `docs(library): reconcile the guide with dev after the critique-10 wave (task-32366)`.

---

## Dispatch order

| Order | Task | Group | Backlog ids |
|---|---|---|---|
| 1 (parallel) | Task 1 | `viewer` | 32346, 32348, 32365 |
| 1 (parallel) | Task 2 | `media-rows` | 32347, 32350, 32364 |
| 1 (parallel) | Task 3 | `onboarding-import` | 32349, 32351 |
| 1 (parallel) | Task 4 | `pagers` | 32352, 32354, 32363, **32057** |
| 1 (parallel) | Task 5 | `export` | 32353, 32362 |
| 1 (parallel) | Task 6 | `layout` | 32355, 32359, 32360, 32361, **32107** |
| 1 (parallel) | Task 7 | `notes-details` | 32356, 32357, 32358 |
| 2 (last) | Task 8 | `docs` | 32366 — **after Tasks 1–7 report** |

Landing order for the controller: Task 1 before Task 5 (Task 5's Find-reason test reads the string Task 1 defines; it passes either way, but the Info-tab assertion is only meaningful after Task 1). Task 2 before Task 6 (both regenerate the CSS bundle — the second one merges `dev` and re-runs `build_css`). Task 8 last, and re-read every page at that point. Tasks 3, 4 and 7 are independent of everything.

## Self-review

**Spec coverage.** 23 ids, each in exactly one branch: 32346 (T1), 32347 (T2), 32348 (T1), 32349 (T3), 32350 (T2), 32351 (T3), 32352 (T4), 32353 (T5), 32354 (T4), 32355 (T6), 32356 (T7), 32357 (T7), 32358 (T7), 32359 (T6), 32360 (T6), 32361 (T6), 32362 (T5), 32363 (T4), 32364 (T2), 32365 (T1), 32366 (T8), 32057 (T4), 32107 (T6).

**File ownership.** No `.py` file is edited by two branches except `library_screen.py` and `library_media_viewer.py`, both split into disjoint, named ranges (see the ownership table and the `library_media_viewer.py` note in the wave constraints). `_agentic_terminal.tcss` is split between Tasks 2 and 6 only. `Docs/User_Guide/` pages are shared by stamp-append, per the wave rule.

**Pins named and dispositioned.** `test_media_secondary_fallback_when_no_type_no_age` (32347 — read it: it pins the no-TYPE fallback, not the age format; the real pins are `test_library_media_state.py:305/306/529/534`, updated); `test_skills_canvas_renders_exact_pager_and_source_wide_trust_count` (32354 — split in two, reversal named and justified); the entry-focus/footer pins in `Tests/UI/test_library_crit8_keyboard.py` and `test_library_crit9_shell.py` (32346 — run before and after, failing-name sets compared); task-31223's printable-key suppression and task-31272's dead-key rule (32346 — **kept**, and the plan's fix is explicitly designed not to reverse them); task-32133 AC#2 (32358 — refined, intent preserved); task-31980's danger-action margin and task-31981's inline-reason shape (32362 — reused, not changed); TASK-716's pressable-with-reason rail grammar (32107 — kept, with the reason stated in the decision text).

---

## Landing record

Eight branches, cut from `origin/dev` and landed in the order below. Ids are the
backlog tasks each PR closed.

| Task | Group | PR | Backlog ids |
|---|---|---|---|
| Task 3 | `onboarding-import` | #2598 | 32349, 32351 |
| Task 4 | `pagers` | #2599 | 32352, 32354, 32363, 32057 |
| Task 1 | `viewer` | #2602 | 32346, 32348, 32365 |
| Task 6 | `layout` | #2603 | 32355, 32359, 32360, 32361, 32107 |
| Task 5 | `export` | #2601 | 32353, 32362 |
| Task 2 | `media-rows` | #2604 | 32347, 32350, 32364 |
| Task 7 | `notes-details` | #2605 | 32356, 32357, 32358 |
| Task 8 | `docs` | this branch | 32366 |

Close-out bookkeeping on the `docs` branch: **task-32360** set Done (AC#2 landed
with #2605); **task-32217** AC#2 ticked and the task set Done, both of its halves
being on dev (`#library-note-body` fills its pane, and the Analysis tab overrides
task-31237's unconditional `1fr` so its action row is not pinned to the pane
floor).

### Riders filed

Carried out of the seven implementation branches and filed at the close, ids
swept across every ref and worktree first (the backlog CLI offered 32367, which
is held on an unmerged branch):

| Id | Rider | From |
|---|---|---|
| 32380 | the Canvas Mermaid asset check downloads unicode.org inputs at check time | wave-wide CI flake |
| 32381 | the Export media-quality knob is inert end to end | Task 5 |
| 32382 | `format_export_bytes` is KB-only | Task 5 |
| 32383 | a Trash filtered to zero says it is empty | Task 4 |
| 32384 | Find on the Read tab marks nothing on a rendered Markdown item | Task 1 |
| 32385 | Settings still ships the retired "Folder Files" name | Tasks 3 / 7 |
| 32386 | `test_closeout_single_app_route_cycle` is red on dev | Tasks 6 / 7 |
| 32387 | two declined review follow-ups from #2598 | Task 3 |
| 32388 | a workspace hop leaves an Undo receipt naming the previous workspace | Task 6 |
| 32389 | below 64 columns the notes reader keeps an empty work pane | Task 7 |
| 32390 | the `#library-notes-template-section` rule is dead after task-32356 | Task 7 |
| 32391 | Ctrl+N paints a transitional Create frame | Task 7 |
| 32392 | stored analyses render through an unsanitized Markdown sink | Task 1 |
| 32393 | Escape on a dirty prompt editor does nothing at all | Task 8 |

Not filed, and deliberately: the Prompts-lane vocabulary and stored "New Chat"
title riders were already filed by Task 2 as **32378** and **32379**; the three
items in the riders list marked "open decisions for the USER" (32303 glyph legend,
32306 Handoff row wrap, 32302 Conversations entry focus after an archive-scope hop)
are decisions, not defects, and stay with the user.

### Snapshot

Critique #10:
`.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md`
(dual-agent live review at dev `1f3184655b`). Task 8's own reconciliation captures
are under the session scratchpad at `crit10/wave/docs/caps/task8/`.
