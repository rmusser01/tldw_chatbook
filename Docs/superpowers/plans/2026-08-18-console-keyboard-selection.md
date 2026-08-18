# Console Keyboard Selection (Phase 5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keyboard-driven single-row text selection in the Console transcript: `s` on the j/k-selected message enters a vim-style mode whose motions drive the SAME SelectionManager as the mouse, so Enter opens the identical action menu (quote / side chat / feedback) with zero new downstream plumbing.

**Architecture:** Pure motion math goes in `console_selection.py` (no Textual imports). `ConsoleTranscript` gains a small mode state + `on_key` interception (the widget's established preempt-bindings pattern). Enter replays the mouse-release path by posting the existing `TranscriptTextSelected` message with row-region coordinates. Riders (notes modal) are a SEPARATE plan after this merges.

**Tech Stack:** Python 3.11+, Textual 8.x, pytest (`.venv/bin/python -m pytest`, `-p no:randomly` for determinism).

## Global Constraints

- Spec: `Docs/superpowers/specs/2026-08-18-console-keyboard-selection-and-note-management-design.md` (Part 1 only).
- ADR-031: `s` is the only NEW static binding this plan adds; mode keys are advertised by the in-mode hint, never the footer.
- Selections are single-row; 1-unit floor; Esc layering = mode first, message selection second.
- All work on branch `feat/console-keyboard-selection` (exists, tracks origin/dev, spec committed).
- Run tests as `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <target> -q -p no:randomly`.
- The repo test venv is uv-managed; NEVER `pip install`.
- File the backlog task FIRST (Task 1 Step 0) with a freshly swept ID: sweep every remote ref + worktree per `backlog/docs/lessons-backlog-hygiene.md`; peer sessions had filed through ≥17386 on 2026-08-18 — re-derive, never reuse a number from this plan.

## Plan-discovered spec refinements (fold into the spec amendment, Task 5)

1. Markdown rows store CHARACTER ranges as-is (`set_selection_range` comment: "the range is stored as-is" — live-spike amended past the design's line-snap wording). So char motions (h/l/w/b/0/$) work on plain AND markdown rows; only diff rows are line-snapped (their `set_selection_range` snaps) and take j/k only.
2. `o` swaps anchor and active end (vim visual's own key) — without it, a text-start anchor can never reach a mid-text span.
3. Keyboard finish must drain `consume_release_click()` + `consume_just_finished()`: `finish_drag()` arms a one-shot release-click suppression token that no keyboard click will consume, and a stale token would eat the next genuine row click.

---

### Task 1: Pure motion helpers

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_selection.py` (append after `offset_for_cell`)
- Test: `Tests/UI/test_console_selection_core.py` (append)

**Interfaces:**
- Produces (Task 3 consumes):
  `word_forward_offset(text: str, offset: int) -> int`
  `word_back_offset(text: str, offset: int) -> int`
  `line_start_offset(text: str, offset: int) -> int`
  `line_end_offset(text: str, offset: int) -> int`
  `next_line_offset(text: str, offset: int) -> int`
  `prev_line_offset(text: str, offset: int) -> int`
  All clamp to `[0, len(text)]`; all are total (never raise) on empty text.

- [ ] **Step 0: File the backlog task** — sweep the true max ID (the lessons-file one-liner over all remote refs + every worktree's `backlog/tasks`), create `backlog/tasks/task-<ID> - Console-keyboard-text-selection-phase-5.md` with status In Progress, description referencing the spec, ACs: (1) `s` enters mode on eligible selected rows only; (2) motions per row kind incl. `o`; (3) Enter opens the identical menu incl. feedback gating; (4) Esc layering; (5) hint truthful per row kind; (6) release-click token drained on keyboard finish; (7) tests green; (8) docs updated. Commit `task(<ID>): file phase-5 keyboard selection task`.

- [ ] **Step 1: Write the failing tests** (append to `Tests/UI/test_console_selection_core.py`):

```python
# --- keyboard motion helpers (phase 5) --------------------------------------

from tldw_chatbook.Widgets.Console.console_selection import (
    line_end_offset,
    line_start_offset,
    next_line_offset,
    prev_line_offset,
    word_back_offset,
    word_forward_offset,
)

TEXT = "alpha beta\ngamma  delta\n\nepsilon"


def test_word_forward_jumps_to_next_word_start():
    assert word_forward_offset(TEXT, 0) == 6        # alpha| -> |beta
    assert word_forward_offset(TEXT, 6) == 11       # beta| -> |gamma (over \n)
    assert word_forward_offset(TEXT, 26) == len(TEXT)  # last word -> end


def test_word_back_jumps_to_previous_word_start():
    assert word_back_offset(TEXT, 6) == 0
    assert word_back_offset(TEXT, 13) == 11         # inside gamma -> its start
    assert word_back_offset(TEXT, 0) == 0           # floor


def test_line_bounds_are_current_line_vim_style():
    assert line_start_offset(TEXT, 8) == 0          # inside line 1
    assert line_end_offset(TEXT, 8) == 10           # before the \n
    assert line_start_offset(TEXT, 13) == 11        # line 2
    assert line_end_offset(TEXT, 13) == 24


def test_line_motions_move_one_line_and_clamp():
    assert next_line_offset(TEXT, 5) == 16          # column-ish landing on line 2
    assert prev_line_offset(TEXT, 16) == 5
    assert next_line_offset(TEXT, 26) == len(TEXT)  # last line -> end clamp
    assert prev_line_offset(TEXT, 3) == 0           # first line -> start clamp


def test_helpers_are_total_on_empty_text():
    for fn in (word_forward_offset, word_back_offset, line_start_offset,
               line_end_offset, next_line_offset, prev_line_offset):
        assert fn("", 0) == 0
```

- [ ] **Step 2: Run to verify they fail** — `... Tests/UI/test_console_selection_core.py -q -p no:randomly -k "word_forward or word_back or line_bounds or line_motions or total_on_empty"` → FAIL (ImportError).

- [ ] **Step 3: Implement** (append to `console_selection.py`; pure stdlib):

```python
def _clamp(text: str, offset: int) -> int:
    return max(0, min(offset, len(text)))


def word_forward_offset(text: str, offset: int) -> int:
    """Vim-w: the start of the next word (whitespace-delimited), else end."""
    i = _clamp(text, offset)
    n = len(text)
    while i < n and not text[i].isspace():
        i += 1
    while i < n and text[i].isspace():
        i += 1
    return i


def word_back_offset(text: str, offset: int) -> int:
    """Vim-b: the start of the previous word, else 0."""
    i = _clamp(text, offset)
    while i > 0 and text[i - 1].isspace():
        i -= 1
    while i > 0 and not text[i - 1].isspace():
        i -= 1
    return i


def line_start_offset(text: str, offset: int) -> int:
    """Vim-0: start of the line containing ``offset``."""
    i = _clamp(text, offset)
    return text.rfind("\n", 0, i) + 1


def line_end_offset(text: str, offset: int) -> int:
    """Vim-$: end of the line containing ``offset`` (before its newline)."""
    i = _clamp(text, offset)
    nl = text.find("\n", i)
    return len(text) if nl == -1 else nl


def next_line_offset(text: str, offset: int) -> int:
    """One line down, preserving the column where the next line allows."""
    i = _clamp(text, offset)
    column = i - line_start_offset(text, i)
    end = line_end_offset(text, i)
    if end >= len(text):
        return len(text)
    nstart = end + 1
    return min(nstart + column, line_end_offset(text, nstart))


def prev_line_offset(text: str, offset: int) -> int:
    """One line up, preserving the column where the previous line allows."""
    i = _clamp(text, offset)
    start = line_start_offset(text, i)
    if start == 0:
        return 0
    column = i - start
    pstart = line_start_offset(text, start - 1)
    return min(pstart + column, line_end_offset(text, pstart))
```

- [ ] **Step 4: Run to verify PASS**, adjusting only test EXPECTATIONS if a hand-computed offset was wrong (verify by hand against TEXT first; the implementation semantics above are the contract).
- [ ] **Step 5: Ruff both files; commit** `feat(console): pure keyboard-motion helpers for selection phase 5`.

---

### Task 2: Mode state machine (enter/exit/layering/hint)

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` + run `./build_css.sh`
- Test: `Tests/UI/test_console_keyboard_selection.py` (new)

**Interfaces:**
- Consumes: existing `selected_message_id`, `selection_manager` (`begin_drag(row_key, offset)`), rows' `get_display_text()/set_selection_range()/clear_selection()`, `_selection_row_for`-style eligibility via `PROTECTED_CLICK_CLASSES` peers.
- Produces (Tasks 3–4 consume): `ConsoleTranscript._kb_selection_row: <row|None>` (None = mode off); `_enter_keyboard_selection() -> bool`; `_exit_keyboard_selection(*, clear: bool = True) -> None`; `_kb_hint` Static with id `console-kb-selection-hint`; `s` in `BINDINGS` → `action_enter_text_selection`.

- [ ] **Step 1: Failing tests** (new file; reuse the harness style of `Tests/UI/test_console_selection_transcript.py` — copy its app/mount scaffolding for a transcript with one plain ASSISTANT row and one protected banner row):

```python
async def test_s_enters_mode_on_selected_eligible_row(pilot, transcript, row):
    transcript.selected_message_id = row.message_id
    await pilot.press("s")
    assert transcript._kb_selection_row is row
    sel = transcript.selection_manager.state.selection
    assert sel is not None and (sel.start, sel.end) == (0, 1)
    assert row.get_selection_text() == row.get_display_text()[0:1]
    hint = transcript.query_one("#console-kb-selection-hint")
    assert hint.display is True

async def test_s_without_selection_or_on_protected_row_is_a_toast_not_a_mode(...):
    # no selected_message_id -> no mode, hint hidden
    # selected id whose row is a banner -> no mode

async def test_escape_layering_mode_first_message_selection_second(...):
    # in mode: Esc -> mode off, text selection cleared, selected_message_id UNCHANGED
    # second Esc -> selected_message_id None (existing binding)

async def test_mouse_down_exits_mode_before_arming_a_drag(...):
    # enter mode, synthesize MouseDown on the row -> _kb_selection_row is None
```

- [ ] **Step 2: Run → FAIL** (no `_kb_selection_row`).
- [ ] **Step 3: Implement.** In `ConsoleTranscript`:
  - `BINDINGS` gains `("s", "enter_text_selection", "Select text")`.
  - State in `__init__`: `self._kb_selection_row = None`.
  - `action_enter_text_selection`: resolve `#console-message-{selected_message_id}` (also try the diff id prefix `console-tool-diff-`? NO — diff rows are separate widgets not addressable from message selection; the SELECTED MESSAGE's own row is the target; diff-row keyboard entry is out of scope, note in docs); require row kind in the three selection classes and not `None`; scroll it visible; `self.selection_manager.begin_drag(row.id, 0)`; `self.selection_manager.extend_drag(row.id, 1 if row.get_display_text() else 0)`; empty-text rows: toast + return; `row.set_selection_range(0, 1)`; set `_selection_origin_row = row` and `_kb_selection_row = row`; show hint (content per row kind, Task 3 sets final copy).
  - `_exit_keyboard_selection(clear=True)`: if clear → `row.clear_selection()`, `self.selection_manager.cancel()`; always `_kb_selection_row = None`, `_selection_origin_row = None`, hide hint.
  - `on_key`: when `_kb_selection_row` is not None and key == "escape": `_exit_keyboard_selection()`, `event.stop(); event.prevent_default()` (keeps message selection — the clear-selection BINDING never fires).
  - `on_mouse_down` first line: `if self._kb_selection_row is not None: self._exit_keyboard_selection()` (mouse takes over cleanly).
  - Row destruction (streaming replacement, prune, session switch): at the TOP of the mode `on_key` branch, `if self._kb_selection_row is not None and not self._kb_selection_row.is_attached: self._exit_keyboard_selection(clear=False); return` — plus a test: enter mode, `transcript.set_messages([])`, press `l` → no crash, mode off, hint hidden.
  - Entry also sets `self._kb_anchor, self._kb_end = 0, 1` (Task 3's motion state; kept next to `_kb_selection_row` so exit clears all three).
  - Hint: compose a `Static("", id="console-kb-selection-hint")` with `display=False` docked at the transcript's bottom via CSS (`dock: bottom; height: 1;` on class `console-kb-selection-hint`, muted palette like the annotations marker); toggle display + update content on enter/exit.
- [ ] **Step 4: Run new tests → PASS; run `Tests/UI/test_console_selection_transcript.py` + `test_console_selection_rows.py` → no regressions.**
- [ ] **Step 5: build_css.sh; ruff; commit** `feat(console): keyboard text-selection mode enter/exit (phase 5)`.

---

### Task 3: Motions

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py` (the `on_key` mode branch)
- Test: `Tests/UI/test_console_keyboard_selection.py` (append)

**Interfaces:**
- Consumes: Task 1 helpers; Task 2 mode state; `extend_drag`.
- Produces: motion handling for `h l w b 0 $ j k o` inside the mode; per-kind key sets `_KB_CHAR_KEYS`/`_KB_LINE_KEYS`; hint copy finalized:
  plain/markdown: `"h/l chars · w/b words · 0/$ line · j/k lines · o swap · Enter menu · Esc cancel"`
  diff: `"j/k lines · o swap · Enter menu · Esc cancel"`

- [ ] **Step 1: Failing tests** — parametrized per row kind:

```python
async def test_l_and_h_extend_and_shrink_by_char_on_plain_rows(...):
    # after s: press l 3x -> selection (0, 4); h once -> (0, 3); h past floor -> stays (0, 1)

async def test_w_b_0_dollar_move_the_active_end(...):
async def test_j_k_move_by_line_preserving_column(...):
async def test_o_swaps_anchor_and_end_so_mid_text_spans_are_reachable(...):
    # s, l l l  -> (0,4); o; w -> anchor stays 4-ish end moves: assert start>0 possible
async def test_char_keys_are_inert_on_diff_rows(...):
    # diff row mode: press l -> selection unchanged; press j -> grows by one line (snapped)
async def test_markdown_rows_take_char_motions(...):
    # live-spike fact: markdown stores char ranges as-is
async def test_mode_keys_do_not_leak_to_bindings(...):
    # in mode press "c" (Copy binding) -> no copy action fired, selection unchanged (inert unknown key)
```

- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Implement** in the `on_key` mode branch (before Textual binding dispatch, `event.stop()`+`prevent_default()` — ruling (corrected after Task 2's review found the fall-through desync): while in mode, stop ALL printable single characters, enter, escape, AND up/down (up/down alias the j/k selection-nav BINDINGS — they do not scroll; letting them through desyncs mode from message selection). Unknown consumed keys are inert. Page-up/page-down and mouse wheel fall through for scrolling. Task 2's no-op consumption of {j,k,down,up,enter} is replaced here by real motions/finish.):

```python
_KB_CHAR_MOTIONS = {"h", "l", "w", "b", "0", "$"}
_KB_LINE_MOTIONS = {"j", "k"}

def _kb_apply_motion(self, key: str) -> None:
    row = self._kb_selection_row
    sel = self.selection_manager.state.selection
    if row is None or sel is None:
        return
    text = row.get_display_text()
    end = self.selection_manager._current_offset  # active end (see note)
    ...
```
  NOTE: do NOT reach into `_current_offset`; use `self._kb_anchor`/`self._kb_end` (initialized 0/1 by Task 2's entry), compute the new end via the helpers (`l`: `min(end+1, len(text))`; `h`: `max(end-1, anchor+1 if end>anchor else ...)` — with `o` swapping `_kb_anchor`/`_kb_end`), floor at 1 unit (`abs(end-anchor) >= 1`), then `begin_drag(row.id, self._kb_anchor)` + `extend_drag(row.id, self._kb_end)` + `row.set_selection_range(*sorted((self._kb_anchor, self._kb_end)))`. Re-begin+extend per motion keeps the manager the single source the menu path reads while the anchor/end pair stays keyboard-owned (mouse never sets `_kb_*`). Diff rows: only `j`/`k`/`o` mutate; char motions return untouched. `j`/`k` on any kind use `next_line_offset`/`prev_line_offset` on the ACTIVE END.
- [ ] **Step 4: PASS + neighbor suites.**
- [ ] **Step 5: Ruff; commit** `feat(console): keyboard selection motions h/l/w/b/0/$/j/k/o (phase 5)`.

---

### Task 4: Enter → the real menu; token drain; e2e

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py`
- Test: `Tests/UI/test_console_keyboard_selection.py` (append) + one journey in `Tests/UI/test_console_selection_end_to_end.py`

**Interfaces:**
- Consumes: `TranscriptTextSelected(selection, screen_x, screen_y)`; `finish_drag()`; `consume_release_click()`; `consume_just_finished()`.
- Produces: Enter-in-mode = mouse-release parity.

- [ ] **Step 1: Failing tests:**

```python
async def test_enter_opens_the_same_menu_with_feedback_gating(...):
    # assistant row + run idle: s, l l, Enter -> ConsoleSelectionMenu mounted on screen,
    # feedback buttons present (assistant prose qualifies), Request/LGTM disabled (no run), Comment enabled.

async def test_keyboard_finish_drains_the_release_click_token(...):
    # s, l, Enter; then assert transcript.selection_manager.consume_release_click() is False
    # (a stale True would eat the next genuine row click)

async def test_menu_anchor_derives_from_row_region_and_stays_in_transcript(...):
    # menu.styles/absolute offset within transcript.region bounds; selection_top == row.region.y

async def test_menu_actions_after_keyboard_selection_dispatch_feedback(...):
    # e2e file: keyboard journey j/k -> s -> l l -> Enter -> click Comment (stubbed modal "kb note")
    # -> queue dispatched ["[Comment]\n> <quote>\nkb note"], sidecar+annotation recorded via _RecordingStore
```

- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Implement** Enter branch in the mode `on_key`:

```python
if event.key == "enter":
    event.stop(); event.prevent_default()
    row = self._kb_selection_row
    selection = self.selection_manager.finish_drag()
    # Keyboard has no release Click to consume the suppression tokens the
    # finish just armed -- drain them or the NEXT genuine row click is eaten.
    self.selection_manager.consume_release_click()
    self.selection_manager.consume_just_finished()
    self._exit_keyboard_selection(clear=False)   # keep highlight + manager state for the menu
    if selection is None or row is None:
        return
    region = row.region
    self.post_message(
        self.TranscriptTextSelected(
            selection=selection,
            screen_x=region.x + min(4, max(0, region.width - 1)),
            screen_y=region.bottom - 1,   # handler's +1 puts the menu just below the row
        )
    )
```
  `_exit_keyboard_selection(clear=False)` hides the hint and nulls `_kb_selection_row` but leaves the row highlight and manager `_finished` state — exactly what the mouse path leaves. NOTE `on_mouse_up` nulls `_selection_origin_row` before `_text_selected` re-resolves it via `_active_selection_row()`; mirror that (null it in the exit).
- [ ] **Step 4: PASS; run the FULL selection + annotation suites** (`test_console_selection_end_to_end.py`, `test_console_selection_menu.py`, `test_console_annotation_markers.py`, `test_console_selection_transcript.py`, `test_console_selection_rows.py`, `test_console_transcript_selection_contract.py`).
- [ ] **Step 5: Ruff; commit** `feat(console): Enter opens the selection menu from keyboard mode (phase 5)`.

---

### Task 5: Docs, spec amendments, live verification, wrap

**Files:**
- Modify: `Docs/User_Guide/console/text-selection-and-feedback.md` (keyboard section replaces the "mouse-only" quirk), `Docs/superpowers/specs/2026-08-18-...-design.md` (the three plan-discovered refinements), `Docs/superpowers/specs/2026-08-14-console-selection-annotations-design.md` (§42 amendment note), `backlog/decisions/068-console-text-selection-and-annotations.md` (amendment 5: keyboard mode; cite the token-drain and swap-ends facts), task file (ACs ticked, notes).

- [ ] **Step 1:** Write all doc updates (user guide gains: `s` entry, per-kind key table, two-stage Esc, "diff rows are reachable by mouse only" note).
- [ ] **Step 2:** Full sweep: the six selection/annotation suites + `Tests/Chat/test_trajectory_capture.py` + ruff over every touched file. Compare any failure against clean dev before attributing.
- [ ] **Step 3: LIVE tmux verification** (scratch profile recipe in memory `console-selection-feedback-program`; llama.cpp :9191): journey = boot → skip wizard → consent modal "Don't check" → Console → send prompt → j/k select reply → `s` → `l l l w` → Enter → menu visible → Comment → note → marker appears. Also verify: Esc twice layering; `s` on empty transcript toasts. Kill server, delete scratch profile.
- [ ] **Step 4:** Task → Done with Implementation Notes; commit docs; push branch; PR against dev titled `feat(console): keyboard text selection (phase 5)`; body cites the spec + the three refinements; then the maintainer's standing flow (Qodo loop → merge) ONLY on explicit instruction.

