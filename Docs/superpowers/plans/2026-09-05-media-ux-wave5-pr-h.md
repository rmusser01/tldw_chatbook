# Media UX fix wave 5 — PR H (the wide layout, task-31633) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** At 235×52 the Library ▸ Media list is at least as wide as at 100×30 and a 98-character title fits or truncates later; no 5-cell dead gutters flank it; `More` does not displace the Reader body; each list item costs two rows, not three.

**Architecture:** The shared layout resolver `resolve_adaptive_reader_layout` in `tldw_chatbook/Utils/adaptive_reader_state.py` pins the Items column at the profile's `list_comfort_width` (`ITEMS_TARGET_WIDTH = 40`, painted as 38) and hands every remaining cell to the Reader, even past `READER_COMFORT_WIDTH`. Task 1 adds an opt-in growth rule to the resolver (`list_grows: bool` on the profile, default False so Conversations/Skills/Collections keep today's geometry): once the Reader has its comfort width, the Items column takes a share of the surplus up to `list_max_width` (72). Task 2 removes the gutters and the per-item spacer row in the Media canvas. Task 3 renders `More` as one compact row inside the toolbar instead of a `Vertical` pushed above the body.

**Tech Stack:** Python 3.12, Textual 8.x, pytest; `Tests/Utils/test_adaptive_reader_state.py` (grep the real name) for the resolver, `Tests/UI/test_library_adaptive_reader_shell.py` (pane geometry, `_painted`), `Tests/UI/test_library_media_render_fixes.py`, `Tests/UI/test_library_media_reader_shell.py`, `Tests/UI/test_library_media_toolbar_adapt.py`.

**Spec:** `backlog/tasks/task-31633 - Library-media-wide-layout-let-the-list-grow-with-width-and-stop-More-displacing-the-body.md` (AC#1-#4); critique #5 P1 "At 235 columns the list is narrower than at 100" (captures 1 and 10); PRODUCT.md "terminal-native density".

## Global Constraints

- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/media-wave5-h`, branch `fix/media-wave5-h` off dev. Every command: `cd <worktree> && PYTHONPATH=<worktree> /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest … -p no:cacheprovider`; absolute paths; UI test files in separate processes; every Bash call begins with the explicit `cd` and `git branch --show-current`.
- Compare failures against the base before claiming them (known: `test_library_ingest_canvas.py::test_progress_detail_paints_below_row…`, `test_library_ingest_retry_last` flake, the `test_library_shell.py` census in task-31249).
- `adaptive_reader_state.py` is shared by Conversations, Skills and Collections (`library_conversations_state.py`, `library_skills_state.py`, `library_collections_state.py`, their controllers, and `config.py` preferences): every change is opt-in per profile, and those three surfaces' existing tests must pass byte-for-byte. The 100×30 composition is the good one — nothing in this PR may change any width at 100×30 (pin it before touching the resolver).
- The Items-pane floor stays 36 cells (PR D's painted pins for the select-mode rows and the choice row assume it); the select-mode rows and receipts keep their positions.
- No new `logger.*`. After any `BUNDLED_CSS` / TCSS edit: `python -m tldw_chatbook.css.build_css` then `python tldw_chatbook/css/check_bundle_sync.py` (exit 0); commit regenerated files. Five-key contract frozen; review-set code and the Find focus token untouched; no new toolbar buttons.
- Live verification: tmux (function `t() { tmux -L w5h "$@"; }` in every call, sleeps inside, `t kill-server` at the end), real config, ONE app instance; seeds via `MediaDatabase.add_media_with_keywords` with salted content including one 98-character title, cleaned with `soft_delete_media`.
- TDD per task; commit per task with the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; backlog task files are flipped by the controller.

---

### Task 1: The Items column grows with the terminal once the Reader is comfortable (task-31633 AC#1)

**Files:**
- Modify: `tldw_chatbook/Utils/adaptive_reader_state.py` — the profile dataclass gains `list_grows: bool = False`; in `resolve_adaptive_reader_layout`, after `items_width` is computed as today, when `profile.list_grows` and the Reader's remainder exceeds `READER_COMFORT_WIDTH`, move `surplus // 2` cells from the Reader to the Items column, capped at `profile.list_max_width` and never below today's value; `required_width` unchanged.
- Modify: the Media surface's profile construction (grep where the media reader profile is built — `library_media_reader_state.py` or `screen_constants.py`) — set `list_grows=True` for Media only.
- Test: the resolver's test file — at width 235 with the Media profile the Items width is ≥ 47 and the Reader keeps ≥ `READER_COMFORT_WIDTH`; at width 100 the layout is byte-identical to today (pin the tuple before the change); Conversations/Skills/Collections profiles at 235 are byte-identical to today; `Tests/UI/test_library_adaptive_reader_shell.py` painted at 235×52: the list pane region width ≥ 47 and a 98-char seeded title paints ≥ 44 characters before `…`.

**Interfaces:**
- Produces: `profile.list_grows`; the growth rule.

- [ ] Step 1: pin today's layouts (100 and 235, all four profiles) as tests; then the failing growth tests.
- [ ] Step 2: run; confirm (Items 40 at 235 for Media).
- [ ] Step 3: implement.
- [ ] Step 4: run the resolver tests, `test_library_adaptive_reader_shell.py`, `test_library_media_reader_shell.py`, `test_library_media_render_fixes.py`, `test_library_media_toolbar_adapt.py`, the Conversations/Skills/Collections UI test files (grep for their names), `test_library_shell.py -k "width or layout or grip"` (compare to base).
- [ ] Step 5: live 235×52 with a 98-char title: the title no longer truncates at 30 characters; then 100×30 identical to before (capture both).
- [ ] Step 6: commit `fix(library): the Media Items column grows with the terminal once the Reader is comfortable (task-31633)`.

---

### Task 2: No dead gutters, two rows per item (task-31633 AC#2, part of AC#1)

**Files:**
- Modify: `tldw_chatbook/css/screen_agentic_library.tcss` / the media canvas `BUNDLED_CSS` — find the rules that produce the 5-cell empty columns between rail and list and between list and Reader at 235×52 (margins/padding on `#library-media-items`, the pane grips' width, or the shell's `Horizontal` spacing) and reduce them to a 1-cell grip plus at most 1 cell of padding; rebuild the bundle.
- Modify: `tldw_chatbook/Widgets/Library/library_media_canvas.py` — the per-item composition: title row + meta row, no blank spacer widget; row separation by a 1-cell top margin on the meta row is NOT allowed (it would still cost a row); rely on the title/meta rhythm and the `▸` marker for separation, exactly as the 100×30 layout already reads.
- Test: `Tests/UI/test_library_adaptive_reader_shell.py` (painted: the columns between the rail's right border and the list's first glyph, and between the list and the Reader frame, are ≤ 2 cells at 235×52); `Tests/UI/test_library_media_render_fixes.py` (15 seeded items → at least 15 title rows visible in a 52-row terminal; no blank row between an item's meta and the next item's title).

**Interfaces:**
- Consumes: Task 1's widths.
- Produces: 2-rows-per-item grammar.

- [ ] Step 1: failing tests (gutter width ×2; rows-per-item).
- [ ] Step 2: run; confirm (5-cell gutters; 3 rows per item).
- [ ] Step 3: implement; rebuild CSS; `check_bundle_sync` exit 0.
- [ ] Step 4: run `test_library_adaptive_reader_shell.py`, `test_library_media_render_fixes.py`, `test_library_multiselect_media.py` (select-mode rows and receipts unchanged), `test_library_media_trash.py`, `test_library_shell.py -k "row or media"` (compare to base).
- [ ] Step 5: live 235×52: count visible items; capture the gutters.
- [ ] Step 6: commit `fix(library): close the Media gutters and drop the per-item spacer row (task-31633)`.

---

### Task 3: `More` is one row, not a push (task-31633 AC#3)

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_media_viewer.py` — the `more_open` branch replaces `Vertical(id="library-media-reader-more-actions")` with a `Horizontal(id="library-media-reader-more-actions", classes="ds-toolbar")` of the same compact buttons (`Edit metadata`, `Open original`, `Open manager`, `Move to trash`) rendered directly under the primary toolbar row; the tab row and body move down by exactly one row while open; `More` reads `More ▴` while open and `More` when closed.
- Test: `Tests/UI/test_library_media_reader_shell.py` / `test_library_media_render_fixes.py` (painted at 235×52 and 100×30: with More open the `Read` tab row is exactly one row lower than with it closed; all four actions readable; at 100×30 the four actions fit or wrap to two rows and the body moves by at most two rows).

**Interfaces:**
- Produces: the one-row More grammar.

- [ ] Step 1: failing tests (displacement ×2 sizes; labels readable).
- [ ] Step 2: run; confirm (~19-row displacement).
- [ ] Step 3: implement; rebuild CSS if any rule is needed.
- [ ] Step 4: run `test_library_media_reader_shell.py`, `test_library_media_render_fixes.py`, `test_library_media_reader_flow.py`, `test_library_shell.py -k "more or menu"` (compare to base).
- [ ] Step 5: live 235×52 and 100×30: open More, capture, close.
- [ ] Step 6: commit `fix(library): the Reader's More actions render on one row instead of pushing the body (task-31633)`.
