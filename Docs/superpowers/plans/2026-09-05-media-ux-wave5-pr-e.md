# Media UX fix wave 5 — PR E (bulk-mutation honesty, task-31220) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A bulk delete in Library ▸ Media never paints `✓` for a write that did not land, never leaves the screen wedged, and always offers a working recovery: Retry that says why it failed, rows that still open under a stale gate, and an Undo that is enabled whenever the receipt says `✓`.

**Architecture:** The existing seam stays: `handle_library_media_delete_selected` → `_delete_library_media_selection` (per-item `delete_media_item` through `_run_library_service_call`, `succeeded`/`failed` lists, receipt from `succeeded`) → `_complete_library_media_mutation` in a `finally` (releases `_library_media_bulk_delete_in_flight`, reconciles, refreshes only with authority). Task 1 adds facts before any change: a real-`MediaDatabase` integration test through that seam with a fresh read connection after the receipt, a WAL-visibility probe that reproduces the assessments' read method, and a two-connection contention probe. Task 2 hardens the wedge in the controller/screen: a failed Retry paints its reason, row opens stay allowed under the mutation gate (reads are safe), and every site that sets the interlock is proven to release it. Task 3 makes the receipt/Undo pair consistent and moves focus to Undo.

**Tech Stack:** Python 3.12, Textual 8.x, pytest + pytest-asyncio; `Tests/UI/test_library_multiselect_media.py` (`_media_fake`, `_bind_media_mutation_seams`, direct `LibraryScreen._delete_library_media_selection(fake, ids)` calls), `Tests/UI/test_library_media_trash.py` (real `MediaDatabase` + `LocalMediaReadingService` + `MediaReadingScopeService(LocalMediaReadingService(db), None)` host pattern near its `local_service = LocalMediaReadingService(db)` sites), `Tests/UI/test_library_media_browse_controller.py` for the controller gate, `_painted` from `Tests/UI/test_library_media_render_fixes.py`.

**Spec:** `backlog/tasks/task-31220 - Media-storage-root-cause-the-session-progressive-silent-degradation.md` (its three new ACs from critique #5 are binding) and the critique #5 snapshot `.impeccable/critique/2026-09-05T06-05-33Z__tldw-chatbook-ui-screens-library-screen-py.md` (P0, corrected mechanism section).

## Global Constraints

- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/media-wave5-e`, branch `fix/media-wave5-e` off dev `f8cb939e2b`. Every command: `cd <worktree> && PYTHONPATH=<worktree> /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest … -p no:cacheprovider`; absolute paths; UI test files in separate processes; begin every Bash call with the explicit `cd` and `git branch --show-current`.
- Compare failures against the base before claiming them. Known pre-existing: `Tests/UI/test_library_ingest_canvas.py::test_progress_detail_paints_below_row…[size0|size1]`, `test_library_ingest_retry_last` registry-ticks flake, and the `test_library_shell.py` census in `backlog/tasks/task-31249 - …md` (whole-file ~224 failing / 825).
- No new `logger.*` calls (`python scripts/check_persistent_diagnostic_inventory.py` exit 0). After any `BUNDLED_CSS` / TCSS edit: `python -m tldw_chatbook.css.build_css` then `python tldw_chatbook/css/check_bundle_sync.py` (exit 0), commit the regenerated files. `./scripts/preflight.sh` green before the PR.
- Workers: never `run_worker(exclusive=True)` without `group=`; the bulk delete/Undo pair keeps ONE flag and ONE group (`library_media_bulk_delete`, ADR-055's one-flag rule) — do not add a second flag.
- The five-key media summary contract (`_MEDIA_SUMMARY_KEYS`) is frozen in this PR (PR I bumps it); review-set code and the Find focus token untouched; no new buttons on the media action toolbar.
- The host may still be out of POSIX semaphores (`multiprocessing.Lock()` → OSError 28): never run an import live; any test spawning a pool fails for that reason, not because of this change.
- Live verification runs the app under tmux (function `t() { tmux -L w5e "$@"; }` defined in every call, sleeps inside the call, `t kill-server` before finishing) against the real config; seeds via `MediaDatabase.add_media_with_keywords` with salted content, cleaned with `soft_delete_media`. ONE instance at a time — if the app's "Another copy of tldw is already using this profile" toast appears, stop and report.
- TDD per task; commit per task with the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; backlog task files are flipped by the controller.

---

### Task 1: Instrumented reproduction — where does a bulk delete's write go, and can a reader miss it? (task-31220 evidence)

**Files:**
- Create: `Tests/UI/test_library_media_bulk_delete_real_db.py`
- Read: `tldw_chatbook/UI/Screens/library_screen.py` (`_delete_library_media_selection`, `_complete_library_media_mutation`, `_run_library_service_call`), `tldw_chatbook/Media/local_media_reading_service.py` (`delete_media_item`: raises `KeyError` when the row is missing and `ValueError` when `mark_as_trash` returns `False`), `tldw_chatbook/DB/Client_Media_DB_v2.py` (`mark_as_trash`: `SELECT … WHERE id=? AND deleted=0`, `UPDATE … WHERE id=? AND version=?`, `ConflictError` on rowcount 0, commit on transaction exit), the real-DB host pattern in `Tests/UI/test_library_media_trash.py`.

**Interfaces:**
- Produces: a `_real_media_host(tmp_path, *, items)` helper returning `(host, screen, db, db_path)` for a `LibraryScreen` whose `media_reading_scope_service` is `MediaReadingScopeService(LocalMediaReadingService(db), None)` over a real `MediaDatabase(db_path, client_id="test")`; reused by Tasks 2 and 3.

- [ ] Step 1: write the three tests, RED first.

```python
# Tests/UI/test_library_media_bulk_delete_real_db.py
"""Real-DB reproduction for task-31220: the bulk-delete write, as seen by a fresh reader."""
import sqlite3
import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Media.local_media_reading_service import LocalMediaReadingService
from tldw_chatbook.Media.media_reading_scope_service import MediaReadingScopeService
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
# import the trash tests' host builder and adapt it; do not copy its body
from Tests.UI.test_library_media_trash import _trash_production_host  # or the real-service host helper used at its LocalMediaReadingService(db) sites


def _seed(db: MediaDatabase, n: int) -> list[int]:
    ids = []
    for i in range(n):
        result = db.add_media_with_keywords(
            url=None, title=f"BulkDelete seed {i}", media_type="article",
            content=f"bulk delete seed body {i} unique", keywords=["bulk"],
        )
        ids.append(int(result[0] if isinstance(result, tuple) else result["id"]))
    return ids


def _fresh_is_trash(db_path, media_id: int) -> int:
    with sqlite3.connect(db_path) as conn:  # a NEW connection, never the app's
        return conn.execute("SELECT is_trash FROM Media WHERE id=?", (media_id,)).fetchone()[0]


@pytest.mark.asyncio
async def test_bulk_delete_write_is_visible_to_a_fresh_reader_and_receipt_matches(tmp_path):
    host, screen, db, db_path = _real_media_host(tmp_path, items=3)
    ids = screen_media_ids(screen)  # canonical "local:media:<id>" ids of the seeded rows
    async with host.run_test(size=(235, 52)) as pilot:
        await LibraryScreen._delete_library_media_selection(screen, (ids[0],))
        await pilot.pause()
        backing = int(ids[0].rsplit(":", 1)[1])
        assert _fresh_is_trash(db_path, backing) == 1
        assert screen._library_media_delete_receipt_ids == (ids[0],)
        assert screen._library_media_bulk_delete_in_flight is False


@pytest.mark.asyncio
async def test_long_lived_reader_connection_can_miss_the_write_the_receipt_reports(tmp_path):
    """Reproduces the critique-#5 assessments' read method: a MediaDatabase opened BEFORE the
    app's commit. Records (does not assert) what it sees; the fresh connection is the oracle."""
    host, screen, db, db_path = _real_media_host(tmp_path, items=2)
    reader = MediaDatabase(db_path, client_id="assessor")
    _ = reader.get_media_by_id(1)  # open a read before the write
    async with host.run_test(size=(235, 52)) as pilot:
        await LibraryScreen._delete_library_media_selection(screen, (ids_of(screen)[0],))
        await pilot.pause()
    stale_view = reader.get_media_by_id(1)["is_trash"]
    fresh_view = _fresh_is_trash(db_path, 1)
    assert fresh_view == 1
    # Document the finding in the report either way:
    print(f"long-lived reader saw is_trash={stale_view}; fresh connection saw {fresh_view}")


@pytest.mark.asyncio
async def test_contended_write_never_paints_a_success_receipt(tmp_path):
    host, screen, db, db_path = _real_media_host(tmp_path, items=2)
    other = sqlite3.connect(db_path, isolation_level=None)
    other.execute("BEGIN IMMEDIATE")  # another instance holds the write lock
    try:
        async with host.run_test(size=(235, 52)) as pilot:
            await LibraryScreen._delete_library_media_selection(screen, (ids_of(screen)[0],))
            await pilot.pause()
            assert screen._library_media_delete_receipt_ids == ()
            assert screen._library_media_bulk_delete_in_flight is False
            assert _fresh_is_trash(db_path, 1) == 0
    finally:
        other.execute("ROLLBACK"); other.close()
```

- [ ] Step 2: run: `pytest Tests/UI/test_library_media_bulk_delete_real_db.py -p no:cacheprovider -v`. Expected: the first test either passes (the app's write is sound single-instance — record that) or fails on `_fresh_is_trash == 1` (a real defect — record the traceback); the contention test must fail RED for `receipt_ids == ()` only if the code paints ✓ under a locked DB (it should not: `delete_media_item` raises); if it passes on the first run, say so in the report — it is a pin, not a behaviour change.
- [ ] Step 3: the ONLY production change permitted in this task: if the first test fails, fix the smallest thing that makes the fresh reader see the write (e.g. a missing commit in `mark_as_trash`'s path); if it passes, change nothing.
- [ ] Step 4: write the facts into the report and into `task-31220`'s evidence section wording you hand back to the controller (do not edit the task file): "single-instance write visible to a fresh reader: yes/no; long-lived reader saw …; contention → receipt …".
- [ ] Step 5: commit `test(library): real-DB reproduction for the bulk-delete receipt (task-31220)`.

---

### Task 2: The wedge — recovery is never gated by what it recovers from (task-31220 AC "interlock released on every path; Retry, row opens and select mode never gated behind it")

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` — (a) `_retry_library_media_browse`: when the controller's request fails or times out, paint the failure into the stale copy (`Retry failed · <reason>` where reason is the timeout/error class, via the controller's existing `stale_copy` setter) instead of leaving the previous copy unchanged; (b) `handle_library_media_row`: under the mutation gate (controller `stale_copy` set, `mutation_pending` False) a row press still opens the item (read-only), only mutating actions stay gated; (c) audit every `self._library_media_bulk_delete_in_flight = True` site (`handle_library_media_delete_selected`, `handle_library_media_bulk_delete_undo`, `handle_library_media_delete_confirm`, the review-set dismiss/undo pair, the Trash restore/permanent-delete pair) and prove each worker clears it through `_complete_library_media_mutation` in a `finally` even when the service raises — add the `finally` where missing.
- Modify: `tldw_chatbook/Widgets/Library/library_media_canvas.py` — `_gate_stale_action` keeps gating mutations; rows are NOT built through it (verify; if they are, split a `_gate_mutation_action` for mutations only).
- Modify: `tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py` — a `fail_request(reason: str)` path (or the existing failure setter) that composes `Retry failed · {reason}` once per failed retry; `request()` clears it on success.
- Test: `Tests/UI/test_library_media_browse_controller.py` (failed retry copy; success clears it), `Tests/UI/test_library_multiselect_media.py` (for each interlock site: a raising service releases the flag; Retry after a failed refresh is not a no-op — the copy changes), `Tests/UI/test_library_media_bulk_delete_real_db.py` (row press under the gate opens the item).

**Interfaces:**
- Consumes: Task 1's `_real_media_host`.
- Produces: controller copy `Retry failed · <reason>`; the invariant "rows open under the gate".

- [ ] Step 1: failing tests: (i) controller: a request that raises `TimeoutError` sets `stale_copy == "Retry failed · Library took longer than 5 s to answer"`, a request that raises `OSError("database is locked")` sets `"Retry failed · database is locked"`, and the next successful request clears it; (ii) screen fakes: for each of the six interlock sites, patch the worker's service to raise and assert `_library_media_bulk_delete_in_flight is False` afterwards; (iii) real-DB: set the gate (`controller.stale_copy = _MUTATION_COPY`), press a row, assert the viewer loaded that item.
- [ ] Step 2: run each file in its own process; confirm the failure reasons (no copy; flag stuck; row press ignored).
- [ ] Step 3: implement (a)-(c).
- [ ] Step 4: run `test_library_media_browse_controller.py`, `test_library_multiselect_media.py`, `test_library_media_bulk_delete_real_db.py`, `test_library_media_trash.py`, `test_library_media_render_fixes.py`, `test_library_shell.py -k "retry or stale or select or delete"` (compare to base).
- [ ] Step 5: live (single instance, tmux 235x52): seed 3, select one, Delete, confirm → `✓ deleted · 1 item · in Trash`, rows still openable, the DB row `is_trash=1` via a fresh `sqlite3` connection; then stop the media DB's parent directory from being readable is NOT allowed — instead simulate the outage by pointing `TLDW_CONFIG_PATH` at a scratch profile whose media DB path is a directory, open Library, press Retry twice and capture `Retry failed · <reason>` painted both times.
- [ ] Step 6: commit `fix(library): Retry says why it failed, rows open under the stale gate, interlock released on every path (task-31220)`.

---

### Task 3: Receipt and Undo agree, and focus lands on Undo (task-31220 AC "Undo is enabled whenever the receipt says ✓, and the receipt never says ✓ when Undo cannot be")

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_media_canvas.py` — the bulk-delete receipt's `Undo` (`#library-media-bulk-delete-undo`) is no longer routed through `_gate_stale_action`: it restores ids the receipt itself names, so it is the recovery for the receipt and stays enabled while the receipt shows `✓`; `Dismiss` unchanged. If an undo fails, the receipt becomes `✗ undo failed · <n> of <m> · <reason>` with `Retry undo` (reuse the receipt grammar; no new CSS).
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` — `_undo_library_media_bulk_delete`: per-item failures collected like the delete path; on any failure paint the `✗ undo failed` receipt; on completion of the delete (`_delete_library_media_selection`'s recompose tail) arm focus on `#library-media-bulk-delete-undo` when the receipt is `✓`, instead of the list entry — the confirm copy promised "You can undo right away", so Enter must undo.
- Test: `Tests/UI/test_library_multiselect_media.py` (Undo enabled while the receipt is ✓ even with `stale_copy` set; focus is on Undo after a full-success delete; a raising restore paints `✗ undo failed` with `Retry undo`), `Tests/UI/test_library_media_render_fixes.py` (painted receipt at 235x52 and 100x30: `✓ deleted · 1 item · in Trash` with `Undo` (no `○`) and `Dismiss` readable).

**Interfaces:**
- Consumes: Task 2's invariant; PR A's receipt classes.
- Produces: receipt states `✓ deleted…` / `✗ undo failed…`; focus contract "receipt ✓ → focus Undo".

- [ ] Step 1: failing tests (four above).
- [ ] Step 2: run; confirm (Undo currently `○ Undo` under the gate; focus currently on the list entry; no `✗ undo failed` state).
- [ ] Step 3: implement.
- [ ] Step 4: run `test_library_multiselect_media.py`, `test_library_media_render_fixes.py`, `test_library_media_bulk_delete_real_db.py`, `test_library_shell.py -k "undo or receipt or delete"` (compare to base); `check_bundle_sync` if any CSS changed (it should not).
- [ ] Step 5: live (single instance, tmux 235x52 then 100x30): select one → Delete → confirm → receipt shows `Undo` enabled and FOCUSED (footer names it), press Enter → item returns, receipt clears; repeat with the DB row made read-only? not possible cheaply — verify the failure receipt in the app-test instead and say so.
- [ ] Step 6: commit `fix(library): Undo is live whenever the receipt says ✓, and focus lands on it (task-31220)`.
