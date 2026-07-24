# Follow-ups correctness batch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Three correctness follow-ups (backlog 152, 153, 156). Branch `claude/followups-correctness` off dev cedd2223. Anchors exact at branch point; grep symbols, lines drift.

**Goal:** Fix three correctness defects: Home Retry requeues the wrong item, recent-only Home items can't be opened, and the chatbook registry write is a lost-update race.

**Global Constraints:** explicit-path staging (NEVER `git add -A`); Fable 5 co-author line; RED-first; parameterized SQL only; venv pytest with isolated HOME. Test command: `HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <files> -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread`.

## Cluster A — Home dashboard controls (152 + 153)

**Root cause (152), already traced:** the rendered Home canvas buttons come from the SELECTION-SCOPED controls (`build_home_triage_state(..., selected_row_id=...)` → `triage.canvas.actions`, built via `build_home_controls(state, selected_item=<selected>)`), which already set `home-retry.target_id` to the SELECTED failed item (dashboard_state.py:353-360, the `selected_item_is_failed` override). But `home_screen.py::_activate_home_control` (~:363) dispatches by looking the pressed control up in `self._current_dashboard.controls` — the UNSCOPED `summarize_home_dashboard(...)` result, whose `home-retry.target_id` is `_first_item_for_status(...)` (the FIRST failed item). So Retry always requeues the first failed item, not the selected one.

### T152 — dispatch from the scoped canvas controls
**Files:** Modify `tldw_chatbook/UI/Screens/home_screen.py` (store the scoped controls at the two `build_home_triage_state` sites ~:153/:299; `_activate_home_control` ~:363). Test: `Tests/UI/test_home_screen.py`.
- Store the scoped canvas controls on the screen whenever the triage is (re)built — e.g. `self._current_canvas_controls: tuple[HomeControl, ...] = triage.canvas.actions` alongside the existing `self._current_dashboard` assignment (both refresh sites). Init it `()` in `__init__`.
- In `_activate_home_control`, resolve the pressed control from `self._current_canvas_controls` FIRST (the scoped set the user actually sees/pressed), falling back to `self._current_dashboard.controls` only when not found there (defensive; count-only fallback canvases have no selected item and their controls live in both). Everything downstream (`method(**kwargs)` with `target_id`) is unchanged — the scoped control simply carries the correct `target_id`.
- RED pilot: build a Home with 2+ retryable failed items, select the SECOND, press its rendered `#home-retry`, assert `app.retry_active_home_item` was called with the SECOND item's `target_id` (not the first). Verify it fails against the current unscoped dispatch. Mirror the existing home_screen pilots' harness (real HomeHarness / RecordingHomeActiveWorkAdapter, bounded pauses).

### T153 — recent-only selected item gets an open control
**Files:** Modify `tldw_chatbook/Home/dashboard_state.py` (`build_home_controls` ~:376-460, the `home-open-details` emission block). Test: `Tests/Home/test_dashboard_state.py`.
- Today `home-open-details` is emitted only inside the `if _pending_approval_count(state) or _active/_running/_paused/_failed count` block, so a selected RECENT-ONLY item (a done import, a chatbook artifact — present only in `recent_work_items`, bumping no active count) gets NO open control on its canvas.
- Fix: when `selected_item is not None` AND no `home-open-details` control was emitted by the count-driven block AND the selected item is not already covered, append a `home-open-details` HomeControl targeting the selected item (`target_route=selected_item.detail_route`, `target_id=selected_item.item_id`, `applies_to="work_details"` — match the existing open-details control's shape; read it in the count-driven block). Keep the count-driven open-details path unchanged for active items. Guard against a double `home-open-details` (only add when absent).
- The dispatch method `open_active_home_item_details` already accepts a `local:ingest:*`/recent id (verified in F1b review — recent path routes through the same handler), so AC#2 (no crash) holds; add a light assertion in the pilot that invoking it for a recent item doesn't raise.
- RED unit: a Home input whose ONLY selectable item is a recent-only item (empty active/failed counts, one `recent_work_items` entry) selected → its canvas controls include `home-open-details` targeting that item. Verify it fails today (no open control).

**Cluster A commit:** `fix(home): Retry requeues the selected failed item; recent-only items get an open control (152,153)`

## Cluster B — chatbook registry write race (156)

### T156 — serialize the registry read-modify-write
**Files:** Modify `tldw_chatbook/Chatbooks/local_chatbook_service.py` (`__init__` :22; every load→mutate→`_save_registry` region: `create_chatbook` :173, and the update/delete methods around :160-223). Test: `Tests/Chatbooks/` (add a concurrency test).
- Concurrency is across OS threads (the export worker calls these via `asyncio.run` on a `@work(thread=True)` worker; a second overlapping export runs on another thread — the disclosed by-design path). `_load_registry`→mutate→`_save_registry` with no lock can lose-update a record or collide on `next_id`.
- Add `self._registry_lock = threading.Lock()` in `__init__` (import `threading`). Wrap EACH read-modify-write region (create/update/delete — every method that calls `_save_registry`) in `with self._registry_lock:` covering the `_load_registry()` through `_save_registry()` span so the whole RMW is atomic. Pure reads (`list_chatbooks`, `_find_record`-only) need no lock. `_save_registry` already uses `atomic_write_json` (keeps the file valid) — the lock closes the lost-update window.
- RED test: spawn N threads (e.g. 20) each calling `create_chatbook` concurrently against one service instance (real temp registry path); after `join`, assert the registry has exactly N records and N distinct `chatbook_id`s (no lost updates, no id collision). Verify it fails without the lock (flaky-fails; run it a few times or with enough threads to force the race — document the pre-fix failure).

**Cluster B commit:** `fix(chatbooks): serialize the registry read-modify-write against concurrent exports (156)`

## Verification & gate

Combined gate: `Tests/Home/ Tests/UI/test_home_screen.py Tests/Chatbooks/ Tests/Library/` + `python -c "import tldw_chatbook.app"`. Visual QA: only 153 adds a visible control — capture the Home canvas with a recent-only item selected showing its open control (cheap, served TUI). Present with the PR. Mark backlog 152/153/156 Done. PR to dev; merge only on explicit user authorization.
