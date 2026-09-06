# Media UX fix wave 5 — PR G (one recovery callout for Media load failures, task-31632) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every Library ▸ Media load failure renders as one recovery callout that names what failed, why, and what to do, with Retry inside it; a source-snapshot timeout is told apart from a hard failure and retries itself on return; the entry canvas's service wall no longer ejects the user to Home.

**Architecture:** The product already has the component and the state: `ds-recovery-callout` (a bordered, tinted `Horizontal` of copy + action, used by the Library hub's "Needs attention" row in `library_entry_canvases.py`) and `DestinationRecoveryState` (`tldw_chatbook/UI/destination_recovery.py`, built today only for policy denials by `policy_denied_recovery_state`). PR G routes the three Media failure copies through that pair: the browse controller's `_SERVICE_ERROR` ("Couldn't load media. Check the local Library and retry."), `_FACET_ERROR` and `Couldn't load page {n}.` (all in `tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py`), and the screen's `LIBRARY_SERVICE_ERROR_COPY` ("Library source services unavailable; retry Library later.", raised at `library_screen.py`'s `asyncio.wait_for(…, timeout=LIBRARY_SOURCE_SNAPSHOT_TIMEOUT_SECONDS)` site and its bare `except Exception`). No new widget class; one builder that turns a failure (kind, reason, retry action) into a `DestinationRecoveryState`, and one canvas fragment that renders it.

**Tech Stack:** Python 3.12, Textual 8.x, pytest; `Tests/UI/test_library_media_browse_controller.py` (controller failure states), `Tests/UI/test_library_shell.py` (`_wait_for_selector`, the hub/entry canvas tests), `Tests/UI/test_library_media_render_fixes.py` (`_painted`), `Tests/UI/test_library_entry_compose_once.py` for the entry canvas.

**Spec:** `backlog/tasks/task-31632 - Library-media-one-recovery-callout-for-load-failures-with-the-reason-and-Retry-adjacent.md` (AC#1-#4); critique #5 P1 "failure states are unbordered, unreasoned, and their recovery is out of sight"; DESIGN.md §"Recovery Callout" (owner, problem, impact, next action; `$warning`/`$error` tinted; text must work without colour).

## Global Constraints

- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/media-wave5-g`, branch `fix/media-wave5-g` off dev. Every command: `cd <worktree> && PYTHONPATH=<worktree> /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest … -p no:cacheprovider`; absolute paths; UI test files in separate processes; every Bash call begins with the explicit `cd` and `git branch --show-current`.
- Compare failures against the base before claiming them (known: `test_library_ingest_canvas.py::test_progress_detail_paints_below_row…`, `test_library_ingest_retry_last` flake, the `test_library_shell.py` census in task-31249).
- No new `logger.*` (the bare `except Exception` you touch already logs — keep exactly that one call). After any `BUNDLED_CSS` / TCSS edit: `python -m tldw_chatbook.css.build_css` then `python tldw_chatbook/css/check_bundle_sync.py` (exit 0); prefer the existing `ds-recovery-callout` rules so no CSS changes.
- The stale gate copy `Media changed; retry to load a current page.` (`_MUTATION_COPY`) and PR E's `Retry failed · <reason>` are NOT failures of this kind — leave them; they are the mutation gate, not a load failure. Five-key contract frozen; review-set code and the Find focus token untouched; no new toolbar buttons.
- Copy rules: name what failed, why, and what to do, in that order; never tell the user to "retry later" without a Retry in reach; a timeout says how long was waited (`Library took longer than 5 s to answer`).
- Live verification: tmux (function `t() { tmux -L w5g "$@"; }` in every call, sleeps inside, `t kill-server` at the end), real config, ONE app instance; to provoke a failure without harming the user's data, point `TLDW_CONFIG_PATH` at a scratch profile under the scratchpad directory whose `[database] media_db_path` is a path inside a directory with no write permission (or a file that is not a database) — say which; never touch the real DB.
- TDD per task; commit per task with the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; backlog task files are flipped by the controller.

---

### Task 1: A recovery state for load failures, and the browse controller produces it (task-31632 AC#1 for the three controller failures)

**Files:**
- Modify: `tldw_chatbook/UI/destination_recovery.py` — add `load_failure_recovery_state(*, what: str, reason: str, retry_id: str, stable_selector: str) -> DestinationRecoveryState` next to `policy_denied_recovery_state`: `message = f"{what} · {reason}"`, next action = Retry with the given button id, tinted `$error` (hard failure) or `$warning` (timeout, decided by the caller passing `kind="timeout"`).
- Modify: `tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py` — replace the three string constants' consumers: the failed-request path (`Couldn't load page {n}.`), `_SERVICE_ERROR`, `_FACET_ERROR` now record a `failure: DestinationRecoveryState | None` on the controller state (alongside the existing `error_copy` field, which keeps carrying the plain sentence for existing consumers), built with `what="Couldn't load page {n}"` / `"Couldn't load media"` / `"Couldn't load media types"` and `reason` derived from the exception class and message the request captured (`TimeoutError` → `Library took longer than 5 s to answer`, `OSError`/`sqlite3.OperationalError` → its message, anything else → the class name). Success clears it.
- Test: `Tests/UI/test_library_media_browse_controller.py` — a failed page request with `TimeoutError` yields a warning-tinted state with message `Couldn't load page 1 · Library took longer than 5 s to answer` and retry id `library-media-retry`; a failed request with `sqlite3.OperationalError("database is locked")` yields an error-tinted state with that reason; a facet failure yields `Couldn't load media types · …`; the next success clears `failure`.

**Interfaces:**
- Produces: `load_failure_recovery_state(...)`, controller field `failure`.

- [ ] Step 1: failing tests (four cases above).
- [ ] Step 2: run; confirm (`failure` attribute missing / builder missing).
- [ ] Step 3: implement.
- [ ] Step 4: run `test_library_media_browse_controller.py`, `test_library_multiselect_media.py` (controller consumers), compare to base.
- [ ] Step 5: no live step (state only).
- [ ] Step 6: commit `feat(library): Media load failures carry a recovery state with the reason (task-31632)`.

---

### Task 2: The Media canvas renders the failure as one callout with Retry inside it (task-31632 AC#1, AC#3 for the page/service/facet failures)

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_media_canvas.py` — where the canvas paints `error_copy` today (the bare `Static` above the rows) render instead a `Horizontal(classes="ds-recovery-callout", id="library-media-load-failure")` with the state's message and the Retry button INSIDE it (`id="library-media-retry"` moves here from the pager strip when a failure is showing; the pager keeps its Retry only for the stale gate). The rows below stay retained-and-gated exactly as today.
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` — `_library_media_canvas_presentation()` passes the controller's `failure` through; `handle_library_media_retry` unchanged (same id).
- Test: `Tests/UI/test_library_media_render_fixes.py` — painted at 235x52 and 100x30: with a forced page failure the callout paints `Couldn't load page 1 · <reason>` and `Retry` on the same row or the row directly below, within 3 rows of the message, and NOT 34 rows below; the pager strip shows no second Retry; pressing the callout's Retry issues a new request (assert on the controller's request log). `Tests/UI/test_library_shell.py -k "retry"` compare to base.

**Interfaces:**
- Consumes: Task 1's `failure` state.
- Produces: `#library-media-load-failure` callout; the rule "Retry lives inside the callout".

- [ ] Step 1: failing tests (painted ×2 sizes; no-second-Retry; Retry re-requests).
- [ ] Step 2: run; confirm (bare sentence; Retry in the pager).
- [ ] Step 3: implement; rebuild CSS only if a rule is needed (prefer none).
- [ ] Step 4: run `test_library_media_render_fixes.py`, `test_library_media_browse_controller.py`, `test_library_multiselect_media.py`, `test_library_media_trash.py`, `test_library_shell.py -k "retry or error or fail"` (compare to base).
- [ ] Step 5: live (scratch profile with an unreadable media DB path, 235x52): open Library ▸ Media → the callout with the reason and Retry adjacent; press Retry → the callout repaints (same or new reason), never silently.
- [ ] Step 6: commit `fix(library): Media load failures render as one recovery callout with Retry inside it (task-31632)`.

---

### Task 3: The source-snapshot timeout is a warning that retries itself, and the entry canvas stops ejecting to Home (task-31632 AC#2, AC#3, AC#4)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` — at the `asyncio.wait_for(asyncio.gather(*gathered_calls), timeout=LIBRARY_SOURCE_SNAPSHOT_TIMEOUT_SECONDS)` site: catch `asyncio.TimeoutError` separately from the bare `except Exception`; the timeout returns a `load_failure_recovery_state(kind="timeout", what="Library sources did not answer", reason=f"waited {LIBRARY_SOURCE_SNAPSHOT_TIMEOUT_SECONDS:g} s", retry_id="library-source-retry", …)`; the generic branch keeps `LIBRARY_SERVICE_ERROR_COPY` as `what` but adds the exception's class/message as `reason`. Returning to Library (screen resume / rail re-entry) with a timeout recovery state present re-runs the snapshot once automatically (AC#2) — find the existing re-entry seam (`on_screen_resume` or the rail handler) rather than adding a timer.
- Modify: `tldw_chatbook/Widgets/Library/library_entry_canvases.py` — the service wall renders through the SAME `ds-recovery-callout` fragment as the hub's attention row, with Retry (`library-source-retry`) inside; `continue_action` is no longer offered as the failure's control — if the hub still offers a Continue for other reasons, its label says where it goes (`Continue to Home`).
- Test: `Tests/UI/test_library_shell.py` (a snapshot that times out paints a warning callout `Library sources did not answer · waited 5 s` with Retry; a snapshot that raises `RuntimeError("boom")` paints an error callout whose reason is `RuntimeError: boom`; re-entering Library after a timeout re-runs the snapshot once — assert the call count), `Tests/UI/test_library_entry_compose_once.py` (the failure fragment composes once and does not eject).

**Interfaces:**
- Consumes: Task 1's builder.
- Produces: `library-source-retry` id; the re-entry auto-retry rule.

- [ ] Step 1: failing tests (timeout callout; hard-failure callout; auto-retry count; compose-once).
- [ ] Step 2: run; confirm (bare sentence + Continue; no auto-retry).
- [ ] Step 3: implement.
- [ ] Step 4: run `test_library_shell.py -k "source or snapshot or unavailable or entry"`, `test_library_entry_compose_once.py`, `test_library_media_render_fixes.py` (compare to base); `python scripts/check_persistent_diagnostic_inventory.py`.
- [ ] Step 5: live (scratch profile, 235x52): make the snapshot time out by pointing the media DB at a FIFO or a very slow path is not reliable — instead patch nothing live; verify the timeout callout in the app-test and, live, verify the hard-failure callout (unreadable DB) shows Retry inside it and no Continue, at 235x52 and 100x30; say which was live.
- [ ] Step 6: commit `fix(library): source-snapshot timeout is a warning that retries on return; the service wall is a callout, not an exit (task-31632)`.
