# De-flake the library note-conflict shell tests (task 192)

**Status:** Design approved (brainstorm — mechanism: condition-based waiting; scope: whole note-conflict family), pending spec review.
**Backlog:** task-192 — "De-flake library note-conflict shell test under CPU contention".
**Discovered:** during task 159 (unrelated to that diff).

## Problem

`test_library_shell_note_conflict_shows_overwrite_reload_and_keeps_user_text` (`Tests/UI/test_library_shell.py`) intermittently fails under concurrent CPU load but passes in isolation. Root cause: it presses Save (which drives an async worker that sets `_library_note_autosave_state = "conflict"` once the conflict seam returns `False`), then waits with a **fixed-iteration** poll loop:

```python
for _ in range(150):
    if screen._library_note_autosave_state == "conflict":
        break
    await pilot.pause(0.02)
else:
    raise AssertionError("The version conflict was never reached.")
```

150 iterations × `pilot.pause(0.02)` gives the worker only ~3s of budget. Under CPU contention the worker is starved, the iterations exhaust before the state flips, and the `else` fires. It passes in isolation because 3s is ample when the CPU is free.

The identical pattern is not unique to that one test: **12 condition-poll loops across 7 note-conflict tests** share it (and ~132 fixed-iteration loops exist file-wide). The approved scope is the note-conflict family.

## Goal / Acceptance

- **AC1** — the note-conflict tests no longer rely on a fixed-iteration budget; their condition waits use a wall-clock deadline that returns immediately when the condition is met and tolerates CPU contention.
- **AC2** — the named test (`..._shows_overwrite_reload_and_keeps_user_text`) and the other six note-conflict tests all still pass.

## Chosen approach: condition-based waiting on a wall-clock deadline

Add one async helper near the existing `_wait_for_*` helpers in the test module (`time` is already imported at `Tests/UI/test_library_shell.py:7`):

```python
async def _wait_for_condition(pilot, predicate, *, timeout=15.0, message, interval=0.02):
    """Await until predicate() is truthy or `timeout` wall-clock seconds elapse."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(interval)
    raise AssertionError(message)
```

This mirrors the original loop **exactly** — check the predicate first, return the instant it is truthy (no extra settle pause — the loops `break` without one), otherwise `await pilot.pause(interval)` and re-check — with the only change being a wall-clock deadline in place of the fixed iteration count. (Unlike `_wait_for_selector`, which does a trailing `pilot.pause()` to settle the widget it returns, a state-poll returns nothing to settle, so no trailing pause is added — keeping the conversion behavior-identical.)

Why a deadline (not "bump `range(150)` → `range(1000)`"): a wall-clock deadline decouples the budget from *iterations × per-pause-time* (which is itself unpredictable under contention), returns the instant the condition is met (so the isolation case stays fast), and replaces a magic number with an intention-revealing call. This is the condition-based-waiting pattern the debugging playbook prescribes. `15.0s` is generous enough for a starved worker yet bounded. Baseline: all 7 note-conflict tests currently pass in isolation, so this is a pure refactor over known-green tests.

Then mechanically replace each conflict-family condition-poll loop:
```python
for _ in range(150):
    if <PREDICATE>:
        break
    await pilot.pause(0.02)
else:
    raise AssertionError("<MESSAGE>")
```
with:
```python
await _wait_for_condition(pilot, lambda: <PREDICATE>, message="<MESSAGE>")
```
The `lambda` closes over the same locals the loop referenced (`screen`, `service`, `calls_before_second_save`, …); the compound predicate (idle-and-editor) becomes `lambda: screen._library_note_autosave_state == "idle" and screen._library_notes_view == "editor"`. Each call keeps that loop's original `AssertionError` message verbatim.

## Components

1. **`_wait_for_condition` helper** — added once, near `_wait_for_library_shell`/`_wait_for_selector` (~`:187-210`).
2. **12 loop conversions across 7 tests** (all `for _ in range(150):` condition polls with a predicate + `else: raise`):
   - `test_library_shell_rail_search_submit_aborts_on_note_conflict` — 1 (`conflict`).
   - `test_library_shell_note_conflict_shows_overwrite_reload_and_keeps_user_text` — 1 (`conflict`) — the named AC test.
   - `test_library_shell_note_conflict_during_preview_reads_live_text` — 2 (`conflict`; `len(service.save_calls) > calls_before_second_save`).
   - `test_library_shell_note_conflict_overwrite_resaves_with_fresh_version` — 2 (`conflict`; `saved`).
   - `test_library_shell_note_conflict_reload_discards_local_edits` — 2 (`conflict`; `idle` and `editor`).
   - `test_library_shell_note_conflict_reload_falls_back_to_list_when_note_missing` — 2 (`conflict`; `_library_notes_view == "list"`).
   - `test_library_shell_note_conflict_overwrite_falls_back_to_list_when_note_missing` — 2 (`conflict`; `_library_notes_view == "list"`).

## Data flow

Unchanged — the tests exercise the same UI flow; only the *wait mechanism* between an action and its assertion changes. No production code is touched.

## Error handling

`_wait_for_condition` raises `AssertionError(message)` on timeout — the same failure type and message the loops raised, so a genuinely-broken flow still fails loudly with the same diagnostic (not silently hangs). The `predicate` is a plain sync callable (all conditions here are attribute checks); it is called on each poll.

## Scope / non-goals

- **Only the 7 note-conflict tests' condition-poll loops.** The bare settle loop `for _ in range(10): await pilot.pause(0.02)` in `..._during_preview_reads_live_text` has no predicate/assertion — it is a deliberate event-loop drain, not a flaky condition wait — and is left untouched.
- The existing `_wait_for_library_shell` / `_wait_for_selector` / `_wait_for_library_rag_query_ready` helpers (also fixed-iteration but not reported flaky) are left as-is; reimplementing them on `_wait_for_condition` is a possible future consolidation, out of scope.
- The other ~120 fixed-iteration loops elsewhere in the file are out of scope.
- No production code changes; no change to what any test asserts.

## Testing

- **Unit test the helper (RED→GREEN):** a fast, deterministic test with a minimal fake pilot (`async def pause(self, delay=0): ...`):
  - an already-true predicate returns without waiting;
  - a never-true predicate with a tiny `timeout` (e.g. `0.05`) raises `AssertionError` carrying the exact `message`.
  RED before `_wait_for_condition` exists (import/attribute error), GREEN after.
- **Regression:** run all 7 converted note-conflict tests — they must still pass (functional behavior preserved). Each still asserts the same conditions; only the wait mechanism changed.
- Reproducing the contention flake deterministically is impractical, so the de-flake is validated by (a) the deadline-based mechanism (no fixed iteration budget) and (b) the 7 tests passing unchanged in behavior.
