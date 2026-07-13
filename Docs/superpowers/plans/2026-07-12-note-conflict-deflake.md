# De-flake the library note-conflict shell tests — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax. Spec: `Docs/superpowers/specs/2026-07-12-note-conflict-deflake-design.md`. Branch `claude/followups-note-conflict-flake` off dev `c5e9886d`. All changes are in `Tests/UI/test_library_shell.py`; line numbers drift as edits land, so key each conversion on the test name + predicate/message shown below, not on line numbers.

**Goal:** Replace the fixed-iteration (~3s) poll loops in the 7 note-conflict shell tests with a wall-clock-deadline `_wait_for_condition` helper, so they stop flaking under CPU contention while staying fast in isolation.

**Architecture:** Add one async helper next to the existing `_wait_for_*` helpers, then mechanically convert all 12 conflict-family `for _ in range(150):` condition-poll loops to call it. Test-only; no production code changes. All 7 tests currently pass in isolation (verified), so this is a pure refactor.

**Tech Stack:** pytest + pytest-asyncio, Textual `Pilot`, `time.monotonic` (already imported at `test_library_shell.py:7`).

## Global Constraints

- **Helper mirrors the original `break` exactly** — check predicate first, return the instant it is truthy with NO trailing settle pause, else `await pilot.pause(interval)` and re-check; the only change vs the loops is a wall-clock deadline replacing `range(150)`.
- **`message` may be a str OR a zero-arg callable** — the callable is evaluated at raise time so the two dynamic diagnostic messages still report the *stuck* state at timeout.
- **Preserve each loop's original `AssertionError` message verbatim** (as a string, or wrapped in `lambda:` for the two f-string diagnostics).
- **Scope: only the 12 conflict-family loops in the 7 named tests.** Do NOT touch the bare `for _ in range(10): await pilot.pause(0.02)` drain in `..._during_preview_reads_live_text` (no predicate). Do NOT touch `_wait_for_library_shell`/`_wait_for_selector`/`_wait_for_library_rag_query_ready` or any other file's loops. No production code changes.
- **Staging:** explicit paths only. Every commit ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- **Test command** (venv, isolated HOME):
  ```
  HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
    .venv/bin/python -m pytest <target> \
    -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
  ```

---

### Task 1: `_wait_for_condition` helper + convert the 7 note-conflict tests

**Files:**
- Modify: `Tests/UI/test_library_shell.py` (add helper ~`:210` after the other `_wait_for_*` helpers; add helper unit tests; convert 12 loops in 7 tests)
- Modify: `backlog/tasks/task-192 - De-flake-library-note-conflict-shell-test-under-CPU-contention.md`

- [ ] **Step 1: Write the failing unit tests for the helper**

Add near the top of `Tests/UI/test_library_shell.py` (after the imports / existing `_wait_for_*` helpers). These are self-contained (a no-op fake pilot):
```python
class _FakePilot:
    """Minimal pilot stand-in for unit-testing _wait_for_condition (pause is a no-op)."""
    async def pause(self, delay: float = 0) -> None:
        return None


@pytest.mark.asyncio
async def test__wait_for_condition_returns_immediately_when_true():
    calls = {"n": 0}

    def pred() -> bool:
        calls["n"] += 1
        return True

    await _wait_for_condition(_FakePilot(), pred, message="must not raise")
    assert calls["n"] == 1  # checked once, returned without pausing


@pytest.mark.asyncio
async def test__wait_for_condition_raises_with_message_on_timeout():
    with pytest.raises(AssertionError, match="boom"):
        await _wait_for_condition(_FakePilot(), lambda: False, timeout=0.05, message="boom")


@pytest.mark.asyncio
async def test__wait_for_condition_evaluates_callable_message_at_raise():
    with pytest.raises(AssertionError, match="dynamic 42"):
        await _wait_for_condition(
            _FakePilot(), lambda: False, timeout=0.05, message=lambda: f"dynamic {6 * 7}"
        )
```

- [ ] **Step 2: Run the unit tests to verify they FAIL**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  .venv/bin/python -m pytest \
  "Tests/UI/test_library_shell.py::test__wait_for_condition_returns_immediately_when_true" \
  "Tests/UI/test_library_shell.py::test__wait_for_condition_raises_with_message_on_timeout" \
  "Tests/UI/test_library_shell.py::test__wait_for_condition_evaluates_callable_message_at_raise" \
  -q -p no:cacheprovider -o addopts="" --timeout=60
```
Expected: FAIL — `NameError: name '_wait_for_condition' is not defined`.

- [ ] **Step 3: Add the `_wait_for_condition` helper**

Add immediately after `_wait_for_selector` (~`Tests/UI/test_library_shell.py:210`):
```python
async def _wait_for_condition(pilot, predicate, *, timeout=15.0, message, interval=0.02):
    """Await until ``predicate()`` is truthy, or raise once ``timeout`` wall-clock seconds elapse.

    A deadline (not a fixed iteration count) so the wait survives CPU contention
    yet returns the instant the condition is met. ``message`` may be a string or a
    zero-arg callable (evaluated at raise time, so dynamic diagnostics report the
    stuck state).
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(interval)
    raise AssertionError(message() if callable(message) else message)
```

- [ ] **Step 4: Run the unit tests to verify they PASS**

Run the Step-2 command. Expected: 3 passed.

- [ ] **Step 5: Convert the 12 conflict-family poll loops**

In each of the 7 tests below, replace every block of the form
```python
        for _ in range(150):
            if <PREDICATE>:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(<MESSAGE>)
```
with a single line
```python
        await _wait_for_condition(pilot, lambda: <PREDICATE>, message=<MESSAGE>)
```
Do them one test at a time. The exact `(PREDICATE, MESSAGE)` pairs — 12 loops:

- **`test_library_shell_rail_search_submit_aborts_on_note_conflict`** (1 loop):
  - `screen._library_note_autosave_state == "conflict"` → `message="The version conflict was never reached."`
- **`test_library_shell_note_conflict_shows_overwrite_reload_and_keeps_user_text`** (1 loop — the named AC test):
  - `screen._library_note_autosave_state == "conflict"` → `message="The version conflict was never reached."`
- **`test_library_shell_note_conflict_during_preview_reads_live_text`** (2 loops; leave the `for _ in range(10): await pilot.pause(0.02)` drain untouched):
  - `screen._library_note_autosave_state == "conflict"` → `message="The version conflict was never reached."`
  - `len(service.save_calls) > calls_before_second_save` → `message="The second Save press never called the seam."`
- **`test_library_shell_note_conflict_overwrite_resaves_with_fresh_version`** (2 loops):
  - `screen._library_note_autosave_state == "conflict"` → `message="The version conflict was never reached."`
  - `screen._library_note_autosave_state == "saved"` → `message="Overwrite never completed."`
- **`test_library_shell_note_conflict_reload_discards_local_edits`** (2 loops; the second is a compound predicate):
  - `screen._library_note_autosave_state == "conflict"` → `message="The version conflict was never reached."`
  - `screen._library_note_autosave_state == "idle" and screen._library_notes_view == "editor"` → `message="Reload never completed."`
- **`test_library_shell_note_conflict_reload_falls_back_to_list_when_note_missing`** (2 loops; the second has a dynamic message → use a `lambda:`):
  - `screen._library_note_autosave_state == "conflict"` → `message="The version conflict was never reached."`
  - `screen._library_notes_view == "list"` → `message=lambda: ("Reload never fell back to the list view for a missing note " f"(stuck: view={screen._library_notes_view!r}, " f"autosave_state={screen._library_note_autosave_state!r}).")`
- **`test_library_shell_note_conflict_overwrite_falls_back_to_list_when_note_missing`** (2 loops; the second has a dynamic message → use a `lambda:`):
  - `screen._library_note_autosave_state == "conflict"` → `message="The version conflict was never reached."`
  - `screen._library_notes_view == "list"` → `message=lambda: ("Overwrite never fell back to the list view for a missing note " f"(stuck: view={screen._library_notes_view!r}, " f"autosave_state={screen._library_note_autosave_state!r}).")`

Verify no conflict-family fixed-iteration loop remains: `grep -nE 'for _ in range\(150\):' Tests/UI/test_library_shell.py` should show only loops OUTSIDE the 7 tests above (the ~120 other-feature loops are out of scope and stay).

- [ ] **Step 6: Run all 7 note-conflict tests + the helper unit tests to verify GREEN**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  .venv/bin/python -m pytest \
  "Tests/UI/test_library_shell.py" -k "note_conflict or _wait_for_condition" \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: 10 passed (7 note-conflict tests + 3 helper unit tests), 0 failed. If a conflict test fails, a predicate/message was mis-mapped — compare against the pairs in Step 5.

- [ ] **Step 7: Mark backlog task 192 Done**

```bash
perl -0pi -e 's/- \[ \] #1/- [x] #1/' "backlog/tasks/task-192 - De-flake-library-note-conflict-shell-test-under-CPU-contention.md" 2>/dev/null
perl -0pi -e 's/^status: .*/status: Done/mi' "backlog/tasks/task-192 - De-flake-library-note-conflict-shell-test-under-CPU-contention.md"
```
(If task-192 has no `## Acceptance Criteria` checkbox, only the status line changes — that's fine.) Add a short `## Implementation Notes` section: replaced the fixed-iteration conflict polls across the 7 note-conflict tests with a deadline-based `_wait_for_condition` helper (str-or-callable message); bare drain loop + other-feature loops left as-is. Confirm the status changed (`grep -n "status:" "backlog/tasks/task-192 - De-flake"*.md`).

- [ ] **Step 8: Commit**

```bash
git add Tests/UI/test_library_shell.py "backlog/tasks/task-192 - De-flake-library-note-conflict-shell-test-under-CPU-contention.md"
git commit -m "test(library): deadline-based _wait_for_condition to de-flake note-conflict polls; task 192 done (192)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Final gate (after Task 1)

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  .venv/bin/python -m pytest \
  "Tests/UI/test_library_shell.py" -k "note_conflict or _wait_for_condition" \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected 10 passed. Then the whole-branch review and finishing-a-development-branch. (Optional extra confidence: run the full `Tests/UI/test_library_shell.py` once to confirm no unrelated regression from the shared-helper edit.)
