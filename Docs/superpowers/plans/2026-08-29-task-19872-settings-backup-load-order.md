# TASK-19872 Settings Backup Load Ordering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make repeated Advanced Config backup loads latest-request-wins without weakening protection for typing that occurs after the newest request is dispatched.

**Architecture:** Add one monotonic request token to `SettingsScreen`, carry it through the existing thread worker, and drop stale callbacks before they can touch editor, validation, or result state. Preserve the existing dispatch-text guard for the newest callback. Prove the race with real mounted Textual workers and deterministic thread/callback handshakes before changing production code.

**Tech Stack:** Python 3.11+, Textual 8.2.8 worker API, pytest/pytest-asyncio, Ruff, repository diagnostic and pre-import artifact guards.

---

## File Map

- Modify: `tldw_chatbook/UI/Screens/settings_screen.py` — own the request token and enforce latest-request arrival semantics.
- Modify: `Tests/UI/test_settings_configuration_hub.py` — reproduce both callback orders, stale errors, serial success, and retained typing protection.
- Modify: `Docs/User_Guide/settings.md` — document latest-request-wins repeated loads and refresh the verification stamp.
- Modify: `backlog/tasks/task-19872 - Double-pressing-Settings-Load-Backup-reports-edits-were-kept-that-never-existed.md` — implementation plan, reproduced evidence, checked ACs, notes, and Done state.
- Conditional modify: `Docs/security/production-diagnostic-inventory.json` — regenerate only if the canonical checker reports line-sensitive drift caused by the Settings edit.
- Verify only: `Tests/Performance/test_screen_preimport_payload_budget.py` — confirm the scoped Settings LOC change remains inside the existing ratchet; do not refresh its diagnostic snapshot.

## Task 1: Reproduce the double-load race with deterministic mounted tests

**Files:**

- Modify: `Tests/UI/test_settings_configuration_hub.py:9750-9850,11158-11226`

- [ ] **Step 1: Add a condition-based callback recorder and per-worker gates**

  Build the test helper locally in the new test, using `threading.Event` objects for `started[0:2]`, `release[0:2]`, and `callback_returned[0:2]`, a lock-protected call index, and distinct old/new `(result, backup_text)` payloads. Await thread events without blocking Textual's loop via bounded calls such as `assert await asyncio.to_thread(event.wait, 5)`. Wrap the real `_apply_advanced_backup_preview_result` with `*args, **kwargs`, call the real method, append the callback observation, and signal the callback-return event in `finally`. Do not use fixed sleeps to establish ordering.

  Put every worker `release.set()` in the mounted test's outer `finally` block so an assertion failure cannot strand an executor thread.

- [ ] **Step 2: Add RED overlap cases**

  Add `test_settings_advanced_config_backup_load_latest_request_wins` as a parameterized mounted test covering:

  1. old callback, then new callback;
  2. new callback, then old callback;
  3. new successful callback, then stale old error callback.

  For each case:

  - wait for worker 1 to enter before the second click;
  - wait for worker 2 to enter before releasing either;
  - await the corresponding callback-return event after each release;
  - assert the newest backup text, newest result copy, and `None` validation state survive.

- [ ] **Step 3: Add the serial characterization case**

  Add `test_settings_advanced_config_backup_load_serial_repeats_report_success`: allow the first click to complete successfully, then dispatch and complete the second. Assert both completions use the ordinary loaded-preview success path. This pins unchanged authorized-replacement behavior; the overlapping test remains the RED oracle.

- [ ] **Step 4: Make the existing typing-protection test deterministic**

  Replace `pilot.pause(0.1)`, the already-satisfied `"Advanced config recovery:"` prefix wait, and `pilot.pause(0.2)` in `test_settings_advanced_config_backup_load_never_clobbers_unsaved_typing`. Gate the backup read with worker-started/release events, type only after the worker-start event is observed through bounded `asyncio.to_thread`, wrap the real callback to signal callback return in `finally`, and release the worker from the test's outer `finally`. Preserve the existing assertions.

- [ ] **Step 5: Run the RED gate and record the actual failure**

  Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
    Tests/UI/test_settings_configuration_hub.py::test_settings_advanced_config_backup_load_latest_request_wins \
    Tests/UI/test_settings_configuration_hub.py::test_settings_advanced_config_backup_load_serial_repeats_report_success -q
  ```

  Expected: the serial case passes; overlap parameters fail on current `dev` because the first/stale callback can paint or replace the latest result with the false preservation/error message. Save the exact observed order and assertion failure for TASK-19872 notes.

## Task 2: Implement the minimal latest-request token

**Files:**

- Modify: `tldw_chatbook/UI/Screens/settings_screen.py:2690-2692,9116-9171,22107-22119`
- Modify: `Tests/UI/test_settings_configuration_hub.py:9828-9850`

- [ ] **Step 1: Initialize one monotonic token**

  Immediately beside `_advanced_config_result` and `_advanced_config_validated_text`, add:

  ```python
  self._advanced_backup_load_token = 0
  ```

  Keep this as a plain integer; do not add a helper class, queue, lock, or new dependency.

- [ ] **Step 2: Capture the newest request at the button boundary**

  In `handle_advanced_load_backup`, increment the token and call the worker with both the current editor text and token:

  ```python
  self._advanced_backup_load_token += 1
  self._advanced_load_backup_worker(
      self._advanced_editor_text(), self._advanced_backup_load_token
  )
  ```

  Update the existing handler unit test so its fake worker accepts both arguments and asserts token `1`.

- [ ] **Step 3: Carry the token through the thread worker**

  Change `_advanced_load_backup_worker` to accept `load_token: int` and pass it as the final argument to `_apply_advanced_backup_preview_result`. Update the Google-style method documentation to name latest-request ownership alongside the existing thread-cancellation fact.

- [ ] **Step 4: Drop stale callbacks before every side effect**

  Add `load_token: int | None = None` to `_apply_advanced_backup_preview_result` for direct-test compatibility, then make its first executable branch:

  ```python
  if load_token is not None and load_token != self._advanced_backup_load_token:
      return
  ```

  This check must precede `final_result = result`, editor reads/writes, validation updates, and result-line writes. Leave the existing dispatch-text protection otherwise unchanged.

- [ ] **Step 5: Run GREEN focused tests**

  Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
    Tests/UI/test_settings_configuration_hub.py::test_settings_advanced_config_backup_load_latest_request_wins \
    Tests/UI/test_settings_configuration_hub.py::test_settings_advanced_config_backup_load_serial_repeats_report_success \
    Tests/UI/test_settings_configuration_hub.py::test_settings_advanced_config_loads_backup_preview_without_saving \
    Tests/UI/test_settings_configuration_hub.py::test_settings_advanced_config_load_backup_handler_uses_worker \
    Tests/UI/test_settings_configuration_hub.py::test_settings_advanced_config_backup_load_never_clobbers_unsaved_typing -q
  ```

  Expected: all cases pass; known dependency and temporary-cleanup warnings may remain.

- [ ] **Step 6: Mutation-check both guards**

  With `apply_patch`, temporarily remove/invert the token mismatch return and rerun the overlap test. Expected: at least the distinct old/new ownership cases fail. Restore it with `apply_patch` and confirm green.

  Separately, temporarily remove the dispatch-text mismatch branch and run `test_settings_advanced_config_backup_load_never_clobbers_unsaved_typing`. Expected: it fails because the typed text is overwritten. Restore it and confirm green.

- [ ] **Step 7: Commit the behavior and tests**

  ```bash
  git add tldw_chatbook/UI/Screens/settings_screen.py \
    Tests/UI/test_settings_configuration_hub.py
  git diff --cached --name-status
  git commit -m "fix(settings): make backup loads latest-request-wins"
  ```

## Task 3: Update user and task documentation

**Files:**

- Modify: `Docs/User_Guide/settings.md:551-555,784-792`
- Modify: `backlog/tasks/task-19872 - Double-pressing-Settings-Load-Backup-reports-edits-were-kept-that-never-existed.md`

- [ ] **Step 1: Update current user guidance**

  Keep the existing post-dispatch typing guarantee and add one sentence: repeated backup loads are latest-request-wins and do not manufacture an unsaved-edit warning. Update the dated verification stamp to name TASK-19872 and both deterministic overlap orders.

- [ ] **Step 2: Record implementation evidence without closing early**

  Add concise Implementation Notes covering the reproduced RED sequence, token decision, serial semantics, real-typing preservation, mutation evidence, and ADR result. Do not check ACs or mark Done until the final gate succeeds.

- [ ] **Step 3: Commit documentation**

  ```bash
  git add Docs/User_Guide/settings.md \
    'backlog/tasks/task-19872 - Double-pressing-Settings-Load-Backup-reports-edits-were-kept-that-never-existed.md'
  git diff --cached --name-status
  git commit -m "docs(task-19872): document backup load ordering"
  ```

## Task 4: Verify generated evidence and close the task

**Files:**

- Conditional modify: `Docs/security/production-diagnostic-inventory.json`
- Verify only: `Tests/Performance/test_screen_preimport_payload_budget.py`
- Modify: `backlog/tasks/task-19872 - Double-pressing-Settings-Load-Backup-reports-edits-were-kept-that-never-existed.md`

- [ ] **Step 1: Check diagnostic inventory before writing**

  Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
    scripts/check_persistent_diagnostic_inventory.py
  ```

  If it fails, inspect the changed Settings statements directly:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
    scripts/check_persistent_diagnostic_inventory.py \
    --statements tldw_chatbook/UI/Screens/settings_screen.py --since origin/dev
  ```

  Regenerate only when that output proves the delta is reviewed line movement from the scoped Settings edit:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
    scripts/check_persistent_diagnostic_inventory.py --write
  git diff -- Docs/security/production-diagnostic-inventory.json
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
    scripts/check_persistent_diagnostic_inventory.py
  ```

  Do not bless unrelated semantic drift.

- [ ] **Step 2: Check the pre-import payload ratchet without refreshing its snapshot**

  Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
    Tests/Performance/test_screen_preimport_payload_budget.py -q
  ```

  The snapshot is diagnostic context, not an equality gate. Do not regenerate `preimport_payload.json` for this fix. If the actual ratchet breaches, reduce imported payload or follow ADR-097's explicit exception process; refreshing the snapshot cannot make the breach pass.

- [ ] **Step 3: Run the focused final gate**

  Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
    Tests/UI/test_settings_configuration_hub.py -q -k \
    'advanced_config and (backup or load_backup)'
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
    Tests/Performance/test_screen_preimport_payload_budget.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
    tldw_chatbook/UI/Screens/settings_screen.py \
    Tests/UI/test_settings_configuration_hub.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check \
    tldw_chatbook/UI/Screens/settings_screen.py \
    Tests/UI/test_settings_configuration_hub.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
    scripts/check_persistent_diagnostic_inventory.py
  git diff --check
  git diff origin/dev...HEAD --check
  ```

  Keep the Settings selection and pre-import guard as separate pytest commands so the `-k` expression cannot filter the performance test. Do not run the full suite without separate user approval.

- [ ] **Step 4: Self-review the complete branch**

  Inspect `git diff --stat origin/dev...HEAD` and `git diff origin/dev...HEAD`. Confirm there is no unrelated Settings refactor, no button-disable behavior, no new abstraction/dependency, and no generated drift beyond reviewed line/LOC changes.

- [ ] **Step 5: Close TASK-19872 only after evidence is green**

  Check all five ACs and finalize Implementation Notes with exact commands/results and any qualified warnings. Then run:

  ```bash
  backlog task edit 19872 -s Done
  backlog task 19872 --plain
  ```

  Capture the exact task path printed by the CLI because it may rewrite the filename.

- [ ] **Step 6: Commit closeout evidence**

  Stage only the exact task path printed in Step 5 and any verified diagnostic inventory artifact. For example, if the path is unchanged:

  ```bash
  git add 'backlog/tasks/task-19872 - Double-pressing-Settings-Load-Backup-reports-edits-were-kept-that-never-existed.md'
  git diff --cached --name-status
  git commit -m "docs(task-19872): close backup load ordering fix"
  ```

  Do not copy the example path blindly. Add `Docs/security/production-diagnostic-inventory.json` only if Step 1 required and verified it.

## Final Review and Branch Handoff

- [ ] Dispatch a fresh whole-branch correctness reviewer after all plan tasks and fix any validated findings.
- [ ] Invoke `superpowers:verification-before-completion` and rerun the final evidence required for any completion claim.
- [ ] Invoke `superpowers:finishing-a-development-branch`; keep this worktree for PR review if the owner selects the PR path.
