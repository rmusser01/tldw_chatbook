# TASK-3401.17 Generated Video Message Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task by task.

**Goal:** Preserve one generated-video storage identity across persistence and
Console restart so unexpired TTL-retained videos remain ready and actionable.

**Architecture:** Reuse ADR-044's existing message-keyed `VideoStore` contract.
Video messages will persist their preallocated native UUID as the database row ID.
After reload, a single Console helper will select `persisted_message_id` for file
lookups, falling back to the native `id` only for live, non-persisted messages.
Card dictionaries and action dispatch continue using native IDs; only filesystem
addressing uses the durable-first key.

**Tech Stack:** Python 3.11+, Textual, SQLite, pytest, Ruff.

---

## Scope and ADR Check

- Production files:
  - `tldw_chatbook/Chat/console_chat_store.py`
  - `tldw_chatbook/UI/Screens/chat_screen.py`
  - `tldw_chatbook/UI/Console_Modules/message.py`
  - `tldw_chatbook/UI/Console_Modules/wiring.py`
- Focused test files:
  - `Tests/Chat/test_console_video_message.py`
  - `Tests/Chat/test_console_video_actions.py`
- Task/documentation files:
  - `backlog/tasks/task-3401.17 - Preserve-generated-video-identity-across-Console-lifecycle.md`
  - this plan

ADR required: no

ADR path: `backlog/decisions/044-ephemeral-generated-video-storage-playback-and-streaming.md`

Reason: ADR-044 already defines stable message-keyed ephemeral storage and TTL
restart behavior. This task repairs conformance without changing storage,
persistence, lifecycle, or schema boundaries.

## Task 1: Add RED persistence-identity and conflict coverage

**Files:**

- Modify: `Tests/Chat/test_console_video_message.py`

- [ ] Update `test_append_video_message_persists_namespaced_payload` to pass a
  fixed preallocated `message_id`, then assert the live ID, forwarded
  `create_message(message_id=...)`, and returned `persisted_message_id` are all
  identical.
- [ ] Add a real-SQLite explicit-ID conflict test that creates an existing row,
  attempts to persist a video message with the same ID, asserts the existing row
  is unchanged, and asserts the persistence exception propagates. Do not assert
  rollback or ephemeral-file cleanup; those are intentionally outside this task.
- [ ] Run only the new persistence tests and confirm RED because video messages
  currently pass `message_id=None`.

Command:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest \
  Tests/Chat/test_console_video_message.py::test_append_video_message_persists_namespaced_payload \
  Tests/Chat/test_console_video_message.py::test_video_message_explicit_id_conflict_propagates_without_overwrite \
  -q
```

Expected: the stable-ID assertion fails before production code changes; the
conflict test pins the existing database behavior.

## Task 2: Persist video messages with their preallocated ID

**Files:**

- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Test: `Tests/Chat/test_console_video_message.py`

- [ ] In `_persist_new_message`, extend the existing explicit-ID condition to
  include `message.video_metadata is not None`.
- [ ] Update the nearby comment so it describes generated image and video
  messages without broadening the rule to ordinary Console rows.
- [ ] Re-run the two Task 1 tests and confirm GREEN.

Command:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest \
  Tests/Chat/test_console_video_message.py::test_append_video_message_persists_namespaced_payload \
  Tests/Chat/test_console_video_message.py::test_video_message_explicit_id_conflict_propagates_without_overwrite \
  -q
```

Expected: both tests pass.

## Task 3: Add RED durable-resolution and true TTL-restart coverage

**Files:**

- Modify: `Tests/Chat/test_console_video_message.py`

- [ ] Extend the real-DB reload test to save bytes under the preallocated ID,
  reconstruct the persisted conversation into a fresh Console store, and prove
  the restored message has a fresh native `id` plus the original durable
  `persisted_message_id`.
- [ ] Construct a fresh TTL-configured `VideoStore` over the same root and call
  `enforce_retention(now=...)` with an in-TTL timestamp before asserting the file
  still exists. This must exercise the retention sweep, not only instantiate a
  TTL store.
- [ ] Drive `_build_video_card_specs` with the restored message and assert the
  native-ID-keyed card is `ready` and points to the retained file.
- [ ] Add `test_video_card_uses_persisted_id_for_storage_resolution`,
  in `test_console_video_message.py`.
- [ ] Add
  `test_handle_console_message_action_routes_video_play_with_persisted_storage_id`
  and
  `test_handle_console_message_action_routes_video_save_with_persisted_storage_id`
  in `test_console_video_actions.py`. These tests must enter through the real
  `handle_console_message_action()` button dispatcher, not call the screen-private
  methods directly. Spy on `VideoStore.resolve` and assert both operations use the
  durable ID while the button/caller still addresses the message by native ID.
  Avoid launching a player or writing outside `tmp_path`.
- [ ] Run only the affected test nodes and confirm RED because the three lookup
  sites currently resolve with `message.id`.

Command:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest \
  Tests/Chat/test_console_video_message.py::test_video_message_reload_round_trip_and_image_reader_isolation \
  Tests/Chat/test_console_video_message.py::test_video_card_uses_persisted_id_for_storage_resolution \
  Tests/Chat/test_console_video_actions.py::test_handle_console_message_action_routes_video_play_with_persisted_storage_id \
  Tests/Chat/test_console_video_actions.py::test_handle_console_message_action_routes_video_save_with_persisted_storage_id \
  -q
```

Expected: the restarted ready-card and action lookup assertions fail on the
fresh native ID.

## Task 4: Centralize durable-first video storage addressing

**Files:**

- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/UI/Console_Modules/message.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Test: the focused files changed in Task 3

- [ ] Add one small Console helper near `_build_video_card_specs` that returns
  `message.persisted_message_id or message.id`.
- [ ] Use that helper only for `VideoStore.resolve` in card construction,
  playback, and save-copy.
- [ ] Add named Play and Save-copy callables to `ConsoleMessageController`, expose
  them through the controller's existing dependency-property pattern, and wire
  them to the two screen-owned methods in `wire_console_controllers`. Keep the new
  constructor parameters optional only to preserve existing isolated controller
  test harnesses; production wiring and the real-dispatch tests must provide them,
  and a reached-but-unwired video branch must fail loudly.
- [ ] Keep `specs[message.id]`, `ConsoleVideoCardSpec.message_id`, and action
  lookup arguments unchanged so Console-native tree/action identity is not
  conflated with the filesystem key.
- [ ] Re-run the Task 3 nodes and confirm GREEN.

Expected: retained cards are ready after the sweep/restart, and all three
resolution boundaries use the durable key.

## Task 5: Mutation-check both load-bearing guards

**Files:**

- Temporarily modify and exactly restore:
  - `tldw_chatbook/Chat/console_chat_store.py`
  - `tldw_chatbook/UI/Screens/chat_screen.py`
  - `tldw_chatbook/UI/Console_Modules/message.py`
  - `tldw_chatbook/UI/Console_Modules/wiring.py`

- [ ] Temporarily remove the video branch from the explicit-ID condition, run
  the persistence identity test, and record the expected failure.
- [ ] Restore the exact production expression and rerun the identity test GREEN.
- [ ] Temporarily make the storage-key helper return only `message.id`, run the
  restart/card/action tests, and record the expected failures.
- [ ] Restore the exact helper and rerun those tests GREEN.
- [ ] Use `python -B` and remove only test-created bytecode if a stale import is
  observed; do not run broad suites to compensate for weak mutation evidence.

## Task 6: Run the final affected-file gates

**Files:** all files touched by Tasks 1–4.

- [ ] Run one fresh combined pytest command containing only the changed test
  files.
- [ ] Run full Ruff on the small touched production/test files. For the legacy
  `chat_screen.py`, run the repository's targeted error rules
  `E9,F63,F7,F82` if unrelated baseline findings prevent full-file Ruff.
- [ ] Run `py_compile` for the two touched production modules with bytecode
  output confined to a temporary directory.
- [ ] Run `git diff --check`, inspect the exact changed-file set, and scan the
  diff for credentials, local server identity, prompt text, generated media,
  and unintended workflow changes.
- [ ] Do not run the full repository suite, broad RuntimePolicy tests, live
  ComfyUI generation, or unrelated Console/UI tests.

Commands:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest \
  Tests/Chat/test_console_video_message.py \
  Tests/Chat/test_console_video_actions.py \
  -q

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check \
  tldw_chatbook/Chat/console_chat_store.py \
  tldw_chatbook/UI/Console_Modules/message.py \
  tldw_chatbook/UI/Console_Modules/wiring.py \
  Tests/Chat/test_console_video_message.py \
  Tests/Chat/test_console_video_actions.py

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check --select E9,F63,F7,F82 \
  tldw_chatbook/UI/Screens/chat_screen.py

PYTHONPYCACHEPREFIX="$(mktemp -d)" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m py_compile \
  tldw_chatbook/Chat/console_chat_store.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/UI/Console_Modules/message.py \
  tldw_chatbook/UI/Console_Modules/wiring.py

git diff --check
git status --short
```

Expected: all affected tests and static checks pass, with no unrelated files or
artifacts introduced.

## Task 7: Review, document, commit, and update the draft PR

**Files:**

- Modify: `backlog/tasks/task-3401.17 - Preserve-generated-video-identity-across-Console-lifecycle.md`
- Review: all changed production, test, design, and plan files

- [ ] Perform a correctness/spec review of the final diff, including the
  persisted/native identity distinction, explicit-ID conflict behavior, and TTL
  retention sweep evidence.
- [ ] Mark all acceptance criteria complete only after the final gates pass.
- [ ] Replace the task's Implementation Notes with concise final evidence,
  exact test/static scopes, ADR-044/design/plan links, touched files, mutation
  results, and any documented deviations.
- [ ] Set TASK-3401.17 to Done via Backlog CLI only after every DoD item is met.
- [ ] Commit in reviewable units, push the branch, and verify draft PR #1460
  reflects the commits and remains mergeable.
- [ ] Do not mark the PR ready or merge it until the remaining workstream tasks
  and requested review cycle are complete.
