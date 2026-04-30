# Event Replay And Local Parity Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden durable event replay metadata and start local parity for remote-only utilities with a translation pilot.

**Architecture:** Keep `EventStateRepository` as the durable event authority and add replay-window metadata to the same repository scope. Add a small local parity registry and promote only Translation to an adapter-backed local parity pilot while preserving all other remote-only contracts.

**Tech Stack:** Python 3.11+, SQLite, dataclasses, existing runtime policy contracts, pytest.

---

## File Structure

- Modify `tldw_chatbook/Notifications/event_state_repository.py`: add replay-window table, dataclass, repository methods, prune integration, cleanup integration.
- Modify `tldw_chatbook/Notifications/server_notification_events.py`: include replay status metadata in feed projection.
- Modify `tldw_chatbook/runtime_policy/domain_edge_contracts.py`: move Translation out of generated remote-only contracts and define it as `local_parity`.
- Modify `tldw_chatbook/Translation_Interop/translation_scope_service.py`: add optional local adapter routing for local mode.
- Modify `Tests/Notifications/test_event_state_repository.py`: repository replay-window red/green tests.
- Modify `Tests/Notifications/test_server_notification_events.py`: feed replay metadata red/green tests.
- Modify `Tests/Translation/test_translation_scope_service.py`: translation local adapter red/green tests.
- Modify `Tests/RuntimePolicy/test_domain_edge_contracts.py`: translation local parity contract tests.
- Modify `Docs/superpowers/trackers/backend-parity-phase-tracker.md`: update Event/notification and Translation rows.
- Modify `Docs/superpowers/handoffs/2026-04-30-backend-parity-ux-handoff.md`: document replay gap metadata and translation pilot state.

## Task 1: Durable Replay Window Metadata

- [ ] **Step 1: Write failing repository tests**

Add tests that prune a stream and expect a replay-window method to report the pruned cursor as `retention_gap`, the newest retained cursor as `available`, and cleanup to remove replay-window rows.

- [ ] **Step 2: Verify tests fail**

Run:

```bash
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest Tests/Notifications/test_event_state_repository.py -q
```

Expected: fail because replay-window methods do not exist.

- [ ] **Step 3: Implement minimal repository support**

Add `EventReplayWindow`, `event_replay_windows`, `get_replay_window(...)`, `get_replay_status(...)`, and update `prune_stream_state(...)` plus `clear_server_profile_state(...)`.

- [ ] **Step 4: Verify repository tests pass**

Run the same focused repository test command.

## Task 2: Feed Projection Replay Metadata

- [ ] **Step 1: Write failing feed test**

Add a server notification feed test that prunes older events, calls `build_server_notification_feed(..., after_cursor="old-cursor")`, and expects `replay.state == "retention_gap"` without adding another store.

- [ ] **Step 2: Verify test fails**

Run:

```bash
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest Tests/Notifications/test_server_notification_events.py -q
```

Expected: fail because feed does not accept or expose replay status.

- [ ] **Step 3: Implement feed metadata**

Add an optional `after_cursor` parameter and include repository replay status in the feed result.

- [ ] **Step 4: Verify feed tests pass**

Run the same feed test command.

## Task 3: Translation Local Parity Pilot

- [ ] **Step 1: Write failing translation tests**

Add tests for `TranslationScopeService(local_service=...)` where local mode calls the local adapter, rewrites backend to `local`, and reports no unsupported local capability. Preserve the existing no-adapter local rejection test.

- [ ] **Step 2: Verify tests fail**

Run:

```bash
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest Tests/Translation/test_translation_scope_service.py -q
```

Expected: fail because `local_service` is unsupported.

- [ ] **Step 3: Implement minimal local adapter routing**

Add `local_service` to the scope service constructor and route local mode through `local_service.translate_text(...)`. Do not change server mode policy behavior.

- [ ] **Step 4: Verify translation tests pass**

Run the same translation test command.

## Task 4: Domain Contract And Tracker Updates

- [ ] **Step 1: Write failing contract tests**

Assert Translation has `authority == "local_parity"`, exposes `("local", "server")`, and is not in `REMOTE_ONLY_DOMAIN_IDS`.

- [ ] **Step 2: Verify tests fail**

Run:

```bash
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest Tests/RuntimePolicy/test_domain_edge_contracts.py -q
```

Expected: fail because Translation is currently generated as remote-only.

- [ ] **Step 3: Implement contract update**

Move Translation to an explicit `DomainEdgeContract` with local parity authority. Keep all other remote-only utilities unchanged.

- [ ] **Step 4: Update docs**

Update tracker and handoff rows to show bounded replay hardening and Translation as adapter-backed local parity pilot.

- [ ] **Step 5: Verify focused suite**

Run:

```bash
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest \
  Tests/Notifications/test_event_state_repository.py \
  Tests/Notifications/test_server_notification_events.py \
  Tests/Translation/test_translation_scope_service.py \
  Tests/RuntimePolicy/test_domain_edge_contracts.py -q
git -C /Users/macbook-dev/Documents/GitHub/tldw_chatbook diff --check
```

Expected: all selected tests pass and diff check is clean.
