# Event Replay And Local Parity Hardening Design

Date: 2026-04-30

## Purpose

This tranche addresses two approved deferred items without reopening the broader backend parity phase:

- Harden durable server event replay beyond the current retained event rows.
- Start local parity for remote-only utilities with a registry and one low-risk pilot.

The work must preserve existing authority decisions. `EventStateRepository` remains the only durable event authority. `ClientNotificationsDB` remains the local inbox authority. Remote-only utilities stay server-owned unless a specific utility is promoted through an explicit local parity contract.

## Non-Goals

- No infinite local event history.
- No second event store or parallel notification inbox.
- No write sync, mutation outbox, or background mutation replay.
- No local CRUD for every remote-only utility in this tranche.
- No UI rebuild work.

## Durable Event Replay Hardening

The current repository stores durable events, dedupe rows, processed cursors, presented high-water cursors, observer status, and retention policy. That is enough for bounded replay, but after retention pruning the client cannot currently explain whether an older requested cursor is still locally replayable or has fallen behind the retained window.

Add replay-window metadata inside `EventStateRepository`, keyed by the same principal-scoped stream key:

- `source_authority`
- `server_profile_id`
- `authenticated_principal_id`
- `stream_name`
- `stream_instance_id`

The replay window records:

- earliest retained cursor and latest retained cursor when retained rows exist.
- last pruned cursor and pruned event count when retention deletes rows.
- updated timestamp.

Expose repository methods that answer whether a stream can be replayed locally from a requested cursor:

- `available`: requested cursor is absent or retained.
- `retention_gap`: requested cursor predates the retained window or was pruned.
- `empty`: no local event rows are retained and no pruning metadata exists.

Presentation builders may include this status as metadata, but they must not synthesize events or read from `ClientNotificationsDB`. If replay status is `retention_gap`, UX can show "older events require server refetch" rather than implying local history is complete.

## Local Parity Registry And Pilot

Remote-only utilities are individually tracked for UX handoff. To avoid another broad rewrite, add a local parity registry that describes each remote-only utility's local parity state:

- `remote_only`: server remains the only supported authority.
- `pilot`: a scoped local implementation exists behind an explicit adapter.
- `planned`: local parity may be added later but is not implemented.
- `blocked`: local parity is blocked by missing dependencies or product decision.

The first pilot is text translation. Translation is low risk because the scope service already has a single operation and a clear server/local mode boundary. The pilot should:

- Accept an optional `local_service` in `TranslationScopeService`.
- Route `mode="local"` to `local_service.translate_text(...)` only when that adapter is provided.
- Preserve the current local unsupported report when no local adapter is configured.
- Keep server policy enforcement and server routing unchanged.
- Mark translation as `local_parity` in the domain edge contract while keeping the other remote-only utilities unchanged.

## Tracker And Handoff Updates

After implementation:

- Update the tracker Event/notification gate remaining work to state that replay-window metadata exists and durable replay is bounded by retention.
- Update Translation from `remote-only` to local parity pilot.
- Keep the remote-only rollup and every other remote-only utility unchanged.
- Update the UX handoff packet so UX knows translation local mode is adapter-dependent and replay gaps can be displayed.

## Test Strategy

Use TDD for behavior changes:

- Add repository tests proving prune records replay-window metadata and reports `retention_gap` for pruned cursors.
- Add feed projection tests proving replay metadata is surfaced without a parallel local store.
- Add translation scope tests proving local mode routes only when a local adapter exists and still rejects when not configured.
- Add domain contract tests proving translation is no longer counted as remote-only and exposes local/server selector states.

Focused verification:

```bash
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest \
  Tests/Notifications/test_event_state_repository.py \
  Tests/Notifications/test_server_notification_events.py \
  Tests/Translation/test_translation_scope_service.py \
  Tests/RuntimePolicy/test_domain_edge_contracts.py -q
```
