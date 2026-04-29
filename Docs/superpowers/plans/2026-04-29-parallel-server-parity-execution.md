# Parallel Server Parity Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the next Chatbook/server parity tranche with maximum safe parallelism while preserving one active-server authority, no write sync, no MCP SDK, and no current UI redesign.

**Architecture:** Land a small shared schema slice first, then run provider migration, realtime/notifications, sync dry-run, domain-edge, and UX contract lanes in isolated worktrees. All lanes must consume the existing runtime-policy/server-context seams and import shared schema outputs rather than creating local replacements. Provider migration is split into service-only sub-batches plus one `app.py` integration branch.

**Tech Stack:** Python 3.11+, dataclasses/enums/protocols, existing `RuntimeServerContextProvider`, existing `TLDWAPIClient`, pytest, existing runtime-policy unsupported-capability reports.

---

## Source Documents

- `Docs/superpowers/specs/2026-04-28-remaining-server-parity-roadmap-design.md`
- `Docs/superpowers/specs/2026-04-29-parallel-server-parity-execution-design.md`
- `Docs/Development/server-client-provider-migration-audit.md`

## Parallel Execution Rules

- Lane 0 must land first.
- Lanes A, C, D, and E may write specs, tests, and skeletons before Lane 0 merges, but must not create local replacement schema stubs.
- Provider migration sub-batches B1a, B1b, and B1c may run in parallel because they have disjoint service ownership.
- Only the B1 integration branch edits `app.py` for migrated service wiring.
- UI tests are not broad blockers; only service-wiring UI tests matter.
- No lane may add new direct legacy `tldw_api` config client construction.
- No lane may enable write sync, replay workers, or remote mutation dispatch from sync.

## Worktree Setup

Run these commands from the `tldw_chatbook` repository root.

Create one worktree per lane:

```bash
git worktree add -b parity-shared-schemas .worktrees/parity-shared-schemas dev
git worktree add -b parity-provider-chat-character .worktrees/parity-provider-chat-character dev
git worktree add -b parity-provider-media-notes .worktrees/parity-provider-media-notes dev
git worktree add -b parity-provider-prompt-chatbook .worktrees/parity-provider-prompt-chatbook dev
git worktree add -b parity-provider-b1-integration .worktrees/parity-provider-b1-integration dev
git worktree add -b parity-events-foundation .worktrees/parity-events-foundation dev
git worktree add -b parity-sync-dry-run-substrate .worktrees/parity-sync-dry-run-substrate dev
git worktree add -b parity-domain-edge-contracts .worktrees/parity-domain-edge-contracts dev
git worktree add -b parity-ux-handoff-contracts .worktrees/parity-ux-handoff-contracts dev
```

Expected: each worktree checks out a dedicated lane branch. If any command fails because the branch/worktree exists, inspect with `git worktree list` and reuse the existing worktree only if it is clean or explicitly assigned. Do not point multiple worktrees at the same `dev` branch.

## File Structure

### Lane 0 Shared Schemas

- Create `tldw_chatbook/runtime_policy/server_parity_models.py`
  - Shared dataclasses/enums for event, notification, sync, and provider migration metadata.
- Modify `tldw_chatbook/runtime_policy/__init__.py`
  - Export the shared records.
- Create `tests/RuntimePolicy/test_server_parity_models.py`
  - Covers defaults, immutability/copy behavior, cursor scoping, dedupe keys, sync readiness failures, and provider migration status records.

### Lane B1a Provider Migration: Chat And Character

- Modify `tldw_chatbook/Chat/server_chat_conversation_service.py`
- Modify `tldw_chatbook/Chat/server_chat_loop_service.py`
- Modify `tldw_chatbook/Character_Chat/server_character_persona_service.py`
- Modify `tldw_chatbook/Character_Chat/server_chat_dictionary_service.py`
- Modify tests:
  - `tests/Chat/test_server_chat_conversation_service.py`
  - `tests/Chat/test_server_chat_loop_service.py`
  - `tests/Character_Chat/test_server_chat_dictionary_service.py`
  - `tests/Character_Chat/test_character_persona_scope_service.py` only if scope behavior changes.

### Lane B1b Provider Migration: Media And Notes

- Modify `tldw_chatbook/Media/server_media_reading_service.py`
- Modify `tldw_chatbook/Notes/server_notes_workspace_service.py`
- Modify tests:
  - `tests/Media/test_server_media_reading_service.py`
  - `tests/Notes/test_server_notes_workspace_service.py`

### Lane B1c Provider Migration: Prompt, Chatbook, Prompt Studio

- Modify `tldw_chatbook/Chatbooks/server_chatbook_service.py`
- Modify `tldw_chatbook/Prompt_Management/server_prompt_service.py`
- Modify `tldw_chatbook/Prompt_Management/prompt_scope_service.py`
- Modify `tldw_chatbook/Prompt_Studio_Interop/server_prompt_studio_service.py`
- Modify tests:
  - `tests/Chatbooks/test_server_chatbook_service.py`
  - `tests/Prompt_Management/test_prompt_chatbook_scope_service.py`
  - `tests/Prompt_Studio/test_prompt_studio_scope_service.py`

### Lane B1 Integration And Audit Guard

- Modify `tldw_chatbook/app.py`
  - Wire migrated services through `self.server_context_provider`.
- Modify `Docs/Development/server-client-provider-migration-audit.md`
  - Move landed services from backlog to migrated list and keep intentional compatibility entries explicit.
- Modify `tests/UI/test_screen_navigation.py`
  - Service wiring only.
- Create `tests/RuntimePolicy/test_server_client_provider_migration_audit.py`
  - Static audit guard for new direct/indirect legacy builders.

### Lane A Events And Notifications

- Create `tldw_chatbook/Notifications/event_cursor_store.py`
- Create `tldw_chatbook/Notifications/event_observer.py`
- Create `tldw_chatbook/Notifications/local_event_producer.py`
- Create `tldw_chatbook/Notifications/notification_presentation.py`
- Modify `tldw_chatbook/Notifications/__init__.py` if present, otherwise do not create package exports unless needed.
- Create tests:
  - `tests/Notifications/test_event_cursor_store.py`
  - `tests/Notifications/test_event_observer.py`
  - `tests/Notifications/test_local_event_producer.py`
  - `tests/Notifications/test_notification_presentation.py`

### Lane C Sync Dry-Run

- Create `tldw_chatbook/Sync_Interop/sync_readiness.py`
- Create `tldw_chatbook/Sync_Interop/sync_state.py`
- Create `tldw_chatbook/Sync_Interop/sync_mirror_report.py`
- Modify `tldw_chatbook/Sync_Interop/sync_scope_service.py` only for read-only readiness/report exposure.
- Create tests:
  - `tests/Sync_Interop/test_sync_readiness.py`
  - `tests/Sync_Interop/test_sync_state.py`
  - `tests/Sync_Interop/test_sync_mirror_report.py`
  - Modify `tests/Sync_Interop/test_sync_scope_service.py` only for unsupported-capability/readiness exposure.

### Lane D Domain Edge Contracts

- Create `Docs/Parity/domain-edge-contracts/chat.md`
- Create `Docs/Parity/domain-edge-contracts/media-reading.md`
- Create `Docs/Parity/domain-edge-contracts/notes-workspaces.md`
- Add or modify service tests only where a domain contract identifies a missing non-UI hard stop.

### Lane E UX Handoff Contracts

- Create `tldw_chatbook/UX_Interop/server_parity_contracts.py`
- Create `tldw_chatbook/UX_Interop/__init__.py`
- Create `tests/UX_Interop/test_server_parity_contracts.py`
- Do not modify current UI screens.

---

## Task 0: Commit Spec Fixes Before Branching

**Files:**

- Modified: `Docs/superpowers/specs/2026-04-28-remaining-server-parity-roadmap-design.md`
- Modified: `Docs/superpowers/specs/2026-04-29-parallel-server-parity-execution-design.md`
- Create: `Docs/superpowers/plans/2026-04-29-parallel-server-parity-execution.md`

- [ ] **Step 1: Verify docs diff**

Run:

```bash
git diff --check -- Docs/superpowers/specs/2026-04-28-remaining-server-parity-roadmap-design.md Docs/superpowers/specs/2026-04-29-parallel-server-parity-execution-design.md Docs/superpowers/plans/2026-04-29-parallel-server-parity-execution.md
```

Expected: no output.

- [ ] **Step 2: Commit docs**

Run:

```bash
git add Docs/superpowers/specs/2026-04-28-remaining-server-parity-roadmap-design.md Docs/superpowers/specs/2026-04-29-parallel-server-parity-execution-design.md Docs/superpowers/plans/2026-04-29-parallel-server-parity-execution.md
git commit -m "Plan parallel server parity execution"
```

Expected: one docs-only commit.

## Task 1: Lane 0 Shared Schema Slice

**Files:**

- Create: `tldw_chatbook/runtime_policy/server_parity_models.py`
- Modify: `tldw_chatbook/runtime_policy/__init__.py`
- Test: `tests/RuntimePolicy/test_server_parity_models.py`

- [ ] **Step 1: Write failing tests for event identity and cursor scoping**

Add tests that prove cursor keys are scoped by source, server profile, stream name, and stream instance:

```python
from tldw_chatbook.runtime_policy.server_parity_models import EventCursor, NormalizedEventRecord


def test_event_cursor_key_is_server_scoped():
    cursor = EventCursor(
        source_authority="server",
        server_profile_id="server-a",
        stream_name="notifications",
        stream_instance_id="default",
        cursor="abc",
    )

    assert cursor.storage_key() == "server:server-a:notifications:default"


def test_normalized_event_requires_server_profile_for_server_events():
    record = NormalizedEventRecord(
        source_authority="server",
        server_profile_id="server-a",
        stream_name="notifications",
        stream_instance_id="default",
        event_kind="notification.created",
        entity_ref={"type": "notification", "id": "n-1"},
        payload_hash="hash",
    )

    assert record.source_authority == "server"
```

- [ ] **Step 2: Write failing tests for sync readiness defaults**

```python
from tldw_chatbook.runtime_policy.server_parity_models import SyncReadinessReport


def test_sync_readiness_defaults_to_not_eligible():
    report = SyncReadinessReport(domain="chat")

    assert report.sync_eligible is False
    assert report.write_enabled is False
    assert "not_registered" in report.reason_codes
```

- [ ] **Step 3: Run failing tests**

Run:

```bash
python -m pytest tests/RuntimePolicy/test_server_parity_models.py -q
```

Expected: fails because the module does not exist.

- [ ] **Step 4: Implement shared models**

Implementation requirements:

- Use frozen dataclasses where practical.
- Keep fields JSON-serializable.
- Do not import UI modules.
- Do not perform network or database work.
- Include these records: `NormalizedEventRecord`, `EventCursor`, `EventDedupeKey`, `NotificationPresentationRecord`, `SyncIdentityMapEntry`, `SyncReadinessReport`, `ProviderMigrationStatus`.

- [ ] **Step 5: Export shared models**

Modify `tldw_chatbook/runtime_policy/__init__.py` to export the new records.

- [ ] **Step 6: Verify Lane 0**

Run:

```bash
python -m pytest tests/RuntimePolicy/test_server_parity_models.py -q
python -m compileall tldw_chatbook/runtime_policy
```

Expected: tests pass and compileall succeeds.

- [ ] **Step 7: Commit Lane 0**

```bash
git add tldw_chatbook/runtime_policy/server_parity_models.py tldw_chatbook/runtime_policy/__init__.py tests/RuntimePolicy/test_server_parity_models.py
git commit -m "Add shared server parity models"
```

## Task 2: Lane B1a Provider Migration For Chat And Character Services

**Files:**

- Modify: `tldw_chatbook/Chat/server_chat_conversation_service.py`
- Modify: `tldw_chatbook/Chat/server_chat_loop_service.py`
- Modify: `tldw_chatbook/Character_Chat/server_character_persona_service.py`
- Modify: `tldw_chatbook/Character_Chat/server_chat_dictionary_service.py`
- Test: `tests/Chat/test_server_chat_conversation_service.py`
- Test: `tests/Chat/test_server_chat_loop_service.py`
- Test: `tests/Character_Chat/test_server_chat_dictionary_service.py`

- [ ] **Step 1: Add provider-backed service tests**

For each service, add a fake provider test following the existing `ServerRuntimeService` pattern:

```python
import pytest


class FakeProvider:
    def __init__(self, client):
        self.client = client
        self.calls = 0

    def build_client(self):
        self.calls += 1
        return self.client


class FakeChatLoopClient:
    async def start_chat_loop_run(self, request_data):
        return {"run_id": "run-1", "messages": request_data.messages}


@pytest.mark.asyncio
async def test_server_chat_loop_service_can_use_context_provider_client():
    from tldw_chatbook.Chat.server_chat_loop_service import ServerChatLoopService

    client = FakeChatLoopClient()
    provider = FakeProvider(client)
    service = ServerChatLoopService.from_server_context_provider(provider)

    result = await service.start_run(messages=[{"role": "user", "content": "hello"}])

    assert result["run_id"] == "run-1"
    assert provider.calls == 1
```

- [ ] **Step 2: Add policy-denied no-build tests where the service has policy enforcement**

Use an exploding provider and assert denied policy raises before `build_client()` is called.

- [ ] **Step 3: Run tests and verify failure**

Run the focused files:

```bash
python -m pytest tests/Chat/test_server_chat_conversation_service.py tests/Chat/test_server_chat_loop_service.py tests/Character_Chat/test_server_chat_dictionary_service.py -q
```

Expected: fails because provider factories do not exist.

- [ ] **Step 4: Implement provider factory pattern**

For each service:

- Add optional `client_provider: Any | None = None`.
- Add `from_server_context_provider(provider, *, policy_enforcer=None)` where policy applies.
- Update `_require_client()` to prefer explicit `client`, then provider-built client, then raise.
- Preserve `from_config()` compatibility.
- Do not edit `app.py` in this task.

- [ ] **Step 5: Verify Lane B1a**

Run:

```bash
python -m pytest tests/Chat/test_server_chat_conversation_service.py tests/Chat/test_server_chat_loop_service.py tests/Character_Chat/test_server_chat_dictionary_service.py -q
python -m compileall tldw_chatbook/Chat tldw_chatbook/Character_Chat
```

Expected: tests pass and compileall succeeds.

- [ ] **Step 6: Commit Lane B1a**

```bash
git add tldw_chatbook/Chat/server_chat_conversation_service.py tldw_chatbook/Chat/server_chat_loop_service.py tldw_chatbook/Character_Chat/server_character_persona_service.py tldw_chatbook/Character_Chat/server_chat_dictionary_service.py tests/Chat/test_server_chat_conversation_service.py tests/Chat/test_server_chat_loop_service.py tests/Character_Chat/test_server_chat_dictionary_service.py
git commit -m "Migrate chat and character services to server context provider"
```

## Task 3: Lane B1b Provider Migration For Media And Notes

**Files:**

- Modify: `tldw_chatbook/Media/server_media_reading_service.py`
- Modify: `tldw_chatbook/Notes/server_notes_workspace_service.py`
- Test: `tests/Media/test_server_media_reading_service.py`
- Test: `tests/Notes/test_server_notes_workspace_service.py`

- [ ] **Step 1: Add provider-backed tests for media and notes**

Add tests for:

- `from_server_context_provider()` exists.
- provider is used for server operations.
- explicit direct client still takes precedence.
- `from_config()` remains compatible.
- denied policy does not build a provider client if these services enforce policy before dispatch.

- [ ] **Step 2: Run tests and verify failure**

```bash
python -m pytest tests/Media/test_server_media_reading_service.py tests/Notes/test_server_notes_workspace_service.py -q
```

Expected: fails because provider factories do not exist.

- [ ] **Step 3: Implement provider factory pattern**

Use the same pattern from Task 2. Do not edit `app.py`.

- [ ] **Step 4: Verify Lane B1b**

```bash
python -m pytest tests/Media/test_server_media_reading_service.py tests/Notes/test_server_notes_workspace_service.py -q
python -m compileall tldw_chatbook/Media tldw_chatbook/Notes
```

Expected: tests pass and compileall succeeds.

- [ ] **Step 5: Commit Lane B1b**

```bash
git add tldw_chatbook/Media/server_media_reading_service.py tldw_chatbook/Notes/server_notes_workspace_service.py tests/Media/test_server_media_reading_service.py tests/Notes/test_server_notes_workspace_service.py
git commit -m "Migrate media and notes services to server context provider"
```

## Task 4: Lane B1c Provider Migration For Prompt, Chatbook, And Prompt Studio

**Files:**

- Modify: `tldw_chatbook/Chatbooks/server_chatbook_service.py`
- Modify: `tldw_chatbook/Prompt_Management/server_prompt_service.py`
- Modify: `tldw_chatbook/Prompt_Management/prompt_scope_service.py`
- Modify: `tldw_chatbook/Prompt_Studio_Interop/server_prompt_studio_service.py`
- Test: `tests/Chatbooks/test_server_chatbook_service.py`
- Test: `tests/Prompt_Management/test_prompt_chatbook_scope_service.py`
- Test: `tests/Prompt_Studio/test_prompt_studio_scope_service.py`

- [ ] **Step 1: Add provider-backed tests**

Cover provider-backed construction and direct-client compatibility for chatbook, prompt, and prompt-studio server services.

- [ ] **Step 2: Add prompt scope provider injection tests**

Add or update a `PromptScopeService` test proving server prompt operations can use an injected provider-backed `ServerPromptService` or existing app-wired service without calling `ServerPromptService.from_config()` inside the operation path.

- [ ] **Step 3: Run tests and verify failure**

```bash
python -m pytest tests/Chatbooks/test_server_chatbook_service.py tests/Prompt_Management/test_prompt_chatbook_scope_service.py tests/Prompt_Studio/test_prompt_studio_scope_service.py -q
```

Expected: fails because provider factories and/or prompt scope provider wiring do not exist.

- [ ] **Step 4: Implement provider factory pattern**

Use the same pattern from Task 2. Preserve all existing direct-client and `from_config()` compatibility.

- [ ] **Step 5: Remove operation-time prompt `from_config()` construction**

Modify `prompt_scope_service.py` so server operations use an injected server service when available. If legacy fallback remains, keep it audited and isolated.

- [ ] **Step 6: Verify Lane B1c**

```bash
python -m pytest tests/Chatbooks/test_server_chatbook_service.py tests/Prompt_Management/test_prompt_chatbook_scope_service.py tests/Prompt_Studio/test_prompt_studio_scope_service.py -q
python -m compileall tldw_chatbook/Chatbooks tldw_chatbook/Prompt_Management tldw_chatbook/Prompt_Studio_Interop
```

Expected: tests pass and compileall succeeds.

- [ ] **Step 7: Commit Lane B1c**

```bash
git add tldw_chatbook/Chatbooks/server_chatbook_service.py tldw_chatbook/Prompt_Management/server_prompt_service.py tldw_chatbook/Prompt_Management/prompt_scope_service.py tldw_chatbook/Prompt_Studio_Interop/server_prompt_studio_service.py tests/Chatbooks/test_server_chatbook_service.py tests/Prompt_Management/test_prompt_chatbook_scope_service.py tests/Prompt_Studio/test_prompt_studio_scope_service.py
git commit -m "Migrate prompt and chatbook services to server context provider"
```

## Task 5: Lane B1 Integration And Migration Audit Guard

**Files:**

- Modify: `tldw_chatbook/app.py`
- Modify: `Docs/Development/server-client-provider-migration-audit.md`
- Modify: `tests/UI/test_screen_navigation.py`
- Create: `tests/RuntimePolicy/test_server_client_provider_migration_audit.py`

- [ ] **Step 1: Add app wiring assertions**

Extend service-wiring tests to assert migrated services use `app.server_context_provider`.

- [ ] **Step 2: Add baseline-diff audit guard test**

Create a static test that scans for direct and indirect client builders and compares matches against the migration audit. The test should fail only on new unlisted builders, not on existing audited compatibility factories.

- [ ] **Step 3: Run tests and verify failure before app wiring**

```bash
python -m pytest tests/UI/test_screen_navigation.py tests/RuntimePolicy/test_server_client_provider_migration_audit.py -q
```

Expected: app wiring assertions fail until `app.py` uses provider-backed factories.

- [ ] **Step 4: Wire migrated services through `self.server_context_provider`**

Modify only the migrated service initialization blocks in `app.py`.

- [ ] **Step 5: Update migration audit**

Move B1a/B1b/B1c services to migrated or partially migrated status. Keep compatibility factories listed if they remain in code.

- [ ] **Step 6: Verify Lane B1 integration**

```bash
python -m pytest tests/UI/test_screen_navigation.py tests/RuntimePolicy/test_server_client_provider_migration_audit.py -q
python -m compileall tldw_chatbook
```

Expected: service-wiring tests pass, audit guard passes, compileall succeeds.

- [ ] **Step 7: Commit Lane B1 integration**

```bash
git add tldw_chatbook/app.py Docs/Development/server-client-provider-migration-audit.md tests/UI/test_screen_navigation.py tests/RuntimePolicy/test_server_client_provider_migration_audit.py
git commit -m "Wire migrated services through server context provider"
```

## Task 6: Lane A Realtime And Notification Foundation

**Files:**

- Create: `tldw_chatbook/Notifications/event_cursor_store.py`
- Create: `tldw_chatbook/Notifications/event_observer.py`
- Create: `tldw_chatbook/Notifications/local_event_producer.py`
- Create: `tldw_chatbook/Notifications/notification_presentation.py`
- Test: `tests/Notifications/test_event_cursor_store.py`
- Test: `tests/Notifications/test_event_observer.py`
- Test: `tests/Notifications/test_local_event_producer.py`
- Test: `tests/Notifications/test_notification_presentation.py`

- [ ] **Step 1: Write event cursor store tests**

Cover per-server cursor isolation, acknowledgement-before-advance, stale cursor reset result, and bounded dedupe retention.

- [ ] **Step 2: Write observer reconnect and cursor-resume tests**

Use fake async event streams and fake sleep/backoff hooks. Cover:

- reconnect uses bounded exponential backoff.
- observer resumes from the last acknowledged cursor when a cursor exists.
- duplicate reconnect events are deduped.
- unacknowledged events do not advance the cursor.
- stale cursor errors produce a typed reset/requery result.
- unsupported cursor streams start without silently reusing another stream cursor.

- [ ] **Step 3: Write observer cancellation tests**

Use fake async event streams. Prove observer cancellation happens when the active server profile changes or credentials are cleared.

- [ ] **Step 4: Write local event producer tests**

Prove local producers emit source-scoped `NormalizedEventRecord` instances without a server profile ID and never write server-owned notification read/dismiss state.

- [ ] **Step 5: Write notification presentation tests**

Prove local notification delivery state is separate from server-owned read/dismiss/reminder state.

- [ ] **Step 6: Run tests and verify failure**

```bash
python -m pytest tests/Notifications/test_event_cursor_store.py tests/Notifications/test_event_observer.py tests/Notifications/test_local_event_producer.py tests/Notifications/test_notification_presentation.py -q
```

Expected: fails because event foundation modules do not exist.

- [ ] **Step 7: Implement minimal event foundation**

Requirements:

- Import `NormalizedEventRecord`, `EventCursor`, `EventDedupeKey`, and `NotificationPresentationRecord` from Lane 0.
- Do not implement broad server subscriptions unless a server contract exists.
- Keep observer transport pluggable: fake stream in tests, SSE adapter later.
- Cursor keys must include active server profile ID for server streams.
- Dedupe retention must be bounded.
- Backoff must be injectable in tests and must not sleep real time during unit tests.
- Observer interfaces must expose cursor resume hooks even when the initial implementation uses fake streams.
- Local event producer must be source-scoped and offline-capable.

- [ ] **Step 8: Verify Lane A**

```bash
python -m pytest tests/Notifications/test_event_cursor_store.py tests/Notifications/test_event_observer.py tests/Notifications/test_local_event_producer.py tests/Notifications/test_notification_presentation.py -q
python -m compileall tldw_chatbook/Notifications
```

Expected: tests pass and compileall succeeds.

- [ ] **Step 9: Commit Lane A**

```bash
git add tldw_chatbook/Notifications/event_cursor_store.py tldw_chatbook/Notifications/event_observer.py tldw_chatbook/Notifications/local_event_producer.py tldw_chatbook/Notifications/notification_presentation.py tests/Notifications/test_event_cursor_store.py tests/Notifications/test_event_observer.py tests/Notifications/test_local_event_producer.py tests/Notifications/test_notification_presentation.py
git commit -m "Add realtime event and notification foundation"
```

## Task 7: Lane C Sync Dry-Run Substrate

**Files:**

- Create: `tldw_chatbook/Sync_Interop/sync_readiness.py`
- Create: `tldw_chatbook/Sync_Interop/sync_state.py`
- Create: `tldw_chatbook/Sync_Interop/sync_mirror_report.py`
- Modify: `tldw_chatbook/Sync_Interop/sync_scope_service.py`
- Test: `tests/Sync_Interop/test_sync_readiness.py`
- Test: `tests/Sync_Interop/test_sync_state.py`
- Test: `tests/Sync_Interop/test_sync_mirror_report.py`
- Test: `tests/Sync_Interop/test_sync_scope_service.py`

- [ ] **Step 1: Write readiness default-false tests**

Every domain must default to `sync_eligible=False` and `write_enabled=False`.

- [ ] **Step 2: Write sync state model tests**

Cover:

- per-server sync profile state keyed by server profile ID.
- remote pull cursor model scoped by server profile, domain, and remote collection.
- identity map entries are included in sync state or mirror reports using `SyncIdentityMapEntry`.
- local outbox model shape with no replay worker or dispatch method.
- conflict strategy enum and default conflict policy.
- per-domain eligibility registry defaulting every unknown domain to not eligible.

- [ ] **Step 3: Write no-mutation dry-run tests**

Use an exploding fake server service with `create`, `update`, and `delete` methods. Dry-run reports must not call any mutation method.

- [ ] **Step 4: Write server-switch and workspace-boundary tests**

Readiness, cursor state, outbox shape, and mirror reports must be scoped by active server profile ID. Workspace-scoped records must preserve workspace ID and must not be reported as general user-space records.

- [ ] **Step 5: Run tests and verify failure**

```bash
python -m pytest tests/Sync_Interop/test_sync_readiness.py tests/Sync_Interop/test_sync_state.py tests/Sync_Interop/test_sync_mirror_report.py tests/Sync_Interop/test_sync_scope_service.py -q
```

Expected: fails because dry-run substrate modules do not exist.

- [ ] **Step 6: Implement dry-run substrate**

Requirements:

- Import `SyncIdentityMapEntry` and `SyncReadinessReport` from Lane 0.
- Define per-server sync profile state.
- Define remote pull cursor state.
- Define local outbox entry model shape only; do not add enqueue-to-replay or dispatch behavior.
- Define conflict strategy enum and default conflict policy.
- Define per-domain eligibility registry where unknown domains are not eligible.
- Do not create replay workers.
- Do not dispatch remote mutations.
- Do not write authoritative local mirror copies of server-owned records.
- Preserve workspace boundaries in readiness and mirror reports.
- Expose unsupported reports for unsyncable domains.

- [ ] **Step 7: Verify Lane C**

```bash
python -m pytest tests/Sync_Interop/test_sync_readiness.py tests/Sync_Interop/test_sync_state.py tests/Sync_Interop/test_sync_mirror_report.py tests/Sync_Interop/test_sync_scope_service.py -q
python -m compileall tldw_chatbook/Sync_Interop
```

Expected: tests pass and compileall succeeds.

- [ ] **Step 8: Commit Lane C**

```bash
git add tldw_chatbook/Sync_Interop/sync_readiness.py tldw_chatbook/Sync_Interop/sync_state.py tldw_chatbook/Sync_Interop/sync_mirror_report.py tldw_chatbook/Sync_Interop/sync_scope_service.py tests/Sync_Interop/test_sync_readiness.py tests/Sync_Interop/test_sync_state.py tests/Sync_Interop/test_sync_mirror_report.py tests/Sync_Interop/test_sync_scope_service.py
git commit -m "Add sync dry-run readiness substrate"
```

## Task 8: Lane D Domain Edge Contracts

**Files:**

- Create: `Docs/Parity/domain-edge-contracts/chat.md`
- Create: `Docs/Parity/domain-edge-contracts/media-reading.md`
- Create: `Docs/Parity/domain-edge-contracts/notes-workspaces.md`
- Modify tests only if contract review finds missing hard-stop behavior.

- [ ] **Step 1: Create chat edge contract**

Include:

- local chat-loop execution decision.
- streaming/persist handoff constraints.
- source-separated history rules.
- unsupported reason codes.
- required service tests.

- [ ] **Step 2: Create media/reading edge contract**

Include:

- ingest-job event/status normalization.
- read-it-later server aggregate-only behavior.
- chunk-level TTS adoption decision.
- saved-view unsupported cases.

- [ ] **Step 3: Create notes/workspaces edge contract**

Include:

- graph semantics.
- workspace-aware sync design boundaries.
- local/offline graph generation decision.
- cross-scope moves deferred.

- [ ] **Step 4: Add tests only for missing hard stops**

If a contract identifies a currently missing unsupported-capability hard stop, add a focused scope/service test. Do not change UI.

- [ ] **Step 5: Verify Lane D**

Run any added tests plus markdown sanity:

```bash
python -m pytest tests/Chat tests/Media tests/Notes -q
```

Expected: relevant backend tests pass. Ignore unrelated broad UI failures.

- [ ] **Step 6: Commit Lane D**

```bash
git add Docs/Parity/domain-edge-contracts
git add tests/Chat tests/Media tests/Notes
git commit -m "Document domain edge contracts"
```

## Task 9: Lane E UX Handoff Contracts

**Files:**

- Create: `tldw_chatbook/UX_Interop/__init__.py`
- Create: `tldw_chatbook/UX_Interop/server_parity_contracts.py`
- Test: `tests/UX_Interop/test_server_parity_contracts.py`

- [ ] **Step 1: Write contract fixture tests**

Cover fixtures for:

- local active source.
- server active source.
- unavailable server.
- unsupported action.
- workspace-scoped record.
- notification presentation.

- [ ] **Step 2: Run tests and verify failure**

```bash
python -m pytest tests/UX_Interop/test_server_parity_contracts.py -q
```

Expected: fails because UX contract module does not exist.

- [ ] **Step 3: Implement UX contract records**

Requirements:

- Use dataclasses or typed dictionaries.
- Consume runtime-policy state and unsupported-capability reports.
- Do not import current UI screens.
- Include contract ID/version, source owner, active source, active server profile ID, capability/action ID, unsupported reason code/message, server reachability/auth state, workspace scope ID, and fixture payload helpers.

- [ ] **Step 4: Verify Lane E**

```bash
python -m pytest tests/UX_Interop/test_server_parity_contracts.py -q
python -m compileall tldw_chatbook/UX_Interop
```

Expected: tests pass and compileall succeeds.

- [ ] **Step 5: Commit Lane E**

```bash
git add tldw_chatbook/UX_Interop tests/UX_Interop/test_server_parity_contracts.py
git commit -m "Add server parity UX handoff contracts"
```

## Task 10: Cross-Lane Integration Gate

**Files:**

- All files changed by Tasks 1-9.

- [ ] **Step 1: Merge in dependency order**

Merge order:

1. Lane 0 shared schemas.
2. Lane B1a/B1b/B1c service sub-batches.
3. Lane B1 integration.
4. Lane A events foundation.
5. Lane C sync dry-run.
6. Lane D domain edge contracts.
7. Lane E UX handoff contracts.

- [ ] **Step 2: Run focused backend suite**

```bash
python -m pytest \
  tests/RuntimePolicy \
  tests/Chat/test_server_chat_conversation_service.py \
  tests/Chat/test_server_chat_loop_service.py \
  tests/Character_Chat/test_server_chat_dictionary_service.py \
  tests/Media/test_server_media_reading_service.py \
  tests/Notes/test_server_notes_workspace_service.py \
  tests/Chatbooks/test_server_chatbook_service.py \
  tests/Prompt_Management/test_prompt_chatbook_scope_service.py \
  tests/Prompt_Studio/test_prompt_studio_scope_service.py \
  tests/Notifications \
  tests/Sync_Interop \
  tests/UX_Interop \
  -q
```

Expected: focused backend/service/contract tests pass.

- [ ] **Step 3: Run service-wiring UI tests only**

```bash
python -m pytest tests/UI/test_screen_navigation.py -q
```

Expected: service-wiring tests pass. Do not expand to broad UI tests in this sprint.

- [ ] **Step 4: Run compile and whitespace checks**

```bash
python -m compileall tldw_chatbook
git diff --check
```

Expected: compileall succeeds and `git diff --check` has no output.

- [ ] **Step 5: Final review checklist**

Verify:

- No new active-server authority exists.
- No secrets are stored in target-store JSON or app config.
- No new direct legacy config client builders were introduced.
- Server switching invalidates provider clients, event cursors, and sync cursors.
- Sync remains dry-run/read-only.
- Current UI screens were not redesigned.

- [ ] **Step 6: Commit or squash final integration**

If working in a single integration branch:

```bash
git status --short
git commit -m "Execute parallel server parity foundation"
```

Expected: final branch has a coherent integration commit or a reviewed stack of lane commits.
