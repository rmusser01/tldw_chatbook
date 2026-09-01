# Personal Context 04 — Sync and Multi-device Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replicate identical canonical Personal Context records among one tldw_server home peer and multiple Chatbook devices, including first-link reconciliation, explicit conflict review, privacy cleanup acknowledgments, device lifecycle, and resurrection-proof global purge.

**Architecture:** Four canonical object domains and one content-free purge domain extend Sync V2 under `server_trusted_v1`. Each runtime materializes accepted whole objects through its existing `PersonalContextService`. Chatbook records a same-database encrypted outbox with each local mutation, then a dispatcher crosses into Sync state idempotently. First link is a reviewed reconciliation transaction. Privacy reductions and purge converge only after explicit device acknowledgments.

**Tech Stack:** Python 3.11+, Shared Profile Core, Chatbook Sync Interop, tldw_server Sync V2, SQLite, FastAPI, Textual 8.x, pytest, Hypothesis.

**Spec:** `Docs/superpowers/specs/2026-08-28-unified-personal-context-profile-design.md`

## ADR check

```text
ADR required: yes
ADR path: backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md
Reason: This plan defines replication domains, conflict and cleanup policy,
device expiry/rebootstrap, first-link ownership, and global purge convergence.
```

## Global Constraints

- Complete Plans 01–03 first. Both runtimes must pin the same supported Shared
  Core schema range and pass the same fixtures before linking is offered.
- The V1 topology is one home tldw_server plus any number of Chatbook devices.
  Do not add server federation or multiple home servers.
- Bind authority to the stable server authority ID, authenticated server user
  ID, and peer-local random Chatbook device ID. URLs, labels, routing IDs, and
  credentials are never profile identity.
- Replicate canonical whole objects, not UI projections, legacy API shapes,
  prompts, embeddings, or semantic-memory rows.
- Exact domains are `personal_context.manifest`, `personal_context.scope`,
  `personal_context.record`, `personal_context.proposal`, and
  `personal_context.purge`.
- Link remains disabled unless the server advertises all five domains,
  compatible schema bounds, `server_trusted_v1`, HMAC-SHA-256 integrity tags,
  privacy-cleanup acknowledgments, purge generations, and required quotas.
- `device_only` objects never enter a Sync envelope. Runtime agent authority
  never synchronizes. `user_only` records may sync when marked `syncable`, but
  must be removed from all agent-facing artifacts before convergence is acked.
- Workspace and global records with the same semantic key are an intentional
  overlay. Only same-scope collisions become Sync review objects.
- Sync conflicts remain separate from profile proposals in storage, APIs, and
  Settings. Agents cannot inspect or resolve them.
- Cross-database work uses an outbox/journal. Never claim one transaction spans
  the profile DB and Sync state DB.
- Purge generations are monotonic and content-free. A device below the current
  generation cannot upload profile objects and must purge/rebootstrap first.

---

### Task 1: Negotiate Personal Context Sync capabilities on both peers

**Files (tldw_server):**
- Modify: `tldw_Server_API/app/core/Sync/v2/models.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/factory.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`
- Modify: `tldw_Server_API/tests/Sync/test_sync_v2_models.py`
- Modify: `tldw_Server_API/tests/Sync/test_sync_v2_service.py`
- Modify: `tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py`

**Files (Chatbook):**
- Modify: `tldw_chatbook/tldw_api/sync_schemas.py`
- Modify: `tldw_chatbook/tldw_api/client.py`
- Modify: `tldw_chatbook/Sync_Interop/server_sync_service.py`
- Modify: `tldw_chatbook/Sync_Interop/sync_readiness.py`
- Test: `Tests/Sync_Interop/test_personal_context_capabilities.py`
- Modify: `Tests/Sync_Interop/test_sync_readiness.py`

**Interfaces:**
- Produces five new `SyncDomain` values and `upsert`/`tombstone` operations.
- Produces capability object:

```json
{
  "personal_context": {
    "min_schema_version": 1,
    "max_schema_version": 1,
    "integrity_algorithm": "hmac-sha256-v1",
    "integrity_key_distribution": "wrapped-bootstrap-v1",
    "privacy_cleanup_ack": "personal-context-cleanup-v1",
    "purge_generation": "personal-context-purge-v1",
    "max_record_bytes": 16384,
    "max_search_results": 20,
    "max_proposals_per_turn": 5,
    "max_proposals_per_session": 25,
    "max_unresolved_proposals": 200
  }
}
```

- [ ] **Step 1: Write failing negotiation tests**

```python
def test_personal_context_link_requires_complete_capability_set(readiness):
    capabilities = complete_capabilities()
    capabilities["supported_domains"].remove("personal_context.purge")
    report = readiness.personal_context(capabilities)
    assert report.write_enabled is False
    assert report.blockers == ("personal_context_domain_missing:personal_context.purge",)


def test_schema_range_must_overlap_local_core(readiness):
    capabilities = complete_capabilities(min_schema_version=2, max_schema_version=3)
    assert readiness.personal_context(capabilities).blockers == (
        "personal_context_schema_incompatible",
    )
```

On the server, test domain literals, operation maps, capabilities endpoint,
server-trusted policy, configured/missing profile master key, and quota values.
On Chatbook, test missing, malformed, downgraded, partially implemented, unknown
future fields, incompatible HMAC algorithm, and absent cleanup/purge contracts.

- [ ] **Step 2: Run tests to verify they fail**

Run in each repository:

```bash
pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py -v
pytest Tests/Sync_Interop/test_personal_context_capabilities.py \
  Tests/Sync_Interop/test_sync_readiness.py -v
```

Expected: the domains and capability object are absent.

- [ ] **Step 3: Extend the server protocol contract**

Add a `PERSONAL_CONTEXT_SYNC_DOMAINS` constant and include it in supported
domains/operations. Add a typed `personal_context` capability field rather than
hiding required gates in generic compatibility flags. Derive its availability
from the pinned Shared Core version and valid server key configuration. If the
key is absent, advertise the domains but mark the feature unavailable with a
stable blocker; never claim write readiness.

- [ ] **Step 4: Add strict Chatbook parsing and readiness**

```python
@dataclass(frozen=True, slots=True)
class PersonalContextSyncReadiness:
    read_enabled: bool
    write_enabled: bool
    blockers: tuple[str, ...]
    negotiated_schema_version: int | None
```

Parse aliases only where Sync V2 already supports them. Require an inclusive
schema intersection and select the highest mutually supported version. Keep
existing domains unaffected when Personal Context is unavailable.

- [ ] **Step 5: Run tests and commit in each repository**

Stage only the files listed above. Use commits:

```bash
git commit -m "feat: advertise personal context sync capabilities"
git commit -m "feat: negotiate personal context sync capabilities"
```

---

### Task 2: Add canonical domain adapters, materializers, and Chatbook outbox

**Files (tldw_server):**
- Create: `tldw_Server_API/app/core/Sync/v2/domain_adapters/personal_context.py`
- Create: `tldw_Server_API/app/core/Sync/v2/materializers/personal_context.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/domain_adapters/__init__.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/materializers/__init__.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/factory.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_adapter.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_materializer.py`

**Files (Chatbook):**
- Create: `tldw_chatbook/Personal_Context/sync_outbox.py`
- Create: `tldw_chatbook/Sync_Interop/personal_context_adapter.py`
- Create: `tldw_chatbook/Sync_Interop/personal_context_dispatcher.py`
- Modify: `tldw_chatbook/Personal_Context/repository.py`
- Modify: `tldw_chatbook/Personal_Context/service.py`
- Modify: `tldw_chatbook/Sync_Interop/envelope_builder.py`
- Modify: `tldw_chatbook/Sync_Interop/envelope_applier.py`
- Modify: `tldw_chatbook/Sync_Interop/local_first_sync_service.py`
- Test: `Tests/Personal_Context/test_profile_sync_outbox.py`
- Test: `Tests/Sync_Interop/test_personal_context_adapter.py`
- Test: `Tests/Sync_Interop/test_personal_context_dispatcher.py`

**Interfaces:**
- Produces `PersonalContextDomainAdapter`,
  `PersonalContextMaterializer`, `ProfileSyncOutbox`, and
  `PersonalContextOutboxDispatcher`.
- Envelope `object_id` is the canonical profile object ID; payload is one exact
  Shared Core whole object plus its keyed integrity tag.
- Produces a dataset-scoped 32-byte Personal Context integrity key through the
  existing Sync key-record enrollment/rewrap path and authenticated wrapped
  bootstrap response. The home server owns it; it is distinct from both peers'
  envelope-encryption keys and identified by `integrity_key_id`.

- [ ] **Step 1: Write failing whole-object and atomic-outbox tests**

```python
def test_local_record_and_outbox_commit_together(service, repository, record):
    repository.fail_after_object_write = True
    with pytest.raises(InjectedFailure):
        service.create_record(record)
    assert repository.get_record_or_none(record.record_id) is None
    assert repository.list_profile_outbox() == ()


def test_device_only_record_never_creates_outbox(service, repository, private_record):
    service.create_record(private_record)
    assert repository.get_record(private_record.record_id) == private_record
    assert repository.list_profile_outbox() == ()
```

Add tests for all five domains, upsert/tombstone, canonical byte equality,
integrity failure, unsupported schema, wrong profile/scope, idempotent retry,
base revision/hash conflict, server authorization before decrypt, dispatcher
crash between databases, poisoned item quarantine, and no raw body in Sync logs.

- [ ] **Step 2: Run tests to verify they fail**

Run the new server and Chatbook tests from the file lists. Expected: adapters,
materializers, and outbox do not exist.

- [ ] **Step 3: Implement the same-database encrypted outbox**

```python
def commit_record_and_enqueue(
    self,
    record: ProfileRecord,
    expected_version_id: str | None,
) -> ProfileRecord:
    with self._repository.transaction(immediate=True) as cursor:
        stored = self._repository.commit_record_version(
            record,
            expected_version_id=expected_version_id,
            cursor=cursor,
        )
        if stored.controls.sync_mode == SyncMode.SYNCABLE:
            self._outbox.enqueue_whole_object(stored, cursor=cursor)
        return stored
```

Encrypt outbox payloads with a dedicated DEK and retain only opaque routing,
version, timing, size, and retry metadata in clear columns. The dispatcher
idempotently copies them into Sync state/envelopes, records the Sync envelope
ID back in the profile DB, and then crypto-shreds acknowledged payloads. A crash
at any point replays without duplicate canonical versions.

- [ ] **Step 4: Implement server and client adapters/materializers**

Adapters validate domain/object relationship, 16 KiB limit, schema range,
HMAC-SHA-256 over canonical bytes with the enrolled dataset integrity key,
purge generation, and base lineage. Materializers call the
runtime's `PersonalContextService` with an authenticated sync actor and expected
version. They never write object tables. Tombstones are content-free Shared Core
deleted records or the content-free purge barrier, not arbitrary empty dicts.

- [ ] **Step 5: Run tests and commit in each repository**

Use commits:

```bash
git commit -m "feat: materialize personal context sync domains"
git commit -m "feat: dispatch personal context sync outbox"
```

---

### Task 3: Implement reviewed first-link reconciliation and scope mapping

**Files (Chatbook):**
- Create: `tldw_chatbook/Personal_Context/link_service.py`
- Create: `tldw_chatbook/Personal_Context/reconciliation.py`
- Create: `tldw_chatbook/Widgets/Settings_Widgets/personal_context_link_modal.py`
- Modify: `tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py`
- Modify: `tldw_chatbook/Sync_Interop/server_sync_service.py`
- Modify: `tldw_chatbook/Sync_Interop/sync_state_repository.py`
- Test: `Tests/Personal_Context/test_profile_reconciliation.py`
- Test: `Tests/Sync_Interop/test_personal_context_first_link.py`
- Test: `Tests/UI/test_personal_context_link_modal.py`

**Files (tldw_server):**
- Modify: `tldw_Server_API/app/core/Sync/v2/profile.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_bootstrap.py`

**Interfaces:**
- Produces:
  - `PersonalContextLinkService.plan(server_profile) -> ReconciliationPlan`
  - `apply(plan_id, decisions) -> LinkReceipt`
  - durable local-to-canonical profile and workspace-scope mapping
  - provisional-profile freeze/journal during reviewed link

- [ ] **Step 1: Write failing first-link tests**

```python
def test_first_link_preserves_same_record_ids_on_both_peers(link_harness):
    local = link_harness.local_record("record-local")
    receipt = link_harness.link_and_accept_all()
    assert receipt.local_profile_id == receipt.server_profile_id
    assert link_harness.server_record(local.record_id).record_id == local.record_id


def test_same_scope_different_ids_same_key_requires_review(link_harness):
    link_harness.local_preference("local-id", "response.detail", "concise")
    link_harness.server_preference("server-id", "response.detail", "detailed")
    plan = link_harness.plan()
    assert plan.key_collisions[0].record_ids == ("local-id", "server-id")
    assert link_harness.agent_context_contains_neither(plan.key_collisions[0])
```

Cover empty/empty, local-only, server-only, identical object/version, diverged
same ID, same-scope same key/different IDs, global/workspace same key overlay,
user-only/private records, proposals, unmapped workspace, cancel/retry, link
interruption, server profile purge generation, provisional-integrity-key
replacement, and concurrent local mutation.

- [ ] **Step 2: Run tests to verify they fail**

Run the three Chatbook tests and server bootstrap test. Expected: there is no
Personal Context reconciliation handshake.

- [ ] **Step 3: Build a read-only reconciliation plan**

Fetch the server manifest/scopes/object heads under one bootstrap cursor. Freeze
new syncable profile mutations locally while allowing read-only use. Compare
exact profile IDs, object IDs, version lineage, canonical hashes, scope IDs, and
structured semantic keys. Do not fuzzy-match content or infer that workspace
labels identify the same scope.

The plan groups exact matches, unilateral additions, same-ID version conflicts,
same-scope key collisions, possible private duplicates, workspace mappings,
and purge-generation blockers. It stores encrypted hashes/references, not a
second plaintext copy.

- [ ] **Step 4: Apply reviewed decisions atomically per database**

The user maps each local workspace to an existing canonical scope or creates a
new random canonical scope. Apply local canonical replacements and an encrypted
outbox journal in one profile-DB transaction; then push idempotently. Server
materializes under its own transaction. Record a link receipt only after pull
confirms the same profile ID, scope map, object IDs/versions, and cursor. Unfreeze
writes and replay journaled user edits afterward. The authenticated wrapped
bootstrap replaces the standalone provisional integrity key with the
server-owned Sync integrity key, then recomputes every Personal Context tag in
a required versioned full integrity rebaseline before normal push/pull begins.

- [ ] **Step 5: Run tests and commit in each repository**

Use commits:

```bash
git commit -m "feat: bootstrap personal context canonical profile"
git commit -m "feat: reconcile personal context on first link"
```

---

### Task 4: Add Personal Context conflict review and deterministic resolution

**Files (Chatbook):**
- Create: `tldw_chatbook/Personal_Context/conflict_service.py`
- Create: `tldw_chatbook/Widgets/Settings_Widgets/personal_context_conflict_modal.py`
- Modify: `tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py`
- Modify: `tldw_chatbook/Sync_Interop/envelope_applier.py`
- Test: `Tests/Personal_Context/test_profile_conflict_service.py`
- Test: `Tests/UI/test_personal_context_conflict_modal.py`
- Test: `Tests/Sync_Interop/test_personal_context_conflicts.py`

**Files (tldw_server):**
- Create: `tldw_Server_API/app/core/Sync/v2/personal_context_conflicts.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_conflicts.py`

**Interfaces:**
- Produces Sync review kinds `version_conflict`, `key_collision`,
  `privacy_transition_blocked`, and `purge_generation_mismatch`.
- Produces user decisions `keep_left`, `keep_right`, `edit_replacement`,
  `keep_both_without_key`, and `dismiss_after_remote_deleted` where valid.

- [ ] **Step 1: Write failing conflict tests**

```python
def test_unacknowledged_key_collision_omits_both_from_context(conflicts, runtime):
    conflict = conflicts.create_key_collision("scope-1", "record-a", "record-b")
    assert runtime.build_snapshot(("scope-1",)).record_ids == ()
    assert conflict.status == "unresolved"


def test_resolution_creates_versions_not_in_place_mutation(conflicts, repository):
    receipt = conflicts.resolve("conflict-1", decision="keep_left", actor="user")
    assert receipt.winner.parent_version_id is not None
    assert repository.get_record(receipt.loser_record_id).state == RecordState.DELETED
```

Cover stale resolution, duplicate resolution request, remote tombstone, archived
record, proposal target conflict, workspace overlay non-conflict, last mutually
acknowledged occupant fallback, offline second resolution, and cross-user access.

- [ ] **Step 2: Run tests to verify they fail**

Run the listed Chatbook/server conflict tests. Expected: generic Sync conflicts
cannot express structured-key review behavior.

- [ ] **Step 3: Implement separate encrypted conflict projections**

Store conflict bodies encrypted in the Sync state DB with canonical object
references, last mutually acknowledged versions, and source envelope IDs.
Expose them in My Profile under a Sync conflicts section distinct from agent
proposals. The runtime may use the last mutually acknowledged occupant; without
one it omits all occupants. Agents and interview providers cannot search or
resolve these objects.

- [ ] **Step 4: Resolve through canonical services and Sync**

Resolution creates new canonical record versions/tombstones using expected
versions, queues them normally, then marks the Sync conflict resolved only after
acknowledgment. A concurrent change returns a fresh review object. Never mutate
canonical rows or Sync conflict bodies in place.

- [ ] **Step 5: Run tests and commit in each repository**

Use commits:

```bash
git commit -m "feat: classify personal context sync conflicts"
git commit -m "feat: review personal context sync conflicts"
```

---

### Task 5: Enforce privacy cleanup acknowledgments and restrictive conversion

**Files (Chatbook):**
- Create: `tldw_chatbook/Personal_Context/privacy_cleanup.py`
- Modify: `tldw_chatbook/Personal_Context/service.py`
- Modify: `tldw_chatbook/Personal_Context/context_service.py`
- Modify: `tldw_chatbook/Sync_Interop/sync_state_repository.py`
- Modify: `tldw_chatbook/Sync_Interop/server_sync_service.py`
- Test: `Tests/Personal_Context/test_profile_privacy_cleanup.py`
- Test: `Tests/Sync_Interop/test_personal_context_cleanup_ack.py`
- Test: `Tests/Sync_Interop/test_personal_context_device_only_conversion.py`

**Files (tldw_server):**
- Create: `tldw_Server_API/app/core/Personalization/personal_context_cleanup.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/models.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/store.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_cleanup_ack.py`

**Interfaces:**
- Produces `PersonalContextCleanupAck(record_id, version_id,
  cleanup_generation, device_id, completed_at)`.
- Produces `convert_to_device_only(record_id, expected_version_id) ->
  RestrictiveConversionReceipt`.

- [ ] **Step 1: Write failing cleanup/conversion tests**

```python
def test_user_only_ack_waits_for_every_agent_facing_owner(cleanup_harness):
    version = cleanup_harness.change_visibility("record-1", "user_only")
    cleanup_harness.fail_owner("context_cache")
    assert cleanup_harness.ack_for(version) is None
    assert cleanup_harness.profile_status() == "sync_attention"


def test_device_only_conversion_uses_new_private_id(conversion_harness):
    old = conversion_harness.syncable_record("shared-id")
    receipt = conversion_harness.convert(old.record_id)
    assert receipt.shared_tombstone.record_id == "shared-id"
    assert receipt.private_record.record_id != "shared-id"
    assert receipt.private_record.controls.sync_mode == SyncMode.DEVICE_ONLY
```

Inventory context caches, preview snapshots, tool result caches, proposal input,
interview adaptive input, diagnostics, exports, logs, crash inputs, outbox, and
run transcripts. Test failure/restart at each cleanup owner, remote order
inversion, repeated ack, revoked device, offline conversion, and canary absence.

- [ ] **Step 2: Run tests to verify they fail**

Run all five listed tests. Expected: no cleanup generation or restrictive
conversion journal exists.

- [ ] **Step 3: Implement cleanup-before-ack**

When a record becomes `user_only`, immediately exclude the new version from all
new agent paths, persist a cleanup job, and delete/invalidate every durable
agent-facing derivative. Only then send the keyed cleanup acknowledgment. The
server retains the privacy transition as pending until every active device has
acked; status remains Sync attention and stale devices may not materialize the
old visibility.

- [ ] **Step 4: Implement shared-to-private conversion**

In one Chatbook profile transaction, create a content-free tombstone for the
shared record ID, create a new private record ID with `device_only`, enqueue only
the tombstone, and store an encrypted linkage receipt for user Undo/review.
Never reuse the shared ID for private content. Other devices observe deletion,
not the new private value.

- [ ] **Step 5: Run tests and commit in each repository**

Use commits:

```bash
git commit -m "feat: track personal context privacy cleanup acks"
git commit -m "feat: converge personal context privacy reductions"
```

---

### Task 6: Add device expiry, local removal, and global purge barriers

**Files (Chatbook):**
- Create: `tldw_chatbook/Personal_Context/purge_service.py`
- Modify: `tldw_chatbook/Personal_Context/service.py`
- Modify: `tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py`
- Modify: `tldw_chatbook/Sync_Interop/local_first_sync_service.py`
- Test: `Tests/Personal_Context/test_profile_local_removal.py`
- Test: `Tests/Sync_Interop/test_personal_context_global_purge.py`
- Test: `Tests/UI/test_personal_context_delete_actions.py`

**Files (tldw_server):**
- Create: `tldw_Server_API/app/core/Personalization/personal_context_purge.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/models.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/store.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/personal_context.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_purge.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_device_expiry.py`

**Interfaces:**
- Produces monotonic `purge_generation`, content-free purge envelopes,
  `remove_local_copy()`, `delete_everywhere()`, `expire_device()`, and
  `rebootstrap_after_purge()`.

- [ ] **Step 1: Write failing purge-generation tests**

```python
def test_offline_pre_purge_device_cannot_resurrect_records(multidevice):
    multidevice.a.go_offline()
    multidevice.a.create_record("offline-record")
    generation = multidevice.b.delete_everywhere()
    multidevice.a.go_online()
    assert multidevice.a.push_result().code == "purge_generation_rebootstrap_required"
    assert multidevice.server.record_count() == 0
    assert multidevice.a.local_generation == generation


def test_remove_local_copy_does_not_delete_server(multidevice):
    record = multidevice.a.create_and_sync_record()
    multidevice.a.remove_local_copy()
    assert multidevice.server.get_record(record.record_id) is not None
    assert multidevice.a.profile_exists() is False
```

Cover purge while another device is online/offline, initiating-device immediate
write freeze, failed network request, repeated purge idempotency, device expiry,
revoked device, stale envelope replay, stale restore/recovery import, tombstone
retention, key crypto-shredding, local-only profile confirmation, and relink.

- [ ] **Step 2: Run tests to verify they fail**

Run the five listed tests. Expected: no content-free generation barrier or
distinct local/everywhere deletion actions exist.

- [ ] **Step 3: Implement device lifecycle and rebootstrap rule**

Extend device status with `expired`. Expiry is an explicit authenticated user or
operator action after displaying last-seen time; there is no silent automatic
expiry in V1. Expired/revoked devices no longer block cleanup/purge retention.
If they return, they must register as a new device and perform a full canonical
bootstrap before writes. Preserve an audit receipt without profile content.

- [ ] **Step 4: Implement the global purge barrier**

The initiating device first persists `purge_pending` and freezes profile writes.
The server atomically increments its per-user generation, crypto-shreds
canonical profile/proposal/conflict/migration-recovery keys, removes materialized
content, and emits only `{profile_id, purge_generation, initiated_at}` in the
purge domain. Every push/pull includes the device generation. Lower-generation
devices purge local owners, ack the barrier, and rebootstrap; their queued older
objects are permanently rejected.

Retain object tombstones and the generation barrier until every active device
acks or is explicitly expired. Purge can complete even if an expired device
never returns.

- [ ] **Step 5: Implement the two Settings deletion actions**

`Remove local copy` is available only for linked profiles or after explicit
confirmation that the only copy will be lost. It shreds local keys/data/outbox
without a server tombstone. `Delete everywhere` shows active/offline devices,
requires typed confirmation, invokes purge, and reports pending acknowledgments.
Neither action is exposed to agents.

- [ ] **Step 6: Run tests and commit in each repository**

Use commits:

```bash
git commit -m "feat: enforce personal context purge generation"
git commit -m "feat: add personal context local and global deletion"
```

---

### Task 7: Prove two-device convergence and stage the rollout

**Files (Chatbook):**
- Create: `Tests/Integration/test_personal_context_multidevice.py`
- Create: `Tests/Integration/test_personal_context_first_link_live.py`
- Modify: `Docs/User_Guide/settings/personal-context-profile.md`
- Create: `Docs/User_Guide/settings/personal-context-sync.md`
- Modify: `tldw_chatbook/config.py`

**Files (tldw_server):**
- Create: `tldw_Server_API/tests/Integration/test_personal_context_sync_live.py`
- Modify: `Docs/Operations/personal-context-profile.md`
- Modify: `Docs/API/sync-v2.md`
- Modify: `tldw_Server_API/app/core/feature_flags.py`

**Interfaces:**
- Produces development feature gates on both peers and release evidence for one
  server plus two isolated Chatbook profiles.

- [ ] **Step 1: Build the real three-peer harness**

Use two independent Chatbook config/data/keyring namespaces and one scratch
tldw_server user/database/master key. Do not share profile DB files. Drive the
real HTTP Sync V2 client, server endpoints, encrypted repositories, outbox,
materializers, and Settings services; mocks may control clocks/failpoints only.

- [ ] **Step 2: Verify canonical convergence**

Exercise and assert:

- A record created on Chatbook A reaches server and B with the same profile,
  scope, record, version, payload, controls, and provenance IDs.
- A server API edit reaches both Chatbooks as the same next version.
- Global/workspace overlay selects the workspace value without conflict.
- Agent proposal remains a proposal everywhere until user acceptance.
- Offline same-ID edit and same-key/different-ID creation become distinct Sync
  review objects, not last-write-wins.
- `user_only` waits for cleanup acknowledgment and disappears from every agent
  path before convergence.
- shared-to-device-only conversion deletes the shared ID elsewhere and keeps a
  new private ID only on the initiating Chatbook.
- Global purge rejects queued old-generation data and leaves no content canary
  in any default durable owner.

- [ ] **Step 3: Run targeted integration suites**

Run:

```bash
pytest Tests/Integration/test_personal_context_multidevice.py \
  Tests/Integration/test_personal_context_first_link_live.py -v
pytest tldw_Server_API/tests/Integration/test_personal_context_sync_live.py -v
```

Expected: every journey passes against the real local server. Ask before a full
Chatbook or tldw_server test sweep.

- [ ] **Step 4: Perform manual live checks**

Begin by probing server capabilities and one authenticated control request.
Then link A, inspect the reconciliation plan, link B, edit/review/conflict,
disconnect A, perform privacy reduction and purge from B, reconnect A, and
verify forced rebootstrap. Fingerprint both real user profiles before/after to
prove scratch isolation. Capture only redacted IDs, statuses, and counts.

- [ ] **Step 5: Document rollout and enablement gates**

Document home-server limitation, server-trusted TLS threat model, schema/capability
requirements, linking/reconciliation, workspace mapping, proposals versus Sync
conflicts, offline behavior, privacy cleanup, device expiry, local removal,
global purge, recovery, and legacy-route deprecation. Keep both feature flags
off by default until Task 7 evidence passes in CI and the server key runbook is
approved. Disclose that authenticated TLS protects transit but the authorized
home server can read syncable Personal Context content before re-encrypting it
at rest.

- [ ] **Step 6: Commit in each repository**

Use commits:

```bash
git commit -m "test: prove personal context multidevice sync"
git commit -m "docs: stage personal context sync rollout"
```

## Plan 04 completion gate

- Chatbook and tldw_server store the same canonical IDs, versions, and bodies;
  Sync transports no application-specific profile projection.
- First link never silently chooses between divergent values or workspace scopes.
- Conflicts are reviewable and separate from agent proposals.
- Privacy reductions converge only after durable cleanup acknowledgments.
- Private conversion uses a new device-only ID and never leaks its value.
- Explicit device expiry enables progress without allowing a returned stale
  device to write before full rebootstrap.
- Global purge generations prevent resurrection from offline outboxes, restores,
  recovery imports, and stale server envelopes.
- The real one-server/two-Chatbook harness passes before rollout flags are enabled.
