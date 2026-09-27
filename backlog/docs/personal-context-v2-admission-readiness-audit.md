# Native V2 memory admission and cutover readiness

Date: 2026-09-26
Task: [TASK-25907.17](../tasks/task-25907.17%20-%20Audit-native-V2-memory-admission-and-cutover-readiness.md)
Inspected baseline: `677268f326`, native Python 3.12.11.
Status: source-backed readiness audit; no V2 model, migration or activation shipped.

## Finding

Exact text/span digests and the complete owner/version binding are available,
but neither can currently be stored as a qualified canonical profile claim.
The shared manifest, record, proposal, schema vocabulary and native consumers
remain V1. Safe foreground source inspection therefore still needs containing
record admission, profile-wide compatibility and binding-metadata lifecycle
controls. The device-only read correction closes one current path; it is not a
V2 activation or metadata-retirement protocol.

The smallest next deliverable is a **reviewed concrete V2 canonical data
contract and conformance matrix**. It must freeze the actual manifest, record,
proposal, digest projections, relation effects and restrictive disclosure
values before implementation. A data-only implementation can be published
without activating profiles; installing it is not permission to migrate, Sync,
resolve sources or send evidence. Native activation and source inspection
remain gated below. This recommendation does not waive server qualification or
approve a new schema/runtime rollout.

## Native owner matrix

| Owner and inspected entry | Current guarantee | Missing V2 qualification |
| --- | --- | --- |
| [Manifest](../../packages/tldw_profile_core/src/tldw_profile_core/models.py#L85), [record](../../packages/tldw_profile_core/src/tldw_profile_core/models.py#L117), [proposal](../../packages/tldw_profile_core/src/tldw_profile_core/models.py#L166) | Canonical objects have schema version exactly 1; V1 provenance has inert legacy IDs/hashes. | Distinct V2 envelopes, typed claim/support/approval/validity/relations, required semantic vocabulary and retirement/disclosure ceilings. Never add optional V2 fields to V1. |
| [Canonical schema dispatch](../../packages/tldw_profile_core/src/tldw_profile_core/schema_export.py#L16) and [package version](../../packages/tldw_profile_core/src/tldw_profile_core/__init__.py#L67) | Root canonical union and public serialized version are V1. | V2 schema/dialect, semantic rules, canonical byte fixtures and explicit object/semantics support. A package version or transport number is insufficient. |
| [Exact span primitive](../../packages/tldw_profile_core/src/tldw_profile_core/evidence.py#L15) and [binding component](../../packages/tldw_profile_core/src/tldw_profile_core/evidence_binding.py#L59) | Strict codepoint/UTF-8 digest identity and 18 required complete owner/version binding fields. | Containing record, trusted source admission, support, human review and privacy lifecycle. Current binding source kind is conversation message; no Notes, captured snapshot or retained excerpt adapter is implemented by this component. |
| [Storage schema](../../tldw_chatbook/Personal_Context/repository.py#L50), [manifest getter](../../tldw_chatbook/Personal_Context/repository.py#L1416) and [snapshot](../../tldw_chatbook/Personal_Context/repository.py#L3201) | SQLite repository schema is 8, authenticated-data schema marker is 1, canonical bodies validate as V1; export uses one explicit SQLite snapshot. | Separate storage migration and version dispatch, atomic manifest/record/relation admission and profile-wide activation barrier. Storage schema 8 is not Profile V2 support. |
| [Per-object getter](../../tldw_chatbook/Personal_Context/repository.py#L2880) | Corrupt/invalid records are quarantined and omitted from ordinary get/list reads. | Quarantine is not a negotiated semantic capability or retirement of an old client. Strict export snapshots independently validate all current rows; an invalid row can fail the whole snapshot. Do not claim quarantine always permits V1 context to proceed. |
| [Authorized context](../../tldw_chatbook/Personal_Context/service.py#L1647) and [agent target guard](../../tldw_chatbook/Personal_Context/service.py#L849) | Current scope/manifest/policy fences; device-only denied before consumer selection and mutation. Context starts from the strict export snapshot. | Registered context consumer acknowledgements, full V2 relation/validity evaluation and destination/purpose admission. An unknown exception may affect a V1 global claim, so filtering one unknown object is not an activation policy. |
| [Bootstrap request/response](../../tldw_chatbook/tldw_api/sync_schemas.py#L621), [schema attention](../../tldw_chatbook/tldw_api/sync_schemas.py#L685), [link default](../../tldw_chatbook/Personal_Context/link_service.py#L154) and [reconciliation](../../tldw_chatbook/Personal_Context/reconciliation.py#L465) | Exact schema requirement defaults to 1 for linking; strict content-free schema/quota/purge attention; bootstrap constructs typed V1 objects. | Object/semantic capability acknowledgements, retirement receipts, stale grant refusal and companion-server conformance. Reconciliation's integer comparison is not feature negotiation. |
| [Sync V2 envelope](../../tldw_chatbook/tldw_api/sync_schemas.py#L862) | Sync transport has its own envelope/version contract. | Its name does not mean canonical Profile V2 objects or evidence semantics are supported. Qualify both layers independently. |
| [Recovery validation](../../tldw_chatbook/Personal_Context/export_service.py#L281) | Independently encrypted recovery parses manifest/record/proposal as V1. | Versioned forward-only restore and retirement fences before usability; no fabricated approval/support/grants and no lossy V2-to-V1 conversion. |
| [Record retirement](../../tldw_chatbook/Personal_Context/repository.py#L2814) and [pending outbox cleanup](../../tldw_chatbook/Personal_Context/repository.py#L2856) | Tombstone lifecycle removes prior record versions and pending outbox bodies; Undo has a separate bounded owner lifecycle. | Source/claim dependency controls, metadata-bearing envelope retirement, reconnect receipts and registered cache/queued payload disposal. Existing deletion/Undo tests do not establish complete V2 forgetting or offline remote erasure. |
| [Accepted foreground inspection design](../../Docs/superpowers/specs/2026-09-26-personal-context-foreground-source-inspection-design.md) | Design defines host-issued independent source authority, narrow current reads, final fresh probes and publication/invalidation gates. | No resolver or mounted caller exists under this task; real V2 admission and binding metadata controls precede implementation. Stored DTOs, profile grants and matching hashes cannot mint source access. |

No required-context-semantics, evidence-retirement-epoch or context-consumer
registry fields were found in the inspected Personal Context/shared-core Python
contract. That is a scoped source finding, not a companion-server audit or proof
that differently named unrelated application registries do not exist.

## Recommended sequence and alternatives

1. **Concrete shared-core contract first — recommended.** Freeze field names,
   strict scalars/defaults/bounds, unknown vocabulary behavior, tombstone and
   proposal rules, claim/binding digest projections and fixtures. Carry forward
   ADR-185 temporal/review semantics and ADR-187 restrictive disclosure ceilings.
   The 18-field binding remains its published component; a governed excerpt or
   new source representation requires explicit versioning/composition rather
   than silently widening it. Keep V1 schemas/bytes and fixture copies intact.
   Test the inactive V2 library contract before any native acceptance path.
2. **Local-only V2 storage first — defer.** A device-only flag stops current
   agent reads/Sync but does not qualify source-metadata access, owner exports,
   recovery/Undo, cached observations or privacy revocation. It also cannot
   retire an old consumer that ignores V2 effects. A local-only rollout would
   need a reviewed change to any applicable existing rollout gates; this audit
   neither chooses nor approves that change.
3. **Resolver over V1 IDs or a parallel binding store — reject for this flow.**
   Legacy IDs lack the complete containing-claim contract and independent source
   authority. A sidecar cannot by itself guarantee atomic claim/evidence Sync,
   admission or retirement. Follow the existing accepted canonical ownership.

After the concrete contract, qualify native version dispatch and atomic
admission together with profile-wide consumer support/retirement. Define the
actual active consumer inventory across context/tools, interviews, exports,
restore, Sync and other memory-producing/replaying paths, then prove that each
acknowledges the required semantics or is explicitly disabled with old grants
retired. A dormant peer is not retired merely because it is offline. Pin shared
canonical/semantic conformance in both Chatbook and the companion server before
rollout, as ADR-185 requires. No server implementation was inspected here.

Bind metadata retirement and restrictive disclosure to the actual native owners
before a V2 evidence claim becomes usable. Startup/restore/reconnect must apply
fences first. Distinguish full-profile purge generation, shared evidence
retirement epochs and local-only retirement revision; these are accepted design
obligations, not implemented fields. A reviewed privacy successor can preserve
permitted assertion text while withholding automatic use and retiring prohibited
binding metadata, without fabricating fresh human approval. Managed recovery
and Undo copies remain governed; delivered external copies cannot be recalled
by a local transaction.

Only then implement the accepted foreground Settings/app/Console source flow
with its real containing-record owner, independent already-open conversation
authority, bounded source read, all relevant writer publication gates, worker
cleanup and disposal. No pre-emptive generic resolver, fake capability DTO,
automatic source capture or provider call is needed for contract work.

## Future qualification matrix — not tests passed today

| Gate | Successful control | Required failure/race controls |
| --- | --- | --- |
| Data contract | Exact V2 bytes/digests match independently fixed fixtures in installed packages. | Unknown schema/semantics, wrong scalar types, unsafe-copy revalidation, bounds, duplicate IDs, altered claim/binding digests; unchanged V1 fixtures. |
| Activation | Every active consumer acknowledges exact object/semantic capabilities before atomic cutover. | One old consumer blocks; explicit disable retires grants; missing registry entry/unknown vocabulary blocks the whole profile including V1; offline reconnect cannot use old grants. |
| Admission | Current authorized heads plus reviewed wording/validity/relations commit in one manifest transition. | Stale/missing/foreign target, concurrent edit, invalid scope direction, cyclic/conflicting effects, failure between record and manifest writes; no partial head/effect. |
| Legacy handling | Unmigrated bytes stay V1; qualified migration leaves authority/support/dates unknown until review. | No invented approval or dates from storage timestamps, no downcast, no automatic source lookup from legacy IDs. |
| Metadata privacy | Binding-free reviewed/qualified successor preserves permitted assertion content under current policy. | Source-policy revocation removes prohibited metadata from registered old envelopes/outboxes/Undo/recovery/caches; stale worker, restore and replay cannot resurrect it; remote receipt pending stays unconfirmed. |
| Disclosure | Qualified destination/purpose explicitly enrolled for admitted governed input. | Device-only remote denial, unknown destination/purpose/custody, old prepared payloads, retries/children/interviews/derived tool arguments; a flag or ordinary tool grant cannot authorize egress. |
| Foreground inspection | Mounted owner-authorized current local span passes both exact digests and final fresh snapshots. | Imported locator, wrong scope/conversation, edit/delete/revocation during work or queued publication, borrowed transaction, time expiry, cancellation before cleanup, escaped display vs unchanged hash input. |

## Native evidence and limits

The audit reran **36 existing targeted cases**, all passed in 7.51 seconds.
Canonical/schema tests include real strict structural/semantic fixtures; native
SQLite tests cover foreign/newer repository schema, corruption quarantine,
bootstrap rejection without mutation, independent recovery, tombstones and
bounded Undo. Reconciliation has a successful exact-version/quota control.
Client attention/path tests use `httpx.MockTransport` or `AsyncMock`; they do
not contact a server or qualify server capability negotiation. Source span and
binding primitives retain their earlier separately recorded qualification.
No production or test Python changed in this audit; there is no new Python
lint scope. Existing test results do not establish V2 behavior. Whitespace, links, code anchors and byte preservation
are checked separately. Native execution preference persists.

```sh
PYTHONPATH=.:packages/tldw_profile_core/src .venv/bin/python -m pytest packages/tldw_profile_core/tests/test_canonical.py packages/tldw_profile_core/tests/test_schema_fixtures.py Tests/Personal_Context/test_repository.py::test_repository_fails_closed_on_foreign_or_newer_schema Tests/Personal_Context/test_repository.py::test_corrupt_record_is_quarantined_and_omitted Tests/Personal_Context/test_runtime_policy.py::test_bootstrap_rejects_unsupported_and_corrupt_databases_without_mutation Tests/Personal_Context/test_profile_reconciliation.py::test_plan_records_exact_content_free_contract_outcomes Tests/Personal_Context/test_profile_reconciliation.py::test_plan_excludes_device_only_and_keeps_remote_workspaces_unlinked Tests/Personal_Context/test_export_service.py::test_recovery_export_is_independently_encrypted_and_round_trips Tests/Personal_Context/test_export_service.py::test_recovery_preserves_current_tombstones_while_plaintext_omits_them Tests/Personal_Context/test_service.py::test_delete_retires_prior_record_and_outbox_content_but_keeps_bounded_undo Tests/Personal_Context/test_service.py::test_undo_body_is_encrypted_and_expires_after_exact_24_hours Tests/Personal_Context/test_profile_sync_outbox.py::test_device_only_record_never_enters_profile_sync_outbox Tests/Personal_Context/test_profile_sync_outbox.py::test_pending_device_only_proposal_never_enters_profile_sync_outbox Tests/tldw_api/test_personal_context_sync_client.py::test_personal_context_bootstrap_schemas_are_strict_and_typed Tests/tldw_api/test_personal_context_sync_client.py::test_personal_context_bootstrap_attention_schema_is_discriminated_and_strict Tests/tldw_api/test_personal_context_sync_client.py::test_personal_context_bootstrap_attention_rejects_semantically_invalid_bodies Tests/tldw_api/test_personal_context_sync_client.py::test_client_raises_only_typed_content_free_bootstrap_attention Tests/tldw_api/test_personal_context_sync_client.py::test_client_uses_exact_personal_context_bootstrap_and_complete_paths -q -o cache_dir=/private/tmp/memory-v2-admission-audit-cache --basetemp=/private/tmp/memory-v2-admission-audit-20260926 --junitxml=/private/tmp/memory-v2-admission-audit-20260926.xml
```

Receipt: `/private/tmp/memory-v2-admission-audit-20260926.xml`. Test config/data
and keyring are isolated by Tests/conftest.py; temporary SQLite uses an in-memory
key protector. No full sweep, app launch, real profile/source/keyring/provider,
network, server audit/test, fetch, push, PR or merge.

## Self-review and ADR check

The review corrected three possible misreadings before closeout: SQLite schema
8 and Sync V2 do not imply Profile V2; ordinary get/list quarantine differs
from strict export/context snapshot validation; a current tombstone plus
bounded Undo does not prove dependency-wide metadata forgetting. The audit
credits current device-only enforcement without extending it to retained or
queued history. Task allocation checked available local object history and 82 worktrees with no
TASK-25907.17 collision. The CLI emitted YAML-title warnings for two unrelated
tasks in a local branch, then created the expected task; no foreign task was
edited. Backlog remote operations and auto-commit remain disabled.
No new schema/default, permission, runtime boundary or ADR is selected by this report. Original memory-task and canonical-fixture bytes stay
unchanged; independent TASK-25907.10 and its roadmap suffix are preserved.

ADR required: no new ADR for this read-only audit/recommendation.
ADR paths: [ADR-102](../decisions/102-personal-context-profile-authority-sync-and-encryption.md),
[ADR-185](../decisions/185-versioned-profile-evidence-and-temporal-claims.md),
[ADR-186](../decisions/186-dependency-aware-personal-context-forgetting.md),
[ADR-187](../decisions/187-personal-context-provider-disclosure-authority.md),
[ADR-191](../decisions/191-foreground-personal-context-source-inspection-authority.md).
A later concrete schema or native admission/activation design needs its own
review and ADR check. Accepted design direction is not runtime rollout approval.
