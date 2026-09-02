# Personal Context Profile

This guide is the Chatbook implementation and integration reference for Personal Context. It documents what the current client ships, what the shared protocol can represent, and what still requires product work. For end-user workflows, see the [Personal Context Profile user guide](../User_Guide/settings/personal-context-profile.md).

The labels in this guide are deliberate:

- **Shipped** means a reachable production caller or user control exists in Chatbook today.
- **Protocol capability** means models, envelopes, or apply paths exist, but that alone does not make a complete user workflow.
- **Desired future behavior** is the accepted direction, not a promise about the current product.

## Contract and ownership

Chatbook bundles `tldw_profile_core` `0.1.0`; the companion server uses its pinned/vendored counterpart. The live packaging, parity, and canonical-fixture tests are the version-and-byte authority—do not copy a package digest into documentation. Shared Core owns canonical object models and bytes. [Sync V2](Sync-v2-client.md) separately owns transport envelopes, cursors, acknowledgement, and dataset staging.

<!-- shared-personal-context-contract:start -->
- `tldw_profile_core` defines the versioned canonical profile object models, exact canonical bytes, interview/tool contracts, serialization, and validation used by both peers. Sync-v2 transport envelopes are a separate contract.
- After successful reviewed first linking, Chatbook and tldw_server converge on the same canonical manifest, scope, record, proposal, and version identities and bytes for the eligible snapshot resulting from the user-approved content-free reconciliation plan.
- Sync V2 defines the `personal_context.manifest`, `personal_context.scope`, `personal_context.record`, `personal_context.proposal`, and content-free `personal_context.purge` domains. Reviewed first linking publishes the eligible snapshot resulting from the user-approved content-free reconciliation plan. Later syncable Chatbook mutations create encrypted local outbox entries, but the current shipped app does not run an ongoing Personal Context sync cycle, so those post-link changes remain queued locally. Purge production and distribution are not wired end to end.
- Each peer retains its own at-rest ciphertext and keys, local database rows, runtime permissions, conflict-review metadata, acknowledgement tracking, and other operational state.
<!-- shared-personal-context-contract:end -->

At rest, each peer encrypts its own rows with its own keys. During linking, Chatbook registers its public wrapping key and receives a wrapped, server-owned Sync integrity key; `PersonalContextLinkKeyCustodian` keeps that key separate from the local profile-at-rest key. Canonical payload bytes and Sync-v2 envelope bytes are therefore different contracts with different key custody.

The governing decisions are the [unified profile design](../superpowers/specs/2026-08-28-unified-personal-context-profile-design.md) and [ADR-102](../../backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md). Server storage and endpoint details belong in the [server developer guide](https://github.com/rmusser01/tldw_server/blob/dev/Docs/Code_Documentation/Personal_Context_Developer_Guide.md) and [server API guide](https://github.com/rmusser01/tldw_server/blob/dev/Docs/API-related/Personal_Context_API.md), rather than being duplicated here.

## Component and key-custody map

- `tldw_chatbook/Personal_Context/bootstrap.py` — `bootstrap_personal_context_service` assembles the unlocked service or a fail-closed locked facade.
- `tldw_chatbook/Personal_Context/key_protector.py` — `ProfileKeyProtector` owns local at-rest key protection, unlock, destruction, and recovery-export key handling.
- `tldw_chatbook/Personal_Context/repository.py` — `PersonalContextRepository` owns the dedicated SQLite schema, encrypted canonical objects, immutable versions, current heads, runtime policy, and atomic commits.
- `tldw_chatbook/Personal_Context/service.py` — `PersonalContextService` is the authorized canonical mutation boundary for records, scopes, controls, and interview batches.
- `tldw_chatbook/Personal_Context/context_service.py` — `ProfileContextService` selects and renders the bounded, effective agent context.
- `tldw_chatbook/Personal_Context/proposal_service.py` — `ProfileProposalService` enforces proposal review and the narrow direct-write correction path.
- `tldw_chatbook/Personal_Context/runtime_policy.py` — `AgentAuthority` defines `read_only`, `propose`, and `direct_write`; the separate global runtime policy supplies the off switch.
- `tldw_chatbook/Personal_Context/interview_coordinator.py` — `ProfileInterviewCoordinator` coordinates fixed/adaptive questions, review, and approved-answer materialization.
- `tldw_chatbook/Personal_Context/interview_draft_repository.py` — `InterviewDraftRepository` stores encrypted, expiring unfinished interview drafts and transcripts locally.
- `tldw_chatbook/Personal_Context/interview_provider.py` — `InterviewQuestionProvider` is the interview model-provider boundary.
- `tldw_chatbook/Personal_Context/link_service.py` — `PersonalContextLinkService` owns capability negotiation, content-free review planning, and reviewed link application.
- `tldw_chatbook/Personal_Context/link_key_custody.py` — `PersonalContextLinkKeyCustodian` owns device wrapping keys and the staged/active server Sync integrity key.
- `tldw_chatbook/Personal_Context/sync_outbox.py` — encrypted `ProfileSyncOutbox` is the canonical repository's source journal for eligible local mutations.
- `tldw_chatbook/Sync_Interop/personal_context_adapter.py` — `PersonalContextSyncAdapter` validates canonical payloads and maps them to separate Sync-v2 envelopes.
- `tldw_chatbook/Sync_Interop/personal_context_dispatcher.py` — `PersonalContextOutboxDispatcher` stages source entries into SyncState and acknowledges/shreds the source copy.
- `tldw_chatbook/Sync_Interop/personal_context_first_link_sync.py` — `PersonalContextFirstLinkSync` publishes and verifies only the user-reviewed first-link lineage.
- `tldw_chatbook/tldw_api/client.py` — `bootstrap_sync_v2_personal_context` and `complete_sync_v2_personal_context_link` are the HTTP bootstrap/completion calls.
- `tldw_chatbook/Agents/profile_tool_provider.py` — `ProfileToolProvider` advertises and reauthorizes profile tools.
- `tldw_chatbook/Chat/console_chat_controller.py` — `ConsoleChatController` owns Console snapshot creation and context injection.
- `tldw_chatbook/Chat/console_agent_bridge.py` — `ConsoleAgentBridge` bridges the authorized profile tools into Console agent runs.
- `tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py` — `PersonalContextSettingsPanel` presents Settings actions only.
- `tldw_chatbook/Widgets/Settings_Widgets/personal_context_link_modal.py` — `PersonalContextLinkModal` presents the content-free reviewed-link plan only.
- `tldw_chatbook/Widgets/Settings_Widgets/personal_context_review_modal.py` — `PersonalContextReviewModal` presents proposal review only.
- `tldw_chatbook/UI/Screens/profile_interview_screen.py` — `ProfileInterviewScreen` presents interview and review state only.

UI, agent, and transport code must use the owning service/repository boundary and must not access profile tables directly.

## Storage, mutation, and authority boundaries

**Shipped.** `PersonalContextRepository` stores encrypted canonical versions and current heads in a dedicated database. Its `BEGIN IMMEDIATE` transaction boundary rolls back on failure. `PersonalContextService` uses compound repository commits so a record and manifest compare-and-swap, optional Undo snapshot, and eligible encrypted source-outbox entry succeed or fail together. Interview batches do the same for all approved records. Proposal acceptance atomically commits the accepted proposal, resulting record and manifest, eligible outbox item, and terminal content-free receipt.

Runtime enablement and scope-to-workspace mappings are peer-local policy, not canonical profile objects. Interview draft deletion and optional runtime enablement happen after the canonical interview batch commits, so retry and recovery code must not pretend they share the canonical transaction. Reviewed first-link rebaselining is atomic inside local SQLite; activating staged key custody is a separate recoverable step.

Record controls are independent:

- `syncable` versus `device_only` decides whether an eligible canonical mutation may enter the encrypted outbox. A device-only record may still be visible to local agents.
- `agent_visible` versus `user_only` decides whether agent context, profile tools, or adaptive interviews may use the record. It does not decide synchronization by itself.
- Scope is either global or an explicitly mapped workspace. A server workspace scope may remain canonical but unavailable to agents until the user maps it locally.
- Active records must also be unexpired and non-conflicted. Working-context records default to a 30-day expiry unless the user explicitly chooses no expiry.
- Archive removes a record from effective context. Delete creates a canonical tombstone; updates create immutable versions rather than overwriting history.

The effective runtime authority is the intersection of the global kill switch, selected scope, workspace mapping, current `AgentAuthority`, live policy generation/revision, record state, expiry, visibility, and conflict state. Defaults fail closed: profile use is off until enabled and a scope defaults to `propose`. Callers must recheck the live revision at execution rather than relying on a stale catalog or preview.

## Shipped lifecycles

### Manual edits and reviewed interviews

Manual Add/Edit operations validate scope, visibility, syncability, expiry, and secret-rejection rules before committing encrypted immutable versions and content-free receipts where the operation defines one. Approved interview output uses the same canonical records and controls; an answer is not agent context merely because it appears in a transcript.

`FixedQuestionProvider` generates the fixed interview locally and makes no model call. An adaptive interview uses the configured default Console provider/model with tools disabled. Each adaptive request sends the interview audience, coverage topics, question attempt, eligible active `agent_visible` and `syncable` records from the exact selected scope, and every prior turn. After the first answer, those prior turns include raw answer text. The screen displays the actual provider and model only after the first provider response completes and before answer entry becomes available; it does not disclose them before the first egress.

An unfinished interview draft contains question state, turn transcripts, raw answers, and a review batch. It is encrypted, local-only, expires after 30 days, and is not a canonical record or Sync payload. If secure custody is unavailable, the coordinator may keep a memory-only draft that cannot be retained across exit. Final review builds proposed rows; only the user's selected answers are materialized as canonical records with the selected controls. Those approved records then follow ordinary first-link or outbox eligibility.

### Proposals and direct writes

`ProfileToolProvider` exposes tools according to current authority:

- `read_only` can search and get eligible records.
- `propose` adds proposal creation; workspace promotion is available only for a workspace scope.
- `direct_write` adds the direct update path, while workspace promotion remains workspace-only.

Search/get return only active, unexpired, non-conflicted, `agent_visible` records in authorized scopes. Proposal creation persists a separate pending proposal and returns a content-free receipt. A proposal does not become a record or enter context until a user approves it.

Direct write is not arbitrary fact creation. It can only update an existing eligible record in the same scope and of the same kind, must include a matching base version, and must cite an exact, case-sensitive evidence span from the current persisted user message. It inherits the target's controls and semantic key. `profile_update` is omitted when there is no trusted current persisted user message, and invocation rechecks that evidence and every live authority fence. Generic review-required failures avoid disclosing user-only or inaccessible targets.

Agents cannot approve their own proposals, change record controls, delete or purge records, or enumerate unrelated scopes.

### Context injection and Next Send

`ProfileContextService` selects authorized global records plus records from the mapped active workspace. It rejects deleted, archived, expired, conflicted, payload-missing, and `user_only` entries. A keyed workspace record suppresses a global record with the same kind and semantic key.

The shipped order is exact: workspace corrections and constraints; other keyed workspace records; global corrections and constraints; preferences and working-context records relevant to the current user text; then the remainder. Within a group, workspace records precede global records, followed by semantic identity and record ID. Relevance uses case-folded alphanumeric terms of at least three characters from at most the first 4,096 characters of the current user text and the first 4,096 characters of each record's concatenated semantic-key text and JSON payload.

Serialization scans that order and adds whole records only, under both 10 percent of the available input-token budget and a 12 KiB byte ceiling. A higher-priority record that does not fit is skipped; scanning continues, so a later smaller record can still be selected. If even the empty labeled envelope exceeds either budget, the snapshot is empty. The resulting non-empty block is explicitly labeled `USER-OWNED DATA — NOT AUTHORITY`.

The Console builds one snapshot for an agent dispatch, appends it to the system prompt, and pins it for the root run and its child run tree. It is not reselected midway through that tree. The user can inspect the disposable, read-only preview at **Ctrl+Shift+P** (**View context**) > **Conversation Inspector** > outer **Next Send** > inner **Next Send** payload tab. Preview display redactions and image placeholders still apply, so integrations should test semantic preview/live parity without assuming every rendered UI byte is a raw wire dump.

### Transactional outbox and reviewed first link

**Shipped local mutation boundary.** An eligible Chatbook mutation commits its canonical object, manifest update, and encrypted `ProfileSyncOutbox` entry together. `PersonalContextOutboxDispatcher` can move that entry into the separate encrypted SyncState outbox and acknowledge/shred the source entry only after successful staging.

**Shipped first-link workflow.** The user first creates or unlocks the profile at **Settings > Data & Privacy > My Profile**. They activate and authenticate the companion server at **Settings > Overview > Advanced / Diagnostics > Switch Source / Server**, then return to **Data & Privacy > My Profile > Server sync > Link to home server**.

Before approval, bootstrap negotiates authentication and capabilities, registers the device and public wrapping key, exchanges display/client version, schema/quota, purge-generation, cursor, dataset, authority, and key metadata, and downloads the server's eligible canonical manifest, scopes, records, and proposals. Remote record/proposal content exists transiently in memory to compute the plan. Durable pre-approval state and the visible plan remain content-free: identifiers, versions, counts, outcomes, and keep-device/keep-server choices. No local profile record or proposal content is uploaded before approval.

On approval, Chatbook unwraps and stages the server-owned Sync integrity key, atomically applies the reviewed reconciliation locally, completes the link on the server, then runs `PersonalContextFirstLinkSync` over only the reviewed lineage. Completion verifies the expected heads and cursor, producing one matching eligible snapshot on both peers.

**Protocol capability, not ongoing shipped behavior.** Sync V2 and `LocalFirstSyncService` understand manifest, scope, record, proposal, and content-free purge envelopes. No startup, background, Settings, or other production caller currently runs an ongoing Personal Context sync cycle. **Overview > Manual Sync** uses only the Notes and Chat domains. Consequently, later eligible Chatbook mutations remain queued in the encrypted canonical outbox.

<!-- personal-context-boundary-matrix:start -->
| Published at reviewed first link when eligible | Not published afterward or peer-local |
| --- | --- |
| Approved eligible canonical manifest | Later syncable Chatbook mutations, which remain queued locally |
| Required global and linked-workspace scopes | Ordinary server REST mutations |
| Controls-eligible record heads and tombstones; eligible proposals and canonical review state; approved interview answers after they are saved as records | Device-only or non-syncable records |
| Exact canonical IDs, versions, and bytes | Runtime agent authority, tool availability, workspace mappings, and enablement |
| — | At-rest and recovery keys; local undo, caches, ciphertext, database row IDs, conflict-review objects, acknowledgement tracking, and operational metadata |
| — | Interview draft and transcript objects; adaptive requests still send prior raw answers to the provider |
<!-- personal-context-boundary-matrix:end -->

### Post-link conflicts and purge limits

Ordinary server REST edits are not currently published to linked Chatbook clients.

Post-link conflicts retain generic Sync metadata but have no dedicated Personal Context resolution screen.

Because the ongoing Personal Context caller is absent, the ordinary shipped workflow also does not generate these post-link conflicts. The only dedicated collision review is the content-free first-link reconciliation surface.

The Personal Context purge domain is protocol-only in the current linked flow: Chatbook has no producer, and end-to-end distribution and acknowledgement are not wired.

The server's authenticated purge route advances server-local purge state, but it does not currently distribute a purge envelope or complete the acknowledgement barrier. Link to the server API guide for endpoint behavior; do not reproduce or infer server calls in a client integration.

## Removal, recovery, and connection boundaries

### Local removal and surviving state

**Shipped.** Remove local profile destroys canonical repository content: encrypted objects and heads, the canonical encrypted outbox, quarantine rows, runtime policy, local mappings, Undo data, and local record links. It then deletes the protected canonical profile key.

This is not a device-wide artifact purge or delete-everywhere operation. The separate encrypted `tldw_chatbook_personal_context_interviews.db` database and its protected per-session draft keys are outside `PersonalContextRepository` and can remain; retained draft payloads may include raw answers and turn transcripts. Separate `SyncStateRepository` artifacts can remain too, including `personal_context_link_state`, `sync_profile_state`, staged `sync_v2_local_outbox` envelopes, remote heads and cursors, conflict reviews, and receipts. Link custody is also separate: the device RSA wrapping private key, an incompletely cleaned staged server integrity key, and dataset-staging keys can survive. The server copy and device registration are unchanged.

If canonical key deletion fails after repository rows are removed, **Finish secure removal** calls the destroyed canonical repository's cleanup again and retries only its protected profile-key deletion. It does not remove interview drafts or draft keys, SyncState rows or envelopes, link-custody keys, the server copy, or device registration. It is a repair for one failed canonical-key step, not a complete local-artifact purge.

### Recovery export

**Shipped export.** The passphrase-encrypted recovery export contains the current manifest, all scopes, current record heads—including device-only records and tombstones—and proposals. `load_recovery_export` is a helper exercised by tests; there is no production UI or caller that imports/restores the export. Treat restore as **desired future behavior**, not a hidden recovery workflow.

### Server URLs, TLS, and Test Connection

Chatbook accepts root server URLs using either `http://` or `https://`; HTTP is unencrypted. Runtime API calls verify certificates by default, can add a configured custom CA, and can disable verification through `[network] ssl_verify`. Invalid verification configuration fails back to the safe default. Disabling verification removes server authentication and permits interception. API calls refuse redirects and carry the configured API key.

**Test Connection** uses a standalone `httpx.AsyncClient(timeout=5.0)` with default certificate verification. It does not use the saved custom-CA or verification-off runtime policy, so its result is not proof that the configured production client follows the same TLS boundary.

## Privacy and logging

Never log profile plaintext, ciphertext, wrapped keys, or raw cryptographic errors.

Keep plaintext out of diagnostics, routing/outbox metadata, temporary artifacts, and unencrypted fixtures. Checked-in canonical fixtures must be synthetic. Sanitized error categories may be logged, but raw exception strings at cryptographic boundaries can disclose sensitive material.

At-rest encryption does not make authorized egress private. A linked companion server can read eligible syncable content after publication. Adaptive interview requests send the data described above to the configured provider. UI and consent copy must identify those trust boundaries without logging the payloads.

## Desired future behavior

The accepted architecture may later add an ongoing Personal Context sync caller, server-origin mutation publication, queue and conflict-status UI, dedicated post-link resolution, an explicit production transport policy, provider/model disclosure before the first adaptive request, recovery import, and complete purge distribution and acknowledgement. None is shipped merely because its underlying model or protocol type exists. Each becomes current behavior only after it has a production caller, reachable user or operator surface, failure recovery, and end-to-end verification.

## Extending Personal Context

Start with the integration mode. A full local-first Sync peer and a server/API-only client have different obligations.

<!-- personal-context-extension-checklist:start -->
1. Decide whether the integration is a full local-first Sync peer or a server/API-only client.
2. Make shared canonical object changes in `tldw_profile_core` first; change Sync transport separately.
3. Preserve canonical identities and explicit syncability whenever the integration persists or transports canonical objects.
4. Route full peers through their owning services; route API-only clients through authenticated public server APIs, never profile tables.
5. Enforce authority, scope, expiry, visibility, and secret-rejection rules at the boundary the integration owns.
6. Keep plaintext out of logs, diagnostics, outbox metadata, and unencrypted fixtures.
7. Add parity/conformance coverage for every shared-core or Sync contract the integration implements.
8. Test only the owned surface: full peers need storage, key, service, Sync, runtime/UI, and recovery coverage; API-only clients need authentication, request/response, error, and privacy coverage.
9. Update the governing ADR for storage, ownership, encryption, Sync, or authority changes.
10. Update both documentation sets whenever the shared contract changes.
<!-- personal-context-extension-checklist:end -->

### Full local-first Sync peer

A full peer owns its local canonical repository, migrations, at-rest key custody, explicit syncability controls, encrypted source outbox, device/link bootstrap, capability negotiation, and content-free reconciliation review. If it offers agent use, it also owns runtime authority, workspace mappings, context injection, tools, interviews, proposal review, local removal, and recovery surfaces. To claim ongoing local-first synchronization, it must ship a scheduler or manual caller, queue/status UI, retries, conflict review, and user-visible recovery—not just adapters.

### Server/API-only client

An API-only client authenticates to the companion server and uses its documented public REST surface. It does not need to implement a local canonical repository, local profile/key custody, Sync-v2 envelopes, first-link reconciliation, interviews, agent injection, profile tools, local removal, or recovery export. It owns API credential handling, TLS policy, request/response validation, privacy-safe errors, and whatever non-authoritative cache its product explicitly documents. It must not claim local-first convergence or peer-local guarantees it does not implement.

### Shared and companion-server responsibilities

**Companion-server responsibility:** own authenticated per-user canonical storage and master-key custody; device registration and wrapped integrity-key bootstrap; server transport and materialization. Publishing server-origin REST mutations, producing/distributing purge envelopes, and aggregating acknowledgements are **desired future behavior** until implemented and tested.

**Shared responsibility for full peers:** keep Shared Core canonical models and bytes conformant; evolve Sync-v2 capabilities/envelopes independently; test the full first-link lineage. An API-only client depends on the public server API contract instead of implementing these peer contracts. A complete future purge feature requires client production/acknowledgement and server distribution/barrier completion.

Do not infer product reachability from protocol classes. Before calling a feature shipped, locate its production caller, user control, status/error surface, and end-to-end test.

## Targeted test map

- `Tests/Packaging/test_profile_core_packaging.py` — embedded Shared Core version, packaging, and canonicalizer pin.
- `packages/tldw_profile_core/tests/` — canonical models, serialization, validation, schemas, and interview/tool contracts.
- `Tests/Personal_Context/` — repository, service, crypto/key custody, runtime policy, interview, export, link, and removal behavior.
- `Tests/Agents/test_personal_context_prompt.py` — context injection and root/child snapshot pinning.
- `Tests/Chat/test_console_personal_context_snapshot.py` — Console snapshot reservation and preview/live behavior.
- `Tests/Sync_Interop/test_personal_context_*.py` — capabilities, adapters, encrypted dispatch, first-link convergence, and SyncState boundaries.
- `Tests/UI/test_settings_personal_context.py` — Settings actions and state.
- `Tests/UI/test_personal_context_*.py` — interview, reviewed link, review, and removal presentation.
- `Tests/tldw_api/test_personal_context_sync_client.py` — authenticated bootstrap/completion client behavior and response validation.

For a shared-contract change, run the equivalent server conformance suite as well. For a client-only change, select the smallest paths above that exercise the changed owner plus its caller; documentation assertions should additionally verify the marked shared block, current-limit sentences, repository paths, and privacy prohibitions.
