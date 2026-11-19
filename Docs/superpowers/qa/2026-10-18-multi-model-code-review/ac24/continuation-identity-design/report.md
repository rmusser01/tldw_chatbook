# TASK-35582: exact selected identity across native continuation

Design only, 2026-10-27. Production source audited at `26c291443ce77a268444a3ed9adc4bc666061c64` (root subsequently committed docs-only evidence at `8f14f54c74a83f16a4ba4134deeffb2e46952b31`, with production source unchanged) in `/Users/macbook-dev/.codex/worktrees/task-35582-multi-model-review/tldw_chatbook`. No production, Git, configuration, credential or original evidence changes; no tests, network requests or child agents. This document supplements `native-continuation-diagnosis.md`, not a new live result.

## Decision

Keep canonical checkpoint `provider` as the execution family. Extend the existing frozen Console destination with the three missing continuation facts, then extend the existing assistant-owned continuation JSON with the exact raw selected provider. Bind at the host adapter boundary, verify against accepted destination again at store/repository barriers, and require the raw identity during history and explicit recovery. Use the existing message owner, transactions, version/CAS, sync and export contracts. No new registry, job engine, table, permission store or generic authority framework.

This is a small cross-owner contract repair, not a repository string-alias fix. It requires JSON format evolution and a managed-resume compatibility change. Existing fields are insufficient after the dispatch checkpoint is deleted.

ADR required: yes (amend existing decisions).
ADR paths: `backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md` and `backlog/decisions/147-agent-provider-routing.md`; cross-reference ADR-179 and ADR-240.
Reason: persistent continuation identity and host/adaptor validation contracts change. The durable owner and trust boundary already exist, so a new numbered ADR would duplicate them.

## Proven topology and an additional mismatch

1. `console_provider_gateway.py::ConsoleProviderResolution` already carries the host-resolved raw `provider`, optional `selected_provider`, canonical `execution_key`, `continuation_protocol`, `model`, full `base_url`, and endpoint provenance. The ready resolver constructs these together; immutable turn context detaches its destination before durable acceptance.
2. `console_library_destination.py::resolve_console_destination` intentionally reduces `base_url` to an **origin** for disclosure. `ConsoleResolvedDestination` has raw provider, model, origin, egress class and in-memory provenance, but no execution family, protocol or full continuation base. `console_turn_context.py::_detached_destination` copies only those fields. Its JSON has four exact keys and a 2,048-byte ceiling.
3. Thus there are TWO mismatches in the actual handoff contract. The existing diagnosis proves `custom-ep:ac24-local != custom-hosted`. Also, `native-continuation-diagnosis/live-readonly-evidence.json` records the actual accepted `endpoint_identity` as `http://127.0.0.1:8010`, while the candidate pins `http://127.0.0.1:8010/v1`. The earlier report's wording “same explicit /v1 endpoint” must not be read as a claim that the persisted destination includes `/v1`. The test double uses the real origin classifier and therefore also contains both mismatches. Removing the provider mismatch alone would still fail the URL comparison. Neither assertion establishes any additional live provider behavior.
4. `console_dispatch_repository.py::_continuation_matches_destination` is shared by first active handoff and complete terminal settlement. Both currently compare canonical checkpoint provider/model/full base to raw destination provider/model/origin. Its surrounding transaction enforces exact assistant/user/conversation ownership, active-path state and expected versions; successful handoff deletes `console_dispatch_checkpoints` atomically with writing the assistant continuation.
5. `console_chat_store.py::persist_provider_continuation_event` resolves the event, invokes the optional managed finalizer, serializes, then performs handoff/settlement or ordinary continuation update. `_continuation_event_value` takes new batch/final candidates from the provider event; call transitions use current stored state. A store-only raw pin is insufficient: `agent_runtime.py` keeps the original successful candidate in its own continuation variable and later submits that candidate's groups to gateway preparation.
6. `_StreamingModelAdapter.chat_call` knows the actual immutable **per-call** `effective_resolution`, including independently routed children. `_StreamingProviderResponse` currently exposes `terminal_metadata.provider_continuation` unchanged to AgentService/runtime. This is the existing host seam for binding neutral V1 candidates before the runtime retains them.
7. `ProviderContinuationCheckpoint` currently admits V1 neutral content and V2 content with a managed-resume envelope. It has no raw selection. `ContinuationRestoreTarget` has canonical provider/protocol/model/base only. Shared restore validation is consumed by controller recovery, history budget, gateway, runtime, and three provider replay owners (hosted engine, Moonshot, ZAI).
8. Managed V2 sealing/verification in `Plugins/continuation.py` hashes a V1 projection by forcibly replacing `schema_version=1, managed_resume=None`. `Plugins/service.py::resume_maximum`, bridge restore, and controller restore identify managed state with `schema_version == 2`. These must change coherently for V3; otherwise V3 would either discard the new identity from its MAC or accidentally obtain a zero-managed ceiling.
9. DB canonicalization, sync, chatbook import/export, Character Chat private export/import and trace private-value projections already reuse the shared parse/dump contract. No alternate durable raw-selection authority was found. AgentRuns routing snapshots describe agent runs, not the portable assistant branch/variant owner; session selections are mutable. Neither is an adequate replacement for the continuation field.

## Concrete data contract

### Frozen accepted destination

Add three optional fields to `ConsoleResolvedDestination`: `continuation_provider`, `continuation_protocol`, `continuation_base_url`. They are an all-or-none tuple. Existing `provider` remains the exact raw selected identity, existing `model` remains the resolved model, and existing `endpoint_identity` remains the disclosure origin.

For a continuation-capable durable target, `resolve_console_destination` copies execution key, continuation protocol and full base from the SAME resolved host object; it validates the canonical provider/protocol pairing and credential-free HTTP(S) base using the existing continuation/endpoint rules. It never re-resolves a slug. `_detached_destination` must preserve them. Unsupported provider resolutions leave all three absent, preserving ordinary destinations.

Version the nested destination JSON: existing exact four-key JSON is legacy V1; new exact JSON adds `schema_version: 2` plus the three continuation fields. A tuple-less destination continues to serialize as the existing V1 shape. Strict parsing retains legacy byte-canonical round trips; partial/unknown versions or fields fail. The existing 2,048-byte ceiling remains unless a separately justified bounded increase is recorded; do not silently truncate a base to fit.

Keep `identity_key` as the existing disclosure identity (callers index its egress element). For dispatch retry, compare the complete frozen destination, including the new tuple, instead of relying on that disclosure key. `_prepared_continuation_block_copy` already uses complete equality; `_resolve_dispatch_retry_context`'s current identity-key comparison must be upgraded. No provider/config state may repair a missing frozen tuple inside repository handoff.

Endpoint provenance remains binding. Never persist a session-only endpoint through the new field: retain the existing omitted-endpoint marker, omit the full continuation tuple for that provenance, and fail durable native handoff before tools. Explicitly ephemeral conversations retain their existing in-memory/non-resumable path. This prevents the repair from opening the currently refused origin/base handoff in a way that leaks session-only endpoint bytes. No automatic conversion of a persistent chat to ephemeral work.

### Durable continuation V3

Add `selected_provider: str | None = None` to the typed checkpoint. V3's exact keys are the V1 keys plus `selected_provider` and `managed_resume`; `selected_provider` is a nonblank bounded string, `managed_resume` is either null or the existing validated managed envelope. V1/V2 keep their exact old shapes and canonical encodings. The canonical `provider` Literal and provider/protocol pairings are unchanged. Existing identity/total payload ceilings apply; repr/errors never print the raw selection or base.

This is the smallest durable addition: execution provider, protocol, model and base are already present. Do not duplicate those facts in a second envelope. The new selected field identifies the host's raw destination; parsing it alone grants no execution permission and does not resolve any endpoint.

Extend `ContinuationRestoreTarget` with `selected_provider: str | None = None`. Production Console targets always capture `resolution.selected_provider or resolution.provider` verbatim, with no custom-slug case folding, prefix inference, thread-kind inference or message-body inference. The existing canonical target provider remains `execution_key`.

Keep the current provider endpoint equivalence behavior in the shared execution comparison: the current validator accepts the exact checkpoint base or its existing `canonical_connection_identity` form. Do not replace that with origin-only, arbitrary URL normalization or fresh config resolution. New accepted continuation bases use the already resolved request base; handoff compares that to the validated candidate using this SAME existing execution comparison. The raw selection comparison is always exact, even when the endpoint contract recognizes two spellings of one API base.

## Validation owners and ordering

### Admission and host candidate binding

The gateway produces the tuple before the existing immutable execution-context and durable-acceptance boundaries. When a continuation-capable resolution with DURABLE_CONFIGURATION provenance is accepted for persistent execution, missing/malformed tuple is an admission error, not permission to fall back to the origin. Preserve provider-intent versus effective-destination semantics; do not make the pre-resolution intent a second resolver.

In `_StreamingModelAdapter.chat_call`, after actual effective resolution and successful provider response but before building `_StreamingProviderResponse`, bind its neutral candidate to the exact per-call raw selection. One small pure helper in `provider_continuation.py` should validate the candidate's canonical execution tuple against the host target and produce V3 with the host selected provider. A provider response cannot supply or replace the mapping. A conflicting existing selection is refused; neutral adapters remain V1 candidate producers. This binding applies to active batches AND complete candidates.

For private request preparation, a primary continues to use its frozen target. A child must use its actual independently resolved per-call target (including model and base overlays) when it has its OWN continuation groups; it must never inherit a custom parent's raw pin merely because execution family matches. The existing parent-sidecar exclusion for children remains. Compute the child target only after `effective_resolution` is finalized inside `_consume`; do not reuse the old model-only replacement of the parent target. Preserve refusal to reroute an already pinned primary continuation. Fallback does not authorize changing an accepted durable destination; if existing admission cannot prove the fallback target, handoff remains refused rather than inventing a new acceptance.

### Store and repository barriers

Pass the captured host continuation target through the existing bridge persistence callback as an explicit host-only argument (never a tool/model kwarg). Store validation precedes the managed finalizer and JSON serialization:

- First batch/final event with a dispatch owner: candidate's exact selected identity and canonical tuple must agree with the frozen accepted destination AND the captured run target. Refuse missing accepted tuple. Stale store state still cannot win repository CAS.
- Later batch/final events: preserve the durable owner's raw selection and canonical identity; validate the candidate against both current owner and frozen run target before replacing it. Provider candidates or event data cannot erase the raw pin or carry a foreign pin.
- Executing/finished events keep that binding via ordinary immutable transitions.
- Persistent paths without a dispatch row use their already host-captured run target and existing exact assistant/generation ownership; no lookup from mutable session selection. If the callback cannot provide that target, refuse new virtual persistence. Legacy direct-provider fixture paths must be explicit, not an implicit production fallback.
- Bind/validate before managed sealing. A failed finalizer cannot downgrade V3+managed to unsealed V1/V3, and provider replacement candidates cannot silently drop required managed custody.

Repository `_continuation_matches_destination` then checks the same canonical tuple using the new frozen fields and requires V3 `selected_provider == destination.provider`. It remains a boolean refusal seam under existing owner/CAS transaction. Both handoff and terminal settlement use it. Do not stamp or change continuation bytes inside the repository: doing so would invalidate the store's managed seal and desynchronize its in-memory checkpoint/hash. The stored V3 must be byte-identical to the value on which durability/sync receipts were computed.

### Restore and history

Separate shared execution validation from full host restore validation using two named functions, not a permissive implicit “if a raw value happens to be present” check. Existing `validate_continuation_restore` remains the full host check; a small extracted `validate_continuation_execution_target` contains its existing canonical family/mode/model/base checks. Only the three lower provider replay owners call the execution-only function: they intentionally receive canonical execution identity and must not learn or resolve custom endpoint configuration.

Controller recovery/history targets and bridge targets include the raw pin. Gateway preparation checks that target's raw selection against the actual effective resolution in addition to canonical provider/model/protocol/base checks. Its present allowance for either execution spelling or display spelling does not establish selected identity. Full restore rejects mismatched or missing V3 raw targets. History's existing behavior remains: incompatible COMPLETE private groups are omitted while visible owner text remains; incompatible ACTIVE state requires explicit recovery and is not silently replayed.

Existing runtime/service revision, round/call, and canonical-target checks remain intact. The host adapter stamps each new runtime candidate, and store enforces immutable durable identity at every event. There is no need to add another agent-runtime identity registry or widen AgentService's canonical service endpoint to raw slugs. Direct AgentService/adapter tests must distinguish neutral V1 candidates from host-bound Console V3; do not turn low-level execution validation into authority to restore a virtual target.

### Managed V2/V3 composition

`seal_checkpoint` and `verify_constraint` must use a shared, version-aware unsigned projection: legacy V2 hashes the byte-identical V1 body it hashes today; V3 hashes the complete V3 body including `selected_provider`, with `managed_resume` set to null. Sealing a bound V3 returns V3 with the managed envelope; sealing a legacy neutral V1 retains V2 behavior. This is a digest compatibility branch, not a new MAC or key.

Replace the three checkpoint `schema_version == 2` managed-presence guards (plugin service, bridge, controller) with validated managed-envelope presence. V3/null receives the existing zero-managed ceiling; V2 and V3/envelope require existing owner, namespace, authority, data and MAC validation. Altering raw slug, model, base, revision or owner must invalidate a managed pin. Remapped/imported owners still cannot restore managed permission.

## Legacy, versions, and migration

- V1 and V2 continuation JSON remains readable, exportable, and byte-canonical. No speculative rewrite of old checkpoints. Existing canonical direct providers retain their current explicit resume/history behavior when the current selection is the same statically known built-in family (existing built-in aliases may be recognized by existing static provider identity helpers).
- Legacy pinless `custom-hosted` state cannot prove a raw virtual owner; fail closed for exact virtual restore/history replay. A legacy direct canonical checkpoint cannot be carried into a `custom-ep:*` target merely because canonical family/model/base match. Do not infer a slug from current registry settings or message metadata. Users retain visible text and explicit discard/new-work options. V2 authentication does not manufacture missing raw identity.
- Legacy four-key dispatch destinations remain readable for existing recovery/discard/non-continuation settlement. They cannot perform a NEW native continuation handoff without a frozen execution tuple. Refuse and retain recovery; do not upgrade already accepted authority from current config. This includes old direct-provider dispatch rows lacking mode/base proof, rather than silently granting them an inferred new contract. This conservative compatibility loss concerns interrupted pre-handoff work, not legacy canonical continuation resume.
- Destination JSON V2 and continuation JSON V3 are **serialization schema** changes inside existing TEXT columns. They do not require DDL, a new SQL column, an AgentRuns migration, or a ChaChaNotes `_CURRENT_SCHEMA_VERSION` bump. Keep the dispatch table's outer `schema_version` at 1 because its relational envelope/ownership does not change; the new nested JSON declares its own version. Document/test readers for both nested formats. If implementation instead adds a SQL field/table or changes constraints, it leaves this design and must add the normal file-backed migration and SQL schema version bump before code lands.
- Newer state is not guaranteed resumable in old binaries: old strict readers reject unknown JSON versions. Preserve historical text through existing tolerant private-state readers; never claim downgrade compatibility or rewrite signed V3 as V1. Explicit sync/chatbook tests must establish whole-record carriage and honest unsupported-state behavior, not new permission transfer.

## Privacy, races and security

The new slug is private continuation identity, not a credential, display label, permission or globally unique endpoint. Two user-configured slugs sharing URL/model remain distinct; identical slug spelling on another device is only an identity comparison and never transfers tool/managed permission. An attacker able to edit whole unencrypted messages already controls ordinary private continuation data; this change adds no new signature claim to unsealed V3. Managed V3 is authenticated as above.

Never serialize keys, headers, credential-bearing/query/fragment URLs, full provider resolution, current registry bodies or session-only endpoints. Keep raw pins/base out of repr, normal logs, errors, FTS, summaries and ordinary text exports. Use current private JSON/chatbook disclosure and encryption boundaries. Trace private projections continue through their existing parser and capture policy.

No new locks or external I/O under SQLite transactions. Candidate binding is pure and uses the actual per-call immutable resolution; store finalization preserves the existing lock ordering. Repository remains final owner/CAS authority. Reopened resume re-evaluates current permissions and exact target; executing calls stay ambiguous, completed results replay exactly, pending work requires normal approval. Failed pin/commit cannot execute a tool, retry a provider, clear ownership, refund budget or reset the approved journey.

## Exact implementation surface

Expected production edits: **14 existing files**, with no new production module:

1. `Chat/provider_continuation.py`: V3 data/codec, selected target, pure host binder, extracted execution validator and full restore rules.
2. `Chat/console_dispatch_checkpoint.py`: optional tuple, strict destination JSON V2/V1 compatibility and provenance-safe serialization.
3. `Chat/console_library_destination.py`: copy host continuation tuple while preserving disclosure origin.
4. `Chat/console_turn_context.py`: detached tuple copy.
5. `Chat/console_dispatch_repository.py`: shared handoff/settlement matching against frozen tuple and exact selected identity.
6. `Chat/console_chat_store.py`: independent immutable owner/target validation before finalizer and every persistence barrier.
7. `Chat/console_agent_bridge.py`: raw targets, per-call candidate binding, actual child targets, host-only callback argument, managed-presence guard.
8. `Chat/console_chat_controller.py`: restore/history raw target, full destination retry comparison, managed-presence guard.
9. `Chat/console_provider_gateway.py`: prepared target/raw-resolution agreement, fail missing tuple for continuation-capable durable admission at its existing seam where appropriate; preserve wire/accounting paths.
10. `Plugins/continuation.py`: V2-compatible/V3-complete signing projection and seal version.
11. `Plugins/service.py`: managed-envelope presence for resume ceiling.
12. `LLM_Calls/hosted_provider_engine.py`: execution-only replay validation; neutral candidate writer unchanged.
13. `LLM_Calls/moonshot.py`: execution-only replay validation.
14. `LLM_Calls/zai.py`: execution-only replay validation.

No planned change to `Agents/agent_runtime.py`, `Agents/agent_service.py`, `DB/ChaChaNotes_DB.py`, SQL migrations, provider registry, config/defaults, model selection, review recipe or UI. Their existing contracts must be exercised. Shared parser consumers requiring targeted round-trip/privacy checks, not speculative source edits: `Sync_Interop/envelope_builder.py`, `envelope_applier.py`, `Chatbooks/chatbook_creator.py`, `chatbook_importer.py`, `Character_Chat/Character_Chat_Lib.py`, `console_history_budget.py`, `console_trace_service.py`, `console_trace_final_values.py`.

Tests/fixtures to update only where these contracts are exercised:

- `Tests/Chat/test_provider_continuation.py`, `test_provider_continuation_history.py`, `test_provider_continuation_privacy.py`.
- `Tests/Chat/test_console_library_destination.py` and `Tests/Chat/test_console_turn_preparation.py` for detached context assertions (no whole-directory run).
- `Tests/Chat/test_console_dispatch_continuation_handoff.py`, `test_console_dispatch_recovery.py`; `Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py`.
- `Tests/Chat/test_console_provider_continuation.py`, `test_console_provider_gateway.py`, `test_provider_continuation_crash_recovery.py`; add the actual custom Console two-turn/reopened cases to the existing handoff test owner.
- `Tests/Plugins/test_continuation.py`; `Tests/Agents/test_provider_continuation_runtime.py` only targeted unchanged runtime contracts/fixture compatibility.
- `Tests/ChaChaNotesDB/test_provider_continuation.py`, `Tests/Sync_Interop/test_provider_continuation_reconciliation.py`, `Tests/Chatbooks/test_provider_continuation_roundtrip.py`; selected trace continuation round-trip assertions in `Tests/Chat/test_console_trace_transform_continuations.py`.
- `Tests/console_provider_doubles.py` already delegates to the real destination classifier. Preserve that; update explicit continuation-capable fixtures with real execution/protocol fields rather than faking full bases into the disclosure origin. Existing neutral provider fixtures stay V1, legacy dispatch fixtures remain explicitly legacy.

Documentation: amend ADR-063/147 with text below, cross-reference in ADR-240/task/plan appendix, update `Docs/Architecture/console.md` and `Docs/Architecture/llm-providers.md` for the two identities and legacy refusal. Add a short lesson only for the evidenced origin-vs-base and actual Console persistence gap; do not claim a live successful repair.

## Focused RED/GREEN plan

Run no tests as part of this design pass. Implementation should record specific node IDs and results, under normal `Tests.conftest` real-profile/network guards; no full suite or live requests.

1. RED: extend the current SQLite handoff fixture with the actual raw slug, canonical `custom-hosted`, disclosure origin and `/v1` full base. Prove first ToolBatchReady fails on current source. Add terminal-complete candidate case. Keep origin and full base separate so the second blocker is not masked.
2. RED: two raw slugs A/B with identical execution/model/base; persist A, close/reopen DB/store, request Resume and completed private-history replay under B. Require rejection/omission before provider/tool action; original A succeeds. Include deletion of the dispatch owner before reopen, proving durability of the new pin.
3. Implement codec/target and destination tuple first. Assert V1/V2 byte-canonical fixtures unchanged, valid V3 round trip, malformed/partial/unknown fields refused, wrong canonical pairing/mode/model/base/raw rejected, session-only base absent. Verify frozen context copying and retry refusal after execution/mode/base changes with unchanged origin.
4. Implement host adapter, store, repository and restore owners. Offline actual Console control emits a valid native read-only tool batch and final response, traverses real admission/dispatch/store/repository/runtime/gateway twice, observes tool execution only after committed V3 and dispatch-row deletion, and completes. Deny sockets/DNS; intercept only the same bounded transport seam as the diagnosis. Do not replace permission/persistence/conflict guards. A separate interrupted pending-call fixture reopens and explicitly resumes through the real controller; no automatic recovery dispatch.
5. Negative matrix on existing owners: foreign assistant/conversation, stale user/assistant/checkpoint version, changed raw slug with same endpoint, changed protocol/family/model/base path, tampered pin, partial tuple, V1 custom-hosted legacy, and duplicate batch/final event. All leave correct owner/recovery and zero unintended side effects. Direct canonical built-in V1 resume still works; new built-in V3 works; explicitly routed built-in child under custom parent is independently resolved and cannot consume the parent's private groups.
6. Managed tests: existing V2 signatures verify unchanged; V3 managed seal covers raw slug and all body fields; every event reseals; altered slug/revision/owner refuses; missing plugin service refuses envelope-bearing V3; V3/null keeps zero-managed ceiling; failed sealing/persistence never downgrades or executes work.
7. Targeted DB/sync/chatbook/private-history/trace round trips prove V3 survives whole-record carriage, branch ownership and private projections; no new field appears in normal logs/FTS/text export. Run selected existing crash/recovery and runtime revision/CAS tests touched by changed fixtures. Ruff/formatter only changed files. No unrelated provider matrix or whole-directory suite.

GREEN means this offline identity contract and actual Console persistence path work. It does not mean AC24 live review qualification passed. Retain the historical diagnostic and failed live attempt as separate evidence.

## Rejected alternatives

- Relabel checkpoint.provider to `custom-ep:*`: violates closed pairings and execution replay/service contracts.
- Accept all raw custom/family aliases or only compare URL/model: lets virtual endpoints and explicit built-in children interchange.
- Read current registry at handoff/restore: mutable settings cannot establish accepted authority.
- Store-only pin: runtime retains provider candidate and loses it on its next private request; first successful write is insufficient.
- Put raw pin in local metadata/AgentRuns/session or retain dispatch rows after handoff: wrong portable owner, second ownership system, or existing exclusive-ownership violation.
- Add a companion SQL column/table: unnecessary; existing private JSON already follows message version/hash/sync/fork/variant ownership.
- Reuse managed V2 without a format branch: V2 implies a required MAC envelope, and its current digest projection erases new fields.
- Change disclosure origin into full API base: confuses egress/UI semantics and still leaves execution/mode identity missing.

## Exact proposed ADR amendments

### Append to ADR-063

> ### Selected destination continuity amendment (TASK-35582, 2026-10-27)
>
> Continuation `provider` remains the canonical execution-family identifier. Console separately binds the exact raw selected provider from the host-resolved request. Before durable acceptance, the existing resolved destination freezes canonical continuation provider, protocol and full API base alongside its raw selection and model. Its disclosure endpoint remains an origin and is not replay authority. Nested destination JSON V2 carries these additional facts; legacy four-key destinations remain readable but cannot authorize a new native handoff without them.
>
> The host adapter binds neutral provider candidates before the agent loop retains them. The existing store validates that binding against captured run authority and the durable assistant owner before managed sealing. Both dispatch-to-continuation handoff and complete terminal settlement independently compare the same frozen mapping under their existing owner/version transaction. Successful handoff transfers the raw pin into assistant-owned continuation JSON V3 before deleting dispatch ownership. Provider data, current endpoint configuration, execution-family aliases and message bodies cannot mint or replace that mapping.
>
> V3 adds an exact selected-provider field to the existing canonical execution tuple and admits either no managed envelope or the existing envelope. Managed V3 authentication covers the selected identity; legacy V2 keeps its existing V1 digest projection and bytes. A managed envelope's presence, rather than equality to format version 2, requires managed authority validation. Failed binding, sealing or persistence never downgrades or permits side effects.
>
> Host history/recovery checks require the selected identity as well as the existing canonical provider/protocol/model/base contract. Two virtual endpoint selections sharing a model and API base are distinct; explicit built-in routing remains distinct from a custom parent. Lower provider replay validates only canonical execution identity after host admission and remains configuration-neutral. Legacy direct-provider V1/V2 continuation remains supported; pinless virtual continuation fails closed because its original raw owner cannot be proved. Old accepted dispatch rows are not upgraded from mutable configuration. Visible history and explicit discard/new-work remain available.
>
> These changes version JSON inside the existing columns and require no SQL migration or second durable owner. Session-only endpoints remain omitted from durable authority and cannot acquire durable native continuation through this amendment. Existing permissions, explicit Resume/takeover, ambiguous executing-call handling, message CAS/hash/sync ownership, private-data projections and durability barriers remain unchanged.

### Append to ADR-147

> ### Durable primary selection identity amendment (TASK-35582, 2026-10-27)
>
> The primary raw selected provider and canonical execution key remain separate through durable Console continuation, not only model-call routing. The host freezes their exact relationship, protocol, model and API base before accepted dispatch and carries the raw selection with the assistant continuation after dispatch ownership is removed, as specified in ADR-063. Neither a shared execution family nor an equal URL/model makes two `custom-ep:<slug>` selections interchangeable.
>
> Candidate binding and request validation use each call's actual resolved destination. A routed child uses its own raw identity and execution tuple and cannot inherit its parent's private continuation pin or history. The canonical checkpoint provider is never relabeled to a raw slug. Current registry aliases, mutable session defaults and message content are not accepted authority; an unprovable legacy virtual binding is refused rather than inferred. This adds no route registry, changes no child budget and grants no execution or destination permission.

## Plan appendix / implementation brief requirements

Root should record the above ADR amendment and append a scoped continuation-identity section to `Docs/superpowers/plans/2026-10-18-multi-model-code-review-implementation.md` and TASK-35582 before implementation. Keep it distinct from review-job continuation and AC23's prior evidence. Include the two mismatch evidence paths, the 14-file scope, explicit V1/V2/V3 compatibility table, destination provenance rule, managed-signature ordering, focused RED/GREEN gates, and the no-SQL-migration rationale. If a fixture or source invariant contradicts this design, stop and amend the contract; do not hide it behind broad aliases, weaker validation or automatic legacy upgrades.

The implementation brief must explicitly require the actual Console second request and reopened same-slug/different-slug checks; a green pure matcher is insufficient. Review should compare serialized accepted destination, host-bound runtime candidate, stored V3, managed unsigned body and restored host target from one offline control. Archive metadata only under existing privacy rules.

Live allowance remains exactly as supplied by root: 71 historical requests consumed, 39 approved requests unspent, three child launches unused. The approved one-journey claim remains consumed. This design and any offline fix authorize no paid call or retry; another live journey requires manual user authorization after concrete fix review. AC24 remains unchecked.
