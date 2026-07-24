# RAG Citation Provenance Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the benchmark, canonical contracts, governed persistence, lifecycle, artifact ownership, and restart-safe legacy migration required before any RAG producer or citation UI writes canonical traces.

**Architecture:** Add frozen Pydantic contracts beside the existing compatibility models, validate source locators through a static data-only registry, and persist only sealed aggregates in the same ChaChaNotes SQLite transaction as their owning message. Governed payloads stay outside immutable trace JSON. Mutable observations, revocation tombstones, owner links, artifact leases, and migration journals remain separate repository concerns. Existing sidecars stay readable; once canonical writes are enabled they become legacy-only inputs, while the disabled recovery mode preserves pre-cutover compatibility behavior without dual-writing.

**Tech Stack:** Python ≥3.11, Pydantic v2, SQLite/FTS5, Textual service wiring, pytest, Hypothesis, stdlib `hashlib`/`hmac`, existing atomic JSON helpers.

**Spec:** `Docs/superpowers/specs/2026-07-23-rag-citation-provenance-design.md`

**Backlog:** `TASK-401`, with foundation children `TASK-401.1` through `TASK-401.9`

**ADR required:** yes

**ADR path:** `backlog/decisions/024-rag-citation-provenance-and-source-resolution.md`

**Reason:** This plan implements the accepted storage, identity, governance, migration, artifact-ownership, and cross-module contract decisions in ADR-024.

---

## Scope boundary

This plan is the first independently shippable subsystem of the citation epic. It intentionally does **not**:

- capture local retrieval runs or prompt-boundary evidence
- seal a trace from a live generation path
- alter streaming, repair, answer rendering, or Console/Library UI
- implement current-source resolver families or open external destinations
- add the server-owned `grounding_trace/v1` producer or client adapter
- add portable provenance export/import, Sync v2 transport, or release qualification

Those are separate plans because they cross different runtime, UI, and server ownership boundaries. This foundation may ship dormant: canonical readers and repositories are available, but no existing answer becomes “Grounded” until a later producer task supplies a coherent sealed aggregate.

## Repository constraints

- Execute each Backlog child as one reviewable PR/commit series; do not combine the nine children into one code commit.
- Use a clean worktree from the current `dev` base. The shared checkout is dirty and must not be reset or cleaned.
- Before starting each child, mark it `In Progress`, add its task-local implementation plan through Backlog.md, and link ADR-024. Mark it `Done` only after its acceptance criteria, notes, tests, lint, and docs are complete.
- Run commands from the repository root with the project environment activated:

```bash
source .venv/bin/activate
```

- At the approved baseline, ChaChaNotes is schema v25. Immediately before `TASK-401.4`, rebase and reserve the next free version. Use v25→v26 only if v25 is still current; rename the SQL file and migration symbols consistently if another migration lands first.
- New provenance tables are local-only in this plan. Do not add them to existing sync triggers, FTS tables, Library indexing, or RAG ingestion.
- Keep `EvidenceReference`, `EvidenceBundle`, `CitationRef`, current server citation arrays, and `chat_rag_context.json` readable. Do not mutate them into falsely complete traces.
- Store no raw query, answer, snapshot, title, source identity, locator, lineage, content hash, or comparison fingerprint in logs or immutable aggregate JSON.
- Repository methods accept a `CitationFingerprintCodec` supplied by a runtime-owned key provider. This plan adds the provider seam and the production keyring adapter; it must never fall back to an unkeyed portable digest.
- `[rag_citations].canonical_writes_enabled` defaults to `false` in this foundation. When false, canonical reads remain available, while repository writes, legacy migration scheduling, and artifact reconciliation fail closed or remain dormant.
- Every bounded JSON field must reject oversize input before a transaction begins. Do not silently truncate immutable provenance structure.

## Foundation file map

| Path | Responsibility |
| --- | --- |
| `Tests/fixtures/rag_citation_provenance/manifest_v1.json` | Deterministic corpus, fixture IDs, sizes, and expected benchmark cases |
| `Tests/fixtures/rag_citation_provenance/corpus_v1.json` | Synthetic, non-sensitive local evidence and answer bodies |
| `Helper_Scripts/Benchmarks/rag_citation_provenance_benchmark.py` | Reproducible local baseline/qualification runner |
| `Docs/Development/RAG/citation-provenance-benchmark-v1.md` | Environment, commands, results, and numeric budgets |
| `Docs/Development/RAG/citation-provenance-baseline-v1.json` | Committed machine-readable baseline used by qualification mode |
| `tldw_chatbook/Chat/citation_trace_models.py` | Frozen canonical trace and governed write-bundle contracts |
| `tldw_chatbook/Chat/citation_trace_adapters.py` | Pure legacy `EvidenceBundle`/`CitationRef` adapters |
| `tldw_chatbook/Chat/citation_trace_identity.py` | Namespaces, stable runtime identity/key seams, opaque IDs, and keyed fingerprints |
| `tldw_chatbook/Chat/citation_provenance_runtime.py` | Shared read/write recovery policy derived from config |
| `tldw_chatbook/Chat/citation_source_locators.py` | Typed inert locator envelope, capabilities, and static inventory |
| `Tests/fixtures/rag_citation_provenance/source_inventory_v1.json` | Versioned local/server source-kind classification |
| `tldw_chatbook/Chat/citation_trace_repository.py` | SQLite aggregate reads/writes and owner operations |
| `tldw_chatbook/Chat/citation_payload_lifecycle.py` | Revocation, tombstones, retention decisions, and GC |
| `tldw_chatbook/Chat/citation_artifact_ownership.py` | Cross-store outbox/lease reconciliation |
| `tldw_chatbook/Chat/citation_legacy_migration.py` | Legacy synthesis, migration journal, and divergence detection |
| `tldw_chatbook/DB/migrations/chachanotes_v25_to_v26_citation_provenance.sql` | Baseline migration name; renumber if needed |

## Dependency order

```text
TASK-401.1 benchmark
  -> TASK-401.2 trace + identity contracts
    -> TASK-401.3 locator contracts
      -> TASK-401.4 schema + sealed transaction
        -> TASK-401.5 idempotency + body binding
          -> TASK-401.6 revocation + retention + GC
          -> TASK-401.7 current observations
          -> TASK-401.8 artifact owner handshake
          -> TASK-401.9 legacy migration
```

`TASK-401.2` defines pure namespace and fingerprint behavior before schema work. `TASK-401.4` then persists the stable local identity context and consumes those contracts. `TASK-401.7` waits for `401.5` because observation keys use the complete `TraceNamespace`. `TASK-401.8` and `401.9` wait for lifecycle behavior because leases and migrated payloads must participate in collection and anti-resurrection rules.

## Frozen v1 bounds

These are contract limits, not benchmark suggestions. Pydantic validation and repository entry points reject larger values before a transaction:

| Value | Maximum |
| --- | ---: |
| Immutable aggregate JSON | 256 KiB per trace |
| Governed evidence snapshot text | 64 KiB UTF-8 per snapshot |
| Governed payload bytes | 4 MiB per trace |
| Selected and diagnostic prompt sets | 8 per trace |
| Evidence entries | 64 per prompt set |
| Answer attempts | 8 per trace |
| Citation occurrences | 512 per selected answer |
| Retrieval candidates recorded per run | 200 |
| Locator envelope JSON | 16 KiB |
| Current-source observation JSON | 8 KiB |
| Sanitized error/reason code | 256 characters |
| External opaque identifier | 256 UTF-8 bytes |
| Governed answer-attempt body | 1 MiB UTF-8 |
| Legacy sidecar input file | 32 MiB |
| Legacy migration batch | 100 messages |

The benchmark corpus contains exact-limit cases and one-unit-over rejection cases for every byte/count boundary, including 64 KiB snapshots, 4 MiB total governed payloads, and 256 KiB aggregate JSON.

## Stable identity and secret ownership

- `TraceNamespace` is part of `TASK-401.2`, not deferred until after persistence. It includes the local profile, origin namespace, authority, optional authenticated tenant, and wire version needed by ADR-024.
- Local persistence uses a singleton `rag_identity_context` row containing random 128-bit `profile_id`, `local_authority_id`, and `fingerprint_key_id`. These survive restart and DB copy. They are never derived from the existing inconsistently-used `client_id`.
- `CitationFingerprintKeyProvider` loads the key identified by `fingerprint_key_id`. The production adapter follows the repository's existing keyring abstraction under service `tldw_chatbook.citation-provenance.v1`; tests inject an in-memory provider. It may provision a new secret only when no fingerprint-bearing canonical row exists. A copied/restored DB whose prior key is missing fails closed rather than silently replacing the key.
- If the key is absent or keyring is unavailable, canonical reads still work, but writes, migration, and reconciliation fail closed with `fingerprint_key_unavailable`. There is no public/default digest.
- Server namespaces are constructed only from the authenticated server profile/connection authority plus tenant/principal/workspace context. This foundation defines and tests the constructor but does not persist server traces or acquire server credentials.

## Exact v1 persistence contract

The migration may use the next free schema number, but it must create this logical schema without leaving later lifecycle tasks to invent columns. IDs are bounded `TEXT`, booleans are `INTEGER CHECK (... IN (0,1))`, ordinals/counts are non-negative `INTEGER`, and timestamps are UTC ISO-8601 `TEXT`; named JSON/text columns carry the frozen bounds above.

| Table | Required columns, keys, and policy |
| --- | --- |
| `rag_identity_context` | `context_name` PK (`CHECK context_name='default'`), unique non-null `profile_id`, `local_authority_id`, `fingerprint_key_id`, and non-null `created_at`. |
| `rag_citation_traces` | PK `(profile_id, trace_id)`; non-null `schema_version`, `request_id`, `generation_id`, `origin_scope_id`, `origin CHECK IN ('local','server','imported','legacy_inferred')`, `lifecycle CHECK IN ('sealed')`, `completeness_at_seal CHECK IN ('complete','partial','redacted','unavailable')`, `selected_attempt_id`, `policy_version`, `aggregate_json`, `visibility_state CHECK IN ('migrating','active')`, `created_at`, `sealed_at`; nullable `connection_authority_id`, `tenant_id`, `server_trace_id`, `wire_schema_version`, `import_package_fingerprint`, `external_trace_id`, `legacy_conversation_id`, `legacy_message_id`. One table `CHECK` enforces exactly one coherent origin shape: local requires `origin_scope_id=profile_id` and all external/import/legacy fields null; server requires authority, server trace, wire version, and non-null normalized `origin_scope_id` (authenticated tenant/principal/workspace ID or literal `authority-root`) while import/legacy fields are null; imported requires package fingerprint/external trace and no server/legacy fields; legacy requires conversation/message IDs and no server/import fields. Partial unique indexes use null-safe non-null tuples `(connection_authority_id, origin_scope_id, server_trace_id, wire_schema_version) WHERE origin='server'` and `(profile_id, import_package_fingerprint, external_trace_id) WHERE origin='imported'`. |
| `rag_evidence_runs` | PK `(profile_id, trace_id, run_id)`; non-null `run_ordinal`, `stage`, `redaction_state`, `started_at`, nullable governed `run_payload_json`, nullable `ended_at`, `purged_at`; `CHECK` requires payload present when available and null when purged; trace FK `(profile_id, trace_id) ON DELETE CASCADE`; unique `(profile_id, trace_id, run_ordinal)`. |
| `rag_evidence_snapshots` | PK `(profile_id, payload_id)`; non-null `governance_scope_id` (local profile or authenticated server tenant/authority-root), `authority_id`, `confidentiality_policy_id`, `revocation_scope_id`, `origin_namespace`, `origin_payload_id`, `storage_mode`, `redaction_state`, `retention_class`, `created_at`; governed nullable `snapshot_text`, `title`, `source_identity_json`, `locator_json`, `lineage_json`, `transformations_json`, `content_hash`, `comparison_fingerprint`; nullable `retain_until`, `purged_at`. Partial unique dedupe index `(governance_scope_id, authority_id, confidentiality_policy_id, revocation_scope_id, content_hash) WHERE content_hash IS NOT NULL`; no nullable namespace component participates in uniqueness. |
| `rag_answer_attempt_payloads` | PK `(profile_id, payload_id)`; non-null `trace_id`, `attempt_id`, `redaction_state`, `retention_class`, `created_at`; nullable governed `answer_body`, `body_integrity_hmac`, `retain_until`, `purged_at`; `CHECK` requires body/integrity present when available and null when purged; trace FK `(profile_id, trace_id) ON DELETE CASCADE`; unique `(profile_id, trace_id, attempt_id)`. |
| `rag_trace_evidence_refs` | PK `(profile_id, trace_id, prompt_set_id, evidence_ordinal)`; non-null `run_id`, `snapshot_payload_id`, `marker_ordinal`, `storage_mode`; trace/run FKs `ON DELETE CASCADE`, snapshot FK `ON DELETE RESTRICT`; unique `(profile_id, trace_id, prompt_set_id, marker_ordinal)`. |
| `rag_message_trace_owners` | PK `(profile_id, message_id, message_revision, trace_id)`; non-null `state CHECK IN ('active','body_mismatch','deleted')`, `body_fingerprint`, `idempotency_key`, `created_at`, `updated_at`; `message_id` uses the exact `messages.id` type and FK `ON DELETE CASCADE`; trace FK `ON DELETE RESTRICT`; unique `(profile_id, idempotency_key)` and partial unique `(profile_id, message_id, message_revision) WHERE state='active'`. |
| `rag_source_observations` | PK `(profile_id, trace_id, prompt_set_id, evidence_ordinal, snapshot_payload_id, resolver_kind, resolver_version)`; non-null independent `availability`, `permission_state`, `content_state`, `location_state`, bounded `capabilities_json`, `request_nonce`, `observed_at`; nullable bounded `error_code`; trace and snapshot FKs `ON DELETE CASCADE`; compare-and-replace retains one row only. Prompt-local ordinals can never collide across reruns. |
| `rag_payload_tombstones` | PK `(profile_id, origin_namespace, origin_payload_id)`; non-null `revocation_scope_id`, `reason_code`, `policy_version`, `revoked_at`, `retain_until`; no governed content columns. |
| `rag_artifact_owner_leases` | Stable-owner PK `(profile_id, artifact_store_id, artifact_id, artifact_revision, trace_id)`; unique non-null `lease_id`; non-null `state CHECK IN ('link_pending','live','unlink_pending','released')`, `created_at`, `updated_at`; nullable `retain_until`; trace FK `ON DELETE RESTRICT`. Every state except `released` is a GC barrier. |
| `rag_artifact_owner_operations` | PK `(profile_id, operation_id)`; non-null stable owner tuple, `operation_kind CHECK IN ('link','unlink')`, `state CHECK IN ('pending','applied','acknowledged')`, `created_at`, `updated_at`; FK to the stable lease `ON DELETE RESTRICT`; unique `(profile_id, artifact_store_id, artifact_id, artifact_revision, trace_id, operation_kind)` so each lease has at most one link and one unlink operation. Applied operation rows remain as bounded idempotency receipts for the lease lifetime. |
| `rag_legacy_migration_journal` | PK `(profile_id, conversation_id)`; non-null `source_fingerprint`, `state CHECK IN ('pending','running','complete','failed','diverged')`, `attempt_count`, `started_at`, `updated_at`; nullable `next_message_cursor`, bounded `error_code`, `completed_at`; conversation FK `ON DELETE CASCADE`. It never stores an unbounded list of created trace IDs. |

Repository `CHECK`s and model validation use the same enum and size constants. Governed rows may be deleted or nulled only through lifecycle methods; direct trace deletion is blocked while an owner or artifact lease exists.

---

### Task 1: Publish the prerequisite benchmark (`TASK-401.1`)

**Files:**

- Create: `Tests/fixtures/rag_citation_provenance/manifest_v1.json`
- Create: `Tests/fixtures/rag_citation_provenance/corpus_v1.json`
- Create: `Helper_Scripts/Benchmarks/rag_citation_provenance_benchmark.py`
- Create: `Tests/Performance/test_rag_citation_provenance_benchmark.py`
- Create: `Docs/Development/RAG/citation-provenance-benchmark-v1.md`
- Create: `Docs/Development/RAG/citation-provenance-baseline-v1.json`
- Exercise by symbol: `ConsoleChatController.submit_draft` and `ConsoleChatController._stream_assistant_response` in `tldw_chatbook/Chat/console_chat_controller.py`
- Reuse tests near: `Tests/Chat/test_console_chat_controller.py`

**Interfaces:**

- Runner command:
  `python Helper_Scripts/Benchmarks/rag_citation_provenance_benchmark.py --mode baseline --samples 30 --warmups 5 --output Docs/Development/RAG/citation-provenance-baseline-v1.json`
- Qualification command requires:
  `--baseline Docs/Development/RAG/citation-provenance-baseline-v1.json`
- JSON result keys:
  `environment`, `fixture_version`, `samples`, `warmups`, `metrics`, `budgets`, `external_network`
- No network access in the local benchmark mode.
- Qualification verifies fixture/schema versions and the documented environment envelope, compares candidate work with the unchanged no-provenance control path in-process, and uses the committed v1 result as the historical reference. An incompatible environment may emit measurements but cannot claim a passing qualification.

- [ ] **Step 1: Put `TASK-401.1` in progress and record its task-local plan**

```bash
backlog task edit 401.1 -s "In Progress"
backlog task edit 401.1 --plan "1. Add deterministic fixtures and manifest\n2. Add a network-free benchmark runner\n3. Record baseline results and numeric budgets\n4. Add reproducibility tests and documentation"
```

- [ ] **Step 2: Write failing manifest and runner contract tests**

Cover:

- a versioned manifest whose referenced files and SHA-256 digests exist
- deterministic fixture IDs and stable byte counts
- exactly 5 warmups and at least 30 measured samples by default
- an isolated temporary ChaChaNotes DB and sidecar
- rejection of an external URL/provider in `--mode baseline`
- machine-readable median and p95 results
- all six required budget families
- qualification mode refusing to run without a compatible committed baseline

Run:

```bash
python -m pytest Tests/Performance/test_rag_citation_provenance_benchmark.py -v
```

Expected: FAIL because the fixture manifest and benchmark module do not exist.

- [ ] **Step 3: Add the deterministic synthetic corpus**

Use only committed synthetic text. Include:

- media, note, and conversation source examples
- 1, 8, 32, and 64 submitted-evidence shapes
- cited and additionally supplied chunks
- Unicode, repeated markers, grouped markers, and a repaired-answer body
- embedded, server-reference descriptor, ephemeral, and redacted storage cases
- legacy `EvidenceBundle`, `CitationRef`, and sidecar records
- exact-limit and one-unit-over cases for every frozen v1 count and byte bound

The manifest records fixture schema version, file digests, character/byte counts, expected source kinds, and expected answer/evidence cardinalities.

- [ ] **Step 4: Implement the network-free runner**

Measure current/proxy seams independently:

1. mocked native Console time-to-first-token through `ConsoleChatController.submit_draft` and `_stream_assistant_response`, using a deterministic gateway/provider stream
2. post-stream bounded serialization/finalization proxy
3. cold and warm stored-inspector read proxy
4. aggregate JSON and governed payload bytes
5. SQLite file growth after message/evidence-shaped writes
6. legacy conversation scan/migration proxy, including interruption and restart

Use `time.perf_counter_ns`, force a fresh temporary DB per sample group, run `PRAGMA wal_checkpoint(TRUNCATE)` before file-size measurements, and report median/p95 rather than minimums. External resolution gets a separate optional mode and never participates in local pass/fail.

- [ ] **Step 5: Record these delivery budgets in the report and result schema**

| Metric | Budget |
| --- | --- |
| Mocked first-token regression | p95 increase ≤ 10% **and** ≤ 25 ms over the recorded v1 baseline |
| Standard finalization (8 × 4 KiB snapshots) | p95 ≤ 75 ms |
| Maximum finalization (exactly 4 MiB total governed payload across ≤64 snapshots) | p95 ≤ 250 ms |
| Inspector initial local load | cold p95 ≤ 100 ms; warm p95 ≤ 25 ms |
| Immutable aggregate JSON | ≤ 256 KiB per trace |
| Governed payload | ≤ 64 KiB per snapshot and ≤ 4 MiB per trace fixture |
| SQLite growth | ≤ governed payload bytes × 1.35 + 256 KiB per grounded answer |
| Legacy migration | ≥ 100 messages/second on the documented reference machine; restart produces zero duplicate canonical rows |

If the measured pre-feature machine cannot satisfy a proposed budget, stop and amend the spec/task with evidence rather than weakening the test silently.

- [ ] **Step 6: Run and publish the baseline**

```bash
python Helper_Scripts/Benchmarks/rag_citation_provenance_benchmark.py \
  --mode baseline \
  --samples 30 \
  --warmups 5 \
  --output Docs/Development/RAG/citation-provenance-baseline-v1.json
python -m pytest Tests/Performance/test_rag_citation_provenance_benchmark.py -v
```

Copy the non-sensitive results, exact command, Python/SQLite versions, CPU/OS, sample rules, and limitations into `Docs/Development/RAG/citation-provenance-benchmark-v1.md`. Commit the machine-readable result beside it. Neither artifact may contain an absolute user path.

- [ ] **Step 7: Complete the Backlog task and commit**

```bash
git add Tests/fixtures/rag_citation_provenance/manifest_v1.json \
  Tests/fixtures/rag_citation_provenance/corpus_v1.json \
  Helper_Scripts/Benchmarks/rag_citation_provenance_benchmark.py \
  Tests/Performance/test_rag_citation_provenance_benchmark.py \
  Docs/Development/RAG/citation-provenance-benchmark-v1.md \
  Docs/Development/RAG/citation-provenance-baseline-v1.json
git commit -m "test(rag): establish citation provenance baseline"
```

Update all `TASK-401.1` acceptance criteria, add the recorded results to Implementation Notes, and mark it `Done`.

---

### Task 2: Define canonical trace contracts (`TASK-401.2`)

**Files:**

- Create: `tldw_chatbook/Chat/citation_trace_models.py`
- Create: `tldw_chatbook/Chat/citation_trace_adapters.py`
- Create: `tldw_chatbook/Chat/citation_trace_identity.py`
- Create: `Tests/Chat/test_citation_trace_models.py`
- Create: `Tests/Chat/test_citation_trace_adapters.py`
- Create: `Tests/Chat/test_citation_trace_identity.py`
- Modify only for compatibility imports if required: `tldw_chatbook/Chat/citation_evidence_models.py`

**Required public contracts:**

- frozen enums/models for origin, lifecycle, completeness, storage mode, marker namespace, claim support, and policy capability
- `EvidenceRun`, `PromptEvidenceEntry`, `PromptEvidenceSet`, `CitationOccurrence`, `AnswerAttempt`, `CitationTrace`
- governed `EvidenceRunPayload`, `EvidenceSnapshotPayload`, `AnswerAttemptPayload`
- `SealedCitationWrite` containing one sealed trace plus all referenced governed payloads
- `reduce_selected_attempt_completeness(trace, payload_index)`
- pure legacy synthesis functions returning `origin=legacy_inferred` and never `complete`
- `TraceNamespace`, `LocalCitationIdentityContext`, and server/import namespace constructors
- `new_opaque_id(prefix) -> str` backed by 128 random bits
- `CitationFingerprintCodec(secret: bytes)` using domain-separated, length-framed HMAC-SHA-256
- `CitationFingerprintKeyProvider` protocol and an existing-keyring-compatible production adapter

- [ ] **Step 1: Add failing round-trip and invariant tests**

Test:

- strict schema versions and `extra="forbid"`
- frozen models and deterministic JSON
- exactly one selected attempt
- every selected attempt references an existing prompt set
- every prompt entry references an existing run and governed payload descriptor
- prompt marker ordinals are unique, positive, and stable
- Unicode codepoint offsets and `chatbook_s_v1` marker grammar
- selected-attempt-only mixed storage reduction
- no governed fields in `CitationTrace.model_dump_json()`
- bounded JSON metadata, prompt sets, attempts, occurrences, and references
- exact-limit acceptance and one-unit-over rejection for every frozen v1 bound

Run:

```bash
python -m pytest Tests/Chat/test_citation_trace_models.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 2: Implement the smallest frozen model graph**

Keep immutable trace JSON limited to:

- opaque IDs and relationships
- ordinals and marker mappings
- storage-policy class and opaque payload refs
- structural/semantic validation summaries
- timestamps, policy version, and bounded timing summaries

Keep submitted text, source/title/locator/lineage, hashes, raw queries, and non-final answer text exclusively in governed payload models. `SealedCitationWrite` validates the complete cross-reference graph before repository code sees it.

- [ ] **Step 3: Implement deterministic completeness reduction**

Use only the final selected attempt and its prompt set:

```text
unavailable > redacted > partial > complete
```

An empty or inconsistent final set is unavailable. Non-final prompt sets remain diagnostic and cannot downgrade the selected answer.

- [ ] **Step 4: Add property and bound tests**

Use Hypothesis to permute non-final attempts, prompt-set ordering, mixed storage modes, repeated occurrences, and unknown markers. Assert:

- the reduction result is stable
- round trips preserve canonical structure
- adding a non-final attempt cannot change selected completeness
- aggregate serialization never acquires governed field names
- oversize metadata is rejected rather than truncated

- [ ] **Step 5: Add pure legacy adapters**

Adapt existing `EvidenceBundle`/`CitationRef` payloads without changing their source module. Missing prompt-boundary evidence, unknown authority, or legacy numeric markers always produce partial or unavailable `legacy_inferred` traces.

Run:

```bash
python -m pytest \
  Tests/Chat/test_citation_trace_models.py \
  Tests/Chat/test_citation_trace_adapters.py \
  Tests/Chat/test_citation_evidence_models.py \
  Tests/Chat/test_answer_citations.py -v
```

Expected: PASS.

- [ ] **Step 6: Add failing pure identity, key-provider, and fingerprint tests**

Test:

- 128-bit random opaque IDs and bounded prefixes/external IDs
- namespace separation across local profiles, authorities, authenticated server tenants, and wire versions
- explicit UTF-8 length framing and distinct HMAC domains for message body, raw query, exact payload, owner operation, and legacy source
- rejection of an empty secret and failure-closed behavior when the key provider cannot load `fingerprint_key_id`
- no key, raw value, or portable raw digest in serialized trace/owner metadata

Run:

```bash
python -m pytest Tests/Chat/test_citation_trace_identity.py -v
```

Expected: FAIL because the identity module does not exist.

- [ ] **Step 7: Implement the pure identity contracts**

Implement only pure constructors, explicit length-framed HMAC, the injectable provider protocol, and the keyring adapter. Do not create database rows or acquire the key during module import. Re-run the identity tests and expect PASS.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Chat/citation_trace_models.py \
  tldw_chatbook/Chat/citation_trace_adapters.py \
  tldw_chatbook/Chat/citation_trace_identity.py \
  tldw_chatbook/Chat/citation_evidence_models.py \
  Tests/Chat/test_citation_trace_models.py \
  Tests/Chat/test_citation_trace_adapters.py \
  Tests/Chat/test_citation_trace_identity.py
git commit -m "feat(rag): define canonical citation trace contracts"
```

Complete `TASK-401.2` hygiene before proceeding.

---

### Task 3: Define inert source locators and inventory (`TASK-401.3`)

**Files:**

- Create: `tldw_chatbook/Chat/citation_source_locators.py`
- Create: `Tests/Chat/test_citation_source_locators.py`
- Create: `Tests/fixtures/rag_citation_provenance/source_inventory_v1.json`
- Read for producer alignment: `tldw_chatbook/RAG_Search/ingestion_indexing.py`

**Required public contracts:**

- `SourceLocatorEnvelope`
- `SourceCapabilityPolicy`
- `SourceInventoryEntry`
- `CitationReadAuthorization` with caller scope, allowlisted authorities, and independent view/resolve/export capabilities
- `LocatorBindingState` with at least `native`, `inert_imported`, and `inert_legacy`
- `validate_native_locator(...)`
- `parse_inert_locator_candidate(...)`
- immutable `SOURCE_INVENTORY_V1`

- [ ] **Step 1: Add failing hostile-input and inventory tests**

Cover:

- unknown envelope/resolver payload versions
- extra `class`, `module`, `command`, absolute-path, handler, and URL-fetch fields
- authority or tenant mismatch
- cross-profile/governance-scope access and missing `view_snapshot` capability
- imported/legacy locator candidates remaining inert
- independent storage and capability decisions
- SQL always snapshot-only
- claims opening only through authorized parent lineage
- every local producer kind (`media_db`, `notes`, `chat_history`)
- explicit current-runtime mapping `media -> media_db`, `note -> notes`, and `conversation -> chat_history`
- every pinned server kind (`character_cards`, `web_content`, `prompts`, `world_books`, `dictionaries`, `sql`, `kanban`, `claims`)

Run:

```bash
python -m pytest Tests/Chat/test_citation_source_locators.py -v
```

Expected: FAIL because the locator module and inventory fixture do not exist.

- [ ] **Step 2: Add data-only locator payload models**

The envelope may select only a canonical source kind and version. It cannot select Python code. Payload models contain bounded opaque IDs, authority/tenant binding, and typed location hints. Local file-backed note payloads use source-root ID plus relative path; no absolute path becomes a native locator.

- [ ] **Step 3: Add the static inventory**

Define `RUNTIME_SOURCE_KIND_TO_CANONICAL_V1` in code as the single source of truth for current local producer mapping:

```text
media        -> media_db
note         -> notes
conversation -> chat_history
```

Load/validate the committed fixture at test time as a contract snapshot of that runtime constant plus the pinned server enum reviewed in the spec. The fixture records:

- producer (`local`, pinned server, or derived)
- required identity and authority fields
- locator version
- default capabilities
- snapshot-only conditions
- authoritative parent requirements

Do not add resolver implementations or navigation callbacks in this task.

- [ ] **Step 4: Add explicit rebinding semantics**

Parsing imported/legacy data yields an inert candidate only. The contract may create a native envelope solely from a fresh current-authority lookup plus an explicit rebind decision; it never mutates the historical trace.

`CitationReadAuthorization` is an immutable request-scoped value created by the trusted composition boundary, never from trace/locator data. It contains the local profile or authenticated tenant scope, permitted authority IDs, and independent capability flags. It is the only governed hydration authorization accepted by the repository; callers cannot substitute a boolean `authorized=True`.

Run:

```bash
python -m pytest Tests/Chat/test_citation_source_locators.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/citation_source_locators.py \
  Tests/Chat/test_citation_source_locators.py \
  Tests/fixtures/rag_citation_provenance/source_inventory_v1.json
git commit -m "feat(rag): define governed citation source locators"
```

Complete `TASK-401.3` hygiene.

---

### Task 4: Add schema and atomic sealed persistence (`TASK-401.4`)

**Files:**

- Create: `tldw_chatbook/DB/migrations/chachanotes_v25_to_v26_citation_provenance.sql` (renumber if necessary)
- Modify symbols: schema-version constant, migration helpers, migration dispatcher, and schema initialization in `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Create: `tldw_chatbook/Chat/citation_trace_repository.py`
- Create: `tldw_chatbook/Chat/citation_provenance_runtime.py`
- Modify symbols: `ChatPersistenceService.__init__`, `ChatPersistenceService.create_message`, and `ChatPersistenceService.update_message_content` in `tldw_chatbook/Chat/chat_persistence_service.py`
- Modify: `tldw_chatbook/config.py`
- Create: `Tests/DB/test_chachanotes_citation_provenance_migration.py`
- Create: `Tests/Chat/test_citation_trace_repository.py`
- Modify: `Tests/Chat/test_chat_persistence_service.py`
- Create: `Tests/test_config_rag_citation_defaults.py`
- Modify: `Helper_Scripts/Benchmarks/rag_citation_provenance_benchmark.py`
- Modify: `Tests/Performance/test_rag_citation_provenance_benchmark.py`

- [ ] **Step 1: Rebase, reserve the next schema version, and add failing migration tests**

The migration must create:

- `rag_identity_context`
- `rag_citation_traces`
- `rag_evidence_runs`
- `rag_evidence_snapshots`
- `rag_answer_attempt_payloads`
- `rag_trace_evidence_refs`
- `rag_message_trace_owners`
- `rag_source_observations`
- `rag_payload_tombstones`
- `rag_artifact_owner_leases`
- `rag_artifact_owner_operations`
- `rag_legacy_migration_journal`

Test the exact logical schema above: columns, origin-coherence `CHECK`s, composite/partial unique indexes, FKs and `ON DELETE` behavior. Include duplicate local snapshots with no tenant, duplicate server traces with authority-root scope, missing origin-required identity fields, and forbidden cross-origin fields. Also test fresh DB creation, previous-version upgrade, a forced DDL failure rollback, and the absence of provenance tables/triggers in FTS/sync inventories.

Run:

```bash
python -m pytest Tests/DB/test_chachanotes_citation_provenance_migration.py -v
```

Expected: FAIL because the schema version and tables do not exist.

- [ ] **Step 2: Add one-source, transaction-safe migration execution**

Do **not** copy the v24→v25 `executescript` behavior: this repository has no standalone v24→v25 SQL file, and `sqlite3.Connection.executescript` can implicitly commit an outer transaction.

Use the new standalone SQL file as the single DDL source. Accumulate lines with stdlib `sqlite3.complete_statement` and execute each complete statement with the active migration cursor inside the existing `TransactionContextManager`; the file contains no `BEGIN`, `COMMIT`, version update, or trigger bodies. The migration method:

1. verifies the DB is exactly the previous version
2. executes every DDL statement through the active cursor
3. creates the singleton identity context with SQLite `randomblob(16)`-derived opaque identifiers
4. updates the schema version as the final statement
5. verifies the final version before the outer context commits

Any injected DDL or version-update failure must roll back tables, indexes, identity context, and version together. Use the exact FKs and `ON DELETE` behavior from the persistence table above. Do not add sync/FTS triggers.

- [ ] **Step 3: Add the dormant recovery switch and identity-context/key seam**

Add `[rag_citations].canonical_writes_enabled = false` to defaults and typed config access. `CitationProvenanceRuntimePolicy` is the shared contract later passed to the repository, migration service, and artifact coordinator. In this task, pass it to the repository and add tests proving:

- canonical repository reads do not require the fingerprint key
- every canonical write returns a bounded disabled/key-unavailable result before opening a transaction when the switch is false or the configured key cannot be loaded
- no key is generated or fetched during module import

`TASK-401.8` and `TASK-401.9` separately test that their own reconciliation/migration entry points honor the same policy before those services are marked Done.

When enabled, a composition root loads the 256-bit secret for the row's `fingerprint_key_id` through the injected provider. It may create one only after a repository check proves there are no fingerprint-bearing canonical rows. A missing key beside existing owners, tombstones, migration journal entries, or imported fingerprints is `fingerprint_key_unavailable`, not an implicit rekey. The secret never enters SQLite, logs, config files, trace JSON, or benchmark results.

- [ ] **Step 4: Add failing repository transaction tests**

Build a complete `SealedCitationWrite` fixture and assert one write creates every row. Inject a failure after each row family and assert:

- no message row remains
- no trace/run/snapshot/attempt/ref/owner row remains
- an incomplete or unsealed write is rejected before opening a transaction
- governed fields are absent from `aggregate_json`
- bounded JSON rejection occurs before any row is written
- missing local identity/key context or disabled canonical writes open no transaction
- aggregate-only reads work without a fingerprint key
- governed hydration rejects cross-profile/scope/authority requests, missing `view_snapshot`, redacted rows, and tombstoned origins without returning partial sensitive fields

- [ ] **Step 5: Implement `CitationTraceRepository`**

The repository:

- accepts only validated `SealedCitationWrite`
- writes rows through the caller's active ChaChaNotes transaction
- supports summary and fully authorized hydration reads
- exposes `get_trace_summary(namespace)` without a key and `hydrate_trace(namespace, authorization: CitationReadAuthorization)` for governed data
- verifies profile/governance scope, authority membership, required capability, row redaction state, and tombstones before selecting governed columns; denial returns only the already-safe summary plus a bounded denial state
- exposes no current-source resolution or artifact behavior yet
- accepts the stable `LocalCitationIdentityContext`, `CitationFingerprintCodec`, and runtime policy explicitly

- [ ] **Step 6: Add an optional sealed-write seam to `ChatPersistenceService.create_message`**

Add optional constructor injection `citation_repository: CitationTraceRepository | None = None` and keyword-only `citation_write: SealedCitationWrite | None = None` on `create_message`. The repository owns the stable identity context, codec, and runtime policy. When a citation write is supplied:

- a missing repository, disabled policy, or missing key raises a bounded `CitationPersistenceUnavailable` before any transaction
- there is no silent message-only fallback inside the service
- the caller may explicitly retry the same message without `citation_write` to persist it as ungrounded
- otherwise message, attachments, feedback, trace rows, and message owner association share one outer `self.db.transaction()`

Existing callers constructing `ChatPersistenceService(db)` and passing no citation write preserve byte-identical behavior.

Run:

```bash
python -m pytest \
  Tests/DB/test_chachanotes_citation_provenance_migration.py \
  Tests/Chat/test_citation_trace_repository.py \
  Tests/Chat/test_chat_persistence_service.py -v
```

Expected: PASS.

- [ ] **Step 7: Re-run the benchmark storage proxies**

Add the sealed repository write/read candidate to qualification mode while retaining the unchanged message-only control. Do not rewrite the committed baseline.

```bash
python Helper_Scripts/Benchmarks/rag_citation_provenance_benchmark.py \
  --mode qualification \
  --baseline Docs/Development/RAG/citation-provenance-baseline-v1.json \
  --samples 30 \
  --warmups 5 \
  --output /tmp/rag-citation-storage-task-401-4.json
```

Record results in task notes. Do not edit the v1 baseline.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/DB/migrations/chachanotes_v*_citation_provenance.sql \
  tldw_chatbook/DB/ChaChaNotes_DB.py \
  tldw_chatbook/Chat/citation_trace_repository.py \
  tldw_chatbook/Chat/citation_provenance_runtime.py \
  tldw_chatbook/Chat/chat_persistence_service.py \
  tldw_chatbook/config.py \
  Tests/DB/test_chachanotes_citation_provenance_migration.py \
  Tests/Chat/test_citation_trace_repository.py \
  Tests/Chat/test_chat_persistence_service.py \
  Tests/test_config_rag_citation_defaults.py \
  Helper_Scripts/Benchmarks/rag_citation_provenance_benchmark.py \
  Tests/Performance/test_rag_citation_provenance_benchmark.py
git commit -m "feat(rag): persist sealed citation traces atomically"
```

Complete `TASK-401.4` hygiene.

---

### Task 5: Add namespaced identity and idempotency (`TASK-401.5`)

**Files:**

- Modify: `tldw_chatbook/Chat/citation_trace_identity.py`
- Modify: `tldw_chatbook/Chat/citation_trace_repository.py`
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py`
- Modify: `Tests/Chat/test_citation_trace_identity.py`
- Modify: `Tests/Chat/test_citation_trace_repository.py`
- Modify: `Tests/Chat/test_chat_persistence_service.py`

**Required repository behavior:**

- idempotency keys for local retry, server wire identity, owner links, and cache reuse
- imported and Sync identity constructors remain pure dormant contracts; portable import and Sync writes are out of scope
- identity comparisons always use the `CitationFingerprintCodec` and stable context defined in `TASK-401.2`/`401.4`

- [ ] **Step 1: Extend failing identity and repository tests**

Test:

- stable idempotency derivation from the already-defined namespaced identities
- identical external IDs in different authorities/tenants do not collide
- owner and cache idempotency domains cannot collide
- no raw text, secret, or portable digest in serialized trace/owner metadata

- [ ] **Step 2: Add uncertain-transaction retry tests**

Simulate a commit whose result is not returned, then retry the same message/trace IDs. Assert:

- one message
- one trace aggregate
- one row per child identity
- one owner link
- a different body or governed payload under the same identity fails closed

- [ ] **Step 3: Add repository idempotency and cache-owner reuse**

On retry, compare immutable identity and integrity columns before returning the existing aggregate. A cache hit adds another message owner to the original trace. It does not clone the trace or rewrite generation ID.

- [ ] **Step 4: Add active body-binding checks**

Store only the keyed owner fingerprint. On message edit/replacement:

- matching content leaves the owner active
- mismatch marks `body_mismatch`
- the historical trace remains readable
- grounded presentation becomes ineligible
- missing/unavailable fingerprint key returns `unverifiable`, never active/grounded, while aggregate-only historical reads remain available

Expose `get_active_trace_for_message(message_id, revision, current_body, codec)` as the only active-presentation lookup and wire the same check into `ChatPersistenceService.update_message_content` without deleting provenance. Tests must prove callers cannot turn a summary or hydrated trace into active grounded state without a verified owner binding.

Run:

```bash
python -m pytest \
  Tests/Chat/test_citation_trace_identity.py \
  Tests/Chat/test_citation_trace_repository.py \
  Tests/Chat/test_chat_persistence_service.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/citation_trace_identity.py \
  tldw_chatbook/Chat/citation_trace_repository.py \
  tldw_chatbook/Chat/chat_persistence_service.py \
  Tests/Chat/test_citation_trace_identity.py \
  Tests/Chat/test_citation_trace_repository.py \
  Tests/Chat/test_chat_persistence_service.py
git commit -m "feat(rag): make citation persistence idempotent"
```

Complete `TASK-401.5` hygiene.

---

### Task 6: Add revocation, retention, tombstones, and GC (`TASK-401.6`)

**Files:**

- Create: `tldw_chatbook/Chat/citation_payload_lifecycle.py`
- Modify: `tldw_chatbook/Chat/citation_trace_repository.py`
- Create: `Tests/Chat/test_citation_payload_lifecycle.py`
- Modify: `Tests/Chat/test_citation_trace_repository.py`

**Required public contracts:**

- `SnapshotDedupeScope`
- `PayloadRetentionPolicy`
- `PayloadTombstone`
- `CitationPayloadLifecycle.revoke(...)`
- `CitationPayloadLifecycle.collect(...)`
- repository tombstone checks on every payload hydration/write/replay seam

- [ ] **Step 1: Add failing dedupe-scope and purge tests**

Assert identical bytes dedupe only when non-null `governance_scope_id` (local profile or authenticated server tenant/authority-root), authority, confidentiality policy, revocation scope, and exact-content identity all match. Cross-tenant or cross-revocation-scope text must create separate payloads. Include the local/no-tenant case so SQLite `NULL` semantics cannot bypass dedupe.

- [ ] **Step 2: Add failing anti-resurrection tests**

After revoke/secure purge:

- text, title, identity, locator, lineage, hashes, and comparison fingerprints are absent
- the allowed non-content tombstone remains
- cache, import, and simulated Sync replay cannot rewrite the payload
- the sealed trace metadata and completeness-at-seal remain unchanged
- run, attempt, snapshot, prompt-evidence reference, and marker identities/counts remain unchanged

- [ ] **Step 3: Implement lifecycle transactions**

Revocation atomically:

1. inserts/updates the tombstone
2. clears `rag_evidence_runs.run_payload_json` and marks the existing run row purged
3. clears all governed snapshot columns, including hashes/fingerprints, while retaining the referenced opaque snapshot row
4. clears governed answer-attempt body/integrity fields while retaining attempt identity
5. leaves `rag_trace_evidence_refs`, marker ordinals, trace aggregate JSON, and completeness-at-seal untouched
6. records reason code, policy version, scope, and time

The revoke path never deletes a row still referenced by a sealed trace and never relies on `ON DELETE CASCADE` for purge. Never log purged values.

- [ ] **Step 4: Add reference-safe GC**

GC must treat as live:

- active and soft-deleted-within-policy message owners
- artifact owner leases
- pending artifact links/unlinks
- Sync retention/tombstone barriers supplied by the caller
- policy retention windows

Ordinary collection may delete an entire unowned trace graph only in a topologically safe transaction after all owner/barrier checks: remove mutable observations and evidence refs, delete the trace-owned run/attempt rows and trace, then delete only snapshots with no remaining refs. Revoked origins retain tombstones through the configured policy window; a snapshot row needed by a surviving trace is cleared, never deleted.

Run:

```bash
python -m pytest \
  Tests/Chat/test_citation_payload_lifecycle.py \
  Tests/Chat/test_citation_trace_repository.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/citation_payload_lifecycle.py \
  tldw_chatbook/Chat/citation_trace_repository.py \
  Tests/Chat/test_citation_payload_lifecycle.py \
  Tests/Chat/test_citation_trace_repository.py
git commit -m "feat(rag): govern citation payload lifecycle"
```

Complete `TASK-401.6` hygiene.

---

### Task 7: Persist bounded current-source observations (`TASK-401.7`)

**Files:**

- Modify: `tldw_chatbook/Chat/citation_source_locators.py`
- Modify: `tldw_chatbook/Chat/citation_trace_repository.py`
- Create: `Tests/Chat/test_citation_source_observations.py`

**Required public contract:**

- `CitationSourceObservation` with independent:
  `availability`, `permission`, `content_state`, `location_state`,
  `capabilities`, `observed_at`, and bounded sanitized error state

- [ ] **Step 1: Add failing observation contract tests**

Cover available, missing, offline, error, revoked, authentication-required, ambiguous, unchanged, changed, relocated, and unknown states. Reject contradictory or unbounded payloads.

- [ ] **Step 2: Add failing repository replacement tests**

Key rows by trace namespace, prompt-set ID, prompt-local evidence ordinal, opaque snapshot payload ref, and resolver kind/version. A newer observation replaces the prior row; an older/stale result cannot overwrite it. Tests use two rerun prompt sets with the same ordinal but different payload refs and prove they cannot collide. No polling history accumulates.

- [ ] **Step 3: Implement observation upsert/read**

Use a compare-and-replace transaction with `observed_at` plus request generation/nonce. Writes must not touch:

- immutable trace JSON
- completeness at seal
- prompt evidence sets
- historical locator payloads
- governed submitted snapshots

Run:

```bash
python -m pytest Tests/Chat/test_citation_source_observations.py -v
```

- [ ] **Step 4: Commit**

```bash
git add tldw_chatbook/Chat/citation_source_locators.py \
  tldw_chatbook/Chat/citation_trace_repository.py \
  Tests/Chat/test_citation_source_observations.py
git commit -m "feat(rag): persist bounded citation observations"
```

Complete `TASK-401.7` hygiene.

---

### Task 8: Add crash-safe artifact owner leases (`TASK-401.8`)

**Files:**

- Create: `tldw_chatbook/Chat/citation_artifact_ownership.py`
- Modify: `tldw_chatbook/Chat/citation_trace_repository.py`
- Modify: `tldw_chatbook/Chat/citation_payload_lifecycle.py`
- Modify: `tldw_chatbook/Chat/console_save_targets.py`
- Modify: `tldw_chatbook/Chatbooks/local_chatbook_service.py`
- Modify symbol: `ChatScreen._save_console_message_as_chatbook` in `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify symbol: `_save_console_chatbook_artifact` in `tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py`
- Modify symbols: `_wire_prompt_chatbook_services`, `_wire_chat_conversation_services`, and `_schedule_deferred_startup_work` in `tldw_chatbook/app.py`
- Create: `Tests/Chat/test_citation_artifact_ownership.py`
- Modify: `Tests/Chat/test_console_save_targets.py`
- Modify: `Tests/Chatbooks/test_local_chatbook_service.py`
- Modify: `Tests/Event_Handlers/Chat_Events/test_chat_events.py`
- Modify: `Tests/Performance/test_app_startup_performance.py`

**Current-backend classification:**

`LocalChatbookService` stores its registry in atomic JSON, not ChaChaNotes. It is therefore a cross-database/cross-store artifact backend. Do not invent a fake SQLite foreign key. The ownership interface and tests require a real artifact FK and shared transaction for any future backend that shares the trace DB; current production behavior and its contract tests use only the outbox/lease path.

- [ ] **Step 1: Add failing artifact-registry outbox tests**

Extend the registry format backwards-compatibly with bounded `provenance_outbox` entries. Test:

- artifact create and pending-link entry appear in one atomic registry replacement
- artifact delete and pending-unlink entry appear together
- artifact ownership revision is immutable for one lease lifecycle; replacement saves allocate the next revision (or a new artifact ID), so one lease has at most one link/unlink pair
- old registries without the field still load
- operation IDs are stable and idempotent
- trace metadata is not copied into the artifact registry
- overlapping create/delete/link operations serialize without dropping another operation
- a fake backend declaring the shared-DB mode is rejected unless it supplies a tested owner-FK/shared-transaction operation; the JSON backend declares cross-store mode

- [ ] **Step 2: Add failing lease/reconciler crash-matrix tests**

Interrupt after each link or unlink handshake phase:

1. artifact registry replacement durably records the artifact mutation plus its unique link/unlink operation ID
2. one repository transaction idempotently records that operation receipt and transitions the **same stable owner lease** (`link_pending -> live` for link; `live -> unlink_pending` for unlink)
3. artifact registry durably marks the operation acknowledged; acknowledged unlink entries remain until trace release is confirmed
4. for unlink, one repository transaction marks the unlink receipt acknowledged and transitions that same lease `unlink_pending -> released`; a final idempotent registry replacement may then prune the entry

The original link and later unlink use different operation IDs in `rag_artifact_owner_operations`, while both target one owner key in `rag_artifact_owner_leases`; operation receipts are never represented as additional leases. Restart and reconcile after every phase. Assert no duplicate lease, both operation IDs remain idempotent, the old live state does not survive release, and no trace collection occurs during pending link, live lease, or unresolved `unlink_pending`. If the artifact registry is unavailable or corrupt, reconciliation fails closed and retains the barrier.

- [ ] **Step 3: Implement the ownership coordinator**

`CitationArtifactOwnershipCoordinator` accepts the artifact store and trace repository. It:

- links/unlinks idempotently by stable operation ID
- derives a stable owner key from profile, artifact store/id/revision, and trace; validates trace and artifact identities
- stores link/unlink idempotency receipts separately from the stable lease state machine
- never trusts trace IDs from unvalidated imported metadata
- leaves failed operations pending with bounded sanitized error state
- exposes a batch-limited `reconcile_pending(limit=...)`
- treats the runtime recovery switch as authoritative and performs no write/reconciliation while disabled

- [ ] **Step 4: Carry active ownership from both Console save paths**

At the native `ChatScreen` and legacy event-handler save seams, resolve provenance only from the persisted message ID/revision. The repository verifies the keyed message-body fingerprint, then returns an opaque owner request. `ChatbookArtifactPayload` and the JSON registry may carry only bounded namespace/trace/operation IDs—not snapshots, locators, titles, or secrets.

Add tests for:

- active matching trace creates a pending link
- missing/body-mismatched trace saves the artifact without a grounded owner
- both save paths use the same coordinator contract
- concurrent registry operations cannot lose provenance outbox records

- [ ] **Step 5: Wire deferred startup reconciliation**

Construct the repository/coordinator only after both local Chatbook and ChaChaNotes services exist. Schedule bounded reconciliation through the existing deferred-startup mechanism so UI readiness is not blocked. Do not schedule it when canonical writes are disabled. Failures log only operation IDs/reason codes and do not crash startup.

Run:

```bash
python -m pytest \
  Tests/Chat/test_citation_artifact_ownership.py \
  Tests/Chat/test_console_save_targets.py \
  Tests/Chatbooks/test_local_chatbook_service.py \
  Tests/Event_Handlers/Chat_Events/test_chat_events.py \
  Tests/Performance/test_app_startup_performance.py -v
```

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Chat/citation_artifact_ownership.py \
  tldw_chatbook/Chat/citation_trace_repository.py \
  tldw_chatbook/Chat/citation_payload_lifecycle.py \
  tldw_chatbook/Chat/console_save_targets.py \
  tldw_chatbook/Chatbooks/local_chatbook_service.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py \
  tldw_chatbook/app.py \
  Tests/Chat/test_citation_artifact_ownership.py \
  Tests/Chat/test_console_save_targets.py \
  Tests/Chatbooks/test_local_chatbook_service.py \
  Tests/Event_Handlers/Chat_Events/test_chat_events.py \
  Tests/Performance/test_app_startup_performance.py
git commit -m "feat(rag): reconcile artifact citation ownership"
```

Complete `TASK-401.8` hygiene.

---

### Task 9: Migrate legacy citation sidecars safely (`TASK-401.9`)

**Files:**

- Create: `tldw_chatbook/Chat/citation_legacy_migration.py`
- Modify symbols: sidecar loading, `record_message_rag_context`, `get_messages_with_context`, and `get_citations` in `tldw_chatbook/Chat/chat_conversation_service.py`
- Modify symbol: legacy citation handling in `tldw_chatbook/Chatbooks/chatbook_importer.py`
- Modify symbol: citation report reads in `tldw_chatbook/Chatbooks/chatbook_creator.py`
- Modify symbols: `_wire_chat_conversation_services` and `_schedule_deferred_startup_work` in `tldw_chatbook/app.py`
- Create: `Tests/Chat/test_citation_legacy_migration.py`
- Modify: `Tests/Chat/test_chat_conversation_service.py`
- Modify: `Tests/Chatbooks/test_chatbook_importer.py`
- Modify: `Tests/Chatbooks/test_chatbook_creator.py`
- Modify: `Helper_Scripts/Benchmarks/rag_citation_provenance_benchmark.py`
- Modify: `Tests/Performance/test_rag_citation_provenance_benchmark.py`

**Cutover rule:**

Canonical DB rows become visible only after a successful journaled cutover. The sidecar remains read-only compatibility input and is not deleted. A later sidecar change is divergence, not an implicit merge. Portable canonical provenance import, imported-origin identity, and authority rebinding remain follow-on work.

- [ ] **Step 1: Add failing pure legacy-synthesis tests**

Cover:

- `EvidenceBundle`
- `CitationRef`
- `citation_validation`
- `chat_rag_context.json`
- malformed/partial records
- numeric legacy markers
- legacy paths, URLs, and `content_ref`
- legacy citation data embedded in a Chatbook package

Every result is `origin=legacy_inferred` with `partial` or `unavailable` completeness. Free-form locators remain inert. A legacy Chatbook package is only another legacy input container; it never produces `origin=imported`, claims a complete trace, or activates a locator in this task.

- [ ] **Step 2: Add failing migration-journal tests**

Per conversation, store:

- keyed source sidecar fingerprint
- state (`pending`, `running`, `complete`, `failed`, `diverged`)
- next stable message cursor
- attempt count and bounded reason code
- start/update/complete timestamps

The journal is one bounded normalized row per conversation and never stores created trace-ID arrays. Canonical rows identify their legacy conversation/message origin through bounded indexed columns. Test interruption after every 100-message batch, restart, retry, malformed conversation isolation, and no duplicate trace/owner rows.

- [ ] **Step 3: Implement bounded staged migration and atomic visibility cutover**

For each conversation:

1. bypass the cached parsed object, reject a sidecar over 32 MiB, read raw bytes, parse once, select the conversation's bounded records, and HMAC their deterministic canonical encoding
2. select at most 100 messages after the stable cursor
3. synthesize, validate, and persist only that batch in one transaction with `visibility_state=migrating`
4. update the cursor in the same transaction
5. after the final batch, re-read raw bytes and compare the fingerprint
6. in one final transaction, mark all rows for the conversation `active` and the journal `complete`

Readers ignore `migrating` rows, so no partially converted conversation is observable. A crash retries the current batch through canonical origin/idempotency constraints. A changed final fingerprint marks the journal `diverged`; hidden staging rows remain ineligible and are deleted/rebuilt only by an explicit retry. Do not block opening a conversation.

- [ ] **Step 4: Make readers canonical-first with legacy fallback**

Update `ChatConversationService.get_messages_with_context` and `get_citations`:

- canonical rows when migration is complete
- synthesized legacy view when missing/incomplete
- explicit divergence state when the post-cutover fingerprint changes
- no silent canonical/sidecar merge

Replace the unbounded whole-file cache with a bounded entry recording `(mtime_ns, size, keyed_content_fingerprint)`. Stat equality may reuse parsed legacy content for ordinary fallback rendering, but it can never establish post-cutover `unchanged`: before a canonical reader returns a verified non-diverged state, it re-reads the bounded raw file and compares the keyed per-conversation content fingerprint with the journal. If verification has not run, expose `verification_pending` rather than asserting unchanged. Migration always bypasses the parsed cache.

Add concurrency tests that:

- change the sidecar between migration batches and force final cutover to `diverged`
- rewrite content to the same size and restore the prior mtime, then prove keyed verification still detects divergence
- never merge canonical and changed legacy records

- [ ] **Step 5: Enforce canonical single-write when enabled and adapt legacy package data**

Keep `record_message_rag_context` as an explicitly deprecated, test/rollback-only legacy seam. When canonical writes are enabled, product callers cannot append new sidecar citation records.

`ChatbookImporter` routes existing package citation data through the same `legacy_inferred` adapter and canonical repository. It verifies local message ownership/body binding and tombstones but does not allocate imported-origin identity, rebind authorities, or interpret a future provenance package. While the recovery switch is off, the importer may use the existing inert sidecar compatibility path so package import remains lossless; enabling canonical writes makes the sidecar path unavailable.

`ChatbookCreator` reads active canonical provenance first but keeps the legacy fallback until later export-policy work replaces the current citation report format. Do not claim that this task delivers portable canonical import or final policy-filtered export.

- [ ] **Step 6: Wire bounded background migration**

Schedule at most one 100-message batch per idle unit after UI readiness. Opening a conversation may enqueue its migration at higher priority but must not await the whole sidecar. Do not schedule or resume migration while canonical writes are disabled. Failures are retryable and sanitized.

Run:

```bash
python -m pytest \
  Tests/Chat/test_citation_legacy_migration.py \
  Tests/Chat/test_chat_conversation_service.py \
  Tests/Chatbooks/test_chatbook_importer.py \
  Tests/Chatbooks/test_chatbook_creator.py \
  Tests/Chat/test_citation_trace_repository.py -v
```

- [ ] **Step 7: Run the migration benchmark and interruption case**

Replace the initial legacy migration proxy with the real bounded migration service candidate while retaining the fixture scan as its control.

```bash
python Helper_Scripts/Benchmarks/rag_citation_provenance_benchmark.py \
  --mode qualification \
  --baseline Docs/Development/RAG/citation-provenance-baseline-v1.json \
  --samples 30 \
  --warmups 5 \
  --output /tmp/rag-citation-migration-task-401-9.json
```

Verify ≥100 messages/second on the documented machine and zero duplicates after interruption/restart.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Chat/citation_legacy_migration.py \
  tldw_chatbook/Chat/chat_conversation_service.py \
  tldw_chatbook/Chatbooks/chatbook_importer.py \
  tldw_chatbook/Chatbooks/chatbook_creator.py \
  tldw_chatbook/app.py \
  Tests/Chat/test_citation_legacy_migration.py \
  Tests/Chat/test_chat_conversation_service.py \
  Tests/Chatbooks/test_chatbook_importer.py \
  Tests/Chatbooks/test_chatbook_creator.py \
  Helper_Scripts/Benchmarks/rag_citation_provenance_benchmark.py \
  Tests/Performance/test_rag_citation_provenance_benchmark.py
git commit -m "feat(rag): migrate legacy citation sidecars safely"
```

Complete `TASK-401.9` hygiene.

---

## Foundation verification gate

After all nine children are implemented, run:

```bash
python -m pytest \
  Tests/Performance/test_rag_citation_provenance_benchmark.py \
  Tests/DB/test_chachanotes_citation_provenance_migration.py \
  Tests/Chat/test_citation_trace_models.py \
  Tests/Chat/test_citation_trace_adapters.py \
  Tests/Chat/test_citation_source_locators.py \
  Tests/Chat/test_citation_trace_identity.py \
  Tests/Chat/test_citation_trace_repository.py \
  Tests/Chat/test_citation_payload_lifecycle.py \
  Tests/Chat/test_citation_source_observations.py \
  Tests/Chat/test_citation_artifact_ownership.py \
  Tests/Chat/test_citation_legacy_migration.py \
  Tests/Chat/test_console_chat_controller.py \
  Tests/Chat/test_console_save_targets.py \
  Tests/Chat/test_citation_evidence_models.py \
  Tests/Chat/test_answer_citations.py \
  Tests/Chat/test_chat_persistence_service.py \
  Tests/Chat/test_chat_conversation_service.py \
  Tests/Chatbooks/test_local_chatbook_service.py \
  Tests/Chatbooks/test_chatbook_importer.py \
  Tests/Chatbooks/test_chatbook_creator.py -v
python Helper_Scripts/Benchmarks/rag_citation_provenance_benchmark.py \
  --mode qualification \
  --baseline Docs/Development/RAG/citation-provenance-baseline-v1.json \
  --samples 30 \
  --warmups 5 \
  --output /tmp/rag-citation-foundation-qualification.json
python -m pytest Tests/ChaChaNotesDB/ Tests/DB/ -q
python -m pytest -q
python -m ruff check tldw_chatbook/Chat tldw_chatbook/Chatbooks \
  tldw_chatbook/DB Tests/Chat Tests/Chatbooks Tests/DB Tests/Performance
git diff --check
```

Also verify manually:

- no provenance table appears in any FTS, Library, or RAG index inventory
- a sidecar-only conversation opens before, during, and after migration
- a revoked payload exposes no text/title/locator/hash through repository dumps or logs
- application startup remains non-blocking with pending artifact and migration work
- setting `canonical_writes_enabled=false` leaves stored canonical traces readable while repository writes, migration, and reconciliation remain dormant

## Follow-on plans

Create separate reviewed plans for:

1. local retrieval/prompt capture, answer attempts, sealing, markers, and visible repair
2. `tldw_server` `grounding_trace/v1` publication/production and Chatbook adaptation
3. shared inspector, Markdown-aware marker UI, source inventory navigation, and one task per resolver family
4. saved-artifact payload carry-through, policy-filtered export/import, and Sync v2 negotiation
5. RAG error analysis, calibrated citation evaluation, security conformance, performance qualification, staged rollout, and eventual recovery-switch default change

Do not begin a follow-on plan until the contracts it consumes are merged and its own Backlog tasks/ADR check are complete.
