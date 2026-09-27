# Inactive V2 canonical profile aggregates implementation plan

> **For agentic workers:** Use superpowers:executing-plans for the already selected native inline execution. Steps use checkbox syntax. One fresh component review closes this unit.

**Goal:** Implement the full ADR-192 data contract without activating native canonical V2.

**Architecture:** An explicit `v2_models` module composes unchanged V1 typed payloads, exact owner-version bindings and the published V2 meaning projection. `v2_contract` supplies separately named validated canonical/hash/integrity, structural export and required semantic dialect interfaces. No default export, consumer or database changes.

**Tech Stack:** Python >=3.12, existing Pydantic 2 and pinned RFC 8785 library; existing pytest/jsonschema test tools; offline wheel build/install.

**Spec:** Docs/superpowers/specs/2026-09-26-personal-context-v2-canonical-data-contract-design.md

## Global Constraints

- ADR required: no new ADR. ADR path: backlog/decisions/192-personal-context-v2-canonical-data-contract.md. Reason: direct inactive implementation of the accepted storage-independent contract; ADR-185/186/187/191 and native activation gates remain binding.
- Python >=3.12; no dependencies, package version, V1 public exports or `SERIALIZED_SCHEMA_VERSION=1` changes.
- All specified keys required, including nulls/arrays; only absent model_disclosure defaults to deny. Nested payload V1 defaults remain V1.
- IDs: exact built-in strings, 1–128 codepoints, <=512 UTF-8 bytes, nonblank, no Cc/Cf/invalid scalars. Digest: 64 lowercase hex.
- Counters/version use existing JSON-integer semantics, 0..9007199254740991; envelope version exactly numeric 2. Existing binding integers remain strictly built-in int.
- Times preserve existing aware whole-minute-offset, millisecond-precision UTC rules. Number 0..1, finite, no bool/string. Method <=64 codepoints/256 bytes, nonblank, no Cc/Cf or credential content.
- Bindings/assessments/edges sorted by ID; purposes and audiences unique sorted. Signed arrays never repaired by sorting.
- Canonical ceilings after defaults: payload/claim/manifest 16384, record65536, proposal98304 UTF-8 bytes.
- Explicit duplicate-aware JSON decoding; unsupported compiled JSON entrypoints refuse before content parsing can claim conformance.
- Public validated serializers revalidate exact full raw-state snapshots before serialization. Unsupported classes/containers/timezones/state dicts cannot invoke callbacks or hide unknown fields.
- Fields are excluded from repr; public aggregate validation/serialization errors are content-free. Structured Pydantic diagnostic input is not a logging API.
- Preserve application, V1 core/schema/fixture, existing binding/meaning bytes and unrelated TASK-25907.10/roadmap suffix. Commit owned paths only.
- Targeted native tests only. No live source/profile/keyring/provider/server/network calls, Sync/storage/migration/grant/UI activation, full sweep, push/PR/merge.

## Task 1: Complete the inactive V2 contract

**Files:**
- Create: packages/tldw_profile_core/src/tldw_profile_core/v2_models.py
- Create: packages/tldw_profile_core/src/tldw_profile_core/v2_contract.py
- Create: packages/tldw_profile_core/tests/test_v2_aggregates.py
- Create: packages/tldw_profile_core/tests/test_v2_aggregates_package.py
- Create identical distribution/package `schemas/personal-context-v2{,-meta}.json` and `fixtures/v2/*.json`.
- Modify: packages/tldw_profile_core/pyproject.toml (resource declarations only).
- Create: backlog/docs/personal-context-v2-aggregates.md.
- Modify: owned roadmap prefix and this unit's tracker/checklist.

**Interfaces:**
- Consumes `OwnerVersionEvidenceBinding`, `Identity`, `Digest`, `CaptureTime`, unchanged V1 payload types/semantic key and `ClaimMeaningV2`/`claim_meaning_digest`.
- Produces `ProfileManifestV2`, `ProfileRecordV2`, `ProfileProposalV2`, controls/provenance/claim/attribution/disclosure shapes, all frozen and extra-forbid.
- `validate_v2_object(value: object) -> ProfileManifestV2 | ProfileRecordV2 | ProfileProposalV2`: structurally/semantically validate a fresh snapshot, rejecting ambiguous/unknown/V1 objects with a generic error. No native admission.
- `validate_v2_json(json_data: str | bytes | bytearray) -> same union`: duplicate-aware JSON ingress.
- `canonical_v2_bytes(value: union) -> bytes`, `v2_object_digest(value: union) -> str`, `v2_integrity_tag(value: union, key: bytes) -> str`: require exact aggregate class, fresh complete validation; keyed integrity requires exact 32-byte built-in bytes.
- `export_v2_json_schema(path: Path) -> None`, `export_v2_meta_schema(path: Path) -> None`; fixed V2 identifiers/required exact semantic map. Structural schemas specify closed wire types, while semantic validation is `validate_v2_object` on the same decoded data.

- [x] **Step 1: Add independent fixtures and positive/failing tests.** Use public synthetic `bytes(range(32))`. Construct data separately from production models; freeze compact sorted stdlib JSON bytes for ASCII aggregates and hashlib/hmac results as static resources. Cover manifest, active supported record, archived, deleted, pending create/update and resolved receipt; invalid fixtures retain structural validity but change one exact digest/time/approval/semantic requirement. Probe missing module with an assertion rather than collection failure.

```python
def api():
    assert importlib.util.find_spec('tldw_profile_core.v2_models'), 'V2 aggregates missing'
    return importlib.import_module('tldw_profile_core.v2_models')

def test_fixed_record():
    record = api().ProfileRecordV2.model_validate(fixture['data'])
    assert contract().canonical_v2_bytes(record).decode() == fixture['canonical_utf8']
    assert contract().v2_object_digest(record) == fixture['sha256']
    assert contract().v2_integrity_tag(record, bytes(range(32))) == fixture['integrity_tag']
```

- [x] **Step 2: Observe missing-feature RED.**
Run: `.venv/bin/python -m pytest -q packages/tldw_profile_core/tests/test_v2_aggregates.py --basetemp=/private/tmp/v2-aggregates-red --junitxml=/private/tmp/v2-aggregates-red.xml`.
Expected: assertion failures for the absent explicit module; no unexpected collection errors.

- [x] **Step 3: Implement the specified complete models.**
Implement exact spec tables. Manifest requires four exact object-version declarations and seven sorted semantic tags, explicit retirement epoch and ordered times. Controls have three closed disclosure variants with unique sorted audience-purpose pairs. Claim has ten required fields. Support names the exact complete binding digest/current meaning and enforces origin/time state, nonempty supporting/contradicting span and captured<=assessed<=updated. Approval binds exact current version/meaning and record times; confidence/salience bind current meaning/version. Bound-source validity IDs name current bindings. Meaning validation supplies payload/validity/edge invariants; containing records reject self-current edges. Legacy requires unknown validity/empty metadata/deny/review hold. Privacy successors require system/reason/null derived, privacy hold, null approval/confidence/salience and no inherited assessments. Deleted records drop all content metadata and deny disclosure. Pending proposals enforce exact90 days, active nested identity/parent, null approval and nested times; receipts drop nested record/provenance.

```python
projection = ClaimMeaningV2.model_validate({
    'projection': 'profile-claim-v2',
    **{key: snapshot[key] for key in ('profile_id','record_id','scope_id','kind','payload')},
    **{key: claim[key] for key in ('claim_basis','temporal_validity','relations')},
})
assert claim['claim_digest'] == claim_meaning_digest(projection)
```

- [x] **Step 4: Validate positive controls and adversarial mutations.**
Parameterize missing/extra fields, bool/string/float distinctions, malformed IDs/methods, duplicate/unsorted arrays, stale receipts, substituted binding fields, mismatched support origin/digest/time, unknown validity, legacy/privacy/deleted shapes, expiry decisions, wrong proposal identity/parent/state and schema requirements. Use already frozen digest oracles, not values generated by production. Test exact canonical ceilings using boundary synthetic UTF-8 strings; recompute meaning independently only for test input to isolate the byte ceiling.
Run: same targeted test file with `--basetemp=/private/tmp/v2-aggregates-models`.
Expected: all model cases pass with fixed vectors unchanged.

- [x] **Step 5: Add raw ingress, unsafe-state and serialization RED cases.**
Reproduce duplicate members at outer/nested depths via dedicated JSON and compiled adapter/container entrypoints. Copy models with invalid nested dict/subclass/extra fields/custom timezones; private raw-state dictionaries must not filter stored data. Every public helper must freshly reject each unchecked aggregate. Validate dictionary dispatch is closed and reject subclasses without callback invocation.

```python
broken = record.model_copy(update={'claim': {'secret': 'private-marker'}})
with pytest.raises(ValueError, match='^invalid V2 profile object$'):
    contract().canonical_v2_bytes(broken)
```

- [x] **Step 6: Implement explicit validation/canonical/hash/integrity boundaries.**
Dispatch only one exact aggregate discriminator (`proposal_id`, `record_id`, `revision`). Snapshot known exact model raw dictionaries before parsing; bounded recursion/nodes/member counts admit every specified wire shape. Decode raw JSON with duplicate-member and nonfinite hooks. Refuse compiled JSON mode. Canonical helpers take exact instances, never unsafe generic `canonical_bytes` as the public boundary. Catch contract rejection and return generic errors without data/traceback chaining. HMAC rejects non-bytes/subclass/non32-byte keys.
Run: targeted file.
Expected: unsafe copies, compiled JSON and malformed raw ingress all reject; Python adapters and supported JSON path pass.

- [x] **Step 7: Add structural/dialect/export and package RED cases.**
Tests require source/package schemas and fixtures, exact semantic constants, structural-positive/semantic-negative disagreement examples, required vocabulary true, all references resolve, and exporter reproduction. Test resource absence before declarations/exports. Offline wheel probe builds a temporary copied package with `pip --isolated wheel --no-deps --no-build-isolation --no-index --no-cache-dir`, target-installs without dependencies, then launches Python `-I`. Check installed origins, Python>=3.12 metadata, V1 default exports/version, fixed bytes/digests/tags, schema validation and package/distribution parity.
Run: `.venv/bin/python -m pytest -q packages/tldw_profile_core/tests/test_v2_aggregates_package.py --basetemp=/private/tmp/v2-aggregates-package-red`.
Expected: schema/resource/API assertions fail after successful wheel build where applicable, never dependency/network failures.

- [x] **Step 8: Implement explicit schema/dialect exports and package resources.**
Generate a combined schema with one `$defs` namespace from the exact V2 aggregate union. Patch scalar time/number definitions as required for portable wire syntax, closed ID patterns and version handling; structural invariants expressible in JSON Schema remain structural, cross-field digest/time/byte/order requirements stay required semantics. Meta schema names all16 frozen rule-map entries with exact const values, false extras and all keys required. Add V2-only package resource declarations, leaving V1 exporter/resources untouched.
Run: both new files plus all shared-core compatibility tests, with isolated `--basetemp=/private/tmp/v2-aggregates-qualified` and XML receipt.
Expected: no errors/failures/skips; offline build/install and all V1/meaning/binding checks pass.

- [x] **Step 9: Self-review and targeted static checks.**
Compare every spec rule to code and concrete test; inspect generated schema structural classification and callback-free snapshot/rejection paths. Record only genuine implementation rulings in the ledger. Run affected-file Ruff check/format with `--no-cache --target-version py312`, `git diff --check`, unchanged prior files (except resource declaration), foreign .10/suffix hashes and resolving new documentation links.
Expected: clean affected static checks and byte guards.

- [x] **Step 10: Commit software and perform one fresh independent component review.**
Commit only new V2 files/resources and owned pyproject declarations. Provide reviewer base `ce795f99cd`, spec/plan/ledger and exact focus: malicious raw/model states, compiled JSON bypasses, schema/model disagreement, runtime-authority claims, attribution rebinding, byte limits and installed-resource conformance. Review only this unit, not earlier unrelated branch work. Each Important/Critical fix must have observed RED/GREEN and complete affected test pass; record Minor deferrals. No second reviewer pass.

- [x] **Step 11: Finish tracker/documentation and local commit.**
Document explicit imports/API, errors, canonical/integrity boundaries, fixed resources/evidence and remaining native retirement/disclosure/source/server gates. Mark AC only after actual evidence; CLI status Done after notes/ADR/docs/static/tests/self-review/fresh-review qualification. Stage owned roadmap prefix separately from foreign suffix. Preserve plan ledger rulings in durable docs before retiring only this plan's scratch. Local commits only; worktree retained.

## Review Focus

Exact scalar/type/state rejection across dictionaries, direct construction, unsafe copies, Python adapters and every JSON path; validation before serialization; raw-state filtering callbacks; unsupported binding float inputs; digest/approval/evidence transplantation; strict legacy/privacy/deleted/resolved shapes; schema structural claims and exact required vocabulary; ceiling tests after defaults and Unicode UTF-8 bytes; actual installed package provenance/resources. No passing data test may imply source access, approval authority, destination enrollment, runtime V2 admission or deletion completion.


## Execution corrections and final evidence

Independent review found per-call strict/extra overrides could erase forbidden
input before canonical revalidation; snapshot-time exact bool/contextual shapes
now guard all supported entrypoints including unchanged composed leaves. Full
offline dialect self-validation found460 nested errors from unconditional
meta required; document-declared dialect now selects the required root map.
Observed review RED31/61 -> final728/728 targeted checks (three defaulted-payload plus six further float
controls), py312 static checks and offline installed-wheel conformance. The
review adds no new policy or native activation. Component notes record every
scope ruling and remaining gates. No deferred Minor findings or re-review.
