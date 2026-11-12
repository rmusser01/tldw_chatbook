# Closed native V2 profile barrier implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for the already selected native inline workflow. Execute this as one implementation unit with one fresh final reviewer.

**Goal:** Prevent unqualified V2 data from bypassing native profile owners while preserving existing V1 behavior.

**Architecture:** An explicit native decoder validates V1/V2 bytes; a compiled consumer registry and repository-owned current-manifest check keep every V2 permit path closed. Service/bootstrap project generic unavailable status while repository reads and transactions enforce the barrier independently. No qualification receipts, storage migration or native admission permits are implemented.

**Tech Stack:** Native Python >=3.12, stdlib, existing Pydantic/profile-core, encrypted SQLite, pytest and Ruff. Native interpreter: `.venv/bin/python` (3.12.11).

**Spec:** [Accepted native design](../specs/2026-09-27-personal-context-native-v2-admission-design.md), Unit A only.

**Tracker:** [TASK-25907.22](../../../backlog/tasks/task-25907.22%20-%20Add-a-closed-native-V2-profile-compatibility-barrier.md).

ADR required: no new ADR; direct implementation of accepted Unit A.
ADR path: [ADR-193](../../../backlog/decisions/193-native-v2-profile-compatibility-and-admission.md).
Reason: implement its closed barrier without choosing a qualification/control storage protocol or activating V2. ADR-102/201/202/203/191/192 remain independent prerequisites.

## Global constraints

- Python >=3.12 is the floor.
- Build on the existing one profile repository and one authorized service.
- Existing binding and meaning components, V1 bytes, scopes and payloads remain unchanged.
- Do not widen the published V2 wire contract or import native controls into shared core.
- No local-only exception to ADR-201/192 is introduced.
- No UI change is part of this design. Use existing unavailable state and canonical F9 Settings behavior.
- No periodic worker, new dependency, external provider or fourth memory store is needed for the barrier itself.
- No physical migration/DDL/envelope version is implemented or approved here.
- All production V2 permit paths stay closed: no creation, import, migration, Sync, source lookup, model use or qualification receipt.
- Tests use temporary encrypted SQLite, in-memory key protection and synthetic data under Tests/conftest.py isolation. No real profile/keyring/source/provider/server/network or full test sweep.
- Worktree: `/Users/macbook-dev/.codex/worktrees/personal-context-memory-baseline/tldw_chatbook`; branch `codex/personal-context-memory-baseline`. Preserve independent TASK-25907.10 and the roadmap suffix, and stage only the owned roadmap prefix.

## Unit boundaries and files

This plan has one independently testable implementation/review unit. The steps
below are its RED/GREEN and integration stages, not separate reviewer dispatches.

| Action | Exact path | Responsibility |
| --- | --- | --- |
| Create | `tldw_chatbook/Personal_Context/native_codec.py` | Bounded explicit raw decoder and fresh canonical V1 insertion validation; lazy V2 imports. |
| Create | `tldw_chatbook/Personal_Context/native_compatibility.py` | Compiled route registry, immutable blocked/legacy view and generic compatibility exception. No grants/receipts or eligible V2 state. |
| Modify | `tldw_chatbook/Personal_Context/repository.py` | Current authenticated manifest guard, protected content reads, canonical insertion denial, transaction/link/outbox coverage. Preserve schema 8 and AAD marker 1. |
| Modify | `tldw_chatbook/Personal_Context/service.py` | Current compatibility status and owner-maintenance projection; never false ABSENT or user-content fallback. |
| Modify | `tldw_chatbook/Personal_Context/bootstrap.py` | Validate registry and initial compatibility before returning an available service. |
| Create | `Tests/Personal_Context/native_barrier_helpers.py` | Synthetic sealed-manifest injection and exact SQL-state comparison for tests; bypass no production guard. |
| Modify | `Tests/Sync_Interop/test_personal_context_dispatcher.py` | Retain synthetic legacy poison dispatch/quarantine while asserting new malformed ingress denial. |
| Modify | `Tests/Personal_Context/test_profile_sync_outbox.py` | Reject malformed new canonical ingress, then seal a synthetic legacy poisoned journal directly to retain the quarantine/shredding control. |
| Modify | `Tests/Personal_Context/test_export_service.py` | Preserve the existing concurrent snapshot test while forwarding the required fixed route and same read connection. |
| Create | `Tests/Personal_Context/test_native_profile_codec.py` | Positive V1/V2 validation, exact bytes and negative raw/model controls. |
| Create | `Tests/Personal_Context/test_native_profile_barrier.py` | Repository, registry, mutation/decryption and rollback tests on real encrypted SQLite. |
| Create | `Tests/Personal_Context/test_native_profile_barrier_routes.py` | Native service/context/tool/proposal/interview/export/recovery/Sync and app wiring integration. |
| Extend | `Tests/App/test_personal_context_wiring.py` | App reuse/retry of unavailable compatibility status, with no automatic recreation. |
| Create | `backlog/docs/personal-context-native-v2-barrier.md` | Final behavior, evidence, omissions and remaining release gates. |
| Update | Task and owned roadmap prefix | Exact final checks/status, review rulings and no activation claim. |

`native_admission.py`, persisted qualification state, migration/cohort wire formats,
retirement/publication owners and destination/source enrollment belong to later
units. Do not scaffold them in this unit. Existing context/tool/interview/export/
Sync consumers should inherit repository/service denial; production changes to
those modules require an observed integration failure and an AC/plan update.

## Interfaces fixed for this unit

- `NativeProfileKind = Literal["manifest", "scope", "record", "proposal"]`.
- `DecodedNativeProfileObject` is frozen and has `kind`, numeric `schema_version`,
  `value: BaseModel` and `canonical: bytes`; value/bytes are excluded from repr.
- `decode_native_profile(kind: NativeProfileKind, raw: bytes | str) -> DecodedNativeProfileObject`.
  No validation-policy/strict/extra kwargs. V1 dispatch uses exact existing classes;
  V2 manifest/record/proposal dispatch uses `validate_v2_json` plus class/kind checks
  and `canonical_v2_bytes`. Scope version 2 is unsupported. Existing V1 missing
  version defaults remain valid only after full V1 validation; a V2-shaped body
  never downgrades by omission. Numeric 1.0/2.0 follow shared JSON integer semantics;
  bool, string, nonfinite and unknown versions deny.
- `native_v1_bytes(kind: NativeProfileKind, value: BaseModel | Mapping[str, Any]) -> bytes`.
  Detach only known exact V1 models/plain built-in values, retaining all fields
  before full validation; reject unknown/extra model state and caller subclasses.
  Do not use a lossy model_dump that hides extra V2 fields. Nested known V1
  payload/control/provenance/semantic-key models are detached with their exact
  model_fields; enum values normalize through their declared V1 classes and
  portable aware datetimes through shared validation. Unknown callbacks/types deny.
  Produce the existing V1 canonical bytes after fresh validation; V2 denies.
- `NativeProfileDecodeError(ValueError)` exposes only `personal_context_payload_invalid`.
- `NativeProfileConsumer` is frozen: `consumer_id`, `owner_id`, `operations` tuple;
  `NATIVE_PROFILE_CONSUMERS` is a compiled, sorted, unique tuple. Registry revision
  and digest describe compiled routes only, not qualification.
- `require_native_consumer(consumer_id: str) -> NativeProfileConsumer` rejects
  unknown IDs with the generic compatibility exception. IDs are fixed native
  caller constants, never user/config/agent/import parameters.
- `ProfileCompatibilityView` is frozen: state `legacy_v1` or `v2_blocked`, schema
  version, profile/manifest IDs (repr=False), purge generation, shared evidence
  epoch (None for V1), compiled registry revision/digest. No qualified V2 state.
- `profile_compatibility(manifest: DecodedNativeProfileObject, *, consumer_id: str) -> ProfileCompatibilityView`
  uses exact decoded manifest requirements and compiled registry. Every valid V2
  result is blocked; invalid/unsupported manifest gives a generic error, never V1.
- `ProfileCompatibilityError(ProfileLockedError)` has fixed reason code
  `personal_context_compatibility_unavailable`; no content, IDs or denied counts.
- Repository `read_compatibility() -> ProfileCompatibilityView | None` is the
  narrow manifest-only maintenance path; None means no profile_meta, never a
  malformed, missing, destroyed or unsupported current manifest.
- Repository `_mutation(*, profile_id=None, allow_empty=False, consumer_id)`
  retains its transaction behavior and requires a fixed compiled native route ID.
  `_head_row`, `_iter_head_rows` and local-body/version helpers likewise carry
  the caller's fixed consumer ID through their guarded connection.
- Repository `_compatibility_on_connection(connection, *, consumer_id)` performs
  one authenticated manifest read and verifies profile ID, manifest version and
  purge generation against profile_meta before computing the immutable view.
- Repository `_require_legacy_on_connection(connection, *, consumer_id, allow_absent=False)`
  rejects blocked/invalid/unregistered states. Explicit absent creation/read cases
  are allowed only when no content-bearing orphan is returned or published.
- Rename existing envelope-authentication body to `_decrypt_row_authenticated(row)`;
  `_decrypt_row(row, *, connection=None, consumer_id)` first checks compatibility
  on the supplied operation connection, then authenticates the requested row.
  Without a connection it opens one guarded read; never consult cached status.
  The only production callers of raw authentication are the manifest-only reader
  and the guarded `_decrypt_row` wrapper. Tests may inspect ciphertext directly.

No external API accepts a compatibility view as authorization. Every repository
operation recomputes it from authenticated current state. A forged dataclass or
valid canonical approval field cannot mint a permit. Avoid runtime stack
inspection, dynamic auto-registration, decorators that hide transactions, or a
per-operation boolean feature flag.

## Task 1: Integrated closed native barrier

### Step 1 — Freeze synthetic fixtures and positive/negative codec controls

- [x] Read the task, accepted spec/ADR and lessons-testing-evidence,
  lessons-live-verification and lessons-backlog-hygiene. Record HEAD/native version
  and hashes of the foreign .10 task/suffix plus untouched shared-core files.
- [x] Create native_barrier_helpers.py with the fixture loader below, then add the
  codec test file importing v2_fixture from that helper. Load fixed source fixtures
  without editing them; fixture shape is `data`, `valid`, `canonical_utf8` and `model`.

```python
import json
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parents[2] / "packages/tldw_profile_core"

def v2_fixture(name):
    return json.loads((CORE_ROOT / "fixtures/v2" / f"{name}.json").read_text())

# In test_native_profile_codec.py; the loader above lives in native_barrier_helpers.py.
from Tests.Personal_Context.native_barrier_helpers import v2_fixture

def test_explicit_v2_decoder_is_validation_only():
    from tldw_chatbook.Personal_Context.native_codec import decode_native_profile
    fixed = v2_fixture("01-manifest")
    decoded = decode_native_profile("manifest", json.dumps(fixed["data"]))
    assert decoded.schema_version == 2
    assert decoded.canonical == fixed["canonical_utf8"].encode()
    assert fixed["data"]["profile_id"] not in repr(decoded)
```

V1 positives: real existing manifest, scope, record and proposal from native
factories; canonical bytes must equal `tldw_profile_core.canonical_bytes(value)`.
V2 positives: fixtures 01–07 with their exact kinds and independent fixed bytes.
Negatives: 08–14, top/nested duplicate fields, wrong selector/kind, scope v2,
missing/wrong selectors, schema true/"2"/3/nonfinite, malformed UTF-8, extra V2
fields on a V1 body, no_expiry 0/"false", oversize/deep input, model_construct
extras at root and composed children, caller model/Mapping subclasses.
A missing V1 schema field is a successful published-default control, not a V2
permit. Include maximum valid V1 Unicode semantic-key/provenance positive controls.

- [x] Run `.venv/bin/python -m pytest -q Tests/Personal_Context/test_native_profile_codec.py`
  and retain RED before implementation. Missing native module is the first RED;
  subsequent negative regressions must fail on the behavior they test.

### Step 2 — Implement only the bounded decoder and fresh V1 serializer

- [x] Implement the interfaces above in native_codec.py, using fixed decoder
  settings and content-free `raise NativeProfileDecodeError() from None`.
  Raw cap: 262144 UTF-8 bytes. Structural walk: 4096 nodes, depth 20, consistent
  with V2's existing detachment bounds. Limits cover valid V1 maximum-size data;
  if a fixed positive proves otherwise, correct the native cap without changing
  published core limits. Do not widen Sync's existing 16384-byte object cap.

```python
def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise NativeProfileDecodeError()
        result[key] = value
    return result

def reject_nonfinite(_value):
    raise NativeProfileDecodeError()
```

Use json.loads with both hooks, exact built-in raw types, strict UTF-8 and bounded
built-in traversal. V1 fresh model validation gets no override kwargs. Lazy-import
V2 helpers only for a numeric schema-2 body; verify the exact returned class for
the requested kind. Fresh V1 detachment reads exact known model __dict__ values
with object.__getattribute__, checks keys against exact model_fields, and detaches
nested values before validation; no callbacks/custom containers are traversed.

- [x] Rerun the codec file GREEN and verify canonical vectors against fixtures,
  not a round-trip oracle produced solely by the new serializer.

### Step 3 — Compile the route registry and blocked compatibility projection

- [x] Add registry/view tests to test_native_profile_barrier.py before production
  code; import v2_fixture from the shared test helper and json/pytest explicitly. The registry contains explicit repository/service/bootstrap route IDs,
  derived from the inspected method inventory below and fixed in code. Its
  construction rejects duplicate IDs/operations, unstable ordering and unknown
  owners. No public register/add/qualified method exists.

```python
def test_valid_v2_manifest_never_qualifies():
    from tldw_chatbook.Personal_Context.native_codec import decode_native_profile
    from tldw_chatbook.Personal_Context.native_compatibility import profile_compatibility
    fixed = v2_fixture("01-manifest")
    decoded = decode_native_profile("manifest", json.dumps(fixed["data"]))
    view = profile_compatibility(decoded, consumer_id="repository.read_compatibility")
    assert view.state == "v2_blocked"
    assert view.schema_version == 2
```

Add V1 success, unknown consumer deny, extra/unknown/reordered semantic requirement
fixtures and unsupported version controls. Supplying data fields such as
`qualified=True` or imported receipt data to a manifest fails validation. The
compatibility API has no kwargs that grant access.

- [x] Observe RED; implement native_compatibility.py with frozen result classes,
  compiled registry and generic exception. Do not import heavyweight app/Sync
  owners or V2 modules at module scope. Run registry/codec files GREEN.

### Step 4 — Install an authenticated blocked profile in a real test database

- [x] Extend native_barrier_helpers.py. `blocked_repository(tmp_path, protector,
  record_factory)` creates an ordinary V1 profile/record through current APIs,
  then seals a fixed V2 manifest as a synthetic pre-existing envelope. It returns
  `(repository, original_record)`. Get the V1 manifest before installing the V2
  fixture; override fixture profile/current-manifest/creation/purge identity with
  valid values, validate through shared V2 helpers, and use a new manifest ID.
  Do not call the newly guarded canonical insert to create an illegal fixture.

```python
from tldw_chatbook.Personal_Context.crypto import EnvelopeCipher

def seal_test_manifest(repository, connection, raw, profile_id, version_id, created_at):
    keys = repository._require_keys()
    aad = repository._aad("manifest", profile_id, version_id)
    sealed = EnvelopeCipher(keys.encryption_key, key_version=keys.key_version).encrypt(raw, aad)
    connection.execute(
        "INSERT INTO encrypted_objects VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("manifest", profile_id, version_id, None, 0, sealed.algorithm,
         sealed.nonce, sealed.ciphertext, sealed.wrap_nonce + sealed.wrapped_dek,
         sealed.key_version, repository._integrity_tag(keys.integrity_key, aad, raw), created_at),
    )
    connection.execute(
        "UPDATE object_heads SET version_id=? WHERE object_type='manifest' AND object_id=?",
        (version_id, profile_id),
    )
    connection.execute(
        "UPDATE profile_meta SET current_manifest_version=? WHERE singleton=1",
        (version_id,),
    )
```

The fixture's manifest purge generation remains the current one. Wrap insertion
in repository._transaction; this is a temporary test database using only memory
keys. Provide `sql_state(repository)` that reads every profile table ordered by
rowid through a private test connection and returns immutable tuples of rows;
include schema/meta/heads/encrypted objects/local policies/Undo/outboxes/quarantine/
first-link markers. The helper implementation is:

```python
from contextlib import closing
from tldw_chatbook.DB.sql_validation import validate_identifier

def sql_state(repository):
    with closing(repository._connect()) as connection:
        names = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )]
        assert all(validate_identifier(name, "table") for name in names)
        return tuple((name, tuple(tuple(row) for row in connection.execute(
            f'SELECT * FROM "{name}" ORDER BY rowid'
        ))) for name in names)
```

Use row/ciphertext equality, not DB-file bytes affected by
normal checkpointing. Add a V1 success control before each denial family.

- [x] Add the first repository regression: a valid V2 manifest with a readable
  V1 record must raise ProfileCompatibilityError on get_record, without decrypting
  the record, writing quarantine, changing any SQL rows or releasing canary text.
  Spy the raw authentication method; permit only manifest authentication during
  the denied operation. Repeat after close/reopen with the same in-memory protector.
- [x] Run the repository regression RED. First prove the fixture authenticates
  and its canonical V2 manifest validates; fixture construction failure is not
  product-denial evidence.

### Step 5 — Guard current manifest and content reads on owner connections

- [x] Implement read_compatibility and the connection helpers. First verify the unchanged supported storage marker in that connection (do not
  run _inspect_schema migrations from a reader). Compare authenticated manifest
  identity/current version/purge generation to profile_meta in the same read
  transaction. Also test a newer storage marker installed after the repository
  handle opened; it blocks before content authentication. Missing/invalid manifest is integrity/unavailable, never absent.
  Destroyed state remains the existing removed/locked lifecycle. The connection
  check is implemented in the repository with existing exception types:

```python
def _compatibility_on_connection(self, connection, *, consumer_id):
    from .native_codec import decode_native_profile
    from .native_compatibility import (
        ProfileCompatibilityError, profile_compatibility, require_native_consumer,
    )
    require_native_consumer(consumer_id)
    storage = connection.execute(
        "SELECT version FROM personal_context_schema WHERE singleton=1"
    ).fetchone()
    if storage is None or storage[0] != SCHEMA_VERSION:
        raise ProfileCompatibilityError()
    meta = connection.execute(
        "SELECT profile_id,current_manifest_version,purge_generation,destroyed FROM profile_meta WHERE singleton=1"
    ).fetchone()
    if meta is None:
        return None
    if meta["destroyed"]:
        raise ProfileDestroyedError("Personal Context is unavailable.")
    row = connection.execute(
        "SELECT encrypted_objects.* FROM encrypted_objects JOIN object_heads USING(object_type,object_id,version_id) "
        "WHERE object_type='manifest' AND object_id=? AND version_id=?",
        (meta["profile_id"], meta["current_manifest_version"]),
    ).fetchone()
    if row is None:
        raise ProfileIntegrityError("Personal Context is unavailable.")
    decoded = decode_native_profile("manifest", self._decrypt_row_authenticated(row))
    manifest = decoded.value
    if (manifest.profile_id, manifest.current_version_id, manifest.purge_generation) != (
        meta["profile_id"], meta["current_manifest_version"], meta["purge_generation"]
    ):
        raise ProfileIntegrityError("Personal Context is unavailable.")
    return profile_compatibility(decoded, consumer_id=consumer_id)

def _require_legacy_on_connection(self, connection, *, consumer_id, allow_absent=False):
    from .native_compatibility import ProfileCompatibilityError
    view = self._compatibility_on_connection(connection, consumer_id=consumer_id)
    if view is None:
        if allow_absent:
            return None
        raise ProfileDestroyedError("Personal Context is unavailable.")
    if view.state != "legacy_v1" or view.schema_version != 1:
        raise ProfileCompatibilityError()
    return view
```

Map bounded decoder failures in the manifest-only reader to the existing generic
integrity/unavailable exception without attaching plaintext input. Import the
compatibility exception inside the functions when needed to preserve lazy boot.

- [x] Split raw authentication without changing AES-GCM, HMAC, key handling or AAD.
  Check the compiled caller route and current profile before any non-manifest
  authentication. Pass the existing operation connection through decrypt call
  sites, including nested current_values; do not read uncommitted write state
  through a second connection. Use one BEGIN read snapshot for bulk reads.
- [x] Preserve V1 quarantine behavior for genuine corrupt V1 objects. Compatibility
  failure must propagate as its distinct generic exception, never be caught as
  integrity failure and converted to an omitted row or a quarantine write.
  V2/unsupported canonical row decoding is unavailable, with no lossy V1 parse.

Read-entry matrix: get_manifest, get_record/list_records, get_scope/list_scopes,
get_proposal/list_proposals, read_export_snapshot, get_record_derivation,
_get_local_body/_get_local_version and scope-binding readers, get_undo/list_undo_ids,
get_outbox_body/list_pending_outbox/list_dispatchable_outbox, first_link_head_rows/
first_link_sync_heads/first_link_reviewed_lineage. Queries that expose protected
IDs/labels also need a registered current guard. Narrow manifest-only status and
existing content-free destruction/key-recovery controls are separately registered;
they do not expose arbitrary content or silently authorize deletion.

- [x] Parameterize all read entries with a positive V1 result and blocked V2
  exception/unchanged SQL state. Include zero-row list/get cases: absence of a
  requested record must not bypass the whole-profile barrier. Run GREEN.

### Step 6 — Reject both current blocked profiles and incoming V2 on writes

- [x] Before implementing write guards, add exact SQL-state rollback regressions
  for ordinary commits under a blocked manifest and for incoming V2 under a V1
  or absent profile; import sql_state from the shared test helper and pytest in
  the test file. The generic encrypted writer can currently serialize a
  BaseModel/Mapping; type annotations alone do not close that path.

```python
def test_incoming_v2_manifest_cannot_create_profile(tmp_path, memory_protector):
    from tldw_profile_core import ProfileScope, ScopeKind
    from tldw_profile_core.v2_contract import validate_v2_object
    from tldw_chatbook.Personal_Context.repository import PersonalContextRepository
    from tldw_chatbook.Personal_Context.native_compatibility import ProfileCompatibilityError
    fixed = v2_fixture("01-manifest")["data"]
    manifest = validate_v2_object(fixed)
    scope = ProfileScope(scope_id="s1", profile_id=manifest.profile_id, kind=ScopeKind.GLOBAL,
                         version_id="scope-v1", created_at=manifest.created_at, updated_at=manifest.created_at)
    repository = PersonalContextRepository(tmp_path / "profile.db", key_protector=memory_protector)
    before = sql_state(repository)
    with pytest.raises(ProfileCompatibilityError):
        repository.create_profile_with_global_scope(manifest, scope)
    assert sql_state(repository) == before
```

- [x] Observe RED; add the registered compatibility guard inside _mutation before
  yielding a connection and directly inside special _transaction writers.
  Validate every canonical insertion through native_v1_bytes before encrypting;
  map native incoming decode/version/identity denial to ProfileCompatibilityError
  from None. Match profile/object/scope to envelope arguments. Manifest version
  is current_version_id; record/scope use version_id. Proposal has no canonical
  version_id: preserve its random local envelope version and existing separate
  Sync-derived hash version, rather than equating it to the nested record version.
  Do not invent scope-existence/graph admission in this closed unit. Noncanonical
  local policy/Undo/outbox bodies retain their current serialization formats;
  their current-profile gate and outbox nested canonical validation still apply.
  The canonical branch at the start of _insert_encrypted is:

```python
if object_type in {"manifest", "scope", "record", "proposal"}:
    from .native_codec import NativeProfileDecodeError, native_v1_bytes
    from .native_compatibility import ProfileCompatibilityError
    try:
        plaintext = native_v1_bytes(object_type, value)
    except NativeProfileDecodeError:
        raise ProfileCompatibilityError() from None
else:
    plaintext = self._canonical_payload(value)
```

Place the envelope-identity checks before the existing cipher.encrypt/SQL insert;
retain the current random DEK, HMAC and all local formats. Add this call to
_mutation after its existing freeze/destroyed/profile checks and before yield:

```python
self._require_legacy_on_connection(
    connection, consumer_id=consumer_id, allow_absent=allow_empty
)
```

  Do not gate profile initialization using a nonexistent manifest, or allow
  `allow_empty` to bypass a present blocked profile.
- [x] Guard acquire_first_link_freeze and apply_reviewed_link inside their existing
  transaction before decrypting/rebaselining bodies. Validate incoming remote
  snapshots as V1 before custody/DB changes. Preserve exactly-once freeze and
  key-custody repair semantics; no bypass to resume blocked V2 content.
- [x] Test record+manifest, manifest-only, scope+binding, proposal acceptance/
  resolution/expiry, interview batch, device-only split, runtime policy/binding,
  Undo and outbox writes. Every denied operation preserves row/ciphertext/head/
  outbox/Undo/quarantine state. Inject failure after the first participating insert
  to verify transaction rollback with a healthy V1 positive and existing CAS tests.
  Run decoder/repository and existing targeted repository/service tests GREEN.

### Step 7 — Consume compatibility in bootstrap and current service status

- [x] Add RED integration tests: direct service over a blocked repository returns
  LOCKED/unavailable with profile_present=True and fixed compatibility reason;
  bootstrap returns a locked facade with the same reason. No ABSENT, auto-created
  profile, changed keys, plaintext exception input or schema write.
- [x] Bootstrap validates the compiled registry, opens the existing repository,
  then checks current compatibility before returning an available service.
  Catch ProfileCompatibilityError separately from corruption/OS errors. In service
  status, recompute the current result; do not permanently cache compatibility
  failure as a generic key failure. Return existing operational states rather
  than adding an unreviewed UI state. Invalid manifest still uses generic integrity
  handling, and removed/disabled/locked V1 semantics remain unchanged.
- [x] Update private native call-site tests to supply fixed registered IDs where
  the protected helper signature changes. Never default an unknown route to a
  known route solely to retain a test. An AST audit must show every production
  _decrypt_row/_mutation consumer in the compiled route inventory and only the
  two approved raw-authentication callers. Run integration GREEN.

### Step 8 — Verify real consumer routes and stale prepared state

- [x] In test_native_profile_barrier_routes.py use the real service, context
  builder and profile tool provider. Healthy V1 controls first, then replace the
  manifest with the sealed V2 fixture and exercise the same paths. Tools must
  advertise no catalog and return the existing generic unavailable outcome;
  no hidden counts/global quarantine metadata enter an automatic result.
- [x] Exercise Settings owner snapshot, proposal generation/acceptance, fixed and
  adaptive interview admission, export/recovery snapshot, first-link planning/
  apply and outbox dispatch through existing synthetic fixtures/native owners.
  Assert no profile bytes reach a provider, destination file or Sync staging owner
  after denial; no outbox acknowledgement or quarantine side effect. Recovery and
  Sync stay schema/format V1 and reject V2 input; no mixed recovery format is added.
- [x] Extend app wiring tests: get_personal_context_service reuses the unavailable
  facade; explicit retry remains unavailable; prepare_personal_context_interview_request
  must not recreate a present blocked profile. Use SimpleNamespace/monkeypatch as
  existing tests do; no UI stylesheet or live app/profile is needed.
- [x] Build a V1 context/explanation/tool scope, install blocked V2, then recheck
  explanation, provider catalog and execution. Final current-owner checks deny
  prepared state. A two-connection/event barrier test changes the manifest before
  a write enters its transaction and verifies no stale permit or partial insert.
  Never claim race-safe enabled V2 publication: that protocol remains unimplemented.
- [x] Run all new files GREEN and the exact affected checks below.

### Step 9 — Native targeted verification and static checks

- [x] Run with native Python; retain JUnit counts, failures/skips and commands.
  First group covers every touched profile owner, including all transaction paths:

```bash
.venv/bin/python -m pytest -q Tests/Personal_Context/ Tests/Agents/test_profile_tool_provider.py Tests/Agents/test_profile_tool_scope.py Tests/Agents/test_personal_context_prompt.py Tests/Chat/test_console_personal_context_snapshot.py Tests/Chat/test_console_profile_tool_provider_integration.py Tests/App/test_personal_context_wiring.py Tests/Sync_Interop/test_personal_context_adapter.py Tests/Sync_Interop/test_personal_context_dispatcher.py Tests/Sync_Interop/test_personal_context_first_link.py Tests/Sync_Interop/test_personal_context_first_link_sync.py Tests/Sync_Interop/test_personal_context_capabilities.py Tests/tldw_api/test_personal_context_sync_client.py --junitxml=/private/tmp/native-v2-barrier-owners.xml
```

- [x] Run affected startup/preimport guards (the registry/codec must not eagerly
  import V2 models/schema exporters into the healthy V1 boot closure):

```bash
.venv/bin/python -m pytest -q Tests/Performance/test_app_import_weight.py::test_app_import_own_module_count_stays_at_the_post_diet_size Tests/Performance/test_app_import_weight.py::test_app_import_does_not_load_full_heavy_dependency_set Tests/Performance/test_ui_ready_module_census.py::test_ui_ready_module_census_stays_at_the_pinned_size Tests/Performance/test_screen_preimport_payload_budget.py --junitxml=/private/tmp/native-v2-barrier-imports.xml
```

- [x] Run `ruff check --no-cache --target-version py312` and `ruff format --check
  --no-cache --target-version py312` through `.venv/bin/python -m ruff` on the three
  modified native files, two new production files and three new tests plus helper
  and modified app/export/outbox/Sync-dispatcher tests (13 owned Python files total). Compare baseline failures before attribution;
  do not waive a failure or raise import budgets to make the change pass.
- [x] Check `git diff --check`, exact consumer call-site coverage, lack of qualified
  V2/receipt/activation flags and unchanged schema/AAD constants. Compare the entire
  packages/tldw_profile_core tree and all untouched runtime/test bytes to the
  execution baseline. Verify foreign task SHA-256
  `a42f0ab6557fabb6f4f29fef2de77c8933b85088d51f1f1598b62e6606c8e176`
  and after-marker suffix SHA-256
  `6bbf9adc885d277aad7746debe0dcc39ce6befe56c1b933588d19134428e7a6f`.
  No full suite or offline wheel rebuild: the shared package is unchanged.

### Step 10 — One final review, evidence and task closeout

- [x] Use the native executing-plans final-review workflow: one fresh read-only
  reviewer for the entire unit, not one per stage. Supply the accepted spec/task,
  complete diff, exact targeted receipts and the closed activation boundary.
  Do not spawn a plan self-review agent.
- [x] Resolve Critical/Important findings in one focused pass using observed RED/
  GREEN controls and current authority/transaction integration. Record every
  scope ruling with its practical cost. Minor deferrals require real rationale,
  not an invented checklist; record unknown future qualification honestly.
- [x] Write final component notes with exact behavior/check evidence and remaining
  ADR-202/203/191/server/cohort gates. Link ADR-193 in task notes. Check all AC only
  after actual implementation/review/static/targeted tests pass; CLI marks Done.
- [x] Commit only owned source/tests/docs/task and owned roadmap prefix. Local
  commits follow native checks; the configured pre-commit hook is absent and no
  provider-backed review is invoked; never push/PR/merge or activate any profile. Keep the existing
  worktree. Retain this unit's ignored execution scratch and receipts for later units.

## Source-backed coverage audit

Execution-baseline repository decrypt callers were: acquire_first_link_freeze, get_manifest,
apply_reviewed_link, commit_manifest_version, commit_record_and_manifest,
commit_interview_batch, commit_device_only_split, get_record, get_record_derivation,
get_scope, commit_proposal, commit_synced_proposal, _expire_due_proposals_in_connection,
get_proposal, list_proposals, read_export_snapshot, _resolve_proposal_in_connection,
accept_proposal_and_record, _require_no_record_collision_in_connection,
_rebuild_newly_linked_scope_outbox, _require_unique_workspace_binding,
_get_local_body, get_undo and get_outbox_body. Nested helpers must carry their
outer operation connection/registered route. Enumeration is checked against AST;
new callers cannot silently inherit an acknowledgement.

The common _mutation wrapper covers most writes; acquire_first_link_freeze and
apply_reviewed_link use _transaction directly and need explicit checks. Schema/key
initialization and content-free destructive/key-custody recovery controls are
separate owner paths: preserve existing authority, do not advertise newly enabled
V2 maintenance, infer cleanup completion or remove blocked data automatically.

## Plan self-review and current evidence

This plan covers Unit A only. The accepted spec's enabled V2 admission stamps,
qualification receipts, history/graph semantics, retirement, destination/source
permission, companion cohort and reviewed cutover intentionally map to Units B–F,
not dummy implementations here. Unit A's successful V2 controls prove decoding
only; its runtime outcome is always blocked. Successful V1 controls distinguish
intentional V2 denial from a broken or universally denying application.

Self-review tightened incoming-envelope validation, transactional first-link
coverage, metadata/outbox reads, zero-result getter bypasses, distinct compatibility
errors, proposal envelope/version separation, post-open storage-marker changes
and lazy imports. The raw authentication primitive cannot become an agent
or export bypass. Exact model/field preservation prevents model_construct extras
from disappearing during native V1 serialization.

Status: TASK-25907.22 complete after required-check repair. Expanded owners: 925 passes plus one harness failure corrected and covered by a 34/34 affected rerun. Barrier/memory: 176/176, including all 68 baseline tests. Import/package: 15/15; raw Settings: 13/13. Production V2 stays blocked. Final notes supersede historical incomplete-check checkpoints; worktree and ignored receipts are retained.

Planning verification used native Python 3.12.11: all 86 local documentation links
resolved, nine Python examples parsed, and the AST audit accounted for all 24
existing decryption callers. At that planning baseline the ten stages retained 34 unchecked execution actions and the implementation task retained eight unchecked acceptance criteria; current implementation status is stated above. Prior runtime,
shared-library, test and accepted-design files, plus foreign work, remain unchanged.
These checks validate the plan and preservation boundaries; they do not substitute
for the targeted implementation tests described above.


## User-requested follow-up review and corrective work

The original final review/fix pass above is historical evidence. The user's subsequent explicit request authorizes a fresh read-only follow-up review of HEAD92a82d0fa6. Native diagnostics found an unresolved Important/P2 known-V1 provenance quarantine regression: string_pattern_mismatch and too_long on recognized source hash/reference fields cause whole-collection compatibility errors instead of isolated quarantine. TASK25907.22 remains In Progress with AC7 and AC8 unchecked. No production implementation change is made in this review.

ADR required: no new ADR.
ADR path: backlog/decisions/193-native-v2-profile-compatibility-and-admission.md.
Reason: restoring known-V1 quarantine with restrictive unsupported-data denial implements the accepted Unit A; efficiency/callback observations are recorded without selecting new architecture or activating V2.

- [x] Add independent encrypted getter/list RED regressions for recognized record/proposal/nested-record provenance damage, with healthy controls and unsupported/mixed privacy/schema/extra-field negatives.
- [x] Correct only the demonstrated known-V1 classifier gap; run the new controls GREEN and targeted affected checks, then update the material review disposition.
- [x] Resolve the already required import/memory verification before task closeout; do not claim unchanged timeout implies no added latency.

Snapshot-local repeated manifest checks and callback-shaped decoded-wrapper comparisons are documented as separate efficiency/hardening opportunities in the component notes. The probe establishes no V2 permit bypass. Future work must preserve guarded operation-owned snapshots, no cached admission, no external callbacks, Python>=3.12 and all later activation gates. Required checks, task Done and evidence cleanup remain incomplete.

Corrective execution:18known-field failures RED,37/37 GREEN;754/754 affected owner selection GREEN (817.94s), including the36new encrypted cases and import provenance. Ruff13files/diff check pass. AC8 resolved and checked; AC7 remains open: fresh import receipt2pass/2fail (76.70s), same974/973 and515/500+382468/378740 breaches, plus the previously documented unresolved/excluded memory baseline. No new review or activation. Deferred efficiency/callback minors remain recorded in component notes and the unit ledger. No task-done or Done/cleanup while requirements remain open.


### Required memory verification: snapshot-local getter repair

Native synthetic timing diagnosed255private SQLite admissions per case in d01–d03 (12.44,12.452,12.745s respectively). Ordinary record/scope/proposal getters split current-head, quarantine and body reads across up to3guarded connections. Consolidate these reads into one operation-owned guarded SQLite snapshot, without retaining admission between calls or bypassing private SQLite helpers. Keep the manifest check before body authentication and quarantine corrupt V1 content only after releasing the reader. This is an existing-boundary efficiency/correctness repair under ADR193; no new ADR, V2 permits, schema, AAD or shared-core changes.

- [x] Add real encrypted positive/missing/quarantined getter controls that require one private admission per read.
- [x] Observe RED, consolidate the existing getters, observe GREEN and rerun native benchmark reproducibility and affected repository/barrier checks.
- [x] Record measurements and remaining required import checks before closeout.


### Required startup verification: first-use import repair

The retained native census proves the untouched baseline also breaches UI-ready974/973 and preimport515/500+382468/378740. AC7 requires real repair before closeout. Direct ADR097 implementation: defer the Next Send selection transport to inspector interaction; keep its type references as forward annotations. Move Library constructor-only state/controller imports into its existing constructor import block and unused/annotation-only imports to TYPE_CHECKING. Preserve construction order, screen registry order, field defaults, model identities and actual first navigation. No UI layout/style/token changes; design-language.md read. ADR required:no new ADR; existing ADR097 and ADR193.

- [x] Observe fresh-process RED import closure controls and the retained budget RED receipts.
- [x] Defer the proven first-use imports; run GREEN closure, real Library construction/wiring, selection-inspector controls and all affected import guards.
- [x] Keep limits unchanged unless ADR097 requires downward tightening; record measured effects and any unresolved costs.

Import follow-up measured973UI-ready (PASS),504preimport modules (FAIL) and378572LOC (inside378740). Tracing confirmed conversation/export controller classmethod bindings legitimately retain their state modules. The type-only capture-reader edge is deferred; the test excludes those2legitimate owners. Four more category-specific Settings model/panel modules can move to their existing render, model and restore handlers without changing class identity or category behavior. Extend the fresh-process RED/GREEN control to Settings and rerun web-search/raw-config lifecycle tests and budgets. This remains direct ADR097 import deferral; no new ADR or settings surface.

Expanded owner verification:920passes,2failures. Existing suspend-hook tests import state classes through the Library screen; preserve those historical aliases with a fixed lazy module export table (PEP562, the ADR097 facade option), retaining exact canonical class identity and construction order. Add a fresh-process alias control, with the existing suspend test providing an independent RED. The empty-state test expects a removed preview node; establish the exact original-source failure before correcting its assertion to the current no-preview/start-in-Console contract. No UI behavior or new service interface is selected.

Legacy aliases now resolve to their exact canonical classes lazily, preserving first-use import savings. The resumed suspend test reveals a second inherited fixture defect:__new__ omits its unavailable-navigation owner (also RED under the exact pre-repair source). Construct the real screen and isolate only the navigation callback requiring an active Textual App; retain every timer stop/clear assertion. The empty-state original-source control also fails with the same missing preview. Its corrected test now requires no preview, no export action and an enabled Start in Console control. No production UI behavior is changed for either stale fixture.

Final expanded selection:925pass/1Pilot failure in743.47s. The failing navigation test had already met its exact selected-ID/visible-title postcondition, then stalled in the helper's global Pilot message-pump barrier; it passed independently in2.28s. Remove that post-success global drain, keeping the same bounded polling, ID/title checks and failure diagnostics. Add independent positive/wrong-ID/wrong-title controls whose unrelated global barrier cannot complete, observe RED/GREEN, and rerun the entire affected Library file. Do not alter application code or raise any timeout, and do not label the failed926-case receipt wholly green.


## Final required-check repair evidence

Completed required-check repairs for the closed native V2 barrier under ADR-193 Unit A and ADR-097. Record/scope/proposal getters now read the head, quarantine state and body within one fresh guarded SQLite snapshot; corrupt V1 quarantine writes follow reader closure. No cached admission, bypass flag, schema/AAD or shared-library changes. Library constructor state and Next Send transport plus four Settings category modules load at first use; fixed lazy Library exports preserve canonical class identity. Two stale preview/suspend fixtures reproduced against exact pre-repair sources and were corrected without changing production UI behavior.

Native Python 3.12.11 evidence: the expanded 926-case selection passed 925 and failed one navigation test at an unrelated Pilot queue drain. A test-only repair retains exact ID/title checks and additionally waits for pending navigation to clear, without increasing timeouts. The complete affected Library/closure rerun passes 34/34, including the failed case, all three helper consumers and four positive/wrong-ID/wrong-title/pending controls. Both queue-drain and premature-settlement defects received observed RED/GREEN controls. The failed 926-case receipt remains historical evidence, not a wholly green receipt.

Separately, 176/176 barrier/memory checks pass, including all 68 memory-baseline tests with unchanged 24-case labels/scoring: report setup 263.90s and reproducibility 251.29s fit the existing 300s timeout. Settings raw-config controls pass 13/13; import/package guards pass 15/15. Budgets remain unchanged: app 641/660 modules, UI-ready 973/973, preimport 498/500 modules and 377,190/378,740 LOC, fattest route 114,673/123,319 LOC. No blanket latency or additional headroom claim.

Ruff check/format with target py312 passes all 14 unit Python files. Seven legacy UI/test files retain inherited style debt: immutable-head comparison finds zero added lint diagnostics (only shifted source-line references in existing F811 messages are normalized), and the existing formatter ratchet passes with no changed-line debt or normalized debt growth. Git diff check, raw crypto AST, schema 8/AAD 1, shared-library and independent task/suffix preservation checks pass. No dependency/license changes; profile denial, rollback and privacy controls remain covered by the targeted owner suites.

ADR-193 native qualification, retirement/publication, provider disclosure, foreground source, exact admission, migration and companion-server/cohort prerequisites remain separate and unqualified. Production V2 stays blocked. Source-based tests do not requalify the installed shared-profile wheel. Redundant manifest authentication within owned snapshots and callback-shaped forged decoded-wrapper comparisons remain deferred minors, with no demonstrated production permit bypass. UI-ready import headroom is zero; preimport headroom is two modules/1,550 LOC. Receipts and worktree are retained; no full sweep, provider-backed hook, push/PR/merge or activation.

Ordinary getters delegate to `_get_canonical_head` with their existing fixed routes and exact V1 models; no consumer permissions change. The navigation helper waits for selected ID, expected visible title and cleared pending navigation. Application code and polling limits are unchanged by that test-only repair. The owned local commit records this closeout; its hash and receipt are retained in the ignored unit ledger.
