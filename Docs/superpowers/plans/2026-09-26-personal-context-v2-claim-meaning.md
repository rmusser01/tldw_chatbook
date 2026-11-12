# Inactive V2 Claim Meaning Component Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task inline. Steps use checkbox (`- [ ]`) syntax for tracking. The user's native inline execution preference is already established; do not ask for that choice again.

**Goal:** Implement the strictly validated, exact V2 claim-meaning projection and digest as an inactive shared-core component.

**Architecture:** Add one explicit `tldw_profile_core.v2_meaning` module. It composes unchanged V1 typed payloads and published identity/time rules with closed V2 validity/relations. It does not enter the default canonical union or any native consumer; a later aggregate contract owns evidence, attribution, approval, controls and containing-record checks.

**Tech Stack:** Python ≥3.12 shared profile library, Pydantic 2, pinned RFC 8785, pytest, stdlib SHA-256/JSON and existing setuptools/wheel tooling. Execute natively using the worktree's Python 3.12 venv.

**Spec:** [Accepted V2 contract](../specs/2026-09-26-personal-context-v2-canonical-data-contract-design.md).

**Tracker:** [TASK-25907.19](../../../backlog/tasks/task-25907.19%20-%20Implement-the-inactive-V2-claim-meaning-component.md).

ADR required: no new ADR; direct implementation of accepted [ADR-192](../../../backlog/decisions/192-personal-context-v2-canonical-data-contract.md) and [ADR-201](../../../backlog/decisions/201-versioned-profile-evidence-and-temporal-claims.md).
Reason: approved inactive data-only projection and temporal shapes; no new native storage, grants, provider or activation boundary.

## Global Constraints

- Worktree: `/Users/macbook-dev/.codex/worktrees/personal-context-memory-baseline/tldw_chatbook`; branch `codex/personal-context-memory-baseline`.
- Baseline: `3417d513cf`, which records explicit written approval of the spec. The software completion/review range begins after the planning commit.
- Projection is exactly `{projection:"profile-claim-v2", profile_id, record_id, scope_id, kind, payload, claim_basis, temporal_validity, relations}`. Every key is required.
- IDs: built-in strings, 1–128 codepoints, ≤512 UTF-8 bytes, nonblank, no Cc/Cf or invalid scalar. Preserve spelling and Unicode normalization form.
- Payload ≤16 KiB JCS after existing V1 defaults. Relations 0–4; bound-source basis IDs 1–8. Arrays are sorted/unique and never silently reordered.
- UTC `.sssZ`, aware years 0001–9999, whole-minute offsets and at most millisecond precision. Unknown validity is not standing.
- Complete fresh validation precedes digesting, including unsafe copied/constructed/nested instances. Unknown fields/tags and duplicate JSON members reject; automated diagnostics do not echo input values.
- `SERIALIZED_SCHEMA_VERSION=1`, default V1 `CanonicalObject`, exports, models, schemas, fixtures, binding helper and its 18-field bytes stay unchanged.
- No native import/admission/migration/Sync/recovery/tool/context integration; no profile key, source access, grant, provider, app launch or server qualification.
- No dependency or package-version change. The explicit submodule and new resource entries are the only package additions.
- Targeted native runs only. Do not run the full application suite without user opt-in.
- Independent TASK-25907.10 and bytes after `## Follow-up: generated-answer effectiveness` are foreign. Preserve and exclude them from commits.
- No fetch, push, PR or merge. This plan is not V2 aggregate/schema or profile rollout approval.

## File Map and Coverage

| File | Responsibility |
| --- | --- |
| Create `packages/tldw_profile_core/src/tldw_profile_core/v2_meaning.py` | Closed validity/effect/edge/projection models, strict safe input snapshot and digest. |
| Create `packages/tldw_profile_core/tests/test_v2_meaning.py` | Real model/digest behavior, fixed vectors, adversarial scalar and unsafe-instance rejection. |
| Create `packages/tldw_profile_core/tests/test_v2_meaning_package.py` | Resource parity, offline wheel and actual isolated target installation; public V1 boundary checks. |
| Create `packages/tldw_profile_core/fixtures/claim_meaning/v2/01-standing-preference.json` and identical `src/tldw_profile_core/fixtures/claim_meaning/v2/01-standing-preference.json` | Independently fixed synthetic projection bytes/hash/integrity vector. |
| Modify `packages/tldw_profile_core/pyproject.toml` | Add only package-data glob and distribution data-file for that resource. |
| Create `backlog/docs/personal-context-v2-claim-meaning.md` | API, checked behavior, native receipts and remaining admission/conformance limits. |
| Update TASK-25907.19 and owned roadmap prefix | Actual execution status/evidence only. |

This plan implements spec canonical scalar/default rules for the projection,
the nine-field meaning hash, validity/effect/relation shapes and their local
cross-checks. It does not implement record/proposal/manifest aggregates,
assessment/approval/policy fields, full V2 semantic dialect, retirement receipts,
native graph/head admission, activation, source factories or server pins.
Those are distinct subsequent implementation units, not missing deliverables
silently claimed by this component. Do not reference nonexistent future tasks.

## Execution Setup

Read the Backlog task and accepted spec. Reuse this worktree. Use
superpowers:executing-plans to create this plan's ignored ledger/workspace and
briefs; record each task's BASE, observed RED/GREEN output and commits. Read
test-driven-development and writing-good-tests before retained code. Use native
commands with `PYTHONPATH=.:packages/tldw_profile_core/src` and caches/basetemp
under `/private/tmp`. The shared profile library follows the repository Python ≥3.12 floor. Use
Ruff `--no-cache --target-version py312`.

Pre-flight interfaces: Task 1 produces the model and nested types consumed by
Task 2; Task 2 produces the exact digest used by Task 3. The common scalar/time
aliases consume existing binding and canonical APIs unchanged. Review input
snapshot safety before calling Pydantic or RFC 8785. No user permission is
needed for this authorized inactive library work; norms still apply to external
side effects. Do not select a runtime cutover path as a plan ruling.

### Task 1: Typed validity, relations and the closed projection

**Files:** Create `packages/tldw_profile_core/src/tldw_profile_core/v2_meaning.py`; create `packages/tldw_profile_core/tests/test_v2_meaning.py`.

**Interfaces:**
- Consumes: published `Identity`, `CaptureTime` from `tldw_profile_core.evidence_binding`; existing `ProfilePayload`, its nine exact concrete classes and `FrozenModel`; `canonical_bytes(value: BaseModel) -> bytes`.
- Produces: `ClaimMeaningV2`, `UnknownValidityV2`, `StandingValidityV2`, `IntervalValidityV2`, `UserReviewedBasisV2`, `BoundSourceBasisV2`, `AllTargetValidityV2`, `OverlapEffectV2`, `ReplaceFromEffectV2`, `CorrectionOfV2`, `ChangeFromV2`, `SupersedesV2`, `WorkspaceExceptionToV2` and discriminated `TemporalValidityV2`, `TemporalRelationV2` aliases, and private `_meaning_snapshot(value: object) -> object`. Wire fields below are exact; suffixes are Python names only.

- [x] **Step 1: Write model tests before creating the production module.**

Use function-local API discovery so missing code produces an assertion failure,
not a collection/import error. Keep the literal example independent of production.

```python
import importlib
import importlib.util
from copy import deepcopy

import pytest
from pydantic import ValidationError
from tldw_profile_core.canonical import canonical_bytes


def api():
    assert importlib.util.find_spec("tldw_profile_core.v2_meaning") is not None, "meaning API missing"
    return importlib.import_module("tldw_profile_core.v2_meaning")


def data():
    return {
        "projection": "profile-claim-v2", "profile_id": "p1", "record_id": "r1",
        "scope_id": "s1", "kind": "preference",
        "payload": {"schema_version": 1, "kind": "preference", "subject": "replies", "polarity": "like", "value": "concise"},
        "claim_basis": "direct_user_assertion",
        "temporal_validity": {"kind": "standing", "basis": {"kind": "user_reviewed"}},
        "relations": [],
    }


CANONICAL = b'{"claim_basis":"direct_user_assertion","kind":"preference","payload":{"kind":"preference","polarity":"like","schema_version":1,"subject":"replies","value":"concise"},"profile_id":"p1","projection":"profile-claim-v2","record_id":"r1","relations":[],"scope_id":"s1","temporal_validity":{"basis":{"kind":"user_reviewed"},"kind":"standing"}}'


def test_fixed_projection_and_all_required_keys():
    model = api().ClaimMeaningV2
    assert canonical_bytes(model(**data())) == CANONICAL
    assert len(CANONICAL) == 337
    for key in data():
        fields = data()
        del fields[key]
        with pytest.raises(ValidationError):
            model.model_validate(fields)


@pytest.mark.parametrize("validity", [
    {"kind": "unknown"},
    {"kind": "standing", "basis": {"kind": "user_reviewed"}},
    {"kind": "interval", "valid_from": "2026-09-15T00:00:00Z", "valid_until": None, "basis": {"kind": "user_reviewed"}},
    {"kind": "interval", "valid_from": None, "valid_until": "2026-09-15T00:00:00Z", "basis": {"kind": "bound_source", "binding_ids": ["b1", "b2"]}},
])
def test_validity_has_distinct_closed_shapes(validity):
    instance = api().ClaimMeaningV2(**(data() | {"temporal_validity": validity}))
    assert instance.temporal_validity.kind == validity["kind"]


@pytest.mark.parametrize("validity", [
    {"kind": "unknown", "valid_from": None},
    {"kind": "standing", "basis": {"kind": "bound_source", "binding_ids": ["b1"]}},
    {"kind": "interval", "valid_from": None, "valid_until": None, "basis": {"kind": "user_reviewed"}},
    {"kind": "interval", "valid_from": "2026-09-15T00:00:00Z", "valid_until": "2026-09-15T00:00:00Z", "basis": {"kind": "user_reviewed"}},
    {"kind": "interval", "valid_from": "2026-09-15T00:00:00Z", "valid_until": None, "basis": {"kind": "bound_source", "binding_ids": ["b2", "b1"]}},
])
def test_invalid_validity_rejects(validity):
    with pytest.raises(ValidationError):
        api().ClaimMeaningV2(**(data() | {"temporal_validity": validity}))
```

Extend before implementation with explicit cases for every relation variant:

```python
EDGE_BASE = {"edge_id": "e1", "target_record_id": "old", "target_version_id": "old-v1"}
EDGE_CASES = [
    EDGE_BASE | {"kind": "correction_of", "effect": {"kind": "all_target_validity"}},
    EDGE_BASE | {"kind": "correction_of", "effect": {"kind": "overlap", "valid_from": "2026-09-15T00:00:00Z", "valid_until": None}},
    EDGE_BASE | {"kind": "change_from", "transition_at": "2026-09-15T00:00:00Z"},
    EDGE_BASE | {"kind": "supersedes", "effect": {"kind": "all_target_validity"}, "reason_code": "user_replacement"},
    EDGE_BASE | {"kind": "supersedes", "effect": {"kind": "replace_from", "replace_from": "2026-09-15T00:00:00Z"}, "reason_code": "user_replacement"},
    EDGE_BASE | {"kind": "workspace_exception_to", "workspace_scope_id": "s1"},
]


@pytest.mark.parametrize("edge", EDGE_CASES)
def test_each_relation_has_its_own_effect(edge):
    fields = data() | {"relations": [edge]}
    if edge["kind"] == "change_from":
        fields["temporal_validity"] = {"kind": "interval", "valid_from": edge["transition_at"], "valid_until": None, "basis": {"kind": "user_reviewed"}}
    assert api().ClaimMeaningV2(**fields).relations[0].kind == edge["kind"]


def test_relation_cross_checks_reject_without_native_target_lookup():
    model = api().ClaimMeaningV2
    bad = [
        data() | {"relations": [EDGE_BASE | {"kind": "change_from", "transition_at": "2026-09-15T00:00:00Z"}]},
        data() | {"relations": [EDGE_BASE | {"kind": "workspace_exception_to", "workspace_scope_id": "other"}]},
        data() | {"relations": [EDGE_CASES[0], EDGE_CASES[0]]},
        data() | {"relations": [EDGE_CASES[0], EDGE_CASES[3] | {"edge_id": "e2"}]},
    ]
    for fields in bad:
        with pytest.raises(ValidationError):
            model(**fields)
```

Also parameterize all field omissions/null misuse, unknown/private discriminators,
duplicate/reversed IDs, >4 edges, >8 basis IDs, two change edges, malformed dates,
kind/payload mismatch, payload overflow and legacy basis with non-unknown validity
or nonempty relations. Give custom scalar objects explicit ordinary-string pytest
IDs, per [testing lesson](../../../backlog/docs/lessons-testing-evidence.md).

- [x] **Step 2: Observe meaningful RED.**

Run: `PYTHONPATH=.:packages/tldw_profile_core/src .venv/bin/python -m pytest packages/tldw_profile_core/tests/test_v2_meaning.py -q -o cache_dir=/private/tmp/v2-meaning-cache --basetemp=/private/tmp/v2-meaning-red --junitxml=/private/tmp/v2-meaning-model-red.xml`.
Expected: assertion failures `meaning API missing`; no collection errors. A missing-fixture or automatic-parameter-name error is a test defect, not RED evidence.

- [x] **Step 3: Implement the closed data models and their local invariants.**

Common base: `extra="forbid"`, `frozen=True`, `revalidate_instances="always"`,
`hide_input_in_errors=True`; all value-bearing fields use `Field(repr=False)`.
Use existing `Identity`/`CaptureTime`, unchanged V1 payload defaults and closed
literal tag unions. A pre-validation safe snapshot accepts only exact built-in
containers/scalars and the exact declared component/V1 payload classes; reject
custom containers, scalar/model subclasses and poisoned extra fields before
calling their serializers/methods. JSON entry rejects duplicate member names
with stdlib `object_pairs_hook` before Pydantic decoding can discard them.

The snapshot must inspect exact model instances before reducing them to field
dicts, reject non-null Pydantic extra state, retain every vars() field, and
recursively validate known nested model classes and exact built-in values.
Converting to dict first must not hide poisoned extra-state metadata.

The following are the complete field declarations to translate into classes:

| Class | Required fields |
| --- | --- |
| `UserReviewedBasisV2` | `kind: Literal["user_reviewed"]` |
| `BoundSourceBasisV2` | `kind: Literal["bound_source"]`; `binding_ids: tuple[Identity, ...]`, length 1–8, strictly sorted/unique |
| `UnknownValidityV2` | `kind: Literal["unknown"]` |
| `StandingValidityV2` | `kind: Literal["standing"]`; `basis: UserReviewedBasisV2` |
| `IntervalValidityV2` | `kind: Literal["interval"]`; `valid_from: CaptureTime | None`; `valid_until: CaptureTime | None`; `basis: UserReviewedBasisV2 | BoundSourceBasisV2` discriminated by kind |
| `AllTargetValidityV2` | `kind: Literal["all_target_validity"]` |
| `OverlapEffectV2` | `kind: Literal["overlap"]`; nullable `valid_from`, `valid_until` using CaptureTime |
| `ReplaceFromEffectV2` | `kind: Literal["replace_from"]`; `replace_from: CaptureTime` |
| `CorrectionOfV2` | `kind: Literal["correction_of"]`; `edge_id`, `target_record_id`, `target_version_id`: Identity; `effect: AllTargetValidityV2 | OverlapEffectV2` discriminated by kind |
| `ChangeFromV2` | `kind: Literal["change_from"]`; same three IDs; `transition_at: CaptureTime` |
| `SupersedesV2` | `kind: Literal["supersedes"]`; same IDs; `effect: AllTargetValidityV2 | ReplaceFromEffectV2` discriminated by kind; `reason_code: Identity` |
| `WorkspaceExceptionToV2` | `kind: Literal["workspace_exception_to"]`; same IDs; `workspace_scope_id: Identity` |
| `ClaimMeaningV2` | `projection: Literal["profile-claim-v2"]`; profile/record/scope Identity; kind equals existing closed RecordKind values; existing ProfilePayload; basis `direct_user_assertion|inference|imported_assertion|legacy_unknown`; typed TemporalValidityV2; `relations: tuple[TemporalRelationV2, ...]`, length 0–4 |

No literal field has a default. Arrays preserve admitted order; require exact
list/tuple input, then compare with sorted identity tuples and reject duplicates.
The interval/overlap validation is:

```python
def validate_interval(start, end):
    if start is None and end is None:
        raise ValueError("interval requires a bound")
    if start is not None and end is not None and start >= end:
        raise ValueError("interval bounds must be ordered")
```

After nested validation, projection checks are exactly:

```python
def validate_projection(value):
    if value.payload.kind != value.kind:
        raise ValueError("payload kind mismatch")
    if len(canonical_bytes(value.payload)) > 16 * 1024:
        raise ValueError("payload exceeds canonical byte limit")
    ids = tuple(edge.edge_id for edge in value.relations)
    targets = tuple((edge.target_record_id, edge.target_version_id) for edge in value.relations)
    if ids != tuple(sorted(set(ids))) or len(set(targets)) != len(targets):
        raise ValueError("relations must be unique and sorted")
    changes = tuple(edge for edge in value.relations if edge.kind == "change_from")
    if len(changes) > 1:
        raise ValueError("only one change transition is allowed")
    for edge in changes:
        if value.temporal_validity.kind != "interval" or value.temporal_validity.valid_from != edge.transition_at:
            raise ValueError("change requires exact interval start")
    for edge in value.relations:
        if edge.kind == "workspace_exception_to" and edge.workspace_scope_id != value.scope_id:
            raise ValueError("workspace exception scope mismatch")
    if value.claim_basis == "legacy_unknown" and (value.temporal_validity.kind != "unknown" or value.relations):
        raise ValueError("legacy meaning remains unknown")
    return value
```

Snapshot/revalidation must retain every complete field and reject unknown keys;
do not filter fields to make an unsafe object valid. Revalidate exact V1 payload
instances from field snapshots because their existing base does not promise
always-revalidation. Bound traversal work for unexpected nested input and use
generic errors for unknown keys/discriminator values, so hide-input formatting
does not accidentally repeat a private tag in an exception message. Preserve
valid payload text exactly; catch malformed canonical UTF-8 with a content-free
error. Pydantic model schema is structural component introspection only, never
the V2 required semantic dialect.

Native target existence, current-version self-edge, graph cycles, authorized
overlap and relation effects on stored heads are intentionally not checked:
the projection contains no record-version/head/repository authority. Do not
add a version/DB/callback parameter to the digest to fake containing admission.

- [x] **Step 4: Observe GREEN and run local validators' negative controls.**

Run the Step 2 command with `--basetemp=/private/tmp/v2-meaning-model-green` and
`--junitxml=/private/tmp/v2-meaning-model-green.xml`.
Expected: all model cases pass; no skips/collection errors. Read the receipt.

- [x] **Step 5: Commit only models/tests.**

Run: `git add -- packages/tldw_profile_core/src/tldw_profile_core/v2_meaning.py packages/tldw_profile_core/tests/test_v2_meaning.py`; then `git commit -m "feat(profile-core): add inactive V2 meaning models"`.
Expected: only two owned paths in commit; V1 bytes and foreign files untouched.
Record tests and commit in the execution ledger.

### Task 2: Exact fresh validated meaning digest

**Files:** Modify the same component/test files from Task 1.

**Interfaces:**
- Consumes: `ClaimMeaningV2` and its full typed snapshot validation; existing canonical bytes.
- Produces: `claim_meaning_digest(projection: ClaimMeaningV2) -> str`, lowercase SHA-256; TypeError for non-exact root type, ValidationError/ValueError for unsafe/malformed complete values. No I/O or mutation.

- [x] **Step 1: Add digest and unsafe-input tests first.**

```python
FIXED_SHA = "4d91768974a491f84e8ef67d9b8577975e8615c47065cd97a16260a7f4c28593"
CHANGED_SCOPE_SHA = "de7ef8e9bffe5f5854390682357e8f8cbc08c4706dfa3e2c288910a08f7b1382"


def test_fixed_digest_and_scope_cannot_transplant_meaning():
    module = api()
    assert hasattr(module, "claim_meaning_digest"), "meaning digest missing"
    assert module.claim_meaning_digest(module.ClaimMeaningV2(**data())) == FIXED_SHA
    assert module.claim_meaning_digest(module.ClaimMeaningV2(**(data() | {"scope_id": "s2"}))) == CHANGED_SCOPE_SHA


def test_digest_rejects_unsafe_copy_before_serialization():
    module = api()
    assert hasattr(module, "claim_meaning_digest"), "meaning digest missing"
    original = module.ClaimMeaningV2(**data())
    for updates in ({"scope_id": True}, {"unexpected": "synthetic-private"}, {"payload": None}):
        with pytest.raises((ValidationError, ValueError)):
            module.claim_meaning_digest(original.model_copy(update=updates))
```

Add independent included-field substitutions for profile/record/scope ID, kind
with matching payload, payload wording, basis, validity and relations. Verify
composed/decomposed text produces distinct digests and timezone-equivalent
intervals produce equal canonical values. Domain tag remains fixed; every
nonliteral included value participates. Excluded metadata (`approval_receipt`,
evidence, confidence, policy, version/time) rejects as extra projection input.

Use exact model subclasses and arbitrary raw dicts to check TypeError; direct
`model_construct`, object-attribute poisoning, unsafe nested payload copies,
unknown private discriminator strings and custom str/int/datetime/container
objects to check rejection before serializer/user methods. Include explicit
poison-method counters/raising callbacks rather than asserting only type names.
Check model repr and formatted error messages omit synthetic private input.
Keep `.errors()` diagnostic input out of logs; hide-input affects formatting,
not a claim that Pydantic's structured diagnostics erase input.

- [x] **Step 2: Observe RED.**

Run the targeted file with new digest tests and receipt
`/private/tmp/v2-meaning-digest-red.xml`.
Expected: `meaning digest missing` assertions; prior model controls still pass.

- [x] **Step 3: Implement only the digest entry point.**

```python
from hashlib import sha256


def claim_meaning_digest(projection: ClaimMeaningV2) -> str:
    if type(projection) is not ClaimMeaningV2:
        raise TypeError("projection must be an exact ClaimMeaningV2 instance")
    validated = ClaimMeaningV2.model_validate(_meaning_snapshot(projection))
    return sha256(canonical_bytes(validated)).hexdigest()
```

Use Task 1's safe snapshot to reject malformed complete/nested values and
Pydantic extra-field poisoning before this serialization. Do not trust a prior
validation flag or clone digest, invoke raw `.model_dump()` on an unvalidated
subclass, omit a field, add approval/support/epoch state, or mutate the input.
Add Google-style Args/Returns/Raises documentation to this public API.

- [x] **Step 4: Observe GREEN plus shared-core compatibility.**

Run: `PYTHONPATH=.:packages/tldw_profile_core/src .venv/bin/python -m pytest packages/tldw_profile_core/tests/test_v2_meaning.py packages/tldw_profile_core/tests/test_canonical.py packages/tldw_profile_core/tests/test_models.py packages/tldw_profile_core/tests/test_schema_fixtures.py packages/tldw_profile_core/tests/test_evidence_binding.py -q -o cache_dir=/private/tmp/v2-meaning-cache --basetemp=/private/tmp/v2-meaning-digest-green --junitxml=/private/tmp/v2-meaning-digest-green.xml`.
Expected: all targeted cases pass, zero errors/failures/skips. V1 fixture bytes
and published binding tests remain unchanged, not rebaselined.

- [x] **Step 5: Commit the exact digest/tests.**

Run `git add` only the two Task 1 paths; `git commit -m "feat(profile-core): bind complete V2 claim meaning digest"`.
Expected: only the component/test delta committed; record results in the ledger.

### Task 3: Fixed resources, isolated installation and truthful closeout

**Files:** Create both fixed fixture copies and package test; modify pyproject resource entries; add component doc; update tracker/owned roadmap prefix.

**Interfaces:**
- Consumes: `ClaimMeaningV2`, `claim_meaning_digest`, generic existing canonical bytes/integrity tag.
- Produces: immutable fixture resource at `fixtures/claim_meaning/v2/01-standing-preference.json` in package and distribution, source/installed parity and a documented explicit component API.

- [x] **Step 1: Fix the synthetic oracle and write resource/wheel tests first.**

Fixture envelope keys: `fixture_version:1`, `data` equal to Task 1's example,
`canonical_utf8` equal to CANONICAL decoded without newline, `claim_sha256`
equal FIXED_SHA, `changed_scope_sha256` equal CHANGED_SCOPE_SHA, and
`integrity_tag` equal
`hmac-sha256-v1:adfefb01c7d615f8b0f1d2540da7cf3b0e5de1f0862eb573bc0e32120c900210`.
Public synthetic integrity key is `bytes(range(32))`. These were calculated
independently from the accepted literal 337-byte ASCII vector before production
implementation. Do not obtain expected fields from the new digest/model.

```python
from pathlib import Path
import json
from tldw_profile_core.canonical import canonical_bytes, integrity_tag

ROOT = Path(__file__).parents[1]
FIXTURE = Path("fixtures/claim_meaning/v2/01-standing-preference.json")


def test_fixed_resource_and_distribution_parity():
    assert (ROOT / FIXTURE).exists(), "meaning fixture missing"
    fixture = json.loads((ROOT / FIXTURE).read_text(encoding="utf-8"))
    module = api()
    model = module.ClaimMeaningV2(**fixture["data"])
    assert fixture["canonical_utf8"].encode() == canonical_bytes(model) == CANONICAL
    assert fixture["claim_sha256"] == module.claim_meaning_digest(model) == FIXED_SHA
    assert fixture["integrity_tag"] == integrity_tag(model, bytes(range(32)))
    assert (ROOT / FIXTURE).read_bytes() == (ROOT / "src/tldw_profile_core" / FIXTURE).read_bytes()
```

In the new package test file, repeat the exact test-only api helper and
CANONICAL/FIXED_SHA/CHANGED_SCOPE_SHA literals locally, keeping the probe
self-contained under either pytest import mode; no production fixture loader. Build from a copied package tree in pytest tmp_path using the
existing binding packaging-test pattern, excluding build/dist/egg-info/caches.

Actual commands issued by the test using `subprocess.run` argument lists:

```text
<sys.executable> -m pip --isolated wheel --no-deps --no-build-isolation --no-index --no-cache-dir --wheel-dir <tmp/wheels> <tmp/project>
<sys.executable> -m pip --isolated install --no-deps --no-index --no-cache-dir --target <tmp/installed> <built.whl>
<sys.executable> -I -c <probe-script> <tmp/installed>
```

Use capture_output, check=False and bounded subprocess timeouts; inspect every
return code and captured output on failure. The isolated probe is:

```python
import importlib.resources
import json
import sys
sys.path.insert(0, sys.argv[1])
import tldw_profile_core
from tldw_profile_core.v2_meaning import ClaimMeaningV2, claim_meaning_digest
assert tldw_profile_core.__file__.startswith(sys.argv[1])
assert tldw_profile_core.SERIALIZED_SCHEMA_VERSION == 1
assert "ClaimMeaningV2" not in tldw_profile_core.__all__
fixture = json.loads(importlib.resources.files("tldw_profile_core").joinpath("fixtures/claim_meaning/v2/01-standing-preference.json").read_text(encoding="utf-8"))
assert claim_meaning_digest(ClaimMeaningV2(**fixture["data"])) == "4d91768974a491f84e8ef67d9b8577975e8615c47065cd97a16260a7f4c28593"
```

The test must also inspect wheel archive bytes for both package and `.data/data`
copies, the installed distribution copy, and original V1 resource parity.
An API import alone is not packaging conformance.

- [x] **Step 2: Observe RED for resource absence.**

Run the new package-test file, receipt `/private/tmp/v2-meaning-package-red.xml`.
Expected: meaning resource missing or missing wheel resource assertions, not a
network, build-tool, dependency or collection error. Fix environmental failures
without classifying them as a feature regression.

- [x] **Step 3: Add the fixed resource copies and only their package entries.**

Append `"fixtures/claim_meaning/v2/*.json"` to existing package-data and add:

```toml
"fixtures/claim_meaning/v2" = ["fixtures/claim_meaning/v2/01-standing-preference.json"]
```

Use exact fixture envelope/content above with a trailing file newline but no
newline in canonical_utf8/hash input. Do not regenerate/edit existing fixtures,
schemas, package version, dependencies, V1 exports or binding resources.

- [x] **Step 4: Run final targeted checks and affected-file static analysis.**

Run the Task 2 targeted list plus `packages/tldw_profile_core/tests/test_v2_meaning_package.py`,
receipt `/private/tmp/v2-meaning-final.xml`, separate basetemp. Expected: all
targeted tests pass, no errors/failures/skips. Run `.venv/bin/python -m ruff check`
and `.venv/bin/python -m ruff format --check` on the new module and two tests.
Expected: clean. Scope is the affected shared-core contract, not full app sweep.

Verify against the planning BASE: unchanged V1 models/public exports/canonical
helpers/schemas/fixtures and complete binding helper/resources, no native
`tldw_chatbook/` or existing `Tests/` changes, no new runtime import, resolving
doc links and exact foreign task/suffix hashes. A pyproject resource diff must
contain only this component's two entries. Capture precise commands/results.

- [x] **Step 5: Document, review and commit this software slice.**

Write component doc with actual API example, fixed digest, input semantics,
test/install/static evidence, fresh-validation limitations, and explicit remaining
aggregate/schema/native/server gates. Record no approval/support/source-authority
inference from a digest. Mark Backlog software criteria only after their checks.
Stage explicit owned paths; stage roadmap prefix via index patch, excluding the
foreign suffix. Commit locally with `feat(profile-core): package inactive V2 meaning conformance`.
Expected: only named paths; no runtime integration or unrelated fixture edits.

Run one fresh-context final review of this plan's complete software range with
the plan/spec and below Review Focus. This is the sole delegated review;
implementation stays native inline. Fix actionable findings with observed
RED→GREEN tests and targeted regression/static checks, record rulings and any
deferred minors. Keep TASK-25907.19 In Progress until every criterion, review,
notes/doc and final verification is complete. Mark Done via CLI, commit closeout
locally, retain worktree/branch and preserve all foreign work. No integration
choice/push/merge is made by this plan.

## Review Focus

Inspect every complete-field digest input and unsafe nested/constructed/copy path;
custom Python input callbacks and raw JSON duplicate keys; private discriminator
or error/repr leakage; Unicode/byte bounds and time normalization; literal defaults
or ordering that alter the fixed oracle; false implications of admission/support/
approval from the typed model; unchanged V1/base package APIs and resources;
isolated actual wheel installation and source-tree contamination; and this unit's
deliberate lack of current-record-version, graph, binding-existence and native
authorization checks. Those checks belong to containing aggregates/runtime and
must not be falsely claimed or emulated by a callback.

## Plan Self-Review and Handoff

Coverage: each promised component outcome maps to Tasks 1–3; excluded aggregate,
schema, privacy/runtime and server obligations remain explicit. Interfaces share
the same exact class/function names and fixed projection. Scalar behavior composes
existing contracts without globally tightening V1. Fixed bytes/hash/tag values
are independent oracles. Native inline execution is the user's established
choice; no additional execution-mode question is needed.

Planning is complete when links/file paths, identity and fixed-vector checks
pass. That is not software completion. Every execution checkbox and TASK-25907.19
software criterion remains unchecked until real RED/GREEN/install/review evidence.


## Owner correction: Python runtime floor

The user clarified that Python 3.12 is the minimum for this repository and its
shared profile library. This supersedes the earlier 3.11 metadata assumption.
TASK-25907.19 was reopened with a seventh criterion before correction. Align
`requires-python` with `>=3.12`, verify the built wheel's actual Requires-Python
metadata, and rerun affected native conformance/static checks at py312. The
existing Annotated aliases use the same assignment form as V1 payload aliases;
removing the obsolete TypeAlias annotation preserves their runtime values and
avoids introducing a new Pydantic dependency requirement.

ADR required: no new ADR.
ADR path: backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md; backlog/decisions/192-personal-context-v2-canonical-data-contract.md.
Reason: correct package metadata inconsistent with the existing project floor
and explicit owner instruction; no new runtime/provider or data boundary.
This correction permits the minimum-version metadata edit beyond the original
two resource entries; package version, dependencies and canonical bytes remain
governed by the original plan.
