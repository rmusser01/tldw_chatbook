# Inactive V2 claim meaning component

[TASK-25907.19](../tasks/task-25907.19%20-%20Implement-the-inactive-V2-claim-meaning-component.md)
implements the first data-only slice of the accepted
[V2 contract](../../Docs/superpowers/specs/2026-09-26-personal-context-v2-canonical-data-contract-design.md).
[ADR-192](../decisions/192-personal-context-v2-canonical-data-contract.md) and
[ADR-201](../decisions/201-versioned-profile-evidence-and-temporal-claims.md)
apply directly; no new architecture decision or runtime activation occurs.
The [implementation plan](../../Docs/superpowers/plans/2026-09-26-personal-context-v2-claim-meaning.md)
is scoped to typed validity/relations, the meaning projection, its digest and
fixed package resources.

## Explicit API

Import the component explicitly; default package exports and serialized schema
dispatch remain V1.

```python
from tldw_profile_core.v2_meaning import ClaimMeaningV2, claim_meaning_digest

meaning = ClaimMeaningV2(
    projection="profile-claim-v2",
    profile_id="p1", record_id="r1", scope_id="s1", kind="preference",
    payload={"kind": "preference", "subject": "replies", "polarity": "like", "value": "concise"},
    claim_basis="direct_user_assertion",
    temporal_validity={"kind": "standing", "basis": {"kind": "user_reviewed"}},
    relations=[],
)
assert claim_meaning_digest(meaning) == "4d91768974a491f84e8ef67d9b8577975e8615c47065cd97a16260a7f4c28593"
```

All nine projection fields are required and non-null. The nested payload uses
the unchanged nine V1 typed payloads and their schema/kind defaults; a missing
payload kind is selected from the required enclosing kind. Explicit kind
mismatches reject. Canonical payload content is limited to 16 KiB UTF-8.

Unknown validity has no dates. Standing validity requires an asserted
`user_reviewed` basis. Interval validity has required nullable start/end,
at least one bound and strictly ordered non-null endpoints. Its basis is either
`user_reviewed` or 1–8 sorted unique asserted binding IDs. Dates use the existing
portable timezone-aware, millisecond-precision contract and normalize to UTC;
Unicode text and identity spelling remain exact.

Up to four sorted, unique relation edges reference exact record/version IDs.
`correction_of` declares all-target or explicit overlap effects; `change_from`
requires a transition matching the new interval start; `supersedes` declares
replacement effect and reason; `workspace_exception_to` matches the containing
scope. Duplicate exact targets and multiple change edges reject. A
`legacy_unknown` basis requires unknown validity and no relations.

## Fresh digest validation

[The component](../../packages/tldw_profile_core/src/tldw_profile_core/v2_meaning.py)
accepts only an exact `ClaimMeaningV2` instance at the digest boundary. It copies
complete field state, rejects poisoned extra state and custom model/container/
scalar subclasses before serialization, and freshly validates nested V1 payload
instances. Datetime inputs permit exact stdlib `timezone`/`ZoneInfo` timezone
objects; custom timezone and metaclass callbacks reject before invocation.
Unsafe `model_construct`, `model_copy` and direct attribute changes
cannot rely on earlier validation. Input is not mutated.

The typed models forbid extra fields, are frozen and hide field values in repr
and formatted validation errors. Duplicate raw JSON member names reject before
last-value-wins decoding. Use the explicit component `model_validate_json`
method for raw JSON; it performs duplicate-aware parsing then Python-mode
validation. Compiled `TypeAdapter(...).validate_json` and containing-model JSON
validation deliberately reject these components, including valid raw JSON,
because Pydantic discards duplicate members before model validators run.
The rejection is a content-free `TypeError`, so default outer wrappers cannot
reformat it with raw input.
Python-mode `TypeAdapter(...).validate_python` remains supported. Future
aggregates must provide their own guarded raw JSON ingress; this component does
not promise drop-in compiled JSON parsing. Snapshot traversal is bounded; errors use generic
messages for unknown/private keys and tags. Pydantic `.errors()` still contains
diagnostic input and must not be treated as content-free telemetry.

The hash is SHA-256 over JCS UTF-8 of exactly projection/profile/record/scope/
kind/payload/claim basis/validity/relations. Scope-only substitution to `s2`
produces `de7ef8e9bffe5f5854390682357e8f8cbc08c4706dfa3e2c288910a08f7b1382`.
Evidence, assessment, approval, confidence, salience, policy, storage versions/
timestamps and privacy metadata are excluded and reject as projection extras.
The typed value and digest grant no source access, support determination,
current-version approval or runtime authority.

## Conformance and verification

The [distribution fixture](../../packages/tldw_profile_core/fixtures/claim_meaning/v2/01-standing-preference.json)
and [package resource](../../packages/tldw_profile_core/src/tldw_profile_core/fixtures/claim_meaning/v2/01-standing-preference.json)
contain the independently fixed 337-byte oracle, hashes and existing generic
integrity tag using the public synthetic key `bytes(range(32))`. File newline
is outside the canonical hash input. Expected values are not generated by the
new production model or digest.

Native evidence on Python 3.12.11:

- Model RED: 97 expected missing-API failures, zero collection errors; GREEN: 97 passed.
- Digest RED: 34 expected missing-digest failures with the prior 97 passing; compatibility GREEN: 396 passed.
- Packaging RED: two missing-resource assertions after a successful offline build; final targeted GREEN after review fixes: 418 passed, zero failures/errors/skips.
- The package test builds offline from a copied package, installs the wheel into a temporary target and runs `python -I` against that installed module. Package/distribution copies, all V1 schemas/fixtures and the existing binding resource match source bytes.
- Self-review callback regressions: two timezone and three metaclass RED failures; GREEN included in the final 406-case run.
- Ruff check uses `--no-cache --target-version py312` for the shared profile library's Python 3.12 floor; format check covers the component and its two new tests.

The targeted list is `test_v2_meaning.py`, `test_v2_meaning_package.py`,
`test_canonical.py`, `test_models.py`, `test_schema_fixtures.py` and
`test_evidence_binding.py` under `packages/tldw_profile_core/tests/`, with
`PYTHONPATH=.:packages/tldw_profile_core/src`. Receipts are
`/private/tmp/v2-meaning-model-red.xml`, `/private/tmp/v2-meaning-digest-red.xml`,
`/private/tmp/v2-meaning-package-red.xml` and `/private/tmp/v2-meaning-final.xml`.
These temporary receipts are local execution evidence, not shipped resources.
A fresh read-only reviewer independently ran the 406-case pre-fix selection and
found two Important boundary gaps. Six root/nested state-dictionary regressions
and four alternate JSON-mode regressions failed before repair; all ten passed
in the final 418-case run. No Critical or Minor findings were returned. The
single fix pass rejects custom model-state dictionaries before copying and
rejects compiled Pydantic JSON-mode entry points before unsafe normalized input
can qualify. A further RED formatting test showed a default outer model repeating rejected
input; generic `TypeError` fixed that within the same pass. All 418 targeted
cases passed with the original static checks. The initial py311 assumption
was superseded by the owner-confirmed Python 3.12 floor; the native interpreter
was already Python 3.12.11. Final
prior-byte/foreign guards preserve 6732 tracked package/application/test files and
exact independent follow-up hashes. TASK-25907.19 is complete as this inactive
component only; all runtime and aggregate gates remain open.

## Remaining gates

This projection is not a V2 canonical record, proposal, manifest, schema or
required semantic dialect. Model JSON schema is structural introspection only.
The component cannot validate target existence, current-record-version
self-edges, graph cycles, actual authorized overlap or binding existence;
containing aggregate/native admission owns those checks. Asserted source-role
and review basis are untrusted inputs, not proof of evidence or consent.

No application import, schema migration, profile storage, source inspection,
retrieval/consolidation, policy/approval flow, consumer retirement, privacy
metadata deletion, Sync/recovery, provider/server or UI path uses this component.
Those remain separately qualified units under ADR-102/201/202/203/191/192.
No dependency, package version, published V1 exports/helpers/schemas/fixtures or
existing complete-binding bytes change. This evidence establishes only the
inactive component; no full application sweep or live-user/provider/server
conformance is claimed.

## Review rulings

- Python floor correction: the owner superseded the original py311 ruling. The shared profile library and application require Python >=3.12; package and wheel metadata plus static validation must agree. Retaining the older claim would misstate supported runtimes.
- Target/binding existence and target authorization remain aggregate/repository checks — this inert projection has no repository authority and grants none — wrong boundary would leave unauthorized future claims unchecked.
- Current-record-version self-edges remain aggregate checks — projection deliberately has no containing version — wrong boundary could let future records supersede themselves.
- Cycles, stale heads, conflicting effects and actual authorized overlap remain native admission checks — only local typed invariants are implemented — wrong boundary could admit inconsistent future stored state.
- Full V2 aggregates/dialect/migration/activation/retirement/Sync/server remain separate units — this software stays inactive — treating it as qualified runtime could cause incompatible rollout.
- Existing V1/base behavior outside this software range is reference-only — 6732 prior bytes and targeted compatibility are preserved — historical bugs outside the scope can remain.
- Foreign .10 task and roadmap suffix stay outside scope — their exact hashes are preserved — reviewing or committing them here could misattribute unrelated work.
- Tracker/document closeout follows final fixes and verification — pending status is intentional until all AC pass — premature completion would falsely certify unfinished work.

## Python floor correction

The shared profile library is the local `packages/tldw_profile_core` Python
package, containing profile models, validation and hashing. Python 3.12 is its
minimum, matching the application. The previous 3.11 claim and metadata are
withdrawn. Verification includes actual built-wheel Requires-Python metadata
and py312 static checks, using the native Python 3.12.11 interpreter.

Floor correction verified: one built-wheel minimum-version assertion failed
before the metadata fix; all 418 targeted cases pass afterwards, including
actual offline wheel installation. Affected Ruff check/format pass with
`--no-cache --target-version py312`. Preservation checks retain all 6732 prior
contract/application/test files and exact independent follow-up bytes.
Correction receipts: `/private/tmp/v2-floor-red.xml`,
`/private/tmp/v2-floor-green.xml`, `/private/tmp/v2-floor-qualified.json`.
