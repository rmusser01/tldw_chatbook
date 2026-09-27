# Inactive V2 canonical profile aggregates

TASK-25907.20 implements [accepted ADR-192](../decisions/192-personal-context-v2-canonical-data-contract.md)
and the [full data specification](../../Docs/superpowers/specs/2026-09-26-personal-context-v2-canonical-data-contract-design.md).
[Native inline plan](../../Docs/superpowers/plans/2026-09-27-personal-context-v2-aggregates.md).
Python >=3.12. No dependencies or package-version change.

## Delivered boundary

Only explicit `tldw_profile_core.v2_models` and `tldw_profile_core.v2_contract`
imports expose the three V2 aggregates. The root package remains V1, including
`SERIALIZED_SCHEMA_VERSION=1`, default exports, existing union/exporters and
schemas/fixtures. No application consumer imports these modules. No storage,
migration, Sync/recovery, provider enrollment, source reader, permission/grant,
UI or server behavior is activated.

V2 retains unchanged V1 typed payloads/semantic keys and exactly composes the
published owner-version binding and claim-meaning components. All envelope
keys are explicit, including nulls/arrays; only absent model_disclosure defaults
to deny. Nested V1 payload defaults are materialized before canonical/digest
checks. Canonical arrays are unique and sorted, never repaired silently.

The manifest requires all four object/version declarations and all seven
semantic tags, plus explicit retirement epoch. Requirements describe a contract;
they do not acknowledge consumer support or completed cleanup.

Records validate complete meaning SHA-256; each assessment names the current
claim and full 18-field binding digest. Attribution times are bounded by the
containing record/capture; empty spans cannot support or contradict. Approval
binds current meaning/version, confidence binds meaning and salience binds
version. Self-current edges reject; same-record earlier versions remain data-
valid. Native target existence, scope direction, current heads, graph/conflict
checks and publication races remain separate.

Legacy claims retain unknown validity, no evidence/relations/attribution,
deny disclosure and a review hold. Privacy-successor provenance and holds cannot
carry inherited assessments/review/confidence/salience. Whether retained
bindings/relations/source-basis IDs are permitted requires the native policy
owner; the library cannot infer a prohibited source from inert identities.
Deleted records and resolved proposals drop content-bearing fields. These
shapes do not establish physical erasure of retained envelopes or offline copies.

Pending create/update candidates have active content, matching identities and
parent/base, null approval, nested times no later than proposal creation and
exact90-day expiry. Archive/promote propose a target/base action without a
nested record. Resolved receipts carry no proposed record/provenance. Serialized
policy ceilings and attribution are assertions, never foreground consent.

## Explicit safe APIs

```python
from tldw_profile_core.v2_contract import (
    validate_v2_json, validate_v2_object,
    canonical_v2_bytes, v2_object_digest, v2_integrity_tag,
)

record = validate_v2_json(raw_json)
wire = canonical_v2_bytes(record)
digest = v2_object_digest(record)
tag = v2_integrity_tag(record, integrity_key)  # exact32-byte built-in bytes
```

`validate_v2_object(object)` returns a fresh manifest/record/proposal. It accepts
only supported built-in values and exact known models, retains every raw field,
and rejects unsupported subclasses, filtered state dictionaries, extra state,
nonfinite values and custom timezone callbacks. Structural bounds admit all
contract arrays while limiting recursive traversal. Model constructors and
Python-mode Pydantic adapters are supported; unsafe copies remain untrusted.

Public helpers return generic `ValueError("invalid V2 profile object")` or
`ValueError("invalid V2 profile JSON")` with no input/chained diagnostics.
Canonical helpers require an exact aggregate class and revalidate a fresh full
snapshot. Integrity requires an exact32-byte key and authenticates bytes only.
Per-call strict/extra overrides cannot coerce expiry booleans or discard/retain wrong-context fields: V2 classes and unchanged composed leaf shapes are checked before configurable parsing. Direct typed model errors hide input in formatted messages; fields are excluded
from repr. Pydantic structured `errors()`/`json()` can contain input diagnostics
and must not be logged. Generic inherited `model_dump`, compiled serializers
and the old V1 `canonical_bytes`/`integrity_tag` are not V2 validated ingress/
serialization boundaries; use these explicit functions for untrusted/copyable data.

Duplicate-aware JSON uses `validate_v2_json` or a component's `model_validate_json`.
All compiled JSON entrypoints, including `TypeAdapter.validate_json` and wrapper
models' compiled parsers, intentionally refuse. A normal parser discards
member duplication; no later schema/model pass can recover it. Binding version/
offset floats remain rejected, even when the decoded envelope version/counters
use integral JSON-number semantics. Cross-language consumers must preserve this
input distinction instead of relying on a parser that erases it.

## Structural schema and required semantics

Separate source/package `personal-context-v2.json` and
`personal-context-v2-meta.json` resources use V2 URNs and the required semantic
vocabulary. `export_v2_json_schema(Path)` and `export_v2_meta_schema(Path)`
reproduce them. The aggregate schema contains one `$defs` namespace and only
three V2 root shapes; scopes keep their unchanged V1 validator/schema.

The closed16-entry semantic map exactly fixes canonicalization, datetime,
I-JSON counters, five canonical byte ceilings, pending expiry, aggregate rules,
meaning/binding projections, attribution, temporal relations, disclosure and
manifest requirements. Ordinary Draft2020-12 acceptance is structural only:
digest equality, cross-field identities/times, sorted uniqueness, exact component
Python scalar distinctions, Unicode categories/UTF-8 limits and native authority
cannot be inferred from it. Run `validate_v2_object` on the same decoded object;
for raw JSON start with duplicate-aware ingress.

Canonical byte ceilings after defaults: payload/claim/manifest16384,
record65536, proposal98304. Tests exercise exact claim/record Unicode UTF-8
boundaries and one byte over; the other scalar limits can make some aggregate
ceilings unreachable, but their explicit checks remain part of the contract.

## Fixed conformance and native evidence

`fixtures/v2/` contains seven full valid aggregate byte/SHA-256/HMAC vectors:
manifest, active supported record, archived record, deleted record, pending
create/update and resolved proposal. Seven structural-positive/semantic-negative
vectors substitute claim/evidence/approval/time/parent/manifest requirements/
expiry. All input is synthetic. The public fixture key is `bytes(range(32))`.
ASCII compact sorted stdlib JSON and stdlib hashlib/hmac supplied the independent
oracles before production models existed; production JCS never generated its
own expected values. Source/package copies are byte-identical.

Observed native evidence:
- 113 expected missing-API RED cases ->113 GREEN before expanded controls.
- 20 expected missing-schema RED cases ->20 GREEN, including an actual offline
wheel build, target installation and Python `-I` import/resource/fixture probe.
- Self-review: one observed private discriminator-message leak and two malformed
method errors repaired with explicit RED/GREEN regression coverage.
- Final pre-review affected run:658/658 shared profile library tests, no failures,
errors or skips; Python3.12.11 and affected Ruff/format at py312.

Receipts: `/private/tmp/v2-aggregates-red.xml`,
`/private/tmp/v2-aggregates-package-red.xml`,
`/private/tmp/v2-aggregates-package-green.xml`,
`/private/tmp/v2-aggregates-private-red.log`,
`/private/tmp/v2-aggregates-method-red.log`,
`/private/tmp/v2-aggregates-reviewed.xml`.
Fresh read-only gpt-6-astra final review independently reproduced658 passing
cases and all seven stdlib byte/hash/HMAC oracles. It found three Important
issues, no Critical/Minor: strict=False expiry coercion, extra=ignore field loss
and460 dialect self-validation errors from recursive required declarations.
The single fix pass enforces exact bool and contextual keys before configurable
parsing, including composed unchanged V1/binding/meaning shapes, and limits the
meta declaration requirement to documents explicitly declaring the V2 dialect.
The closed16-entry map remains mandatory/exact. Sixty-one review regression
cases initially produced31 expected failures and30 prior rejection passes.
The final run adds six float controls and three defaulted-payload controls and passes728/728 targeted tests with no
errors/skips, actual offline installation, full offline dialect self-validation
and affected py312 static checks. A further three observed RED cases completed the same contextual-key fix for payloads whose V1 kind/schema defaults were absent; containing record kind selects the exact payload shape before defaults. No second reviewer pass or deferred minors.
Receipts: `/private/tmp/v2-aggregates-review-red.xml`,
`/private/tmp/v2-aggregates-complete.xml`,
`/private/tmp/v2-aggregates-guards.json`.
Guard qualifies6737 original core/application/test bytes (resource declarations
excepted), foreign .10/suffix bytes and local documentation links.

## Implementation rulings

- V2-only validator/schema includes the three V2 aggregates. Scopes remain V1,
following the separate V2-only interface requirement. Cost if wrong: a later
explicit mixed-version envelope must negotiate additional dispatch.
- Attribution methods reject recognized credential patterns using the existing
secret validator. Cost if wrong: an otherwise inert name may need replacement.

## Final review scope rulings

Each declined review area stays an explicit future qualification gate, as
required by the accepted inactive boundary:

- Native transactions, heads/scopes/graphs, migration and consumer acknowledgments
are unqualified. Treating them as complete could admit conflicting/stale data.
- Source permission/content, evidence support truth, actor authenticity and
foreground approval are unqualified. Treating serialized assertions as authority
could permit unauthorized reads or fabricated approval.
- Destination enrollment, disclosure enforcement and grant lifecycle are
unqualified. Treating ceilings as consent could expose private data.
- Physical deletion, offline-copy erasure and retirement completion are
unqualified. Treating empty shapes/epochs as completion could resurrect data.
- Companion-server conformance and earlier branch work are outside this unit's
review range. Treating this review as covering them could ship incompatible
server behavior or regressions elsewhere; earlier native evidence stands only
for its own recorded scope.

## Remaining qualification

Full V2 data validation does not enable storage or usable memory. Native atomic
manifest/record admission, version migration/dispatch, all consumer acknowledgments
or retirement with grants withdrawn, dependency-aware metadata retirement,
provider destination/purpose disclosure, approved source inspection/publication
fences and required companion-server fixed-byte/semantic conformance still need
implementation and verification. No server code was inspected. Consolidation,
feedback/repair and generated-answer evaluation are separate work.

No full application sweep, real profile/source/keyring/provider/server/network,
push, PR or merge was performed. Independent .10 work remains untouched.
