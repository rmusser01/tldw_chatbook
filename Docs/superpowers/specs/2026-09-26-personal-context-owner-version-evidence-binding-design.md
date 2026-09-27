# Personal Context owner-version evidence binding component

Status: Implemented locally for the reviewed data-only scope, 2026-09-26; no runtime or profile schema rollout
Date: 2026-09-26
Task: TASK-25907.14

ADR required: no new ADR for this scoped data-only component.
ADR path: [ADR-185](../../../backlog/decisions/185-versioned-profile-evidence-and-temporal-claims.md).
Reason: implement a subset of its accepted exact-binding convention without
changing repository ownership, source-access policy, persistence or consumers.
Any later adapter/service boundary still needs its own scoped ADR check.

## Goal and scope

Supply a strict, immutable shared-core value that binds one conversation-message
representation to an opaque authority/scope/object identity, an owner version,
a half-open span and its exact digests. This is the input contract needed before
a native memory source adapter can validate an exact binding.

This component is a proposed building block for Profile V2, not a V2 profile
manifest, record or proposal. Its component format version 1 is independent of
SERIALIZED_SCHEMA_VERSION, which remains 1. No existing V1 record can acquire a
binding by pairing its separate legacy IDs/hashes. Imported values remain data.

The [native source audit](../../../backlog/docs/personal-context-source-readiness-audit.md)
found that live conversation text has an owner revision identity, but retained
historical projections may be sanitized or absent. The first component handles
owner-version identity only; it promises no historical text retention. Source
resolution remains separate because a profile record-read grant does not grant
conversation access.

## Alternatives and recommendation

1. **Standalone owner-version component (recommended).** Fix exact fields,
   bounds and identity semantics now; a future source owner can validate them.
   It delivers a shared, testable contract without installing a new fact store.
2. **Build a live resolver immediately.** Requires a trusted source-read action,
   host ownership checks and publication fencing that existing profile grants
   do not supply. A caller boolean or imported authority ID cannot fill that gap.
3. **Roll out all Profile V2 together.** Includes manifest negotiation, temporal
   admission, retirement, disclosure and server conformance. It is too large
   for this independently testable prerequisite.

## Public surface

Add `packages/tldw_profile_core/src/tldw_profile_core/evidence_binding.py`.
Its public model is `OwnerVersionEvidenceBinding`, a `FrozenModel` subclass with
exactly the required fields and validation below. Its digest function has the
signature `owner_version_evidence_binding_digest(binding: OwnerVersionEvidenceBinding) -> str`;
it returns lowercase SHA-256 of `canonical_bytes(binding)` after revalidation.

Import both names explicitly from this module. Use the existing `canonical_bytes`
function for canonical serialization. Do not add package-root exports, alter
the canonical-object union, SERIALIZED_SCHEMA_VERSION, V1 schema/fixture
selection or current ProfileProvenance. The digest helper requires the exact component class instance,
rejecting subclasses, dicts, raw source text and other BaseModel types with
TypeError. Its field revalidation rejects malformed data with Pydantic
ValidationError; validated construction uses the same validation error family.

## Required fields and semantics

Every field below is required in parsed input and is always serialized. There
are no omitted defaults, null values, optional handles or derived digest fields.

| Field | Accepted value and meaning |
| --- | --- |
| `component_version` | Built-in integer `1`; independent component format. |
| `binding_id` | Bounded opaque identity of this binding, included in its digest. |
| `authority_kind` | `local_profile` or `authenticated_tenant`; a recorded assertion, never a credential. |
| `authority_id` | Bounded opaque source-authority identity; not assumed equal to a profile ID. |
| `governance_scope_id` | Bounded opaque source-policy scope identity. |
| `source_kind` | Exactly `conversation_message`. |
| `source_container_id` | Bounded opaque conversation identity, part of complete source identity. |
| `source_object_id` | Bounded opaque message identity within that owner/container. |
| `version_kind` | Exactly `owner_immutable`; captured-representation tokens are outside this format. |
| `source_version_id` | Bounded opaque immutable owner revision token; runtime must verify its actual meaning. |
| `representation_id` | Exactly `message_content_text_v1`: the unmodified, decoded message content text, excluding role, title, attachments and provider/UI transformations. |
| `representation_sha256` | Exactly 64 lowercase ASCII hexadecimal characters: SHA-256 of strict UTF-8 for the entire representation. |
| `offset_unit` | Exactly `unicode_codepoint`. |
| `span_start` | Built-in integer, inclusive zero-based offset. |
| `span_end` | Built-in integer, exclusive zero-based offset. |
| `span_sha256` | Exactly 64 lowercase ASCII hexadecimal characters: SHA-256 of strict UTF-8 for representation[start:end]. |
| `source_role` | `direct_user_message`, `quoted_material`, `attachment`, `tool_result` or `imported_material`; recorded origin annotation, not proof of authorship or intent. |
| `captured_at` | Existing portable RFC3339 datetime semantics, normalized to UTC milliseconds by canonical serialization. |

For opaque identities, accept only built-in strings of 1–128 Unicode codepoints
and at most 512 strict UTF-8 bytes. Reject wholly whitespace strings, unpaired
surrogates, and Unicode Cc/Cf control/format characters; preserve all other
characters exactly without trimming, case folding or normalization. Do not
change V1's more permissive identity or hexadecimal types.

String scalars, including enum values and timestamp strings, reject bytes,
numeric coercion and custom string subclasses. Offsets and component_version
reject bool, floats (including integral floats), numeric strings and custom
integer subclasses. For captured_at, accept an exact built-in datetime or a
built-in RFC3339 string under the existing timezone/precision constraints;
reject datetime subclasses. Reuse UTC normalization rather than creating a
second date grammar. A timezone offset for the same instant yields identical
canonical bytes. A caller-supplied time is not a trusted capture clock.

Require `0 <= span_start <= span_end <= 2**53 - 1`, preserving portable exact
JSON integers. An empty span is valid identity metadata and cannot by itself
establish meaningful support. Since source text is deliberately absent, core
cannot validate end <= actual text length or either digest against that text.
No normalization, line-ending conversion, joined fields or UTF-16/grapheme
conversion occurs in this component.

The model is frozen, forbids unknown keys and validates cross-field offset
ordering. Supported admission is normal validated construction or model_validate
/ model_validate_json, with equivalent scalar semantics on both paths. Unsafe
Pydantic model_construct and model_copy(update=...) are not validation APIs;
the digest boundary must revalidate complete stored field data, including any
unexpected keys inserted by unsafe copying, so these paths cannot produce an
accepted digest for malformed fields. Validate before serialization; dropping
unknown keys through model_dump would hide an invalid copy. This check does
not authenticate data. Existing canonical_bytes still expects already validated
models and is not a new admission gate for unchecked objects.

Set instance revalidation to always for model_validate(existing_instance).
The digest additionally validates a detached complete field snapshot before
canonical serialization. Frozen fields prevent normal assignment, not arbitrary
Python object mutation; this value is not a security capability.

JSON admission follows the existing Pydantic parser, including last-key-wins
handling of duplicate JSON keys. This component hashes validated field values,
not the original input spelling. It is not a strict I-JSON transport parser;
future signed/wire admission must reject duplicate keys before model admission.
Imported role/authority labels remain untrusted on every admission path.

All fields are bounded scalars. Future containing records must separately
validate binding uniqueness/count and their canonical byte budget; this task
does not add a binding list or claim metadata to any record. Repr/logging must
not automatically reveal the binding's source identities; field errors and
canonical bytes must not be logged by this component. Suppress field repr and
use hide_input_in_errors for the human-readable validation exception. Structured
ValidationError.errors() still exposes supplied input; callers must not log it
as a sanitized error.

## Complete identity and fixed synthetic vector

The binding digest is lowercase SHA-256 of RFC8785 JCS UTF-8 bytes for **all 18
fields above**. It is computed, not serialized as a nineteenth field. No source
text, access observations, support assessment or approval receipt is included.
Changing any field, including binding ID, role or capture time, changes the
identity. Equivalent timezone spelling is normalized before hashing.

For exact text `Hi 👋 — café`, the source has 11 codepoints and `[7, 11)` selects
`café`. The proposed component's canonical bytes are this single UTF-8 line:

```json
{"authority_id":"synthetic-authority-01","authority_kind":"local_profile","binding_id":"synthetic-binding-01","captured_at":"2026-09-26T16:00:00.000Z","component_version":1,"governance_scope_id":"synthetic-scope-01","offset_unit":"unicode_codepoint","representation_id":"message_content_text_v1","representation_sha256":"bdd355f021e0095bbd9ce9da729cfd0b9556b01bc01c5b59461ebc4e12e55bef","source_container_id":"synthetic-conversation-01","source_kind":"conversation_message","source_object_id":"synthetic-message-01","source_role":"direct_user_message","source_version_id":"synthetic-revision-01","span_end":11,"span_sha256":"850f7dc43910ff890f8879c0ed26fe697c93a067ad93a7d50f466a7028a9bf4e","span_start":7,"version_kind":"owner_immutable"}
```

Expected binding SHA-256: `dae4dd8a13189b7bf86559f7adaab2d6992fe3040a1d255b2c212dc7b15801fb`.

This fixed vector was calculated during design review from the existing JCS
serializer and independently checked against sorted, compact standard-library
JSON for this ASCII-only metadata. It is a specification example, not evidence
that the proposed model exists or that general JCS equals json.dumps. The
source/span hashes come from the already implemented exact-text helper.

## Ownership, privacy and integration boundaries

Core validates structure and complete identity. A future trusted runtime must
check authority/scope, allowed owner/container/object, immutable version,
representation availability, actual codepoint bounds and both text digests.
It must authorize before body access and fence publication against owner,
policy, profile and lifecycle changes. Neither local_profile nor an owner
version label inside data grants authority or certifies an immutable revision.

A user-role message can contain quotes or attached instructions. Native capture
must establish actual source origin; source_role remains an untrusted assertion
when imported. Hash equality does not establish semantic support, current
truth, reviewed validity, human approval or trusted capture time. The component
contains no capability, native flag, path, URL or executable resolver hint.

Canonical binding metadata can itself expose relationships and source identity.
Do not persist, sync, inject into prompts, expose to agent tools or resolve it
from a real profile in this task. Future runtime adoption must satisfy ADR-185's
retirement/forgetting and disclosure controls before portable evidence ships.
Local-only sources must not become portable through this component. An
attachment origin annotation does not make attachment bytes resolvable here.

Retained excerpts, captured-representation tokens, Notes, media and cross-source
transforms are excluded. Adding them requires an explicit reviewed component
format change and updated canonical fixtures; it must not extend these format-1
bytes silently. Full V2 admission and server/runtime negotiation remain separate.

## Verification requirements for the implementation

- Positive validated construction and JSON round trips, fixed canonical bytes
  and digest, model immutability and strict offset boundaries including empty spans.
- Negative controls for missing/extra fields, scalar coercion, subclasses,
  malformed IDs/hashes/dates, non-UTF8 text and reversed/unsafe integer offsets.
  Include valid controls adjacent to each rejection class.
- IDs at 128 codepoints/512 bytes accepted where permitted; 129 codepoints,
  control/format characters, blank identities and surrogates rejected. The
  512-byte limit is implied at 128 strict UTF-8 codepoints, but remains explicit.
- Change each bound field independently while retaining validity and require
  the digest to differ; independently test full/span hash separation. Fixed
  discriminator changes are rejected rather than used as valid mutations.
- Equivalent UTC instants serialize identically; NFC/decomposed identities
  remain distinct. Empty span metadata is not asserted to be supported evidence.
- Digest admission revalidates unsafe constructed/copied objects, including
  unknown keys from model_copy(update=...); rejects unsupported input and
  malformed values instead of accepting an invalid object identity.
- Check model JSON-schema structure and packaged/test fixture parity. JSON
  Schema alone is not claimed to enforce UTF-8 byte limits or semantic rules;
  native model conformance remains required before any server/runtime use.
- Preserve all existing V1 core tests, root exports, schemas and both fixture
  trees byte-for-byte. Run only the affected core tests under native .venv
  Python with dedicated temporary roots; Ruff and format on modified Python.
  No provider calls, full application sweep or real-profile access.

The implementation may add only these concrete component artifacts and the
owned task, roadmap, implementation plan and execution-review entries:

- `packages/tldw_profile_core/src/tldw_profile_core/evidence_binding.py`
- `packages/tldw_profile_core/tests/test_evidence_binding.py`
- `packages/tldw_profile_core/pyproject.toml` (new fixture paths only)
- `packages/tldw_profile_core/fixtures/evidence_binding/v1/01-unicode-message.json`
- `packages/tldw_profile_core/src/tldw_profile_core/fixtures/evidence_binding/v1/01-unicode-message.json`
- `backlog/docs/personal-context-owner-version-evidence-binding.md`

The two fixture copies contain the same synthetic vector above and are checked
for byte parity. They are selected explicitly by the new component tests; the
existing V1 profile fixture reader is unchanged. Add the new resource glob to
setuptools package-data and its mirrored data-files entry without changing V1
entries, dependencies or version. Build a wheel offline in a temporary copy and
load its own module/fixture from that wheel; source-tree parity alone does not
prove packaging. Do not rewrite historical
baseline evidence or independent TASK-25907.10 additions.

## Initial specification review record (before implementation)

Inline specification review checked scope, ADR alignment, field/digest coverage,
strict input semantics, privacy boundaries and testable negative controls. It
resolved five potential ambiguities: owner-version identity does not promise
original historical text; format version 1 is not Profile V2 support; unsafe
Pydantic construction must not bypass digest-boundary validation; the exact
artifact paths are scoped; and every documented field is required on admission.
No production code, fixture, schema or source-access interface was implemented
for this review.

Native documentation checks passed: 56 local links, 15 unique task-family IDs,
seven unchecked implementation criteria, all 18 documented fields, the fixed
739-byte canonical vector and its complete/source/span SHA-256 values. Prior
tasks, runtime paths and the independently owned evaluation task/roadmap suffix
were preserved. No application test run is claimed for this documentation-only
change; the component's targeted tests are implementation acceptance criteria.

## Requested second review, 2026-09-26

Resolved the packaging omission: the existing package-data configuration names
only fixtures/v1, so the component's new fixture path must be explicitly added
and qualified from an offline built wheel. Clarified instance revalidation,
human-readable versus structured error privacy, and the existing JSON parser's
duplicate-key limitation. A synthetic native Pydantic 2.12.5 probe rejected both
wrong-typed and unknown-key copies with revalidate_instances=always; it confirmed
last-key-wins JSON parsing. No V1 parser or runtime transport is changed.

The user's Looks good subject to review authorizes this corrected data-only
scope and native planning/execution. Package configuration is an implementation
of the promised packaged fixture, not a new dependency or runtime boundary.
