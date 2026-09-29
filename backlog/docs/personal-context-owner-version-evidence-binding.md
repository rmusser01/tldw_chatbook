# Personal Context owner-version evidence binding API

The explicit shared-core component binds one asserted conversation-message
identity to its authority/scope/container/object, immutable owner version,
exact representation/span digests, source role and capture time. It does not
resolve a source or admit anything into a Personal Context profile.

Governance: [ADR-201](../decisions/201-versioned-profile-evidence-and-temporal-claims.md).
Contract: [reviewed specification](../../Docs/superpowers/specs/2026-09-26-personal-context-owner-version-evidence-binding-design.md).
Task: [TASK-25907.14](../tasks/task-25907.14%20-%20Add-bounded-owner-version-evidence-bindings-to-shared-profile-core.md).

## Import and synthetic example

```python
import json
from importlib.resources import files

from tldw_profile_core.canonical import canonical_bytes
from tldw_profile_core.evidence_binding import (
    OwnerVersionEvidenceBinding,
    owner_version_evidence_binding_digest,
)

resource = files("tldw_profile_core").joinpath(
    "fixtures/evidence_binding/v1/01-unicode-message.json"
)
example = json.loads(resource.read_text(encoding="utf-8"))
binding = OwnerVersionEvidenceBinding(**example["data"])
assert canonical_bytes(binding) == example["canonical_utf8"].encode("utf-8")
assert owner_version_evidence_binding_digest(binding) == (
    "dae4dd8a13189b7bf86559f7adaab2d6992fe3040a1d255b2c212dc7b15801fb"
)
```

The packaged fixture is entirely synthetic. Its original text `Hi 👋 — café`
and selected `[7, 11)` span demonstrate Unicode codepoint identity. Source text
is fixture data, never a model field or retained real-user excerpt. Actual wheel
conformance tests load both the code and resource from the wheel itself without
installing it or contacting a package index.

## Admission and complete identity

All 18 fields in the specification are required, scalar and non-null. Unknown
keys, including native, source_path, excerpt_ref and binding_digest, are rejected.
component_version is the integer 1; source_kind is conversation_message,
version_kind is owner_immutable, representation_id is message_content_text_v1
and offset_unit is unicode_codepoint. This component format is independent of
Profile V1 and V2 versions.

The six opaque identities are 1–128 Unicode codepoints and at most 512 strict
UTF-8 bytes, nonblank, and exclude Cc/Cf control/format characters. They are not
trimmed or normalized. Scalar validation rejects bytes, bool/float integer
coercion and custom string/integer/datetime subclasses. V1 types are unchanged.

Offsets satisfy `0 <= span_start <= span_end <= 2**53 - 1`. Core has no source
text with which to check the actual end bound or either source/span hash. Empty
spans and syntactically valid arbitrary hashes are structural metadata, not
support. Both SHA-256 fields require exactly 64 lowercase ASCII hex characters.

Capture time uses existing portable RFC3339 timezone/millisecond rules and
normalizes to UTC. Equivalent instants hash identically. The complete binding
digest is lowercase SHA-256 of all 18 validated fields in existing RFC8785
canonical bytes; it includes binding ID, authority, role and capture time, and
is not stored as a self-referential nineteenth field.

The model is frozen and instance admission always revalidates. The digest API
requires the exact component class, then validates a detached copy of its
complete stored field dictionary before serialization. Invalid unchecked
model_construct/model_copy(update=...) results cannot skip digest admission;
valid unchecked data can pass only after this fresh validation. Generic
canonical_bytes still expects an already validated model and is not a new
admission guard for unchecked objects.

| Failure | Public error |
| --- | --- |
| Constructor/model_validate/model_validate_json rejects malformed data | pydantic.ValidationError |
| Digest receives a dict, string, None, another model or subclass | TypeError |
| Digest snapshot has missing, unknown, wrong-typed or inconsistent fields | pydantic.ValidationError |

No automatic field repr or component logging exposes the metadata. Human-readable
validation exceptions suppress input values. Structured ValidationError.errors()
still contains supplied input; callers must not log it as sanitized output.

JSON admission follows Pydantic's existing parser, including last-key-wins for
duplicate keys. The digest identifies validated field values, not raw input JSON
spelling. This component is not a strict I-JSON transport parser. Future signed
or wire admission must reject duplicate keys before model validation.

## Caller-owned evidence checks

The future trusted source owner must authorize before body access and validate
actual scope, container/object, immutable version, exact unmodified content,
codepoint bounds and both digests. It must fence publication against owner,
policy, profile and lifecycle changes. An imported authority or source_role
label, even direct_user_message, is still an untrusted assertion; user messages
can contain quotations and attached instructions. A caller-supplied capture time
is not a trusted clock.

Shape validation and digest equality establish neither source access nor
source ownership, historical original-text retention, semantic support, truth,
reviewed validity or user approval. Frozen Python fields are a convenience,
not an authorization mechanism. Identity metadata itself may disclose sensitive
relationships and must eventually receive source/claim policy and retirement
controls.

This task adds no source lookup, Profile V2 object, record/proposal field, root
export, existing consumer, permission grant, persistence, sync, model prompt,
agent tool or real-user capture. Captured representations, Notes, excerpts and
other source kinds require a reviewed format extension. V2 adoption, source
adapters, server conformance, forgetting/disclosure controls and the known
device-only disclosure gap remain separate future work. V1 schemas/fixtures,
canonical bytes and legacy unverified provenance retain their existing meanings.
