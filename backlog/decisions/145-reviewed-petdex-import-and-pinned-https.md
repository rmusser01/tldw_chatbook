# ADR-145: Reviewed Petdex imports and pinned HTTPS

Status: Accepted
Date: 2026-09-07
Task: TASK-32031

## Decision

Import Petdex public metadata and sprite atlases directly into unpublished native
Persona Visual drafts. A saved local Persona remains the explicit destination.
Local folder/ZIP and remote URL/slug sources converge on one immutable source model,
atlas parser and reviewed mapping. No downloaded program is executed, no gallery or
CLI is installed, and no third-party assets are published into the project collection.

The remote transport admits only HTTPS on petdex.dev (registry) and assets.petdex.dev
(assets), port 443, with no credentials, fragments or environment proxies. Resolve
all addresses and apply the existing egress IP classifier; connect a socket directly
to an approved numeric address and wrap TLS with the original hostname and normal
certificate verification. Check the connected peer and revalidate every redirect.
Bound wire and decoded bytes, timeout, redirect count and cancellation. Ordinary
egress settings cannot relax these Petdex-specific requirements. Use the standard
library HTTP/TLS stack; no new runtime dependency or global network-policy change.

Classic nine-row atlas semantics are pinned to upstream 5d1844be. Eleven-row sheets
require explicit supported declarations or a user-reviewed row/count/duration map.
Geometry alone never supplies missing semantics. Exact loop duration is distributed
across integer frame durations; every frame remains within native limits. Required
native states use explicit reviewed mappings and disclosed idle fallbacks.

Carry original terms as a canonical validated artwork record in a dedicated string
entry of existing Persona Visual source context, rather than broadening generic
provenance fields. Keep the generic keys' path restrictions. Native import/export,
authoring save/edit and Buddy-to-character conversion preserve that record. Mapping
origin and source digest use bounded non-path scalar provenance. Pack-level artwork
credits remain attached when users edit the pack; asset-level character lineage still
clears when the corresponding image is replaced, per ADR-074.

No schema migration or live runtime coupling. Unknown terms remain unspecified;
source-code licensing is never inferred as an artwork license. Existing native and
Actor Pack exports must carry notices or reject unsupported preservation explicitly.

## Alternatives

An embedded gallery and Petdex CLI would add unrelated discovery/runtime and write
another application's directories. Existing egress fetch alone checks DNS before a
separate resolution by its client, so it cannot satisfy connection-pinning requirements.
Disabling URL import permanently would avoid that implementation but fail the approved
paste-a-pet workflow; the local path remains usable when networking is unavailable.

## Verification

Fixtures cover connection address/SNI/peer, redirects, private and mixed DNS answers,
compressed and streamed size caps, cancellation, classic/scaled/declared/manual
atlases, sentinel pixels, malformed packages, stale destination and actual native
publication with attribution roundtrips. A permitted real fetch is recorded separately
from fixture success. No server preservation claim without its own verification.
