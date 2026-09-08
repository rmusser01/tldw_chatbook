# Import Petdex companions as native Buddies

Status: Implemented in the isolated branch; verification linked below
Date: 2026-09-07
Creator: tldw-project
Programme: [Buddy imports and characters](2026-09-07-buddy-character-programme-design.md)

## Entry points and alternatives

Add Petdex as an import source in Persona Visual authoring. Users paste a public
pet page URL or slug, or choose a local Petdex package/folder. URL and local file
routes converge on one parser and native import review. A saved local Persona is
the destination; use the existing Persona creation path if one does not exist.

Recommend direct metadata/image import. An embedded gallery adds discovery UI and
remote pagination before proving conversion. Requiring Petdex's CLI adds a runtime
dependency and writes another application's pet directories. Neither is needed for
this first release. Available agent tools should call the same review/import service,
not execute downloaded shell installers or treat a website command as authority.

## Verified upstream basis

Petdex source revision: `5d1844be151bb4e613b62b2e50a70bc8e1540d65`.

- [Manifest parser](https://github.com/crafter-station/petdex/blob/5d1844be151bb4e613b62b2e50a70bc8e1540d65/packages/petdex-cli/src/manifest.ts)
  supports object entries and a compact v2 manifest with declared columns and asset base.
- [Atlas detector](https://github.com/crafter-station/petdex/blob/5d1844be151bb4e613b62b2e50a70bc8e1540d65/src/lib/sprite-atlas.ts)
  recognizes eight columns, nine or eleven rows, and integer-cell scaled variants
  of 192×208 frames.
- [Named state definitions](https://github.com/crafter-station/petdex/blob/5d1844be151bb4e613b62b2e50a70bc8e1540d65/src/lib/pet-states.ts)
  specify actual row counts and timings; not every row uses all eight cells.

The website fetch returned 403 during research, so this design is based on public
source inspection, not a successful end-to-end live Petdex download. Public compact
and authenticated full manifests differ; do not require the latter for this flow.

## Acquisition and validation

For a URL/slug, resolve one exact public registry entry and fetch its metadata and
sprite image. Capture creator, source link, available license, version and hashes.
Preserve an absent license as unspecified; Petdex's source-code license is not an
asset license. Never mirror newly imported third-party art into tldw-stuff automatically.

Use the existing HTTP client/network policy with bounded timeout, redirect count and
download size. Validate HTTPS destinations and redirects against the configured
Petdex API/asset hosts; reject private/local targets and credentials in URLs. Do not
execute the returned install command, follow arbitrary metadata file paths, or
fetch optional external dependencies. Unsupported registry versions fail clearly.

For local folders/ZIPs, pin source bytes and validate normalized relative paths,
links, duplicate members, metadata size, image size and total decoded pixels before
conversion. Read only declared metadata and image payloads, treating other text as
data; executable files are never run or imported into a native asset directory.
Malformed declarations or ambiguous sprite choices stop review rather than selecting
the first file. Preserve package-local license/notice files as bounded data in the
reviewed provenance carrier, rather than discarding everything except images/JSON.
Reuse existing private staging and archive validation primitives.

Metadata JSON is limited to 2 MiB, downloaded image bytes to 25 MiB and dimensions
to the native 4096-pixel edge limit; native decoded-pixel and total-pack budgets
also apply. Registry responses have a 10 MiB cap after content decoding. Enforce
streaming read limits rather than trusting Content-Length; reject compressed
responses that exceed the decoded cap. Capture destination authority before fetch
and recheck after every asynchronous stage and before publication. The existing
`Utils/egress.py` guards URLs and redirects but explicitly does not pin the checked
DNS result to the connection. Reuse its policy and bounded-fetch behavior; do not
claim it already validates the actual peer. Before enabling remote import, define
and test a connection-pinning adapter that preserves HTTPS hostname verification,
checks every redirect, and does not allow a proxy to bypass destination checks.
Keep local package import independently usable if that adapter is unavailable.
This transport addition belongs in the Petdex ADR and implementation plan.
Failed or cancelled network/import
operations clean only their owned staging. Authentication/rate-limit errors are
reported explicitly without turning a failed remote import into a false success.

## Native conversion and review

Detect v1/v2 atlas geometry, validate every region and supported state declaration,
then retain the source sprite sheet as a native `sprite_sheet` asset with frame
regions. Preserve per-state frame counts and timing, converting whole-loop duration
to integer frame durations whose sum matches the original. Do not stretch or crop
away character features to force an unsupported layout.

Use only supported, validated state declarations or an explicitly recognized
version's pinned layout. The nine-row classic mapping is established by the source
review; eleven-row geometry alone does not prove row semantics, used-cell counts or
timing. Do not apply the classic mapping to v2 by assumption. If v2 metadata is
insufficient, keep import unpublished and offer manual row/frame-count/timing
mapping with a visual preview. Cancellation creates nothing. A tested official v2
mapping can be added without manual mapping once backed by fixtures. Reject
conflicting metadata/version/dimensions; do not prefer whichever happens to parse.
Persist the mapping source (declared, pinned classic, or user-reviewed).

Suggested required-state mappings:

| Native Buddy state | Petdex source |
| --- | --- |
| idle | idle |
| thinking | review |
| error | failed |
| listening | waiting; otherwise disclosed idle fallback |
| speaking | disclosed idle fallback unless the user maps a suitable sequence |

Retain waving, jumping, directional runs and other available states as custom
animations with original labels. Preview all mappings and identify defaults,
fallbacks and missing states. Manually mapped rows require exact frame counts and
timing; users may exclude them. Honor native minimum/maximum frame durations rather than accepting
invalid timing metadata. Final publication requires the existing activatable-state
validator to succeed; no fabricated happy/sad artwork fills gaps.

Publish through the native Persona Visual draft/review service and existing user
action. Keep local actor ownership, source attribution and optional activation
unchanged. Import creates a new draft; it never replaces a customized pack silently.
Offer Create character from Buddy after saving, using the independent conversion
workflow without a separate Petdex-specific character pathway.

## Acceptance and evidence

- Local v1, fully declared v2, manually mapped v2, scaled sheets and public
  object/compact manifests share a verified source-to-native conversion path, with
  correct region pixels, frame counts and timing. Undeclared v2 never guesses rows.
- Distinct sentinel colors/pixels in rows catch row swaps, off-by-one crops and use
  of unused atlas cells. Missing speaking animation is visibly an idle fallback.
- Unknown versions, oversized/decompression-bomb images, invalid paths, symlinks,
  duplicate ZIP members and untrusted redirects fail without profile mutation.
- Cancel, changed destination Persona, stale source and publication failure leave
  existing packs/preferences intact and remove only owned temporary files.
- A permitted live public pet import is separately recorded when available; fixture
  transport success must never be called live-service verification.
- The saved Buddy works offline and can be exported/reimported. Converting it to
  a character preserves creator/source/terms and renders both motion modes.

ADR required: yes. Define the external data-to-native import boundary, attribution
and reviewed operational mappings before implementation. No Petdex runtime or CLI
dependency, no new shared state catalog, and no automated third-party publication.

## Implementation record

[ADR-134](../../../backlog/decisions/134-reviewed-petdex-import-and-pinned-https.md),
[implementation plan](../plans/2026-09-07-petdex-import.md), and
[verification](../reviews/2026-09-07-petdex-import-verification.md) record the local
Chatbook implementation. No server-native import or notice-preservation claim is made.
