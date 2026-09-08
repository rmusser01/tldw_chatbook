# Import Petdex companions as native Buddies

Status: Design review
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
the first file. Reuse existing private staging and archive validation primitives.

Metadata JSON is limited to 2 MiB, downloaded image bytes to 25 MiB and dimensions
to the native 4096-pixel edge limit; native decoded-pixel and total-pack budgets
also apply. Registry responses have a 10 MiB cap. Failed or cancelled network/import
operations clean only their owned staging. Authentication/rate-limit errors are
reported explicitly without turning a failed remote import into a false success.

## Native conversion and review

Detect v1/v2 atlas geometry, validate every region and supported state declaration,
then retain the source sprite sheet as a native `sprite_sheet` asset with frame
regions. Preserve per-state frame counts and timing, converting whole-loop duration
to integer frame durations whose sum matches the original. Do not stretch or crop
away character features to force an unsupported layout.

Use declared supported named states when available. For absent declarations, use
the pinned nine-row classic state mapping and timings; in an eleven-row sheet,
preserve additional rows as explicitly unnamed custom sequences requiring preview,
not invented emotions. Unknown versions/layouts are rejected rather than guessed.
Persist how defaults were selected in conversion provenance.

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
fallbacks and missing states. Require explicit review of unnamed rows; users may
exclude them. Honor native minimum/maximum frame durations rather than accepting
invalid timing metadata. Final publication requires the existing activatable-state
validator to succeed; no fabricated happy/sad artwork fills gaps.

Publish through the native Persona Visual draft/review service and existing user
action. Keep local actor ownership, source attribution and optional activation
unchanged. Import creates a new draft; it never replaces a customized pack silently.
Offer Create character from Buddy after saving, using the independent conversion
workflow without a separate Petdex-specific character pathway.

## Acceptance and evidence

- Local v1, v2, scaled sheets and public object/compact manifests share a verified
  source-to-native conversion path, with correct region pixels, frame counts and timing.
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
