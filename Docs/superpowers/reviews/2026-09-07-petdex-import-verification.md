# Petdex import verification

Creator: tldw-project
Task: TASK-32031
Decision: [ADR-134](../../../backlog/decisions/134-reviewed-petdex-import-and-pinned-https.md)
Plan: [Petdex import](../plans/2026-09-07-petdex-import.md)
Branch: codex/buddy-import-design, based on the completed Buddy conversion commit 451137c93.

## Delivered behavior

A saved active local Persona exposes Petdex import and native Buddy export.
Public URL/slug and downloaded ZIP/folder sources become immutable reviewed inputs.
The dialog shows source statements, exact state definitions, required mappings,
per-state preview and idle fallbacks. Use draft hands a validated native archive to
the existing unpublished importer; Save Pack remains the publication action.

Network reads pin public DNS answers to the actual socket, verify TLS for the
original host, verify the peer, ignore proxies, and enforce redirect, wire, decoded
and total-time limits. Local admission rejects traversal, duplicates, links,
special files, ambiguous sprite declarations, NUL-truncated ZIP names and bounded
package expansion. Nested notices and differing source credit statements survive.

Classic atlas facts are pinned to upstream commit
5d1844be151bb4e613b62b2e50a70bc8e1540d65:
[states](https://github.com/crafter-station/petdex/blob/5d1844be151bb4e613b62b2e50a70bc8e1540d65/src/lib/pet-states.ts),
[geometry](https://github.com/crafter-station/petdex/blob/5d1844be151bb4e613b62b2e50a70bc8e1540d65/src/lib/sprite-atlas.ts),
and [manifest](https://github.com/crafter-station/petdex/blob/5d1844be151bb4e613b62b2e50a70bc8e1540d65/packages/petdex-cli/src/manifest.ts).
No upstream source code or pet artwork is bundled in this change. Eleven-row
semantics are accepted only from explicit declarations or manual review.

Native source context now carries a dedicated bounded canonical artwork record.
Import, authoring, save, offline snapshot, native export/reimport and independent
character conversion retain it. Runtime graphs still exclude source context.
Actor Pack export refuses substantive native Buddy credits it cannot represent,
with a native-export instruction; unknown/empty credits remain compatible.

## Automated evidence

With the repository virtualenv and PYTHONPATH=packages/tldw_profile_core/src:

```sh
TLDW_BUDDY_COLLECTION=/Users/macbook-dev/Documents/GitHub/tldw-stuff/buddy-packs \
python -m pytest Tests/Petdex Tests/Persona_Visual \
  Tests/UI/test_petdex_import_review.py \
  Tests/UI/test_personas_persona_visual_authoring.py \
  Tests/UI/test_personas_persona_visual_pack.py \
  Tests/UI/test_buddy_character_review.py \
  Tests/Character_Chat/test_buddy_conversion.py \
  Tests/Character_Chat/test_buddy_conversion_lineage.py -q --tb=short
```

Result: **782 passed**, no failures/skips, 189.68 seconds. After adding four further
archive/inventory admission cases, the final **Tests/Petdex** run passed **103 tests**.
The overlapping runs are not summed. The sole warning is the existing Requests
urllib3/charset-normalizer dependency warning.

The integrated run includes 21 Petdex UI tests covering actual native-draft handoff,
cancelled and stale source/destination, final source checks, atomic export failure,
early root/service failure cleanup, preview image cleanup and static codec fallback.
Sentinel pixel tests decode the generated character animation: thinking uses row 8,
six correct colored cells and exactly 1030ms, rather than trusting manifest fields.
Native artwork tests exercise real SQLite publication, reopening, subsequent edits,
source deletion, native export/reimport and independent character publication.

Ruff check and format pass for all 15 new Python files. The eight modified existing
Python files add no Ruff diagnostics compared with HEAD. All generated CSS bundles
reproduce from their sources; git diff --check passes. Independent transport and
source/native reviews found no remaining actionable issues. A slow-body probe
verified the 30-second request deadline and cancellation at read boundaries.

Earlier broad Actor Pack runs have four established unrelated baseline failures,
recorded in [attribution verification](2026-09-07-artwork-attribution-verification.md).
The current integrated command above passes in full; it is a targeted regression
run, not the entire repository suite.

## Live and offline evidence

The public compact registry returned 4681 entries, 830199 decoded bytes, SHA-256
675665e122c3fa92d5cd2d6e22721b23049799b11df322d38eb5ed4af8a8dd76.
An authorized live fetch of the exact `homelander` slug used the real pinned
transport and yielded original 192×208 cells in a classic nine-row atlas.
Source digest: 02c4b7e6bb3320a6b74e58761694f88a73bbc795106d7dda91fc25863270168f.
Native archive: 2178859 bytes, SHA-256
f68165117c54d274ad9a3b438891ce56e2b5d42ba08fb839831963fc43cd573d.
Creator remains Serhat; license remains unspecified; source remains
https://petdex.dev/pets/homelander.

A second offline probe imported and saved that archive into a disposable profile,
deleted its copied source, exported/reimported exact original sheet bytes and
credits, and published an independent local character with 14 animated expressions.
Carried credits were checked on the character pack and every expression asset.
No real user profile or collection content was changed by these probes.

Disposable evidence paths, not repository fixtures:
- /tmp/tldw-petdex-live-probe.py and /tmp/tldw-petdex-offline-probe.py
- /tmp/petdex-live-offline.log
- /private/var/folders/sn/m80n2j152t9gw3w8qwk2nykh0000gn/T/petdex-offline-proof-8pp5n97v/receipt.json
- /tmp/petdex-live-source.png and /tmp/petdex-live-preview.png: mounted 90×35 review,
  source credits, visible mappings/fallbacks, fixed action row and actual pet preview.
  Inspected screenshots are UI evidence, not physical-terminal verification.

The public `boba` entry had registry version 1 and metadata version 2; the importer
correctly refused that contradiction. CDN filenames can differ from pet.json's
logical spritesheetPath: the adapter fetches the exact validated registry URL and
retains the validated logical name, with image-format verification.

## Boundaries

This implements profile-local Chatbook imports. No server-native Petdex endpoint,
server metadata preservation, third-party collection publication or gallery is
claimed. Existing native image and character conversion budgets remain enforced;
oversized sheets can be refused even when their grid geometry is recognizable.
Unknown artwork terms are never inferred from the Petdex source-code license.
