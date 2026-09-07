# Canvas runtime profile compatibility

Canvas revisions store a short immutable `runtime_profile` identity. The identity
selects an exact packaged runtime contract; it is never interpreted as a version
range and never resolves through a “latest” fallback. Unknown, retired, revoked,
missing, or integrity-failed profiles remain readable source but are not executable.

## Packaged authority

`profile-catalog.json` is strict JSON capped at the runtime-manifest byte ceiling.
Its top-level schema is closed:

| Field | Contract |
| --- | --- |
| `schema_version` | Integer `1` |
| `build_id` | Lower-case SHA-256 of the canonical sorted manifest/library inventory projection |
| `policy_id` | Lower-case SHA-256 of the canonical default/execution/refusal policy projection |
| `default_diagram_profile` | Admitted exact profile ID, or `null` |
| `profiles` | Non-empty list of unique closed profile entries |

Each profile entry contains exactly `profile_id`, `manifest`,
`manifest_sha256`, `executable`, `reason`, and `library`. `manifest` is an exact
safe packaged JSON filename, not a path or archive member; each profile may name
its own manifest. `library` contains exactly a
verified aggregate `bytes` value and a closed `files` mapping whose values contain
exactly `bytes` and lower-case `sha256`. Duplicate JSON keys, duplicate profile
IDs, unsafe IDs, unknown fields, missing files, byte mismatches, and inconsistent
aggregate sizes fail the complete snapshot closed.

The catalog entry is only a bounded, source-free admission projection. For every
profile, the runtime-asset loader owns one frozen object containing the immutable
verified manifest bytes, parsed manifest, engine, worker, renderer, and profile
library bytes. The process-owned `ProfileSnapshot` retains those objects in a
private tuple and exposes exact-profile lookup through the pure
`runtime_assets_for()` accessor. This keeps bytes attached to the verified policy
without a mutable global registry or a filesystem reread. Archives carry revision
source and profile IDs only; they are never accepted by the profile loader and
cannot supply manifests or executable assets.

## Runtime manifest contract

Every referenced manifest binds its `runtime_profile` and contains a closed
`profile_contract` with these required sections:

- `engine`, `facade`, `plan`, `grammar`, and `layout`: exact non-empty IDs and
  lower-case SHA-256 identities.
- `unicode`: exact data version, segmentation rule, width rule, and SHA-256
  identity.
- `quotas`: an exact quota-contract ID and positive integer values for HTML and
  script bytes, guest heap and stack, startup/event interruption, pending jobs,
  DOM nodes, CSS rules, and patches per operation.

Changing any pinned semantic input requires a new profile ID. A security-revoked
profile remains source-only rather than silently receiving a patched engine.

## Selection and lifetime

Creation selects `canvas-v1` unless diagrams require the admitted diagram default.
An update of V1 may select that exact admitted default; updates of any other parent
retain the parent profile even when all diagrams are removed. Rename and load keep
the exact parent identity. If a required diagram default is unavailable, selection
returns the bounded `profile-unavailable` refusal and does not fall back to V1.

`load_profile_snapshot()` verifies the complete packaged inputs once for the
process owner. Callers retain that frozen snapshot and pass it to the pure
`resolve_profile()` function. Updating packaged catalog, policy, manifest, or asset
files cannot mutate an owned snapshot; applying an update requires restarting the
native process or the served parent and all children. A browser refresh is not a
policy reload. `runtime_snapshot_id()` is the canonical, source-free cross-process
identity for parent/child consistency and cache binding; it is not a browser
credential.

The production catalog admits only `canvas-v1`. Candidate
`canvas-v2-mermaid-1` remains disabled with `default_diagram_profile: null`:
the final release gate is blocked by owned-child native SQLite crashes, recorded
in [V2 verification](V2_VERIFICATION.md). V2 candidate manifest
SHA-256 is `17717bcab7c7bba4a28e0069354f6ecbf895d2ca58f4b8d1c0355b7726e2f466`.
It shares the exact pinned QuickJS engine with V1 and owns separate V2 facade,
plan, grammar and layout identities. V1 still has zero library bytes and remains
the creation default when diagrams are absent. Released semantic bytes cannot
change under this profile ID; changed semantics require a new qualified identity.

## Mermaid build and private interface

`scripts/vendor_canvas_mermaid.py` uses Python 3.12.11 and the standard library.
`Canvas/mermaid/inputs.json` declares every download, exact byte/hash inventory,
the full Mermaid archive member allowlist, two exact source-map entries and their
source hashes, Unicode 16.0.0 data and UAX29 revision45 rules. No npm install,
lifecycle hook, upstream renderer, native guest evaluation or code generator runs.
The authenticated Mermaid 11.17.2 archive is scanned under a 1,200-member / 96 MiB
uncompressed scan bound; only its two maps, package metadata and MIT license are
read under the existing 32 MiB selected-input cap. Its exact generated Jison
0.4.18 ESM export suffix is validated and replaced with a private return, then
deterministic property tables and authored modules are concatenated.

Use `--input-dir` for a previously verified offline input directory; omitting it
downloads only the declared HTTPS inputs with redirects refused. `--output-dir`
supports independent builds. Unicode tables merge adjacent explicit ranges and
implement extended grapheme rules GB3–GB999, including GB9c and GB11. The fixture
retains all 1,093 official Unicode 16.0.0 grapheme conformance rows. Width is a
profile-owned conservative logical cell rule (16 CSS px per cell), not a platform measurement.
Original label code points are retained; Unicode normalization and `Intl` are not
used. Escaped Mermaid entity spellings in plain labels remain literal text.

The manifest's Mermaid provenance records source/input hashes, not qualification
status. Catalog execution/default/refusal policy is separate from immutable
manifest identity. Regeneration preserves an existing admitted or revoked policy
only for the exact unchanged manifest and complete library/notices inventory;
changed bytes produce a disabled candidate and no diagram default. No build
argument or environment variable grants admission. Final recorded qualification
scope and measurements are in [V2 verification](V2_VERIFICATION.md).

`mermaid-subset.json` has exactly `schema_version`, `profile_id`, `source`,
`source_bytes`, `source_sha256`, `inputs_sha256`, and `inventory` (the seven authored
module digests). The catalog's library inventory binds this JSON and the separate
`MERMAID_THIRD_PARTY_LICENSES.txt`. `ProfileRecord.library_bytes` counts all those
packaged bytes. The actual evaluated-script charge is **`source_bytes`**, verified
against the UTF-8 `source`, and shares the 256 KiB document script ceiling with
authored scripts. JSON escaping and notice bytes are not evaluated JavaScript.

Evaluating the verified source returns private handles `parseMermaid`,
`DiagramBudget`, `DiagramError`, `segmentGraphemes`, `graphemeWidth`,
`layoutDiagram`, and `renderDiagrams`; none is
installed on the guest global. A worker retains one `new DiagramBudget()` for a
document. `parseMermaid(source, budget)` calls `beginDiagram(source)` and returns
the closed flow/sequence model. `beginDiagram` increments the ordinal, charges
the raw UTF-8 input before comment processing, and resets only per-diagram counts.
`charge(kind, amount=1)` charges both scopes for `input`, `labels`, `nodes`,
`edges`, `participants`, `messages`, or `notes`; `label(text)` also checks the
individual 512-byte ceiling. Failed startup discards the entire VM/budget.
Layout extends that same owner with `work`, `elements`, `output`, and `area`.

The V2 quota manifest adds exactly `document_declarations`, `label_bytes`,
`diagram_width`, `diagram_height`, and diagram/document pairs for `input_bytes`,
`nodes`, `edges`, `participants`, `messages`, `notes`, `label_bytes`, `svg_elements`,
`output_bytes`, `work_units`, and `area`. V1's quota schema stays unchanged.
All accepted ceilings are recorded and the layout ceilings are enforced by the
profile layout. In the test-only `candidate_snapshot` fixture, execution policy is
replaced while all real verified manifest and asset hashes remain attached.

## Candidate deterministic scenes (Task 3)

`layoutDiagram(model, budget)` returns `{width, height, root, metrics}`. Scene
nodes contain only `tag`, attribute pairs, plain `text`, and `children`; they
contain no IDs, executable source, links, or SVG reference attributes. An
ordinary HTML description and exact source accompany an intrinsic-size SVG
inside an overflow wrapper. `renderDiagrams([{source}, ...])` creates one budget,
parses and lays out each declaration in order, and returns scenes only when all
declarations succeed. The private worker startup applies these scenes through
the existing virtual allocator and typed mutations before authored scripts.

Flow diagrams use Kahn ranks with source-order ties, two forward/backward
barycentric ordering sweeps, and orthogonal rank lanes. Edge labels reserve
lane space in an outer strip beyond all node columns; labeled and rank-skipping
edges route through trunks beyond that strip. A charged segment/label-rectangle
check refuses any intersection before returning the scene. Sequence headers
keep declared order and messages/notes share their original event order. Side
notes reserve horizontal margins, two-participant notes span their columns, and
every note reserves a separate vertical interval. Arrows use ordinary paths,
including geometric arrowheads and dashed sequence messages; no marker is used.

Wrapping preserves the pinned extended grapheme clusters and original code
points. The candidate changes the logical cell to 16 CSS px for conservative
extents under the approved 16px monospace font, normal weight/style, and 24px
line spacing. Logical widths determine geometry, with no native measurements,
`Intl`, or fetched fonts. Missing glyphs can use local fallback; source retains
their original spelling. Explicit authored overrides can change typography or
dimensions and do not recompute layout; that appearance is outside the default
fidelity contract.

SVG elements and shallow serialized records charge the shared budget during
construction, with the final scene envelope included in exact UTF-8 output
accounting. `metrics` records the per-diagram input, labels, semantic counts,
work units, SVG elements, output bytes, and logical area. Coordinates, positive
dimensions, arrowhead extents, and final viewBox bounds are checked before any
worker patches exist. Budget failures never truncate labels or return partial
scene arrays. Additional refusal codes are `work-limit`, `elements-limit`,
`output-limit`, `area-limit`, and `geometry-limit`.

The exact QuickJS scene/model corpus is `Tests/Canvas/fixtures/mermaid/layout.json`.
Scene-only Chromium qualification in `test_mermaid_scene_readability.py` uses the
existing V1 renderer and isolated loopback harness. It checks actual default-font
extents, inherited typography, narrow scrolling, source identity and zero generated
egress; it is not evidence of V2 startup integration or cross-platform pixel
identity. These historical scene-only checks are not sufficient for admission;
the complete Task 8 product qualification is recorded separately.

## Candidate compilation and startup (Task 4)

`compile_canvas_document` defaults to unchanged V1. Explicit V2 compilation
requires the exact retained verified snapshot. `prepare_canvas_document`
performs one bounded HTML parse, inspects structural declarations, resolves the
profile, and compiles inside the caller's existing compilation admission owner.
The source-preserving frozen V2 plan contains the six V1 wire fields plus
`diagrams` and `profile_manifest_sha256`. Diagnostics are never wire fields.
Each diagram is exactly `{ordinal, target_node_id, kind, source}`; the source is
decoded text from a nonempty, text-only Mermaid `pre`. The records must correspond
one-to-one, in document order, with their virtual pre nodes.

V2 pins `canvas_runtime_worker_v2.js` and `canvas_renderer_v2.js`. The V1 worker,
renderer and manifest bytes remain frozen. Both profile layouts are fixed by the
trusted loader; a browser plan cannot select asset paths. The Mermaid vendor
build copies the V2 worker/renderer into independent output directories and pins
their hashes. Separate files deliberately duplicate the established sandbox;
future runtime maintenance must qualify each affected immutable closure.

The private V2 renderer init envelope is exactly
`{type: "canvas:init", nonce, plan, runtime_data}`. The worker prepare envelope
is exactly `{type: "prepare", plan, runtime_data}`.
`runtime_data` is exactly `{manifest, library, source}`, all UTF-8 strings:
the exact captured manifest JSON, exact captured `mermaid-subset.json`, and
the exact revision source. The trusted parent acquires these bytes before the
execution acknowledgement. Renderer and worker independently verify the closed
plan, manifest SHA-256, library JSON size/digest, evaluated source size/digest,
exact revision source identity, declaration target/text correspondence, and
combined library/authored byte ceiling. No library module is imported or
evaluated in the native realm; no CSP or resource privilege changes.
Task 6 supplies the product parent delivery; Task 4's owned harness exercises
this exact consumer seam with recorded parent fetch acknowledgements.

Startup begins the existing 250 ms operation before evaluating the library in
QuickJS. One private budget prepares every scene, then applies them with virtual
IDs and typed mutations (including per-property style setters). Authored scripts
and bounded jobs run under that same deadline and transaction. Only successful
startup publishes a transaction; diagram or authored failures publish no diagram
mutations. Handles are disposed before authored code and never become globals.
Later source/attribute mutation does not rerender. These browser tests qualify
this consumer integration only; release admission additionally requires the
complete product qualification recorded in V2 verification.

Worker admission also independently checks every generic plan collection and
node record before creating the VM: namespace-specific tag/attribute vocabulary,
element versus text-node slots, uniqueness, asset references, bounded base64 and
raster metadata, aggregate bytes/pixels, and CSS text/nested-rule quotas. Actual
CSSOM paint semantics and browser image decode remain renderer responsibilities;
the worker does not gain DOM/CSSOM construction. Installation removes the original
declaration text children through typed mutations before appending its scene, so
the virtual text getter sees the same descendants as the rendered DOM.

## Exact revision ownership (Task 5)

The Console controller and its durable service share a captured profile snapshot.
If both dependencies are supplied explicitly, they must reference the same snapshot
object; construction rejects distinct snapshots, even when their values compare equal.
Creation and replacement preparation use the profile resolver in the compiler's
single bounded parse. An internal preparation value carries that selected profile,
source identity, parent profile and snapshot through the existing owner gate.
An offered legacy prepared plan is compared with an independent preparation;
its own fields cannot choose the profile. Normal native imports pass the internal
value and do not parse again during mutation. Owner, parent, selection, cancellation
and temporary-incarnation checks still run after compilation.

V1 children upgrade when diagram declarations require the admitted default. A V2 child
retains its exact profile after diagram removal, while an explicit historical V1
branch retains V1 semantics. Native stored reads compile the exact stored profile.
Unknown, revoked and uninstalled sibling profiles remain source-only; title-only
rename preserves source/profile without compiling them, and HTML updates refuse.
Repository and tool projections validate bounded profile IDs as data, independently
of execution admission. Promotion and turn contributions preserve every profile
inside their existing transactions, including rollback/retry and origin remapping.

Canvas-bearing conversation exports use ChatbookCreator/ChatbookImporter and
archive format 3.0, for one or multiple selected conversations. Round trips preserve
source, profiles, branches, renames and deleted origins without installing runtime
bytes or changing the snapshot. Legacy plain-text/JSON conversation exports are
not Canvas graph archives. Schema 68, archive format 3.0 and sync exclusion are
unchanged. Qualification additionally exercises fixed test-only candidate and
revoked policies in separately owned replacement processes; these wrappers are
not production enabling mechanisms. Source-only recovery followed by explicit
new-Canvas creation preserves old exact source/profile/history without substitution.
