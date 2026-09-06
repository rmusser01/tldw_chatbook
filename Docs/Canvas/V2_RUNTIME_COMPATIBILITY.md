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

The production catalog currently admits only `canvas-v1`, with zero library bytes
and `default_diagram_profile: null`. Task 2 first registers
`canvas-v2-mermaid-1` as non-executable after its real verified candidate manifest
and grammar assets exist; Task 1 does not fabricate a placeholder byte identity.
V2 remains unavailable until all qualification gates are complete.
