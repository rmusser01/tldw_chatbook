# Artwork attribution verification

Historical evidence: the results below describe the original source branch, not
the current dev integration. Combined-code results are recorded in
[the integration verification](2026-09-10-buddy-feature-integration-verification.md).

Date: 2026-09-07
Creator: tldw-project
Task: TASK-32024
Decision: [ADR-074 amendment](../../../backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md)
Plan: [Artwork attribution](../plans/2026-09-07-artwork-attribution.md)

## Delivered

Public creator, source URL, license and full notice text now have a bounded,
versioned carrier in Actor Packs. It is covered by existing checksums, requires an
explicit supported feature, and binds image records to output SHA-256. The importer
validates it before activation and shows complete notices as plain text. Unknown
profile source-context fields are never exported by the carrier.

The visual editor keeps the source manifest license, using `unspecified` when
legacy data has no license. Retained images and profile copies preserve their
credits; replacements drop old image credits. Imported visual packs receive a
private publication identifier, allowing subsequent edits without exporting that
identifier. Same-actor import review hashes the exact stored notice and manifest
JSON independently of the portable actor payload's short-string limit, so long
notices remain eligible and any change invalidates a pending review.

## Executed checks

- New decoder/record tests: canonical JSON, unknown fields/versions, source URL
  restrictions, UTF-8 lengths, per-notice/total/asset-count limits, output hashes,
  unknown expression keys and empty-carrier omission.
- Real SQLite/filesystem round trip: independent archive → review → activation →
  database close/reopen → same-actor re-review → expression edit → export → second
  installation. An 8,800-character notice and original creator survive exactly.
- Real publication tests: a shared builtin graph forks only the target actor,
  retains credits on unchanged images, drops replaced-image credits and preserves
  MIT; absent legacy terms no longer become AGPL.
- Rejection tests: feature/member mismatch, unknown feature version, modified
  archive bytes, wrong output hash, changed reviewed notices, and a notice changed
  between export snapshot reads.
- Mounted import review at 80×24 retains complete literal notice text, scrolls,
  and keeps Create/Cancel reachable. Existing review/workflow cases also ran.
- The final focused command passed **48 tests** across attribution records,
  attribution archives, mounted review, and Actor Pack architecture. This includes
  both original golden archives byte-for-byte with the producer version pinned to
  the version recorded inside each fixture; the fixture files were not regenerated.
- The broader targeted run recorded **242 passed, 1 platform skip, 5 failures**
  across eleven publication/Actor Pack/UI modules. All five failures reproduce on
  unchanged merge-base `d6d792ecdfe8b201e70e25a71e9b77544cbcfa49`, extracted into
  disposable storage. They are three existing Persona export rejections and two
  golden-byte comparisons whose fixtures contain an older producer version.
- Architecture/import-closure selection: **12 passed**, with the separate app
  subprocess test failing because its harness overrides PYTHONPATH and cannot find
  the uninstalled bundled `tldw_profile_core`. That failure also reproduces on the
  unchanged merge-base. The stronger Actor Pack module import-closure test passes.
- New modules/tests pass Ruff and formatting. Changed existing Python files add
  no Ruff diagnostics relative to the pre-change commit; only changed ranges were
  formatted. `git diff --check` passes.

Interpreter: repository `.venv/bin/python`, with
`PYTHONPATH=packages/tldw_profile_core/src`. Existing requests/Pillow dependency
warnings remain. No full-suite pass is claimed.

## Scope

This completes the attribution prerequisite for Buddy conversion. The conversion
snapshot, animation encoding, Create character UI, conversion-specific lineage and
Petdex importer are not implemented by this change. There is no new server import
claim: the native visual manifest remains unchanged, and older Actor Pack readers
reject the required attribution feature rather than discard it. No data schema,
runtime binding, source artwork, user profile or external repository was changed.
