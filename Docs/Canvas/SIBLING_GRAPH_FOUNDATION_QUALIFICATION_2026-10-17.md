# Local Canvas sibling graph foundation qualification

TASK-35376 implements ADR-239's local compatibility foundation only. Schema 82
admits complete tied graphs; Canvas exports use Chatbook 4.0 / extension 2.0.
Valid legacy 3.0 / 1.0 remains readable; no-Canvas exports remain 2.0. Sync,
server transport, runtime profiles and execution capabilities are unchanged.

Qualification used native macOS Python 3.12.11 and SQLite 3.49.1 from the existing
project environment, synthetic files/private profiles, no dependency downloads
and the existing loopback-only test guard. Base: 2b37746902a8fb8a9059fdec107d1107b0b7953c.

## Evidence

- RED: seven new controls failed on unchanged production (tied imports,
  unknown 4.0, missing 82 upgrade/rollback), then all seven passed. An additional
  oversized-sequence RED exposed unbounded range allocation; the bounded
  contiguous-distinct check now rejects it without allocating from supplied max.
- Genuine schema-81 files preserve exact rows, roll back DDL/version on injected
  failure, retry, match fresh-82 catalogs, deny a second root, refuse incompatible
  predecessor indexes and then accept tied imports. Ten migration tests passed.
- Both repository import transaction paths retain unordered sibling IDs/source;
  ordinary and authored turn batches allocate sequence 3 after [1,2,2], with stale
  claims refused. Service UUID tie-break/branch projection remains deterministic.
- Real creator/importer ZIP round trips preserve source/profile/origin metadata,
  history and tombstones; fixed manifest timestamps yield deterministic bytes.
  Re-exported tied graphs restore idempotently. Valid legacy archives import;
  legacy ties and version mismatches refuse before extraction/database access.
  The actual pre-change reader, executed from base Git bytes, refuses new 4.0.
- Recovery validates current core/hybrid files, rejects corrupt/custom catalogs,
  rewrites valid files, retains restricted-connection denials, and upgrades all
  four frozen v81 core lineages. Genuine shared-file 81 validates and native
  opening upgrades it to 82 while preserving source. Eleven recovery tests passed.
- Broad affected run: 264 passed, one inherited migration-source-census failure.
  Post-format focused qualification: 71 passed; subsequent strengthened migration
  and archive-rejection controls: 10 and 5 passed. No full-suite claim.
- Seven package checks passed: real sdist and direct/rebuilt wheels contain the
  exact new migration and installed fresh/v35 bootstrap reaches current schema.
  Packaging/check_manifest.py validated the actual source distribution and wheel.
- Derived preflight and whitespace checks passed. Ruff: 762 inherited diagnostics
  in both base/candidate, no new per-file diagnostic multiplicities. Twelve
  baseline-clean files pass formatting; four retain existing formatting debt.

## Limits and attribution

Initial baseline: 43 failed / 183 passed. Existing admitted-profile test markers
resolve 34 archive and four provider setup failures; the sandbox required scoped
loopback permission for two existing controls. Corrected unchanged-base run:
223 passed / three inherited hybrid catalog failures. Exact installed 81/82
hybrid catalogs were added under an explicit scope ruling, retaining older ones.

The unrelated migration census fails identically on base/candidate for seven
Dreams, Guardian and collections files. It remains unchanged. A default-sandbox
Hypothesis run warned that its normal example database was unwritable and used
memory; no assertion or quota was weakened.

Raw local evidence and built artifacts: /private/tmp/chatbook-canvas-sync-foundation.KPZ9cL.
The full implementation handoff lives in the plan's local SDD task-1-report.md.
No host-profile migration, full test sweep, external network, dependency install,
server qualification, renderer change, push or merge was performed.

## Review correction I1

The independent review found that the inherited V1/V2 parser ignored a present
Canvas member, allowing import to drop Canvas. The parsing boundary now rejects
any such member before extraction/database access, including null/malformed
members and normalized V1 aliases 1.0.0/1.1.0. Only 3.0/1.0 and 4.0/2.0 pairings
are admitted. Ordinary no-Canvas V1/V2/alias conversation imports remain valid.

Fix-round RED: 20 failed / six passed against 0cba3e2831 (eight real ZIP pairings
reached the database-access tripwire, twelve key-presence model cases silently
parsed). Four ordinary no-Canvas conversation imports passed before the fix.
GREEN: 103 model/archive tests passed without warnings using an owned Hypothesis
store; seven fresh package/installed checks passed. Both direct and sdist-built
installed parsers passed sixteen refusal and four no-Canvas controls each.
The final model bytes match the fresh sdist and both wheels; package manifest
validation, scoped Ruff, formatter and whitespace checks passed. New evidence is
under the original evidence root's fix1-* paths; earlier receipts are retained.

## Independent closeout

The task review and scoped I1 re-review are complete. A fresh whole-branch review
of `272cad956855..346687690cde` found no Critical or Important issue. Its sole
remaining Minor finding was a stale approval-status paragraph in the earlier
maintenance closeout; commit `ee778b9e29ce` labels that historical snapshot and
records the current approved contract. A scoped re-review found M2 addressed,
with no new breakage or remaining findings.

The controller independently reran 30 final-parser/version-pairing/legacy-import
controls: all passed without warnings. Source/package identity was checked for
the fixed model across the working tree, sdist and both direct/rebuilt wheels
(SHA-256 `27f2d46087a7a8726d58f0ececdf52d3d791dcaf3953085ec6d1897b8b8f0160`).
All 5,504 task records pass identity/path/readability checks. Renderer assets and
their pinned input manifest are unchanged from the branch base.

TASK-35376 is the completed local foundation; TASK-35377–35383 remain separate
server/client transport and conformance work. These local checks do not establish
server compatibility, current-head hosted checks or permission to migrate the
normal user profile. Schema/archive allocations must be revalidated before
integration; the branch and its evidence remain local and unmerged.
