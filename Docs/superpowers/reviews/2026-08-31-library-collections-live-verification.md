# Library Collections Capture Reader Live Verification

**Task:** TASK-18919

**Date:** 2026-09-01

**Branch:** `codex/task-18919-collections-reader`

**ADR:** `backlog/decisions/107-collections-capture-authority-and-legacy-boundary.md`

## Outcome

- **Local authority: PASS.** The production-shaped mounted walkthrough used a disposable test
  profile/data root and a real on-disk Collections database seeded with 45 captures. Its resolved
  database-path fingerprint matched before and after the walkthrough; no private path is retained
  in this record.
- **Server authority: PASS.** An isolated loopback tldw_server profile advertised exact
  `hasReadingSnapshotPagesV1: true` through the versioned docs-info endpoint. The mounted reader
  then completed the 45-plus-capture source-replacement, paging, geometry, focus, mode, workspace,
  save, archive/Undo, and controlled-unknown-save walkthrough without bypassing capabilities.
- **Integration status: PASS.** The walkthrough found and corrected a SQLite schema-memo defect in
  the tldw_server prerequisite. PR #2851 merged at
  `8140c679f3ea0334cea2dc1be32feb5b80e22ebe`, completing the cross-repository gate.

## Local walkthrough evidence

The live test `Tests/Live/test_library_collections_capture_walkthrough.py` exercises the actual
Library route, production stylesheet, capture repository/service, and mounted Textual widgets.
Condition-based waits are used for asynchronous state changes.

| Terminal | Pinned pure geometry (Library / Items / Work) | Mounted result |
| --- | --- | --- |
| 160×50 | 30 / 40 / 80 | PASS: all panes, both grips, Work focus via F6, no descendant overflow |
| 120×35 | 0 / 56 / 54 | PASS: Items reclaims Library width, Work remains mounted, no overflow |
| 100×30 | 0 / 42 / 48 | PASS: compact Items and Work, keyboard focus and containment preserved |
| 80×24 | 0 / 0 / 70 | PASS: Work-only posture, both grips remain accounted for, no overflow |

The run also verified:

- untouched Library startup does not pre-mount the capture reader;
- route activation, every Library/Items collapse combination, wide resize restoration, and F6 Work
  traversal;
- exact pages of 20, 20, and 5 rows from a coherent total of 45;
- closing Library expands Items so longer titles receive the reclaimed width;
- Quick Capture commits the URL, tags, and freeform note before extraction completes;
- a controlled extraction failure persists as `failed`, same-item re-selection refreshes it, and an
  explicit Retry reaches `ready` with authoritative recovered text;
- Read, Highlights, Notes, and Info modes;
- archive and Undo, offline-copy cleanup through confirmed hard delete, and complete recovery export
  of all 45 legacy Collections rows plus all 45 memberships;
- simulated unknown Server-save UI behavior: the draft remains visible, no automatic retry occurs,
  the user is told to refresh first, and only the explicit warning confirmation issues one retry.

## Server walkthrough evidence

The enabled-Server live test used a disposable loopback profile and test principal. Local contained
exactly 3 captures while Server contained 45 captures before the mounted save and 46 afterward;
switching authority replaced the dataset rather than merging either identity space.

The run verified:

- exact docs-info attestation and capability reasons, including fail-closed unavailable actions;
- coherent pages of 20, 20, and 5 rows before the mounted save;
- 160×50, 120×35, 100×30, and 80×24 geometry, reclaimed Items width, Work expansion, descendant
  containment, and F6 Work focus;
- Read, Highlights, Notes, and Info modes;
- a Local workspace switch that left the active Server authority and dataset unchanged;
- one confirmed Quick Capture through the mounted UI, followed by archive and Undo;
- a controlled unknown save outcome that retained its complete draft, issued no automatic retry,
  required explicit warning confirmation, and allowed Back and Cancel without another request; and
- switching back to Local restored only the original 3 Local captures.

## Findings corrected during the walkthrough

1. The Items toolbar could overflow a compact pane even while the shell's child widths summed
   correctly. Sort now owns a separate toolbar row, and the live assertion checks every visible
   descendant against its pane.
2. The Work action toolbar could similarly push Open Original outside the pane. Primary and
   secondary actions now use separate rows.
3. Re-selecting a capture whose background extraction changed did not refresh the loaded detail.
   Same-item selection and Retry now reload authoritative detail.
4. A late summary/audio/offline result could paint onto a newly selected capture. Content,
   highlight, capture-note, and linked-Note completions now verify identity after awaits.
5. An indeterminate Server save could lose its draft or invite an accidental retry. The reader now
   retains the draft, exposes Refresh first, and requires a second explicit confirmation with the
   current Server default-reapplication warning.
6. The real versioned docs-info response does not contain the invented `api_version` field used by
   the original capability gate. The gate now trusts the versioned endpoint boundary and still
   requires exact snapshot attestation for browse.
7. The Server's Collections schema memo indexed SQLite mapping rows positionally, so verification
   failed and replayed the complete schema bootstrap for every database adapter. It now reads the
   named `name` field, with a regression proving the second adapter skips bootstrap; the correction
   landed in tldw_server PR #2851.
8. Background reader recomposition could erase text already entered into Quick Capture. Input and
   note changes now update the retained draft immediately, with a mounted regression covering all
   fields.
9. Final verification found the action-persistence test could observe highlight state before the
   reader's scheduled recomposition remounted its mode controls. The test now waits for the Notes
   control to return before pressing it, matching the user-visible boundary.

## Automated evidence

- Focused controller, mounted reader, and Local live walkthrough: **29 passed**.
- Complete capture feature and Local live gate: **206 passed**.
- Production-shaped cross-reader closeout after the final save-recovery hardening: **490 passed**.
- Final Local/Server service, configuration, mounted-reader, and enabled live gate: **61 passed**.
- Post-prerequisite-merge focused client gate after the remount-wait hardening: **58 passed**.
- Post-review import-closure and lazy-wiring regressions: **17 passed**.
- Post-review focused controller, mounted-reader, and Local live gate: **44 passed, 1 enabled-Server
  walkthrough skipped because no isolated endpoint was configured**.
- Complete GitHub UI-latency and boot-budget workflow after deferring the capture runtime past first
  paint: **23 passed**.
- tldw_server snapshot, docs-info, and schema-memo regression gate: **51 passed**.
- Static validation covers the edited Python modules, CSS regeneration, bytecode compilation, and
  whitespace/error checks.

## PR performance-gate correction

The first PR run measured 976 project modules resident at `_ui_ready` against ADR-097's pinned
maximum of 972. The six newly resident modules were the capture models, repository, service, legacy
recovery, offline store, and Server adapter. The implementation now preserves the same package and
app-level service API through lazy exports and a first-use scope proxy, and schedules normal capture
runtime composition just after first paint. The ratchet itself was not raised. A subprocess import
closure test prevents the implementation graph from returning to the plain app or headless
controller import path, and the exact failing census plus the complete performance workflow now pass.

## Final PR review closeout

The final review pass moved the lease migration to a packaged SQL artifact, reused the database's
write-transaction context, bounded Local highlight and linked-Note reads, and connected Quick
Capture to the guarded article extractor without widening first-paint imports. Extraction workers
are now fenced by processing state and lease owner, so a concurrent favorite/status edit cannot
strand completed content. Archive Undo carries its capture identity and remains visible after the
archived row leaves the current scope.

Server update/archive actions now require exact `hasReadingOptimisticUpdatesV1=true` and send the
expected revision in the mutation request; without that attestation they are visibly unavailable.
Server hard-delete remains unavailable because the current API has no atomic delete precondition.
Highlight deletion verifies the selected capture before mutation, and saved-search mutations map
transport/response failures to bounded content-free reasons. The post-review evidence is 107 focused
regressions passed, 19 import/Local-live checks passed with one isolated-Server skip, and all 23
performance checks passed.

## Integration closure

tldw_server PR #2851 merged at `8140c679f3ea0334cea2dc1be32feb5b80e22ebe` after its required
checks passed and all review threads were resolved. The implementation and required Local/Server
walkthrough evidence are complete, and TASK-18919 is Done.
