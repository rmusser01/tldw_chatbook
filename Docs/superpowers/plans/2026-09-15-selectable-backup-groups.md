# Selectable backup groups implementation

Task: TASK-32628. Spec: ../specs/2026-09-15-selectable-backup-groups-design.md.
Baseline: 58e9c632021e474d58a7fe5ee2f9043c0860f38f. PR #2642 targets dev.
Latest dev integration: 657f70ffe7 at 17d8b5fc75, following the earlier
48d40df8ce, a9cb557674 and 4e4558bff2 merges on 2026-09-15.

ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: selectable data ownership and replacement scope amend the existing
backup architecture; the TASK-32628 amendment records the approved migration.

## Stage 1: Installed group catalog and selection closure
**Goal:** Define user-facing groups using installed owner contracts.
**Success Criteria:** Everything default; strict named selections; whole-owner,
shared-store, and dependency closure; config support distinguished from Settings.
**Tests:** Pure behavioral tests for complete owner coverage, duplicate/empty/
unknown selections, dependency cycles, shared stores, and support configuration.
**Status:** Complete

Add `Backup_Recovery/data_groups.py` with frozen `BackupGroup` (group_id, label,
description), `ResolvedGroups` (requested_groups, effective_groups,
required_groups, member_ids, support_ids), `group_for_owner(owner_id)`, and
`resolve_inventory_groups(items, group_ids)`. None means Everything; an explicit
empty tuple refuses. Policy is installed and deterministic. Non-user transient
owners have no selectable group. A config dependency becomes support unless
Settings was explicitly selected. Tests use real StorageItems without app boot.

## Stage 2: Capture and versioned archive scope
**Goal:** Capture only reviewed groups and necessary support in one archive.
**Success Criteria:** Preview/execution digest includes selection; excluded groups
are explicit; coherent subset is not mislabeled whole-profile; v1 still readable.
**Tests:** Selective capture payload/exclusion/closure assertions; changed selection
refusal; malformed scope rejection; full/v1 reader regression; encryption policy.
**Status:** Complete

Extend DiscoverySelections, capture option validation, inventory selection,
capture preview and manifest generation. Preserve discovery defects and native
source protection. Introduce strict v2 scope metadata in archive_models and
archive_reader without weakening producer-inventory or dependency validation.

## Stage 3: Selective restore and retained local configuration
**Goal:** Restore selected groups while preserving the remaining target data.
**Success Criteria:** Reviewed roots and dependency closure bind the plan; target
reverse dependencies are included or refused; Settings-unselected config remains
byte/identity unchanged; safety copy, Finish/Abort/later rollback honor this scope.
**Tests:** Real target/incoming configurations with deliberately different paths
and settings; local dependency validation; changed bytes/identity refuses; native
replacement, restart Finish/Abort, and later rollback; Settings-only reachability.
**Status:** Complete

Integrate group selection in destinations and restore_plan. Add a typed local
config association (separate from later-rollback LocalSnapshotSource), include it
in plan digest/persistence with explicit old-plan compatibility, stage private
support copies with verified provenance, and propagate selectors into native
replacement/publication. Preserve existing security and activation proof. Adapt
projection retirement and owner dependency review so unselected groups cannot be
silently changed. No arbitrary retained-owner grafting.

## Stage 4: Service, UI, and command-line workflow
**Goal:** Expose the same selection and review contract through existing surfaces.
**Success Criteria:** Everything default, named checkboxes, required-group review,
clear completeness and preservation messages; changing any selection invalidates
review; destination forms reflect selected groups; keyboard-first operation.
**Tests:** Service and mounted Textual flows for novice defaults and power-user
selection, invalidation, dependencies, safety-copy distinction, and restore.
**Status:** Complete

Use the installed catalog for labels. Reuse canonical F9 Backup and Restore and
existing form widgets/design tokens. Update relevant docs and CLI arguments only
where an existing backup/recovery entry point exists.

## Stage 5: Native verification, UAT, and PR update
**Goal:** Verify the migrated workflows on macOS, Linux, and Windows.
**Success Criteria:** Native selective/full/legacy tests and first-time/power-user
UAT pass, preservation has byte/identity readback, identified findings fixed,
touched-scope Bandit and review pass, task and PR report exact evidence.
**Tests:** One local app/test boot at a time; disposable real profiles; Linux raw
data stays on host; authorized Windows Actions; no broad unrelated suite sweep.
**Status:** Complete

Preserve prior evidence and the requester's Change summary. Update PR #2642
against dev when verified; do not merge. Commit working increments with task and
plan references and normal hooks. Record verification and review below.

2026-09-16 qualification follow-up: Windows diagnostic `35068720686` passes
123/124 product cases and all 59 native checks. All 27 fixture portability
failures are resolved. The remaining installed credential journey correctly
refuses a review invalidated by ordinary startup Agent Runs storage creation.
The fixture now completes that real startup retention pass before source review
and requires the owner to appear in coverage; native guards and timers remain.
Independent review is clean; the installed macOS credential journey and 11
native capture checks pass (12 total, 85.64s). Linux's earlier fixture repeat
passes 124/124 in 299.60s. The final installed Linux credential journey passes in
92.95s. Final Windows run `35070813386` at `39c4c286d8` passes all 124 product
cases and 59 native checks with zero failures/errors/skips. All 500 distinct
product cases have passing evidence across the original full run and corrective
rerun; production and packaging sources are unchanged between them. Artifact,
source and installed-package audits pass. See the final verification report at
`Docs/Development/backup-selectable-groups-verification-2026-09-15.md` and its
JSON receipt. PR #2642 remains open against dev; no merge is included.

## Progress and decisions

- 2026-09-15: The user selected replacement of chosen groups with preservation of
  unselected groups. Architecture migration and this restore rule are authorized.
- Independent design review identified six retained-config proof boundaries;
  these are incorporated in Stage 3 and the spec.
- TASK-32628 is the authoritative tracking record. The duplicate auto-allocated
  task was demoted and archived using official Backlog commands before edits.
- Stage 1: 29 catalog tests pass; independent review and scalability re-review
  found no remaining issues. Scoped Ruff and Bandit pass.
- Stage 2: 16 selective capture/discovery tests pass, including actual native
  prompt capture and preserved live config/media bytes and inodes. Scope v2 is
  emitted for explicit selections; the default full capture keeps v1 compatibility.
- Stage 3: retained-config foundation passes 20 new native tests and 103 existing
  focused regressions, including fresh-process Finish/Abort and later rollback.
  Independent review found no remaining foundation issues. Group integration and
  Settings-only reachability remain in progress.
- Stage 2 final check: capture scope has no new Ruff findings against HEAD and
  Bandit reports zero findings across all eight production files. Formatting
  changes are confined to the added scope code. Native selected capture remains
  covered by the 16-case passing batch.
- Stage 3/4: 25 preserved-path tests, 13 CLI tests, and 22 restore-group/error
  tests pass. Actual fresh-profile selective replacement exposed a first-use
  binding-order defect; the existing native enrollment flow is being reused.
  Returning a newly introduced group to absence needs a retirement-only later
  rollback; 19 focused journal/lifecycle checks pass, with full integration and
  finalization absence checks still in progress.
- Stage 5 preparation: the installed F9 workflow now has selected-group plain
  and encrypted cases, and the existing Windows runner has a finite
  selectable-groups selection. These additions have not yet been executed.
- Native Linux capture checkpoint: committed f591cc455e passes 56/56 catalog,
  archive-scope, selected-capture and native-discovery cases, with no failures,
  skips or unfinished cases; all 16,779 source files remain unchanged. See
  Docs/Development/backup-selectable-groups-linux-capture-20260915.json.
- Stage 3 retirement-only lifecycle now passes its real native workflow:
  introduce previously absent data, return to absence through later rollback,
  and restore the later edits from the new safety copy. Eight finalization
  drift/idempotence cases also pass. Per-volume private storage includes
  retirement targets without creating fake publication rows.
- Stage 2 reopened after installed F9 selected Conversations UAT failed
  `capture_sqlite_source_changed`. Offline capture evidence remains valid;
  maintained live-app discovery needs investigation before completion.
- The first-use binding proposal remains unapplied following automatic approval
  review rejection. Independent safety review identified the need to repeat
  absent-database/companion checks under the acquired native maintenance lease.
  The corrected proposal is being reviewed before resubmission.
- Follow-up: selected SQLite rediscovery now reuses its verified captured image.
  All 17 native discovery/WAL cases pass; the installed F9 selected Conversations
  plaintext backup, isolated restore, and fresh Open workflow passes in 64.21s.
  Source mutation checks remain intact and Bandit reports zero findings.
- The corrected first-use proposal was accepted on resubmission after independent
  review and the additional held absence check. All 24 focused first-use and
  adversarial cases pass in 5.72s. Public-service replacement/later rollback is
  being rerun; no final cross-platform migration-complete claim is made.
- Stage 3/4: 12 restore-group and 8 mounted UI tests pass. Exact writer per-owner
  dependency records follow the producer graph direction; unknown or modified
  cohort records retain atomic closure. Shared configuration does not implicitly
  select unrelated dependents. Native whole-service and cross-platform UAT remain.
- Following the first-use change, established retained-config namespaces again
  resolve while the operation owns its pending fence. Strict first-use review
  remains unchanged. All 128 restore-plan, finalization, mapped-publication and
  retained-config regressions pass in 239.92s, including fresh-process recovery.
- Public selected Prompts capture and replacement now preserve unselected
  bytes/identities. Later rollback still needs grouped footprint eligibility for
  never-created unselected databases. Review additionally requires refusal when
  a new active dependent group appears, and historical Settings path validation
  before live publication. Focused regressions and service UAT are in progress.
- Installed-package F9 UAT passes encrypted selected Conversations, default full
  plaintext, and encrypted full with credentials: 3 cases in 222.04s. Together
  with the earlier selected plaintext case, all four exercised backup variants
  restore and open through actual controls. Cross-platform migration runs remain
  in Stage 5.
- Actual post-replacement rollback discovery was blocked by Evals assuming every
  restore publishes Settings. The narrow fix must recognize the authenticated
  retained config relation and preserve prior inactive Evaluation paths. Current
  config bytes must remain editable after commit; transaction-boundary drift
  checks continue unchanged.
- Latest-dev merge checks pass 177 cases. The one failing Notes production-builder
  assertion reproduces unchanged on pre-merge 31a9cc490e; no unrelated correction
  is included. Both Console timer regressions pass, Notes producer lifetime and
  incoming sync logic are retained, and Bandit matches all five baseline findings.
- Integrated selectable tests pass 97/99: all 26 native private-copy cases,
  plaintext/encrypted Prompts replacement and rollback, and return-to-absence then
  undo, plus dependency and scope regressions. Both Settings cases pass staging
  but still fail during replacement; diagnosis is continuing.
- Settings finalization now uses the authenticated staged preservation observation
  while activation remains pending, with no startup bypass. All 58 preservation
  checks pass, including new WAL/SHM/journal, content, inode and parent drift.
  All 43 service/finalization cases pass, including plain/encrypted Settings and
  Prompts replacement, later rollback, and fresh-process Settings Finish.
- Fresh later rollback now reviews the current retained config instead of locking
  ordinary preferences to historical bytes. All 29 targeted cases pass, including
  real config-writer preference changes, locator-change refusal, and post-review
  config drift. Independent review found no remaining issue in this fix.
- Selected empty-owner declarations now contribute exact reviewed retirement
  targets; ambiguous cross-location slots and conflicting incoming publications
  refuse. Pure mapping/message checks pass 19 cases. Native empty-history
  replacement exposed a protected-root recheck that does not recognize its own
  completed retirement; the existing journal-backed recovery proof is being reused.
- Refreshed dev again and integrated both newer Notes commits at a93e4cd296.
  The only merge conflict was generated diagnostic aggregate totals; both deltas
  are preserved. Final generator, native platform checks and PR update remain.
- Final diagnostic inventory rebuild matches the merged source exactly. Scoped
  parsing covers 18 production files, Ruff introduces no new findings against
  HEAD, and Bandit reports zero findings/errors. These are checkpoints before the
  final redundant-root lifecycle correction, not its completion evidence.
- Empty-history retirement requires terminal admission to recognize an absent
  redundant file alias under its already owned profile directory. Independent
  design review selected non-mutating effective-root validation: preserve raw
  registry/profile roots and all historical tokens, require native parent and
  per-profile coverage, and retain strict actual source reads. This deliberately
  also accepts external absence of the same redundant child; uncovered or foreign
  roots still refuse. It avoids a new registry/activation transaction. Automatic
  approval review rejected applying this boundary change; only an unconnected
  helper and direct native regressions have been implemented. The full proposal
  passed independent static safety review at v3 with no remaining actionable
  findings. Seven direct native helper tests pass. The user explicitly approved
  the reviewed caller integration; it is applied and native lifecycle validation
  is running. Platform validation remains pending. Earlier applied-scope checks parse 19 production files,
  introduce no Ruff findings against HEAD, and report zero Bandit findings.
- Final selected-owner lifecycle: 40 admission/absence/capture cases pass; 31
  strict missing-root/refusal regressions pass. Four actual history lifecycle
  cases pass with undo and fresh-process Finish at three interruption points,
  preserving the raw registry and profile mappings.
- Empty-only saved groups remain available in review. All 40 new cases pass,
  including three real Writing capture, empty replacement, and later-rollback
  workflows. Metadata-less SQLite primary mapping is limited to installed schema
  and canonical current-config locators, replayed after fingerprinting; raw files
  and tree roots keep their metadata requirement. Independent review is clean.
- Settings/RAG selection fixes pass 15 cases and independent review; nine legacy
  projection publication cases also pass. Actual non-config dependencies and
  concrete unselected retirements still refuse.
- Installed F9 four-variant UAT plus selection controls and CLI pass 27 cases in
  229.43s. This covers selected plaintext/encrypted and full plaintext/encrypted
  with credentials through isolated restore and installed Open/readback.
- Refreshed dev advanced to 657f70ffe7. Its merge preserves backup and Workflows
  shutdown, appends Workflows as SQLite census C92 without renumbering existing
  recovery records, and rebuilds the diagnostic inventory. Targeted integration
  checks and final macOS/Linux/Windows migration qualification remain Stage 5.
- Final combined macOS selection passes all 500 cases in 827.51s with no failures,
  errors or skips. This includes four installed F9 workflows, selected replacement,
  preservation, empty-group retirement, interruption recovery, later rollback,
  CLI/UI controls and the latest-dev integration regressions. All 54 changed
  Python files parse; Ruff and Bandit introduce no new findings against the merged
  baseline. Derived-artifact checks pass. Final Linux and Windows runs follow
  against the committed implementation before Stage 5 is complete.
- Native Linux passes the same 500 cases in 901.84s, with unchanged source and
  matching installed-package evidence. Windows run 35062911011 passes 59 native
  and 472 product cases, with 22 failures and six setup errors. Reviewed test
  fixture corrections preserve native ownership and reparse-point refusals;
  nine changed Python test/runner files pass static checks and all 160 local
  diagnostic/CI cases pass in 279.87s. A fixed Windows diagnostic selection and
  existing delegating observers will verify the corrections and expose the
  remaining capture-scope transition. No production checks are relaxed.
  [Verification record](../../Development/backup-selectable-groups-verification-2026-09-15.md).
