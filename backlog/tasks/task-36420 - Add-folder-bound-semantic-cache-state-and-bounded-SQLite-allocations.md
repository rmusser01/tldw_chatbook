---
id: TASK-36420
title: Add folder-bound semantic cache state and bounded SQLite allocations
status: Done
assignee:
  - '@codex'
created_date: '2026-10-23 03:26'
updated_date: '2026-10-23 15:44'
labels: []
dependencies:
  - TASK-36006
references:
  - backlog/decisions/241-workspace-project-context-and-folder-bound-index.md
  - Docs/superpowers/plans/2026-10-20-workspace-project-semantic-index.md
type: feature
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare optional semantic storage without changing existing project access, keyword behavior or saved Library choices.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh v4 and supported upgrade chains preserve keyword rows, policies, frontier, claims and authority; injected DDL failures leave the prior valid catalog intact.
- [x] #2 Semantic settings and consent have an independent revision; their changes preserve keyword epoch, readiness and Library policies/scopes, and status opens no embedding/model objects.
- [x] #3 Semantic-capable allocations and all competing new-version mutations use serialized binding-wide admission with a source-qualified pre-write main/journal/temp/control regular-file-length bound within1GiB; unsupported geometry, unretired ownership or observed mixed writers refuse before semantic mutation.
- [x] #4 Body, commit, rollback or native-close failure retains uncertain ownership and blocks reuse; read/status/search paths perform no hidden unreserved writes.
- [x] #5 Legacy cleanup v2/v3 and newv4 remain available without migration/allocation; existing v1 quarantine is preserved, and old compatible Ready storage is never discarded to fund staging.
- [x] #6 Pure keyword-only operation retains established behavior without embedding imports or semantic-specific runtime qualification; keyword search remains usable during real scanner waits outside finite SQL units, and missing/degraded semantics preserves saved Library choices.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read Task36420, amended semantic plan/ADR241, current project-index architecture, quota-foundation-proposal source-pinned refinement and actual baseline/store/lifetime tests. Existing ADR applies; no duplicate decision.
2. Establish meaningful RED tests using real SQLite/native binding ownership for additive v4/fresh and old migration preservation, independent settings/revision, model-free status and unchanged keyword-only availability; then bounded allocation, all-control-write, pre-open journal, two-namespace contention and commit/rollback/close uncertainty cases.
3. Implement narrow project allocation.py using stable allocation-v1.lock through BindingLifetime for finite try-only SQL units outside registry fences (retained job lifetime may precede it), source-qualified main/journal/temp/control allowance on actual held connections and positive native settlement. Admit every semantic-capable competing allocation; preserve legacy keyword-only mode and record unsupported/mixed-version boundary honestly. No generic kernel change or embedding/provider/model construction.
4. Add004/fresh-v4 exact catalog and semantic immutable settings/generation/member records under same owner, read-only status, independent semantic revision CAS and preserved prior keyword state. Extend exact v2/v3/v4 cleanup without migration/allocation, retain v1quarantine and oldReady headroom.
5. Run targeted new/affected keyword DB/service/recovery/cleanup/native nodes only, scoped lint/format/source guards and diff. Retain raw/child receipts as each process retires outside rotating temp; no dependency/model install, benchmark corpus rebuild, fullsuite or external operations. Include analytical+actual cap evidence and unresolved limits; commit exact task paths and report for independent spec/quality review.
ADR required: yes, existing amended decision.
ADR path: backlog/decisions/241-workspace-project-context-and-folder-bound-index.md.
Reason: direct implementation of accepted folder-bound semantic storage/native allocation contract; amendment written before code. No new Library owner, dependency or generic runtime boundary.
6. Final pre-code consistency correction: actual project SQLite timeout/busy_timeout0; no allocation gate/store/cursor across scanner IPC, provider wait, Capture, joins or retry sleep. Positively close each native store and inventory before releasing its gate; every gate-held lifetime/receipt acquisition is try-only and unwinds on busy. Verify keyword lookup while the real scanner is waiting. This supersedes the draft allocation-first whole-job order; source-backed proposal found finite units feasible.

Review round1: implement binding-wide semantic-required preflight before target native store for write-capable units, existing-only status lifetime observation, and versioned provider/model catalog-key refs plus SHA-256 v1 digests per amended ADR241/semantic plan. Disabled settings without generations remain pure keyword; unexpected read-to-write nesting refuses. Actual catalog resolution stays in the embedding-scope deliverable. Keep one native store at a time, all targeted native/keyword safety gates, and saved Library choices.

R1 implementation refinement (recorded before owner-policy edit): the actual first GREEN failed because db.workspace_project_index permits only PRIVATE_FILE and refuses READ_ONLY_URI. Authorize only this existing owner's allowed kinds to become _PRIVATE_AND_READ_ONLY, using the existing checked mode=ro, must_exist=True path for serial peer inspection. Keep its production_module, private-path validation, backup/sync exclusion, foreign-source flag, recovery flags, memory exclusion and every other owner unchanged. This introduces no source grant or generic-kernel behavior. Record the new WorkspaceProjectIndexDB._probe_peer call as a separate read_only_uri C99 row in backlog/docs/sqlite-private-owner-inventory.md and deliberately extend only the exact connection census range to include C99 in Tests/DB/test_private_sqlite_inventory.py, retaining its guards. Add actual project-owner read-only/missing-target/write-refusal controls and run matching inventory/registry nodes; no inventory regeneration or unrelated baseline absorption.

Missing allocation gate bootstrap: before first gate creation, a regular write-capable constructor must inspect every existing namespace serially through existing-only read-only handles, with validated inventory and positive native close. An existing v4 target OR peer refuses without creating the missing gate, including repeated/restarted attempts. Only an empty or fully validated legacy v2/v3 group may proceed to a separate gate-held write-capable unit, which rechecks current peers before opening the target. Read/status never creates a gate. Ambiguous aliases, unknown catalogs, recovery sidecars or close uncertainty refuse; uncertain ownership is retained. The preliminary scan grants no write authority and cannot substitute for the gate-held recheck. Costs if wrong: safe refusal or additional finite peer-open work; no repair-by-retry of missing v4 control.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented additive semantic cache v4 and immutable settings/state under the existing project owner, independent semantic revision/consent, and finite binding-wide SQLite allocation/native settlement. Keyword data, saved Library choices and optional-dependency behavior remain intact. All competing namespaces use semantic-qualified admission when the group retains enabled consent or semantic generation data. Status is read-only; existing-only legacy bootstrap and uncertain close retain fail-closed behavior. Persisted refs use versioned catalog keys and SHA256-v1 digests; actual resolver/embeddings come next. Existing ADR241 governs storage/native authority; the amended plan records the source-conditioned quota and cooperative-version limits. Supported incremental v2/v3/v4 cleanup remains nonmigrating; v1 quarantine stays.

Source commits42a75186 and439f905f. Independent Task1 review found3Important, all fixed and independently re-reviewed with no new Critical/Important findings. Targeted real SQLite/native/schema/settings/keyword/scanner-wait/cleanup controls and scoped static checks cover the implementation. The unrelated raw-connection census site in unchanged app_service_wiring.py and six unchanged private_sqlite.py Ruff findings are documented baseline conditions, not absorbed or claimed green. The sampled12MiB capacity result is separate from the conditional analytical1GiB envelope. No full suite, embedding/model/provider execution or external integration. Relevant architecture docs/ADR/plan and testing-evidence lesson updated. Exact evidence/disposition: task-1-report.md, task-1-fix-round-1-review-report.md and controller-task1-cannot-verify-disposition.md under this plan's private SDD; retained raw children and source hashes remain. Actual provider-wait/semantic jobs belong to the separate generation/search task.
<!-- SECTION:NOTES:END -->
