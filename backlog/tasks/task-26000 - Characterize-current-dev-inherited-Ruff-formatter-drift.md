---
id: TASK-26000
title: Characterize current-dev inherited Ruff formatter drift
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-30 15:39'
updated_date: '2026-08-30 16:22'
labels:
  - maintenance
  - formatting
  - quality
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md
  - Docs/superpowers/plans/2026-08-30-task-26000-ruff-formatter-debt.md
documentation:
  - Docs/superpowers/plans/2026-08-30-task-26000-ruff-formatter-debt.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-22514 proved that its closeout introduced no Ruff formatter regressions while leaving a historical 61-file residue on its pinned base. Re-census current origin/dev and define conflict-safe atomic cleanup batches so formatter debt stops obscuring feature-owned changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A pinned current `origin/dev` census records every Python file failing the repository-supported Ruff format check.
- [ ] #2 The current census is compared with the 61-file TASK-22514 historical residue and every difference is explained.
- [ ] #3 A mechanically checked batch manifest assigns every current failure exactly once to one atomic cleanup record; every record requires behavior preservation, and one final record requires an explicit repository-wide zero-exit Ruff format check after its lower-ID dependencies.
- [ ] #4 TASK-26000 changes no Python source; `git diff --check` over its recorded task boundary and `Tests/CI/test_backlog_task_id_uniqueness.py` pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Recheck duplicate work, rebase only the TASK-26000 range onto current `origin/dev`, and record the exact task base, current pin, and common ancestor with TASK-22514's closeout branch.
2. Build and self-test a temporary standard-library census tool that records exact revision-local Git paths, blob IDs, Ruff exit-code status, blockers, configuration provenance, and aggregate control results without modifying any checkout.
3. Run whole-repository base/pre-closeout/closeout/common/current censuses, reconstruct TASK-22514's scoped `M/B/C/H` identity sets, resolve revision-path lineage, and prove the projected final-closeout invariant.
4. Generate one canonical point-in-time JSON manifest, mechanically derive every current classification, define owner-aligned stable batches, prove validator negative cases, and append the exact counts and stable labels to both plans before cleanup records exist.
5. Allocate collision-safe Backlog IDs, create every non-final cleanup record before the lower-ID-dependent final record, bind records to batches, and make the positive manifest checker plus task-ID guard pass.
6. Obtain independent subagent approval of the evidence, lineage, batches, and task contracts; verify and correct every finding before re-review.
7. Recheck current `origin/dev`, run documentation/evidence closeout gates, check all TASK-26000 criteria, add implementation notes, and mark only the characterization task Done.

Task 1 authority state (2026-08-30):

- `task_base`: `05c858e87cc1f11c96d6b384b34fdaf914efc51e`
- `current_pin`: `05c858e87cc1f11c96d6b384b34fdaf914efc51e`
- `common_ancestor`: `f0e8961222fe1a7a3ac7566f7f78142e717358f3`

ADR required: no.

ADR path: N/A.

Reason: the task records and schedules behavior-preserving formatter cleanup without changing runtime, storage, security, dependency, or cross-module architecture.

Task 4 manifest record (2026-08-30): pins are task base/current
`872a325483679d2880fcfe2a6e2b9fc82e12f42d`, common
`f0e8961222fe1a7a3ac7566f7f78142e717358f3`, historical base
`31ed49bb368f54211d6482599e00a5c1340f80b2`, pre-closeout
`1f4f72ac5ff02f5237a4946745e82e8932cd41cf`, and closeout
`642b1c782fe6c066a781314dae669a55b05b62ad`. Counts are `M=99`, `B=64`,
`C=77`, `C-B=16`, `B-C=3`, `H=61`, `F_closeout=1,738`, `F_common=1,746`,
current `=1,923`, `historical_still_current=47`,
`historical_no_longer_current=14`, `shared_ancestor_debt=1,617`,
`current_line_drift=259`, and blockers `=0`.

The 38 stable labels are `ruff-agents-runtime`, `ruff-api`,
`ruff-api-client-large`, `ruff-app-shell-large`, `ruff-chachanotes-db-large`,
`ruff-character-persona`, `ruff-chat-core`, `ruff-chunking`,
`ruff-ci-workflow-active`, `ruff-console-composer-active`,
`ruff-console-performance-active`, `ruff-console-trace-ledger-active`,
`ruff-console-ui`, `ruff-core-runtime`, `ruff-database`, `ruff-evals`,
`ruff-generation-media`, `ruff-ingestion-web-media`, `ruff-integration-live`,
`ruff-library`, `ruff-library-screen-large`, `ruff-notes`, `ruff-performance`,
`ruff-personas-screen-large`, `ruff-providers-prompts`, `ruff-rag-research`,
`ruff-root-ci-architecture-final`, `ruff-scheduling-notifications`,
`ruff-settings-screen-large`, `ruff-speech-audio`, `ruff-tests-misc`,
`ruff-tools-workspaces`, `ruff-tts-windows-active`, `ruff-ui-screens`,
`ruff-utils-config`, `ruff-watchlists-screen-large`,
`ruff-watchlists-subscriptions`, and `ruff-widgets`.
<!-- SECTION:PLAN:END -->

### Task 1 Repin Record (2026-08-30)

- Recorded base/current pin `c2f64f690bf4a712b604a1a1db348398df932f36` advanced to `ceac56e06eda4d3d2995a2f5ac8010a7a1821ed2`.
- After stashing only the Task 2 plan/task edits, the clean-index eleven-commit recorded slice was verified to touch only the approved task/spec/plan files; the upstream README/screenshot/TASK-2803 delta had no path or TASK-26000 conflict.
- Rebased only that slice with `git rebase --onto ceac56e06eda4d3d2995a2f5ac8010a7a1821ed2 c2f64f690bf4a712b604a1a1db348398df932f36`; derived common ancestor remains `f0e8961222fe1a7a3ac7566f7f78142e717358f3`.

### Pre-Task 3 Repin Record (2026-08-30)

- Before a real census may begin, refreshed the recorded base/current pin from `ceac56e06eda4d3d2995a2f5ac8010a7a1821ed2` to `3e5e75e4aa884d4f362aa63c1e151c3855f07a36`.
- The clean fourteen-commit TASK-26000 task/spec/plan slice rebased only with `git rebase --onto 3e5e75e4aa884d4f362aa63c1e151c3855f07a36 ceac56e06eda4d3d2995a2f5ac8010a7a1821ed2`; common ancestor remains `f0e8961222fe1a7a3ac7566f7f78142e717358f3`.

### Task 2 Execution Record (2026-08-30)

- Temporary root: `/tmp/task26000.b0z8M0` (created with the required `mktemp -d /tmp/task26000.XXXXXX` pattern).
- Hardened Appendix A SHA-256: `dc665997e31040be0b16701a83b83890fbc555f93116430691f1e6eb1f860cc0` (mechanically rematerialized after the Task 2 atomic-publication, executable-provenance, and direct-executable portability regressions and hardening changes).
- Supplied and canonical absolute invocation executable: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`; the replayable virtual-environment symlink is intentionally not dereferenced.
- Version gates: `Python 3.12.11`; `ruff 0.15.22`.
- Hardened `--self-test`: zero exit, `census self-tests: 20 cases passed` (the original fixture/blocker probes plus exact snapshot exit-2 checks, direct and replayable symlinked-launcher provenance, relative/non-executable and toolchain rejection, abnormal `core.excludesFile`, hostile Git environment, checkout-root, and atomic success/write/file-sync ownership probes).

### Task 3 Execution Record (2026-08-30)

- The initial evidence run pinned `origin/dev` at
  `3e5e75e4aa884d4f362aa63c1e151c3855f07a36`. Before commit, authority advanced
  first to `57ffb893670ebee744da00c85c0c2c87318357d5`, then the final pre-stage fetch
  advanced to `857747d3d4e8d048d7c763a65d2a05d9104fc52e`. Before the spec-review correction,
  authority advanced again to `ae863bfc0e5b33d29a9423e4dcc70664d490cc12`, then
  the executable-provenance correction gate advanced it to
  `747042659706d68861d6e8d88da7a3bbc139f247`, and the direct-executable portability
  correction gate advanced it to `fa0017351ceb375fcb70a0af7cce82dc3d3d4814`,
  and the bounded pin refresh advanced it to
  `4ae04314c49c54d9241aae8275b5d4b8e14b254e`, followed by the current-only
  refresh to `872a325483679d2880fcfe2a6e2b9fc82e12f42d`, followed by the current
  refresh to `05c858e87cc1f11c96d6b384b34fdaf914efc51e`;
  the clean task/spec/plan slice rebased only onto each fresh SHA, current evidence
  and lineage were regenerated each time, and common remained
  `f0e8961222fe1a7a3ac7566f7f78142e717358f3`. Historical pins were base
  `31ed49bb368f54211d6482599e00a5c1340f80b2`, pre-closeout
  `1f4f72ac5ff02f5237a4946745e82e8932cd41cf`, closeout
  `642b1c782fe6c066a781314dae669a55b05b62ad`.
- Isolated evidence lives outside Git under `/tmp/task26000.b0z8M0/`:
  `evidence-repo/`, five clean detached `checkouts/`, five full `raw/*.json`
  snapshots, and canonical `m-identities.json` (SHA-256
  `c0ab0afef8c9726ef3751c28f41d91a0e9c90d66b4040cf0468392463f018f78`).
  Snapshot entries/failures were base `4,648/1,741`, pre-closeout
  `4,653/1,754`, closeout `4,653/1,738`, common `4,643/1,746`, and current
  `4,988/1,903`; all blockers were zero and every aggregate control reconciled.
  Corrected raw SHA-256 values are base `7d2c0b02695fc6a05ebe294f629389348b68403f8433466f2ca6bd4d88f8ae17`,
  pre-closeout `073db424a2bc1ba7d0af7a047120c9d3e996eb1f71934fd8f83e823e68fd77ae`,
  closeout `5d29afd7294cbf7149676287edbf7b1f1c3a13824634d98eea7668579fd74e56`,
  common `c34c5fe9d8e3154c3450f1cf28d4c9a6f1f631feb4735296fc6b891af5de1b15`,
  and current `a4484533728c2caae2a90454bd7f4e09720ad956c1afd4f54036a939fc4c864f`.
  All five were rerun with the final portability-corrected Appendix A. The four
  historical snapshots remained byte-identical because their schema does not embed
  the producer-source digest; current changed with the required authority repin.
  Each snapshot now preserves `.venv/bin/python` identically in toolchain and command
  provenance. Verbatim Appendix B remained SHA-256
  `b16cfb7bdbd94fe0946cad99a4225f8981de87c27df324e78516f5556459a413`;
  its self-test and callable census validation accepted all five snapshots.
  The earlier repin added two tracked Python files, added no failures, and resolved
  `tldw_chatbook/Utils/input_validation.py` relative to its superseded snapshot; the
  portability-correction repin held the count at 4,947 files and added two failures.
- Historical arithmetic passed exactly: `M=99`, `B=64`, `C=77`, `C-B=16`,
  `B-C=3`, `H=61`. Complete lineage categories were `unchanged=2,123`,
  `add=5`, `delete=4`, `rename=0`, `copy=0`, `ambiguous=0`; all 1,746 common
  failures were projected (1,742 unchanged, four interval-proven deletes).
  Target-anchored follow evidence plus exact NUL source/target interval rows require
  commits `38dbb58a21`, `f9a06ff625` (two paths), and `489a57b050` while preserving
  source blob IDs and zero exact-current-blob matches. The derivation now authenticates
  the isolated Git repository, full pins/ancestry and unique closeout/current merge
  base, canonical closed-schema
  snapshots and tree/configuration inventories, approved toolchain/scope, aggregate
  controls, and M identities against the authentic historical diff. It sanitizes Git
  authority inputs, rejects source-descended same-path A/D replacement history, and
  propagates R/C identity state through source-descended commit-parent A/D/R/C events.
  Partial deletion removes and records only one active copy path; identity death occurs
  only when the final active path is deleted, and later resurrection fails closed.
  Merge parents must agree on the identity-bearing active/dead state; historical
  aliases and tombstones are conservatively unioned so a retired alias on either
  parent still blocks later replacement or resurrection. ADRC candidate filtering
  preserves full topology propagation while reducing the 20-unrelated-commit control
  from 21 per-parent proof diffs to one. Exact proof commands, raw digests, and parsed
  rows are persisted and replayed; deletion proofs identify the actual parent and
  full-row index. Full endpoint blob/path maps still make stationary duplicates
  ambiguous, and publication uses Appendix A's owner-safe atomic writer.
  The four real intervals contain neither R/C nor same-path replacement projections.
  The temporary helper/test digests are
  `13f8718bcfc59d96bdd7221a7875fe0806c83566752888003a908cf32b03de67`
  and `2a7f8c519a70ebfc956a50a9c5a3f3db7c62184f3df75c0d3abbfdd8ad89f60a`;
  all 49 controls pass across merge-parent retired-alias union and later
  alias-resurrection rejection, direct/merged surviving copies, three-copy partial
  deletion, all-path resurrection blocking, merge-deletion parent/row pointers,
  candidate-filter call counts, prior direct/merged rename and copy deletion/reuse,
  exact proof replay, merge-base authority, multi-hop/merge-parent R/C chains,
  stationary duplicates, end-to-end D/R/C, authority-mutation, hostile-environment,
  strict-NUL, and atomic-output cases.
  `F_closeout & project(M, closeout) == project(H, closeout)` passed with exactly
  61 projected identities.

## Renumbering provenance

This formatter characterization task renumbered from `TASK-24653` to
`TASK-26000` under TASK-19601. The older holder,
`backlog/tasks/task-24653 - Network-TLS-trust-policy-corp-DPI.md` (Network TLS
trust policy (corp DPI)), keeps `TASK-24653`: it was created on 2026-08-29 22:51,
while this formatter task was created on 2026-08-30 15:39. Per the owner rule, the
younger task renumbers regardless of status. Only citations within the pre-renumber
formatter commit range `1d2cd6bec1..dceb79f19f` and the pre-renumber versions of
this task record, its design, and its plan refer to this formatter task; unrelated
historical `TASK-24653` citations retain their own local meaning.
