# Canvas V2 Mermaid acceptance closeout

Date: 2026-09-08 (local). Task: TASK-31941.
Governance: [ADR-124](../../../backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md),
extending ADR-121. No new ADR: qualification and the recovery correction implement
the approved contract without adding authority.

## Outcome

The exact `canvas-v2-mermaid-1` profile is locally qualified and admitted. Task
review, whole-branch review, and the single scoped final-fix review are complete.
There are no open Critical or Important findings. This is not a hosted-CI result,
PR approval, rebase, push, or merge; integration remains a separate user choice.

Admission commit: `745811a33538c32746d18ca7987bc0cf7e01affe`.
Recovery fix: `7464fe0c6c1f840417767d6036f4b08e2bca82da`.
Full review base: `1d49796fea3c8cd1c81200ba0ff458c882ebcf74`.

The whole-branch review found I1: strict runtime verification could prevent native
Console, served-parent, and served-child startup, making source-only recovery
unreachable. The fix preserves strict loader rejection and gives application
owners an immutable empty admission snapshot. Matched unavailable parent/child
states retain authenticated source access; HTML mutations and runtime delivery
remain refused, mismatches fail closed, and file repair cannot reactivate an
already-running owner. All immutable runtime assets and catalog policy are
unchanged by this fix. The scoped reviewer marked I1 **ADDRESSED** and found no
new breakage.

## Evidence

| Gate | Result |
| --- | --- |
| Chromium candidate selection | 176 passed; 2 optional engine skips |
| Complete candidate / admitted Canvas selections | 1383 passed each; 2 optional skips each |
| Packaged admission assertion sequence | 2 expected RED failures, then 2 GREEN passes |
| Recovery owner RED / strict-loader controls | 12 expected startup failures; 4 rejection controls passed |
| Post-fix complete selected Canvas gate | 1400 passed; 2 optional skips; 1 existing warning; 712.32s; exit 0 |
| Final production-faithful recovery fixtures | 17 passed; 1 existing warning; 6.07s; exit 0 |
| Static gate | Clean scoped lint/format/whitespace; zero added legacy diagnostics |

The full post-fix run contains 1402 JUnit cases with zero failures/errors. After
that run, self-review corrected only a served-test fixture's picker/lineage shape
and assertions; the final 17-case run verifies those amendments. Product diff
bytes are identical to the full passing run. Intermediate failing test assertions
are retained, not relabeled as successful runs.

Required Chromium containment, actual-child V1/V2 create/update, restart
revocation, confirmed unsent repair, browser isolation, archives, offline rebuilds,
distribution closure, and CI workflow contracts ran. See
[full qualification record](../../Canvas/V2_VERIFICATION.md) for exact scope,
timings, visual inspection, immutable identities, and historical failed gates.
The separately accepted SQLite prerequisite remains covered by its
[own closeout](2026-09-08-sqlite-acceptance-closeout.md); it was not re-audited here.

Retained local evidence:

- Qualification: `/private/tmp/mermaid-qualification.lga1Y7` (160-file manifest).
- Recovery: `/private/tmp/mermaid-recovery.MikmXD` (86-file manifest).
- Recovery manifest SHA-256:
  `ceff67ae6a1fd63f143a812786ac89a0404b3d939df8b6fb9a96ff3b52d9bfb7`.
- Full-gate patch SHA-256:
  `4f3e459dffd41061d421f98b0ac5b9dc9f1b1f9930ad08190463acce983c9322`.
- Final patch SHA-256:
  `88058217db337e1d147ec24cc3c1015e790423d64f2f8eeedef3720391c81b2e`.

The scoped reviewer independently verified all 86 recovery evidence files and
the exact fixture-only difference. Raw review reports remain in the preserved
plan-owned `.superpowers/sdd/2026-09-06-chatbook-canvas-v2-mermaid-implementation/`
directory (`task-8-review.md`, `final-review.md`, `final-fix-rereview.md`).

## Explicit limits and rulings

- Existing M1 dependency/compiler warning debt is non-blocking, disclosed, and
  deferred; no warning was suppressed or dependency changed. Warning noise remains.
- Legacy owner files retain 481 baseline lint diagnostics across the recorded
  eight-file inventory. The scoped gate requires zero new diagnostics, full clean
  checks for small/new files, and formatting of changed logical ranges in legacy
  files. This avoids unrelated cleanup; the cost is that existing debt remains.
- No full-repository sweep, newly executed hosted CI, optional Firefox/WebKit pass,
  cross-platform font parity, or pristine whole-repository lint claim is made.
- No PR, rebase, push, merge, worktree removal, or evidence cleanup was performed.
