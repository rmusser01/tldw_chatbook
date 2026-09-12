# dev branch-protection baseline (2026-09-12)

Deliberate owner decision recorded after the PR #2634 merge flow. Do not
re-tighten these without the owner's say-so.

## Standing state

- **Required approving reviews: 0.** Set to 0 by the owner on 2026-09-12
  after a same-day change to 1 blocked every PR authored from the owner's
  own account (GitHub refuses self-approval, and no reviewing bot has
  write access — Qodo posts findings but cannot approve). If reviews are
  re-required, owner-authored bot-driven PRs like the backlog small-batch
  series become unmergeable without manual help.
- **"Require branches to be up to date" (strict): false.** Relaxed the same
  day by the owner's instruction after four consecutive merge windows were
  lost to a structural race: dev merges PRs roughly every ten minutes, a
  full CI run takes about thirteen, and the frequently-regenerated
  `Docs/security/production-diagnostic-inventory.json` conflicts on nearly
  every dev merge. The required check ("Derived artifacts reproduce from
  their sources") still runs on every PR head; only the must-contain-
  latest-dev condition is gone.
- **Only required status check:** "Derived artifacts reproduce from their
  sources". **enforce_admins:** on. Force pushes and deletions: off.

## Verified the same day

- No workflow, script, or tool in this repository calls the branch
  protection or rulesets APIs (searched `.github/` and `scripts/`), so
  nothing automated can re-tighten these settings — only a manual
  repository-settings action can.
