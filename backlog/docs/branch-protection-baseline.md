# dev branch-protection baseline (2026-09-12)

Deliberate owner decision recorded after the PR #2634 merge flow. Do not
re-tighten these without the owner's say-so.

## 2026-10-03: in-repo merge queue (ADR-218)

Protection is unchanged: strict, required conversation resolution, enforce_admins, and one required check. The queue
works within these settings. It rebases the front armed PR with `GITHUB_TOKEN`, dispatches `derived-artifacts.yml`
(`pr=<n>`), and leaves the merge to auto-merge. Its mode is the repository variable `MERGE_QUEUE` (off, dry or on), and
setting it needs an admin. Agent rules are in `CLAUDE.md` and `AGENTS.md`, under "Merging into `dev`".

## 2026-09-27: strict re-enabled (owner decision)

**"Require branches to be up to date" (strict): true** again. The owner chose
this after stale-base merges (PRs merged on a green run against an older
`dev`) had caused breakage.

Why it was needed:

- PR CI tests the PR merged into `dev` as `dev` stood when the run started.
- On a push to `dev` only the artifact checkers run; both fast lanes are
  `if: github.event_name == 'pull_request'`. So a broken combination was
  rarely caught after merge.
- 4 of the 84 required-check failures on 2026-09-19..26 were inventory drift on
  pushes to `dev` itself.

What changed since the 2026-09-12 relaxation:

- The account-wide runner starvation was traced to tldw_server's
  `LICENSE_FIRST_CI_ENABLED` duplicate CI lane. The variable was deleted on
  2026-09-27.
- The inventory conflict churn is addressed by
  `Docs/superpowers/specs/2026-09-27-ci-conflicts-and-waste-design.md`
  (drop the committed `summary` totals).

The cost is serialization: every merge makes the other ready PRs stale, so
they must re-sync and re-pass before merging. Merge the instant the required
check is green on a head that contains the current `dev`.

The same day, the owner also enabled:

- **Repo `allow_update_branch=true`**, so `gh pr update-branch` re-syncs a behind PR without
  a local rebase.
- **Repo `allow_auto_merge=true`.** An opted-in PR still has to pass every protection above.
- **`dev` "Require conversation resolution before merging": true.** This is the guard that
  makes auto-merge safe: nothing merges, auto or manual, while a Qodo or other review thread is
  unresolved. That enforces the owner's per-PR step "address all Qodo comments" server-side.
  When enabled, 7 open PRs had unresolved Qodo threads (#2026, #2059, #2196, #2427, #2595,
  #2838, #2841).

The rules for when agents may use `--auto` are in `CLAUDE.md`, under "Merging into `dev`".

**`Nightly Deep` disabled (2026-09-27, owner decision).** It was disabled with
`gh workflow disable nightly-deep.yml`, and its state is `disabled_manually`. This changes no file,
so `nightly-deep.yml` and its contract tests are untouched.

- **Why:** 0 of 8 complete runs, while using about 22% of the account's runner-minutes and 64% of
  its macOS minutes.
- **Restoring it:** `gh workflow enable nightly-deep.yml`, once CI throughput sub-project 3 makes
  a run able to finish and report. The ADR-103 amendment for this cadence change shipped in #2860.

## Standing state (as of 2026-09-12; strict superseded above)

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
