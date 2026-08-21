---
id: TASK-19572
title: >-
  CI has not completed successfully since 2026-06-26 and dev has no branch
  protection — ship one install-free required check
status: To Do
assignee: []
created_date: '2026-08-21 20:25'
labels:
  - ci
  - process
  - infrastructure
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 7 (process, tooling & repo health) —
its **F1** and the recommendation from **F2**. The lane called F1 **the root
cause under most of its other findings**, and this filing's verification found
it is **worse than reported**.

**The Tests workflow has not completed successfully since 2026-06-26** — 56
days. Last success anywhere: `2026-06-26T18:00:01Z` (branch
`codex/personas-rail-collapse`); last success on `dev`: `2026-06-26T00:17:48Z`.

Measured now, over the last **500** runs (2026-08-17 → 2026-08-21):
**423 cancelled / 62 failure / 15 in-flight / ZERO success.**

Mechanism — `.github/workflows/test.yml:20-22`:

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.event_name }}-${{ github.ref }}
  cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}
```

**Correction that makes this materially worse than the review stated:** the
required quiet window is **not ~60 minutes**. The last two *successful* runs
took **200.1 and 226.8 minutes**. Recent cancelled runs die at 9–63 minutes.
Measured merge rate to `origin/dev`: **39 / 23 / 12 / 45 / 10 merges** on
Aug 17–21 (peak 62 on Aug 11). A 3.5-hour uninterrupted window on `dev` is
unattainable. `main` is exempt from cancellation but never receives pushes.

**And even a completed red run could not stop anything.**
`gh api repos/rmusser01/tldw_chatbook/branches/dev/protection` → **404 Branch
not protected**. Same 404 for `main`. No required status checks. No git hooks
(`core.hooksPath` → `.git/hooks`, **0 non-`.sample` files**). No
`.pre-commit-config.yaml`.

This is the reason the repo's standing "CI is intentionally cancelled, verify
locally" workaround exists — and the reason TASK-19568's schema pins were
**merged red**. The suite that would have caught them exists, is in the CI job,
and has produced **no verdict in eight weeks**.

**Cost is not the obstacle.** All three derived-artifact checkers are
**stdlib-only** (verified two ways: static import analysis, and a runtime
`sys.modules` audit finding nothing from site-packages) and cheap:

| checker | measured |
|---|---|
| `tldw_chatbook/css/check_bundle_sync.py` | **0.80 s** |
| `scripts/check_profile_owned_path_inventory.py` | **10.81 s** |
| `scripts/check_persistent_diagnostic_inventory.py` | **20.50 s** |
| **total** | **32.1 s** (~90 s with checkout) |

Only `check_bundle_sync.py` runs in CI today
(`.github/workflows/css-bundle-guard.yml:63`). The other two are reachable
**only** through `Tests/Architecture/…` — i.e. through the suite that never
completes.

**The checker is also unactionable when it does fire.**
`scripts/check_persistent_diagnostic_inventory.py:326-354` has exactly one
flag, `--write` — no `--diff`, no `--verbose`. On mismatch it prints one
sentence naming no file, no owner and no call site:
*"production diagnostic owners or persistent-sink topology changed; review the
diff before running --write"*. The only route to detail is to run `--write`
(overwriting the checked artifact) and then `git diff` by hand. Every past
burn-down task hand-rebuilt a ~30-line diff probe. This is not new:
**task-2768 (2026-08-07) is the same incident**, and 15 commits since 08-08
exist whose only change is this JSON.

**The lane's recommendation, with alternatives already rejected on evidence:**
make the checker **teach the fix** (`--diff` on every non-zero exit:
rows-only-committed / only-rebuild / changed, with old→new counts and digests,
plus the next command), carry all three in a **new install-free
`derived-artifacts` job**, and make **that** the one required check on `dev`.

Alternatives the lane rejected, with its reasons — do not relitigate without
new evidence:
- **pre-commit hook** — 22.6 s per commit, bypassable with `-n`, does not
  survive a clone.
- **auto-sync-with-review** — forbidden by the artifact's own design rationale
  (it trains regeneration without reading).
- **classification-only pin** — measurably worse: catches PR #1869 only,
  misses #1877 and #1880, which are the majority of the risk surface.
- **adding it to `test.yml`** — already there; that is precisely why it caught
  nothing.

Per the owner's standing ruling, this is the durable/pragmatic choice: one
small check that actually runs beats a comprehensive suite that never reports.

## Acceptance Criteria

- [ ] A `derived-artifacts` CI job exists that runs the three stdlib-only
      checkers with no dependency installation, completing in roughly 90 s
      including checkout
- [ ] That job is a **required status check** on `dev`, so a red guard blocks a
      merge
- [ ] Branch protection exists on `dev` (and a decision is recorded for `main`)
- [ ] `check_persistent_diagnostic_inventory.py` gains a `--diff` mode that
      runs automatically on every non-zero exit and names what changed —
      rows-only-committed / only-rebuild / changed, with old→new counts and
      digests, and the exact next command
- [ ] The other two checkers are held to the same standard: when they fail they
      name the offending file
- [ ] The Tests workflow's cancel-in-progress behaviour is addressed so that
      *some* full-suite verdict is produced on a regular cadence — e.g. a
      scheduled run on a quiet branch, sharding, or exempting a nightly ref.
      The 200–227 minute runtime is the constraint to design around (see also
      TASK-19425, which covers the runtime growth itself)
- [ ] The first completed full-suite run since 2026-06-26 is recorded, with its
      pass/fail counts, so the repo knows its actual baseline
