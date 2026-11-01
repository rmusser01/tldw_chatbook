# PR3309 backlog integration — 2026-11-01

Source head: `1b3c46389c5db35d38c6a550cfe736a2f0e7449b`.
Tested dev base: `cf5fda70d3fe75e442078b37aa8234a83b7aaf9b`.

The rebase preserves current dev's received-input custody, native commit
retirement, recovery and transaction owners. A failed commit before reservation
pauses only its exact current attempt; another reservation or recorded commit
is untouched. A legacy chat's missing Never/Blocked policy row is repaired in
the same transaction as its accepted turn.

Independent review found a restart defect: the checkpoint retained missing/None
while hydration read the repaired durable/1 policy, so response recovery refused.
The integration permits only that safe first-persistence bookkeeping transition.
Every remaining frozen policy, provider and Library authority field must still
match. No checkpoint or accepted user/assistant row is replaced.

## Fresh local evidence

Receipts are beside this isolated worktree; raw command, XML, logs and before/after
hashes are retained. All runs used private profiles and unchanged source hashes.

- `r-3b670a8f`: two RED restart cases fail at the intended changed-authority refusal
  before the matcher repair, with zero errors/skips.
- `r-5e6be45c`: 128 passed, 12 failed, zero errors/skips across 12 directly affected
  modules. All 11 real hydration/restart cases, 25 durable acceptance cases,
  15 native commit cases, 17 native integration cases, six new mounted commit
  recovery cases and four unreserved-failure controls pass. All 18 added restart,
  authority-boundary and late-SQL-rollback cases pass. Retry uses one provider call
  and the original accepted row identities in both accepted/dispatch_started states.
- All 12 failure IDs reproduce on the exact dev tree in the retained PR3306
  worktree: `r-d135f903` (seven recovery controls), `r-20a666a6` (one contention
  responsiveness control), `r-95663057` (four mounted Library/acknowledgement
  controls). Both base commits resolve to tree
  `fc5eca7da4860e9af4d5c6972916200e8934b127`. Seven cases encounter the pre-existing
  Hooks-unavailable guard before their intended recovery paths. The contention
  test exceeds its 0.5s loop bound on both arms (candidate 0.687s, base 0.922s).
  Mounted failures are the same pending-pause/draft-transfer/early-clear assertions;
  the changed acknowledgement case fails before its new post-commit assertions.
  These are baseline failures, not a claim that the broader suites are green.
- `r-b275c4ca`: 338 passed, three size-limit failures, zero errors/skips. Controller
  growth is eight lines for the narrow matcher (29,285 versus 29,277). The other two
  failures are unchanged dev files: trace service 6,787 > 6,765 and app command
  providers 1,133 > 1,123. The user's explicit size waiver applies. The source PR's
  store/persistence budget adjustments are reconciled with current dev; no
  production guard or behavioral test is weakened.
- Census checker passes at 1,263, retaining all 1,258 base entries in order plus 5 source
  entries. All three new non-UI recovery selectors are in admission_2 and its
  matching contract. Diagnostic inventory rebuild has no drift: 715 owners, 16 sinks.
- Scoped Ruff adds no diagnostics across 16 changed Python files. Three incoming UI
  tests were formatted after the frozen run; before/after ASTs are identical and
  Ruff passes. `pr3309-format-only.json` retains hashes. Remaining formatter/lint
  debt is present on the exact base. `git diff --check` passes.

No full suite or fresh native/real-provider UAT ran. Historical source UAT remains
historical; current mounted tests qualify the integrated recovery behavior.

## Scope and review gate

Existing follow-ups 33621.56–.59 (paused-send toast, other custody exits, background
start failure, Stop-during-failed-commit) and 33621.17 (Temporary Save & Send) remain
separate user-impact limitations. This change does not claim to resolve them.

ADR required: no. Existing `backlog/decisions/079-console-library-conversation-authority.md`
and `033-application-session-state-ownership.md` govern this correction.
Independent review uses `superpowers:requesting-code-review`; the exact-head verdict
is retained beside the worktree as `pr3309-integration-review.md`.

The user's later authorization replaces Qodo and waives hosted CI and size limits.
A fresh non-draft/head/base/owner/auto-merge/thread check precedes publishing and
merging. Drafts and PRs #3267 and #3297 remain excluded. No release or main promotion.
