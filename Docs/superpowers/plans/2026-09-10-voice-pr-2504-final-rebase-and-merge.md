# PR 2504 current-dev integration and approved merge

**Goal:** Preserve the complete reviewed audio work while rebasing PR #2504 onto
current dev, verify the combined source, resolve new concrete findings and merge
the same PR after checks pass.

**Authority:** The user explicitly approved updating to current dev and merging
after verification on 2026-09-10. This supersedes the earlier merge hold only.
All restrictions on hardware, native/local model execution, full-suite runs,
installs, unrelated files and worktree cleanup remain in force.

**Architecture:** Compose existing voice custody and process isolation with
upstream archive, reasoning replay and lock-safe SQLite contracts. No new owner,
runtime, acoustic policy, release qualification or provider architecture.

ADR required: no new ADR
ADR path: existing ADR-094, ADR-098, ADR-090, ADR-097 semantic trace, ADR-125 and
`backlog/decisions/147-conversation-archive-and-exact-resume.md`
Reason: integrate accepted contracts and advance only the unmerged voice migration
after the already-landed archive migration; preserve upstream SQL bytes.

## Frozen inputs and ownership

- Active worktree: `.worktrees/speculative-duplex-voice-dev`.
- PR branch: `codex/speculative-duplex-voice-dev`.
- Reviewed published head / initial push lease:
  `098f904355cf18ff523e9931fa449406827ab646`.
- Previous base: `2e3389e694e93592a1c66e5c3416bf29a1057d6c`.
- Fetched target: `0fcb79e596b3e527778c56867cf7ee0c62c17894`.
- Root owns publication and merge. One integration worker owns the local rebase,
  conflict composition and targeted verification. Read-only reviewers own bounded
  analysis; at most one code writer operates at once.
- Keep the original feature worktree, main checkout, all backups and scratch.
  No reset, checkout-discard, stash, garbage collection or worktree removal.

## Steps

1. Verify exact clean published head, target and changed-path intersection. Read
   the relevant task/ADRs. Review schema and provider overlaps independently and
   review this plan before implementation. Record any concrete additional seams.
2. Preserve the reviewed head in `codex/voice-pr2504-before-sep10-rebase`. Commit
   this plan and the existing task's approved merge note with explicit paths.
   Rebase normally with `--onto` the pinned target from the previous base.
   Resolve actual conflicts by retaining both accepted behaviors, not by taking
   entire old files. Preserve native files and hard-off authority unchanged.
3. Keep dev's archive schema 71 and its SQL artifact unchanged. Move the unmerged
   voice migration to 71→72, updating its method/map, current schema, test filename,
   live expectations, index census and both source/lint inventories. Preserve
   historical migration fixtures. Exercise real predecessors 70 and 71, including
   archive columns, index, triggers, rows and rollback behavior. Update only the
   incoming archive test's final-current-schema assertion as necessary.
4. Compose provider reasoning replay with bounded voice delivery: preserve both
   request fields, pruning/reconstruction, ContextVar worker setup/reset, exact
   cleanup/custody and winning-only trace promotion. Preserve upstream archive
   activation/lifecycle guards and all earlier stale-claim/idle-startup fixes.
   Add a focused regression before any non-mechanical behavioral correction.
5. Review changed logging statements before using the official diagnostic
   generator. Regenerate CSS with its official builder; do not raise budgets,
   snapshots, weaken scope checks, or add late-loading solely to evade a census.
   Prove non-overlapping incoming paths and native/authority bytes survive.
6. Run only relevant software/real-SQLite/fake-provider tests selected by overlap
   review: migration/archive, reasoning/provider/voice promotion, retained bridge,
   visible-control admission, initial polling and original boot guards. Keep
   incoming Torch/MPS/runtime model tests uncollected. Run source-list, index,
   version, diagnostic and CSS guards plus scoped lint/diff checks.
7. Obtain bounded independent spec/quality review of conflict resolutions and
   subsequent fixes. Record commands, results and limitations; do not sum
   overlapping cohorts or label earlier evidence as final-head execution.
8. Commit explicit integration paths, verify clean source digest, push the same
   PR with an exact observed-head lease, and preserve the fresh generated PR-body
   section. Allow normal automatic PR CI only; no manual dispatch/rerun.
9. Wait for required/applicable checks, refresh all review threads and comments,
   and address concrete failures. Recheck head and base immediately before merge.
   If the target changed materially, inspect and verify that delta first. Merge
   the exact verified head through GitHub without bypassing branch protections,
   auto-merge shortcuts or manual branch deletion. Verify the resulting merge
   commit and report it. Preserve local worktrees and backups.

## Reviewed target advance and archive composition

After the pinned replay, the reviewed twelve-file portrait-fit change advanced
the target to `3afa68f1b99103ec8b5b5f19921ac480ad9344ac`. A second normal rebase
preserved that change before integration verification. Frozen original inputs
above remain historical evidence.

Archive admission follows the existing runtime custody owner, with original
conversation reservations released by actual task cleanup, including cancellation
before first execution. Hands-free lifecycle checks consult retained process
sessions through actual parent-effects retirement. Direct voice preparation
revalidates durable archive state; completed-pair transactions reject a newly
archived target after reconciling already-committed identities. These compose
ADR-094/098/147 without a new ownership or runtime boundary.

## Verification limits

Only targeted software and file-only provenance checks run locally. No full test
suite, live app/audio/device/route, provider request, model inference, native
build/configure/compilation/loading/binding/GIL/callback/installed-wheel check,
qualification harness, repeated conversation or soak. Hosted native builds are
only the previously approved automatic PR CI. Packaged qualification and build
identity bytes remain hard off; merging into dev does not qualify a release.
