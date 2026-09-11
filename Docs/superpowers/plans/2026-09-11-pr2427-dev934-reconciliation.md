# PR 2427: reconcile dev 934b28f39a

Continuation of TASK-31932 steps 1–5 and the completed Library runtime-repair
plan. The user requested continuation of the latest-dev rebase, fixes, review,
and normal protected merge. This is integration of accepted behavior, not a new
feature or owner-extraction proposal.

ADR required: no new ADR.
ADR path: existing `094-console-turn-lifetime-and-navigation-boundary.md`,
`098-low-latency-speculative-duplex-voice-pipeline.md`, ADR-086 and ADR-141.
Reason: retain incoming app-owned turn custody and the PR's existing UI owners,
test-resource boundaries, and Library lifecycle contracts.

## Frozen identities and scope

- Published and recoverable PR checkpoint:
  `ab59e4684ab25d2f5771effebd5ffa325ad00f4b`.
- Previous qualified base: `423ae37c1b5394cb596d0d975c1579c9e6f790d3`.
- Fetched target: `934b28f39a5c386ca95de49a67abdf16a54251a8`.
- Replay all 272 PR commits; do not squash, drop reviewed repairs, or replay
  unrelated dev ancestry.
- Incoming delta: 290 commits, 1,051 changed paths. The exact branch overlap is
  106 paths: 36 production, 68 tests, one lessons document and one diagnostic
  inventory. The read-only merge preview reports 33 conflicted paths.
- Work only in `.worktrees/pr2427-review-recovery`. Preserve the unrelated
  untracked `2026-09-08-pr2427-library-reader-paydown.md` without opening it.
- Root alone stages and advances Git. Read-only investigations may run in
  parallel; only one implementation owner edits an overlapping runtime group.

## Reconciliation checklist

- [x] Fetch and verify exact heads, ancestry, published checkpoint and path census.
- [x] Inspect the non-mutating merge-tree conflict preview and independently map
  Library, Console submission and test-fixture overlap.
- [x] Complete the ordinary rebase, resolving each conflict rather than choosing
  a whole side. Review automatic merges where a method changed owners.
- [x] Preserve incoming synchronous runtime custody, non-destructive captured
  drafts, token/session admission, recoverable refused turns and view detach.
  Adapt those methods into the existing submission controller with explicit
  callbacks; remove superseded screen-worker/stash-watchdog paths. Keep current
  event hooks in the Screen. Intermediate compatibility wrappers may remain
  only until their later reviewed removal commits replay.
- [x] Reconcile the moved prompt-history accessor with the incoming runtime
  history owner, preserving one shared composer/accepted-turn history and its
  existing factory seam. An AST census of all 33 incoming changed Screen
  methods identified this additional moved-body overlap.
- [x] Preserve settings-return owner delegation and exact claim lifecycle,
  terminal-receipt metadata plus legacy exchange persistence, user-denied
  presentation plus exact persisted redaction, and deterministic worker waits.
- [x] Keep Handoff and bounded import-diff implementations in their existing
  PR helpers, applying incoming copy and wikilink normalization there. Keep
  Media's requested/applied state in its current state owner while retaining
  the failed-Clear retry behavior. Retarget new tests to that same owner.
- [x] Retain published size ceilings. Any actual excess remains a failure;
  do not adopt incoming higher pins merely to make checks pass.
- [ ] Check review-created task IDs against the target. Initial exact-path
  census finds only TASK-31249's intentional title rename, not a new identity
  collision. Repeat all-ref/current-tree checks before publication.
- [ ] Review final incoming source changes and changed-owner call sites, then
  run complete targeted affected files with frozen sources. Use native FD
  observation for real-app/database cohorts. No repository-wide sweep, hardware
  qualification, paid inference, model download, forced GC or global cleanup.
- [ ] Regenerate derived artifacts only from reviewed source changes and run
  all existing preflight checks. Keep known resource, size/preload/CSS failures
  separate from any new failures; no merge-ready claim until all gates pass.
- [ ] Obtain independent spec and quality review, record exact terminal evidence,
  publish with a fresh exact lease, refresh all Qodo threads and final-head CI,
  and merge only through ordinary branch protection when genuinely eligible.

## Evidence status

The published checkpoint's 1,074 distinct targeted cases and separate 44-case
regression run remain evidence for that checkpoint only, not this rebased tree.
The following sections distinguish intermediate evidence from qualification
after the complete dev934 replay. No partial replay is a publishable checkpoint.

### Intermediate integration evidence

- Step 55: preserved incoming runtime submission custody in the existing
  submission owner. Complete controller-wiring file: 44 passed in 21.80s.
  Direct token/session checks and normalized moved-body checks also passed.
  Corrected early runtime creation and a stale settings-suspend receiver.
- Step 144: reconciled private delegate retirement with app-owned history.
  Initial targeted run: 45 passed, one obsolete history-path patch failed.
  The follow-up exposed a real stale conversation receiver and four obsolete
  architecture inventory rows plus an obsolete acceptance-hook fixture:
  82 passed, seven failed. Evidence:
  `/private/tmp/pr2427-step144-verification.dQ6BP9/pytest.log`.
  The receiver now resolves the session controller; regression checks require
  late-bound session ownership and absence of superseded view-owned APIs.
  Final complete four-file run: **134 passed**, three warnings, 41.66s;
  `/private/tmp/pr2427-step144-verification.Gw5CPZ/pytest.log`.
  These runs used isolated temporary directories,
  not a native FD observer; they do not qualify resource cleanup.

### Completed replay and final-source qualification

All 272 commits replayed successfully onto the frozen target. Local head is
`b5f8d0d7c96c4b39c5daad75ee3dd4d7436f9616`; Git confirms target ancestry and
272 replayed commits. The remote remains the recoverable `ab59e4684` checkpoint.
The only post-rebase production edit removes the newly introduced, unused
`complete_durable_units` import. Two other unused imports also exist in the
published baseline and remain untouched. All 840 changed Python paths parse;
fatal scoped Ruff checks pass.

Independent final Library review confirms the mapped runtime integrations.
Current Media/import/sync counts remain 4733/4646, 651/587, and 2024/2023.
Initial preflight passes five checks, including 3,929 unique Backlog task IDs;
the diagnostic manifest needs regeneration and Mermaid lacked its declared
input. The existing verified offline input cache reproduces all six assets.
Diagnostic statement review found fixed copy, booleans and exception categories
only: Screen removes one obsolete restoring-draft warning and adds three current
attach/keep-live-draft diagnostics; session removes one policy-fallback warning.
Rebuilt manifest: 607 owners, 12 sinks; all seven checks subsequently passed.

Initial frozen native-observed complete-file cohort report locations:

- Console: 291 cases, `/private/tmp/pr2427-dev934-console.Bo7hu8/pytest.log`.
- Library: 236 cases, `/private/tmp/pr2427-dev934-library.YGPOYW/pytest.log`.
- Backend/boot contracts: 416 cases,
  `/private/tmp/pr2427-dev934-contracts.PpHUJh/pytest.log`.

Each report directory contains the unchanged observer's `fd_identity.jsonl`.
This is an initial 943-case integration cohort, not all remaining changed files
or a full repository sweep. Initial Console/backend failures are under
investigation after terminal completion. Publication follows qualified repairs;
normal protected merge remains blocked by the separately recorded gates.

### Terminal findings and bounded repair plan

All three initial cohorts are terminal: Console 281 passed / 10 failed,
Library 234 passed / 2 failed, backend/boot 410 passed / 6 failed. The Library
failures are stale full Handoff-copy expectations. Console failures are missing
snapshot/captured skill prerequisites and retired owner/inventory expectations.
Five backend failures are fixture signature/resolution drift; the sixth is the
unchanged 1,000-turn workload reaching its 300-second timeout after 988 turns.
An unchanged isolated rerun is required before changing that fixture. All seven
derived preflight checks passed in `/private/tmp/pr2427-dev934-preflight-final.log`.

Native final attribution separates behavior from resource evidence: Library
retains no SQLite/instance-lock handles; Console retains 392 SQLite and 30
instance-lock handles; backend retains 14 SQLite handles. The unchanged size
guards separately report 86 passed / 15 failed. No cap will be raised.

1. Correct only the demonstrated fixture contracts, retaining exact assertions;
   obtain separate spec and quality review and rerun complete affected files.
2. For Console resources, reuse `close_owned_console_resources` and
   `close_owned_console_test_apps` in send-draft snapshot, composer undo and
   composer history tests. Attribute all 392 SQLite / 30 locks to these modules
   (351/27, 27/2, 14/1 respectively). Route snapshot's send-app helper through
   its local builder with the same real database attachment; route undo's
   ready-host helper through the existing injectable builder seam, as already
   done in dictation-streaming tests. Do not change shared fixture behavior,
   production code, test assertions, deadlines or application ownership.
3. Native-verify those complete files plus existing cleanup fault controls;
   review the exact lifecycle patch before reporting any cleanup success.
4. Backend attribution identifies ten remaining worker-thread handles in two
   test-owned CharactersRAGDB instances, plus four handles from workspace-context
   lookup opening a cached global Workspace DB. In the agent-bridge test module,
   import the existing Console resource fixture. The attempted run-log-only
   repair did not remove those four handles; discard that fixture and its four
   controls. Supply a real test-owned registry through the existing explicit
   `_registry_factory` seam, without modifying the cached global. Reuse
   `OwnedAppExecutor` for that exact DB's worker handles, register its drain before
   auxiliary database close, and reuse the same registry in the existing named
   workspace-note test. Preserve real logging and all workspace/prompt assertions.
   Verify startup-rider, manual trace and named workspace controls, then the
   complete bridge file natively. No production or shared-fixture changes.

ADR required: no. ADR path: N/A. Reason: test-only reuse of established exact
resource owners and fixture contracts, not new runtime or cleanup boundaries.

### Follow-up verification

- Backend signature/provenance controls: six passed in 3.04s, report
  `/private/tmp/pr2427-dev934-fixture-controls.mNqwJY`. The report still retains
  three default-workspace handles, so passing assertions did not prove cleanup.
- Console fixture controls: ten passed in 9.13s after correcting synchronous
  skill catalog capture; `/private/tmp/pr2427-test-reconciliation.e5JQYp/pytest.log`.
  This run did not use native FD observation.
- The unchanged 1,000-turn test passed alone in 287.52s under the unchanged
  300-second timeout, with zero final SQLite/instance-lock handles:
  `/private/tmp/pr2427-dev934-tombstone-isolated.jeddw2`. No workload, diagnostic
  owner, retention assertion or production code was modified. Timing margin
  remains small; bounded diagnostic probes were not full-test qualification.
- Complete Handoff owner file: 13 passed in 19.89s, zero final SQLite/locks:
  `/private/tmp/pr2427-dev934-library-verified.7SCiFm`.
- Initial complete three-file backend repair run: 376 passed in 51.36s,
  `/private/tmp/pr2427-dev934-backend-verified.wb40pv`. The ten owned
  CharactersRAGDB handles are gone, but four default-workspace handles remain.
  A constructor-stack probe identifies `workspace_context_note` in first-request
  planning, before AgentService/logging construction, as the actual allocation
  path. Reassess the temporary log-root fixture rather than adding global cleanup.
- Static checking also found a dead provider-selection fake callback referring
  to a removed Screen helper. Removing that unused assignment preserves the
  actual controller-owned test path and clears F821. Independent review approved
  this deletion; all affected-file fatal Ruff checks pass.
- All seven final derived checks pass again in
  `/private/tmp/pr2427-dev934-closeout-preflight.log`.
- The complete nine-file Console/cleanup cohort passes 313 tests in 301.72s,
  with zero final SQLite or instance-lock handles and seven final descriptors:
  `/private/tmp/pr2427-dev934-console-verified.LDwaIV`. The extra regular handle
  is the intentionally process-lifetime faulthandler log; no forced collection
  or global cleanup was used.
- Final backend plus existing worker-fault controls: **382 passed** in 60.16s,
  zero final SQLite/instance-lock handles, six final descriptors;
  `/private/tmp/pr2427-dev934-backend-final.UOHQh7`. This uses the real owned
  workspace registry and existing executor; the unnecessary log fixture and
  its four tests are absent. Separate spec and quality reviews approve the
  final replacement and all other scoped patches.
- Extra original/regression cohort on this rebase: **38 passed / 6 failed** in
  121.75s, zero final SQLite/locks;
  `/private/tmp/pr2427-dev934-original-regressions.G1FiDJ`. The failures are
  browse-return scroll 0 instead of 5, two compact chrome authority-copy checks,
  two updated footer contracts, and a scroll snapshot of 7.000000010566575
  versus restored 7. Keep these open pending independent diagnosis; do not
  weaken scroll/focus assertions or call the earlier 44-case evidence current.

This is a verified repair checkpoint, not merge readiness: 708 tests passed in
complete changed-file/cleanup cohorts (313 + 382 + 13), plus the unchanged
isolated 1,000-turn test. Six additional Library failures, 15 unchanged size
guards, previously recorded resource/preload/CSS gates and final PR checks remain.
