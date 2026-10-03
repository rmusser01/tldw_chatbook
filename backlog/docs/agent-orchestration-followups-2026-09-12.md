# Agent orchestration remaining work — 2026-09-12

## September 29 authorized burn-down — complete

The user subsequently authorized all remaining routing/progress follow-ups and
three formerly optional communication extensions. The historical September 12
completion below describes that earlier scope. Its optional-design language does
not defer the newly authorized tasks.

| Work | Tasks | Current disposition |
| --- | --- | --- |
| Selected endpoint identity and provider sampling | TASK-32929, TASK-33001.9 | Done; targeted checks and independent review passed. |
| Progress timer, refusal diagnostics and shared parser | TASK-32517, TASK-32639, TASK-32499 | Done; mounted/routing checks and independent review passed. |
| Truthful child targets and explicit pre-tool fallback | TASK-32497, TASK-32508 | Done; all three independent review findings corrected and re-reviewed with 50 passing checks. |
| Historical task closeouts and guide guard | TASK-22061, TASK-32477, TASK-32520 | Done; actual navigation-away behavior qualified and historical statuses reconciled. |
| Scoped sibling messaging | TASK-33430 | Done; exact ownership, bounded delivery, privacy checks and independent review passed. |
| Durable saved-chat progress | TASK-33431 | Done; atomic persistence, responsive lifecycle and all four new SQL worker cache entries pass targeted checks and independent physical-drain review. |
| Progress-triggered supervisor wakes | TASK-33432 | Done; final source-scope, Canvas and mixed scheduling corrections passed 72 affected checks and independent re-review. |
| Shared replay admission and publication | TASK-33663 | Done; required hooks and shared readiness refuse before clearing/worker launch; authorized queued recovery and early transcript polling pass real regressions and independent review. |
| Deferred Resend startup imports | TASK-33664 | Done; three imports defer to actual consumers and all five original startup/storage/CSS cases pass without changing budgets. |

Contracts: [ADR-199](../decisions/199-scoped-peers-durable-progress-and-wakes.md)
and [ADR-200](../decisions/200-preset-pre-tool-fallback-targets.md).
The [implementation plan](../../Docs/superpowers/plans/2026-09-29-agent-orchestration-burndown.md)
tracks this scope. All 13 tasks are Done with checked acceptance criteria and implementation
notes. The [final review and qualification record](../../Docs/superpowers/reviews/2026-09-29-agent-orchestration-burndown.md)
records the corrections, independent approvals, passing targeted selections and disclosed limits.
The current branch is integrated with dev `74bd039d607b4c961f95d1fb4f1c3d89b4552d99` (PR2964 queue shelf repair). Immutable source `b01835aeffdf13c8992e90ee63c8558a20b0d58e` independently authenticates 214 Git/disk hashes, exact incoming UI/CSS/test/user/task bytes, full testing lessons and deletion. Queue consumers 63 PASS in 261.439s including 17 painted/mounted cases, governance 8 PASS in 32.101s and ORIGINAL5 guards5/5 in 119.747s at681/686imports and 1033/1033 UI-ready; artifacts and 4,823 task guards pass. All prior positive/NON-GREEN/skip/style/physical-behavior limits and separate task states remain; root adds no new source repair or live capture. Detailed evidence remains in the review.

October 2 integration requalifies all affected tasks: shared spawn/resume initialization retains required hooks and eligible peers; exact wake refund/nonreplay and physical SQL cleanup remain. Independent41 schema and19 runtime cases approve3d14b4a, plus4 actual v2 wake-refusal cases. Root10 repair,24 runtime,4 originalstartup/storage,7 modal-quit and corrected mounted checks pass. Original import/UI limits remain681/686 and1033/1033, with no ceiling changes. Initial fixture/profile/import failures are disclosed in the final review. Separate reproduced upstream host-disposal SessionEnd suppression is tracked as **TASK33648, To Do**; no authority redesign or broader plugin/keybinding closure is claimed. Final replay integration passes40 neighboring cases,35 readiness cases,10 wake/close cases and both independent final reviews. Final upstream composition passes13 Resend UI cases54.462s and5 original budget cases75.893s, including both incoming storage variants; imports681/686 and UI-ready1033/1033 retain original drift warnings and no UI headroom. Raw profile/bare-fixture and scratch probe failures remain disclosed. Separate TASK33662 real-relaunch recovery stays To Do. Incoming TASK33621.28 is documented Done upstream; no new real-app Ctrl+Q claim. Fresh published-head Qodo, all four CI jobs and protected merge remain.

Final schema qualification passes 31 cases; mounted saved reopen/close passes two; independent compaction/wake review passes ten. Independent schema review approves 39 overlapping checks, including genuine upgrades, rollback, dictionary variants, frozen histories and exact staged shared recovery. Startup/CSS/guide passes four cases with imports 679/686 and UI-ready 1032/1033; original pins remain unchanged. Final 79-file changed-code checks and audited diagnostic inventory pass. Selections overlap and are not summed. The two initial schema qualification setup failures are corrected and preserved in the review record; no recovery validator was weakened.

Qodo's four findings remain addressed and independently reviewed. The earlier boot CSS breach is repaid with one scoped editor class, preserving actual paint and the original 608090 B limit. Readiness rebase test corrections and their negative controls remain qualified; historical failed combined runs are not claimed green. All 13 original tasks are Done with checked criteria and notes.

Delivery is through [PR #2918](https://github.com/rmusser01/tldw_chatbook/pull/2918) against dev. The user authorized merge after fresh Qodo review and remote checks, plus temporary five-minute background checks until verified merge. The preceding queue/logging integration passes 11 wake/close/Save checks, four schema checks and five startup/CSS/freeze checks; independent queue and logging/maintenance review passes 14 and 25 checks. After admission changes, seven consumers and 16 independent evidence-reuse cases pass; all 79 patch Python bytes remain unchanged from the queue/logging qualification. Selections overlap. The extra full-registry warning assertion remains unqualified because optional pydub is absent; the review record retains its earlier profile setup failure and separate passing freeze observations. No full suite, live-provider, Windows, aggregate-resource or merge result is claimed.

TASK-33647 was identified by fresh-head CI during final integration and is also Done: the legacy read-only probe and write fallback share existing repository admission, preserving the original storage ceiling, SQL lock behavior and worker retirement. Controlled cold-worker RED, unchanged ratchet GREEN, affected and independent checks are recorded in the final review. All 13 original tasks remain Done. Final Delete/Undo integration passes nine persistence/Save/close/admission checks, four unchanged storage/startup/CSS guards and two independent mounted/nonreplay checks. The 81-file static scan, ten new-file checks, all CSS bundles and the incoming 632-owner/15-sink diagnostic inventory pass. Published e8474 passes all four CI jobs and exact-head Qodo; strict protection requires updating after dev advances. The required keep-alive/project-folder integration passes ten first-send/Save/close/nonreplay/admission cases, four unchanged storage/startup/CSS guards, four independent persistence/Inspector cases and six independent lifecycle cases. Both reviews approve; the 81-file static, worker-contract and 633-owner diagnostic guards pass. Imports/UI-ready remain within original pins at 680/686 and 1033/1033. Published 64d770 passes all four jobs/Qodo; the subsequent upstream conflict preserves both exact testing-lesson append blocks. Latest hook-review integration passes 12 real consumers, four unchanged storage/startup/CSS guards, six independent controller cases and nine independent mounted modal/diagnostic cases. Both reviews approve; 81-file/new-file static, worker-contract and 634-owner diagnostics pass. Upstream TASK-33621.28 remains open; harness pump checks do not establish real-app Ctrl+Q under modals. The subsequent 1 Hz probe integration identifies and repairs an exact preacceptance maintenance refusal under existing TASK-33432 AC3. The progress/completion regression is RED before the guard; 11 repair/neighbor checks and seven maintenance caller/drain checks pass, with independent real refund/resume approval. Original census passes after byte-for-byte restoration; its earlier unexplained typing-helper breach remains disclosed. Original pins, worker contract and audited diagnostics pass. TASK-33432 is again Done after review and task hygiene; separate upstream TASK-33560 stays open. Delivery still awaits fresh published-head Qodo, all four CI jobs and protected merge.

## September 12 completion status (historical)

PR #2641 merged the earlier five reliability/verification follow-ups at
`d66908a69ef03066fed77a92edf77a326f44bd89`. The remaining approved work is now
complete locally on `codex/agent-orchestration-remaining`. This section
supersedes the historical pending-work labels retained below.

- **Done and reviewed:** approval verification TASK-13154.4, bounded segment-log
  paging TASK-18601, denial breaker TASK-18929, live per-run usage TASK-18923,
  bounded reusable webhook delivery TASK-31511, and the Settings/presets/caps
  children TASK-13154.5/.6/.7. Earlier integration corrections remain recorded
  at `7852cf47ba`.
- **Done and reviewed:** TASK-31210 visible worktree confirmation and TASK-31211
  earlier-turn recovery. Ordinary selected-authority Git creation, schema 20
  ownership, positive physical completion, shared confirmed Apply/Merge/Discard
  and the actual Console picker/card path are delivered. Preview/live schemas
  now agree for tested progress and virtual CLI/raw-shell availability.
- **Final integration complete:** `e135a085f2` fixes partial Git-reader startup
  cleanup and manual executor-submission ownership. The single scoped re-review
  approved both findings with no new issues. Final verification passed 88
  affected/capacity cases and 2 actual reopened card/Git flows after 11 RED
  failures. Earlier 25-case mounted UI, 23-case preview/raw-shell and 60-case
  lifetime selections overlap and are not summed.
- **Parent closed:** TASK-13154 is Done. All seven actual children have checked
  acceptance criteria and implementation notes; both recovery tasks are Done
  with all 9 and 8 criteria checked. No approved functional task remains open
  in this workstream.

The [complete restoration review, evidence and decisions](../../Docs/superpowers/reviews/2026-09-12-agent-worktree-restoration.md)
retains every finding, correction and qualification. The
[earlier remaining-wave review](../../Docs/superpowers/reviews/2026-09-12-agent-orchestration-remaining.md)
remains the record for the preceding changes. All current work is local; no new
PR, push or merge was performed during this completion.

Verification limits remain: inherited Requests dependency warning; UI-ready at
973/973 with zero headroom and its intentional drift warning; existing
ChatScreen size/no-growth failures, including the disclosed 21-line/3-method
addition; and eight stale historical diagnostic-label expectations already
absent at the pre-restoration baseline. The current production diagnostic
inventory/sink guard passes, and changed-file static checks add no diagnostic
identities with edited-range formatting passing. No full-suite, Windows or
live-provider result is claimed.

Ordinary Git retains the documented concurrent external metadata-replacement
limit. Automatic worktree deletion stays disabled; discard leaves a disclosed
baseline checkout, and held/uncertain or legacy work remains protected.
Communication remains bounded process-local steering/progress, explicit
supervisor relay and continuation, and versioned session tasks. Durable inboxes,
arbitrary direct peer routing and progress-triggered wakes are optional designs
outside ADR-136 and this completed scope.

## Historical investigation and delivery notes

Baseline: `origin/dev` at `8ab21ecaf372ad0b5cc8bca98dd12c427f4aef87`.
[PR #2631](https://github.com/rmusser01/tldw_chatbook/pull/2631) merged the
25-item audit scope. The supervisor and its managed sub-agents remain one
agent-orchestration workstream; these older open tickets are part of that area.

This inventory distinguishes current code observations from old ticket claims.
It does not treat every old failure as a current product defect.

The current follow-up branch is `codex/agent-orchestration-followups`.
TASK-15666, TASK-13215, TASK-2155, TASK-22720, and TASK-19642.8.3 are Done on
that branch after task review, final review, and scoped review of the headless
lifecycle amendment. Integration is pending.

## Reliability and resource follow-ups

| Task | Current evidence | Remaining work |
| --- | --- | --- |
| [TASK-13215](../tasks/task-13215%20-%20Fleet-approval-revocation-add-a-revoked-run-tombstone-and-close-the-residual-arm-read-windows.md) | Reproduced late approval after revoke, including a real local write, and mixed multi-row verdict snapshots. The old cross-lock premise was stale. | **Done on this branch** (`838e0fb1c3`, documentation correction `13456c5393`): per-kind late-arm fences, atomic verdict snapshots, content-free unowned warnings, mutation-sensitive sibling recovery. Host-lifetime tombstones intentionally retain revoked IDs; safe reclamation requires physical-worker drain proof. |
| [TASK-15666](../tasks/task-15666%20-%20busy_fleet_session_count-prunes-the-fleet-as-a-side-effect-of-a-read.md) | Coordinator terminal pruning was already at turn start; the remaining defect was retained-owner cleanup during the count. | **Done on this branch**: observational snapshot (`726409de10`) plus explicit headless next-turn cleanup (`b377eca2ff`, tests hardened in `8f9f8619a8`). Final fleet selection: 12 passed. A real placement mutation failed the intended disabled-path assertion; live-owner cancellation remains covered. All reviews approved. |
| [TASK-18601](../tasks/task-18601%20-%20Agent-run-step-log-is-a-single-JSON-blob-column-and-does-not-scale-to-the-raised-step-budget.md) | DB child-table storage and metadata-only reads already shipped; three AC are checked. `ConsoleRunLogModal` still receives/stores a complete `log_text` and builds one `TextArea`. The current full-log path loads filesystem log segments through `load_run_log_text`, not the DB step table; the load runs in a worker, but all records and rendered text are materialized. | Finish viewer paging/bounded memory across the actual log source and modal. Reconcile the old child-table paging proposal with the current segment-log viewer; do not redo the DB migration. |
| [TASK-18929](../tasks/task-18929%20-%20Agent-loop-consecutive-denial-circuit-breaker.md) | Existing run budgets bound execution; a separate consecutive-denial streak guard remains an open proposal. | Add a per-run breaker with honest terminal messaging, reset and sibling-isolation tests. Resolve the ticket's suggested small default versus its “0 or absent disables” AC before implementation. |
| [TASK-31511](../tasks/task-31511%20-%20Agent-run-webhook-delivery-spawns-a-thread-and-event-loop-per-run.md) | `schedule_run_webhook` starts a daemon thread running `asyncio.run` per delivery. | Reuse bounded delivery workers and avoid repeated unchanged-settings reads while keeping finalization nonblocking. Requires a runtime-lifecycle ADR check. |

## User-facing capabilities and recovery

| Task | Current evidence | Remaining work |
| --- | --- | --- |
| [TASK-31210](../tasks/task-31210%20-%20Console-confirm-card-for-agent-worktree-merge.md) | Merge/discard backend and controller confirmation seam exist; the Console does not wire `set_pending_worktree_merge`. Tool disclosure fails closed. | Add the visible diffstat Allow/Deny card, correct disclosure, parked/remounted state, and a real controller-to-agent-run test. |
| [TASK-31211](../tasks/task-31211%20-%20Cross-turn-persistence-for-agent-worktrees.md) | `AgentService.run_turn` initializes a fresh `_agent_worktrees` map. | Recover previous-turn unmerged work through a confirmed path while preserving DB-backed live-owner protections. Persistent handles versus a recovery UI needs an explicit design/ADR decision. |
| [TASK-18923](../tasks/task-18923%20-%20Agent-rail-live-per-run-status-line-elapsed-and-streaming-tokens.md) | Elapsed/activity rendering already exists. Fleet rows document that token totals arrive on finish, rather than providing growing live usage. | Reconcile the partial implementation, then add honest live usage where observable and verify the requested cadence/idle teardown. |

## Older verification and program records

The following six exact old failure nodes were run on the merged baseline using
the isolated worktree interpreter: **4 passed, 1 failed, 1 XPASS**. Counts are
not added to the merged PR's CI results.

| Task | Fresh evidence | Disposition |
| --- | --- | --- |
| [TASK-19642.8.3](../tasks/task-19642.8.3%20-%20Restore-Console-fleet-and-headless-wake-authority-tests.md) | All three named `test_console_fleet_wake_safety.py` nodes and `test_a_headless_wake_takes_the_same_agent_dispatch_and_budget` pass. | **Done on this branch** in `eae3599460`; verification reconciled and reviewed; current gates and frozen run-budget assertions remain, with ADR-134 automatic restrictions documented. No reproduced defect from the four original nodes. |
| [TASK-22720](../tasks/task-22720%20-%20Agent-bridge-placeholder-replacement-test-trips-the-unresolved-recovery-guard.md) | The old recovery regression XPASSed on merged dev; its task correction already rejected the swallowed-exception premise. | **Done on this branch** in `eae3599460`; stale marker removed; all replacement assertions remain and 7 agent citation cases pass normally. |
| [TASK-2155](../tasks/task-2155%20-%20Agent-branch-console-send-never-invokes-agent-bridge-pre-existing-dev-failure.md) | The manually bound conversation lacked a durable Library policy. Hydration alone failed; real policy insert plus hydration fixed test setup. | **Done on this branch** in `eae3599460`; harness repaired; both dictionary branches pass and verify raw transcript text. Production acceptance remains unchanged. |
| [TASK-13154](../tasks/task-13154%20-%20Supervisor-agent-fleet-program.md) | Parent remains In Progress; discovered definition, wake, and steering children are Done. Older deferred notes include spawn validation, Settings DB ownership, and tool-filter feedback. | Reconcile the original six-phase acceptance criterion and each deferred note with merged work before closing the parent. |

Open PR #2427 was checked for overlap: it lists none of the four test files
above. No open PR or remote branch name matched TASK-13215 or TASK-15666 when
this wave started. Repeat the check before later changes or integration.

The parent program's deferred notes were also inspected:

- The mutually exclusive named-agent/skill spawn guard now raises `ValueError`;
  that old implementation note is superseded. A supplied roster has a real
  no-reread regression (`test_run_turn_reuses_planned_agent_roster_without_db_reread`).
  The older `test_definitions_load_once_per_turn_roster_in_protocol` checks
  roster disclosure, not a read count; direct-load call-count coverage still
  needs reconciliation before closing the parent's testing note.
- Settings still constructs an `AgentsSettingsPanel` on category rendering;
  its `_derive_runs_db` opens an owned `AgentRunsDB`, and the panel has no
  explicit close hook. This ownership cleanup remains open.
- `_form_definition` still drops `RUNTIME_TOOL_NAMES` from the typed allowlist;
  the Save path does not explain those omissions. Per-save feedback remains open.
- Settings category placement is a product preference, not a correctness bug.

## Additional approval-harness failures observed in this wave

The broad affected-file run for TASK-13215 exposed these **seven inherited
failure nodes**, each reproduced with the original production files restored.
They remain verification follow-ups; these observations alone do not establish
seven product defects:

| Test file | Exact node | Observed failure |
| --- | --- | --- |
| `Tests/UI/test_console_parked_payload_rekey.py` | `test_bridges_do_not_share_a_head` | Legacy mounted-payload readiness times out; failing test leaves a waiter alive. |
| Same | `test_promoted_round_mounts_with_remaining_time_not_original_timeout` | Expects wall-clock countdown, sees the original 30-second answerable-time budget. Reconcile with current decision visibility before changing assertions. |
| `Tests/UI/test_console_mcp_approval.py` | `test_finishing_card_is_not_counted_and_keyboard_focuses_the_card` | Expected card is hidden. |
| Same | `test_alt_a_focuses_the_pending_approval_decision_select` | Focus stays on `console-setup-modal-action`. |
| Same | `test_alt_a_reaches_the_card_at_80_columns_with_inspector_closed` | Same setup-modal focus mismatch. |
| Same | `test_batch_row_widgets_have_nonzero_geometry_and_do_not_overlap_under_bundled_css` | Expected row header has zero geometry. |
| Same | `test_single_row_fast_buttons_have_nonzero_geometry_and_do_not_overlap_under_bundled_css` | Expected fast-approve button has zero geometry. |

`test_falsy_run_id_is_normalized` also fails after those leaking harness tests,
but passes alone; it is not counted as an independent human-wait defect.
The affected same-session skill-script test's preregistration/publication race
was repaired within TASK-13215 by waiting for its actual badge/payload and
ensuring worker shutdown in `finally`.

The final approval selection passed **197 tests, 5 deselected** (the five mounted
MCP cases above); a separate approval/payload selection passed **30**, and the
local-provider selection passed **16**. These selections overlap and are not
summed. Both parked-payload failures above remain outside that green selection.
Whole-file formatter/lint debt and environment dependency warnings are recorded
separately; changed lines introduce no static-check findings.

## Boundaries and order

The cancellation/read-path repairs and three stale verification tickets in this
wave are closed locally. Next, reconcile the seven inherited approval UI
failures, then address bounded log rendering and denial behavior. Worktree confirmation
precedes durable worktree recovery; live status and webhook delivery can follow
as separate, reviewable tasks.

Other existing agent-area programs (persistent memory, scheduled server work,
MCP resources/prompts, code-execution tools, and broader Settings work) require
their own scope review. They are not additional defects proven by this audit.
Module-wide formatting debt also stays separate from correctness repairs.

Direct peer addressing, progress-triggered wakes, and durable progress inboxes
remain optional designs outside [ADR-136](../decisions/136-scoped-child-progress-and-supervisor-relay.md).
The implemented communication remains bounded process-local steering/progress,
explicit supervisor relay, and versioned session tasks.

## Decisions made during this wave

These preserve the decisions and their costs if mistaken, including the
headless-cleanup assumption corrected by final revalidation:

1. Include retained-owner mutation in TASK-15666's observational-read contract.
   The initial assumption that other cleanup paths bounded retention proved
   incomplete for headless turns; see decision 7.
2. Continue the user-authorized reliability repairs without repeating design
   approval. Persistent-worktree and peer-messaging designs stay separate; the
   cost of this scope choice is that those capabilities remain unfinished.
3. Keep revoked-run fences for the interrupt host lifetime. This prevents an
   abandoned worker from regaining approval, at the cost of memory proportional
   to distinct revoked IDs. Reclamation requires physical-drain evidence.
4. Treat the atomic final verdict snapshot as the approval commitment point.
   Revocation before it denies the batch; revocation after it cannot retract a
   completed approval or side effect.
5. Include the one-line configuration-callback assertion improvement in the
   reviewed test-reconciliation batch. There is no runtime change; the tradeoff
   is broader test-file scope in that batch.
6. Normalize the old task's multiline command to a fenced block, while keeping
   dependency and temporary-directory warnings disclosed. Formatting has no
   runtime cost; the environment warnings remain unresolved.
7. Reopen TASK-15666 after a real two-turn probe showed headless owner
   accumulation. Add cleanup at the next turn boundary while preserving live
   owners and terminal-handle timing. A settled owner may wait until that
   boundary or another lifecycle cleanup; releasing a live owner would lose
   cancellation authority, so the regression covers both states.


Final preserving PERF-07 integration retains all 96 reviewed source/CSS files and eight incoming files exactly. Independent config 11, replay 13 and custody 12 cases pass; original five budgets pass in 83.786s at the unchanged 681/686 and 1033/1033 module limits. Root 24 passes/one config-retarget fixture failure and its exact incoming reproduction remain disclosed in the review. TASK-33664 AC3 is requalified and closed through CLI; all 16 scoped tasks are Done. Incoming TASK-33266 remains as shipped, and separate open work remains. Fresh published-head Qodo, CI and protected merge remain.

Final preserving release integration retains all 96 reviewed Python/CSS and eight PERF-07 files exactly, with complete upstream/local testing lessons and exact remaining incoming files. Source 353afb6ad48b7336945f4ca959c66c7192e25b7b is independently approved; 46 incoming/mounted cases pass in 35.230s and all five original budgets pass in 75.284s at 681/686 imports and 1033/1033 UI-ready. The incoming Resend absence guard is preserved with unchanged limits/workload. Final static/CSS/diagnostic/worker and 4,791 task guards pass. TASK-33664 AC3 is requalified and closed through CLI; all 16 scoped tasks are Done. Incoming release/API tasks retain shipped statuses; no installed/package/index/native release qualification or broader open-work closure is claimed. Full evidence and all earlier NON-GREEN limits remain in the review. Fresh published-head Qodo, four CI jobs and protected merge remain.


## Known-work trace integration qualification — October 2

Actual dev `2612fc56b26510630190ecacdc3855c4bffb1786` includes PR #2959/TASK-33801 known-work trace passes and PR #2925 audit documentation. The prior published `a0708efc6290351822733e8e45a9195d2aef78e8` had all four required jobs PASS and edited exact-head Qodo clear before GitHub reported the actual conflict and live strict up-to-date protection required the update. TASK-33664 AC3 was reopened and planned through CLI before the preserving 32-commit rebase. Incoming TASK-33801 remains Done, TASK-33802 remains To Do; QA scripts/data/report remain exact as shipped, without claiming a wider audit rerun or typing-flake attribution.

The sole source composition places upstream work-hint consumption and conditional idle probe inside the existing admitted batch, with initialization and generation signalling exact incoming. Provider refusal still precedes admission; the write transaction still rechecks durable state. All SQL, the 100ms bound, physical custody/cleanup, schema 76/both stamp gates/frozen AgentRuns, privacy, hooks, replay/readiness, exact twelve-line AGENT_WAKE refund and accepted nonreplay remain approved. Existing ADR097/199 apply; no new ADR, owner, permission, schema, dependency, gate or ceiling.

Initial root qualification is **NON-GREEN: 53 passes/one failure among 54 cases, 154.235s** (`/private/tmp/pr2918-tracepass-functional.{log,xml}`). The owned cold admission-count case returned an admitted zero-row/incomplete batch at its first-row assertion. Its unchanged isolated rerun passed (one case, 5.280s). The contract already permits elapsed-time yield before the first row; an external passing real-clock probe recorded 72.919ms cold lookup and 74.083ms row-guard clock delta. The original failure did not record its clock, so no deterministic host-load cause is claimed. All five incoming work-flag tests and original compaction/runtime/refund/close/custody neighbors passed in the initial selection; selections overlap and are not summed.

Only the owned admission-count test now uses the existing injected clock seam, preserving all 86 previous assertion ASTs and real SQLite/worker/admission/normalization/retirement behavior. A separate real private-profile cold-worker test verifies legal zero-row yield, pending exchange/checkpoint preservation, subsequent normalization, at most two admissions per attempt and zero registered worker handles. The external ignore-time negative control is **RED** at the intended zero-row assertion (one failure, 5.016s); no production mutation shipped. Both affected legacy/parking modules pass all **29 cases in 37.151s**, exit 0, zero failures/errors/skips. The exact incident lesson preserves the full prior lesson prefix and all positive/non-green evidence.

Final immutable source `e963c33d0476ba3ac8c868a8202232e44acfe1a4` has independent source/functional approval with no findings. Independent blob/disk authentication covers all 131 final hashes: 129 entries remain exact against the approved preserving composition; only the owned test fixture and append-only lesson differ. All package/native production and every Performance byte remain exact. Initial composition and supplemental independent reviews, manifests, AST/SQL and negative-control proofs are `/private/tmp/pr2918-tracepass-*.{json,md,log,xml}`.

Final ORIGINAL five storage/import/UI-ready/boot-CSS cases pass **5/5 in 107.723s**, exit 0, zero failures/errors/skips, after functional tests, independent reviews and artifact checks settled (`/private/tmp/pr2918-tracepass-original-budgets.{log,xml}`). Original four guard sources are exact incoming2612 with child cwd/PYTHONPATH pinned to REPO_ROOT; imports681/686 and UI-ready1033/1033 retain original drift warnings, stale pytest cleanup warnings and zero UI headroom. No pins, ceilings, warmup, counts, timeout or measured-work changes.

Final original 93 patch-Python fatal/added-line and ten new full Ruff/format checks pass, with supplemental owned fixture checks and whitespace passing. Whole-file inherited I001 and original incoming assertion wrapping remain exact; no blanket whole-file lint/format claim is made. All twelve CSS bundles reproduce. Diagnostics remain 638 owners/16 sinks (1429/56/7604 calls); workers remain 340 lookups/161 functions and 69 waits/27 roots, with no new sites. All 4,793 task IDs/paths are unique/readable. No source change follows independent approval; documentation closure preserves all 131 reviewed hashes.

All 16 scoped tasks are Done/checked after TASK-33664 AC3 qualification through CLI. Earlier PERF-07 config-retarget fixture failure and exact incoming reproduction, optional pydub warning, inherited lint/format/size/AST/profile debt, snapshot drift/stale cleanup warnings and separate open programs remain disclosed. No full-suite, live-provider, Windows, aggregate-resource, package/index/installed/native release or wider audit result is claimed. Fresh published-head Qodo, all four source-reproduction jobs and protected head-pinned merge remain delivery gates; no merge is claimed.

## Roleplay frame documentation preservation qualification — October 2

All FOUR exact published231d30d jobs completed successfully and edited Qodo remained clear with four resolved threads before the required update. Actual remote dev1d566b9d92b655168e743c96b8f14e69bec3f86d (PR2960) created BEHIND under live strict up-to-date protection enforced for admins. TASK33664 AC3 was reopened/planned through CLI before the clean39-commit rebase; no preemptive update occurred while CI ran.

Immutable source b306302e340dc1ece408b75c414314cdfaa4bfe2 preserves all177 prior approved Git/disk hashes plus31 exact incoming Markdown additions (208 total), with the capture_cloud.py deletion unchanged. The incoming spec, review and29 TASK33910 records remain byte-exact against actual1d; every incoming task stays shipped To Do/unchecked. Of26459 prior tracked paths, only the two owned plan/task qualification records differ; the new tree has26490 paths. All tldw_chatbook/native/packages/Packaging/Tests/Performance/scripts/.github trees, modes and blob IDs remain exact against231. No Roleplay-frame implementation or future slice completion is claimed.

Actual task-ID/path/readability guards pass all4823 task files; whitespace passes against exact1d. Independent immutable Git/disk/source/evidence review is Ready with no findings, authenticating proportional documentation preservation. Proofs are /private/tmp/pr2918-frame-{pre-rebase-manifest,reviewed-source-manifest,composition-proof}.json and reports are pr2918-frame-independent-review.{md,json}; actual task guard logs are pr2918-frame-task-{ids,files}.log. Existing ADR031/097/199 apply to unchanged shipped code; no new ADR, owner, authority, schema, dependency, source repair, gate or ceiling.

The original five budget result5/5 in107.713s, 681/686 imports and1033/1033 UI-ready, capture145PASS/51SKIP, source-derived diagnostic/CSS/worker checks and original static qualification carry forward by exact source identity. No new runtime/performance measurement was performed for this documentation-only addition. Original Roleplay editor/bare-profile setup failures reproduced on exact incoming dev, interrupted selections, remaining unexecuted neighbors, rejected uncreated/unexecuted readiness adapter, PERF07 config-retarget failure/baseline and cold-worker positive/NON-GREEN/negative-control evidence remain retained. New Roleplay behavior/physicalCtrlQ/relaunch remain unqualified. All prior inherited lint/format/size/AST/profile debt, typing flake, optional pydub warning, snapshot drift/stale cleanup warnings and zero UI headroom remain; selections overlap and are not summed. No full-suite/live-provider/Windows/aggregate-resource/package/index/installed/native-release/wider-audit certification follows. Separate task states, including TASK33640 In Progress/uncheckedAC5 and TASK33648/33662 To Do, remain unchanged.

TASK33664 AC3 is rechecked/closed through CLI only after guards and independent approval; all16 scoped tasks return Done/checked. The four-owned-Markdown closure preserves all208 approved hashes and deletion. Publish once with EXACT observed231d30d810d17c41030bc9fe6813bd03761852b4 lease; fresh current-head Qodo/no actionable threads, all FOUR jobs and actual live strict protection precede normal --match-head-commit merge. Verify MERGED parents/tree/concurrent changes and pause heartbeat. No merge is claimed.

## Queue shelf preserving integration qualification — October 2

All FOUR exact published `cf50848faffcc62ed41d594e686b88a165d6e5db` jobs PASS and edited exact-head Qodo is clear with four resolved threads/all pagination complete before the required update. Actual remote dev `74bd039d607b4c961f95d1fb4f1c3d89b4552d99` (PR2964/TASK33625.4 queue shelf clipping) creates BEHIND under freshly verified strict up-to-date protection enforced for administrators. TASK33664 AC3 was reopened/planned through CLI before the clean 41-commit preserving rebase. No update began while CI ran.

Immutable source `b01835aeffdf13c8992e90ee63c8558a20b0d58e` preserves all 207 prior non-overlap approved Git/disk hashes and six non-overlap incoming paths exactly. The sole lesson overlap equals the independently authenticated clean three-way composition (SHA256 `9e5732fa40c93bde513d7afbaafc847e1226a52009ca1448a5ffb7b3cac91d13`, 1127707 bytes), retaining every old incident and the incoming dynamic-Button-layout lesson. The final 214 manifest, modes/blob IDs, all 26,490 tracked/index paths and capture_cloud.py deletion authenticate. Only seven incoming paths plus the two owned prospective plan/task records differ against prior cf50848; package changes are only exact incoming prompt_queue.py and its two generated CSS streams. Native/packages/Packaging/Tests/Performance/scripts/.github trees are unchanged. Independent immutable source/evidence review is Ready with no actionable findings. Proofs and reports are `/private/tmp/pr2918-shelf-{pre-rebase-manifest,reviewed-source-manifest,composition-proof,independent-review}.{json,md}`.

The unmodified incoming queue shelf module passes **63 cases in 261.439s**, exit 0, zero failures/errors/skips; this includes all 17 new/changed painted-label/mounted-width cases and the original recovery, admission, navigation and pause/resume neighbors. All 173 original assertion ASTs remain, with 182 final. The source AST outside its CSS constant and two Button layout-refresh calls remains exact. Design-token governance passes 8 cases in 32.101s, zero failures/errors/skips. Raw/XML/hash evidence is `/private/tmp/pr2918-shelf-consumers.{log,xml}`, design-governance.{log,xml} and functional-evidence.json. Incoming TASK33625.4 stays Done/checked as shipped; root performs no new live capture, physical-key or Roleplay/relaunch certification. Selections overlap and are not summed.

All 12 CSS bundles reproduce. Diagnostics remain 639 owners/16 sinks and 1429/56/7604 calls; workers remain 340 lookups/161 functions/69 waits/27 roots, with no new sites. Actual task-ID/path/readability checks pass 4,823; fatal checks for the two incoming Python files and whitespace pass. Whole-file Ruff on those files retains 10 findings, including one exact incoming added-line RUF012 at the CSS_PATH list; the UI test module is formatted, while the exact incoming production file retains four wrapping-format hunks. This is preserved incoming style debt, not blanket lint/format cleanliness. Earlier 93 patch-Python fatal/added-line and ten new full Ruff/format qualification carry by unchanged source identity, with 1,794 historical whole-file findings retained separately. Restricted default Ruff cache writes initially failed before validation; only an isolated scratch cache was used for the completed checks, and raw invocation failures remain retained.

After all functional/review/artifact activity settled, the ORIGINAL five tested/untested storage/app-import/UI-ready/boot-CSS cases pass **5/5 in 119.747s**, exit 0, zero failures/errors/skips. All four actual guard sources are byte-exact incoming 74bd039; child cwd/PYTHONPATH pin the exact REPO_ROOT. Counts remain 681/686 imports and 1033/1033 UI-ready, preserving zero UI headroom, original snapshot drift and stale pytest cleanup warnings. No pins, ceilings, warmup, counts, timeouts or measured work changed. Raw/XML and source/hash/count receipts are `/private/tmp/pr2918-shelf-original-budgets.{log,xml}` and budget-verification.json.

Existing ADR031/097/150/199 apply; no new ADR, source repair, owner, authority, schema, dependency, dispatch rule, gate or ceiling. Schema76/both stamp gates/frozen AgentRuns, exact preacceptance wake refusal/refund/nonreplay, child initialization/scoped tools/peers, immediate app revocation/finite admitted SQL custody, Shared Send hook/replay/readiness and three deferred Resend imports remain unchanged. All prior original Roleplay failures/exact baselines/interrupted/unexecuted limits, rejected uncreated/unexecuted readiness adapter, PERF07 config-retarget failure/baseline, cold-worker positive/NON-GREEN/clock negative evidence, capture 145 PASS / 51 SKIP and optional pydub/typing/lint/format/size/AST/profile debt remain. No full-suite/live-provider/physical Roleplay CtrlQ/relaunch/Windows/aggregate-resource/package/index/installed/native-release/wider-audit qualification follows. All separate task states remain unchanged, including TASK33640 In Progress/unchecked AC5, TASK33648/33662 To Do and 29 TASK33910 To Do/unchecked.

TASK33664 AC3 is rechecked/closed through CLI after actual guards and independent approval; all 16 scoped tasks return Done/checked. The four-owned-Markdown closure must preserve all 214 reviewed hashes/deletion. Publish once with EXACT observed `cf50848faffcc62ed41d594e686b88a165d6e5db` lease, verify actual refs/body/current-head Qodo/no actionable threads and ALL FOUR fresh jobs before normal protected --match-head-commit merge. Verify GitHub MERGED plus actual parents/tree/concurrent changes, qualify concurrency, then pause heartbeat. No merge is claimed.

## Provider presentation and hook visit final integration qualification — October 3

All FOUR published f8bcd94f6b50249e8e07356b9a4320483362406e jobs passed and edited exact-head Qodo was clear (four resolved threads, complete pagination) before strict-base updates began. Live strict up-to-date protection is enforced for administrators. The required preserving provider rebase integrated actual dev420b53a63df54d97e774a4f3bfa3b09df9d32e93 (PR2970/TASK33922), then the clean49-commit hook update integrated actual dev0409592a2db8d50825d3482c144bc84ef6d1d523 (PR2967/TASK33642). TASK33664 AC3/4 were prospectively planned/open before integration. No rebase began while published-head jobs were running.

Provider composition retains the exact structured400/404 model-unavailable classifier before incoming labelled HTTP fallback. The incoming HTTP test double was missing requests.Response.json; only that faithful empty-response method was added, preserving all ten original assertions. Independent review then identified the actual typed-provider Console presentation gap. ADR211 defines the separate ephemeral sanitized display field on ChatAPIError and keyword-only repr/compare-excluded RunOutcome.console_copy. Direct toast/system-row and primary-agent failure rows now receive the existing provider display/model/recovery copy without changing exception class/default str/message/status/provider/retry_after, model/fallback semantics or content-free diagnostic STEP_ERROR/SQL/run-log bytes. The existing diagnostic helper, projector, classifier, persistence and safe-step ASTs remain exact. No presentation is persisted, logged, sent into agent/child context or used as launch/fallback authority; post-hook replacement discards stale presentation. Existing ADR063/179/197/097/150/199/200 remain applicable, with new accepted ADR211 linked from TASK33664 and the plan.

Source and actual behavior have independent Ready approval at d9f315107052369751cfbb6d550d1dd878fc758f; that source carries byte-exact into final immutable23b1803e366f7393babf7547b288b834d8e7d204. Final provider presentation selection passes12 in3.707s, fallback45 in10.786s, real service/audit neighbors5 in2.082s, controller diagnostic/stuck neighbors6 in0.848s, gateway9 in2.100s, and final-head actual agent400/404 confirmation2 in1.883s. Earlier incoming consumers735PASS79.350s, actual numeric-loopback transport15PASS8.150s and faithful response-copy11PASS1.666s remain authenticated. Selections overlap and are never summed. The excluded unchanged old httpx agent fixture is not newly qualified.

NON-GREEN evidence is retained: original response-copy7PASS/4FAIL2.057s for missing double.json, initial gateway8PASS/1FAIL2.642s for uncaught typed presentation, new-fixture RED6FAIL1.563s and first-green6FAIL1.735s stopped at hook admission and do not prove display. That production attempt was discarded before corrected admitted-profile RED6FAIL2.019s. Admitted first-green4PASS/2FAIL2.973s had two incorrect literal test expectations, then selected11PASS3.483s. Actual default-agent RED2FAIL2.377s precedes GREEN2PASS2.137s; the append-only lesson erratum corrects earlier2.383/2.140 duration transcriptions, preserving every prior byte and unchanged raw/XML. The original controller selection2PASS/2FAIL1.989s still fails before delivery at inherited hook admission; no old markers, source guards, config retargets or readiness adapters were altered. Raw/XML and receipts are /private/tmp/pr2918-provider-* and provider-presentation-*; initial/final independent reports preserve every positive and NON-GREEN limitation.

Hook integration has no prior238-manifest or actual feature-patch overlap across its ten incoming files. All237 other prior Git/disk hashes remain exact; the sole lesson change is the authenticated605-byte append-only duration erratum. All ten incoming040 files are byte-exact, yielding248 approved source hashes. All26,495 tracked/index paths, modes/blob IDs equal the prospectively planned024b9982cb tree with exactly ten incoming paths substituted, and capture_cloud.py stays deleted. All42 original hook functions and115 assertion ASTs remain exact, with151 final assertions. Visit reuse remains presentation-only: shared Send/review/Settings/launch continue authoritative full reads; validated selection, no-follow file/directory/both-lock posture, data-root/admission/seal/recovery invalidation and locked memory recheck are preserved as shipped. Hooks Settings loads at its actual category with name-routed events and existing staged Save/Revert/review behavior. Incoming TASK33642 stays Done/checked, as does incoming TASK33922; no wider incoming work is closed.

Unchanged bounded hook permission consumers pass67 in25.935s, Send admission20 in8.287s, Console hook review and Settings23 in83.333s, and screen preimport1 in3.632s, all with zero failures/errors/skips. Preimport is556/556 modules (zero headroom),411931 LOC, Settings45 modules. Incoming guards tighten visit43→39 config,134→112 storage,44712→35351 opens (helper9/slack1.05 unchanged) and preimport557→556; root did not edit or weaken a pin. Upstream184PASS/two Settings-search baseline failures are incoming historical notes, not a root search sweep or wider audit. Source/evidence and independent approval are /private/tmp/pr2918-hooks-{pre-rebase-manifest,reviewed-source-manifest,lesson-erratum,functional-evidence,independent-review}.{json,md}.

After every functional/artifact/review activity settled, ORIGINAL tested/untested storage, app-import, UI-ready and boot-CSS cases pass5/5 in85.719s, exit0, zero failures/errors/skips. All four actual guard sources are byte-exact incoming040, and import/UI subprocess cwd/PYTHONPATH pin exact REPO_ROOT. Counts remain681/686 imports and1033/1033 UI-ready, with zero UI headroom. Both actual storage variants satisfy the shipped tightened visit limits. Two in-suite snapshot-drift warnings and the post-summary stale pytest rm_rf cleanup tail against existing garbage remain in the authenticated raw log; no foreign cleanup occurred. The initial receipt warning description omitted that tail and was corrected as metadata only, with initial-description receipt retained and raw/XML hashes unchanged. Supplemental independent review reauthenticates all248 source hashes, five original names, four guards, counts, raw/XML and deletion after the run. Evidence is /private/tmp/pr2918-hooks-original-budgets.{log,xml}, budget-verification.json and budget-independent-review.{md,json}. No count, ceiling, warmup, timeout or measured work was changed.

Diagnostics rebuild exactly639 owners/16 sinks with1429/56/7604 calls; workers remain340 lookups/161 functions and69 waits/27 roots, none new. All twelve CSS bundles carry their reproduced source identity; CSS/build/native/SQL/schema/fallback sources remain exact. SupportedPython3.12.11 task-ID/path/readability guards pass4826 and whitespace/fatal checks pass. The seven incoming hook Python files retain118 whole-file Ruff findings and three wrapping-format-debt files exactly as shipped; four are formatted. First Ruff invocation placed cache-dir at an unsupported global position and exited before validation; the raw failed invocation remains, and corrected validation uses an isolated scratch cache. Prior provider13-file98-findings/one incoming I001, owned eight-file325-findings/zero owned-added findings, original93-patch/ten-new-file qualification and1794 historical findings remain separate overlapping selections. No blanket whole-file style cleanliness is claimed.

All schema76/both stamp gates/frozen AgentRuns, exact twelve-line preacceptance AGENT_WAKE refusal/refund/nonreplay, child initialization/scoped tools/peers, immediate app revocation and finite admitted SQL custody, Shared Send admission/hook/replay/readiness/held-slot recovery and three deferred Resend imports remain approved. All earlier Roleplay failures/exact incoming baselines/interrupted/unexecuted limits and rejected uncreated/unexecuted readiness adapter, PERF07 config-retarget failure/baseline, cold-worker positive/NON-GREEN/negative-clock evidence, capture145PASS/51SKIP, optional pydub/typing/lint/format/size/AST/profile/snapshot/stale cleanup/large-UI/flat-store limits remain disclosed. No full-suite/live-provider/user-key/physicalCtrlQ/relaunch/Windows/aggregate-resource/package/index/installed/native-release/wider-audit qualification follows. Separate TASK33640 In Progress/uncheckedAC5, TASK33648/33662 To Do and29TASK33910 To Do/246unchecked remain unchanged.

TASK33664 AC3/4 are rechecked and Done through CLI only after independent source/functional/original-budget approval; all16 scoped tasks are Done/checked with notes. The four-owned-Markdown closure preserves all248 approved source/lesson/ADR/index hashes and deletion; no source edits follow approval. Publish once with EXACT observed f8bcd94f6b50249e8e07356b9a4320483362406e lease, verify actual refs/body/current-head Qodo and ALL FOUR fresh jobs, then satisfy actual live strict protection before normal --match-head-commit merge. Verify MERGED parents/tree/concurrency, then pause heartbeat. No merge is claimed.

## Audio refusal and Roleplay review strict-base qualification — October 3

All FOUR published 30c1595985b9abb6281b652a99b0df0e8c6a2c89 jobs passed and exact-head Qodo was clear with four resolved threads and complete pagination before strict-base work. Actual dev f87153b799eb6285f2fb417674af1954429ace26 caused CONFLICTING/DIRTY under freshly verified strict up-to-date protection enforced for administrators. TASK33664 AC3 was prospectively reopened/In Progress through CLI and planned at 40c5d42 before the preserving 51-commit rebase. Final immutable source fe6534932ffd159822dd435cc6e4571ca94adbae has independent source/targeted-functional/artifact-preservation and supplemental original-budget approval. No runtime/test repair follows approval.

All 248 prior approved Git/disk hashes remain exact, with ZERO incoming manifest overlap. Of 105 incoming paths, 104 remain byte-exact and the sole actual feature overlap is the shared live-verification lesson. Its entire 231,961-byte common prefix, 5,706-byte prior suffix and 876-byte incoming suffix are preserved at 238,543 bytes; stage3 of the only rebase conflict equaled the entire prior published lesson. The 353-hash union and all 26,594 tracked/index paths, modes/blob IDs equal planned40c5 tree plus exact incoming replacements and the authenticated composed lesson; capture_cloud.py stays deleted. All 208 original assertions (61/65/82 by module, now 65/67/85) and all original audio test functions remain exact.

The only incoming production change is the exact server audio authentication-refusal translation: existing runtime admission and active-client resolution still precede dispatch; AuthenticationError from passive/warm STT health, warm capability lookup or streaming status becomes existing PolicyDeniedError/auth_required with preserved cause/server authority. Current administrator checks, warm403→admin_required, older-server behavior, non-auth errors, local audio and other operations retain their contracts. Existing Proposed ADR178 remains Proposed and relevant; no new ADR, owner, authority, storage, schema, dependency, UI caller, gate or ceiling. The independent review's required filename correction is applied in the owned plan/task links to backlog/decisions/178-server-audio-diagnostic-admin-boundary.md.

The unchanged incoming three audio modules, local audio and auth/redirect/core policy neighbors pass 46 in 2.247s, exit 0, zero failures/errors/skips using fake clients/MockTransport/private pytest profiles. Raw/XML and command/exit/hash receipts remain /private/tmp/pr2918-audio-functional*. No live server/provider/user key/microphone or physical-key probe was run. Raw retains post-summary stale pytest rm_rf cleanup warnings against existing garbage. The initial metadata description incorrectly mentioned an emitted pydub warning; corrected before independent review, initial description retained, raw/XML hashes unchanged. Historical optional pydub limitation remains. Initial pre-rebase preparation incorrectly expected 83 unchecked new Roleplay criteria and stopped BEFORE task/source mutation; actual 73 authenticated and corrected before prospective plan commit, not behavior or test failure evidence. The independent review's stalled bulk cat-file request ended with exit130 and a read-only ps diagnostic was sandbox-denied; neither counts as positive evidence or an application failure. The completed disk SHA256/mode/recomputed Git blob address proof authenticated source instead. The prospective added-section hash is now named explicitly in the receipt alongside authenticated whole-plan/task hashes; the initial ambiguous-scope receipt remains, with no source or head change.

Four incoming runtime/test Python files pass fatal/Ruff/format with zero findings and all four formatted; no owned Python edits. Actual supportedPython3.12.11 task-ID/path/readability guards pass 4,840. The whole incoming QA documentation diff retains 274 whitespace findings reproduced byte-for-byte from exact upstream; all owned plan/task/lesson changes pass whitespace checks. All prior source-specific fatal/added-line/format, inherited whole-file style debt, diagnostics 639 owners/16 sinks and 1429/56/7604 calls, workers 340 lookups/161 functions/69 waits/27 roots and twelve CSS reproductions carry by exact relevant source identity. No blanket repository style or wider QA cleanliness is claimed.

After all functional/artifact/review activity settled, ORIGINAL tested/untested storage, app-import, UI-ready and boot-CSS cases pass 5/5 in 73.376s, exit 0 zero failures/errors/skips. Four actual guard sources remain byte-exact incoming f871 and prior 040; import/UI subprocess cwd/PYTHONPATH pin exact REPO_ROOT. Fresh counts remain 681/686 imports and 1033/1033 UI-ready, zero UI headroom; historical preimport 556/556 remains separately identified. Both actual storage variants preserve tightened 39 config/112 storage/35,351 opens/helper 9/slack 1.05 limits. Two actual snapshot-drift warnings, the post-summary stale pytest cleanup tail and emitted PyAudio-unavailable message remain in authenticated raw evidence. No pydub message was emitted in this budget selection; no install, mask, optional audio/microphone qualification or foreign cleanup. No pins, ceilings, counts, warmup, timeout or measured work changed. Supplemental review reauthenticates 353 hashes after the run. Evidence: /private/tmp/pr2918-audio-original-budgets.{log,xml}, budget-verification.json and budget-independent-review.{md,json}.

Incoming TASK33660 and33621.44 remain Done/checked as shipped. All 13 TASK33781–33793 remain To Do with 73 unchecked criteria; incoming Roleplay QA reports/harnesses are authenticated historical records, NEITHER executed nor newly behavior-certified by root. All 29 TASK33910 remain To Do/246 unchecked. No wider incoming work is closed. All sixteen scoped orchestration tasks return Done/checked only after independent approval. Existing schema 76/stamp gates/frozen AgentRuns, exact 12-line preacceptance AGENT_WAKE refusal/refund/nonreplay, child initialization/scoped tools/peers, app authority revocation/finite SQL custody, shared Send admission/hook/replay/readiness/HELD recovery and three deferred Resend imports remain approved unchanged.

Every earlier positive/NON-GREEN/baseline/interrupted/unexecuted source qualification and metadata correction remains in the canonical records: Roleplay original failures/exact incoming baselines/rejected uncreated-unexecuted readiness adapter; PERF07 config-retarget baseline; cold-worker legal zero-row/unrecorded-clock/negative-clock controls; capture 145 PASS/51 SKIP; provider presentation/fallback/audit controls; hook visit/Settings and original budget evidence; optional pydub/typing/style/size/AST/profile/snapshot/stale cleanup/large-UI/flat-store limits. Overlapping selections are never summed. Separate TASK33640 In Progress/unchecked AC5, TASK33648/33662 To Do and other incoming/separate statuses remain unchanged. No full suite, live providers/user keys, physicalCtrlQ/relaunch, Windows, aggregate-resource, package/index/installed/native-release or wider-audit qualification follows.

Documentation-only closure changes only the four owned Markdown records, repairs the ADR178 links and preserves all 353 reviewed hash/deletion. Fresh published-head Qodo, all FOUR jobs and actual live strict protection remain required for a normal protected --match-head-commit merge. Do not rebase while those fresh jobs run. Verify MERGED state/parents/tree/concurrency before pausing the heartbeat.

## Legacy flat deletion and root-fork strict-base qualification — October 3

All FOUR published 941715c876f11697df4e755bc726633bd0ace27c jobs passed and exact-head Qodo was clear, with all four threads resolved and complete pagination, before strict-base work. Actual dev 22628b0f3c466c8593ecf95c255cd77498f1778f caused BEHIND under freshly verified strict up-to-date protection enforced for administrators. TASK33664 AC3 was prospectively reopened/In Progress through CLI and planned at d2bf1adbeed47eae8c6f084ebb66071db83109ee before the clean preserving rebase. Immutable source 915a549fa1ccde40db54c14955372fcefeab6ee1 has independent source, targeted-functional, composition and artifact approval plus supplemental original-budget approval. No runtime or test repair follows approval.

Incoming PR2965/TASK33628.6 changes eleven paths. All 351 prior non-overlap approved Git/disk hashes and nine incoming non-overlap paths remain exact. Both actual overlaps, console_chat_store.py and ChaChaNotes_DB.py, equal independently reproduced clean three-way compositions, with no changed-function intersection. The prior eight store/two DB feature functions and incoming six store/five DB functions remain exact. All 362 union hashes and 26,595 tracked/index paths, modes and blob IDs authenticate against the prospectively planned tree plus exact incoming replacements and the two composed files. The entire live-verification lesson remains unchanged and capture_cloud.py stays deleted. All 167 original assertion ASTs remain exact (checkpoint87→90, deletion37→158, metadata43→53;301 final).

The incoming root_fork metadata defaults false and serializes only when true, preserving exact unmarked JSON. Marked first-message sibling/before-first USER roots remain separate branches and carry the marker through whole-record rewrites and durable acceptance; user_root_fork rejects a parent. Legacy unmarked mixed/childless USER roots retain chaining. Older/synced/imported unmarked forks, voice and image/video before-first rows retain the incoming documented limits and are not newly certified. Persisted IDs of the complete in-memory deletion subtree, including off-path roots, feed the original service/transaction boundary; missing IDs remain lenient. Same-conversation live seeds and recursive parent traversal include hidden children, preserve the selected-version fence, atomic tombstone/Undo/trace/Sync custody and use the parent index. Variable binding uses min(500,live SQLite variable limit) batches within the original transaction. The lazy helper is imported only at its existing operation sites. Existing schema76, progress, permission, recovery, provider and diagnostic boundaries remain unchanged. No new ADR is required; the prospective plan links existing boot-budget/semantic-trace ADR097, ADR199 and ADR211. No new owner, schema, authority, dependency, ceiling or dispatch boundary is introduced.

The unchanged three-module selection is NON-GREEN:140 PASS/9 FAIL in41.990s XML, exit1, zero errors/skips. Deletion persistence39 and metadata56 pass completely; checkpoint45 pass with nine older fixture failures. Those exact nine nodes on an archived exact incoming226 snapshot reproduce9 FAIL in6.786s with byte-exact failure messages at existing semantic-write authorization, unresolved-dispatch cursor or terminal-receipt guards. The new root-fork durable acceptance case passes. No production/test/marker/readiness/config override or masking was introduced to turn old fixtures green. Bounded private-profile progress close, durable fork/reload/durability and trace-detachment neighbors pass10 in17.221s, exit0, zero failures/errors/skips. Selections overlap and are never summed. Real SQLite, controller Resend/before-first Send, deletion/Undo/rollback, low-variable limits and query plans are covered by the passing scoped cases; no live server/provider/key or physical-key probe ran.

The root three-module run also reports file descriptors14→217, growth203 above200. Its cause remains unqualified: the shorter nine-node baseline is not a comparable resource selection and does not establish an inherited cause. No aggregate-resource claim follows. Raw post-summary stale pytest rm_rf cleanup tails against existing garbage remain, with no foreign cleanup. Functional/baseline/neighbor command, exit, raw/XML and exact-baseline receipts are /private/tmp/pr2918-flat-functional-evidence.json and the corresponding flat-{functional,incoming-baseline,neighbors} files. Independent source review is flat-independent-review.{md,json}.

Ten incoming/composed Python files pass fatal checks with zero findings. Whole composed and exact incoming selections each retain801 Ruff findings; the sole added-line RUF059 is exact incoming test code/message/source. Three files, including the new lazy helper, are formatted; seven retain inherited wrapping debt. No owned Python edits or blanket style cleanliness are claimed. Actual rebuilt diagnostics remain639 owners/16 sinks and1429/56/7604 calls; workers340 lookups/161 functions and69 waits/27 roots have no new sites. SupportedPython3.12.11 task-ID/path/readability guards pass4,840 and owned/current diff whitespace passes. Existing twelve CSS reproductions and native/workflow/script qualifications carry by exact relevant source identity. Evidence is flat-static-proof.json; the non-existent diagnostic-script read exited before validation, followed by the correct actual guard.

After ALL functional/artifact/review activity settled, ORIGINAL unchanged tested/untested storage, app-import, UI-ready and boot-CSS cases pass5/5 in71.134s XML, exit0, zero failures/errors/skips. Four actual guard sources remain byte-exact prior941/incoming226; import/UI subprocess cwd/PYTHONPATH pin exact REPO_ROOT. Fresh counts remain681/686 imports and1033/1033 UI-ready, zero UI headroom; historical preimport556/556 is separately identified. Storage39 config/112 storage/35,351 opens/helper9/slack1.05 limits remain unchanged. Two actual snapshot-drift warnings, emitted PyAudio-unavailable message and the post-summary stale cleanup tail remain in authenticated raw evidence. No pydub message was emitted in this selection; no installation, masking, optional audio/microphone or foreign cleanup qualification. The initial generic warning-list field was labelled inSuiteWarningLines although it also contained cleanup lines; corrected before supplemental review to rawWarningLines plus two actual in-suite warnings, with initial receipt retained and raw/XML hashes unchanged. No pin, ceiling, count, warmup, timeout or measured work changed. Supplemental review reauthenticates all362 hashes/modes/blob addresses and full tree/index after the run. Evidence: flat-original-budgets.{log,xml}, budget-{command,exit,verification}.json and budget-independent-review.{md,json}.

Incoming TASK33628.6 remains shipped Done/three checked criteria. TASK33664 AC3 remains In Progress and unchecked for the subsequently required strict-base recovery update; AC4 remains checked. The other fifteen scoped tasks stay Done/checked. All separate/incoming task states remain unchanged, including TASK33640 In Progress/AC5 unchecked, TASK33648/33662 To Do, thirteen Roleplay tasks To Do/73 unchecked and twenty-nine TASK33910 To Do/246 unchecked. All prior schema/stamp/frozen AgentRuns, exact AGENT_WAKE refusal/refund/nonreplay, child initialization/scoped tools/peers, app authority revocation/finite SQL custody, shared Send admission/hook/replay/readiness/HELD recovery and three deferred Resend imports remain approved.

Every earlier positive/NON-GREEN/baseline/interrupted/unexecuted result and metadata correction remains in the canonical records: Roleplay failures/exact baseline/rejected uncreated-unexecuted readiness adapter; PERF07 config-retarget baseline; cold-worker legal zero-row/unrecorded-clock/negative controls; capture145 PASS/51 SKIP; provider presentation/fallback/audit; hook/Settings/audio and original budgets; optional/runtime/typing/style/size/AST/profile/snapshot/stale cleanup/large-UI/flat-store limits. No full-suite, live-provider/user-key, physicalCtrlQ/relaunch, Windows, aggregate-resource, package/index/installed/native-release or wider-audit certification follows.

This completed flat qualification is preserved before the further required recovery update, without publication or source repair. The subsequent prospective plan controls next actions; remote941715 remains the exact publication lease.
