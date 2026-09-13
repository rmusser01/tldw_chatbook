# Agent worktree restoration review record

Status: complete. TASK-31210 (9/9 criteria), TASK-31211 (8/8), and parent TASK-13154 (1/1) are Done. All five restoration slices passed task review; the broad integration review's two startup-lifetime findings were fixed in `e135a085f2` and independently re-reviewed with no new issues. The work is committed locally on `codex/agent-orchestration-remaining`.

The delivered path uses ordinary selected-authority Git worktrees, durable original-base/ownership records, positive physical completion, exact visible confirmation, and earlier-turn Console recovery. Discard retains its disclosed detached baseline checkout. Communication remains bounded process-local steering/progress and explicit supervisor relay; durable inboxes, arbitrary peer routing and progress-triggered wakes are outside the approved scope.

Final correction verification: 88 affected/capacity cases and 2 actual reopened card/Git flows passed after 11 behavioral RED failures. Earlier reviewed selections and all raw-evidence pointers below remain applicable; overlapping counts are not summed. The current diagnostic inventory/sink check passes. Inherited ChatScreen size guards, eight stale historical diagnostic-label expectations, the Requests warning and startup at 973/973 with no headroom remain explicitly disclosed. This is not a full-suite, all-guards-green, Windows or live-provider claim.

This record supplements the earlier agent-orchestration-remaining review; historical findings, failed attempts, chronology deviations and rulings remain preserved below.

## Completed review gates

- Creation: a11db916ff plus test-only review fix d63d4f2a19. Targeted affected-neighbor run260passed; post-format behavioral selection7passed; review-fix selection3passed. Scoped static added no diagnostic identities and edited-hunk formatting passed. Independent review found two missing test proofs; both are addressed.
- Durable records/drain: 4756f88793 plus review fix94f7110db1. Focused31passed and directly affected DB neighbors104passed; review-fix5passed. Scoped static added no diagnostic identities and edited-hunk formatting passed. Branch-shape validation and actual-v19 migration proof findings are addressed.

Every listed pytest run has one inherited RequestsDependencyWarning for the existing environment. No dependency changed, no full suite ran, and this record does not claim pristine test output. Exact commands/output remain in the plan-specific scratch reports/evidence.

## Process qualifications

The creation worker wrote production before the new positive regression; there is no legitimate pre-implementation RED for that case. Obsolete negative assertions failing after production edits are not TDD evidence. This chronology was disclosed, retained and not reconstructed. The durable-record slice and its branch-validation fix have valid pre-implementation failures.

Ordinary local Git is the approved implementation. Application checks detect observable authority/root/metadata changes at operation boundaries; they do not prove atomic protection against another process replacing Git metadata during a command. Automatic checkout deletion stays disabled.

## Independent verdicts

### 2026-09-12-agent-worktree-creation-restoration

### Spec Compliance

- ❌ Issues found: the production implementation covers the requested creation and routing path, but the patch does not provide the required controller/bridge proof that the accepted turn's exact writable selected authority is forwarded, nor the required retention proofs for provider-admission and worktree thread-start failures. The only changed controller test asserts the negative `None` case (`Tests/Chat/test_console_turn_execution_context.py:731`); direct service fixtures begin below the controller/bridge boundary (`Tests/Agents/test_fleet_runtime.py:4109`, `Tests/Agents/test_fleet_runtime.py:4169`, `Tests/Agents/test_fleet_runtime.py:4207`).

### Strengths

- `tldw_chatbook/Chat/console_chat_controller.py:25384` obtains authority through `capture_run_admitted_workspace_roots` using the accepted `project_selection`, and `tldw_chatbook/Chat/console_chat_controller.py:25402` composes a fresh fail-closed kill-switch check with the captured authority guard. The bridge passes that same object directly into the service at `tldw_chatbook/Chat/console_agent_bridge.py:6755`.
- `tldw_chatbook/Agents/agent_service.py:4145` refuses when the real local provider is absent, `tldw_chatbook/Agents/agent_service.py:4152` refuses absent source authority without a fallback root, and `tldw_chatbook/Agents/agent_service.py:4159` checks write permission, the live guard, and source identity before Git. It stores the created checkout before the post-create guard and provider admission at `tldw_chatbook/Agents/agent_service.py:4182`, and the child guard pins both source and child identity at `tldw_chatbook/Agents/agent_service.py:4194`.
- `tldw_chatbook/Agents/agent_worktree.py:105` validates the run identifier before using it in a branch or path; `tldw_chatbook/Agents/agent_worktree.py:122` adds a full random destination suffix; and `tldw_chatbook/Agents/agent_worktree.py:131` uses the captured base SHA with generated hooks disabled. The focused advancing-HEAD regression verifies the base behavior at `Tests/Agents/test_agent_worktree.py:179`.
- The real-service regression uses distinct selected and fallback repositories and verifies that the child bytes land only in the registered isolated checkout (`Tests/Agents/test_fleet_runtime.py:4094`, `Tests/Agents/test_fleet_runtime.py:4146`). The no-authority/plain-sibling test also proves refusal does not prevent ordinary fleet work (`Tests/Agents/test_fleet_runtime.py:4045`).
- Inline/no-fleet worktree isolation still refuses without shared execution at `tldw_chatbook/Agents/agent_service.py:5751`. Automatic retirement remains routing-only at `tldw_chatbook/Agents/agent_service.py:4223`.

### Issues

#### Critical (Must Fix)

None.

#### Important (Should Fix)

- `Tests/Chat/test_console_turn_execution_context.py:731`: the sole changed application-boundary assertion proves only that a read-only accepted selection produces `worktree_repo_authority=None`. There is no positive test showing that a writable selected binding becomes the exact authority passed to `ConsoleAgentBridge.run_reply`, and no controller-level test showing removal, retargeting, root replacement, or a newly enabled kill switch makes that forwarded guard fail closed. The direct `RunAdmittedWorkspaceRoot` fixtures at `Tests/Agents/test_fleet_runtime.py:4109` bypass the source-selection and guard-composition code where a wrong repository or stale binding could be introduced. Add an accepted-turn controller regression with selected and unrelated bindings, assert the forwarded object's binding/root/identity are the selected values, then mutate each live condition and assert its guard refuses. This is explicit acceptance coverage, and without it the task's central exact-authority boundary is unverified.
- `Tests/Agents/test_fleet_runtime.py:4202`: the added retention test covers only a post-create authority-guard failure. The brief separately requires retention when `LocalToolProvider.admit_run_workspace_root` fails and when `Thread.start()` fails after successful worktree admission. Existing generic thread-start tests do not inject worktree authority or create a checkout, so they cannot show the new record survives (`Tests/Agents/test_fleet_runtime.py:3078`). Add focused tests that force provider admission and fleet thread start to fail after real creation and assert the checkout record, directory, and branch remain while per-run routing is absent/retired.

#### Minor (Nice to Have)

- The reported targeted runs contain one inherited `RequestsDependencyWarning`. The report identifies it as environment dependency skew and this patch changes no dependency surface, so it is non-blocking, but the test output is not pristine.

### Assessment

**Task quality:** Needs fixes

**Reasoning:** The implementation itself is compact and follows the selected-authority, double-check, child-containment, exact-base, and preservation design. Required tests are missing at the most important application boundary and for two post-creation failure paths, so the package does not yet establish the full requested behavior.


### Finding Verdicts

- **Accepted writable selected authority and live invalidations lack controller/bridge proof** — ADDRESSED. `Tests/Chat/test_console_turn_execution_context.py:754` drives a real accepted Console turn with distinct selected and unrelated writable bindings; `Tests/Chat/test_console_turn_execution_context.py:878` verifies the exact selected binding, resolved root, and captured root identity passed to `run_reply`. The forwarded guard is then exercised against binding removal, same-ID retargeting, root replacement, a newly enabled kill switch, and a kill-switch read failure at `Tests/Chat/test_console_turn_execution_context.py:884`.
- **Real Git retention after provider-admission and `Thread.start()` failures lacks proof** — ADDRESSED. `Tests/Agents/test_fleet_runtime.py:4239` forces real post-creation provider admission failure and asserts the retained checkout directory and branch plus absent routing at `Tests/Agents/test_fleet_runtime.py:4277`. `Tests/Agents/test_fleet_runtime.py:4287` forces the first fleet `Thread.start()` to fail after successful admission and asserts retained checkout/branch, retired routing, and completed coordinator cleanup at `Tests/Agents/test_fleet_runtime.py:4330`.

### New Breakage in the Fix Diff

None.

### Out-of-Scope Observations

None.

### Verdict

**Fix round:** All findings addressed, no new Critical/Important breakage.


### 2026-09-12-agent-worktree-records-and-drain

### Spec Compliance

- ❌ Issues found: the required Git branch-shape validation is incomplete at `tldw_chatbook/DB/agent_worktrees.py:75–87` (Important finding below).
- ✅ The six repository methods borrow the provided DB, retain complete identity tuples, constrain ownership and operation transitions, and perform metadata-only scoped queries (`tldw_chatbook/DB/agent_worktrees.py:90–326`). All requested production files and test-file changes are represented in the package.
- ✅ Physical completion atomically detaches callbacks with owner removal and invokes them outside the lock; uncertainty and late-registration outcomes are retained (`tldw_chatbook/Agents/execution_capacity.py:277–344`). Creation, recovery operations, and UI integration remain later scope.

### Strengths

- `tldw_chatbook/DB/agent_worktrees.py:155–188, 242–326`: parameterized writes and guarded state transitions prevent duplicate ownership replacement, foreign execution completion, competing claims, and stale operation finalization.
- `tldw_chatbook/Agents/execution_capacity.py:277–344`: callback failure isolation and one-time detachment are simple and preserve existing physical-owner accounting. Tests exercise a callback that re-enters `snapshot()` and actual delayed worker completion (`Tests/Agents/test_execution_capacity.py:77–140`; `Tests/DB/test_agent_worktree_recovery.py:234–301`).
- `tldw_chatbook/DB/agent_worktrees.py:121–124`: construction only borrows the database; no reconciliation silently changes durable recovery state.

### Issues

#### Critical (Must Fix)

- None.

#### Important (Should Fix)

- `tldw_chatbook/DB/agent_worktrees.py:75–87`: `_branch` checks a dot only at the start of the entire name, so `agent/.hidden` passes even though a Git ref component cannot start with a dot. It also accepts `-agent`, which is invalid as a branch name. These values can be persisted as supposedly validated durable ownership metadata, violating this task's explicit valid-branch boundary requirement and leaving later recovery with an unusable branch. Reject dot-prefixed components and leading hyphens, and add focused boundary cases alongside the existing bad-space test.

#### Minor (Nice to Have)

- `Tests/DB/test_agent_worktree_recovery.py:50–100`: the test named `test_v19_upgrade_preserves_definition_cap_and_structural_record_on_reopen` starts with the current schema and never restores a v19 database. The separate reference-SQL test does reconstruct v19, and the existing v18 runtime neighbor crosses this migration, but the named test does not directly demonstrate its claimed runtime v19 upgrade. Seed v19 before reopening through `AgentRunsDB`, or rename it to describe its actual reopen coverage.
- The implementer's reported targeted runs contain an inherited `RequestsDependencyWarning`. This is recorded environment noise, not a regression attributed to this patch; no dependency change is requested.

### Checks and Scope

- Read the supplied immutable review package once; the tool truncated its middle, so recovered only the omitted test/callback hunks from the same package. No changed source was reread to repeat the diff review.
- Named concrete risk: callback-driven persistence must preserve DB thread ownership and write serialization. Checked `tldw_chatbook/DB/AgentRuns_DB.py:314–362`; the borrowed contexts use thread-local connections and `BEGIN IMMEDIATE`.
- Named concrete risk: adding final-drain notification must not permit new work after notification or leave old prune callers invoking callbacks under lock. Checked the previously unshown reservation block at `tldw_chatbook/Agents/execution_capacity.py:247–274` and searched `_prune_if_finished` call sites: root-finished owners reject reservation, and both callers detach under lock and invoke afterward.
- Named concrete risk: durable tuple representation must match the existing identity producer. Checked `tldw_chatbook/Agents/agent_worktree.py:149–161`; its ancestor-chain tuples match the stored four-field components.
- No tests rerun: the code-level branch counterexamples are directly apparent from the predicate, and duplicating the supplied focused runs would not answer a new question. No source, Git, or dependency mutations performed; only this requested scratch report was written.

### Assessment

**Task quality:** Needs fixes.

**Reasoning:** The ownership repository and physical-drain callback implementation are cohesive and respect the required transaction and lock boundaries. Complete the explicitly required branch validation before approving the slice.


### Finding Verdicts

- **Persisted Git branch validation accepts invalid branch shapes** — ADDRESSED. `tldw_chatbook/DB/agent_worktrees.py:74-92` now rejects a leading hyphen, lone `@`, and every dot-prefixed or dot-suffixed slash component; `Tests/DB/test_agent_worktree_recovery.py:118-128` exercises the cited `agent/.hidden` and `-agent` counterexamples and verifies rejection leaves no durable row.
- **The purported v19 runtime migration test did not start from schema 19** — ADDRESSED. `Tests/DB/test_agent_worktree_recovery.py:349-385` now removes the only v20 schema object and v20 audit row, asserts the predecessor is version 19, reopens through `AgentRunsDB`, and verifies both the version-20 worktree table and the saved v19 definition wall cap. The original reopen test was renamed at `Tests/DB/test_agent_worktree_recovery.py:50` to match its actual coverage.

### New Breakage in the Fix Diff

None.

### Out-of-Scope Observations

None.

### Checks

- Confirmed the supplied focused final evidence names the amended branch and runtime-migration tests and records `5 passed`, exit 0, with only the inherited `RequestsDependencyWarning`.
- No tests rerun: the supplied focused evidence covers both fixes, and the diff raised no unanswered behavior doubt.

### Verdict

**Fix round:** All findings addressed, no new Critical/Important breakage.


## Rulings made during restoration

Listed in plan execution order; later-plan rulings describe the remaining approved implementation contracts, not completed features.

### 2026-09-12-agent-worktree-creation-restoration

Ruling: Restore ordinary local Git with exact selected writable authority and command-boundary identity checks; supersede the blanket backend qualification blocker — this implements the user-approved scope — concurrent external replacement during Git remains an explicitly documented risk and is not claimed prevented.
Ruling: Keep automatic and failed-start deletion disabled; leave confirmation/merge closures unavailable until subsequent ownership/UI slices — preserves unfinished work while allowing useful isolated creation — interim users still need manual recovery for newly created work until the next slices land.
Ruling: Reuse the existing verification interpreter and copy only the stdlib test runner into this plan workspace — keeps the established test isolation and unique basetemp — this does not certify or modify installed dependencies.
Ruling: Preserve the disclosed missing pre-implementation RED for the new real-service test; do not rewind source to manufacture chronology — actual final behavior and review evidence remain useful — this slice did not follow the promised TDD order and cannot claim it did.
Ruling: Author creation review fixes only in disjoint test files while the independent DB/capacity slice is implemented; serialize validation at a stable-source checkpoint — avoids duplicate source work and idle time — shared-tree test imports still require that coordination.

### 2026-09-12-agent-worktree-records-and-drain

Ruling: Preserve in-flight mutation state across opening another DB handle; treat it as unavailable for replay instead of resetting it in every constructor — another handle can open while the original operation is live — post-crash in-flight work needs manual inspection and remains non-actionable.
Ruling: Creation records need complete base/identities before child execution; a crash during Git creation can leave unrecorded retained material — preserves the existing no-adoption/no-deletion policy without speculative metadata — those rare partial checkouts remain manual-only.

### 2026-09-12-agent-worktree-ownership-integration

Ruling: Keep callback errors conservative and preserve held records — a lost positive persistence result must never authorize recovery — cost is manual recovery after a DB failure.
Ruling: Preserve all scratch evidence and reports through workstream completion — earlier durable review already established this retention policy — cost is local scratch storage.

### 2026-09-12-agent-worktree-confirmed-operations

Ruling: Release an exact claim to unresolved only when absence of destination effects is positively verified — ordinary conflicts should permit a fresh user-confirmed attempt without pretending an ambiguous effect was absent — cost is replay risk if the no-effect proof is implemented incorrectly, tested through conflict/persistence cases.
Ruling: Preserve existing source capture commit after an oversized patch and describe it accurately — no destination bytes changed and original base remains known — cost is an extra local child commit.
Ruling: Keep historical blanket-qualification claims superseded — implement ordinary Git with command-boundary checks and explicit concurrent-replacement limit — cost is no atomic guarantee against a hostile concurrent metadata replacement.

### 2026-09-12-agent-worktree-console-recovery

Ruling: Treat card/list as an Operate-mode local Console extension — user already approved the concrete restoration and recovery design — cost is no separate visual concept selection; incumbent tokens/cards are authoritative.
Ruling: Do not expand work into Impeccable metadata repair or a design-world exercise — context reports stale sidecar/deprecated Register/unset buildPath but these do not affect native confirmation behavior — cost is existing design metadata drift remains for a separate requested cleanup.
Ruling: Reuse the task-scoped independent reviewer for both code and provided native screenshot evidence — one review seat avoids duplicate audits while retaining the visual review requirement — cost is the review packet must explicitly contain all viewport captures and native craft floor.


## Ownership call-site integration gate

Production8efca6ee05; test improvements c8e1d119b8 and eed41ac5d2. Final14 targeted tests passed with the inherited Requests warning; review-fix selections3 and2 passed. Root inspected exact output and zero added diagnostic/format failures. The real child stores ownership before its first write; delayed physical workers remain held after logical timeout; cleanup uncertainty survives reopen; callback-owned connections close while borrowed connections remain usable; failed insertion preserves work and prevents execution.

### task-1-review.md

### Spec Compliance

- ✅ Spec compliant for the creation/drain integration. `agent_service.py:4158` binds the supplied child owner before creation; `agent_service.py:4197`–`4242` captures matching source/child Git common-directory structure and inserts exact source authority, child identity, branch/base and execution ID before routing at `agent_service.py:4302`.
- ✅ Physical completion is the persistence trigger (`agent_service.py:4254`–`4270`), and cleanup uncertainty marks the captured owner before attempting persistence (`agent_service.py:4272`). Provider observer delivery precedes refusal translation (`local_tool_provider.py:1899`). No recovery UI or mutation was added.
- ✅ Failed ownership insertion returns refusal without routing, retaining the checkout (`agent_service.py:4190`, `4303`). Existing failed-launch finally completes the exact owner (`agent_service.py:5536`–`5546`); callback registration occurs only after insertion.

### Strengths

- Full structural record fields are supplied explicitly, with fixed `rev-parse --git-common-dir` queries and canonical relative-path handling (`agent_worktree.py:167`–`179`). This fits the approved ordinary Git command-boundary contract.
- Callback connection handling preserves an existing per-thread connection and closes one opened by the callback (`agent_service.py:4244`–`4252`). This matches `AgentRuns_DB.py:324`–`361`, where transactions retain the current thread's handle.
- Cleanup failure remains conservative even when persistence raises: owner marking precedes the DB call; repository state transitions cannot upgrade uncertain to drained (`agent_service.py:4272`–`4286`; `DB/agent_worktrees.py:258`–`276`). Observer exceptions preserve the original refusal (`local_tool_provider.py:1905`–`1911`).
- Real temporary Git and reopened SQLite tests cover held ownership, exact owner binding, real child execution before first write, physical timeout completion and durable uncertainty (`Tests/Agents/test_fleet_runtime.py:4160`, `4293`, `4403`, `4469`).

### Issues

#### Critical (Must Fix)

- None.

#### Important (Should Fix)

- None.

#### Minor (Nice to Have)

- `Tests/Agents/test_fleet_runtime.py:4456` waits for `coordinator.all_finished` and immediately asserts durable drain at line 4459. That predicate can become true at `agent_service.py:5421`, before `run_child_owned` calls `child_owner.finish_root` at line 5488. The test can therefore fail under a legitimate scheduling interleaving despite correct production behavior. Join the child before asserting drain, or wait for the durable drained predicate. Static ordering establishes the race; no stress loop is needed.
- `Tests/Agents/test_fleet_runtime.py:4621` and `4669`: existing provider-admission/thread-start failure tests now exercise durable admission but still assert only retained Git and absent routing. Adding reopened durable-row assertions would directly guard the new failed-start ownership contract; the production finally/callback ordering is correct by inspection.
- `.superpowers/sdd/2026-09-12-agent-worktree-ownership-integration/final-evidence/final-targeted-3/stdout.txt:4`: reported verification includes a RequestsDependencyWarning from the existing environment. This is pre-existing dependency noise, not a regression from this patch, but the output is not pristine. No dependency changes are requested within this task.

### Checks

- Read task brief, report, Global Constraints, and the provided diff package once; recovered a tool-truncated middle section without re-reading production hunks. No Git commands or test suites run.
- Named lifetime risk: whether callback persistence closes borrowed DB handles or nests transactions unsafely. Checked `DB/AgentRuns_DB.py:290`–`361` and `DB/agent_worktrees.py:247`–`277`; borrowed handles survive, and persistence failures are contained conservatively.
- Named sticky-state risk: whether drain callbacks run before owned operations finish or can upgrade uncertainty. Checked `Agents/execution_capacity.py:277`–`331`; callbacks follow root plus operation drain outside the capacity lock, with sticky outcome.
- Named failed-start/synchronization risk: the diff cuts off admission caller teardown. Checked `agent_service.py:5250`–`5546` to establish exact owner propagation, failed-launch finish and fleet-terminal-before-drain ordering. Inspected existing failed-start test bodies because their diff hunks omit assertions.
- Read stored final targeted output: 14 passed, one pre-existing Requests dependency warning, no stderr. Root independently verified the static baseline comparison; not duplicated here.

### Assessment

**Task quality:** Approved with minor test improvements.

**Reasoning:** The integration records ownership before exposure and connects durable release to the exact child execution's physical lifetime. No production correctness blocker was found; the new gated-child test should synchronize on physical completion instead of the earlier fleet terminal transition.

### task-1-rereview-1.md

### Finding Verdicts

- **The gated real-child regression can assert durable drain before physical child completion** — ADDRESSED. `Tests/Agents/test_fleet_runtime.py:4456` now calls `join_fleet_children(service)` before the durable `writer_state == "drained"` assertion at line 4461; the intervening `coordinator.all_finished()` check occurs only after the child thread has joined.
- **Failed provider-admission and thread-start tests do not reopen SQLite and verify retained durable ownership** — NOT ADDRESSED. Both tests now reopen SQLite and exactly compare `base_sha`, `binding_id`, and `writer_state` (`Tests/Agents/test_fleet_runtime.py:4665-4674`, `Tests/Agents/test_fleet_runtime.py:4730-4739`), but each checks only that `record["execution_id"]` is truthy at lines 4673 and 4738. Neither captures the failed child owner's expected execution ID and compares equality, so a wrong nonempty owner ID would pass and the requested exact execution binding remains unproved.

### New Breakage in the Fix Diff

None.

### Out-of-Scope Observations

None.

### Verdict

**Fix round:** Findings remain open — the failed provider-admission and `Thread.start` regressions must compare the reopened row's `execution_id` with the exact captured child `ExecutionOwner.execution_id`, rather than asserting only nonemptiness.

### task-1-rereview-2.md

### Finding Verdicts

- **Failed provider-admission and thread-start tests do not reopen SQLite and verify retained durable ownership** — ADDRESSED. Each regression now wraps the real `AgentService._admit_agent_worktree`, captures the exact `ExecutionOwner` supplied for the child, asserts exactly one owner was admitted, and compares the independently reopened durable row's `execution_id` with `admitted_owners[0].execution_id` (`Tests/Agents/test_fleet_runtime.py:4652-4659,4678-4682` and `Tests/Agents/test_fleet_runtime.py:4724-4731,4751-4755`). The wrapper forwards the original handle, child run ID, and owner unchanged, so it observes rather than substitutes the admission path. The reopened lookup remains keyed by the sub-agent run row and conversation, while the production admission records that same `child_run_id` with the supplied owner's exact execution ID.

### New Breakage in the Fix Diff

None.

### Out-of-Scope Observations

None.

### Verdict

**Fix round:** Approved — all scoped findings are addressed, and the exact retained owner equality is now proved in both failed-start regressions.


### Additional ownership rulings

Ruling: Include callback connection-ownership proof before review — the last tool callback can open a DB connection on a thread that then exits — cost is small callback wrapper bookkeeping; tests require newly opened connections close and borrowed ones remain usable.
Ruling: Start confirmed-operation implementation while read-only re-review finishes the committed minor test fixes — no production blocker or source-writing overlap remains — cost is possible focused test rework if the small fix has a defect.
Ruling: Let disjoint engine work continue while the small failed-start test-only correction is made — no production overlap and the shared test file has one explicit owner — cost is serialization of that file's later operation tests.


## Confirmed-operation implementation gate

Commit720bc15f92; independent review pending. Root inspected the final 91-case combined gate, four additional CAS/hooks cases and three final bounded administrative-read cases, plus zero added scoped diagnostics and passing edited formatting. These overlapping selections are not summed.

# Task 1 implementation report

Status: implemented and ready for independent review. Source left unstaged. TASK-31210 and TASK-31211 are not Done: the visible Console card, call-site parity, retained recovery list and manual operation lifetime remain the next slice.

Initial source base: c8e1d119b8. Static/review comparison base: eed41ac5d2 (excludes the separately reviewed exact-owner fixture assertion follow-up). Current root commits during this work were governance-only. No Git/Backlog/governance mutations performed by this worker.

## Delivered interface

`tldw_chatbook/Agents/agent_worktree_recovery.py` exports the requested unchanged interface:

```python
@dataclass(frozen=True)
class WorktreeRecoveryOutcome:
    action: str
    message: str
    state: str
    commit_sha: str | None = None

def recover_agent_worktree(db, *, authority, conversation_id, run_id, action,
                          request_confirmation, should_cancel): ...
```

The caller retains and activates its ExecutionOwner and invokes the synchronous function on its worker thread. The function borrows AgentRunsDB and never closes the caller connection. Exact `allow is True` is required. Payload includes run_id/action/branch/worktree/source/destination/diffstat; discard adds retains_checkout=True and retained_baseline text. Same-turn adapters add handle_id.

Metadata eligibility precedes confirmation: exact conversation, terminal status, durable positive writer drain, unresolved claim state, selected writable authority, workspace/binding/fingerprint/root and complete identity chains, common Git directory, standard linked checkout and exact original branch/base ancestry. A second snapshot after Allow refuses observed drift. A generated transactional operation ID prevents competing mutation claims.

Apply captures dirty/new source work using a fixed agent identity, streams a binary original-base patch to a capped owned temporary file, checks it before applying, and lands unstaged changes without replacing unrelated parent index state. Merge requires a clean destination with no existing merge/rebase/cherry-pick operation, disables hooks, makes an explicit no-ff commit, and aborts only a MERGE_HEAD demonstrably belonging to its source. Positive HEAD/status/operation restoration permits a no-effect refusal. Clean work refuses instead of issuing a false merge receipt.

Logical discard inventories special/nested entries before mutation, deletes descendants through no-follow directory handles, restores a detached original-base checkout, and deletes only the exact recorded branch with expected-old-SHA CAS. Root and administrative link remain. CAS failure after cleanup leaves uncertain state and preserves the changed ref. No forced root removal is used in confirmed recovery.

Known-no-destination-effect failures can release the exact claim to unresolved; uncertain effects and failed receipt persistence remain protected. A disclosed source-only capture commit may survive an oversized-patch refusal. Source/destination identities and live authority are checked at operation boundaries. Ordinary local Git retains the ADR-155 documented external replacement/config/filter limitations.

## Files and implementation choices

- New `Agents/agent_worktree_recovery.py`: shared validation, snapshots, consent, claims, actions and receipts.
- New private `Agents/agent_worktree_git.py`: bounded Git stdout/stderr, fixed host environment, disabled generated hooks/signing, timeout and owned process-group retirement; unproven cleanup marks the active ExecutionOwner before returning.
- `Agents/agent_service.py`: exact current-turn handle/run mapping, frozen selected authority, real confirmation callback, and identical fleet/primary/callable runtime/schema gate.
- `Agents/tool_catalog.py`: current-turn handles, older Console recovery and retained baseline descriptions.
- `DB/agent_worktrees.py`: exact-owner unresolved completion with caller-proven no-destination-effect contract.
- New real Git/SQLite engine tests; actual run_turn apply/discard tests that join physical children and assert durable writer drain; updated Console disclosure and missing-card/authority assertions.
- Existing lifecycle tests now place created children under their own temporary fixture directory.

Deviation: existing legacy helpers in `agent_worktree.py` remain unchanged. Confirmed production operations exclusively use the new shared engine/private bounded helper; legacy helper deletion is still used by older fixture cleanup and remains unreachable from restored service operations. No automatic cleanup was enabled.

## Limits

- Preview text: 8192 characters. Git output and streamed binary patch: 32 MiB each. Changed/untracked snapshot content: 32 MiB per inspected tree; entry inventory: 10,000. Oversized or special entries refuse explicitly rather than silently truncating the fingerprint.
- Unchanged baseline file bytes are not scanned into snapshot content limits. Tracked HEAD/index/diff evidence plus no-follow changed/untracked bytes and discard ignored entries detect content drift. Ignored discard entries appear in preview.
- Read snapshots require actual no-follow descriptor read primitives. Discard separately requires POSIX descriptor unlink/rmdir. A missing discard-only primitive does not disable apply (tested); platforms without safe snapshot reads refuse recovery. Creation is untouched. Evidence is macOS POSIX, not a Windows qualification claim.
- Process limits are 30 seconds plus bounded cleanup waits. A residual background filter process is killed even after its Git leader and pipes complete. If process-group retirement cannot be proven, the owner is marked cleanup-unproven and recovery refuses conservatively.
- No public read-only Git allowlist, dependency, network/provider path, live config, or user repository was changed.

## Evidence

All runs used the supplied unique-label runner and existing integration interpreter. Exact commands, isolated basetemp paths, stdout/stderr and exit status are in `final-evidence/<label>/`.

- `engine-red-1`: 19 expected failures before engine/DB implementation.
- `engine-green-1`: 19 passed, establishing real apply/merge/discard.
- `adapters-red-1`: 5 expected failures for disclosure/missing authority and large unchanged baseline.
- `service-integration-1`: 2 passed; actual run_turn, real child fs_write and durable physical-drain assertions.
- `engine-neighbors-2`: 60 passed across engine, runtime, Console and DB.
- `hardening-red-1`: three expected failures (clean merge receipt, ignored discard preview, administrative symlink), one oversized-patch behavior pass.
- `hardening-green-1`: 27 passed.
- `process-capabilities-red`: capability failure plus a test harness `ps` permission error. The process check was corrected to a temporary child heartbeat; no permission escalation or process census was attempted.
- `process-red-corrected`: expected failure proving the old helper returned while a filter descendant was still writing its owned heartbeat.
- `final-targeted-1`: **91 passed, 1 inherited warning, 37.52s**. Includes new engine, existing lifecycle, dispatch, Console confirmation, selected actual service nodes and all directly affected DB recovery tests.
- `final-cas-hooks`: **4 passed, 1 inherited warning, 3.97s**. Exact-ref mutation race, pre-existing merge preservation, and actual apply/merge with refusing repository hooks prove hooks are disabled.
- `bounded-admin-final`: **3 passed, 1 inherited warning, 2.88s**; final focused recheck for capped no-follow administrative link reads and unaffected apply/capability behavior.

The inherited warning is RequestsDependencyWarning for urllib3/chardet/charset_normalizer versions in the existing interpreter. No new dependencies were installed and no full suite was run.

`static-final.json` records the supplied scoped utility comparison: no added Ruff diagnostic identities and all edited hunk/new-file formatting passes. Existing diagnostic counts remain unchanged (agent_service 51, tool_catalog 9, lifecycle test 1, fleet test 13, Console confirmation test 5; new files and DB recovery repository zero). `git diff --check` passes.

## Review priorities and next integration

Review guarded mutation/no-effect classifications, no-follow discard and exact-ref CAS, process cleanup owner marking, same-turn handle ownership, and capability separation. Console should call the public function with a freshly admitted selected authority and its independently retained/activated manual ExecutionOwner, preserving the exact confirmation payload and cancellation signal. Do not reuse a finished primary turn's cancellation lifetime or revive old fleet handles. Return the retained baseline message unchanged enough for the user to understand cleanup residue.

Final-source evidence note: final-targeted-1 covers the complete runtime implementation before only the bounded administrative-link read and its early read-capability gate were tightened. bounded-admin-final covers that last source change. final-cas-hooks covers the four subsequently added requirement cases; the only following test edit was explicit check=False and formatting. static-final.json was regenerated after all final source/test edits. No pending test processes remain.


### Additional rulings: 2026-09-12-agent-worktree-confirmed-operations

Ruling: Keep the confirmed lifecycle implementation in the new recovery and private Git modules, leaving the legacy agent_worktree.py merge/preview/discard helpers unchanged — a focused caller search found no confirmed production caller of those legacy functions; existing fixture cleanup and legacy helper tests still use them — cost is retaining unused legacy product helpers until a separately justified cleanup. The planned strengthening applies to the confirmed production path rather than requiring a mechanical edit to every listed file.

Ruling: Separate read-snapshot and discard-delete capability checks — missing unlink/rmdir must not disable apply/merge — cost is explicit refusal on platforms lacking the actual no-follow read primitives, with ordinary creation still available. This is a narrow supported-operation check, not renewed backend qualification.


### Additional rulings: 2026-09-12-agent-worktree-console-recovery

Ruling: Stamp each decision control with its immutable round ID and test an old queued button press — a mutable card field alone can relabel the old press as a new approval — cost is rebuilding small action controls when the round changes.


## Diagnostic inventory review before Console integration

Read-only inventory comparison at720bc15f92 against the last pin commit7852cf47ba identified exactly five added calls: execution_capacity1, agent_service3 and local_tool_provider1. The statement-level command showed only fixed messages plus exception type names; no exception bodies, paths, locators, provider content or secrets were added. No sink topology changed. Root inspected every statement. The pin is intentionally not regenerated yet; repeat the delta check after the Console slice and write the combined reviewed inventory once.


### Confirmed-operation independent review at720bc15f92

### Spec Compliance

- ❌ Issues found: `tldw_chatbook/Agents/agent_worktree_recovery.py:517-519` can persist a successful explicit-merge receipt when Git created no commit. The original-base change guard at lines 391-406 does not exclude a child already contained in destination history.
- ✅ The requested public outcome and worker interface are present (`agent_worktree_recovery.py:27-34,302-311`); exact consent and post-consent snapshot comparison precede claim (`:373-415`). Ownership, terminal state, positive drain and unresolved state are checked before confirmation (`:42-68,326-338`).
- ✅ The legacy-helper deviation and read/delete capability split comply with the explicit progress rulings. The Console card/manual lifetime remain the later slice and are not missing requirements of this engine review.
- ⚠️ Physical ownership activation by every future Console caller cannot be established from this engine diff; that integration must retain its independently activated ExecutionOwner as specified. This is a later-slice check, not a blocker here.

### Strengths

- `agent_worktree_recovery.py:458-475` streams an original-base binary patch to an owned capped spool, checks it first, then applies without replacing the destination index. The real Git test `Tests/Agents/test_agent_worktree_confirmed_recovery.py:116` covers advanced parent history, unrelated staged content and binary/newline-name child additions.
- `agent_worktree_recovery.py:500-515` limits merge abort to a matching source MERGE_HEAD and requires restored HEAD, clean status and no remaining operation before releasing a conflict claim. `:535-564` preserves uncertainty or in-flight protection when effect/persistence cannot be established.
- `agent_worktree_recovery.py:141-222,427-440` uses descriptor-relative no-follow deletion, retains root and administrative link, restores a detached baseline and deletes the exact ref with expected-old-SHA CAS. The new tests exercise external symlink target preservation and CAS failure after cleanup.
- `agent_worktree_git.py:91-145` checks the process group even after leader/pipes finish and marks cleanup-unproven when retirement cannot be established. The new heartbeat test measures a real filter descendant rather than equating logical completion with physical drain.
- `agent_service.py:6160-6206` requires current-turn handle membership and agreement with its created run, passes the conversation and selected authority into the shared engine, and preserves refusal codes. `:4816-4820,7675-7681` aligns callback exposure with the real-confirmation/fleet/primary predicate.

### Issues

#### Critical (Must Fix)

- None identified.

#### Important (Should Fix)

- **Incorrect successful merge receipt for an already incorporated child** — `tldw_chatbook/Agents/agent_worktree_recovery.py:517-519` (triggering guard at `:391-406`). A child can have a real commit beyond its recorded base while that commit is already an ancestor of the destination HEAD, for example after the user manually merges or fast-forwards the retained branch. The original-base diff is nonempty, so recovery proceeds. `git merge --no-ff <child-head>` then exits successfully with “Already up to date”; it does not create a commit. The engine nevertheless stores `merged`, returns the unchanged destination SHA and claims an explicit merge commit. This violates the requested honest receipt and explicit merge behavior. Detect this ancestry/no-op before mutation and return a specific no-effect refusal, and positively verify the successful merge actually produced the expected new two-parent commit before persisting `merged`. Add a real Git regression with a child commit already included in the destination, asserting unchanged destination HEAD and no false merged receipt.

#### Minor (Nice to Have)

- None identified.

### Assessment

**Task quality:** Needs fixes.

**Reasoning:** The authority, consent, claim, discard and physical-cleanup boundaries are thoughtfully implemented and backed by focused real Git/SQLite evidence. The remaining merge no-op path can issue a durable receipt for an operation that never happened and needs correction before approving this task.

### Review checks and scope

- Reviewed immutable package `review-d358eb3de8..720bc15f92.diff`, base `d358eb3de8`, head `720bc15f92`; the initial tool output truncated its middle, so the omitted ranges were retrieved before completing the review. Line references were derived from that package; changed source files were not independently reread and no Git commands were rerun.
- Read task brief, implementation report, progress rulings, restoration spec and amended ADR-155. No outside-diff product-code checks were necessary; no broader codebase crawl occurred.
- No suites or tests were rerun. Controller-inspected evidence is 91 + 4 + 3 passing targeted cases, no added static diagnostic identities and passing edited formatting/diff checks. The disclosed inherited RequestsDependencyWarning is nonblocking under the controller's ruling; no dependency change is requested.
- Source, index and branch remain unchanged. Only this requested review report was written.


### Confirmed-operation fix and approval

 — explicit merge receipt (base 720bc15f92)

Addressed the Important finding in task-1-review.md. A committed child already included in destination history previously passed the original-base difference check, and Git's successful “Already up to date” response was incorrectly persisted as a new explicit merge.

Changes are limited to `Agents/agent_worktree_recovery.py` and `Tests/Agents/test_agent_worktree_confirmed_recovery.py`:

- A clean child whose HEAD is already the destination's ancestor now returns `already_merged` before the mutation claim. The destination and source commits remain unchanged and the record stays unresolved.
- An additional ancestry check immediately before destination mutation handles the post-source-capture path. Its confirmed no-destination-effect refusal can release the exact claim to unresolved.
- A successful Git command must now produce a new commit with exactly the previous destination SHA as first parent and the confirmed child SHA as second parent. Unchanged HEAD, wrong parents, or failed verification cannot persist `merged`; after possible effects these outcomes stay uncertain and cannot replay after reopen.
- Dirty new work on an already incorporated child remains mergeable and gets the expected two-parent commit.

Evidence, using the same isolated runner/interpreter:

- `final-evidence/fix1-merge-red`: **5 expected failures**, 6.49s, before production edits. Real Git cases cover equal destination/child HEAD and an advanced destination ancestor; injected command-boundary cases reproduce an unchanged successful Git response, actual post-merge external advancement and unavailable verification.
- `final-evidence/fix1-merge-green`: **14 passed**, 15.75s. Covers all new cases plus ordinary explicit merge, dirty incorporated child, clean no-change refusal, apply/merge conflicts, pre-existing merge preservation, disabled hooks and failed-persistence protection. This evidence covers the final fix source/tests; no code edits followed it.
- `static-fix1.json`: comparison against **720bc15f92**, zero added Ruff identities and passing edited-range formatting for both scoped files.
- `git diff --check`: passes.

Both pytest runs retain the same single RequestsDependencyWarning; stderr is empty. No whole-suite run, dependencies, live configuration, network/provider use, Git mutation or Backlog/governance edit. Source remains unstaged for re-review. The broader initial 91-test gate was not repeated because the fix is confined to merge ancestry/receipt classification and directly affected focused cases passed.


### Finding Verdicts

- **Incorrect successful merge receipt for an already incorporated child** — ADDRESSED. `tldw_chatbook/Agents/agent_worktree_recovery.py:407-416` refuses a clean child already contained by destination history before a mutation claim, and `:500-508` repeats the ancestry check after source capture at the destination boundary. `:530-545` requires a changed HEAD whose exact parents are the previous destination and confirmed child commits before `merged` can be persisted. `Tests/Agents/test_agent_worktree_confirmed_recovery.py:464-548` covers equal/advanced incorporated history, unchanged successful Git output, wrong parents, unavailable verification, reopen protection, and dirty work on an incorporated child.

### New Breakage in the Fix Diff

None.

### Out-of-Scope Observations

None.

### Review Checks

- Reviewed immutable package `review-720bc15f92..fc9c01473e.diff` once, base `720bc15f92`, head `fc9c01473e`; no Git commands or source rereads were needed.
- The fix report records 14 passing focused cases, zero added scoped static diagnostic identities, passing edited-range formatting, and passing `git diff --check`. The inherited Requests dependency warning remains nonblocking under the controller ruling. No tests were rerun.

### Verdict

**Fix round:** All findings addressed, no new Critical/Important breakage.


### Additional rulings: 2026-09-12-agent-worktree-console-recovery

Ruling: Begin Console implementation while the committed narrow merge-receipt fix receives read-only rereview — the recovery API is unchanged, no other source writer remains, and the two file scopes are separate — cost is possible focused caller adjustment if rereview finds a contract defect.


## Console implementation gate

Committedc5edf2741b; independent review pending. Root verified raw final-functional39passing cases, final-lifetime12, and authority-and-startup31 (overlapping selections, not summed); all exit0/stderr0. Root verified the18-file scoped static comparison and staged diff-check.

# Task 1 implementation report — Console agent-worktree confirmation and recovery

Source/static base: `fc9c01473e`. Work was confined to the existing isolated `agent-orchestration-pr` worktree. Source/tests remain unstaged. Root owns Git, Backlog, ADR/spec/plan edits and diagnostic inventory. No commits, verification checkout, dependency installation, live provider/config, network, or foreign cleanup were performed.

## Implemented behavior and public contracts

- `WorktreeConfirmCard` is lazy, hidden at construction, and renders plain text for the action, separately labeled source/destination, bounded 8192-character scrollable diffstat, and retained-checkout consequence. Control/format characters are escaped. Each Allow once/Deny button carries its own immutable request ID. Old queued `Button.Pressed` instances cannot authorize replacement rounds; answered controls remain disabled under the same payload. The controller now makes the first exact decision immutable under its round lock.
- `TaskResumeState.pending_worktree_merge` is volatile. It participates in lazy card routing/visibility but is absent from `to_dict` and not restored by `from_dict`. No automatic worktree locators or confirmation source bodies are added to durable resume metadata.
- `ConsoleRuntime` has the real disposable `set_pending_worktree_merge` hook slot. Worktree rounds remount independently before unified-approval early returns. A failed disposable worktree projection does not abort the other pending-decision projection. Existing accepted rounds survive session switching/navigation/detachment and reappear with their exact IDs.
- `ConsoleChatController.worktree_confirmation_enabled` requires an actual worktree hook and app. Both bridge preview entry points accept `worktree_merge_enabled=False`; both controller preview calls and live call admission use the same actual-surface gate. Generic retained unified-approval support alone does not advertise worktree tools. Already-armed rounds retain the existing park/remount host path.
- `request_worktree_merge_confirm(..., operation_cancel_event=None)` accepts an explicitly supplied manual Event. Supplying a manual Event requires an explicit owning session; it does not borrow the primary turn's Stop Event. Model-driven calls retain their existing cancellation behavior.
- `ConsoleRuntime.worktree_recovery` lazily owns `ConsoleWorktreeRecovery`. Its `list_work(session_id, after_run_id=None)` returns a typed `RecoveryPage`, and `start(session_id, run_id, action)` returns the shared outcome/refusal. `cancel_session`, `begin_close`, and async `close` own recovery cancellation/drain. The synchronous runtime disposal fence closes manual admission before async teardown.
- `RecoveryIntent` contains only session ID, workspace ID, persisted conversation ID, ephemeral flag and immutable project-instruction state. Capture is pure UI-thread memory work. Binding/root/filesystem/DB validation runs in the worker and marshals only fresh pure session capture back to the UI. It requires an explicit selected writable named binding, exact current locator/root identity, and a fresh clear kill switch. No viewed-session or arbitrary sole-binding fallback supplies authority.
- A manual operation allocates and activates an existing `ExecutionOwner` from the bridge's public `runtime_capacity`, and calls the unchanged synchronous shared Git engine using its own worker-opened `AgentRunsDB` handle. The physical worker's `finally` closes that handle and finishes the root. Shielding and the retained task registry preserve the owner after cancellation of a disposable asyncio waiter. Borrowed bridge DB handles are not closed by the helper. No old fleet is reconstituted and no second capacity ledger is introduced.
- Listing uses only `AgentWorktreeRepository.list_for_conversation` metadata, fetching 51 rows to display 50 and an exact next cursor. Rows describe held writers as unconfirmed completion, ambiguous/in-flight results as needing manual review, and applied/merged/discarded-with-baseline states accurately. Nonactionable rows have disabled actions.
- `Console: Recover agent work…` opens the paged modal for the exact current conversation/repository. A row action dismisses the list and launches the existing inline Allow/Deny flow; selection itself never approves. Receipts are bounded to 32 and keyed by their owning conversation. A live toast additionally checks the original session, conversation and current attached view; reopening the owning conversation's list can show its retained receipt.
- ChatScreen contains only three thin additions: the lazy projection hook, palette delegation, and exact decision forwarding. No new keybinding or architecture budget was added.

## Root-approved import refinement

`guards-one` exposed UI-ready 976/973. Root traced three modules to prior work in this same restoration stream and explicitly assigned the narrow fix:

1. Move the **unchanged** `AGENT_WORKTREES_SCHEMA` literal from the metadata repository module to `AgentRuns_DB`, removing the eager repository-module import. Schema behavior/version is unchanged.
2. Move `ConsoleRunLogModal`'s import into the existing modal publication handler, deferring the modal and `run_log_paging` until actual use.

`authority-and-startup` then passed the real worktree migration/reopen tests, actual mounted paged-log UI tests, and UI-ready census at **973/973**. No unrelated imports were shed and the cap/snapshot were not changed.

## Real RED and GREEN evidence

All commands used the supplied `run_pytest.py LABEL Tests/...` wrapper and its existing isolated Python 3.12 interpreter. Every label below has exact argv/cwd in `final-evidence/LABEL/command.json`, stdout/stderr, and exit code files. Product imports occurred only under Tests/conftest isolation.

| Label | Result and meaning |
| --- | --- |
| `card-red` | 2 failed: missing visible worktree card/state and no restored volatile field. |
| `card-green` | 1 failed/1 passed: test-only incorrect private Static introspection; changed to assertions on actual rendered literal markup/control text. |
| `lifecycle-red` | 1 failed: controller lacked explicit independent operation Event parameter. |
| `physical-red` | 1 failed: retained manual helper did not exist. |
| `initial-green` | 18 passed: mounted immutable old-button behavior, plain text, volatile state, existing controller/bridge contracts, old primary Stop independence and physical ownership after waiter cancellation. |
| `wiring-red` | 2 failed: absent runtime hook/remount and both preview flags. |
| `dialog-red` | 1 failed: recovery presentation absent. |
| `wiring-green` | 6 passed. |
| `reopened-first` | 2 failed: **product defect** in initial implementation, whole-session deepcopy encountered a retained thread lock. Replaced it with pure `RecoveryIntent`. |
| `reopened-green` | 2 passed: real reopened SQLite record → controller → mounted Allow → actual Git Apply/Discard, retained source, duplicate refusal. |
| `duplicate-red` | 1 failed: **product defect** allowing a second resolver call to overwrite Deny. Fixed first-decision immutability under the round lock. |
| `real-gate-red` | 1 failed: shared actual-surface predicate absent; generic retained target now explicitly fails the gate test. |
| `full-flow-first` through `full-flow-six` | Harness prerequisites were corrected: durable named Workspace registry, correct project-instruction consent seam, actual runtime view-hook attachment, waiting for the background run after `submit_draft`, and a scripted provider obeying `load_tools` plus deferred project-instruction reconsideration. These are not product fallback or hidden retries. Exact failed outputs remain preserved. |
| `full-flow-seven` | 2 passed: real controller submit → real bridge → AgentService.run_turn → spawned/drained child filesystem write → visible actual Allow/Deny button → actual Git landing or preservation. Provider responses are scripted and the local catalog is a real preapproved LocalToolProvider fixture. |
| `retained-lifecycle` | 21 passed: includes session switch and detached remount retaining the exact round, and old Stop observed across a full poll interval. |
| `paging-negative` | 1 passed: 53 real SQLite records are returned exactly once in pages of 50 and 3; no transcript hydration, held refusal, and wrong-conversation refusal. |
| `session-close` | 1 passed: two physical manual workers remain capacity-owned; runtime closing A cancels only A, leaving B untouched. |
| `manual-drift` | 3 passed/2 failed: harness waited for registered round before actual button mount; changed its condition to the real mounted button. No production workaround. |
| `remount-failure-red` | 1 failed: initial projection exception could abort attach. Added the same content-free best-effort handling used by neighboring remounts. |
| `final-functional` | **39 passed**, exit 0, empty stderr. Includes mounted Apply/Discard, Deny, source changes and selection revocation during human wait; full live child/Git flow; exact pagination and lifetime contracts. |
| `dispose-fence-red` | 1 failed: initial synchronous runtime disposal fence did not yet close the already-created helper. |
| `final-lifetime` | **12 passed**, exit 0, empty stderr, after adding synchronous `begin_close` before async drain. |

Every successful functional run above has the existing `RequestsDependencyWarning` from the provided environment. This was not suppressed.

## CSS, startup, architectural and static evidence

- CSS sources were edited only in `css/components/_agentic_terminal.tcss`; the bundle was rebuilt using `css/build_css.py` with the existing Python 3.12 interpreter. The host `python3` is older and initially rejected the builder's existing PEP-604 annotation; that invocation did not build anything. No stylesheet budget or token value was changed.
- `guards-one`: **14 passed, 3 failed, 3 deselected**, exit 1. CSS token/bundle checks passed. Failures were startup 976/973 plus the existing ChatScreen size ceiling and historical task-22507 no-growth guard.
- `authority-and-startup`: **31 passed**, exit 0, empty stderr. UI-ready is **973/973, headroom 0, snapshot drift +16/-16**. It has **two warnings**: the existing Requests dependency warning and the census's intentional headroom/snapshot-drift UserWarning. No rebaseline or suppression.
- Final AST-only ChatScreen measurement: **24,410 whole-file lines / 746 direct methods**. Saved baseline: **24,389 / 743**. This slice adds **21 lines and 3 methods** after scoped formatting. Unchanged inherited ceilings: **16,966 / 563**. The existing overage and historical no-growth test remain failures; the added lines/methods are explicitly not attributed to the baseline.
- `static-final.json`: scoped comparison of **18 Python files** to `fc9c01473e` reports **zero added diagnostics and zero scoped-format failures**. All seven new Python files have zero diagnostics. Existing diagnostic counts remain unchanged: controller 188, screen 197, runtime 32, bridge 27, AgentRunsDB 10, Console agent adapter 12, task cards 2, state 1, existing confirmation tests 5. Existing files were formatted only over changed ranges; no whole-file debt cleanup.

## Native visual packet

Root performed the one batched wide/narrow inspection and one final confirming inspection. Root reports no material visual findings. Include `visual-review.md` in the independent reviewer packet.

Owned artifacts under `visual-evidence/` (SVGs and root-rendered `.svg.png` files):

- `card-120.svg`, `card-60.svg`: long wrapping source and scrollable 30-line diffstat.
- `list-120.svg`, `list-60.svg`: ready, held and uncertain rows with accessible actions/footer.
- `discard-60.svg`: narrow discard consequence visible above Allow/Deny.
- `resolved-60.svg`: scrolled resolved/discarded row accurately naming retained baseline, disabled actions.

`visuals-one`: 2 passed. `visuals-confirm`: 1 passed. Visual fixture code is in its own `Tests/UI/test_console_worktree_visuals.py` so ordinary final functional gates do not create additional captures. No more inspection rounds were performed.

## Source/test files

New:

- `tldw_chatbook/Chat/console_worktree_recovery.py`
- `tldw_chatbook/UI/Console_Modules/worktree.py`
- `tldw_chatbook/Widgets/Chat_Widgets/worktree_confirm_card.py`
- `tldw_chatbook/Widgets/Chat_Widgets/worktree_recovery_dialog.py`
- `Tests/Chat/test_console_worktree_recovery_lifetime.py`
- `Tests/UI/test_console_worktree_recovery.py`
- `Tests/UI/test_console_worktree_visuals.py`

Modified:

- `tldw_chatbook/Chat/console_agent_bridge.py`
- `tldw_chatbook/Chat/console_chat_controller.py`
- `tldw_chatbook/Chat/console_runtime.py`
- `tldw_chatbook/DB/AgentRuns_DB.py`
- `tldw_chatbook/DB/agent_worktrees.py`
- `tldw_chatbook/UI/Console_Modules/agent.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`
- `tldw_chatbook/UI/Screens/chat_screen_state.py`
- `tldw_chatbook/UI/console_command_provider.py`
- `tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py`
- `tldw_chatbook/css/components/_agentic_terminal.tcss`
- `tldw_chatbook/css/tldw_cli_modular.tcss` (generated)
- `Tests/Chat/test_console_worktree_merge_confirm.py`

## Limits and inherited issues

No full suite, whole-UI sweep, live user app/provider, or Windows qualification was run. Git tests use owned temporary repositories and real SQLite on the available macOS/POSIX platform. The shared engine's documented command-boundary root/metadata checks, POSIX logical-discard restrictions, retained baseline cleanup, held/uncertain refusal, and no replay limitations remain unchanged. Provider/capability composition is deliberately supplied by test doubles in the full-flow fixture; the actual controller, bridge, service, child filesystem executor, confirmation card and Git effects execute.

The existing ChatScreen architectural overage remains, with this slice's exact small increment disclosed above. UI-ready is at its unchanged cap with no headroom. Existing linter debt and the provided environment's dependency warning remain. Tests own and join their worker threads/tasks; the manual helper closes worker-opened SQLite handles on that same worker, and the isolated test runner processes all completed. No external fleet or user data was cleaned up. Independent review and final documentation/Backlog closeout are root-owned next steps.


# Native visual review

Round1: root rendered the four isolated fixture SVGs with macOS Quick Look, then inspected card-120.svg.png, card-60.svg.png, list-120.svg.png and list-60.svg.png using view_image. This is Textual fixture output, not a screenshot of the user's app. Source viewports120x40 and60x28; Quick Look adds square thumbnail margins, unrelated to actual UI geometry.

No material layout finding in the supplied states. Long source paths wrap at narrow width, diffstat/list have visible scrolling, decision/footer actions remain visible, and wide list distinguishes ready, completion-unconfirmed and manual-review states. Native inherited terminal tokens/control states are appropriate. The brief's protected-focus modal is an approved local extension; no web design-world rules are substituted.

Evidence gap to finish within the second/final inspection batch: narrow discard card showing retained-baseline consequence, and a list scrolled to its resolved/discarded row (outside the initial viewport). Root requested those two captures; no layout patch requested from round1.

Round2 (final): root rendered and inspected discard-60.svg.png and resolved-60.svg.png. Narrow discard shows the retained baseline consequence above visible Allow/Deny controls; resolved list row says “Discarded; baseline checkout retained.” with disabled action buttons and visible Close. No material visual findings. Two native inspection rounds complete. Functional behavior is separately verified by tests/review; screenshots alone do not establish it.


Ruling: Fix the three owned startup imports within the Console slice — the fresh UI-ready guard is976against973; move the unchanged worktree schema literal to its sole AgentRuns_DB consumer and defer the bounded-log modal import to its sole open handler, removing that modal and paging module from startup — cost is touching two earlier-slice files and one related Console adapter, with real migration/log-open/census checks required; no unrelated module shedding, copied schema or ratchet increase.


### Console independent review atc5edf2741b

### Spec Compliance

- ❌ Issues found: manual recovery is not fenced against new admission while its owning session closes. `tldw_chatbook/Chat/console_worktree_recovery.py:167` and `:234`, together with `tldw_chatbook/Chat/console_runtime.py:3955`, leave the close/drain window described below.
- ✅ The requested Console slice is otherwise represented file by file: real disposable hook and independent remount (`console_runtime.py:874`, `:3727`); matching controller preview/live predicate (`console_chat_controller.py:16633`, `:19342`, `:19518`, `:25823`); both bridge preview entry points forward the flag (`console_agent_bridge.py:5283`, `:5395`); volatile state and lazy card routing (`chat_screen_state.py:41`, `chat_task_cards.py:156`); exact decision controls (`worktree_confirm_card.py:26`, `:105`); retained helper, paged picker, palette entry and thin screen delegation (`console_worktree_recovery.py:117`, `worktree_recovery_dialog.py:44`, `UI/Console_Modules/worktree.py:14`, `UI/console_command_provider.py:25`, `chat_screen.py:23441`). Paths without a prefix in this report are under `tldw_chatbook` in their corresponding package.
- ✅ Root's explicit startup refinement is confined to moving the unchanged DDL literal into its existing DB consumer (`DB/AgentRuns_DB.py:40`) and deferring the existing log-modal import to its publication handler (`UI/Console_Modules/agent.py:1457`). No schema increment or architecture/import cap change appears in the diff.
- ⚠️ Missing real boundary evidence: the preview test at `Tests/Chat/test_console_worktree_recovery_lifetime.py:84` checks method signatures/defaults only. Neither that test nor the added live flow compares the actual preview and live tool schemas with the surface present and absent. The code forwards the same predicate, but the brief's explicit schema-equality requirement is not established by these tests.
- ⚠️ The real reopened-DB/card/Git test invokes `helper.start` directly (`Tests/UI/test_console_worktree_recovery.py:128`); the visual tests open a prebuilt picker. No added mounted test exercises the actual palette → asynchronous list → picker selection → inline confirmation adapter boundary, nor its stale-view/receipt delivery behavior. These are coverage limits, not evidence that the adapter is generally broken.
- ⚠️ Shared engine claim/effect/reopen/platform behavior belongs to the independently reviewed prior slice. This diff consumes it; it does not independently establish all engine guarantees. Broad restoration integration remains root-owned.

### Strengths

- Immutable button-owned request IDs and the controller's locked first-decision write prevent a queued old button from authorizing a replacement round (`Widgets/Chat_Widgets/worktree_confirm_card.py:26`, `:105`; `Chat/console_chat_controller.py:16799`). The mounted test actually posts an old `Button.Pressed`, rather than merely constructing a stale decision message (`Tests/UI/test_console_worktree_recovery.py:39`).
- Plain rendering escapes control/format characters and disables Rich markup; confirmation is hidden at construction, lazy, and excluded from durable resume state (`worktree_confirm_card.py:15`, `:36`, `:42`; `chat_screen_state.py:41`; `Tests/UI/test_console_worktree_recovery.py:78`). Source/destination are separately labeled and diffstat is bounded.
- Physical ownership sits with the worker, not the disposable waiter: worker-created DB handles close on that worker, `owner.finish_root()` follows worker completion, and `asyncio.shield` prevents waiter cancellation from prematurely dropping capacity (`Chat/console_worktree_recovery.py:187`, `:209`, `:232`). The deterministic held-worker test verifies this boundary (`Tests/Chat/test_console_worktree_recovery_lifetime.py:14`). The focused unchanged `AgentRunsDB.close` check confirmed it only closes the calling thread's held connection (`DB/AgentRuns_DB.py:351`).
- Authority capture copies only pure session identity/selection, and blocking selection/root/DB validation runs off the UI thread. Every action recaptures the owning session; a list row is not authority (`Chat/console_worktree_recovery.py:38`, `:52`, `:128`, `:167`). Real reopened SQLite/Git tests cover Apply, Discard, Deny, changed source and revoked selection (`Tests/UI/test_console_worktree_recovery.py:128`).
- Pagination uses 51 metadata rows for a 50-row page and preserves the exact cursor; receipt storage is bounded and keyed to the owning conversation (`Chat/console_worktree_recovery.py:146`, `:218`). Successful toast publication checks view, session and conversation (`UI/Console_Modules/worktree.py:53`). Held, ambiguous and resolved row copy preserves uncertainty and accurately discloses retained checkout state (`worktree_recovery_dialog.py:18`).
- ChatScreen grows by the disclosed 21 lines/3 methods, all hook/delegation/decision forwarding. Actual recovery policy and lifecycle remain in the helper/adapter/widgets. This is an appropriate narrow addition despite the inherited whole-screen overage (`UI/Screens/chat_screen.py:23441`).

### Issues

#### Critical (Must Fix)

- None found in this task-scoped review.

#### Important (Should Fix)

- **Fence new manual admission during session close.** `Chat/console_worktree_recovery.py:167` rejects only globally closed helpers or a currently registered operation; `:234` signals only an operation that already exists. `Chat/console_runtime.py:3955` calls that signal once before asynchronous voice/turn/fleet draining. The owning session remains in the store during that drain (`console_runtime.py:3989`), so a late queued recovery-picker callback can enter `start`, create a fresh unsignalled Event, and validate the still-current binding. Neither `capture_intent` nor the authority snapshot check includes the session-close fence (`console_worktree_recovery.py:38`; `console_chat_controller.py:1477`), and confirmation cancellation reads only the fresh operation Event and visit Event (`console_chat_controller.py:13325`). This allows a new confirmation/operation after the owning session has begun closing; it can remain pending or proceed during the drain, contrary to the required close boundary. The same placement also cancels existing manual work before voice drain and revision/close acceptance: a stale revision or refused/provisional close can cancel recovery even though the session stays open. Move cancellation to the accepted irreversible close boundary and have manual admission consult the existing authoritative runtime admission fence; avoid a second permanent per-session ledger. Preserve rollback behavior for refused/provisional closes and the existing physical-owner drain behavior. Add a deterministic test that gates real session close/drain, attempts a late `start`/queued picker callback, and verifies refusal with no new worker or confirmation, while another session remains usable. Also verify that a refused/stale close leaves the operation signal clear. The current close test (`Tests/Chat/test_console_worktree_recovery_lifetime.py:230`) replaces `_close_session_after_voice_drain` wholesale and starts workers before close, so it cannot catch this window.

#### Minor (Nice to Have)

- **Add the requested schema equality assertion.** `Tests/Chat/test_console_worktree_recovery_lifetime.py:84` proves only signature presence. Compare actual first-request tool schemas from both preview paths and live planning under wired and absent surfaces; retain the generic retained-target negative case. No schema mismatch was established from this diff.
- **Output/guard debt remains explicit.** The reported successful runs contain the inherited `RequestsDependencyWarning`; the startup selection also intentionally warns about 973/973 headroom and snapshot drift. The two historical size guards remain failures. These are not pristine outputs or a full green guard sweep, but do not justify unrelated dependency work, a rebaseline, or whole-screen decomposition in this slice. The new increment is 21 lines/3 thin methods, not part of the inherited baseline.

### Native Visual Assessment

- ✅ Independently inspected all six supplied `.svg.png` artifacts under this task's `visual-evidence`: card/list at 120×40 and 60×28, plus narrow discard and scrolled resolved state. Long paths wrap, long content scrolls, actions and Close remain visible, and ready/disabled states are distinguishable. The narrow discard states that a baseline checkout is retained; the resolved row accurately repeats that consequence with disabled actions. No material visual finding.
- ✅ Applied the existing terminal-native Operate world and approved protected-focus modal, using the provided craft floor in that context. The standard terminal typography, token colors, spacing and control states fit the incumbent interface. White thumbnail margins are export framing, not application geometry. No browser redesign rules were imposed.
- ⚠️ The static captures do not establish keyboard traversal, every empty/error/loading state, or functional event routing. No new screenshots or live-app session were requested or run.

### Checks and Evidence Scope

- Reviewed immutable diff `17635484ba..c5edf2741b` once in sequential chunks. Recovered the lifetime-test portion omitted by the first tool-output truncation; no missing production hunk remained. Used diff line counters only to locate report references.
- Focused outside-diff checks addressed two named risks only: (1) close/drain admission and cancellation ordering, checking runtime close, controller begin-close/cancellation and authority-current predicates; (2) worker DB cleanup ownership, checking `AgentRunsDB.close`. Initial guessed interrupt/preview filenames were absent; those read-only lookup failures are not product/test failures. No Git commands, source edits, Backlog changes, subagents, checkouts, or test reruns occurred. Only this requested report was written.
- Consumed root-verified evidence: final-functional 39 passed; final-lifetime 12 passed; authority-and-startup 31 passed, each exit 0/empty stderr; 18-file static comparison zero added diagnostic identities and passing changed-range formatting, seven new Python files zero diagnostics. Read the final-functional command record to confirm its three targeted test files. Warnings above remain in pytest output despite empty stderr.
- Retain the implementer's raw evidence classifications: missing-feature REDs; actual deepcopy/duplicate-decision/remount/dispose defects; test introspection and mount-wait harness corrections; six full-flow harness setup iterations before the passing real controller/bridge/service/child/Git flow. This review does not reinterpret harness failures as product failures or manufacture a cleaner TDD chronology. The initial old-host CSS builder invocation failed; the subsequent isolated Python 3.12 build and targeted CSS checks are the applicable evidence.

### Assessment

**Task quality: Needs fixes.**

The Console responsibilities are well separated, exact consent is robust, and the native presentation is usable. The session-close admission window must be fenced; the explicitly requested preview/live schema comparison should be supplied before the task's verification claims are complete.


Diagnostic inventory delta atc5edf2741b: the five earlier reviewed fixed-message/type-name warnings are unchanged. The sole additional statement is ConsoleRuntime debug “Worktree remount failed (exception_type={})” with type(exc).__name__; root inspected statement-level output against7852cf47ba and found no body/path/secret interpolation or new sink. Pin write remains deferred until source fix completion.


## Console fix round 1 at7faf6d2742

Root inspected the actual command/output/exit records: final lifetime60passed, final mounted UI25passed, preview/raw-shell neighbors23passed. Every successful run exited0 with empty stderr and the inherited Requests warning. The nine-file static comparison againstc5edf2741b has zero added diagnostic identities and passing edited-range formatting. git diff --check passed before the local source commit. Source is committed; the implementer report below retains its original unstaged-at-handoff wording.

## Review fix round 1 — base c5edf2741b

Read `task-1-review.md` fully. The Important session-close finding is corrected. Public `close_session` no longer cancels manual recovery before voice draining or revision validation. `_close_session_after_voice_drain` signals the owning manual Event immediately after the controller accepts the exact irreversible close ticket. Pure `capture_intent` consults the app runtime's existing `_raise_if_disposed_or_session_fenced` check; both new manual admission and the worker's marshaled authority recheck therefore reject a fenced owner. No second permanent session ledger was added. Existing shielded physical task/ExecutionOwner/owned-DB cleanup is unchanged.

The accepted-close regression gates the actual fleet-drain await inside the real close implementation, verifies an already-owned Event is signalled, removes its completed operation, and attempts a fresh late start while the closing session still exists in the store. It asserts refusal before any engine worker or confirmation, while another session can still run. Three additional cases cover stale revision, explicit controller refusal, and provisional voice close; all preserve the existing Event. The earlier test that replaced the entire close/drain method was removed in favor of this actual boundary coverage. The physical-worker cancellation test and runtime shutdown neighbors remain in the final passing selection.

The requested mounted adapter path uncovered a second real defect: `_RecoveryButton.action` collided with Textual's built-in Button action routing. `Button.press()` consequently ran the string action instead of emitting `Button.Pressed`, so the actual picker never launched recovery. The payload field is now `recovery_action`. No layout/CSS change was made. The mounted fixture calls the actual ChatScreen command action, enters the actual async adapter/list and modal, presses the real row button, shows the real controller inline card, and invokes the shared engine against reopened SQLite and actual Git. Normal apply succeeds. The stale-session variant retains its outcome only under the owning conversation and suppresses notification to the newly viewed conversation. A queued row press during a gated actual accepted close creates no helper operation, receipt, confirmation, or Git change. Existing reopened apply/discard/deny/changed-source/changed-authority coverage remains passing.

### Preview-schema evidence and explicit limit

The new wired/unwired test runs `controller.build_context_snapshot`, both real bridge preview planners, then `controller.submit_draft` through the live service. The observed service calls its real request builder. A generic retained decision target is present in both cases; only the real worktree hook toggles disclosure. Exact `spawn_subagent`, `merge_agent_worktree`, and `discard_agent_worktree` schemas match between both preview plans and the live plan, and project-preview native request tools match the live request for that complete worktree family. Merge/discard are present only with the real hook.

This establishes **worktree-family equality, not equality of the entire tool list**. A broader assertion exposed an existing progress-inbox difference: live first-request planning includes progress tools such as `read_agent_messages`, whereas an initially unopened project preview does not; the personal preview does not forward the progress-inbox flag. Root has been notified for scope disposition. No progress-tool production changes were made and no all-schema equality claim is made. The first two schema runs were harness setup corrections (default run-log-enabled preview deliberately unavailable; project instruction budgeting builds two actual requests). The third run exposed the broader non-worktree mismatch. There was no fabricated missing-worktree RED.

### Exact evidence

All paths below are relative to this plan directory. Each runner label contains exact arguments in `command.json`, full `stdout.txt`, `stderr.txt`, and `exit.txt` under `final-evidence/<label>/`.

- `fix1-close-red`: exit 1, **4 failed**. Real product failures: late start returned a completed outcome instead of refusing, and stale/refused/provisional closes set the existing Event. Run before the runtime/helper production edits.
- `fix1-close-green`: exit 0, **4 passed**, 12 deselected after those source fixes.
- `fix1-picker-one`: exit 1, **1 failed**, real picker failed to start (KeyError observing missing operation).
- `fix1-picker-two`: exit 1, **1 failed**, added diagnostic assertion confirmed no operation, receipt, or notification after actual button press. Both preceded the reserved Button.action field fix.
- `fix1-picker-three`: exit 1, **1 failed**, picker now reached engine/confirmation; test incorrectly assumed stale-session outcome type.
- `fix1-picker-green`: exit 1, **1 passed / 1 failed**, stale receipt assertion corrected separately from the production button fix.
- `fix1-schema-one`: exit 1, **2 failed**, log-enabled preview is intentionally unavailable; harness pinned logs off.
- `fix1-schema-two`: exit 1, **2 failed / 2 passed**, preview builder emits two requests for instruction budgeting; both mounted adapter cases passed.
- `fix1-schema-three`: exit 1, **2 failed**, entire tool-list comparison exposed progress-inbox difference described above.
- `fix1-schema-worktree`: exit 0, **2 passed**, exact actual worktree-family schema comparison, wired/unwired.
- `fix1-picker-final`: exit 0, **3 passed**, mounted action/picker/confirmation, stale receipt, and queued picker during accepted close.
- `fix1-lifetime-final`: exit 0, **60 passed** across `Tests/Chat/test_console_worktree_recovery_lifetime.py`, `test_console_runtime_shutdown.py`, and `test_console_worktree_merge_confirm.py`.
- `fix1-ui-final`: exit 0, **15 passed**, whole targeted `Tests/UI/test_console_worktree_recovery.py`, including both real controller/bridge/service child-write flows and all added adapter/schema cases.

Every run above has empty stderr; successful pytest output contains the inherited `RequestsDependencyWarning`. No dependency/config changes or warning suppression. No full suite, live app/provider/network, new checkout, subprocess imaging, screenshots, Git mutations, Backlog edits, or subagents. The two earlier approved native visual rounds remain the applicable visual evidence; see root's `visual-review.md`.

### Source/static scope

Production files changed in this round:

- `tldw_chatbook/Chat/console_runtime.py` — accepted-ticket cancellation placement.
- `tldw_chatbook/Chat/console_worktree_recovery.py` — existing runtime admission fence in pure capture/current authority.
- `tldw_chatbook/Widgets/Chat_Widgets/worktree_recovery_dialog.py` — nonreserved row action payload field.

Tests changed:

- `Tests/Chat/test_console_worktree_recovery_lifetime.py`.
- `Tests/UI/test_console_worktree_recovery.py`.

`static-fix1-first.json` recorded only new test import/unused-variable and formatting issues; these were corrected in the changed test blocks. `static-fix1-final.json` compares all five files against **c5edf2741b**: **zero added diagnostic identities; every changed-range formatter check passed**. Runtime retains exactly 32 baseline diagnostics; other four files have zero. No whole-file production formatting/debt cleanup. ChatScreen has no new production change in this round, so the earlier disclosed +21 lines/+3 methods and inherited ceiling failures remain unchanged. All source/tests remain unstaged for root packaging and independent re-review.

### Fix round 1 continuation — complete preview parity (supersedes the earlier open schema limitation)

Root recorded TASK-31210 AC8/AC9 and the plan/ledger refinements before these production edits. The previously reported complete-schema discrepancy is now repaired, rather than filtered out. The earlier worktree-family-only evidence remains historical; the final test compares **all runtime and active schemas** in both actual Console preview plans against the live first-request plan, plus the complete native tool tuple of the actual project-preview request against the live request. No tool-name filtering or shell-provider fixture restriction remains.

The pure shared first-request planner now reserves progress disclosure when an existing inbox is available **or** the enabled fleet will open its inbox before the actual request. This matches the already-existing service behavior without opening an inbox in a preview. Personal preview receives the exact owning session ID and forwards its noncreating inbox lookup; this also covers retained inboxes after the fleet size is reduced to one. No queue consumption, inbox allocation, progress ownership, delivery semantics, or ADR-136 policy changed.

A second complete-schema assertion exposed a real personal-preview omission of `virtual_cli`: project preview/live already composed the available provider, but personal preview neither composed nor forwarded it. Personal preview now uses the same existing composition method with captured configuration, selected repository identity, and admitted roots, then forwards the optional provider to the planner. The available raw-shell provider had the same omission. Its existing composition contract accepts an already permitted/armed app-owned runtime and builds provider/callback objects; it does not execute a command or allocate a runtime. Personal preview now forwards it through that same contract. Authority, Ask/Off policy, kill switch and execution paths are unchanged. Tests use an inert runtime protocol fixture whose `execute` raises a test failure if called, with real provider construction and catalog schemas.

Final matrix: worktree hook present/absent with a generic retained target; fleet size one/three; inbox unopened/empty/queued. Queued cases also have the existing raw runtime armed; other cases are unarmed. Both previews preserve exact inbox identity and queued content, including not allocating an unopened inbox. Full schema equality includes builtins, local files, virtual CLI, conditional raw shell, progress and conditional worktree tools. Hook-driven merge/discard disclosure is additionally asserted against the actual fleet gate. The real controller, bridge, planner, service request builder and mounted app are used; the provider response is scripted and no network/provider process runs.

Additional production files in this continuation:

- `tldw_chatbook/Chat/console_agent_bridge.py`: pure prospective-inbox schema reservation and optional `session_id`, `virtual_cli_provider`, `raw_shell_provider` inputs forwarded by personal preview.
- `tldw_chatbook/Chat/console_chat_controller.py`: personal preview composes/forwards available virtual CLI and raw shell using the existing selection/configuration and passes owning session ID. ChatScreen remains unchanged in this round.

Additional targeted fixture files, explicitly authorized by root:

- `Tests/Chat/test_console_personal_context_snapshot.py`: replaces the legacy `object.__new__` bridge with a real bridge/owned SQLite DB, closes worker-owned DB connections on their worker and the main connection in `finally`. Its missing `_db` prerequisite predated this continuation: the preexisting personal preview already reads runtime definitions through the DB. The new optional forwarding inputs require a fully constructed bridge, rather than silent production fallbacks.
- `Tests/Chat/test_console_raw_shell_revocation.py`: uses the controller's real typed execution context instead of a namespace missing `tool_policy_profile_id`; the unchanged raw composition method already required that field. Once this prerequisite was repaired, the previously unreachable approval test revealed its stale registry alias (host owns registration) and obsolete string assertion (current result is a `ToolReviewDecision`). Both test observations now follow the real contracts while preserving the exact denial verdict, absent approval decision, unrelated-round preservation, revocation, and zero-execution assertions. Its waiter cleanup is unconditional even on a failed registration/assertion.

Additional exact evidence under `final-evidence/`:

- `fix1-progress-red`: exit 1, **6 failed**, complete-schema cases before the progress repair (unopened/empty/queued, wired/unwired). Preserves the actual reproduced mismatch, not a fabricated RED.
- `fix1-progress-green-one`: exit 1, **6 failed**. Progress/native request parity fixed; complete personal-plan comparison revealed the actual `virtual_cli` omission. This is a product disclosure finding, not a harness correction.
- `fix1-progress-green-two`: exit 1, **9 passed / 3 failed** during an intermediate fixture restriction. The failures were the test's missing fleet-size gate in its explicit worktree-presence assertion. That assertion was corrected, the fixture restriction was removed, and the real virtual CLI omission repaired.
- `fix1-preview-complete-green`: exit 0, **12 passed**, complete unfiltered schemas including actual virtual CLI, before enabling the raw-shell test cases.
- `fix1-raw-preview-red`: exit 1, **4 failed**, armed raw-shell complete-schema mismatch before the raw forwarding production edit.
- `fix1-preview-final`: exit 0, **12 passed**, final complete matrix including actual available/unavailable raw-shell provider.
- `fix1-preview-neighbors`: exit 1, **6 failed / 17 passed**, stale bridge/context fixture prerequisites described above.
- `fix1-preview-neighbors-two`: pytest reported **2 failed / 21 passed**, then a stale-registry test waiter kept its process alive. A cleanup call in the newly constructed bridge fixture also tried to allocate capacity after closing it; removed because the preview never allocated capacity. The exact owned pytest PID **22851**, identified by its unique `--basetemp` and runner parent, was terminated with SIGTERM after reporting; raw `exit.txt` is **-15** (shell wrapper status 241). Sandbox denied ordinary `ps`/`kill`; narrowly escalated identification and termination were approved. No foreign process was inspected for cleanup or signalled. This is not counted as a passing or normal-exit run.
- `fix1-preview-neighbors-three`: exit 1, **1 failed / 22 passed**; typed denial-result assertion was the remaining stale fixture check. Unconditional cleanup now let this failure terminate normally.
- `fix1-preview-neighbors-final`: exit 0, **23 passed**, exact two affected neighbor files; threads drain and no command execution occurs.
- `fix1-ui-final-expanded`: exit 0, **25 passed**, entire targeted mounted UI file including live controller/bridge/child/Git flows, picker/close/receipt cases and the complete preview matrix.

All these successful runs have empty stderr and the inherited `RequestsDependencyWarning` in pytest stdout. No dependency fixes, live config, schema/storage changes, tool execution, new screenshots, checkouts, Git mutations, or unrelated UI work. Earlier `fix1-lifetime-final` remains the applicable 60-pass close/confirmation/shutdown evidence; this continuation changed only preview planning and targeted fixtures.

**Final static artifact:** `static-fix1-final-all.json`, all **nine** round-1 source/test files versus **c5edf2741b**. Zero added diagnostic identities and all changed-range formatter checks pass. Baseline diagnostic counts are unchanged: runtime 32, bridge 27, controller 188, personal-context test 2, raw-shell test 1; the other four files remain zero. `static-fix1-final-expanded.json` was the seven-file intermediate result before the two neighbor fixtures; it is superseded by the nine-file artifact. One test-only changed-range format correction followed the last neighbor run; no behavior changed. Source/tests remain unstaged, and root's other docs/governance were left untouched. No open scoped functional finding remains for implementer action; independent re-review is root-owned.


## Final diagnostic inventory verification

Root repeated the current inventory diff and reviewed every added statement since7852cf47ba. There are six added calls: one execution-drain callback warning, three agent-worktree persistence/routing warnings, one local cleanup-observer warning and one Console remount debug message. Each uses fixed text plus an exception type, without exception text, user bodies, paths or locators. No persistent sink topology changed. The refreshed pin records607owner files,1392TASK492calls and7726TASK494calls.

The targeted final-diagnostic-inventory run passed the current production inventory/sink guard and failed the separate historical metadata-label guard (1passed/1failed, inherited Requests warning, exit1, empty stderr). That second guard expects eight labels at stale locations: one in console_agent_bridge.py and seven in console_fleet_wake.py. An AST-only comparison of current source with git show9861170ff9 confirmed all eight were absent from their expected owner before restoration too. The focused repeated historical guard failed identically; no test or production rebaseline was made. This is an inherited guard limitation, not a green sweep. Exact statements and comparison results remain in the Console plan scratch diagnostic-statements-final.txt and diagnostic-metadata-stale-labels.json; raw tests remain in final-evidence.


## Additional restoration rulings

Ruling: Include the actual picker/adapter boundary test with the close-fence correction — it is the user entry path and can deliver the late queued action at issue — cost is one additional focused UI integration case; no new visual rounds or product scope.

Ruling: Preserve the three thin ChatScreen binding methods and disclose their21-line increment — independent review confirms all recovery policy remains outside the screen, while the unchanged historical size guards already fail by thousands of lines — cost is a small additional composition burden in the existing oversized class, not a disguised baseline or a raised budget.

Ruling: Fix the actual progress-inbox preview disclosure mismatch found by the required live comparison — queued-agent progress is part of this same orchestration workstream, and excluding its mismatched schema would conceal a reproduced defect — cost is extending the narrow preview forwarding correction; TASK31210 AC8 was added via CLI before source edits. Existing ADR136 progress semantics remain unchanged; no new ADR.

Ruling: Include the narrow virtual CLI/raw shell preview provider forwarding correction if the actual caller omits already-available providers — the whole-schema comparison has exposed this adjacent disclosure boundary and TASK31210 AC9 now records it before edits — cost is a small preview argument-surface extension and targeted provider-present/absent coverage; no provider policy or runtime redesign.

Ruling: Repair the six stale raw-shell/personal-preview fixture prerequisites exposed by the adjacent caller verification — tests need real accepted-context fields and bridge DB ownership to reach the contracts they assert — cost is a bounded extension to the affected test files and their static checks, with failure provenance retained; production fallbacks and skipped authority assertions are not a remedy.

Ruling: Preserve and disclose the eight inherited historical diagnostic-label expectations for final review rather than silently refreshing them as part of the new pin — the restored inventory and six changed statements pass the actual current-source inspection, and all8label absences precede this restoration — cost is one known historical architecture guard failure; the broad reviewer receives the evidence and can assess its relevance.


## Console scoped re-review verdict at7faf6d2742

# Task 1 fix round 1 re-review — Console worktree recovery

Fix base: `c5edf2741be0adbcecf75eb3d1ec24bbac894339`
Immutable head: `7faf6d274202d2b4f560ea8c10eff42093445ce8`

## Finding verdicts

### ADDRESSED — Fence new manual admission during accepted session close

The close path now installs the runtime's existing per-session admission fence before asking the controller for the exact close ticket, rolls that fence back if ticket issuance fails, and signals manual recovery only after the ticket is accepted (`tldw_chatbook/Chat/console_runtime.py:3996`, `:4000`, `:4002`, `:4006`, `:4010`). Public `close_session` no longer signals recovery before voice drain or revision acceptance (`tldw_chatbook/Chat/console_runtime.py:3945`). A provisional voice close returns before reaching the irreversible boundary (`tldw_chatbook/Chat/console_runtime.py:3969`, `:3973`).

Manual admission now consults that authoritative runtime fence in pure intent capture (`tldw_chatbook/Chat/console_worktree_recovery.py:38`). The initial start checks it before allocating an owner (`tldw_chatbook/Chat/console_worktree_recovery.py:181`), and worker authority validation marshals the same check back to the UI thread (`tldw_chatbook/Chat/console_worktree_recovery.py:79`). `start` performs intent capture, owner allocation, task creation, and operation registration without an intervening await (`tldw_chatbook/Chat/console_worktree_recovery.py:173`, `:186`, `:221`, `:222`): a start admitted first is visible to close and receives cancellation, while a start scheduled after the fence is refused. This reuses the runtime ledger and adds no second permanent session fence.

The deterministic lifetime test reaches the real close implementation, gates its actual fleet drain while the session is still stored, observes cancellation of existing work, attempts a fresh late start, verifies no engine or confirmation, and proves another session remains usable (`Tests/Chat/test_console_worktree_recovery_lifetime.py:260`, `:286`, `:295`, `:299`, `:300`, `:302`, `:303`, `:305`). The stale revision, explicit refusal, and provisional voice cases all assert that the existing recovery event stays clear and the runtime fence rolls back (`Tests/Chat/test_console_worktree_recovery_lifetime.py:315`, `:331`, `:333`, `:340`, `:350`, `:358`, `:359`).

### ADDRESSED — Actual picker action, accepted-close queueing, and stale-view receipt evidence

The picker button payload no longer shadows Textual's reserved `Button.action`; it uses `recovery_action`, and the real `Button.Pressed` handler returns that value (`tldw_chatbook/Widgets/Chat_Widgets/worktree_recovery_dialog.py:37`, `:41`, `:108`, `:114`). The production adapter lists asynchronously, rejects a stale view before mounting, starts recovery only after a row selection, and publishes a toast only when the original view, session, and persisted conversation still match (`tldw_chatbook/UI/Console_Modules/worktree.py:14`, `:30`, `:36`, `:39`, `:51`, `:53`, `:57`, `:62`, `:68`). Selection still enters the existing inline confirmation rather than granting consent.

The mounted test invokes the actual `ChatScreen.action_recover_agent_work`, waits for the real dialog, presses the real row button, observes the helper operation and inline card, and reaches the real reopened SQLite/Git engine (`Tests/UI/test_console_worktree_recovery.py:198`, `:223`, `:231`, `:239`, `:244`, `:278`, `:283`, `:289`, `:306`, `:309`). Its accepted-close variant queues the real picker press during the gated close drain and verifies no operation, receipt, confirmation, or Git change (`Tests/UI/test_console_worktree_recovery.py:246`, `:257`, `:266`, `:267`, `:269`, `:273`). Its stale-view variant switches to another conversation before confirmation, then verifies no toast and a receipt only under the owning persisted conversation (`Tests/UI/test_console_worktree_recovery.py:302`, `:306`, `:325`, `:327`, `:328`, `:329`).

### ADDRESSED — Requested actual preview/live schema equality

The new mounted matrix covers the real hook present and absent, fleet sizes one and three, and unopened, empty, and queued inbox states (`Tests/UI/test_console_worktree_recovery.py:559`). It observes the shared planner through both real controller preview paths and live submission (`Tests/UI/test_console_worktree_recovery.py:598`, `:612`, `:617`, `:750`, `:762`), retains the generic decision-target negative case (`Tests/UI/test_console_worktree_recovery.py:732`, `:735`), and compares the complete runtime and active schema tuples from both previews to live without name filtering (`Tests/UI/test_console_worktree_recovery.py:755`, `:768`, `:771`, `:773`). It also compares the actual project-preview native request tool tuple with the live first request (`Tests/UI/test_console_worktree_recovery.py:757`, `:760`, `:765`, `:767`) and separately asserts conditional worktree disclosure (`Tests/UI/test_console_worktree_recovery.py:778`).

This satisfies the original worktree-family assertion and the expanded all-schema requirement. No mismatch is concealed by filtering.

### ADDRESSED — Complete progress, virtual CLI, and raw-shell preview parity

The shared planner now reserves progress disclosure from a noncreating inbox observation or prospectively when an enabled fleet will create its inbox before the request (`tldw_chatbook/Chat/console_agent_bridge.py:4671`). Project preview uses the exact noncreating session lookup (`tldw_chatbook/Chat/console_agent_bridge.py:5300`); personal preview now receives the owning session and uses the same lookup (`tldw_chatbook/Chat/console_agent_bridge.py:5335`, `:5415`). Live planning forwards its current lookup through the same planner (`tldw_chatbook/Chat/console_agent_bridge.py:5727`, `:5769`), and the existing live path opens the fleet inbox before constructing the service request (`tldw_chatbook/Chat/console_agent_bridge.py:6759`, `:6766`, `:6774`).

Personal preview now composes virtual CLI and raw shell from the same captured turn configuration, selected root, root identity, and admitted roots used by live composition, then forwards both providers to the shared preview planner (`tldw_chatbook/Chat/console_chat_controller.py:19237`, `:19265`, `:19293`, `:19298`, `:19311`, `:19353`, `:19368`). Project preview and live already use the corresponding composition/forwarding paths (`tldw_chatbook/Chat/console_chat_controller.py:19485`, `:19486`, `:19504`, `:19513`, `:19533`, `:19548`; `tldw_chatbook/Chat/console_chat_controller.py:25471`, `:25483`, `:25499`, `:25770`). Both preview providers are registered only in the disposable schema plan (`tldw_chatbook/Chat/console_agent_bridge.py:5381`, `:5389`); inbox inspection itself does not allocate a message store or inbox (`tldw_chatbook/Chat/console_agent_bridge.py:8121`).

The matrix preserves inbox identity and content across both previews (`Tests/UI/test_console_worktree_recovery.py:748`, `:753`, `:754`), makes shell execution fail the test if invoked (`Tests/UI/test_console_worktree_recovery.py:708`, `:712`), and asserts complete parity plus the expected conditional shell and virtual CLI names (`Tests/UI/test_console_worktree_recovery.py:767`, `:775`, `:776`).

### ADDRESSED — Test fixture ownership in the expanded scope

The personal-context neighbor now constructs a real bridge with an owned SQLite handle, closes the worker-thread connection in the preview wrapper, and closes controller/progress/main-thread ownership in `finally` (`Tests/Chat/test_console_personal_context_snapshot.py:371`, `:375`, `:376`, `:383`, `:386`, `:439`, `:455`). The raw-shell neighbor registers its fixture registry on the actual interrupt host, unconditionally shuts down the controller, and joins both owned threads even after an assertion failure (`Tests/Chat/test_console_raw_shell_revocation.py:161`, `:162`, `:196`, `:198`, `:208`, `:210`, `:211`, `:214`). The final neighbor evidence terminated normally, so the earlier retained waiter was not carried into the accepted evidence.

## New fix breakage

No new Critical, Important, or in-scope Minor breakage found.

## Out-of-Scope Observations

- The unchanged planner docstring still says both preview call sites intentionally omit worktree disclosure because no production confirmation surface exists (`tldw_chatbook/Chat/console_agent_bridge.py:4515`). That description was already stale at the fix base and is therefore non-blocking for this scoped re-review; it should be corrected in the forthcoming broad integration/documentation review.
- The inherited `RequestsDependencyWarning`, UI-ready 973/973 headroom and drift warning, and existing ChatScreen size/no-growth failures remain as disclosed. This fix adds no ChatScreen production change.
- Root's current documentation/pin edits are outside this immutable fix artifact and were not reviewed here.

## Checks and evidence limits

- Read immutable diff `review-c5edf2741b..7faf6d2742.diff` once in bounded sequential chunks `1-250`, `251-500`, `501-750`, `751-1000`, and `1001-1249`; no diff output was truncated. Read unchanged surrounding close/admission, preview composition, inbox creation, adapter, and fixture cleanup code only for the named concerns.
- No suite or isolated test was rerun. Saved command/output/exit records verify: `fix1-lifetime-final` 60 passed; `fix1-ui-final-expanded` 25 passed; `fix1-preview-final` 12 passed; `fix1-preview-neighbors-final` 23 passed. Each exited 0 with empty stderr and the inherited Requests warning in stdout. Counts overlap and are not summed.
- `static-fix1-final-all.json` covers the nine scoped source/test files against `c5edf2741b`: zero added diagnostic identities and every changed-range formatter exit is 0. Whole-file lint exits remain nonzero only where the recorded baseline diagnostics remain unchanged (runtime 32, bridge 27, controller 188, personal-context test 2, raw-shell test 1).
- No full suite, live app/provider/network, new checkout, screenshots, source/index/HEAD/Backlog mutation, dependency/configuration change, cleanup, or subagent work was performed.

## Verdict

**All findings addressed; no new Critical or Important breakage.** The fix is ready to return to the broader integration review, subject to the already disclosed inherited warnings and guard debt.


### Final planner documentation correction

The stale surface docstring identified by scoped re-review is corrected before the broad integration package. It now describes both previews and live dispatch forwarding the actual confirmation-surface gate. Behavior is unchanged; no test rerun was required. The one-file static-final-docstring.json records27baseline/current diagnostics, zero added identities and all changed-range formatter exits0.

Ruling: Include the stale planner docstring correction in final documentation packaging — behavior and full schema equality are already reviewed, but the comment contradicts the new callable surface — cost is a tiny source-doc hunk for broad review, without redundant behavioral tests or another task-level review seat.

Ruling: Qualify the completed full-flow checklist item as delivered behavioral proof, not an invented pre-implementation RED — the six original full-flow failures were harness prerequisites after wiring, while the missing card/hook/owner cases had real REDs — cost is explicitly retaining a full-flow TDD chronology deviation rather than reconstructing history.


## Broad restoration integration review at256507e387

# Final agent-worktree restoration integration review

Review base: `9861170ff9` (before restoration). Immutable head: `256507e3875875f6d4d02614edceef8872a0e1df`. Package: `review-9861170ff9..256507e387.diff`, 10,266 lines, 19 commits.

## Assessment

**Needs fixes: two Important failed-start lifetime defects.** No Critical finding. The five slices otherwise form a coherent restoration under the current ADR-155 ordinary-local-Git contract. These findings concern application-owned process/thread startup, not the expressly accepted concurrent external repository-metadata replacement limit. They do not require another Git backend or a new architecture.

Final task/plan checkbox and status reconciliation remains root-owned and should follow the narrow fixes and their scoped re-review. This review does not reopen the previously approved remaining-wave work through the base.

## Strengths

- **Selected authority reaches creation and recovery without fallback.** The controller captures the selected writable binding and adds a fail-closed kill-switch guard, the bridge forwards it unchanged, and the service refuses missing authority or a missing real local provider. Creation preserves the captured base and stores the known checkout before later admission can fail. Child filesystem routing uses the captured source guard and child identity. Automatic retirement remains routing-only. The accepted-turn test exercises removal, retargeting, root replacement and kill-switch failures; real service tests distinguish selected and unrelated fallback repositories.
- **Durable ownership and physical drain are integrated.** Schema 20 records original base, exact run/execution/binding and complete identity chains before provider admission and child execution. The repository borrows the DB, performs parameterized metadata-only joins, uses transactional exact claims, and preserves held/uncertain/in-flight state on reopen. Completion callbacks detach outside the capacity lock; local cleanup uncertainty poisons the captured owner before the tool returns. Existing failed child-thread admission explicitly finishes the owner, and callbacks close only callback-created connections. Real delayed-worker and failed-start tests verify retained work and exact owner equality.
- **Both action entry points consume one confirmed engine.** Current-turn handle membership is checked against the actual created run. Earlier-turn Console recovery recaptures current selected authority rather than reviving handles. Exact Allow, source/destination snapshot comparison, eligibility and a transactional claim precede mutation. Original-base apply preserves unrelated destination staging; merge verifies a new commit with exact expected parents, and already-incorporated work does not receive a false merged receipt. Ambiguous effects and failed persistence stay protected.
- **Discard matches its disclosed contract.** Descriptor-relative no-follow cleanup retains the checkout root and administrative link, restores the original detached baseline, and removes only the recorded branch with expected-old-SHA CAS. Unsupported read and discard primitives are separate refusals. Nested repository/submodule refusals and symlink/CAS tests support the retained-work boundary; no automatic or forced root deletion is reintroduced in the confirmed path.
- **Console presentation and ordinary operation lifetime connect correctly.** Immutable button-owned round IDs prevent queued old presses from approving a newer round. The payload is volatile and rendered without markup. Worktree remount runs independently of the unified-approval early return. The retained helper owns its own cancellation signal and worker DB handle; shielding preserves an already-running worker after waiter cancellation. Accepted close fences admission and then cancels only the owning operation; stale/refused/provisional close preserves it. The actual command → asynchronous list → real picker → confirmation → Git tests cover action routing, queued selection during close and owning-conversation receipt delivery.
- **Both real previews match live planning for the tested capability matrix.** Complete active/runtime schemas and the actual project-preview request tool tuple are compared without filtering under hook present/absent, fleet size one/three, and unopened/empty/queued inbox conditions. Available virtual CLI and conditional raw-shell schemas use the existing composition path. Tests preserve inbox identity/content and prohibit shell execution during preview. The final planner docstring now describes that real surface gate accurately.
- **The startup refinement is narrow.** The unchanged worktree DDL literal moves to its actual DB consumer and the existing run-log modal import moves into the opening handler. The retained evidence covers migration/reopen, paged-log opening and the unchanged startup ceiling. The diagnostic-pin change corresponds to six fixed-text/type-only statements, with no new sink.

## Findings

### Critical

None identified.

### Important

1. **Git reader startup can bypass all process retirement.** `tldw_chatbook/Agents/agent_worktree_git.py:93-97`, specifically `reader.start()` at **line 94**. The subprocess already exists at line 51, but reader construction/start happens before the cleanup `try/finally` begins. If either reader fails to start (for example, `RuntimeError: can't start new thread`), `run_git` escapes without killing/waiting for Git, closing the remaining pipes, joining the reader that may already have started, or marking cleanup unproven. This applies to source capture and destination mutations as well as preview reads. The shared engine catches the exception, and the caller can finish its execution owner while that Git process or its descendants still run. An uncertain mutation row prevents replay but does not restore physical process ownership; a pre-claim failure leaves the row unresolved too. **Smallest remedy:** establish retirement protection immediately after successful `Popen`, include reader construction/start inside it, track which readers actually started, retire the process group and close/join owned resources on partial startup, and latch cleanup unproven whenever retirement cannot be established. Do not join an unstarted thread. Add deterministic first-reader and second-reader startup-failure cases, including a gated mutating command, proving no later Git effect/live worker remains after a proven completion and that an unproven retirement poisons the owner.

2. **Manual recovery leaks its execution owner when worker submission fails before entry.** `tldw_chatbook/Chat/console_worktree_recovery.py:187-221`, specifically the `asyncio.to_thread` task creation at **line 221**. `begin_execution` registers the owner before executor submission, while the only `finish_root` is inside `worker` at line 219. A definite submission refusal before `worker` runs (for example, a shut-down default executor) completes the asyncio task exceptionally; `completed` removes the operation at line 225 and retains a failure receipt, but never releases that owner. `close()` can then finish with no registered operation although the capacity ledger permanently reports an active execution. `RuntimeCapacity.close()` only fences new admission; it does not repair leaked owners. Existing bridge capacity replacement/rebinding checks treat that phantom owner as active. **Smallest remedy:** explicitly own the pre-start-to-worker handoff and release the exact owner when execution is positively known not to have started; keep worker-finally ownership for admitted work and preserve shielding after entry. Account conservatively for a submission error whose work may already have been queued—do not blindly finish the owner while a worker might still start. Add a deterministic rejected-submission case asserting the engine is never entered, the operation registry is cleared, and the capacity snapshot contains no leaked execution; preserve the existing running-worker cancellation test.

Both findings are established by control-flow inspection. No new runtime reproduction is claimed; the supplied tests do not inject these two startup failures. The existing child-thread failed-start handling is sound, but it does not cover these separate Git-reader and manual-executor startup boundaries.

### Minor

No additional actionable in-scope Minor finding. The qualifications below are retained limitations/debt, not requests for unrelated repair.

## Evidence inspected and limits

- Read the supplied immutable diff in bounded sequential passes, including all source, tests, five current plans, restoration design, retained slice reports/rulings, ADR amendments and documentation changes. Recovered every tool-truncated middle range from that same package; did not regenerate the diff. Read the current ADR-155 in full to distinguish incorporated ownership/consent policy from superseded execution qualification.
- Focused outside-diff checks answered named integration concerns only: actual child launch/failed-launch owner finalization (`agent_service.py:5480-5556`); ordinary creation's existing Git runner (`Workspaces/git_workspace.py:198-252`); accepted session-close ordering (`console_runtime.py:3945-4060`); thread-local DB close (`DB/AgentRuns_DB.py:321-361`); capacity admission/close/snapshots and bridge active-owner replacement checks (`Agents/execution_capacity.py:82-204`, `console_agent_bridge.py:8033-8101`). No whole-repository repeat review occurred.
- Consumed the retained implementation/re-review evidence and root's stated inspection of exact command/output/exit records: creation 260-case affected selection and scoped fixes; storage 31-case and 104-case neighbors; ownership 14-case final selection and exact-owner fixes; confirmed engine 91-case gate plus focused 4/3-case additions and 14-case merge fix; Console final 60 lifetime, 25 mounted UI and 23 preview/raw-shell neighbors, with the complete 12-case preview matrix. **These selections overlap and are not summed.** This final review independently read the test bodies and retained reports; it did not rerun those commands or claim to have freshly reproduced their counts.
- Reused the documented two root visual inspections and independent inspection of six native wide/narrow captures. No new visual defect arose from source inspection. No additional visual session or new screenshot inspection was performed by this reviewer. Static captures do not establish every interaction; actual picker and old-button tests provide separate functional evidence.
- Read `diagnostic-statements-final.txt` and `diagnostic-metadata-stale-labels.json`. All six added logger statements are fixed text with exception-type values. The saved AST comparison records zero matches both at the restoration baseline and current head for each of the eight historical labels. The refreshed current inventory/sink guard passes; the separate stale-label guard remains a disclosed failure. This is not a blanket green architecture-guard claim, and neither a rebaseline nor unrelated diagnostic repair is requested.
- The inherited `RequestsDependencyWarning` remains. UI-ready passes at **973/973**, with **zero headroom** and the intentional **+16/-16 drift** warning. Historical ChatScreen size/no-growth guards remain failed: **24,410 lines/746 methods** versus the captured **24,389/743** baseline and unchanged **16,966/563** ceilings. The restoration adds **21 lines/3 thin methods**; those increments are not mislabeled inherited. Scoped static reports record zero added diagnostic identities and passing edited-range formatting; they do not establish whole-file lint cleanliness.
- Creation's missing positive pre-implementation RED and the Console full-flow harness failures remain process qualifications. Passing final behavior was not reinterpreted as a cleaner TDD chronology. The earlier timed-out neighbor process and its qualified cleanup remain historical evidence, not an accepted passing run.
- No tests/suites were rerun: inspection established the two unanswered startup boundaries and repeating passing selections would not test them. No product imports, live configuration/provider/network access, verification checkout, dependency changes, cleanup, source/index/HEAD/Backlog mutation or subagents occurred. Only this requested report was written.
- Platform evidence remains owned temporary Git/SQLite on macOS/POSIX. No Windows certification, full-suite result or live-provider validation is claimed. Ordinary Git's accepted external metadata/config/filter limitations and retained manual-only uncertain/legacy work remain explicit.


## Final startup-lifetime correction ate135a085f2

Root inspected the exact command/output/exit and four-file static records before committing the correction. The report below preserves its unstaged-at-handoff wording.

# Final failed-start lifetime fix report

Base: `256507e3875875f6d4d02614edceef8872a0e1df`.
Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/agent-orchestration-pr`.

Both Important findings in `final-review.md` are addressed. This worker changed only the four authorized source/test files and this scratch evidence/report. Source/tests were left unstaged; no Git mutations, Backlog edits, documentation edits, dependency changes, provider/network work, live configuration reads, new checkouts, foreign cleanup or subagents were performed. The explicitly supplied scoped-static helper performs its authorized read-only Git comparisons.

ADR required: no new ADR.
ADR path: `backlog/decisions/155-agent-worktree-recovery.md`.
Reason: these changes complete the accepted physical-lifetime and failed-start contract without changing the ordinary-local-Git boundary.

## Changes and decisions

- `tldw_chatbook/Agents/agent_worktree_git.py`: prepare pipe-reader state before process creation and establish `try/finally` immediately after successful `Popen`. Construct/start readers inside that protection, retain each reader/pipe pair, and separately track threads with a physical identity so an unstarted reader is never joined. Retire the owned process group and explicitly wait/reap even when signaling finds that the process disappeared. Join started readers, close pipes without a live reader, and retain cleanup-unproven state for unretired processes/readers or failed close. A live `BufferedReader` is never closed externally while its reader can hold the buffer lock; that branch remains bounded and poisons the active owner.
- `tldw_chatbook/Chat/console_worktree_recovery.py`: a standard concurrent `Future` provides an atomic pending-to-worker admission gate and completion acknowledgment. On failed submission, cancellation of an unclaimed gate positively prevents engine/DB entry even when the executor already queued the wrapper; only then does the submitting path finish the exact owner. A claimed gate leaves finalization with the worker. If submission raises after entry, the retained operation awaits the worker acknowledgment before reporting the error/removing the operation. The external waiter remains shielded. Worker DB close is nested under finalization so a close exception cannot skip owner completion/acknowledgment. This is an admission handshake on the existing owner, not another capacity ledger.
- `Tests/Agents/test_agent_worktree_confirmed_recovery.py`: eight startup regressions cover construction/start failure at reader one/two, with proven and deliberately denied retirement.
- `Tests/Chat/test_console_worktree_recovery_lifetime.py`: three real executor-boundary regressions cover definite shutdown refusal, queue-then-raise, and engine-entry-then-raise, including cancellation while an admitted worker remains held.

No logger statement changed; the diagnostic pin needs no regeneration for this fix.

## Physical ownership and cleanup evidence

The new Git fixture launches one real process/session using the supplied fixed Git argv behind a pipe-gated Python `execv`. It acknowledges reaching the gate before reader failure is injected, so opening the gate would otherwise permit the exact `update-ref` mutation. In proven cases, assertions occur before fallback test cleanup: the process has been reaped with `-SIGKILL`, its process group is absent, all constructed readers are stopped, both pipes are closed, and opening the gate cannot create the ref. A subsequent ordinary Git read also confirms the ref is absent.

In denied-retirement cases the patched signal path actually refuses SIGKILL and leaves the real gated child alive. The test observes owner drain as `False` and the actual process still live before its unconditional finalizer uses the saved real `killpg` on the exact owned group, waits/reaps it, joins only started captured reader handles and closes its pipes. Patched helpers remain installed until those owned resources retire. The timeout in the production bounded `wait` is an intentional injected cleanup failure, not an abandoned test runner. The test never fabricates successful process retirement.

The manual fixture routes `loop.run_in_executor` to a test-owned real `ThreadPoolExecutor`, retaining every accepted concurrent Future and shutting down that exact executor with `wait=True` before patched helpers are restored. Definite refusal uses an actually shut-down executor. Queue-then-raise is held behind an earlier job until the submitting side has revoked admission; after release, no engine entry or worker DB file exists. Entry-then-raise holds the real worker after observing its activated exact owner. The operation and capacity entry remain while held, including after cancelling the asyncio waiter; only worker completion releases them. All three cases end with no operation, no active capacity execution and a bounded failure receipt. The existing accepted-worker waiter-cancellation regression remains green.

## Exact test evidence

All commands ran from the worktree above using the supplied wrapper. It invokes the existing interpreter `.superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python` and creates a unique evidence directory/basetemp per label. Each directory under `final-evidence/` contains exact expanded `command.json`, complete `stdout.txt`, `stderr.txt` and `exit.txt`.

1. **Pre-source RED** — `final-fix-red-startup`:
   ```sh
   python3 .superpowers/sdd/2026-09-12-agent-worktree-console-recovery/run_pytest.py final-fix-red-startup Tests/Agents/test_agent_worktree_confirmed_recovery.py Tests/Chat/test_console_worktree_recovery_lifetime.py -k 'partial_reader_startup or submission_failure' -q
   ```
   **11 failed, 54 deselected, 1 warning in 1.88s; normal runner exit 1.** Four proven-cleanup cases found a still-live gated process; four denied-cleanup cases found incorrectly proven owner drain; definite and queued submission cases found an unreleased owner; the admitted submission case found the operation removed while its worker was still held. These were behavioral failures, not fixture/import prerequisites. Source changes began only after inspecting these failures.

2. **Initial targeted GREEN** — `final-fix-green-startup`:
   ```sh
   python3 .superpowers/sdd/2026-09-12-agent-worktree-console-recovery/run_pytest.py final-fix-green-startup Tests/Agents/test_agent_worktree_confirmed_recovery.py Tests/Chat/test_console_worktree_recovery_lifetime.py -k 'partial_reader_startup or submission_failure or physical_manual_worker_retains' -q
   ```
   **12 passed, 53 deselected, 1 warning in 27.68s; normal runner exit 0.** Includes the unchanged accepted-worker cancellation regression.

3. **Final affected modules/capacity neighbors** — `final-fix-green-neighbors`:
   ```sh
   python3 .superpowers/sdd/2026-09-12-agent-worktree-console-recovery/run_pytest.py final-fix-green-neighbors Tests/Agents/test_agent_worktree_confirmed_recovery.py Tests/Chat/test_console_worktree_recovery_lifetime.py Tests/Agents/test_execution_capacity.py -q
   ```
   **88 passed, 1 warning in 79.69s; normal runner exit 0.** Runs the final test assertions, strengthened after initial GREEN to check exact `-SIGKILL`/absent group before test cleanup and no worker DB creation for revoked admission. It also covers the existing descendant-filter cleanup, successful and refused Git recovery, retained session lifecycle and capacity behavior.

4. **Root-requested actual caller neighbors** — `final-fix-green-real-ui`:
   ```sh
   python3 .superpowers/sdd/2026-09-12-agent-worktree-console-recovery/run_pytest.py final-fix-green-real-ui 'Tests/UI/test_console_worktree_recovery.py::test_reopened_record_manual_action_uses_real_controller_card_and_git[apply-adapter]' 'Tests/UI/test_console_worktree_recovery.py::test_reopened_record_manual_action_uses_real_controller_card_and_git[discard-normal]' -q
   ```
   **2 passed, 1 warning in 4.97s; normal runner exit 0.** Actual reopened SQLite record → manual helper → real controller/card → Git apply/discard path. No whole-UI sweep or new visual inspection.

These selections overlap and should not be summed as unique coverage. All four runner stderr files are empty. No test runner had an abnormal/signal/timeout exit. The only warning in each run is the inherited RequestsDependencyWarning; output is not pristine.

## Scoped static and self-review

The supplied CLI contract was read before use. Final command:

```sh
python3 .superpowers/sdd/2026-09-12-agent-orchestration-remaining/scoped_static.py 256507e3875875f6d4d02614edceef8872a0e1df .superpowers/sdd/2026-09-12-agent-worktree-console-recovery/final-fix-static.json tldw_chatbook/Agents/agent_worktree_git.py tldw_chatbook/Chat/console_worktree_recovery.py Tests/Agents/test_agent_worktree_confirmed_recovery.py Tests/Chat/test_console_worktree_recovery_lifetime.py
```

Helper exit 0. Each of the four files has base/current diagnostic counts **0/0**, zero added identities, and all edited-range format exits 0. The initial static report (`final-fix-static-initial.json`) found only appended test-range formatting failures; only those ranges were formatted. No whole-file debt cleanup occurred.

Self-review traced successful process creation through each partial reader failure and both proven/unproven retirement branches, including pipe-lock behavior and reap ownership. It traced the manual gate's atomic cancellation/claim race, the queue-after-error no-entry guarantee, admitted-worker finalization/DB close, shielded external cancellation, retained operation cleanup and failure receipt. The deterministic tests reject removal of process retirement, cleanup poisoning, admission revocation or admitted-worker retention. Both Important findings are fully covered in scope; no further source changes remain.

Limits: POSIX/macOS owned temporary processes/Git/SQLite only. The startup mutation fixture delays exec of Git rather than claiming a live Git filter is already running; the unchanged adjacent descendant-filter test separately exercises actual Git descendant cleanup. No Windows certification, new backend qualification, live provider test, full suite or additional screenshot review is claimed. Existing ordinary-Git external metadata/config/filter limitations remain as accepted in ADR155.


## Final correction scoped re-review ate135a085f2

# Final failed-start lifetime fix scoped re-review

Review base: `256507e3875875f6d4d02614edceef8872a0e1df`.
Immutable head: `e135a085f251795ea5379ddbfec5605dda862039`.
Package: `review-256507e387..e135a085f2.diff` (four source/test files).

## Assessment

**All findings addressed; no new Critical or Important breakage.** The single fix wave closes both failed-start lifetime defects from `final-review.md` without changing the accepted ADR-155 boundary or adding another capacity ledger.

## Original finding verdicts

### 1. Git reader startup can bypass all process retirement — ADDRESSED

Locations: `tldw_chatbook/Agents/agent_worktree_git.py:26-31`, `:71-165`; regression coverage at `Tests/Agents/test_agent_worktree_confirmed_recovery.py:554`.

Reader collections and cleanup state now exist before process creation, and the cleanup `try/finally` begins immediately after the successful `Popen` at lines 74-92. Both reader construction and `start()` are inside that protection. Each constructed reader retains its pipe, while `started_readers` receives a reader only when it has a physical thread identity; cleanup therefore never joins an unstarted thread and can still close its pipe.

Every partial construction/start failure reaches process-group retirement, a separate wait/reap attempt, descendant-group absence checking, started-reader joins, and pipe closure. A pipe held by a still-live buffered reader is deliberately not externally closed, avoiding a buffer-lock hang; the live reader instead makes cleanup unproven. Signal, wait, group-absence, close, or reader-join uncertainty calls `_cleanup_unproven()`, which resolves `current_execution_owner()` and poisons that actual owner before its eventual drain result. A disappearing process during signaling still proceeds through `wait()`, so the parent does not mistake signal lookup failure for reaping.

The eight-case regression crosses construction versus start failure, first versus second reader, and proven versus deliberately denied retirement. Its gated mutating command establishes that proven completion leaves the owned process reaped, its group absent, readers stopped, pipes closed, and no later ref update; denied retirement leaves the real child observable and reports owner drain as unproven before fixture cleanup.

### 2. Manual recovery leaks its execution owner when worker submission fails before entry — ADDRESSED

Locations: `tldw_chatbook/Chat/console_worktree_recovery.py:188-262`; regression coverage at `Tests/Chat/test_console_worktree_recovery_lifetime.py:370`.

The existing execution owner is now paired with one standard `concurrent.futures.Future` admission gate. The worker must atomically claim that gate with `set_running_or_notify_cancel()` before validation, DB construction, or engine entry. On a definite submission refusal, or a queued wrapper that has not claimed admission, `handoff.cancel()` proves non-entry; only that winning path releases the exact owner. A queued wrapper that later runs observes the cancelled gate and returns before any engine or DB effect, so release cannot be followed by late recovery work.

If the wrapper claimed admission before submission raised, cancellation loses and the submitting task shield-waits for the handoff acknowledgment. The worker retains physical ownership while it validates, opens/closes its DB, and runs recovery; nested finalization closes the DB, finishes the owner even if DB close raises, and only then acknowledges completion. The original submission or worker exception is subsequently propagated, after which the existing callback removes the operation and stores the bounded failure receipt. The public waiter and `close()` continue to shield the actual operation task at lines 262 and 274-278, preserving the previously accepted running-worker cancellation behavior.

The three-case regression covers definite refusal, queued-then-raise, and entered-then-raise. It checks exact owner identity, no engine entry or worker DB creation after revoked admission, retained registry/capacity ownership while entered work is held, final owner drain, registry cleanup, and the failure receipt.

## New fix breakage

None identified. No new Critical, Important, or Minor issue was found in the four-file fix diff. No logger statement changed, and the reviewed diagnostic pin remains unchanged.

## Out-of-scope observations

No new outside-diff issue was identified. Root's uncommitted documentation is outside this immutable fix and was not adjudicated here. The accepted ordinary-Git external metadata/config/filter limitations remain governed by ADR-155.

## Evidence inspected and limits

- Read the re-review brief, the two original Important findings verbatim, the implementation/evidence report, and the supplied immutable diff once in bounded chunks; recovered only the tool-truncated opening ranges from that same package. No Git diff was regenerated.
- Inspected unchanged source only for the named owner poisoning, admission/cancellation, registry, result propagation, and physical shielding concerns: `ExecutionOwner.activate`, `mark_cleanup_unproven`, `finish_root`, and the manual helper's `start`/`close` paths.
- Consumed root's reviewed command/output/exit record: pre-source RED had **11 intended failures**; the final affected/capacity-neighbor selection had **88 passed**; the two actual reopened card/Git paths had **2 passed**. The focused initial GREEN had **12 passed**. All runners exited normally, all stderr files were empty, and each run retained the inherited Requests warning.
- Consumed the four-file scoped-static record: base/current diagnostic counts **0/0**, zero added identities, and all edited-range formatting exits **0**. No logger change or diagnostic-pin refresh occurred.
- No suite or test was rerun. No product import outside Tests isolation, live configuration/provider/network access, dependency change, checkout, cleanup, source/index/HEAD/Backlog mutation, or subagent was used. This report is the only written artifact.
- Evidence is limited to POSIX/macOS owned temporary process/Git/SQLite behavior. The mutation fixture gates execution before Git `exec`; the retained adjacent descendant test supplies the actual Git-descendant coverage. There is no Windows, full-suite, live-provider, or new visual claim.
- The inherited Requests warning, zero startup headroom, inherited ChatScreen size guards, and eight absent historical diagnostic labels remain disclosed. Static edited-range results do not establish whole-file lint cleanliness or all architecture guards green.

## Final verdict

**All findings addressed; no new Critical or Important breakage.**


## Final task and plan reconciliation

Root used the Backlog CLI to check all 9 TASK-31210 criteria, all 8 TASK-31211 criteria and the parent criterion, add implementation notes and mark each Done. A fresh local file check verified each status and checklist. The seven actual parent children were separately verified Done with zero unchecked criteria and implementation notes. The Console plan and parent closeout plan are complete. No additional historical child or runtime evidence was created to force closure. The prior approved records review and earlier-wave final review remain preserved in their linked records.

Ruling: Keep both failed-start lifetime corrections in one final fix dispatch under existing ADR155 — they complete already-approved physical ownership, without changing ordinary Git or provider boundaries — cost is focused process/executor failure injection plus their adjacent lifetime checks; ambiguous accepted work must remain protected.

Ruling: Use a standard concurrent Future as the manual pre-entry admission gate — cancelling an unclaimed gate prevents even queued-after-error work from acquiring engine/DB authority, while a claimed worker retains physical-finally ownership — cost is one small operation-local handoff state, not a second capacity ledger; tests must cover definite refusal, queued-after-error and entered-before-error.

Ruling: Finish as a verified local branch using the task's targeted checks — current authorization completes implementation and the repository requires opt-in for a full suite; no new PR or merge was requested for this remaining wave — cost is that remote integration and full-suite coverage are separate future actions. Preserve the worktree and evidence under the existing retention ruling.
