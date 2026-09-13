# Agent worktree restoration review record

Status: implementation continues. Creation, durable-record prerequisites, actual ownership call sites and confirmed operations are reviewed; the Console flow is not yet complete. This supplements the earlier agent-orchestration-remaining review rather than replacing its historical findings or rulings.

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
