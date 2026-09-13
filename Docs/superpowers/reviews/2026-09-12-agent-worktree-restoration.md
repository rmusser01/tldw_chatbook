# Agent worktree restoration review record

Status: implementation continues. Creation and durable-record prerequisites are reviewed; ownership call sites, confirmed operations and the Console flow are not yet complete. This supplements the earlier agent-orchestration-remaining review rather than replacing its historical findings or rulings.

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

