# Goal runs: preimplementation review

Date: 2026-09-08
Scope: Review the proposed native Console goal workflow before implementation.
Baseline: Chatbook HEAD `22aa927f0287e84c76ab39836a05a028f2a73e82` plus the shared working tree inspected during this conversation. The agent-budget changes remain uncommitted; implementation must reconcile that baseline.
Artifacts: [Design](../specs/2026-09-08-gnhf-inspired-goal-runs-design.md), [plan](../plans/2026-09-08-gnhf-inspired-goal-runs.md), [proposed ADR-141](../../../backlog/decisions/141-native-console-goal-runs.md).

The native outer loop remains the recommended approach. Seven design issues needed correction. The documents now include those corrections and their acceptance tests; this is not a claim that the runtime changes have been implemented or tested.

## R1 — P1: A successful tool invocation is not a successful, current verification

**Evidence:** [The Console script bridge](../../../tldw_chatbook/Chat/console_agent_bridge.py), lines 3779–3808, renders `ScriptRunResult` into text and returns `ToolResult(ok=True)` even when the process exited nonzero or timed out. The [typed result](../../../tldw_chatbook/Skills_Interop/skill_script_runner.py), lines 98–125, already exposes the actual exit, timeout and truncation fields.

**Failure:** A goal could accept a failed command if it treats tool success as check success. Recording an exit code alone also leaves two gaps: the model can choose an unrelated passing command, and a real passing check can become stale after a later file edit.

**Correction:** Bind each objective check to a launch-approved `VerificationSpec`. Capture typed results before display formatting, with operation identity, verifier fingerprint, arguments and checked artifact version. Match the expected result and verify input freshness before/after checking and before completion or quality approval. An opaque MCP success string requires a qualified evidence adapter; it cannot silently acquire the same meaning as a typed process result.

**Acceptance:** Nonzero exit with `ok=True`, timeout, spoofed stdout, changed verifier/arguments, passing check followed by an edit, and manual edits before review never yield verified completion. A real unchanged verifier passing against the identified final artifact can do so. Slices 2/3/5.

## R2 — P1: The proposed tool subset missed runtime tools

**Evidence:** [`build_first_request_schema_plan`](../../../tldw_chatbook/Agents/agent_service.py), lines 553–584, filters catalog schemas by `allowed_tools`, then independently adds runtime schemas. [`run_skill_script` dispatch](../../../tldw_chatbook/Agents/agent_runtime.py), lines 1581–1595, directly invokes its callback. It does not pass through catalog invocation.

**Failure:** A goal selecting a narrow catalog subset could still advertise or execute a runtime tool, including a script. Filtering only initial schemas also misses later tool loading and restored calls. Existing permission grants can be broader than this particular goal's selected scope.

**Correction:** Use one immutable `GoalToolScope` with explicit catalog/runtime/script/server identities. Enforce it during discovery/loading, schema preparation and actual invocation after waits or continuation restoration. Existing grants remain necessary. Direct CLI execution stays in the first milestone.

**Executor limitation:** The skill runner's scratch directory and resource limits are not a filesystem/network sandbox. Local file-tool confinement does not confine trusted script code. The plan now states the actual executor authority and tests the fixture's expected effects without claiming OS isolation.

**Acceptance:** Unadvertised runtime calls, disallowed tool loading, changed skills, retargeted MCP servers and post-approval scope changes are refused before execution; selected CLI tools still work through existing grants. Slice 2.

## R3 — P1: Idempotent goal creation did not cover conversation creation

**Evidence:** [`ChatPersistenceService.create_conversation`](../../../tldw_chatbook/Chat/chat_persistence_service.py), lines 216–339, creates chat history and separately links workspace membership. [`ChaChaNotesDB.add_conversation`](../../../tldw_chatbook/DB/ChaChaNotes_DB.py), starting at line 7585, accepts a caller-provided UUID, but the current persistence-service method does not expose an idempotent launch operation.

**Failure:** If the UI creates a conversation before calling the goal service, a double Start or crash between stores can leave duplicate conversations or a goal whose history was never provisioned. An AgentRunsDB transaction cannot make these other stores atomic.

**Correction:** Persist a launch intent and preallocated conversation UUID together with the chain. Provision that exact conversation and membership through their existing owners, verifying ownership when reconciling an existing row. Keep the goal in Starting until all bindings are ready. The UI submits the same launch ID on retry.

**Acceptance:** Crash injection around each store write and duplicate/conflicting Start delivery produce at most one owned goal/conversation and zero premature model calls. Slice 1.

## R4 — P1: Checkpoint settlement and recovery needed one authoritative contract

**Evidence:** The [automatic ledger transaction](../../../tldw_chatbook/DB/automatic_work.py), starting at line 135, uses `synchronous=FULL` and rejects nested transactions. Its `recover`, starting at line 889, replaces the singleton owner and fences foreign work. The [fleet coordinator](../../../tldw_chatbook/Chat/console_fleet_wake.py), lines 676–694, already initiates that audit once. `AutomaticWorkContext` also currently permits completed fleet attempts for surviving work, which is not appropriate for a settled childless goal iteration.

**Failure:** Separately saving a report, incrementing the goal and completing its attempt leaves crash windows with inconsistent state. Calling global recovery from a newly created goal service could revoke live fleet work. The former generic `review(accepted: bool)` contract did not distinguish judging output quality from resolving a command that may still be running or may already have changed external state.

**Correction:** Complete the goal checkpoint and accepted attempt in one FULL transaction, idempotent on identical result identity and rejecting conflicting delivery. Preserve previously settled call accounting. Share one startup audit/owner across both coordinators and discriminate attempt kind at every authority boundary. Separate `awaiting_result_review` from `recovery_required`; quality approval applies to one fresh artifact/checkpoint and cannot clear uncertain charges, release a worker or authorize replay. Recovery preserves unknown effects and requires its own typed resolution.

**Acceptance:** Fail each checkpoint write; deliver duplicate/conflicting results; initialize goals during a live fleet; call wake APIs on goal attempts; attempt review on uncertain work; and terminate/restart a real process after acceptance. No case may duplicate effects, refresh allowance or settle another owner's attempt. Slices 2/3/4.

## R5 — P2: Explicit goals inherited unrelated fleet policy and oversized native turns

**Evidence:** [`AutomaticWorkContext`](../../../tldw_chatbook/Agents/automatic_work_runtime.py), lines 85–148, reads `autowake_enabled` and fleet limits in checks, output caps and call admission. [`AutomaticWorkLimits.from_settings`](../../../tldw_chatbook/Agents/automatic_work_budget.py) reads `max_autowake_*`. The [Console bridge](../../../tldw_chatbook/Chat/console_agent_bridge.py) deliberately has large ordinary turn limits; the [agent loop](../../../tldw_chatbook/Agents/agent_runtime.py), lines 1009–1026, reports several budget exits as `stuck` plus prose.

**Failure:** Turning off fleet follow-ups would disable an explicitly enabled goal. Changing only goal preparation would not fix later context checks. A single ordinary Console turn could also use the entire goal allowance before its first checkpoint, undermining the small-iteration workflow.

**Correction:** Keep independent goal enablement and finite policy settings while sharing accounting/capacity. Resolve applicable policy from trusted origin at every check. Add per-iteration limits (initially 8 model turns, 64 steps, 240 seconds) within the total goal allowance, and typed termination reasons. Preserve normal manual and fleet defaults.

**Acceptance:** Exercise both combinations of fleet/goal enablement, live settings changes during a wait, per-iteration exhaustion before the total bound, and helper-call accounting. Partial work at a limit is saved and paused, not blindly retried. Slices 2/4/5.

## R6 — P2: A compact handoff did not guarantee compact provider requests

**Evidence:** [The controller's normal submission path](../../../tldw_chatbook/Chat/console_chat_controller.py), around line 3678, calls `_provider_messages_for_session` and applies further context preparation. The [run-log eviction module](../../../tldw_chatbook/Agents/run_log_eviction.py) explicitly operates only on the outgoing request and preserves the underlying history.

**Failure:** Appending a 16 KiB handoff to a dedicated conversation can still resend every prior iteration and an earlier provider continuation. Testing the prompt builder's length alone would miss the growing actual request.

**Correction:** Give goal origin an explicit request-history builder. Keep transcript rows for inspection, but begin each new native iteration from the bounded handoff and currently authorized context. Keep continuation within an iteration coherent without restoring a settled prior iteration's continuation. Validate the final prepared request after all context injections.

**Acceptance:** Capture actual second/third provider requests and show that old transcript/continuation bodies are absent, mandatory objective/criteria remain intact and the complete payload meets limits. Slices 2/3.

## R7 — P2: Evidence retention targeted the wrong owner and lacked an aggregate bound

**Evidence:** [`run_log_eviction.py`](../../../tldw_chatbook/Agents/run_log_eviction.py), lines 1–25, trims send context and is pure with respect to I/O; it does not retain files. [`LocalSkillsService`](../../../tldw_chatbook/Skills_Interop/local_skills_service.py), around lines 2380–2404, retains script output directories and prunes older runs.

**Failure:** Pinning context would not protect a referenced script artifact from pruning. A 4 MiB per-goal cap also does not bound accumulation across many completed goals. The first plan left these responsibilities ambiguous.

**Correction:** Copy only the small checkpoint evidence into private goal records. Keep large outputs as references with honest unavailable/stale states. Reserve bounded result space before execution and use a default 128 MiB aggregate goal-payload cap. Explicitly remove settled payloads to reclaim space while retaining accounting tombstones and protecting active/uncertain work. Avoid a new run-log pinning framework and do not claim to cap unrelated existing logs.

**Acceptance:** Remove/prune an original artifact, exceed each record/goal/aggregate cap, and remove settled history. Required evidence either remains available in its bounded copy or is visibly unavailable; history cleanup cannot delete workspace files, remove uncertain work or reset accounting. Slices 3/5.

## Delivery improvements and remaining verification

The plan now exercises a real CLI process through the actual controller and agent dispatch in slice 2, before UI construction. The existing skill end-to-end test is retained as regression coverage; its fixture intercepts `AgentService` to capture the bridge closure, so it is not evidence for the complete controller-to-agent route. Slice 5 extends the early fixture into the two-iteration failed-check/edit/passing-check demonstration.

One active goal and zero spawned subagents remain sensible first-release limits. Existing direct CLI/MCP tools remain available. Workflows invocation, repository commit policy and whole external-agent backends remain later integrations.

This review changed only the design, plan, proposed ADR and this report. Verification here is source inspection and document consistency/link checking; no application tests or live model runs were executed. Planned fault-injection, real SQLite, real subprocess, restart and mounted Textual tests remain implementation acceptance work. The shared checkout's unrelated changes and the user's edit in the plan were preserved.
