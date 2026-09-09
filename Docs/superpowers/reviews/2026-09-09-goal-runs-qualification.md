# Native goal runs: implementation qualification

Status: **whole-branch review pending**. All five implementation slices have independent specification and quality approval. One nonblocking form-state finding from Console re-review is carried into the final review and fix wave. The configured local model was exercised and did not complete the task; its observed failure is recorded below.

Scope: the goal feature on `codex/native-goal-runs`, relative to preserved prerequisite baseline `77bc58dc171c3dcd2178f4433d19a6ceb59b1e7b`. The original shared checkout was extensively dirty; its prerequisite changes were preserved separately before implementation. Integration must reconcile those owners and the independent Workflows branch before merging.

Authority: [accepted ADR-141](../../../backlog/decisions/141-native-console-goal-runs.md), [design](../specs/2026-09-08-gnhf-inspired-goal-runs-design.md), [implementation plan](../plans/2026-09-08-gnhf-inspired-goal-runs.md), and [original seven-finding review](2026-09-08-goal-runs-preimplementation-review.md).

The [Console qualification assets](../qa/native-goals/task5/README.md) preserve the actual CLI and endpoint traces, final fixture/diff, rendered controls, targeted test output and baseline diagnostic proofs.

## Original review findings

| Finding | Implemented contract and evidence |
| --- | --- |
| R1: false or stale verification | Launch-bound exact verifier selection, typed process outcomes, latest owned observation per check, input-version manifests, and fresh checkpoint/artifact review. Native tests cover failed exits despite successful invocation, spoofed output, changed arguments/verifiers, edits after checks, and older-pass/newer-failure ordering. See `Tests/Chat/test_goal_cli_verification.py`, `Tests/Agents/test_goal_progress.py`, and `Tests/Agents/test_goal_run_service.py`. |
| R2: runtime tools outside scope | One immutable goal scope narrows discovery, schemas, loading, restored calls, runtime callbacks and invocation after waits. Existing permissions, script trust and provider/MCP identities remain authoritative. Native CLI execution remains supported with its actual executor authority. See `Tests/Chat/test_console_goal_authority.py` and `test_console_goal_dispatch.py`. |
| R3: duplicate conversation creation | Durable launch intent and preallocated UUID precede cross-store provisioning; exact retries reconcile the same conversation/membership. Fault-injection and mounted tests cover partial provisioning, duplicate Start, Retry setup and navigation/shutdown while provisioning is held. See `Tests/Chat/test_goal_conversation_provisioning.py`, `Tests/DB/test_goal_runs.py` and `Tests/UI/test_console_goal_setup.py`. |
| R4: split settlement/recovery authority | Checkpoint, private evidence and matching attempt transition commit together. Fleet and goals share one startup audit. Result review cannot resolve uncertain effects or authorize replay. Actual subprocess restart tests observe zero replacement-runtime provider/tool operations at accepted and checkpointed boundaries. See `Tests/DB/test_goal_evidence_retention.py`, `test_goal_restart_process.py`, and `Tests/Chat/test_console_goal_recovery.py`. |
| R5: policy and budget leaks | First and successor increments share the original goal-origin allowance, including helper calls. Goal/fleet enablement is independent while capacity/manual reserves remain shared. Typed iteration limits pause immediately; waits and Resume retain the original deadline. F9 Save/Revert persists the actual goal policy atomically. See `Tests/Chat/test_console_goal_scheduling.py`, `Tests/DB/test_goal_attempts.py` and `Tests/UI/test_console_goal_settings.py`. |
| R6: growing provider requests | Later native requests contain bounded goal handoff and currently authorized context; retained transcript and previous iteration continuation are excluded from the actual request. Tests inspect second/third gateway requests. See `Tests/Agents/test_goal_memory.py`. |
| R7: unbounded or unavailable evidence | Small evidence records are copied privately with pre-execution reservations and per-record/per-goal/aggregate ceilings. Missing originals remain distinguishable from retained proof. Settled removal preserves accounting tombstones and user files; active/uncertain payloads are protected. See `Tests/DB/test_goal_evidence_retention.py` and mounted removal in `Tests/UI/test_console_goal_controls.py`. |

## Backend review and verification

All four backend slices received independent specification and quality approval. Review fixes included typed wrapper failures, live goal disablement at acceptance, exact verifier selection, immediate pauses for native iteration limits, and shutdown admission/draining under persistence failures.

The continuation slice's broad final gate passed 164 targeted tests. Its final amended scheduling/runtime/goal-and-fleet admission gate passed 78 tests; these scopes overlap and are not additive. Earlier slices' summaries and verification outcomes are recorded in their Backlog implementation notes: TASK-32116 through TASK-32119. No full test-suite sweep was requested or run.

The shutdown regression covers five controlled orderings: a delayed initial read, prepared reservation awaiting return, provider resolution before acceptance, blocked acceptance before the ledger transaction, and committed acceptance awaiting return. All prevent provider dispatch after closure. The committed case retains its generation charge. This does not establish atomic cancellation against every cross-thread timing between a guard and SQLite commit.

Restart qualification uses real isolated SQLite files, actual native adapter operations, a trusted CLI subprocess, and an owned child process terminated at observed durable boundaries. It demonstrates no redispatch on process restart; it does not certify power-loss behavior or exactly-once external effects. Stop remains cooperative and retains ownership through actual provider/tool cleanup.

Scoped lint/format checks passed without new diagnostics in touched legacy code. The shared environment emits an existing `RequestsDependencyWarning`; one earlier run also exposed existing splash-screen invalid-escape warnings. No unrelated dependency or splash changes were made.

## Configured local model

The available OpenAI-compatible endpoint at `http://127.0.0.1:9099/v1` advertises `Qwen2.5-0.5B-Instruct`. Chatbook used its existing `llama_cpp` adapter and AgentService fenced tool protocol. The adapter name does not establish the serving binary or model-weight provenance.

Two setup attempts stopped at the erroneous provider-native-tools capability check before making any HTTP/model call. That check was narrowed to the actual AgentService runtime contract. The first networked run then made two HTTP 200 model calls but dispatched no tools. Inspection identified conflicting instructions for intermediate tool calls and the final report; the final-report wording was clarified without relaxing parsing or evidence requirements.

One bounded retest sent the corrected instructions in both actual requests. It again made two accepted iterations/model calls, received malformed reports, dispatched no tools, and paused with `no_progress`. One response recommended completion despite having no verification evidence. The evidence gate rejected that recommendation. The recorded final file was still `invalid\n`, the diff was empty, and the external sentinel was unchanged.

Each networked run allowed at most two iterations, eight model calls, 50,000 budget tokens, 1,024 output tokens per call and 60 elapsed seconds; iteration caps were four turns, 32 steps and 30 seconds. There was no cloud fallback or model download. These traces establish transport and failure handling for this endpoint, **not successful autonomous CLI execution by this model**. The deterministic provider fixture's real CLI correction is separate evidence.

## Console qualification

Mounted tests exercise actual Start, interrupted-setup retry, lifecycle controls, draft preservation, Settings Save/Revert and exact checkpoint/run navigation through the existing runtime. The final verification pass exposed a Pause event lost during an awaited state read; its fix preserves the event and the saved retry deadline. Review selection and duplicate Start received additional targeted regressions before the independent gate.

The amended affected gate passed **98 tests**, with one intentional opt-in live skip. Subsequent selected-review/removal/navigation checks passed **10 tests**, and final setup controls passed **6**; these runs overlap. The earlier broad gate's untouched Change Review and Settings cases had passed. Scoped Ruff and formatting passed, and six large incumbent files add no lint diagnostics compared with their baseline.

The Console review then identified missing cold-restart conversation hydration, tool choices frozen from the first project binding, and a stale launch-review summary. A focused root check also found older goals unreachable behind the newest 50 entries, including removed tombstones. Commit `e98f6c68aa` adds runtime-owned Resume hydration, current-binding discovery with stale-result rejection, the frozen submitted selection summary, and bounded Older/Newer history controls. The [affected regression gate](../qa/native-goals/task5/review-fix-affected.txt) passed **84 tests**, followed by **6** strengthened, overlapping checks; eight-file Ruff and formatting passed. Independent re-review approved all four findings. A remaining nonblocking form-state issue can disable tool choices after a project changes during validation; final review will carry it into the fix wave.

The [actual CLI trace](../qa/native-goals/task5/deterministic-cli.json) records the trusted verifier exiting **7**, an authorized `fs_edit`, and the same verifier exiting **0** across two increments and five charged model calls. Root inspected the [final file](../qa/native-goals/task5/fixture.txt), [diff](../qa/native-goals/task5/fixture.diff) and verifier source/hash: the result is exactly `valid\n`, and the check remained unchanged. This uses a deterministic provider with real tools, process execution, controller and SQLite. It is POSIX local-skill qualification; the process retains its existing host authority.

The setup and review modals were rendered using the production CSS bundle at **80×24 and 160×44**, with keyboard focus and fixed action-row access checked. One layout correction moved objective/criteria before long authority details, removed the empty optional source list and clarified wait reasons. Root inspected the four regenerated captures. The 80×24 review capture shows its body scrolled while actions remain visible. The dimmed backdrop retains the fixture's fresh-profile Console setup state, so these captures qualify modal layout, not a complete operational rail walkthrough.

## Inventory baseline

Before Console implementation, the diagnostic scan found two goal-related owner changes: the new coordinator's Stop-persistence warning and the runtime's goal-drain warning. It found no persistent-sink topology change.

The remaining non-goal mismatch is `tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py`: its 50-call inventory digest is `f7faf8b3c1e797ef8684`, while scanning the preserved source yields `95d2b6a88660ad3919a2`. That source is byte-identical to baseline `77bc58dc17` (SHA-256 `c22919b572b227e2d80565f6dec6a5c4b0bcd59cdea66432a3b03e27b9652b92`). It predates all goal commits and requires the prerequisite owner's review. The final goal slice must update its own inventory entries and disclose this existing mismatch if it remains; a failing whole-manifest check is not a passing result.

The final source scan confirms the updated goal entries/counts match and TTS is the only manifest mismatch; top-level metadata and persistent-sink topology match. The broader architecture gate returned **97 passed, 3 failed, 1 skipped**. The other two failures pin diagnostic labels that were already absent in the preserved bridge/fleet code. Executing the unchanged tests' exact source assertions against both baseline source copies and the current tree reproduced identical lists of seven and one missing labels. The skipped historical test requires two unavailable commits; current-source assertions still ran.

This gate also emitted existing invalid-escape warnings from `Tools/patch_tool_impls.py` and `Utils/Splash_Screens/environmental/train_journey.py`; both files are byte-identical to the feature baseline. These findings were classified, not silently repaired or treated as passing checks.

## Remaining review

- Whole-branch independent review of all seven original findings and their cross-module contracts.
- Resolve the remaining Console form-state finding with any final-review fixes.
- Final Backlog/plan bookkeeping and integration handoff after those gates.

## Implementation decisions

- **Preserved prerequisites:** isolate the reviewed dirty baseline before feature work. Cost if wrong: integration must resolve those prerequisite changes before the goal branch can be merged.
- **Human judgment:** immutable `human_review_required=True` by default; explicit false requires launch-bound checks and declares them sufficient. Cost if wrong: changing that decision requires a new launch, and objective checks only establish what they actually test.
- **Exact verifier selection:** reject indistinguishable invocation declarations while preserving historical request readability and exact replay bytes. Cost if wrong: a user must combine the declaration or use distinguishable arguments in a new launch.
- **Settings ownership:** use canonical F9 Console behavior's draft and atomic Save/Revert for actual `[agents]` goal keys. Cost if wrong: the feature's placement must change; named-agent SQLite CRUD and manual zero/unlimited semantics remain separate owners.
- **Retry proof:** only trusted adapter-local rejection before dispatch qualifies for automatic retry; no generic remote-rate-response inference. Cost if wrong: some provider failures require explicit handling instead of retry, with conservative accounting retained.
- **Restart review:** explicit Resume gates another increment, while a settled result already awaiting review can still be decided without another call. Cost if wrong: recovery UI must be revised; freshness and uncertainty checks remain mandatory.
- **Diagnostic baseline:** update goal-owned inventory entries and retain the unrelated TTS mismatch for its prerequisite owner. Cost if wrong: the global manifest gate remains red until that separate change is reviewed; this qualification must not claim a clean global result.
- **Verifier setup:** let the existing skill owner expose a read-only immutable verifier reference; the form does not manufacture trust or grant permission. Cost if wrong: the setup seam needs revision, while existing execution-time checks still reject stale authority.
- **Launch lifetime:** the coordinator owns one bounded setup task through provisioning and hydration; a view only waits on it. Cost if wrong: cancellation and shutdown need rework, and the durable launch identity must still prevent duplicate conversations.
- **Provider protocol:** allow the existing AgentService fenced tool path as well as provider-native function calls. Cost if wrong: a provider may fail to follow the tool/report protocol; it must pause or fail without gaining a plain-provider execution path.
- **Report instructions:** constrain the final iteration report to strict JSON while explicitly allowing intermediate tool replies in the existing protocol. Cost if wrong: models may still fail the protocol; runtime evidence and the strict parser remain the completion authority.
