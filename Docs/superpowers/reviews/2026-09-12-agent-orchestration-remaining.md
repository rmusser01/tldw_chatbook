# Agent orchestration remaining work: final review and decisions

The four integration findings are fixed in `7852cf47ba` and passed the one
independent scoped re-review. Definition wall caps are also implemented and
independently reviewed across storage, runtime and canonical Settings. All seven
actual TASK-13154 child tasks are Done. The parent remains In Progress because
functional worktree merge confirmation (TASK-31210) and prior-turn recovery
(TASK-31211) still lack a qualified execution boundary. Current agent-driven
worktree create/merge/discard refuses, while existing checkouts and branches are
retained. Ordinary agents remain available.

Current delivery status is maintained in the
[workstream inventory](../../../backlog/docs/agent-orchestration-followups-2026-09-12.md).
The working branch is `codex/agent-orchestration-remaining`; this continuation
created local commits, with no new PR, push or merge.

## Final verification

The final affected continuation module passed 45 tests with no teardown errors;
the adapter accounting/provider-error selection passed 13, and the runtime
denial/cancellation selection passed 6. The new telemetry fault regression had
2 actual pre-fix failures, followed by 2 passes. These runs overlap and are not
an aggregate suite count. The diagnostic inventory matches the final source;
scoped static checks add no diagnostics and every edited formatting range
passes. These are targeted checks, not a full-suite or live-provider result.

The inherited RequestsDependencyWarning and separate baseline-controlled modal
inventory failures remain qualified. Earlier preset probes outside Tests read
local configuration before a sandboxed lock-file append failed. An earlier cap
test run without the required basetemp caused pytest to attempt and fail foreign
garbage-tree removal. No zero-prior-side-effect claim is made for those incidents.
Later tests used the mandatory isolated runner with unique owned basetemps; no
manual foreign cleanup was performed. A host-Python3.9 AST-check failure is also
preserved separately from the successful supported-Python diagnostic check.

All source workspaces and raw evidence are retained. The sections below preserve
the original review and decision text, including duplicate or superseded rulings.
The original review's four open findings describe its pre-correction checkpoint;
the following scoped review gives their final disposition. The decision table
separates shipped behavior from rejected or unimplemented worktree proposals.

## Scoped correction review — final disposition

**Ordinary budget stops capture a different continuation boundary** — ADDRESSED. `tldw_chatbook/Agents/agent_runtime.py:3081-3097` preserves post-batch cancellation precedence and advances `coherent_len` only inside actual denial-circuit termination. The unchanged ordinary loop-top contract remains pinned at `Tests/Agents/test_fleet_continuation.py:431-459`. Retained evidence shows the complete continuation module passed (`45 passed`, no teardown errors) and six denial/cancellation/history neighbors passed.

**Two continuation tests still require prohibited worktree admission** — ADDRESSED. `Tests/Agents/test_fleet_continuation.py:915-938` seeds a genuine coordinator-retained isolated transcript through reserve/attach/finish. `Tests/Agents/test_fleet_continuation.py:941-989` exercises the actual continuation call and verifies no Git call, provider child execution, worktree allocation, or shared-tree fallback; it also preserves the original history/handle and checks the failed resumed row and refusal returned to the parent. `Tests/Agents/test_fleet_continuation.py:992-1052` independently verifies shell/virtual-CLI filtering and deterministic settlement without a child script. The retained module gate passed all 45 cases without teardown or unused-script errors.

**Per-chunk telemetry can abort a successful provider stream** — ADDRESSED. `tldw_chatbook/Chat/console_agent_bridge.py:3626-3653` contains snapshot and output-count extraction exceptions around optional per-chunk observation, disables further attempts for that call after the first fault, emits a fixed content-free warning, and continues the normal text append path. The catch is `Exception`, so cancellation/system-exit semantics are not absorbed, and genuine provider failures still propagate before the unchanged terminal accounting containment at `tldw_chatbook/Chat/console_agent_bridge.py:3691-3717` and `:3746-3785`. `Tests/Chat/test_console_agent_bridge.py:11123-11192` reaches an attributed two-chunk provider text stream for both fault types and asserts complete response/store text, started/finished observation, cleared adapter state, and one new per-call warning. Retained evidence records an actual `2 failed` RED followed by `2 passed` GREEN; the 13-case adapter gate also passed, including the genuine-provider-failure control.

**Required production diagnostic inventory is stale** — ADDRESSED. `Docs/security/production-diagnostic-inventory.json:326-330`, `:452-456`, and `:11565-11571` now record bridge 35, gateway 19, and total TASK-492 1391. The retained statement review identifies exactly three added fixed-string warnings, no `agent_runtime.py` diagnostic change, and no sink change; the new per-chunk warning is bounded by disabling observation after its first failure. The final supported-interpreter `--diff` exited 0 with no drift: 606 owners, 1391 TASK-492 calls, 55 TASK-31551 calls, 7721 TASK-494 calls, and 12 sink files.

### New Breakage in the Fix Diff

None. The scoped static report records no added diagnostics in any of the four changed Python files and no edited-format failures; removal of eight test diagnostics follows from replacing the obsolete positive-worktree tests. `git diff --check` is reported clean. I did not rerun tests because the retained raw outputs directly cover the amended behavior and named controls.

### Out-of-Scope Observations

- Functional worktree execution/recovery remains incomplete by deliberate policy; this fix correctly tests the current refusal boundary and does not satisfy TASK-31210/TASK-31211 or close the parent program.
- The sole warning in each retained green pytest gate is the inherited `RequestsDependencyWarning`; it is not pristine-output evidence and does not arise from this fix.

### Verdict

**Fix round:** All four findings addressed, no new Critical/Important breakage. Parent closure remains out of scope and must not be claimed from this correction wave.


## Correction evidence and implementation report

# Final combined correction wave report

Baseline reviewed: `564e5f27db2617da7f3ef15288d6ffdb974263d4`.

## Changed files and causal repairs

- `tldw_chatbook/Agents/agent_runtime.py`: advances the extra coherent transcript capture only on actual denial-breaker termination. Post-batch cancellation retains precedence and ordinary loop-top budget termination again uses the established drain boundary.
- `Tests/Agents/test_fleet_continuation.py`: preserves the unchanged ordinary budget-boundary regression and replaces the two obsolete successful-worktree-resume contracts. The replacements create a durable prior run, reserve/attach/finish an isolated handle through the coordinator API, retain its real transcript, then exercise `send_to_agent`. They assert the closed boundary invokes neither Git nor a child provider/shared-tree fallback, leaves original history and handle unchanged, writes an exact failed resumed row, returns the refusal to the parent, filters shell/virtual CLI from the attempted child config, and settles with no unused child script.
- `tldw_chatbook/Chat/console_agent_bridge.py`: catches snapshot and output-count extraction exceptions only around optional per-chunk provider-usage observation. The first fault emits one fixed, content-free warning and disables further per-chunk attempts for that call; transcript append, provider completion, outer finished observation, genuine provider exceptions, and terminal accounting keep their existing paths.
- `Tests/Chat/test_console_agent_bridge.py`: adds an actual attributed `ConsoleProviderGateway` text stream for both snapshot and extractor faults. It requires the complete response/store text, started/finished observations, cleared adapter state, and exactly one new per-call warning.
- `Docs/security/production-diagnostic-inventory.json`: regenerated from the final source using the supported isolated Python 3.12 interpreter with `-I`; no ratchet changed.

## RED and GREEN evidence

- New telemetry RED: `final-evidence/final-telemetry-red`. Exact mandatory runner selected `Tests/Chat/test_console_agent_bridge.py::test_live_usage_chunk_observation_failure_does_not_truncate_provider_text`; both parameters failed before source repair because `RuntimeError` escaped the per-chunk path. Exit 1, `2 failed, 1 warning in 1.04s`.
- Telemetry GREEN: `final-evidence/final-telemetry-green`. Same exact node, exit 0, `2 passed, 1 warning in 0.46s`.
- Required affected continuation module after final formatting: `final-evidence/final-continuation-module-after-format`. Exit 0, `45 passed, 1 warning in 6.17s`; zero failures and zero teardown errors. The earlier pre-format successful run is separately preserved at `final-evidence/final-continuation-module`.
- Denial/cancellation/history neighbors: `final-evidence/final-runtime-neighbors`. Six exact nodes across provider continuation and ordinary runtime cancellation, exit 0, `6 passed, 1 warning in 0.34s`.
- Final adapter accounting/error control: `final-evidence/final-adapter-accounting-after-format`. Thirteen cases covering the new two-parameter text test, terminal snapshot containment, final signal fallback, post-stream accounting failure, genuine provider failure, and raw provider-output validation; exit 0, `13 passed, 1 warning in 1.58s`.

Every pytest command above used `.superpowers/sdd/2026-09-12-agent-orchestration-remaining/run_pytest.py` with a unique label and a `Tests/...` selection. The only warning in each green gate is the inherited `RequestsDependencyWarning`. The required continuation module produced no teardown error, unused script error, worker leak report, or resource warning. Only fixture-owned lifelines/workers were joined; no foreign cleanup was performed.

## Diagnostic inventory

The final statement review is preserved in `final-evidence/final-diagnostic-statement-review` with exact argv/stdout/stderr/exit. Relative to inventory revision `5e7dac433a6957c77f47115ce60d15be419d68cb`:

- `agent_runtime.py`: 9 -> 9, no statement change.
- `console_agent_bridge.py`: 33 -> 35, two fixed warnings: existing live-usage callback containment and new per-call provider-usage observation containment.
- `console_provider_gateway.py`: 18 -> 19, the previously reviewed fixed emission-observer warning.

All three additions are fixed strings. They interpolate no user content, secret, path, URL, exception text, provider data, or payload, and introduce no sink change. The new warning is bounded once per attributed call by `live_provider_usage_enabled = False` after the first fault.

The supported interpreter `--write` operation is preserved at `final-evidence/final-diagnostic-inventory-write`, exit 0. The final `--diff` operation is preserved at `final-evidence/final-diagnostic-inventory-diff`, exit 0: `606 owners, 1391 TASK-492 calls, 55 TASK-31551 calls, 7721 TASK-494 calls, 12 sink files`; committed inventory and rebuild match exactly.

## Scoped static and self-review

`final-evidence/final-scoped-static/report.json` compares all four changed Python files with `564e5f27db`:

- `agent_runtime.py`: 19 -> 19 diagnostics, no tuple deltas, no format failures.
- `test_fleet_continuation.py`: 22 -> 14 diagnostics, no additions; removed two inherited F811 tuples, and two occurrences each of three RUF059 tuples from the obsolete positive tests. No format failures.
- `console_agent_bridge.py`: 27 -> 27 diagnostics, no tuple deltas, no format failures.
- `test_console_agent_bridge.py`: 20 -> 20 diagnostics, no tuple deltas, no format failures.

`git diff --check` exited 0. The final working tree contains only the five owned source/test/generated-artifact files. No staging, commit, Backlog/plan/lesson edit, provider/network call, config mutation, dependency change, full suite, guard increase, or worktree cleanup was performed.

Self-review found the source changes confined to the two causal boundaries. The runtime assignment remains after the post-batch cancellation check, so denial termination captures the fully settled batch without masking cancellation. The adapter catches `Exception` rather than `BaseException`, keeping cancellation/system-exit semantics out of telemetry containment. The final accounting block remains unchanged and is exercised together with a real provider-failure control. Worktree tests do not fake admission: the coordinator history exists before the real continuation call, while the real service refusal creates the new failed database row and terminal coordinator handle.

Residual qualifications are unchanged: the Requests dependency warning is inherited; no full-suite or live-provider claim is made; functional worktree execution/recovery remains intentionally unavailable and outside this correction wave.


## Original broad review — before correction

# Whole-branch code and integration review

Reviewed BASE `d66908a69ef03066fed77a92edf77a326f44bd89` through HEAD `564e5f27db2617da7f3ef15288d6ffdb974263d4`. HEAD was verified read-only. Used the supplied 60-commit combined diff, current source, tests, plans/ADRs, current-status document, ruling extract and retained evidence. No duplicate diff generation, source mutation, product imports, test runs, environment changes, cleanup or helpers. The only write is this report.

## Strengths

- The worktree prerequisite consistently refuses automatic create/merge/discard and retains existing checkouts and branches. Service routing retirement remains independent of the handle record. Preview and live schema planning share the omission. This is an appropriate interim boundary, not completed recovery.
- Definition caps are validated at CRUD, normalize present fingerprint values while preserving absent fingerprints, apply after helper floors, and propagate the admitted ceiling through coordinator retention. Using `replace` in child helpers also preserves retry, warning and denial settings. The continuation clock remains per invocation; automatic accounting and physical ownership remain separate.
- Log reads now retain bounded fragments, seek over unrelated bodies, verify complete frames, retain only cursor history, and perform target resolution/availability work outside the UI thread. The bridge reacquires scratch authority per chunk/page; publication checks exact selection and bridge identity.
- Approval metadata preserves raw compatible accessors and exact-owner key semantics. Denial accounting is run-local, uses authoritative results rather than error text, and allows an approved tail to reset the streak. The noncancelled restored-pending test now proves another model call is reached.
- Webhook admission is bounded and nonblocking for queue capacity, captures configuration, uses one Runner per generation, and does not mark retirement complete until the Runner closes. Settings explicitly distinguishes owned/borrowed DB lifetime and reports runtime-tool omissions, including the all-filtered inheritance consequence.

## Issues

### Critical

None verified.

### Important — must address before declaring this branch integration clean

1. **P2: ordinary budget stops capture a different continuation boundary.** `tldw_chatbook/Agents/agent_runtime.py:3083` updates `coherent_len` after every completed batch, before checking whether the denial breaker actually trips. A normal successful calculator round that reaches `max_steps=3` now returns that round in `final_messages`, whereas the previous contract captures exactly the last model/drain boundary (`:1718`, after loop-top cancellation/budget checks). `Tests/Agents/test_fleet_continuation.py:433` explicitly pins that contract and its `:461` assertion fails in the retained full-module output. The test AST is unchanged from BASE; the supplied diff shows this unconditional write was added by this branch. This is a branch regression, not inherited failing coverage. Move the extra capture into actual denial termination, preserving cancellation precedence and the full settled native/fence denial batch. Keep the unchanged budget-boundary regression and add/retain adjacent cancellation and denied-batch history evidence; do not redefine the test to bless unrelated semantics.

2. **P2: two continuation tests still require prohibited worktree admission.** `Tests/Agents/test_fleet_continuation.py:918` and `:970` script an isolated child running twice, wait for its retained transcript, and expect fresh worktree admission on resume. Current `AgentService._admit_agent_worktree` (`tldw_chatbook/Agents/agent_service.py:4123`) deliberately refuses before Git/child execution. Consequently the tests fail while waiting for nonexistent successful history and leave unused parent/child scripts, producing the two teardown errors in the retained module gate. This is incomplete test integration caused by this branch's intentional capability removal, not a reason to restore unsafe admission. Replace these positive expectations with active, exact refusal coverage: seed retained isolated history through the coordinator for the resume case; assert refusal/no shared-tree fallback, no child provider execution or Git admission, original retained history preservation, honest failed-row/handle outcome, and deterministic harness settlement. Keep ordinary continuation/tool-containment coverage. No skips, blanket deletion of the contract, or fake successful admission.

3. **P2: per-chunk telemetry can abort a successful provider stream.** `tldw_chatbook/Chat/console_agent_bridge.py:3625` calls `call_signals.usage_snapshot()` and `_provider_output_count()` outside observation containment. A snapshot/extraction exception on an attributed text stream exits `_consume` before `append_stream_chunk` at `:3641`, propagates through `future.result()` at `:3684`, and becomes a failed model call. `_emit_live_usage` only isolates the sink; the protected terminal block at `:3748` is never reached. The terminal fault test at `Tests/Chat/test_console_agent_bridge.py:11063` yields only a `ProviderToolCalls` sentinel, whose early `continue` bypasses the faulty per-chunk path, so it cannot qualify this case. This is a current-wave observational seam, even though the scoped ledger called it “unchanged” relative to that later fix round. Severity is P2: it requires an observation implementation fault, not routine valid provider data, but turns optional telemetry into an execution dependency in violation of ADR-156. Contain extraction separately with bounded content-free diagnostics and disable repeated failed observation for the call. Test a real attributed adapter consuming text while the snapshot/extractor raises, asserting complete text, successful response, finished/cleared observation state, and bounded diagnostics. Preserve real provider exceptions and existing final accounting semantics.

4. **P2: required production diagnostic inventory is stale.** `Docs/security/production-diagnostic-inventory.json:326` still records bridge count 33/digest `a9bf40cc6440a639630b`; `:452` records gateway count 18/digest `1828d489dec64c4989a0`, and `:11570` records total 1388. The controller's supported-Python AST-only gate reports actual counts 34/19/1390 and exits 1; I read its exact stderr/exit in `final-evidence/diagnostic-inventory-python312`. Current source includes the two fixed warnings at `tldw_chatbook/Chat/console_agent_bridge.py:3164` and `tldw_chatbook/Chat/console_provider_gateway.py:5102`. They are content-free and bounded per call; the root's exact statement review confirms only these two intended additions and no sink-topology change. This is derived-artifact integration drift from this branch, not a diagnostic privacy defect. Refresh the inventory after the telemetry repair's final diagnostic shape and rerun its supported-interpreter check without increasing any ratchet. The earlier system-Python 3.9 attempt stopped parsing unrelated existing match syntax; it supplied no inventory result and is separate from the later confirmed drift.

### Minor / inherited qualifications

- Existing RequestsDependencyWarning, foreign pytest garbage-directory warnings and module-wide formatting debt remain inherited/environment qualifications. No dependency edits, foreign cleanup or ratchet increases are justified by this review.
- The three modal inventory assertions remain separately baseline-controlled failures (launch declarations, transitive/AST inventory, expected 15 versus actual 16 contract rows). Do not present them as newly introduced solely because the branch edits the viewer, or present the whole modal inventory as green.
- Historical missed preimplementation RED and sensitivity-only controls remain process limitations; later success does not retroactively create a RED/GREEN sequence. The actual continuation module gate is **41 passed, 3 failed, 2 teardown errors**, not green or an incomplete run. No pristine-base execution is claimed.
- The unsafe outside-Tests preset probes read real config before a blocked lock-file append. The default-basetemp gate attempted and failed foreign garbage-tree removal. Both incidents remain disclosed; neither is grounds for cleanup or a claim of zero prior side effects. Current isolated evidence is qualified on its own.

## Mandatory/deferred triage

Mandatory observation 1 is issue 1; observation 2 is issue 2; observation 3 is issue 3. The source-only reasoning and existing failed output are sufficient to identify these issues, so this review did not rerun tests merely to reconfirm them.

Deferred denial combined-hook annotation is fixed: `build_combined_review_hook` and its nested callable accept `(calls, run_id)` and `ToolReviewValue`. Deferred restored-pending coverage is fixed by `Tests/Agents/test_provider_continuation_runtime.py:1871`, which uses limit 1, no cancellation masking, and requires one further model call and `RUN_DONE`. Usage snapshot docstring, fleet snapshot annotation and token docstring are corrected. Settings hardcoded capture writes were removed; current paint tests retain in-memory assertions and owned historical captures. Webhook ownership/test-report fixes are recorded with exact-generation teardown; no additional product finding was established. All remaining warning/process minors above are retained, not silently cleared.

## Disposition of all 50 source rulings

References below identify the original `progress.md` line numbers quoted in `final-rulings-source.md`. “Accepted” means the decision is coherent with the current source and scope; it does not imply that this review reran its historic verification. Duplicate rulings are explicitly mapped.

| Workspace / source lines | Disposition |
| --- | --- |
| orchestration-approval-verification 12 | Accepted bounded harness repair; source retains real answerable-clock, focus/geometry and worker-finally assertions. No product permission widening. |
| orchestration-approval-verification 13 | Accepted serialized root Git ownership; no review Git mutation. |
| bounded-console-run-log 13 | Accepted actual filesystem segment viewer source; no unnecessary DB storage rewrite. |
| bounded-console-run-log 14 | Accepted fragment, UTF-8, sparse-segment and complete-frame contract; real-file reconstruction/read instrumentation assertions inspected. |
| bounded-console-run-log 15 | Accepted source serialization and evidence retention. |
| bounded-console-run-log 28 | Accepted optional cancellation callback, checked between bounded chunks. |
| bounded-console-run-log 30 | Accepted UI-only target predicate, captured bridge/run; authority remains in bridge reader. |
| bounded-console-run-log 32 | Accepted completed-negative one-second retry; positive cache and collapsed no-I/O behavior retained. |
| bounded-console-run-log 42 | Accepted I/O-free map token and worker-only metadata resolution; primary writer binding supplies current identity. |
| bounded-console-run-log 45 | Accepted exact-generation queued/running cancellation settlement; no claim of forcibly stopping an entered read. |
| agent-denial-breaker 25 | Accepted narrow model fixture and actual three-tool trace assertions. |
| agent-denial-breaker 35 | Accepted owner provenance extension across builtin/raw/virtual paths; opaque runtime policy refusals remain unannotated. |
| agent-denial-breaker 36 | Accepted bare-context fixtures and behavioral MCP refusal/audit coverage; no weakened composition guard. |
| agent-denial-breaker 45 | Accepted frozen stamp helper/raw accessors; selected exact/name key remains owner-specific and metadata cannot grant permission. |
| agent-denial-breaker 48 | Accepted default profile fixture and exact-only raw/managed/virtual selection. |
| agent-denial-breaker 55 | Completed annotation carry into controller, verified above. |
| agent-denial-breaker 80 | Completed independent noncancelled restoration regression, verified above. |
| agent-worktree-recovery 26 | Accepted test-only nested-interpreter prerequisite; it does not qualify a new Git authority backend. |
| agent-worktree-recovery 34 | Rejected absolute/common-Git-directory reopening remains rejected. No production fallback should be revived. |
| agent-worktree-recovery 40 | Accepted preserved scratch/test prerequisite and independent task ordering. Its “no ADR158 created” statement is historical, superseded by line 52. Functional tasks remain open. |
| agent-worktree-recovery 52 | Accepted ADR158 ordering: shipped definition cap 18→19, future qualified recovery 19→20. No placeholder migration. |
| agent-worktree-interim-safety 15 | Accepted uniform refusal and retention; not functional recovery completion. |
| agent-worktree-interim-safety 19 | Correct decision to replace unreachable positive templates with active boundaries; incompletely propagated to continuation tests, issue 2. |
| agent-worktree-interim-safety 23 | Preview and inline refusal corrections present; exact failed-task persistence remains necessary in amended continuation evidence. |
| live-per-run-usage 18 | Accepted exact service run scope and pre-step primary/child observation; no finished-ID registry. |
| live-per-run-usage 25 | Accepted optional gateway synthetic emission observer, excluding synthetic strings without provider-specific inference. |
| live-per-run-usage 41 | Accepted existing secondary wrapping for live labels; no new UI clock. |
| live-per-run-usage 42 | Accepted tuple-return fixture correction; no production transcript guard change. |
| live-per-run-usage 50 | Accepted exact fixture-owned DB/lock closure. Zero regular-FD growth does not mean zero socket/pipe residuals or pristine BASE. |
| agent-settings-followups 31 | Accepted observed preset-row/instructions viewport repair, scoped to Agents. |
| agent-settings-followups 35 | Accepted compact input widths and owning rows; source keeps local CSS scope. |
| agent-settings-followups 37 | Accepted three-line Switch owner with actual state/clip assertions, not border-glyph proof. |
| definition-wall-cap 19 | Accepted duplicate of recovery 52; ADR158 is authoritative for ordering. |
| definition-wall-cap 21 | Accepted exact inherited runtime-tool set assertion repair, not catalog permission change. |
| agent-fleet-program-closeout 16 | Accepted partial record reconciliation, all actual children without invented child; parent stays open pending integration and recovery. |
| agent-orchestration-remaining 16 | Accepted independent plans/review gates within authorized work; not blanket external/destructive authority. |
| agent-orchestration-remaining 17 | Accepted isolated interpreter and targeted evidence policy; obeyed by this review, no test run needed. |
| agent-orchestration-remaining 18 | Accepted workspace/evidence preservation; no cleanup. |
| agent-orchestration-remaining 41 | Parked functional logical-discard/CAS design only. Current automatic discard refuses on every platform. Not a shipped capability. |
| agent-orchestration-remaining 42 | Parked recovery physical-drain/sticky-uncertainty design. Existing execution-owner accounting remains relevant, but no recovery admission proof exists. |
| agent-orchestration-remaining 43 | Parked Git environment/preview/patch design only. It must be reconsidered with qualified execution; no current agent Git path is enabled. |
| agent-orchestration-remaining 44 | Accepted bounded scalar/sequence usage and strict explicit integer zero; extraction containment still needs issue 3. |
| agent-orchestration-remaining 45 | Accepted unsaved presets and parent tool intersection; empty allowlists inherit only available parent tools, not invented MCP names. |
| agent-orchestration-remaining 46 | Accepted post-floor definition minimum, helper `replace`, retained capped lineage and absent fingerprint compatibility. |
| agent-orchestration-remaining 49 | Parked patch-cap timing refinement supersedes older draft wording only. No child capture commit or destination mutation capability is shipped. |
| agent-orchestration-remaining 53 | Accepted duplicate of bounded log 28. |
| agent-orchestration-remaining 55 | Accepted duplicate of bounded log 30. |
| agent-orchestration-remaining 57 | Accepted duplicate of bounded log 32. |
| agent-orchestration-remaining 64 | Accepted duplicate of bounded log 42. |
| agent-orchestration-remaining 67 | Accepted duplicate of bounded log 45. |

## Recommendations and assessment

Use one combined correction wave for the four issues, with narrow regression evidence under the mandatory runner. Preserve successful isolated gates, failed historical gates and current workspaces. No broad retest, dependency change, unsafe Git workaround or fixture bypass is warranted. Refresh/check the required generated diagnostic metadata after the correction rather than increasing limits.

**Ready to merge? No, at this checkpoint.** The ordinary continuation regression, telemetry containment gap, obsolete continuation tests and stale diagnostic inventory need correction and review. Even after those fixes, this branch must remain an explicitly partial delivery: TASK-31210/31211's qualified worktree execution/recovery and the full parent program outcome are still unfinished. The current refusal/retention policy is intentional and should remain enabled until a valid execution design is demonstrated.


## Original controller rulings — preserved verbatim

# Source rulings for the current remaining-work wave

Exact ledger wording is preserved, including historical decisions later superseded. All workspaces and underlying evidence are retained. Previous PR2641 follow-up rulings are outside this wave. Final review explicitly triaged all 50 entries; the scoped correction added no new policy ruling.

## orchestration-approval-verification

Source: .superpowers/sdd/2026-09-12-orchestration-approval-verification/progress.md:12

Ruling: User's instruction to address all listed tasks authorizes the described bounded harness repair, so no repeat permission gate. Repair setup and contract assertions while preserving production guards; any new product defect requires a concrete root-cause scope update. Cost if wrong: a harness-only repair might conceal a product issue, so retain rendered geometry/focus and meaningful clock tests.

Source: .superpowers/sdd/2026-09-12-orchestration-approval-verification/progress.md:13

Ruling: Root serializes all commits. Workers leave unstaged edits, avoiding the shared-index hazard recorded in lessons-backlog-hygiene. Cost if wrong: task commit boundaries could mix; inspect exact paths before/after each commit.

## bounded-console-run-log

Source: .superpowers/sdd/2026-09-12-bounded-console-run-log/progress.md:13

Ruling: filesystemlogs are actualviewer source, so page those rather than thealready-fixedDBtable — retainsfullstoredcontent — wrongchoicewould hide orduplicateevidence.

Source: .superpowers/sdd/2026-09-12-bounded-console-run-log/progress.md:14

Ruling: fragments preserveoversizedUTF8records, sparse segment discovery andcomplete-recordterminatorvalidation — acceptedcodeccontract — wrongboundswould truncateorstall; realreadinstrumentationtestsare required.

Source: .superpowers/sdd/2026-09-12-bounded-console-run-log/progress.md:15

Ruling: rootcommits and sourceworkers runsequentially; preserveledgerartifacts — sharedindex/evidencelessons — wrongscopingwould mixcommits.

Source: .superpowers/sdd/2026-09-12-bounded-console-run-log/progress.md:28

Task 2 Ruling: optional keyword-only cancellation callback on run_log_available preserves existing bool callers while allowing the new UI worker to stop before a subsequent bounded metadata chunk. Task 1 had no cancellation source. This executes the existing spec requirement without changing authority; cost if wrong is premature false availability or continued stale scanning, covered by gated cancellation and stale-publication tests.

Source: .superpowers/sdd/2026-09-12-bounded-console-run-log/progress.md:30

Task 2 Ruling: optional modal target_is_current predicate runs only on the UI thread before admission/publication; production captures exact bridge/run while standalone callers default true. Reader authority remains independent. Cost if wrong is stale content publication or unnecessarily blocked navigation; gated target replacement and usable Close must be covered.

Source: .superpowers/sdd/2026-09-12-bounded-console-run-log/progress.md:32

Task 2 Ruling: confirmed inherited negative availability cache has no write/lifecycle invalidation, so a first record arriving after the probe can remain hidden forever. TASK18601 AC5 added before implementation. Use a one-second negative retry deadline on existing expanded ticks, beginning only after completion, with exact-generation pending state preventing overlap. Positive cache and collapsed no-I/O remain. Cost if wrong: unnecessary metadata rescans or delayed discoverability; real append, held-probe and collapsed tests qualify it.

Source: .superpowers/sdd/2026-09-12-bounded-console-run-log/progress.md:42

Task 2 Ruling: use an I/O-free bridge token backed by existing primary turn/run maps to compare UI selection, and a worker-only metadata target resolver. Primary-only writer binding records run identity before first step (existing callback gains explicit conversation_id); no added registry/timer or authority recreation. Initial open and modal predicates compare bridge/conversation/drill/token only. Cost if wrong: stale primary selection or missed cache invalidation; gated resolver, same-conversation new-turn/bind, stale return and cancellation tests must qualify it. This fixes the review's metadata-threading gap without widening unrelated rail readers.

Source: .superpowers/sdd/2026-09-12-bounded-console-run-log/progress.md:45

Task 2 Ruling: current Textual Worker handle lets existing expanded ticks settle matching cancellation before function entry, while generation guards protect replacement and stale publication. AC5 clarified before implementation: a still-admitted probe is not restarted; cancellation permits replacement and does not promise forced termination of an already-entered Python read. Cost if wrong: stuck retry state or overlapping stale read work; gated queued/running cancellation and generation tests must qualify it, with bounded chunk cancellation retained.

## agent-denial-breaker

Source: .superpowers/sdd/2026-09-12-agent-denial-breaker/progress.md:25

Ruling: Include the narrow known-model fixture repair and explicit three-tool successful-execution assertions in Task1 trace compatibility verification — the planned test otherwise fails before reaching the boundary it claims to verify; costs one test-only scope extension if wrong, and must not change product disclosure/permission policy. Plan and regenerated brief contain the correction.

Source: .superpowers/sdd/2026-09-12-agent-denial-breaker/progress.md:35

Ruling: Extend Task2 to builtin gate/provider, virtual CLI and raw-shell invocation facts, preserving unanswered metadata at their existing stamp boundaries — read-only scout confirmed review-only metadata is unreachable for virtual denied calls and Off bypasses builtin/raw review. Keep Task3 result-only accounting; opaque raw-runtime refusals remain unannotated. ADR154 already governs the extension; no new ADR. Cost if wrong: four additional source-file changes and stamp compatibility rework, bounded by actual owner-to-runtime tests. AC8 and spec/Task2/Task3 plan are corrected before their implementation. Scout report scratch/producer-gap-review.md records exact source evidence; no tests/source edits from scout.

Source: .superpowers/sdd/2026-09-12-agent-denial-breaker/progress.md:36

Ruling: Repair six bare-context fixture failures and replace the brittle MCP source-literal test with behavioral refusal/audit evidence in Task2 — root isolated fixture control passed8 cases and source constant already preserves denied-unresolved. Cost if wrong: a test-only scope extension; product composition/permission guards must remain unchanged. Baseline309pass/7fail and control output retained.

Source: .superpowers/sdd/2026-09-12-agent-denial-breaker/progress.md:45

Ruling: Accept worker-proposed pure approval_provenance.py with frozen ApprovalStamp/raw decision+fact at existing keyed stamp entries and compatible detailed/raw accessors; builtin detailed refusal+fact uses one check and legacy check unwraps it — avoids Chat import cycles and repeated owner metadata logic without new shared state. Unanswered metadata must use the actual selected call/name key; only selected name-wide deny is conservatively unresolved if a contributing deny was unanswered, never changing approved scope. Cost if wrong: helper/stamp compatibility rework or conservative undercount, covered by exact-key fallback, clear/pop/nested scope and mixed-row tests. Root approved via message before source edits and updated spec/plan.

Source: .superpowers/sdd/2026-09-12-agent-denial-breaker/progress.md:48

Task2 supplemental fixture ruling: extend the already qualified fake turn-context completion to Tests/Agents/test_raw_shell_integration.py, where the unchanged raw composition path likewise requires tool_policy_profile_id. Ruling: add only the missing default profile field after baseline qualification — same actual composition contract as the virtual fixture, no guard changes. Cost if wrong: one test-only fixture edit to reconsider. Worker owns qualification and implementation; plan updated. Worker also caught owner-specific key selection: exact-only raw/managed/virtual builders must not infer name fallback; new raw RED and owner-specific lookup correction are part of required metadata authority, not a scope expansion.

Source: .superpowers/sdd/2026-09-12-agent-denial-breaker/progress.md:55

Ruling: Carry nonblocking Task2M1 combined-hook annotation into Task4, which already edits the controller — the existing annotation contradicts the implemented compatible typed contract, and a one-line correction can be independently reviewed with that slice without extending I1's scoped fix loop. Cost if wrong: annotation-only rework. Task4 brief regenerated; M2 baseline warnings remain disclosed, not hidden.

Source: .superpowers/sdd/2026-09-12-agent-denial-breaker/progress.md:80

Ruling: carry minorTask3restorationcoverage into Task4 retained-history tests, adding one noncancelled restored_pending case that reaches anothermodelcall atlimit1. Cost if wrong: one test-only scope extension; no extra source/refactor authorized. This avoids treating cancellation as proof of independent exclusion. Task4plan/brief updated beforedispatch; noTask3fixloop.

## agent-worktree-recovery

Source: .superpowers/sdd/2026-09-12-agent-worktree-recovery/progress.md:26

Ruling: Include the narrow nested-runtime dependency and outer test-harness isolation repair in Task1 verification; use real Pipe-gated process tests for new race evidence while naming the inherited19 semaphore failures — the original harness cannot reach its intended boundary safely, and the production protocol has no semaphore requirement. Cost if wrong: test-only harness rework; no production permission changes or host resource manipulation. Plan and regenerated Task1 brief include the correction.

Source: .superpowers/sdd/2026-09-12-agent-worktree-recovery/progress.md:34

Ruling: Reject absolute common-Git-dir reopening after source pin — a real Pipe-gated replacement experiment created a child from the replacement repository. Retaining source and child cwd alone also failed: a commit from the original pinned child advanced a replacement repository ref after source relocation. These are top-level admission races, not the spec's excluded hostile descendant metadata edits. Cost if wrong: additional boundary implementation work; no user mutation is enabled.

Source: .superpowers/sdd/2026-09-12-agent-worktree-recovery/progress.md:40

Ruling: Preserve and independently review the useful test-isolation prerequisite, park intentionally RED design/probe files in owned scratch, and advance independent live-usage/Settings tasks while redesigning worktree execution. Do not mark TASK31210/31211 Done or enable a known pathname mutation fallback. Cost if wrong: task ordering/rework; recovery remains unfinished until its authority guarantee is qualified. Accepted ADR155 stays unchanged; ADR158 allocation scan found no existing158+ claim across174 unique refs, but NO ADR158 has been created or adopted.

Source: .superpowers/sdd/2026-09-12-agent-worktree-recovery/progress.md:52

Ruling: ADR-158 supersedes only ADR-157's unimplemented migration order: definition caps use18→19 independently; recovery is planned later19→20 after its execution qualification. No placeholder schema or speculative recovery storage is introduced. Cap policy, root authority, physical drain and uncertainty requirements are unchanged. Cost if wrong: renumber still-unimplemented migrations/tests if task order changes again; shipped migrations must never be rewritten. Allocation checked223 local ref tips and38 registered worktrees with no158+ claims; adr-sequencing-allocation.json is in the cap plan workspace. Source plans/specs and CLI task plans updated before implementation. Earlier preflight schema-order rows are historical and superseded by this ruling; recheck actual DB before edits.

## agent-worktree-interim-safety

Source: .superpowers/sdd/2026-09-12-agent-worktree-interim-safety/progress.md:15

Ruling: implement the already-required unsupported-boundary refusal and data-retention policy now rather than leave known pathname mutation/GC live while recovery is redesigned. Explicit worktree spawn fails without shared-tree fallback; normal agents remain supported. Cost if wrong: temporarily unavailable worktree isolation and additional retained disk data; no user data deletion or weakened authority. No new ADR is required because ADR155 already chooses these boundaries and ADR158 explicitly adopts no replacement backend.

Source: .superpowers/sdd/2026-09-12-agent-worktree-interim-safety/progress.md:19

Root completion check returned first report before commit: focused6 and affected11 passed but17 directly invalidated legacy service tests were skipped. Ruling: replace obsolete reachable-behavior assertions with active refusal/retention coverage and remove superseded unreachable service tests, rather than retaining a skipped future implementation template. Preserve manual low-level helper tests and meaningful no-Git/no-child/no-confirm/routing/ordinary-sibling assertions; test confirmation absent/allow/deny and state-independent refusal where useful. Cost if wrong: future qualified worktree implementation must reintroduce/update integration expectations from its design and Git history instead of unskipping stale unsafe contracts. This follows plan step5 and changes no product policy. Original worker resumes this bounded test completion, not an independent review fix round.

Source: .superpowers/sdd/2026-09-12-agent-worktree-interim-safety/progress.md:23

Task1 independent review at93ee16a144: Needs fixes/specfailed. Important I1 actual Console preview test still expects enabled schemas (reviewer exactnode failed); I2 inline/no-fleet route keeps old no_fleet code/copy. Root confirms minor exact refused-row assertion as a required verification gap from plan step1: successful plain sibling can currently satisfy row-existence assertion. Ruling: include exact refused task RUN_ERROR/persisted reason assertion in fix1 together with I1/I2 — matches existing AC, cost one focused assertion correction. Explicitly add Console test filename to scope; no new policy/ADR. Original worker fix round1/5, previous reviewedFIX_BASE93ee16a144.

## live-per-run-usage

Source: .superpowers/sdd/2026-09-12-live-per-run-usage/progress.md:18

Root named interface check for pre-first-step publication: console_agent_bridge.py5952 seeds only the primary per-turn live key;6199 publishes child snapshots only from on_step;8209live_run_snapshot returnsNone when no child step slot exists. The new run scope is therefore needed to make the spec's first observable child count visible before model completion, while the primary real runID must resolve through its separate current-turn key. Ruling: add explicit real-boundary pre-first-step primary/child assertions to Task1 — this directly verifies the already accepted all-active-runs contract and prevents a feature that appears only after its useful window; cost if wrong is one test/setup seam to reconsider. No new ADR or product scope; ADR156 governs. Late started-after-terminal rejection must remain bounded by active scope/run ownership, never a finished-ID map. Plan updated before usage source work.

Source: .superpowers/sdd/2026-09-12-live-per-run-usage/progress.md:25

Ruling: extend Task1's file scope narrowly to the existing gateway emission provenance and its targeted tests, because its omission from the file list prevents the specification's already-required exclusion of synthesized copy. Keep existing consumers backward compatible and accounting unchanged; no new provider policy or ADR is needed beyond ADR156. Cost if wrong: gateway seam/test rework; no provider-specific branches, additional text retention, or broadened task functionality. Update plan/brief before the fix. Fix round1/5 will resume the original implementer; all four findings must be addressed before UI work.

Source: .superpowers/sdd/2026-09-12-live-per-run-usage/progress.md:41

Task2 root painted inspection: Quick Look rendered owned SVGs to PNG after sandbox initialization failed; outside-sandbox rendering approved/succeeded, no dependencies/userapp data changed. Both180x48 and100x32 hide128provideroutputtok behind the task ellipsis. Ruling: enable existing wrap_secondary only while a nonzero live usage label is present, preserving appended task/result/error text and existing terminal unread wrapping, with no new component/API/CSS/timer. Confirm both rendered widths; cost if wrong: active long task rows use more vertical space while usage is visible and may need later density tuning. No new ADR; direct presentation of ADR156 scalar.

Source: .superpowers/sdd/2026-09-12-live-per-run-usage/progress.md:42

Task2 verification prerequisite: four affected transcript sync nodes fail before reaching activity assertions because _sync_stub returns a bare MagicMock from sync_selected_fork_eligibility, whose real API unpacks a2tuple. This includes three existingtests and onenewtest. Baseline classification exactAST for _sync_stub and unchanged chat_screen stored in fixture-baseline-classification.json. Ruling: fix only this owned stub to return the actual no-selection tuple, retaining transcript/repaint assertions; no production ChatScreen change. Cost if wrong: one fixturevalue correction; do not mask other failures or call this a new product regression.

Source: .superpowers/sdd/2026-09-12-live-per-run-usage/progress.md:50

Task2 resource qualification found actual new-node growth despite no sentinel warning: fifteen new usage tests retained39 regular descriptors plus2sockets/1pipe. The module-owned _real_fleet_recovery_database fixture constructs unused Tldw apps and attaches a ChaCha DB; exact constructor/attached owners outlive the harness. Ruling: close those exact fixture-owned database and profile-lock handles after mounted harness teardown using the existing test close helper, leaving production ownership and foreign resources unchanged. Cost if wrong: premature fixture disposal could invalidate late worker access; the owning-module targeted check must cover teardown and existing tests. Before/after disposable pytest census shows regular2→41 then2→2 for the same fifteen new nodes; socket/pipe residuals remain disclosed rather than claimed zero aggregate. The corrected old-node comparison is on pre-cleanup current source, not pristine BASE; no full-suite or repository-wide leak claim.

## agent-settings-followups

Source: .superpowers/sdd/2026-09-12-agent-settings-followups/progress.md:31

Task2 root paint inspection at currentunstagedsource: original120x40/70x40 PNGs retainedin task-2-evidence/svg-render. Narrowgeometry showsSelect/Loady34h3 ends37 butactionsstarty36, actualone-rowoverlap; loadedbulk instructionsTextArea also hasno paintedcontent. Ruling: correctowningpresetrowsize tofitcontrols andpreserveusableinstructionsviewport with narrowcanonicalSettingsCSS/existingformscroll; no newlayoutarchitecture. Addpairwisedisjointgeometry andactualTextArea contentvisibility checks, confirmbothwidths onceafterfix. Costifwrong: a fewmoreformrows/scroll orslightlylessdefinitionlistspace; preserveseditablepresetcontract and avoids clippedcontrols. Current14UI/2containmentpasses donotqualifythisgeometrygap.

Source: .superpowers/sdd/2026-09-12-agent-settings-followups/progress.md:35

Task2 fielddiagnosis: narrowDescriptionregion x24width70 exceedsviewport70; contentx27width64. Ruling: Agents-scoped compactInput min-width0 andSwitchauto sizing are withinpreserveeditableformcontract; loader cursorstart maypresentloadedprefix. Alsoqualifyparentrowclip because Inputh3 againstgenericrowh1canhidecontent. Fixexactowningrows orscopedcompactstyle,notglobalSettingsrules. Costifwrong: localrow/scrollspacing andcursorplacement rework; noauthoritychange. Await allfieldpaint/geometry andoneconfirmedbatchbefore review.

Source: .superpowers/sdd/2026-09-12-agent-settings-followups/progress.md:37

Task2 rootfinal-qualified-controlsPNGinspection: Inputs/instructions/preset/actions nowvisible; EnabledSwitchstillpaints onlydarktopborder (no thumb/state) atbothwidths. ReverseglyphfromSwitch.render_line(0) isborder,falsepositive forvisiblecontrol. Ruling: fitexistingthree-lineSwitch withinitsowningEnabledrow usingexistingselect-rowheight3/scopedrowrule; requireSwitchcontentinsideparentclip andON/OFFvisibledifference. Costifwrong: twoextraformrows/scroll, no newcomponent orpermission. Preservecurrentqualifiedcaptures; finalvisualapproval withheldonlyforswitch.

## definition-wall-cap

Source: .superpowers/sdd/2026-09-12-definition-wall-cap/progress.md:19

Ruling: ADR-158 supersedes only ADR-157's unimplemented migration order: definition caps use18→19 independently; recovery is planned later19→20 after its execution qualification. No placeholder schema or speculative recovery storage is introduced. Cap policy, root authority, physical drain and uncertainty requirements are unchanged. Cost if wrong: renumber still-unimplemented migrations/tests if task order changes again; shipped migrations must never be rewritten. Allocation checked223 local ref tips and38 registered worktrees with no158+ claims; adr-sequencing-allocation.json is in the cap plan workspace. Source plans/specs and CLI task plans updated before implementation. Earlier preflight schema-order rows are historical and superseded by this ruling; recheck actual DB before edits.

Source: .superpowers/sdd/2026-09-12-definition-wall-cap/progress.md:21

Schema18 baseline at925aa1445b: isolated pytest Tests/Agents/test_agent_models.py Tests/DB/test_agent_runs_db.py -q --basetemp=.superpowers/sdd/2026-09-12-definition-wall-cap/pytest-baseline returned110passed/1failed4.81s exit1 (task-1-schema18-baseline.txt/.exit). The single failure is test_runtime_tool_names: expected set omits existing report_to_supervisor/read_agent_messages. Entire test file is byte-identical to merged branch base d66908a69; RUNTIME_TOOL_NAMES assignment AST is identical there and here (task-1-baseline-classification.json). No product catalog regression is claimed. Ruling: reconcile this exact inherited assertion in cap Task1 while preserving exact-set coverage and all original names, so targeted model verification is meaningful. Cost if wrong: one expectation rework; no production disclosure/permissions change. Preserve baseline warning qualification; no full-suite claim.

## agent-fleet-program-closeout

Source: .superpowers/sdd/2026-09-12-agent-fleet-program-closeout/progress.md:16

Ruling: reconcile verified historical and child records now while leaving the parent AC and status open — all seven actual children qualify, but refusing to record that until an unqualified worktree backend exists would leave stale status — cost if wrong: the partial checkpoint needs a later refresh; no functionality or completion is fabricated. The plan explicitly retains its parent-closure prerequisites. Root reused six historical ancestry checks, current runtime122 evidence and task reviews rather than repeating tests. Records-only independent review is next; final branch review still pending.

## agent-orchestration-remaining

Source: .superpowers/sdd/2026-09-12-agent-orchestration-remaining/progress.md:16

Ruling: split independent subsystems into their own plans and task review gates; continue all authorized work without checking in between tasks. User approved the eight outcomes; default choices preserve prior conservative caps and existing safety boundaries. Real destructive/external actions still require their ordinary authority.

Source: .superpowers/sdd/2026-09-12-agent-orchestration-remaining/progress.md:17

Ruling: reuse verified isolated Python .superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python. No shared venv edits or full suite. Keep all test imports under pytest isolation and clean up owned workers/SQLite/repositories.

Source: .superpowers/sdd/2026-09-12-agent-orchestration-remaining/progress.md:18

Ruling: preserve scratch and durable ledger rather than delete worktree artifacts; prior repository lesson shows evidence dies with disposable directories. Distill relevant notes to committed task/spec/docs before final handoff.

Source: .superpowers/sdd/2026-09-12-agent-orchestration-remaining/progress.md:41

Ruling: safe logicaldiscard retainsbaselinecheckout andexactbranchCAS; unsupportedWindows/missingPOSIXdescriptors refuse — pathnameforcedrootdeletecannotpinidentity — coststoragecleanupresidue/platformlimitation, explicitUIreceipt.

Source: .superpowers/sdd/2026-09-12-agent-orchestration-remaining/progress.md:42

Ruling: reuseExecutionOwner positivephysicaldrain withstickyuncertainty insteadofnewregistry — existingphysical tool/modelleasealreadytracksabandonedworkers — callback/DBfailuresmuststaynon-actionable; costconservativeblockedrecovery.

Source: .superpowers/sdd/2026-09-12-agent-orchestration-remaining/progress.md:43

Ruling: worktreeGit usesonlyhostHOME/XDG_CONFIG_HOME foruseridentity/config plus existingexternalPATH, disablesgeneratedhooks andnoinjection/networkcredentials — preserveuserGitidentitywithoutarbitraryexecutor — costdocumentednon-hostilemetadata/filtertrust. Preview8192chars, streambinarypatchcap32MiBbeforemutation; oversizedworkrefuseshonestly.

Source: .superpowers/sdd/2026-09-12-agent-orchestration-remaining/progress.md:44

Ruling: liveusage usesboundedsequenceallocator, explicitproviderintegerincludingzero, scalarcurrentcallstate andexistingtimers — avoidfinished-runmapleak/chunkroundinginflation — costapproximateUTF8/4untilproviderobserved.

Source: .superpowers/sdd/2026-09-12-agent-orchestration-remaining/progress.md:45

Ruling: researcher/ingest-runner presetsinheritonlyparentavailabletools (noinventedMCPnames);critic/bulkreaderlocalreadallowlist; allunsavededitable — providerinstallationsdiffer — costbroaderparenttoolavailabilityclearlysubjecttoexistingpermissions.

Source: .superpowers/sdd/2026-09-12-agent-orchestration-remaining/progress.md:46

Ruling: definitioncapsapplyafterexistinghelperfloors; preservethreadedoutliveparent vsinlineparentremainder; retainadmittedcap onlyforcappedlineages — nosemanticwidening — costnewlineagerequiresfreshspawn toraisebound. Use replaceinchildhelpers tofixdroppedretry/warning/denialfields; extraACaddedbeforecode. Preserveuncappedfingerprinthashes viaomittedNonekey.

Source: .superpowers/sdd/2026-09-12-agent-orchestration-remaining/progress.md:49

Ruling: the 32 MiB worktree patch cap applies before destination mutation, not before a confirmed child capture commit. Capturing tracked/untracked changes first preserves existing exact-patch semantics without a second temporary-index mechanism. Oversize preserves the child commit, original base and unresolved recovery record and reports the refusal. Cost if wrong: child work is committed on its isolated branch despite a refused destination operation; no destination change or source loss is allowed. This supersedes the earlier blanket before-mutation wording in draft ADR 155 and its spec/plan.

Source: .superpowers/sdd/2026-09-12-agent-orchestration-remaining/progress.md:53

Task 2 Ruling: optional keyword-only cancellation callback on run_log_available preserves existing bool callers while allowing the new UI worker to stop before a subsequent bounded metadata chunk. Task 1 had no cancellation source. This executes the existing spec requirement without changing authority; cost if wrong is premature false availability or continued stale scanning, covered by gated cancellation and stale-publication tests.

Source: .superpowers/sdd/2026-09-12-agent-orchestration-remaining/progress.md:55

Task 2 Ruling: optional modal target_is_current predicate runs only on the UI thread before admission/publication; production captures exact bridge/run while standalone callers default true. Reader authority remains independent. Cost if wrong is stale content publication or unnecessarily blocked navigation; gated target replacement and usable Close must be covered.

Source: .superpowers/sdd/2026-09-12-agent-orchestration-remaining/progress.md:57

Task 2 Ruling: confirmed inherited negative availability cache has no write/lifecycle invalidation, so a first record arriving after the probe can remain hidden forever. TASK18601 AC5 added before implementation. Use a one-second negative retry deadline on existing expanded ticks, beginning only after completion, with exact-generation pending state preventing overlap. Positive cache and collapsed no-I/O remain. Cost if wrong: unnecessary metadata rescans or delayed discoverability; real append, held-probe and collapsed tests qualify it.

Source: .superpowers/sdd/2026-09-12-agent-orchestration-remaining/progress.md:64

Task 2 Ruling: use an I/O-free bridge token backed by existing primary turn/run maps to compare UI selection, and a worker-only metadata target resolver. Primary-only writer binding records run identity before first step (existing callback gains explicit conversation_id); no added registry/timer or authority recreation. Initial open and modal predicates compare bridge/conversation/drill/token only. Cost if wrong: stale primary selection or missed cache invalidation; gated resolver, same-conversation new-turn/bind, stale return and cancellation tests must qualify it. This fixes the review's metadata-threading gap without widening unrelated rail readers.

Source: .superpowers/sdd/2026-09-12-agent-orchestration-remaining/progress.md:67

Task 2 Ruling: current Textual Worker handle lets existing expanded ticks settle matching cancellation before function entry, while generation guards protect replacement and stale publication. AC5 clarified before implementation: a still-admitted probe is not restarted; cancellation permits replacement and does not promise forced termination of an already-entered Python read. Cost if wrong: stuck retry state or overlapping stale read work; gated queued/running cancellation and generation tests must qualify it, with bounded chunk cancellation retained.

