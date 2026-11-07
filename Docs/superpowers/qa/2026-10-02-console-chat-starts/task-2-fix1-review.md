# Task 2 fix round 1 — scoped independent review

Reviewer: /root/integration_review, gpt-6-astra/xhigh. Range c9b3738434b9a1f95a37e0930491b47aa96a779f..71e5fe1b9c95dfd1e21908409e5868dbac0eb795.

Spec verdict: Issues found. Task quality: Needs fixes. All six original obligations are addressed; one new Important fix-diff regression remains.

## Original finding verdicts

1. Physical worker ownership — ADDRESSED. console_chat_start.py:149 shields exact worker; cleanup:327 drains before completing attempt/releasing claim; controller integration:27233. Tests start:1470 and actual adapter:1998 hold threads after Stop and assert capacity remains claimed.
2. Required project decisions before acceptance — ADDRESSED. controller:10756 checks frozen binding/live authority/consent, call:10843 precedes both receipts; guards:26573 refuse later automatic dialogs. Tests start:1580 one-/two-folder cases retain pending draft, no rows/charge/callbacks.
3. Visible prepared Manual Send — ADDRESSED. prompt_queue.py:635 exposes enabled Send. Mounted runtime test:2959 clicks actual control, confirms withdrawal/refund and ordinary manual request.
4. Initial preparation/refund uncertainty — ADDRESSED. start:241 registers owner before await; drain:278; refund:347. Six barriers at test:1522 cover source/caller/shutdown false/raised abort. Publication:175 drains exact write and propagates cancellation before provider; test:1843 asserts no provider calls.
5. Lost durable launch outcomes — ADDRESSED. metadata:37 bounded validation; persistence:1403 transactional facts; store:2987 hydration; switcher:1233 History label; controller:17093 early publication. Metadata is display only. New issue below concerns completed manual recovery.
6. Native start→child→wake integration — ADDRESSED. test start:1671 uses actual runtime/gateway with network-boundary transport double:2 generations,1 child,4 calls,12 confirmed tokens, original root limits/deadline and exhaustion refusal without another request. Deterministic integration, not real-provider recursion.

## New Important finding

### Historical refusal permanently marks a successfully recovered chat as blocked

console_chat_controller.py:5264 always publishes the saved launch label. UI/Console_Modules/workspace.py:3010 converts Not started/Review required to blocked activity without checking whether the handoff is unresolved. Ordinary Manual Send can consume a runtime-disabled draft and complete successfully, yet Active still shows INPUT NEEDED / Waiting for you with no live/queued work.

Focused real native-rig probe, exit0: refusal=not_started; manual accepted/completed, origin=MANUAL/provider_started=True; run_status=completed; handoff_state=consumed; handoff_status=Not started; accepted_live_turn=False; queued=0. Initial unawaited fixture misuse did not exercise behavior and is not product evidence.

Preserve historical launch facts, but derive blocked activity from the current unresolved handoff. Add refusal→successful Manual Send→completed/unblocked real projection regression.

Exact probe saved separately in task-2-fix1-recovery-probe.py. No existing suite rerun.

## Evidence, strengths and qualifications

Physical completion/logical Stop now have explicit owners; outcome facts use existing transactions and bounded display vocabulary. Actual controls/cancellation barriers and test consolidation are accurately documented. The reviewer read the fix package once, inspected acceptance ordering and unchanged shutdown/early-result/project semantics, and confirmed report claims against tests/output.

102 core/migration passed;2 actual mounted controls passed; final current-session edit has3 passing cases after the102 run, not a claim that102 ran afterward. Postcommit lint/format/all8ratchets/whitespace exit0. Actual normal-app captures/receipt confirm saved refusal, blocked activity, draft/no replay and corrected History. Physical drain is deterministic proof; streaming uncertainty and separate confirmed non-streaming settlement remain qualified.

## Out-of-scope observations

Two direct AgentService fixtures reject existing reasoning_replay keyword identically at archived FIX_BASE and current code. Classification as preexisting fixture debt is supported by budget-baseline.log:16/:39 and budget-probe.log:16. Earlier FD-growth/escape warnings remain separately qualified; no repository-wide cleanup is established.

No new Critical or additional in-scope Minor finding. Resolve false blocked activity before task approval.
