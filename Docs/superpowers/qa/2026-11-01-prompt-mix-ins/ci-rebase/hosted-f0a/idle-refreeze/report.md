# f0a UI18 native idle-refreeze oracle race

Baselinef0a856d1b914c668a6d8ad28baafd91adb9b43a6. Only Tests/UI/test_console_idle_heap_refreeze.py changed. Task38544 investigation plan was recorded by parent before edits. Existing ADR198/230 and owner semantics apply; no new ADR for this test-only correction. No production, custody, physical retirement, GC policy, import/performance budget, census, or timeout change.

## Diagnosis

Hosted exact test_mounted_received_custody_blocks_completion_and_idle_timer_reclaims failed its final exact-reason assertion: the real refreezer had performed run_settled, not yet first_send_preimport. All prior custody/retirement assertions passed. The test waited only until a weakly referenced cycle died, then immediately required the specific first-send action. Those are independent asynchronous outcomes: a previously armed native run-settled timer may collect the cycle before a second warmup callback's timer fires.

Connected source confirms intended behavior. app._finish_first_send_warmup delegates to app_idle_warmup, which requests the existing attached Console refreezer. IdleHeapRefreezer.request explicitly replaces earlier requests, _fire drops busy requests, and actions records performed maintenance rather than pending intent. The transcript poll requests run_settled when it stops. Production does not promise that a particular requested timer must be the first collector after custody retires. No production defect was demonstrated.

## Causal evidence

One original native run with observational request/fire/maintain/test-phase tracing passed16.59s. It showed run_settled pending when the test creates the cycle; first_send_preimport then replaces that request before its timer fires and performs collection.

One bounded controlled ordering reproduced the exact failure8.87s. The private diagnostic delays ONLY the second asyncio.to_thread submission until the already-armed native run_settled timer completes, then calls the original host.call_from_thread and original warmup completion callback. It never fires a timer manually, invokes collection, alters busy/readiness, or edits owner state. Native timer run_settled collects the cycle. The subsequent original warmup callback requests first_send_preimport with busyFalse. The old weakref-only wait exits immediately while that request is still pending, and the original reason assertion fails with the same hosted deque.

Exact RED trace: run_settled maintain ends at monotonic1871200.303789; second warmup request arrives1871200.325330; cycledead/assertion at1871200.342090 while pendingfirst_send_preimport and actions onlyrun_settled. This reproduces the failing scheduling condition, not the precise hosted scheduler timing.

## Minimum correction

The final existing10s until now waits for both original outcomes: weakref is None AND exact(refreeze,first_send_preimport) is in actions. The final reason assertion remains unchanged. Every earlier received-custody, no-maintenance, physical retirement, native timer/weakref collection check is unchanged. No added retry, timeout extension, new case or relaxed action classification.

Same controlled ordering with corrected predicate passes8.27s: cycledead at1871273.536071 while first-send remains pending; actual first-send timer fires1871273.557322 and maintenance completes1871273.583156 before final assertion. Ordinary exact without any observational/order plugin passes5.59s. No full-module or broader tests were needed for this single predicate change; no test processes remain active.

Artifacts preserve original and final probe source, controlled RED/GREEN, ordinary GREEN, hosted log, source snapshots and exact commands/cwd/return/elapsed values. Original observation used broad Python line tracing; controlled runs narrowed trace to the target test. Timings are functional evidence only, not performance qualification. Controlled ordering was privately enabled only through UI18_ORDERED=1; ordinary GREEN has no plugin installed. No unsupported claim about actual provider success: this established test exercises received-custody retirement with its existing refusal-capable native fixture.

Static: all original assertions and timeout arguments AST-identical. Ruff baseline0/current0. Existing unrelated file-format debt retained; diff check clean. Source SHA2566b579c6ace596b17a192731d32914a0f6ccd0c6f2e0c1bc91dc39dbf7e73a085. Root owns task notes, durable packaging, commit/push and fresh hosted monitoring; no Qodo requested. Independent reviewer inspected source/timelines and reported no actionable source findings; final report supplied separately.
