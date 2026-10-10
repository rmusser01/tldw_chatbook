# Console hook preparation native lifetime

Task: TASK-34563.13. Base: 6d8d56e75c819d993e0e1df299e6a7a2727e5a52.

ADR required: yes, existing ADR-225 applies.
ADR path: backlog/decisions/225-console-send-preparation-and-io-ownership.md.
Reason: implement finite domain I/O retirement required before enabling early received Send capture; preserve ADR-197 permission authority and ADR-126 native ownership.

## Contract

Own exactly the original controller hook-admission read and runtime v2-configuration/context-key reads. Keep one physical read record containing its captured callback/source, issuing Task, optional session key, private executor Future and retirement Future. Controller and captured runtime observe the same record; runtime-only reads need no controller. These sets contain issued native work only, never admission, permission or provider state. Replacing a controller/store cannot detach an already issued record. Attaching a controller with an issued standalone read adds that same record to runtime observation.

Register synchronously before executor submission. Use the existing default executor, copied context, a submission-resolved seal (enqueue-then-raise cannot enter native work), and a private Future retained through repeated cancellation. Finish only after the original callback and its operation-owned native cleanup return; consume actual result/error and then re-deliver caller cancellation. No new executor, scheduler, timeout policy or whole-submit drain.

Keep hook_admission_reason() zero-argument and existing injected async callbacks. Known submit, queue and postcommit call sites provide private same-Task session binding around the exact call; inherited child Tasks cannot borrow it. Unbound direct reads are app-scoped and conservatively included in session teardown. Snapshot original store/controller/session and permission sources before handoff; worker and post-await checks refuse source drift or session successors. The context-key stock body uses the captured source rather than a later mutable runtime lookup. Custom callbacks retain their selected invocation and fail closed on a replaced owning source.

Stop seals received admission and cancels the exact issuing waiter, including before submit registration. Close/dispose preserve their existing generic two/three-second grace, then drain only physical read retirement notices before releasing custody, removing sessions or ending store ownership. Standalone controller shutdown drains its own issued reads. Repeated cancellation cannot detach native resources. A provider/custom submit tail after native completion remains governed by existing bounded behavior. Existing ordinary commit and initial review ownership stays unchanged.

Preserve .12 demand-driven context absence, native plugin hook selection, hook reconciliation/revocation, WAL/NORMAL, durable saved-turn failure/draft retention, checkpoints and awaited history. This slice does not enable early unconfigured UI receipt or claim 100ms feedback/subsecond dispatch; reference expansion and Library native reads remain separately scoped.

## Parallel ownership

- Shared preparation lane: only new Chat/console_hook_preparation.py and Tests/Chat/test_console_hook_preparation_reads.py. Implement finite record/execution/drain and private binding helpers against this contract. No native runs.
- Controller/provider integration lane: only Chat/console_chat_controller.py and Chat/console_runtime.py. Wire the exact three reads, source checks, Stop/Close/dispose and standalone/attachment lifetime. No native runs, no submit-registry conversion or broad formatting.
- Baseline verification lane: only new Tests/Chat/test_console_hook_preparation_lifetime.py. Actual runtime.accept_turn with held original permission and workspace producers, physical lease assertions, source drift and unrelated-tail controls. Existing fixture files stay unchanged. No native runs.
- Root: integration owner, plan/task/report, source review and all sequential native checks. Implementations wait for meaningful RED on the original source; final integration checks after both implementations are ready.

## Verification

1. Review contract before product edits. Add actual original snapshot cancellation/Close regression, observe meaningful RED against unchanged base in one contained native run.
2. Implement concurrently within ownership above. Helper tests cover repeated cancellation, callback failure, executor enqueue-then-raise, same-Task binding and exact observation/attachment.
3. Integrated original producers: snapshot Stop/repeated cancellation; v2 Close past original grace; actual WorkspaceDB context dispose cancelled during original grace (real retained empty engine); controller replacement; permission/store/same-session drift; post-read custom/provider tail boundedness.
4. Targeted existing controls: .12 cold empty and retained configured owner; permission epoch/revoke; .10 held original review close; .8 native drain ends before unrelated submit tail; .11 custom submit/receipt. Run actual mounted Send control if integration touches its behavior. No full suite or parallel native/timing runs.
5. Review complete diff, scoped lint/format deltas and unchanged architecture caps; inspect containment identities, original resource retirement and source hashes. Record failures and scope accurately. Commit tested candidate only, no primary edits/push/merge.

Source-only review approved physical-only dual observation over retaining old controllers or converting submit indexes. Open implementation review points: pin original stock context sources, preserve callback shapes and app-scoped standalone teardown, avoid self-wait deadlock, retain records when attaching an already-active controller, and ensure cancellation during dispose grace still reaches physical drain.

Initial source/contract review passed. Agreed helper API: ConsoleHookPreparationRead; run_hook_preparation_read; hook_preparation_reads_for; observe_hook_preparation_reads; drain_hook_preparation_reads; bind_hook_preparation_session / hook_preparation_session_for; worker-only hook_preparation_source_for. Existing creator observer sets hold the same record. The original-source RED failed at the intended assertion: cancelled runtime custody Task became terminal while actual snapshot raw operations/leases remained held. Original product hashes matched 6d8d56; original producer and contained process resources retired normally after fixture release, force/overflow/lookup races: 0. Evidence: .superpowers/sdd/2026-10-07-console-hook-preparation-lifetime/red-audit.json. Product implementation authorized after this RED.

## Completed implementation and verification

All planned implementation/check steps completed. Final integrated selection: 46 passed, no failures/errors/skips, after a test-only Python 3.12 observer correction and fresh-process rerun. Both peer source findings were fixed: controller app/context-provider source capture, and keeping finite source witnesses out of the reusable lifecycle. Cold canonical accessor affinity and StopIteration-to-Future transport have focused controls. Exact physical native and process retirement are recorded separately; first integrated run had one diagnostic PID lookup race, all runs retired normally without force or overflow.

No introduced lint/format issues; existing module caps and controller lint debt remain. No broader latency/native ownership completion claimed. Report: Docs/Development/2026-10-07-console-hook-preparation-lifetime-verification.md. ADR-225 remains governing.
