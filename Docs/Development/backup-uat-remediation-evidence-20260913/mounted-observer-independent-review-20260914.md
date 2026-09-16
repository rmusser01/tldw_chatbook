# Mounted observer wiring review

Scope: five inserted Python statements (six diff lines) in Tests/Backup_Recovery/test_mounted_console_backup.py versus 5af7381f65. SHA256 75213b2bf80d7eb344dfbd0f376e3c452b27a867e610577805c397a58d3a206a.

One actionable finding: diagnostic cleanup can replace the native wait failure. The newly inserted `stop_loop()` precedes `service.status` in the wait finally. The existing helper's stop disables profiling, then calls `_write`, which can raise. If native wait is already raising, that write failure becomes the propagated exception and skips the original status/checkpoint evidence. The newly registered admission observer's stop likewise has deliberate raising final-write/sampler-failure paths and can replace an active failure during ExitStack cleanup. Preserve the active original failure at these new cleanup call sites while retaining diagnostic failures as separate bounded evidence. Successful-path observer failures can remain visible. This requests a caller cleanup correction, not product/helper redesign.

Otherwise verified: removing precisely the added statements produces an identical embedded-child AST; the current child compiles. Original service calls, assertions, installed-origin check, native waits and parent deadlines remain intact. Admission native_calls=False instruments only existing groups/tokens; no Windows native-call wrappers are enabled. The existing helpers emit bounded numeric thread/call/root-count and code-name metadata, with no new path/value disclosure from this wiring. Existing ExitStack and main/service finally structure remains.

Treat resulting timings as instrumented diagnostic evidence. The loop's duration=5 schedules same-thread stop after five seconds; a blocked loop can deliver that stop late. It is not a hard wall-clock observation limit. Existing output records actual elapsed time. This does not change product/test deadline constants or establish Windows startup acceptance.

Read-only review and AST/compile checks only; no source edits or additional mounted run. Root owns ongoing observer and native tests.

## Final cleanup re-review — approved

The earlier cleanup finding is resolved. `stop_observer` captures `sys.exception()` before invoking cleanup, uses bare raise for a diagnostic-only failure, and returns without reraising an existing primary. A fixed note is the only added primary metadata; even a failing overridden add_note cannot replace the primary. The two new mounted observer cleanup sites use this helper; other call sites and product behavior are unchanged.

Independent targeted execution: `python -m pytest Tests/Backup_Recovery/test_thread_diagnostics.py -k observer_cleanup -q` — **7 passed, 46 deselected in 1.01s**, exit 0. Log: /private/tmp/uat-mounted-observer-cleanup-independent.log. Cases prove original exception identity and exact traceback preservation through direct finally and ExitStack cleanup for RuntimeError and cancellation, diagnostic-only propagation, failing note attachment, and successful cleanup without extra note. Environment emitted requests dependency and unrelated pytest retention-cleanup warnings; no test failure.

Final reviewed SHA256:
- thread_diagnostics.py: d59ce2f3e93275bb060a3f34d8d3526458b577df0e18c83d0f0babbb94b5e6a3
- test_thread_diagnostics.py: bd5485093e2050ea55288c68a85b1e3b7d09cc98f14316febe9bc033ae681f57
- test_mounted_console_backup.py: 40a4b18ac9bde4fdf0e9ccfddfe1dd65ebd3312e16a8944b879fee5676e67cc3

No remaining actionable finding in this bounded delta. Root's earlier three installed macOS routes predate this cleanup wiring; the final wired Settings repeat and native Windows acceptance remain separately owned by root. Timing remains instrumented evidence, not an uninstrumented performance result.
