# TASK-40608 bounded pre-plan attribution artifact

Artifact: `task62-preplan-profile.py.draft` (479 lines), SHA256 `f658d7f615e230356e2d250582788f2fff50dde468a7607cb198eb97fed01220`.
Based on provider diagnostic `260138d19cc0abbc518868ead597c2298252564d12983feedd2f6d71f1304d67`; retains every prior selector and the 512-line emission cap.

Use the same temporary worker installer and final cleanup call as the approved provider diagnostic. Root owns installation, freeze and the one actual-process run. The exact artifact has been executed only against synthetic stdlib functions/modules on Python 3.12.10. No product imports, native authority, actual-user execution or live product edits occurred here.

The original received coroutine opens a diagnostic ContextVar gate. Existing `copy_context().run(invoke)` carries it to the finite preparation worker. Only the exact original worker bodies enable cProfile. The two original received/custodied owner coroutine active slices record wall/thread-CPU durations without cProfile. Nesting preserves the outer slice; suspension stops its timing. First original controller acceptance START closes the gate and immediately stops the current thread's active timing/profile. PY_UNWIND also exits active slices. Cleanup disables current-thread profiling and retains exit callbacks until any already-active original remote worker stops; it never cancels, joins or replaces a product operation.

Python 3.12 requires PY_UNWIND to be global. The diagnostic enables only that global event and immediately filters by exact registered code; normal markers remain local. Cleanup clears both global and local masks before freeing the tool. cProfile itself uses interpreter-wide tool slot 2, so only one finite worker may profile at once. Existing/overlapping profiler ownership emits a coverage-invalid marker and is preserved. All nine original cProfile diagnostic callbacks are filtered to the exact worker thread, including dependent C_RETURN/C_RAISE events omitted from get_events(). Product functions and identities remain unchanged.

Cleanup reports active elapsed time, thread CPU, segment counts and the top ten cumulative/five self-time source-code rows per bounded profile. Rows contain only module/qualified source names and numeric counts/durations. No arguments, locals, bodies, raw paths, return objects or built-in descriptive strings are emitted. Any `profile.error` marker makes that run's profiling coverage invalid. cProfile overhead means these timings are attribution evidence, not an accepted latency benchmark.

Source proof: `console_runtime.py:3317` lazily imports the received helper before its original call; local CALL discovery registers that code before the Task can start. `console_received_dispatch.py:425` directly awaits the custodied coroutine (`:889`) in the same context. `runtime_producer_lifetime.py:258` defines the exact original finite `invoke`; its `:293` copy-context and executor submission convey the diagnostic gate. Controller acceptance is `console_chat_controller.py:12484`.

Read-only source pins at handoff:

| Path under tldw_chatbook | SHA256 |
| --- | --- |
| Chat/console_runtime.py | 3da077c4d01d6e8d2582a10d3a64cffd8a3bcbb25ad6dc6fb1b450f9686d258e |
| Chat/console_received_dispatch.py | acc9e54982c9ac1cda26e565e27cbc80eb2719b7a55d5298d7ebde4601a3fbf0 |
| Backup_Recovery/runtime_producer_lifetime.py | 1c9a1ef861989151ca8bc0f30d6a8f59ff0469cb41d4a36b612219a5d66bddc4 |
| Chat/console_chat_controller.py | 9ded0e138bead44af2b64682faafdd7ef5b22e009b4143b9dafc6bb4387a5aa6 |

Validation: AST parse/compile PASS; all selected qualified names exist in current source, including functions nested inside control flow; previous selectors are a subset; imports are stdlib; no frame inspection or tracing APIs. The complete artifact passed synthetic normal completion, owner exception, cancellation, worker exception, acceptance cutoff, global/local/tool cleanup, concurrent unrelated-thread exclusion, and overlapping-profiler refusal/preservation. Activation is proved through exact probe call counts, because sys.getprofile() remains None for cProfile on this interpreter. Smoke source: `task62-preplan-profile-smoke.py`, SHA256 `01d2a26b585b7a6184aa04aa81a66c7612335afd4ee6ddcfb93e1dcf90eee6dd`. Command: `C:/Python312/python.exe -I -S .superpowers/sdd/2026-11-13-task-40608-sub-five/task62-preplan-profile-smoke.py`.

Superseded artifact `804dd893...` failed installation in actual run a1: PY_UNWIND was incorrectly included in set_local_events, which raises ValueError on Python 3.12 and 3.13. That run is invalid profiling evidence. The synthetic investigation additionally proved concurrent per-thread cProfile instances conflict on interpreter slot 2. This revision follows root's approved finite-worker-only scope and emits bounded exception class/phase labels without exception contents. The callback inventory is corroborated by CPython 3.12's [original profiler implementation](https://github.com/python/cpython/blob/3.12/Modules/_lsprof.c). Neither synthetic correctness nor profiled actual timing qualifies the performance target.
