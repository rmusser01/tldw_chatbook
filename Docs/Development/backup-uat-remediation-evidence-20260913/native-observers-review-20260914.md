# Native diagnostic observers — independent review

Final disposition: APPROVED after the correction documented below. The initial RED evidence and finding are retained. No source edits by reviewer.

## Finding

startup_timing_diagnostic.py main(), final emit(): a snapshot or stderr-write failure replaces the original pytest exception or return outcome. Independent temporary probe made pytest.main raise a sentinel RuntimeError and snapshot raise OSError during final emission; main propagated OSError instead of the original RuntimeError. Evidence /private/tmp/uat-startup-diagnostic-independent-error.log (1 failing test, 2.32s), probe /private/tmp/test_startup_diagnostic_review.py. The same boundary enables cProfile and starts the thread before entering try/finally, so a thread-start failure can leave profiling enabled. Make setup/cleanup and emission preserve the original outcome, keep diagnostic failures bounded/type-only, and add exception/exitcode/setup cleanup regressions.

## Other reviewed properties

The unavailable-dependency observer follows inventory.py's actual dependency_unavailable predicate, including unvalidated deletion. It observes at most 4096 items and 64 dependencies each, emits at most 64 edges and eight records, hashes both logical identifiers, and does not infer missing dependencies from a truncated tail. Output carries bounded owner/status strings, not paths or payload/config contents.

The startup profiler emits only source-relative code coordinates, call counts and aggregate times, top 20 by total/self time. It explicitly identifies profiling overhead; these are diagnostic measurements, not an unprofiled latency acceptance result. It starts a fresh process with the selected source first on sys.path and changes cwd before importing pytest; both source trees' conftests select private fixture roots before product imports. source_matches reports the loaded product package root. The dev/candidate test fixture implementations differ historically, which remains a comparison qualification, not a guard bypass.

The workflow runs only after a Windows job failure, retains the original failing step, adds no dependency, and uses continue-on-error only for the subsequent diagnostic. It selects the exact dev4631 source in a detached worktree and runs the same named GGUF case on dev and candidate in separate interpreters. Existing --timeout=60 and 20-minute job ceiling are unchanged. No blanket admission deferral, product mode or runtime guard change is present.

## Final correction and independent verification

The sampler starts inside the profiling cleanup boundary. A RuntimeError during start is retained by class while the original pytest call still runs. Optional snapshot/stderr failures retain only the latest four exception class names; they cannot replace the original pytest exception or exit code. Finally disables profiling, stops/joins a started sampler with a bounded wait, and performs best-effort final emission. No exception values or local variables were added to output.

The independent original-error probe that previously failed now passes unchanged. Together with the seven final observer regressions, independent verification passed 8 tests in 0.67s: /private/tmp/uat-startup-observer-independent-final.log. These cover snapshot, output, and sampler-start failures against both original exception identity and a nonzero return value; profiling is disabled afterward. The provided final Bandit receipt contains zero findings. Existing unrelated dependency and pytest temporary-directory cleanup warnings did not change the test outcomes.

No remaining actionable finding in this five-file test/diagnostic/workflow scope. This approval does not assert native Windows execution or interpret profiled duration as startup-performance acceptance.

Final SHA-256 receipt:

```json
{
  "Tests/Backup_Recovery/thread_diagnostics.py": "3f752216934c3a9203f82083fde265ad23d7983360b32da297583b2e49554a15",
  "Tests/Backup_Recovery/test_thread_diagnostics.py": "c7bd9b9a64089d3980c1087c9b7cda34ba1fdf9e46642ae48513dd6a67201029",
  "Tests/Backup_Recovery/startup_timing_diagnostic.py": "3b1a8ff0d4b8e55dad9343e8e0d7e43967d738bfc26f0162788f7eb4ee6c4ebc",
  "Tests/Backup_Recovery/test_startup_timing_diagnostic.py": "dd7f3c4e0dd9a05d642059365002ecc758972c00e073b558325eaf84fb9a459a",
  ".github/workflows/task-2062-2-gguf-source-evidence.yml": "0e68781765dfff8240b313aeec0a179c9a85a88e71f89100e30635a89cd98759"
}
```
