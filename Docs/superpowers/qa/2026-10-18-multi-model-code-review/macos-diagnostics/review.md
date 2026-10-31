What this change does: The retained-eval restore child replaces its repeating C traceback watchdog with the existing Python-owned thread observer. The restore body remains inside a try/finally, with cleanup delegated to the existing exception-preserving stop_observer helper.

Review scope: uncommitted .github/workflows/eval-recovery-regressions.yml, Tests/Backup_Recovery/test_eval_retained_definitions.py, Tests/Backup_Recovery/test_eval_restore_diagnostics.py, and the referenced existing observer/child-runner helpers. Base HEAD verified as 73112b77977b64979fda79c341907807dd5ecfda. Static review only; no source edits, additional test runs, live requests, qualification, PR changes, or subagents.

Must fix:

1. [P2] Preserve the newly relocated diagnostics in hosted CI artifacts (Tests/Backup_Recovery/test_eval_retained_definitions.py:66; .github/workflows/eval-recovery-regressions.yml:99).
   What this is: The old watchdog wrote to the restore child's captured output. The replacement writes restore-stacks.json and restore-stacks-fatal.log under the child's private restored/home directory.
   Problem: The child runner reads only restart-output.log. The workflow uploads JUnit on every OS and three output-log paths only on Windows; neither new file is collected anywhere. If the macOS child times out or segfaults again, its new periodic snapshots and fatal trace disappear with the hosted runner, leaving less accessible crash evidence than before.
   Fix: Preserve these two exact owned files on workflow failure for every OS, using a predictable/private pytest root or another safely resolved fixture location. Keep file selection narrow; no product or deadline change is needed.
   If we skip it: A recurrence cannot be investigated using the diagnostics this repair adds.

Other checks: The product restore script and all its assertions are unchanged apart from indentation inside try/finally. Its sole existing journey caller still relies on _run's unchanged 45-second restore timeout. The observer samples at 10 seconds, bounds records to four snapshots/32 threads/64 frames, records only basename/function/line metadata, writes atomically with private file modes, and retains native fatal reporting without dump_traceback_later. Cleanup joins the observer and preserves an active product exception if cleanup fails. The two new controls execute the actual assembled child script, forbid the C watchdog, wait for actual observer output, inject failure at the first guarded child entry before restore file/network work, and check cleanup/error identity. The workflow includes those controls. No new dependency, product permission, storage boundary, retry, timeout extension, or weakened restore assertion was introduced.

Evidence reused: /private/tmp/task35582-macos-diagnostics-green.log reports 10 passed in 84.62s, including the original journey (83.27s). git diff --check for the reviewed tracked paths passed. Parent-reported RED controls and earlier baseline/accelerated runs were not independently rerun.

Verdict: Fix finding 1 before publishing the diagnostic repair.
Not checked: Hosted macOS execution and the native SIGSEGV cause were not reproduced or proven; local green tests establish the diagnostic contract, not native crash causality. No full suite or live qualification was run by this reviewer.


Follow-up disposition — 2026-10-29

Finding 1 is resolved in the revised workflow. The locator runs after Windows TEMP/TMP setup, uses Python's actual tempfile.gettempdir(), and appends EVAL_TEMP_ROOT to GITHUB_ENV without overwriting prior values. The existing test launcher does not substitute a pytest base directory. The failure-only artifact step now runs on every OS with matrix-specific names and preserves the three existing output-log leaves plus exactly restored/home/restore-stacks.json and restored/home/restore-stacks-fatal.log under pytest-owned fixture roots. Test invocation, private Windows setup, restore body, and deadlines remain unchanged.

Reviewed only this workflow delta and the existing launcher behavior. Parent reports YAML parsing, actual locator execution/environment-preservation/glob checks, Ruff, format, and diff checks passed; those checks were not rerun here. No additional source edits, tests, live work, or PR changes.

Final verdict: No remaining actionable findings within the scoped diagnostic repair. Clear to publish for normal same-head CI; this does not establish the native macOS crash cause or authorize a retry/qualification/merge.
Not checked: Actual hosted artifact upload and native SIGSEGV causality remain unverified locally.
