# Windows recovery handoff lifetime — bounded source review

One actionable conditional test-harness gap; no demonstrated production launcher defect. No edits, native run or remote operations.

## Established production behavior

recovery_restart.restart validates the exact request, uses fixed interpreter/-P/-c and separate nonsecret hints with the existing filtered environment. Windows Popen explicitly forwards only standard descriptors0/1/2 with close_fds=True. After successful creation, os._exit(0) ends the old process; it deliberately does not wait while retaining old native leases. Spawn failure propagates and does not call _exit. POSIX retains execve. Ordinary CLI requests reach restart only after the guarded app shutdown returns a RecoveryRestart object.

History ae901141df documents actual Windows CRT exec failure (access violation/truncated script), the reason for CreateProcess-based Popen, and the existing PID/creation-time observation fixture. Current Windows unit tests verify exact argv/environment/standard-handle options and spawn-failure ordering; they mock the platform branch and do not prove interactive console focus behavior.

The old Windows PID returning0 proves successful spawn, not child completion or success. test_f9_replacement_workflow already accounts for this: stdout/stderr go to a file, then wait_for_restart identifies the fresh child by PID/create_time under the same total deadline. It also requires the actual successful replacement result and restored-content assertions. That helper kills only the verified child on timeout; an already-gone child still requires the separate successful product result.

## Conditional test-harness defect

Tests/Backup_Recovery/test_recovery_restart.py::test_actual_handoff_execs_fresh_recovery_ui delegates to test_home_citation_retirement._run, which uses subprocess.run(capture_output=True, timeout=70). Correction verified by root against exact4244 artifact d061c3e25d6c54c2.txt: _run, subprocess.run and communicate orig_timeout all use70s. There is no110s override. The113.091s testcase duration includes approximately43s of native_package fixture/build setup. The conditional cleanup limitation concerns the original70s bound.

The fresh Windows child inherits the captured stdout/stderr handles. Thus communication normally stays open until the fresh child closes/exits, despite old PID0 having already exited. The fresh child's required success marker and actual origin/UI assertions prevent treating the old exit alone as normal successful coverage.

But CPython3.12 subprocess.run's Windows TimeoutExpired branch kills its directly tracked process and then calls communicate() without a timeout. Windows _communicate joins pipe-reading threads until EOF. If the fresh child remains alive holding those inherited pipes, killing the already-exited old parent does not terminate it, and this timeout cleanup can wait beyond the intended test bound. The handoff test does not identify/retire that fresh child. This is a source-proven conditional gap, not a reproduction or attribution of any specific Windows failure.

Smallest existing-pattern correction to consider separately: use the already tested file-output plus exact child identity/completion observation for the restart test, preserving its original total deadline and actual child assertions. Do not change production exit/lease behavior to make a test wait for the old PID. No implementation is proposed here without root approval.

CPython source inspected locally: cpython3.12.11 subprocess.py549–564 (run timeout cleanup) and1598–1633 (Windows reader threads); no platform simulation or remote execution.

## Not established

Whether a particular Windows shell displays a prompt or competes for console input before the replacement RecoveryApp exits depends on the actual shell/console launch lifecycle. Popen's shared standard handles and early old-PID exit alone do not prove an observed foreground/input defect. Current headless tests do not establish that interactive behavior. No launcher redesign is justified from this review.

The default later driver writes stdout to a file and performs no restart. The pipe-descendant condition above cannot explain its timeout. Any attribution there requires its own retained execution evidence.

Reviewed hashes: recovery_restart.py893ebe229f366d243e372f4c4765512e329e55cd0c6ddf463bd299d76c74d809; test_recovery_restart.pyfd81f1c6425883caf82f601139eccdb8ac45e9eca6f4df3977f5539869b4317d; restart_observation.py5bdbf3019bd71c5bdd4dd6e271dfdae3d74d67a44708368e425606ee47c8f724.
