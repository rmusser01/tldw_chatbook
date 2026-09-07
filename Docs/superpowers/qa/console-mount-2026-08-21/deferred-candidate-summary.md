# Console deferred-Context candidate — 2026-08-21

Command:

```bash
.venv/bin/python Tests/Performance/run_console_mount_profile.py \
  --phase production --iterations 30 \
  --output Docs/superpowers/qa/console-mount-2026-08-21/deferred-candidate-report.json
```

Environment: isolated scratch profile, Python 3.12.11, Textual Pilot viewport
170x48, fresh `ChatScreen` per navigation, 30 balanced warm samples per
variant. Variant order rotates every iteration. The profiler injects the
candidate only inside the measurement process; production remains eager.

The first-interactive timestamp is the first `call_after_refresh` boundary.
Full-ready waits for the candidate hydration callback to finish. Input probes
run only after that full-ready condition, so hydration work cannot be
misclassified as key or Enter latency.

| Metric | Eager median / p95 | Deferred median / p95 | Change |
| --- | ---: | ---: | ---: |
| First interactive | 774.9 / 1,295.3 ms | 633.0 / 1,155.5 ms | 18.31% faster median |
| Full ready | 1,237.5 / 1,736.4 ms | 1,239.1 / 1,663.9 ms | 0.13% slower median |
| Focus restored | 1,237.5 / 1,736.5 ms | 1,239.1 / 1,664.0 ms | 0.13% slower median |
| Outgoing detached | 1,048.4 / 1,578.6 ms | 734.6 / 1,249.4 ms | 29.93% faster median |
| Key to echo | 140.5 / 429.2 ms | 134.4 / 142.5 ms | 66.79% better p95 |
| Enter to worker | 21.3 / 22.6 ms | 21.2 / 25.4 ms | 12.39% worse p95 |
| Screen unmount handler | 0.002 / 0.003 ms | 0.002 / 0.002 ms | no material change |

First-interactive widget medians:

| Variant | Screen | Context rail | Inspector rail | Transcript | Composer |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eager | 401 | 110 | 75 | 56 | 20 |
| Deferred | 293 | 2 | 75 | 56 | 20 |

Both variants converge to the same final medians: 405 screen widgets, 111
Context-rail widgets, 75 Inspector widgets, 59 transcript widgets, and 20
composer widgets.

Decision: reject the candidate. It clears the 15% first-interactive target and
the full-ready/key-latency budgets, but Enter-to-worker p95 regresses 12.39%,
above the 10% ceiling in TASK-19505. No deferred mount behavior remains in
production.
