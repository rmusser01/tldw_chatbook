# Console mount baseline — 2026-08-21

Command:

```bash
.venv/bin/python Tests/Performance/run_console_mount_profile.py \
  --phase controls --iterations 30 \
  --output Docs/superpowers/qa/console-mount-2026-08-21/baseline-report.json
```

Environment: isolated scratch profile, Python 3.12.11, Textual Pilot viewport
170x48, fresh `ChatScreen` per navigation, 30 balanced warm samples per variant.
Variant order rotates each iteration to avoid systematically favoring a warmed
candidate. Each sample also sends a real key through the focused composer and
routes Enter through the prompt controller to its worker-launch seam.

| Metric | Baseline median / p95 | Inspector-empty median / p95 | Context-empty median / p95 |
| --- | ---: | ---: | ---: |
| First interactive | 766.1 / 1,188.2 ms | 681.1 / 1,218.9 ms | 602.8 / 1,038.3 ms |
| Full ready for measured tree | 1,423.7 / 1,660.1 ms | 1,136.2 / 1,644.7 ms | 893.2 / 1,325.0 ms |
| Focus restored | 1,423.7 / 1,660.1 ms | 1,136.3 / 1,644.8 ms | 893.2 / 1,325.0 ms |
| Outgoing detached | 1,022.6 / 1,473.4 ms | 937.4 / 1,519.5 ms | 835.3 / 1,267.6 ms |
| Screen unmount handler | 0.002 / 0.002 ms | 0.002 / 0.002 ms | 0.002 / 0.002 ms |
| Key to echo | 133.4 / 155.3 ms | 132.4 / 161.3 ms | 131.1 / 135.6 ms |
| Enter to worker | 21.8 / 23.9 ms | 21.0 / 24.0 ms | 22.1 / 24.6 ms |

Widget medians:

| Variant | Screen | Context rail | Inspector rail | Transcript | Composer |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 405 | 111 | 75 | 59 | 20 |
| Inspector empty | 331 | 111 | 1 | 59 | 20 |
| Context empty | 295 | 1 | 75 | 59 | 20 |

Decision: Inspector's maximum measured first-interactive improvement is 11.10%,
below TASK-19505's 15% threshold. Context's maximum is 21.32%, so it proceeds
to a real first-refresh hydration experiment under ADR-078. The empty controls
are not production implementations and their reduced full-ready times are not
used as evidence for the final deferred full-ready budget.
