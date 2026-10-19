# Console return input queueing — TASK-35614

Baseline: `191fb7d7a3368062d8c4adea2f053ba4588d4f45`.
Python 3.12.10, Textual 8.2.8, native Windows, 170×48 headless viewport.

## Diagnosis and changes

The saved 2,285.60 ms return in `plain-ready-subject-final.json` was mostly
waiting: 152 of 178 main-thread samples in that stage were in the event-loop
poll. Textual's `Screen._invoke_and_clear_callbacks` awaited Console attachment
reconciliation inside the screen message pump. Its native-read await allowed
loop progress but prevented that pump from dispatching the typed key.

A mounted regression holding reconciliation reproduced the blocked key. The
after-refresh callback now synchronously launches a deduplicated finite
`console-attach-reconciliation` worker. The reconciler itself retains its visit
checks, phases, bounded retries, source checks and runtime answerability gate.
The new group participates in existing exact-worker capture and shutdown drain;
an older visit's physical read is not cancelled to launch its replacement.

Send/Stop action updates also performed nine immediate style refreshes for one
combined class transition. Native `update_classes` reduces that to two while
preserving unrelated classes, final styles, disabled state and tooltips.

The first ABBA profiles exposed a second issue: resume scheduled hook display
refresh before attachment and then again from reconciliation's completed resume
body. The early publication could synchronously enter config admission while
FULL was awaiting its read. Resume now schedules that display refresh after
reconciliation. Independent session-change and Send refresh paths remain.

## Preserved safety

No storage/config/hook authority implementation is changed. The eight normalized
source hashes in `return-safety-sources.json` match the pinned baseline. No TTL,
path posture, ownership, source/session/visit, pause, nofollow or physical
retirement check is removed. The worker-group addition retains existing custody.

Input becoming available does not acknowledge runtime attachment. Send may enter
its existing Preparing state and still performs fresh configuration, hook review
and admission checks. If attachment fails, the prior hook indicator count may
remain until successful reconciliation or explicit Hooks/Send refresh; that
presentation snapshot grants no permission.

## Initial serial ABBA experiment

These runs include the worker and class batching, before removing the duplicate
resume-triggered hook refresh. Six Home/Console returns per fresh process:

| Run | Median input | Worst input | Median attach | Median largest loop stall |
| --- | ---: | ---: | ---: | ---: |
| Original 1 | 551.95 ms | 1354.34 ms | 650.15 ms | 89.90 ms |
| Updated 1 | 153.10 ms | 172.49 ms | 968.40 ms | 91.14 ms |
| Updated 2 | 157.26 ms | 171.52 ms | 967.79 ms | 89.94 ms |
| Original 2 | 575.37 ms | 1186.89 ms | 579.92 ms | 102.89 ms |

Input improved 72–73%, but attachment regressed. This prompted the additional
hook scheduling fix rather than treating early typing as proof all work sped up.
Updated run 2 retained an attachment outlier of 1,588 ms, with 169 ms input and
294 ms maximum loop overshoot. Hook-driven control refresh appeared in 19 samples
(9 ending in a native wait), versus 9 in updated run 1 and 3 in original run 1.
These are overlapping stack counts, not CPU durations; stage samples cover both
input and attachment and cannot place individual stacks before/after the key.

## Final combined-code comparison

A final updated/original pair uses all three fixes and the same pinned baseline:

| Run | Median input | Worst input | Median attach | Median largest loop stall |
| --- | ---: | ---: | ---: | ---: |
| Final updated | 177.79 ms | 200.64 ms | 976.42 ms | 105.36 ms |
| Final original control | 721.84 ms | 1532.84 ms | 727.33 ms | 115.10 ms |

Observed input improved **75.4%**, and the worst input sample improved **86.9%**.
The historical 2.29-second input outlier did not recur in the updated runs; this
is a small controlled workload, not a guarantee about all future outliers.
**Attachment remains 34.2% slower in the final pair.** Removing the duplicate
hook refresh did not establish faster completed reconciliation. Earlier usable
input is the verified benefit; background completion remains the next target.

All **36** profiled returns retained their typed draft and completed attachment.
All six processes exited zero, with zero network attempts and profile refusals.
The final run has 21 inclusive CSS-application samples and 9 config-posture
samples, versus 27/9 in the final control. Upstream resume-time full styling and
required fresh config work remain. The final resume stages have no hook repaint
samples; that does not establish zero hook cost outside the sampled stage.
No broader CSS-refresh suppression or cross-await config cache was introduced.

## Verification

- The input queueing, redundant class refresh and early hook-refresh regressions
  each failed before their corresponding fix and passed afterward.
- Targeted mounted attachment/retry, stale-visit, original FULL/workspace native
  read, cancellation/physical retirement, draft, Send/Stop and hook freshness
  checks passed after correcting fixture issues described below.
- The timer census retains the original rearming root through the exact
  same-class `self.run_worker(self.reconcile)` edge. A mutation test proves the
  repaint remains reachable; no inventory pin was removed. Portable relative
  paths fix the census's pre-existing Windows separator mismatch. Four targeted
  inventory checks passed.
- A bare retry fixture lacked the workspace preparer and failed on the original
  code too; it now supplies that unrelated dependency. Four domain test doubles
  were adapted to the synchronous launcher. The Stop widget test now asserts its
  synchronous artificial projection before the real idle runtime repaints it.
- The hook unsafe-directory test used POSIX chmod, which does not change Windows
  DACLs. It now uses the existing native ACL fixture, verifies real refusal, and
  restores the exact original DACL. The native Windows assertion passed.
- New diagnostic/regression files pass Ruff lint and format checks. Changed
  existing Python files add no lint findings; the three product modules retain
  their baseline 22/18/1 findings and pre-existing formatting debt. See
  `return-lint.json`. No full-suite sweep ran.
- Independent read-only review found no product blockers and reviewed the worker
  custody, draft/Send behavior, fixture changes and timer inventory approach.

`return-verification.json` retains initial failures and successful rechecks.
The broad targeted run was 42 pass/5 fail; four launcher fixtures and the
synchronous Stop projection were corrected. The follow-up native checks passed,
then the timer inventory's portability/worker-edge issues were fixed and all four
targeted inventory cases passed. The hook follow-up was 9 pass/1 fail; replacing
its ineffective Windows chmod fixture with a real ACL change passed. An earlier
known-xfail bare-screen overlap test remains xfailed, unchanged.

## Reproduction and limits

```powershell
python Docs/Development/evidence/windows-performance-2026-10-18/run_return_profiles.py
python Docs/Development/evidence/windows-performance-2026-10-18/run_return_profiles.py --final
python Docs/Development/evidence/windows-performance-2026-10-18/run_return_profiles.py --final-control
python Docs/Development/evidence/windows-performance-2026-10-18/summarize_return.py
```

The original arm loads the exact three target modules from the baseline in
memory. Receipt source hashes describe the checkout, not substituted bytes.
The original ABBA table above represents the intermediate patch; running the
runner on the final patch produces a new experiment. Keep tests and profiles
serial. The clock-safe 31.25 ms polling/heartbeat floor is retained. Measurements
are posted-key observation and attachment completion, not terminal paint latency,
long conversations or completed Send/Stop journeys. All outliers are retained.
