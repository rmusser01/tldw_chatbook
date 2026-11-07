# Console skill-context follow-up

The original eight reported failures are repaired (TASK-35589), and Console
skill discovery now runs its original checked context read off the UI loop
(TASK-35590). Storage admission, policy, trust and source checks are unchanged.

## What changed

`UI/Console_Modules/skill.py` retains the existing cold-trust preparation, then
recaptures the current stock source. The original decorated scope context call
runs through the existing `run_preparation_read` worker lifetime. Its physical
read is tracked by the original runtime through cancellation and shutdown.
Source and attachment checks precede publication. Custom context/catalog,
policy, disabled-builtins and trust-status implementations keep their direct
execution path. There is no new catalog cache or reused permission verdict.

The context body was async without suspending: most sampled time was Windows
storage admission before catalog enumeration. Moving only the body beneath its
decorator would have left those checks blocking the loop.

## Native Windows measurements

Python 3.12.10, Textual 8.2.8, Windows 10, 170×48 headless Textual viewport.
Fresh private profiles, actual native file/storage checks, null keyring, blocked
network and protected real profile. No mocked catalog or storage reader. No
concurrent test workload during timing runs. Sampling selects the main thread.

| Focused context workload, six reads | Before | After |
| --- | ---: | ---: |
| Median per-read maximum event-loop overshoot | 142.08 ms | 14.46 ms |
| Worst event-loop overshoot | 183.46 ms | 176.62 ms |
| Median total context-read time | 151.79 ms | 189.14 ms |
| Main-thread samples in context storage admission | 60 | 0 |

The typical UI stall decreased about 90%; the underlying checked reads did not
get faster. The first fixed read overlapped other UI work (570 ms total), and
its 177 ms loop stall must not be hidden. Later fixed overshoots were 7–15 ms.
Every read returned one available builtin and no blocked skills. This is an
empty-profile responsiveness measurement, not a large-library throughput claim.
`input_observed_after_fetch_ms` is an upper bound recorded after awaiting the
fetch; it is **not** independent keystroke latency.

| Full navigation/typing tour | Previous Settings/Library fix | Console fix |
| --- | ---: | ---: |
| Median of 12 keystrokes | 3.07 ms | 2.36 ms |
| Worst of 12 keystrokes | 1,167.21 ms | 30.34 ms |
| Main-thread context-admission samples, whole tour | 22 | 0 |
| Idle CPU, approximately 3 seconds | 0.656 s | 0.344 s |
| Worst idle loop overshoot | 26.53 ms | 144.84 ms |

These are one tour per version, with scheduling variability. The improvement
does not establish a global latency bound. Both focused and full fixed profiles
record zero network attempts and zero real-profile guard refusals. Timing runs
preceded final additional custom-object classification checks; the final source
is verified by the targeted native tests.

Remaining main-thread work in the fixed idle interval includes context/spend
summary token counting, `get_tiktoken_encoding`, and custom-tokenizer manager
storage admission. That is the next bounded investigation. Active-workspace and
durable attention reads remain other candidates from the previous tour; no
authority/cache shortcut was applied to them.

## Verification and limits

- Original eight failures: 37 authoring/resolver/lasting-sync tests pass. All 22
  Notes lasting-sync/review/root cases also passed in related coverage.
- Console: 38 targeted tests pass, including actual mounted native context
  reads, event-loop progress during a held original callback, repeated
  cancellation with physical-read retention, stale source rejection, custom
  policy thread affinity and custom callback qualification.
- Independent review identified missing wrapped-body and inner trust-status
  checks; both were added and tested. The native hold is correlated with its
  exact issuing task, avoiding interference from background discovery.
- New test files pass Ruff lint and formatting. The Console module retains the
  same four existing E721 diagnostics as HEAD; no new lint diagnostics. New
  implementation ranges are formatted; compilation and diff checks pass.
- No full test suite was run. A wider related prompt run exposed 14 additional
  inherited failures; all reproduce with the exact original HEAD fixture in a
  separate process. They are outside the requested original eight. See
  `compatibility-repair-related.log` and
  `compatibility-related-fixture-baseline.log`.
- Existing tiktoken asset edits predate this work and were left untouched.

## Reproduction and retained artifacts

From the repository root, with the normal user's native Windows fixture owner:

```powershell
.venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/probe.py --ready --skills-only --tag skills-fixed
.venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/probe.py --ready --tag console-skills-fixed
```

The focused baseline was captured before replacing the original awaited context
call; the qualification helpers were already present but unused. Raw timing and
stacks: `plain-ready-skills-original.*`, `plain-ready-skills-fixed.*`, and
`plain-ready-console-skills-fixed.*`. `summarize_skill_context.py` reproduces
`skill-context-comparison.json`. Final targeted output:
`console-context-final-verified.log`; original repaired failures:
`compatibility-repair-final.log`. Earlier failed diagnostic test runs are retained
with their names; they are not passing evidence.

ADR required: no. This applies the existing finite native ownership and
authority separation contracts in ADR-126/230/231 without changing service,
storage, permission or persistence semantics. Architecture notes are updated in
`Docs/Architecture/console.md`.
