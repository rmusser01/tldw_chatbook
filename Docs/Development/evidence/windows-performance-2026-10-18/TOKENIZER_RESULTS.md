# Console tokenizer initialization on Windows (TASK-35591)

Cold token-estimation backend loading now runs through the existing Console
readiness projection's finite worker. The original custom-manager constructor
still performs native storage admission; tiktoken still loads its original
encoding. Context and cost displays wait for their selected backend instead of
initializing it synchronously on the UI loop. Live action counting is unchanged.

Readiness is bound to model/provider, manager and mapping identities, stock
callback sources, and the existing session/config/runtime owner. Cancellation
retains the physical read until it exits. No permission verdict or storage
lease is cached. Config admission ends before tokenizer preparation starts.

Custom callbacks and heuristic custom-tokenizer directory selection retain the
direct path. Cold discovery of a heuristic mapping invalidates its request so
the next capture takes that same fallback. Failed backend loading keeps optional
estimates unavailable and retries through the existing display cadence; it does
not repeatedly retry the failed loader on the UI thread. This can suppress a
rough character estimate that the live counter would otherwise produce.

## Before/after evidence

Windows 10, Python 3.12.10, Textual 8.2.8, real native storage checks, fresh private
profiles, headless 170x48 viewport. Same route tour, twelve input keys, and
approximately three idle seconds. No provider requests, no guard refusals,
and no concurrent test workload during timing runs. Ordinary desktop/background
activity was not controlled; these are diagnostic samples, not statistical bounds.

| Measurement | Before | After |
| --- | ---: | ---: |
| Main-thread samples in tokenizer loading, whole tour | 7 | 0 |
| Worst idle event-loop overshoot | 158.29 ms | 70.98 ms |
| Median key-to-draft observation | 2.96 ms | 3.50 ms |
| Worst key-to-draft observation | 33.64 ms | 34.67 ms |
| Process CPU during idle window | 0.391 s | 0.281 s |

The cold-loading stall moved off-loop; typing was already fast after the prior
skill-context fix and did not improve here. The measured idle maximum decreased
55%, but this is one before/after pair. Remaining samples include checked config
admission, unseen attention-mark reads on resume, and workspace reconciliation.
Overlapping stack counts are not independent costs or elapsed-time attribution.

`tokenizer-comparison.json` and `plain-ready-tokenizer-{before,after}.json`
retain the measurements. `summarize_tokenizer.py` reproduces the comparison from
local `*-samples.json` files. Large raw stack dumps, profiler binaries and verbose
application logs remain local rather than inflating the PR. The probe and compact
timings are committed; run `probe.py --ready --tag tokenizer-<variant>` from the
repository root to reproduce with a new private profile.

## Verification

- 21 backend and projection controls pass, including actual exact-mapping JSON
  loading/counting, missing backend handling, cold heuristic fallback, custom
  callback rejection, source/model/session changes, coalescing and cancellation.
- A real mounted-app control holds the original manager's `_load_mappings` after
  its original storage admission. The UI keeps running, and the read is tracked
  by the actual runtime until retirement; this passed in two isolated runs.
- The related Console run recorded 149 passes, four deliberately deselected
  baseline failures, and one skill-control startup timeout before its assertions.
  The unchanged test passed separately on the final saved source. The startup
  timeout's cause was not established; a passing retry does not erase that
  observation (`console-pr-verification.log`, `console-skill-final-recheck.log`).
- The four baseline failures reproduce with original HEAD Console modules:
  one mode-bar fake lacks `_console_rate_reading`; three old token-limit tests
  expect 4096 where the current documented fallback is 32000. These are separate
  from the original eight failures repaired in TASK-35589.
- New files pass Ruff lint/format; modified ranges are formatted and `git diff
  --check` passes. Existing E721 findings in pre-existing source remain unchanged.
  No full-suite sweep was run.

ADR required: no. This is a scheduling fix applying existing finite ownership
and authority separation (ADR-126/230/231), with no new storage/service contract.
Independent review found the cold heuristic fallback issue; it is fixed and
covered by an original-constructor regression test.
