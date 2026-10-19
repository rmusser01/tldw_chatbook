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

## Continued profiling after PR #3166

A second fresh-profile tour on commit `4c9755fe51` again recorded **zero**
main-thread tokenizer-loading samples. Worst idle overshoot was **19.55 ms**,
median typing **3.83 ms**, worst typing **33.50 ms**, and idle process CPU
**0.219 s**. The final actual display was prepared and reported **11 tokens**,
so the improvement did not come from permanently withholding the estimate.
No network attempts or real-profile refusals occurred. See
`plain-ready-tokenizer-post-pr.json` and the updated comparison.
The four disposable profiling homes created for this follow-up were removed
after their processes exited. Baseline scripts pin original revision
`69771c1811e734421f0bdf31ed90969f8c1f0c63` so later commits cannot change the control.

`warm_token_math.py` then isolated original counting with a prepared backend
using generated ASCII code-like text (seven repetitions per size):

| Characters | Uncached median, backend warm | Memo-hit median |
| --- | ---: | ---: |
| 1,000 | 0.12 ms | 0.003 ms |
| 10,000 | 1.02 ms | 0.004 ms |
| 100,000 | 5.30 ms | 0.006 ms |
| 1,000,000 | 64.94 ms | 0.018 ms |

The 1 MB maximum was 90.53 ms; readiness source checking averaged 0.032 ms.
This is a single synthetic workload and excludes TUI rendering, provider work
and non-ASCII/custom encoders. Existing memoization is already effective for
unchanged text; these measurements do not justify another token cache. Large
changing drafts remain a CPU-counting candidate, distinct from backend loading.

The repeated tour still samples synchronous
`ConsoleRuntime.recompute_console_attention` → `list_console_unseen_marks`
on resume (18 overlapping samples) and `get_active_workspace` (19). Inspection
confirms attention recomputation also publishes notifications, handles exact
receipt visibility, and serializes acknowledgements under its operation lock.
The next bounded investigation should measure original attention reads and
coalesce their *read-only input preparation* without moving the entire callback
or weakening acknowledgement/authority boundaries. Workspace reconciliation is
also authoritative; a display memo cannot replace its live checks. Separately,
the marks query fetches every matching row before applying its valid-receipt
limit in Python; large-history profiling is needed before changing that path.
