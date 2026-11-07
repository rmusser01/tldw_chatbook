# Console return and CSS matching follow-up (TASK-35611)

Native Windows 10, Python 3.12.10, Textual 8.2.8, 170×48 headless real TldwCli.
Baseline: `f9cb81835d80c272a2eaba665bdba95f92f020a8` (PR3166 before this follow-up).

## Investigation and change

The previous six-return comparison regressed from 1066 to 1129 ms median and
1617 to 2523 ms worst. The slowest final return was first; the unchanged repeat
(`matching-before`) had a 444 ms first return and a 1749 ms slowest last return.
This does not reproduce a stable first-return penalty. Both versions' sampled
CSS work comes predominantly from Textual's `ScreenResume` full stylesheet
refresh. That refresh remains intact.

A separate actual-matcher observer counted 16960 rule checks with 2059 matches
in one Console update. The leading-name-only filter let absent widget-type
scopes through; a comma alternative with an unrelated bare subject also kept
other impossible alternatives alive. The filter now requires all known static
type/class/id names in each alternative, splitting the final subject compound
from ancestor requirements at the last child/descendant boundary. Every
alternative must be impossible before a rule is rejected.

Candidates alone are cached, under the existing bounded LRU. Types follow
Textual's CSS inheritance. Live names and stylesheet parsing invalidate the same
cache; upstream still decides co-location, ancestry order, pseudo-state,
specificity, theme values and recursive component styles. No storage, config,
native worker ownership, source identity, permission or authority code changed.

## Same-screen benchmark

Four rounds alternate original/current execution order. Each arm starts with its
own empty candidate cache, then five warm full updates supply its round median.
Widget and component style maps are compared after each arm. Separate matcher
counters run outside the timed sections. Source: `matching_benchmark.py` and
`plain-ready-matching-benchmark.json`.

| Round | Original warm median ms | Updated warm median ms | Original cold ms | Updated cold ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 355.56 | 222.15 | 231.59 | 664.15 |
| 2 | 196.74 | 124.03 | 238.82 | 196.87 |
| 3 | 214.03 | 136.29 | 302.00 | 169.04 |
| 4 | 177.78 | 125.68 | 219.18 | 197.66 |

Median of warm round medians: **205.38 → 130.98 ms (36.2% faster)**.
Matcher calls: **16960 → 4233 (75.0% fewer)**; matching calls stay **2059**.
**526 widgets, zero base/component style mismatches**. The first updated cold
pass is worse; the other three improve. This is a focused CSS measurement,
not an end-to-end guarantee. The diagnostic run has no return-latency samples.

## Verification

- Three initial regressions failed on the original filter. The first direct
  pytest invocation never reached them because shared TEMP storage was refused;
  the valid red run uses genuine user-owned native fixtures.
- Initial CSS-inheritance test assumed inherit_css=False retained base-type
  matching. Inspection of Textual's actual matcher disproved that assumption.
  The corrected tests cover both inherited base types and explicit opt-out.
- **12 targeted tests passed**, including all-node real Console/Library/Settings
  style fidelity, same-ancestor-name unions in different orders, subject
  compounds, comma alternatives, dynamic disabled/focus/order states, source
  replacement, themes, component styles, cache bounds and no node retention.
- Independent read-only review found no blockers; its stale overview comment
  finding was corrected. No full-suite sweep was run.

## Plain return profiles

Each row is six Home/Console returns in a fresh private profile. The final four
runs execute original→updated, then updated→original, serially without concurrent
tests. Both complete pairs improve median and worst observed input delay. The
earlier original repeats are retained rather than selected away.

| Run | Median input ms | Worst input ms | Median attach ms | Median per-return worst heartbeat ms |
| --- | ---: | ---: | ---: | ---: |
| Unchanged discovery repeat (`before`) | 1376.08 | 1749.29 | 1458.50 | 426.13 |
| Initial control (`control1`) | 935.27 | 1815.20 | 938.14 | 271.71 |
| Pair 1 original (`control2`) | 972.30 | 1199.68 | 977.04 | 320.32 |
| Pair 1 updated (`after2`) | 858.42 | 1043.25 | 863.49 | 208.71 |
| Pair 2 updated, run first (`after3`) | 814.91 | 1257.81 | 817.29 | 230.09 |
| Pair 2 original, run second (`control3`) | 911.49 | 1445.55 | 912.60 | 293.79 |

Median input improves **11.7%** and **10.6%** in the two completed pairs.
Worst input improves **13.0%** in each; median worst heartbeat improves **34.8%**
and **21.7%**. All 24 paired returns retain their drafts and finish attach, with
zero network attempts or real-profile refusals. All four processes exit 0.
Clock-safe probe polling/heartbeat remains 31.25 ms, above the 15.625 ms loop
clock resolution; no product timer or storage policy is relaxed.

Paired main-thread `_check_selectors` samples total **69 → 29**. These are
overlapping stacks, not additive times. Config posture, inspector construction
and Home notification reads still appear. The measurements establish less CSS
work and improvements in these small native headless pairs, not universal or
production tail guarantees. Draft updates do not measure completed paint or
Send/Stop actions. See `matching-comparison.json` for per-return values and
source SHA256. Lint and format checks pass on all nine touched Python files.

One plain updated run (`matching-after1`) terminated abnormally after four
completed returns and a fifth input observation. Its 60-second diagnostic stack
dump is truncated in `windows_files._stat_handle`, with no Python exception or
final report/sample publication. The shell reported failure but did not retain
the native exit code. It is **incomplete and excluded from timing comparisons**,
not silently treated as a successful faster run or attributed to the baseline.
Further original/current runs retain explicit exit-code records.
Both subsequent updated runs and both repeated original controls complete; the
abnormal exit was not reproduced. Its cause remains unclassified, and this CSS
change does not claim to fix it. The partial JSON and stack dump are retained.

## Reproduction

From the repository root (native user-owned fixture execution):

```powershell
.venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/matching_tests.py Tests/Performance/test_css_static_requirements.py Tests/Performance/test_css_candidate_reuse.py Tests/Performance/test_textual_css_fastpath.py::test_ancestor_requirements_only_claims_what_it_can_prove Tests/Performance/test_textual_css_fastpath.py::test_upstream_apply_still_has_the_shape_the_fastpath_assumes Tests/Performance/test_textual_css_fastpath.py::test_fastpath_computes_identical_styles_for_every_node -q
.venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/probe.py --ready --resume-only --clock-safe --matching-benchmark --tag matching-benchmark
.venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/probe.py --ready --resume-only --attach-poll --clock-safe --matching-baseline --tag matching-control2
.venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/probe.py --ready --resume-only --attach-poll --clock-safe --tag matching-after2
.venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/probe.py --ready --resume-only --attach-poll --clock-safe --tag matching-after3
.venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/probe.py --ready --resume-only --attach-poll --clock-safe --matching-baseline --tag matching-control3
.venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/summarize_matching.py
```

The baseline loader reads the exact previous CSS module through `git show`, in
memory, without replacing the working tree. Matcher diagnostic and benchmark
timings are never mixed into the plain-return summary. Fresh profile paths are
recorded in each JSON and removed only after its process exits.
