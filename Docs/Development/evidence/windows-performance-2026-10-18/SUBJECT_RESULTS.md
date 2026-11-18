# Config derivation and CSS subject matching — TASK-35613

Current-code repeat: [SUBJECT_CONFIRMATION.md](SUBJECT_CONFIRMATION.md) preserves
the later Console return fix in both comparison arms. It confirms 18.5% faster
warm CSS application and 19 passing targeted checks, while recording flat/slower
overall input results rather than claiming a general latency gain.

Baseline: `f152dfc4a6c4fbf2fcd277543757a46a0d2874e2`, the rebased PR head.
Python 3.12.10, Textual 8.2.8, native Windows, 170×48 headless viewport.

## Changes and safety boundaries

- Independent readiness and provider/model-label builds each called settings
  loading three times. Both now enter the existing synchronous derivation scope:
  one current read per build, reuse of an enclosing scope, and cleanup on return
  or exception. Real Settings saves are observed by the next independent build.
  No memo crosses an await or becomes action/storage authority.
- The remaining matcher spent much of its time scanning ancestors for
  `App *:disabled:can-focus` and `*:focus`, even when the subject failed that
  state. The stylesheet matcher now checks a final pseudo-bearing selector on
  the subject first, using Textual's live `Selector.check`. False rejects early;
  every possible match still delegates the full upstream ancestry matcher.
- Dynamic rules remain in the candidate list so upstream hover, focus-within,
  sibling-order and odd/even refresh flags are preserved. No dynamic result is
  cached, DOM query matching is unchanged, and comma alternatives/specificity,
  themes, source order, `initial`, and component resolution stay upstream.
- Config path posture, storage admission, checked-display lifetime, source,
  ownership and pause checks are unchanged. `subject-safety-sources.json` records
  identical normalized source hashes for these safety modules. Required posture
  reads remain visible in profiles; reducing them would require separate design
  work, not extending this memo or weakening validation.

## Focused CSS benchmark

`subject_benchmark.py` alternates four original/updated rounds on the same mounted
Console, measuring one cold update and the median of five warm updates per arm.
Both `Stylesheet.apply` and the stylesheet matcher are restored for controls.

| Metric | Original | Updated |
| --- | ---: | ---: |
| Median of warm round medians | 165.34 ms | 129.41 ms |
| Warm round 1 | 127.69 ms | 109.37 ms |
| Warm round 2 | 145.51 ms | 121.36 ms |
| Warm round 3 | 185.17 ms | 137.46 ms |
| Warm round 4 | 191.14 ms | 143.80 ms |
| Rule checks / matching rules | 4,233 / 2,059 | 4,233 / 2,059 |

Warm styling improved **21.7%**, with **0 base/component style mismatches across
526 widgets**. Cold timings were noisy: original 185/211/210/228 ms, updated
142/181/**290**/205 ms. The slower updated cold round is retained, not discarded.

A separate instrumented update attributed matcher time to exact parsed selectors
(the old observer's `Selector.css` hid universal-selector pseudo suffixes).
Total observed matcher time was 94.00 → 36.56 ms; the disabled/can-focus rule
39.36 → 8.42 ms and focus rule 13.48 → 1.68 ms. These instrumentation timings are
diagnostic attribution, separate from the plain benchmark and return profiles.

## Serial plain return profiles

`run_subject_profiles.py` runs original → updated → updated → original, six
Home/Console returns per fresh process. Both arms retain the clock-safe Windows
polling floor and identical diagnostic machinery. Exact subprocess exits and
checkout source hashes are recorded; original modules are loaded from the pinned
commit in memory. Source hashes describe the checkout, not the substituted bytes.

| Run | Median input | Worst input | Median attach | Median largest loop stall |
| --- | ---: | ---: | ---: | ---: |
| Original 1 | 654.56 ms | 1682.37 ms | 663.90 ms | 117.87 ms |
| Updated 1 | 676.34 ms | 1567.21 ms | 681.50 ms | 90.80 ms |
| Updated 2 | 671.07 ms | 1277.10 ms | 673.43 ms | 98.11 ms |
| Original 2 | 685.24 ms | 1481.92 ms | 832.99 ms | 139.81 ms |

The first median pair is 3.3% slower; the reverse pair is 2.1% faster. **No
consistent median return-latency gain is established.** Worst input improves
6.8%/13.8% and loop stalls 23.0%/29.8% in the two pairs. All 24 drafts were
retained, all attaches completed, all four processes exited zero, and network
attempts/profile refusals were zero. These measurements concern observed input
and attach completion, not paint latency or completed Send/Stop journeys.

Remaining config-posture samples are 11/10 in controls and 5/12 in updated runs;
they have not been eliminated. Upstream matching samples are 14/6 versus 3/7,
with the new precheck appearing in 5/7 updated samples (overlapping stacks, not
additive time). Required config checks, CSS application and rendering remain.

## Verification and integration

- Three initial regressions failed for the expected redundant-work conditions.
- 21 distinct targeted checks passed: provider/readiness/inspector freshness,
  session selection, exception cleanup, static candidate reuse, dynamic pseudo
  states, upstream matcher contract, and real Console/Library/Settings fidelity.
- A first native readiness run timed out at 45 seconds during cold database
  initialization before its assertion. The integration fixture now has a bounded
  90-second allowance; the successful readiness rerun took 28 seconds. The initial
  failure remains in the local `subject-green.log` evidence.
- After the first rebase, all five focused config/CSS checks passed again.
- Modified tests, CSS helper and diagnostic helpers pass Ruff lint/format checks.
  ChatScreen retains exactly its baseline 22 lint findings and pre-existing
  formatting debt (`subject-lint.json`); no full-suite sweep was run.
- Independent review found no blockers and requested the added upstream matcher
  success-condition guard. All storage safety implementation checks remained intact.
- A concurrent rebase was detected in the shared checkout. Work moved into an
  attached isolated worktree; no rebase state was changed in the other checkout.
  After measurement, another task added the recorded-spend correction
  `1bebf20b97196df1d847a7a24f7276c5cb483cdd`. This follow-up was rebased on it.
  The ABBA table above remains evidence against `f152dfc4`, not an A/B claim about
  that independent spend change. Auto-merge was disabled before this PR update,
  as required by repository instructions.

Final combined-code validation added two measured-spend compatibility cases to the
21 checks above (**23 distinct targeted checks passed**). All five final spend/CSS
checks passed. An updated-only six-return run then measured median input 885.85 ms,
worst 2285.60 ms, and median largest stall 143.81 ms. Because this was slower than
the earlier runs, a following control used the same combined app with original
target modules (the `f152dfc4` target files are byte-identical to `1bebf20b`). Its
median was 909.63 ms, worst 1471.62 ms, and stall 237.59 ms. This extra pair again
does **not** establish a robust median gain, and the updated first-return outlier
is substantially worse. No general worst-latency improvement is claimed across
both integration states. All 12 additional drafts/attaches succeeded, both exits
were zero, and guard counts remained zero. See `subject-final*-exit.json` and the
raw `plain-ready-subject-final*.json`; the checkout revision is recorded explicitly.

Reproduce the controlled comparison with the original surrounding app at
`f152dfc4` plus this follow-up's product changes:

```powershell
python Docs/Development/evidence/windows-performance-2026-10-18/run_subject_profiles.py
python Docs/Development/evidence/windows-performance-2026-10-18/summarize_subject.py
```

Use `run_subject_profiles.py --final` for an updated-code-only integration run.
Use `--final-control` for the corresponding original-target-module control.
Do not run tests concurrently with timing profiles. Private diagnostic profiles
are owned fixtures under the real home; remove only exact recorded fixture paths
after their processes exit.
