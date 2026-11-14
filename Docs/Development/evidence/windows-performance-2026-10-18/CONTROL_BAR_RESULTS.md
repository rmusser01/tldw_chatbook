# Direct control-bar native configuration entry — TASK-35646

Baseline: `473c32b4edbcf4c17e982045a749b5e7c6bc49b7` (preceding PR head).
Product commit: `e9c894e9bc9696aca60317fc2029f6709dcf9d0b`.
Native Windows 10, Python 3.12.10, Textual 8.2.8, 170×48 headless viewport.
Auto-merge remains **off**.

## Cause and correction

The previous packet's updated worst-input return included original main-thread
stacks through `_run_coalesced_control_bar_sync` → `_sync_console_control_bar` →
`run_console_presentation_sync` → native config operation → initialization wait.
This direct callback did not take FULL's awaited preparation route.

The precise branch was `_checked_display_status`'s genuine expiry case. A refresh
that was already pending refused display, but a direct call starting its own
refresh returned `None`, selecting native fallback. Nonblocking Python-lock
probes did not make the subsequent native admission nonblocking. The original
direct coalesced regression observed a real main-thread config entry/native open
and failed its zero-I/O assertion.

The qualified expired case now returns `False` in both pending states. The
existing projection `run()` already schedules finite checked preparation, and
the existing coalesced retry can render only after that preparation is fresh.
Refused worker submission remains cold/retryable. Unused pending-at-entry
bookkeeping was deleted. No new worker, cache, lock, TTL or native permission
decision is introduced. Custom/unqualified and unscoped callers retain their
original native route and callback affinity.

Eleven storage/config/ownership modules are unchanged. Within the modified
projection, seven preparation/owner/render helper functions and every check
before the final expiry branch are AST-identical to the baseline. This includes
qualification, owner/source, pricing, coordinator and proof-clock checks. Native
read retirement, cancellation and live action paths retain their original code.
See `control-bar-safety.json`.

## Verification

Real-deadline regressions first produced **3 failures / 1 pass**: the original
newly-expired case rendered through native fallback, the actual coalesced method
entered native config work, and failed worker submission fell back synchronously.
Already-pending refresh was already safe. All **4 passed** after the fix, including
fresh replay after checked preparation and retry after submission refusal.

The primary targeted native batch passed **103 checks in 175.88 seconds**:
checked displays, original native fallback/nested readers, owner/source/generation
and maintenance drift, config preparation/retirement, tokenizer/pricing and held
attachment input. The four focused checks overlap this count. Connected
contention, original pending-reader and direct pending-facts checks are recorded
separately below. No full suite ran.

The connected batch initially passed **10** contention/pending-reader cases and
failed four mounted cases. Three cost-source-retry tests timed out before their
target behavior: worker connections to the fixture's `:memory:` database lacked
the conversations table, trace capture refused, and the provider was not
contacted. Exact original-projection controls reproduce those three errors.
Their existing fixture shape is not changed by this work; they remain known
pre-target validation failures, not passing coverage.

The fourth candidate failure was a diagnostic recorder overflow after repeated
identical branches. Its old-code in-memory control was **invalid for overflow
attribution**: the source-fidelity observer refused a code/on-disk-source mismatch
before reaching that assertion. No source check was bypassed. The direct
observer now uses its sibling's existing scalar-observation coalescing, retaining
multiplicity and the original 64-distinct-row bound. The recorder itself is
source-pinned. A new RED/GREEN unit verifies counts, input-copy independence,
repeat-at-limit, overflow refusal and nonplain-value refusal. The original
mounted pending-facts test then passed with every UI, source, native lifetime and
retirement assertion intact. Unit plus mounted case passed in **63.13 seconds**;
the final recorder unit passed again after exact-type style cleanup. In total,
**115 distinct targeted cases passed**; the three baseline-reproduced fixture
failures remain. Focused/repeated runs are not added to that distinct count.

The two old contention tests deliberately expected expiry to enter native work.
They now select the still-supported unqualified copied-proof route, restore the
genuine proof inside that native body, and retain the exact assertions that busy
metadata refuses and native resources retire before retry publication. They do
not relax the original cleanup or coordinator checks.

Independent read-only review found no required product/test/harness changes.
New/changed test and diagnostic files pass Ruff checks/formatting. The projection
module has the same preexisting E721 exact-type finding and 71-line formatter
delta on baseline and candidate; no storage/type check was changed to silence
them. `control-bar-lint-comparison.json` retains both outputs. Diff whitespace
checks pass.

Baseline test loads are audited in parent and private child in
`control-bar-baseline-receipts.jsonl`. The fourth control's source-fidelity
refusal is retained separately from the three reproduced fixture errors and
from all plain profiling results.

## Native diagnostics and serial comparison

Diagnostic counters observe original code identities with
`sys.monitoring`, separately from plain timing. They count main-thread native
config entry and native-open calls inside the exact synchronous coalesced
callback. Successful scalar spans retain callback duration and stage; no frames,
locals or native handles are retained. The filtered global unwind event uses the
validated Python 3.12 API.

Plain ABBA restores **only the projection module** at the baseline in controls.
Home suppression, CSS matching, reminder and all other fixes stay active in both
arms. Six original Home/Console returns per fresh private profile retain 31.25 ms
clock-safe polling/heartbeat, actual draft input, attachment completion and all
existing storage/profile/network guards. Tests and diagnostics run separately.
Every plain process has an exact-exit and product-source receipt; adverse results are
retained without attributing unrelated latency variance to this component.

All four plain processes exited **0**, with all **24/24** drafts retained and
attachments reconciled and **zero** network/profile guard events. Results in ms:

| Run | Median input | Median attachment |
| --- | ---: | ---: |
| Original control 1 | 217.04 | 1015.51 |
| Updated 1 | 167.15 | 1369.77 |
| Updated 2 | 201.27 | 1093.83 |
| Original control 2 | 199.95 | 1131.17 |
| Original pooled (12 returns) | 200.73 | 1063.05 |
| Updated pooled (12 returns) | 181.36 | 1169.22 |

Pooled input median improved **9.65%**, while attachment median worsened
**9.99%**. Worst input worsened **351.06 → 447.44 ms**; worst attachment changed
**1778.59 → 1750.47 ms**. These small serial samples verify a mixed outcome, not
an overall Console or attachment speedup. Every journey is retained in
`control-bar-comparison.json`; no outliers are removed.

Both separate diagnostic processes exited **0**, retained all six drafts and
attachments each, and recorded zero guard events. Original/current return-stage
coalesced callbacks numbered **17/13**; both recorded **zero** native config
entries and **zero** native-open calls inside those callbacks. Their longest
callback spans were **7.08/9.06 ms**. Thus these natural runs did **not reproduce**
the intermittent native-entry path in the original arm, and cannot establish its
removal or a callback-duration improvement. The forced real-deadline direct
regression's RED/GREEN result establishes the corrected branch behavior.

The updated diagnostic also recorded **one main-thread config entry outside the
coalesced callback** on return 0. This narrow change does not eliminate every
main-thread native config route. Checked worker callback wall times still vary
substantially; instrumented timing is discovery evidence only. Detailed scalar
spans, status branches and the outside entry remain in
`control-bar-diagnostics.json`. Broader return-tail and attachment latency remain
unresolved; the mixed plain results do not justify changing storage safeguards.

## Reproduction

```powershell
.venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/run_native_wait_profiles.py --control-bar
.venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/summarize_native_waits.py --control-bar
```

Separate diagnostics add `--control-bar-observe --native-wait-observe` to the
same probe journey; original controls also add `--control-bar-baseline`.
Raw sampled stacks and complete instrumented payloads remain local; compact
scalar receipts and all plain journeys are committed with this report.

ADR required: no. ADR path: N/A. This narrows an existing disposable display
refusal within ADR-126/230 without changing storage/service/authority boundaries.
