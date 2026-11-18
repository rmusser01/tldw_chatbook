# Console transcript context reads — TASK-35679

Baseline: `d6dffdf7270bcc9e31f8b048e86581e576cd9ae7`, the preceding PR head.
Product commit: `9220460203` (complete SHA in all six final process receipts).
Native Windows 10, Python 3.12.10, Textual 8.2.8, 170×48 headless viewport.
PR3166 remains open with auto-merge **off**.

## What the slower results establish

The preceding window had a lower input median but worse attachment median and
worst input. Its main-thread samples group navigation through attachment, so
they cannot place every sampled stack before the posted key. They do not prove
that the preceding direct-control-bar correction caused those regressions.

One retained original stack does identify a remaining synchronous native path:
`_sync_native_console_chat_ui` → `_sync_native_console_transcript` →
`context_control_inputs` → `_global_context_policy_overrides` → checked config
operation → storage initialization wait. The transcript used only the returned
effective memory, yet entered the live global-policy reader outside FULL's
checked synchronous presentation scope.

A separate current-code discovery run added original-code scalar spans and
navigation/key timestamps without replacing product callables. It reproduced
that live transcript policy call at **567.69 ms** while leaving Console. Its
slowest input return spent **215 ms** in `Stylesheet.update_nodes` before the
key was posted; navigation-to-input was **324 ms**. Its longest attachment return
was **2202 ms**: FULL awaited configuration preparation until about **1811 ms**,
then workspace preparation occupied about **367 ms**. Queuing and concurrent
finite work remain part of those wall intervals; they are not pure CPU costs.

These observations distinguish styling before input, native preparation before
attachment, and another synchronous transcript call. They do not retrospectively
explain every difference in the previous small timing window. The next concrete
target is the unnecessary synchronous transcript context read.

## Correction and safety

Qualified stock transcript rendering now consumes the controls' already-wired
`ConsoleContextReadSnapshot.inputs`. Its original finite reader retains native
config, memory, lineage, source and retirement checks. No worker, cache, TTL or
native authority is added. Cold preparation hides only the disposable banner;
message painting continues. The published effective-memory value joins the
transcript gate, allowing the existing publication refresh to repaint otherwise
unchanged messages. The transcript never awaits preparation or holds streaming
chunks behind it.

Exact controller/callback and presentation-reader binding/code/default checks
retain the original post-gate synchronous route and cadence for custom readers.
Tuple unpacking remains inside the optional display error boundary. Independent
review caught a source-restoration edge: a custom async or live reader could
populate this existing memo, then restoring stock code within its TTL could
reuse the custom banner. The memo's existing key now also retains reader
binding, code and default-container identities. Existing pre/post-read key checks
discard changed-source work. Strong references prevent identity reuse; no
freshness deadline is extended.

Fourteen storage/config/controller/repository modules are unchanged. All thirty
other existing projection definitions, the snapshot's methods other than `_key`,
and 706 other ChatScreen methods are AST-identical to the preceding head. No
action/Send read, memory selector, physical handle retirement or admission gate
was altered. See `memory-safety.json`.

## Verification

The actual transcript regressions first failed **2/2**: original rendering read
unused live policy and did not request finite context publication. Both passed
after adoption. The initial publication test was corrected to observe both
existing preparation stages (config, then context), rather than assuming the
first completed worker meant both had retired.

New coverage exercises original native work held in a worker while transcript
messages continue painting, actual branch-validated legacy banner parity,
original custom-reader thread/cadence, async override compatibility, malformed
optional display data and publication through the unchanged-token gate. Review's
two real cached-custom→stock restoration cases first failed **2/2** and then
passed after source-key invalidation.

The final context/source/native ownership and gate batch passed **44 checks in
77.63 seconds**. An earlier integration batch passed **55 checks in 125.42
seconds**, overlapping seven new tests: FULL configuration preparation, original
direct-display/fallback/expiry/retirement and held-attachment keyboard controls.
No full suite ran. Additional mounted/owner/wiring results are recorded with
their setup failures in `memory-verification.json`.

Ten existing settings-context owner checks and one wiring check passed. The
mounted banner case initially refused a profile retarget before app construction;
an old-code control refused it too. Adding its neighbours' private-profile wrapper
first exposed the wrapper's required `request` fixture, then a separate pending
FULL/tick-gate setup issue. The test now forces the original transcript method
after each injected memory change (which does not bump the store token), under
the existing private-profile lifetime. All banner, row identity, selection and
plain-export assertions are unchanged. It passes on current and exact original
code (**35.60/34.27 seconds**).

In total **104 distinct targeted checks passed** across these batches; overlapping
runs are not added again. The final nine new cases passed again after the final
source-identity change. **Three connected tab-publication controls remain
unqualified**: all also fail on the exact preceding code, but at different
warmup/held-reader stages. Thus this comparison does not establish identical
failure causes or a passing tab-publication/retirement result. Their assertions,
deadlines and source checks were not weakened; they remain validation gaps.

Those controls execute original Git bytes from actual matching source files,
with compile filename, module file and spec origin aligned. Both pinned modules'
hashes were verified in **six parent/child processes (12 receipts)**. Earlier
in-memory controls are retained separately; they are not used as native
source-fidelity qualification. See `memory-source-baseline-receipts.jsonl`.

Independent review found no remaining required product changes after the cadence,
optional-value and restored-reader corrections. Changed test and harness files
pass Ruff. New test/harness formatting passes; the existing large transcript
fixture retains its baseline formatter differences. Product lint finding
multiplicities and formatter differences match their exact baseline per file
(`memory-lint-comparison.json`). Exact-type safety checks and unrelated formatting
debt are retained.


## 2026-10-31 final integrated tab followup

The three tab-publication gaps recorded above now have passing targeted functional controls on product head `0c39d774c13e0860a0afeb6ed174991dc1afd4cc`, based on actual PR3162 merge `a0dbc1174c3dbd4d2732b0212786ec9e08bfc3b5`: after_close 28.04s, coalesced 31.40s and same_session_drain 34.27s. The original reader/native proof, real Close click, keeper-only DOM and exact non-cancelled physical retirement assertions remain. Test-only event waits retain existing deadlines; the two deliberately held-Surface phases first finish the identified deferred startup publication within the same setup budget. All original failures and source patches are retained. See `../pr3166-final-integration-2026-10-31/RESULTS.md` for the exact diagnosis, receipts and limits. This does not change the historical timing results or attribute the old cadence 6-versus-5 failure. TASK35679 AC2 remains open pending fresh independent review.

## Serial profiling

Plain original/current runs restore only ChatScreen and
the spend-projection module in controls. All earlier fixes remain in both arms.
ABBA uses six Home/Console returns in each fresh private profile, real posted
draft input, attachment completion, 31.25 ms polling and heartbeat, exact process
exits, eight product source hashes and unchanged network/real-profile guards.
Diagnostics run separately and do not enter the plain timing comparison.

All four plain processes exited **0**, retained all **24/24** drafts and
attachments, and recorded **zero** network/profile guard events. Results in ms:

| Run | Median input | Median attachment |
| --- | ---: | ---: |
| Original 1 | 170.07 | 1137.72 |
| Updated 1 | 185.81 | 1170.11 |
| Updated 2 | 186.68 | 864.36 |
| Original 2 | 183.09 | 970.53 |
| Original pooled (12 returns) | 177.02 | 1023.92 |
| Updated pooled (12 returns) | 186.68 | 1038.45 |

Median input worsened **5.46%**, median attachment worsened **1.42%**, and worst
input worsened **203.79 → 247.49 ms**. Worst attachment improved
**1973.95 → 1590.37 ms (19.43%)**. The first attachment pair worsened; the reverse
pair improved. No consistent overall, median or input-tail improvement is
established. Every journey and adverse result is retained in
`memory-comparison.json`.

Both separate final diagnostic processes exited **0**, retained their twelve
drafts/attachments and recorded zero guards. The original trace recorded **one
main-thread live policy/context-control call, about 479 ms, during startup**;
the updated trace recorded **zero**. Neither arm called that path during its six
settled return stages. This supports removal of the cold/dirty transcript wait,
not a claim that it dominated these idle returns or explains their median
movement. The original direct regression separately observed the actual checked
config entry before the fix; current direct rendering observes none. Configuration
and workspace preparation and screen styling remain measured costs.

Final diagnostics retain original callback events, navigation/key timestamps and
scalar native preparation spans in `memory-diagnostics.json`. All six final
processes retain the same product commit and eight unchanged source hashes;
control packets additionally record both actually loaded original module hashes.

## Reproduction

```powershell
.venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/run_native_wait_profiles.py --memory
.venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/summarize_native_waits.py --memory
```

Separate diagnostics add `--return-observe --native-wait-observe` to the same
probe journey; exact controls also add `--memory-baseline`. Complete instrumented
payloads and sampled stacks stay local; scalar summaries and every plain journey
are retained. Failed diagnostic/test setup is distinguished from product behavior.

All 26 exact disposable test/profile roots and the two verified original source
files were removed after checking that no owned test/profile process remained.
No unrelated temporary files were removed. See `memory-cleanup.json`.

ADR required: no. ADR path: N/A. This adopts existing disposable presentation and
strengthens its reader-source key under ADR126/230 without changing storage,
permission, service or native ownership boundaries.


## 2026-10-31 integration qualification note

The checked config proof now preserves its original issue time across optional tokenizer preparation (48 focused checked-display/tokenizer tests passed). This does not qualify the three connected tab-publication controls above. On the b1ca1b11 integration base, a bounded original-reader trace returned a real native proof after 10.641s, beyond the unchanged ten-second setup predicate. The original control assertions remain intact; root deferred qualification until the final current-dev integration. TASK-35679 AC2 remains open. Exact source patches, adverse receipts, native-owner diagnostics and the unrelated historical cadence 6-versus-5 limit are in [the review-repair record](../pr3166-review-2026-10-31/RESULTS.md). Historical timings and speedup conclusions are unchanged.
