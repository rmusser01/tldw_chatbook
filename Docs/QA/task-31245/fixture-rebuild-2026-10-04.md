# Fresh fixture tooling checkpoint

Status: implementation and bounded automated verification; **not a full-scale,
native, Windows, participant or performance pass**. TASK-31245 remains In Progress.

## Source integration

The two local rename/ownership repair commits were replayed without conflict
onto dev `49206beea90d35ea9e6842ffa44e4b274db29d8a`. Range-diff reports both patches
equivalent. Upstream Console recovery/send changes are retained. The affected
rename, reuse, presentation and activation five-file check passed 85 tests in
317.79 seconds, no warnings, strict descriptor gate exit 0 and no retained
fixture database files. Raw log: `/tmp/switcher-dev4920-corrected-tests.log`.
An earlier command named a nonexistent test file and collected no tests; it is
not verification evidence (`/tmp/switcher-dev4920-tests.log`).

## Tooling and independent expectations

`Tests/Benchmarks/character_qualification_fixture.py` reserves only fresh explicit
OS-temporary destinations, checks clean exact source before production imports,
establishes disposable HOME/config/data/cache, strips inherited credentials and
Python path, selects null keyring, and installs the existing real-profile and
network guards. It exposes build, standalone Keyword, existing UI matrix,
small native preparation and manually invoked native launch commands.

The new fixture version is `task31245-rebuilt-v1`. Full scale is 10,000 chats,
250,000 selected user/assistant messages and four excluded messages. Thinking
and attachment canaries are sidecars on an eligible assistant message, not
extra visible messages. Message writes and indexing use production APIs; only
synthetic conversation dates are normalized through parameterized SQL with
production triggers intact. No raw message write or semantic-guard bypass.

The checked-in 30-query JSON is the independently declared historical oracle,
separated from its historical timings. Exact IDs, text, category, expectation
and target records compare equal to that oracle. New corpus bytes/digest and
timings must be measured fresh; no old digest is reproduced or timing relabelled.
Identity/content, explicit message timestamps and conversation ordering are
deterministic. Authority and generation UUIDs are production-generated, so byte
identity between builds is not promised; each receipt records its own digest.

Standalone retrieval uses a checkpointed immutable source backup. It retains
all measured samples and correctness failures, checks readiness and integrity,
drains the exact worker on cancellation before reporting retirement, and
records actual owned descriptors. Tiny receipts are smoke only. Full scale
requires explicit matching source head, five discarded warmups per query,
ten measurements per query and all 300 durations. Limits remain nearest-rank
P95 300 ms and maximum event-loop gap 50 ms. The existing compositor matrix
retains its 100 ms busy-paint and 50 ms loop-gap limits.

Source is rechecked at measurement/CLI completion. Any dirty source or HEAD
drift invalidates retained receipts without erasing raw samples. An unbound
tiny fixture cannot be promoted to full-scale evidence. Native preparation and
return are explicitly not qualified outcomes.

The separate native corpus contains four ordinary cards with seven chats each,
two chats whose card is unavailable, and an empty card. Unique transcript
markers support exact destination proof. Its manual launcher prepares two
saved Console tabs through installed resume APIs; actual startup/input/quit
has not yet been observed. No Terminal control is automated.

## Bounded verification and retained failures

Initial scaffold RED: 11 missing-feature failures. Receipt/launcher RED:
3 failures. Real held-worker cancellation RED proved premature terminal return.
Native dataset RED proved its missing implementation. Source-input refusal RED
proved an outside path was read before refusal. Independent review found missing
head acceptance, missing completion source fence and nondeterministic message
timestamps; four focused REDs and eight real-Git receipt-fence REDs are retained.

After the review fixes, the first combined check had 39 passes and two assertion
failures: the driver converts TIMESTAMP values to datetime, while the assertions
expected strings. Assertions now inspect stored text using CAST; their independent
expected dates were not changed. Final covering check: **41 passed in 8.70s**,
no warnings, strict gate exit 0, zero retained database files after each teardown.
Command used the new fixture module plus
`Tests/Benchmarks/test_console_character_switcher_latency_measurement.py` and
the opt-in read-only descriptor census. Logs: `/tmp/switcher-fixture-*-red.log`,
`/tmp/switcher-fixture-reviewed-green.log` (failed assertions) and
`/tmp/switcher-fixture-reviewed-corrected.log` (final pass). No full sweep.

## Source-bound commands after review and commit

Run from this worktree only, after recording a clean exact commit. Every output
directory below must be absent; commands refuse reuse. Run measurements alone,
not alongside tests, other benchmarks or native activity.

```sh
qualification_python=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python
qualification_head=$(git rev-parse HEAD)
qualification_container=$(mktemp -d /tmp/task31245-qualification-XXXXXX)

"$qualification_python" -m Tests.Benchmarks.character_qualification_fixture build \
  --root "$qualification_container/scale" --expected-head "$qualification_head" --size scale

"$qualification_python" -m Tests.Benchmarks.character_qualification_fixture keyword \
  --root "$qualification_container/keyword" --expected-head "$qualification_head" \
  --corpus "$qualification_container/scale/corpus.sqlite" \
  --source-receipt "$qualification_container/scale/build-receipt.json"

"$qualification_python" -m Tests.Benchmarks.character_qualification_fixture ui \
  --root "$qualification_container/ui" --expected-head "$qualification_head" \
  --corpus "$qualification_container/scale/corpus.sqlite" \
  --source-receipt "$qualification_container/keyword/keyword-receipt.json"

"$qualification_python" -m Tests.Benchmarks.character_qualification_fixture prepare-native \
  --root "$qualification_container/native-source" --expected-head "$qualification_head"

# Operator only: manually open a dedicated Terminal window, cd to this worktree,
# and run the source-bound native command. Do not operate existing windows.
"$qualification_python" -m Tests.Benchmarks.character_qualification_fixture native \
  --root "$qualification_container/native-run" --expected-head "$qualification_head" \
  --corpus "$qualification_container/native-source/native.sqlite" \
  --source-receipt "$qualification_container/native-source/native-receipt.json"
```

First run an equivalent tiny build/Keyword smoke command to verify the launcher
in a clean source-bound subprocess. Keep all failed receipts. Native launch is
not permission to infer input, normal quit or resource success: follow
`native-qualification-checklist.md` and preserve case results independently.

ADR required: no new ADR. Existing ADR-120 eligibility/privacy/navigation and
ADR-198 GC-policy constraints apply. No production policy, dependency, schema,
embedding model or authority boundary was introduced. Frozen baseline and
observer-cost comparison precede any later GC remedy. Actual Windows Terminal
and three first-time participants remain external; TASK-31246's dependency is
not waived and no final PR has been created yet.

## First clean-process scale attempt and fixture correction

Frozen source `f7b4227e36f0d317d43c988cea13c7e5a2788757` completed the tiny
CLI smoke and full production-API corpus build. Full counts were
10,000/250,004/10,000, integrity OK and index ready. Standalone Keyword passed
all 300 measured identities, P95 108.303041 ms and maximum loop interval
17.039625 ms, no retained database descriptors or registered handles. Source
digest was `a8da4228b4e4526b578687af49a33e82bbd4c805f39e3a3dbbf29029fca05921`.
All raw receipts are retained under `/tmp/task31245-freeze-5uPHpo`.

The first UI attempt failed before collecting samples: the normal provider-setup
overlay refused the Console switcher action. A separate read-only action
observer reproduced `setup_blocking=true`, `decision_blocking=false`,
`first_send_completed=false` and ChatScreen both before and after the action.
This is a fixture precondition failure, not evidence of search/paint latency.
The global setup-wizard flag does not represent Console's separate first-send
state. A focused regression failed on this distinction before the correction.

The checked-in synthetic config now explicitly represents an existing-chat user
through `[console.onboarding] first_send_completed=true`. This does not assert
provider readiness, enable sending, inject credentials or bypass a production
control. Navigation-discovery participants are not application/provider
onboarding qualification. The provider remains unavailable and network guard
remains enforced. New source-bound receipts must be measured after re-freezing;
the earlier passed Keyword receipt is not relabelled for the new head.

The fixture/measurement regression pair passed 42 tests in 8.72s, no pytest
warnings, strict descriptor gate exit 0 and zero retained database files after
each teardown (`/tmp/task31245-config-green.log`). RED receipt:
`/tmp/task31245-config-red.log`. Changed Python lint and format are clean.

The failed UI receipt retains its post-run-test-unmount database descriptors and
three registered handles; this is not terminal resource proof. One optional
pydub warning appears in its private application log. No warning suppression,
manual sweeping of database owners or increased timing limit was applied.

## Corrected-head scale baseline and setup-frame race

Frozen `c25ea51e57a404a7628ad60f3620243c8b4b6f9f` rebuilt the corpus through
production APIs and passed fresh standalone Keyword qualification: 300 measured
identities correct, P95 131.421958 ms, maximum loop interval 6.3825 ms,
registered handles 0, owned database descriptors empty and ending source clean.
Receipt: `/tmp/task31245-freeze-5uPHpo/keyword-c25/keyword-receipt.json`.

The full real-owner UI matrix now ran both sizes, all 60 search cases and eight
exact OPENED activations. Both sizes satisfied their expected page geometry
(50 fetched, four visible at 52x20 and 11 at 120x50). Maximum busy paint was
34.928375 ms. All eight activation loop gaps failed the unchanged 50 ms gate:
70.528583, 69.089958, 77.304875, 88.939959, 75.369125, 102.358042,
134.493208 and 90.637958 ms. This is a failed UI qualification, not a pass.
Source/corpus remained unchanged. Post-run-test-unmount registered handles
were 18; no normal native quit or final retirement was observed.
Raw receipt: `/tmp/task31245-freeze-5uPHpo/ui-c25/ui-evidence/ui-latency-evidence.json`.

Read-only diagnostics remain separate from acceptance evidence. The first
all-generation GC observer stopped at a blank-frame assertion before activation.
A narrower run reached all cases but exhausted its 500-event main-thread
generation-1/2 buffer before the last three wide activations (246 dropped).
Do not interpret their empty event lists as no-GC evidence. A stack-sampled
run retained all 62 generation-2 events and 8,314 stack samples (no drops),
showing rendering/layout, guarded config/storage reads and GC during some long
gaps. Its sampling-created objects and up-to-65 ms wall observation interval
can perturb the heap/schedule; its 14–80 ms GC spans do not prove uninstrumented
production cost. A later low-allocation GC-only comparison again failed before
activation. No forced collection, GC threshold, cache policy or production
performance fix was applied. Diagnostic roots are `ui-gc-diagnostic-c25`,
`ui-gc-bounded-c25`, `ui-activation-diagnostic-c25`, `ui-gc-only-c25` and
`ui-gc-frame-c25` beneath the retained container above.

The last diagnostic captured the exact failing blank compositor frame: pending
true, no mounted empty-state node, and `Loading local chats…`. The harness had
observed earlier settled flags, then yielded in `pilot.pause()` while a live
activity projection reconciliation restarted loading. It was asserting the
stale startup observation, not a failed completed query. The probe now waits
for the actual ready blank compositor frame after that pause. This is untimed
setup only; measured windows, production loaders/activation, timing limits and
all failed receipts are unchanged. Five focused REDs precede the correction;
the fixture/probe pair passed 47 tests in 17.94s with no pytest warnings and
strict zero retained database files (`/tmp/task31245-blank-green.log`).
Changed Python lint and formatting pass. This routine harness correction needs
no new ADR; ADR-120 and ADR-198 remain controlling. Repeat the frozen baseline
and complete low-allocation observer comparison before selecting a performance
remedy. Native, Windows, participant, resource and semantic dependency gates
remain open.

## Frozen 8c source: real CLI walkthrough and full-scale results

Source `8c2c6b16a8d9e40c26e026f7cdacc7439a689671` remained clean throughout
these runs. The 47 fixture/measurement tests passed again in 10.86s, no pytest
warnings, strict exit 0 and zero retained fixture database files at each
teardown (`/tmp/switcher-current-8c-fixture-tests.log`). This does not extend the
fixture resource result to an application's whole lifetime.

The actual installed app ran in a new CLI PTY using only the private native
fixture. No Terminal.app/iTerm GUI or alternate native-control driver was used.
Observed actual input and output:

- Ctrl+K distinguished CURRENT from the MRU other tab; Enter switched from
  Indigo 07 to Amber 07 and displayed its unique transcript marker.
- The Character mode button was keyboard reachable. Keyword body search for
  `NATIVE_MARKER_AMBER_01` offered the exact Amber 01 conversation. Cold Enter
  opened it and increased Active from three to four. Repeating the search
  showed CURRENT TAB; warm Enter resumed it without increasing Active again.
- F2 from the query refused truthfully. Focusing the open Amber 07 result then
  F2 opened the rename dialog. The accepted title `QA Amber renamed PTY`
  appeared in the native tab and switcher. An independent read-only query of
  the private mutable SQLite file confirmed that exact conversation's durable
  title. The immutable source corpus was unchanged.
- Escape followed by Ctrl+Q exited normally, process return code 0; no kill or
  controller interrupt was used.

Limits and findings: this is CLI-functional evidence, **not macOS native GUI,
font, Windows Terminal, participant or release qualification**. The injected
legacy Shift+F3 sequence `ESC [ 1 ; 2 R` did not cycle modes. Installed Textual's
parser recognizes this ambiguous sequence as a cursor-position report before
key dispatch, so simply adding an F15 binding would not repair this path.
Mode buttons worked; no global parser or keybinding policy was changed.
A combined Tab/text input chunk initially did not populate the query; standalone
text delivery did. The receipt retains this input timing limitation.

At `App.run()` return, while its app owner was retained, the receipt still
listed 35 owned SQLite/main/WAL/SHM descriptors. That is not zero-resource
retirement, nor alone proof of a cumulative leak. Process exit is not a
substitute for proving owner retirement before interpreter exit. No manual
sweeping of app-owned databases was applied. Raw startup/return and observations:
`/tmp/task31245-freeze-5uPHpo/pty-functional-8c/{native-startup.json,native-return.json,functional-observations.json}`.

Fresh full-scale Keyword qualification passed on the same frozen head:
10,000 conversations, 250,000 eligible messages, all 300 measured identities
correct, warm P95 **112.412334 ms**, maximum loop gap **15.228834 ms**, index
ready, zero registered handles and an empty owned-database descriptor list
after cleanup. Source digest:
`e574781b032e9dab13c5298cbae40c766029ffb5bd2ddebf8a092748910d28d5`.
Receipt: `/tmp/task31245-freeze-5uPHpo/keyword-8c/keyword-receipt.json`.

The corrected UI command consumes that **passed Keyword receipt**, not the
build receipt. An initial wrong-input invocation failed its `status == passed`
assertion before collecting samples (`ui-baseline-8c`); it is not UI evidence.
The fresh `ui-baseline-keyword-8c` matrix completed all 60 searches and eight
exact OPENED activations at 52x20 and 120x50, with unchanged source/corpus and
no app exception. The expected 50-fetched/four-visible and 50-fetched/11-visible
geometries passed. Search maximum loop gap was 49.454708 ms; maximum busy paint
was 66.257875 ms. Activation loop gaps were **118.825917, 49.644667, 51.326750,
87.713709, 53.206459, 85.814792, 85.979916, 77.967834 ms**: seven failures
against the unchanged 50 ms limit. This is a failed UI qualification despite
correct identities. Post-run-test-unmount registered handles were 16; that
boundary is not terminal app resource qualification. Raw receipt:
`/tmp/task31245-freeze-5uPHpo/ui-baseline-keyword-8c/ui-evidence/ui-latency-evidence.json`.

### Bounded stall diagnostics, not acceptance or a production remedy

The complete main-thread generation-2 timing comparison retained 60 events,
zero drops or unmatched pairs, with maximum direct callback cost 0.008708 ms.
It observed 57–66 ms collections during two wide activation windows, but also
a 70.454958 ms activation gap with **no** main-thread generation-2 event.
Collections contribute; they do not explain every failed gap. The normal paint
observer in this diagnostic had maximum per-activation durations 0.52–1.29 ms.
The baseline acceptance probe, without added GC/span instrumentation, separately
had a 34.99 ms maximum paint observer duration, so neither run is a universal
observer-cost bound.

A first-activation cProfile diagnostic substantially perturbed timing and
retained implausible interleaved caller attribution (including stylesheet calls
attributed to database methods). Its caller graph is not reliable allocation
ownership evidence. A direct main-thread synchronous-span observer was used
instead. Its first 5 ms recording floor exhausted the 200-span cap (133 dropped)
before wide activations; their empty span lists are **not** no-work evidence.
The app-first-import-order 20 ms-floor rerun retained all 87 spans and 54 GC
events, zero drops. Across all eight activations it recorded five to nine
synchronous Console control refresh spans per window, mostly 20–48 ms, with
some 50–74 ms spans. These include nested work and automatic GC, not exclusive
function cost. Readiness, exact activation and owner guards remain installed.

Retained diagnostic roots under `/tmp/task31245-freeze-5uPHpo`:
`ui-gc-diagnostic-8c`, `ui-activation-profile-8c`, `ui-sync-spans-8c` and
`ui-sync-spans-v2-8c`. Temporary diagnostic scripts record their own digests;
they introduce no source patch, forced collection, threshold increase, GC
disable, renderer replacement or fake activation result. Existing desktop
background load remained; these were serial runs, not alongside my own tests
or CLI walkthrough. Attribution and a reviewed bounded correction remain
necessary before claiming the latency/resource gates passed. TASK-31246's
qualification dependency and the final combined PR remain pending.

### Stable control-refresh attribution on the small fixture

Documentation-only successor `5a43164923` preserves the production, test,
script and package inputs byte-for-byte from the measured 8c head. A newly
prepared, exact-head small synthetic corpus supported five direct calls to the
installed control refresh after real saved-chat resume. This is a diagnostic,
not a full-scale latency, native input or resource qualification. Original
readiness/config/ownership checks remain in place and sources stayed unchanged.

The first observer mistakenly rebound a static method as an instance method
and failed during composition, before the intended observations. That failure
is retained at `control-steps-5a`; it is not a production regression. Corrected
observers preserve the original descriptor and filter to the main thread.
The five steady refreshes in `control-steps-v2-5a` took 19.019583, 20.810458,
22.442000, 21.773208 and 20.064375 ms, with zero dropped records. Composer
reason/geometry work was small rather than the main cost; no speculative
composer-wide optimization was implemented.

Adding the remaining speech/library/inspector projection substeps in
`control-steps-v3-5a` retained all observations and no app exception. Its five
refreshes took 35.908834, 40.469959, 23.480750, 28.880583 and 29.450875 ms.
Of 158.128 ms summed control-refresh time, 94.219 ms was inside
`ConsoleControlBar._set_recovery_height` (90.677 ms main-thread CPU). These
instrumented inclusive durations are not a before/after production comparison
or a universal timing bound. The existing method unconditionally removes
its `h-*` class and re-adds the same `h-1` on every unchanged disabled-speech
refresh; Textual synchronously restyles that control-bar subtree for both
mutations. This gives a concrete bounded correction to test, without a global
GC/cache policy or dropping the authoritative speech/config refresh.

The proposed correction replaces only the owned height-class set atomically,
preserves unrelated classes and the inline-height reset, and retains the
existing one-/two-row recovery transitions. The user approved proceeding, and
the bounded correction below was implemented; this does not itself establish
the remaining 50 ms activation limit. TASK-31966 is In Progress, with criteria open.
Diagnostic receipts/scripts remain in the retained temporary container above;
each successful receipt records its full source head, source digest, raw calls,
windows and script digest. ADR-120/150/161/198 remain controlling.

### Bounded recovery-bar correction

The mounted RED had three expected failures and three passes: five unchanged
speech refreshes caused ten real node restyles for both hidden and visible
recovery, and each transition first removed its height class before adding the
replacement. Atomic replacement of the owned `h-*` set now lets Textual skip an
unchanged class set. Unrelated classes, the inline-height reset, min/max bounds,
recovery visibility, retry/resume behavior and replacement widget setup remain
intact. No global GC/cache policy or activation/readiness guard changed.

Initial GREEN: six mounted tests pass in 2.61s, no pytest warnings. First setup
attempts exposed the documented source-bound config admission trap; the new
widget-only suite uses the existing bootstrap-profile marker without profile
mutations. Logs: `/tmp/task31966-recovery-red-bound.log` and
`/tmp/task31966-recovery-green.log`. Both paths format clean; the new tests are
Ruff clean. The production file retains the same three inherited Ruff findings
as HEAD (FLY002 and two B010), confirmed on the committed source.

The covering speech/coalescing/design-token run had 117 passes and four config
admission failures in 54.14s. All four reproduce with this production fix
removed (4.65s): both coalescing mounts and the two full-app speech suspend
cases fail with `raw_source_selection_changed`. This is not a clean covering
suite or a recovery regression. No guard was bypassed to turn them green.
The strict read-only census retained no database files after any teardown;
admission/lease descriptors remain, so this is not an all-descriptor-zero claim.
Logs: `/tmp/task31966-recovery-cover.log` and
`/tmp/task31966-inherited-baseline.log`. After restoring the correction and
formatting, all six mounted regressions pass again in 3.37s with strict
database retirement required and no pytest warnings. All eleven artifact guards
pass, including the newly gated regression file. Logs:
`/tmp/task31966-recovery-final.log` and `/tmp/task31966-recovery-preflight.log`.
Fresh frozen performance remains to run.

### Frozen post-correction evidence — `0ab325187e`

All following runs were serial at clean source
`0ab325187e5b5ca838cd795d2545b738e0a8415c`. No forced collection, GC threshold
change, renderer/activation replacement, real-profile access or concurrent test
run was used. Ordinary desktop background load remained; this is not an idle
machine claim. Every finished receipt confirms exact clean source and unchanged
input corpus.

The identical small-fixture diagnostic script (`fd2983eb86ffd69909ebc2b85bb1919401e86a9e411b802aecfbfdcfe753ebeb`)
retained all observations, with no app exception. Five steady control refreshes
took 20.086125, 11.763209, 12.282041, 12.643750 and 12.616000 ms. Recovery-height
spans took 0.095375, 0.104125, 0.074250, 0.052791 and 0.048250 ms, versus
94.219 ms summed across five baseline calls. This proves the measured local
improvement, not a universal speed bound or complete activation qualification.
Root: `/tmp/task31245-freeze-5uPHpo/control-steps-v3-0ab`.

The fresh full corpus has 10,000 conversations, 250,000 eligible messages and
four excluded canaries; integrity is OK and index status ready. Digest:
`e08867648fca24cd9aa1280a620410948b465f32112fcd2d2ad7c390786614c9`.
The 300-query Keyword run passed with exact results, warm P95 **240.415500 ms**
against 300 ms, maximum scheduling interval **25.307375 ms** against 50 ms,
zero registered handles and no owned database descriptors after cleanup.
Roots: `scale-0ab` and `keyword-0ab` in the retained container.

The ordinary full-owner matrix still **failed**. All 60 searches and eight exact
`opened` activations completed without an app exception. The 52×20 and 120×50
geometry cases retained 50 fetched results with four and eleven visible rows.
Maximum search interval was 48.518959 ms and busy paint 37.289709 ms. Activation
intervals were, in order: **88.615792, 51.870000, 69.553792, 57.377209 ms** at
52×20 and **68.284458, 78.751333, 93.816959, 86.245042 ms** at 120×50. All eight
exceed the unchanged 50 ms limit; narrow preparation also failed at 52.510417 ms.
Paint-observer maximum was 44.539375 ms, so observation overhead cannot be
ignored or confused with exclusive production cost. The app remained retained
after `run_test` unmount with 15 registered handles: this is not terminal
application-owner retirement evidence. Root: `ui-0ab`, raw receipt
`ui-evidence/ui-latency-evidence.json`.

A single current-head GC diagnostic also failed the matrix. It retained 60
main-thread generation-2 callback events (30 matched pairs), zero drops or
unmatched events; callback maximum was 0.011125 ms. Wide activation windows
contained collections of roughly 62–88 ms wall time (61–86 ms CPU); none
reported uncollectable objects. But the first narrow scheduling interval was
450.808333 ms while its two contained collections were only 14.737 and
29.745 ms. GC contributes; these observations do not explain every stall or
identify collected-object owners. Do not relabel diagnostic timings as pure
production latency. Root: `ui-gc-diagnostic-0ab`, with both the ordinary matrix
receipt and `gc-timing-diagnostic.json` retained. No policy change was made.

Logs: `/tmp/task31966-control-steps-0ab.log`, `/tmp/task31966-scale-0ab.log`,
`/tmp/task31966-keyword-0ab.log`, `/tmp/task31966-ui-0ab.log` and
`/tmp/task31966-ui-gc-0ab.log`. Raw temporary artifacts must be preserved before
cleanup; paths alone are not portable evidence. TASK-31966 remains In Progress;
native macOS/Windows, unfamiliar-participant, remaining stall attribution and
terminal application-owner qualification remain unwaived. TASK-31246 remains
dependent on TASK-31245 qualification; no final combined PR was created.

### Untimed heap and stable-restyle attribution — `ed124369f1`

This documentation-only successor has byte-identical production, test, script
and package inputs to `0ab325187e`. A fresh small native-source corpus was
prepared at the exact clean head; it is **prepared, not native-qualified**.
Digest: `28d2600c9d91877d5057ff6028f2d0f038d394063481ff0eee0efe7dc5f7e39f`.
Both following observers use real offline Console owners and retain their
throwaway scripts, source digests and limitations. Neither is a latency pass.

The untimed heap census completed two saved-chat resumes, three cold Character
activations and one warm reuse through actual Enter dispatch and exact-ready
proof, without an app exception or source mutation. Before saved-chat resume,
it found 131,104 unfrozen tracked objects, including 3,439 Strips and 24,173 FIFO
caches (23,359 empty). After two resumes: 213,012 objects, 6,617 Strips and
46,473 FIFO caches (45,201 empty). Partial traversal from widget render/style
caches reached 6,389 of those 6,617 unfrozen Strips, principally in mounted
Console widgets. This supports render-cache attribution, not a leak diagnosis.

After four Character activations, the unfrozen count was 75,500 and the frozen
count 655,991, versus 478,105 frozen at the preceding snapshot. Existing boot
pre-import freezes can run after UI ready; these censuses are not equivalent
steady-state heap samples and do not prove reclaimed memory. The census itself
allocates and holds tracked objects, may include unreachable cyclic garbage,
and deliberately runs outside timed windows. Its partial cache traversal does
not classify all compositor/global roots or already collected objects. After
unmount the app remains retained: no terminal-owner retirement claim follows.
Receipt: `heap-owners-ed124/heap-owner-diagnostic.json` in the retained container;
SHA256 `8d16fff13755c257899fa05e427b67132e28303afdf05e7821ebfe07bcdee1ae`.

A separate bounded observer around real `DOMNode.update_node_styles` found
**40 Send-reason, ten voice-status and ten attachment-indicator restyles during
five unchanged control refreshes**. Send-reason restyles took 17.116916 ms summed
(inclusive, observer-perturbed). The recovery bar no longer restyled. No records
dropped, no app exception, and the source corpus stayed unchanged. Complete
refresh durations were 29.549916, 31.471792, 25.107958, 17.340750 and 17.875959 ms;
these small diagnostic windows do not substitute for the failing scale matrix.
The first observer attempt used the wrong framework class for the method and
failed before app construction; it is a diagnostic setup failure, not a
production regression. Corrected receipt:
`control-restyles-ed124-attempt2/control-steps-diagnostic.json`, SHA256
`f18fc73189df2d756f699eaeb5098c86acf9aa8153c1badd7f3334a6ce264955`;
script `diagnose_control_restyles_ed124.py`, SHA256
`5c59222c9aefa33479b82cdc3195db2ca1a9c8ff645205526d14f1d24dfe78db`.

All four production callers of `_sync_send_disabled_reason` reach its same
remove/re-add width/height logic: action refresh, resize and two voice repaint
paths. The proposed next bounded correction is atomic replacement of that
owner's actual size classes, preserving unrelated classes, inline reset, live
width budget, advisory copy/link safety and voice-preparation behavior. Mounted
RED/GREEN should cover unchanged visible/hidden/empty states, transitions,
resize, conflicting inline/classes and existing voice/disabled-state contracts.
**Await design approval before implementation**, per the brainstorming skill.
No global GC/cache change, framework patch or new dependency is proposed;
existing ADR-120/150/161/198 apply. The voice and attachment observations remain
separate candidates, not silently bundled fixes. All qualification gaps remain
open. Logs: `/tmp/task31966-heap-source.log`, `/tmp/task31966-heap-owners.log`,
`/tmp/task31966-control-restyles.log` and
`/tmp/task31966-control-restyles-attempt2.log`.

### Approved bounded Send-reason correction

After user approval, the existing shared reason owner now replaces its actual
`w-*`/`h-*` set atomically with Textual's native `set_classes`. Unrelated
classes, inline-size resets, measured width budget, escaped setup copy/link,
empty/narrow suppression and full-width voice preparation remain intact. All
four callers still share this owner; no GC/cache policy, dependency, activation
authority or voice/attachment correction was bundled. Existing ADR120/150/161/198
apply; no new ADR is required for this behavior-preserving rendering fix.

Mounted RED: four failures and two passes; five unchanged calls caused twenty
real node restyles and a transition caused four rather than one. Initial GREEN:
six passes. A read-only independent review found no production blocker; its
canonical composer-ID suggestion was applied to the harness. After removing
only this production hunk, the final harness again produced the four expected
RED failures. The added real full-width voice transition retains cached reason
copy while suppressing its layout, then restores its bounded one-row guidance.

Final focused check: **25 passed in 164.77s**, no pytest warnings, strict
descriptor gate exit 0 and no retained database files at any of its 25 teardown
observations in the parent pytest process. This did not census its private test
children; the Model-row check below covers that additional boundary.
Admission/lease descriptors remain; this is not whole-app final
resource retirement. Command covered the new mounted module, the existing
private-profile Send-disabled contracts and design-token governance. The new
test/checker paths format clean and the new test is Ruff clean. The production
file retains the same seven inherited Ruff findings; its remaining format
suggestions are a subset of the baseline's, with none added. Whitespace passes.
The new regression module is included in the UI PR census and its floor raised
by one. Logs: `/tmp/task31966-send-reason-red.log`,
`/tmp/task31966-send-reason-green.log` and
`/tmp/task31966-send-reason-final.log`.
All eleven derived-artifact guards pass after the final test addition
(`/tmp/task31966-send-reason-final-preflight.log`); UI census is 142 files,
floor 140. This is not a full test sweep.

The broader affected run was interrupted after 101 failures, 42 passes and one
thread warning; it is **not a passed covering suite**. Representative existing
reason-width/voice tests fail before mounting because source-bound config sees
`raw_source_selection_changed`. The retry-thread warning has the same admission
cause, not a ResourceWarning. A bounded comparison with the production fix
removed reproduces both config failures, the retry failure/warning and the
dimension-governance failure in unchanged agentic-terminal, splash-theme and
workflows stylesheets (counts 1/2/11). That baseline run had eight failures,
two passes and one warning, including the four intended RED cases, in 13.23s.
This proves those representative failures are inherited, not that every
interrupted case was individually qualified. No warning suppression, source
guard bypass or unrelated stylesheet repair was applied. The separate combined
governance attempt was interrupted at this inherited dimension failure after
19 passes. Logs: `/tmp/task31966-send-reason-cover.log`,
`/tmp/task31966-send-reason-reviewed.log` and
`/tmp/task31966-send-reason-baseline.log`. Initial default-basetemp RED also
reported unrelated old pytest garbage-directory cleanup warnings; later runs
use fresh explicit private basetemp roots without deleting those directories.

Fresh clean-head restyle and full-scale measurements remain to run. No
qualification pass, native/Windows/participant waiver, semantic work or final
combined PR follows from this local regression result. TASK31966 remains
In Progress with all criteria open.

### Frozen Send-reason correction measurements — `ca793ee190`

All runs below were serial at clean exact source
`ca793ee1900d3d24decea8f363ee8ac3b7896827`. Every finished receipt confirms
unchanged corpus and clean source at completion. No production edits, concurrent
tests/benchmarks, real-profile access, forced collection or GC/cache change was
used; ordinary desktop background load remained. These timings are not an
idle-machine or universal before/after speed claim. Raw roots are retained under
`/tmp/task31966-send-measure-v0Z4Cs`; do not delete them during cleanup.

The fresh small corpus digest is
`504a0dc3e70bde4a8b26ef166b20c0cfbaa5dfd851058bbdb66d0e6ce628711d`.
The **identical** observer script
(`5c59222c9aefa33479b82cdc3195db2ca1a9c8ff645205526d14f1d24dfe78db`)
completed real saved-chat resumes and five unchanged control refreshes with
zero dropped records or app exceptions. Send-reason restyles fell from **40 to
zero**; recovery remains zero. Voice and attachment each still restyled ten
times (inclusive sums 5.028123 and 5.687376 ms); they were deliberately not
bundled. Refresh durations were 16.730667, 17.208458, 30.525208, 13.646791 and
17.017416 ms. The structural restyle elimination is proven; observer-perturbed
timings do not prove an overall activation improvement. Root `control-restyles`,
receipt `control-steps-diagnostic.json`. This is headless real-owner diagnostics,
not native GUI input or resource qualification. `native-source` is prepared only.

The new full corpus has 10,000 chats, 250,000 eligible selected-branch messages
and four excluded canaries; integrity OK, index ready, registered handles zero
after builder cleanup. Seed took 408.926881s and indexing 8.285102s; the prior
seed took 196.720125s, another reason not to infer a controlled timing comparison.
Digest: `d91a0d1b1a08d5a7cc2a9a8256a8a5e5367188ed30002f4a06db4d36fbb8ecf8`.
Fresh standalone Keyword **passes**: all 300 measured identities correct,
P95 **172.685959 ms** against 300 ms, maximum loop gap **17.772709 ms** against
50 ms, zero registered handles and no owned database descriptors after cleanup.
Roots `scale` and `keyword`, receipts `build-receipt.json` and
`keyword-receipt.json`. Query IDs named `semantic-*` are historical Keyword
oracle categories, not enabled local-semantic-search qualification.

The full real-owner UI matrix still **fails** despite completing all 60 exact
search cases and eight exact `opened` activations without an app exception.
Both layouts retain 50 fetched results, four visible rows at 52×20 and eleven
at 120×50. Narrow activation loop gaps were **74.434958, 73.399792, 89.488875,
85.862250 ms**; wide were **152.296875, 184.211125, 187.347208, 142.758333 ms**.
All eight exceed the unchanged 50 ms limit. Preparation also fails at 71.162584
and 63.044042 ms; wide historical `semantic-1` Keyword search fails at 77.674 ms.
One wide activation busy paint fails at **127.125750 ms** against 100 ms.
Maximum paint-observer duration was **75.940500 ms**: retain observer overhead
as a limitation, not exclusive production cost or an excuse to waive failure.
The retained post-unmount app has 18 registered handles; this is not final
application-owner retirement proof. Root `ui`, receipt
`ui-evidence/ui-latency-evidence.json`. Its private qualification log reports
optional python-frontmatter unavailable; no warnings were suppressed.

Logs: `/tmp/task31966-send-native-source.log`,
`/tmp/task31966-send-restyle-measure.log`, `/tmp/task31966-send-scale.log`,
`/tmp/task31966-send-keyword.log` and `/tmp/task31966-send-ui.log`.
Raw failed timings are preserved, not relabelled or selectively rerun.
Remaining work is attribution/review of the residual stalls and the separate
native/Windows/participant/terminal retirement gaps. TASK31966 and TASK31245
remain In Progress; dependent TASK31246 and the final combined PR remain pending.
No further voice/attachment fix or global policy remedy is authorized by this
bounded correction's approval.

### Same-source residual-stall attribution after the Send-reason fix

The documentation-only `cb95979312` successor preserves all production, test,
script and package inputs from the measured `ca793ee190` head. Rather than
rebuild identical scale data or restamp old receipts, three existing throwaway
observers ran serially at the exact clean measured commit against its original
verified corpus/Keyword receipt. Only this isolated worktree was temporarily
detached; its original branch and `cb95979312` head were restored after each run,
including failure exits. No source edits, real-profile access, forced GC,
threshold change, dependency change or guard bypass occurred. Each matrix
retained clean-source/corpus completion proof, all 60 exact searches and eight
exact opened activations, no app exception. These are failed diagnostics,
not replacement acceptance receipts or native/resource passes.

The unmodified low-allocation GC observer retained **30 matched generation-2
collections**, no dropped/unmatched events, default thresholds 700/10/10 and
maximum callback overhead **0.021166 ms**. No event reported uncollectable
objects. Wide activations 2/3/4 contained collections lasting roughly **66–79
ms**, supporting GC contribution. But wide activation 1 had a **58.262666 ms**
loop gap and **no generation-2 collection** in its window. Its paint observer
maximum was 1.298125 ms. Narrow activation 1 had a 70.906333 ms gap with two
collections of 12.560708 and 22.264833 ms. Window overlap is not exact attribution
of the longest individual scheduler interval; GC does not explain every stall.
One narrow activation interval was 47.215541 ms in this diagnostic; the run
still fails, and that single sample does not supersede the ordinary matrix.
Root `ui-gc-diagnostic`, script digest
`24b49690b2e10619f85b4175c77efa96317beb9a5c79836c95fa13823cc442df`.

The existing bounded synchronous observer retained **66 spans >=20 ms**, with
zero dropped span or GC records. Within activation windows, the shared
`ChatScreen._run_console_config_sync` boundary retained **20–60 ms** spans
without generation-2 overlap. This boundary includes guarded entry, rendering
and guarded exit; it does **not** isolate config IO from the rendered body.
Selected reflows lasted 80.145750, 125.053166 and 105.855250 ms, with measured
generation-2 overlap of 58.913625, 104.265250 and 99.471875 ms respectively.
Activation paint-observer maxima were 0.930458–3.244083 ms. Thus neither the
paint observer alone nor generation-2 alone accounts for the residual problem.
Wrappers still alter allocation/scheduling; inclusive spans must not be added
as independent costs or promoted to pure production timings. Root
`ui-sync-spans-diagnostic`, script digest
`44ecf448f4aede9c37ac3d2b0413833bc2fb66036c9bbe755e662f07d18ae1b5`.

The existing first-activation cProfile observer captured its real activation
await and retained the binary profile and diagnostic receipt. However, nine
profile rows have cumulative time below internal time, and some caller edges
report zero calls despite attributed time. Do **not** infer exclusive cost,
subprocess/DNS caller ownership, or a safe config/memo/cache remedy from those
inconsistent tables. The async scope also includes interleaved loop work and
substantially perturbs timing. Raw profile is retained for audit, not relied on
to select a production fix. Root `ui-first-activation-profile`, script digest
`a287fd4cf4ef5084665494e057699fd0bb5c0ad39282a7c8a8778f3806e96339`;
summary `/tmp/task31966-send-activation-profile-summary.log`.

Next diagnostic design: time **entry, synchronous rendering body and exit** of
the existing checked config lifetime separately, plus its rendering substeps.
Every original context-manager/lock/source check must still execute, with no
await or retained config snapshot added. This is a throwaway bounded probe,
not permission to change authority, introduce a cache policy or patch Textual.
The existing derivation helper is only a candidate; cost/freshness must be proven
before extending it. Existing ADR120/150/161/198 remain controlling; a global
GC/cache/authority remedy requires its separate architectural review.
All qualification criteria remain open. Logs:
`/tmp/task31966-send-ui-gc.log`, `/tmp/task31966-send-ui-sync-spans.log` and
`/tmp/task31966-send-activation-profile.log`; roots share the retained
`/tmp/task31966-send-measure-v0Z4Cs` container.

### Checked config phases and summary-query attribution

The approved throwaway entry/render/exit probe and its rendering-substep
extensions ran at the same exact clean `ca793ee190` source, using the original
verified scale corpus and Keyword receipt. Each run restored the isolated
`codex/switcher-workstream-burndown` branch to `6e637de859`; no production,
test, script or package input changed. Native context-manager entry, body,
exception suppression and exit were forwarded, including all source checks
and locks. A runnable forwarding self-check covered success, entry failure,
body failure, suppression, nesting and exit failure. No config payloads,
paths, authority bypass, retained snapshot or GC-policy change was added.

The first probe retained 84 refresh roots, 112 complete scopes, 535 substep
records and 28 paired main-thread gen-2 collections; all drop counters were
zero. In the 58 refreshes inside activation windows, primary entry took
**2.248500–29.051625 ms**, rendering **2.925458–46.569666 ms**, and exit
**0.032000–1.292667 ms**. None of those scopes overlapped gen-2. Wall and
thread CPU clocks are retained separately; entry wall time is not proof of
pure config IO or lock contention. The settings-summary substep reached
35.509250 ms. Root `ui-config-phases-diagnostic`; script
`/tmp/task31966-config-phases.py`, SHA256
`94a59b23c36a2c270b789e4ba6d85302fee48126f911dd6ab57c40d9aabc9634`.

The summary build/apply extension retained 81 roots, 109 complete scopes,
594 substeps and 26 paired collections, no drops. Its 13 summary refreshes
inside activation windows spent 260.167375 ms inclusively in summary sync:
35.168917 ms in state building and 224.927667 ms in widget application.
The application substep peaked at 21.928791 ms. Substeps overlap their parents
and must not be added as independent costs. Existing readiness/context memo
expansion is not justified by these measurements. Root
`ui-config-summary-phases-diagnostic`; script
`/tmp/task31966-config-phases-v2.py`, SHA256
`f3d7d56051ffc91cc86ea531b57bff9c43c94c9dbf8ded364851d9c0e7960362`.

The final forwarding observer timed native `DOMQuery.nodes` evaluation and
`Static.update` during summary application, retaining only calls >=0.1 ms.
No query/update or enclosing-phase record was dropped. In each of the 14
rooted summary applications inside activation windows, **four uncached
ChatScreen-wide queries** consumed **6.501790–15.040541 ms** within application
spans of 12.959291–19.341708 ms. Across all summary applications in activation
windows (including those outside a tracked config-refresh root), 88 retained
queries totalled 262.259913 ms. Retained updates peaked at 0.213000 ms; absence
of below-threshold updates is not proof of zero cost. The source confirms
three compound Model-value lookups and one recovery lookup evaluate the whole
screen tree; the native query evaluator walks all descendants. Root
`ui-summary-query-cost-diagnostic`; script
`/tmp/task31966-summary-query-cost.py`, SHA256
`6b9d72a7ec14ce42e5631a1299879b5b7ee5a8387569fad910d582edbd88d5d1`;
its receipt also pins the unchanged parent probe's digest.

All three failed matrices completed all 60 exact searches and eight exact
opened activations, with no app exception, clean/unchanged source and unchanged
corpus digest. Every activation still exceeded 50 ms: respective narrow/wide
ranges were **81.973167–130.844042 / 142.210250–258.761375 ms**,
**55.353833–109.873875 / 103.075167–159.956750 ms**, and
**64.060375–98.809167 / 88.148041–179.260875 ms**. First-probe wide activation
3 also failed busy paint at 154.760625 ms. Thresholds remained 700/10/10.
Wrappers perturb allocations/scheduling; these are attribution receipts, not
controlled speed comparisons or substitute acceptance evidence. The final
retained post-unmount app has 14 registered handles and open DB descriptors,
not terminal application-owner retirement proof.

The proposed next bounded correction is to resolve the three Model-value rows
through their mounted section containers using native lookups, rather than
scan the entire screen three times. Preserve missing rows, fresh structured
values, remount/recompose behavior, hidden-rail updates and every config/ownership
check. Do not bundle recovery lookup, update equality, cross-pass memo, cache
policy or GC changes. Obtain approval before implementation and prove mounted
RED/GREEN plus the unchanged scale limits afterward. This is not claimed to
remove all stalls: first-probe narrow activation 1's longest 81.973167 ms
interval overlaps neither gen-2 nor a tracked config refresh; other intervals
contain substantial GC. Existing ADR120/150/161/198 apply; no new ADR for this
behavior-preserving local-query proposal. TASK31966 and TASK31245 stay In
Progress; native/Windows/participant/resource and dependent semantics remain
unqualified. Logs: `/tmp/task31966-send-config-phases.log`,
`/tmp/task31966-send-config-summary-phases.log`,
`/tmp/task31966-send-summary-query-cost.log`. Raw diagnostic and normal
`ui-evidence/ui-latency-evidence.json` receipts are retained under each root.

## Approved Model-value lookup correction — mounted verification

The user approved narrowing only the three Model-value lookups after the
query-cost attribution above. The shared summary application now resolves each
mounted row by native ID lookup, then queries that row for its value. Missing
containers/children still skip safely; replacement widgets receive fresh
structured values. The recovery query, update frequency, config scopes,
ownership, geometry and GC/cache policy are unchanged.

Mounted RED at 52x20 and 120x50 observes the real native query evaluator while
forwarding every call: both cases fail `4 == 1`, after confirming actual hidden
row contents and unchanged focus. Missing/remounted child and container coverage
already passes before the fix. Log: `/tmp/task31966-model-query-red.log` (two
failed, one passed in 19.44s). Its pytest exit cleanup reports unrelated old
garbage-directory removal warnings; these were not hidden or cleaned up by this
work. The first GREEN is four passed in 18.87s, no pytest warnings, in a fresh
explicit temporary root (`/tmp/task31966-model-query-green.log`).

A stricter run loaded the read-only census in the private pytest children as
well as the parent. All six mounted bodies passed, but the process gate correctly
failed on unretired harness-owned `library_collections.sqlite` and
`workspaces.sqlite` files. The new tests originally used an imported factory
binding outside the existing opt-in ownership fixture; the older Model cases
also had not opted in. Reusing the module's factory and the existing
`owned_console_apps` fixture repairs test retirement after harness/workers stop,
without new production cleanup. Failed log:
`/tmp/task31966-model-final-tests.log`.

Final affected Model-section and design-token run: **15 passed in 55.98s**, no
pytest warnings, required retirement gate exit 0 in parent and private children.
Each of the six mounted Model child logs records an empty private-file census;
the parent retains only admission/lease descriptors. Child logs are under
`/tmp/task31966-model-owned-71i643/pytest`; parent log:
`/tmp/task31966-model-owned-tests.log`. The committed observer
`Docs/QA/task-31245/descriptor_census_probe.py` was loaded through `PYTEST_PLUGINS`
so disabling child plugin autoload did not omit it. This is test-fixture retirement
evidence, not terminal app-owner/native/Windows qualification.

All eleven artifact guards pass (`/tmp/task31966-model-preflight.log`). The changed
test file is Ruff/format clean, the changed production range is format clean,
and whitespace is clean. Full-file production Ruff retains the same 200 existing
diagnostics as HEAD, comparing code/message after normalizing the line number in
the existing duplicate-definition message; no new diagnostic. JSON receipts:
`/tmp/task31966-model-ruff-base-exact.json` and
`/tmp/task31966-model-ruff-current.json`.

Independent read-only review found no Critical, Important or Minor finding in
this bounded production/test patch. It confirmed native mutation-versioned
lookup freshness and declined to judge the still-pending full qualification.

Fresh clean-head results follow below. Mounted verification alone does not close
the 50ms/100ms limits or any native, Windows, participant or final application-owner
resource gap. TASK31966/TASK31245 remain In Progress with all qualification
acceptance criteria open. Existing ADR120/150/161/198 apply; no new architectural
decision or semantic gate waiver.

## Frozen Model-value scale and query-cost receipts

Measured clean source: **a0772b7c2a75cb1127f6138e9360402c6765f5b7**.
Fresh fixture container: `/tmp/task31966-model-measure-l9EZAY`. Every build,
Keyword and UI receipt verifies the exact clean source again at CLI settlement;
the original corpus digest remains unchanged:
`9f3395e3cb692987d08c3270b36394bcc93cb4bf488c4ed527e51a71108cb27f`.
This is not evidence for a future rebase or changed production head.

- `scale`: 10,000 conversations, 250,000 eligible selected-branch messages plus
  four exclusions, 10,000 indexed documents; seed 107.645329s, index build
  3.090368s, ready, zero registered handles after cleanup.
- `keyword`: all 300 exact manifest queries pass after five warmups per query;
  P95 **128.731708ms**, max loop interval **15.789042ms**, no correctness failure,
  zero registered handles and no owned database descriptors after cleanup.
- `ui`: both real-owner viewports complete all 60 exact searches and eight exact
  OPENED activations, no app exception. Both preparation windows pass
  (43.377250/41.134000ms); all busy-paint limits pass. Seven of eight activation
  windows still fail 50ms; wide `unicode-long-1` search also fails at
  **60.254791ms**. The ordinary paint observer's maximum duration is
  **52.958875ms**, retained as an overhead limitation rather than attributed
  entirely to production. The retained post-unmount app has 16 registered
  handles and open database descriptors: not terminal application retirement.

Activation maximum event-loop intervals (ms), unchanged 50ms limit:

| Viewport | First open | Other chat | Unicode chat | Reuse first chat |
| --- | ---: | ---: | ---: | ---: |
| 52x20 | 85.193250 | 51.468708 | 44.424708 | 51.794166 |
| 120x50 | 64.160042 | 70.511458 | 81.063958 | 111.111000 |

The unchanged throwaway native-query observer then ran at the same exact source
and corpus under `ui-summary-query-cost`. Its entrypoint and parent hashes remain
the previously recorded `6b9d72a7…` / `f3d7d560…`; forwarding self-check passes.
It retains 86 config roots, 115 scopes, 499 substeps, 25 paired main-thread gen-2
collections and 47 query/update observations, with no drops and thresholds
unchanged at 700/10/10. In every one of **15** rooted activation-window summary
applications, exactly **one** cold screen-wide query remains (the unmodified
recovery lookup), versus four in every baseline application. Its retained query
cost is **1.958292–3.975500ms** within total apply cost
**3.442042–9.092125ms**; baseline corresponding ranges were
6.501790–15.040541ms and 12.959291–19.341708ms. Retained Static updates max
0.152875ms. Below-threshold observations are not zero-cost, and overlapping
inclusive spans cannot be added independently.

This diagnostic also completes all 60 exact searches/eight exact activations,
without app exception or source/corpus mutation, but six activation loop windows
still fail: narrow **56.666708 / 38.686958 / 35.810125 / 56.262834ms**, wide
**101.339167 / 99.960958 / 94.205250 / 124.529209ms**. Paint observer max is
1.382459ms; all busy limits and preparation windows pass. In its longest wide
intervals, gen-2 overlaps last 49.680375, 61.246708, 53.798084 and 90.526208ms.
Narrow activation 1's longest failed interval contains neither gen-2 nor a
tracked config root; narrow activation 4's failed interval contains two config
roots and no gen-2. GC contributes but is not the only residual cause. Wrappers
perturb allocation/scheduling, so diagnostic and ordinary runs are not a
controlled whole-app speed comparison. The diagnostic's retained post-unmount
app has 18 registered handles and open DB descriptors, not retirement proof.

Raw build/Keyword receipts, normal and diagnostic `ui-evidence` receipts, painted
frames, config-phase and query-cost records remain under the named container.
Logs: `/tmp/task31966-model-scale.log`, `/tmp/task31966-model-keyword.log`,
`/tmp/task31966-model-ui.log`, `/tmp/task31966-model-summary-query-cost.log`.

The approved lookup correction is confirmed, but the broader latency acceptance
criterion remains unsatisfied. After the three bounded recovery-height,
Send-reason and Model-query corrections, pause for a Console refresh/GC and
measurement-overhead architecture discussion before attempting another
production correction. No speculative GC/cache policy or fourth fix is approved
or applied. Native, Windows, participant and terminal-resource qualification
remain unwaived; no final combined PR or dependent semantics work is ready.

## Current-dev integration and measured Enter barrier correction

The architecture checkpoint rebased all 15 follow-up patches onto dev
`3146bbd8da2f30eaa97d72bd0ab860de4c40de75`, producing clean measured head
`ac3a4f131561972189f6c2df47df70316907d88a`; the patch range-diff is unchanged.
The 16 affected integration files passed **288 tests in 484.67s**, with the
required read-only private-file retirement gate enabled and no warnings.
All 288 census rows retain only admission/lease files, not private databases.
Log: `/tmp/switcher-rebase-ac3-targeted.log`. This is test-fixture evidence,
not terminal application-owner retirement.

Fresh exact-head container: `/tmp/task31966-ac3-qualification-ZJXybB`.
Its corpus has 10,000 conversations, 250,000 eligible messages plus four
exclusions and 10,000 index documents; digest
`939cf2732ae8f6fc8e57f2743a03de824b2503cfbe663ed2b3e52bc040f3c80e`.
Keyword passes all 300 timed queries (P95 117.960333ms; loop maximum
17.335625ms), with zero owned database descriptors/registered handles after
cleanup. Normal UI completes all 60 searches and eight exact OPENED
activations without an app exception, but seven activation intervals and one
wide body search still fail 50ms; all preparation/busy limits pass. Raw roots
are `scale`, `keyword` and `ui` beneath that container. These failures remain
failures, not a post-correction latency receipt.

The bounded diagnostic roots `ui-tail-capture`, `ui-native-refresh`,
`ui-config-acquisition` and `ui-gc-caller` retain their raw records and failed
UI matrices at the same exact source/corpus. Final-frame capture spans are
1.640500–8.246458ms and do not overlap the worst failed intervals in that run.
All 249 observed native lock acquisitions succeed, with zero retry branches;
this does not prove zero contention or explain high wall/low CPU entry spans.
Other failed intervals contain native rendering and/or gen-2 work, sometimes
neither tracked config nor native refresh. Inclusive overlaps are not additive,
and instrumentation changes scheduling/allocation; no time is subtracted.

The Enter-specific native diagnostic identifies a **73.944541ms** gen-2
collection triggered inside `Pilot._wait_for_screen`, after dispatching Enter.
That barrier walks all descendants and queues up to **751** callbacks; direct
walk/registration inclusive cost peaks at 76.979216ms. There are 24 observed
waits, 26 paired gen-2 events and zero drops, with unchanged 700/10/10 GC
thresholds. Trigger stack tags identify the allocating call, not ownership of
the scanned or collected objects. Other stalls remain unexplained.
Diagnostic entrypoint: `/tmp/task31966-gc-caller-probe.py` (SHA256
`ece43d63d6a4ee383cca27195355695bd892bb6352ac30c18a5bd0711390835b`);
forwarding/result/error self-check passes. Raw diagnostic SHA256
`25988ed619471b90be367f6a4a5265be6ec2a2a3400aacf2e53d9c5f53310342`.
Log: `/tmp/task31966-ac3-gc-caller.log`.

The bounded test-only correction uses Textual's same native
`App._press_keys(("enter",))` dispatch, preserving its idle/animation waits and
removing only Pilot's extra descendant barrier from measured activation.
Typed OPENED, modal unregister, exact exposure, transcript ownership/readiness,
composer focus, source/corpus guards and 50ms/100ms limits remain unchanged.
Untimed setup and other Pilot operations are untouched; no production or GC
policy change is made. Existing ADR120/198 apply; no new ADR is required.

A real mounted Input regression first receives the exact native submission but
fails on four barrier callbacks (`DispatchApp`, `Screen`, `Input`, `Static`),
then passes after the correction. RED: `/tmp/task31966-pilot-barrier-red.log`.
The complete two-file affected run passes **48 tests in 10.84s**, no warnings,
strict private-file gate enabled; every census row contains only admission and
lease descriptors. GREEN: `/tmp/task31966-pilot-barrier-green.log`.
Independent read-only review finds no actionable issue and confirms the actual
activation settlement gates remain intact. This is harness evidence only.

Before freezing this correction, all eleven artifact guards pass
(`/tmp/task31966-pilot-preflight.log`); both changed Python files are Ruff/format
clean and whitespace is clean. No full test sweep was requested or run.

Next: freeze the corrected clean source, build fresh source-bound scale and
Keyword receipts, and rerun the unchanged real-owner UI matrix. The ac3a4f1
receipts cannot be retagged as evidence for this changed source. Native,
Windows, participant and terminal resource qualification remain unwaived;
TASK31966/TASK31245 remain In Progress and their open criteria are unchanged.

## Corrected-driver qualification at clean 81c0d22

Fresh source-bound container: `/tmp/task31966-pilot-measure-PKKkEE`, measured
head `81c0d22ff9538f75135c09eb0c43a3f31b7a0ccc`. The scale receipt records
10,000 conversations, 250,000 eligible selected-branch messages plus four
excluded canaries, and 10,000 index documents. Corpus SHA256:
`cfaefa4ead2e1f7d844c3354498a391e58b1cb9f4a5b6a4689a43b22752ad54b`.
The fresh Keyword receipt passes all 300 timed manifest queries: warm P95
128.429459ms, maximum loop interval 16.524ms, no correctness failures, zero
owned database descriptors and zero registered handles after cleanup.
Source-clean/exact and unchanged-corpus guards pass. Raw roots: `scale` and
`keyword`; logs `/tmp/task31966-pilot-scale.log` and
`/tmp/task31966-pilot-keyword.log`. This is Keyword-only evidence, not Meaning
or UI latency qualification.

The unchanged real-owner UI matrix completes both preparations, all 60
searches and eight exact OPENED activations, with no app exception. Preparation,
search and busy-paint limits pass; maximum busy paint is 74.517042ms. Six of
eight activation loop intervals still fail the unchanged 50ms limit:

| Viewport | Activation loop intervals (ms), in order |
| --- | --- |
| 52×20 | 72.963958 FAIL; 51.315000 FAIL; 47.356833 PASS; 48.625583 PASS |
| 120×50 | 60.630625 FAIL; 90.121500 FAIL; 79.306833 FAIL; 116.870292 FAIL |

Raw failed receipt: `ui/ui-evidence/ui-latency-evidence.json`; log
`/tmp/task31966-pilot-ui.log`. The retained post-test app still has 16 registered
handles, not terminal ownership retirement. Do not retag these receipts to a
later documentation head or infer a controlled whole-app speedup.

The same external GC-trigger diagnostic is retained under `ui-gc-caller`
(`/tmp/task31966-pilot-gc-caller.log`). It records 16 Pilot waits, 27 paired
generation-2 collections (54 start/stop records), zero dropped wait/GC records
and unchanged 700/10/10 thresholds. No measured activation overlaps a Pilot
descendant barrier, confirming that contribution was removed. Remaining
collections occur inside native layout/rendering; one allocating trigger is the
benchmark's extra full-frame `render_strips()` capture. This identifies a
candidate observer contribution, not scanned/dead-object ownership or a proven
production remedy. Inclusive spans are not additive, timings are not corrected
by subtraction, and the instrumented matrix remains failed. No fourth
production correction or global GC/cache policy is selected.

## Partial native macOS walkthrough at clean 81c0d22

The approved dedicated Terminal window (PID 2934, window 118726) ran a freshly
prepared disposable profile at the same frozen head. Actual observed viewport
was **244×73**, not any required qualification viewport. Host: macOS 26.5.2
arm64, Python 3.12.11, Textual 8.2.8. Font/zoom/remapping and exact Terminal
version were not recorded. Native source digest:
`dfbdfa4fc80e5a22f4a13fa383df85486e3baa01953cd01b74ef52ef11202ee1`;
30 conversations, 60 messages and 28 Keyword index documents. Synthetic
Amber/Indigo/Cedar/Copper, Unavailable and empty-character fixtures were used;
no real profile or provider requests. Screenshot directory: `native-shots`
(00–65) under the named container.

| Native case | Observed result at 244×73 | Evidence |
| --- | --- | --- |
| Blank Active MRU Enter | Switched Indigo-07 to exact Amber-07 marker, composer focused, three tabs unchanged. Reopening distinguished CURRENT Amber-07 from highlighted MRU Indigo-07. | Screenshots 21–27 |
| History cold open | Pointer selected History; native digit keys accepted query `01`. Enter opened exact `NATIVE_MARKER_AMBER_01`, with composer focus; tabs increased once from three to four. | 33–43 |
| History warm reuse | History Enter reopened the exact Amber-01 marker; four tabs unchanged and composer focused. | 44–50 |
| Mode-cycle/Character pointer | Shift+F3, shifted Fn+F3 and Character-pointer attempts did not yield an observed Character-mode frame. Not passed. | 28–32, 51–56 |
| Text insertion | CUA AX `type_text` reported insertion but no query change was observed. Native individual digit keys worked separately. Not a text-entry pass. | 37–41 |
| Escape | Later capture returned to unchanged Amber-01 and four tabs. Delayed background captures do not identify a controlled pre/postcommit boundary. | 57–62 |
| Normal quit | Actual native Ctrl+Q, then settled Terminal `[Process completed]`; receipt return code zero. No controller signal was used. | 63–65; `native-macos/native-return.json` |

Immediate background captures sometimes retained an older frame, so CUA action
acknowledgements alone are not counted as observed results. Required 52×20,
120×50, 72×35 and 80×24 native checks remain unrun; the earlier resize attempt
did not change actual cells. Character/Context/Roleplay/recovery/rename and
controlled cancellation cases, Windows and three actual unfamiliar
participants remain unqualified. ADR031 moved mode cycling to **Shift+F3**;
the stale checklist and task wording are corrected, without changing bindings.

The native receipt's status is `returned-not-qualified`: source-clean/exact and
unchanged-corpus guards pass, but **37 owned database descriptors remain open
at `App.run()` return**. They include Chat, evaluation, collections,
subscriptions and workspace databases and WAL/SHM files. Later OS process exit
is not app-owner retirement evidence. Startup's optional missing
`python-frontmatter` notice is separate from this finding; no warning was
suppressed. Read-only shutdown tracing found no general app-cache retirement
in the installed lifecycle; database `close_connection()` is current-thread
only, whereas existing file-owner quiescence is a separate stronger boundary.
Do not blindly close global/foreign owners or use test-fixture cleanup to claim
a production lifecycle fix. Preserve this failed resource receipt while tracing
the actual ownership and worker-settlement contract.

TASK31966 and TASK31245 stay In Progress. No qualification waiver, semantic
work, full test sweep or follow-up PR is implied by this partial evidence.

## First-busy observation correction: targeted receipt

Clean 81c0d22 normal activation windows captured the full modal 3–18 times
apiece although the contract retains only the first actual busy frame. The
native trigger diagnostic separately located a gen-2 allocation trigger in
this extra benchmark full-frame rendering. This is not proof that removing
captures eliminates all production pauses; a collection can move elsewhere.

The bounded test-only correction keeps actual native display and all owner/
mode/batch gates, captures until the first activation busy frame, then stops
only subsequent redundant activation modal captures. Search/preparation paint
capture continues. The entire-operation sentinel, typed OPENED, strict exact
readiness, actual modal unregister and final Chat compositor capture remain
unchanged; no time is subtracted and no limits or production policies change.

The real mounted RED forwards compositor output unchanged and observes a later
native repaint, but fails because full captures increase from one to two after
the first actual busy frame. The search control passes. Original tool-output
excerpt retained at `/tmp/task31966-busy-capture-red-excerpt.txt` (explicitly an
excerpt, not a full log): one failed, one passed, 20 deselected in 0.60s.

Initial GREEN: 50 tests pass in 8.65s with an observation-only census; that run
did **not** enforce retirement. Corrected strict rerun: **50 pass in 9.15s**, no
pytest warnings, `FILE-RETIREMENT-REQUIRED True`, all 50 read-only census rows
contain no database files. Log: `/tmp/task31966-busy-capture-strict.log`.
Both changed Python files are Ruff/format clean; whitespace is clean.
Independent read-only review finds no actionable issue and confirms all
existing acceptance boundaries remain. This is a harness correction, not full
branch qualification. Fresh frozen-head scale evidence is required next.

All eleven derived-artifact guards pass
(`/tmp/task31966-busy-capture-preflight.log`). Both Backlog integrity guards
pass again after the final task-note edits. No full suite was run.

## Frozen first-busy correction: full-scale result

Measured head `0abbd7bf16f58249ab0e5a5f6b9c8f5d0ef8ff0c`; fresh container
`/tmp/task31966-busy-measure-pGwqbB`. Source remained clean/exact throughout
both ordinary and diagnostic measurements. The new real-API corpus has counts
10,000/250,004/10,000, integrity `ok`, Keyword index `ready`, and digest
`ff8fa926743839b41a0f888426207dee8bbb30e22588431c0f4203294b224c6c`.
Seeding took 154.699196s, index construction 3.960759s; registered handles after
build cleanup: zero. Raw build receipt is in `scale/build-receipt.json`;
log `/tmp/task31966-busy-scale.log`.

Fresh standalone Keyword passes all 300 timed queries: P95 123.915667ms,
maximum loop interval 13.286542ms, no correctness or acceptance failures,
unchanged corpus, zero owned DB descriptors/registered handles after cleanup.
Receipt `keyword/keyword-receipt.json`; log `/tmp/task31966-busy-keyword.log`.
This does not qualify Meaning or native interaction.

The ordinary real-owner UI matrix completes both preparations, all 60 exact
searches and eight exact OPENED activations with no app exception. Every
activation now records **one** actual busy capture, at most 0.783792ms observer
duration. All preparation, search and busy-paint gates pass, but **seven**
activation intervals still fail the unchanged 50ms limit:

| Viewport | Activation loop intervals (ms), in order |
| --- | --- |
| 52×20 | 66.633042 FAIL; 39.285375 PASS; 68.128625 FAIL; 51.300791 FAIL |
| 120×50 | 68.768917 FAIL; 113.711542 FAIL; 114.820666 FAIL; 102.904625 FAIL |

Failed receipt: `ui/ui-evidence/ui-latency-evidence.json`; log
`/tmp/task31966-busy-ui.log`. The retained post-test app has 15 registered
handles, not terminal retirement. The correction demonstrably removes redundant
captures, but these separate runs are not a controlled speedup comparison and
the broader latency criterion is still unsatisfied.

The unchanged external GC/native-refresh/config observer passes its forwarding/
exception self-checks and runs once at this same source/corpus. Raw diagnostic
root: `ui-gc-caller`; log `/tmp/task31966-busy-gc-caller.log`; absolute interval
overlap analysis: `/tmp/task31966-busy-analysis.json`. The outer trigger observer
retains 26 paired gen-2 collections (52 records), 16 Pilot waits, valid pairing
and zero drops. The nested config observer retains 25 pairs over its shorter
lifetime. Thresholds remain 700/10/10. The instrumented matrix still fails six
activation windows and one Keyword semantic-category query window; the latter
expects zero Keyword matches and is not a Meaning execution.

Worst failed narrow activation 1 (62.326667ms) and wide activation 1
(65.269542ms) contain no gen-2 collection. Their tracked native/config spans
leave other work unattributed. In the other failed activation intervals,
allocating triggers are native transcript Static construction, partial-update
Strip construction, Markdown mount task creation and layout arrangement:

| Activation | Collection wall / thread CPU (ms) | Collected objects |
| --- | --- | --- |
| 52×20 #3 | 39.819625 / 39.004458 | 13,091 |
| 120×50 #2 | 74.807125 / 74.289416 | 34,525 |
| 120×50 #3 | 67.138333 / 66.943541 | 12,985 |
| 120×50 #4 | 60.689125 / 60.498708 | 12,621 |

Trigger metadata locates the allocation, not ownership of scanned/dead objects.
Inclusive native/config/GC spans overlap and are not additive; no acceptance
time is subtracted. A separate gen-2 trigger occurs in the required final Chat
capture elsewhere in the diagnostic; it is not proof that all worst intervals
are observer work. Native first-busy and final exact-readiness/transcript gates
remain intact. No additional production fix, global GC/cache policy or
threshold change is selected. Existing architecture checkpoint remains in
force; native/resource/Windows/participant qualification is still open.

Read-only resource tracing additionally distinguishes process-global lazy
Chat/Prompt/Media owners from app-created Library/Workspace/Subscriptions
owners. `SubscriptionsDB.close_all_connections()` explicitly checkpoints and
closes the calling thread only, reporting other live-thread caches; its name
does not promise cross-thread retirement. This is why the native 37-descriptor
receipt cannot justify a blanket cleanup or a fixture-based production claim.
An ownership-specific shutdown remedy is not yet established.

## Shutdown-owner census and exact opening identities — clean `d7b1e915dc`

This is a **headless ownership diagnostic**, not native, performance, Windows
or participant qualification. The existing guarded small native fixture was
prepared at `d7b1e915dc7ecd27d16bee8d21eac7979feeff19` under
`/tmp/task31966-shutdown-owner-qyE6oG/prepared`: 30 conversations, 60 messages,
28 Keyword documents, zero registered handles after preparation. Corpus digest:
`e3cff3d6140fdd094d2ea0a1a7a7f6b95fd54182966b2d21180efaec1945411b`.
Every completed attempt retained clean/exact source and an unchanged corpus.

The throwaway driver uses the real synthetic app, actual saved-chat preparation,
`App.run_test`, ordinary `action_quit`, Textual shutdown and `asyncio.run`'s
default-executor join. The census reads the already-installed storage registry,
native transaction flags and owned descriptors. It never closes a handle,
collects garbage, changes GC/cache policy, suppresses warnings, reads real
profiles or touches the other checkout. Normal CLI entry points were also
inspected: both arm the exit watchdog after `App.run`; neither contains an
additional database cleanup that the native fixture omitted.

Retained headless receipts, all below that temporary container:

| Attempt | After Textual: connections / DB descriptors | After runner join: connections / DB descriptors | Limit |
| --- | --- | --- | --- |
| `headless` | Not captured | 13 / 38 | Invalid quit driver: awaited synchronous `action_quit`, raising TypeError; not normal-quit evidence |
| `headless-corrected` | 12 / 37 | 12 / 37 | Correct actual quit; initial registry inspection skipped absent repository referents |
| `headless-all-repositories` | 14 / 39 | 14 / 39 | Includes absent referents; none found in this run |
| `headless-first-open` | 15 / 42 | 15 / 42 | Scalar object-ID origin matches are insufficient because IDs can be reused |
| `headless-weak-identity` | 10 / 34 | 10 / 34 | Exact weak-key connection tokens; final origin receipt below |

The scalar-origin attempt still observed one active storage operation and three
pending acquisitions immediately after Textual shutdown; both were zero after
the runner joined. In the final weak-identity run, both counters were zero at
both end snapshots. These boundaries are not interchangeable physical-settlement
proofs. Counts differ across separately scheduled runs; this is not a controlled
speedup or resource-reduction comparison.

The final probe uses monotonically assigned tokens in a WeakKeyDictionary keyed
by the actual installed native connection. It retains only scalar stack metadata,
not frame or connection references. All ten retained connections have distinct
nonnull tokens, each matched to an observed successful original registration;
108 registrations were retained, none dropped. Original call arguments, return
identities and exceptions remain forwarded. The final script is retained as
`owner-probe-weak-identity.py`, SHA-256
`c31e69b6519d76c56d144e4f891ad62305ccdd1f3ccc5de480dbd4bfa3652fc7`.
Raw receipt: `headless-weak-identity/owner-shutdown-diagnostic.json`; normal fixture
return receipt alongside it. Log: `/tmp/task31966-shutdown-owner-weak-identity.log`.

| Retained installed owner | Native connections after runner join | Proven origin / thread |
| --- | --- | --- |
| Shared Chat DB | 6 | Main-thread citation-service construction; five exited workers: two local-marks `unread_ids_for`, one local-marks `unread_token`, one world-book summary, one cached RAG conversation metadata read |
| Evaluation DB | 1 | Main-thread evaluation-service construction / schema setup |
| Library collections DB | 1 | Main-thread app constructor / schema setup |
| Subscriptions DB | 1 | Main-thread app constructor / schema setup |
| Workspace DB | 1 | Main-thread app constructor / schema setup |

All ten remain native-open, with live leases and no active native transaction.
The five retained Chat worker threads have actually exited. Chat is the exact
instance referenced by both `app.chachanotes_db` and the config cache; the
Library/Subscriptions/Workspace rows match the app's original attributes.
No absent-repository participant was found. Evaluation is reached through its
service rather than a direct app DB attribute. This identifies held connection
owners, not a one-to-one attribution of every SQLite/WAL/SHM descriptor or the
rendering heap scanned by GC. One separate theme executor is still present in
the final thread census; default-executor join does not prove every producer
has stopped.

The next resource correction must distinguish finite worker reads from
constructor-held app caches and the shared process cache. Existing finite-worker
retirement is a candidate only where its real callback/borrower contract holds;
an app-wide blanket close is not justified. No production remedy, fourth
performance correction, new global lifecycle/GC/cache policy, qualification
waiver or PR was introduced by this diagnosis. All remaining native/resource,
latency, Windows and participant gates stay open.

### Finite unread-reader retirement — 2026-10-05

The exact weak-key origin trace above identified finite local-marks reads among
the retained Chat worker handles. Both shared readers (`unread_token` and
`unread_ids_for`) now use the installed operation-owned boundary only for the
exact file-backed `CharactersRAGDB` class. Their manual lock, query, chunking,
validation, token/revision semantics and returned materialized values remain
unchanged. Already registered caller caches and transactions, memory databases,
custom owners and other mark readers/writers retain their previous lifetimes.
Imports remain lazy; no new owner abstraction, public API or UI change was added.
Existing ADR120 and the installed finite-operation contract apply; no new ADR or
global lifetime/GC/cache policy is introduced.

Real file-backed, physically joined executor regressions first failed against
the unchanged production readers: four failures each observed two registered
handles instead of the expected one. Eight borrowed-transaction/memory/custom
controls already passed. The failure cases execute a real missing-table query
after acquisition and preserve its `sqlite3.OperationalError`.
Raw RED: `/tmp/task31966-marks-red-pWi8uO/tests.log` (4 failed, 8 passed, 7.31s).
Initial GREEN: `/tmp/task31966-marks-green-DfEAWi/tests.log` (12 passed, 7.25s,
no pytest warnings, required read-only descriptor gate clean).

The first wider 97-case run passed every body but **failed** the required
retirement gate. Old local-marks unit tests had no terminal database teardown:
85 observations retained database files; the final observation held 43 Chat
handles plus WAL/SHM descriptors, and the unchanged FD sentinel warned of growth
by 206. This is not a warning-free or resource-qualified pass.
Raw failed gate: `/tmp/task31966-marks-cover-rIMHvo/tests.log`.

The unit module now captures only its actual constructor alias and quiesces
those exact test-owned databases after bodies and their worker pools stop.
Final wider verification: `/tmp/task31966-marks-final-oA64Gu/tests.log`:
97 passed in 39.58s, no pytest warnings, all 97 required read-only observations
contain no database files. This covers local marks, timestamp shape, attention
projection, installed finite-offload/cancellation contracts and the new ownership
regressions. Fixture retirement is not a production shutdown policy.

The first mounted reminder test's child body also passed, but its child census
retained constructor databases and the custom marks database, so the process
gate correctly failed. A separate sandbox-only pytest cache-write warning was
also retained. Raw failure:
`/tmp/task31966-marks-mounted-red-jgvvxN/tests.log`. The module now opts into the
existing `owned_console_apps` fixture and registers each exact marks database
for retirement after final runtime disposal. An explicit writable parent cache
directory does not reach the private child, which clears `PYTEST_ADDOPTS`.
No warning filter or threshold change is used.
Full mounted verification: `/tmp/task31966-marks-mounted-final-zgoSqI/tests.log`,
14 passed in 125.64s. The parent has no pytest warnings, but inspection of the
actual child logs still finds cache-write warnings; this is not warning-free
verification. Every one of the 14 actual child
processes also has a required clean read-only database-file census. Coverage
includes stale acknowledgements, duplicate/cancelled navigation, actual unread
menu dispatch, composer/row focus and Unicode/ASCII action geometry at three
sizes. These mounted tests are not native Terminal qualification.

The properly authorized rerun at
`/tmp/task31966-marks-mounted-clean-OFaktB/tests.log` passes all 14 cases in
138.11s. The parent and each actual child log have no pytest warnings; all 14
child resource observations contain no database files. No code, warning filters
or thresholds changed between those runs. Child cache writes required normal
worktree write authorization because the private runner deliberately clears the
parent's `PYTEST_ADDOPTS`; the failed/non-warning-free receipts above remain.

Independent read-only reviews found no actionable issue in the production
scope, real regression tests, unit constructor capture or mounted fixture
registration. All eleven derived-artifact guards pass in
`/tmp/task31966-marks-green-DfEAWi/preflight.log`. Changed Python passes Ruff;
test files and the modified production method range are formatted. The unchanged
`_now` expression elsewhere in the production file retains its pre-existing
full-file formatting difference. Whitespace is clean.

Fresh clean-source shutdown evidence remains pending. Neither these tests nor
fixture cleanup proves all app/process caches closed, latency qualification,
native input/geometry, Windows or participant usability. All corresponding
workstream gates remain open; no final follow-up PR is ready yet.

### Latest-dev integration and unread-owner shutdown checkpoint — 2026-10-05

All twenty task-owned commits were rebased from dev `3146bbd8da2f` onto
`74557e202ac38c6d29510d0940a062ca7cc7f38b`. Eighteen patches replay unchanged.
The rename patch retains upstream project-state validation, fork admission and
all resume-controller arguments alongside its title fences and rename delegate.
The fixture patch keeps upstream's explicit stale-model setup. Independent
read-only review finds no actionable conflict-resolution issue.

Clean combined source `78345fc75e7e5cd3bc521f161053ca51d31629af` passes the
affected rename/controller/unread/finite-owner batch: **267 passed in 413.28s**,
process exit zero. Parent and actual child logs contain no pytest warnings.
Seventeen census reports retain 283 observations, none with database files.
All eleven artifact guards pass. Raw receipts:
`/tmp/switcher-rebase-74557-VeyPxU/tests.log`, private child logs under its
`pytest` directory and `preflight.log`. A duplicated bootstrap marker from the
replay is subsequently removed, and inherited import ordering is normalized;
production code is unchanged by this mechanical test cleanup.

The installed guarded 30-conversation/60-message/28-card fixture is freshly
prepared for that clean head. Corpus digest:
`390de49d79688f88b682f2a2942aa07428e453f68cbbfbc89b47f89005d1f666`;
preparation retires every registered handle. The unchanged weak-key origin
observer (digest `c31e69b6519d76c56d144e4f891ad62305ccdd1f3ccc5de480dbd4bfa3652fc7`)
performs actual headless quit and joins the default executor. Return code is
zero; source/corpus guards hold; 131 distinct registrations, zero dropped.
After runner join, **13 native-open connections and 39 database descriptors
remain**, with zero active storage operations or pending acquisitions.

Eight retained Chat worker handles originate from two world-book summaries,
three cached RAG metadata reads, two annotation reads and one citation-count
read. No retained handle originates from the corrected unread readers. Five
main-thread constructor owners remain: shared Chat/citation context, Evals,
Library collections, Subscriptions and Workspaces. All eight worker threads
have exited; the separate theme executor is still live. This does not prove
every application producer has stopped or justify blanket owner closure.
Because upstream changed before this run, total counts are **not** a controlled
same-source improvement over the old ten-connection trace. The narrow unit
RED/GREEN remains the unread correction's causal evidence.

Raw source-bound artifacts: `/tmp/task31966-marks-shutdown-7SGQDg/prepared`
and `headless/owner-shutdown-diagnostic.json`, `headless/native-return.json`,
with the invocation logs alongside. The shutdown observer ran separately from
latency measurement; another interpreter was running targeted tests, so this is
resource-identity evidence, not a timing measurement. Native, latency, Windows,
participant and terminal app-cache qualification remain open.

### Remaining finite Console readers — 2026-10-05

The eight exited-worker origins above are finite, materialized reads. Six
offloads now reuse installed `run_owned_db_call`: cached/fresh conversation
metadata, world-book summary, citation counts, annotation previews and the
annotation browser's sibling initial read. Captured database ownership,
authority/privacy reads, memory/custom/borrowed routes, physical cancellation
completion, stale-result fences and fail-soft/closed outcomes are preserved.
Writers, constructor caches, global GC/cache/lifetime policy and timing limits
are unchanged. This is resource correction, not a fourth performance remedy.

`Tests/UI/test_console_finite_read_ownership.py` exercises real file-backed SQL
twice per owner, both with healthy data and with its real query table made
unavailable. All twelve cases have a valid original-offload RED: materialized
results are correct but two registered connections remain instead of the
original main-thread one after the runner physically joins its worker. Initial
citation fixture errors occurred before the query and are not counted as RED;
their corrected actual callbacks then reproduce the same handle failure.
The sibling browser cases additionally prove normal dismissal/failure releases
its inflight guard. Each fixture quiesces only its own DB after runner join.

The first affected caller batch reports 111 passes and 18 stale-fixture
failures. Citation shells lacked extracted session hooks, the turn-action and
memory-banner dependencies; mounted harnesses also switched away from the
collection-bound isolated profile. Their declared dependencies are repaired,
the hydration harness reuses the existing shell, and these non-profile-selecting
tests retain the isolated bootstrap profile. No authority checks or citation
assertions are removed. Those failed logs remain, not relabelled as green.

The final targeted command includes both changed UI tests, retrieval/review
controllers, review-notes modal, installed finite-owner contracts, fresh-scope
identity privacy, cached malformed-scope semantics and the memory routing
regression. With `PYTEST_PLUGINS=descriptor_census_probe`, a fresh explicit
`--basetemp`, `-p no:cacheprovider` and
`TLDW_TEST_REQUIRE_FILE_RETIREMENT=1`, **129 pass in 32.99s**, process exit zero,
no pytest warnings. All 129 terminal observations contain no database files.
Both changed tests are Ruff/format clean. Four production-file diagnostic
multisets have no additions (only embedded line references are normalized for
an inherited duplicate-handler diagnostic); modified method ranges are format
clean. All eleven artifact guards pass; the gated UI census now includes the
new file (143 paths, unchanged floor 140). Independent read-only production and
fixture reviews find no actionable issue. Whitespace is clean.

Raw RED receipts: `/tmp/switcher-finite-read-red-ko4Po5/tests.log`,
`citation-session-hooks.log` and `browser-red.log`. Final caller and guard logs:
`/tmp/switcher-finite-read-verification-LIrtV3/final-tests.log` and
`preflight.log`; prior fixture-failure receipts remain alongside. A new guarded
clean-head shutdown probe is next. This checkpoint does not prove terminal
application-cache retirement, native/Windows/participant qualification or a
passing latency matrix; all corresponding task criteria remain open.

### Finite-reader shutdown and native retry — 2026-10-05

Clean `4448cbb2101914968036a1e6e7f879b7fc8c6ea5` is freshly prepared with
the guarded 30-chat/60-message/28-card native fixture. Corpus digest:
`dadd831afc30bb5e01b81c51c6e45563efe86b13ea99b6dcba3ed4ba67ffc936`.
The unchanged weak-key observer records 134 unique registrations, zero drops,
actual headless quit and physical default-executor join. Return zero and both
source/corpus guards hold. After join, six native-open connections and 27 DB
descriptors remain, with no active operations or pending acquisitions. None
originates from the six repaired finite-reader offloads. Five are unchanged
main-thread constructor caches; the sixth is an exited Chat worker at
`project_workspace_membership`, reached by the actual asynchronous workspace
projection retry. The separate theme executor remains live. These origin
observations do not establish global producer shutdown or justify blanket
cache closure; total counts are not a controlled latency comparison.

CUA permissions are granted and actual native control works in a new dedicated
Terminal window (ID 121371); original terminal windows and real data are
untouched. At the observed 244x73 geometry, native Ctrl+K shows distinct CURRENT
Indigo and selected OTHER OPEN Amber rows; Enter resumes Amber's exact
`NATIVE_MARKER_AMBER_07` transcript with the same three tabs. Native Ctrl+Q
exits normally, returning zero with unchanged source/corpus. The post-Enter
composer contains stray `;2d`; its source is not yet established, so this is
partial MRU evidence, not clean focus/keyboard qualification. App.run return
still retains 29 DB descriptors, not terminal owner-retirement proof. Required
compact/wide viewports and Character journeys, Windows and unfamiliar-user
qualification remain open.

Raw root: `/tmp/task31966-finite-read-shutdown-CnUfDM`; headless origin and
return JSON are under `headless`, native source-bound receipts under `native`,
and `native-start.png`, `native-switcher.png`, `native-mru-enter.png` and
`native-normal-quit.png` retain actual native frames. No controller kill,
global GC/cache change or timing-limit waiver was used.

### Shared projection-authority read retirement — 2026-10-05

The existing four two-database tests had added an outer Chat ownership wrapper
not present in the actual retry entry point, and checked only the registry
cache. Invoking the real direct service offload and store reconcile produces
four valid REDs: membership/retry/failure and registry assertions hold, but a
second Chat connection remains. Three borrowed-transaction, memory and custom
owner controls already pass before the correction.

The shared materialized authority read now reuses installed
`operation_owned_connection` only for a captured exact native file-backed Chat
database. Registry writes, their existing ownership, exceptions/retry behavior,
all sibling callers, borrowed transactions and memory/custom ownership are
unchanged. No new helper, global lifetime policy or fourth performance remedy.

The actual caller/finite-owner batch passes **59 tests in 21.44s**, exit zero,
with no pytest warnings and no DB files in all 59 strict terminal census
observations. Both changed tests are Ruff/format clean; the production modified
range is format clean and its 72 inherited diagnostics have no additions. All
eleven artifact guards and whitespace pass. Independent scoped review finds
no actionable production or test-integrity issue. Raw RED, pre-fix controls,
GREEN and guards are `projection-red.log`, `projection-controls-before.log`,
`projection-green.log` and `projection-preflight.log` in the root above.

Fresh clean-head shutdown is next. Application-cache retirement, native input
and required geometry, latency, Windows and participant gates remain unwaived;
all TASK-31966 criteria stay open and the final combined PR is not ready.

### Projection shutdown and native control limits — 2026-10-05

Clean `18acbb41836a3262e0cdc94b3f7346cd1f89db9b` is freshly prepared with
the guarded 30-chat/60-message/28-card native fixture. The unchanged weak-key
observer records 154 unique registrations, zero drops, actual headless quit
and physical default-executor join. Return zero and source/corpus guards hold.
After join, eight native-open connections and 30 DB descriptors remain, with
no active operations or pending acquisitions. None originates from the repaired
projection or six earlier finite-reader offloads. Five are the unchanged
main-thread constructor caches; three exited-worker Chat handles originate in
the shared current visual-identity resolver. The separate theme executor remains
live. Origin tracing, not fluctuating total counts, identifies this next finite
read boundary; it does not establish global producer shutdown or justify blanket
cache closure.

CUA permissions are granted. The dedicated synthetic Terminal window 121428
again shows distinct current/selected Active rows at 244x73. Pointer selection
opens Character chats, but background query typing and geometry controls do
not reliably commit in the observed frames. Neither keyword navigation nor the
required viewports is passed. Native Ctrl+Q exits normally, returning zero with
both source guards true; 28 DB descriptors remain at App.run return. The normal
quit frame is retained, not relabelled as terminal owner-retirement proof.
Original Terminal sessions and real conversations remain untouched. Explicit
permission to foreground only the dedicated QA window has been requested;
background no-ops are not permission denials or product-failure diagnoses.

Raw root: `/tmp/task31966-projection-shutdown-njK8UJ`. Prepared receipts are
under `prepared`, unchanged origin/return receipts under `headless`, native
return receipts under `native`, with `native-start.png`, `native-normal-quit.png`
and the intervening actual native frames retained. The observer script hash
remains `c31e69b6519d76c56d144e4f891ad62305ccdd1f3ccc5de480dbd4bfa3652fc7`.
No controller kill, GC/cache change, timing-limit change or qualification waiver.

### Shared finite visual-identity reads — 2026-10-05

Current avatar, reaction inventory and immutable historical-avatar callbacks
return materialized values. Each shared Console boundary now reuses installed
`operation_owned_connection` only for exact native file-backed Chat databases.
Borrowed handles/transactions, memory/custom owners, writers, immutable asset
guards, Persona authority revalidation and all original failure/fallback and
stale-result behavior are preserved. No new helper or global lifetime policy.

Six real Character success/SQL-failure cases reproduce the original worker
handle retention after physical runner join. The nine borrower/memory/custom
controls pass before the production fix. An initial borrower fixture updated
all cards and hit a unique-name constraint; those three setup failures are
excluded from RED, and the corrected exact-card controls then pass. Independent
review catches the linked-Persona inventory captures outside the first proposed
scope. Two additional real local-service success/failure cases reproduce that
leak; the final scope includes both captures and the graph query. Review of the
corrected production and test ownership has no actionable findings.

The first affected run records 134 passes, four stale UI assertion/geometry
failures and 56 profile-admission setup errors. Retaining the collection-bound
isolated bootstrap profile, exact constructor ownership and shipping app-tier
sizing CSS removes those fixture errors. The next run records 156 passes and 40
avatar failures: most paths were genuinely blocked by first-run Console setup,
one failure injected a removed ChatController.close_session method, and one
empty-state assertion targeted the obsolete hidden caption. Existing ready-
Console configuration, the actual awaited Runtime.close_session and the visible
identity row restore the intended paths; no setup/authority gate or behavioral
assertion is bypassed. All failed logs remain, not relabelled as green.

Final six-file affected run: **196 pass in 257.02s**, exit zero, with unchanged
strict file-retirement requirement. All 196 terminal observations contain no
database files. No resource warnings occur. Three third-party deprecation
warnings remain at textual_image's Pillow Image.getdata call; they are neither
suppressed nor presented as warning-free verification. New finite-reader tests
are Ruff/format clean; modified ranges are format clean, and full production
and old caller-fixture diagnostic multisets remain unchanged at 47 and 27.
All eleven derived-artifact guards pass; UI census 143, unchanged floor 140.

Raw root: `/tmp/task31966-projection-shutdown-njK8UJ`, including `visual-red.log`,
`visual-controls-before.log`, `persona-red.log`, failed `visual-green.log` and
`visual-green-final.log`, focused `visual-focus.log`/`avatar-ready-focus.log`,
final `visual-green-ready.log`, static JSON receipts and
`visual-preflight-final.log`. A fresh clean-head shutdown comparison is next.
Application-cache retirement, required native viewports/input, latency, Windows
and participant criteria remain open; no fourth performance remedy or waiver.

### Visual-reader clean-head resource and scale checks — 2026-10-05

Frozen source `68e57dc67cccd09abe92ba81ff7b636bacde9051` passes the guarded
startup-only quit diagnostic: 180 unique registrations, zero drops, return0,
unchanged source/corpus and physical default-executor join. Five main-thread
constructor caches and26 DB descriptors remain, with no worker-owned handle,
active operation or pending acquisition. This small startup journey does not
exercise exact Character activation and is not full resource qualification.
Raw root: `/tmp/task31966-visual-shutdown-VRidTG`, with prepared/native receipt,
`headless/owner-shutdown-diagnostic.json`, return receipt and `shutdown.log`.
The separate theme executor remains live; no global producer-stop proof.

Fresh normal scale root: `/tmp/task31966-visual-measure-0tOKKR`. Its guarded
corpus has10,000 conversations/250,000 eligible messages plus four excluded
messages, ready index/quick-check, and digest
`315340b027a8f941d2055a0346d459df86de02e2991232a68d3f03e2690206e7`.
The standalone Keyword check passes300 exact measured queries, zero correctness
failures, P95107.553750ms and loop maximum15.419083ms; owned handles/descriptors
are zero after cleanup. This is not UI latency evidence.

The normal, uninstrumented UI matrix fails, exit1. All60 exact searches and eight
exact OPENED activations complete at52x20 and120x50; source/corpus guards hold
and app exception is None. Seven of eight activation loop windows exceed50ms
(maximum80.175542ms), and one wide body search reaches50.049ms. Both preparation
windows remain below50ms (maximum43.432792ms); every busy paint is below100ms
(maximum58.636792ms). Narrow/wide lists show four/eleven rows from50 fetched.
Eight registered handles at run_test unmount are not a physical runner-join
census. Keep `corpus/build-receipt.json`, `keyword/keyword-receipt.json`,
`ui/ui-evidence/ui-latency-evidence.json` and all three logs; limits are unchanged.

The first broader origin probe installed its registration hook after importing
the app. It records only five registrations while all12 retained handles lack
trace tokens: seven exited workers/34 DB descriptors after runner join. Counts
are observations, but origin coverage is invalid; zero dropped records does not
repair that omission. Its script, log and receipt are retained unchanged under
`ui-owner-probe.py`, `ui-owner.log` and `ui-owner-diagnostic` (script SHA256
`7f22f056b06b0b9cec285de425f85a5501dadd4a55098368f2923b5edb1fb649`).

The corrected hook runs before app imports. It records506 unique registrations,
zero drops and no untraced retained handle. After Textual shutdown and physical
runner join, ten native-open handles/33 DB descriptors remain, without active
operations or pending acquisitions. Five are constructor caches; all five
exited-worker handles originate from the shared exact Character-target
revalidation transaction in workspace.py, not the repaired visual/projection or
earlier finite readers. Tokens208/240/427/440/465 retain the exact acquisition
stacks. Corrected script `ui-owner-probe-v2.py` SHA256
`7b3437928708b597648892b686fd5ba002b46c05842d41f2ca11892ac55a9402`, log
`ui-owner-v2.log`, receipt `ui-owner-diagnostic-v2/ui-owner-diagnostic.json`.
Neither instrumented matrix's timings count as latency/native qualification.

### Shared exact-target revalidation read — 2026-10-05

The canonical coordinator, Roleplay preflight and post-commit validation all
reach the same finite synchronous transaction. Four real typed/raw async
success/SQL-failure regressions reproduce a cold worker cache after physical
runner join (count2 rather than1); three borrower/memory/custom controls pass
before correction. The first run also attempts shared pytest-temp garbage
cleanup and emits unrelated removal warnings. It is retained, not warning-free
evidence. Repeating in a fresh explicit task-owned temporary root gives the
same four valid REDs/three controls with no pytest warnings.

Only this shared transaction now reuses installed `operation_owned_connection`
for the exact native file-backed Chat owner. Materialized values, transaction
snapshot, authority/card/revision decisions, fail-closed outcomes, ordinary
to_thread dispatch/cancellation and every borrower/application cache stay
unchanged. The seven new cases pass the required read-only descriptor gate;
independent scoped review finds no actionable issue. Broader affected verification
and fresh clean-head full-activity comparison follow; no global lifetime policy,
fourth performance remedy or qualification waiver. Raw root:
`/tmp/task31966-revalidation-KL6lMh` (`red.log`, `red-isolated.log`, `green.log`).

Final four-file finite-reader, activation coordinator, installed presentation
and reuse/mode run: **98 pass in145.00s**, exit0, no pytest warnings and no DB
files in all98 strict teardown observations (`affected.log`). New tests are
Ruff/format clean, changed production range is format clean, and all69 inherited
workspace Ruff diagnostics retain exactly the same normalized identity and
multiplicity. All11 artifact guards pass (`preflight.log`), UI census143/floor140
unchanged, whitespace clean. Existing ADR120/finite native-operation ownership
governs this routine correction; no new architectural policy. Fresh clean-head
full-activity resource comparison remains next; all qualification criteria stay
open rather than extrapolating fixture cleanup to production shutdown.

### Exact revalidation clean-head comparison — 2026-10-05

Frozen `de7479ccad5c43cad799f9001aa8e586ac04e96d` is rebuilt and measured
alone using a fresh 10,000-chat/250,000-eligible-message corpus, ready index,
valid quick-check and digest
`3460b66c2e7dcf74d00647d274cecaca5294bfe351665b2b7f56785e70afa5e6`.
All source/corpus guards hold. Standalone Keyword passes 300 exact queries,
P95 108.510209 ms, loop maximum 12.788625 ms, with no owned handles/descriptors
after cleanup. Raw root: `/tmp/task31966-revalidation-KL6lMh`, including
`host-before-measure.log`, `build.log`, `corpus/build-receipt.json`,
`keyword.log` and `keyword/keyword-receipt.json`.

The normal UI matrix retains every exact search/activation assertion: all
60 searches and eight OPENED outcomes complete without app exception. Exit1
still records six activation windows over50 ms (maximum79.397542 ms), plus
wide preparation57.331583 ms; all busy paints meet100 ms (maximum57.180375 ms).
Three registered handles sampled at unmount are not a full ownership census.
Retain `ui.log` and `ui/ui-evidence/ui-latency-evidence.json`; thresholds and
measured input/paint/readiness boundaries are unchanged.

The separate unchanged full-activity origin observer records585 unique
registrations, no drops and no untraced retained handle. After physical runner
join, six native-open handles/25 DB descriptors remain, with zero active
operations/acquisitions. No worker originates in production revalidation or
any earlier repaired finite reader. Five handles are main constructor caches.
The one exited-worker handle, token329, instead originates from the benchmark's
own `_keyword_evidence` navigation-service constructor authority capture. This
read can borrow a previously leaked worker cache and become visible only after
production owners retire; do not turn that shared-worker accident into cleanup
coverage. Its instrumented timing failures remain diagnostic only. Retain
`ui-owner.log` and `ui-owner-diagnostic/ui-owner-diagnostic.json`; observer script
and hash are unchanged from the preceding v2 receipt.

### Benchmark-owned Keyword evidence read — 2026-10-05

Both boot and post-preparation snapshots use one finite benchmark callback.
Installed exact-file operation ownership now begins before its authority-
capturing service construction and encloses the original materialized snapshot
transaction. No production code, status/count/revision/generation assertion,
maintenance contract, measured event/window, threshold, or global retirement
policy changes. Three genuine plain-worker REDs cover success, constructor SQL
failure and snapshot SQL failure; borrower-transaction, memory and subclass
controls pass before correction. No test adds an ownership wrapper.

Affected measurement/fixture verification passes **56 tests in 11.46s**, exit0,
no pytest warnings and no DB files in all 56 strict teardown observations. Both
changed Python files are Ruff/format clean. All eleven artifact guards pass
(`keyword-owner-preflight.log`), UI census 143/floor 140 unchanged. Independent
read-only review finds no actionable issue. Raw RED/GREEN logs are
`keyword-owner-red.log` and `keyword-owner-green.log` in the root above.
Fresh clean-head full-activity ownership remains next; this test-only correction
does not waive constructor retirement, latency, native, Windows or participants.

### Final finite-reader comparison — 2026-10-05

Frozen source `b5e0f37bb0cdea13c8b67d067246dab8cd63d4f8` is measured
alone against a fresh ready-index, quick-check-valid corpus of 10,000
conversations/250,000 eligible messages plus four excluded canaries. Counts are
10,000/250,004/10,000; digest
`ca0505616a8f631ea28643e0417cbdef78d3eb97e17ef9b175ac3c8a9ea84c52`.
Source/head/corpus guards hold. Standalone Keyword passes 300 exact measured
queries, no correctness failures, P95 103.089667 ms and loop maximum 5.883167 ms.
After corpus cleanup, registered handles and owned database descriptors are zero.
Raw root: `/tmp/task31966-keyword-owner-pkvILn`, including
`host-before-measure.log`, `build.log`, `corpus/build-receipt.json`,
`keyword.log` and `keyword/keyword-receipt.json`.

The uninstrumented UI matrix still fails, exit1. All 60 exact searches and eight
exact OPENED activations complete without app exception, with unchanged
source/corpus and 50 ms loop/100 ms busy-paint limits. Five activation loop
windows exceed 50 ms: narrow 65.683792/51.853708 ms; wide
70.857833/78.352583/76.840667 ms. Preparation passes at
40.201583/41.844083 ms; all busy paints pass (maximum 63.851458 ms).
Narrow/wide lists show four/eleven rows from 50 fetched. The reported one Chat
handle at run_test unmount is property-specific, not a whole-app resource census.
Keep `ui.log` and `ui/ui-evidence/ui-latency-evidence.json`; no causal latency
improvement is inferred from variation between runs.

Two independent fresh-process full-activity origin traces use the unchanged
pre-import v2 observer (SHA256
`7b3437928708b597648892b686fd5ba002b46c05842d41f2ca11892ac55a9402`).
Each records 636 unique registrations, zero drops, no untraced retained handle,
and exact clean source. At both Textual shutdown and physical default-executor
join, each has five native-open main-thread constructor connections, zero
worker-owned connections, active operations or pending acquisitions. Database
descriptor counts are 22 in the first process and 24 in the repeat, not an
identical-file-count plateau. The remaining connection origins are shared
Chat/config, evaluation orchestration, Library collections, Subscriptions and
Workspace construction; no repaired finite production or benchmark reader
retains a worker cache. After runner join only MainThread and the daemon storage
admission thread remain in the captured thread list.

Keep `ui-owner.log`, `ui-owner-diagnostic/ui-owner-diagnostic.json`,
`ui-owner-repeat.log` and `ui-owner-repeat/ui-owner-diagnostic.json`. These traces
exit1 for their instrumented timing failures and explicitly report
diagnostic-not-qualified. Their timings are not substituted for the normal
matrix. Retained constructor caches alone do not prove a leak or authorize
blanket close; application versus process ownership and producer-stop ordering
still need a bounded architectural decision. All TASK-31966 criteria remain
open, as do native, Windows and participant qualification. No fourth production
performance remedy, global GC/cache policy, evidence waiver or final PR.

### Approved Console allocation/refresh diagnostic — 2026-10-05

Frozen clean source `5c8c69babbc58024076dcc69148bb1c8acebc26d` contains the
approved diagnostic plan. All production, Tests, scripts and package inputs are
byte-identical to the preceding `b5e0f37bb0` normal/resource checkpoint. The
user approved this Console-only diagnostic, not a fourth production remedy or
global GC/cache/lifetime change. Existing ADR120 and ADR198 apply; no new ADR.
All runs use the existing synthetic offline real-profile/network/keyring guards
and run sequentially, not concurrently with other app probes or tests.

Raw root: `/tmp/task31966-console-allocation-z8neRS`. Fresh scale corpus counts
are 10,000/250,004/10,000, ready index and valid quick-check; digest
`7b986a63b08ca1f87f929647a20dc38b7e05c5a33332fd6423e66b9956b95258`.
Standalone Keyword passes all 300 exact measurements, P95 108.649 ms, maximum
loop interval 6.963625 ms, no correctness failures and zero owned database
descriptors/registered handles after cleanup. Build/Keyword source guards pass.
Keep `build.log`, `corpus/build-receipt.json`, `keyword.log` and
`keyword/keyword-receipt.json`. This is retrieval evidence, not UI latency.

The unchanged forwarding-tested config/native/GC observer reaches all 60 exact
searches and eight exact OPENED activations, with 70 retained matrix samples,
no app exception and clean exact source/unchanged corpus. Exit1 retains seven
failures: two narrow activations, wide preparation, one wide Keyword query in
the historical semantic-category oracle, and three wide activations. That
oracle label does not enable Meaning search. There are 28 paired outer
main-thread generation-2 collections and zero dropped GC/wait/config/native
records. GC thresholds stay 700/10/10 before and after; no forced collection.

Absolute interval comparison reconstructs each worst sentinel gap from raw
timestamps and matches the recorded maximum. It never adds inclusive nested
spans or subtracts diagnostic timing from a normal measurement:

| Activation | Worst loop gap (ms) | Gen-2 overlap (ms) | Bounded observation |
| --- | ---: | ---: | --- |
| 52x20 #1 | 87.656125 | none | Native layout 22.286292; enclosing timer 23.842250; remaining work unattributed by these wrappers |
| 52x20 #4 | 52.677000 | none | Native layout 19.901834; enclosing timer 21.317833 |
| 120x50 #1 | 40.586250 | none | Passing window; config callback 12.549750 contains summary 6.259833 and widget application 4.664083 |
| 120x50 #2 | 87.047792 | 32.518542 | Collection triggered inside native geometry/compositor reflow |
| 120x50 #3 | 120.749083 | 70.438375 | Collection triggered inside native CSS matching/widget mount |
| 120x50 #4 | 114.459958 | 53.983708 | Collection triggered inside native visual/render/style-cache work |

The trigger identifies the allocation site, not ownership of scanned or
reclaimed objects. Native/config span totals are inclusive, not exclusive CPU
costs. Whole-run config scope-entry ranges are not activation-only attribution.
Keep `ui-gc.log`, `ui-gc-diagnostic/ui-evidence/ui-latency-evidence.json`, its
`gc-caller-diagnostic.json`, `native-refresh-diagnostic.json`,
`config-phases-diagnostic.json` and `interval-analysis.json`. The latter SHA256
is `2bd8076aad0fffcdd198b3e571e3e0eef261b0d68b5a10ecf4c15ef195b1341f`.

Because the narrow no-GC gaps remained partly unattributed, one bounded follow-on
sampler observes code identifiers/line numbers only; it retains no frame/local
values, conversation text or credentials. It forwards original key dispatch,
summary and run_test arguments/results/errors, samples only activation windows
with a late diagnostic heartbeat, and stops/joins its thread at completion.
Fresh forwarding checks pass. It changes no production source, measured
readiness boundary or guard. This is an extension of the approved diagnostic,
not a production-fix plan.

The sampler run again reaches all 60 searches/eight OPENED outcomes, clean exact
source/unchanged corpus and no app exception. Exit1 retains seven failed
activation windows: narrow 69.286875/65.505417/50.238250/83.785625 ms and wide
90.037125/125.068458/100.143375 ms. The first wide activation passes at
46.990458 ms. It retains 163 samples/eight windows and 28 paired gen-2 events,
zero dropped records, stopped sampler and maximum capture cost 0.405542 ms.
Thresholds remain 700/10/10. Samples locate code while the heartbeat is late;
counts are not CPU shares. Sampling perturbs scheduling; GIL-held GC can prevent
sampling, and selector/wait samples are not proof of application CPU work.

In narrow #2, the 65.505417 ms worst gap has no gen-2 overlap. Samples include
checked storage/pinned-directory work, actual card SQL through
`_console_browser_character_labels_for`, the mandatory real-profile guard, and
the asyncio selector. Narrow #3 also has no gen-2 overlap and includes registry
active-workspace reconciliation plus checked path resolution. Narrow #1 shows
fresh provider-config posture and storage startup/admission work. Wide #2–4
overlap native Strip/compositor/style-cache rendering and 62.417250/71.498292/
71.083500 ms gen-2 spans. These are separate instrumented intervals, not timings
substituted for the preceding normal matrix. Keep `code-location.log`,
`ui-code-location/code-location-diagnostic.json`, its original UI matrix and
`sample-analysis.json` (SHA256
`2fa3f2fd1c7903f51d068a6b75998adbd64e8151faa5222d68eb9098a1b70322`).

The separate untimed heap census uses a newly prepared small source, counts
30/60/28, digest
`12eb7cd51c41522469f374d9da98873323e59684da870d03bfa832d50bf5194c`.
No native Terminal window is operated. Two saved resumes followed by three cold
and one warm exact Character activations succeed; session counts 4/5/6/6 prove
warm reuse in this headless journey only. App exception is None, corpus unchanged,
fixture exit0 and final source verification clean/exact. The diagnostic JSON
has no `source_clean_and_exact_after` field; the launcher enforces the final
source guard even when no standard receipt is present.

| Untimed snapshot | Unfrozen tracked | Frozen | Strip | FIFO / empty FIFO | Strip reached / not reached by partial widget traversal |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ready before resumes | 128842 | 493557 | 3446 | 24222 / 23404 | 3432 / 14 |
| After two saved resumes | 84323 | 620166 | 4304 | 30267 / 29269 | 3965 / 339 |
| After three cold + one warm | 113975 | 612651 | 5404 | 38039 / 36297 | 4465 / 939 |
| After unmount, app still owned | 102060 | 603991 | 4579 | 32264 / 30652 | 512 / 4067 |

Direct cache owners include mounted ChatScreen/layout containers, buttons,
provider transcript regions and Console session surfaces. Approximately 95–97%
of observed FIFO caches are empty; the installed Strip constructor creates seven
FIFO caches. Late boot freeze and reference-count disposal change the frozen
population, so these are not equivalent steady-state samples or leak/growth
proof. The partial traversal can include unreachable widget cycles, excludes
other global/compositor roots and cannot identify already-collected ownership.
Census strong references/allocations are untimed; retaining the app after unmount
is not terminal resource proof. Keep `small-build.log`, the prepared native
receipt, `heap.log` and `heap-diagnostic/heap-owner-diagnostic.json` (SHA256
`17e81277a941c7e219b1e93e3adc5bacc054e4dec49dee359e191d3d641a6ebc`).

Diagnostic script provenance (original bytes retained):

| Script | SHA256 |
| --- | --- |
| `/tmp/task31966-gc-caller-probe.py` | `ece43d63d6a4ee383cca27195355695bd892bb6352ac30c18a5bd0711390835b` |
| `/tmp/task31966-native-refresh-probe.py` | `50ee94d65f8f048177a5e72e63b6be81dd4ad9644fef97a78f06b011704d1197` |
| `/tmp/task31966-config-phases-v2.py` | `f3d7d56051ffc91cc86ea531b57bff9c43c94c9dbf8ded364851d9c0e7960362` |
| `/tmp/task31245-freeze-5uPHpo/diagnose_heap_owners_ed124.py` | `2bc0e22a0e6b3a3184fa92510bdaa7a0ed8b7210aca504ef909b38621eedfd77` |
| Raw-root `analyze_intervals.py` | `c63266fd483f5bfb7d1f2db6ba72f86daf004d6acb10b3e56c53bbdbd636b649` |
| Raw-root `sample_console_intervals.py` | `e25232940237cadfd97484ef1c8653016aecc17ceded4465f9c7519ed4522192` |
| Raw-root `analyze_samples.py` | `c9b32755a416c6394f0924f7afd3586340f50c318fe36d04c7e7c2628a691b65` |

All three new throwaway scripts are Ruff clean; interval pairing/source/head and
worst-gap reconstruction assertions pass. Original observer forwarding checks
also pass. No dependency, source UI, GC, cache, threshold or authority change.
The preceding uninstrumented `b5e0f37bb0` matrix remains the normal acceptance
receipt; no new normal matrix or full test sweep is claimed for this diagnostic.

Read-only caller tracing confirms candidate-only labels (two shared callers),
fresh provider settings with an existing synchronous derivation memo, and
immediate registry/session reconciliation on ordinary resume. Those contracts
must survive a correction. The next bounded design candidate is redundant
Console label/config derivation work, reusing installed checked-operation and
per-pass helpers; quantify required versus redundant reads before selecting it.
No scope may cross an await or replace fresh authority with stale cached state.
General rendering/heap work and constructor-lifetime policy are separate
architectural questions; this evidence does not authorize blanket GC/cache
changes. All TASK-31966 criteria remain open, along with native, Windows,
unfamiliar-participant and final app-owner retirement qualification. No final PR.
### Native authoritative-label fallback reads — 2026-10-05

The allocation/refresh diagnostic identified guarded card reads in native row
rebuilds. The native row already prefers the session's trimmed character name;
fetching that card's label cannot affect the displayed result. The shared native
builder now submits only missing/whitespace-name candidates to the existing
distinct-ID, checked fallback resolver. Persisted/workspace rows, exact native
identities, target revalidation, config/registry freshness and storage lifetime
remain unchanged. No cache, dependency, async boundary or GC policy was added.
Existing ADR120/ADR198 apply; no new ADR is needed for this redundant read removal.

Real file-backed SQLite plus the real ConsoleChatStore records actual card
SELECTs and asserts unchanged labels/identities. Valid isolated RED has three
count failures and one passing blank-name control; label/identity assertions
already pass. After correction, named sessions issue zero card SELECTs, mixed
named/fallback and deleted-fallback cases issue one, and two blank names issue
two. GREEN plus existing native/page/shared-resolver controls: seven pass in
3.41s. Both RED and GREEN use the strict read-only resource gate and retain no
database files after their test-owned owner retires.

Affected installed Character activation, measurement contracts and native/
workspace projection cases pass **53 tests in 79.52s**, process exit0, no pytest
warnings. All53 strict teardown observations contain no database files;
FILE-RETIREMENT-REQUIRED is True. The changed test range and production file are
format clean. All seven inherited test and 69 production Ruff diagnostics have
the same normalized identity/multiplicity; no diagnostic was added. All eleven
artifact guards and whitespace checks pass. Raw root:
`/tmp/task31966-native-label-8xIitV` (`red.log`, `green.log`, `affected.log`,
`preflight.log`). Independent read-only review finds no actionable issue. Its
seven-node test run also passes but encounters unrelated shared pytest
garbage-cleanup warnings; it does not substitute for the isolated warning-free
receipt above or establish broader merge readiness.

Fresh clean-head scale/UI measurements follow this commit. Eliminating unused
queries is not evidence that the unchanged 50ms activation/100ms busy limits
now pass, nor native/Windows/participant or application-cache retirement proof.
All TASK-31966 acceptance criteria remain open.

### Native-label correction: frozen-source requalification — 2026-10-05

Frozen `b738054edf475e2128b9b8d80b52d13a2e68492c` uses a fresh
10,000-conversation/250,000-eligible-message corpus plus four excluded canaries.
Ready index, counts10,000/250,004/10,000 and quick-check pass; corpus SHA256
`27437f2ea7de6c9e763e4d65a3b1f35fa45eaad4916fafddf97bb85107a3dafd`.
All300 standalone Keyword identities pass, P95105.448875ms, loop9.916375ms;
registered handles and owned DB files after cleanup are zero. Source fences hold.
Raw root is `/tmp/task31966-native-label-8xIitV`: `build.log`,
`corpus/build-receipt.json`, `keyword.log`, `keyword/keyword-receipt.json`.

The normal UI matrix still fails, exit1. All60 exact searches/eight OPENED
activations complete at52x20/120x50, no app exception and unchanged corpus/source.
Six activation loop windows exceed50ms: narrow62.180750/56.246083/61.527750ms;
wide62.025958/89.041708/78.595167ms. One wide body search also fails51.097ms.
Preparation39.248/37.417333ms and all busy paints (maximum34.058417ms) pass their
unchanged limits. Observer maximum46.2925ms is retained as an overhead limit,
not exclusive production cost; cross-run variation does not prove a causal
latency improvement. Four/eleven rows paint from50 fetched. One Chat handle at
run_test unmount is not whole-app retirement. Retain `ui.log` and
`ui/ui-evidence/ui-latency-evidence.json`; no failing sample is discarded.

Actual CUA macOS attempt uses only new Terminal window121584, prepared synthetic
source30conversations/60messages/28index documents (digest
`45855f5b92f67eefb1429df5463e4973dea8297465ade69181dd471e779dea12`).
Accessibility/Screen Recording are granted; foregrounding the dedicated QA
window was authorized. Native Ctrl+K shows CURRENT Indigo-07 versus highlighted
MRU Amber-07; native Return displays `NATIVE_MARKER_AMBER_07`, three tabs
unchanged. Actual cells remain244x73, not a required viewport. Evidence:
`native-07.jpg` and `native-08.jpg`; startup is `native-macos/native-startup.json`.
Host is macOS26.5.2 arm64, Terminal2.15, Python3.12.11/Textual8.2.8. SHA256:
launcher `1d897dabe88a1e35eb3fc69f54feb909eea96f03fa0574af38aeadcb51f3a65b`;
Active frame `3fc5481ae999c2d4c6b9b5558f116836a74862d938d3184e35c36e2ddf8260ac`;
resumed frame `6462539cdb1d489ca4c5689a33f94837467fd55e0945fc92d9811cab45bfcf41`.
The Inspector's initial snapshot is Not Applicable; actual Shell > Show
Inspector later reveals Columns244/Rows73. Its panel controls are absent from
the CLI AX tree. Pixel/AX-text attempts do not change size; text insertion is
acknowledged against a shell AXTextArea instead of the intended field. Do not
count these attempts as resizing or text-entry success.

The integrated computer-use selector then explicitly denies
`com.apple.Terminal` for safety reasons. Terminal UI operations stop immediately;
no alternate UI path follows that denial. The controller verifies only disposable
Python PID80035's full command and sends SIGINT. `ps` confirms it has stopped.
Its returned-not-qualified receipt has null return_code, unchanged corpus and
clean/exact final source fences, with26 DB descriptors at app return. This is
controller interruption, not normal native quit or retirement. Required native
viewports/workflows, font/zoom/remapping, Windows and actual participants remain
unqualified. Read-only app inventory finds installed VMware/Parallels, neither
running; that is not proof of an available prepared Windows qualification host.

The tested code is available for draft PR review; this receipt does not establish
merge/release readiness or waive the remaining TASK-31245/TASK-31966 gates.
