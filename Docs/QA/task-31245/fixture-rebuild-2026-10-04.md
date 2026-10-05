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

Fresh clean-head scale and query-attribution measurements remain pending. This
does not close the 50ms/100ms limits or any native, Windows, participant or final
application-owner resource gap. TASK31966/TASK31245 remain In Progress with all
qualification acceptance criteria open. Existing ADR120/150/161/198 apply; no
new architectural decision or semantic gate waiver.
