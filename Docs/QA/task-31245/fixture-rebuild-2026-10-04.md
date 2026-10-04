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
