# PR #3569 integration after CI feedback

The diagnostic-inventory correction was published as
`64e46a5584d189f27d7af88536a22a65a585dbf9`. While its predecessor's CI queued,
`dev` advanced to `cc6006fcf59328353004206bba8724c399216749`; conflicts then
prevented a new pull-request CI run. Both PR commits were rebased onto that
revision with autostash and shared rerere disabled. The published pre-resync head
is preserved as `codex/pr3569-before-ci-resync-64e46a5584`.

ADR required: no new ADR. Existing Library ownership contracts and
ADR-303/304/161 govern the product integration; ADR-103 governs CI capacity.

## Integration

The Library conflict combines retained Media filter/results ownership with
incoming empty-page focus recovery. Other incoming Library changes preserve
compact Notes layout, Media Find, Collections rail updates and focus retries.
Incoming Settings custom-endpoint recovery is separate from Speech and context
memory. Exact combined module/line ratchets follow the resulting source rather
than adding a discretionary allowance. Both sides' incident notes and capacity
task history are retained.

The [fresh CI model](ci/resync-cc6006fc/README.md) retains all 2145 incoming UI
selectors and the original 27 PR selectors. It uses the new upstream measurement
window, explicit recorded charges and 13.0-minute target. Forty shards at
concurrency 19 fit the model; deadlines and incoming lanes are preserved. This
is a projection, not a completed run of the new matrix.

## Focus findings and verification record

The first mounted Media/context integration batch ran 17 cases: **15 passed,
two failed**. All original Media/context cases and incoming Find, Collections
and wide-layout focus checks passed. Compact empty-page recovery failed to
reach Import in both its ordinary and queued-grip variants. Parent JUnit:
`d4c2a9ad7ad50ee2f3f1b36d1fde2fa73223f4104fa80a22e9910cce5971d9da`.

Review also found that checking a newer focus owner before calling a helper was
insufficient: that helper queued another callback before writing focus. Held
callbacks reproduce the overwrite of newer pending or settled focus. The
incoming helper has the same final-write weakness, but the new integration must
preserve focus across its additional results boundary.

The merged outer callback also read a nonexistent screen-level Find flag. The
canonical value belongs to `LibraryMediaState.find_focus_pending`. Its lightweight
test double supplied the nonexistent attribute and masked the owner error. A
two-case observational probe repeated that mistake and raised before its layout
snapshot; that run is retained as contaminated diagnostic evidence, not proof of
a geometry defect. No layout change is justified by that probe.

The corrected observation traced a targetless empty publication after Textual's
prune automatically parked focus in the retained filter. Implicit focus capture
then treated that parking as a requested filter restore and never admitted the
recovery helper. The correction skips that implicit capture only for a Reader
reset without an explicit target. Native filter edits still request the filter
explicitly. A targetless empty reset follows the incoming stranded-Media policy;
this is not a promise to preserve every prior deliberate focus inside Media.

Recovery captures the accepted result, entry generation and focus before results
completion, and rechecks canonical state both there and at the final focus write.
New pending entry/Find, a changed result or settled owner suppresses the stale
callback. The existing route, foreign-focus, pruning and focus-chain checks remain.
There is no layout change, retry, timer or new focus-tracking framework.

The final regression uses one real mounted screen for twelve held-callback
scenarios. An earlier unbound-test form would have increased the repository's
722-call coupling pin, so it was replaced; that pin remains unchanged. An isolated
negative control omitting the final predicate fails at the intended stale Import
write after a newer pending-entry owner. It is expected-failure evidence, not a
failed qualification run.

## Settings fixture findings

The initial 21-case Speech/Settings batch passed 19 cases: all eighteen Speech
cases and the incoming factory off-loop check. Two incoming Settings cases failed.
Its parent JUnit is
`d4c73ed356c128ef776e7bad5e8495e5f5a22f5f8e0b6b4d30718449c268a7ae`.

The endpoint case saved successfully, then queried a status child during section
recomposition and raised `NoMatches`. The bounded helper now catches that
exception only around its completion predicate. Three native attempts, each
four-second window, final assertion and endpoint security/persistence assertions
are unchanged. Exact-helper controls pass transient absence after two checks and
still fail permanent absence after three attempts (12.10 simulated seconds).

The supplied-state diagnostics case timed out at its original five-second wait.
One unchanged observational replay found its worker blocked on target-store
storage admission behind another initializer; target lookup and its caught
missing-profile exception occurred after the assertion. Profiling overhead means
this does not prove the original run's exact cause or a production deadlock. The
probe separately hit the existing executor teardown limit. TASK-1378.4 makes this
presentation fixture explicit: its fabricated server has no live context,
sync-repository or manual-sync services. Existing production fallbacks derive the
dry-run states; real runtime policy, worker, publication and rendering remain.
Configured-server storage is outside this test's assertions. The clean final
replay has no diagnostic plugin or executor teardown error.

## Final focused receipts

These runs use native Python 3.12.10 and Textual 8.2.8. Parent counts below do not
double-count private child executions. Root independently checked source hashes,
XML hashes/counts and zero failures/errors/skips for the listed clean runs.

| Run | Result | Parent JUnit SHA-256 |
| --- | --- | --- |
| Mounted ownership, unchanged coupling pin and exact size/no-slack guards, before test closure lint cleanup | 8 passed | `40427d737e9c12e58f826ff7df573117701d8f3405048fe25c026d24a19d5a26` |
| Original compact empty-page recovery controls | 2 passed | `2171395ff22053a1f4bf0ba0667fb7d650f12f08fce50aa7622c137a6ec8c76e` |
| Original endpoint recovery case | 1 passed; private child clean | `a2451a47ab2a7348e40d6524cc1904d681ae1a1d34bb1dec01ecfff4a88bac14` |
| Diagnostics presentation and source-contract companion | 2 passed; private child clean | `66802b8268e3b3a74962ddb5912c15cdd19e0e6be929c08f33997d57e1e5b8b7` |
| Final Media filter preservation, wide rail focus and Find | 7 passed | `b54b1381ab37e3bb0f398ebed685cf49b33d3acafc2e7409376b1c6e570946e8` |
| Final mounted ownership regression after explicit closure binding | 1 passed, twelve scenarios | `5d9d9b9a7567f810d54b445ab30295884285084f66887245a26d53d86b98df12` |

The final Library screen is 38,211 lines, SHA-256
`67a232b2cabc6477de2d977a3ab4ed000ea9b6a48f4f33dd42095856d78f70a4`;
Media controller is 5,320 lines,
`71c39c06edc62645fd0e58a7a136567153ddffa014884f9664009216e84aea4b`.
Removing only the changed sync method and grip import leaves the Library screen
AST equal to pinned dev; removing only the landing helper does the same for the
controller. Incoming Find/Collections and other focus code remain intact.

Independent final source/test review found no actionable product or test issue.
Fatal Ruff, AST compilation and scoped whitespace checks pass. Existing formatter
debt in the large Settings test file is unchanged; no unrelated rewrite was made.
Earlier failed, contaminated and pre-import runs remain historical evidence and
are not relabeled as successful final verification. No full local suite was run.

The ordinary diagnostic-inventory check passes on the final production source:
743 owners, 1,100 TASK-492 calls, 56 TASK-31551 calls, 7,731 TASK-494 calls and
16 sink files, with no drift. It took 267.58 seconds; source hashes were unchanged
before and after. The extra TASK-492 call relative to the earlier published fix
comes from the integrated dev tree. No inventory regeneration was needed here.

Scoped formatter checks pass for both changed production methods and both
Settings test edits. Full Library screen lint has the same 24 diagnostics as
HEAD, matched by code, message and source line; none were introduced here. The
new mounted test's five loop-capture lint findings are corrected with explicit
per-iteration callback bindings, followed by a focused replay.
That final test passes full Ruff and scoped formatting, and its JUnit duration
is 68.533 seconds. Its source SHA-256 is
`db39a65102abc45ec227b3bf4f1327373263e5ec1e791b4454d58a6179712af6`.
The final seven Media cases retained all 32 captured source/test/harness hashes.

The exact mounted regression selector runs at the Architecture lane's existing
bootstrap-profile tail, after terminal privacy checks. The final three ordering
and selector contracts pass (JUnit
`a76dc837cc43fd3287797521dadd27075b945b3bddd6ea129c93399bff223560`).
It adds no UI-census entry; the UI model remains 2,172 selectors, forty shards,
concurrency nineteen. The actual new remote matrix still needs observation.
