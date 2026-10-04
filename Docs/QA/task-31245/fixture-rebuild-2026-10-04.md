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
