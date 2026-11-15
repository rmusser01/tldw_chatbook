# Textual follow-up verification

Baseline: `ecad404bd4567b57139716a6deaa6b37fe833749`, fetched `origin/dev` on
2026-11-14. [Plan](../../plans/2026-11-14-textual-pattern-followups.md).
Verification used Python 3.12.10 and Textual 8.2.8 on Windows, with this
worktree supplying product code and the existing private-profile/bootstrap
guards enabled. Only targeted tests were run.

## Media filter — TASK-37360

The filter remains mounted while its results subtree recomposes. A native value
watcher claims a live edit before queued `Input.Changed` delivery; accepted older
results cannot replace that raw draft. Scope chrome has its own pending display
snapshot, so an earlier Reader-state update cannot conceal a needed scope update.
Zero results update the clean Reader without replacing the whole screen.

Mounted regressions first reproduced replaced Input identity, lost paced prefixes,
and a publication before Changed delivery that changed `P` to an empty string.
The final five new cases cover ordinary and paced typing, identity, focus,
caret/selection, accepted and superseded older results, actual worker-thread
drain, wide/compact empty results, backspace and settled Clear. The wide empty
Reader assertion checks compositor-painted placeholder text.

Final affected run: **7 passed**, including all five new cases and the two
existing applied-scope tests that exposed a scope-publication regression during
implementation. Independent scoped review approved the correction.

Broader targeted verification initially returned **52 passed / 5 failed**. All
five failures occurred before controls mounted because those legacy tests changed
the admitted config source. Retrying exactly those nodes with the existing
`bootstrap_profile` marker, without changing guards, exposed two scope regressions
(fixed and included in the final seven) and one early-Clear timing assertion.
The retry-Clear and type-selection cases passed. The remaining legacy
`test_clearing_the_applied_filter_also_clears_the_box` also fails against an isolated
archive of the exact baseline: it observes the emptied field before accepted
result publication, encountering the old scope or a temporarily absent scope
node. It was not weakened. New settled native Clear checks pass.

The 52 passing checks include Reader identity/dirty pinning, toolbar/layout,
scoped RAG choices, sort/pager/generations, entry/continue restoration, design
tokens, component patterns and generated CSS synchronization. CSS was rebuilt
from its source module. Ruff passes for the canvas, callback module and new
tests; formatting passes for the canvas and new tests. The screen's 24 existing
Ruff findings and inherited screen/callback formatting debt were compared with
the baseline and left unchanged.

| Run under `.test-tmp/textual-37360/` | Result | Parent JUnit SHA-256 |
| --- | --- | --- |
| `red` | 2 failed before repair | `3eaf75f4c4b486acb65062ca3bd50577809e4bcaeb1b24c1513697e7dd889c15` |
| `red-event-order` | 1 failed before synchronous admission | `751db259ff1f1356d87aff65251fe01d4c5a677b1d2f9c0e1d14c1bbcb93ba9d` |
| `targeted` | 52 passed, 5 bootstrap refusals | `002edfe9fb6bfc292e784d3288a7da9bb15d9d3f4b95aa34b52d9154954ffa34` |
| `baseline-three` | 2 passed, early-Clear timing failure | `856f0ffadc3e96a836fad1fbd4ac38b0f6366b945f98bf653539997867a31c1f` |
| `red-scope-owner` | 1 failed before scope-owner correction | `fa693c82acfeba1f1fb0c14e09eb6402dd48b10e3d6ef91dd111312d9ee368ff` |
| `green-scope-owner` | 7 passed, 0 failures/errors/skips | `c2647e6c510a040562ec1e077a8ab627da89ca3ca853b45edca4242c001f1abe` |

## Context-memory events — TASK-40969

The investigation confirmed three defects: a programmatic backing refresh added
all nine untouched context values to an unrelated draft; refresh could overwrite
a genuine native edit before its Changed event; semantically equivalent refresh
reformatted raw percentage input such as `80.` to `80`.

All four Inputs and five Selects now use the existing tracked-control seam.
Native message suppression prevents projection echoes, pending-value tracking
protects real edits, and parsed-value comparison retains equivalent raw numeric
text. Validation, percentage conversion and the existing staged-save owner remain
unchanged.

Final verification: **4 new regressions + 12 existing cases passed**, with
zero failures/errors/skips in both parent and private child reports. Coverage
includes repeated backing updates with an unrelated draft, all nine queued edits,
actual keyboard percentage edits, raw invalid text, focus/caret/selection,
validation, persisted normalized Save and confirmed Revert. Independent review
approved the change; source hashes were unchanged when compatibility finished.

Full Ruff/format checks pass for the new test, production F checks pass, and diff
whitespace checks pass. The screen's nine inherited Ruff findings and whole-file
formatting debt were compared against baseline rather than rewritten.

| Run under `.test-tmp/textual-40969/` | Result | Parent JUnit SHA-256 |
| --- | --- | --- |
| `red-context-01` | 3 failed before repair | `97318ec2ea512883e5705a2b0658f256cc43e9dc9be252058af8a7c54a1a3f66` |
| `green-context-01` | 4 passed | `950467ea118d641b6544fc19a475a25ecaf82129f6067e31ef351556fde7bd12` |
| `compatibility-context-01` | 12 passed | `f22ee00b4808ced4090ba3c23f1776969ab2265adcbebb35310bad0fffbed1ac` |

## Speech category ownership — TASK-1378.3

The existing registered Speech panel now owns rendering dispatch, dirty/banner
and help policy, bounded provider navigation and category/destination leave
decisions. Runtime/model-library workers, leases, persistence and global-versus-
Studio ownership remain with their existing owners. Parent TASK-1378 remains open
for the remaining Settings categories.

One safe snapshot supplies the existing dirty predicate and field count. They
remain distinct: a cleanup-only proposal can be dirty with zero counted fields.
Ordinary repaint does not collect live controls. Save/Revert/navigation capture
pending controls synchronously; owner identity and the existing complete draft
revision reject stale messages after newer capture, discard or replacement.

Baseline mounted tests reproduced queued messages resurrecting the cache after
Discard or successful Save-and-leave, and an older projection replacing a newly
captured pending edit. A separate RED showed direct widget Save reading the new
field while the shell retained the old snapshot.

The **nine new regressions plus existing category Cancel passed**. Existing
compatibility checks also exposed a migration regression: sanitized retained
snapshots lack live credential/source metadata, so using them as constructor
metadata caused `KeyError('openai')` on dirty same-category rebuild. The correction
uses fresh configuration for metadata through the existing constructor restore
path, while preserving snapshot edits and their original saved baseline. Both
deep-link and search cancellation/rebuild cases pass after the correction;
independent scoped review approved it.

Audio FIFO bypass checks pass in both orders with the supported-UI capability
fixture. Native Windows lacks the scanner's required `O_NOFOLLOW` capability;
the unchanged scanner refuses the managed-package setup before UI behavior.
Cleanup is therefore separately qualified with a validated accepted-package
input and the **unchanged inherited `preack_overlap` scenario**, using the real
handoff/cleanup/rollback owners. That check passes: pending cleanup disables
mutations, blocks both panel and screen leave, preserves the overlapping edit,
requeues the handoff and eventually releases the fence. This is UI orchestration
evidence, not native Windows scanner, filesystem-security or runtime support.

Nine scoped architecture checks pass. New-test and registry Ruff checks,
new-test formatting, changed-module compilation and diff whitespace checks pass.
Full affected-file Ruff comparison found no new diagnostics: Settings 9→9,
registry 0→0, widget 1→1, snapshot module 1→1. Inherited whole-file formatting
debt was not rewritten.

An inherited focused Save/Revert click test passed its first compatibility run,
then failed an immediate mock-call assertion. An unchanged replay passed. One
instrumented run showed the same live owner and successful Save/Pressed delivery,
then an out-of-bounds Revert target after scrolling. The earlier zero-call cause
is not fully attributed. The test now waits for native scroll/animation completion
and actual handler acknowledgment, while asserting current owner, field focus,
enabled visible target, hit-test identity and the original exact-once calls.
It does not retry clicks or change production timing. The final original test
passes; its parent and private child hashes were independently verified.

| Run under `.test-tmp/textual-1378-3/` | Result | Parent JUnit SHA-256 |
| --- | --- | --- |
| `final-regressions-01` | 10 passed | `4f2bbeeee2e1d02f594b59f46eb5d28ff56a67a6044d7ba2eeb4ac94bce17078` |
| `fix1-navigation` | 8 passed; 1 legacy focused-click failure described above | `d5570f0e2789ea45aab6fad4f952334b0be4ee7d6c59e8271b737a91cd8e7bb0` |
| `focused-click-synchronized` | 1 passed after completion-based test synchronization | `e6e956cbe4be3a8a9da7b0591f3c1db7fbb182899cf575748b5333000744ec38` |
| `cleanup-ui-boundary` | 1 passed | `2ab55e6e5d02b81e3db2182d5792dc76bd07f070fb7855a987281f49978bc69d` |
| `architecture-green` | 9 passed | `8ed9706fe39381b95c1739c8a197e12d863c96673059c37f4db5bb9d8d824fc9` |

## PR gate routing

The gate now includes 27 verified cases (5 Media, 4 context-memory, 18 Speech),
with 2061 unique selectors and the matching census floor. It retains every
inherited selector. The smallest modeled count meeting the existing 13.3-minute
pytest target is 38 shards; maximum UI concurrency remains 18, introducing a
third scheduling wave. Recorded timings, explicit reserve/calibration assumptions,
34-swap replay and reproduction commands are in the [CI report](ci/README.md).

The 104 existing checker/replay tests and nine final count/workflow guards pass.
The coordinator independently verified receipt hashes, case counts and unchanged
hashes for all 13 CI files. The new shard order has **not** run in an actual PR;
that post-change observation remains part of TASK-39144. No workflow was
dispatched, and nothing was pushed.

## Evidence boundaries

Independent task reviews and the [final whole-branch review](final-review.md)
approve the final implementation with no actionable findings. [Scope and
verification decisions](decisions.md) record the tradeoffs and retained limits.

These are mounted Textual tests with production styles and controlled services;
they do not measure natural race frequency or live remote-service performance.
No full-suite or real-terminal manual-verification claim is made. The existing
pytest-asyncio, Pydantic and audioop warnings remain disclosed in raw run logs.
Private runner receipts, commands and JUnit files are retained in the task scratch
directories named above. The final report does not relabel intermediate failed
runs as green.
