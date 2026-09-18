# Console capture gateway contract — TASK-32817

Console reads cached context-window metadata while composing its controls.
`CapturingGateway` predated that contract, so archive restore/resume/send tests
failed before they reached the workflow. The fixture now exposes the matching
synchronous cached and asynchronous resolution methods using the production
pure catalog/fallback resolver. It supplies no server metadata and performs no
network probe. Stream recording and production code are unchanged.

## Failure evidence and setup correction

[Saved source 2cc702b85c](../2026-09-18-console-footer-shutdown/archive-baseline-results.json)
failed both original archive variants. The [fresh red run](red-results.json)
reproduces both failures at saved efe900eb99; [first](red-001.txt) and
[second](red-002.txt) logs show missing cached_context_window.

The gateway repair reached the original workflow. The first variant then
[failed its pre-navigation draft assertion](fixture-race-red.txt), while the
workspace-collision variant passed. A direct native sync call can return after
coalescing a request onto an incumbent worker, before the store is updated.
Fixture setup now waits for composer/session ownership and the actual stored
draft through the existing bounded helper. It does not directly seed the store.
[All 32 original assertions in the archive module](archive-assertion-parity.json) remain unchanged,
including history, unrelated drafts, workspace identity and exact durable rows.

## Other direct consumer assumptions

The [initial thirteen-case run](initial-consumer-results.json) recorded nine
passes and four failures: the archive setup race above and three native-flow
consumers. Their original logs are retained individually.

- [Lifecycle](lifecycle-consumer-red.txt) and [workspace-switch](workspace-default-consumer-red.txt)
  fixtures set legacy provider display fields and selected the first session,
  but later sessions correctly read saved defaults and selected OpenAI.
  Both now use the existing persisted llama.cpp fixture helper before mount.
  All lifecycle, tab, history, workspace and persistence assertions remain.
  The lifecycle log alone did not prove why its tab was absent; the original
  tab assertion is retained in the final rerun.
- [Rail refresh](rail-consumer-red.txt) expected the complete `active session`
  phrase in a deliberately cell-truncated subtitle (`active sessi…`). The test
  still checks visible title, selection, the `active` state cue and exact
  matching row identity; the full status is asserted on that row's normalized
  state, alongside the existing age check at the same boundary. This does not
  claim that the full phrase paints in the narrow rail.

[Eight passing consumer functions and the gateway class](unchanged-passing-consumers.json)
are AST-identical between the initial run and the final source. The final subset selects both archive variants plus the three changed native-flow
consumers: four pass and lifecycle advances to a restored-screen pointer failure.
The [retained failure](final-subset-003.txt) places the tab at y=49 outside the
48-row screen. `RestoredConsoleHarness` lacked the application stylesheets loaded
by the initial `ConsoleHarness`; it now uses the same CSS_PATH. The [lifecycle
rerun](lifecycle-green-results.json) passes with its original pointer click,
draft, tab and transcript assertions. No scroll/click or assertion bypass was
added. This is test-harness stylesheet parity, with no production visual change.
The [only other restored-harness consumer](restored-adjacent-results.json) also
passes its real local-service conversation resume and recreation checks.

## Verification scope

The [selected cases](selected-cases.txt) include both archive variants and all
eleven direct CapturingGateway consumers in the native-flow module. Each case
runs serially in a fresh private profile through the previously recorded
[isolated runner](../2026-09-18-css-consolidation/run_consumer_cases.py).
The [final coverage ledger](qualified-cases.json) contains **14 distinct passing
cases**: eight unchanged consumers from the initial run, four from the corrected
subset, the final lifecycle rerun and the adjacent restored-harness consumer.
Later receipts supersede earlier failures by exact test ID; those failures remain
available above. [Final checks](final-verification.json) confirm the tested source
hashes, unchanged consumer ASTs and all archive-module assertions.
These are mounted tests with deterministic gateway responses and real SQLite;
they do not establish provider availability or a new native visual qualification.
No full suite was run. Existing screenshots retain their original source bounds.

[Lint comparison](lint-delta.json) retains 22 existing diagnostics and introduces
none in the large native-flow test file; the archive file is clean. All seven changed
ranges pass Ruff formatting and the authored diff passes whitespace checks.
[Independent review](independent-review.txt) found no actionable issue in the
gateway contract, bounded fixture-state waits, consumer corrections or restored
stylesheet parity. The root's subsequent adjacent test also passed.

ADR required: no. This changes test setup only, with no runtime, provider,
storage or UI contract change. Draft PR2707 remains unmerged; the broader
component/feature review remains active.
