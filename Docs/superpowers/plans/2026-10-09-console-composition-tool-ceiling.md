# Establish Console tool definitions at composition

TASK34601 AC21; OPT98. Root owns integration and every native run.

ADR required: yes.
ADR path: backlog/decisions/225-console-send-preparation-and-io-ownership.md.
Reason: move stock MCP definition/eligibility observation from receipt to the
existing execution consumer's composition; revise earlier attempt-wide ceiling.

## Design and boundaries

The app already shares controller/provider preparation and retains invocation
gates. The remaining duplicate stage is initial capture's fresh policy/catalog/
builtin scan followed by another fresh composition. Remove that initial stage
for qualified stock owned capture, rather than add a cross-attempt cache.

Add validated `mcp_definition_capture` with literal values `captured` (default)
and `composition` to the detached configuration and execution-context facade.
Only the explicit composition mode means deferred: IDs must be None and hashes
empty. Legacy None and captured empty keep their current meanings. The common
synchronous capture remains unchanged; owned capture supplies an explicit empty
maximum to suppress it, then replaces the frozen result with the selected mode.
An ephemeral session remains captured-empty. Unknown/custom entry routes retain
their initial capture; a later change to an unsupported custom route refuses the
deferred composition rather than treating it as an unbounded custom request.

Existing shared `prepare_console_tools` observes fresh policy/catalog/inventory
once. Both its adoption and the ordinary stock fallback issue the concrete IDs/
hashes at the provider's successful source-current catalog publication. A private
default-off freeze-on-first-compose flag distinguishes this from legacy callers.
Successful empty/killed results freeze empty; failure/cancellation before successful
publication must not latch partial data. Recomposition of that provider can only
narrow against its issued IDs/hashes. Actual invocation remains freshly gated.

The stock ceiling is now per existing execution consumer's composition lifetime:
the hook executor and live agent run each compose under current policy; disposable
previews remain separate and cannot publish live counts or execution authority.
Reconstruction before a new execution consumer is a fresh preparation, not a
refresh of an already dispatched provider. No existing live-run path should
replace its provider halfway through execution; verify this in source review.
Do not introduce a mutable attempt ceiling registry or retain native leases.
Plugin snapshots retain their separate captured narrowing. Saved acceptance,
draft failure handling, hook effect checks and recovery order remain unchanged.

## Parallel ownership and review

1. Shared lane owns only Chat/console_turn_context.py and
   Chat/console_configuration_preparation.py. It prepares the explicit mode,
   validation/forwarding and qualified owned-capture omission outside the worktree.
2. Integration lane owns only Chat/console_chat_controller.py and
   Agents/mcp_tool_provider.py. It prepares stock qualification, live/hook/preview
   propagation, common successful freeze, empty/killed and custom-drift refusal.
   No new preparation class, catalog cache, I/O owner or policy cache.
3. Baseline lane owns new Tests/Chat/test_console_composition_tool_ceiling.py.
   Reuse real existing profile/stock-capture fixtures and original-body observers;
   prove no initial policy/catalog work and one fresh demanded composition, actual
   saved output/retirement, and add boundary tests. No product substitutions that
   bypass the work counted. Other test migrations belong to root after inspection.
4. All lanes are source/static only. Root reviews agreed API and plan first,
   runs original read-count regression, integrates both implementations, then
   runs targeted capture/shared/fallback/empty/plugin/current-source/cancellation/
   invocation and required saved-draft controls. Preserve original deadlines.
5. Boundary tests cover permission/catalog additions before composition; denied,
   redefined and newly added tools after provider publication; empty/killed
   first results; explicit legacy/captured empty; malformed mode; ephemeral and
   no service; custom at entry and drift after capture; source replacement and
   repeated cancellation; preview isolation and fresh actual invocation.
6. Only after both implementations and final tests are ready, root runs sequential
   quiet full-default baseline/candidate/candidate/baseline comparisons. Freeze
   source/HEAD, guard overlap, prove all turns persisted/settled, record raw cold/
   warm timings and limits. Do not infer physical feedback or p95 from headless
   traces. Retain only a justified simplification; record all rejected options.

## Review requirements

Resolve provider reconstruction and prospective-hook lifetime explicitly; do not
claim receipt-time eligibility is preserved. Check supported custom signatures,
plugin constraints and final publication after source checks. No initial-catalog
cache or metadata-only execution schema. Root/independent review precedes code.
Overall under-one-second application and100ms actual-feedback targets stay open.


## Final qualification and retained result

Retained after root integration and independent source review. The original
read-count control failed causally at one permission/catalog read versus zero;
the earlier first RED only exposed a temporary-session fixture assumption and
is not product evidence. The final boundary run passes 22 composition cases plus
three existing durable-ordering, postcommit-error and draft-recovery controls.
The broader integration run passed 231 of 234 cases. Its three old initial-MCP
expectations were migrated explicitly and pass: temporary capture is empty;
the detached-view lifetime case now holds the real retained Workspace SQL reader,
preserving the four-second entry deadline, live native lease, off-loop execution,
GC, and physical connection/lease retirement. Total:259 distinct qualified cases.
No full suite was run. Final review's deferred no-service hook refusal is fixed
and covered with the original real hook lifecycle and current-task reservation.

Compilation and new/changed-line lint/format checks pass. Existing controller
lint remains60 diagnostics; unrelated whole-file formatting debt remains in the
controller, provider and turn context. The new tests and migrated tests format
cleanly. One provider assignment was formatted before the qualified timing
cohort; no semantic changes followed the boundary checks.

Quiet full-default-profile ABBA, baseline b5f9bc4c8f versus candidate on
7a4a82206c plus the four frozen product files:

| Run, in execution order | Cold / warm / warm Send-to-adapter seconds |
| --- | --- |
| composition-ceiling-a3 | 4.916454 / 3.514851 / 2.627913 |
| composition-ceiling-b1 | 3.848684 / 2.690795 / 2.666949 |
| composition-ceiling-b2 | 4.040550 / 2.670157 / 2.842759 |
| composition-ceiling-a2 | 4.853725 / 3.259906 / 2.722104 |

Cold mean:4.885089 to3.944617s.
Warm mean:3.031194 to2.717665s
(10.34% lower in this small sample).
All12 turns have three saved user/assistant pairs and linked complete traces per
process, zero pending checkpoints, nonstreamed/nonstreamed/streamed dispatch,
unchanged source/HEAD and no detected overlap. Loaded module membership matches;
only the intended four product files differ after newline normalization.
The initial a1 receipt passed functionally but is excluded from timing because
another chat's Python evidence-reading process tripped the overlap guard. It was
replaced by a3 before the two candidate runs; no sample was replaced for speed.

The removed stage is demonstrated by original native bodies: initial policy and
catalog reads1/1 to0/0; demanded shared composition retains1/1. Actual invocation
remains freshly gated. This is a small sequential headless adapter-stubbed sample,
not a percentile, physical-terminal-feedback or under-one-second acceptance claim.
Candidate warm saved-to-trace still takes1.133-1.562s. The task remains In Progress.

Evidence directory:C:/Users/GDesktop-1/.codex/visualizations/2026/10/06/01a10fff-d063-7810-bebc-65f55409bcec/claude-watch-final-gate-review/
Receipts:composition-ceiling-{red-2,green-1,integrated-1,boundaries-2,migration-1,
a1,a3,b1,b2,a2}; summaries:composition-ceiling-{comparison,test-summary,
static-final}.json. The comparison records exact receipt and loaded-source hashes.
The native CI workflow now includes the new boundary file with unchanged budgets.
No current-head remote CI result is asserted.


## Committed-source feedback and remaining-cost check

On24d3d773fd, both existing original held-reader Enter/button feedback controls
pass with unchanged source. Natural Preparing frames appear at34.671/41.205ms;
input mutation1.925/2.210ms and input frames26.718/9.178ms. This brings the targeted
set to261 distinct cases. These supplied headless compositor frames do not prove
physical-terminal flush or a percentile bound. Receipt:composition-ceiling-feedback-1.

The separate existing DetailSpans diagnostic on the same commit completes three
saved replies/traces with stable sources, no detected overlap, current original
bindings and retired monitoring. Warm saved-to-trace takes2.140/1.828s under this
instrumentation. Inclusive original boundaries are tool composition .537/.455s,
final hook admission .229/.274s, history .111/.221s, initial chain maintenance
.174/.249s, dispatch checkpoint .151/.136s and run-log bind .164/.064s. These are
instrumented, nested/concurrent wall intervals, not additive savings or the quiet
comparison. No additional duplicate whole-stage call is established; the source
audit retains distinct history, final authority and run-start ordering.

The observer has zero row overflow/unmatched returns; seven ordinary starts lack
returns because exception unwinds are not observed. Hook/raw ancestry has92 depth
misses, while context reads and generator-entry maps have no recorded gaps. Do
not describe this as complete native attribution. Exact spans, original source
hashes, and the derived post-save report are under
composition-ceiling-postsave-detail-1/{run,probe,spans,postsave-summary}.json.
No product optimization follows from this diagnostic. Further contraction of
these owners requires a justified lifecycle/freshness decision; OPT96/99 retain
the unimplemented longer-lived connection/definition-owner alternatives. Current
remote checks show only CodeRabbit; native CI at this head remains unqualified.
