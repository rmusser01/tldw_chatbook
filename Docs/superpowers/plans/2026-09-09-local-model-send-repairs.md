# Local-model send repairs

**Goal:** Restore captured llama.cpp tool runs and stop repeated failed searches.

**Architecture:** Reuse the canonical fenced-tool parser, the existing LocalToolError/ToolResult failure boundary, and the per-run repeated-call threshold. Preserve trace ownership and permission checks; no database reset or new dependency.

**Tech stack:** Python, Textual Console controller, SQLite, pytest, httpx.

The user authorized the diagnosed repairs with “fix it” after reviewing the evidence in this session. Existing planning-provenance design/plan documents are reference material. Implementation is in the current workspace so the editable installation receives the repair; the separate pre-existing planning worktree is untouched.

## Chunk 1: Captured agent tool calls — TASK-32188

- Reuse the existing worktree's candidate classifier/test changes, correct the fixture API and use a saved user revision.
- Prove failure on visible planning text before a tool fence and misclassification of malformed look-alike fences.
- Apply the shared parser; preserve native calls, ordinary text, exact provider input and provenance validation.
- Extend real-controller discovery coverage across capture on/off, project context on/off, cold factories and following sends.
- Investigate old conversation failures using copied databases and deterministic fixtures before any history repair.

ADR required: no new ADR.
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md
Reason: classifier alignment implements the existing exact semantic-provenance contract. Reassess if old-history recovery needs a new proof contract.

## Chunk 2: Honest search failures — TASK-32189

- Reproduce DuckDuckGo challenge pages, HTTP errors, missing parser, backend error envelopes and malformed responses.
- Detect failures at the backend/tool boundary and raise existing LocalToolError so Console and external local-tool consumers receive failed results.
- Keep valid empty results separate and preserve successful-result caching.
- Tell the agent to stop repeating unavailable searches and explain the configuration requirement.

ADR required: no new ADR.
ADR path: backlog/decisions/078-structured-agent-tool-outcome-provenance.md; backlog/decisions/032-local-agent-tool-permission-boundary.md
Reason: repair existing ordinary-failure reporting without changing permission authority or wire schema.

## Chunk 3: Repeated failing calls — TASK-32190

- Reproduce repeated failures with different arguments in the real pure loop.
- Stop after three consecutive ordinary failures of the same tool; reset on success or tool switch.
- Keep final result recording, native batch coherence, permission outcomes and per-run isolation.
- Verify failure, recovery and independent-run controls.

ADR required: no new ADR.
ADR path: backlog/decisions/078-structured-agent-tool-outcome-provenance.md
Reason: bound repeated execution failures using existing outcome facts and loop termination mechanics.

## Chunk 4: Existing failed tool-run history — TASK-32191

- Preserve the four real-controller failures for failed empty replies, with project instructions and uncaptured follow-ups on/off.
- Extend the existing bounded closure proof to durable failed empty replies and exact saved uncaptured turns; keep historical calls immutable.
- Require exact revision and parent ownership, closed response state, matching policy, no active checkpoint or intervening captured calls, and a bounded suffix.
- Revalidate at final binding; reject partial responses, sidecars, ambiguous or changed history, active delivery and over-limit chains.
- Verify cold reload, subsequent sends and existing explicit-discard behavior.

ADR required: yes, amendment to existing ADR097 before implementation.
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md
Reason: extend the durable closure evidence accepted by the existing tool-suffix proof without adding storage or inventing captures for uncaptured turns.

## Validation and handoff

Target the touched request/trace, search/provider and agent runtime modules. Check changed-range formatting, introduced lint diagnostics, derived inventories and an independent review. Run a bounded real local-model check with disposable databases if the server is reachable. Do not run the full suite. Document credential/challenge limitations and any unresolved historical failure separately from repaired behavior.

## Verification evidence

- The canonical-parser regression failed before the classifier repair; 21 targeted cases passed afterward. Independent review also checked native calls and the controller matrix (22 passed).
- Search failures, changing-argument retries, challenge ordering, and genuine empty Searx/Yandex searches each reproduced before their fixes. Combined search, provider, runtime, continuation and local-MCP verification: 637 passed, three opt-in live skips, one unrelated filesystem-read ledger failure. That failure also reproduces with the original web-tool module; its provider/ledger implementations are unchanged.
- A broader initial trace run passed 456 cases with two unrelated live-versus-resumed display/redaction failures. Both also fail with the original bridge/runtime modules. The final capture regression selection excludes those two test functions, with the exclusion recorded rather than changing their expectations.
- Final combined bridge, prepared request, provenance, project context, trace service/runtime and witness verification: 611 passed, three cases deselected by the two documented baseline test-function exclusions.
- Settings budget verification passed 27 cases; the existing save-click integration failure also reproduces with the original Settings module.
- Real-controller failed-history, explicit-Discard and rendered-system verification passed 95 cases. Four additional real-controller cases prove that three failed searches produce a stopped run and the next captured send succeeds, with project context on/off and warm/cold history.
- Final standalone recovery module: all 39 cases passed, including the stopped-search integration, immutable original captures, mixed discarded/uncaptured history and final-binding rejection controls.
- Two bounded live llama.cpp checks passed using disposable databases: captured greeting, calculator result 323, and captured follow-up. No live profile or credentials were changed.
- Independent reviews accepted all four repairs. The Searx empty-result issue found in review was corrected and rechecked.
- Changed Python files parse successfully; Ruff reports no introduced diagnostics (five inherited findings). Changed ranges and new files are formatted. Diff whitespace checks pass.
- Derived styles, profile-owned paths, diagnostic inventory, Backlog IDs, schema allowlist and index-plan pins pass. The diagnostic inventory change is one reviewed removal: missing lxml now raises instead of logging and returning an empty search. The unrelated Mermaid reproducibility check requires its pinned Python 3.12.11; available interpreters are 3.14.7 and 3.12.14, so that check remains unverified. No Canvas assets changed.

The full suite was not run. Restart the application to load the local repair.
For an existing pending response, use its normal Discard recovery before a new
captured send. Searches still require an available backend and any credentials
that backend needs; the repair reports those failures and bounds retries.


## PR scope and verification

ADR required: no additional ADR.
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md; backlog/decisions/078-structured-agent-tool-outcome-provenance.md; backlog/decisions/032-local-agent-tool-permission-boundary.md
Reason: this proposal contains the four capture and failed-tool-loop repairs
above, including the error classification required for unavailable searches to
reach the loop guard. Genuine empty search results remain successful.

The proposal is based on upstream dev `86a8054ed`. Only TASK32188–32191 are
included. Their plans and acceptance criteria define the implementation scope.
Final verification targets capture recovery, failed outcomes and retry handling.

Fresh verification of this final proposal: **425 passed, 3 opt-in
live-network skips**; 313 unrelated cases were deselected in the search-only
selection. Commands:

```bash
python -m pytest -q \
  Tests/Agents/test_agent_runtime.py \
  Tests/Agents/test_web_search_failure_run.py \
  Tests/Chat/test_console_trace_failed_tool_run.py \
  Tests/Chat/test_console_trace_discarded_tool_run.py \
  Tests/Chat/test_console_trace_final_values.py \
  Tests/Chat/test_console_trace_runtime.py \
  Tests/Chat/test_console_trace_service.py \
  Tests/Web_Scraping/test_search_backends.py --tb=short
# 388 passed, 3 skipped

python -m pytest -q Tests/Tools/test_web_tool_impls.py \
  Tests/Agents/test_local_tool_provider.py -k search --tb=short
# 37 passed, 313 deselected
```

All 17 changed/new Python files parse. Ruff comparison against the upstream
base confirms no introduced diagnostics. Diagnostic inventory, task-ID and
whitespace checks pass. Independent review confirmed that the retained loop
fix has no dependency on the removed changes. The earlier exploratory baseline
failures and live-service limitations above remain disclosed; no full-suite
run was performed.
