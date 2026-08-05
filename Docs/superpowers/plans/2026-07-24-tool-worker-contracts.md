# Tool Worker Contracts Plan — Superseded

Date: 2026-07-24
Reconciled: 2026-07-27
Status: Superseded on current dev; do not execute

**ADR:** [ADR-031](../../../backlog/decisions/031-bounded-evaluation-and-tool-worker-execution.md)

**Current task:** [TASK-545](../../../backlog/tasks/task-545%20-%20Wire-built-in-tool-executor-into-MCP-permission-gate.md)

**Historical design:** [Tool worker contracts](../specs/2026-07-24-tool-worker-contracts-design.md)

## Reconciliation

This plan originally proposed adding concurrency, public cancellation,
terminal history, and explicit batch-child cleanup to `ToolExecutor`. It must
not be executed against current `dev`.

TASK-545 later verified that `ToolExecutor.execute_tool_calls()` and the
executor's other execution APIs had zero production callers. Current `dev`
therefore retired System A by deleting `ToolExecutor`, `ToolResultCache`,
`get_tool_executor()`, and `reload_tool_executor()` while retaining the
load-bearing `Tool` ABC and built-ins used by the live, permission-gated tool
provider.

The historical review correctly identified these defects in the deleted
implementation:

- the configured worker limit did not bound actual async tool execution;
- cancellation could leave non-terminal history;
- `return_exceptions=True` could turn child cancellation into batch data;
- parent or child control-flow failure needed explicit sibling cancellation
  and drain;
- reload owned a thread-pool lifecycle that never executed tools; and
- the MCP bridge could leak a coroutine when cross-thread submission failed
  before ownership transfer.

Deletion resolved the first five without adding lifecycle machinery to a dead
parallel system. The final MCP ownership defect applies to the live provider
and is the only implementation retained from this plan.

## Live Implementation and Verification

1. Keep System A's retired symbols absent and covered by
   `Tests/Tools/test_system_a_is_retired.py`.
2. In `tldw_chatbook/Agents/mcp_tool_provider.py`, retain ownership of the
   `execute_hub_tool()` coroutine until
   `asyncio.run_coroutine_threadsafe()` returns a future. Close the coroutine
   if submission raises before that transfer.
3. Verify the closed-loop path in
   `Tests/Agents/test_mcp_tool_provider.py` with unawaited-coroutine warnings
   promoted to errors.
4. Do not recreate the deleted ToolExecutor tests, callback API, public
   cancellation API, cache/history contracts, or batch execution machinery.

## Authoritative Successor

- [Retire System A design](../specs/2026-07-26-retire-system-a-design.md)
- [Retire System A implementation plan](2026-07-26-retire-system-a.md)
