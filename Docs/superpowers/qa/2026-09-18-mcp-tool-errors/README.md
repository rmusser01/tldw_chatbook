# MCP tool error propagation — TASK-32831

A local stdio server's `isError: true` result now reaches the existing failure
path. Test Tool shows the error and Audit records failure; a corrected invocation
can succeed on the same connection. Previously the transport dropped the flag,
so the real app displayed `OK` and recorded a successful invocation.

Base: merged dev `149acda36be8939fe8cd5e589bf77d13462257e7`.
Branch: `codex/mcp-tool-execution-review`, independent of draft PR2711–PR2714.
ADR required: no; existing ADR-111 and ADR-161 apply. No UI, CSS, token, permission,
storage, transport-interface or dependency change. The implementation follows
the client's pinned [2025-03-26 MCP tool-result contract](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2025-03-26/server/tools.mdx).

## Verification

- [Eleven distinct targeted cases pass](targeted-final.txt): eight isolated real
  stdio client/control-plane cases and three existing JSON-RPC client neighbors.
  [Inventory](test-results.json). Absent/false flags keep the prior success shape;
  true flags retain only nonblank text blocks, with a generic fallback for empty
  or nontext content. Tests verify error-body sentinels stay out of the execution
  log and application logging, and a second call succeeds without disconnecting.
- On unchanged dev, [six new cases fail and two pass](red-before.txt).
  A [real native baseline](baseline-false-success.json) independently shows false
  `OK` and `ok: true` audit metadata for the server error;
  [capture](baseline-false-success.svg), [clean shutdown](baseline-lifecycle.json).
- Two older control-plane cases fail at profile setup with
  `RecoveryRequired: raw_source_selection_changed` on both
  [current code](neighbors-current.txt) and [unchanged dev](neighbors-baseline.txt),
  before reaching tool execution. They are retained as baseline test debt, not
  counted as passing. No full-suite run was performed.
- [All seven derived-artifact guards pass](preflight.txt). New Python files pass
  Ruff and formatting; changed production ranges pass formatting.
  [Scoped Ruff comparison](static.json) shows the same 123 existing diagnostics,
  with none introduced. [Independent review](independent-review.md) found no
  actionable issue. [Pre-allocation ID census](id-census.json) checked 333 refs
  and 33 worktrees before assigning TASK-32831.

## Native and visual evidence

The [runner](native_check.py) uses real TldwCli, LinuxDriver, attached TTY streams,
private configuration/data, the real control plane and stdio transport, and a
[local protocol fixture](../../../../Tests/MCP/fixtures/stdio_tool_result_server.py).
Each invocation uses the normal one-shot Ask approval action. No service/client
operation is replaced. Two dark/light cells at 170×48 each fail, then succeed
using the same live subprocess. [Results and audit records](native-result.json),
[wire trace](fixture-trace.jsonl). Four rendered SVG captures were visually
inspected: complete failure details, the retry action, and subsequent `OK` are
visible. Raw-response rendering beyond its collapsed control is not qualified.

| Theme / size | Server-reported failure | Successful retry |
| --- | --- | --- |
| Dark / 170×48 | [Failed](textual-dark-170x48-failed.svg) | [Recovered](textual-dark-170x48-recovered.svg) |
| Light / 170×48 | [Failed](textual-light-170x48-failed.svg) | [Recovered](textual-light-170x48-recovered.svg) |

[Shutdown](native-lifecycle.json) confirms app return/exit 0, absent app and
fixture PIDs, released instance lock, ten healthy private databases, zero
conversations/messages, unchanged default config/UI/policy fingerprints, no
error/faulthandler output and matching production/fixture hashes. The recorded
runner hash also matches. [Export provenance](export-manifest.json) records
original and whitespace-normalized artifact hashes.

## Compact boundary and next review

The planned 80×24 journey failed before tool invocation: the focused argument
Input was absent from the compositor's visible widgets. Explicit scroll reveal
did not repair it. [First attempt](compact-attempt-001.json),
[second attempt](compact-attempt-002.json), [inspected capture](compact-unreachable.svg).
Both attempts returned the app cleanly and reaped their fixture process
([first lifecycle](compact-lifecycle-001.json), [second](compact-lifecycle-002.json));
their harness exit 1 records failed qualification.

The task's native scope was narrowed to wide execution based on this observation.
**Compact inspector reachability remains unresolved and is the next bounded UI
review.** This client-only repair does not qualify compact execution, complete
permission flows, remote transports, schema/raw argument recovery, diagnostic
actions or full Audit navigation. Current-head CI, accumulated review and owner
visual approval remain merge gates.
