# MCP connected catalog refresh — TASK-32830

Historical qualification follows. Current-dev integration and qualification are
recorded in [current-dev](current-dev/README.md); the obsolete executable runner
is retained only through its immutable source link.

Refresh tools now reconnects and discovers the server's current tools, resources
and prompts. Previously a connected profile returned its cached catalog and
reported success without any discovery request. A connected profile stays
connected; a disconnected profile returns to its disconnected state after discovery.
Observe and launch gates remain in force before replacing an existing connection.

Base: merged dev `149acda36be8939fe8cd5e589bf77d13462257e7`.
Branch: `codex/mcp-connection-recovery-review`; separate draft PR against dev.
The inspector, server-action and cancellation changes in draft PR2711, PR2712 and
PR2713 are not included. ADR required: no; existing ADR-161 and ADR-111 apply.
No new transport API, policy grant, persistence format, token or CSS change.

## Verification

- **15 distinct targeted cases pass**: nine new isolated real-stdio cases and six
  adjacent service/client cases. [Inventory](test-results.json),
  [service run](targeted-service.txt), [atomic-fixture rerun](targeted-atomic-fixture.txt).
  Coverage includes changed catalogs, both original connection states, observe/launch
  denials, failed reconnect/retry, persistence failure cleanup, and another caller's
  pending connection surviving either a busy rejection or launch denial.
- [Four failing dev cases](red-discovery.txt) reproduce the stale catalog and missing
  launch check, failed reconnect expectation and leaked temporary session on save
  failure. The original disconnected and observe-denial cases already passed.
- Independent review caught cleanup of another caller's pending connection in the
  first repair. [Two failing regressions](red-pending-ownership.txt) reproduce it.
  Failed post-connect discovery/persistence now cleans up only its established
  session identity; refresh disconnects temporary sessions only after success.
  [Review record](independent-review.md).
- [All seven preflight guards pass](preflight.txt). The first sandbox attempt could
  not fetch pinned Mermaid inputs; that [failed attempt](preflight-initial-network-failure.txt)
  is retained. Repeating with authorized network access verified the pinned inputs
  and all six generated Mermaid outputs. [Ruff comparison](static.json) has no
  introduced diagnostics (five baseline diagnostics in the existing service file).
  Changed production ranges and new tests/runner pass formatting; new files pass lint.
- The [ID census](id-census.json) checks 327 refs and 31 worktrees before saving.
  No full repository test sweep was run.

## Native and visual evidence

The [original runner](https://github.com/rmusser01/tldw_chatbook/blob/970f568c2fab5266ccd6c6ac8b403fff617d136e/Docs/superpowers/qa/2026-09-18-mcp-connection-refresh/native_check.py) drives real TldwCli, LinuxDriver and attached TTY
streams in a fresh private profile. It uses the real service, store, client and
stdio subprocess transport with the [local JSON-RPC fixture](../../../../Tests/MCP/fixtures/stdio_catalog_server.py).
The fixture changes all three catalog sections and deliberately refuses initialization
for the failure phase. There is no substituted client/service operation and no
external network or tool execution in this journey.

[Four passing cells](native-result.json) cover dark/light at 80×24 and 170×48.
Each connects, refreshes the changed catalog, fails a second refresh, recovers with
Refresh tools, and completes another Connect/Disconnect cycle. [Wire trace](fixture-trace.jsonl)
records 20 initialize attempts and 16 complete tools/resources/prompts discoveries.
All sixteen SVG captures were rendered with Quick Look and visually inspected:
Refresh tools remains visible after failure, recovered disconnected profiles show
Connect, and wide captures show original → updated → recovered catalog names.

Compact captures qualify the inspector's lifecycle action and status. Long command
text places catalog names below the fold; compact catalog scrolling is not qualified
here. The compact detail toolbar retains the clipping repaired separately by PR2712.
This service repair does not claim complete MCP destination, remote transport,
connected tool execution, cancellation, permission-editor or source-switch coverage.

[Shutdown receipt](native-lifecycle.json) verifies returned app/exit 0, absent app
and all 20 fixture PIDs, released instance lock, ten healthy private databases,
zero conversations/messages, unchanged default config/UI/policy fingerprints,
no error/faulthandler output, and matching production/fixture/runner hashes.
[Export hashes](export-manifest.json) preserve original provenance through trailing
whitespace normalization.

The [first native attempt](native-initial-selector-failure.json) stopped after
successful disconnected recovery because the harness expected Refresh tools rather
than the correct Connect action. [Its exact app/child PIDs exited](native-initial-exit.json).
The final fresh profile uses the correct action and waits for app notifications to
expire before capturing unobscured states.

| Theme / size | Connected | Fresh catalog | Failed refresh | Recovered |
| --- | --- | --- | --- | --- |
| textual-dark / 80x24 | [Connected](textual-dark-80x24-connected.svg) | [Refreshed](textual-dark-80x24-refreshed.svg) | [Retry available](textual-dark-80x24-failed-retry-available.svg) | [Connect available](textual-dark-80x24-recovered.svg) |
| textual-dark / 170x48 | [Connected](textual-dark-170x48-connected.svg) | [Refreshed](textual-dark-170x48-refreshed.svg) | [Retry available](textual-dark-170x48-failed-retry-available.svg) | [Connect available](textual-dark-170x48-recovered.svg) |
| textual-light / 80x24 | [Connected](textual-light-80x24-connected.svg) | [Refreshed](textual-light-80x24-refreshed.svg) | [Retry available](textual-light-80x24-failed-retry-available.svg) | [Connect available](textual-light-80x24-recovered.svg) |
| textual-light / 170x48 | [Connected](textual-light-170x48-connected.svg) | [Refreshed](textual-light-170x48-refreshed.svg) | [Retry available](textual-light-170x48-failed-retry-available.svg) | [Connect available](textual-light-170x48-recovered.svg) |

Current-head CI, accumulated remote review and owner visual approval remain merge gates.
