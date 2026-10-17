# Managed plugin operations

Plugins keeps inspection, trust, workspace activation, configuration and runtime
permission separate. Inspect a local folder or controlled catalog result; review
the selected components and exact source before installation. Review trust and
the captured workspace. Enabling a component does not grant its tools permission.

## Configure and use

After creating saved data or a connection profile, use Refresh to load the current
root/profile choices before the next review. Status is a refreshed snapshot.

Save MCP configuration without connecting. Review connection and discovery as a
separate action, then review exact mapping references. Explicit hook/connection
tests need an existing idle Console chat in the captured workspace and use normal
permissions and resource custody. A changed, deleted or busy workspace/chat may
refuse the test. Use Console, then return to Plugins; the session draft retains
its original identity. Changes to inputs invalidate the previous review. In Console, Alt+A opens the
pending approval decision; Tab stays within the current pane, while F6 moves
between panes. Choose Approve once or Deny through that normal permission card.

Updates and rollback need fresh review of the exact revision and component
selection. Unsupported optional components do not disable unrelated ready ones;
required dependencies block affected behavior. Selection changes and newly
supported adapters do not silently opt in components.

### Configure declared values

In the installed plugin's actions, choose **Save or revoke declared value**. Select the
name, enter its declared type (whole number, `true`/`false`, or text), and choose
**Save value**. Secret input is masked. Saving creates a protected reference;
**Review action**, then **Confirm** applies the complete mapping separately.
The filled Configure JSON retains current connections, discovered tools, model
choices, component tool constraints and other variable references. When editing
that JSON manually, supply the complete desired mapping, using reference handles
only. Omitting a mapping removes it from the proposed configuration.

Input clears after saving, invalid input, closing and navigation. An in-flight
save retains only its reference metadata when complete; reopening can select that
reference. **Review revoke**, then **Confirm revoke** invalidates the protected
reference and clears any pending configuration review. Consuming components need
a replacement; a revoked or wrong-type reference never falls back to a default.
The host checks identity and generation again at actual launch and does not put
resolved values in packages, snapshots, handler definitions or result logs.

## Disable, uninstall and saved data

Disable here fences the captured scope immediately; Disable everywhere and
uninstall affect every scope. A persistence failure may leave a session-only
stop: it is not a durable disable. Independent authorized workspace work may
continue on a shared connection. A cancelled waiter does not establish that the
server stopped or that an uncertain request can be retried. Cancellation visits
every owned lease, including several connections belonging to one run. A failed
stdio cleanup and its retry keep root custody until the original direct child
is observed terminated and retained cleanup has settled.

A completed Console turn may leave an owned idle MCP connection. That connection
continues to count as a possible data writer until reviewed disconnect/drain
settles it; a successful tool response does not prove all custody ended.

Uninstall keeps saved data. History can manage roots under their original
installation identity. Review exact roots and all affected users before deletion.
New users are fenced; active readers, late writers and idle writer-capable
processes must drain, or explicitly authorized cancellation must confirm cleanup.
Cancelling the wait never forces other work. Cleanup pending is not success.

A status of cleaned saved data means the authenticated owner observed completed
absence under its current exact root and boot checks. It is a point-in-time
status, not future absence, quiescence, or permission to skip review. Recovery
reasons and pending cleanup take precedence. Unexpected recreated leaves and
stale generations remain refused.

## Recovery

Clean shutdown and an uncertain shutdown differ. Reboot reconciliation requires
a separately protected matching clean checkpoint and exact qualified native
root identity. Dirty, unknown, legacy or remote-uncertain custody cannot be
repaired by checking a box, observing no rows, or guessing that a process exited.
Reduced marker backends cannot provide the protected rollback guarantee. Actual
OS reboot and keychain behavior need separate platform evidence.

Durable intent, certificates and complete authority snapshots decide restart
recovery; a prepared intent alone cannot advance authority. A missing response
may follow a committed operation: look up its original operation ID instead of
replaying execution. Credential refresh within the same authorized binding does
not authorize replay of an uncertain call. Changed account, endpoint, scope or
binding reference requires reviewed current authority.

See [compatibility](compatibility.md) for unqualified environments and
[authoring](authoring.md) for component and resource contracts.
