# Audit catalog freshness — TASK-32838 / PR2726

Audit navigation can wait while the catalog replaces the selected tool. Both
drilldowns now resolve that identity again under the existing publication lock,
then select the row and render its current description, schema and availability.
Removed identities clear detail and warn. Merged row-selection checks, control
ownership and post-selection profile validation remain intact.

PR2724 merged at `cccf0acdad8e939a55cb003588ff1406cef5d1f4`. This existing
PR2726 draft is rebased onto that commit. [Conflict choices and visual review](CURRENT-DEV-REVIEW.md)
cover the combined implementation; [original evidence](ORIGINAL-README.md)
is retained as historical qualification against the earlier base.

ADR required: no. Existing ADR-150/161 apply. This repairs existing publication
synchronization without changing permission policy, persistence, interfaces,
styles or token values.

## Current verification

[168 distinct targeted cases pass](current-dev/final-cases.json): 14 catalog
freshness, 32 Audit navigation, 23 permission navigation, 49 native launcher
argument checks and 50 adjacent/governance cases. [Exact cases](current-dev/selected-cases.txt)
and [output](current-dev/final-tests.txt) define the scope; no full suite ran.
Two warnings concern cleanup of unrelated old pytest temporary directories.

The two new profile-switch tests change the active profile during destination
row selection. Both [fail when the preserved post-selection checks are removed](current-dev/profile-negative-tests.txt)
and pass with the exact integrated source restored. The original replacement,
removal, disconnection and incomplete-publication regressions remain covered.
[All seven preflight guards](current-dev/preflight.txt) pass. [Ruff](current-dev/static-analysis.json)
adds no diagnostics; workbench/inspector retain their 65/13 baseline findings.
New files and [changed production ranges](current-dev/formatting.json) are
formatted. [Independent review](current-dev/independent-review.txt) found no
blockers in the navigation integration or QA refresh.

## Native evidence and limits

The [current gallery](GALLERY.md) contains eight rendered and inspected captures
from a real TldwCli/LinuxDriver terminal, dark/light at 120×40 and 170×48.
Shared argument validation precedes application imports and output writes.
The runner uses a fresh private HOME/USERPROFILE/XDG/TLDW profile, the supported
terminal warm-up, grouped deferred imports and an outbound network guard.
Keyboard-activated controls must be fully visible, painted and own their hit
position. Paths and tmux names are validated; output reuse is rejected.

One synthetic metadata record targets the real built-in `chat_with_llm` tool.
A controlled collector replacement publishes `Refreshed catalog`, a new
description and `review_query` schema while each Audit action clears its source
panel. Both destinations select the exact row and use the fresh definition.
This is controlled UI publication evidence, not connected-server qualification.
No tool executes and no permissions change.

Both private journeys pass these behavioral assertions. The replay [receipt](current-dev/native/result.json)
and [lifecycle check](current-dev/lifecycle.json) record eight replacements,
zero network attempts, one metadata record, unchanged permission profiles,
normal App.run return, exit 0, absent process, released instance lock, ten
healthy databases, zero conversations/messages and unchanged user-default files.
All eleven application source hashes and both QA script hashes match.

The first run exposed an intermittent light-wide Tools header misalignment;
the selected row and fresh detail were correct. An unchanged-source replay
shows aligned headers. [Both views and the investigation boundary](HEADER-FOLLOWUP.md)
are retained: the replay does not prove the layout issue fixed. ToolsMode is
unchanged from merged dev; a matching Textual header-cache hazard was found,
but baseline reproduction is still needed to establish attribution. This PR
qualifies catalog-current navigation, not general table-header reliability.

[Export hashes](current-dev/export-manifest.json) and [normalization receipts](current-dev/log-export-manifest.json)
identify the saved evidence. The [replay launch note](current-dev/replay-launch-note.txt)
records a rejected reused-profile wrapper before the corrected private replay.

## Remaining work

PR2726 still requires current-head CI/review and its own owner visual approval.
PR2724's approval does not authorize this merge. The header race is a bounded
follow-up before Permissions restored-roots review. Already-open inspector
refresh, 80×24 inspector reachability, connected-runtime journeys and the wider
component workstream remain open in the [MCP ledger](../../reports/2026-09-18-mcp-review.md).
