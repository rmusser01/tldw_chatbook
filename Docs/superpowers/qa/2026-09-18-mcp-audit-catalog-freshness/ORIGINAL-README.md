# Audit catalog freshness — TASK-32838

Audit drilldowns kept the tool object captured before their clearing awaits.
A completed catalog refresh could therefore leave a current row next to an old
description, schema or connection state, or show detail for a removed tool.
Both destinations now acquire the existing catalog publication lock, validate
the profile and resolve the current identity before selecting/rendering detail.
The lock remains held through inspector waits, with the existing workbench to
inspector ordering. A vanished identity clears detail and warns.

This repairs existing workbench synchronization under ADR-150/161. No new ADR,
interface, persistence, permission policy, CSS or token change is introduced.

## Verification

[62 distinct targeted cases pass](qualified-cases.json): 12 new cases, 24 adjacent
inspector/workbench/table cases and 26 design/component governance checks.
[Exact cases](selected-cases.txt) and [final output](tests/final-001.txt) record
the scope. No full suite ran. Two warnings concern unrelated old pytest temporary
folder cleanup; there are no test failures.

The initial [eight focused cases failed as intended](tests/red-001.txt): old
descriptions, retained removed tools, and navigation finishing before a paused
catalog publication. The [12 new cases then passed](tests/green-001.txt).
They exercise the real catalog derivation and publication path for same-ID
replacement, disconnection and removal, a publication paused after Tools rows
but before permission state, and actual Audit actions in a dark/light real app.
Adjacent cases preserve profile-change rejection and shared permission jumps.

All [seven preflight guards](preflight.txt) pass. [Static analysis](static-analysis.json)
adds no diagnostics; the workbench retains its 65 baseline diagnostics and new
files are clean. New files and [changed production ranges](formatting.txt) pass
formatting. Only formatting changed after the final test run started.
[Independent review](independent-review.txt) found no concrete blockers.

## Native visual evidence

The real TldwCli runs with LinuxDriver and TTY streams inside a fresh private
HOME, USERPROFILE, XDG and TLDW profile validated before imports. One synthetic
metadata record identifies the real built-in `chat_with_llm` tool. The runner
wraps the catalog collector to supply a controlled same-ID replacement during
the action's Audit-clear await, then invokes the real catalog publication path.
The replacement changes the label to `Refreshed catalog`, description to
`Catalog refreshed during Audit navigation.`, and schema to a `review_query`
field. This is controlled UI publication evidence, not a connected server
changing its discovery output.

In each dark/light × 120×40/170×48 cell, direct focus plus Enter activates the
real Audit row and both action buttons. Full visibility of activated controls
is asserted. Each journey starts from the original catalog, publishes its
replacement during navigation, and verifies the selected identity, refreshed
label/description/schema and exact destination row. Both destinations visibly
use `Refreshed catalog` in the [eight inspected captures](GALLERY.md). Tools
also shows the refreshed description. Schema, disconnection/removal, and a
partially published catalog are covered by targeted tests; no tool executes.

The [receipt](native/result.json) and [lifecycle check](lifecycle.json) record
eight controlled replacements across four passing cells, unchanged permission
profiles, normal App.run return, exit 0, absent process, released lock, ten
healthy private databases, zero conversations/messages, unchanged default
config/UI state/runtime policy and no app errors or faulthandler output. Eleven
source hashes and the runner hash match final source. [Export hashes](export-manifest.json)
distinguish originals from copies normalized only for trailing whitespace.

## Bounds and next review

This independent branch starts at dev `cef6bd2a3e3f0b8de0e166146acb4900ca7ea2b6`.
PR #2724 separately repairs retired Audit controls and missing destination rows;
its changes are not included here. Existing inspector-refresh, compact layout
and guidance follow-ups remain separate. This repair covers navigation against
pending publication; ongoing refresh of an already-open inspector remains with
that earlier inspector-refresh follow-up. The 80×24 inspector, connected-runtime
journeys and the wider component review are not qualified by this slice.

Next: Permissions restored-roots review. Current-head CI and final visual
approval remain required before merging this draft. [Allocation](allocation-census.json),
[fresh census](precommit-census.json) and [sole ownership](allocation-owner-check.json)
checks cover the task ID.
