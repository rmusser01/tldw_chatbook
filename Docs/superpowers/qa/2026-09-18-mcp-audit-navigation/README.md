# Audit navigation ownership — TASK-32837

Queued presses from retired Audit buttons could open a replacement record.
Mutating the original entry dictionary could also redirect a still-displayed
button. Each mounted action now captures its rendered tool and profile identity;
replacement/clear invalidates those controls before pruning yields. Both
navigation destinations now honor row-selection failure, clear detail and warn
when the tool disappears during navigation. The Permissions repair also applies
to its existing shared jump helper's other two callers.

This is an implementation repair under ADR-150/161; no new ADR is required.
There are no CSS, design-token, permission-policy or persistence changes.

## Verification

[64 distinct targeted cases pass](qualified-cases.json): 14 new regression cases,
24 adjacent inspector/workbench/table cases and 26 design/component governance
cases. [Exact case selection](selected-cases.txt) and [final output](tests/final-001.txt)
record the scope. No full suite ran. The final run has two cleanup warnings
for unrelated old pytest temporary folders and no test failures.

The initial [red run](tests/red-001.txt) has eight intended failures, two passing
clear-only cases and four deselected full-app cases. It reproduced retired and
mutated identity plus missing destination rows. All [14 new cases passed](tests/green-001.txt)
after the repair. Tests also cover same-record re-render, profile capture,
profile switches before/during navigation, both shared permission jump callers,
and real-app dark/light navigation to rows hidden by destination filters.

All [seven preflight guards](preflight.txt) pass. [Static analysis](static-analysis.json)
adds no diagnostics: inspector/workbench retain their 13/65 baseline diagnostics;
new files are clean. New files and [changed production ranges](formatting.txt)
pass formatting. [Independent read-only review](independent-review.txt) found
no concrete blockers.

## Native visual and lifecycle evidence

The real TldwCli runs with LinuxDriver and TTY streams inside a fresh private
HOME, USERPROFILE, XDG and TLDW profile validated before app imports. Only two
synthetic execution metadata records are appended: one for the real built-in
`chat_with_llm` catalog entry and one for an unavailable tool. No tool executes,
no external server connects and no permission profile changes.

In each dark/light × 120×40/170×48 cell, the journey seeds a nonmatching destination
filter, selects the valid Audit row, and activates each real action button with
Enter after direct focus. It verifies the selected identity, exact destination
row key, cleared filter and removed Audit controls. It then selects the missing
record and checks that both actions stay in Audit and show the warning.
Full visibility of each activated control is asserted. Destination-row loss
during an await is verified by deterministic tests, not by a native race.

All [16 captures](GALLERY.md) were rendered and inspected. Tool and permission
destinations select `chat_with_llm`; missing destinations show the unavailable-tool
warning. Existing inspector guidance and Audit filter layout are visible because
their separate PRs are not included. Compact 80×24 inspector reachability remains
with PR #2718 and is not qualified here.

The [native receipt](native/result.json) and [lifecycle check](lifecycle.json)
record four passing cells, normal App.run return, exit 0, absent process,
released lock, ten healthy private databases, zero conversations/messages,
unchanged default config/UI state/runtime policy, and no app errors or
faulthandler output. All eleven captured source hashes and the runner hash
match final source. [Export hashes](export-manifest.json) distinguish original
private evidence from repository copies normalized only for trailing whitespace.

## Scope and follow-ups

This independent branch starts at dev `cef6bd2a3e3f0b8de0e166146acb4900ca7ea2b6`,
after PR #2707 merged. Audit selection (#2720), filter layout (#2721) and inspector
guidance (#2722) remain separate draft PRs. Current-head CI and final visual
approval remain required before merging this follow-up.

Next: review catalog replacement while an Audit drilldown is already in flight.
This repair qualifies missing rows, not same-ID tool-definition freshness across
an await. Connected-runtime and the wider component review remain open.
[Allocation](allocation-census.json), [fresh census](precommit-census.json) and
[sole ownership](allocation-owner-check.json) checks cover the task ID.
