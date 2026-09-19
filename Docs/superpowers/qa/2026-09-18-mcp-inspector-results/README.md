# MCP raw responses and result replacement — TASK-32833

A local argument-validation failure changed the status line but left the previous
raw response and interpretation visible. It now uses the existing complete
result renderer, preserving the draft, preview and exact permission-profile
context while clearing the old outcome. Corrected input can run again normally.

The Raw response disclosure also extended beyond the compact inspector. Its title
now wraps within the available width, its contents lose unnecessary indentation,
and its scroll viewport uses the existing eight-row token in compact mode. Wide
layouts retain the existing twelve-row body. Focus styling, keyboard expansion,
plain-text rendering and the existing raw-output cap are preserved.

This branch depends on [PR2718](https://github.com/rmusser01/tldw_chatbook/pull/2718),
head ba4b418c5883bed4a74aa896da02a034d480bb21, for inspector reachability. Its draft
PR targets that branch to isolate this diff. Retarget onto dev after the parent
merges, then check integration and obtain final visual approval before merging.
The repository GitHub workflows target dev, so their checks must run on the
retargeted PR before merge; the stacked draft has the local verification below.
It does not include the independent PR2716 tool-error propagation change.
ADR required: no; this repair applies ADR-150/161 and ADR-031.

## Automated evidence

[50 distinct passing targeted cases](qualified-cases.json): four dark/light
compact/wide disclosure and body-scroll/resize journeys, four schema/raw result
replacement and correction cases under default/named permission context, fourteen
parent reachability cases, and 28 design/CSS performance guards. Emitted requests
retain their exact arguments and profile context; invalid attempts emit none.
Mismatched tool results do not replace the current result, and Escape still closes.
No full suite was run. [All seven preflight guards](preflight.txt) pass without
raising budgets or exceptions. New tests and runner pass Ruff and formatting;
the changed inspector range is formatted and adds no Ruff diagnostics to its
13-diagnostic baseline. Independent review found no remaining concrete blocker.

[Initial and intermediate attempts](test-results.json) remain in the receipt.
The corrected baseline reproduces clipped compact title/body geometry and stale
raw/note visibility after validation. The first harness also raced deferred
mount focus and expected unescaped Unicode from an ASCII-escaped fixture. Those
harness issues were corrected before qualification. Waiting for disclosure
animations exposed continued clipping of the twelve-row compact body; the
compact height repair addresses that geometry. A later correction press arrived
inside Textual Button's activation debounce; the test now waits on its bounded
active state, as the native runner already did. Intermediate passes and reruns
are not counted twice. See the final geometry/parent results and final validation
results alongside the governance log.

## Native qualification

The real TldwCli runs through LinuxDriver and native TTYs with a fresh private
profile and the real permission store, service, client and a local JSON-RPC stdio
fixture. The runner selects the actual tool, then directly focuses controls and
uses Enter/End. In each dark/light 80×24/170×48 cell it executes a long response,
reads its full disclosure title and final content, submits invalid required input,
and verifies that old raw/note content disappears without another execution.
Corrected input produces a fresh response on the same live connection. Closing
and reopening clears the displayed outcome. There are eight actual tools/call
requests and eight successful audit records; the four invalid attempts run none.

All [16 captures](GALLERY.md) were rendered and inspected: title, raw end,
validation failure and corrected response for each cell. Compact title wrapping
is complete when focused, the focused raw viewport fits, and the final response
content is readable. The corrected capture focuses the body, so its title may
scroll above the viewport. Tests separately qualify resize and literal markup/
Unicode handling. This is not an all-controls keyboard tour or qualification of
all schema, permission, runtime-error or remote transport combinations.

The [native result](native/result.json) and [lifecycle receipt](lifecycle.json)
record success on the first native run, normal App.run return, exit 0, terminated/
reaped fixture, absent app and fixture PIDs, released instance lock, ten healthy
private databases, zero conversations/messages, unchanged default config/UI-state/
runtime-policy fingerprints, and no app error or faulthandler output. All thirteen
source/fixture hashes and the runner hash match the final files. Exported capture
and log hashes are recorded in the [manifest](export-manifest.json); only trailing
line whitespace and final newlines are normalized.

The task has one allocation owner across refs/worktrees. Broader schema edge
cases, preview/execution-state ownership, Audit filtering and exact-event/tool
drilldown remain separate reviews. Nothing is merged by this qualification.
