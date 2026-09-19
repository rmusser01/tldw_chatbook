# MCP inspector reachability — TASK-32832

At 80×24, the inspector hid Test Tool arguments below a non-scrolling viewport.
Its default button minimum also clipped actions under the new scrollbar; resize
could leave the focused Run action outside the visible region. The existing
inspector now scrolls vertically, reveals its current focused child after resize,
and uses token-backed content sizing for its form and owned buttons. Drafts and
handlers are preserved. ADR required: no; applies ADR-150/161 and ADR-031.

## Targeted verification

The [case ledger](qualified-cases.json) records **42 distinct passing cases**:
eight schema/raw form journeys across compact/wide dark/light, six compact
readiness/permission/Advanced action cases, and 28 design/CSS performance guards.
The final permission fixture explicitly includes Re-allow, Remove and Revoke;
each must be exercised and fully painted. The form cases cover Tab, approval
label wrapping, retained arguments through resize, one emitted request, and Escape.
All seven [preflight guards](preflight.txt) pass; no budget or exception changed.
New tests, native runner and fixture pass Ruff and formatting. Inspector Ruff
has the same 13 pre-existing diagnostics, with no added diagnostics; its new
handler range is formatted. [Independent review](independent-review.txt) is clear.

[Initial and intermediate results](test-results.json) are retained. The original
production source fails all eight form cases. Overflow alone leaves buttons too
wide; content sizing alone leaves focused controls offscreen after resize.
Sibling action checks exposed the same clipping beyond Test Tool. The first
attempt lacked the required private-profile fixture annotation and stopped in
setup. Two later raw-form cases raced deferred mount focus; the minimal test app
now waits for its mount worker and preview before Tab. These intermediate runs
are not added to the final passing count.

Six selected older inspector tests stop before UI creation with
`RecoveryRequired: raw_source_selection_changed`, identically on unchanged
149acda36be8939fe8cd5e589bf77d13462257e7 and modified production source.
[Comparison](neighbors-comparison.json), [baseline log](tests/neighbors-baseline.txt)
and [modified log](tests/neighbors.txt) retain that harness limitation. They do
not count as passing behavior coverage. No full suite was run.

The branch rebased without conflicts onto dev `cef6bd2a3e3f0b8de0e166146acb4900ca7ea2b6`.
That intervening commit changes only an unrelated Console test and its task.
All [14 geometry/behavior cases pass again](tests/rebased.txt); native source,
runner and exported evidence hashes still match. Reruns do not increase the
42-case distinct count.

## Native visual and lifecycle evidence

The real TldwCli runs through LinuxDriver and native TTYs under a fresh private
profile. The retained runner uses the real permission store, service, MCP client
and local JSON-RPC stdio subprocess; production transport/execution is not mocked.
A required argument first fails validation without a wire execution or audit
success; correcting it and pressing Approve & run once produces an actual
`tools/call`, successful audit record and visible OK. Close and Test Tool reopen
work in the same process/connection. Each of four dark/light 80×24/170×48 cells
passes. Four further connected-server readiness cells check complete button paint.

All [12 final captures](GALLERY.md) were rendered and inspected. Compact approval
text wraps completely, Close remains reachable and connected-server actions fit.
The runner directly focuses individual controls then uses Enter; mounted tests
cover Tab traversal and resize draft retention. Summary captures explicitly reveal
the result region. This is bounded inspector qualification, not an end-to-end
keyboard traversal of every MCP destination.

The [native result](native/result.json) records four real calls and four successful
audit entries on one connection. [Lifecycle evidence](lifecycle.json) records
App.run returning 0, process exit 0, terminated/reaped fixture, both PIDs absent,
released instance lock, ten healthy private databases, zero conversations/messages,
unchanged default config/UI state/runtime policy fingerprints, and no app errors
or faulthandler output. All 13 source/fixture hashes and the retained runner hash
were [independently matched](source-verification.json) after final verification.

The [first native attempt](initial-native/result.json) completed its four tool
cells, then hit duplicate `mcp-builtin-enable` IDs while the harness directly
called `_select_server_key` during a mode change. The final runner instead uses
the ordinary rail selection and waits for the readiness action. The first failure
and clean shutdown are retained; this change does not establish that every
concurrent overview-refresh path is fixed.

## Remaining bounds

This branch is independent of draft PRs 2711–2716 and contains no changes from
them. MCP `isError` propagation belongs to PR2716. Permission mutations, Audit
workflows, remote transports and full schema/result recovery remain unqualified.
The compact Raw response disclosure title is still clipped, visible in these
captures; result disclosure/content deserves its own bounded review. The wider
Servers toolbar has separate follow-up work. No merge is authorized here: final
visual approval still applies.

Task allocation ownership and normalized export hashes are retained alongside
this receipt. Exports remove trailing line whitespace only; the
[manifest](export-manifest.json) records original and exported hashes.
