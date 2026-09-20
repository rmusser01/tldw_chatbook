# Selected MCP inspector catalog refresh — TASK-32823

Saved as [PR2757](https://github.com/rmusser01/tldw_chatbook/pull/2757).

**Latest qualification:** [Qodo closeout](qodo/README.md) adds policy-input
currentness, API docs and real-mount coverage: 316 current-source targeted passes
and fresh native captures matching the approved gallery apart from caret blink.
Earlier evidence below records the original approved implementation.

Resumed saved prototype `135f226888` on the verified PR2730 merge
`802809947b`. [PR2730 receipt](pr2730-closeout.json) confirms all current-head
gates, resolved Qodo findings and an actual merge tree equal to the tested tree.

Catalog refresh now updates the existing selected detail, clears removed tools,
and preserves equal definitions' mounted structured/raw argument drafts, cursor,
focus and permission preview. Changed definitions retire the old form and display
reopening guidance. A form owner token is invalidated before awaited teardown;
late previews and queued requests cannot publish into a retiring or replacement
form. Accepted successor requests retire old workers even when profile validation
fails. Refresh does not replace a newer selection or steal newer keyboard focus.

**475 distinct targeted passes; one reproduced baseline failure.**
[Case census](qualified-cases.json): 89 refresh/selection/navigation/governance
passes, 295 adjacent inspector/prepared-tool passes and 91 runner-boundary passes.
All 13 new cases pass. [The sole failure](baseline-comparison.json), eleven raw
Workflows dimensions, also fails on unchanged `802809947b`; the test and offending
stylesheet are byte-identical. No full sweep. [Run logs](tests/targeted-001.txt).
[All seven artifact guards pass](preflight.txt). New files are Ruff-clean; changed
production ranges pass formatting. [78 pre-existing production diagnostics, zero
introduced](lint-comparison.json). Independent review reproduced and checked the
focus and invalid-profile successor fixes and found no remaining blocker.

[Eight inspected native captures](GALLERY.md) show the real private app in dark
and light at 120×40 and 170×48. The controlled collector modifies a real built-in
`search_notes` definition; real service previews are prepared, and no tool runs.
The journey asserts draft/cursor/focus preservation, replacement guidance and
removed-tool clearing. [Result](native/result.json) records actual loaded module
origins and current source hashes. [Lifecycle](lifecycle.json) proves normal exit,
released lock, ten healthy private DBs, zero chats/messages, unchanged defaults,
no app errors and no network attempts. This is UI/catalog projection evidence,
not external discovery or connected-runtime qualification.

Earlier attempts remain explicit: 001 stopped before app startup on an import
provenance lookup; 002 selected an agent-only calculator absent from the MCP Tools
catalog and exited normally; 003 passed before formatter-only line wrapping.
004 is the final source-matching native evidence. Saved September 18 prototype
logs are historical and do not qualify the completed repair.

Existing ADR-161/170 and ADR-032 local tool permission boundaries apply; no new
ADR, service admission policy, stylesheet or token change. The owner approved
the eight-capture gallery and merge. [Conflict-free current-dev integration](approved-integration.json)
preserves all approved MCP sources/styles; all 13 refresh regressions pass again.
Current-head CI/Qodo and final live-dev verification remain before merge. Remaining
work includes compact/long-path presentation, connected-runtime journeys and
other destination reviews. The unrelated Workflows dimension floor stays visible
as a follow-up; this PR does not expand into that screen.

Text/SVG exports trim trailing line whitespace; the manifest retains both raw and
export hashes for normalized files. Original captures remain in the recorded
private run directory. This changes no rendered text or source code.
