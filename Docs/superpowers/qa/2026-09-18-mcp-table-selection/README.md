# MCP table refresh and inspector selection — TASK-32796

Server navigation cleared the inspector, but delayed table redraw highlights
selected rows again and reopened obsolete tool/finding details. ADR-170 replaces
the shared after-refresh flag with a synchronous Textual prevent context at
message publication. All existing consumers and Audit Findings now use it.
Deduplication includes the owning table, and external Tools/Permissions drills
suppress their final cursor move while admitting the next real Enter.

## Targeted evidence

**275 distinct cases pass:** [273 selected consumer cases](final-consumers-273.txt)
and the two original Workbench inspector-clearing regressions, included in the
[14-case regression run](inspector-regressions-14.txt). Counts overlap. The final
consumer run includes all 14 new cases, adding delayed cell highlights and equal
row keys across different tables to the earlier regression run. Only formatting,
with unchanged Python ASTs, followed the production repair. No full suite ran.

- [Nine delayed-event failures before repair](queued-highlight-red.txt) hold real
  published highlights until after refresh callbacks, rather than adding sleeps.
- [Three review regressions before repair](review-regressions-red.txt) cover hidden
  Audit Findings interrupting an Executions arrow/Enter gesture and focused
  external Tools/Permissions drills posting unwanted selection messages.
- [Independent review](independent-review.txt) found no remaining bounded issue.
  All [nine publication contexts](synchronous-contexts.json) contain zero awaits.
- [Ruff comparison](lint-comparison.json) has no introduced findings. Audit,
  Permissions, Servers and Voice Cloning retain 2, 3, 4 and 23 baseline findings.
  Shared helper, Tools, new tests and native runner are clean; changed functions
  and new files [format](format-checks.txt). The runner was checked again after
  the native retry. [Backlog validation](backlog-check.txt) and
  [diagnostic inventory](diagnostic-inventory.txt) pass.

The [initial wider run](initial-consumers.txt) passed 260 cases and failed six.
Four Audit CSS assertions still require literals where unchanged source uses
matching tokens (70%, 24, 20); [baseline evidence](audit-css-baseline.jsonl)
reproduces all four at 20de195131. They are explicitly excluded from the final
consumer run. Two Speech harness cases fail during profile setup with
`raw_source_selection_changed`, before the UI is built; they were not requalified
here. Existing direct Voice Cloning consumer tests pass. These exclusions do not
qualify a green full suite.

## Native evidence and visual bounds

Final run `/private/tmp/tldw-32796-native-002`, PID 84687, used the real app,
LinuxDriver, TTY streams, real catalog and keyboard activation. Dark/light at
80×24 and 170×48 each selected a tool, pressed `r` with its table focused,
verified the exact inspector node remained mounted, activated All servers,
refocused/refreshed the table, verified the tool detail stayed cleared, and
selected again with an arrow. All eight final captures were rendered and
visually inspected. Table names and State remain visible; cleared captures
show the inspector prompt. Compact detail content continues below its viewport;
this is not a full inspector-content or action-reachability review.

[Native result](native-result.json), [lifecycle](lifecycle.json) and
[capture hashes](capture-manifest.json) record final source/runner identities,
exit 0, normal return/shutdown, process absence before terminal closure,
reacquired instance lock, ten healthy private databases, zero conversations and
messages, unchanged default files and permission profiles, and no error/fault
log entries. No connected external server or tool execution was exercised.

The first run stopped at a full-button visibility assertion: the existing 80×24
rail paints `All` from `All servers`. [Failed capture](failed-run001-state.svg),
[result](failed-run001-native-result.json) and
[clean failed-run shutdown](failed-run001-lifecycle.json) are retained. The final
runner explicitly records that clipping, requires visible paint and genuine
focus, and verifies keyboard activation. It does not waive or approve the rail
layout. Wide rail labels are complete; compact rail clipping remains open.

| View | Selected detail after refresh | Cleared detail after refresh |
| --- | --- | --- |
| Dark 80×24 | ![Selected Dark 80×24](textual-dark-80x24-selected-after-refresh.svg) | ![Cleared Dark 80×24](textual-dark-80x24-cleared-after-refresh.svg) |
| Dark 170×48 | ![Selected Dark 170×48](textual-dark-170x48-selected-after-refresh.svg) | ![Cleared Dark 170×48](textual-dark-170x48-cleared-after-refresh.svg) |
| Light 80×24 | ![Selected Light 80×24](textual-light-80x24-selected-after-refresh.svg) | ![Cleared Light 80×24](textual-light-80x24-cleared-after-refresh.svg) |
| Light 170×48 | ![Selected Light 170×48](textual-light-170x48-selected-after-refresh.svg) | ![Cleared Light 170×48](textual-light-170x48-cleared-after-refresh.svg) |

## Remote verification and remaining work

Saved head 20de195131 [Fast Lane](https://github.com/rmusser01/tldw_chatbook/actions/runs/35368901609)
failed only these two inspector cases (1,150 passed), and its derived-artifact
aggregate gate failed because Fast Lane failed. [Logs](prior-head-ci-failures.txt)
and [check snapshot](prior-head-checks.json) preserve that evidence. UI latency,
Backlog IDs, CSS and all six GGUF checks passed on that head. This repair passes
both failures locally; fresh remote checks are required after push.

Selected definitions/schema/drafts, execution, Servers, Audit/Permissions flows,
the baseline harness/assertion failures and compact rail clipping remain in the
[MCP review ledger](../../reports/2026-09-18-mcp-review.md). PR2707 remains draft,
unmerged and subject to its own visual review and merge approval. The broader
component review is still active.
