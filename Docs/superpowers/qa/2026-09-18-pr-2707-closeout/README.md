# PR #2707 closeout — TASK-32824

The current PR stops at its existing component-review scope. The remaining eleven
destinations, Settings categories and Console subflows move to follow-up PRs.
**Do not merge until the owner approves the final visual/conflict review.**

Start with [the visual and conflict review](REVIEW.md). The integrated production
source is merge commit `863e56b5e9799e9a8dda37b8da9c666e00a69c22`, combining PR
`157b19ee9f` and current dev `53b56384cc`. Later closeout changes contain evidence,
ledgers and the final test-only unused-variable cleanup; production source stays
identical to the nineteen hashes in [the native receipt](native/result.json).

## Integration and targeted checks

- Sole text conflict: `backlog/docs/lessons-testing-evidence.md`. Retain the whole
  PR document and insert the complete incoming watcher-testing lesson. Neither
  side was discarded. [Exact parents and union proof](conflict-proof.json).
- Four overlapping automatic merges were reviewed: diagnostic inventory,
  Console screen, native gateway tests and MCP workbench tests. The incoming
  recovery completion wiring and existing PR behavior are preserved.
  [Independent integration review](integration-review.txt).
- **53 distinct targeted integration cases pass in final results:** 43 unaffected
  passing cases from [the initial run](integration-initial-results.json), plus all
  ten Send-state cases in [the corrected run](send-final-results.json). Runs were
  serial and each case selected a fresh private profile before imports. Counts
  are not additive across the initial and corrected runs.
- The initial integration run passed 51/53. Both failures injected child-only
  state and then yielded while the live parent projected its actual ready state.
  Clean dev reproduced the setup-block failure; its markup case passed on that
  attempt ([baseline results](dev-baseline-results.json)). The setup test now
  supplies the screen's blocker and uses its real projection; the markup test
  inspects its synchronous formatting result before the parent replaces it.
  Assertions and async blocked/unblocked checks remain. Production code is unchanged.
- Seven derived-artifact preflight guards pass ([output](preflight.txt)). Fresh
  ID checks cover 278 refs, 32 worktrees and 4,159 task files. Scoped Ruff and
  formatting checks pass. No full repository test suite was requested or run.

## Native and visual qualification

Sixteen captures were rendered and inspected: dark/light at 80×24 and 170×48,
showing Console Steps, Tool Profiles Import, Deep research saved on and Ask user
focused after restoring Deep research off. All eleven MCP gate buttons were
traversed by Tab and asserted fully painted inside their clipping region.
Settings navigation is real; individual target controls are focused directly.
MCP uses real destination/Servers/built-in selection and private config writes.

The real `TldwCli` uses LinuxDriver with stdout, stderr and its render stream on
the terminal. Ctrl+Q returns from App.run with exit0; the PID is absent and the
instance lock can be reacquired. All eleven private databases pass quick_check,
with zero conversations/messages, unchanged default config/UI/policy files,
empty fault output and no app errors. All nineteen source hashes match.
[Lifecycle receipt](lifecycle.json). The first lifecycle assertion mistakenly
expected the previous MCP-only run's ten databases; Settings also creates the
sync-state database. Every actual database was healthy. The checker was corrected
to include that eleventh database, without rerunning or weakening the journey.

This replay does not qualify connected MCP servers, tool execution, provider
calls, recovery dispatch, broader Settings persistence or all destinations.
Earlier galleries below retain their original source and fixture boundaries.

## Reviews, CI and follow-up

GitHub had zero review threads and zero submitted reviews at closeout inspection.
CodeRabbit's success status represents a skipped draft review, not a code review.
The scoped independent integration review found no merge defect. The subsequent
composer diagnosis was read-only; final harness edits were self-reviewed.

On pre-integration head `157b19ee9f`, all applicable Actions checks passed except
[Windows GGUF source evidence](https://github.com/rmusser01/tldw_chatbook/actions/runs/35398114314/job/105771503467).
Its external-copy case first failed to settle visible keyboard focus; teardown
also raised a missing `SelectOverlay` during inventory publication. The same
intermittent family is already retained in the MCP ledger. It is not dismissed
as infrastructure and no repair is claimed here. Final pushed-head CI is checked
separately in the PR; prior-head results cannot qualify the final head.

TASK-32823 is safely pushed on `codex/mcp-inspector-refresh-followup` at
`135f226888`. Its prototype is excluded from this PR and remains incomplete:
retiring the test panel must synchronously invalidate delayed preview minting.
After the approved merge is confirmed, continue on a fresh branch from merged
`dev`, beginning with MCP server lifecycle/inspector and reusing that work.
See the [completion ledger](../../reports/2026-09-17-design-system-completion-audit.md)
and [MCP ledger](../../reports/2026-09-18-mcp-review.md).

ADR required: no. ADR path: N/A. This is routine integration, test-harness
correction and closeout of the existing ADR-150/161/168/170 implementation.
