# Compact restored MCP root review — TASK-32879

At 80×24 the old dialog initially focused an offscreen Cancel button. Its plain
Container could scroll with the mouse but supplied no keyboard scroll bindings,
so Page Down could not expose the remaining paths. A recovery-specific subclass
now puts the unchanged review text in a focused VerticalScroll, keeps both
standard decisions outside that body, and left-aligns wrapped literal paths.
Existing tokens size and color this dialog; other confirmations are untouched.
The workbench no longer changes its layout after mounting.

ADR required: no. Existing ADR-126 and ADR-150/161 apply. Approval ownership,
native recovery writes, historical retention and safe dismissal are unchanged.

## Verification

116 distinct targeted cases pass: four new compact/theme regressions and 112
adjacent recovery/design checks. No full suite ran. Exact run receipts:

- [Four meaningful pre-fix failures](tests/red.txt), followed by
  [four passes](tests/green-001.txt). A first test attempt omitted its markup-path
  fixture and is retained separately as unqualified evidence.
- [Initial adjacent run](tests/adjacent-001.txt): 110 passes, four failures.
  [Merged-dev comparison](tests/baseline-failures.txt), at
  `62d43190ce3ca21bda0f0ff03e5148971eedc7ca`, reproduces the unbound-fresh-root
  assertion and the existing Workflows dimension ratchet (11 raw literals).
  Those two baseline failures remain outside this presentation change.
- The other two failures came from a catalog test checking `!busy` before the
  queued confirmation event admitted work. [Repetition](tests/catalog-repeat.txt)
  showed the timing dependence. The test now waits for both token retirement and
  idle state, preserving every outcome assertion; [all six catalog cases pass](tests/catalog-settled.txt).
- [Six preflight guards pass](preflight.txt); the seventh initially lacked its
  network-fetched input, then [passed with the declared pinned inputs](mermaid-preflight.txt).
  Ruff check/format passes for new/changed test and runner files; the large existing
  workbench retains [the same 65 diagnostics](static-analysis.json).
  [Independent review](independent-review.txt) found no blocker, including the
  catalog test's completion condition.

## Native visual and lifecycle evidence

[Sixteen inspected captures](GALLERY.md) cover dark/light at 80×24 and 170×48.
The real TldwCli and recovery owner run under LinuxDriver with all terminal
streams attached, in a fresh private HOME/USERPROFILE/XDG/config profile and an
actual isolated restore. The path fixture includes literal markup and CJK text.
Every message row is collected from the compositor while pressing Down; complete
review text, full action labels and actual hit targets are asserted. Each cell
cancels, reopens, and explicitly confirms fresh Ask/local defaults. Historical
bytes remain intact, only MCP owners are approved, and connection/discovery/tool
execution guards and the network guard record no calls.

[Final run 002](native/result.json) records source hashes, actual imported module
origins and the exact saved runner hash. [Lifecycle](lifecycle.json) verifies
normal return/exit 0, absent process, reacquired lock, ten healthy private SQLite
databases, zero conversations/messages and unchanged user defaults. One Evals
`storage_scope_not_enrolled` diagnostic is inherited from this restored fixture;
Evals recovery is unqualified here. No other app errors; empty faulthandler log.
The first native run also passed; run 002 repeats qualification after runner lint
cleanup. An initial lifecycle reader expected the ordinary profile's DB filename;
its corrected check reads the actual restored ChaChaNotes database.

The gallery is ready for owner visual approval. Current-head CI, accumulated PR
review, current dev/conflict review and final approval remain before merge.
Connected-runtime journeys and other unqualified MCP screens remain separate.
