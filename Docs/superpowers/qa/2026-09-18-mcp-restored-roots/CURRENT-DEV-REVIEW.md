# PR2727 integrated restored-roots review

[PR2749 merged](integration/pr2749-closeout.json) at
`65d79cc2b7bd8b7b1d86ee87cc31f1823475ab3a`. Its actual merge tree equals the
verified, owner-approved Tools header tree. PR2727 is rebased onto that state.
Only the two review ledgers conflicted: each retains all current-dev history and
the saved restored-roots checkpoint. There were no product conflicts.
[Exact choices](integration/rebase-conflicts.json) refer to the resolved rebase
state, before this new checkpoint was added.

The integrated review found a same-mode ownership gap: selecting another
Permissions row or policy profile while an accepted native review completed
could still let its old receipt clear the newer detail or reset the profile.
The four keyboard regressions, including away-and-back selections, all failed
on the rebased baseline and pass after invalidating the existing receipt at the
validated selection boundary. The native write still completes; invalid contexts
and unchanged profile selections retain their existing behavior.
[Red](integration/tests/selection-red-001.txt),
[green](integration/tests/selection-green-001.txt).

## Current evidence

Qodo on `bb824eba9a` found no product bugs and two evidence-runner rule issues.
The entry point now documents its CLI/exit contract and reuses the existing
shared `native_runner_args.py` boundary before app imports or output writes.
Only its returned canonical profile, validated identifiers and PATH-resolved
tmux are used. [Seven entry-point failures](integration/tests/runner-red-001.txt)
are reproduced before the fix; all [70 shared boundary cases pass](integration/tests/runner-green-001.txt).
Together with the unchanged product checks below, 155 distinct cases pass.
The fresh native captures/lifecycle in this packet use the corrected runner.
The first integrated capture set remains in commit `bb824eba9a`.

- [85 targeted cases](integration/tests/integrated-001.json) pass in 203.81s:
  the original 45, four new selection cases and 36 adjacent profile/navigation
  cases. [Exact selection](integration/selected-cases.txt). No full suite ran.
- All [seven artifact guards](integration/preflight.txt) pass.
  [Static analysis](integration/static-analysis.json) adds no diagnostics;
  existing baseline counts are retained. Changed ranges and new Python files
  pass [formatting checks](integration/formatting.txt).
- [Independent review](integration/independent-review.txt) covers the retained
  native write, validation order, selection round trips and regression routing.
- [Sixteen current native views](GALLERY.md) cover dark/light at 120×40 and
  170×48. Both action labels remain visible and reachable by keyboard. Actual
  owner approval creates fresh Ask/local defaults, preserves historical bytes,
  clears old Servers/Audit rows and approves only MCP owners. The first cell
  activates the generation; the others repeat review of that same generation.
- The [native lifecycle](integration/lifecycle.json) records normal App.run
  return, exit 0, absent process, released lock, ten healthy private SQLite
  databases, zero conversations/messages and three unchanged user-default
  files. Native source hashes and runner hash match the current files.

The restored Evals owner's single `storage_scope_not_enrolled` diagnostic is
retained and outside this MCP qualification; there are no other app errors.
Two pytest warnings concern unrelated old temporary-directory cleanup.
An [early runner failure](integration/unqualified-startup.txt) imported a private
image-library helper absent in this environment and never started the app. The
runner now uses the app's own image-protocol warm-up. A fresh profile produced
the qualified evidence; the original runner is retained in
[historical-native-check.py.txt](integration/historical-native-check.py.txt).

## Visual scope and remaining work

This is a receipt-lifetime and passive-view repair under existing ADR-126 and
ADR-150/161. No new ADR, tokens, styles, owner policy or service contracts.
The gallery retains the existing narrow, scrollable long-path review and
persistent restored-review guidance after success. Long-path presentation,
catalog repopulation/status guidance, 80×24 and connected-runtime journeys are
follow-up UX work, not claims made by these captures. No MCP connection,
discovery, execution or permission grant is performed by this journey.

The original evidence below this directory's historical sections qualifies its
older source only. [Current source verification](integration/verification.json)
and [export hashes](integration/export-manifest.json) identify this integration.
PR2727 requires its own current-head CI, accumulated review and final visual
approval before merge. PR2707's continuation heartbeat remains paused.
