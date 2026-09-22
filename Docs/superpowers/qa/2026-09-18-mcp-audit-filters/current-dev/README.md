# Audit filters — current merged-dev qualification, 2026-09-21

This is the pre-approval checkpoint. The [approved, rebased Qodo closeout](final-review/README.md)
contains the latest qualification and normalized publication receipts.

PR2721 / TASK-32835 resumes from PR2720's actual merge
`7a758b8196bd3083ae9f878551d01dde7f4ec1ad`, also the latest fetched dev at
publication preparation. Product/CSS commit `78b6d27bf3` reapplies saved commit
`8ba5f72f0b6c045756e6b1761e83bbf819410744`. The native run also includes the
uncommitted runner/admission modernization listed in [launch.json](launch.json).
All [15 captured source and runner hashes](source-verification.json) match the
published sources; subsequent changes only retain evidence and update notes.

The bounded change stacks full-width Audit filters using existing tokens and
reveals the current focused control/table cursor after focus or resize. No token
values, selection identity, permission policy, execution behavior or budgets change.
Existing ADR-150 and ADR-161 apply; no new ADR is required.

## Verification and limits

**294 distinct targeted cases pass; one pre-existing governance case fails.**
No full suite was run.

- [110 integration cases](tests/integration.txt): 10 real-app layout cases,
  69 legacy Audit cases, 15 mounted selection cases, 8 isolated identity cases
  and 8 design-token guards. The layout matrix covers dark/light at 80×24,
  100×30, 120×40 and 170×48, with painted labels and keyboard/resize behavior.
- [133 shared runner-admission cases](tests/admission-green.txt) pass.
  The new Audit-filter runner first produced [seven expected failures](tests/admission-red.txt)
  with its obsolete CLI. Its [final seven-case rerun](tests/admission-final.txt)
  also passes; these repeats are not counted again.
- [51 governance/build cases pass and one fails](tests/governance.txt).
  `test_dimension_literal_ratchet` reports one declaration in
  `components/_agentic_terminal.tcss` and eleven in `features/_workflows.tcss`.
  The [same test regex applied to base and current source](upstream-dimensions.json)
  proves all twelve are unchanged from merged dev. No allowance was widened.
  This is retained upstream debt, not a green governance claim.
- All [nine derived-artifact preflight guards](preflight.txt) pass, including
  CSS and widget-bundle reproduction, Canvas assets and backlog identity.
- [Ruff](static-analysis.json) introduces no diagnostics: the legacy production
  file retains two and legacy tests retain three. New/updated runner and new
  tests are clean; all [14 file/changed-range formatting checks](formatting.json)
  pass. [Independent review](independent-review.txt) found no actionable issue.

PR2720's separate [actual-merge closeout](../../2026-09-18-mcp-audit-selection/merge-closeout/README.md)
records five inherited code-size budget failures already owned by TASK-32809.
Those files and allowances are unchanged by this filter repair. These receipts
do not claim full-repository or full-architecture qualification of PR2721.

## Native review

[All 18 captures](GALLERY.md) were rendered and inspected, including compact
keyboard menus. The real TldwCli uses LinuxDriver and real TTY streams with
private HOME/USERPROFILE/XDG/profile paths admitted before app imports. Its
actual execution-log service reads 48 synthetic metadata records; the journey
executes no tools and connects no servers.

At 80×24 and 170×48 in both themes, terminal keystrokes type `review`, select
`Blocked (kill switch)` and `Test`, traverse controls with Tab and reach the
last of 24 filtered rows with Ctrl+End. Full labels paint and all filter values
persist. Compact result access scrolls the execution pane as intended. The
first text field is focused directly; subsequent navigation uses keystrokes.

The [result](native/result.json) and [lifecycle receipt](lifecycle.json) verify
normal App.run return, exit 0, absent process, released instance lock, ten healthy
private databases, zero conversations/messages, unchanged default config/UI state/
runtime policy and fixture sentinels, matching source/runner provenance, zero
network attempts and no error/faulthandler output. The private terminal session
was closed after qualification. [Export hashes](export-manifest.json) distinguish
original captures from copies normalized for trailing whitespace and subsequently sanitized for host paths.

## Conflict choices and remaining gate

Only the two review-history documents conflicted. Both histories were retained:
current merged-work checkpoints remain at the top; saved filter checkpoints are
explicitly historical. Product code and source styles applied without conflicts;
generated styles were rebuilt and reproduced from source.

PR2720's owner approval does not approve this new visual slice. TASK-32835 stays
In Progress until fresh visual approval, current-head CI, accumulated Qodo review
and final current-dev/conflict checks are complete. PR2722 inspector-guidance
ownership and other component reviews remain separate. The guidance visible above
unrelated Audit detail in these captures is the known next slice, not a new claim
that inspector review is complete.
