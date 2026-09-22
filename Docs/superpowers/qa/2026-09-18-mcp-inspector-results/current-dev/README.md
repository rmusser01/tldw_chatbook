# PR2719 current-dev result review

PR2719 resumes the saved result repair on dev `9e33252708` (including merged
PR2722 and PR2808). The rebase was conflict-free. Qualified implementation head:
`587931456a960e158e5f758c15d7a2b0b79a8ba0`. Later changes in this continuation
record evidence only. TASK-32913 remains In Progress pending its own owner visual
approval, current-head CI/review and final dev/merge verification.

The inspector now uses its existing result renderer when argument collection
fails. This replaces the previous status, interpretation and raw response while
preserving the draft and captured permission context, without requesting tool
execution. Existing tool/server/profile identity guards remain in place.
Token-backed styles let the Raw response title wrap, remove redundant content
padding and give the compact raw body an eight-row viewport.

## Verification

- [Eight regressions fail before the repair](checks/32913-results-red.txt) and
  [pass afterward](checks/32913-results-green.txt). They cover raw/schema input,
  exact profile context, correction, stale tool results, both themes and sizes,
  literal markup/CJK content, keyboard End, resize and collapse.
- [281 related inspector/Workbench cases pass](checks/32913-result-integration-fixed.txt).
  Running the inspector selection independently exposed a collection-to-fixture
  config source change. Its existing `bootstrap_profile` marker now retains the
  source, like the Workbench harness. Production recovery admission is unchanged;
  [the first failing case also passes alone](checks/32913-isolated-harness.txt).
- [43 layout/design/bundle cases pass](checks/32913-layout-governance.txt).
  One existing dimension-ratchet test fails: one declaration in
  `_agentic_terminal.tcss` and eleven in `_workflows.tcss` are byte-equivalent
  matches on the starting dev. [Baseline comparison](checks/32913-dimension-baseline.json).
  No budgets or governance assertions were relaxed.
- [21 runner admission cases pass](checks/32913-runner-admission-final.txt).
  [No new Ruff diagnostics](checks/32913-static-analysis.json); new files are
  clean. The product and existing inspector suite retain their baseline findings.
- After incoming PR2808, [17 cases pass](integration-checks/32913-dev-integration-tests.txt):
  the eight new regressions plus nine incoming startup/Actor Pack/model cache
  cases. [All nine artifact guards pass again](integration-checks/32913-dev-integration-preflight.txt).
  Total: **362 distinct targeted passes**, plus the one inherited governance
  failure. No full suite ran.

## Native evidence and review

The [final native run](integration-native/result.json) launched a clean checkout
at the qualified head in an exclusively owned private profile, through the real
terminal driver and a real local stdio fixture. All four dark/light × 80×24/170×48
journeys pass: previous success, invalid input, corrected long response and
close/reopen. Eight calls appear in both the wire trace and audit; four invalid
attempts add none. Keyboard interaction reaches the fully painted disclosure
title and the final raw `END` result. The helper explicitly scrolls controls into
view before focusing them; these receipts do not claim automatic focus scrolling
for every transition.

All sixteen final SVG/text capture pairs were inspected. Twelve whole-frame
rasters equal the pre-rebase run; the four light/wide changes are confined to
the top navigation's horizontal scroll position. The inspector views are
unchanged. The approval gallery uses the final post-rebase captures.

[Lifecycle](integration-native/lifecycle.json) confirms clean app return and
exit, stopped fixture process, absent fixture session/profile, released instance
lock, ten healthy private databases, zero conversations/messages, unchanged real
defaults and unrelated sentinels, matching source/runner hashes, empty
faulthandler logs and zero network attempts/errors. The private tmux session was
closed afterward. Earlier `native/` receipts are the pre-rebase qualification.

Independent product, runner and incoming-dev review is clear. Its cleanup finding
was fixed: disconnect/delete success, stopped process, and absent active/pending
sessions and stored profile are asserted before a successful result survives
cleanup. Runner calibration also corrected a fixture assumption (the service
normalizes external results to content) and allowed hit-testing a scrollable
container's actual child. These were QA changes, not product changes.

Raw evidence remains private; exported text paths are normalized. Manifests in
each evidence directory retain original and exported hashes. Shared CLI admission
runs before application imports and rejects reused/unsafe profiles and tmux names.

## Scope and history

The saved PR2719 head is `730e4651f8ac7d5c3efd786443f59ca62e19122e`.
Its old scrolling prerequisite already merged as PR2770/TASK-32882; no stale
PR2718 implementation was imported. Saved task 32833 collided with the landed
Workspace exclusions task, so this continuation uses TASK-32913 after an
all-ref/worktree ID sweep. ADRs 150, 161 and 031 govern the bounded repair; no new
architecture decision or authority/runtime/storage boundary is introduced.

PR2722's approved merge closeout is retained separately. Other MCP and destination
reviews remain open. The old PR2707 heartbeat stays paused.
