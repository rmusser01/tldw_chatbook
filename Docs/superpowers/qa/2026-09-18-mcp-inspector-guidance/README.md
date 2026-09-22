# MCP inspector guidance ownership — TASK-32836

Current integration: [fresh dev review and evidence](current-dev/README.md).
The sections below retain the original 2026-09-18 checkpoint and counts; their
base, source hashes, draft status and follow-up statements are historical.
The maintained native runner now uses the shared admission/bootstrap contract.

Opening a detail hid the readiness badge but left its explanation and action
buttons above unrelated content. The existing visibility owner now controls
all three widgets together. Tool, permission, Audit and finding detail all hide
server guidance; clearing the last detail reveals its current content/actions.
Background readiness updates remain content-only. No action routing, settings,
permissions, CSS or design-token values change. This applies the existing
TASK-2270 detail-context contract and ADR-150/161; no new ADR is required.

## Verification

[62 distinct targeted cases pass](qualified-cases.json): nine new regressions,
17 existing inspector cases, ten workbench transitions/routing cases and 26
language/component governance checks. [Selected cases](selected-cases.txt) and
[final output](tests/final-001.txt) retain the exact scope. No full suite ran.
New cases cover all four detail types, empty/ready/checking refreshes, rebuilt
buttons excluded from focus, restoration of the current Cancel/action set,
partial clear, and real-app dark/light layouts at 80×24 and 170×48.

The [initial red test](tests/red-001.txt) caught `mcp-inspector-message` remaining
visible after tool detail opened. All [nine new cases then passed](tests/green-001.txt).
The final run has two pytest cleanup warnings for unrelated old temporary
folders; no test failures. All [seven preflight guards](preflight.txt) pass.
[Static analysis](static-analysis.json) adds no diagnostics; the production
module retains its 13 baseline diagnostics. New files and changed helper range
pass formatting. [Independent review](independent-review.txt) found no blocker.

## Native visual and lifecycle evidence

Real TldwCli, LinuxDriver and TTY streams run inside a fresh private HOME,
USERPROFILE, XDG and TLDW profile validated before app imports. The service
execution log receives one synthetic metadata record, `review_metadata`; no
tools execute or external servers connect. The Tools journey uses the real
built-in catalog. No databases were copied into the profile.

Native mode buttons and rail/table selections are activated with Enter after
direct focus. In each of four dark/light × compact/wide cells, the journey
selects Servers/built-in, Audit/the synthetic row, Tools/the first built-in row,
then Servers again. It verifies readiness visibility, excludes hidden actions
from keyboard focus, and checks that the original server explanation returns.
Direct `scroll_visible` reveals the inspector heading/explanation for captures;
this qualifies content ownership, not automatic compact inspector navigation.
Permission/finding details and background refreshes are covered by targeted
tests, not claimed as native connected-runtime journeys.

All [16 captures](GALLERY.md) were rendered and inspected. Audit and tool details
start beneath the Inspector heading without the unrelated built-in explanation
or Add server action. Server guidance returns on leaving detail in every cell.
Baseline compact filter clipping and broader inspector scrolling remain visible
because their separate PRs are not included in this fresh-dev branch.

The [native receipt](native/result.json) and [lifecycle check](lifecycle.json)
record four passing cells, normal App.run return, exit 0, absent process,
released instance lock, ten healthy private databases, zero conversations or
messages, unchanged default config/UI state/runtime policy and no app errors or
faulthandler output. All eleven captured source hashes and the runner hash
match final source. The lifecycle checker needed read-only process access
outside the sandbox; the native app itself did not require a retry.
[Export hashes](export-manifest.json) distinguish original private captures from
repository copies normalized only for trailing whitespace.

## Scope and follow-ups

This independent branch starts at dev `cef6bd2a3e3f0b8de0e166146acb4900ca7ea2b6`,
after PR #2707 merged. Separate [PR #2720](https://github.com/rmusser01/tldw_chatbook/pull/2720)
contains Audit selection and [PR #2721](https://github.com/rmusser01/tldw_chatbook/pull/2721)
contains Audit filter layout; both saved heads had green checks when this review
began. Current-head CI and final visual approval remain required before merging
this follow-up. Next: Audit Open tool / Adjust permission drilldowns, including
missing/stale context. Connected-runtime and wider component reviews stay open.

The [task allocation check](allocation-owner-check.json) confirms sole ownership
across local refs/worktrees; no unrelated working files were included.
