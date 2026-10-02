# Console tool UX qualification — 2026-10-01

Scope: TASK-33625.2 approval layout, TASK-33621.16 confirmed Close and
TASK-32367 pending-kind projection. TASK-33625.1 Stop was already merged
in PR #2934; its 33 focused run-control checks pass unchanged.

## Review and implementation

Existing Select/Button instances reflow by card width and the existing
Console compact-height mode. Compact rows use the existing scroll region,
bounded to eight rows; bulk actions remain visible while Tab reveals the
optional reason input. Tall/wide layout retains its original order and
15-row row limit. Height-only resizing preserves choice, focus and identity.

Close names the tab, omits zero consequences, states pending cancellation,
defaults to Stay, and reports/reconfirms failures. Cleanup revokes and wakes
exact-session questions; hosts retain teardown. Inspector uses existing
kind-aware copy and a locked session approval-round count including queued
rounds. Questions and confirmations do not count as tool approvals.

Independent review found crowded Close copy could hide the title at 80×24.
Shorter consequence lines and removing a repeated question corrected the
27/29-row overflow; short and 60-character titles now fit. Final review
checked scrolling reason-input focus, height transitions and control reuse.
No unresolved local review finding.

## Targeted receipts

- Approval/card/denial-reason/budget/token: **52 passed**;
  /tmp/console-tool-ux-approval-qualified-final.txt.
- Final seven compact cases plus UI latency/responsiveness: **25 passed**;
  /tmp/console-tool-ux-height-final.txt.
- Close/impact: **17 passed**; affected routing: **3 passed**;
  final real-kind Close and maximum-risk geometry: **6 passed**.
- Pending-kind projection: **18 passed**, including two real full-app cases
  with worker rounds, queued count 2→1, precedence, sibling isolation,
  navigation and fresh remount.
- Stop: **33 passed**; /tmp/console-tool-ux-stop.txt.
- Derived-artifact preflight passes; generated CSS and UI census121 verified.
  Both new suites are added to that existing gate.
- New Python files pass Ruff/format. Modified legacy files add zero diagnostics;
  [comparison](2026-10-01-console-approval-layout/lint-delta.json).

RED receipts retain actual control clipping, question Inspector mismatch,
orphan-question cleanup and dialog overflow. Three inherited approval-test
assumptions were repaired: extracted CSS ownership, token width and real
private-profile/round ownership. Native refreshed second-tab retries exposed
scope wrapping and short-height chrome pushing Submit off-screen; existing
scrolling/height mode fixed them. Failed receipts remain in /tmp.

CSS whitespace was removed only in the owning source sheet to pay for compact
rules within the unchanged startup budget. Other visual declarations were
retained; generated CSS was rebuilt.

## Native evidence

[Approval result](2026-10-01-console-approval-layout/native/result.json):
**9/9** actual TldwCli/LinuxDriver/TTY journeys at 80×24 Inspect open/closed
and 235×52 Inspect open. At each size, fast Deny and Deny all + Submit return
authoritative refusal; Approve once reads the disposable file through the
production local executor. Deny all leaves its worker pending until Submit.
All five actions have complete painted labels and native hit/clip checks.
App/process exit0; no exception or network attempt; source/runner hashes recorded.

The real controller-composed LocalToolProvider and private permission store
gate fs_read as Ask. A disposable workspace is supplied explicitly at provider
composition; no accepted model turn or external MCP server is claimed.
Session projection is synchronized and its pending count asserted before capture.
[80×24 denial](2026-10-01-console-approval-layout/native/80x24-inspect-True-deny-all-decision.svg),
[80×24 Inspect closed](2026-10-01-console-approval-layout/native/80x24-inspect-False-approve-once-pending.svg),
[235×52 wide](2026-10-01-console-approval-layout/native/235x52-inspect-True-fast-deny-pending.svg).
Matching terminal text is retained. Approval choices use Pilot key events in the live native app; Close uses tmux SGR mouse input. Raster previews were inspected locally; Cairo font fallback is not pixel-perfect terminal evidence. Attempts01–06 remain under
/tmp/console-tool-ux-approval-native-*; 06 qualified the final patch.

[Close report](2026-10-01-console-close-followup/README.md): four real decision
worker closes, owning task cancellation, sibling isolation and explicitly
synthetic maximum-risk geometry. No provider/server execution is claimed
for Close. Its isolation receipt verifies the real profile was unchanged.

## Inherited optional-governance failures

Two broader component-pattern checks fail on original dev83c2c9810d:
dimension_literal_ratchet has14 unchanged diagnostics in agentic/settings/
workflows sheets; python_style_ratchet flags an unchanged Library skill-pane
width assignment. The [comparison](2026-10-01-console-approval-layout/baseline-governance.json)
proves identical matched declarations and byte-identical Python source.
This patch introduces neither violation. Checks were not removed, relaxed
or suppressed; affected token/budget checks pass. No full-suite sweep.

ADR required: no. Existing ADR-043/150/161 layout, ADR-067/094 lifetime and
ADR-195 live-tool boundaries apply; task plans link them. Final PR checks
and merge verification are tracked in the PR.
