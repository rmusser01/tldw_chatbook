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
No unresolved local review finding. Qodo follow-ups add all five pending kinds
to Close copy, validate QA arguments before profile mutation, reconcile exact
provisional fleet fences, and route review to the visible decision before a
queued approval. A separate failed-provisional marker blocks unsafe retries
while preserving surviving-child usage in the retained session.

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
- Derived-artifact preflight passes; generated CSS and UI census122 verified.
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

## Review follow-up receipts

- Real mounted Close file: **19 passed**. Final provisional retry/refusal and
  surviving-usage correction: **2 passed**; close/usage boundaries: **21 passed**.
- All three keyboard/Inspector/tab review routes with a real skill confirmation
  and queued approval: **3 mounted cases passed**.
- Native Close runner malformed CLI: **7 passed** without traceback or profile writes.
- Independent review repeated full-finalize usage 120→165 and same-generation
  retry refusal; no remaining actionable finding.
- Diagnostic inventory reviewed against latest dev: one new rollback warning
  records only `type(exc).__name__`, with no exception text, user content, paths
  or secrets and no new sink. Its existing owner pin was regenerated after
  reviewing /tmp/console-tool-ux-diagnostic-review.txt.
- Preceding combined Console, Close, controller-attribute and latency invocation:
  **44 passed**, one expected empty-exemption parameter set skipped;
  /tmp/console-tool-ux-final-dev-qualified.txt. Fresh final preflight passes:
  /tmp/console-tool-ux-final-dev-preflight-qualified.txt.
- Latest-dev combined token/startup-budget checks: **9 passed**. Declaration-neutral
  whitespace in the owning sheet preserves the unchanged 608,090-byte cap;
  final census is **608,077 bytes**. Dev's model-switcher and both new UI suites
  remain in the 122-file census.

Receipts: /tmp/console-tool-ux-close/qodo-{close,usage,usage-boundary,args}-green.log,
/tmp/console-tool-ux-labels/qodo-focus-final.log,
/tmp/console-tool-ux-latest-dev-tokens.txt. No full-suite sweep.

## Latest-dev integration

The clean rebase onto dev84247cb843 includes the healthy-run/provider-readiness
fix from PR #2948. Shared pending-kind, focus and layout behavior is retained.
Compact approval layout, startup ratchets and design-token governance pass:
**26 tests**, /tmp/console-tool-ux-dev842-layout-qualified.txt. Fresh preflight
passes, /tmp/console-tool-ux-dev842-preflight.txt; Ruff comparison against this
base remains identical with zero new diagnostics in 16 modified Python files.
All six mounted pending-projection and latest-dev readiness cases pass, including
the healthy-run, held-regenerate and missing-key control. Exact-node private
child receipts are /tmp/console-tool-ux-combined-rfotoln6/run-000..005/pytest.log;
aggregate /tmp/console-tool-ux-labels/combined-readiness.log. Each child uses
the designated temporary profile and disables pytest caching.
Native approval run08 and Close run9 requalify the changed Console screen;
source hashes match the combined tree and the real profile stays unchanged.

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
Matching terminal text is retained. Approval choices use Pilot key events in the live native app; Close uses tmux SGR mouse input. Raster previews were inspected locally; Cairo font fallback is not pixel-perfect terminal evidence. Attempts01–08 remain under
/tmp/console-tool-ux-approval-native-*; 08 requalified all nine journeys after
the Qodo fixes and healthy-run/readiness dev84247cb843 rebase. Exported source hashes match that tree.

[Close report](2026-10-01-console-close-followup/README.md): four real decision
worker closes, owning task cancellation, sibling isolation and explicitly
synthetic maximum-risk geometry. No provider/server execution is claimed
for Close. Final run9 requalified the four real closes and all-five-kind geometry
after the Qodo fixes and healthy-run/readiness dev84247cb843 rebase, including target fleet/wake fence release. Its
isolation receipt verifies the real profile was unchanged.

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
