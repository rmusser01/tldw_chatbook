# Console tool follow-up verification — 2026-09-29

Scope: TASK-18920 optional denial reasons and TASK-33562 partial tool output.
TASK-33082 was already merged in PR #2908; its affected approval/audit checks
passed separately (11 passed).

## Automated evidence

- Final focused run: **48 passed**. Covers normalized/quoted reasons, real
  controller/model/provider denial flow, two simultaneous rounds with reused
  call IDs, all mounted Deny paths, held skill stdout/stderr and UTF-8 decoding,
  real concurrent native MCP progress, bounded observation and worker timeout,
  visible disclosure focus, and the unchanged startup CSS byte budget.
- Broader affected run: **301 passed, 1 failed**. The new blocked-observer test
  failed before reaching its callback. Its corrected setup explicitly holds the
  callback before checking scope close and passes in the final focused run.
  No runtime change was needed. This run also covers the existing runtime review
  hooks, MCP/virtual owners, live/final transcript identity, and token governance.
- The broader run kept the collection-time disposable profile using the existing
  `bootstrap_profile` marker. Six legacy transcript tests initially tripped
  `raw_source_selection_changed`; current and unchanged renderers both pass
  those six when run with the same selected profile.
- Startup module census: **1030 / 1033**, passed. Other selected boot guards
  passed; the final CSS-byte guard passed after consolidating identical approval
  declarations. The ratchet was not raised.
- New Python modules pass Ruff; changed-line checks in existing files report no
  findings. Changed Python ranges were formatted. `git diff --check` passes.
- `scripts/preflight.sh` passes its derived-artifact and repository guards.

Reproduction: the final focused run selected
`Tests/Agents/test_approval_denial_reasons.py`,
`Tests/Agents/test_tool_output_stream.py`,
`Tests/Chat/test_console_denial_reason_flow.py`,
`Tests/MCP/test_tool_progress.py`, `Tests/Skills/test_skill_script_runner.py`,
`Tests/UI/test_console_denial_reasons.py`, and
`Tests/Performance/test_boot_css_byte_budget.py` with `PYTHONPATH=.`,
`pytest -p no:cacheprovider`, and a disposable `--basetemp`.

## Native evidence

Owned native terminal sessions at **80×24** and **100×40** loaded the changed
controller/runtime/producer modules from this worktree, using Textual's
LinuxDriver and an actual TTY. Keyboard entry submitted a per-row denial reason;
a real held fixture subprocess then supplied stdout/stderr before completion.
Expanded Live output, collapse, authoritative final replacement, and clean
application exit were verified. Both receipts are `{"returned": true,
"result": 0}`. Captures were visually inspected.

Local receipts and captures: `/tmp/console-followups-native/run5` (80×24) and
`/tmp/console-followups-native/run6` (100×40). These are disposable verification
artifacts, not committed user data. The harness mounts the production widgets
and drives the real controller and producer; it does not qualify full Console
application navigation or an external model provider.

## Existing limitations

The full repository suite was not run. An unchanged MCP client reproduces
16 failures in legacy bare-client fixtures missing ProducerLifetime. An unchanged
approval widget reproduces three legacy checks: two stylesheet-literal assertions
and the full-app 80×24 navigation test landing on nav-home. These were not counted
as passing feature checks or silently removed from the suite.

Read-only reviews found and verified fixes for denial propagation through virtual
owners, interrupted-output retention, blocking observer cleanup, and final-only
fallback when the dispatcher cannot start. ADR-067/195 govern denial scoping;
ADR-205 records the new session-only producer/observer contract.

Qodo review remediation: explicitly typed the verdict map, reused the named output cap,
and moved MCP progress into a strict shared Pydantic model. Integer counter precision
is retained; malformed strings, booleans, nonfinite numbers and overflowing integers
are quietly rejected without logging validator bodies. The final focused selection
above passes all 48 checks after these changes.


## CI repair and latest-dev qualification — 2026-09-30

Required UI CI passed 840 cases and failed two mixed-submit ownership cases.
Both failures reproduce: their namespace fixture bypassed widget initialization
and omitted the new reason input/collector. The repaired fixture binds the real
collector and asserts that the reason input locks before publishing and that
approval excludes its text. The derived-artifact job failed only its UI-result
aggregation step; every artifact/inventory guard passed.

Rebase onto dev `dfee4bf4c6` preserved both implementation patches unchanged
(`git range-diff`). Approval/denial-flow qualification: **42 passed**. Streaming
and startup qualification: **34 passed**, with the CSS-byte case initially
over by 65 B. Removing 71 whitespace bytes from existing approval rules and
rebuilding preserved every non-whitespace source character; the final UI, token
and CSS-budget selection then passed **39 checks**. Boot CSS is **608,084 /
608,090 B** and module census **1031 / 1033**. No budget was raised. Latest-dev
preflight passed; the rebuilt stylesheet's sync guard is checked before push.
