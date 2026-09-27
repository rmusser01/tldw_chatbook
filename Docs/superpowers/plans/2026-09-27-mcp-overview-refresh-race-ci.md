# MCP overview refresh race — TASK-32917.1

ADR required: no
ADR path: N/A
Reason: Routine lifecycle repair of the existing Servers-mode canvas; data,
authorization, service and visual contracts remain unchanged under ADR-150/161.

## Stage 1: Reproduce the failing refresh
**Goal:** Explain the PR Fast Lane failure without timing retries.
**Success Criteria:** A controlled overlap reproduces duplicate callout IDs.
**Tests:** Held-removal overlap cases for built-in Enable and recovery buttons.
**Status:** Complete

Run 36333168869 on PR #2822 passed 1167 cases but failed the stale Settings
tool-profile deep-link test. Its resize worker overlapped a workbench refresh
in `MCPServersMode.update_overview`; both removed before either mounted, then
both mounted `mcp-builtin-enable`. The existing sequential refresh regression
did not exercise independently scheduled callers. Controlled overlap tests
reproduced duplicate Enable and recovery IDs; cancellation of a queued refresh
also failed to exercise a serialized replacement (three red, one green).

## Stage 2: Serialize callout replacement
**Goal:** Keep removal through mounting exclusive per Servers-mode widget.
**Success Criteria:** Overlapping refreshes show only latest callouts; active
or queued cancellation cannot strand subsequent refreshes.
**Tests:** Four controlled overlap/cancellation cases.
**Status:** Complete

One `asyncio.Lock` surrounds awaited callout removal, construction from the
latest shared snapshots, and batch mounting. The async context releases on
cancellation. Four controlled cases pass after the fix. Table rendering,
permission gates, style tokens and visibility policy are unchanged.

## Stage 3: Qualify and integrate
**Goal:** Verify the actual failing path and related overview interactions,
then require fresh final-head CI and review before merge.
**Success Criteria:** Relevant overview/workbench cases pass, artifacts and
security checks pass, independent/Qodo review is clean and GitHub gates pass.
**Tests:** Overview, callout, resize, readiness, source/form and Settings
deep-link cases; inventory and task guards; lint baseline comparison, scoped
format checks, compilation, Bandit and diff checks; required remote CI.
**Status:** In Progress

A broader Servers-mode plus deep-link probe passed 65 cases and failed three
untouched gate-display tests. Those same three failures reproduce on exact
dev source/tests at `c041b6d81`: their legacy per-setting getter fakes no longer
configure the current batched gate snapshot. This repair does not change those
fixtures or gates. The controlled CI failure is independently reproduced and
fixed; no assertions, test collection or CI gate are weakened.

The existing inventory reproduces unchanged (603 owners, 1395 TASK-492 calls,
56 TASK-31551 calls, 7545 TASK-494 calls, 14 sinks); the task guard passes 4438
files. Compilation, changed-range formatting and diff checks pass. Bandit
reports zero findings. Ruff comparison against HEAD shows no new diagnostics;
the two touched import blocks are sorted, and existing unrelated findings are
preserved. The targeted overview/callout, resize/readiness, source/form and
Settings deep-link run passes **45 cases**. Independent review and fresh
final-head remote CI/Qodo remain pending.
