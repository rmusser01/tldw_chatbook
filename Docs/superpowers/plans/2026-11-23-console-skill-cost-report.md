# Console skill cost and usage report implementation plan

> For agentic workers: use superpowers:executing-plans or superpowers:subagent-driven-development.

**Goal:** Implement the current-session Console report in TASK-38931.

**Architecture:** Attach observations to the existing session, using its
workspace and conversation binding revision to reject stale results. Derive
costs from the prepared Context breakdown and use the existing Manual Send
and agent skill runner for observation. Open the report using a Console slash
command and a read-only modal.

**Tech stack:** Python, Textual, pytest.

**Spec:** [Approved design](../specs/2026-11-23-console-skill-cost-report-design.md)

ADR required: yes (amend existing).
ADR path: backlog/decisions/230-console-send-preparation-and-io-ownership.md
Reason: define what metadata crosses the preparation and display boundary.

1. Write focused tests for prepared-cost reconciliation, unavailable evidence,
   accepted usage deduplication and scope changes.
2. Implement the session observations in Chat. Use planner-selected schema
   identities, original request accounting and the accepted invocation paths.
3. Wire the command and modal. Verify command origin, literal rendering and
   dismissal. No transcript entry or provider call.
4. Run targeted Chat/tool-surface/command tests, rebuild and verify CSS, run
   the applicable preflight checks, update the guide and task, and open a PR.
