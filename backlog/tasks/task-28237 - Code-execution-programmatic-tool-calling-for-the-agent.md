---
id: TASK-28237
title: Code execution / programmatic tool calling for the agent
status: To Do
assignee: []
created_date: '2026-09-02 06:39'
labels:
  - agents
  - tools
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Hermes-parity deferred row C6, promoted by TASK-26041's review: still the highest-leverage unfiled row. Let the model write a short script that composes existing tools programmatically instead of one JSON round-trip per call. Both preconditions matured on dev: registry.invoke_by_name is the single tool-dispatch choke point (Agents/agent_service.py:3013, run_tool_policy.py:5), and sandboxed skill-script execution already exists with discard-writes semantics (Agents/tool_catalog.py:458). This composes them: a script runtime whose tool bindings route through the SAME permission gate as individual calls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A script can call multiple registered tools in one execution, each call passing through the existing permission gate and execution log
- [ ] #2 Script execution reuses the existing sandboxed skill-script runtime; no new execution surface
- [ ] #3 A gated tool inside a script raises the same approval card as a direct call; denial fails the script honestly
- [ ] #4 Resource bounds (wall time, output size) match or tighten the existing per-call bounds
<!-- AC:END -->

## Renumbering provenance

This task previously held id TASK-28227, colliding with the
"Agent-loop-active-turn-redirect-keeping-completed-tool-results" task that arrived on origin/dev first (dev minted 28226-28228
within the hour after this batch's sweep; re-verified at the wave-2 dev merge,
2026-09-02). Per the TASK-19601 owner rule the younger task renumbers with
provenance; it is now TASK-28237.


## Design review 2026-09-02 (deferred, security finding)

Brainstormed toward a spec; deferred after a design review surfaced a blocking
issue. Decision: the "live tool bridge" (script runs sandboxed, calls tools via a
gated request/response channel, branches on results) is the right capability
(matches hermes / Anthropic code-execution-with-MCP) over the one-way
plan-then-execute alternative. BUT:

- **The gate is bypassable for model-authored code.** The reused sandbox
  (Skills_Interop/skill_script_runner.py, run_script_subprocess) is
  best-effort only: setrlimit + scrubbed env + discarded scratch writes, with
  NO network jail and NO filesystem-read jail (no namespaces/seccomp). That is
  acceptable for run_skill_script because skills are user-vetted and confirmed
  per run. 28237 runs code the MODEL authored, which can open('/etc/passwd')
  and open a network socket directly in Python -- never touching invoke_by_name,
  so no gate, no approval card, no execution-log entry. So AC#1/#3 ("every call
  through the gate") governs only the explicit tools.x() calls; direct file
  reads and network egress route around the gate. That is a privilege
  escalation past the gate if the gate is meant to be a real boundary.

Resolution options recorded: (1) treat run_agent_script as one high-trust
gateable builtin (OFF by default) that raises a per-run approval card like
run_skill_script, and spec honestly that the sandbox is not network/fs-read
jailed (inner per-call gate = defense-in-depth + logging for mutating tools);
(2) add real OS confinement (Linux netns+seccomp+fs view) -- heavy, Linux-only,
macOS dev loses it; (3) a non-arbitrary tool-only DSL -- hard to make safe in
Python, kills "let the model write a script". Owner chose to DEFER and revisit;
a future spec should start from option (1) unless real OS confinement is wanted.

Other design points already settled for the eventual spec: reuse the existing
per-call invoke_tool(ToolCall) dispatch closure rather than reimplementing the
gate/approval/log/cross-thread path; exclude meta/runtime tools
(run_agent_script, run_skill_script, sub-agent spawn, human-input) from the
script-exposed set to prevent re-entrant fork-bombs; strictly serial,
length-framed request/response over a fixed pass_fds socket with a `tools` stub
in the scratch dir; script calls consume the run's RunToolPolicy caps + a new
per-script max-calls ceiling; approval card gains "allow this tool for the rest
of this script run".
<!-- keep task To Do; this is the review trail, not completion -->
