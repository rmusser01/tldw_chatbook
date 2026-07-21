---
id: TASK-350
title: Fix sub-agent run rendering - duplicated answer text and mid-word truncated tool entries
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The first ordinary prompt was routed through a 'spawn_subagent' tool call. Consequences in the transcript: (1) the identical 600-word answer appears twice back-to-back - once as the italic tool result ('spawn_subagent -> ### Understanding SQLite...') and once streamed into the assistant bubble; (2) the tool-call args and tool result are hard-truncated mid-word ('the traditional rollba', 'the main database remains in') with no ellipsis and no expand affordance; (3) all this happened while the header chips claimed 'Tools: 0 ready' and 'Approvals: 0 pending' - only the Inspector reveals 'MCP: 10 tools ready'. The hidden sub-agent also made the perceived first-token latency ~100s because its own streaming is invisible.

**Repro:** Fresh provider home -> ask the 600-word WAL question -> model spawns a sub-agent -> compare the tool-result block with the assistant bubble below/above it; read the 'Tools: 0 ready' chip vs Inspector 'MCP: 10 tools ready'.

**Verifier note:** All three sub-claims verified in j4-33 + code: (1) duplicate full answer is structural — format_agent_step_marker renders '⚙ {tool_name} → {result}' with the UNtruncated tool result (console_agent_bridge.py:145-148), so a spawn_subagent whose result IS the answer prints it twice; (2) spawn summary cut mid-word with no ellipsis ('the traditional rollba'); (3) header chip 'Tools: 0 ready' vs Inspector 'MCP: 10 tools ready' — chip counts _console_tool_count only, MCP counted separately. Not prior art: task-231 was an efficiency review (Done, spawned perf tasks 243-245 only); mcp-chat-bridge-deferred no longer applies since the agent runtime made MCP tools live in Console. The auto-spawn routing itself is model behavior under the agent operating prompt, not a UI defect — the transcript/chip consequences are the finding.

**Source:** Console UX expert review 2026-07-20 (finding j4-subagent-detour-duplicate-text-truncated-tools; P2, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J4 streaming journey. Evidence: `j4-10-reply1-complete.png`, `j4-14-after-stop-settled.png`, `j4-32-inspector-midrun.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tool provenance should be collapsed/summarized (not duplicate the full answer), truncation should be marked with an expand affordance, and the Tools chip should reflect the tools that can actually run
<!-- AC:END -->
