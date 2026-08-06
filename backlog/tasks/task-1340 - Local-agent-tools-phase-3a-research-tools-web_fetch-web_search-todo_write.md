---
id: TASK-1340
title: 'Local agent tools phase 3a: research tools (web_fetch/web_search/todo_write)'
status: Done
assignee: []
created_date: '2026-08-05 15:09'
updated_date: '2026-08-05 15:10'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec: Docs/superpowers/specs/2026-08-05-local-agent-tools-phases-3-4-replan.md (phase 3a). Plan: Docs/superpowers/plans/2026-08-05-local-agent-tools-phase3a.md. ADRs: 032, 033.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 web_fetch refuses private/loopback/link-local targets and non-http(s) schemes, including on redirect hops
- [x] #2 web_fetch enforces redirect cap, timeout, byte caps, per-domain rate limit, and TTL cache
- [x] #3 web_search delegates to perform_websearch with bounded per-result size
- [x] #4 todo_write mutates per-session state and renders in the transcript
- [x] #5 Agent system prompt hints at find_tools/load_tools discovery
- [x] #6 All new tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-08-05-local-agent-tools-phase3a.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

Implemented on branch `feat/local-agent-tools-p2` (stacked on PRs #1352/#1358) via subagent-driven development with per-task spec + quality review.

- `Tools/web_tool_impls.py` (new): `validate_outbound_url` SSRF guard (scheme allowlist, DNS-resolve, all-IPs-public via ipaddress, per-redirect-hop re-validation, explicit 100.64/10 + 192.0.0.0/24 blocks, IPv4-mapped normalization) and `web_fetch` (ported behaviors from tldw_server @ 5605b9d9 — manual redirect loop, 30s timeout, 1 MiB/5 MiB byte caps with truncation marker, per-domain min-interval rate limit, in-memory TTL cache, trafilatura extraction with regex tag-strip fallback; `trust_env=False` closes the env-proxy bypass). `web_search` core delegates directly to `perform_websearch` with the legacy config wiring, byte-exact 4 KiB/result + 24 KiB total caps, and the `snippet or content` key fix (the legacy tool carries the same latent bug — left for a follow-up).
- `Agents/local_tool_provider.py`: `web_fetch`/`web_search` specs (no tags, network-classed), conditional `todo_write` spec via new `todo_store`/`on_todo_change` seams (spec absent without a store), full todo validation (shape, status enum, at-most-one in_progress, 50-item/500-char caps, key whitelist, validate-before-mutate), mutates tag.
- `Chat/console_chat_store.py`: `ConsoleChatSession.todos` (session-lifetime, never persisted). `Chat/console_agent_bridge.py`: `format_todo_marker` + `append_todo_marker` transcript rendering (200-char/item truncation, newline flattening, markup-off discipline), `FIND_LOAD_DISCOVERY_HINT` appended by `compose_agent_system_prompt` only when the run's catalog crosses DIRECT_DISCLOSE_THRESHOLD ("git" dropped from the hint until 3b-ii lands).
- `Chat/console_chat_controller.py`: `_compose_local_provider(session_id)` wires the session's live todos list + marker callback.
- Tests: SSRF guard incl. bypass-class regressions (decimal/hex/octal IPs, IPv4-mapped, userinfo), hermetic web_fetch tests (MockTransport + stubbed DNS + fake clock), web_search handler tests with real payload shapes, todo provider/controller/bridge tests, e2e find/load runs for web_fetch and todo_write (risk_floored approval path), discovery-hint gating test.

Review-driven hardening beyond the plan: CGNAT/IETF blocks, trust_env=False, bad-port/InvalidURL wraps, snippet-key bug (critical — real payload shape), byte-exact caps, todo bounds/whitelist.

Deviations: web_search bounding lives in a `web_search()` core wrapper in web_tool_impls.py (plan said handler-direct; the cap logic needs a home); todo rendering reuses the bridge's in-memory marker path rather than call_from_thread (same worker thread); disclosure boundary test now uses the real 10-entry default catalog instead of synthetic padding (strictly better).

Final whole-implementation review: Ready to merge; all 6 ACs verified. Worktree runs: 356 passed (Agents+Tools), 1386 passed (Chat+Utils) — only the two known pre-existing base failures.
