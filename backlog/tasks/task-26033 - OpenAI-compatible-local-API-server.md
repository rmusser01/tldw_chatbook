---
id: TASK-26033
title: OpenAI-compatible local API server
status: Done
assignee: []
created_date: '2026-08-31 15:47'
updated_date: '2026-09-02 13:58'
labels:
  - interop
  - agents
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Chatbook's agent cannot be driven by anything but its own TUI. Verified on origin/dev: a named grep for FastAPI, aiohttp.web, uvicorn and starlette across tldw_chatbook returns only doc comments and TTS references; the only HTTP surface is textual-serve rendering the TUI in a browser (Web_Server/serve.py:224,232). Hermes exposes /v1/chat/completions and a run lifecycle so editors and CLIs can drive it. Even single-user and local-first, pointing an editor at your own configured agent is a real capability - and the run loop it would wrap already exists behind AgentService.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A local HTTP server exposes an OpenAI-compatible chat completions endpoint backed by the existing agent run loop
- [ ] #2 Streaming responses are supported in the standard server-sent-events shape
- [ ] #3 The server binds to loopback by default and requires an explicitly configured token; it never starts unauthenticated
- [ ] #4 It is disabled by default and starting it is an explicit user action, not a side effect of installing a dependency
- [ ] #5 Requests run under the same permission gate and tool policy as a Console run - an API caller cannot bypass approvals
- [ ] #6 The listening address, port and auth state are visible in the UI while the server is running
- [ ] #7 Shutting down the application stops the listener
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/<next> - local-api-server-and-auth-boundary.md. Reason: this opens a network surface that can drive the agent and spend the user's tokens; the authentication boundary and what an API caller may do relative to a Console user are decisions to record, not to infer. Sweep backlog/decisions for a free number at authoring time - ADR numbers in this repo collide routinely.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
REJECTED by owner (2026-09-02): a local HTTP API inside the TUI duplicates tldw_server, which IS the API surface of this ecosystem — anything that wants to drive an agent over HTTP should talk to the server, not to a second embedded server in the chatbook process. Filed from hermes parity, not from a real need. The 26041 fold-ins recorded onto this task (inbound webhooks, IPC control surface, non-interactive fail-closed policy, sandbox-urgency trigger) do NOT get a new home: the webhook family was already owner-rejected, IPC/control belongs to tldw_server if anywhere, and the policy/sandbox notes only mattered if an unattended API existed here. The framework question (FastAPI vs aiohttp) is moot.
<!-- SECTION:NOTES:END -->
