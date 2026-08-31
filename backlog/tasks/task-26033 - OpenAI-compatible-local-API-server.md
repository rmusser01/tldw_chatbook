---
id: TASK-26033
title: OpenAI-compatible local API server
status: To Do
assignee: []
created_date: '2026-08-31 15:47'
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
