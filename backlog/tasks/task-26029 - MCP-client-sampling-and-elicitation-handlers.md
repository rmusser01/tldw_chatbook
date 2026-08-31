---
id: TASK-26029
title: 'MCP client: sampling and elicitation handlers'
status: To Do
assignee: []
created_date: '2026-08-31 15:46'
labels:
  - mcp
  - interop
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Server-initiated MCP requests are all refused. Verified on origin/dev: MCP/client.py:744,754 answers ping and returns -32601 method-not-found for everything else, so a server that asks the client to run a completion (sampling/createMessage) or to ask the user a question (elicitation/create) simply cannot work. This silently narrows which MCP servers are usable. Chatbook has both halves already: a chat provider for sampling and an approval-card surface for elicitation - the gap is the two handlers wiring them to the protocol.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A server-initiated sampling request is fulfilled through the existing chat provider and its result returned to the server
- [ ] #2 Sampling requests are gated: the user controls whether a given server may request completions, and the default is not silent consent
- [ ] #3 Sampling requests are bounded (rate and token budget) so a server cannot drain the user's account
- [ ] #4 A server-initiated elicitation request is presented to the user through the existing approval-style surface and the response returned
- [ ] #5 Elicitation requests that ask for credentials or free-form secrets are refused rather than presented
- [ ] #6 A declined or timed-out request returns a well-formed protocol error, never a hang
- [ ] #7 Servers that never use these methods are unaffected
<!-- AC:END -->
