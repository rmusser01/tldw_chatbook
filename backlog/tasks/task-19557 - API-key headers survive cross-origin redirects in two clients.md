---
id: TASK-19557
title: >-
  API-key headers survive cross-origin redirects in two clients
status: To Do
assignee: []
created_date: '2026-08-21 20:07'
labels:
  - security
  - credentials
  - http
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 2 (security & privacy) — its **Tier 1
#3 and #4**, filed together because they are one bug shape in two clients.
Re-verified at this branch base.

Both `httpx` and `requests` strip only `Authorization` (and `Cookie`) when a
redirect crosses origins. **Custom API-key headers are not stripped** — they
are forwarded to whatever host the redirect names.

**Client 1 — the tldw server client.**
`tldw_chatbook/tldw_api/client.py:1144` sets `headers["X-API-KEY"] =
self.token` on a client constructed at line 1149 with
`follow_redirects=True`. `api-key` is the **default auth mode**, and this
client backs Notes, Chatbooks, Character, Sync and MCP traffic. **No test pins
this.**

**Client 2 — Anthropic.** `LLM_Calls/LLM_API_Calls.py:1434` sets
`"x-api-key": final_api_key` with the same exposure (also
`LLM_API_Calls.py:1570`, and the same shape in
`LLM_Calls/Summarization_General_Lib.py`).

The sharpest detail: **the fix is already present one function over.**
`LLM_API_Calls.py:3528-3549` handles this correctly and carries a comment
naming the hazard. This is, again, a seam that never adopted a primitive the
repo already owns — not a missing idea.

A user pointed at a hostile or compromised endpoint (or a legitimate one that
starts redirecting) hands over their API key.

## Acceptance Criteria

- [ ] Neither `X-API-KEY` nor `x-api-key` is sent to an origin other than the
      one the request was authorized for — on redirect, the credential header
      is dropped or the redirect is refused
- [ ] The repair reuses the existing correct pattern at
      `LLM_API_Calls.py:3528-3549` rather than introducing a second mechanism
- [ ] Applied to both clients: `tldw_api/client.py` and the Anthropic call
      sites in `LLM_API_Calls.py` / `Summarization_General_Lib.py`
- [ ] A regression test drives an actual cross-origin redirect and asserts the
      credential header is absent on the second hop, for each client
- [ ] The test is mutation-checked: restoring the header makes it red
- [ ] The remaining custom-credential headers in the codebase are swept for the
      same shape and either fixed or recorded as clean
