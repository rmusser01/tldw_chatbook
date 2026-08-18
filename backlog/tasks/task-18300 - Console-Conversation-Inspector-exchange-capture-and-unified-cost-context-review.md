---
id: task-18300
title: Console Conversation Inspector — exchange capture + unified cost/context review
status: In Progress
assignee: ['@claude']
created_date: '2026-08-18'
labels: ['console', 'ui', 'db', 'transparency']
dependencies: []
---

## Description (the why)

Clicking the Console token/cost chip today opens a numbers-only breakdown; the
full next-send payload viewer hides behind Ctrl+Shift+P; and nothing records
what actually went over the wire on past turns. Users cannot answer "what
exactly is being sent back and forth in my conversation, and what did each
piece cost." Owner-approved spec:
`Docs/superpowers/specs/2026-08-18-console-conversation-inspector-design.md`
(six recorded owner decisions, 2026-08-18).

## Acceptance Criteria (the what)

- [ ] Every Console provider call (each tool-loop iteration, direct sends, and
      the llama.cpp branch) is captured — request and response — and persisted
      locally per turn, anchored to the turn's assistant message.
- [ ] Captured requests are allowlist-sanitized by construction: credentials
      never persist, dropped key names are visible to the user, and binary/
      base64 content is stubbed with mime/size/sha256.
- [ ] Captures survive Stop (partial, marked stopped) and abandoned
      regenerations (kept, marked abandoned); ephemeral sessions never persist
      captures; deleting a conversation removes its captures.
- [ ] Clicking the cost chip (and Ctrl+Shift+P, and the command-palette entry)
      opens one Conversation Inspector with Costs, Exchange, and Next Send
      tabs; the old cost and context modals are retired with their behavior
      pins migrated.
- [ ] Per-piece token figures are labeled estimates alongside the provider's
      reported buckets; turns without captures render an explicit
      "no capture recorded" row.
- [ ] `[console] exchange_capture` kill-switch (default on) disables capture
      end-to-end, read through the resolved settings layer.
- [ ] ChaChaNotes gains a local-only `message_exchanges` table (schema bump
      with migration); no sync_log or FTS coupling; writes are idempotent.
- [ ] Live verification against a real provider covers a multi-call tool turn,
      a mid-stream Stop, an abandoned regeneration, and the kill-switch.

## Implementation Plan (the how)

Execute `Docs/superpowers/plans/2026-08-18-console-conversation-inspector.md`
(11 TDD tasks; re-anchored 2026-08-18 against origin/dev @ 1bdbcac61 —
scoped call signals, PreparedProviderRequest kwargs, schema v40→v41).
