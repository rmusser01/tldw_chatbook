---
id: TASK-18932
title: 'Console: toggleable live reasoning display'
status: To Do
assignee: []
created_date: '2026-08-19 09:55'
updated_date: '2026-08-19 09:55'
labels:
  - console
  - ux
  - streaming
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Surface provider reasoning/thinking streams in the Console transcript — identified as a major gap in the 2026-08-19 hermes-release review (hermes streams thinking live by default; chatbook discards the display entirely). Render reasoning as a visually distinct, dim, collapsible block attached to its assistant reply while it streams. Gated by a persisted toggle (Settings ▸ Console Behavior), default decided in implementation (recommendation: ON where the provider streams reasoning on the wire). Scope honestly per provider: only providers that return reasoning content on the wire can display anything (e.g. GLM reasoning via the hosted path); Kimi K3 preserved-thinking is private by existing contract and must never be displayed; no placeholder or fabricated "thinking" for providers without wire reasoning. Must not disturb the existing private continuation-data design (reasoning replay stays omitted from transcript/logs/exports per the QwenCloud/Kimi/Z.ai contracts).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A persisted setting toggles reasoning display; changing it applies to subsequent streams without an app restart
- [ ] #2 While streaming, reasoning renders live in a dim, visually distinct collapsible block attached to its assistant reply; the collapsed-by-default or expanded-by-default choice is pinned and tested
- [ ] #3 Display is presentation-only: shown reasoning never changes what is sent to the provider beyond what the provider's own replay contract already requires, and it never appears in exports, summaries, search, or usage displays unless it is already stored content today (pin the storage rule: recommendation — display-only, not persisted, unless a provider contract requires storage)
- [ ] #4 Provider honesty: models/providers without wire-level reasoning show nothing (no placeholder); private reasoning (Kimi preserved thinking) is excluded by contract and covered by an explicit test
- [ ] #5 Tests cover toggle persistence, live render, collapse interaction, per-provider gating, and non-leakage into exports/search
- [ ] #6 The user guide documents the toggle and the per-provider honesty rules (which providers can show reasoning, which never can and why)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A.
Reason: presentation-layer change over reasoning content that already flows through the provider seams (zai/moonshot/hosted_chat reasoning_content handling); the existing provider contracts (ADR-063 hosted-provider wire and durable tool continuation) already govern what reasoning is retained and replayed — this task displays only what those contracts already allow.

1. Inventory which providers surface reasoning_content on the streaming path today and thread it into the transcript stream events
2. Transcript block (dim, collapsible) + streaming render
3. Settings toggle (Console Behavior) with persistence
4. Privacy tests (no export/search/summary leakage, Kimi exclusion), provider gating tests, docs
<!-- SECTION:PLAN:END -->
