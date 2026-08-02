---
id: TASK-1990
title: 'Console transcript: full markdown rendering for assistant replies (streaming-aware)'
status: To Do
assignee: []
created_date: '2026-08-02 22:30'
labels:
  - console
  - chat
  - markdown
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Assistant replies in the Console render through a hand-rolled safe-subset span renderer (TASK-372: headings, `**bold**`, `` `code` `` + roleplay flavor). Everything else arrives as literal text: fenced code blocks show their ` ``` ` fences, tables show raw `|` pipes, lists show raw `-`/`1.` markers, links show `[text](url)`, blockquotes show `>`. The chat-behavior checklist items "Messages support markdown rendering" and "Code blocks are properly formatted" have never been checkable.

A Frogmouth-vs-chatbook comparison (2026-08-02) established that the installed Textual (8.2.7) ships the missing piece Frogmouth predates: `Markdown.append()` / `Markdown.get_stream()` (`MarkdownStream`), purpose-built for LLM streaming — incremental block parsing instead of a full re-parse per tick.

**Spike evidence — branch `spike/console-markdown-stream` (env gate `TLDW_CONSOLE_MD_SPIKE=1`):** assistant rows swapped to a `Vertical(header Static, Markdown)` widget, streaming prefix-diffed deltas through `Markdown.append()`. Live-verified in tmux (235x52, scratch profile, local SSE stub at 0.12s/chunk): fenced Python block, bordered table, ordered/unordered lists, blockquote and link all rendered correctly **mid-stream and at completion**; tail-follow held through growth; click-to-select worked through the Markdown widget (action row mounted); completion caused no re-render blink (identical content short-circuits). Smoke test asserts append-only on prefix growth (zero `update()` calls), full-update fallback on non-prefix edits, and gate-off leaves rows untouched. A/B baseline capture of the identical reply shows the span renderer's literal fences/pipes/brackets.

**Remaining engineering (why the spike is not mergeable as-is):**
- Attachment chips, citation notices, variant/sibling headers and selected-state styling need parity with `ConsoleTranscriptMessage`.
- Safety review: the span renderer guaranteed literal text (styles via `(text, style)` tuples, never markup parsing — Qodo #823). The Markdown widget *parses* assistant content; confirm USER/SYSTEM/TOOL rows stay verbatim and assistant markdown cannot trigger unintended link handling (`open_links` policy needed).
- `to_plain_text` exports and reconciliation signatures must stay exact.
- Long-transcript cost: one Markdown widget per assistant row vs today's single Static; measure mount/scroll behavior on a large session.
- Decide default-on vs a Console setting; the env gate is spike-only.

**Timing constraint:** Console changes intersect the held screen-decomposition program (owner ruling 2026-08-02: execution held until Console churn settles). Coordinate before starting.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Assistant replies render fenced code blocks, tables, ordered/unordered lists, blockquotes and links with real structure (no literal marker characters) in the live Console
- [ ] #2 Streaming applies deltas incrementally — no full-message re-parse per sync tick (asserted by test, as in the spike smoke test)
- [ ] #3 USER, SYSTEM and TOOL rows keep verbatim text rendering (their #/**/backtick characters may be literal and meaningful)
- [ ] #4 Tail-follow, click-to-select, j/k selection, the action row, and plain-text export behavior are unchanged
- [ ] #5 Attachment chips, citation notices and variant/sibling headers render with parity to current assistant rows
- [ ] #6 Link clicks follow an explicit policy (open browser or no-op) — never silent partial handling
- [ ] #7 Live tmux capture at 235x52 recorded before merge, including a mid-stream capture (render-verification convention)
<!-- AC:END -->
