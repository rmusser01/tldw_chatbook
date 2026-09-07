---
id: TASK-1990
title: 'Console transcript: full markdown rendering for assistant replies (streaming-aware)'
status: Done
assignee:
  - '@claude'
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
- [x] #1 Assistant replies render fenced code blocks, tables, ordered/unordered lists, blockquotes and links with real structure (no literal marker characters) in the live Console
- [x] #2 Streaming applies deltas incrementally — no full-message re-parse per sync tick (asserted by test, as in the spike smoke test)
- [x] #3 USER, SYSTEM and TOOL rows keep verbatim text rendering (their #/**/backtick characters may be literal and meaningful)
- [x] #4 Tail-follow, click-to-select, j/k selection, the action row, and plain-text export behavior are unchanged
- [x] #5 Attachment chips, citation notices and variant/sibling headers render with parity to current assistant rows
- [x] #6 Link clicks follow an explicit policy (open browser or no-op) — never silent partial handling
- [x] #7 Live tmux capture at 235x52 recorded before merge, including a mid-stream capture (render-verification convention)
<!-- AC:END -->

## Implementation Plan (the how)

1. Rebase the live-verified spike (`spike/console-markdown-stream`) onto post-wave-1 dev (prune window TASK-1365 + diff rows TASK-1366 landed since; pruning measures `outer_size.height` per mounted row and groups by message id, so variable-height markdown rows compose with it unchanged).
2. Replace the spike env gate with `[chat_defaults] assistant_markdown` (default true), resolved via a pure `get_console_assistant_markdown(app_config)` mirroring `get_console_prune_watermarks`; document in the config template.
3. Parity: dim role/status header (sibling i/n, Generating…, [streaming]/[stopped]/[failed]); chips + citation notice in a dim footer Static below the body (never in the markdown source — would break prefix appends); selected-state classes unchanged.
4. Link policy: `open_links=False`; http(s) → system browser + notify; other schemes notify-only.
5. Renderer-agnostic row-text helpers in the test suites that queried rows as `Static`.
6. Live tmux verification, both gate positions, against a local OpenAI-compatible SSE stub.

## Implementation Notes

Assistant rows render via `ConsoleMarkdownMessage` (Vertical: header Static + `Markdown(open_links=False)` + footer Static), streaming prefix-diffed deltas through `Markdown.append()` — O(delta) per 0.2s sync tick, full `update()` only on non-prefix changes (variant switch/edit); identical content is a no-op (no completion blink). USER/SYSTEM/TOOL rows keep the verbatim renderer; `to_plain_text` exports are unchanged. Config-off restores the TASK-372 span renderer (kept for roleplay flavor colors) — both paths live-verified in tmux at 235x52 against a local SSE stub (mid-stream capture shows the half-streamed fence rendering as a code block; config-off capture shows literal ``` and pipe rows again).

Files: `Widgets/Console/console_transcript.py` (widget + gate), `config.py` (template), new `Tests/UI/test_console_transcript_markdown_widget.py` (7 tests: gate resolution, role scoping, append-only streaming, footer, link policy, export stability), renderer-agnostic helpers in `test_console_native_transcript.py` / `test_console_transcript_pruning.py` / `test_console_native_chat_flow.py`. 412 tests green across the transcript/chat-flow suites; 31,443 collected clean. Pre-existing failure noted (not this change, fails identically on clean dev `75bc25db3`): `Tests/Chat/test_console_generation_actions.py::test_generate_image_handler_restores_draft_when_batch_raises` (`ChatScreen._nodes` AttributeError).

Trade-off recorded: markdown mode drops the span renderer's speech/action flavor colors for roleplay sessions; the config switch is the escape hatch. Frogmouth-inspired follow-ups remain tasks 1991-1995.
