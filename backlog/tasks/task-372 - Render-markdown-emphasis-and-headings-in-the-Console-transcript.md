---
id: TASK-372
title: Render markdown emphasis and headings in the Console transcript
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Section headings requested in the prompt arrive as markdown and are displayed raw: '### Understanding SQLite Write-Ahead Logging (WAL) Mode', '#### The Checkpointing Process', etc., in both the live-streamed assistant bubble and the persisted transcript. Long replies with many sections read as noisy plain text; other inline markdown (backticks) is also shown literally.

Also observed independently in J2 returning power user as `j2-markdown-asterisks-raw`: Assistant replies show raw markdown markers ('**local RAG**') in the transcript.

**Repro:** Ask for any answer 'with section headings' -> observe literal #/## characters in the transcript.

**Verifier note:** Confirmed in j4-33/38 (literal ####). The Console transcript deliberately renders markup-off plain text (Content.assemble, never markup-parsed — documented in console_agent_bridge.py's marker docstring and the transcript-visual ledger item's 'plain text unchanged'), but that is an injection-safety implementation choice; no ledger/backlog decision says assistant markdown must render raw, and legacy chat renders markdown. Legit rendering gap, P3 polish.

**Source:** Console UX expert review 2026-07-20 (finding j4-raw-markdown-headings-in-transcript, j2-markdown-asterisks-raw; P3, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J4 streaming journey. Evidence: `j4-10-reply1-complete.png`, `j4-33-streaming3.png`, `j4-38-after-tab-flip.png`, `j2-24-resumed-long.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Render headings with terminal-appropriate emphasis (bold/underline/color) or strip the marker characters
<!-- AC:END -->
