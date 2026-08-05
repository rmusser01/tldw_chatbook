---
id: TASK-1536
title: 'Roleplay flavor styling in the Console transcript'
status: Done
assignee: []
created_date: '2026-07-30 17:20'
labels: [enhancement, roleplay, console, ux]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Assistant replies now style roleplay flavor distinctly: "quoted speech"
(straight or curly, quotes kept visible) in sky blue, *action/inner
monologue* in violet italic (markers stripped), **bold** in warm gold;
plain narration unchanged. Built on the transcript's existing
literal-text + (text, style)-tuple seam (`_inline_markdown_spans`), so the
no-markup-injection guarantee is preserved; unclosed markers stay literal
mid-stream, and a quote swallows markers inside it. Assistant-only, like
the existing markdown emphasis (Qodo #823 rationale).

Colors are concrete values (Content span styles never resolve CSS $theme
variables); chosen to read on the dark default theme.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Bold, quoted speech, and action/italics render mutually distinct and distinct from narration.
- [x] #2 Unclosed markers stay literal; injected Rich markup stays literal.
- [x] #3 Straight and curly double quotes both style as speech.
- [x] #4 Span-level tests pin the exact segments; live-verified with screenshots.
<!-- AC:END -->
