---
id: TASK-32313
title: Implement offer-first Canvas guide tool and inline skill
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-11 04:53'
updated_date: '2026-09-11 05:31'
labels:
  - canvas
  - skills
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-09-10-canvas-guidance-skill-design.md
  - Docs/superpowers/plans/2026-09-10-canvas-guidance-skill-implementation.md
  - backlog/decisions/149-offer-first-canvas-guide-and-inline-skill.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Offer Canvas before proactive authoring and provide focused compatible authoring guidance after acceptance through the owning Console run and an optional trusted inline skill.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ordinary Canvas discovery and loaded guidance offer before proactive source generation and respect explicit requests, requested edits, refusal, and absent consent.
- [x] #2 The scoped canvas_guide tool exposes only four fixed documentation topics, caps serialized results at 12 KiB, preserves tool authority and metadata-only records, and refuses stale or disabled calls.
- [x] #3 The optional canvas skill imports through existing trust controls and runs inline without a child run or expanded tool authority.
- [x] #4 Packaged guides load from an installed wheel and all exact complete examples compile and execute their intended browser interactions with existing runtime profiles.
- [ ] #5 Targeted automated checks, static checks, documentation, independent reviews, and a recorded model-behavior sample support the delivered feature; unavailable evidence is explicitly retained as outstanding.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes (existing accepted decision)
ADR path: backlog/decisions/149-offer-first-canvas-guide-and-inline-skill.md
Reason: Direct implementation of the user-approved guide/skill decision, preserving ADR-121, ADR-124, and ADR-009.
Follow Docs/superpowers/plans/2026-09-10-canvas-guidance-skill-implementation.md in five stages with spec and quality review after each. Independent browser/package verification and skill/trust files may proceed alongside provider work; shared provider/guidance edits remain sequential:
1. Package fixed-topic bounded guides and exact examples, including unchanged Mermaid reuse.
2. Add the scoped guide tool, coordinated reservations, closed safe projections, and live disable/context checks.
3. Wire shared offer-first guidance and optional trusted inline canvas skill with targeted integration tests.
4. Verify wheel resources and actual browser interactions of the exact guide examples.
5. Record authorized model-behavior samples, final targeted checks/review, documentation, and truthful task closeout.
Use test-first changes for new logic and existing local Python3.12 environment. No full-suite runs. No renderer/profile/storage changes or skill-child authority expansion. The parent owns Backlog and plan checkboxes; stage workers own their explicitly assigned implementation files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented offer-first Canvas guidance, read-only canvas_guide(topic), and an optional trusted inline $canvas skill. The guide reads four fixed packaged topics, caps the complete result at 12 KiB, preserves live run authority, and stores only metadata. Existing create/update tools, pinned renderer/profile assets, and skill-child authority remain unchanged.

Files cover Canvas/guide.py and guides, the scoped provider/catalog and Console discovery path, Docs/Examples/skills/canvas/SKILL.md, user documentation, and focused provider/trust/browser/package tests.

Validation: 222 core regression checks; 40 skill/substitution checks; 10 policy/request checks; 49 reader/package checks including a fresh wheel/sdist; four exact-example Chromium checks plus eight inspected screenshots. These groups overlap. Ruff passes all 13 changed Python files. Nine files pass full formatting; four files have verified pre-existing formatting drift, with no new drift in changed lines. Independent stage spec/quality reviews and final overall review completed. No full suite was run.

Browser evidence exposed CSS shorthand expansion outside the renderer allowlist; the examples now use supported longhands. Two stale test assumptions were corrected against existing production behavior, without changing the controller.

ADR required: yes; implemented accepted backlog/decisions/149-offer-first-canvas-guide-and-inline-skill.md, preserving ADR-121, ADR-124, and ADR-009. Evidence: Docs/superpowers/qa/2026-09-10-canvas-guidance-skill.md.

Outstanding: all ten real-model scenarios remain unrun. The authorized llama.cpp endpoint at http://192.168.5.196:9191 is unreachable from this Mac (OSError 65: No route to host, also outside the sandbox). No generation request was sent. Keep AC #5 unchecked and task In Progress until live qualification is recorded; guidance tests do not prove model obedience or measured token savings.
<!-- SECTION:NOTES:END -->
