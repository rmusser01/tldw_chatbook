---
id: TASK-2376
title: Media and conversation handoff snippets are thin generic labels, not excerpts
status: To Do
assignee: []
created_date: '2026-08-04 20:07'
labels:
  - console
  - rag
dependencies: []
priority: low
---

## Description

`capture_console_staged_evidence_for_chat`'s snippet formula prefers `display_summary` over `body` when both are present. Media and conversation handoff builders set `display_summary` to a short generic label (for example, "Media staged: {title}"), so the snippet the model actually receives is a label, not an excerpt of the real content. Notes are unaffected, since their builder does not set a competing `display_summary`.

PR-T1 Task 9 (task-2374) fixed the underlying zero-content bug for these handoff kinds and self-flagged this as a residual: the content delivered today is thin but honest (a real, correctly attributed reference), not silently empty. This task is about upgrading fidelity, not about a correctness regression.

## Acceptance Criteria

- [ ] Media and conversation handoffs deliver a real content excerpt in the snippet sent to the model, not just a generic label
- [ ] Notes' existing (unaffected) snippet behavior is preserved
- [ ] A test pins the excerpt's actual content, not merely that a snippet is present
