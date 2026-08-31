---
id: TASK-26017
title: 'Compaction: surface a manual preview before committing'
status: To Do
assignee: []
created_date: '2026-08-31 15:45'
labels:
  - console
  - context
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Manual compaction commits without showing what it will do. Verified on origin/dev: plan_manual_prefix and plan_manual_range (Chat/console_context_compaction.py:1016, :1047) build a ManualMemoryPlanResult and it is invoked from Chat/console_chat_controller.py:14051, but a grep for --preview or dry_run across Chat/ returns zero - the plan object exists and is simply never shown. Hermes offers /compress --preview. This is presentation over a value already computed before the commit point.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A manual compaction can be previewed: what will be summarized, what will be retained, and the estimated token change
- [ ] #2 The preview does not perform the compaction and leaves no memory record or provenance entry behind
- [ ] #3 The user can commit or discard directly from the preview without re-specifying the range
- [ ] #4 Preview honors the same range semantics as the commit path, so what is previewed is what happens
- [ ] #5 Previewing costs no model call, or if it must call, that cost is stated before it is incurred
<!-- AC:END -->
