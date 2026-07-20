---
id: TASK-335
title: Fix Console provider-failure UX - raw 500 discards server detail and failed image poisons the conversation
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux, regression]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Sending an image to llama.cpp produced transcript rows: 'Assistant [failed]' + 'Agent run failed: Server error 500 Internal Server Error for url http://127.0.0.1:9099/v1/chat/completions / For more information check: https://developer.mozilla.org/...Status/500.' The server body actually said 'image input is not supported - hint: you may need to provide the mmproj' (verified by curl) but was thrown away. Worse: a later plain-text message ('line one') in the same conversation failed identically because history re-sends the image forever; nothing tells the user the image is the cause and there is no affordance to drop it from history. On reload the same errors re-render under the role 'Tool' instead of 'System'.

**Repro:** Mark local-gemma vision-capable in config, attach a PNG, send. Observe the generic 500 + MDN link. Then send any plain text message in the same conversation - it fails with the same 500.

**Verifier note:** Ledger says provider failures render as transcript-only system rows with CLASSIFIED copy (demonstrably worked 2026-07-11). The now-default agent-runtime path (agent_runtime_enabled=True, console_chat_controller.py:278) formats failures at line 2326 as 'Agent run failed: {raw reason}', bypassing describe_stream_failure (line 172) — j3-63 confirms raw httpx text + MDN link, no classification prefix; and the rows come back on reload rendered as 'Tool' (j3-77/79), contradicting 'transcript-only'. The response BODY was discarded even under the old classified copy, and the pointing-at-offending-attachment / drop-from-history recovery never shipped — those sub-asks are new, but the headline copy/persistence behavior contradicts shipped remediation behavior. P1 stands: default path, and the re-sent image poisons every later send with an undiagnosable 500.

**Source:** Console UX expert review 2026-07-20 (finding j3-provider-error-discards-detail-poisons-conversation; P1, verdict REGRESSION, medium confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J3 attachments journey. Evidence: `j3-63-response-final.png`, `j3-67-shift-enter.png`, `j3-70-reopened-conv.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Surface the provider's error body, translate it ('this model can't accept images'), point at the offending attachment, and offer recovery (retry without images / remove from context). Never emit MDN links in a chat transcript
- [ ] #2 A regression test pins the restored behavior
<!-- AC:END -->
