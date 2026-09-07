---
id: TASK-335
title: >-
  Fix Console provider-failure UX - raw 500 discards server detail and failed
  image poisons the conversation
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
updated_date: '2026-07-21 14:20'
labels:
  - console
  - ux
  - regression
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
- [x] #1 Surface the provider's error body, translate it ('this model can't accept images'), point at the offending attachment, and offer recovery (retry without images / remove from context). Never emit MDN links in a chat transcript
- [x] #2 A regression test pins the restored behavior
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Route agent-path failures through describe_stream_failure (kills raw httpx text + MDN links)
2. Surface the provider's response body (the actionable hint, e.g. 'provide the mmproj') in the classified copy
3. When the failed request carried image attachments, append a recovery hint naming the image as likely cause and pointing at existing remove/delete affordances
4. Check the reload-renders-as-Tool claim; RED-first tests for copy + body + hint
<!-- SECTION:PLAN:END -->

## Implementation Notes

The raw copy leaked through TWO paths: (a) `describe_stream_failure`'s
`(detail)` parenthetical appended the full `str(exc)` — for httpx status
errors that IS the status line + MDN boilerplate; (b) the agent service's
catch-all (`agent_service.py` run_turn) stamped raw `str(exc)` into the
STEP_ERROR summary that becomes `Agent run failed: {reason}` — the
review's exact live path.

Fixes: the classifier moved to a shared `Chat/provider_failures.py`
(controller re-exports for existing importers); for status errors the
detail is now the provider's RESPONSE BODY (best-effort JSON message
extraction, whitespace-collapsed, truncated) — surfacing hints like
"you may need to provide the mmproj" — and the MDN boilerplate is
stripped from any detail. The agent service classifies at capture where
the exception is live. Both failure sites (exception path and
failed-outcome path) append a recovery hint when the session history
carries an image and the failure is HTTP-status-shaped: names the image
as likely cause and points at the existing Delete affordance /
vision-capable-model switch (history re-sends the image every turn, so a
rejecting provider fails ALL later sends — the poisoning half of the
finding).

Verified: 4 RED-first tests (JSON body, plain-text truncation, no-MDN,
and a controller-level integration with a real httpx 500 carrying the
review's exact body + an image message in history → classified row with
body + recovery hint). Live E2E was attempted but headless-clipboard →
xterm paste friction blocked attaching a real image through the served
app (documented harness limitation class); the provider body itself was
already live-proven by the review's curl evidence (j3-63). Reload-as-Tool
rendering claim: rows persist as SYSTEM role; the Tool-render on reload
belongs to task-350's sub-agent rendering cluster. Files:
`Chat/provider_failures.py` (new), `Chat/console_chat_controller.py`,
`Agents/agent_service.py`.
