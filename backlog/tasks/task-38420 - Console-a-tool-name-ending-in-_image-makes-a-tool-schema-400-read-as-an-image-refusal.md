---
id: TASK-38420
title: >-
  Console: a tool name ending in _image makes a tool-schema 400 read as an image
  refusal
status: Done
assignee: []
created_date: '2026-10-31 03:29'
labels:
  - console
  - errors-recovery
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The image hint on a failed reply follows a typed verdict: a 400, 413, 415 or 422 whose provider body names image or vision input. The body check matches 'image' after an underscore in any of the error's message, type, code or param fields, so it also matches a tool name like generate_image. A 400 that rejects such a tool's schema, in a conversation that holds an image, then shows the image hint beside the tool-definition copy, advising a vision model or /rewind for a failure that has nothing to do with the image. Found in review of TASK-33621.11.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A 400 that rejects a tool definition whose name contains 'image' does not add the image hint, even when the conversation holds an image
- [x] #2 A real image refusal (for example OpenAI's 'invalid_image_format' or Anthropic's content image error) still adds the hint
- [x] #3 Tests drive both cases through the real gateway and failure-copy code
<!-- AC:END -->

## Implementation Plan

ADR required: no new ADR.
ADR path: backlog/decisions/211-ephemeral-provider-failure-presentation.md.
Reason: repair existing transient classification; diagnostics and contact/custody stay separate. Existing ADR094/230 remain applicable.

Follow Docs/superpowers/plans/2026-11-01-pr3330-backlog-integration.md: causal gateway/schema and structured-seam RED, bounded strict parser, targeted GREEN; preserve current accounting and received custody.

## Implementation Notes

Bounded error classification now requires exact image codes, rooted input paths or anchored rejection clauses; schema/tool names are negative in JSON, provider_reason and plain text. The hosted attach seam retains only the transient boolean before suppressed HTTP context. Agent/direct admitted-image payload tests retain tool recovery without image/model advice; code/path positives retain image advice. Typed diagnostic messages and SQL/run-log custody remain unchanged.

Integrated onto actual dev7d2fc1ea with current contact/refused-stream/native ownership preserved. Also repaired the selected checked received-image preflight's missing plain-string refusal before promotion, using frozen evidence and original session; no attachment transfer, user admission or provider contact. Existing [ADR211](../decisions/211-ephemeral-provider-failure-presentation.md), [ADR094](../decisions/094-console-turn-lifetime-and-navigation-boundary.md), [ADR230](../decisions/230-console-send-preparation-and-io-ownership.md) apply; no new ADR.

Verification:126 distinct focused cases have full PASS receipts across stable identity/hash-bound runs; no full sweep. QA: [current-base integration](../../Docs/QA/2026-11-01-pr3330-current-base-integration.md). The Backlog CLI was unavailable; authorized direct Markdown updates preserve IDs/schema. TASK38424 remains open for measured narrow strip/editor limits.
