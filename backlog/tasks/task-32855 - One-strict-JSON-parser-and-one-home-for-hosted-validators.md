---
id: TASK-32855
title: One strict-JSON parser and one home for the hosted validators
status: To Do
assignee: []
created_date: '2026-09-19 08:24'
labels:
  - core-review
  - review-cascade
dependencies: []
parent_task_id: TASK-32850
references:
  - qa/cascade-review-2026-09-19/report.md
  - backlog/decisions/062-hosted-chat-completions-provider-boundary.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Strict-JSON parsing exists in three drifted families (verified by the 2026-09-17 review, unfiled until now): family A with depth/node caps but no duplicate-key rejection (`hosted_chat.py:633-678`, `qwencloud_streaming.py:61-109`, `moonshot.py:920-948`, `zai.py:913-941`); family B with dup-key rejection but no caps (`Chat/thinking_blocks.py:221-235`, `Chat/provider_continuation.py:204-219`); family C parse-constant-only (~10 sites). The drift is a live defect: a provider emitting a duplicate key in tool arguments passes the wire family and throws an uncaught `ContinuationValidationError` at the continuation checkpoint (`moonshot.py:403`).

Separately, `moonshot.py` and `zai.py` duplicate ~14 pure validators verbatim (~203 LOC, `_json_shape_is_bounded` byte-identical) — mechanics, not the per-provider builders ADR-062 keeps separate.

One public `strict_json_loads(text, *, max_depth, max_nodes, reject_duplicate_keys=True)` in `Utils/input_validation.py` (combining A's caps with B's dup-key hook), families A and B as thin wrappers; validators move to `hosted_chat.py` parameterized by provider label. ADR required: yes (small) — the shared acceptance rule for wire and storage is a contract decision; the 2026-09-17 finding already scoped it ("pick one rule and pin it").

Source: cascade review 2026-09-19 — prior evidence `qa/core-code-review-2026-09-17/slices/LLM.md` P2 finding; cascade framing in `qa/cascade-review-2026-09-19/report.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One public strict-JSON parser in `Utils/input_validation.py` owns depth/node caps, finite-float and str-key checks, and duplicate-key rejection; families A and B are thin wrappers
- [ ] #2 Wire and continuation storage agree on duplicate-key acceptance — one rule, pinned by a test that round-trips a dup-key tool-arguments payload end to end
- [ ] #3 The moonshot/zai shared validators live once in `hosted_chat.py` parameterized by provider label; per-provider builders stay per provider (ADR-062)
- [ ] #4 The family-C parse-constant-only sites adopt the shared parser where their semantics allow, or are enumerated with reasons
- [ ] #5 Existing moonshot/zai/hosted-chat/thinking-blocks/continuation tests pass; new pins cover dup-key and depth-cap edges
<!-- AC:END -->
