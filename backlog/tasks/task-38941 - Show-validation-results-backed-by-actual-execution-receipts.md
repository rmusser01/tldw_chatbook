---
id: TASK-38941
title: Show validation results backed by actual execution receipts
status: To Do
created_date: 2026-11-05 16:09
updated_date: 2026-11-22 15:51
labels:
- console
- validation
- agents
priority: medium
references:
- https://uxplanet.org/anthropic-just-shipped-claude-2-0-bbb46a9c1383
- https://code.claude.com/docs/en/plugins/mods/overview
- TASK-35582
documentation:
- Docs/Architecture/console.md
- Docs/Architecture/agent-runtime.md
- backlog/decisions/281-primary-agent-multi-model-code-review.md
- tldw_chatbook/Agents/raw_shell_tool_provider.py
- tldw_chatbook/Agents/code_review_models.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users distinguish checks that were proposed from checks that actually ran and understand what each result establishes about a generated change. Keep missing evidence and incomplete verification visible.

A check here is one of two recorded things, and nothing else:

- a `shell_exec` run (the ask-only raw-shell process tool, `Agents/raw_shell_tool_provider.py`): its tool call while it awaits approval or runs, then its `RawCliResult` (`invocation_id`, `exit_code`, `terminal_state` such as `exited`, `timed_out`, `cancelled` or `refused`, and `truncated`). It records no target revision or content identity.
- where a multi-model code review ran, ADR-281's `CheckProposal` and `CheckReceipt` records (`Agents/code_review_models.py`). A receipt carries a `target_digest` and an outcome (`executed`, `declined`, `unavailable`, `cancelled`, `expired` or `uncertain`); an `executed` receipt must name its native execution receipt. ADR-281's Slice 9 note says stock executable review checks resolve `unavailable` before launch, so receipts showing `executed` need the external executor it describes.

Other tool calls, suggested commands, and a model's check labels or prose are not checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A bounded current-turn/run validation view shows proposed, running, completed, failed, cancelled, and unknown checks only to the extent existing host/tool receipts can establish those states.
- [ ] #2 An executed check links to its `shell_exec` tool call and result or its ADR-281 receipt, and shows the outcome and, for a receipt, its `target_digest`; a `shell_exec` run's target identity shows as unknown.
- [ ] #3 Assistant prose or a model-authored checklist alone cannot mark a check as executed or passed, and stale checks are not presented as verification of later changes.
- [ ] #4 Viewing the panel never executes commands, reruns tests, or broadens tool/file authority; existing explicit actions retain their usual review gates.
- [ ] #5 Errors, partial output, unavailable receipts, and session switches preserve truthful status without exposing unauthorized output.
- [ ] #6 Targeted and mounted tests prove the distinction between claimed and observed success and cover target drift, partial/cancelled checks, and missing receipts.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Implementation planning records ADR required: yes/no, ADR path, and reason; any required new or amended ADR is accepted and linked before implementation begins.
- [ ] #2 Automated tests (unit + integration) cover any new logic, the linter and formatter pass, and `./scripts/preflight.sh` passes, per CLAUDE.md's Definition of Done.
- [ ] #3 Relevant user guidance and living Architecture documentation describe the delivered behavior and its limits.
<!-- DOD:END -->
