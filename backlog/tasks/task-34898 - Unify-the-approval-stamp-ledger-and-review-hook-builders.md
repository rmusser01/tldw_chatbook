---
id: TASK-34898
title: Unify the approval-stamp ledger and review-hook builders
status: To Do
created_date: 2026-10-12 17:53
labels:
- simplification
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The ask-once-per-batch approval protocol is implemented five times on the provider side (stamp ledgers with snapshot restore and scope-ranked merging) and seven times as near-identical review-hook builders in console_interrupt_rounds.py wired through 324 read_global_ parameters.

Evidence (verified on dev df4985e): Provider stamp copies: Agents/local_tool_provider.py:997-1010+1678-1712 Agents/mcp_tool_provider.py:585+877-1027 Agents/builtin_tool_gate.py:131-264 Agents/raw_shell_tool_provider.py:299+430 Agents/virtual_cli_provider.py:258. Hook builders: Chat/console_interrupt_rounds.py:5396-6474 (seven build_*_review_hook functions ~1080 lines; build_tool_review_hook takes 20 read_global_ callables). Forwarding shims in Chat/console_chat_controller.py:3008-3147. Refusal constants are deliberately byte-pinned per provider (TASK-28238 TASK-34351) and become adapter data. Tests: Tests/Agents/test_builtin_tool_gate.py Tests/MCP/test_permission_prompt_reducer.py.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One ApprovalStampLedger helper owns stamp bookkeeping for all gated providers,One generic review-hook builder parameterized by a small provider adapter,Per-provider refusal copy preserved byte-for-byte as adapter data,read_global_ seams remain test-patchable,Agents and MCP approval tests pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
