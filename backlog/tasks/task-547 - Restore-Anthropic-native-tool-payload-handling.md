---
id: TASK-547
title: Restore Anthropic native tool payload handling
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 20:35'
updated_date: '2026-07-24 20:35'
labels:
  - providers
  - tools
  - reliability
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the Anthropic request converter contract so valid native input_schema tools survive normalization without mutating caller input and malformed tools are rejected without leaking caller-controlled data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Valid Anthropic-native tool definitions survive request normalization
- [ ] #2 Normalized native definitions are defensive copies and caller input is unchanged
- [ ] #3 Malformed tool definitions are dropped with Anthropic-specific metadata-only diagnostics
- [ ] #4 Anthropic tool-contract and adjacent mocked request tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the full-suite native-tool failure as RED evidence and add regression coverage for defensive copying and secret-safe malformed-shape diagnostics.
2. Restore strict Anthropic-native input_schema handling while leaving OpenAI-shape conversion unchanged.
3. Run the focused Anthropic and adjacent mocked request suites plus Ruff and diff checks.
4. Resume the project-wide fail-fast verification and close only after the contract remains green.

ADR required: no
ADR path: N/A
Reason: This repairs the implementation of the existing Anthropic provider/tool conversion boundary established by TASK-263 without changing the public interface or choosing a new provider architecture.
<!-- SECTION:PLAN:END -->
