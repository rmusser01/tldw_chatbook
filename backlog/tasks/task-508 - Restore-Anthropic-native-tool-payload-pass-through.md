---
id: TASK-508
title: Restore Anthropic-native tool payload pass-through
status: Done
assignee: []
created_date: '2026-07-24 17:26'
updated_date: '2026-07-24 17:30'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the Anthropic request converter so valid native input_schema tool definitions survive request normalization while malformed entries remain local failures and prompt-caching behavior stays correct.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Valid Anthropic-native tools survive conversion without mutating caller input.
- [x] #2 Cache-capable Claude requests add the configured last-tool cache breakpoint.
- [x] #3 Non-caching Claude requests preserve native tools without `cache_control`.
- [x] #4 Malformed native and unrelated tool shapes are dropped with Anthropic-specific bounded diagnostics.
- [x] #5 Anthropic native-tool and adjacent mocked request tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the failing native-tool request as RED evidence and add cache/non-cache plus malformed-shape contract coverage.
2. Restore strict Anthropic-native `input_schema` conversion using fresh dictionaries and provider-specific diagnostics.
3. Run the full Anthropic native-tool suite and adjacent dispatch tests.
4. Run lint, format, and diff checks; record the no-new-ADR rationale and complete only after verification.

ADR required: no
ADR path: N/A
Reason: This restores the existing Anthropic provider-conversion and TASK-323 caching contracts without creating a new provider boundary or interface.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Restored Anthropic-native tool normalization in _anthropic_tools_payload after tracing the regression to the Cohere-only hardening hunk accidentally applied in commit 5d803ef25. Valid native entries now require a nonblank string name and dict input_schema, then survive through a fresh dict copy with native fields preserved. Invalid entries remain local drops and emit an Anthropic-specific diagnostic with a 120-character preview. OpenAI-shaped conversion and the TASK-323 request-layer cache annotation remain unchanged: cache-capable Claude adds cache_control to the last copied tool, while claude-2.1 preserves copied native tools without it. Updated the stale caching-model test and added bounded-diagnostic and non-cache/immutability coverage. Verification: full Tests/Chat/test_anthropic_native_tools.py plus Tests/Chat/test_chat_mocked_apis.py passed (34 tests); Ruff check passed; Ruff format check passed; git diff --check passed. ADR required: no. ADR path: N/A. Reason: this restores existing provider-conversion and TASK-323 caching contracts without a new boundary or interface.
<!-- SECTION:NOTES:END -->
