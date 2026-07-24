---
id: TASK-547
title: Restore Anthropic native tool payload handling
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 20:35'
updated_date: '2026-07-24 20:47'
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
- [x] #1 Valid Anthropic-native tool definitions survive request normalization
- [x] #2 Normalized native definitions are defensive copies and caller input is unchanged
- [x] #3 Malformed tool definitions are dropped with Anthropic-specific metadata-only diagnostics
- [x] #4 Anthropic tool-contract and adjacent mocked request tests pass
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Restored the existing TASK-263 Anthropic request-conversion contract. Valid native name/input_schema mappings are preserved via fresh outer dictionaries, while OpenAI-shaped conversion remains unchanged. Malformed mappings and non-mappings are rejected locally with fixed Anthropic-specific diagnostics that neither render custom repr values nor expose secret-bearing payloads; blank OpenAI function names now receive the same bounded diagnostic instead of disappearing silently. Added immutability, raising-repr, secret, malformed-native, and blank-name coverage. Verification: focused Anthropic plus adjacent mocked-provider suites passed (28 tests); the full non-gateway Chat suite passed (982 tests, 69 expected skips); the gateway file passed separately (68 tests); Ruff check/format, py_compile, and git diff --check passed. ADR required: no. ADR path: N/A. Reason: this repairs the existing provider/tool conversion boundary without a new interface or architecture.
<!-- SECTION:NOTES:END -->
