---
id: TASK-2118
title: HuggingFace tools debug log dumps full tool schemas verbatim
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-03 16:35'
updated_date: '2026-08-09 14:10'
labels:
  - llm-calls
  - observability
  - security
dependencies:
  - TASK-2116
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`chat_with_huggingface`'s "HuggingFace Tools" debug log emits full tool definitions —
schemas and descriptions — verbatim when the request is not classified sensitive.

This is the same bug class TASK-2116 and the PR #1295 Qodo round closed for the nine
provider request-payload logs, but this call site was outside that task's stated scope
(it logs tools, not the request payload) so it was left untouched and flagged instead.

The fix is small because the machinery now exists: `safe_llm_request_payload_summary`
in `tldw_chatbook/Utils/sensitive_llm_logging.py` already reduces tool definitions to
**names only**, dropping descriptions and schemas, and PR #1295 added
`test_tool_definitions_log_names_only_never_schema_or_description` pinning that
behavior. This site needs to route through the same helper rather than formatting the
raw definitions.

Worth stating the general lesson, since this code has now produced three leaks in one
session: redaction here must be **allowlist-shaped**. Denylists ("strip messages",
"strip contents") failed twice — once by missing `input`/`system`/`system_instruction`,
once by missing tool schemas. Anything logged from a provider payload should be an
explicitly enumerated safe field, so an unrecognized field is dropped by construction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The "HuggingFace Tools" debug log emits tool names only — never descriptions, parameter schemas, or enum values
- [ ] #2 The site routes through the existing `safe_llm_request_payload_summary` / tool-name helper rather than reimplementing redaction
- [ ] #3 A sentinel test proves a distinctive string planted in a tool description and in a parameter schema appears nowhere in log output, on BOTH the sensitive and non-sensitive paths
- [ ] #4 `tldw_chatbook/LLM_Calls/` swept for any remaining log site that formats raw tool definitions or raw payload dicts; each hit fixed or justified in the notes
<!-- AC:END -->
