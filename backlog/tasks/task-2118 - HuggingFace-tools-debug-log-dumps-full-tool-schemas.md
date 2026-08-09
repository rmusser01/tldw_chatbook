---
id: TASK-2118
title: HuggingFace tools debug log dumps full tool schemas verbatim
status: Done
assignee:
  - '@codex'
created_date: '2026-08-03 16:35'
updated_date: '2026-08-09 20:22'
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
- [x] #1 The "HuggingFace Tools" debug log emits tool names only — never descriptions, parameter schemas, or enum values
- [x] #2 The site routes through the existing `safe_llm_request_payload_summary` / tool-name helper rather than reimplementing redaction
- [x] #3 A sentinel test proves a distinctive string planted in a tool description and in a parameter schema appears nowhere in log output, on BOTH the sensitive and non-sensitive paths
- [x] #4 `tldw_chatbook/LLM_Calls/` swept for any remaining log site that formats raw tool definitions or raw payload dicts; each hit fixed or justified in the notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a parameterized real-function Loguru sentinel test covering ordinary and sensitive HuggingFace tool logging; prove the current ordinary path is red.
2. Route the ordinary Final Payload and Tools diagnostics through safe_llm_request_payload_summary without changing the sensitive branch or helper contract; run focused green and static gates.
3. Sweep every LLM_Calls logger for raw request-payload dictionaries and raw tool definitions, classify all candidates, and repair any additional AC 4 match.
4. Mutation-test the tool-definition and Final Payload guards independently, restoring the corrected implementation after each red proof.
5. Run touched-file and affected-functionality tests plus lint, edited-range format, compilation, and diff gates, per the user-approved closeout scope.
6. Record exact sweep/mutation/test evidence, check all acceptance criteria, and mark TASK-2118 Done only after every gate passes.

ADR required: no new ADR
ADR path: backlog/decisions/029-local-private-data-boundary.md
Reason: ADR-029 already excludes provider payloads and tool definitions from persistent logs while permitting bounded metadata such as tool names. This task applies that accepted contract.
Detailed plan: Docs/superpowers/plans/2026-08-09-task-2118-huggingface-tool-log-privacy.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the narrow HuggingFace privacy repair: the ordinary Final Payload diagnostic now uses safe_llm_request_payload_summary, and the separate Tools diagnostic summarizes a tools-only projection through the same existing helper. Sensitive requests remain metadata-only and emit no Tools line. No helper contract changed and no dependency was added.

Regression evidence: the parameterized test drives the real chat_with_huggingface function directly with only transport/config seams replaced; it covers ordinary and sensitive contexts and plants distinct description, parameter-enum, and unallowlisted-user sentinels. Before the repair the ordinary case was red while the sensitive case passed; after the repair both cases passed. Independent cache-disabled mutations separately restored raw tool interpolation and the former Final Payload denylist: each mutation produced exactly 1 failed and 1 passed parameter case for the intended sentinel, and each restored implementation returned to 2 passed.

AC 4 sweep evidence: every one of 763 logger/logging calls across eight LLM_Calls modules was source-reviewed: LLM_API_Calls.py 171, LLM_API_Calls_Local.py 41, Local_Summarization_Lib.py 242, Summarization_General_Lib.py 281, huggingface_api.py 9, pricing_catalog.py 4, realtime/openai_session.py 13, and realtime/transport.py 2. Function-scoped correlation found 57 outbound body-bearing calls across 33 scopes: json= 55, data= 2, content= 0. All 57 body construction and reassignment flows were reviewed. Forty-one logger sites correlated with an exact body name or the limited simple-alias set: 27 exact and 14 alias-only, distributed 9/1/20/11 across LLM_API_Calls, LLM_API_Calls_Local, Local_Summarization_Lib, and Summarization_General_Lib. They classify as nine existing safe allowlist summaries, one local keys-only diagnostic, and 31 summarization metadata or input-preview diagnostics that do not render the constructed provider request dictionary. A separate lexical tool/schema/definition scan found ten expressions in LLM_API_Calls: eight constant validation/event warnings, one tool_call_id diagnostic, and the corrected HuggingFace names-only summary; no raw schema or definition structure remained.

The supplementary identifier-filtered inventory remained exactly 35 candidates across four modules: 13/1/15/6. Eleven are helper-routed request/tool summaries, fourteen are bounded metadata-only diagnostics, and ten are separately owned content diagnostics rather than raw provider request dictionaries or tool definitions. The ten exact content sites are Local_Summarization_Lib.py lines 48, 175, 364, 825, 1050, 1548, and 1798, plus Summarization_General_Lib.py lines 82, 1477, and 2485; a separately filed follow-up owns them. This 35-site inventory is heuristic evidence only. Static evidence covers Python calls rooted at logger/logging plus the enumerated HTTP method/body-keyword shapes and limited direct/copy aliases; it does not prove dynamic runtime values, custom logger aliases, positional or custom transport bodies, or generated code. The AC 4 conclusion is the combined complete source review, body/tool correlation, real-function sentinel, and independent mutations.

Latest-dev reconciliation: final base is origin/dev fccb3af6b. The LLM_Calls tree and the three focused test files have identical Git object IDs to the prior independently reviewed 575d4cd8d baseline, so no intervening scoped change invalidated the reviewed evidence; the 35- and 763-call counts were also rerun unchanged.

Final touched-scope verification after the final rebase: sensitive logging 70 passed; HuggingFace chat subset 8 passed and 60 deselected; debug-log f-string hygiene 2 passed. Ruff lint passed; edited-range format checks at production 4341-4365 and test 40-70/675-820 each reported already formatted; py_compile and git diff --check passed. The user explicitly limited completion tests to files and functionality touched, so the earlier repository-wide gate in the task plan was replaced and is not completion evidence. No test application or simplified application was used.

ADR required: no new ADR. ADR-029 already governs the metadata-only log boundary and permits tool names while excluding payload values and tool definitions. The existing Loguru temporary-sink and mutation-testing lesson applied; this task discovered no new general lesson requiring a lessons-file update. Modified scope: LLM_API_Calls.py, test_sensitive_llm_logging.py, TASK-2118 spec/plan/task documentation, and the separately committed follow-up record.
<!-- SECTION:NOTES:END -->
