---
id: TASK-2118
title: HuggingFace tools debug log dumps full tool schemas verbatim
status: Done
assignee:
  - '@codex'
created_date: '2026-08-03 16:35'
updated_date: '2026-08-09 21:18'
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

The supplementary identifier-filtered inventory remained exactly 35 candidates across four modules: 13/1/15/6. Eleven are helper-routed request/tool summaries, fourteen are bounded metadata-only diagnostics, and the remaining candidate subset contains separately owned content diagnostics rather than raw provider request dictionaries or tool definitions. It is heuristic evidence only and is not the remediation inventory. Final review inspected all 523 logger calls in Local_Summarization_Lib.py and Summarization_General_Lib.py and assigned the complete 199-site private-diagnostic boundary to the standalone follow-up, grouped by module, enclosing function, stable label, and category. Local has 100 sites: 13 raw/processed/extracted input, 8 prompt, 8 credential-fragment, 6 private endpoint/path, 29 response/output, and 36 exception/error-detail. General has 99: 8 input, 9 prompt, 13 credential-fragment, 5 private endpoint/path, 42 response/output, and 22 exception/error-detail. Combined category totals are 21/17/21/11/71/58. Static evidence covers Python calls rooted at logger/logging plus the enumerated HTTP method/body-keyword shapes and limited direct/copy aliases; it does not prove dynamic runtime values, custom logger aliases, positional or custom transport bodies, or generated code. The AC 4 conclusion is the combined complete source review, body/tool correlation, real-function sentinel, and independent mutations.

Latest-dev reconciliation: final-review base is origin/dev f6911b37b. No upstream commit since the prior reviewed base changed the scoped production file, focused tests, or either summarization module; only the testing-lessons file changed in scope. The 35-, 523-, and 763-call counts were rerun unchanged. The reviewed LLM_API_Calls.py diagnostic change preserves 171 calls and changes only its manifest digest from dc16bf5efed6e22426f0 to 246b5c982ddb1910cc8d. After that one-entry reconciliation, generated-versus-stored inventory drift is exactly the same 16 unrelated owners on branch and detached f6911b37b (normalized SHA-256 fa24957505c91fa2be8cf3426e3b86572ee8015b34be6715f72ff38eba62db41); sink topology is unchanged. A separate To Do task owns that current-dev baseline incident, and TASK-2118 does not bless those entries.

Final touched-scope verification after the final-review rebase: sensitive logging 70 passed; HuggingFace chat subset 8 passed and 60 deselected; debug-log f-string hygiene 2 passed. The diagnostic-inventory architecture file produced 1 failed and 7 passed on the branch, exactly matching the detached f6911b37b baseline; normalized generated-versus-stored drift is identical on both at 16 entries and SHA-256 fa24957505c91fa2be8cf3426e3b86572ee8015b34be6715f72ff38eba62db41, with no branch delta after the LLM_API_Calls.py reconciliation. Ruff lint passed; edited-range format checks at production 4341-4365 and test 40-70/675-820 each reported already formatted; py_compile and both diff checks passed. The user explicitly limited completion tests to files and functionality touched, so the earlier repository-wide gate in the task plan was replaced and is not completion evidence. No test application or simplified application was used.

Merge-integration refresh: the completed branch rebased conflict-free onto `origin/dev` `37e634cbb`. The intervening Console-rail change added no persistent logger calls, so the reviewed diagnostic population and manifest stayed unchanged. Fresh direct/function-level verification passed: 99 combined architecture/privacy/debug-log/RAG-keyword tests, 8 HuggingFace tests (60 deselected), 11 roleplay store/controller tests (331 deselected), 4 chat-display-name config tests (25 deselected), and 7 FFmpeg trim-argument tests. The manifest checker remained green at 485 owners, 1,167 TASK-492 calls, 6,962 TASK-494 calls, and 6 sinks; Ruff lint passed for branch-owned lines, all 70 changed Python hunks were formatter-clean, all 26 changed Python files compiled, and both diff checks passed. No repository-wide suite or test/simplified application was used.

ADR required: no new ADR. ADR-029 already governs the metadata-only log boundary and permits tool names while excluding payload values and tool definitions. The existing Loguru temporary-sink and mutation-testing lesson applied. Final review added an incident-based testing-evidence lesson because a deliberately heuristic candidate list had been incorrectly promoted into a complete remediation inventory. Modified scope: LLM_API_Calls.py, test_sensitive_llm_logging.py, the one branch-owned production-diagnostic manifest digest, TASK-2118 spec/plan/task documentation, the complete summarization follow-up record, the unrelated baseline-owner record, and the testing-evidence lesson.
<!-- SECTION:NOTES:END -->
