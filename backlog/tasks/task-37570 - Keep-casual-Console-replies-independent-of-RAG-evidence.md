---
id: TASK-37570
title: Keep casual Console replies independent of RAG evidence
status: Done
created_date: 2026-10-29 19:35
assignee:
- '@codex'
documentation:
- Docs/Architecture/console.md
updated_date: 2026-10-31 16:25
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A workspace user received a no-relevant-files response to How are you today. Clarify the Console evidence prompt so social conversation remains natural while answers about Library and workspace content remain grounded.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console instructions explicitly answer greetings, small talk, thanks and acknowledgements naturally without unnecessary searches, citations or missing-file notices.
- [x] #2 Source-dependent and mixed requests retain relevant evidence grounding, correct Library versus file citations and scoped statements about missing support.
- [x] #3 Dedicated Library answer grounding remains strict and targeted citation and provider-payload checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace Console evidence formatting and its initial, staged and retry provider-payload callers; compare the dedicated Library answer contract.
2. Clarify only the shared Console prompt: social replies need no sources, source-dependent claims still need relevant evidence, and mixed requests handle each part appropriately. Keep retrieval, permissions and citation validation unchanged; update the Console architecture reference.
3. Run the targeted citation formatter, canonical Console payload and strict Library answer checks, scoped lint/format and diff checks; self-review the prompt and record the limits of automated evidence.
ADR required: no
ADR path: N/A (existing ADR-024 and ADR-241 apply)
Reason: prompt wording clarification within existing grounding and authority boundaries; no new runtime policy, storage, interface, dependency or UI structure.
Merge qualification: rebase the existing prompt-only PR onto latest dev, rerun the original targeted citation/payload/Library checks in a fresh private test root, obtain the user-requested independent code review in place of Qodo, address findings and merge with the user-authorized CI/admin override. ADR required: no; wording-only change within existing ADR-024/241.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Clarified the shared CONSOLE_CITATION_INSTRUCTIONS in tldw_chatbook/Chat/answer_citations.py: greetings (including How are you today), small talk, thanks and acknowledgements get natural replies without source searches/citations/missing-file notices; mixed requests ground their source-dependent portion. Retrieved snippets must support the request, and empty results describe only the searched scope. Docs/Architecture/console.md records the clarification.
Existing Library-only answer instructions, automatic retrieval, tool permissions, evidence identity and citation repair are unchanged. ADR required: no; existing ADR-024 and ADR-241 apply. This is prompt wording only, with no new executable branches; no implementation-mirroring tests were added.
Static validation passed: ruff check, ruff format --check and git diff --check for the changed source. Self-review traced the common formatter and canonical Console initial/staged payload paths; dedicated Library formatting retains its strict contract.
Independent pre-PR review found no actionable issues: shared Console payload paths use the revised instructions, mixed requests preserve grounding, empty-result statements remain scoped, and dedicated Library grounding remains strict.
Initial verification (2026-10-29) was blocked before execution. The targeted command selected Tests/Chat/test_answer_citations.py, test_canonical_console_file_answer_uses_optional_library_markers, and Tests/Library/test_library_rag_answer_service.py. It stopped in collection/startup: sandboxed storage admission reported recovery_scope_uncertain; native reruns rejected the default Temp parent and then the checkout parent as shared_writable_parent (mode 0o766). A formatter-only attempt without conftest also stopped during imports on a sandbox DLL access denial. The task remained In Progress pending a supported private test location; the successful qualification below supersedes this block.
Merge qualification (2026-10-31): rebased PR3305 from 85f37825d3 onto latest dev 92ecd2c0aa73 (including merged SSE fix3308) as 7c2fa54400 without conflicts. Range-diff confirms the prompt/documentation patch is identical. No production branches, retrieval policy or citation validation were added or altered.
The original targeted runtime selection now passes: 93 tests in 79.22s across Tests/Chat/test_answer_citations.py, all eight direct/agent/marker cases of test_canonical_console_file_answer_uses_optional_library_markers, and Tests/Library/test_library_rag_answer_service.py. Native TEMP/TMP used a fresh private .codex/.tmp-pr3305-merge-check directory. Ruff check, Ruff format check and git diff --check pass. This supersedes the earlier environmental runtime block; existing Pydantic and pytest-asyncio deprecation warnings remain. No full suite or live-model qualification is claimed.
Fresh independent requesting-code-review review, explicitly replacing Qodo at the user's request, found no critical, important or minor issues. It traced initial, canonical staged and dispatch-retry payloads to the revised instructions and verified strict Library grounding, malformed-marker validation, permission and evidence-identity boundaries remain unchanged. Live conversational compliance and comprehensive prompt-injection qualification were explicitly outside deterministic review evidence.
ADR required: no; existing ADR-024/241 apply. The user carried forward the immediate merge authorization with CI/admin protection bypass from PR3308; protection enforcement is restored after the merge.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
