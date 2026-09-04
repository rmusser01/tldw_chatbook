---
id: TASK-31214
title: Add guided vLLM environment and model preflight
status: Done
assignee:
  - '@codex'
created_date: '2026-09-03 22:32'
updated_date: '2026-09-04 00:30'
labels:
  - vllm
  - lab
  - onboarding
dependencies:
  - TASK-31213
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent predictable first-run vLLM launch failures by making environment, model-source, network, and managed-argument readiness visible before Start.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Users can select either a Hugging Face repository ID or a local model directory through source-appropriate controls.
- [x] #2 Preflight reports interpreter resolution, vLLM module availability, model-source validity, port availability, and bind-address exposure without starting a server.
- [x] #3 Start is disabled with a visible field-adjacent reason until required checks pass.
- [x] #4 Managed host, port, and model flags cannot be duplicated or overridden through raw arguments.
- [x] #5 Focused unit and mounted Textual tests cover success, failure, preservation, and recovery states.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Source: .superpowers/sdd/2026-09-03-vllm-lab-console-complete-redesign/task-1-brief.md
ADR required: no
ADR path: backlog/decisions/115-vllm-lab-console-readiness-and-profiles.md
Reason: This task directly implements the accepted runtime and UX boundaries.

1. Add pure vLLM launch/preflight contract tests first, then record their expected RED result.
2. Implement immutable contracts, validation, bounded preflight, and public CLI command construction.
3. Replace the inline pane with VllmSetupView and add mounted workflow tests.
4. Run the specified focused GREEN suites and incumbent deferred-view checks.
5. Check acceptance criteria, record evidence and no-ADR rationale, mark the task Done, and commit Task 1 files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the immutable vLLM launch/preflight contracts and focused deferred setup view. Local starts resolve the matching public vllm CLI, reject managed/secret raw argument overrides, use a source-specific local-directory picker, reserve the existing server lifecycle claim, and never perform readiness or Console adoption.

Tests: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Management/test_vllm_setup.py Tests/UI/test_vllm_lab_workflow.py -k "preflight or initial or mode or command or source" (19 passed, 10 deselected); /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Management/test_gguf_server_sources.py Tests/UI/test_llm_deferred_views.py (78 passed); full focused vLLM files (29 passed); compileall and git diff --check passed.

ADR required: no. Existing ADR path: backlog/decisions/115-vllm-lab-console-readiness-and-profiles.md. Reason: directly implements accepted runtime and UX boundaries.

Modified: tldw_chatbook/UI/LLM_Management/{__init__.py,vllm_setup.py,vllm_setup_view.py}, tldw_chatbook/UI/LLM_Management_Window.py, tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_vllm.py, and focused tests.
<!-- SECTION:NOTES:END -->
