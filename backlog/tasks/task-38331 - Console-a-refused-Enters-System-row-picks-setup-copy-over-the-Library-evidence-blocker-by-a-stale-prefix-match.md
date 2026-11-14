---
id: TASK-38331
title: >-
  Console: a refused Enter's System row picks setup copy over the
  Library-evidence blocker by a stale prefix match
status: Done
assignee:
  - '@claude'
created_date: '2026-10-31 01:16'
updated_date: '2026-10-31 03:33'
labels:
  - console
  - errors-recovery
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When Enter is refused, prompt_queue chooses which sentence the System row shows by checking whether the send gate's reason starts with "Console send blocked: Library Search/RAG". The gate's evidence copy now reads "Console send blocked: Library search has no available evidence ...", so the prefix never matches and a setup sentence wins whenever both block. Found while removing substring-matched copy selection in TASK-33621.11 (not one of its named seams). The refusal should be chosen from a typed blocker, as the Send strip now is.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A refused Enter's System row names the blocker the send gate actually stopped on, chosen from a typed code rather than the sentence's wording
- [x] #2 A test drives a send blocked by both missing Library evidence and a provider setup gap through the real gate and asserts the System row
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Code the send gate's refusal (ChatScreen._console_send_gate) with the blocker it stopped on.
2. Let prompt_queue show the setup sentence only when that code is provider setup.
3. Mounted test: a send blocked by both missing Library evidence and a provider setup gap.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Folded into TASK-33621.11 at its review (same branch), because that task's AC#5 forbids choosing a failure row by matching display copy.

- The send gate now returns its sentence together with a code: archive_inflight, library_evidence_unavailable, provider_setup or image_unsupported (ChatScreen._console_send_gate; _console_send_blocked_reason returns the sentence alone). prompt_queue receives it through blocked_reason_accessor.
- A refused Enter's System row uses the setup sentence only when the gate's code is provider_setup; otherwise it uses the gate's own sentence. The setup toast follows the same code. The dead 'Console send blocked: Library Search/RAG' prefix check and the visible == setup_reason equality check are gone.
- Test: Tests/UI/test_console_send_strip_codes.py::test_a_refused_enter_names_the_blocker_the_gate_stopped_on (mounted ChatScreen; OpenAI with no key plus a staged bundle whose only reference is blocked). On the pre-fix code the row read 'Add API key in Settings > Providers & Models before sending.'; it now reads the evidence sentence.
<!-- SECTION:NOTES:END -->
