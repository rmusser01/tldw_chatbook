---
id: TASK-33793
title: >-
  Console: every send logs a false RAG capture_provider_failure error from a
  missing launch argument
status: To Do
assignee: []
created_date: '2026-10-02 04:57'
labels:
  - console
  - ux-review-2026-10-01
  - bug
dependencies: []
references:
  - Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/report.md
  - Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/findings.json
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the 2026-10-01 Roleplay NN/g + HCI review: finding RP-097 (P3, severity 1, effort S). Filed as a Console task because the defect is in the Console, which the Roleplay review depends on. Report: `Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/report.md` (section 5 and Appendix A), with the machine-readable entry in `findings.json` beside it. The review ran on origin/dev @ 84247cb843 (worktree `.worktrees/roleplay-library-ux`).

Line refs below were re-checked on origin/dev @ ab4df99959. Paths are under `tldw_chatbook/` (`personas_screen.py` is `UI/Screens/personas_screen.py`; the `personas_*` and `persona_*` widgets are in `Widgets/Persona_Widgets/`). Capture names refer to the review's capture set; those copied into the QA folder are marked `evidence/`.

This is also the Console review ledger's G2-37: TASK-33621 lists it among related P2/P3 findings and TASK-33621.20 names it out of scope, but no task owns it. The Roleplay review added the root cause.

**What happens.** The Console wires a RAG capture function that takes three arguments (draft, turn context, launch) but calls it with two. The resulting TypeError is swallowed and logged as "Console RAG capture unavailable; reason=capture_provider_failure" on every send, successful or not, and a "Retrieval failed" Trace event is written. The Roleplay review's first round mistook this error for the cause of the blocked Chat-now sends; the real cause is tracked in TASK-33621.2.

**Who it hurts.** Anyone diagnosing a Console send from the logs or the Trace: an ERROR on every send hides real failures and sends investigations the wrong way. It is otherwise benign, because without a launch the capture would return nothing anyway.

**Evidence:**
- `tldw_chatbook/Chat/console_runtime.py:2503-2516`: `_capture_frozen_console_staged_rag(draft, turn_context, launch)` requires `launch`; it is wired as the RAG capture provider at `:3677`.
- `tldw_chatbook/Chat/console_chat_controller.py:23840-23872`: the controller sees a context parameter and calls the provider with draft and turn context only; the TypeError is caught, logged without its type, and recorded as "Retrieval failed".
- Review logs `rv-gap1-run1-app.log` lines 390 and 444: the same ERROR on a completed send and on a blocked send. A standalone replica of the dispatch raised "missing 1 required positional argument: launch".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Console send with nothing staged logs no capture_provider_failure error and writes no "Retrieval failed" Trace event.
- [ ] #2 A Console send with staged evidence still captures it exactly as before.
- [ ] #3 When the capture provider genuinely fails, the log line names the exception type and contains no draft or evidence content.
- [ ] #4 A test exercises the Console's RAG capture with the provider exactly as production wires it, not a stub with a different signature; it fails on the current code.
<!-- AC:END -->
