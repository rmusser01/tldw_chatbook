---
id: TASK-34937
title: Stop-spawning-webbrowser-open-and-pyperclip-on-the-UI-thread
status: In Progress
assignee: []
created_date: 2026-10-13 14:57
labels:
- performance
- ui-perf
- wave-4-blocking-rebuilds
dependencies: []
priority: low
updated_date: 2026-10-15 15:19
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Link handlers spawn the OS open command on the UI thread (settings_screen.py:23600, console_transcript.py:1964, library_media_controller.py:4204, library_collections_controller.py:1583) and the inspector copy spawns pbcopy (console_conversation_inspector.py:1910). Each spawn can cost 50-150ms of UI freeze.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] Link and copy actions dispatch subprocess spawns via to_thread or executor
- [ ] No subprocess spawn on the event loop in the named handlers
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
PR3023 current-dev integration correction (2026-10-15): Reopen the five named browser/clipboard handlers after current Textual run_worker ABI review exposed duplicate name arguments and a missing Collections screen delegate. Read design language/component patterns and testing/live lessons; freeze actual integrated sources. Bind only the existing guarded callable and argument with functools.partial, preserving selected values, validation, notification order, privacy formatting, existing screen owner, thread/name/exit flags and all other bodies. Add one five-row targeted control using the actual imported handlers, genuine owning DOM/WorkerManager and finite external side-effect recorders; join the exact worker and verify original argument, off-issuer thread and Inspector instruction omission. Run that focused control and affected existing tests/static checks. No actual browser/clipboard or native paint/latency claim. ADR required: no. ADR path: N/A. Reason: routine correction to the existing worker API and owner, with no new runtime contract.
Preserve the existing About URL-policy cases and assertions; align only their positive asynchronous completion wait with the exact issued about-link worker before checking the external recorder. Include that existing module in targeted qualification; no fixed sleep or all-App worker join.
Actual baseline9 is1P8F: all five new handler rows reach ABI/no-worker failures; three About cases stop before their oracles at raw_source_selection_changed because the existing module lacks the required bootstrap_profile declaration. Add only that documented module marker, preserving every original body/assertion and the reviewed exact-worker wait. Retain original baseline negatives; do not claim those three setup failures were ABI regressions. Distinct review before candidate9, same new five-row TEST3579 in both variants.
<!-- SECTION:PLAN:END -->
## Implementation Notes

- All five spawn sites (settings link handler, console transcript `_open_link`, library media + collections URL openers, inspector `pyperclip.copy`) now dispatch through `run_worker(..., thread=True, exit_on_error=False)` — the OS spawn (50-150ms freezes) never runs on the loop, and a spawn failure can no longer kill the app.
- Behavior note: the inspector's success toast now fires when the copy is DISPATCHED (worker failures are swallowed by exit_on_error rather than caught inline); the collections opener's except branch remains for the pre-spawn state reads.
- Verification: all five edited modules import clean; `test_console_raw_cli_transcript.py` (exercises the transcript link-handler file) 17/17 green.
- ADR required: no — spawn dispatch change at five call sites.

- PR3023 current-dev correction: all five existing guarded browser/copy callables now use functools.partial with the original argument; Collections delegates to its existing screen WorkerManager. Targeted actual mounted-handler controls plus the four existing About URL-policy cases pass 9/9 (8.572s), including exact worker completion, off-loop invocation, unchanged URL policy and Inspector instruction omission. The existing About module now declares bootstrap_profile and waits for its exact worker. New TEST and affected About static checks pass; inherited product lint diagnostics are unchanged. Separate canonical COST regression formatting preserves the full AST and all three current cases pass (16.770s). Independent reviews 5ece1dbb and83ba2045 accepted these bounded results. Earlier baseline 1P8F remains evidence, with three About setup refusals distinguished from the five worker ABI failures. Real OS browser/clipboard, native paint/latency and whole-PR acceptance remain unchecked. ADR required: no; routine worker API/owner correction under the existing boundary. Status/AC remain In Progress/unchecked pending batch closeout.
