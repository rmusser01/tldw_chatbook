# Phase 3.9 Artifacts Chatbook Console Launch

Task: `TASK-3.9`
Branch: `codex/unified-shell-phase3-artifacts-console-launch`

## Goal

Make Artifacts a real Console live-work source for local Chatbook artifacts. A user who has generated or imported a Chatbook can launch the latest local Chatbook into Console from Artifacts, while users without a local Chatbook get an honest unavailable state instead of a false generic handoff.

## Implementation Summary

- Added latest-local-Chatbook lookup to the Artifacts destination using the existing `local_chatbook_service.list_chatbooks` seam.
- Changed `artifacts-use-in-console` from generic chat handoff behavior to a typed `open_console_for_live_work` launch when a Chatbook record exists.
- Kept the Artifacts Console action disabled with recovery copy when no local Chatbook exists or the service is unavailable.
- Updated Console source readiness to mark `Artifacts` connected for the local Chatbook launch path.

## Verification

- Red result: focused tests failed because Artifacts still showed `Use in Console` as a generic handoff, did not disable the action without Chatbooks, Console readiness reported `Artifacts: Not wired`, and this evidence file did not exist.
- Focused green command: `.venv/bin/python -m pytest Tests/UI/test_console_live_work_handoffs.py -q`
- Focused green result: `41 passed, 1 warning in 16.17s`.
- Broader focused command: `.venv/bin/python -m pytest Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_destination_shells.py -q`
- Broader focused result: `76 passed, 1 warning in 24.15s`.
- Whitespace check: `git diff --check` completed with no output.
- Review-fix red result: three focused regressions failed for unsafe Chatbook metadata, lexicographic latest-Chatbook tie-break, and service errors rendering as empty-state copy.
- Review-fix focused green command: `.venv/bin/python -m pytest Tests/UI/test_console_live_work_handoffs.py::test_artifacts_destination_sanitizes_chatbook_metadata_before_console_launch Tests/UI/test_console_live_work_handoffs.py::test_artifacts_destination_uses_numeric_id_tie_break_for_latest_chatbook Tests/UI/test_console_live_work_handoffs.py::test_artifacts_destination_distinguishes_chatbook_service_failure_from_empty_state -q`
- Review-fix focused green result: `3 passed, 1 warning in 8.87s`.
- Review-fix broader focused command: `.venv/bin/python -m pytest Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_destination_shells.py -q`
- Review-fix broader focused result: `79 passed, 1 warning in 26.35s`.

## QA Walkthrough Notes

- Environment: focused Textual mounted-window tests using the repo virtualenv.
- Entry path: Artifacts destination screen.
- Visual check: Artifacts preserves the `Open Chatbooks` primary route and shows either `Console launch available` with the latest Chatbook title or `Console launch unavailable` with recovery copy.
- Functional result: selecting `artifacts-use-in-console` with a local Chatbook stages a typed Console launch payload containing Chatbook identity, file path, description, tags, categories, and updated timestamp.
- Recovery result: without a local Chatbook, the Console launch control is disabled and does not stage chat or Console context.
- Review-fix recovery result: if the local Chatbook service fails, Artifacts now shows distinct service-unavailable recovery copy instead of claiming there are no local Chatbooks.
- Review-fix security result: Chatbook metadata is sanitized through `input_validation.py` before being staged into Console launch payloads.

## Residual Risk

- This slice covers local Chatbook artifacts only; server output artifacts, report bundles, datasets, and generated-output live streams remain future Artifacts work.
- Console stages the Chatbook payload but does not yet open a dedicated Chatbook detail action from the Console primary-action button.
