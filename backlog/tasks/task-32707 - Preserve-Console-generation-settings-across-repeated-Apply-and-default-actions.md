---
id: TASK-32707
title: Preserve Console generation settings across repeated Apply and default actions
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 01:07'
updated_date: '2026-09-17 01:31'
labels: []
dependencies: []
documentation:
  - Docs/Development/console-model-modal-investigation-2026-09-16.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Previously applied generation values must survive later settings edits and be saved as shown.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Repeated quick and full modal Apply preserve untouched conversation generation values
- [x] #2 Make default for new chats persists the displayed settings without losing earlier edits
- [x] #3 Provider/model switching and explicit inheritance retain their intended behavior; nearby retention defects are covered
- [x] #4 Same-target quick and full Apply retain supported hidden values and live endpoints; A-B-A restores untouched conversation values as well as unfinished edits
- [x] #5 Full Make default without an explicitly authorized endpoint edit writes its model profile successfully and leaves the configured endpoint unchanged
- [x] #6 Quick-to-full settings transfer saves displayed fields newly exposed by the full modal without converting them to Inherit
- [x] #7 Rebase and remembered settings drafts retain exact custom endpoint registry IDs including hyphenated slugs
- [x] #8 Make default for a named endpoint keeps its exact hyphenated registry ID and configured endpoint through config reload and blank-chat resolution
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Reproduce the four mounted settings-loss cases; fix same-target revalidation without changing target-switch semantics; add inheritance and unexposed-field regressions; inspect nearby retention paths; run scoped tests and lint.
ADR required: no
ADR path: backlog/decisions/095-conversation-owned-console-generation-settings.md
Reason: Restore the accepted live-conversation/default ownership contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Same-target Apply and remembered A-B-A targets now retain their full conversation snapshot, including supported values hidden by quick settings and the live endpoint. Unseen targets still use target defaults; explicit Inherit edits resolve refreshed lower-precedence defaults. Exact registry provider IDs now survive controller rebase and remembered model drafts (console_settings_provider_key).

Nearby fixes found by mounted regressions: keep an absent endpoint-save intent absent during commit so full Make default no longer rejects its profile write as an unauthorized endpoint patch; when Quick transfers to Full settings, newly exposed fields retain the displayed effective value as their profile intent. These extend the original implementation plan through AC4–7 under the existing ADR-095 ownership contract; registry IDs follow ADR-146.

Files: tldw_chatbook/Chat/console_chat_controller.py; Chat/console_settings_apply.py; isolated _updated_field_draft fallback in Widgets/Console/console_settings_modal.py (coordinated with endpoint owner); new Tests/Chat/test_console_settings_retention.py and Tests/UI/test_console_settings_retention.py.

Evidence: initial 9 retention regressions failed with expected resets; subsequent mounted tests separately exposed full-default write rejection and Quick-to-Full profile loss, then passed after fixes. Final combined command: .venv/bin/python -m pytest -q Tests/UI/test_console_popover_context_window.py Tests/UI/test_console_settings_retention.py Tests/Chat/test_console_settings_retention.py Tests/Chat/test_console_settings_apply.py Tests/Chat/test_console_settings_defaults.py --tb=short --show-capture=no — 134 passed. Existing Apply mouse/keyboard, lifecycle persistence/resume/promotion, default runtime publication, and existing-chat ownership controls: 5 passed. The combined count includes 4 Quick context tests owned by TASK-32709.

Verification limits: a broader targeted store run had 71 passed and one unrelated existing roleplay argument assertion (persona_system_template=None absent from expectation); the identical failure was reproduced with the unchanged HEAD rebase method installed in-process. New tests pass Ruff lint and formatting; changed controller/popover/apply regions are formatted, git diff --check passes, and comparison against HEAD found no added Ruff diagnostics (controller 192, settings_apply 1, popover 4 pre-existing diagnostics). No full suite, live provider generation, or commit was performed.

Named endpoint defaults: preserved exact custom-ep IDs and immutable hyphenated slugs in both ChatScreen intent reservation paths and default-writer validation/endpoint matching via console_settings_provider_key. Mounted test_make_default_retains_hyphenated_entry_and_endpoint_on_reload passed: select custom-ep:gpu-node and registry-model, Make default, reload config, and resolve next-chat provider/model/endpoint without changing the registry ownership defined by ADR-146. Registry per-model generation defaults retain their existing explicit unsupported behavior.

Final named-entry integration verification after the coordinated screen/default-writer fix: .venv/bin/python -m pytest -q Tests/UI/test_console_endpoint_discovery.py::test_make_default_retains_hyphenated_entry_and_endpoint_on_reload Tests/Chat/test_console_settings_retention.py Tests/Chat/test_console_settings_defaults.py --tb=short --show-capture=no — 84 passed. All eight acceptance criteria are complete; existing ADR-095 and ADR-146 apply. Self-review and scoped diff/format/lint comparison completed; no unrelated source edits or commit.
<!-- SECTION:NOTES:END -->
