# Task 3 brief — Console thinking visibility and replay controls

## Scope

Implement only plan Task 3: default-on device visibility in canonical F9 Settings, current-conversation Auto/Include/Exclude replay policy in the existing Context & memory modal, new-conversation default propagation, and explicit assistant-edit provenance-loss copy.

## Constraints

- Reuse the existing Settings worker/refresh, Context modal/result, session/store persistence, transcript reconciliation, and edit modal paths.
- No legacy settings surface, new controller, dependency, binding, or footer hint.
- Visibility is presentation-only and must not alter capture, persistence, replay, token accounting, sync, or exports.
- Conversation policy remains conversation-owned; device defaults affect only new conversations.
- ADR-090 is the governing decision. A planned foundation getter/setter is absent at base `450361d9cf`; add only the minimal parallel store/repository seam.

## TDD contract

1. RED: config default/coercion, mounted F9 control, visibility projection, policy control/result, durable setter/new-session default, and edit provenance tests fail for missing behavior.
2. GREEN: implement the smallest changes through incumbent seams.
3. Verify plan-targeted tests plus nearest transcript/config/session tests, Ruff, and `git diff --check`.
