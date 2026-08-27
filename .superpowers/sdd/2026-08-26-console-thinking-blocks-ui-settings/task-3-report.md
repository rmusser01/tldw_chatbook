# Task 3 report — Console thinking visibility and replay controls

## Outcome

Added the default-on, device-local `Show model thinking` control to canonical F9
Console Behavior. It applies immediately through the existing appearance refresh
seam, persists through the existing bounded Settings adapter, and rolls back with an
error if persistence fails. Hiding filters only projected thinking activities; it
retains captured history, Assistant/Tool widget identity and expansion state, and
restores historical rows collapsed or still-pending live rows expanded.

Added current-conversation Auto/Include/Exclude replay policy to the existing Context
& memory modal, including an effective Required state with continuation-specific
reason, a bounded new-conversation-default action, and a backward-compatible modal
result. Fresh sessions copy the resolved device default once; existing Auto/NULL
sessions are not rewritten. Explicit Assistant edits now name the thinking and
continuation provenance that Save clears through the existing edit path.

## TDD evidence

- RED: collection failed because the settings loaders did not exist.
- RED: invalid hand-edited `show_model_thinking` resolved False instead of the
  required safe True fallback; the lifecycle test also exposed a missing `pytest`
  import.
- GREEN: 21 Settings/context-memory tests passed.
- GREEN: 66 store lifecycle/edit-modal tests passed.
- GREEN: 3 mounted F9 rollback and transcript visibility/lifecycle tests passed.
- GREEN: 2 bounded default-write/new-session-copy tests passed.
- Scoped Ruff lint passed. `git diff --check` passed before this report update.
- The exact plan filter completed 117/118 assertions; its sole failure is the
  unchanged `test_console_inspector_hosts_staged_context_above_source_readiness`
  project-instruction DOM-parent expectation and reproduced alone. Task 3 does not
  touch that widget or test, so completion relies on the 92 scoped GREEN tests above.
- Whole-file Ruff format remains an inherited baseline on large legacy Settings,
  transcript, and adjacent test files; formatter output included broad unrelated
  churn, so Task 3 did not reformat those files wholesale.

Pytest emitted post-success temporary-directory cleanup warnings for unrelated sync
promotion fixtures; no Task 3 assertion failed in the completed focused runs.

## Review fix round 1

The first independent review found three Priority-1 gaps and all three were reproduced
before implementation:

- RED: three continuation-policy cases failed because the controller had no
  effective-policy seam, and the UI builder rejected the new effective-policy input.
  GREEN: 5 focused cases passed. The modal/popover now ask the existing provider
  resolver for the actual frozen send target and reuse the controller's existing
  continuation-group selector. One shared pure helper maps the saved optional value
  plus compatible completed continuation groups to the effective value. Compatible
  completed replay is Required; active recovery, incompatible history, a changed
  target, and provider-resolution failure retain the saved Auto/Include/Exclude value.
- RED: the store rejected the proposed live-default provider argument and the runtime
  created Auto instead of the configured Exclude value. GREEN: 2 focused store/runtime
  cases passed. ConsoleChatStore.create_session now reads one runtime-wired live
  default on every omitted-policy creation, covering ordinary, character, workspace,
  first-chat, and other callers through their shared store boundary. Explicit values
  and restored durable values bypass the provider; legacy durable NULL remains Auto.
  The one-path post-create setter and its duplicate UI resolver were removed.
- RED: the rapid-toggle regression raised on the legacy worker signature and left the
  optimistic value inconsistent. GREEN: the focused coalescing and worker-overlap
  cases passed. Visibility persistence now has one in-flight writer, a newest desired
  revision/value, and a confirmed persisted baseline. Completion drains only the
  newest differing value; a latest failure rolls the checkbox, app config, label, and
  transcript projection back to the confirmed baseline and emits the existing error.
  The worker-overlap test also proves the surviving persisted value is the value a
  restart loads.

Final review-round verification: the exact affected Settings/Context/session/edit
filter completed **128 passed, 460 deselected** in 67.30s. Scoped Ruff lint passed on
all 12 changed implementation/test files. Pytest again emitted only the documented
post-success temporary-directory cleanup warnings.

## Architecture and deviation

ADR-090 is the governing accepted decision; Task 3 introduces no additional
architecture. The planned foundation conversation-policy getter/setter was absent at
base `450361d9cf`. With parent approval, Task 3 added only the missing conversation-
owned store getter/setter and direct `ChatPersistenceService` write-through parallel
to existing context-policy overrides. No controller, dependency, binding, legacy
settings surface, or footer hint was added.

Review round 1 remains a direct implementation of ADR-090: it makes the UI's Required
projection consume the existing provider/send ownership boundary, moves future-session
default resolution to the store boundary that owns session creation, and serializes an
already-existing device-local presentation write. No new ADR is required.

## Impeccable / Ponytail review

Operate mode preserved the incumbent Neon Workbench Settings and Context modal
vocabulary. The implementation reuses the canonical Checkbox, Select, status row,
existing persistence worker, transcript reconciliation, and edit modal. No new visual
system or speculative abstraction was introduced.
