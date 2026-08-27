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

## Architecture and deviation

ADR-090 is the governing accepted decision; Task 3 introduces no additional
architecture. The planned foundation conversation-policy getter/setter was absent at
base `450361d9cf`. With parent approval, Task 3 added only the missing conversation-
owned store getter/setter and direct `ChatPersistenceService` write-through parallel
to existing context-policy overrides. No controller, dependency, binding, legacy
settings surface, or footer hint was added.

## Impeccable / Ponytail review

Operate mode preserved the incumbent Neon Workbench Settings and Context modal
vocabulary. The implementation reuses the canonical Checkbox, Select, status row,
existing persistence worker, transcript reconciliation, and edit modal. No new visual
system or speculative abstraction was introduced.
