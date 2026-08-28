# TASK-17067 workspace root drift filter design

## Goal

Keep file-tool authority, change-review tracking, and the agent workspace note on one call-time folder-binding validity rule so those surfaces cannot drift apart.

## Approved design

Add one private iterator in `tldw_chatbook/Tools/workspace_file_roots.py`. It receives a registry and workspace id, walks bindings in registry order, and yields `(binding, Path)` only when the locator is an existing directory, is not a symlink, and resolves to itself.

The iterator owns one standardized warning for a symlinked or drifted binding. The warning is metadata-only and does not interpolate the user-controlled filesystem path. Missing directories remain silently excluded, matching current behavior.

The three consumers keep their existing responsibilities:

- `allowed_file_roots` resolves the run workspace, applies read/write access filtering, and prepends the private sandbox.
- `folder_binding_roots` applies the global and per-workspace change-review gates and includes both read-only and read-write bindings.
- `workspace_context_note` renders relative, sanitized labels and read-only annotations.

Registry failures continue to be handled by each caller because their fallback behavior differs. The shared iterator does not catch registry or filesystem exceptions that callers currently treat as a whole-operation failure.

## Alternatives rejected

- A shared predicate would leave three binding loops and three opportunities for policy drift.
- Structured accepted/rejected result objects would add types and branching that no caller needs.
- Caller-specific logging callbacks would preserve old wording but keep diagnostic ownership split; standardized logging was explicitly selected instead.

## Verification

Add a red-first routing regression that substitutes the private iterator and proves all three consumers use it. Keep the existing behavior tests unchanged and run the complete `Tests/Tools/test_workspace_file_roots.py` module. Regenerate the persistent-diagnostic inventory for the consolidated warning, then run its focused architecture check, scoped Ruff, Python compilation, and `git diff --check`.

## Governance

ADR required: no.

ADR path: `backlog/decisions/028-settings-workspaces-category-and-folder-roots.md`.

Reason: ADR-028 already requires call-time validation and run-bound roots. This refactor consolidates that policy without changing storage, permissions, ownership, or an external contract.
