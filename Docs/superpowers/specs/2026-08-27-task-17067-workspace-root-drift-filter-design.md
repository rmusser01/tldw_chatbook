# TASK-17067 workspace root drift filter design

## Goal

Keep file-tool authority, change-review tracking, and the agent workspace note on one call-time folder-binding validity rule so those surfaces cannot drift apart.

## Approved design

Add one private iterator in `tldw_chatbook/Tools/workspace_file_roots.py`. It receives a caller-supplied binding iterable and yields `(binding, Path)` in that iterable's order only when the locator is an existing directory, is not a symlink, and resolves to itself. Registry lookup stays outside the iterator so each caller retains its current fallback boundary.

The iterator preserves the current probe order for each supplied binding:

1. Build `Path(binding.locator)`.
2. If `is_dir()` is false, skip silently. This intentionally includes a broken symlink, matching current behavior.
3. If `is_symlink()` is true, warn and skip.
4. If `resolve() != folder`, warn and skip.
5. Otherwise yield `(binding, folder)`.

Both warned cases use the exact metadata-only message `Workspace folder binding excluded because its path no longer resolves to itself (symlink or mount drift)`. The warning has no formatting arguments and never interpolates the user-controlled filesystem path. Filesystem exceptions propagate to each caller's existing whole-operation fallback.

The three consumers keep their existing responsibilities:

- `allowed_file_roots` resolves the run workspace, filters read-only bindings before passing the remaining iterable to the helper when `write=True`, and prepends the private sandbox. This preserves the current rule that a write-only lookup never probes an irrelevant read-only path.
- `folder_binding_roots` applies the global gate before registry construction and the per-workspace gate before listing or validating bindings, then includes both read-only and read-write bindings.
- `workspace_context_note` renders relative, sanitized labels and read-only annotations.

Registry and filesystem failures continue to be handled by each caller because their fallback behavior differs. The shared iterator does not catch exceptions that callers currently treat as a whole-operation failure.

## Alternatives rejected

- A shared predicate would leave three binding loops and three opportunities for policy drift.
- Structured accepted/rejected result objects would add types and branching that no caller needs.
- Caller-specific logging callbacks would preserve old wording but keep diagnostic ownership split; standardized logging was explicitly selected instead.

## Verification

Add red-first regressions that substitute the private iterator and prove all three consumers use it, verify `allowed_file_roots(write=True)` filters read-only bindings before helper iteration, preserve both change-review gates before binding validation, and assert each consumer produces the exact standardized warning without the raw locator. Pin the `is_dir()`-before-symlink ordering with a broken-symlink case that remains silent. Keep the existing behavior tests unchanged and run the complete `Tests/Tools/test_workspace_file_roots.py` module. Regenerate the persistent-diagnostic inventory for the consolidated warning, then run its focused architecture check, scoped Ruff, Python compilation, and `git diff --check`.

## Governance

ADR required: no.

ADR path: `backlog/decisions/028-settings-workspaces-category-and-folder-roots.md`.

Reason: ADR-028 already requires call-time validation and run-bound roots. This refactor consolidates that policy without changing storage, permissions, ownership, or an external contract.
