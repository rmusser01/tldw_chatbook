# ADR-032: Adopt textual-diff-view for tool-call diff rendering

## Status

Accepted (2026-08-05), implemented under TASK-1351 on branch `feat/toad-ui-improvements`.

## Context

Tool calls that write files (e.g. `WriteFileTool`) currently render in chat as truncated plain-text arguments/results (`tool_message_widgets.py`), with no structured view of what changed. Users cannot review an edit before or after execution in any meaningful way.

The `textual-diff-view` package (by Will McGugan, used by the `toad` ACP client) provides a standalone `DiffView` Textual widget: unified/split/auto view modes with width-based auto-switching, syntax highlighting, intra-line char diffs, hunk folding via `difflib.get_grouped_opcodes`, and strip-based rendering that scales to large diffs without per-line widgets. It has no coupling to `toad`.

## Decision

1. Add `textual-diff-view` as a pinned runtime dependency (exact pin, matching the deliberate-pin policy set for `textual` in TASK-1353).
2. Transport pattern: capture **before/after file contents** on file-writing tool-call records and diff client-side with `difflib` (via the widget). Do not parse or apply unified-diff patches in the UI.
3. Compute diffs off the UI thread (`await diff_view.prepare()`) before mounting.
4. Rendering site: the tool-execution area of the chat window, alongside/within the existing `ToolExecutionWidget` flow.

## Consequences

- New runtime dependency with a single-author bus factor; pinned exactly, so upgrades are deliberate. The widget is small enough to vendor later if the package is abandoned.
- File-writing tools must capture pre-write content, which adds a read before the first write per file per turn (bounded — capture is skipped entirely above a 256 KB per-side cap (`DIFF_CAPTURE_MAX_BYTES`), and only for tools that opt in).
- Raw before/after contents are display-only: they are stripped from DB-persisted tool messages and from the LLM continuation payload (`strip_diff_contents`), so diffs render from the in-memory record during the live session only; reloaded conversations show the text result.
- Lays the foundation for a future notes-sync conflict diff view (adjacent to task-97) using the same widget and split-view pattern.

## Alternatives considered

- **Hand-rolled diff rendering**: rejected — significant effort to match hunk folding, split/auto modes, and scalable strip rendering; the package is purpose-built and already proven in `toad`.
- **Patch-based transport (unified diffs from tools)**: rejected — requires a patch parser and loses syntax-highlighted full-context rendering; before/after contents with client-side `difflib` is simpler and more robust.
- **rich/textual built-ins only**: no maintained diff widget exists in core Textual.
