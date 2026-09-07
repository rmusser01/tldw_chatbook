# diff_widgets.py
"""
Diff rendering helpers for tool-call file edits.

Uses the `textual-diff-view` package (`DiffView` widget) to render file
before/after contents captured by file-writing tools as syntax-highlighted
unified/split diffs. See ADR-032
(`backlog/decisions/032-textual-diff-view-for-tool-call-diff-rendering.md`).
"""

from typing import Any, Dict, Literal, Optional, Tuple

from textual_diff_view import DiffView

from tldw_chatbook.Tools.file_operation_tools import DIFF_CONTENT_KEYS

DiffViewMode = Literal["unified", "split", "auto"]

# Default presentation for tool-call diffs. "auto" starts unified and flips
# to split when the terminal is wide enough (DiffView handles resize itself).
# Module-level on purpose: a settings UI is out of scope (TASK-1351).
DEFAULT_DIFF_VIEW_MODE: DiffViewMode = "auto"
DEFAULT_DIFF_WRAP = False
DEFAULT_DIFF_ANNOTATIONS = False


def set_default_diff_view_mode(mode: DiffViewMode) -> None:
    """Set the default view mode for newly created diff widgets.

    Args:
        mode: One of "unified", "split", or "auto".

    Raises:
        ValueError: If the mode is not recognized.
    """
    global DEFAULT_DIFF_VIEW_MODE
    if mode not in ("unified", "split", "auto"):
        raise ValueError(f"Unknown diff view mode: {mode!r}")
    DEFAULT_DIFF_VIEW_MODE = mode


def make_diff(
    path: str,
    code_before: Optional[str],
    code_after: Optional[str],
    *,
    id: Optional[str] = None,
    classes: Optional[str] = None,
) -> DiffView:
    """Make a diff view widget from before/after file contents.

    Args:
        path: Path of the edited file (used for the title and language guess).
        code_before: File content before the edit ("" for new files).
        code_after: File content after the edit.
        id: Textual CSS id.
        classes: Textual CSS classes.

    Returns:
        A configured DiffView widget. Callers mounting large diffs should
        `await diff_view.prepare()` first so the diff is computed off the UI
        thread.
    """
    mode = DEFAULT_DIFF_VIEW_MODE
    return DiffView(
        path,
        path,
        code_before or "",
        code_after or "",
        split=mode == "split",
        annotations=DEFAULT_DIFF_ANNOTATIONS,
        auto_split=mode == "auto",
        wrap=DEFAULT_DIFF_WRAP,
        id=id,
        classes=classes,
    )


def extract_diff_from_result(
    result: Dict[str, Any],
) -> Optional[Tuple[str, str, str]]:
    """Extract (path, old_content, new_content) from a tool result, if present.

    Tool results carry before/after contents as `result.old_content` /
    `result.new_content` (see `WriteFileTool`). Results without them, error
    results, and non-dict payloads return None.

    Args:
        result: A tool execution result record (`{"tool_call_id", "result"|"error"}`).

    Returns:
        (path, old_content, new_content) or None when there is nothing to diff.
    """
    if not isinstance(result, dict) or "error" in result:
        return None
    data = result.get("result")
    if not isinstance(data, dict):
        return None
    old_content = data.get("old_content")
    new_content = data.get("new_content")
    if not isinstance(old_content, str) or not isinstance(new_content, str):
        return None
    path = data.get("file_path") or "file"
    return str(path), old_content, new_content


def strip_diff_contents(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of a tool result record without raw diff contents.

    The before/after contents captured for UI diff rendering (TASK-1351) are
    display-only: they must not be persisted to the conversation DB or echoed
    back to the LLM (DB bloat, context bloat, replay leaks). This helper is
    non-mutating — the in-memory record keeps its contents so the live
    session can still render the diff.

    Args:
        result: A tool execution result record (`{"tool_call_id", "result"|"error"}`).

    Returns:
        A shallow copy with `result.old_content`/`result.new_content` removed,
        or the original record when there is nothing to strip.
    """
    if not isinstance(result, dict):
        return result
    data = result.get("result")
    if not isinstance(data, dict) or all(key not in data for key in DIFF_CONTENT_KEYS):
        return result
    stripped = dict(result)
    stripped["result"] = {
        key: value for key, value in data.items() if key not in DIFF_CONTENT_KEYS
    }
    return stripped
