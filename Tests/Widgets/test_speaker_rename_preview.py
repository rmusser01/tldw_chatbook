"""A speaker rename must not replace the metadata preview with the transcript.

TASK-32811.4 (first of three Library sub-defects). Renaming a speaker in
Library ▸ Media called `_refresh_after_speaker_rename`, which wrote the
entire rewritten `Media.content` (the whole transcript) into
`#library-media-preview-lines` -- a pane composed from `canvas.preview_lines`,
which is exactly three metadata lines (title/type/date). The rename rewrites
transcript text, not that metadata, so the preview must keep rendering the
three lines.

The Media canvas needs a mounted app to exercise at runtime, which this
worktree's storage-admission gate blocks, so this is a structural pin over
the method's source: it must render `preview_lines`, never the raw
`content`.
"""

from __future__ import annotations

import ast
import inspect

from tldw_chatbook.Widgets.Library import library_media_canvas


def _method_source(name: str) -> str:
    cls = library_media_canvas.LibraryMediaCanvasBinding if hasattr(
        library_media_canvas, "LibraryMediaCanvasBinding"
    ) else None
    # The method lives on the canvas widget class; find it by name in source.
    source = inspect.getsource(library_media_canvas)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"{name} not found")


def test_refresh_renders_preview_lines_not_the_transcript():
    body = _method_source("_refresh_after_speaker_rename")
    # It must render the three-line metadata preview...
    assert "preview_lines" in body, (
        "the refresh no longer renders the metadata preview lines"
    )
    # ...and must NOT write the raw Media.content into the preview pane.
    assert 'update(content)' not in body, (
        "the refresh still dumps the whole transcript into the preview pane"
    )
    assert 'row["content"]' not in body, (
        "the refresh still reads the transcript body to display it"
    )
