"""Exact local sample editing and bounded file admission."""

from __future__ import annotations

from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Static, TextArea

from tldw_chatbook.Utils.path_validation import validate_path_simple

SAMPLE_BYTES = 2 * 1024 * 1024


def read_sample_file(selected: str) -> tuple[str, dict]:
    """Read one explicitly selected UTF-8 file, refusing unbounded reads.

    Args:
        selected: User-selected absolute file path.

    Returns:
        Exact text and descriptive source metadata.
    """
    path = validate_path_simple(selected, require_exists=True)
    if not path.is_file():
        raise ValueError("Choose a regular UTF-8 text file")
    with path.open("rb") as stream:
        payload = stream.read(SAMPLE_BYTES + 1)
    if len(payload) > SAMPLE_BYTES:
        raise ValueError(
            "File exceeds 2 MiB. Choose an explicit excerpt with start/end character positions."
        )
    return payload.decode("utf-8", errors="strict"), {
        "kind": "file",
        "name": Path(selected).name,
    }


def read_sample_excerpt(selected: str, start: int, end: int) -> tuple[str, dict]:
    """Read a user-selected character range, retaining exact line endings."""
    if start < 0 or end <= start or end - start > SAMPLE_BYTES:
        raise ValueError("Choose a nonempty character range within the 2 MiB limit")
    path = validate_path_simple(selected, require_exists=True)
    pieces, position = [], 0
    with path.open("r", encoding="utf-8", errors="strict", newline="") as stream:
        while position < end:
            text = stream.read(min(65536, end - position))
            if not text:
                raise ValueError("Excerpt end exceeds the file's character count")
            if position + len(text) > start:
                pieces.append(text[max(0, start - position) :])
            position += len(text)
    excerpt = "".join(pieces)
    if len(excerpt.encode("utf-8")) > SAMPLE_BYTES:
        raise ValueError("Selected excerpt exceeds 2 MiB UTF-8; choose a smaller range")
    return excerpt, {
        "kind": "file_excerpt",
        "name": path.name,
        "start": start,
        "end": end,
    }


class SampleRegion(Vertical):
    """Sample editor emits copied edit requests; the screen owns transitions."""

    BUNDLED_CSS = """
    SampleRegion { height: 1fr; min-height: 0; padding: 0 1; }
    SampleRegion Static { height: auto; }
    SampleRegion TextArea { height: 1fr; min-height: 4; }
    """

    class Changed(Message):
        def __init__(self, text: str):
            self.text = text
            super().__init__()

    def compose(self) -> ComposeResult:
        yield Static("Sample · copied text", markup=False)
        yield Static(
            "Paste text below, or use Files / Recovery to load UTF-8 text.",
            markup=False,
        )
        yield TextArea(id="lab-sample-text", soft_wrap=True)
        yield Static(
            "2 MiB UTF-8 maximum · full text and results stay in local recovery",
            markup=False,
        )

    @on(TextArea.Changed, "#lab-sample-text")
    def sample_changed(self, event: TextArea.Changed) -> None:
        event.stop()
        self.post_message(self.Changed(event.text_area.text))
