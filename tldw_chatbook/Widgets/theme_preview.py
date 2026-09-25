"""Console-shaped theme preview, painted from a colours mapping (TASK-31259, TASK-32948)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

PREVIEW_ROWS = (
    ("rail", " Console ▸ Conversation · ready"),
    ("user", " You: summarise the attached paper"),
    ("assistant", " Assistant: Here is the summary…"),
    ("success", " ✓ tool web_search finished"),
    ("warning", " ! approval needed before the next call"),
    ("error", " ✗ provider returned 401"),
    ("accent", " [ Send ]   Ctrl+P palette"),
)
PREVIEW_STYLE = {
    "rail": ("panel", "foreground"),
    "user": ("primary", "foreground"),
    "assistant": ("surface", "foreground"),
    "success": ("background", "success"),
    "warning": ("background", "warning"),
    "error": ("background", "error"),
    "accent": ("background", "accent"),
}
_COMPACT_ROWS = ("rail", "accent")


class ThemePreview(Vertical):
    def __init__(self, prefix: str, *, compact: bool = False, **kwargs: Any) -> None:
        kwargs.setdefault("classes", "settings-theme-preview")
        super().__init__(**kwargs)
        self._prefix = prefix
        self._rows = tuple(r for r in PREVIEW_ROWS if not compact or r[0] in _COMPACT_ROWS)

    def compose(self) -> ComposeResult:
        for suffix, text in self._rows:
            yield Static(
                text,
                id=f"{self._prefix}-{suffix}",
                classes="settings-theme-preview-row",
                markup=False,  # task-32946: "[ Send ]" is literal text
            )

    def paint(self, colours: Mapping[str, str]) -> None:
        for suffix, _text in self._rows:
            background_key, foreground_key = PREVIEW_STYLE[suffix]
            try:
                row = self.query_one(f"#{self._prefix}-{suffix}", Static)
            except Exception:  # noqa: BLE001 - not mounted yet
                return
            try:
                if background := colours.get(background_key):
                    row.set_styles(background=background)  # ds-runtime: previewed palette
                if foreground := colours.get(foreground_key):
                    row.set_styles(color=foreground)  # ds-runtime: previewed palette
            except Exception:  # noqa: BLE001, S112 - a half-typed hex must not break painting
                continue
