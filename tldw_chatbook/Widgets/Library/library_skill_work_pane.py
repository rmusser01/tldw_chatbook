"""Permanent Skills work pane for Overview, Edit, Trust, Files, and import."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from tldw_chatbook.Library.library_skills_state import (
    SkillReaderMode,
    coerce_skill_reader_mode,
    skill_invocation_copy,
    skill_review_identity_line,
)

from .library_skills_canvas import (
    LibrarySkillsListCanvas,
    skill_supporting_files_text,
)


class LibrarySkillWorkPane(LibrarySkillsListCanvas):
    """Render non-list Skills content while the concrete list stays mounted."""

    def __init__(
        self,
        *,
        reader_mode: SkillReaderMode | str = "overview",
        **kwargs: Any,
    ) -> None:
        """Initialize the retained Skills work pane.

        Args:
            reader_mode: Initial Overview, Edit, Trust, or Files projection.
            **kwargs: Additional arguments forwarded to the Skills list canvas.
        """
        super().__init__(
            show_editor_trust=False,
            show_editor_files=False,
            **kwargs,
        )
        self.reader_mode = coerce_skill_reader_mode(reader_mode)
        self.remove_class("library-adaptive-reader-items")
        self.styles.min_width = 0

    def compose(self) -> ComposeResult:
        """Compose the active task, load state, or explicit reader mode.

        Returns:
            The widgets for import, loading, empty, or active reader content.
        """
        if self.import_open:
            yield Static(
                "Import skills",
                id="library-skill-import-heading",
                classes="destination-section",
                markup=False,
            )
            yield from self._compose_import_row()
            return
        if self.mode == "loading":
            yield from super().compose()
            return
        if self.mode != "editor" or self.editor_state is None:
            yield Static(
                "Select a skill to inspect it here.",
                id="library-skill-work-empty",
                classes="destination-purpose",
                markup=False,
            )
            return

        yield from self._compose_mode_strip()
        state = self.editor_state
        if self.reader_mode == "overview":
            region = Vertical(id="library-skill-overview-region")
            region.styles.height = "auto"
            with region:
                yield Static(state.name, classes="destination-section", markup=False)
                yield Static(
                    state.description or "No description set.",
                    classes="destination-purpose",
                    markup=False,
                )
                yield Static(
                    skill_invocation_copy(
                        state.user_invocable,
                        state.disable_model_invocation,
                    ),
                    id="library-skill-overview-invocation",
                    markup=False,
                )
                yield Static(
                    f"Version: {state.version if state.version is not None else 'unknown'}",
                    id="library-skill-overview-version",
                    markup=False,
                )
                yield Static(
                    f"Trust: {state.trust_status.replace('_', ' ')}",
                    id="library-skill-overview-trust",
                    markup=False,
                )
            return
        if self.reader_mode == "edit":
            region = Vertical(id="library-skill-edit-region")
            region.styles.height = "auto"
            with region:
                yield from self._compose_editor()
            return
        if self.reader_mode == "trust":
            region = Vertical(id="library-skill-trust-region")
            region.styles.height = "auto"
            with region:
                review_identity = skill_review_identity_line(self.active_review)
                yield Static(
                    review_identity,
                    id="library-skill-trust-review-identity",
                    markup=False,
                )
                yield from self._compose_trust_panel(state)
            return
        region = Vertical(id="library-skill-files-region")
        region.styles.height = "auto"
        with region:
            yield Static(
                "Supporting files", classes="destination-section", markup=False
            )
            yield Static(
                "Read-only in Library. File contents are not editable here.",
                id="library-skill-files-read-only",
                classes="destination-purpose",
                markup=False,
            )
            yield Static(
                skill_supporting_files_text(state.supporting_files),
                id="library-skill-supporting",
                markup=False,
            )

    def _compose_mode_strip(self) -> ComposeResult:
        """Render the four explicit Skills work modes."""
        toolbar = Horizontal(id="library-skill-mode-strip", classes="ds-toolbar")
        toolbar.styles.height = "auto"
        with toolbar:
            for mode, label in (
                ("overview", "Overview"),
                ("edit", "Edit"),
                ("trust", "Trust"),
                ("files", "Files"),
            ):
                classes = "library-canvas-action"
                if self.reader_mode == mode:
                    classes = f"{classes} -active"
                yield Button(
                    label,
                    id=f"library-skill-mode-{mode}",
                    classes=classes,
                    compact=True,
                    disabled=self.is_create and mode != "edit",
                )

    def sync_state(
        self,
        *,
        reader_mode: SkillReaderMode | str = "overview",
        **kwargs: Any,
    ) -> None:
        """Apply the current work projection and recompose only on change.

        Args:
            reader_mode: Requested Overview, Edit, Trust, or Files projection.
            **kwargs: Additional state forwarded to the Skills list canvas.
        """
        requested = coerce_skill_reader_mode(reader_mode)
        if requested == self.reader_mode and all(
            getattr(self, key, object()) == value for key, value in kwargs.items()
        ):
            return
        self.reader_mode = requested
        super().sync_state(**kwargs)
