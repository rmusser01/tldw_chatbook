"""Library skills canvas: list mode (rows + filter + sort).

Structural template copy of ``library_prompts_canvas.py``'s list-view
``compose`` -- only the list shape (header count line, filter Input, single
``ds-toolbar`` toolbar row, escaped row rendering) is mirrored here; skills
have no in-canvas editor yet (a later Skills task builds the detail/trust
editor on top of Task 2's ``SkillEditorState``/``build_skill_editor_state``).

Unlike the prompts list (where the secondary line is packed into the same
Button label as the name), each skill row renders its flags/description
line as a SEPARATE ``Static`` sibling right below the row Button -- per the
Task 3 brief's interface: the Button label is just ``f"{glyph} {name}"``.
"""

from __future__ import annotations

from typing import Any

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_skills_state import SkillsListState

_SORT_LABELS = {"name": "Name", "status": "Status"}
_EMPTY_SKILLS_COPY = "No skills yet — create them in Library ▸ Skills."
_EMPTY_SKILLS_FILTER_COPY = "No skills match your filter."


class LibrarySkillsListCanvas(Vertical):
    """Render the Library skills canvas's list view.

    Attributes:
        state: List-view display state (rows, count, sort). ``None``
            renders nothing (mirrors ``LibraryPromptsListCanvas``'s guard
            for a not-yet-available list state).
        sort_mode: Current skills sort mode key (``"name"``/``"status"``),
            used to label the sort control.
        filter_value: Current skills filter text, prefilled into the
            filter ``Input``.
    """

    def __init__(
        self,
        state: SkillsListState | None = None,
        *,
        sort_mode: str = "name",
        filter_value: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.sort_mode = sort_mode
        self.filter_value = filter_value
        self.styles.width = "1fr"
        self.styles.min_width = 40

    def compose(self) -> ComposeResult:
        state = self.state
        if state is None:
            return
        yield Static(
            f"Skills ({state.count})",
            id="library-skills-header",
            classes="destination-section",
            markup=False,
        )
        yield Input(
            placeholder="Filter skills… (Enter)",
            id="library-skills-filter",
            value=self.filter_value,
        )
        # One horizontal ds-toolbar row for sort/Import -- mirrors
        # library_prompts_canvas.py's toolbar exactly (same render-safe
        # shape: every child is a fixed-width compact Button). Import… has
        # no handler wired yet (a later Skills task) -- same
        # inert-but-selectable posture Task 1 gave the rail row itself
        # before this canvas existed.
        toolbar = Horizontal(classes="ds-toolbar")
        toolbar.styles.height = "auto"
        with toolbar:
            yield Button(
                f"sort: {_SORT_LABELS.get(self.sort_mode, 'Name')} ▸",
                id="library-skills-sort", classes="library-canvas-action", compact=True,
            )
            yield Button(
                "Import…", id="library-skills-import",
                classes="library-canvas-action", compact=True,
            )
        if not state.rows:
            yield Static(
                _EMPTY_SKILLS_FILTER_COPY if self.filter_value else _EMPTY_SKILLS_COPY,
                id="library-skills-empty",
                markup=False,
            )
            return
        with Vertical(id="library-skills-list"):
            for row in state.rows:
                # Skill names are unique + name-shaped (lowercase
                # alphanumerics and hyphens only, per
                # ``local_skills_service._AGENT_SKILL_NAME_PATTERN``,
                # enforced at save time), so they're safe verbatim as a DOM
                # id suffix -- same posture as the prompt row's integer
                # ``prompt_id``, just a string here instead.
                name = escape_markup(row.name)
                classes = "library-skill-row"
                if row.blocked:
                    classes = f"{classes} library-skill-row-blocked"
                button = Button(
                    f"{row.trust_glyph} {name}",
                    id=f"library-skill-row-{row.name}",
                    classes=classes,
                    compact=True,
                )
                button.skill_name = row.name
                yield button
                if row.secondary:
                    # The flags/description line is user-controlled (the
                    # skill's free-text description) and rendered as its
                    # own Static, NOT packed into the Button label above --
                    # escaped the same way the prompts canvas escapes its
                    # secondary line, so a description containing "[x]"
                    # renders verbatim instead of being eaten as an
                    # (unmatched) Rich markup tag.
                    yield Static(
                        escape_markup(row.secondary),
                        classes="library-skill-row-secondary",
                    )
