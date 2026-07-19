"""Library skills canvas: list mode (rows + filter + sort) + detail editor.

Structural template copy of ``library_prompts_canvas.py``'s list-view
``compose`` -- only the list shape (header count line, filter Input, single
``ds-toolbar`` toolbar row, escaped row rendering) is mirrored for the list
view. Unlike the prompts list (where the secondary line is packed into the
same Button label as the name), each skill row renders its flags/description
line as a SEPARATE ``Static`` sibling right below the row Button -- per the
Task 3 brief's interface: the Button label is just ``f"{glyph} {name}"``.

Task 4 adds the in-canvas SKILL.md detail/trust editor (``mode="editor"``),
structurally templated on ``LibraryPromptsListCanvas._compose_editor``: a
Back button, stacked full-width fields, a warnings line, a trust panel, and
a single plain ``ds-toolbar`` action row. Two deliberate deviations from the
brief's parenthetical widget hints, matching this canvas family's own
documented render-safety discipline (see
``library_notes_canvas.py._compose_sync``'s docstring: "Notably absent:
``Select``... and ``Switch``... neither renders reliably in this canvas"):
``user_invocable``/``disable_model_invocation`` are toggle Buttons (not
Checkbox/Switch) and ``context`` is a cycling Button (not Select), the same
"cycling/toggle Buttons instead" posture the media type filter and notes
sort control already use.

Task 5 adds the list view's inline Import row (``import_open``), a
structural template copy of ``LibraryPromptsListCanvas``'s own Import row.
"""

from __future__ import annotations

from typing import Any, Mapping

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Static, TextArea

from tldw_chatbook.Library.library_skills_state import (
    SkillEditorState,
    SkillsListState,
    save_marks_needs_review,
    skill_name_shadows_builtin,
)

_SORT_LABELS = {"name": "Name", "status": "Status"}
_EMPTY_SKILLS_COPY = "No skills yet — create them in Library ▸ Skills."
_EMPTY_SKILLS_FILTER_COPY = "No skills match your filter."

# Trust panel copy/gating (Task 4). Mirrors ``skills_screen.py``'s own
# ``_skill_trust_copy``/``SKILLS_TRUST_REVIEWABLE_STATUSES`` values (kept a
# separate, smaller copy here rather than importing that screen's private
# helpers -- this editor only needs the trust STATE line + two gating
# predicates, not that screen's fuller blocked-reason copy).
_TRUST_REVIEWABLE_STATUSES = frozenset((
    "quarantined_modified",
    "quarantined_added",
    "quarantined_deleted",
))
_TRUST_STATE_COPY = {
    "trusted": "Trust: trusted",
    "trust_uninitialized": "Trust: not initialized",
    "trust_locked": "Trust: locked",
    "quarantined_modified": "Trust: changed since trusted baseline",
    "quarantined_added": "Trust: new untrusted file",
    "quarantined_deleted": "Trust: trusted file missing",
    "quarantined_manifest_error": "Trust: manifest cannot be verified",
    "quarantined_unsupported_path": "Trust: unsupported file path",
}
# Exact copy pinned by the Task 4 brief -- both the canvas's initial render
# and the screen's targeted (no-recompose) live updates
# (``LibraryScreen._update_library_skill_warnings_static``) must agree on
# this literal text, so it lives in ONE place (``skill_editor_warning_lines``
# below), imported by both.
_SHADOW_WARNING_TEMPLATE = (
    'Name shadows a built-in command/tool ("{name}") — it will not be '
    "invocable as /{name} or as an agent tool."
)
_NEEDS_REVIEW_WARNING = (
    'Saving marks this skill "needs review" — re-approve it in the trust '
    "panel after saving."
)
MODEL_HINT_COPY = "Not applied in v1."

# Fix wave (Skills Phase-1 gate, FIX 2): a brand-new install has no trust
# manifest at all (``trust_status == "trust_uninitialized"``) -- the Library
# editor's Unlock action only ever unlocks an EXISTING manifest, so the
# normal Unlock/Review/Approve row would render as a permanent dead end.
# This copy/predicate pair backs a dedicated first-run panel state instead
# (see ``_compose_trust_panel``): an explanation line plus a single "Set up
# skill trust" action that drives the real ``bootstrap_trust`` primitive
# through a confirm-passphrase modal.
_TRUST_SETUP_EXPLANATION_COPY = (
    "Local skill trust hasn't been set up yet. Set a trust passphrase to "
    "start reviewing and approving local skills — current local skill "
    "files become the trusted baseline."
)


def skill_trust_needs_setup(trust_status: str) -> bool:
    """Return whether the trust panel should render its first-run setup state."""
    return trust_status == "trust_uninitialized"


def skill_trust_state_line(trust_status: str, changed_files: tuple[str, ...] = ()) -> str:
    """Render the trust panel's current-state line.

    Args:
        trust_status: The skill's current trust status.
        changed_files: Files changed since the trusted baseline (only
            meaningful while blocked); appended parenthetically when
            non-empty.

    Returns:
        A one-line human-readable trust state summary.
    """
    line = _TRUST_STATE_COPY.get(trust_status, "Trust: blocked")
    if changed_files:
        line = f"{line} ({', '.join(changed_files)})"
    return line


def skill_trust_unlock_enabled(trust_status: str) -> bool:
    """Return whether the trust panel's Unlock action should be enabled."""
    return trust_status == "trust_locked"


def skill_trust_review_enabled(trust_status: str, trust_blocked: bool) -> bool:
    """Return whether the trust panel's Review changes action should be enabled."""
    return bool(trust_blocked) and trust_status in _TRUST_REVIEWABLE_STATUSES


def skill_editor_warning_lines(
    *, live_name: str, trust_status: str, trust_blocked: bool,
) -> tuple[str, ...]:
    """Build the editor's non-blocking warning lines.

    Args:
        live_name: The Name field's current (possibly unsaved) value.
        trust_status: The open skill's current trust status.
        trust_blocked: Whether the open skill is currently trust-blocked.

    Returns:
        Zero, one, or both of: the shadow-name warning (live, name-driven)
        and the save-marks-needs-review warning (only while currently
        trusted and not already blocked -- see ``save_marks_needs_review``).
    """
    lines: list[str] = []
    shadow = skill_name_shadows_builtin(live_name)
    if shadow:
        lines.append(_SHADOW_WARNING_TEMPLATE.format(name=shadow))
    if save_marks_needs_review(trust_status, trust_blocked):
        lines.append(_NEEDS_REVIEW_WARNING)
    return tuple(lines)


def skill_user_invocable_label(value: bool) -> str:
    """Render the user-invocable toggle Button's label."""
    return f"user invocable: {'yes' if value else 'no'} ▸"


def skill_disable_model_label(value: bool) -> str:
    """Render the disable-model-invocation toggle Button's label."""
    return f"disable model invocation: {'yes' if value else 'no'} ▸"


def skill_context_toggle_label(context: str) -> str:
    """Render the context-cycling Button's label."""
    return f"context: {context} ▸"


def next_skill_context(context: str) -> str:
    """Cycle the skill editor's ``context`` field between ``inline``/``fork``."""
    return "fork" if context == "inline" else "inline"


def skill_supporting_files_text(supporting_files: tuple[tuple[str, int], ...]) -> str:
    """Render the read-only supporting-files list as plain text."""
    if not supporting_files:
        return "No supporting files."
    return "\n".join(f"{name} ({size} bytes)" for name, size in supporting_files)


class LibrarySkillsListCanvas(VerticalScroll):
    """Render the Library skills canvas: the list view, or the skill editor.

    ``VerticalScroll`` root (the L3a clipping lesson -- a plain ``Vertical``
    canvas clips content past the fold, and the editor's Trust panel/
    Save-Delete row sit below the fold at ordinary terminal sizes): same
    house pattern already used by ``LibraryExportCanvas``/
    ``LibraryIngestCanvas``. This gives mouse-wheel scroll, the default
    keyboard scroll bindings (up/down/pageup/pagedown/home/end), and
    focus-jump-into-view (e.g. tabbing into the Trust panel) for free --
    all built into Textual's ``ScrollableContainer``, not custom code here.

    Attributes:
        state: List-view display state (rows, count, sort). ``None``
            renders nothing. Only used when ``mode == "list"``.
        sort_mode: Current skills sort mode key (``"name"``/``"status"``),
            used to label the sort control.
        filter_value: Current skills filter text, prefilled into the
            filter ``Input``.
        mode: ``"list"`` renders the skills list; ``"editor"`` renders the
            in-canvas SKILL.md detail/trust editor for ``editor_state``.
        editor_state: The skill to render in editor mode. Required when
            ``mode == "editor"``.
        warnings: Screen-computed warning text (see
            ``skill_editor_warning_lines``), joined with ``"\\n"``; ``""``
            when there is nothing to warn about.
        status: Save-outcome status text shown below the warnings line
            (e.g. ``"Saved."``), or ``""`` when idle. Not shown while
            ``conflict`` is set.
        conflict: When ``True`` (editor mode only), renders the save
            conflict banner (a quiet explanatory line plus a Reload action)
            in place of the normal Save/Delete action row.
        active_review: The trust panel's currently-captured review mapping
            (from ``capture_review``'s result), or ``None`` when no review
            has been captured for the open skill yet. Only its
            ``changed_files`` entry is rendered; presence/absence alone
            gates the Approve action.
        is_create: Whether the open editor is creating a brand-new skill
            (reached via the Create rail's "New skill" row) rather than
            editing one that already exists on disk. The service has no
            rename primitive, so an EXISTING skill's Name Input is
            disabled (with a dim hint) instead of letting a user silently
            corrupt the skill by changing it -- only the create branch
            renders it editable.
        import_open: List-view only (Task 5). When ``True``, renders the
            inline Import row (a path Input for a SKILL.md file OR a
            skill's own directory, plus Browse/Import/Cancel actions)
            below the sort/Import… toolbar -- structural template copy of
            ``LibraryPromptsListCanvas``'s own Import row.
        import_path: The Import row's path ``Input`` prefilled value. Only
            meaningful while ``import_open`` is ``True``.
        import_status: Muted outcome line shown below the Import row
            (e.g. ``"1 imported · re-review it in the trust panel"``), or
            ``""`` when idle/not yet run.
    """

    def __init__(
        self,
        state: SkillsListState | None = None,
        *,
        sort_mode: str = "name",
        filter_value: str = "",
        mode: str = "list",
        editor_state: SkillEditorState | None = None,
        warnings: str = "",
        status: str = "",
        conflict: bool = False,
        active_review: Mapping[str, Any] | None = None,
        is_create: bool = False,
        import_open: bool = False,
        import_path: str = "",
        import_status: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.sort_mode = sort_mode
        self.filter_value = filter_value
        self.mode = mode
        self.editor_state = editor_state
        self.warnings = warnings
        self.status = status
        self.conflict = conflict
        self.active_review = active_review
        self.is_create = is_create
        self.import_open = import_open
        self.import_path = import_path
        self.import_status = import_status
        self.styles.width = "1fr"
        self.styles.min_width = 40

    def compose(self) -> ComposeResult:
        if self.mode == "editor":
            yield from self._compose_editor()
            return
        yield from self._compose_list()

    def _compose_list(self) -> ComposeResult:
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
        # shape: every child is a fixed-width compact Button).
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
        if self.import_open:
            yield from self._compose_import_row()
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

    def _compose_import_row(self) -> ComposeResult:
        """Render the inline Import row: a path Input, then a Run/Cancel
        action toolbar, then the outcome line.

        Structural template copy of
        ``LibraryPromptsListCanvas._compose_import_row``: the path
        ``Input`` is its own full-width sibling -- NOT packed into a
        ``Horizontal`` alongside the action Buttons -- same render-safe
        shape this canvas family documents throughout (mixing a 1fr-width
        Input with fixed-width compact Buttons in one ``Horizontal`` is
        this family's known non-rendering failure mode).

        Unlike the prompts Import row, the placeholder copy mentions a
        skill's own directory too: every real skill package (e.g. the
        ``superpowers`` skillset) is a directory named after the skill
        containing a literally-named ``SKILL.md`` file, so pointing the
        path Input at either the ``SKILL.md`` file itself or its parent
        directory both resolve to the same skill name (see
        ``_run_library_skills_import``).
        """
        yield Input(
            placeholder="SKILL.md file or skill folder path…",
            id="library-skills-import-path",
            value=self.import_path,
        )
        toolbar = Horizontal(classes="ds-toolbar")
        toolbar.styles.height = "auto"
        with toolbar:
            # Browse… picks a FILE via the same FileOpen dialog the
            # prompts/media-ingest Browse actions use -- that dialog has no
            # directory-selection mode, so importing a skill BY ITS FOLDER
            # path still has to be typed by hand into the path Input above.
            yield Button(
                "Browse…", id="library-skills-import-browse",
                classes="library-canvas-action", compact=True,
            )
            yield Button(
                "Import", id="library-skills-import-run",
                classes="library-canvas-action", compact=True,
            )
            yield Button(
                "Cancel", id="library-skills-import-cancel",
                classes="library-canvas-action", compact=True,
            )
        yield Static(
            self.import_status,
            id="library-skills-import-status",
            markup=False,
        )

    def _compose_editor(self) -> ComposeResult:
        """Render the SKILL.md editor: Back, fields, warnings, trust panel, actions.

        Structural template copy of
        ``LibraryPromptsListCanvas._compose_editor``: stacked full-width
        widgets plus a single plain ``ds-toolbar`` action row. See the
        module docstring for the Checkbox/Switch/Select deviations.
        """
        editor_state = self.editor_state
        if editor_state is None:
            return
        yield Button(
            "‹ Back to list",
            id="library-skill-back",
            classes="library-canvas-action",
            compact=True,
        )
        yield Static("Name", classes="library-prompt-field-label", markup=False)
        yield Input(
            value=editor_state.name,
            id="library-skill-name",
            disabled=not self.is_create,
        )
        if not self.is_create:
            yield Static(
                "Rename isn't supported — create a new skill instead.",
                id="library-skill-name-hint",
                classes="library-prompt-field-hint",
                markup=False,
            )
        yield Static("Description", classes="library-prompt-field-label", markup=False)
        yield Input(value=editor_state.description, id="library-skill-description")
        yield Static("Argument hint", classes="library-prompt-field-label", markup=False)
        yield Input(value=editor_state.argument_hint or "", id="library-skill-argument-hint")
        yield Static("Allowed tools", classes="library-prompt-field-label", markup=False)
        yield Input(
            value=editor_state.allowed_tools_csv,
            placeholder="Allowed tools (comma-separated)",
            id="library-skill-allowed-tools",
        )
        yield Button(
            skill_user_invocable_label(editor_state.user_invocable),
            id="library-skill-user-invocable",
            classes="library-canvas-action",
            compact=True,
        )
        yield Button(
            skill_disable_model_label(editor_state.disable_model_invocation),
            id="library-skill-disable-model",
            classes="library-canvas-action",
            compact=True,
        )
        yield Button(
            skill_context_toggle_label(editor_state.context),
            id="library-skill-context",
            classes="library-canvas-action",
            compact=True,
        )
        yield Static("Model override", classes="library-prompt-field-label", markup=False)
        yield Input(value=editor_state.model or "", id="library-skill-model")
        yield Static(
            MODEL_HINT_COPY,
            id="library-skill-model-hint",
            classes="library-prompt-field-hint",
            markup=False,
        )
        yield Static("Body", classes="library-prompt-field-label", markup=False)
        yield TextArea(editor_state.body, id="library-skill-body")
        yield Static("Supporting files", classes="library-prompt-field-label", markup=False)
        yield Static(
            skill_supporting_files_text(editor_state.supporting_files),
            id="library-skill-supporting",
            markup=False,
        )
        yield Static(self.warnings, id="library-skill-warnings", markup=False)
        if self.conflict:
            yield Static(
                "This skill changed elsewhere — Reload discards your edit and refetches it.",
                id="library-skill-conflict-copy",
                classes="destination-purpose",
                markup=False,
            )
        else:
            yield Static(self.status, id="library-skill-save-status", markup=False)
        yield from self._compose_trust_panel(editor_state)
        toolbar = Horizontal(classes="ds-toolbar")
        toolbar.styles.height = "auto"
        with toolbar:
            if self.conflict:
                yield Button(
                    "Reload",
                    id="library-skill-conflict-reload",
                    classes="library-canvas-action",
                    compact=True,
                )
            else:
                yield Button(
                    "Save",
                    id="library-skill-save",
                    classes="library-canvas-action",
                    compact=True,
                )
                yield Button(
                    "Delete",
                    id="library-skill-delete",
                    classes="library-canvas-action library-media-action-danger",
                    compact=True,
                )

    def _compose_trust_panel(self, editor_state: SkillEditorState) -> ComposeResult:
        """Render the trust panel: state line, changed-files, Unlock/Review/Approve.

        The changed-files Static is ALWAYS present (empty text when no
        review is active) rather than mounted/removed on demand -- simpler
        than a D3-style targeted mount/remove, and matches how
        ``#library-skill-save-status`` is always present too.

        Fix wave (Phase-1 gate, FIX 2): while ``trust_status ==
        "trust_uninitialized"`` (a brand-new, never-bootstrapped trust
        store), the normal Unlock/Review/Approve row is replaced entirely
        by a first-run setup state -- an explanation line plus a single
        "Set up skill trust" action -- since Unlock only ever unlocks an
        EXISTING manifest and would otherwise render as a permanent dead
        end (there is nothing yet to unlock, review, or approve).
        """
        active_review = self.active_review or {}
        changed_files = active_review.get("changed_files") or []
        with Vertical(id="library-skill-trust-panel", classes="ds-panel"):
            yield Static("Trust", classes="destination-section", markup=False)
            state_classes = (
                "library-skill-trust-state-blocked" if editor_state.trust_blocked else ""
            )
            yield Static(
                skill_trust_state_line(editor_state.trust_status, editor_state.trust_changed_files),
                id="library-skill-trust-state",
                classes=state_classes,
                markup=False,
            )
            if skill_trust_needs_setup(editor_state.trust_status):
                yield Static(
                    _TRUST_SETUP_EXPLANATION_COPY,
                    id="library-skill-trust-setup-explanation",
                    markup=False,
                )
                setup_toolbar = Horizontal(classes="ds-toolbar")
                setup_toolbar.styles.height = "auto"
                with setup_toolbar:
                    yield Button(
                        "Set up skill trust",
                        id="library-skill-trust-setup",
                        classes="library-canvas-action",
                        compact=True,
                    )
                return
            yield Static(
                ", ".join(str(item) for item in changed_files),
                id="library-skill-trust-review-files",
                markup=False,
            )
            toolbar = Horizontal(classes="ds-toolbar")
            toolbar.styles.height = "auto"
            with toolbar:
                yield Button(
                    "Unlock",
                    id="library-skill-trust-unlock",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=not skill_trust_unlock_enabled(editor_state.trust_status),
                )
                yield Button(
                    "Review changes",
                    id="library-skill-trust-review",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=not skill_trust_review_enabled(
                        editor_state.trust_status, editor_state.trust_blocked
                    ),
                )
                yield Button(
                    "Approve",
                    id="library-skill-trust-approve",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=self.active_review is None,
                )
