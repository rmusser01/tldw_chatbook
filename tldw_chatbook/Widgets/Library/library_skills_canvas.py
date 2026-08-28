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
from textual.css.query import NoMatches, QueryError
from textual.widgets import Button, Input, SelectionList, Static, TextArea
from textual.widgets.selection_list import Selection

from tldw_chatbook.Library.library_shell_state import (
    library_choice_label,
    library_choice_tooltip,
    library_disabled_action_label,
    library_toggle_label,
)
from tldw_chatbook.Library.library_pager_state import LibraryPagerDisplay
from tldw_chatbook.Widgets.Library.library_choice_strip import (
    compose_library_choice_strip,
)
from tldw_chatbook.Library.library_skills_state import (
    SkillEditorState,
    SkillEditorSupportingFile,
    SkillsListState,
    coerce_skill_editor_mode,
    save_marks_needs_review,
    skill_invocation_copy,
    skill_allowed_tools_sequence,
    skill_name_shadows_builtin,
    skill_trust_requires_details,
    skill_trust_header_line,
)
from tldw_chatbook.Widgets.Library.library_canvas_sync import (
    PostRecomposeCallback,
)

_SORT_LABELS = {"name": "Name", "status": "Status"}
LIBRARY_SKILLS_FILTER_ID = "library-skills-filter"
LIBRARY_SKILLS_PAGE_PREVIOUS_ID = "library-skills-page-previous"
LIBRARY_SKILLS_PAGE_NEXT_ID = "library-skills-page-next"
LIBRARY_SKILLS_RETRY_ID = "library-skills-retry"
# task-418: the old copy ("create them in Library ▸ Skills") pointed at
# the exact list the user was already looking at; name the real paths.
_EMPTY_SKILLS_COPY = (
    "No skills yet — use Create ▸ New skill in the rail, or Import… above."
)
_EMPTY_SKILLS_FILTER_COPY = "No skills match your filter."

# Trust panel copy/gating (Task 4). Mirrors ``skills_screen.py``'s own
# ``_skill_trust_copy``/``SKILLS_TRUST_REVIEWABLE_STATUSES`` values (kept a
# separate, smaller copy here rather than importing that screen's private
# helpers -- this editor only needs the trust STATE line + two gating
# predicates, not that screen's fuller blocked-reason copy).
_TRUST_REVIEWABLE_STATUSES = frozenset(
    (
        "quarantined_modified",
        "quarantined_added",
        "quarantined_deleted",
    )
)
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
# task-2859 item 9: "v1"/"round-tripping" is internal-version talk (DESIGN.md
# plain-language rule) -- what the field actually does is unaffected by
# renaming it: the value has no effect when the skill runs, and is kept
# read-only here only so re-saving doesn't drop it from the imported file.
MODEL_HINT_COPY = (
    "Not used when running this skill — kept so saving doesn't lose the value."
)

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

# Task 4: labels for the Skills-list adaptive trust header's single inline
# action Button. Keyed by ``skill_trust_header_line``'s ``action_id``
# (``library_skills_state.py``); the "" key never renders a button (see
# ``_compose_list``).
_TRUST_HEADER_ACTION_LABELS = {
    "setup": "Set up skill trust",
    "resetup": "Set up skill trust",
    "retry": "Retry",
    "unlock": "Unlock",
    "review": "Review",
}
# Task 5: the standalone destructive Reset action -- distinct from the
# header's own single action button above (``setup``/``resetup`` already
# resets internally before it bootstraps; this is the explicit escape hatch
# for ``needs_resetup``/``locked`` postures, and the editor's
# ``quarantined_manifest_error`` trust panel, which has no header at all).
_RESET_TRUST_BUTTON_LABEL = "Reset skill trust…"
# Postures (Task 3's ``trust_posture()`` values) that surface the standalone
# Reset action in the Skills-list header, alongside its own action button.
_TRUST_POSTURES_WITH_RESET = frozenset(("needs_resetup", "locked"))


def skill_trust_needs_setup(trust_status: str) -> bool:
    """Return whether the trust panel should render its first-run setup state."""
    return trust_status == "trust_uninitialized"


def skill_trust_state_line(
    trust_status: str, changed_files: tuple[str, ...] = ()
) -> str:
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


def skill_trust_remediation_copy(trust_status: str, skill_path: str) -> str:
    """Guidance for the two blocked-but-non-reviewable trust states (task-421).

    ``quarantined_manifest_error``/``quarantined_unsupported_path`` are
    ``trust_blocked`` yet excluded from review eligibility and not
    unlockable -- without this, the panel offered no way forward at all.

    Args:
        trust_status: The skill's current trust status.
        skill_path: The skill's on-disk directory, for external repair.

    Returns:
        State-specific next-step guidance naming the on-disk location, or
        ``""`` for every state that has in-panel remediation already.
    """
    where = skill_path or "the local skills directory"
    if trust_status == "quarantined_manifest_error":
        return (
            "The local skill trust manifest can't be verified, so this "
            f"skill stays blocked. Inspect the files under {where} and the "
            "trust store next to it; if the manifest is beyond repair, "
            "remove the trust store to start over with Set up skill trust."
        )
    if trust_status == "quarantined_unsupported_path":
        return (
            "This skill contains files at unsupported paths (for example "
            f"nested folders or links). Open {where} and remove or flatten "
            "them, then reopen this skill to re-check."
        )
    return ""


# Task 5 (skills-foundation): ``quarantined_manifest_error`` now has a real
# in-panel recovery -- the Reset action below -- so the long "go inspect
# files, maybe delete the trust store by hand" guidance
# ``skill_trust_remediation_copy`` still returns (kept there verbatim, and
# still directly tested, since it is reused as-is for
# ``quarantined_unsupported_path``, which has no Reset path) is replaced by
# a short line pointing at the button that actually fixes it.
_TRUST_MANIFEST_ERROR_SHORT_COPY = (
    "The local skill trust manifest can't be verified, so this skill stays "
    "blocked. Reset skill trust to start over -- your skills themselves "
    "are not touched."
)
# Exact copy pinned by the Task 5 brief for the destructive Reset action's
# inline confirm row (mirrors ``skill_delete_confirm_copy``'s two-step
# pattern): every skill drops back to needs-review, but nothing on disk is
# deleted.
_TRUST_RESET_CONFIRM_COPY = (
    "Reset skill trust? Every skill will need re-approval. Your skills are not deleted."
)


def skill_trust_panel_remediation_copy(trust_status: str, skill_path: str) -> str:
    """Guidance text for the trust panel's remediation line (Task 5).

    Same contract as ``skill_trust_remediation_copy`` for every state
    except ``quarantined_manifest_error``, which now renders a short line
    naming the in-panel Reset action instead of that function's longer
    "inspect the files / remove the trust store by hand" guidance.

    Args:
        trust_status: The skill's current trust status.
        skill_path: The skill's on-disk directory, for external repair.

    Returns:
        ``_TRUST_MANIFEST_ERROR_SHORT_COPY`` for ``quarantined_manifest_error``,
        else ``skill_trust_remediation_copy``'s own return value.
    """
    if trust_status == "quarantined_manifest_error":
        return _TRUST_MANIFEST_ERROR_SHORT_COPY
    return skill_trust_remediation_copy(trust_status, skill_path)


def skill_script_grant_line(granted: bool) -> str:
    """Return the trust-panel line describing this skill's script permission.

    Task 7 (skills-script-execution): surfaces the standing "always allow
    this skill's scripts" grant a confirm card (Task 6) may have recorded --
    a grant the user cannot see or revoke here would be a real hole.

    Args:
        granted: Whether a standing script-execution grant is in effect.

    Returns:
        A single plain-text line for the trust panel.
    """
    if granted:
        return (
            "Scripts: this skill may run its bundled scripts without asking. "
            "Any change to its files cancels this automatically."
        )
    return "Scripts: you are asked to confirm each time this skill runs a script."


# task-414: per-file preview cap. Generous enough for any realistic
# SKILL.md, small enough that a pathological file can't wedge the panel.
_TRUST_REVIEW_PREVIEW_FILE_CHAR_CAP = 4000
# PR #750 review (Qodo): aggregate caps so a review with many changed files
# can't build/render an unbounded string during trust-panel refreshes. The
# per-file cap alone bounds one file, not the total.
_TRUST_REVIEW_PREVIEW_MAX_FILES = 20
_TRUST_REVIEW_PREVIEW_TOTAL_CHAR_CAP = 20000


def skill_trust_review_preview(active_review: Mapping[str, Any] | None) -> str:
    """Render the captured review's changed-file CONTENT for human review.

    task-414: ``capture_review`` has always returned ``current_files``
    (filename -> full text), but the panel only ever showed the filename
    list -- Approve was blind sign-off. This renders one labelled block
    per changed file so the user can actually read what they are about to
    trust. The trust store keeps fingerprints, not baseline text, so this
    is an as-is content preview rather than a before/after diff.

    Args:
        active_review: The captured review mapping (or ``None`` while no
            review is active).

    Returns:
        Labelled per-file content blocks. A changed file absent from
        ``current_files`` (text) is disambiguated using
        ``current_fingerprints``: a present binary renders
        ``(binary file — N bytes, sha256 ...)`` while a genuinely
        missing file renders ``(deleted)``. Each text file is capped at
        ``_TRUST_REVIEW_PREVIEW_FILE_CHAR_CAP`` chars; empty string while
        no review is active. Rendering is bounded to
        ``_TRUST_REVIEW_PREVIEW_MAX_FILES`` files and
        ``_TRUST_REVIEW_PREVIEW_TOTAL_CHAR_CAP`` total chars -- any
        remaining files are summarized in a trailing "N more files
        omitted" line.
    """
    if not active_review:
        return ""
    changed_files = [str(item) for item in (active_review.get("changed_files") or [])]
    if not changed_files:
        return ""
    raw_files = active_review.get("current_files")
    current_files = dict(raw_files) if isinstance(raw_files, Mapping) else {}
    fingerprints = {
        str(fp.get("relative_path")): fp
        for fp in (active_review.get("current_fingerprints") or [])
        if isinstance(fp, Mapping)
    }
    blocks: list[str] = []
    total_chars = 0
    rendered = 0
    for file_name in changed_files:
        # Stop once either aggregate budget is reached; the remaining files
        # are accounted for in the trailing omission notice below.
        if (
            rendered >= _TRUST_REVIEW_PREVIEW_MAX_FILES
            or total_chars >= _TRUST_REVIEW_PREVIEW_TOTAL_CHAR_CAP
        ):
            break
        content = current_files.get(file_name)
        if content is None:
            fp = fingerprints.get(file_name)
            if fp is not None:
                block = (
                    f"── {file_name} ──\n"
                    f"(binary file — {fp.get('byte_length', 0)} bytes, "
                    f"sha256 {str(fp.get('sha256', ''))[:12]}…)"
                )
            else:
                block = f"── {file_name} ──\n(deleted — no longer on disk)"
        else:
            text = str(content)
            if len(text) > _TRUST_REVIEW_PREVIEW_FILE_CHAR_CAP:
                text = (
                    text[:_TRUST_REVIEW_PREVIEW_FILE_CHAR_CAP]
                    + f"\n… truncated ({len(text)} chars total) — open the file on disk to read the rest."
                )
            block = f"── {file_name} ──\n{text}"
        blocks.append(block)
        total_chars += len(block)
        rendered += 1
    omitted = len(changed_files) - rendered
    if omitted > 0:
        blocks.append(
            f"… {omitted} more file{'s' if omitted != 1 else ''} omitted "
            "— open on disk to review."
        )
    return "\n\n".join(blocks)


def skill_trust_review_enabled(trust_status: str, trust_blocked: bool) -> bool:
    """Return whether the trust panel's Review changes action should be enabled."""
    return bool(trust_blocked) and trust_status in _TRUST_REVIEWABLE_STATUSES


# F-018: every disabled trust action says why (reason while disabled,
# action description while enabled) -- the workspaces handoff button's
# pattern, applied to the trust panel's live-patched buttons.
def skill_trust_unlock_tooltip(trust_status: str) -> str:
    """Return the trust panel Unlock action's tooltip for the given state.

    Args:
        trust_status: The open skill's current trust status (e.g.
            ``"trust_locked"``, ``"trusted"``).

    Returns:
        The action description while Unlock is enabled, otherwise the
        reason it is disabled (F-018: every disabled action says why).
    """
    if skill_trust_unlock_enabled(trust_status):
        return "Unlock this skill so it can run."
    return "Only a locked skill needs unlocking — this one isn't locked."


def skill_trust_review_tooltip(trust_status: str, trust_blocked: bool) -> str:
    """Return the trust panel Review changes action's tooltip.

    Args:
        trust_status: The open skill's current trust status.
        trust_blocked: Whether the open skill is currently trust-blocked.

    Returns:
        The action description while Review changes is enabled, otherwise
        the reason it is disabled for this state (nothing to review, or a
        state that cannot be reviewed in place).
    """
    if skill_trust_review_enabled(trust_status, trust_blocked):
        return "Review the pending changes to this skill."
    if not trust_blocked:
        return "Nothing to review — this skill isn't trust-blocked."
    return "This trust state can't be reviewed in place — see the guidance above."


def skill_trust_approve_tooltip(has_active_review: bool) -> str:
    """Return the trust panel Approve action's tooltip.

    Args:
        has_active_review: Whether a review is currently open (the
            Approve action's enablement condition).

    Returns:
        The action description while Approve is enabled, otherwise the
        reason it is disabled.
    """
    if has_active_review:
        return "Approve the reviewed changes."
    return "Review the changes first, then approve."


# F-018: the Discard action's tooltip pair (reason while clean/disabled,
# action while dirty) -- shared by the canvas compose and the screen's
# live ``_set_library_skill_discard_enabled`` patcher.
SKILL_DISCARD_TOOLTIP_CLEAN = "No unsaved changes to discard."
SKILL_DISCARD_TOOLTIP_DIRTY = "Leave the editor without saving the current changes."

# F-019: the editor's accelerators, advertised inline (the file-notes git
# panel's guide-line pattern) -- ctrl+s/escape are otherwise undiscoverable
# (the footer carries only the Library-wide contexts).
SKILL_EDITOR_SHORTCUT_HINTS = "ctrl+s Save · esc Back to list"


def skill_delete_confirm_copy(name: str, supporting_count: int) -> str:
    """Build the inline delete-confirmation line (task-415).

    Args:
        name: The skill's name.
        supporting_count: Number of supporting files that would be removed
            along with ``SKILL.md``.

    Returns:
        One line naming exactly what a confirmed delete removes.
    """
    scope = "the skill's directory"
    if supporting_count == 1:
        scope += " and 1 supporting file"
    elif supporting_count > 1:
        scope += f" and {supporting_count} supporting files"
    return f'Delete "{name}"? This removes {scope} and cannot be undone.'


def skill_editor_warning_lines(
    *,
    live_name: str,
    trust_status: str,
    trust_blocked: bool,
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
    """Render the user-invocable toggle Button's label (task-418 copy).

    task-14902: a kept one-press toggle -- the full yes/no option set is
    on the label with the ``✓`` marker on the active value.

    Args:
        value: Whether a user can invoke the skill directly.

    Returns:
        The toggle Button's label text.
    """
    return library_toggle_label("User can invoke", ("yes", "no"), 0 if value else 1)


def skill_disable_model_label(value: bool) -> str:
    """Render the disable-model-invocation toggle Button's label.

    task-418: display polarity is inverted -- the stored field stays
    ``disable_model_invocation``, but the label answers the question the
    user actually has ("can the agent invoke this?") instead of the
    double-negative "disable model invocation: no". task-14902: a kept
    one-press toggle with the full option set on the label.

    Args:
        value: The stored ``disable_model_invocation`` flag (``True`` means
            the agent is barred from invoking the skill).

    Returns:
        The toggle Button's label text, phrased as "Agent can invoke".
    """
    return library_toggle_label("Agent can invoke", ("yes", "no"), 1 if value else 0)


def skill_context_toggle_label(context: str) -> str:
    """Render the context-cycling Button's label.

    task-418: keeps the SKILL.md spec value (``inline``/``fork``) visible
    for round-tripping, framed in plain language.

    Args:
        context: The skill's execution context (``"inline"`` or ``"fork"``).

    Returns:
        The cycle Button's label text.
    """
    # task-14902: a kept one-press toggle -- both spec values on the label
    # with the ✓ marker on the active one. The task-418 plain-language
    # hint stays, but on the ACTIVE option only so the label survives
    # 60-column compact widths.
    hints = {"inline": "this conversation", "fork": "sub-agent"}
    options = tuple(
        f"{value} ({hints[value]})" if value == context else value
        for value in ("inline", "fork")
    )
    return library_toggle_label("Runs in", options, 0 if context == "inline" else 1)


def next_skill_context(context: str) -> str:
    """Cycle the skill editor's ``context`` field between ``inline``/``fork``."""
    return "fork" if context == "inline" else "inline"


def skill_supporting_files_text(
    supporting_files: tuple[SkillEditorSupportingFile, ...],
) -> str:
    """Render the read-only supporting-files list as plain text.

    Nested paths (e.g. ``"references/api.md"``) render as-is. Binary rows
    (``is_text is False``) get a ``(binary)`` marker -- the list is already
    read-only, so this is purely informational: binaries were never
    editable as text in the first place.
    """
    if not supporting_files:
        return "No supporting files."
    lines = []
    for file in supporting_files:
        if file.is_text:
            lines.append(f"{file.name} ({file.size} bytes)")
        else:
            lines.append(f"{file.name} — {file.size} bytes (binary)")
    return "\n".join(lines)


class LibrarySkillsTrustHeader(Vertical):
    """Recompose the asynchronous trust header without replacing list rows."""

    def __init__(
        self,
        *,
        has_skills: bool,
        blocked_count: int,
        trust_posture: str,
        confirming_reset: bool,
        **kwargs: Any,
    ) -> None:
        """Initialize the retained Skills trust header.

        Args:
            has_skills: Whether any Skills are available.
            blocked_count: Number of Skills currently blocked by trust policy.
            trust_posture: Aggregate trust-store posture.
            confirming_reset: Whether to show the destructive reset confirmation.
            **kwargs: Additional arguments forwarded to ``Vertical``.
        """
        super().__init__(**kwargs)
        self.has_skills = has_skills
        self.blocked_count = blocked_count
        self.trust_posture = trust_posture
        self.confirming_reset = confirming_reset
        self.styles.height = "auto"

    def compose(self) -> ComposeResult:
        """Render only the posture-dependent header controls.

        Returns:
            The trust summary, action, and optional reset controls.
        """
        if not self.has_skills:
            return
        header = skill_trust_header_line(self.trust_posture, self.blocked_count)
        if header is None:
            return
        copy, action_id = header
        yield Static(copy, id="library-skills-trust-header", markup=False)
        if action_id:
            button = Button(
                _TRUST_HEADER_ACTION_LABELS[action_id],
                id="library-skills-trust-action",
                classes="library-canvas-action",
                compact=True,
            )
            button.trust_action = action_id
            yield button
        if self.trust_posture in _TRUST_POSTURES_WITH_RESET:
            yield Button(
                _RESET_TRUST_BUTTON_LABEL,
                id="library-skills-trust-reset",
                classes="library-canvas-action library-media-action-danger",
                compact=True,
            )
            if self.confirming_reset:
                yield Static(
                    _TRUST_RESET_CONFIRM_COPY,
                    id="library-skills-trust-reset-confirm-copy",
                    markup=False,
                )
                toolbar = Horizontal(classes="ds-toolbar")
                toolbar.styles.height = "auto"
                with toolbar:
                    yield Button(
                        "Reset",
                        id="library-skills-trust-reset-confirm",
                        classes=("library-canvas-action library-media-action-danger"),
                        compact=True,
                    )
                    yield Button(
                        "Cancel",
                        id="library-skills-trust-reset-cancel",
                        classes="library-canvas-action",
                        compact=True,
                    )

    def sync_state(
        self,
        *,
        has_skills: bool,
        blocked_count: int,
        trust_posture: str,
        confirming_reset: bool,
    ) -> None:
        """Apply a posture result while leaving sibling Skill rows mounted.

        Args:
            has_skills: Whether any Skills are available.
            blocked_count: Number of Skills currently blocked by trust policy.
            trust_posture: Aggregate trust-store posture.
            confirming_reset: Whether to show the destructive reset confirmation.
        """
        values = (has_skills, blocked_count, trust_posture, confirming_reset)
        if values == (
            self.has_skills,
            self.blocked_count,
            self.trust_posture,
            self.confirming_reset,
        ):
            return
        (
            self.has_skills,
            self.blocked_count,
            self.trust_posture,
            self.confirming_reset,
        ) = values
        self.refresh(recompose=True)


class LibrarySkillsListCanvas(PostRecomposeCallback, VerticalScroll):
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
            (e.g. ``'Imported "executing-plans" · re-review it in the trust panel'``), or
            ``""`` when idle/not yet run.
        import_in_flight: Whether an accepted import is still running. The
            import controls are disabled while true; Library navigation is
            owned by the surrounding screen and remains available.
        trust_posture: List-view only (Task 4). The Skills trust service's
            current posture (``SkillTrustService.trust_posture()``'s
            return value -- Task 3), used to render the adaptive trust
            header above the toolbar via ``skill_trust_header_line``.
            ``""`` (the default) hides the header -- the screen (Task 5)
            supplies the real posture.
        confirming_reset: ``True`` while the destructive Reset action is
            armed and awaiting its own confirm/cancel (Task 5) -- renders
            the inline confirm row (``_compose_trust_reset_confirm_row``)
            below whichever Reset button is showing, in either the
            list-view header (posture ``needs_resetup``/``locked``) or the
            editor's ``quarantined_manifest_error`` trust panel.
    """

    def __init__(
        self,
        state: SkillsListState | None = None,
        *,
        sort_mode: str = "name",
        filter_value: str = "",
        mode: str = "list",
        trust_posture: str = "",
        confirming_reset: bool = False,
        editor_state: SkillEditorState | None = None,
        warnings: str = "",
        status: str = "",
        conflict: bool = False,
        active_review: Mapping[str, Any] | None = None,
        is_create: bool = False,
        dirty: bool = False,
        confirming_delete: bool = False,
        scroll_to_actions: bool = False,
        skill_path: str = "",
        import_open: bool = False,
        import_path: str = "",
        import_status: str = "",
        import_review_name: str = "",
        import_in_flight: bool = False,
        sort_choices_visible: bool = False,
        editor_mode: str = "basic",
        tool_catalog: tuple[str, ...] = (),
        tool_filter: str = "",
        mutation_in_flight: bool = False,
        more_actions_open: bool = False,
        trust_details_open: bool = False,
        script_access_granted: bool = False,
        show_editor_trust: bool = True,
        show_editor_files: bool = True,
        detail_notice: str = "",
        detail_retryable: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.sort_mode = sort_mode
        self.sort_choices_visible = sort_choices_visible
        self.filter_value = filter_value
        self.mode = mode
        self.trust_posture = trust_posture
        self.confirming_reset = confirming_reset
        self.editor_state = editor_state
        self.warnings = warnings
        self.status = status
        self.conflict = conflict
        self.active_review = active_review
        self.is_create = is_create
        self.dirty = dirty
        self.confirming_delete = confirming_delete
        self.scroll_to_actions = scroll_to_actions
        self.skill_path = skill_path
        self.import_open = import_open
        self.import_path = import_path
        self.import_status = import_status
        self.import_review_name = import_review_name
        self.import_in_flight = import_in_flight
        self.editor_mode = coerce_skill_editor_mode(editor_mode)
        self.tool_catalog = tuple(dict.fromkeys(tool_catalog))
        self.tool_filter = tool_filter
        self.mutation_in_flight = mutation_in_flight
        self.more_actions_open = more_actions_open
        self.trust_details_open = trust_details_open
        self.script_access_granted = script_access_granted
        self.show_editor_trust = show_editor_trust
        self.show_editor_files = show_editor_files
        self.detail_notice = detail_notice
        self.detail_retryable = detail_retryable
        self.rebuilding_tool_picker = False
        self.add_class(
            "library-skills-list-mode"
            if mode == "list"
            else "library-skills-editor-mode"
        )
        self.styles.width = "1fr"
        self.styles.min_width = 40

    def compose(self) -> ComposeResult:
        if self.mode == "loading":
            yield Static(
                self.detail_notice or "Loading skill…",
                id="library-skill-loading",
                classes="destination-purpose",
                markup=False,
            )
            if self.detail_retryable:
                yield Button(
                    "Retry",
                    id="library-skill-detail-retry",
                    classes="library-canvas-action",
                    compact=True,
                )
            return
        if self.mode == "editor":
            yield from self._compose_editor()
            return
        yield from self._compose_list()

    def sync_state(
        self,
        *,
        state: SkillsListState | None,
        sort_mode: str,
        filter_value: str,
        mode: str,
        trust_posture: str,
        confirming_reset: bool,
        editor_state: SkillEditorState | None,
        warnings: str,
        status: str,
        conflict: bool,
        active_review: Mapping[str, Any] | None,
        is_create: bool,
        dirty: bool,
        confirming_delete: bool,
        scroll_to_actions: bool,
        skill_path: str,
        import_open: bool,
        import_path: str,
        import_status: str,
        import_review_name: str,
        import_in_flight: bool,
        sort_choices_visible: bool,
        editor_mode: str = "basic",
        tool_catalog: tuple[str, ...] = (),
        tool_filter: str = "",
        mutation_in_flight: bool = False,
        more_actions_open: bool = False,
        trust_details_open: bool = False,
        script_access_granted: bool = False,
        detail_notice: str = "",
        detail_retryable: bool = False,
    ) -> None:
        """Apply a complete skills snapshot within the mounted canvas.

        Args:
            state: Skills list snapshot, or ``None`` outside list mode.
            sort_mode: Active skill sort identifier.
            filter_value: Current skill filter text.
            mode: Canvas surface to render.
            trust_posture: Trust state for the selected skill.
            confirming_reset: Whether reset confirmation is armed.
            editor_state: Skill editor snapshot.
            warnings: Current skill validation warning copy.
            status: Current skill editor status copy.
            conflict: Whether the selected skill has an edit conflict.
            active_review: Active trust-review data, if any.
            is_create: Whether the editor is creating a skill.
            dirty: Whether the skill editor has unsaved changes.
            confirming_delete: Whether delete confirmation is armed.
            scroll_to_actions: Whether to reveal the editor action row.
            skill_path: Filesystem path for the selected skill.
            import_open: Whether the skill import form is expanded.
            import_path: Current skill import path.
            import_status: Current skill import outcome copy.
            import_review_name: Skill awaiting post-import trust review.
            import_in_flight: Whether an accepted import is still running.
            sort_choices_visible: Whether the sort chooser is expanded.
        """
        header_only = bool(
            self.mode == mode == "list"
            and self.state == state
            and self.sort_mode == sort_mode
            and self.filter_value == filter_value
            and self.import_open == import_open
            and self.import_path == import_path
            and self.import_status == import_status
            and self.import_review_name == import_review_name
            and self.sort_choices_visible == sort_choices_visible
        )
        self.state = state
        self.sort_mode = sort_mode
        self.filter_value = filter_value
        self.mode = mode
        self.set_class(mode == "list", "library-skills-list-mode")
        self.set_class(mode != "list", "library-skills-editor-mode")
        self.trust_posture = trust_posture
        self.confirming_reset = confirming_reset
        self.editor_state = editor_state
        self.warnings = warnings
        self.status = status
        self.conflict = conflict
        self.active_review = active_review
        self.is_create = is_create
        self.dirty = dirty
        self.confirming_delete = confirming_delete
        self.scroll_to_actions = scroll_to_actions
        self.skill_path = skill_path
        self.import_open = import_open
        self.import_path = import_path
        self.import_status = import_status
        self.import_review_name = import_review_name
        self.import_in_flight = import_in_flight
        self.sort_choices_visible = sort_choices_visible
        self.editor_mode = coerce_skill_editor_mode(editor_mode)
        self.tool_catalog = tuple(dict.fromkeys(tool_catalog))
        self.tool_filter = tool_filter
        self.mutation_in_flight = mutation_in_flight
        self.more_actions_open = more_actions_open
        self.trust_details_open = trust_details_open
        self.script_access_granted = script_access_granted
        self.detail_notice = detail_notice
        self.detail_retryable = detail_retryable
        if header_only:
            rows = state.rows if state is not None else ()
            title_count = (
                state.pager.title_count
                if state is not None and state.pager is not None
                else len(rows)
            )
            try:
                self.query_one(
                    "#library-skills-trust-region", LibrarySkillsTrustHeader
                ).sync_state(
                    has_skills=bool(state and state.source_summary_fresh)
                    and (
                        bool(rows)
                        or bool(title_count)
                        or bool(state and state.blocked_total)
                    ),
                    blocked_count=(
                        state.blocked_total
                        if state is not None and state.pager is not None
                        else sum(1 for row in rows if row.blocked)
                    ),
                    trust_posture=trust_posture,
                    confirming_reset=confirming_reset,
                )
                return
            except (NoMatches, QueryError):
                pass
        self.refresh(recompose=True)
        self._schedule_scroll_to_actions()

    def on_mount(self) -> None:
        """task-417: a recompose lands a fresh canvas scrolled to the top.

        When the screen armed ``scroll_to_actions`` (the create-save
        snapshot recompose), bring the action row back into view so the
        user still sees the Save button and its status line they just
        acted on.
        """
        self._schedule_scroll_to_actions()

    async def set_editor_mode(self, mode: str) -> None:
        """Switch the mounted Skill presentations without rebuilding the draft."""
        requested = coerce_skill_editor_mode(mode)
        focused = self.app.focused
        basic = self.query_one("#library-skill-basic-fields")
        advanced = self.query_one("#library-skill-advanced-fields")
        hiding_focused = bool(
            focused is not None
            and (
                (requested == "advanced" and basic in focused.ancestors_with_self)
                or (requested == "basic" and advanced in focused.ancestors_with_self)
            )
        )
        if hiding_focused:
            self.screen.set_focus(None)
        self.editor_mode = requested
        basic.display = requested == "basic"
        advanced.display = requested == "advanced"
        state = self.editor_state
        self.query_one("#library-skill-argument-fields").display = bool(
            requested == "advanced" or (state is not None and state.user_invocable)
        )
        mode_button = self.query_one("#library-skill-editor-mode", Button)
        mode_button.label = "Show basic" if requested == "advanced" else "Show advanced"
        if hiding_focused:
            self.call_after_refresh(
                self._restore_editor_mode_focus,
                focused,
                requested,
            )

    def _restore_editor_mode_focus(self, prior_focus, editor_mode: str) -> None:
        """Focus the mode control unless a newer visible user target won."""
        live_focus = self.app.focused
        hidden_region = self.query_one(
            "#library-skill-basic-fields"
            if editor_mode == "advanced"
            else "#library-skill-advanced-fields"
        )
        live_focus_is_hidden = bool(
            live_focus is not None and hidden_region in live_focus.ancestors_with_self
        )
        if (
            live_focus is not None
            and live_focus is not prior_focus
            and live_focus.id != "library-skill-editor-mode"
            and not live_focus_is_hidden
        ):
            return
        target = self.query_one("#library-skill-editor-mode", Button)
        self.screen.set_focus(target, scroll_visible=False)

    def sync_lifecycle_actions(
        self,
        *,
        dirty: bool | None = None,
        conflict: bool | None = None,
        confirming_delete: bool | None = None,
        mutation_in_flight: bool | None = None,
        more_actions_open: bool | None = None,
        is_create: bool | None = None,
    ) -> None:
        """Patch lifecycle-valid actions without replacing editor fields."""
        if dirty is not None:
            self.dirty = bool(dirty)
        if conflict is not None:
            self.conflict = bool(conflict)
        if confirming_delete is not None:
            self.confirming_delete = bool(confirming_delete)
        if mutation_in_flight is not None:
            self.mutation_in_flight = bool(mutation_in_flight)
        if more_actions_open is not None:
            self.more_actions_open = bool(more_actions_open)
        if is_create is not None:
            self.is_create = bool(is_create)

        busy = self.mutation_in_flight
        conflict_active = self.conflict and not busy
        delete_armed = (
            self.confirming_delete
            and not self.conflict
            and not self.is_create
            and not busy
        )
        create = self.is_create and not busy and not self.conflict
        dirty_active = (
            self.dirty
            and not self.is_create
            and not busy
            and not self.conflict
            and not self.confirming_delete
        )
        clean = (
            not self.dirty
            and not self.is_create
            and not busy
            and not self.conflict
            and not self.confirming_delete
        )
        visibility = {
            "#library-skill-mutation-progress": busy,
            "#library-skill-mutation-reason": busy,
            "#library-skill-conflict-reload": conflict_active,
            "#library-skill-delete-confirm": delete_armed,
            "#library-skill-delete-cancel": delete_armed,
            "#library-skill-save": create or dirty_active,
            "#library-skill-cancel": create,
            "#library-skill-discard": dirty_active,
            "#library-skill-back": clean,
            "#library-skill-more-actions": clean,
            "#library-skill-delete": clean and self.more_actions_open,
            "#library-skill-delete-confirm-copy": delete_armed,
            "#library-skill-conflict-copy": self.conflict,
            "#library-skill-save-status": not self.conflict,
        }
        for selector, visible in visibility.items():
            self.query_one(selector).display = visible
        self.query_one("#library-skill-save", Button).label = (
            "Save skill" if self.is_create else "Save changes"
        )

    def _tool_picker_selections(self, filter_value: str = "") -> list[Selection]:
        """Build unique chooser rows while retaining raw content separately."""
        state = self.editor_state
        if state is None:
            return []
        query = filter_value.strip().casefold()
        captured = skill_allowed_tools_sequence(state.allowed_tools_csv)
        captured_set = set(captured)
        known_set = set(self.tool_catalog)
        rows = [
            Selection(name, name, name in captured_set)
            for name in self.tool_catalog
            if not query or query in name.casefold()
        ]
        rows.extend(
            Selection(f"{name} (unavailable)", name, True, disabled=True)
            for name in dict.fromkeys(captured)
            if name not in known_set and (not query or query in name.casefold())
        )
        return rows

    def set_tool_filter(self, value: str) -> None:
        """Filter only picker rows; never rewrite the captured Skill allowlist."""
        self.tool_filter = value
        picker = self.query_one("#library-skill-tool-picker", SelectionList)
        self.rebuilding_tool_picker = True
        picker.clear_options().add_options(self._tool_picker_selections(value))
        self.rebuilding_tool_picker = False

    def _schedule_scroll_to_actions(self) -> None:
        """Preserve the post-save scroll receipt across canvas-only syncs."""
        if not (self.scroll_to_actions and self.mode == "editor"):
            return

        def _scroll_to_action_row() -> None:
            # During the delete confirmation the Save button is replaced by
            # the Delete/Cancel row, so anchor on the confirm copy that is
            # actually present; otherwise the Save row (review finding).
            for selector in (
                "#library-skill-delete-confirm-copy",
                "#library-skill-save",
            ):
                # Narrow to the query miss (PR #750 review): a missing anchor
                # is expected (the Save button is absent in confirm mode), but
                # any other error should surface rather than be swallowed.
                try:
                    target = self.query_one(selector)
                except (NoMatches, QueryError):
                    continue
                if not target.display:
                    continue
                target.scroll_visible(animate=False)
                return

        self.call_after_refresh(_scroll_to_action_row)

    def _compose_list(self) -> ComposeResult:
        state = self.state
        if state is None:
            return
        title_count = (
            state.pager.title_count if state.pager is not None else state.count
        )
        yield Static(
            "Skills" if title_count is None else f"Skills ({title_count})",
            id="library-skills-header",
            classes="destination-section",
            markup=False,
        )
        # The posture read may settle after rows become interactive. Keep it
        # in its own retained region so that update cannot invalidate a row's
        # already-posted Button.Pressed event.
        yield LibrarySkillsTrustHeader(
            has_skills=state.source_summary_fresh
            and (bool(state.rows) or bool(title_count) or state.blocked_total > 0),
            blocked_count=(
                state.blocked_total
                if state.pager is not None
                else sum(1 for row in state.rows if row.blocked)
            ),
            trust_posture=self.trust_posture,
            confirming_reset=self.confirming_reset,
            id="library-skills-trust-region",
        )
        yield Input(
            placeholder="Filter skills… (Enter)",
            id=LIBRARY_SKILLS_FILTER_ID,
            value=self.filter_value,
        )
        # One horizontal ds-toolbar row for sort/Import -- mirrors
        # library_prompts_canvas.py's toolbar exactly (same render-safe
        # shape: every child is a fixed-width compact Button).
        toolbar = Horizontal(classes="ds-toolbar")
        toolbar.styles.height = "auto"
        # task-14902: the sort choice strip replaces this toolbar row while
        # open (the Notes Sort precedent).
        toolbar.display = not self.sort_choices_visible
        with toolbar:
            yield Button(
                library_choice_label("sort", _SORT_LABELS.get(self.sort_mode, "Name")),
                id="library-skills-sort",
                classes="library-canvas-action",
                compact=True,
                tooltip=library_choice_tooltip(
                    "the sort order", tuple(_SORT_LABELS.values())
                ),
            )
            yield Button(
                "Import…",
                id="library-skills-import",
                classes="library-canvas-action",
                compact=True,
            )
        if self.sort_choices_visible:
            yield from compose_library_choice_strip(
                strip_id="library-skills-sort-choices",
                choice_class="library-skills-sort-choice",
                options=tuple(
                    (f"library-skills-sort-{mode}", mode, label)
                    for mode, label in _SORT_LABELS.items()
                ),
                active_value=self.sort_mode,
            )
        if self.import_open:
            yield from self._compose_import_row()
        if not state.rows:
            yield Static(
                _EMPTY_SKILLS_FILTER_COPY if self.filter_value else _EMPTY_SKILLS_COPY,
                id="library-skills-empty",
                markup=False,
            )
        else:
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
                    if row.selected:
                        classes = f"{classes} is-selected"
                    button = Button(
                        f"{'› ' if row.selected else ''}{row.trust_glyph} {name}",
                        id=f"library-skill-row-{row.name}",
                        classes=classes,
                        compact=True,
                        disabled=state.actions_disabled,
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
        if state.pager is not None:
            yield from self._compose_pager(state.pager)

    def _compose_pager(self, pager: LibraryPagerDisplay) -> ComposeResult:
        """Render the controller-derived Skills pager without recalculation."""
        reasons = tuple(
            dict.fromkeys(
                reason
                for disabled, reason in (
                    (pager.previous_disabled, pager.previous_reason),
                    (pager.next_disabled, pager.next_reason),
                )
                if disabled and reason
            )
        )
        with Vertical(id="library-skills-pager", classes="library-source-pager"):
            yield Static(
                pager.range_copy,
                id="library-skills-range",
                classes="library-source-pager-status",
                markup=False,
            )
            yield Static(
                pager.page_copy,
                id="library-skills-page",
                classes="library-source-pager-status",
                markup=False,
            )
            status_copy = " · ".join(
                copy for copy in (pager.status_copy, *reasons) if copy
            )
            if status_copy:
                yield Static(
                    status_copy,
                    id="library-skills-pager-status",
                    classes="library-source-pager-status",
                    markup=False,
                )
            with Horizontal(classes="library-source-pager-controls"):
                yield Button(
                    library_disabled_action_label("Previous", pager.previous_disabled),
                    id=LIBRARY_SKILLS_PAGE_PREVIOUS_ID,
                    classes="library-canvas-action",
                    compact=True,
                    disabled=pager.previous_disabled,
                    tooltip=pager.previous_reason or None,
                )
                if pager.retry_visible:
                    yield Button(
                        "Retry",
                        id=LIBRARY_SKILLS_RETRY_ID,
                        classes="library-canvas-action",
                        compact=True,
                    )
                yield Button(
                    library_disabled_action_label("Next", pager.next_disabled),
                    id=LIBRARY_SKILLS_PAGE_NEXT_ID,
                    classes="library-canvas-action",
                    compact=True,
                    disabled=pager.next_disabled,
                    tooltip=pager.next_reason or None,
                )

    def _compose_trust_reset_confirm_row(self) -> ComposeResult:
        """Inline confirm row for the destructive Reset action (Task 5).

        Shared by both places the Reset button can render -- the
        list-view header (``needs_resetup``/``locked`` postures) and the
        editor's ``quarantined_manifest_error`` trust panel -- so the two
        never drift. Same render-safe shape as ``_compose_editor``'s own
        delete-confirm row: the copy is a full-width ``Static`` ABOVE the
        toolbar, never mixed into the same ``Horizontal`` as the Buttons.
        """
        yield Static(
            _TRUST_RESET_CONFIRM_COPY,
            id="library-skills-trust-reset-confirm-copy",
            markup=False,
        )
        toolbar = Horizontal(classes="ds-toolbar")
        toolbar.styles.height = "auto"
        with toolbar:
            yield Button(
                "Reset",
                id="library-skills-trust-reset-confirm",
                classes="library-canvas-action library-media-action-danger",
                compact=True,
            )
            yield Button(
                "Cancel",
                id="library-skills-trust-reset-cancel",
                classes="library-canvas-action",
                compact=True,
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
            placeholder="SKILL.md file or skill folder path… or GitHub/zip URL",
            id="library-skills-import-path",
            value=self.import_path,
            disabled=self.import_in_flight,
        )
        toolbar = Horizontal(classes="ds-toolbar")
        toolbar.styles.height = "auto"
        with toolbar:
            # Browse… picks a FILE via the shared FileOpen dialog;
            # task-422 adds the folder variant beside it (SelectDirectory)
            # since a real skill package is a directory named after the
            # skill -- the common shape no longer has to be typed by hand.
            yield Button(
                "Browse…",
                id="library-skills-import-browse",
                classes="library-canvas-action",
                compact=True,
                disabled=self.import_in_flight,
            )
            yield Button(
                "Browse folder…",
                id="library-skills-import-browse-folder",
                classes="library-canvas-action",
                compact=True,
                disabled=self.import_in_flight,
            )
            yield Button(
                "Import",
                id="library-skills-import-run",
                classes="library-canvas-action",
                compact=True,
                disabled=self.import_in_flight,
            )
            yield Button(
                "Cancel",
                id="library-skills-import-cancel",
                classes="library-canvas-action",
                compact=True,
                disabled=self.import_in_flight,
            )
        yield Static(
            self.import_status,
            id="library-skills-import-status",
            markup=False,
        )
        if self.import_review_name:
            # task-422: the success copy says "re-review it in the trust
            # panel" -- this is the direct path there.
            yield Button(
                f'Review "{self.import_review_name}"…',
                id="library-skills-import-review",
                classes="library-canvas-action",
                compact=True,
                disabled=self.import_in_flight,
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
        advanced = self.editor_mode == "advanced"
        yield Button(
            "Show basic" if advanced else "Show advanced",
            id="library-skill-editor-mode",
            classes="library-canvas-action",
            compact=True,
        )
        if not self.conflict and not self.confirming_delete:
            # F-019: the editor's ctrl+s/escape accelerators (task-424),
            # advertised inline -- the file-notes git panel's guide-line
            # pattern. Hidden during the conflict banner and the delete
            # confirmation, where ctrl+s is gated off and the hint would
            # be a lie.
            yield Static(
                SKILL_EDITOR_SHORTCUT_HINTS,
                id="library-skill-editor-hints",
                classes="library-prompt-field-hint",
                markup=False,
            )
        yield Static("Name", classes="library-prompt-field-label", markup=False)
        yield Input(
            value=editor_state.name,
            id="library-skill-name",
            disabled=not self.is_create,
            # task-424: the format rule is known upfront -- say it before
            # the save-time error can fire.
            placeholder=(
                "lowercase letters, numbers, hyphens (e.g. code-review)"
                if self.is_create
                else ""
            ),
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
        if editor_state.description_derived:
            # task-419: the record's description was auto-derived (no
            # frontmatter description on disk); say so instead of quietly
            # pre-filling the field with text the user never wrote.
            yield Static(
                "No description set — lists show the skill's first body line "
                "automatically. Type here to set your own.",
                id="library-skill-description-hint",
                classes="library-skill-field-hint",
                markup=False,
            )
        basic_fields = Vertical(id="library-skill-basic-fields")
        basic_fields.styles.display = "none" if advanced else "block"
        with basic_fields:
            yield Static(
                skill_invocation_copy(
                    editor_state.user_invocable,
                    editor_state.disable_model_invocation,
                ),
                id="library-skill-invocation-copy",
                markup=False,
            )
        argument_fields = Vertical(id="library-skill-argument-fields")
        argument_fields.styles.display = (
            "block" if advanced or editor_state.user_invocable else "none"
        )
        with argument_fields:
            yield Static(
                "Argument hint", classes="library-prompt-field-label", markup=False
            )
            yield Input(
                value=editor_state.argument_hint or "",
                id="library-skill-argument-hint",
            )
        # task-14902: kept one-press toggles -- the labels now carry the
        # full option set with the ✓ active marker, so the old
        # option-enumerating tooltips (task-4023 AC#5's stopgap for a
        # hidden option space) are redundant; the tooltips now say what a
        # press does instead.
        yield Button(
            skill_user_invocable_label(editor_state.user_invocable),
            id="library-skill-user-invocable",
            classes="library-canvas-action",
            compact=True,
            tooltip="Press to switch user invocation.",
        )
        yield Button(
            skill_disable_model_label(editor_state.disable_model_invocation),
            id="library-skill-disable-model",
            classes="library-canvas-action",
            compact=True,
            tooltip="Press to switch agent invocation.",
        )
        yield Static("Body", classes="library-prompt-field-label", markup=False)
        yield TextArea(editor_state.body, id="library-skill-body")
        advanced_fields = Vertical(id="library-skill-advanced-fields")
        advanced_fields.styles.display = "block" if advanced else "none"
        with advanced_fields:
            yield Static(
                "Allowed tools", classes="library-prompt-field-label", markup=False
            )
            yield Static(
                "Restricts which currently available tools this Skill may use; "
                "it never grants permission.",
                id="library-skill-tool-help",
                classes="library-prompt-field-hint",
                markup=False,
            )
            yield Input(
                value=self.tool_filter,
                placeholder="Filter tools",
                id="library-skill-tool-filter",
            )
            yield SelectionList(
                *self._tool_picker_selections(self.tool_filter),
                id="library-skill-tool-picker",
            )
            yield Static(
                editor_state.allowed_tools_csv,
                id="library-skill-tool-captured",
                classes="library-prompt-field-hint",
                markup=False,
            )
            yield Button(
                skill_context_toggle_label(editor_state.context),
                id="library-skill-context",
                classes="library-canvas-action",
                compact=True,
                tooltip=(
                    "Press to switch the execution context: inline runs in "
                    "this conversation, fork runs in a sub-agent."
                ),
            )
            if editor_state.model:
                yield Static(
                    "Imported model",
                    classes="library-prompt-field-label",
                    markup=False,
                )
                yield Input(
                    value=editor_state.model,
                    id="library-skill-model",
                    disabled=True,
                )
                yield Static(
                    MODEL_HINT_COPY,
                    id="library-skill-model-hint",
                    classes="library-prompt-field-hint",
                    markup=False,
                )
            if self.show_editor_files:
                yield Static(
                    "Supporting files",
                    classes="library-prompt-field-label",
                    markup=False,
                )
                yield Static(
                    skill_supporting_files_text(editor_state.supporting_files),
                    id="library-skill-supporting",
                    markup=False,
                )
            yield Static(self.warnings, id="library-skill-warnings", markup=False)
        conflict_copy = Static(
            "This skill changed elsewhere — Reload discards your edit and refetches it.",
            id="library-skill-conflict-copy",
            classes="destination-purpose",
            markup=False,
        )
        conflict_copy.display = self.conflict
        yield conflict_copy
        save_status = Static(self.status, id="library-skill-save-status", markup=False)
        save_status.display = not self.conflict
        yield save_status
        # task-416: no trust panel in create mode -- a never-saved skill
        # has no on-disk files, so the panel could only show a false state
        # ("Trust: trusted") with dead buttons. The post-create snapshot
        # refresh recomposes with is_create=False, which renders the real
        # panel for the just-saved skill.
        if not self.is_create and self.show_editor_trust:
            yield from self._compose_trust_panel(editor_state)
        # task-415: inline two-step delete, mirroring the notes/media
        # confirming-delete pattern. The confirm copy is a full-width
        # Static ABOVE the toolbar (mixing a Static into the toolbar's
        # Buttons is the known non-rendering failure mode the media
        # viewer documents).
        confirming_delete = (
            self.confirming_delete and not self.conflict and not self.is_create
        )
        delete_copy = Static(
            skill_delete_confirm_copy(
                editor_state.name, len(editor_state.supporting_files)
            ),
            id="library-skill-delete-confirm-copy",
            markup=False,
        )
        delete_copy.display = confirming_delete and not self.mutation_in_flight
        yield delete_copy
        toolbar = Horizontal(id="library-skill-lifecycle-actions", classes="ds-toolbar")
        toolbar.styles.height = "auto"
        with toolbar:
            busy = self.mutation_in_flight
            conflict = self.conflict and not busy
            delete_armed = confirming_delete and not busy
            create = self.is_create and not busy and not self.conflict
            dirty = (
                self.dirty
                and not self.is_create
                and not busy
                and not self.conflict
                and not confirming_delete
            )
            clean = (
                not self.dirty
                and not self.is_create
                and not busy
                and not self.conflict
                and not confirming_delete
            )
            progress = Static(
                "Saving changes…",
                id="library-skill-mutation-progress",
                markup=False,
            )
            progress.display = busy
            yield progress
            reason = Static(
                "Editor actions are unavailable until saving finishes.",
                id="library-skill-mutation-reason",
                markup=False,
            )
            reason.display = busy
            yield reason
            reload_button = Button(
                "Reload",
                id="library-skill-conflict-reload",
                classes="library-canvas-action",
                compact=True,
            )
            reload_button.display = conflict
            yield reload_button
            confirm_delete = Button(
                "Delete",
                id="library-skill-delete-confirm",
                classes="library-canvas-action library-media-action-danger",
                compact=True,
            )
            confirm_delete.display = delete_armed
            yield confirm_delete
            confirm_cancel = Button(
                "Cancel",
                id="library-skill-delete-cancel",
                classes="library-canvas-action",
                compact=True,
            )
            confirm_cancel.display = delete_armed
            yield confirm_cancel
            save = Button(
                "Save skill" if self.is_create else "Save changes",
                id="library-skill-save",
                classes="library-canvas-action",
                compact=True,
            )
            save.display = create or dirty
            yield save
            cancel = Button(
                "Cancel",
                id="library-skill-cancel",
                classes="library-canvas-action",
                compact=True,
            )
            cancel.display = create
            yield cancel
            discard = Button(
                "Discard changes",
                id="library-skill-discard",
                classes="library-canvas-action",
                compact=True,
                tooltip=SKILL_DISCARD_TOOLTIP_DIRTY,
            )
            discard.display = dirty
            yield discard
            back = Button(
                "Back to list",
                id="library-skill-back",
                classes="library-canvas-action",
                compact=True,
            )
            back.display = clean
            yield back
            more = Button(
                "More actions",
                id="library-skill-more-actions",
                classes="library-canvas-action",
                compact=True,
            )
            more.display = clean
            yield more
            secondary_delete = Button(
                "Delete",
                id="library-skill-delete",
                classes="library-canvas-action library-media-action-danger",
                compact=True,
            )
            secondary_delete.display = clean and self.more_actions_open
            yield secondary_delete

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
                "library-skill-trust-state-blocked"
                if editor_state.trust_blocked
                else ""
            )
            yield Static(
                skill_trust_state_line(
                    editor_state.trust_status, editor_state.trust_changed_files
                ),
                id="library-skill-trust-state",
                classes=state_classes,
                markup=False,
            )
            show_details = (
                self.trust_details_open
                or self.script_access_granted
                or skill_trust_requires_details(
                    editor_state.trust_status,
                    editor_state.trust_blocked,
                    editor_state.trust_changed_files,
                )
            )
            if not show_details:
                yield Button(
                    "View details",
                    id="library-skill-trust-view-details",
                    classes="library-canvas-action",
                    compact=True,
                )
                return
            # task-421: always present (empty for states with in-panel
            # remediation) so the screen's no-recompose panel patch can
            # keep it current, same contract as the review-files line.
            # Task 5: ``skill_trust_panel_remediation_copy`` (not the plain
            # ``skill_trust_remediation_copy``) so ``quarantined_manifest_error``
            # gets the short line pointing at the Reset button below,
            # instead of the old "go inspect files by hand" guidance.
            yield Static(
                skill_trust_panel_remediation_copy(
                    editor_state.trust_status, self.skill_path
                ),
                id="library-skill-trust-remediation",
                markup=False,
            )
            # Task 7 (skills-script-execution): the standing script-execution
            # grant a confirm card (Task 6) may have recorded for this skill,
            # plus a way to revoke it. Compose-time default is "not granted"
            # (the screen only knows the real state after an off-thread
            # fingerprint check -- see ``local_skill_trust_service
            # .script_execution_granted``, which re-scans the skill's
            # directory and must never run on this synchronous compose
            # path); the screen patches both widgets in place moments later
            # via ``_render_library_skill_trust_panel``, same contract as
            # the always-present review-files/review-content lines above.
            yield Static(
                skill_script_grant_line(self.script_access_granted),
                id="library-skill-script-grant",
                markup=False,
            )
            yield Button(
                "Revoke script access",
                id="library-skill-script-grant-revoke",
                classes="library-canvas-action",
                compact=True,
                disabled=not self.script_access_granted,
            )
            if editor_state.trust_status == "quarantined_manifest_error":
                # Task 5: the manifest itself can't be verified, so nothing
                # in the normal Unlock/Review/Approve row below can ever
                # apply to this skill -- Reset is the only real way
                # forward, rendered here since this state has no list
                # header to surface it from.
                reset_toolbar = Horizontal(classes="ds-toolbar")
                reset_toolbar.styles.height = "auto"
                with reset_toolbar:
                    yield Button(
                        _RESET_TRUST_BUTTON_LABEL,
                        id="library-skills-trust-reset",
                        classes="library-canvas-action library-media-action-danger",
                        compact=True,
                    )
                if self.confirming_reset:
                    yield from self._compose_trust_reset_confirm_row()
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
            # task-414: the content actually under review. Always present
            # (empty when no review is active) so the screen's no-recompose
            # trust-panel patch can fill it in place, same contract as the
            # changed-files line above. Without this, Approve was blind
            # sign-off on a filename list.
            yield Static(
                skill_trust_review_preview(self.active_review),
                id="library-skill-trust-review-content",
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
                    # F-018: reason/action tooltip (kept current in place by
                    # the screen's no-recompose trust patcher).
                    tooltip=skill_trust_unlock_tooltip(editor_state.trust_status),
                )
                yield Button(
                    "Review changes",
                    id="library-skill-trust-review",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=not skill_trust_review_enabled(
                        editor_state.trust_status, editor_state.trust_blocked
                    ),
                    tooltip=skill_trust_review_tooltip(
                        editor_state.trust_status, editor_state.trust_blocked
                    ),
                )
                yield Button(
                    "Approve",
                    id="library-skill-trust-approve",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=self.active_review is None,
                    tooltip=skill_trust_approve_tooltip(self.active_review is not None),
                )
