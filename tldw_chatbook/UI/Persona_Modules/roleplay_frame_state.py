"""Pure state for the Roleplay destination frame (spec sections 1.3 and 5.11).

The Roleplay screen gathers inputs and pushes the views computed here into its
widgets. Everything in this module is a plain value or a pure function: no
Textual import, no widget query, no I/O, so every rule is unit-tested without
mounting a screen. Later frame slices extend it (rail state, the work-session
reducer, the keyboard projection); slice B1 brings the one-row header.

Untrusted text (spec R33). Item names and server labels arrive raw. They are
measured and cut as PLAIN text with resolved glyphs, then escaped only on the
way into a markup-on surface: the shared header's status chip gets
``escape_markup``. The fitted item label and the chips render literal
``Content`` and are never escaped (a backslash would paint).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rich.cells import cell_len

from tldw_chatbook.UI.Workbench.workbench_state import WorkbenchHeaderState
from tldw_chatbook.Utils.input_validation import escape_markup
from tldw_chatbook.Widgets.glyph_fallback import resolve_glyph
from tldw_chatbook.Widgets.Persona_Widgets.personas_state import MODE_LABELS

#: The destination's one public name (F-034: it matches the nav label).
HEADER_TITLE = "Roleplay"

#: Cells the inline header spends around its parts. They mirror
#: ``css/features/_roleplay.tcss``; ``Tests/UI/test_roleplay_header.py`` pins
#: them against the mounted widgets' computed styles, so a CSS edit that
#: forgets these fails there instead of clipping the status chip.
HEADER_PADDING_CELLS = 2  # #personas-header: padding 0 1
KIND_GAP_CELLS = 2  # the kind subtitle: margin-left $ds-space-2
ITEM_GAP_CELLS = 1  # #personas-header-item: margin-left $ds-space-1
CHIP_CHROME_CELLS = 3  # each chip: margin-left 1 + padding 0 1
STATUS_CHROME_CELLS = 3  # the status chip: margin-left 1 + padding 0 1

#: Below this many columns the unsaved chip takes its short form (spec 1.3).
UNSAVED_SHORT_BELOW_COLUMNS = 100
UNSAVED_CHIP = "Unsaved changes"
UNSAVED_CHIP_SHORT = "Unsaved"
#: A server label longer than this is cut (with the resolved ellipsis).
SERVER_LABEL_MAX_CELLS = 32
#: Degrade steps when the chips and the status do not fit (spec 1.3):
#: 0 full forms; 1 the status drops the server label; 2 the blocked chip
#: drops "Settings"; 3 the status reads just "Server".
LAST_DEGRADE_STEP = 3

#: Roleplay's destination classes for the shared adaptive pane shell, in
#: ``AdaptivePaneClasses`` field order (shell, nav, items, work, grip). Each
#: carries a Roleplay split prefix, so the shell rules in
#: ``css/features/_roleplay.tcss`` stay lazy (spec 2.12 item 2). Plain strings,
#: so this module stays free of Textual; frame slice B7 mounts the shell with
#: ``AdaptivePaneClasses(*ROLEPLAY_PANE_CLASS_NAMES)``.
ROLEPLAY_PANE_CLASS_NAMES: tuple[str, str, str, str, str] = (
    "roleplay-shell",
    "roleplay-nav",
    "roleplay-items",
    "roleplay-shell-work",
    "roleplay-shell-grip",
)

_ELLIPSIS = "…"
_SEPARATOR = "·"
_GO = "›"

#: One-line "what this kind is" copy: the purpose line under the header and
#: the mode chips' tooltips, until frame slice B6's rail glosses replace both
#: (spec G8). Moved verbatim from ``personas_screen.py`` in B1.
MODE_DESCRIPTORS: dict[str, str] = {
    "characters": "Characters — who the AI plays.",
    # F-034: the descriptor teaches the genre convention (characters = who
    # the AI plays, personas = who YOU play) instead of the vague "assistant
    # profiles" - without reviving the retired human-identity framing.
    "personas": "Personas — who you play in the chat.",
    "prompts": "Prompts — moving to the Library.",
    "dictionaries": "Dictionaries — text find/replace rules.",
    "lore": "Lore — world facts injected on keywords.",
}


class DraftSnapshotLike(Protocol):
    """The one property of ``RoleplayDraftSnapshot`` the predicate reads.

    A Protocol, not an import: the snapshot's module defines Roleplay's
    navigation dialogs, which the screen imports lazily, and this module must
    not drag them onto the route's pre-import census.
    """

    @property
    def is_clean(self) -> bool:
        """True when no domain is dirty and no save is in flight."""
        ...


def roleplay_has_unsaved_work(snapshot: DraftSnapshotLike) -> bool:
    """The one ADR-046 unsaved predicate (spec R24, section 3.12).

    True while any Roleplay draft domain is dirty OR a save is still in
    flight: ``is_clean`` already counts in-flight saves, so a chip driven by
    this stays on until the save completes, never just ``has_unsaved_changes``.

    Args:
        snapshot: ``PersonasScreen._aggregate_roleplay_draft_snapshot()``.

    Returns:
        Whether the destination holds work that is not yet safely saved.
    """
    return not snapshot.is_clean


def ellipsize_cells(text: str, budget: int) -> str:
    """Cut ``text`` to ``budget`` terminal cells, ending in the resolved ellipsis.

    Measures cells, not characters, so wide (CJK, emoji) and zero-width
    characters fit by what they paint. The same rule as the rail rows'
    fitter in ``Widgets/adaptive_pane_shell.py``, kept here so this module
    stays free of Textual.

    Args:
        text: Plain (unescaped) text.
        budget: Cells available.

    Returns:
        ``text`` when it fits; ``""`` when not even one character fits before
        the ellipsis; otherwise the longest fitting head plus the ellipsis.
    """
    if cell_len(text) <= budget:
        return text
    ellipsis = resolve_glyph(_ELLIPSIS)
    room = budget - cell_len(ellipsis)
    head = ""
    for character in text:
        if cell_len(head + character) > room:
            break
        head += character
    head = head.rstrip()
    return f"{head}{ellipsis}" if head else ""


def _one_line(text: str) -> str:
    """``text`` with each whitespace run (newline, tab, spaces) as one space.

    The header is one row: a ``Static`` paints only a text's first line and
    ``cell_len`` measures a newline as 0 cells, so a server label
    ``"home\\nevil"`` would paint ``Server: home`` without its ``read-only``.
    """
    return " ".join(text.split())


def runtime_server_label(app_instance: object) -> str:
    """The active server's display label, read the way the Library reads it.

    Args:
        app_instance: The app (or a test double); only attributes are read.

    Returns:
        ``runtime_policy.state.last_known_server_label``, else its
        ``active_server_id``, else ``""``, on one line (each whitespace
        run as one space). Non-string values count as absent.
    """
    runtime_policy = getattr(app_instance, "runtime_policy", None)
    state = getattr(runtime_policy, "state", None)
    for name in ("last_known_server_label", "active_server_id"):
        value = getattr(state, name, None)
        if isinstance(value, str) and value.strip():
            return _one_line(value)
    return ""


@dataclass(frozen=True)
class RoleplayHeaderInputs:
    """Everything the one-row header shows, gathered by the screen.

    Attributes:
        mode: The active kind (``"characters"``, ``"personas"``,
            ``"dictionaries"`` or ``"lore"``).
        edit_mode: ``"view"``, ``"create"`` or ``"edit"``.
        item_name: The selected item's name, raw (untrusted), or ``""``.
        unsaved: ``roleplay_has_unsaved_work`` of the aggregate snapshot.
        provider_blocked: The destination-wide block: no ready chat provider
            for character chats (``console_handoff_readiness()``, which
            reads no selection).
        runtime_source: ``"local"`` or ``"server"``.
        server_label: ``runtime_server_label`` (raw), or ``""``.
    """

    mode: str
    edit_mode: str = "view"
    item_name: str = ""
    unsaved: bool = False
    provider_blocked: bool = False
    runtime_source: str = "local"
    server_label: str = ""


@dataclass(frozen=True)
class RoleplayHeaderView:
    """The header as it should paint at one width.

    Attributes:
        state: For ``DestinationHeader.sync_state``: the title, the kind as
            the subtitle and the status chip text (escaped: that chip is a
            markup-on ``Static``).
        item: The fitted item label's value, ``(text, editing)``, for
            ``fit_header_item`` (spec 5.3 interim: ``› <item>`` until B5b-2).
        unsaved_chip: ``"Unsaved changes"``, ``"Unsaved"`` or ``""`` (hidden).
        blocked_chip: The blocked-destination chip text, or ``""`` (hidden).
        status_plain: The status chip text before escaping, as it paints.
    """

    state: WorkbenchHeaderState
    item: tuple[str, bool]
    unsaved_chip: str
    blocked_chip: str
    status_plain: str


def mode_descriptor(mode: str) -> str:
    """The kind's one-line descriptor (falls back to its label, then its id)."""
    return MODE_DESCRIPTORS.get(mode, MODE_LABELS.get(mode, mode))


def purpose_line(mode: str, count: int | None) -> str:
    """The descriptor plus the live item count on one line (F-033).

    Args:
        mode: The active kind.
        count: Its item count, or ``None`` for a kind without one.

    Returns:
        ``"Characters — who the AI plays · 2"``, or the bare descriptor.
    """
    descriptor = mode_descriptor(mode)
    if count is None:
        return descriptor
    return f"{descriptor.rstrip('.')} · {count}"


def header_kind(mode: str) -> str:
    """The kind noun the header names (spec 1.3, RP-029); never cut."""
    return MODE_LABELS.get(mode, mode)


def header_item(inputs: RoleplayHeaderInputs) -> str:
    """The interim item text (spec 5.3): the new item's noun while creating.

    The name is put on one line (each whitespace run as one space), so all
    of it paints on the one-row header.
    """
    if inputs.edit_mode == "create":
        return "New persona" if inputs.mode == "personas" else "New character"
    return _one_line(inputs.item_name)


def initial_header_state(mode: str, runtime_source: str) -> WorkbenchHeaderState:
    """The state the header is composed with, before any input is gathered.

    It already names the kind and the data source, so the shared header's
    default "Ready" chip never paints, whatever order the first mount and the
    first ``_update_title`` run in.

    Args:
        mode: The active kind.
        runtime_source: ``"local"`` or ``"server"``.

    Returns:
        The title, the kind as the subtitle and the bare data-source word.
    """
    return WorkbenchHeaderState(
        title=HEADER_TITLE,
        subtitle=header_kind(mode),
        status_label="Server" if runtime_source == "server" else "Local",
    )


def fit_header_item(value: tuple[str, bool], width: int) -> str:
    """Fit ``› <item>[ · editing]`` into ``width`` cells.

    The name is cut first (ending in the resolved ellipsis), then dropped
    with its marker; `` · editing`` survives while it fits, so the header
    still reads ``Characters · editing``. The kind is not in this text: it
    is the header subtitle and is never cut.

    Args:
        value: ``(item, editing)`` from ``RoleplayHeaderView.item``; an empty
            value (``FittedText``'s default, before the first paint) paints
            nothing.
        width: The label's content width in cells.

    Returns:
        The plain text to paint (never markup).
    """
    item, editing = value or ("", False)
    state = f"{resolve_glyph(_SEPARATOR)} editing" if editing else ""
    suffix = f" {state}" if state else ""
    if item:
        marker = f"{resolve_glyph(_GO)} "
        whole = f"{marker}{item}{suffix}"
        if cell_len(whole) <= width:
            return whole
        cut = ellipsize_cells(item, width - cell_len(marker) - cell_len(suffix))
        if cut:
            return f"{marker}{cut}{suffix}"
    return state if cell_len(state) <= width else ""


def _status_text(inputs: RoleplayHeaderInputs, step: int) -> str:
    """The status chip at one degrade step (spec 1.3; DESIGN.md:115)."""
    if inputs.runtime_source != "server":
        return "Local"
    read_only = f"{resolve_glyph(_SEPARATOR)} read-only"
    label = ellipsize_cells(inputs.server_label, SERVER_LABEL_MAX_CELLS)
    if step == 0 and label:
        return f"Server: {label} {read_only}"
    if step < LAST_DEGRADE_STEP:
        return f"Server {read_only}"
    return "Server"


def _blocked_text(step: int) -> str:
    """The blocked-destination chip at one degrade step (never "Ready")."""
    go = resolve_glyph(_GO)
    if step < 2:
        return f"No chat provider {resolve_glyph(_SEPARATOR)} Settings {go}"
    return f"No chat provider {go}"


def _required_cells(kind: str, chips: tuple[str, ...], status: str) -> int:
    """Cells the header needs with an empty item label."""
    return (
        HEADER_PADDING_CELLS
        + cell_len(HEADER_TITLE)
        + KIND_GAP_CELLS
        + cell_len(kind)
        + ITEM_GAP_CELLS
        + sum(cell_len(chip) + CHIP_CHROME_CELLS for chip in chips if chip)
        + cell_len(status)
        + STATUS_CHROME_CELLS
    )


def build_header_view(inputs: RoleplayHeaderInputs, width: int) -> RoleplayHeaderView:
    """The header view for ``inputs`` on a ``width``-column header.

    The unsaved chip is short below ``UNSAVED_SHORT_BELOW_COLUMNS``. Then the
    longest forms that fit win, degrading in this order until they fit:
    drop the server label, shorten the blocked chip, then show the status as
    ``Server``. The title and the kind never change, and the item label takes
    whatever is left (``fit_header_item``).

    Args:
        inputs: Gathered by the screen.
        width: The header's outer width in cells.

    Returns:
        The view to paint.
    """
    kind = header_kind(inputs.mode)
    unsaved = ""
    if inputs.unsaved:
        unsaved = (
            UNSAVED_CHIP_SHORT if width < UNSAVED_SHORT_BELOW_COLUMNS else UNSAVED_CHIP
        )
    for step in range(LAST_DEGRADE_STEP + 1):
        blocked = _blocked_text(step) if inputs.provider_blocked else ""
        status = _status_text(inputs, step)
        if _required_cells(kind, (unsaved, blocked), status) <= width:
            break
    return RoleplayHeaderView(
        state=WorkbenchHeaderState(
            title=HEADER_TITLE,
            subtitle=kind,
            status="ready",
            status_label=escape_markup(status),
        ),
        item=(header_item(inputs), inputs.edit_mode != "view"),
        unsaved_chip=unsaved,
        blocked_chip=blocked,
        status_plain=status,
    )
