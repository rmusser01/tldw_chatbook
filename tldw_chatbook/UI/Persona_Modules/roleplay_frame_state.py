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

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Protocol

from rich.cells import cell_len

from tldw_chatbook.UI.Persona_Modules.roleplay_list_window_state import (
    _ELLIPSIS,
    _SEPARATOR,
    DELETE_MODAL_MAX_NAMES,
    ITEM_KINDS,
    KIND_FOR_MODE,
    MODE_FOR_KIND,
    NAME_TAIL_CELLS,
    NEW_BLANK_FOR_MODE,
    PORTRAIT_NEEDS_LOCAL,
    SCENT_MAX_CELLS,
    SERVER_CHARACTERS_READ_ONLY,
    SNIPPET_MAX_CELLS,
    TAG_MAX_CELLS,
    TWO_LINE_MAX_ITEMS,
    TWO_LINE_MIN_WIDTH,
    UNMEASURED_WIDTH,
    DeleteModalCopy,
    ItemPrompt,
    MarksBarView,
    NewChoice,
    _one_line,
    build_item_prompt,
    character_scent,
    delete_modal_copy,
    dictionary_scent,
    ellipsize_cells,
    gutter_mark_columns,
    hidden_mark_count,
    import_landed_toast,
    import_landing,
    imported_toast,
    item_density,
    item_gutter,
    items_border_title,
    loaded_position,
    lore_scent,
    marks_bar_view,
    middle_ellipsize_cells,
    new_chooser_options,
    persona_scent,
    plural,
    relative_age,
    turned_off_toast,
)
from tldw_chatbook.UI.Persona_Modules.roleplay_list_window_state import (
    _LOADED_GLYPH as _LOADED_GLYPH,  # noqa: PLC0414 -- preserve the original import seam.
)
from tldw_chatbook.UI.Persona_Modules.roleplay_list_window_state import (
    _MARK_GLYPH as _MARK_GLYPH,  # noqa: PLC0414 -- preserve the original import seam.
)
from tldw_chatbook.UI.Persona_Modules.roleplay_list_window_state import (
    ListAnchor as ListAnchor,
)
from tldw_chatbook.UI.Persona_Modules.roleplay_list_window_state import (
    ListFence as ListFence,
)
from tldw_chatbook.UI.Persona_Modules.roleplay_list_window_state import (
    ListWindowRequest as ListWindowRequest,
)
from tldw_chatbook.UI.Persona_Modules.roleplay_list_window_state import (
    ListWindowState as ListWindowState,
)
from tldw_chatbook.UI.Persona_Modules.roleplay_list_window_state import (
    WindowIntent as WindowIntent,
)
from tldw_chatbook.UI.Persona_Modules.roleplay_list_window_state import (
    new_option_ids as new_option_ids,
)
from tldw_chatbook.UI.Persona_Modules.roleplay_list_window_state import (
    same_list as same_list,
)
from tldw_chatbook.UI.Persona_Modules.roleplay_list_window_state import (
    window_wants_more as window_wants_more,
)
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


# ---- The items toolbar (frame slice B2b, spec 1.5.1; owner decision D2) ----

#: The search box never gets fewer cells than this (spec 1.5.1).
TOOLBAR_SEARCH_MIN_CELLS = 14
#: Cells the toolbar spends around its parts. They mirror the CSS:
#: ``Tests/UI/test_roleplay_item_list_app.py`` pins them against the mounted
#: widgets, so a CSS edit that forgets these fails there instead of clipping.
SEARCH_CHROME_CELLS = 3  # .settings-compact-input: border-left + padding 0 1
CONTROL_GAP_CELLS = 1  # .roleplay-items-control: margin-left $ds-space-1
RAIL_COLLAPSE_CELLS = 3  # .console-rail-collapse-button: width $ds-size-3
RAIL_TITLE_CHROME_CELLS = 2  # .personas-column-title: padding 0 1
#: The rail title, which yields to the interim Import (spec 5.3) when both
#: do not fit (below about 100 columns).
RAIL_TITLE = "Library"
#: The sort the server's own order shows under (it has no local key).
SERVER_SORT_KEY = "server"
#: Each sort's label, full and short (spec 1.5.1: labels shorten, never wrap).
SORT_LABELS: dict[str, tuple[str, str]] = {
    "relevance": ("Relevance", "Best"),
    "name_asc": ("Name", "Name"),
    "modified_desc": ("Recent edit", "Edited"),
    "created_desc": ("Recent add", "Added"),
    SERVER_SORT_KEY: ("Server order", "Server"),
}
#: The controls each kind's toolbar row shows, left to right (spec 1.5.1);
#: a mode without an items list keeps only ``+ New``.
TOOLBAR_CONTROLS: dict[str, tuple[str, ...]] = {
    "characters": ("sort", "tag", "new"),
    "personas": ("sort", "new"),
    "lore": ("new",),
    "dictionaries": ("new",),
}


def toolbar_controls(mode: str) -> tuple[str, ...]:
    """The controls a mode's toolbar shows (``sort``, ``tag``, ``new``)."""
    return TOOLBAR_CONTROLS.get(mode, ("new",))


def sort_label(sort_key: str, *, short: bool = False) -> str:
    """The sort button's label; an unknown key reads as ``Name`` (as on dev)."""
    full, short_form = SORT_LABELS.get(sort_key, SORT_LABELS["name_asc"])
    return short_form if short else full


def sort_tooltip(sort_key: str) -> str:
    """The sort button's tooltip: the whole sort name and the key (``s``)."""
    return f"Sort: {sort_label(sort_key)}. Click to change the order. (s)"


def search_placeholder(mode: str, *, short: bool) -> str:
    """The search box's placeholder: ``Search characters…`` or ``Search…``.

    No ``/`` hint until frame slice B3 binds the key (owner decision D4).
    """
    ellipsis = resolve_glyph(_ELLIPSIS)
    kind = KIND_FOR_MODE.get(mode)
    if short or kind is None:
        return f"Search{ellipsis}"
    return f"Search {ITEM_KINDS[kind][1]}{ellipsis}"


def import_label(*, short: bool = False) -> str:
    """The interim ``Import…`` on the rail's title row (spec 5.3; owner D2).

    ``short`` drops the ellipsis: the one rung where the header row cannot
    hold ``Import...`` beside the collapse button (12 cells, ASCII mode).
    """
    return "Import" if short else f"Import{resolve_glyph(_ELLIPSIS)}"


@dataclass(frozen=True)
class ToolbarFit:
    """How the items toolbar lays out at one width (spec 1.5.1).

    Attributes:
        rows: The toolbar's rows: 1 (the search beside the controls), 2 (the
            controls on the row under the search) or one more per control
            (stacked).
        labels: The words to paint, by part: the search placeholder
            (``search``), ``import`` and each shown control.
        stacked: Each control takes its own row (only where one row of
            controls does not fit: 80x24).
        title: The rail title shows beside ``Import…``.
    """

    rows: int
    labels: Mapping[str, str]
    stacked: bool
    title: bool


def fit_items_toolbar(mode: str, *, width: int, sort_key: str) -> ToolbarFit:
    """The toolbar's layout at ``width`` cells: a ladder, never a clipped label.

    One row with the full labels, then one row with the short ones, then two
    rows (full, then short), then every control on its own row. The search
    keeps at least ``TOOLBAR_SEARCH_MIN_CELLS`` beside the controls and shows
    its long placeholder only where it fits. On the header row, Import loses
    its ellipsis only where it and the collapse button cannot otherwise fit.

    Args:
        mode: The workbench mode (``toolbar_controls``).
        width: The pane's content width.
        sort_key: The list's sort (``SERVER_SORT_KEY`` for the server's order).

    Returns:
        The rows, the labels to paint, whether the controls stack, and
        whether the rail title shows.
    """
    controls = toolbar_controls(mode)

    def labels_for(short: bool) -> dict[str, str]:
        words = {
            "sort": sort_label(sort_key, short=short),
            "tag": "Tag",
            "new": "+ New",
        }
        return {control: words[control] for control in controls}

    def row_cells(labels: Mapping[str, str]) -> int:
        return sum(CONTROL_GAP_CELLS + cell_len(label) for label in labels.values())

    def placeholder(search_cells: int) -> str:
        full = search_placeholder(mode, short=False)
        fits = cell_len(full) <= search_cells - SEARCH_CHROME_CELLS
        return full if fits else search_placeholder(mode, short=True)

    import_words = import_label()
    if CONTROL_GAP_CELLS + cell_len(import_words) + RAIL_COLLAPSE_CELLS > width:
        # The header row holds Import and the collapse button whole, or the
        # ellipsis goes: at 80x24 (12 cells) ASCII's "Import..." took 9
        # cells and the button spilled one cell past the pane (new with B2b,
        # closed in its final fix wave; the Unicode "Import…" always fit).
        import_words = import_label(short=True)
    beside_title = width - CONTROL_GAP_CELLS - cell_len(import_words)
    title = beside_title - RAIL_COLLAPSE_CELLS >= (
        cell_len(RAIL_TITLE) + RAIL_TITLE_CHROME_CELLS
    )

    def fit(
        rows: int, labels: dict[str, str], search_cells: int, stacked: bool
    ) -> ToolbarFit:
        words = {"search": placeholder(search_cells), "import": import_words}
        return ToolbarFit(rows, {**words, **labels}, stacked, title)

    full, short = labels_for(False), labels_for(True)
    for labels in (full, short):
        if TOOLBAR_SEARCH_MIN_CELLS + row_cells(labels) <= width:
            return fit(1, labels, width - row_cells(labels), False)
    for labels in (full, short):
        if row_cells(labels) <= width:
            return fit(2, labels, width, False)
    widest = max(CONTROL_GAP_CELLS + cell_len(label) for label in full.values())
    labels = full if widest <= width else short
    return fit(1 + len(controls), labels, width, True)


# ---- Empty, no match, loading (frame slice B2b, spec 1.5.4) ----

#: ``Loading <kind>…`` shows only once a kind's read has taken this long.
LOADING_DELAY_SECONDS = 0.25
#: The echoed query or tag is cut to this many cells, on one line (R33).
NO_MATCH_ECHO_MAX_CELLS = 24
#: ``Also in:`` lists the other kinds in the rail's order (spec 1.4.7 item 5).
ALSO_IN_ORDER: tuple[str, ...] = ("character", "persona", "lore", "dictionary")
ALSO_IN_LABEL = "Also in:"
NOT_FOUND_ELSEWHERE = "Not found in other kinds either."


def empty_kind_text(kind: str) -> str:
    """The list's line when the kind holds nothing at all (its unfiltered total is 0).

    It keeps the New/Import hint until frame slice B5b-2's kind empty state
    takes the explanation (spec R25), in the canonical nouns.
    """
    return f"No {ITEM_KINDS[kind][1]} yet - use New or Import to add one."


def loading_text(kind: str) -> str:
    """``Loading characters…`` (spec 1.5.4)."""
    return f"Loading {ITEM_KINDS[kind][1]}{resolve_glyph(_ELLIPSIS)}"


@dataclass(frozen=True)
class AlsoInTarget:
    """One ``Also in:`` target: another kind the search matches (spec 1.5.4).

    Attributes:
        kind: The entity kind (``ITEM_KINDS`` keys).
        count: How many of that kind match.
        label: ``Lore books 1 ›`` (fixed words and a count: nothing untrusted).
    """

    kind: str
    count: int
    label: str


@dataclass(frozen=True)
class NoMatchView:
    """What the list says when a search or tag leaves it empty (spec 1.5.4).

    Attributes:
        lines: What found nothing (``No characters match "kepler".``) and the
            way out (``28 in total · esc clears the search``). Plain text: the
            pane paints it with markup off (the query and tag are untrusted).
        targets: The other kinds the query matches, in rail order.
        none_text: ``Not found in other kinds either.`` once every other
            kind was counted and none matches; None otherwise.
    """

    lines: tuple[str, str]
    targets: tuple[AlsoInTarget, ...] = ()
    none_text: str | None = None


def no_match_view(
    kind: str,
    *,
    query: str,
    tag: str | None,
    total: int | None,
    counts: Mapping[str, int] | None = None,
    failed: Collection[str] = (),
    capped: Collection[str] = (),
) -> NoMatchView:
    """The no-match message for a search, a tag, or both (spec 1.5.4, RP-022).

    Args:
        kind: The list's entity kind (``ITEM_KINDS`` keys).
        query: The search as typed (cut to one line of
            ``NO_MATCH_ECHO_MAX_CELLS`` cells).
        tag: The active tag, if any (cut the same way).
        total: The kind's unfiltered total; None when it is unknown (a
            server search), which leaves ``N in total`` out.
        counts: The other kinds' match counts once they are in (None while
            they are counted, or for a tag alone).
        failed: Kinds whose count failed: never listed, and they keep
            ``Not found in other kinds either.`` from being claimed.
        capped: Kinds whose count is a lower bound (a server character
            search answers at most 100 rows with no total, as its list's
            ``100+`` says; the cap exists on dev): the label reads ``100+``.

    Returns:
        The two lines, the ``Also in:`` targets and the not-found line.
    """
    noun = ITEM_KINDS[kind][1]

    def echo(text: str) -> str:
        return ellipsize_cells(_one_line(text), NO_MATCH_ECHO_MAX_CELLS)

    if query and tag:
        found = f'No {noun} match "{echo(query)}" with the tag "{echo(tag)}".'
        cleared = "the search and tag"
    elif tag:
        found = f'No {noun} have the tag "{echo(tag)}".'
        cleared = "the tag"
    else:
        found = f'No {noun} match "{echo(query)}".'
        cleared = "the search"
    way_out = f"esc clears {cleared}"
    if total is not None:
        way_out = f"{total} in total {resolve_glyph(_SEPARATOR)} {way_out}"
    if counts is None:
        return NoMatchView((found, way_out))
    others = [other for other in ALSO_IN_ORDER if other != kind]
    targets = tuple(
        AlsoInTarget(
            other,
            counts[other],
            f"{ITEM_KINDS[other][2]} {counts[other]}"
            f"{'+' if other in capped else ''} {resolve_glyph(_GO)}",
        )
        for other in others
        if other not in failed and counts.get(other, 0) > 0
    )
    counted = all(other in counts and other not in failed for other in others)
    none_text = NOT_FOUND_ELSEWHERE if counted and not targets else None
    return NoMatchView((found, way_out), targets, none_text)


__all__ = [
    "ALSO_IN_LABEL",
    "ALSO_IN_ORDER",
    "CHIP_CHROME_CELLS",
    "CONTROL_GAP_CELLS",
    "DELETE_MODAL_MAX_NAMES",
    "HEADER_PADDING_CELLS",
    "HEADER_TITLE",
    "ITEM_GAP_CELLS",
    "ITEM_KINDS",
    "KIND_FOR_MODE",
    "KIND_GAP_CELLS",
    "LAST_DEGRADE_STEP",
    "LOADING_DELAY_SECONDS",
    "MODE_DESCRIPTORS",
    "MODE_FOR_KIND",
    "MODE_LABELS",
    "NAME_TAIL_CELLS",
    "NEW_BLANK_FOR_MODE",
    "NOT_FOUND_ELSEWHERE",
    "NO_MATCH_ECHO_MAX_CELLS",
    "PORTRAIT_NEEDS_LOCAL",
    "RAIL_COLLAPSE_CELLS",
    "RAIL_TITLE",
    "RAIL_TITLE_CHROME_CELLS",
    "ROLEPLAY_PANE_CLASS_NAMES",
    "SCENT_MAX_CELLS",
    "SEARCH_CHROME_CELLS",
    "SERVER_CHARACTERS_READ_ONLY",
    "SERVER_LABEL_MAX_CELLS",
    "SERVER_SORT_KEY",
    "SNIPPET_MAX_CELLS",
    "SORT_LABELS",
    "STATUS_CHROME_CELLS",
    "TAG_MAX_CELLS",
    "TOOLBAR_CONTROLS",
    "TOOLBAR_SEARCH_MIN_CELLS",
    "TWO_LINE_MAX_ITEMS",
    "TWO_LINE_MIN_WIDTH",
    "UNMEASURED_WIDTH",
    "UNSAVED_CHIP",
    "UNSAVED_CHIP_SHORT",
    "UNSAVED_SHORT_BELOW_COLUMNS",
    "AlsoInTarget",
    "Collection",
    "DeleteModalCopy",
    "DraftSnapshotLike",
    "ItemPrompt",
    "ListAnchor",
    "ListFence",
    "ListWindowRequest",
    "ListWindowState",
    "Literal",
    "Mapping",
    "MarksBarView",
    "NewChoice",
    "NoMatchView",
    "Protocol",
    "RoleplayHeaderInputs",
    "RoleplayHeaderView",
    "Sequence",
    "ToolbarFit",
    "WindowIntent",
    "WorkbenchHeaderState",
    "build_header_view",
    "build_item_prompt",
    "cell_len",
    "character_scent",
    "dataclass",
    "datetime",
    "delete_modal_copy",
    "dictionary_scent",
    "ellipsize_cells",
    "empty_kind_text",
    "escape_markup",
    "fit_header_item",
    "fit_items_toolbar",
    "gutter_mark_columns",
    "header_item",
    "header_kind",
    "hidden_mark_count",
    "import_label",
    "import_landed_toast",
    "import_landing",
    "imported_toast",
    "initial_header_state",
    "item_density",
    "item_gutter",
    "items_border_title",
    "loaded_position",
    "loading_text",
    "lore_scent",
    "marks_bar_view",
    "middle_ellipsize_cells",
    "mode_descriptor",
    "new_chooser_options",
    "new_option_ids",
    "no_match_view",
    "persona_scent",
    "plural",
    "purpose_line",
    "relative_age",
    "resolve_glyph",
    "roleplay_has_unsaved_work",
    "runtime_server_label",
    "same_list",
    "search_placeholder",
    "sort_label",
    "sort_tooltip",
    "timezone",
    "toolbar_controls",
    "turned_off_toast",
    "window_wants_more",
]
