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
    ListAnchor as ListAnchor,
    ListFence as ListFence,
    ListWindowRequest as ListWindowRequest,
    ListWindowState as ListWindowState,
    WindowIntent as WindowIntent,
    new_option_ids as new_option_ids,
    same_list as same_list,
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


# ===== Items list (frame slice B2a; spec section 1.5) =======================
#
# Everything the line-based items list paints is computed here as plain text
# or as ``(text, style)`` spans; ``RoleplayItemList`` turns spans into literal
# ``Content`` (markup off, spec R33). Untrusted names are measured and cut as
# plain text, never escaped (a backslash would paint).

#: Each entity kind's list words: singular noun, plural noun, border-title word.
ITEM_KINDS: dict[str, tuple[str, str, str]] = {
    "character": ("character", "characters", "Characters"),
    "persona": ("persona", "personas", "Personas"),
    "lore": ("lore book", "lore books", "Lore books"),
    "dictionary": ("chat dictionary", "chat dictionaries", "Chat dictionaries"),
}
#: The workbench mode that lists each kind (``PersonasWorkbenchState.active_mode``).
KIND_FOR_MODE: dict[str, str] = {
    "characters": "character",
    "personas": "persona",
    "lore": "lore",
    "dictionaries": "dictionary",
}
#: 2-line rows need at least this many cells inside the list (spec 1.5.2) ...
TWO_LINE_MIN_WIDTH = 40
#: ... and at most this many items in the kind; 1-line rows otherwise.
TWO_LINE_MAX_ITEMS = 40
#: A cut name keeps this many cells of its end (spec 1.5.2: "the last 9-10").
NAME_TAIL_CELLS = 10
#: A character scent's description snippet and first tag are cut to these.
SNIPPET_MAX_CELLS = 72
TAG_MAX_CELLS = 24
#: No scent is longer than this as data; the prompt cuts it to the row.
SCENT_MAX_CELLS = 96
#: The delete modal names at most this many items, then "and N more" (R32).
DELETE_MODAL_MAX_NAMES = 5
#: The width prompts are built at before the list is laid out: nothing is cut,
#: and the first real width rebuilds only the prompts it cuts.
UNMEASURED_WIDTH = 10_000

_LOADED_GLYPH = "▸"
_MARK_GLYPH = "✓"


def plural(count: int, singular: str, plural_form: str | None = None) -> str:
    """The count and the noun in the number that matches it (RP-070: "1 entry").

    Args:
        count: How many.
        singular: The noun for one.
        plural_form: The noun for any other count; ``singular + "s"`` if omitted.

    Returns:
        ``"<count> <noun>"``.
    """
    noun = singular if count == 1 else (plural_form or f"{singular}s")
    return f"{count} {noun}"


def relative_age(timestamp: object, *, now: datetime) -> str | None:
    """A timestamp as a short relative age (``now``, ``5m``, ``3h``, ``3d``, ``2mo``, ``2y``).

    Raw UTC dates read as noise in a list (RP-070). A naive timestamp is UTC,
    as the database writes them; a future one reads ``now``.

    Args:
        timestamp: An ISO-8601 string from the page query, or anything else.
        now: The current time (injected, so the result is testable).

    Returns:
        The age, or ``None`` when ``timestamp`` is not a parseable date.
    """
    if not isinstance(timestamp, str) or not timestamp.strip():
        return None
    try:
        moment = datetime.fromisoformat(timestamp.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    seconds = (now - moment).total_seconds()
    if seconds < 60:
        return "now"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    days = hours // 24
    if days < 30:
        return f"{days}d"
    if days < 365:
        return f"{days // 30}mo"
    return f"{days // 365}y"


def middle_ellipsize_cells(
    text: str, budget: int, *, tail: int = NAME_TAIL_CELLS
) -> str:
    """Cut ``text`` to ``budget`` cells in the middle, keeping its last cells.

    ``Highspire Gazette (imported from card) 2`` at 28 cells reads
    ``Highspire Gazette…om card) 2``: the end of a name tells copies apart
    (spec 1.5.2). Measures cells, so wide and zero-width characters fit by
    what they paint; the ellipsis is the resolved glyph (``...`` in ASCII mode).

    Args:
        text: Plain (unescaped) one-line text.
        budget: Cells available.
        tail: Cells of the end to keep, at most half of what is left.

    Returns:
        ``text`` when it fits, else head + ellipsis + tail within ``budget``
        (just as much of the ellipsis as fits when ``budget`` is tiny).
    """
    if cell_len(text) <= budget:
        return text
    ellipsis = resolve_glyph(_ELLIPSIS)
    room = budget - cell_len(ellipsis)
    if room <= 0:
        return ellipsis if cell_len(ellipsis) <= budget else ""
    keep = min(tail, room // 2)
    end = ""
    for character in reversed(text):
        if cell_len(character + end) > keep:
            break
        end = character + end
    head = ""
    for character in text:
        if cell_len(head + character) > room - cell_len(end):
            break
        head += character
    return f"{head.rstrip()}{ellipsis}{end}"


def item_gutter(*, loaded: bool, marked: bool) -> str:
    """The row gutter: col 1 ``▸`` (loaded), col 2 ``✓`` (marked), col 3 a space.

    In ASCII mode the glyphs resolve to ``>`` and ``[x]``, so the gutter is
    5 cells (``>[x] ``, spec 1.5.2). Unset columns are spaces of the same width.
    """
    loaded_glyph = resolve_glyph(_LOADED_GLYPH)
    mark_glyph = resolve_glyph(_MARK_GLYPH)
    return (
        (loaded_glyph if loaded else " " * cell_len(loaded_glyph))
        + (mark_glyph if marked else " " * cell_len(mark_glyph))
        + " "
    )


def gutter_mark_columns() -> range:
    """The content columns of the mark cell: a click there toggles the mark (spec 4.10)."""
    start = cell_len(resolve_glyph(_LOADED_GLYPH))
    return range(start, start + cell_len(resolve_glyph(_MARK_GLYPH)))


def item_density(*, inner_width: int, kind_total: int) -> int:
    """2-line rows when the list is at least 40 cells wide and the kind holds at most 40 items."""
    if inner_width >= TWO_LINE_MIN_WIDTH and kind_total <= TWO_LINE_MAX_ITEMS:
        return 2
    return 1


@dataclass(frozen=True)
class ItemPrompt:
    """One row's prompt as styled spans: ``lines[i]`` is ``((text, style), ...)``.

    ``full_cells`` is the row's widest line with nothing cut; ``cut`` says some
    text gave way to ``width``; ``name_cut`` says the name did (its hover
    tooltip then shows it whole).
    """

    lines: tuple[tuple[tuple[str, str], ...], ...]
    full_cells: int
    cut: bool
    name_cut: bool
    width: int
    density: int

    def needs_rebuild(self, *, width: int, density: int) -> bool:
        """Whether this prompt reads differently at ``width``/``density`` (spec 1.5.6 RC-6).

        A prompt that fitted and still fits is reused as it is, so a resize
        rebuilds only the rows it actually changes.
        """
        if density != self.density:
            return True
        if self.cut:
            return width != self.width
        return self.full_cells > width


def build_item_prompt(
    *,
    name: str,
    state: str | None,
    meta: str | None,
    short: str | None,
    loaded: bool,
    marked: bool,
    width: int,
    density: int,
) -> ItemPrompt:
    """One row of the items list (spec 1.5.2, 1.5.3).

    Line 1 is the gutter, the name and ``· <state>`` (``on``/``off``; the word
    never drops, the name gives way through the middle ellipsis). At 1-line
    density a book adds ``· <short>`` (its entry count) only when the full name
    still fits. At 2-line density line 2 is the dim scent, or blank, so every
    row of the list has the same height. The loaded row's name is bold; an
    ``off`` book's name is muted. All text stays plain (spans, no markup).

    Args:
        name: The item's name (untrusted; any whitespace run paints as one space).
        state: ``"on"``/``"off"`` for books, else ``None``.
        meta: The scent for line 2, or ``None``.
        short: The 1-line extra for books (``"250"``), or ``None``.
        loaded: The row is loaded in the work pane.
        marked: The row is marked.
        width: The list's inner width in cells.
        density: 1 or 2 (``item_density``).

    Returns:
        The prompt's spans and what a later width change must rebuild.
    """
    gutter = item_gutter(loaded=loaded, marked=marked)
    separator = f" {resolve_glyph(_SEPARATOR)} "
    clean_name = _one_line(name)
    state_tail = f"{separator}{state}" if state else ""
    short_tail = f"{separator}{short}" if density == 1 and short else ""
    full_cells = cell_len(gutter + clean_name + short_tail + state_tail)
    if full_cells <= width:
        shown, tail, name_cut = clean_name, short_tail + state_tail, False
    else:
        tail = state_tail
        room = max(0, width - cell_len(gutter) - cell_len(tail))
        shown = middle_ellipsize_cells(clean_name, room)
        name_cut = shown != clean_name
    cut = name_cut or (short_tail != "" and short_tail not in tail)
    name_style = "bold" if loaded else ("dim" if state == "off" else "")
    lines: list[tuple[tuple[str, str], ...]] = [
        ((gutter, ""), (shown, name_style), (tail, ""))
    ]
    if density == 2:
        blank = " " * cell_len(gutter)
        scent = _one_line(meta or "")
        full_cells = max(full_cells, cell_len(blank + scent))
        room = max(0, width - cell_len(blank))
        if cell_len(scent) > room:
            scent = ellipsize_cells(scent, room)
            cut = True
        lines.append(((blank, ""), (scent, "dim")) if scent else ((blank, ""),))
    return ItemPrompt(tuple(lines), full_cells, cut, name_cut, width, density)


def character_scent(record: Mapping[str, object], *, now: datetime) -> str | None:
    """Line 2 of a character row from the page query's own fields (spec 1.5.2 v1 scent).

    ``<age> · <first tag> · <description snippet>``. A missing part is absent,
    never ``0``; the chat count arrives with frame slice B6's aggregates.
    """
    parts: list[str] = []
    age = relative_age(record.get("last_modified"), now=now)
    if age:
        parts.append(age)
    tags = record.get("tags")
    if isinstance(tags, (list, tuple)) and tags:
        tag = _one_line(str(tags[0]))
        if tag:
            parts.append(ellipsize_cells(tag, TAG_MAX_CELLS))
    snippet = _one_line(str(record.get("description") or ""))
    if snippet:
        parts.append(ellipsize_cells(snippet, SNIPPET_MAX_CELLS))
    if not parts:
        return None
    return ellipsize_cells(
        f" {resolve_glyph(_SEPARATOR)} ".join(parts), SCENT_MAX_CELLS
    )


def persona_scent(record: Mapping[str, object]) -> str | None:
    """Line 2 of a persona row: the first words of its system prompt (the field that reaches the model)."""
    prompt = _one_line(str(record.get("system_prompt") or ""))
    return ellipsize_cells(prompt, SNIPPET_MAX_CELLS) if prompt else None


def lore_scent(entry_count: int) -> str:
    """Line 2 of a lore-book row (``· N off`` and ``used by`` arrive with B6 and C)."""
    return plural(entry_count, "entry", "entries")


def dictionary_scent(entry_count: int) -> str:
    """Line 2 of a chat-dictionary row (regex and invalid counts arrive with B6)."""
    return plural(entry_count, "rule")


def items_border_title(
    kind: str,
    *,
    total: int | None,
    matched: int | None = None,
    tag: str | None = None,
    enabled: int | None = None,
    total_known: bool = True,
    failed: bool = False,
) -> str:
    """The items pane's top-border title (spec 1.5.1, 1.4.2).

    ``Characters 28``; ``Characters 3 of 28`` while a search or tag narrows the
    list; ``Characters 20 of 348 · tag fantasy``; ``Characters (…)`` before the
    first count; ``Characters 100+`` while a server total is unknown;
    ``Characters (?)`` when the list failed; books add ``· N on``. Plain text:
    the pane paints it as literal ``Content`` (the tag is untrusted).
    """
    word = ITEM_KINDS.get(kind, ("item", "items", "Items"))[2]
    separator = f" {resolve_glyph(_SEPARATOR)} "
    if failed:
        return f"{word} (?)"
    if total is None:
        return f"{word} ({resolve_glyph(_ELLIPSIS)})"
    count = f"{total}{'' if total_known else '+'}"
    text = f"{word} {matched} of {count}" if matched is not None else f"{word} {count}"
    if tag:
        text += f"{separator}tag {_one_line(tag)}"
    if enabled is not None:
        text += f"{separator}{enabled} on"
    return text


def loaded_position(index: int | None, total: int | None) -> str:
    """The bottom border's ``#4 of 28``; blank with nothing loaded or the row not listed (RC-10)."""
    if index is None or total is None:
        return ""
    return f"#{index + 1} of {total}"


def hidden_mark_count(
    marked_ids: Collection[str], loaded_ids: Collection[str], *, filtered: bool
) -> int:
    """Marks the current search or tag hides (spec 1.5.5 "(2 hidden)").

    With no search or tag every marked row is reachable by scrolling, so none
    is hidden. While a filter is on, a marked id absent from the loaded rows
    is counted: exact when the filtered result is fully loaded (≤ 200 matches
    at first load), and an unloaded match counts until it is scrolled in.
    """
    if not filtered:
        return 0
    return len(set(marked_ids) - set(loaded_ids))


@dataclass(frozen=True)
class MarksBarView:
    """What the one-row marks bar shows (spec 1.5.5)."""

    count_text: str
    count_tooltip: str
    labels: tuple[str, ...]
    separator: str
    #: The action each label belongs to (``export``, ``delete``, ``clear``).
    actions: tuple[str, ...]


def marks_bar_view(
    count: int, hidden: int, width: int, *, can_export: bool
) -> MarksBarView | None:
    """The marks bar's text at ``width`` cells, or ``None`` while nothing is marked.

    Full form ``✓ 2 marked · Export… · Delete… · Clear esc``. When it does not
    fit the bar degrades in a fixed order: the separators and the ``esc`` hint
    go, then ``marked``, then the hidden count (the tooltip keeps the full
    count). Narrower still, whole actions drop and none is ever cut: Clear
    first (Escape on the list clears the marks too), then Export; then the
    count yields to Delete (it moves into Delete's tooltip, empty
    ``count_text``). Delete never drops while marks exist: it is the bulk
    path (R32). ``width`` is the bar's own content width. Export applies to
    characters and personas.
    """
    if count <= 0:
        return None
    mark = resolve_glyph(_MARK_GLYPH)
    dot = f" {resolve_glyph(_SEPARATOR)} "
    hidden_part = f" ({hidden} hidden)" if hidden else ""
    actions = ("export", "delete", "clear") if can_export else ("delete", "clear")
    long_labels = {"export": "Export…", "delete": "Delete…", "clear": "Clear esc"}
    short_labels = {"export": "Export", "delete": "Delete", "clear": "Clear"}
    tooltip = f"{count} marked" + (
        f", {hidden} hidden by the search or tag" if hidden else ""
    )
    short = f"{mark} {count}"
    ladder = (
        (f"{short} marked{hidden_part}", actions, long_labels, dot),
        (f"{short} marked{hidden_part}", actions, short_labels, " "),
        (f"{short}{hidden_part}", actions, short_labels, " "),
        (short, actions, short_labels, " "),
        (short, tuple(a for a in actions if a != "clear"), short_labels, " "),
        (short, ("delete",), short_labels, " "),
    )
    for text, shown, names, separator in ladder:
        labels = tuple(names[action] for action in shown)
        cells = cell_len(text) + sum(cell_len(separator + label) for label in labels)
        if cells <= width:
            return MarksBarView(text, tooltip, labels, separator, shown)
    # Too narrow for the count beside Delete: the count yields (its tooltip
    # moves onto Delete), never Delete, even narrower than its own label.
    return MarksBarView("", tooltip, (short_labels["delete"],), "", ("delete",))


@dataclass(frozen=True)
class DeleteModalCopy:
    """The one delete modal's strings (R32); shown by ``ConfirmationDialog`` with markup off."""

    title: str
    message: str
    confirm_label: str
    cancel_label: str = "Cancel"


def delete_modal_copy(
    kind: str, names: Sequence[str], *, bulk: bool
) -> DeleteModalCopy:
    """Copy for every Roleplay delete (R32): it names up to 5 items plus "and N more".

    A single delete keeps dev's wording. A bulk delete names what it will
    delete, so it never hits an unseen row (RP-039, F-040), and its button
    says ``Delete N marked``.
    """
    singular, plural_noun, _word = ITEM_KINDS.get(kind, ("item", "items", "Items"))
    clean = [_one_line(name) or "Unnamed" for name in names]
    if not bulk:
        subject = clean[0] if clean else f"this {singular}"
        return DeleteModalCopy(
            title="Delete",
            message=f"Delete {subject}? This cannot be undone here.",
            confirm_label="Delete",
        )
    count = len(clean)
    question = (
        f"Delete this {singular}?"
        if count == 1
        else f"Delete these {count} {plural_noun}?"
    )
    lines = [f"{question} This cannot be undone here.", ""]
    lines += [f"  {name}" for name in clean[:DELETE_MODAL_MAX_NAMES]]
    if count > DELETE_MODAL_MAX_NAMES:
        lines.append(f"  and {count - DELETE_MODAL_MAX_NAMES} more")
    return DeleteModalCopy(
        title=f"Delete {count} marked",
        message="\n".join(lines),
        confirm_label=f"Delete {count} marked",
    )


def turned_off_toast(name: str) -> str:
    """R33's one undo copy string, shown when Space turns a book off."""
    return f"{_one_line(name)} turned off {resolve_glyph(_SEPARATOR)} space to undo"


# ---- The New chooser (frame slice B2b, spec 1.4.1 "New…", 1.4.6) ----

#: The screen's read-only reason while browsing server characters; the New
#: chooser's character rows give the same words.
SERVER_CHARACTERS_READ_ONLY = "Server characters are read-only here."
#: Why "Persona · From a portrait" is off: the portraits are local characters.
PORTRAIT_NEEDS_LOCAL = "Portraits come from local characters."
#: The Blank row each kind's chooser opens on (spec 1.4.6: Enter Enter gives a
#: blank draft of the current kind).
NEW_BLANK_FOR_MODE: dict[str, str] = {
    "characters": "character:blank",
    "personas": "persona:blank",
    "lore": "lore",
    "dictionaries": "dictionary",
}


@dataclass(frozen=True)
class NewChoice:
    """One row of the New chooser.

    Attributes:
        choice_id: The create variant (``character:blank``, ``lore``, ...).
        mode: The workbench mode that lists what it creates; the screen
            switches there (inside the unsaved guard) before it creates.
        label: The row's words, painted literally.
        hint: A dim note after the label, or None.
        enabled: False when the variant cannot run here.
        reason: Why it is disabled (shown instead of the hint).
        preselected: The row the chooser opens on.
    """

    choice_id: str
    mode: str
    label: str
    hint: str | None
    enabled: bool
    reason: str | None
    preselected: bool


def new_chooser_options(
    mode: str, *, local_characters: bool, local_runtime: bool
) -> tuple[NewChoice, ...]:
    """Every create variant in the rail's order, the current kind's Blank first-picked.

    Character: Blank · From a concept · From a portrait (Actor Pack); Persona:
    Blank · From a portrait; Lore book; Chat dictionary (spec 1.4.1). The rows
    are the same in every kind, so the chooser never reshuffles. Lore books and
    chat dictionaries keep dev's immediate create until B9a (spec 5.3).

    Args:
        mode: The active workbench mode; its Blank is preselected.
        local_characters: Whether local character seams own the screen (False
            while browsing server characters).
        local_runtime: Whether personas run locally (a persona's portrait is
            picked from local characters).

    Returns:
        The rows, exactly one preselected and enabled.
    """
    sep = resolve_glyph(_SEPARATOR)
    rows = [
        ("character:blank", "characters", f"Character {sep} Blank", None),
        (
            "character:concept",
            "characters",
            f"Character {sep} From a concept",
            "describe it, then generate the card",
        ),
        (
            "character:portrait",
            "characters",
            f"Character {sep} From a portrait (Actor Pack)",
            None,
        ),
        ("persona:blank", "personas", f"Persona {sep} Blank", None),
        (
            "persona:portrait",
            "personas",
            f"Persona {sep} From a portrait (Actor Pack)",
            None,
        ),
        ("lore", "lore", "Lore book", "created at once"),
        ("dictionary", "dictionaries", "Chat dictionary", "created at once"),
    ]

    def reason_for(choice_id: str) -> str | None:
        if choice_id.startswith("character:") and not local_characters:
            return SERVER_CHARACTERS_READ_ONLY
        if choice_id == "persona:portrait" and not local_runtime:
            return PORTRAIT_NEEDS_LOCAL
        return None

    reasons = {choice_id: reason_for(choice_id) for choice_id, *_ in rows}
    wanted = NEW_BLANK_FOR_MODE.get(mode, "character:blank")
    if reasons.get(wanted) is not None:
        # The current Blank is off (server characters): the first row that runs.
        wanted = next(choice_id for choice_id, *_ in rows if reasons[choice_id] is None)
    return tuple(
        NewChoice(
            choice_id=choice_id,
            mode=row_mode,
            label=label,
            hint=hint,
            enabled=reasons[choice_id] is None,
            reason=reasons[choice_id],
            preselected=choice_id == wanted,
        )
        for choice_id, row_mode, label, hint in rows
    )


# ---- Where an import lands (frame slice B2b, spec 1.4.6) ----

#: The workbench mode that lists each kind (``KIND_FOR_MODE`` reversed).
MODE_FOR_KIND: dict[str, str] = {kind: mode for mode, kind in KIND_FOR_MODE.items()}


def import_landing(
    *, clean: bool, intent_current: bool, presentation_current: bool
) -> Literal["select", "toast"]:
    """Whether an import that landed is selected, or only named in a toast.

    Quick import is kept: the new item is selected (in its own kind, which
    the screen switches to) only when nothing can be lost or overridden.

    Args:
        clean: Every Roleplay draft is clean (R24's aggregate predicate).
        intent_current: No newer intent (a kind switch, a create, a load, a
            cursor move) arrived since the picker opened.
        presentation_current: The screen still shows (attached, on top).

    Returns:
        ``"select"`` when all three hold, else ``"toast"``.
    """
    if clean and intent_current and presentation_current:
        return "select"
    return "toast"


def import_landed_toast(name: str, kind: str, *, existed: bool = False) -> str:
    """R33's literal copy for where an import went: ``Imported <name> · in <Kind>``.

    The screen paints it with markup off, so ``name`` goes in as typed (on
    one line).

    Args:
        name: The imported item's name.
        kind: Its entity kind (``ITEM_KINDS`` keys).
        existed: A character card that was already in the library (the
            importer returned the existing one).

    Returns:
        The toast text.
    """
    head = (
        f"{_one_line(name)} already existed"
        if existed
        else f"Imported {_one_line(name)}"
    )
    return f"{head} {resolve_glyph(_SEPARATOR)} in {ITEM_KINDS[kind][2]}"
