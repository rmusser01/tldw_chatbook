"""Pure Roleplay item projections, list-window fences, and cursor/activation state."""

from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Literal

from rich.cells import cell_len

from tldw_chatbook.Widgets.glyph_fallback import resolve_glyph


@dataclass(frozen=True)
class ListFence:
    """Everything a window must still match when it arrives (spec 1.5.6 "Window fence")."""

    mode: str
    query: str
    sort_key: str
    tag: str | None
    runtime_source: str
    server_id: str | None
    generation: int


@dataclass(frozen=True)
class ListAnchor:
    """Where the list stood before a reload from the anchor (spec 1.5.6 "Writes")."""

    cursor_id: str | None
    cursor_index: int | None
    top_index: int
    loaded: int


def window_wants_more(
    *,
    loaded: int,
    last_window: int,
    window_rows: int,
    total: int | None,
    target_index: int | None = None,
    target_found: bool = True,
    to_end: bool = False,
) -> bool:
    """Whether the window worker fetches another window before it applies (load-through).

    One window per request, unless the request moves to a row past the loaded
    end (an index, an id not yet seen, or the end), in which case windows load
    until the target is in. A short window means the list is exhausted.
    """
    if last_window < window_rows:
        return False
    if total is not None and loaded >= total:
        return False
    if to_end or not target_found:
        return True
    return target_index is not None and loaded <= target_index


def new_option_ids(
    existing: Collection[str], incoming: Sequence[str]
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Indexes of ``incoming`` to append, and the duplicate ids dropped (spec 1.5.6 "No duplicates").

    Textual 8.2.8 appends options one at a time and raises ``DuplicateID``
    part-way through a batch, so ``add_options`` must only ever see new ids.
    """
    seen = set(existing)
    keep: list[int] = []
    dropped: list[str] = []
    for index, option_id in enumerate(incoming):
        if option_id in seen:
            dropped.append(option_id)
            continue
        seen.add(option_id)
        keep.append(index)
    return tuple(keep), tuple(dropped)


@dataclass(frozen=True)
class ListWindowRequest:
    """One run of the window worker (spec 1.5.6).

    ``replace`` reloads from row 0 (``set_options``); otherwise the run
    appends after ``offset`` (``add_options``). ``anchor`` makes a reload load
    through every previously loaded row; ``target_id`` (an item id) and
    ``to_end`` make an append load through to that row or to the end.
    """

    fence: ListFence
    offset: int
    replace: bool
    anchor: ListAnchor | None = None
    target_id: str | None = None
    to_end: bool = False


def same_list(first: ListFence, second: ListFence) -> bool:
    """Whether two fences name the same list (mode, query, sort, tag, source), any generation."""
    return replace(first, generation=second.generation) == second


@dataclass(eq=False)
class WindowIntent:
    """A cursor move the items list owes the user once rows arrive (spec 1.5.6).

    ``reason`` is ``"end"`` (End past the loaded rows) or ``"id"`` (a row to
    load, select or restore; ``target_id`` is its option id) or ``"first"``
    (Enter in the search box: the cursor goes to the first match of the
    searched list once its rows land; ``activate`` retains a subsequent
    Enter until that publication, TASK-33910.35; ``fence`` is set when it
    is owed and no window is asked for it). ``read`` says a window was asked
    for it; ``fence`` is the list it belongs to, set when the screen sees
    that request (None until then). After publication, an activated first
    intent becomes ``"activate"`` until the matching selection is accepted.
    """

    reason: str
    target_id: str | None = None
    read: bool = False
    fence: ListFence | None = None
    activate: bool = False


class ListWindowState:
    """The items list's window state, with one owner (TASK-33910.34).

    The screen creates it and its items list shares it, so every reset is a
    transition here: a reload (``reload_started``, then ``landing``), a kind
    switch, a leave and the return. Plain state: no Textual, no I/O.

    Attributes:
        generation: Bumped by every reload and leave. It is part of each
            fence, so a read still in flight is dropped when it lands.
        rows_fence: The list the loaded rows belong to (None: no list's).
        more_available: Rows past the loaded end exist.
        requested: A window is asked for and not answered yet (the latch
            that asks once per approach to the loaded end).
        intent: The cursor move owed to the user, if any.
        reload_on_return: A leave found the list's rows not landed; the
            return reloads it.
        moves: How many cursor moves and selections the user made
            (``user_moved``). An import compares it with the count when its
            picker opened: a newer move keeps the cursor where the user put
            it (frame slice B2b, spec 1.4.6).
    """

    def __init__(self) -> None:
        self.generation = 0
        self.rows_fence: ListFence | None = None
        self.more_available = False
        self.requested = False
        self.intent: WindowIntent | None = None
        self.reload_on_return = False
        self.moves = 0
        self._painting = False

    # ---- the rows ----

    def reload_started(self) -> None:
        """A reload (or a source change) starts: reads in flight are fenced out."""
        self.generation += 1

    def rows_current(self, live: ListFence) -> bool:
        """Whether the loaded rows are ``live``'s list, painted: only then may an append extend them."""
        return self.rows_fence == live and not self._painting

    def landing(self, fence: ListFence) -> None:
        """A replace's rows are ``fence``'s list's rows now.

        A move owed on another list (another sort, search, tag or source) is
        dropped before they paint, so it never moves the cursor on this one;
        one owed on this list stays for ``take_replay``.
        """
        self.rows_fence = fence
        owed = self.intent.fence if self.intent is not None else None
        if owed is not None and not same_list(owed, fence):
            self.intent = None

    @contextmanager
    def painting(self) -> Iterator[None]:
        """While a replace paints, its rows are not current (a leave reloads on the return)."""
        self._painting = True
        try:
            yield
        finally:
            self._painting = False

    def forget_rows(self) -> None:
        """The loaded rows are no list's (a server reload's blank, or a paint the fence outran)."""
        self.rows_fence = None

    def set_more(self, more: bool) -> None:
        """Whether rows past the loaded end exist; the asked-for window is answered."""
        self.more_available = more
        self.requested = False

    def answered(self) -> None:
        """An appended window answered the request."""
        self.requested = False

    def read_failed(self) -> None:
        """A window read failed or could not start: the request is answered,
        and an End, a load-through or a first-match landing that waited for
        the read is owed no more."""
        self.requested = False
        intent = self.intent
        if intent is not None and (
            intent.read or intent.reason in {"first", "activate"}
        ):
            self.intent = None

    def kind_switched(self) -> None:
        """A new kind starts with no window state (B2a's final review, I1)."""
        self.rows_fence = None
        self.more_available = False
        self.requested = False
        self.intent = None

    # ---- the owed cursor move ----

    def ask(self) -> None:
        """One window is asked for (the cursor or scroll neared the loaded end)."""
        self.requested = True

    def want(
        self, reason: str, target_id: str | None = None, *, read: bool
    ) -> WindowIntent:
        """Owe the user a move (End or a load-through); ``read`` asks for a window too."""
        self.intent = WindowIntent(reason, target_id, read)
        if read:
            self.requested = True
        return self.intent

    def owe_first(self, fence: ListFence) -> None:
        """Enter in the search box: owe the cursor on ``fence``'s first match.

        A newer move than any End or load-through owed, which it replaces.
        No window is asked for: the searched list's own replace lands it.
        """
        self.intent = WindowIntent("first", fence=fence)

    def landing_owed(self) -> bool:
        """Whether the searched list still owes its first match before activation is safe."""
        return self.intent is not None and self.intent.reason == "first"

    def defer_first_activation(self) -> bool:
        """Retain one Enter until the searched rows publish; repeated Enter coalesces."""
        if not self.landing_owed():
            return False
        self.intent.activate = True
        return True

    def take_first(self, fence: ListFence) -> WindowIntent | None:
        """Take a matching landing, retaining an activation until its exact receipt."""
        intent = self.intent
        if (
            intent is None
            or intent.reason != "first"
            or intent.fence is None
            or not same_list(intent.fence, fence)
        ):
            return None
        if intent.activate:
            intent.reason = "activate"
            intent.fence = fence
        else:
            self.intent = None
        return intent

    def take_activation(
        self, intent: WindowIntent, fence: ListFence, target_id: str
    ) -> bool:
        """Accept only this owner's exact published activation; retire a stale receipt."""
        if self.intent is not intent:
            return False
        self.intent = None
        return (
            intent.reason == "activate"
            and intent.fence == fence
            and intent.target_id == target_id
        )

    def landed_elsewhere(self, fence: ListFence) -> None:
        """Another list's rows landed without a replace of this module's windows.

        A first-match landing owed on a list other than ``fence``'s can never
        land (its render was superseded), so it is dropped, as ``landing()``
        does for a character replace. ``rows_fence`` is not touched.
        """
        intent = self.intent
        if (
            intent is not None
            and intent.reason == "first"
            and intent.fence is not None
            and not same_list(intent.fence, fence)
        ):
            self.intent = None

    def asked_for(self, intent: WindowIntent | None, fence: ListFence) -> None:
        """The screen saw ``intent``'s request: it belongs to the list the user waits for."""
        if intent is not None and intent is self.intent:
            intent.fence = fence

    def user_moved(self) -> None:
        """A newer cursor move or selection: the owed move is no longer wanted."""
        self.intent = None
        self.moves += 1

    def arrived(self, loaded: Collection[str]) -> str | None:
        """The owed row's option id once it is loaded (the move is then met)."""
        intent = self.intent
        if intent is None or intent.reason != "id" or intent.target_id not in loaded:
            return None
        self.intent = None
        return intent.target_id

    def take_end(self) -> bool:
        """Whether End's load-through still owes the move to the last row (it is then met)."""
        if self.intent is None or self.intent.reason != "end":
            return False
        self.intent = None
        return True

    def take_replay(self, fence: ListFence) -> WindowIntent | None:
        """The move to ask for again once a replace of its list has landed.

        End or a load-through whose read that reload deferred or cancelled. A
        move with no read, or whose request the screen has not seen yet, stays.
        """
        intent = self.intent
        if intent is None or not intent.read or intent.fence is None:
            return None
        if not same_list(intent.fence, fence):
            return None
        self.intent = None
        return intent

    # ---- the visit ----

    def leave(self, live: ListFence) -> None:
        """The visit ends (spec 4.11): fence reads in flight, forget the owed move.

        Rows that were ``live``'s list, painted, stay usable after the return;
        otherwise (a replace was reading or painting) the return reloads.
        """
        current = self.rows_current(live)
        self.generation += 1
        if current:
            self.rows_fence = replace(live, generation=self.generation)
        else:
            self.reload_on_return = True
        self.intent = None
        self.requested = False

    def returned(self) -> bool:
        """The visit resumed: whether the list reloads (answered once)."""
        reload, self.reload_on_return = self.reload_on_return, False
        return reload


_ELLIPSIS = "…"

_SEPARATOR = "·"

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
    count: int, hidden: int, width: int, *, can_export: bool, filtered: bool = False
) -> MarksBarView | None:
    """The marks bar's text at ``width`` cells, or ``None`` while nothing is marked.

    Full form ``✓ 2 marked · Export… · Delete… · Clear esc``. When it does not
    fit the bar degrades in a fixed order: the separators and the ``esc`` hint
    go, then ``marked``, then the hidden count (the tooltip keeps the full
    count). Narrower still, whole actions drop and none is ever cut: Clear
    first (Escape on the list clears marks after any filter), then Export; then the
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
    long_labels = {
        "export": "Export…",
        "delete": "Delete…",
        "clear": "Clear" if filtered else "Clear esc",
    }
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


def imported_toast(name: str) -> str:
    """The plain copy for an import that was selected: ``<name> imported.``

    Where it went is named only when it was not selected
    (``import_landed_toast``); lore books and chat dictionaries said
    "Imported X · in Kind" either way until B2b's final fix wave. Painted
    with markup off, so ``name`` goes in as typed (on one line).

    Args:
        name: The imported item's name.

    Returns:
        The toast text.
    """
    return f"{_one_line(name)} imported."
