"""Pure list-window fences and the single Roleplay cursor/activation intent owner."""

from __future__ import annotations

from collections.abc import Collection, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace


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
    """

    def __init__(self) -> None:
        self.generation = 0
        self.rows_fence: ListFence | None = None
        self.more_available = False
        self.requested = False
        self.intent: WindowIntent | None = None
        self.reload_on_return = False
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
