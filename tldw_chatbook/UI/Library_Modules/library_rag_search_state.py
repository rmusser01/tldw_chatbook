"""``LibraryRagSearchState`` -- the combined Search+RAG subsystem's own fields.

State PR of the combined Search+RAG extraction series (wave-3 task 2, recipe:
``backlog/docs/library-decomposition-recipe.md``; export/collections series
are the worked examples this mirrors). Every field here was moved verbatim
out of ``LibraryScreen.__init__`` in ``tldw_chatbook/UI/Screens/library_screen.py``
-- same default, same type. ``library_screen.py`` originally kept every
original ``_library_rag_<field>``/``_library_search_<field>`` attribute name
alive as a generated getter/setter ``@property`` shim pointing at
``self._rag_search_state.<field>`` (a sentinel-wrapped block right after the
``LibraryScreen`` class body). A future controller PR (wave-3 task 3) will
delete that screen-side shim block once the subsystem's methods have all
moved to a controller and the screen's own remaining references have been
retargeted; that controller will carry its OWN generated shim block in its
place, exactly the export/collections precedent.

Why ONE combined state object, not two (search-only + rag-only): wave-2
task 8's entanglement census (57.1% of the 14 search-cluster candidates
cross-call RAG-named methods; the top search bar's submit path *is* the RAG
query entry point) already forced the combined series decision at the
METHOD level. This task's own field-level census confirms the same
conclusion holds at the FIELD level and goes further: ``_library_rag_panel_
state`` (the single presentation-object builder every render path funnels
through) reads across literally all 20 fields in one call --
``_library_rag_scope_deselected``/``_library_loaded`` gate the scope
selection, ``_library_rag_query``/``_library_rag_searched_query``/``_library_
rag_mode``/``_library_rag_results``/``_library_rag_selected_result_id``/
``_library_rag_retrieval_status``/``_library_rag_answer_in_flight`` feed the
retrieval half, and a sibling read of the answer fields happens in the same
builder's continuation. There is no field subset one half of a two-object
split could own without the OTHER half's builder reaching across a
controller boundary for it on every single render. The plan's own
hypothetical clean seam ("rag-answer pipeline vs search/history surface")
does not hold under this evidence: even the 6 fields used ONLY by
``_library_rag_answer*``-named methods (no search-cluster-named caller) are
still read by ``_library_rag_panel_state`` in the SAME call that reads the
retrieval/history fields. See the task-2 report's ownership table for the
full per-field consumer breakdown that established this.

Every field's ``__init__`` assignment sat in ONE contiguous, unentangled
block (`library_screen.py` lines 2146-2243 at this task's own measurement) --
unlike the conversations/collections exemplars' entangled reader-preferences
trios, nothing here interleaves with another subsystem's initialization
code, so no field needed the "keep the original __init__ line untouched"
accommodation those two subsystems required.

Every field uses the ``_library_rag_`` prefix except ONE:
``_library_search_history`` (dataclass field name ``history``, since the
alternate prefix ``_library_search_`` already carries the word "search").
``SEARCH_PREFIXED_STATE_FIELDS`` below is the single authoritative home for
that one-field exception, so a future controller's own generated shim block
can import it rather than keeping an independent copy -- the conversations
exemplar's own ``CONVERSATIONS_PLURAL_STATE_FIELDS`` lesson (task 8 fix
round 1: two independent copies of the same two-name set drifted silently)
applied here from the start instead of being rediscovered.

One field, ``history``, has a genuinely computed (not static-literal)
default in the original code -- ``self._load_library_search_history()``, a
same-subsystem method (one of the 14 search-cluster candidate methods, not
an entangled cross-subsystem call). Per the recipe's "computed defaults
become constructor arguments" rule, ``__init__`` still calls that method (at
the exact position the removed ``self._library_search_history = ...`` line
occupied) and passes the result into the ``LibraryRagSearchState(...)``
constructor call; this dataclass's own ``history`` default (an empty tuple,
via ``default_factory``) is therefore a momentary placeholder, identical in
spirit to the export state object's own ``form`` field.
"""
from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ...Library.library_rag_answer_service import LibraryRagAnswer
from ...Library.library_rag_state import LibraryRagResultRow
from ..destination_recovery import DestinationRecoveryState

#: See the module docstring's note on this constant: the single
#: authoritative home for which field names use the ``_library_search_``
#: shim prefix instead of the cluster's default ``_library_rag_`` prefix.
SEARCH_PREFIXED_STATE_FIELDS: frozenset[str] = frozenset({"history"})


@dataclass
class LibraryRagSearchState:
    """Every field the combined Search+RAG subsystem exclusively owns."""

    mode: str = "search"
    query: str = ""
    results: tuple[LibraryRagResultRow, ...] = ()
    retrieval_status: str = ""
    recovery_state: DestinationRecoveryState | None = None
    selected_result_id: str = ""
    # Task 8: the current results' non-result-shaped retrieval
    # diagnostics (e.g. `semantic_scope_coverage`) -- travels with
    # `_library_rag_results` through every reset/outcome/save-restore
    # path below so the coverage note built from it can never drift
    # from the results it describes.
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    # task-15 finding I3: the query the CURRENT `_library_rag_results`/
    # `_library_rag_retrieval_status` were actually retrieved for --
    # travels with them through every reset/outcome/save-restore path
    # exactly like `_library_rag_diagnostics` above, so the quiet
    # no-match line (`library_rag_empty_state_quiet_copy`) can never
    # quote query text the "empty" outcome it explains wasn't actually
    # run against.
    searched_query: str = ""
    # PR-3 Task 4: the grounded answer generated for the CURRENT results
    # (`generate_library_rag_answer`), plus the query/mode it was
    # generated for. Those two are the answer worker's staleness guards,
    # mirroring `_apply_library_rag_search_outcome`'s query/mode guards
    # for retrieval: every transition that invalidates an answer (a new
    # search, a mode toggle) clears them, so an answer that lands
    # afterwards is discarded instead of overwriting a newer panel.
    answer: LibraryRagAnswer | None = None
    answer_query: str = ""
    answer_mode: str = ""
    # Whether the single provider call is running right now. Kept
    # SEPARATE from `_library_rag_retrieval_status` (rather than
    # overwriting it with "answering") so the retrieval's own settled
    # status is never destroyed and settling the answer is just clearing
    # a flag -- there is no "restore the right previous value" step to
    # get wrong. Deliberately NOT persisted by `save_state`: a restored
    # screen is a new instance with no worker running, so a restored
    # "answering" status could never be resolved by anything.
    answer_in_flight: bool = False
    # The provider `resolve_library_rag_answer_provider` resolved for
    # the answer call `_library_rag_answer_in_flight` is currently
    # tracking (PR-3 Task 3) -- feeds the in-flight "Asking
    # <provider>..." line. Cleared everywhere the flag above is
    # cleared; the panel-state builder additionally only ever forwards
    # this value while the flag is True, so a value left behind by a
    # reset gap could never surface on its own.
    answer_in_flight_provider: str = ""
    # What the Answer region currently shows, as
    # `(mode, is_answering, answer_object)` -- the three inputs
    # `library_rag_answer_children` reads. `_refresh_library_rag_answer_
    # widgets` skips its teardown/remount when this still matches, which
    # is what keeps a landed answer (up to
    # `LIBRARY_RAG_ANSWER_DISPLAY_MAX_LENGTH` characters of `Static`)
    # from being remounted on every keystroke in rag mode -- the exact
    # churn class task-284 removed for results/history. `None` means
    # "unknown, rebuild" and is set on every compose.
    answer_render_key: tuple[str, bool, Any] | None = None
    # B2: source types the user has toggled OFF (deselected) in the
    # scope region. Empty = every available source is in scope (the
    # default). Persists across rail switches within the session, same
    # as mode, but is never written to config.
    scope_deselected: set[str] = field(default_factory=set)
    # D1: whether the `Recent searches` collapsible should render
    # collapsed. Only `_apply_library_rag_search_outcome` (the
    # results-arrival transition) is allowed to change this; every
    # other refresh must leave the user's manual expand/collapse alone.
    history_collapsed: bool = False

    # Placeholder default only -- see module docstring: the original
    # `__init__` line's `self._load_library_search_history()` call still
    # runs, at the position of this field's removed assignment, and is
    # passed into the state constructor call directly. Uses the
    # `_library_search_` shim prefix (see `SEARCH_PREFIXED_STATE_FIELDS`
    # above), not the cluster's default `_library_rag_` prefix.
    history: tuple[str, ...] = field(default_factory=tuple)

    # Serializes history-collapsible content rebuilds: the "searching"
    # status refresh (called synchronously before the search worker is
    # scheduled) and that worker's own "outcome" refresh can otherwise
    # interleave mid-rebuild and mount duplicate row IDs.
    history_refresh_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Serializes whole-panel refreshes (PR-3 Task 4). Two-phase answering
    # made an always-latent race reachable on ordinary input: the
    # retrieval outcome's refresh and the answer's own refresh can now be
    # in flight at once (the no-evidence path resolves generation almost
    # immediately after retrieval), and both tear down and remount the
    # same fixed-id widgets -- each captures its removal list, awaits,
    # and then mounts, so interleaving raises `DuplicateIds`
    # (`library-rag-query-quiet-line`, observed). One lock around the
    # whole sequence means a refresh always finishes before the next
    # starts, and the next then rebuilds from state that is by then
    # settled.
    panel_refresh_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # (task-2075 D5) Cache of the last `library_rag_scope_shows_recovery`
    # result actually mirrored into the DOM, read by
    # `_sync_library_rag_scope_toggle_and_run_gate_widgets` to change-gate
    # the recovery-block mirror it schedules. `None` until the first
    # in-place snapshot sync runs -- deliberately distinct from both
    # `True`/`False` so that first call always reconciles the DOM
    # against whatever `compose()` actually rendered (cheap: at most an
    # empty remove + empty mount when nothing needs to change), while
    # every later snapshot with an unchanged value takes the no-op path
    # RAG-27 requires.
    scope_recovery_visible: bool | None = None
