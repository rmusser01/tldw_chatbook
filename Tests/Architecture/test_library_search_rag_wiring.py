"""Combined Search+RAG extraction series: state object and controller are screen-wired.

Wave-3 Task 2 (state PR -- search+RAG series 1/3) and Task 3 (controller PR
-- search+RAG series 2/3; recipe: backlog/docs/library-decomposition-recipe.md;
export/collections series precedent: Tests/Architecture/
test_library_export_wiring.py / test_library_collections_wiring.py; the
conversations exemplar's own test_library_conversations_wiring.py is the
precedent for a state object whose fields split across TWO shim prefixes).

Task 2's ``test_state_object_fields_match_the_shim_surface`` (every
``LibraryRagSearchState`` field <-> a matching generated property shim on
``LibraryScreen``) is unchanged below -- it stays green through the
controller move (the screen's own shim block is a Task-4 cleanup deletion,
not a Task-3 concern, per the conversations/export/collections precedent).
Task 3 adds the full-cluster ownership/same-name-delegator-forwarding checks
(``_RAG_SEARCH_CLUSTER_METHOD_NAMES``, 42 names) and the controller-side
state-field-coverage check, mirroring ``test_library_collections_wiring.py``
exactly.
"""
from __future__ import annotations

import dataclasses
import inspect
import re

import pytest

from tldw_chatbook.UI.Library_Modules.library_rag_search_state import (
    SEARCH_PREFIXED_STATE_FIELDS,
    LibraryRagSearchState,
)


@pytest.mark.unit
def test_state_object_fields_match_the_shim_surface() -> None:
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    field_names = {f.name for f in dataclasses.fields(LibraryRagSearchState)}
    assert field_names, "state object is empty"
    missing = []
    for name in field_names:
        prefix = (
            "_library_search_"
            if name in SEARCH_PREFIXED_STATE_FIELDS
            else "_library_rag_"
        )
        shim = getattr(LibraryScreen, prefix + name, None)
        if not isinstance(shim, property):
            missing.append(prefix + name)
    assert not missing, f"no screen shim property found for: {missing!r}"


@pytest.mark.unit
def test_search_prefixed_state_fields_are_real_state_fields() -> None:
    """Guards the single-source prefix mapping against drift: every name it
    lists must actually be a `LibraryRagSearchState` field (a typo or a
    stale entry here would otherwise silently shim nothing under the
    intended prefix and everything under the wrong one instead).
    """
    field_names = {f.name for f in dataclasses.fields(LibraryRagSearchState)}
    unknown = SEARCH_PREFIXED_STATE_FIELDS - field_names
    assert not unknown, f"SEARCH_PREFIXED_STATE_FIELDS names unknown fields: {unknown!r}"


#: Every method Task 3 moved into `LibraryRagSearchController`, under its
#: original `LibraryScreen` name. Derived from a full `ast` census of every
#: `LibraryScreen` method whose name contains "search" or "rag" (60 raw
#: matches, matching Task 2's own census), minus 3 Prompts-owned + 7
#: Media-owned (50 combined-cluster candidates), minus 3 `@work`-decorated
#: framework-decorator-hazard methods, 1 module-globals-coupling exclusion
#: (`_load_library_search_history`), and 4 test-bypass (instance-attribute
#: monkeypatch) exclusions -- NOT a prefix/substring shortcut. See
#: `library_rag_search_controller.py`'s module docstring for the full
#: per-name reasoning behind every exclusion.
_RAG_SEARCH_CLUSTER_METHOD_NAMES: tuple[str, ...] = (
    "_apply_library_rag_answer",
    "_apply_library_rag_scope_recovery_block",
    "_apply_library_rag_search_outcome",
    "_focus_library_search_input",
    "_focused_library_rag_result_card_index",
    "_library_rag_answer_chat_kwargs",
    "_library_rag_scope_summary",
    "_library_rail_search_placeholder",
    "_open_library_rag_result_by_index",
    "_persist_library_search_history",
    "_record_library_search_history",
    "_refresh_library_rag_answer_widgets",
    "_refresh_library_rag_history_widget",
    "_refresh_library_rag_query_status_widgets",
    "_refresh_library_rag_results_widgets",
    "_reset_library_rag_answer_state",
    "_reset_library_rag_in_flight_status",
    "_reset_library_rag_retrieval_state",
    "_reveal_library_rag_results",
    "_select_library_rag_result_by_index",
    "_stage_library_rag_result_in_console",
    "_start_library_rag_answer",
    "_start_library_rag_query",
    "_sync_library_rag_scope_toggle_and_run_gate_widgets",
    "_use_library_rag_result_in_console",
    "action_library_rag_result_card_open",
    "action_library_rag_result_card_select",
    "action_library_rag_use_in_console",
    "clear_library_search_history",
    "cycle_library_rag_mode",
    "handle_library_search_changed",
    "handle_library_search_submitted",
    "open_import_export_from_library_rag",
    "open_library_rag_result",
    "rerun_library_search_from_history",
    "run_library_rag_query",
    "select_library_rag_result",
    "submit_library_rag_query",
    "sync_library_rag_history_collapsed",
    "toggle_library_rag_scope_source",
    "update_library_rag_query",
    "use_selected_library_rag_result_in_console",
)

#: The 1 name above that is a `@staticmethod` on `LibraryScreen`. Its
#: delegator forwards straight to the module-level `LibraryRagSearchController`
#: CLASS (per the conversations/export/collections wiring tests' "static-
#: method delegator pattern" precedent), not through
#: `self._rag_search_controller`.
_RAG_SEARCH_CLUSTER_STATICMETHOD_NAMES: frozenset[str] = frozenset(
    {
        "_library_rag_scope_summary",
    }
)


@pytest.mark.unit
def test_rag_search_controller_owns_its_cluster() -> None:
    """Every one of the 42 moved names is a callable on the controller.

    Covers the whole cluster, not a hand-picked sample -- mirrors
    `test_collections_controller_owns_its_cluster`.
    """
    from tldw_chatbook.UI.Library_Modules.library_rag_search_controller import (
        LibraryRagSearchController,
    )

    missing = [
        name
        for name in _RAG_SEARCH_CLUSTER_METHOD_NAMES
        if not callable(getattr(LibraryRagSearchController, name, None))
    ]
    assert not missing, f"LibraryRagSearchController is missing: {missing!r}"


@pytest.mark.unit
def test_screen_delegates_rag_search_handlers() -> None:
    """Every one of the 42 moved names is a one-line screen delegator that
    forwards to the SAME-NAMED controller method (or, for the 1
    staticmethod, to the module-level controller CLASS).

    Mirrors `test_screen_delegates_collections_handlers`: a same-name
    forwarding check, not a loose "the controller is referenced somewhere"
    substring check.
    """
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    not_delegators = []
    for name in _RAG_SEARCH_CLUSTER_METHOD_NAMES:
        method = getattr(LibraryScreen, name, None)
        if method is None:
            not_delegators.append(f"{name!r} (missing entirely)")
            continue
        src = inspect.getsource(method)
        escaped = re.escape(name)
        if not re.search(
            rf"_rag_search_controller\.{escaped}\(", src
        ) and not re.search(rf"LibraryRagSearchController\.{escaped}\(", src):
            not_delegators.append(name)
    assert not not_delegators, f"not delegators yet: {not_delegators!r}"


@pytest.mark.unit
def test_rag_search_cluster_staticmethods_forward_to_the_controller_class() -> None:
    """The 1 staticmethod name in the cluster forwards to the CLASS, not an instance."""
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    not_class_forwarding = []
    for name in _RAG_SEARCH_CLUSTER_STATICMETHOD_NAMES:
        src = inspect.getsource(getattr(LibraryScreen, name))
        if not re.search(rf"LibraryRagSearchController\.{re.escape(name)}\(", src):
            not_class_forwarding.append(name)
    assert not not_class_forwarding, (
        f"expected class-forwarding delegators: {not_class_forwarding!r}"
    )


@pytest.mark.unit
def test_rag_search_controller_exposes_every_state_field() -> None:
    """The controller's generated shim loop covers every state field.

    Mirrors `test_collections_controller_exposes_every_state_field`, but
    checks BOTH prefixes (this cluster, like the conversations exemplar,
    splits its fields across `_library_rag_`/`_library_search_`).
    """
    from tldw_chatbook.UI.Library_Modules.library_rag_search_controller import (
        LibraryRagSearchController,
    )

    field_names = {f.name for f in dataclasses.fields(LibraryRagSearchState)}
    assert field_names, "state object is empty"
    missing = []
    for name in field_names:
        prefix = (
            "_library_search_"
            if name in SEARCH_PREFIXED_STATE_FIELDS
            else "_library_rag_"
        )
        if not isinstance(
            getattr(LibraryRagSearchController, prefix + name, None), property
        ):
            missing.append(prefix + name)
    assert not missing, (
        f"no rag+search controller shim property found for state field(s): {missing!r}"
    )
