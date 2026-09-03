"""Collections extraction series: the state object and controller are screen-wired.

Wave-2 Task 5 (state PR -- collections series 1/3) and Task 6 (controller
PR -- collections series 2/3; recipe:
backlog/docs/library-decomposition-recipe.md §13; export series precedent:
Tests/Architecture/test_library_export_wiring.py). Task 5's
`test_state_object_fields_match_the_shim_surface` pins that every
LibraryCollectionsState field has a matching generated property shim on
LibraryScreen under the `_library_collections_<field>` name -- the single
prefix every Collections field uses (unlike Conversations, no field needed
a different prefix variant). Task 6 adds the full-cluster/same-name-
delegator-forwarding shape, mirroring the export wiring test's own
Task 3-era additions exactly (minus a `_safe_text` binding test -- no
moved Collections body calls `self._safe_text(...)`).
"""
from __future__ import annotations

import dataclasses
import inspect
import re

import pytest

from tldw_chatbook.UI.Library_Modules.library_collections_state import (
    LibraryCollectionsState,
)


@pytest.mark.unit
def test_state_object_fields_match_the_shim_surface() -> None:
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    field_names = {f.name for f in dataclasses.fields(LibraryCollectionsState)}
    assert field_names, "state object is empty"
    for name in field_names:
        shim = getattr(LibraryScreen, "_library_collections_" + name, None)
        assert isinstance(shim, property), (
            f"no screen shim property found for state field {name!r}"
        )


#: Every method Task 6 moved into `LibraryCollectionsController`, under its
#: original `LibraryScreen` name. Derived from a full `ast` census of every
#: `LibraryScreen` method whose name contains "collection" (67 methods,
#: matching Task 5's own census) followed by reading each candidate's body
#: -- NOT a prefix/substring shortcut (the recipe's own documented trap).
#: 3 of the 67 belong to a DIFFERENT feature (Prompts' own "Prompt
#: Collections" grouping, unrelated to this capture-reader cluster) --
#: see `library_collections_controller.py`'s module docstring for the full
#: per-name reasoning. All other 64 move onto this controller -- no
#: `@work` framework-decorator hazard, no unbound-fake-self/silent-Mock
#: test bypass, and no class-level/instance-attribute monkeypatch was
#: found for any of the 64 (confirmed by running this task's own
#: verification battery, not assumed).
_COLLECTIONS_CLUSTER_METHOD_NAMES: tuple[str, ...] = (
    "_sync_library_collections_reader_layout_from_shell",
    "_mirror_library_collections_reader_preference",
    "_restore_library_collections_page",
    "_library_collections_capture_presentation",
    "_library_collections_capture_request",
    "_refresh_library_collections_capture_reader",
    "_load_library_collections_capture_entry",
    "_ensure_library_collections_capture_controller",
    "_run_library_collections_capture_transition",
    "_notify_library_collections_warning",
    "select_library_collection_capture",
    "_select_library_collection_capture",
    "select_library_collection_capture_scope",
    "filter_library_collection_captures",
    "toggle_library_collection_quick_capture",
    "_capture_library_collection_quick_capture_draft",
    "retain_library_collection_quick_capture_input",
    "retain_library_collection_quick_capture_note",
    "_reset_library_collection_quick_capture_draft",
    "cancel_library_collection_quick_capture",
    "save_library_collection_quick_capture",
    "retry_library_collection_quick_capture",
    "cancel_library_collection_quick_capture_retry",
    "refresh_library_collection_quick_capture",
    "_submit_library_collection_quick_capture",
    "toggle_library_collection_capture_filters",
    "_library_collection_capture_filter_request",
    "_apply_library_collection_capture_request",
    "apply_library_collection_capture_filters",
    "clear_library_collection_capture_filters",
    "cycle_library_collection_capture_sort",
    "_page_library_collection_captures",
    "previous_library_collection_captures",
    "next_library_collection_captures",
    "retry_library_collection_captures",
    "retry_library_collection_capture_detail",
    "set_library_collection_capture_mode",
    "toggle_library_collection_capture_more",
    "inspect_library_collection_legacy_recovery",
    "close_library_collection_legacy_recovery",
    "choose_library_collection_legacy_recovery_export",
    "_export_library_collection_legacy_recovery",
    "_update_selected_library_collection_capture",
    "_library_collection_loaded_capture",
    "_library_collection_capture_is_current",
    "_load_library_collection_capture_highlights",
    "save_library_collection_capture_highlight",
    "delete_library_collection_capture_highlight",
    "save_library_collection_capture_note",
    "link_library_collection_capture_note",
    "unlink_library_collection_capture_note",
    "_run_library_collection_capture_content_action",
    "summarize_library_collection_capture",
    "listen_to_library_collection_capture",
    "save_library_collection_capture_offline",
    "mark_library_collection_capture_read",
    "favorite_library_collection_capture",
    "archive_library_collection_capture",
    "undo_library_collection_capture_archive",
    "retry_library_collection_capture_extraction",
    "open_library_collection_capture_original",
    "arm_library_collection_capture_hard_delete",
    "cancel_library_collection_capture_hard_delete",
    "confirm_library_collection_capture_hard_delete",
)

#: The 1 name above that is a `@staticmethod` on `LibraryScreen`. Its
#: delegator forwards straight to the module-level `LibraryCollectionsController`
#: CLASS (per the conversations/export wiring tests' "static-method
#: delegator pattern" precedent), not through `self._collections_controller`.
_COLLECTIONS_CLUSTER_STATICMETHOD_NAMES: frozenset[str] = frozenset(
    {
        "_restore_library_collections_page",
    }
)


@pytest.mark.unit
def test_collections_controller_owns_its_cluster() -> None:
    """Every one of the 64 moved names is a callable on the controller.

    Covers the whole cluster, not a hand-picked sample -- mirrors
    `test_export_controller_owns_its_cluster`.
    """
    from tldw_chatbook.UI.Library_Modules.library_collections_controller import (
        LibraryCollectionsController,
    )

    missing = [
        name
        for name in _COLLECTIONS_CLUSTER_METHOD_NAMES
        if not callable(getattr(LibraryCollectionsController, name, None))
    ]
    assert not missing, f"LibraryCollectionsController is missing: {missing!r}"


@pytest.mark.unit
def test_screen_delegates_collections_handlers() -> None:
    """Every one of the 64 moved names is a one-line screen delegator that
    forwards to the SAME-NAMED controller method (or, for the 1
    staticmethod, to the module-level controller CLASS).

    Mirrors `test_screen_delegates_export_handlers`: a same-name forwarding
    check, not a loose "the controller is referenced somewhere" substring
    check.
    """
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
    from tldw_chatbook.UI.Library_Modules.library_collections_controller import (
        LibraryCollectionsController,
    )

    not_delegators = []
    for name in _COLLECTIONS_CLUSTER_METHOD_NAMES:
        method = getattr(LibraryScreen, name, None)
        if method is None:
            not_delegators.append(f"{name!r} (missing entirely)")
            continue
        src = inspect.getsource(method)
        escaped = re.escape(name)
        if not re.search(
            rf"_collections_controller\.{escaped}\(", src
        ) and not re.search(rf"LibraryCollectionsController\.{escaped}\(", src):
            not_delegators.append(name)
    assert not not_delegators, f"not delegators yet: {not_delegators!r}"


@pytest.mark.unit
def test_collections_cluster_staticmethods_forward_to_the_controller_class() -> None:
    """The 1 staticmethod name in the cluster forwards to the CLASS, not an instance."""
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    not_class_forwarding = []
    for name in _COLLECTIONS_CLUSTER_STATICMETHOD_NAMES:
        src = inspect.getsource(getattr(LibraryScreen, name))
        if not re.search(rf"LibraryCollectionsController\.{re.escape(name)}\(", src):
            not_class_forwarding.append(name)
    assert not not_class_forwarding, (
        f"expected class-forwarding delegators: {not_class_forwarding!r}"
    )


@pytest.mark.unit
def test_collections_controller_exposes_every_state_field() -> None:
    """The controller's generated shim loop covers every state field.

    Mirrors `test_export_controller_exposes_every_state_field`. Collections
    uses a single `_library_collections_` prefix for every field (task 5's
    report: no field needed a plural variant), so unlike Conversations
    there is no prefix branch to check.
    """
    from tldw_chatbook.UI.Library_Modules.library_collections_controller import (
        LibraryCollectionsController,
    )

    field_names = {f.name for f in dataclasses.fields(LibraryCollectionsState)}
    assert field_names, "state object is empty"
    missing = [
        name
        for name in field_names
        if not isinstance(
            getattr(LibraryCollectionsController, "_library_collections_" + name, None),
            property,
        )
    ]
    assert not missing, (
        f"no collections controller shim property found for state field(s): {missing!r}"
    )
