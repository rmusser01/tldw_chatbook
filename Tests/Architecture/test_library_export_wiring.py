"""Export extraction series: state object + controller are screen-wired.

Wave-2 Task 2 (state PR), Task 3 (controller PR), and Task 4 (cleanup PR --
export series 3/3; recipe: backlog/docs/library-decomposition-recipe.md;
conversations series precedent: Tests/Architecture/test_library_conversations_
wiring.py). Task 2's own screen-shim assertion
(`test_state_object_fields_match_the_shim_surface`) is GONE as of Task 4:
the screen's generated `_library_export_<field>` property shim block was
deleted wholesale in cleanup (`self._export_state` is a real
`LibraryExportState` instance now, not a shimmed screen attribute), so
there is nothing left on `LibraryScreen` for that assertion to check --
exactly the conversations exemplar's own Task 9 precedent (see that
controller module's `test_reader_controller_exposes_every_state_field`
docstring). `test_export_controller_exposes_every_state_field` below
already covers the equivalent job on the controller side and needed no
change. Task 3's full-cluster controller-ownership and same-name-
delegator-forwarding checks are unchanged in shape; Task 4 adds the
`_EXPORT_CLUSTER_SCREEN_DELEGATOR_PRUNED` skip/absence-assertion pair to
`test_screen_delegates_export_handlers`, mirroring
`_BROWSE_CLUSTER_SCREEN_DELEGATOR_PRUNED` in the conversations wiring test.
"""
from __future__ import annotations

import dataclasses
import inspect
import re

import pytest

from tldw_chatbook.UI.Library_Modules.library_export_state import (
    LibraryExportState,
)


#: Every method Task 3 moved into `LibraryExportController`, under its
#: original `LibraryScreen` name. Derived from a full `ast` census of every
#: `LibraryScreen` method whose name contains "export" (51 methods, matching
#: Task 2's own census) followed by reading each candidate's body -- NOT a
#: prefix/substring shortcut (the recipe's own documented trap). Only 22 of
#: the 51 are genuinely Export-owned AND move; the other 29 are excluded
#: (see `library_export_controller.py`'s module docstring for the full,
#: per-name reasoning):
#:
#: - 18 belong to a DIFFERENT subsystem (Notes/Prompts/Media/Conversations/
#:   Collections/Search-RAG) despite the name match -- verified by reading
#:   what state each body actually touches, not by name.
#: - 2 more (`_run_library_export_counts_worker`, `_run_library_export_worker`)
#:   are genuinely Export-owned but stay on `LibraryScreen`, UNMOVED: both
#:   carry `@work(thread=True, ...)`, whose decorator asserts
#:   `isinstance(self, DOMNode)` at call time -- a plain controller instance
#:   would fail that assertion (the module docstring's "framework-decorator
#:   self-type assertion" exclusion note).
#: - 9 more, found only by running this task's own verification battery
#:   (not static analysis, and not `Tests/UI` alone -- four of the nine
#:   surfaced only once the sweep was widened to `Tests/Library/`), are
#:   genuinely Export-owned but ALSO stay on `LibraryScreen`, UNMOVED:
#:   `_apply_library_export_cancelled`, `handle_library_export_cancel`,
#:   `_apply_library_export_progress`, `_apply_library_export_counts`,
#:   `_build_library_export_state`, `_update_library_export_canvas_after_run`,
#:   `_start_library_export_counts_worker`, `_start_library_export_worker`,
#:   `_apply_library_export_success` -- each reached by an unbound
#:   `LibraryScreen.<name>(fake, ...)` call (a `SimpleNamespace` lacking
#:   `_export_controller`, or -- for `_apply_library_export_success` alone --
#:   a `unittest.mock.Mock()`, whose auto-attribution silently swallows a
#:   delegator instead of raising) in one of six test files. All 9 were
#:   confirmed, via `git stash -u`, to PASS on a pristine baseline before
#:   being excluded (genuine regressions this move introduced, not
#:   pre-existing reds).
_EXPORT_CLUSTER_METHOD_NAMES: tuple[str, ...] = (
    "_default_library_export_form",
    "_reset_library_export_transient_state",
    "_open_library_export_canvas",
    "_library_export_is_server_mode",
    "_resolve_library_export_chachanotes_db",
    "_compute_library_export_counts",
    "handle_library_export_submit",
    "_build_library_export_payload",
    "_run_library_export_via_service",
    "_marshal_library_export_success",
    "_marshal_library_export_failure",
    "_marshal_library_export_cancelled",
    "_build_library_export_success_message",
    "_apply_library_export_failure",
    "_refresh_library_export_status_line",
    "action_library_export_back",
    "handle_library_export_name_changed",
    "handle_library_export_description_changed",
    "handle_library_export_quality",
    "handle_library_export_quality_choice",
    "handle_library_export_choose_destination",
    "_apply_library_export_destination",
)

#: Task 4 cleanup: `_library_export_is_server_mode` has ZERO references
#: anywhere except its own one-line screen delegator (checked via a
#: repo-wide census across `tldw_chatbook/` and `Tests/`) -- every call is
#: either the controller's own internal `self._library_export_is_server_
#: mode()` (twice, both moved-body-internal) or the now-deleted screen
#: delegator itself; nothing outside the controller ever called
#: `screen._library_export_is_server_mode()`. Its screen delegator was
#: deleted as dead weight; the name remains in
#: `_EXPORT_CLUSTER_METHOD_NAMES` above (the controller still genuinely
#: owns and uses it) but is excluded from the delegation-forwarding check
#: below, same shape as the conversations exemplar's own
#: `_BROWSE_CLUSTER_SCREEN_DELEGATOR_PRUNED` (Task 9).
_EXPORT_CLUSTER_SCREEN_DELEGATOR_PRUNED: frozenset[str] = frozenset(
    {
        "_library_export_is_server_mode",
    }
)

#: The 5 names above that are `@staticmethod`s on `LibraryScreen`. Their
#: delegators forward straight to the module-level `LibraryExportController`
#: CLASS (per task-8-report.md's "static-method delegator pattern"
#: correction, cited in the conversations wiring test), not through
#: `self._export_controller`.
_EXPORT_CLUSTER_STATICMETHOD_NAMES: frozenset[str] = frozenset(
    {
        "_default_library_export_form",
        "_compute_library_export_counts",
        "_build_library_export_payload",
        "_run_library_export_via_service",
        "_build_library_export_success_message",
    }
)


@pytest.mark.unit
def test_export_controller_owns_its_cluster() -> None:
    """Every one of the 22 moved names is a callable on the controller.

    Covers the whole cluster, not a hand-picked sample -- mirrors
    `test_browse_controller_owns_its_cluster` in the conversations wiring
    test.
    """
    from tldw_chatbook.UI.Library_Modules.library_export_controller import (
        LibraryExportController,
    )

    missing = [
        name
        for name in _EXPORT_CLUSTER_METHOD_NAMES
        if not callable(getattr(LibraryExportController, name, None))
    ]
    assert not missing, f"LibraryExportController is missing: {missing!r}"


@pytest.mark.unit
def test_screen_delegates_export_handlers() -> None:
    """Every one of the 22 moved names is a one-line screen delegator that
    forwards to the SAME-NAMED controller method (or, for the 5
    static/classmethods, to the module-level controller CLASS).

    Mirrors `test_screen_delegates_browse_handlers` in the conversations
    wiring test: a same-name forwarding check, not a loose "the controller
    is referenced somewhere" substring check.

    Skips `_EXPORT_CLUSTER_SCREEN_DELEGATOR_PRUNED` (Task 4 deleted that
    1 screen delegator as dead weight -- zero external references) and
    instead asserts that name is genuinely ABSENT from `LibraryScreen`, so
    a future accidental re-add would fail loudly here rather than silently
    reintroducing dead code.
    """
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
    from tldw_chatbook.UI.Library_Modules.library_export_controller import (
        LibraryExportController,
    )

    not_delegators = []
    for name in _EXPORT_CLUSTER_METHOD_NAMES:
        if name in _EXPORT_CLUSTER_SCREEN_DELEGATOR_PRUNED:
            assert getattr(LibraryScreen, name, None) is None, (
                f"{name!r} was pruned from the screen (task 4) but is back -- "
                "either wire it as a delegator again or drop it from "
                "_EXPORT_CLUSTER_SCREEN_DELEGATOR_PRUNED"
            )
            continue
        method = getattr(LibraryScreen, name, None)
        if method is None:
            not_delegators.append(f"{name!r} (missing entirely)")
            continue
        src = inspect.getsource(method)
        escaped = re.escape(name)
        if not re.search(rf"_export_controller\.{escaped}\(", src) and not re.search(
            rf"LibraryExportController\.{escaped}\(", src
        ):
            not_delegators.append(name)
    assert not not_delegators, f"not delegators yet: {not_delegators!r}"


@pytest.mark.unit
def test_export_cluster_staticmethods_forward_to_the_controller_class() -> None:
    """The 5 staticmethod names in the cluster forward to the CLASS, not an instance."""
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    not_class_forwarding = []
    for name in _EXPORT_CLUSTER_STATICMETHOD_NAMES:
        src = inspect.getsource(getattr(LibraryScreen, name))
        if not re.search(rf"LibraryExportController\.{re.escape(name)}\(", src):
            not_class_forwarding.append(name)
    assert not not_class_forwarding, (
        f"expected class-forwarding delegators: {not_class_forwarding!r}"
    )


@pytest.mark.unit
def test_export_controller_exposes_every_state_field() -> None:
    """The controller's generated shim loop covers every state field.

    Mirrors `test_browse_controller_exposes_every_state_field` in the
    conversations wiring test. Export uses a single `_library_export_`
    prefix for every field (task 2's report: no field needed a plural
    variant), so unlike Conversations there is no prefix branch to check.
    """
    from tldw_chatbook.UI.Library_Modules.library_export_controller import (
        LibraryExportController,
    )

    field_names = {f.name for f in dataclasses.fields(LibraryExportState)}
    assert field_names, "state object is empty"
    missing = [
        name
        for name in field_names
        if not isinstance(
            getattr(LibraryExportController, "_library_export_" + name, None),
            property,
        )
    ]
    assert not missing, (
        f"no export controller shim property found for state field(s): {missing!r}"
    )


@pytest.mark.unit
def test_export_controller_safe_text_is_bound_via_screen_import() -> None:
    """Importing `library_screen` installs `_safe_text` on the export controller.

    Mirrors `test_browse_controller_safe_text_is_bound_via_screen_import` in
    the conversations wiring test: `handle_library_export_submit` calls
    `self._safe_text(...)` on a regular instance method, so the SAME
    class-level rebinding shape (`LibraryExportController._safe_text =
    staticmethod(LibraryScreen._safe_text)`) is required here too.
    """
    import tldw_chatbook.UI.Screens.library_screen  # noqa: F401  (import side effect installs the binding)
    from tldw_chatbook.UI.Library_Modules.library_export_controller import (
        LibraryExportController,
    )

    bound = LibraryExportController.__dict__.get("_safe_text")
    assert isinstance(bound, staticmethod), (
        "LibraryExportController._safe_text must be the staticmethod "
        "installed by library_screen.py's trailing class-level rebinding -- "
        f"got {type(bound)!r}"
    )
    assert callable(LibraryExportController._safe_text), (
        "LibraryExportController._safe_text must be callable"
    )
