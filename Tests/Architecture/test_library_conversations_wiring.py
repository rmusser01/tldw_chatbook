"""Conversations extraction series: state object exists and is screen-wired."""
from __future__ import annotations

import pytest

from tldw_chatbook.UI.Library_Modules.library_conversations_state import (
    LibraryConversationsState,
)

#: Every method Task 7 moved into `LibraryConversationReaderController`, under
#: its original `LibraryScreen` name (including the five `@on`-decorated
#: handlers, which keep screen-side delegators). A module-level constant, not
#: inlined per-test, so Task 8's browse-controller wiring test can follow the
#: same shape: one full-cluster ownership test plus one full-cluster
#: delegator test, both driven off one authoritative name list.
#:
#: A prior version of this file asserted only 2 of these 21 names and one
#: delegator's source -- cheap enough to write, but it proved too narrow to
#: catch a real class of bug: task-7-report.md's "a real bug found and
#: fixed" section documents a missing import that produced a swallowed
#: `NameError` inside a method these 2 names never reached. Looping the full
#: cluster does not catch that exact bug (it was already fixed before this
#: strengthening landed), but it closes the same shape of gap for the other
#: 19 names and for any future controller in this series -- see
#: `test_reader_controller_exposes_every_state_field` below for the
#: plural-state-field variant of the same risk.
_READER_CLUSTER_METHOD_NAMES: tuple[str, ...] = (
    "_bootstrap_library_conversation_reader",
    "_conversation_reader_bootstrap_is_current",
    "_conversation_reader_list_summary",
    "_conversation_reader_record",
    "_conversation_reader_record_version",
    "_conversation_reader_request_is_current",
    "_conversation_reader_service",
    "_ensure_library_conversation_reader_selection",
    "_invalidate_library_conversation_reader_authority",
    "_load_library_conversation_reader",
    "_mirror_library_conversation_reader_preference",
    "_retry_library_conversation_reader",
    "_start_library_conversation_reader_selection",
    "_sync_library_conversation_reader",
    "_sync_library_conversation_reader_layout_from_shell",
    "library_conversation_reader_messages_synced",
    "retry_library_conversation_reader",
    "show_library_conversation_reader_info",
    "show_library_conversation_reader_read",
    "find_in_library_conversation",
    "_finish_library_conversation_find_focus",
)

#: Task 9 cleanup: the 9 reader-cluster names below have ZERO references
#: anywhere except their own one-line screen delegator (checked via a
#: repo-wide census across `tldw_chatbook/` and `Tests/`, plus the `__init__`
#: wiring-lambda block) -- nothing outside `LibraryConversationReaderController`
#: itself ever called `screen.<name>(...)`; every internal cluster call
#: already goes controller-to-controller via `self.<name>`. Their screen
#: delegators were deleted wholesale as dead weight; they remain in
#: `_READER_CLUSTER_METHOD_NAMES` above (the controller still legitimately
#: owns and uses them) but are excluded from the delegation check below.
_READER_CLUSTER_SCREEN_DELEGATOR_PRUNED: frozenset[str] = frozenset(
    {
        "_bootstrap_library_conversation_reader",
        "_conversation_reader_bootstrap_is_current",
        "_conversation_reader_record",
        "_conversation_reader_record_version",
        "_conversation_reader_request_is_current",
        "_conversation_reader_service",
        "_load_library_conversation_reader",
        "_retry_library_conversation_reader",
        "_finish_library_conversation_find_focus",
    }
)


@pytest.mark.unit
def test_reader_controller_owns_its_cluster() -> None:
    """Every one of the 21 moved names is a callable on the controller.

    Covers the whole cluster, not a 2-name sample: a name present on
    `LibraryScreen` (as a delegator) but missing on the controller would
    otherwise only surface as an `AttributeError` at call time, inside
    whichever delegator reached for it -- the same swallowed-failure shape
    task-7-report.md's fix-report section documents for a missing import.
    """
    from tldw_chatbook.UI.Library_Modules.library_conversation_reader_controller import (
        LibraryConversationReaderController,
    )

    missing = [
        name
        for name in _READER_CLUSTER_METHOD_NAMES
        if not callable(getattr(LibraryConversationReaderController, name, None))
    ]
    assert not missing, (
        f"LibraryConversationReaderController is missing: {missing!r}"
    )


@pytest.mark.unit
def test_screen_delegates_reader_handlers() -> None:
    """Every one of the 21 moved names is a one-line screen delegator that
    forwards to the SAME-NAMED controller method.

    Task 9 strengthened this from a loose "the controller is referenced
    somewhere in the source" substring check to a same-name forwarding
    check (a recorded review gap from task-7-report.md's fix round): the
    old check would have passed a delegator that called the controller for
    something unrelated to `name`. Still source-level, not behavioral -- a
    delegator could still forward the wrong arguments and this would not
    catch it -- but it catches the concrete regressions this test guards
    against: a cleanup pass that re-inlines a body onto the screen, a new
    method added under one of these names that never gets wired to the
    controller, or a delegator that silently calls a DIFFERENT controller
    method than its own name.

    Skips `_READER_CLUSTER_SCREEN_DELEGATOR_PRUNED` (task 9 deleted those
    9 screen delegators as dead weight -- zero external references) and
    instead asserts those names are genuinely ABSENT from `LibraryScreen`,
    so a future accidental re-add would fail loudly here rather than
    silently reintroducing dead code.
    """
    import inspect
    import re

    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    not_delegators = []
    for name in _READER_CLUSTER_METHOD_NAMES:
        if name in _READER_CLUSTER_SCREEN_DELEGATOR_PRUNED:
            assert getattr(LibraryScreen, name, None) is None, (
                f"{name!r} was pruned from the screen (task 9) but is back -- "
                "either wire it as a delegator again or drop it from "
                "_READER_CLUSTER_SCREEN_DELEGATOR_PRUNED"
            )
            continue
        method = getattr(LibraryScreen, name, None)
        if method is None:
            not_delegators.append(f"{name!r} (missing entirely)")
            continue
        src = inspect.getsource(method)
        if not re.search(rf"_conversation_reader_controller\.{re.escape(name)}\(", src):
            not_delegators.append(name)
    assert not not_delegators, (
        f"not delegators yet: {not_delegators!r}"
    )


#: Every method Task 8 moved into `LibraryConversationsController` (the
#: browse cluster: list/paging, row selection/multiselect, export, filter,
#: empty/retry states, and the "Use in Console"/"Use as source" handoff),
#: under its original `LibraryScreen` name. Same shape as
#: `_READER_CLUSTER_METHOD_NAMES` above -- one full-cluster ownership test
#: plus one full-cluster delegator test, both driven off this one
#: authoritative list, instead of a narrow hand-picked sample.
#:
#: Excludes seven names:
#:
#: - `_set_library_destination_with_conversation_fence` -- a deliberate
#:   ownership decision: despite its name, it is the shared
#:   rail/destination-switch helper every subsystem's row-open and
#:   rail-switch dispatch calls, not a Conversations-exclusive method -- it
#:   stays on `LibraryScreen`, unmoved.
#: - `handle_library_conversation_row`, `_library_conversation_loaded_preview_selected`,
#:   `handle_library_conversations_export_selected`,
#:   `handle_library_conversations_empty_console`,
#:   `handle_library_conversations_empty_clear_filter` -- found only by this
#:   task's own `-k "conversation and library"` sweep, not by static
#:   analysis: `Tests/UI/test_library_multiselect_conversations.py` calls
#:   these directly on hand-built `SimpleNamespace` fakes lacking a
#:   `_conversations_controller` attribute (a sibling failure mode to the
#:   recipe's §3 monkeypatch-routing rule, reached through unbound-class-
#:   method access instead of `monkeypatch.setattr`). See
#:   `library_conversations_controller.py`'s module docstring for the full
#:   trace.
#: - `_selected_conversation_handoff_payload` -- found only by this task's
#:   paired-baseline xdist sweep (recipe §7):
#:   `Tests/UI/test_post_release_workspaces_library_depth.py` does
#:   `screen._selected_conversation_handoff_payload = lambda: payload` on a
#:   REAL screen instance, expecting `_open_selected_conversation_handoff`
#:   (which DOES stay moved) to observe the patch on its next internal
#:   call -- an instance-attribute monkeypatch, a third bypass shape
#:   distinct from both the class-level monkeypatch in recipe §3 and the
#:   unbound-fake-self calls above.
_BROWSE_CLUSTER_METHOD_NAMES: tuple[str, ...] = (
    "_library_conversation_focus_region",
    "_library_conversation_escape_label",
    "_adopt_library_conversation_state_selection",
    "_carry_selected_conversation_into_snapshot",
    "_conversation_records",
    "_conversation_record_id",
    "_ensure_selected_conversation_id",
    "_selected_conversation_record",
    "_conversation_message_count_label",
    "_conversation_workspace_label",
    "_conversation_updated_label",
    "_build_library_conversations_state",
    "_sync_library_conversation_canvas",
    "_normalize_library_conversation_page",
    "_start_library_conversation_page_request",
    "_prepare_library_conversation_page_request",
    "_library_conversation_page_needs_recovery",
    "_finish_library_conversation_request_focus",
    "_finish_library_conversation_page_apply",
    "_fail_library_conversation_request",
    "_conversation_out_of_range_total",
    "_library_conversation_absence_fence_is_current",
    "_confirm_library_conversation_page_absence",
    "_load_library_conversation_page",
    "handle_library_conversations_select_toggle",
    "handle_library_conversations_select_all",
    "handle_library_conversations_select_clear",
    "handle_library_conversations_export",
    "handle_library_conversations_filter_submitted",
    "handle_library_conversations_retry",
    "_retry_pending_library_conversation_open",
    "handle_library_conversations_previous",
    "handle_library_conversations_next",
    "_focus_library_conversations_filter",
    "_refocus_library_conversations_filter_after_sync",
    "_notify_library_conversation_unavailable",
    "_validate_library_conversation_locator",
    "_open_selected_conversation_handoff",
    "open_selected_conversation_in_console",
    "use_selected_conversation_as_source",
)

#: Task 9 cleanup: same shape as `_READER_CLUSTER_SCREEN_DELEGATOR_PRUNED`
#: above -- these 9 browse-cluster names have ZERO references anywhere
#: except their own one-line screen delegator (repo-wide census). Their
#: screen delegators were deleted; they remain in
#: `_BROWSE_CLUSTER_METHOD_NAMES` (still genuinely owned by the controller)
#: but are excluded from the delegation check below.
_BROWSE_CLUSTER_SCREEN_DELEGATOR_PRUNED: frozenset[str] = frozenset(
    {
        "_ensure_selected_conversation_id",
        "_finish_library_conversation_request_focus",
        "_finish_library_conversation_page_apply",
        "_conversation_out_of_range_total",
        "_library_conversation_absence_fence_is_current",
        "_confirm_library_conversation_page_absence",
        "_retry_pending_library_conversation_open",
        "_focus_library_conversations_filter",
        "_refocus_library_conversations_filter_after_sync",
    }
)


@pytest.mark.unit
def test_browse_controller_owns_its_cluster() -> None:
    """Every one of the 40 browse-cluster names is a callable on the controller.

    Mirrors `test_reader_controller_owns_its_cluster` above, for
    `LibraryConversationsController` (task 8) instead of
    `LibraryConversationReaderController` (task 7).
    """
    from tldw_chatbook.UI.Library_Modules.library_conversations_controller import (
        LibraryConversationsController,
    )

    missing = [
        name
        for name in _BROWSE_CLUSTER_METHOD_NAMES
        if not callable(getattr(LibraryConversationsController, name, None))
    ]
    assert not missing, (
        f"LibraryConversationsController is missing: {missing!r}"
    )


@pytest.mark.unit
def test_screen_delegates_browse_handlers() -> None:
    """Every one of the 40 browse-cluster names is a one-line screen delegator
    that forwards to the SAME-NAMED controller method.

    Mirrors `test_screen_delegates_reader_handlers` above (61 delegators
    total across both clusters in this series: 21 reader + 40 browse). Five
    of the 40 names are `@staticmethod`/`@classmethod` on `LibraryScreen`
    (`_normalize_library_conversation_page`,
    `_validate_library_conversation_locator`, plus the three `@classmethod`
    label helpers `_conversation_message_count_label`,
    `_conversation_workspace_label`, `_conversation_updated_label`) -- per
    task-8-report.md's correction to a task-7 minor, those five delegators
    forward straight to the module-level `LibraryConversationsController`
    class (not through the `self._conversations_controller` instance
    attribute), so the same-name forwarding check (task 9 strengthening,
    see `test_screen_delegates_reader_handlers`) accepts either spelling.

    Skips `_BROWSE_CLUSTER_SCREEN_DELEGATOR_PRUNED` (task 9 deleted those
    9 screen delegators as dead weight -- zero external references) and
    instead asserts those names are genuinely ABSENT from `LibraryScreen`.
    """
    import inspect
    import re

    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    not_delegators = []
    for name in _BROWSE_CLUSTER_METHOD_NAMES:
        if name in _BROWSE_CLUSTER_SCREEN_DELEGATOR_PRUNED:
            assert getattr(LibraryScreen, name, None) is None, (
                f"{name!r} was pruned from the screen (task 9) but is back -- "
                "either wire it as a delegator again or drop it from "
                "_BROWSE_CLUSTER_SCREEN_DELEGATOR_PRUNED"
            )
            continue
        method = getattr(LibraryScreen, name, None)
        if method is None:
            not_delegators.append(f"{name!r} (missing entirely)")
            continue
        src = inspect.getsource(method)
        escaped = re.escape(name)
        if not re.search(rf"_conversations_controller\.{escaped}\(", src) and not (
            re.search(rf"LibraryConversationsController\.{escaped}\(", src)
        ):
            not_delegators.append(name)
    assert not not_delegators, (
        f"not delegators yet: {not_delegators!r}"
    )


@pytest.mark.unit
def test_browse_controller_exposes_every_state_field() -> None:
    """The browse controller's generated shim loop covers every state field.

    Mirrors `test_reader_controller_exposes_every_state_field` above,
    against `LibraryConversationsController` (task 8) instead of
    `LibraryConversationReaderController` (task 7). Both controllers'
    shim generators now import the SAME `CONVERSATIONS_PLURAL_STATE_FIELDS`
    set from `library_conversations_state` (task 8 promoted it to one
    shared home instead of adding a third local copy) -- this test still
    checks the controller's actual generated properties, not the shared
    constant directly, so it still catches a broken generator loop even
    though the drift-between-two-copies risk itself is now closed.
    """
    import dataclasses

    from tldw_chatbook.UI.Library_Modules.library_conversations_controller import (
        LibraryConversationsController,
    )

    field_names = {f.name for f in dataclasses.fields(LibraryConversationsState)}
    assert field_names, "state object is empty"
    missing = []
    for name in field_names:
        for prefix in ("_library_conversation_", "_library_conversations_"):
            if isinstance(
                getattr(LibraryConversationsController, prefix + name, None),
                property,
            ):
                break
        else:
            missing.append(name)
    assert not missing, (
        f"no browse controller shim property found for state field(s): {missing!r}"
    )


@pytest.mark.unit
def test_reader_controller_exposes_every_state_field() -> None:
    """The controller's generated shim loop covers every state field.

    Mirrors `test_browse_controller_exposes_every_state_field` above, against
    `LibraryConversationReaderController` (task 7) instead of
    `LibraryConversationsController` (task 8). Guards the drift risk task
    7's review originally flagged: the controller module used to keep its
    own `_READER_PLURAL_STATE_FIELDS` set (which field names use the
    `_library_conversations_` plural prefix vs. the singular
    `_library_conversation_` prefix), independently duplicated from a copy
    that used to live on the screen. Task 8 closed that specific drift by
    promoting both copies to the one shared `CONVERSATIONS_PLURAL_STATE_FIELDS`
    constant in `library_conversations_state.py`, which every controller's
    shim generator now imports (the screen's own shim block -- and its
    field-name literal -- is gone entirely as of task 9's cleanup: `_conversations_state`
    is a real dataclass instance, not a shimmed screen attribute). This test
    still exercises the controller's actual generated properties rather than
    asserting the shared constant directly, so it still catches a broken
    shim-generation loop for any one field -- on this controller specifically,
    since task 9 deleted the screen-side test with the equivalent job.
    """
    import dataclasses

    from tldw_chatbook.UI.Library_Modules.library_conversation_reader_controller import (
        LibraryConversationReaderController,
    )

    field_names = {f.name for f in dataclasses.fields(LibraryConversationsState)}
    assert field_names, "state object is empty"
    missing = []
    for name in field_names:
        for prefix in ("_library_conversation_", "_library_conversations_"):
            if isinstance(
                getattr(LibraryConversationReaderController, prefix + name, None),
                property,
            ):
                break
        else:
            missing.append(name)
    assert not missing, (
        f"no controller shim property found for state field(s): {missing!r}"
    )
