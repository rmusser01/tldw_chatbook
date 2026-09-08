"""Explicit Media presentation ownership stays independent of request fencing."""

from dataclasses import fields

from tldw_chatbook.Library.library_media_state import (
    MediaBrowseScope,
    build_media_browse_result,
)
from tldw_chatbook.UI.Library_Modules.library_media_browse_state import MediaBrowseState
from Tests.UI.test_library_media_browse_controller import (
    _Screen,
    _Service,
    _controller,
    _page,
)


def test_browse_state_instances_are_independent_while_controllers_fence_results():
    first = _controller(_Screen(), _Service())
    second = _controller(_Screen(), _Service())
    assert first.state is not second.state
    assert first.state.requested_scope is not second.state.requested_scope
    first.state.type_options = ("video",)
    assert second.state.type_options == ()

    scope = MediaBrowseScope()
    first_generation = first.begin(scope)
    second_generation = second.begin(scope)
    first.invalidate()
    result = build_media_browse_result(scope, _page(1, 1))

    assert (
        first._apply(result, generation=first_generation, focus_identity=None) is False
    )
    assert (
        second._apply(result, generation=second_generation, focus_identity=None) is True
    )
    assert first.state.applied_result is None
    assert first.state.retained_items == ()
    assert second.state.applied_result is result
    assert second.state.retained_items[0]["id"] == "local:media:1"
    assert second.state.pager.title_count == 1


def test_state_write_is_the_same_retained_page_read_by_projection():
    controller = _controller(_Screen(), _Service())
    state = controller.state
    scope = MediaBrowseScope()
    result = build_media_browse_result(scope, _page(1, 1))
    state.applied_result = result
    state.retained_items = result.items
    state.freshness = "fresh"

    assert state.applied_scope is scope
    assert state.pager.title_count == 1
    assert state.note_analysis_state("local:media:1", has_analysis=True) is True
    assert state.retained_items[0]["has_analysis"] is True
    assert state.freshness == "fresh"


def test_state_owns_only_presentation_data_without_runtime_backreferences():
    controller = _controller(_Screen(), _Service())
    names = {field.name for field in fields(MediaBrowseState)}
    assert names == {
        "requested_scope",
        "inflight_scope",
        "applied_result",
        "retained_items",
        "freshness",
        "loading",
        "error_copy",
        "stale_copy",
        "stale_reason",
        "page_failure",
        "facet_failure",
        "_page_fault_reason",
        "_page_fault_context",
        "_facet_fault_reason",
        "_facet_fault_context",
        "type_options",
        "facet_loading",
        "facet_error_copy",
        "facet_fingerprint",
    }
    assert set(vars(controller.state)) == names
    assert not names.intersection(vars(controller))
    assert "_page_generation" in vars(controller)
    assert "_facet_generation" in vars(controller)
    assert not any(callable(value) for value in vars(controller.state).values())
    for name in (
        "failure",
        "applied_scope",
        "mutation_refresh_scope",
        "scope_for_page",
        "pager",
        "_failure_copy",
        "retain_stale_items",
        "note_analysis_state",
        "reconcile_committed_mutation",
        "clear_fault_episode",
    ):
        assert name in vars(MediaBrowseState)
        assert name not in vars(type(controller))
