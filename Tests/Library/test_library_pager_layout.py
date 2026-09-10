"""The one-page pager rule, stated once (task-32104).

``single_page`` shipped as one flag on ``LibraryPagerDisplay`` and three
independent readings of it -- Media, Conversations and Prompts each spelled
out "drop the page counter, drop the boundary reasons, drop the controls
unless a Retry needs the row" in their own ``_compose_pager``. A fourth
surface needed a fourth copy, and the three had already drifted apart in
how they filtered the reasons.
"""

from __future__ import annotations

import inspect

import pytest

from tldw_chatbook.Library.library_pager_state import (
    build_library_pager_display,
    library_pager_layout,
)


def _display(*, total: int, page_size: int = 20, applied_page: int = 1, **kwargs):
    row_count = min(page_size, max(0, total - (applied_page - 1) * page_size))
    return build_library_pager_display(
        applied_page=applied_page,
        requested_page=applied_page,
        page_size=page_size,
        row_count=row_count,
        total=total,
        freshness="fresh",
        **kwargs,
    )


@pytest.mark.unit
def test_a_one_page_list_keeps_only_its_range() -> None:
    """Nowhere to page to: no counter, no boundary reasons, no controls."""
    layout = library_pager_layout(_display(total=2))

    assert layout.status_parts == ("1-2 of 2",)
    assert layout.boundary_reasons == ()
    assert layout.controls_hidden is True


@pytest.mark.unit
def test_a_multi_page_list_keeps_the_counter_and_its_boundary_reason() -> None:
    """Everything returns the moment a second page exists."""
    layout = library_pager_layout(_display(total=25))

    assert layout.status_parts == ("1-20 of 25", "Page 1 of 2")
    assert layout.boundary_reasons == ("Already on the first page.",)
    assert layout.controls_hidden is False


@pytest.mark.unit
def test_a_retry_keeps_the_control_row_on_a_single_page() -> None:
    """A stale one-page list still needs somewhere to put Try again."""
    stale = build_library_pager_display(
        applied_page=1,
        requested_page=1,
        page_size=20,
        row_count=1,
        total=None,
        freshness="stale",
        stale_copy="List may be out of date.",
    )
    assert stale.retry_visible is True
    assert library_pager_layout(stale).controls_hidden is False

    # ...and a surface that renders its Retry elsewhere says so, rather than
    # keeping an empty control row (Media puts a FAILED fetch's Retry in its
    # load callout, task-31632).
    assert (
        library_pager_layout(_display(total=2), retry_visible=False).controls_hidden
        is True
    )


@pytest.mark.unit
def test_every_pager_surface_reads_the_rule_from_the_helper() -> None:
    """No canvas carries its own reading of ``single_page`` any more."""
    from tldw_chatbook.Widgets.Library.library_conversations_canvas import (
        LibraryConversationsCanvas,
    )
    from tldw_chatbook.Widgets.Library.library_media_canvas import LibraryMediaCanvas
    from tldw_chatbook.Widgets.Library.library_prompts_canvas import (
        LibraryPromptsListCanvas,
    )

    for canvas, member in (
        # Conversations renders its pager inline in ``compose``; the other
        # two have a ``_compose_pager`` of their own.
        (LibraryConversationsCanvas, "compose"),
        (LibraryMediaCanvas, "_compose_pager"),
        (LibraryPromptsListCanvas, "_compose_pager"),
    ):
        source = inspect.getsource(getattr(canvas, member))
        assert "library_pager_layout(" in source, canvas.__name__
        assert "pager.single_page" not in source, (
            f"{canvas.__name__} still spells the one-page rule out itself"
        )
