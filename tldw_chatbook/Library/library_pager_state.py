"""Pure display derivation for top-level Library pagers."""

from dataclasses import dataclass
from typing import Literal


PageFreshness = Literal["uninitialized", "fresh", "stale"]

_FIRST_PAGE_REASON = "Already on the first page."
_FINAL_PAGE_REASON = "No more results."
_LOADING_REASON = "Page is loading."
_UNKNOWN_BOUNDARY_REASON = "Page boundary is unknown."


@dataclass(frozen=True)
class LibraryPagerDisplay:
    """Immutable copy and control state for a source-owned Library pager."""

    title_count: int | None
    range_copy: str
    page_copy: str
    status_copy: str
    previous_disabled: bool
    next_disabled: bool
    previous_reason: str
    next_reason: str
    retry_visible: bool
    #: task-28016: a fresh, non-loading result that fits one page. Sources may
    #: use it to drop pager chrome (Page 1 of 1, boundary reasons) that is pure
    #: noise when there is nowhere to page to.
    single_page: bool = False


@dataclass(frozen=True)
class LibraryPagerLayout:
    """What the one-page rule leaves on screen for one pager (task-32104).

    Attributes:
        status_parts: The status line's parts, already stripped of empties
            and joined by the caller (" · ").
        boundary_reasons: The de-duplicated boundary reasons to render, or
            empty when there is nowhere to page to.
        controls_hidden: Whether the Previous/Next row renders at all.
    """

    status_parts: tuple[str, ...]
    boundary_reasons: tuple[str, ...]
    controls_hidden: bool


def library_pager_layout(
    pager: LibraryPagerDisplay,
    *,
    retry_visible: bool | None = None,
) -> LibraryPagerLayout:
    """Apply the one-page pager rule once, for every Library pager.

    task-28016 + task-31237: a list that fits one page has nowhere to page
    to, so "Page 1 of 1", the boundary reasons ("Already on the first
    page.", "No more results.") and the two dead "○ Previous ○ Next" forms
    all say the same nothing. The item range stays, and every part of it
    returns the moment a second page exists. A Retry still needs its row.

    task-32104: this used to be spelled out separately in Media's,
    Conversations' and Prompts' own ``_compose_pager``, over the one shared
    ``single_page`` flag -- three readings that had already drifted, and a
    fourth surface would have meant a fourth copy.

    Args:
        pager: The derived pager display.
        retry_visible: Overrides ``pager.retry_visible`` for a surface that
            renders its Retry somewhere else -- Media puts a FAILED fetch's
            Retry in its load callout, beside the reason (task-31632), and
            must not keep a control row here just to hold one.

    Returns:
        The three decisions the rule makes for this pager.
    """
    visible_retry = pager.retry_visible if retry_visible is None else retry_visible
    if pager.single_page:
        return LibraryPagerLayout(
            status_parts=tuple(copy for copy in (pager.range_copy,) if copy),
            boundary_reasons=(),
            controls_hidden=not visible_retry,
        )
    return LibraryPagerLayout(
        status_parts=tuple(
            copy for copy in (pager.range_copy, pager.page_copy) if copy
        ),
        # A reason is non-empty only while its own control is disabled (see
        # ``build_library_pager_display``), so this needs no second
        # disabled check to say the same thing.
        boundary_reasons=tuple(
            dict.fromkeys(
                reason
                for reason in (pager.previous_reason, pager.next_reason)
                if reason
            )
        ),
        controls_hidden=False,
    )


def simple_library_pager_display(
    *,
    range_copy: str,
    page: int,
    total_pages: int,
    has_previous: bool,
    has_next: bool,
) -> LibraryPagerDisplay:
    """Build a pager display for a source that pages itself (task-32354).

    ``build_library_pager_display`` validates row counts against an exact
    total and raises when they disagree -- correct for the sources that own
    their paging end to end, fatal for one whose service may return a short
    page. This carries the same copy and the same boundary reasons so those
    sources can still go through ``library_pager_layout``.

    Args:
        range_copy: The already-formatted item range line.
        page: The applied page number.
        total_pages: Known page count, or 0 when it is unknown.
        has_previous: Whether a previous page can be loaded.
        has_next: Whether a next page can be loaded.

    Returns:
        A display whose ``single_page`` is True when neither direction can move.
    """
    return LibraryPagerDisplay(
        title_count=None,
        range_copy=range_copy,
        page_copy=f"Page {page} of {total_pages}" if total_pages else "",
        status_copy="",
        previous_disabled=not has_previous,
        next_disabled=not has_next,
        previous_reason="" if has_previous else _FIRST_PAGE_REASON,
        next_reason="" if has_next else _FINAL_PAGE_REASON,
        retry_visible=False,
        single_page=not has_previous and not has_next,
    )


def build_library_pager_display(
    *,
    applied_page: int | None,
    requested_page: int,
    page_size: int,
    row_count: int,
    total: int | None,
    freshness: PageFreshness,
    loading: bool = False,
    error_copy: str = "",
    stale_copy: str = "",
) -> LibraryPagerDisplay:
    """Validate page metadata and derive its complete display state.

    Args:
        applied_page: Last successfully applied page, if one exists.
        requested_page: Page targeted by the current or last request.
        page_size: Maximum number of rows in a page.
        row_count: Number of retained rows.
        total: Exact applied total, available only while fresh.
        freshness: Whether metadata is absent, authoritative, or stale.
        loading: Whether a request is in progress.
        error_copy: Source-owned recoverable error copy.
        stale_copy: Source-owned reason for stale retained rows.

    Returns:
        Immutable title, pager, status, control, and retry display values.

    Raises:
        TypeError: An input has the wrong scalar type.
        ValueError: Page metadata or presentation state is contradictory.
    """

    if isinstance(requested_page, bool) or not isinstance(requested_page, int):
        raise TypeError("requested_page must be an integer")
    if requested_page < 1:
        raise ValueError("requested_page must be at least 1")
    if isinstance(page_size, bool) or not isinstance(page_size, int):
        raise TypeError("page_size must be an integer")
    if page_size < 1:
        raise ValueError("page_size must be at least 1")
    if isinstance(row_count, bool) or not isinstance(row_count, int):
        raise TypeError("row_count must be an integer")
    if row_count < 0:
        raise ValueError("row_count must be at least 0")
    if applied_page is not None:
        if isinstance(applied_page, bool) or not isinstance(applied_page, int):
            raise TypeError("applied_page must be an integer")
        if applied_page < 1:
            raise ValueError("applied_page must be at least 1")
    if total is not None:
        if isinstance(total, bool) or not isinstance(total, int):
            raise TypeError("total must be an integer")
        if total < 0:
            raise ValueError("total must be at least 0")
    if freshness not in ("uninitialized", "fresh", "stale"):
        raise ValueError("freshness must be uninitialized, fresh, or stale")
    if not isinstance(loading, bool):
        raise TypeError("loading must be a boolean")
    if not isinstance(error_copy, str):
        raise TypeError("error_copy must be a string")
    if not isinstance(stale_copy, str):
        raise TypeError("stale_copy must be a string")
    if error_copy and not error_copy.strip():
        raise ValueError("error_copy cannot be whitespace-only")
    if stale_copy and not stale_copy.strip():
        raise ValueError("stale_copy cannot be whitespace-only")
    if row_count > page_size:
        raise ValueError("row_count cannot exceed page_size")
    if loading and error_copy:
        raise ValueError("loading and error_copy cannot both be set")

    single_page = False
    if freshness == "fresh":
        if applied_page is None or total is None:
            raise ValueError("fresh state requires applied_page and total")
        if stale_copy:
            raise ValueError("fresh state cannot include stale_copy")
        if requested_page != applied_page and not (loading or error_copy):
            raise ValueError("idle fresh state requires matching pages")
        total_pages = max(1, (total + page_size - 1) // page_size)
        single_page = not loading and not error_copy and total_pages == 1
        if applied_page > total_pages:
            raise ValueError("applied_page exceeds the final page")
        expected_rows = min(page_size, max(0, total - (applied_page - 1) * page_size))
        if row_count != expected_rows:
            raise ValueError("row_count is inconsistent with applied_page and total")

        range_copy = (
            f"{(applied_page - 1) * page_size + 1}-"
            f"{(applied_page - 1) * page_size + row_count} of {total}"
            if row_count
            else "0 of 0"
        )
        page_copy = f"Page {applied_page} of {total_pages}"
        status_copy = (
            f"Loading page {requested_page}…" if loading else error_copy
        )
        if loading:
            previous_disabled = next_disabled = True
            previous_reason = next_reason = _LOADING_REASON
        else:
            previous_disabled = applied_page == 1
            next_disabled = applied_page == total_pages
            previous_reason = _FIRST_PAGE_REASON if previous_disabled else ""
            next_reason = _FINAL_PAGE_REASON if next_disabled else ""
    else:
        if total is not None:
            raise ValueError("non-fresh state cannot expose total")
        if freshness == "uninitialized":
            if applied_page is not None or row_count:
                raise ValueError("uninitialized state cannot include an applied page")
            if stale_copy:
                raise ValueError("uninitialized state cannot include stale_copy")
            range_copy = (
                f"Loading page {requested_page}…"
                if loading
                else "No page loaded · Total unavailable"
            )
            status_copy = error_copy
        else:
            if applied_page is None:
                raise ValueError("stale state requires applied_page")
            if not stale_copy:
                raise ValueError("stale state requires stale_copy")
            if error_copy:
                raise ValueError("stale state uses stale_copy, not error_copy")
            range_copy = "List may be out of date"
            status_copy = (
                f"Loading page {requested_page}…" if loading else stale_copy
            )
        page_copy = ""
        previous_disabled = next_disabled = True
        reason = _LOADING_REASON if loading else _UNKNOWN_BOUNDARY_REASON
        previous_reason = next_reason = reason

    return LibraryPagerDisplay(
        title_count=total if freshness == "fresh" else None,
        range_copy=range_copy,
        page_copy=page_copy,
        status_copy=status_copy,
        previous_disabled=previous_disabled,
        next_disabled=next_disabled,
        previous_reason=previous_reason,
        next_reason=next_reason,
        retry_visible=not loading and (bool(error_copy) or freshness == "stale"),
        single_page=single_page,
    )
