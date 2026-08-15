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


def _require_int(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


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

    requested_page = _require_int("requested_page", requested_page, minimum=1)
    page_size = _require_int("page_size", page_size, minimum=1)
    row_count = _require_int("row_count", row_count, minimum=0)
    if applied_page is not None:
        applied_page = _require_int("applied_page", applied_page, minimum=1)
    if total is not None:
        total = _require_int("total", total, minimum=0)
    if freshness not in ("uninitialized", "fresh", "stale"):
        raise ValueError("freshness must be uninitialized, fresh, or stale")
    if not isinstance(loading, bool):
        raise TypeError("loading must be a boolean")
    if not isinstance(error_copy, str):
        raise TypeError("error_copy must be a string")
    if not isinstance(stale_copy, str):
        raise TypeError("stale_copy must be a string")
    if row_count > page_size:
        raise ValueError("row_count cannot exceed page_size")
    if loading and error_copy:
        raise ValueError("loading and error_copy cannot both be set")

    if freshness == "fresh":
        if applied_page is None or total is None:
            raise ValueError("fresh state requires applied_page and total")
        if stale_copy:
            raise ValueError("fresh state cannot include stale_copy")
        if requested_page != applied_page and not (loading or error_copy):
            raise ValueError("idle fresh state requires matching pages")
        total_pages = max(1, (total + page_size - 1) // page_size)
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
    )
