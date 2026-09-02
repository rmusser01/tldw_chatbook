from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Library.library_pager_state import (
    LibraryPagerDisplay,
    build_library_pager_display,
)


def _fresh_display(**overrides: object) -> LibraryPagerDisplay:
    values: dict[str, object] = {
        "applied_page": 2,
        "requested_page": 2,
        "page_size": 20,
        "row_count": 20,
        "total": 45,
        "freshness": "fresh",
    }
    values.update(overrides)
    return build_library_pager_display(**values)  # type: ignore[arg-type]


def test_fresh_middle_page_has_exact_range_and_enabled_boundaries():
    display = _fresh_display()

    assert display == LibraryPagerDisplay(
        title_count=45,
        range_copy="21-40 of 45",
        page_copy="Page 2 of 3",
        status_copy="",
        previous_disabled=False,
        next_disabled=False,
        previous_reason="",
        next_reason="",
        retry_visible=False,
    )


def test_first_page_has_visible_previous_boundary_reason():
    display = _fresh_display(applied_page=1, requested_page=1)

    assert display.range_copy == "1-20 of 45"
    assert display.page_copy == "Page 1 of 3"
    assert display.previous_disabled is True
    assert display.previous_reason == "Already on the first page."
    assert display.next_disabled is False
    assert display.next_reason == ""


def test_partial_final_page_has_visible_next_boundary_reason():
    display = _fresh_display(
        applied_page=3,
        requested_page=3,
        row_count=5,
    )

    assert display.range_copy == "41-45 of 45"
    assert display.page_copy == "Page 3 of 3"
    assert display.previous_disabled is False
    assert display.next_disabled is True
    assert display.next_reason == "No more results."


def test_exact_multiple_final_page_has_exact_range():
    display = _fresh_display(total=40)

    assert display.title_count == 40
    assert display.range_copy == "21-40 of 40"
    assert display.page_copy == "Page 2 of 2"
    assert display.next_disabled is True
    assert display.next_reason == "No more results."


def test_one_row_page_has_both_visible_boundary_reasons():
    display = _fresh_display(
        applied_page=1,
        requested_page=1,
        row_count=1,
        total=1,
    )

    assert display.range_copy == "1-1 of 1"
    assert display.page_copy == "Page 1 of 1"
    assert display.previous_reason == "Already on the first page."
    assert display.next_reason == "No more results."


def test_single_page_fresh_result_is_flagged_single_page():
    # task-28016: one full page (total <= page_size) is a single page, so the
    # media canvas can drop the "Page 1 of 1" / boundary-reason chrome.
    display = _fresh_display(applied_page=1, requested_page=1, total=3, row_count=3)

    assert display.single_page is True
    assert display.page_copy == "Page 1 of 1"


def test_empty_fresh_result_is_flagged_single_page():
    display = _fresh_display(applied_page=1, requested_page=1, total=0, row_count=0)

    assert display.single_page is True


def test_multi_page_result_is_not_single_page():
    # _fresh_display defaults to 45 rows over a 20-row page (3 pages).
    assert _fresh_display().single_page is False


def test_loading_single_page_is_not_flagged_single_page():
    display = _fresh_display(
        applied_page=1, requested_page=2, total=3, row_count=3, loading=True
    )

    assert display.single_page is False


def test_uninitialized_state_is_not_flagged_single_page():
    display = build_library_pager_display(
        applied_page=None,
        requested_page=1,
        page_size=20,
        row_count=0,
        total=None,
        freshness="uninitialized",
    )

    assert display.single_page is False


def test_successfully_empty_collection_is_not_uninitialized():
    display = _fresh_display(
        applied_page=1,
        requested_page=1,
        row_count=0,
        total=0,
    )

    assert display.title_count == 0
    assert display.range_copy == "0 of 0"
    assert display.page_copy == "Page 1 of 1"
    assert display.previous_disabled is True
    assert display.next_disabled is True
    assert display.retry_visible is False


def test_initial_loading_does_not_fabricate_exact_metadata():
    display = build_library_pager_display(
        applied_page=None,
        requested_page=1,
        page_size=20,
        row_count=0,
        total=None,
        freshness="uninitialized",
        loading=True,
    )

    assert display.title_count is None
    assert display.range_copy == "Loading page 1…"
    assert display.page_copy == ""
    assert display.status_copy == ""
    assert display.previous_reason == "Page is loading."
    assert display.next_reason == "Page is loading."
    assert display.retry_visible is False


def test_page_only_loading_retains_last_good_exact_metadata():
    display = _fresh_display(requested_page=3, loading=True)

    assert display.title_count == 45
    assert display.range_copy == "21-40 of 45"
    assert display.page_copy == "Page 2 of 3"
    assert display.status_copy == "Loading page 3…"
    assert display.previous_disabled is True
    assert display.next_disabled is True
    assert display.previous_reason == "Page is loading."
    assert display.next_reason == "Page is loading."
    assert display.retry_visible is False


def test_uninitialized_failure_never_fabricates_zero_total():
    display = build_library_pager_display(
        applied_page=None,
        requested_page=1,
        page_size=20,
        row_count=0,
        total=None,
        freshness="uninitialized",
        error_copy="Couldn't load conversations.",
    )

    assert display.title_count is None
    assert display.range_copy == "No page loaded · Total unavailable"
    assert display.page_copy == ""
    assert display.status_copy == "Couldn't load conversations."
    assert display.previous_disabled is True
    assert display.next_disabled is True
    assert display.previous_reason == "Page boundary is unknown."
    assert display.next_reason == "Page boundary is unknown."
    assert display.retry_visible is True


def test_recoverable_page_failure_retains_fresh_metadata():
    display = _fresh_display(
        requested_page=3,
        error_copy="Couldn't load page 3.",
    )

    assert display.title_count == 45
    assert display.range_copy == "21-40 of 45"
    assert display.page_copy == "Page 2 of 3"
    assert display.status_copy == "Couldn't load page 3."
    assert display.previous_disabled is False
    assert display.next_disabled is False
    assert display.retry_visible is True


@pytest.mark.parametrize(
    "stale_copy",
    [
        "List may be out of date",
        "Source changed again; try again.",
    ],
)
def test_stale_display_suppresses_exact_metadata_and_actions(stale_copy: str):
    display = build_library_pager_display(
        applied_page=3,
        requested_page=3,
        page_size=20,
        row_count=5,
        total=None,
        freshness="stale",
        stale_copy=stale_copy,
    )

    assert display.title_count is None
    assert display.range_copy == "List may be out of date"
    assert display.page_copy == ""
    assert display.status_copy == stale_copy
    assert display.previous_disabled is True
    assert display.next_disabled is True
    assert display.previous_reason == "Page boundary is unknown."
    assert display.next_reason == "Page boundary is unknown."
    assert display.retry_visible is True


def test_display_is_immutable():
    display = _fresh_display()

    with pytest.raises(FrozenInstanceError):
        display.title_count = 99  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("requested_page", True),
        ("page_size", False),
        ("row_count", True),
        ("total", False),
        ("applied_page", True),
        ("requested_page", "1"),
        ("page_size", 20.0),
        ("row_count", "20"),
        ("total", 45.0),
        ("applied_page", "2"),
    ],
)
def test_integer_inputs_reject_bools_and_non_integers(field: str, value: object):
    with pytest.raises(TypeError):
        _fresh_display(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("requested_page", 0),
        ("requested_page", -1),
        ("page_size", 0),
        ("page_size", -1),
        ("row_count", -1),
        ("total", -1),
        ("applied_page", 0),
        ("applied_page", -1),
    ],
)
def test_integer_inputs_reject_out_of_range_values(field: str, value: int):
    with pytest.raises(ValueError):
        _fresh_display(**{field: value})


def test_row_count_cannot_exceed_page_size():
    with pytest.raises(ValueError, match="row_count"):
        _fresh_display(row_count=21)


@pytest.mark.parametrize("freshness", ["", "unknown", "Fresh", 1, None])
def test_freshness_must_be_an_exact_permitted_literal(freshness: object):
    with pytest.raises(ValueError, match="freshness"):
        _fresh_display(freshness=freshness)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("loading", 1),
        ("loading", "yes"),
        ("error_copy", None),
        ("error_copy", 1),
        ("stale_copy", None),
        ("stale_copy", 1),
    ],
)
def test_loading_and_copy_inputs_require_exact_types(field: str, value: object):
    with pytest.raises(TypeError):
        _fresh_display(**{field: value})


@pytest.mark.parametrize(
    "overrides",
    [
        {"applied_page": None},
        {"total": None},
        {"row_count": 19},
        {"applied_page": 3, "requested_page": 3, "row_count": 0, "total": 40},
        {"applied_page": 2, "requested_page": 2, "row_count": 0, "total": 0},
    ],
)
def test_fresh_state_requires_coherent_exact_metadata(overrides: dict[str, object]):
    with pytest.raises(ValueError):
        _fresh_display(**overrides)


@pytest.mark.parametrize("freshness", ["uninitialized", "stale"])
def test_non_fresh_state_rejects_an_exposed_total(freshness: str):
    with pytest.raises(ValueError, match="total"):
        build_library_pager_display(
            applied_page=None if freshness == "uninitialized" else 2,
            requested_page=2,
            page_size=20,
            row_count=0,
            total=0,
            freshness=freshness,  # type: ignore[arg-type]
            stale_copy="List may be out of date" if freshness == "stale" else "",
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"loading": True, "error_copy": "Failed."},
        {"stale_copy": "List may be out of date"},
        {"requested_page": 3},
    ],
)
def test_fresh_state_rejects_contradictory_status(overrides: dict[str, object]):
    with pytest.raises(ValueError):
        _fresh_display(**overrides)


def test_uninitialized_state_rejects_applied_rows_or_stale_copy():
    base = {
        "requested_page": 1,
        "page_size": 20,
        "total": None,
        "freshness": "uninitialized",
    }

    with pytest.raises(ValueError):
        build_library_pager_display(applied_page=1, row_count=0, **base)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_library_pager_display(applied_page=None, row_count=1, **base)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_library_pager_display(
            applied_page=None,
            row_count=0,
            stale_copy="Stale.",
            **base,  # type: ignore[arg-type]
        )


def test_stale_state_requires_applied_page_and_source_owned_copy():
    base = {
        "requested_page": 1,
        "page_size": 20,
        "row_count": 0,
        "total": None,
        "freshness": "stale",
    }

    with pytest.raises(ValueError):
        build_library_pager_display(applied_page=None, stale_copy="Stale.", **base)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_library_pager_display(applied_page=1, stale_copy="", **base)  # type: ignore[arg-type]


def test_error_copy_rejects_whitespace_only_recovery_copy():
    with pytest.raises(ValueError, match="error_copy"):
        _fresh_display(error_copy=" \t ")


def test_stale_copy_rejects_whitespace_only_recovery_copy():
    with pytest.raises(ValueError, match="stale_copy"):
        build_library_pager_display(
            applied_page=1,
            requested_page=1,
            page_size=20,
            row_count=0,
            total=None,
            freshness="stale",
            stale_copy=" \t ",
        )


def test_stale_state_rejects_error_copy_even_with_meaningful_stale_copy():
    with pytest.raises(ValueError, match="error_copy"):
        build_library_pager_display(
            applied_page=1,
            requested_page=1,
            page_size=20,
            row_count=0,
            total=None,
            freshness="stale",
            error_copy="Couldn't refresh page.",
            stale_copy="List may be out of date",
        )
