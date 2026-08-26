"""Pure contracts for the bounded-fractional Library rail width policy."""

from __future__ import annotations

import pytest

from tldw_chatbook.Utils.library_rail_width import (
    LIBRARY_CANVAS_MIN_WIDTH,
    LIBRARY_CUSTOM_MAX_WIDTH,
    LIBRARY_DEFAULT_MAX_WIDTH,
    LIBRARY_EMERGENCY_WIDTH,
    LIBRARY_MIN_WIDTH,
    LIBRARY_REFERENCE_WIDTH,
    OrdinaryRailPresentation,
    OrdinaryRailStyleContract,
    ordinary_emergency_required,
    project_default_library_width,
    resolve_ordinary_rail_contract,
)


def test_width_policy_constants_express_the_approved_bounds() -> None:
    assert (
        LIBRARY_REFERENCE_WIDTH,
        LIBRARY_MIN_WIDTH,
        LIBRARY_DEFAULT_MAX_WIDTH,
        LIBRARY_CUSTOM_MAX_WIDTH,
        LIBRARY_CANVAS_MIN_WIDTH,
        LIBRARY_EMERGENCY_WIDTH,
    ) == (31, 24, 34, 48, 40, 64)
    assert LIBRARY_EMERGENCY_WIDTH == LIBRARY_MIN_WIDTH + LIBRARY_CANVAS_MIN_WIDTH


@pytest.mark.parametrize(
    ("content_width", "expected"),
    [
        (1, 24),
        (24, 24),
        (127, 24),
        (128, 24),
        (152, 29),
        (163, 31),
        (165, 31),
        (178, 33),
        (181, 34),
        (10000, 34),
    ],
)
def test_project_default_library_width_is_bounded_fractional(
    content_width: int, expected: int
) -> None:
    assert project_default_library_width(content_width) == expected


@pytest.mark.parametrize("content_width", [0, -1, True, False, 1.0, "64", None])
def test_width_helpers_reject_non_positive_or_non_integer_content_width(
    content_width: object,
) -> None:
    with pytest.raises(ValueError, match="content_width must be a positive integer"):
        project_default_library_width(content_width)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="content_width must be a positive integer"):
        ordinary_emergency_required(content_width)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("content_width", "expected"), [(1, True), (63, True), (64, False), (10000, False)]
)
def test_ordinary_emergency_is_required_only_below_sixty_four_columns(
    content_width: int, expected: bool
) -> None:
    assert ordinary_emergency_required(content_width) is expected


def test_default_alongside_contract_uses_native_fractional_width() -> None:
    assert resolve_ordinary_rail_contract(
        64, OrdinaryRailPresentation.ALONGSIDE, False, 31
    ) == OrdinaryRailStyleContract(True, "3fr", 24, 34)


@pytest.mark.parametrize(
    ("content_width", "saved_width", "expected"),
    [
        (74, 35, 34),
        (75, 35, 35),
        (64, 48, 24),
        (80, 48, 40),
        (87, 48, 47),
        (88, 48, 48),
    ],
)
def test_custom_alongside_contract_compresses_without_mutating_saved_width(
    content_width: int, saved_width: int, expected: int
) -> None:
    contract = resolve_ordinary_rail_contract(
        content_width, OrdinaryRailPresentation.ALONGSIDE, True, saved_width
    )

    assert contract == OrdinaryRailStyleContract(True, expected, expected, expected)
    assert saved_width in range(LIBRARY_MIN_WIDTH, LIBRARY_CUSTOM_MAX_WIDTH + 1)


def test_rail_only_and_hidden_contracts_clear_or_fill_inline_width_rules() -> None:
    assert resolve_ordinary_rail_contract(
        1, OrdinaryRailPresentation.RAIL_ONLY, False, 31
    ) == OrdinaryRailStyleContract(True, "1fr", 0, None)
    assert resolve_ordinary_rail_contract(
        1, OrdinaryRailPresentation.HIDDEN, False, 31
    ) == OrdinaryRailStyleContract(False, None, None, None)


def test_alongside_is_invalid_during_emergency_widths() -> None:
    with pytest.raises(ValueError, match="requires content_width of at least 64"):
        resolve_ordinary_rail_contract(
            63, OrdinaryRailPresentation.ALONGSIDE, False, 31
        )


@pytest.mark.parametrize("custom_widths_enabled", [1, 0, "true", None])
def test_contract_rejects_non_boolean_custom_width_flags(
    custom_widths_enabled: object,
) -> None:
    with pytest.raises(TypeError, match="custom_widths_enabled must be a boolean"):
        resolve_ordinary_rail_contract(
            64,
            OrdinaryRailPresentation.ALONGSIDE,
            custom_widths_enabled,  # type: ignore[arg-type]
            31,
        )


@pytest.mark.parametrize("saved_width", [True, "31", None])
def test_contract_rejects_non_integer_saved_width(saved_width: object) -> None:
    with pytest.raises(TypeError, match="saved_width"):
        resolve_ordinary_rail_contract(
            64,
            OrdinaryRailPresentation.ALONGSIDE,
            True,
            saved_width,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("saved_width", [23, 49])
def test_contract_rejects_out_of_range_saved_width(saved_width: int) -> None:
    with pytest.raises(ValueError, match="saved_width"):
        resolve_ordinary_rail_contract(
            64,
            OrdinaryRailPresentation.ALONGSIDE,
            True,
            saved_width,  # type: ignore[arg-type]
        )


def test_contract_rejects_non_presentation_values() -> None:
    with pytest.raises(TypeError, match="presentation"):
        resolve_ordinary_rail_contract(64, "alongside", False, 31)  # type: ignore[arg-type]
