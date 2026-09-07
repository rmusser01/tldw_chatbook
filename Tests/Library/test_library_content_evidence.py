"""Pure Library content evidence and lifecycle transition contracts."""

import pytest

from tldw_chatbook.Library.library_content_evidence import (
    LibraryContentEvidence,
    LibraryEvidenceStatus,
)
from tldw_chatbook.Library.library_rail_state import (
    LibraryLifecycle,
    aggregate_library_lifecycle,
    explore_library_lifecycle,
    return_library_lifecycle_to_starter,
)


def test_content_evidence_and_status_values_are_stable():
    assert [value.value for value in LibraryContentEvidence] == [
        "unknown",
        "empty",
        "has_user_content",
    ]
    assert [value.value for value in LibraryEvidenceStatus] == [
        "loading",
        "settled",
        "partial_failure",
    ]


def test_any_usable_content_graduates_and_graduation_is_sticky():
    evidence = (
        LibraryContentEvidence.UNKNOWN,
        LibraryContentEvidence.EMPTY,
        LibraryContentEvidence.HAS_USER_CONTENT,
        LibraryContentEvidence.UNKNOWN,
        LibraryContentEvidence.EMPTY,
        LibraryContentEvidence.EMPTY,
    )

    for lifecycle in (
        LibraryLifecycle.UNKNOWN,
        LibraryLifecycle.STARTER,
        LibraryLifecycle.EXPANDED,
        LibraryLifecycle.GRADUATED,
    ):
        assert (
            aggregate_library_lifecycle(lifecycle, evidence)
            is LibraryLifecycle.GRADUATED
        )

    assert (
        aggregate_library_lifecycle(
            LibraryLifecycle.GRADUATED,
            (LibraryContentEvidence.EMPTY,) * 6,
        )
        is LibraryLifecycle.GRADUATED
    )


def test_starter_requires_every_source_to_report_empty():
    all_empty = (LibraryContentEvidence.EMPTY,) * 6

    assert (
        aggregate_library_lifecycle(LibraryLifecycle.UNKNOWN, all_empty)
        is LibraryLifecycle.STARTER
    )
    with pytest.raises(ValueError, match="exactly six"):
        aggregate_library_lifecycle(LibraryLifecycle.UNKNOWN, all_empty[:-1])
    with pytest.raises(ValueError, match="exactly six"):
        aggregate_library_lifecycle(
            LibraryLifecycle.UNKNOWN,
            all_empty + (LibraryContentEvidence.EMPTY,),
        )


def test_unknown_evidence_never_claims_starter():
    evidence = (LibraryContentEvidence.EMPTY,) * 5 + (LibraryContentEvidence.UNKNOWN,)

    assert (
        aggregate_library_lifecycle(LibraryLifecycle.UNKNOWN, evidence)
        is LibraryLifecycle.UNKNOWN
    )


def test_evidence_aggregation_accepts_only_enums():
    with pytest.raises(TypeError, match="LibraryContentEvidence"):
        aggregate_library_lifecycle(
            LibraryLifecycle.UNKNOWN,
            (LibraryContentEvidence.EMPTY,) * 5 + ("empty",),
        )


@pytest.mark.parametrize(
    "evidence",
    [
        (LibraryContentEvidence.EMPTY,) * 6,
        (LibraryContentEvidence.EMPTY,) * 5 + (LibraryContentEvidence.UNKNOWN,),
    ],
)
def test_aggregation_does_not_automatically_regress_expanded(evidence):
    assert (
        aggregate_library_lifecycle(LibraryLifecycle.EXPANDED, evidence)
        is LibraryLifecycle.EXPANDED
    )


@pytest.mark.parametrize(
    "blocking_evidence",
    [
        LibraryContentEvidence.UNKNOWN,
        LibraryContentEvidence.HAS_USER_CONTENT,
    ],
)
def test_return_to_starter_requires_authoritative_empty_evidence(blocking_evidence):
    evidence = (LibraryContentEvidence.EMPTY,) * 5 + (blocking_evidence,)

    assert (
        return_library_lifecycle_to_starter(LibraryLifecycle.EXPANDED, evidence)
        is LibraryLifecycle.EXPANDED
    )


def test_explore_expands_separately_and_empty_expanded_can_return_to_starter():
    preferences = {"browse_open": False, "details_open": True}

    assert (
        explore_library_lifecycle(LibraryLifecycle.UNKNOWN) is LibraryLifecycle.EXPANDED
    )
    assert (
        explore_library_lifecycle(LibraryLifecycle.STARTER) is LibraryLifecycle.EXPANDED
    )
    assert (
        explore_library_lifecycle(LibraryLifecycle.GRADUATED)
        is LibraryLifecycle.GRADUATED
    )
    assert preferences == {"browse_open": False, "details_open": True}

    assert (
        return_library_lifecycle_to_starter(
            LibraryLifecycle.EXPANDED,
            (LibraryContentEvidence.EMPTY,) * 6,
        )
        is LibraryLifecycle.STARTER
    )
    assert (
        return_library_lifecycle_to_starter(
            LibraryLifecycle.GRADUATED,
            (LibraryContentEvidence.EMPTY,) * 6,
        )
        is LibraryLifecycle.GRADUATED
    )
