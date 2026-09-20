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
        LibraryContentEvidence.EMPTY,
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
            (LibraryContentEvidence.EMPTY,) * 7,
        )
        is LibraryLifecycle.GRADUATED
    )


def test_starter_requires_every_source_to_report_empty():
    all_empty = (LibraryContentEvidence.EMPTY,) * 7

    assert (
        aggregate_library_lifecycle(LibraryLifecycle.UNKNOWN, all_empty)
        is LibraryLifecycle.STARTER
    )
    with pytest.raises(ValueError, match="exactly seven"):
        aggregate_library_lifecycle(LibraryLifecycle.UNKNOWN, all_empty[:-1])
    with pytest.raises(ValueError, match="exactly seven"):
        aggregate_library_lifecycle(
            LibraryLifecycle.UNKNOWN,
            all_empty + (LibraryContentEvidence.EMPTY,),
        )


def test_unknown_evidence_never_claims_starter():
    evidence = (LibraryContentEvidence.EMPTY,) * 6 + (LibraryContentEvidence.UNKNOWN,)

    assert (
        aggregate_library_lifecycle(LibraryLifecycle.UNKNOWN, evidence)
        is LibraryLifecycle.UNKNOWN
    )


def test_evidence_aggregation_accepts_only_enums():
    with pytest.raises(TypeError, match="LibraryContentEvidence"):
        aggregate_library_lifecycle(
            LibraryLifecycle.UNKNOWN,
            (LibraryContentEvidence.EMPTY,) * 6 + ("empty",),
        )


@pytest.mark.parametrize(
    "evidence",
    [
        (LibraryContentEvidence.EMPTY,) * 7,
        (LibraryContentEvidence.EMPTY,) * 6 + (LibraryContentEvidence.UNKNOWN,),
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
    evidence = (LibraryContentEvidence.EMPTY,) * 6 + (blocking_evidence,)

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
            (LibraryContentEvidence.EMPTY,) * 7,
        )
        is LibraryLifecycle.STARTER
    )
    assert (
        return_library_lifecycle_to_starter(
            LibraryLifecycle.GRADUATED,
            (LibraryContentEvidence.EMPTY,) * 7,
        )
        is LibraryLifecycle.GRADUATED
    )


def test_artifact_only_profile_graduates_in_the_seventh_source_slot():
    evidence = (LibraryContentEvidence.EMPTY,) * 6 + (
        LibraryContentEvidence.HAS_USER_CONTENT,
    )
    assert aggregate_library_lifecycle(LibraryLifecycle.UNKNOWN, evidence) is (
        LibraryLifecycle.GRADUATED
    )


@pytest.fixture
def artifact_owners(tmp_path):
    from tldw_chatbook.Chatbooks.local_chatbook_service import LocalChatbookService
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB

    subscriptions = SubscriptionsDB(tmp_path / "subscriptions.db", "evidence")
    kept = CharactersRAGDB(tmp_path / "kept.db", client_id="evidence")
    registry = LocalChatbookService(registry_path=tmp_path / "registry.json")
    try:
        yield {
            "subscriptions_db": subscriptions,
            "chachanotes_db": kept,
            "local_chatbook_service": registry,
        }
    finally:
        subscriptions.close()
        kept.close_connection()


@pytest.mark.asyncio
async def test_artifact_empty_reads_and_unconfigured_owners_are_empty(artifact_owners):
    from tldw_chatbook.Library.library_content_evidence import (
        get_library_artifact_content_evidence,
    )

    assert await get_library_artifact_content_evidence(**artifact_owners) is (
        LibraryContentEvidence.EMPTY
    )
    assert await get_library_artifact_content_evidence() is LibraryContentEvidence.EMPTY


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["live_reports", "kept_reports", "chatbooks"])
async def test_each_owned_artifact_store_can_supply_positive_evidence(
    artifact_owners, owner
):
    from tldw_chatbook.Library.library_content_evidence import (
        get_library_artifact_content_evidence,
    )
    from tldw_chatbook.Subscriptions.watchlist_bundle_service import (
        WatchlistBundleService,
    )

    if owner == "live_reports":
        subscriptions = artifact_owners["subscriptions_db"]
        watchlist = WatchlistBundleService(subscriptions).create(name="Evidence")
        subscriptions.insert_briefing(watchlist["id"], status="complete")
    elif owner == "kept_reports":
        artifact_owners["chachanotes_db"].create_kept_briefing(
            source_briefing_id=777,
            watchlist_name="Deleted source",
            body_markdown="A durable report",
            origin="manual",
        )
        artifact_owners["subscriptions_db"].close()
        artifact_owners["subscriptions_db"] = None
    else:
        await artifact_owners["local_chatbook_service"].create_chatbook(name="Research")
    assert await get_library_artifact_content_evidence(**artifact_owners) is (
        LibraryContentEvidence.HAS_USER_CONTENT
    )


@pytest.mark.asyncio
async def test_artifact_read_failure_is_unknown_but_kept_content_wins(artifact_owners):
    from tldw_chatbook.Library.library_content_evidence import (
        get_library_artifact_content_evidence,
    )

    artifact_owners["local_chatbook_service"].registry_path.write_text("invalid JSON")
    assert await get_library_artifact_content_evidence(**artifact_owners) is (
        LibraryContentEvidence.UNKNOWN
    )
    artifact_owners["chachanotes_db"].create_kept_briefing(
        source_briefing_id=888,
        watchlist_name="Deleted source",
        body_markdown="A durable report",
        origin="manual",
    )
    assert await get_library_artifact_content_evidence(**artifact_owners) is (
        LibraryContentEvidence.HAS_USER_CONTENT
    )


@pytest.mark.asyncio
async def test_configured_but_missing_artifact_owner_is_unknown():
    from tldw_chatbook.Library.library_content_evidence import (
        get_library_artifact_content_evidence,
    )

    assert (
        await get_library_artifact_content_evidence(unavailable_sources=("chatbooks",))
        is LibraryContentEvidence.UNKNOWN
    )
