"""Preflight computes treatments without reading or moving item content."""

import pytest

from Tests.Research_Workspace.test_copy_models import LOCAL, SERVER, TIME
from tldw_chatbook.Research_Workspace.copy_manifest import (
    ResearchCopyCandidate,
    ResearchCopyPreflightBuilder,
)
from tldw_chatbook.Research_Workspace.copy_models import (
    CopyConflictChoice,
    CopyItemKind,
    CopyOwnerRef,
    CopyReason,
    CopyTreatment,
    ResearchCopyConfirmation,
)


def preflight(*items, replace_supported=()):
    return ResearchCopyPreflightBuilder().build(
        LOCAL,
        SERVER,
        items,
        available={
            CopyItemKind.MEDIA: frozenset(
                {CopyTreatment.CONTENT, CopyTreatment.REFERENCE}
            )
        },
        versioned_replacements=frozenset(replace_supported),
        created_at=TIME,
    )


def test_preflight_keeps_unknown_and_device_only_records_out_of_executable_rows():
    manifest = preflight(
        ResearchCopyCandidate(
            CopyOwnerRef(LOCAL, "media", "1", version=3), estimated_bytes=100
        ),
        ResearchCopyCandidate(CopyOwnerRef(LOCAL, "conversation", "7", version=2)),
        ResearchCopyCandidate(
            CopyOwnerRef(LOCAL, "note", "9", version=2), transfer_policy="local-only"
        ),
    )
    assert manifest.estimated_bytes == 100
    by_id = {row.source.owner_id: row for row in manifest.items}
    assert by_id["1"].actionable
    assert by_id["7"].reason is CopyReason.OWNER_UNSUPPORTED
    assert by_id["9"].treatment is CopyTreatment.OMIT
    assert all(not by_id[key].actionable for key in ("7", "9"))


def test_preflight_requires_conflict_choice_and_projects_replace_capability():
    source = CopyOwnerRef(LOCAL, "media", "1", version=3)
    destination = CopyOwnerRef(SERVER, "media", "2", version=10)
    unresolved = preflight(ResearchCopyCandidate(source, destination=destination))
    assert unresolved.items[0].reason is CopyReason.CHOOSE_CONFLICT
    with pytest.raises(ValueError, match="conflict"):
        ResearchCopyConfirmation(unresolved.manifest_hash, confirmed=True).validate(
            unresolved
        )
    request = ResearchCopyCandidate(
        source, destination=destination, conflict=CopyConflictChoice.REPLACE_DESTINATION
    )
    assert preflight(request).items[0].reason is CopyReason.REPLACE_UNAVAILABLE
    accepted = preflight(request, replace_supported=(CopyItemKind.MEDIA,))
    assert accepted.items[0].actionable and accepted.requires_replacement_confirmation
    assert accepted.items[0].destination.version == 10


def test_preflight_requires_source_revision_and_blocks_reference_without_owner_support():
    no_revision = CopyOwnerRef(LOCAL, "media", "1")
    result = preflight(ResearchCopyCandidate(no_revision))
    assert result.items[0].reason is CopyReason.SOURCE_REVISION_REQUIRED
    missing_support = preflight(
        ResearchCopyCandidate(
            CopyOwnerRef(LOCAL, "note", "1", version=1), transfer_policy="reference"
        )
    )
    assert missing_support.items[0].treatment is CopyTreatment.BLOCKED
