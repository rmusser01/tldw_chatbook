"""Copy manifests retain only qualified, immutable owner references."""

from dataclasses import FrozenInstanceError, replace

import pytest

from tldw_chatbook.Research_Workspace.contracts import QualifiedWorkspaceRef
from tldw_chatbook.Research_Workspace.copy_models import (
    CopyConflictChoice,
    CopyOwnerRef,
    CopyTreatment,
    ResearchCopyConfirmation,
    ResearchCopyItem,
    ResearchCopyManifest,
)

LOCAL = QualifiedWorkspaceRef("local", "reading-group")
SERVER = QualifiedWorkspaceRef("server", "reading-group", "server-1", "account-1")
TIME = "2026-11-24T19:00:00Z"


def item(owner_id="41", **kwargs):
    return ResearchCopyItem(
        CopyOwnerRef(LOCAL, "media", owner_id, version=1),
        CopyTreatment.CONTENT,
        **kwargs,
    )


def test_manifest_is_stable_across_item_order_and_time_but_changes_with_versions():
    first = ResearchCopyManifest(LOCAL, SERVER, (item("41"), item("42")), TIME)
    replay = ResearchCopyManifest(
        LOCAL, SERVER, (item("42"), item("41")), "2026-11-25T19:00:00Z"
    )
    assert first.transfer_id == replay.transfer_id
    assert first.manifest_hash == replay.manifest_hash
    assert first.item_idempotency_key(first.items[0]) == replay.item_idempotency_key(
        replay.items[0]
    )
    assert first.item_idempotency_key(first.items[0]) != first.item_idempotency_key(
        first.items[1]
    )
    changed = replace(first.items[0], source=replace(first.items[0].source, version=2))
    assert (
        replace(first, items=(changed, first.items[1])).transfer_id != first.transfer_id
    )
    with pytest.raises(FrozenInstanceError):
        first.items[0].source.version = 3
    renamed = replace(first, target_name="Reading group copy")
    assert renamed.manifest_hash != first.manifest_hash
    with pytest.raises(ValueError, match="different manifest"):
        ResearchCopyConfirmation(first.manifest_hash, confirmed=True).validate(renamed)


def test_manifest_rejects_ambiguous_authority_duplicate_and_wrong_item_scope():
    with pytest.raises(ValueError, match="different authorities"):
        ResearchCopyManifest(
            LOCAL, replace(LOCAL, workspace_id="other"), (item(),), TIME
        )
    with pytest.raises(ValueError, match="principal"):
        ResearchCopyManifest(LOCAL, replace(SERVER, principal_id=""), (item(),), TIME)
    with pytest.raises(ValueError, match="duplicate"):
        ResearchCopyManifest(LOCAL, SERVER, (item(), item()), TIME)
    with pytest.raises(ValueError, match="source workspace"):
        ResearchCopyManifest(
            replace(LOCAL, workspace_id="other"),
            SERVER,
            (item(),),
            TIME,
        )


@pytest.mark.parametrize(
    "value", ("/Users/alex/notes", "C:\\private\\note", "file://owner", "sk-synthetic")
)
def test_copy_identifiers_reject_private_paths_and_secret_shaped_values(value):
    with pytest.raises(ValueError):
        CopyOwnerRef(LOCAL, "note", value, version=1)


def test_device_overlays_and_mutable_content_are_ineligible():
    with pytest.raises(ValueError, match="item type"):
        CopyOwnerRef(LOCAL, "overlay", "device-record", version=1)
    with pytest.raises(ValueError, match="stable source revision"):
        ResearchCopyItem(CopyOwnerRef(LOCAL, "note", "41"), CopyTreatment.CONTENT)


def test_replacement_requires_destination_version_and_second_confirmation():
    existing = CopyOwnerRef(SERVER, "media", "101", version=2)
    with pytest.raises(ValueError, match="destination version"):
        item(
            conflict=CopyConflictChoice.REPLACE_DESTINATION,
            destination=replace(existing, version=None),
        )
    proposed = ResearchCopyManifest(
        LOCAL,
        SERVER,
        (item(conflict=CopyConflictChoice.REPLACE_DESTINATION, destination=existing),),
        TIME,
    )
    with pytest.raises(ValueError, match="confirm replacement"):
        ResearchCopyConfirmation(proposed.manifest_hash, confirmed=True).validate(
            proposed
        )
    with pytest.raises(ValueError, match="different manifest"):
        ResearchCopyConfirmation(
            "a" * 64, confirmed=True, replacement_confirmed=True
        ).validate(proposed)
    ResearchCopyConfirmation(
        proposed.manifest_hash, confirmed=True, replacement_confirmed=True
    ).validate(proposed)
