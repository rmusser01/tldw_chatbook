"""Replace restore and an owner's excluded directory listed as one row (TASK-40132).

Its own module, free of Hypothesis, so the Architecture guard lane runs it.
"""

import pytest

from Tests.Backup_Recovery.test_archive_reader import sealed
from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem


def _rows(archived):
    def mutate(doc):
        doc["producer_inventory"] = [
            {
                "logical_id": "root",
                "owner_id": "ui.state",
                "status": "included_directory",
                "dependencies": [],
                "shared_group": None,
            },
            {
                "logical_id": "file",
                "owner_id": "ui.state",
                "status": "included",
                "dependencies": ["root"],
                "shared_group": None,
            },
        ]
        if archived:
            doc["producer_inventory"].append(
                {
                    "logical_id": "kept",
                    "owner_id": "ui.state",
                    "status": "intentionally_excluded",
                    "dependencies": ["root"],
                    "shared_group": None,
                }
            )
            doc["exclusions"] = [
                {"logical_id": "kept", "reason": "intentionally_excluded"}
            ]

    return mutate


@pytest.mark.parametrize("archived", [True, False], ids=["archived", "added"])
def test_replace_keeps_an_excluded_directory_whole_without_listing_it(
    tmp_path, archived
):
    """An owner can exclude a directory as one row without listing it (the plugin
    fetch folder's repository.git). When the archive excluded it too, replace
    preserves it whole, so its unlisted children are kept with it. When the
    archive has no such row (it appeared since the backup), its children stay
    unknown and the review refuses (r13g-rev3 M-2), not Start."""
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

    archive = sealed(tmp_path, mutate=_rows(archived))
    destination = tmp_path / "live"
    kept = destination / "kept"
    kept.mkdir(parents=True)
    (kept / "object").write_text("never listed")
    target = Inventory(
        (
            StorageItem("ui.state", "root", destination, "included_directory", ()),
            StorageItem("ui.state", "kept", kept, "intentionally_excluded", ("root",)),
        ),
        True,
        "local",
        (),
    )
    if archived:
        plan = plan_restore(
            archive, mode="replace", destinations={"root": destination}, target=target
        )
        assert ("kept", kept) in plan.preserve
    else:
        with pytest.raises(ValueError, match="target_owner_unclassified"):
            plan_restore(
                archive, mode="replace", destinations={"root": destination}, target=target
            )
    assert (kept / "object").read_text() == "never listed"
