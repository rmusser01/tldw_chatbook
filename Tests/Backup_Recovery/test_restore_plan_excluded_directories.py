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


@pytest.mark.parametrize("archived", [True, False], ids=["archived", "added"])
def test_replace_keeps_a_plugin_reset_aside_whole_even_when_the_archive_predates_it(
    tmp_path, archived
):
    """TASK-40308 review r14plg-rev1: a post-restore reset sets the restored plugin
    registry, download records and downloads aside in ``reset-<id>/``, which the
    plugins owner excludes whole. A later replace restore keeps it whole, also
    from an archive made before the reset; it is never listed as newer downloads
    (``refuse_added_fetch_folders``) and nothing in it is deleted."""
    from tldw_chatbook.Backup_Recovery.models import FileMetadata
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

    def mutate(doc):
        _rows(archived)(doc)
        for row in doc["producer_inventory"] + doc["files"] + doc["owners"]:
            row["owner_id"] = "plugins"
        doc["owners"][0]["schema_version"] = 5  # The registry catalog's version.
        if archived:
            doc["producer_inventory"][-1]["logical_id"] = "aside"
            doc["exclusions"] = [
                {"logical_id": "aside", "reason": "intentionally_excluded"}
            ]

    archive = sealed(tmp_path, mutate=mutate)
    destination = tmp_path / "live"
    aside = destination / "reset-0123456789abcdef"
    for relative in ("registry.sqlite3", "staging/old/repository.git/HEAD"):
        (aside / relative).parent.mkdir(parents=True, exist_ok=True)
        (aside / relative).write_text("never listed")

    def metadata(relative, kind):
        return FileMetadata(
            1,
            "root",
            relative,
            None if not relative else "root",
            kind,
            0o700,
            0,
            "private",
        )

    target = Inventory(
        (
            StorageItem(
                "plugins",
                "root",
                destination,
                "included_directory",
                (),
                metadata=metadata("", "directory"),
            ),
            StorageItem(
                "plugins",
                "aside",
                aside,
                "intentionally_excluded",
                ("root",),
                metadata=metadata(aside.name, "directory"),
            ),
        ),
        True,
        "local",
        (),
    )
    plan = plan_restore(
        archive, mode="replace", destinations={"root": destination}, target=target
    )
    assert ("aside", aside) in plan.preserve
    assert (aside / "registry.sqlite3").read_text() == "never listed"
    assert (aside / "staging/old/repository.git/HEAD").read_text() == "never listed"
