"""Synthetic file parents can contain independent selected owner directories."""

import json
from pathlib import Path
from threading import Event

import pytest

from Tests.Backup_Recovery.test_restore_plan import sealed
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.staging import stage_restore


def _nested_archive(
    tmp_path, *, synthetic=True, filename="note.txt", replacement=False
):
    def topology(doc):
        doc["directories"][0]["synthetic"] = synthetic
        doc["files"][0]["relative_path"] = filename
        doc["directories"].append(
            {
                **doc["directories"][0],
                "logical_id": "child-root",
                "root_id": "child-root",
                "synthetic": False,
            }
        )
        doc["files"].append(
            {
                **doc["files"][0],
                "logical_id": "child-file",
                "root_id": "child-root",
                "parent_id": "child-root",
                "relative_path": "saved.txt",
                "payload": "payload/2",
            }
        )
        doc["dependency_groups"][0]["members"].append("child-file")
        if replacement:
            from Tests.Backup_Recovery.test_restore_plan import producer

            producer(doc)
            doc["producer_inventory"].extend(
                [
                    {
                        "logical_id": key,
                        "owner_id": "ui.state",
                        "status": status,
                        "dependencies": dependencies,
                        "shared_group": None,
                    }
                    for key, status, dependencies in (
                        ("child-root", "included_directory", []),
                        ("child-file", "included", ["child-root"]),
                    )
                ]
            )

    return sealed(tmp_path, mutate=topology)


@pytest.mark.parametrize("child", ["skills", "a/skills"])
def test_synthetic_parent_stages_independent_owner_child_in_one_candidate(
    tmp_path, child
):
    archive = _nested_archive(tmp_path)
    destination = tmp_path / "new"
    plan = plan_restore(
        archive,
        mode="isolated",
        destinations={"root": destination, "child-root": destination / child},
        target=None,
    )
    from tldw_chatbook.Backup_Recovery.control_records import register_pending
    from tldw_chatbook.Backup_Recovery.journal import Journal
    from tldw_chatbook.Backup_Recovery.publication import publish_candidate

    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    journal = Journal(control, "nested")
    stage = stage_restore(archive, plan, tmp_path / "work", Event(), journal=journal)
    rows = json.loads((stage / "candidate.json").read_bytes())["artifacts"]
    units = [row for row in rows if row["publication_unit"]]
    assert len(units) == 1
    candidate = Path(units[0]["candidate"])
    assert (candidate / "note.txt").read_bytes() == b"durable"
    assert (candidate / child / "saved.txt").read_bytes() == b"durable"
    assert not destination.exists()
    bootstrap = tmp_path / "bootstrap"
    selector = destination / "config.toml"
    register_pending(bootstrap, "nested", ("profile",), control, (selector,))
    journal.prepare_publication(
        stage,
        plan,
        bootstrap_root=bootstrap,
        namespaces=("profile",),
        selectors=(selector,),
        generation="nested",
    )
    publish_candidate(stage, plan, journal, None)
    assert (destination / child / "saved.txt").read_bytes() == b"durable"
    assert (destination / "note.txt").read_bytes() == b"durable"


@pytest.mark.parametrize(
    ("synthetic", "filename", "parent", "child"),
    [
        (False, "note.txt", "new", "new/skills"),
        (True, "skills", "new", "new/skills"),
        (True, "skills", "new", "new/skills/nested"),
        (True, "Skills", "New", "new/skills"),
    ],
)
def test_overlapping_owned_bytes_still_refuse(
    tmp_path, synthetic, filename, parent, child
):
    archive = _nested_archive(tmp_path, synthetic=synthetic, filename=filename)
    with pytest.raises(ValueError, match="destination_overlap|destination_collision"):
        plan_restore(
            archive,
            mode="isolated",
            destinations={"root": tmp_path / parent, "child-root": tmp_path / child},
            target=None,
        )
    assert not (tmp_path / parent).exists()


def test_existing_synthetic_parent_keeps_required_missing_local_container(tmp_path):
    from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem

    archive = _nested_archive(tmp_path, replacement=True)
    destination = tmp_path / "existing"
    destination.mkdir(mode=0o700)
    (destination / "note.txt").write_bytes(b"previous bytes")
    target = Inventory(
        (
            StorageItem("ui.state", "root", destination, "included_directory", ()),
            StorageItem(
                "ui.state", "file", destination / "note.txt", "included", ("root",)
            ),
        ),
        True,
        "local",
        (),
    )
    plan = plan_restore(
        archive,
        mode="replace",
        destinations={"root": destination, "child-root": destination / "a" / "skills"},
        target=target,
    )
    assert destination / "a" in dict(plan.containers).values()
    stage = stage_restore(archive, plan, tmp_path / "work", Event())
    document = json.loads((stage / "candidate.json").read_bytes())
    assert [row["destination"] for row in document["containers"]] == [
        str(destination / "a")
    ]
    assert {
        row["destination"] for row in document["artifacts"] if row["publication_unit"]
    } == {str(destination / "note.txt"), str(destination / "a" / "skills")}
    assert (destination / "note.txt").read_bytes() == b"previous bytes"
