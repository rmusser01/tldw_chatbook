"""Declared shared trees must stay identical rather than merge during restore."""

import json
from copy import deepcopy
from pathlib import Path
from threading import Event

import pytest

from Tests.Backup_Recovery.test_restore_plan import producer, sealed
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.staging import stage_restore


def shared_archive(tmp_path, *, difference=None, empty=False):
    """Build untrusted archive inputs for the planner's alias boundary."""

    def declarations(doc):
        producer(doc)
        doc["files"][0]["metadata"] = {"version": 1, "mode": 0o600, "mtime_ns": 9}
        doc["directories"].append(
            {
                "logical_id": "empty",
                "root_id": "root",
                "parent_id": "root",
                "relative_path": "empty",
                "metadata": {"version": 1, "mode": 0o700, "mtime_ns": 7},
            }
        )
        doc["producer_inventory"].append(
            {
                "logical_id": "empty",
                "owner_id": "ui.state",
                "status": "included_directory",
                "dependencies": ["root"],
                "shared_group": None,
            }
        )
        for item in doc["producer_inventory"]:
            item["shared_group"] = "shared-" + item["logical_id"]
        for key in ("directories", "files", "producer_inventory"):
            for original in tuple(doc[key]):
                row = deepcopy(original)
                row["logical_id"] = "other-" + row["logical_id"]
                if key == "producer_inventory":
                    row["dependencies"] = [
                        "other-" + value for value in row["dependencies"]
                    ]
                else:
                    row["root_id"] = "other-root"
                    if row["parent_id"] is not None:
                        row["parent_id"] = "other-" + row["parent_id"]
                    if key == "files":
                        row["payload"] = "payload/2"
                doc[key].append(row)
        if empty:
            doc["files"] = []
            doc["directories"] = [
                row for row in doc["directories"] if row["parent_id"] is None
            ]
            doc["producer_inventory"] = [
                row
                for row in doc["producer_inventory"]
                if row["logical_id"] in {"root", "other-root"}
            ]
        by_id = {row["logical_id"]: row for row in doc["producer_inventory"]}
        rows = {row["logical_id"]: row for row in (*doc["directories"], *doc["files"])}
        if difference in {"root-metadata", "child-metadata", "file-metadata"}:
            key = {
                "root-metadata": "other-root",
                "child-metadata": "other-empty",
                "file-metadata": "other-file",
            }[difference]
            rows[key]["metadata"]["mtime_ns"] += 1
        elif difference == "undeclared":
            by_id["other-root"]["shared_group"] = None
        elif difference == "child-group":
            by_id["other-empty"]["shared_group"] = "unrelated-directory"
        elif difference == "undeclared-children":
            by_id["empty"]["shared_group"] = None
            by_id["other-empty"]["shared_group"] = None
        elif difference == "child-owner":
            doc["owners"].append(
                {"owner_id": "config", "schema_version": 1, "capabilities": []}
            )
            by_id["other-empty"]["owner_id"] = "config"
        elif difference == "extra-member":
            row = deepcopy(rows["other-empty"])
            row.update(logical_id="extra", relative_path="extra")
            doc["directories"].append(row)
            doc["producer_inventory"].append(
                {
                    "logical_id": "extra",
                    "owner_id": "ui.state",
                    "status": "included_directory",
                    "dependencies": ["other-root"],
                    "shared_group": None,
                }
            )
        doc["dependency_groups"] = [
            {
                "group_id": "tree",
                "complete": True,
                "members": [row["logical_id"] for row in doc["producer_inventory"]],
            }
        ]

    return sealed(tmp_path, mutate=declarations)


def test_shared_concrete_tree_stages_one_physical_candidate(tmp_path):
    archive = shared_archive(tmp_path)
    destination = tmp_path / "new"
    plan = plan_restore(
        archive,
        mode="isolated",
        target=None,
        destinations={"root": destination, "other-root": destination},
    )
    candidate = stage_restore(archive, plan, tmp_path / "stage", Event())
    rows = json.loads((candidate / "candidate.json").read_bytes())["artifacts"]
    by_id = {row["logical_id"]: row for row in rows}
    for first, second in (
        ("root", "other-root"),
        ("empty", "other-empty"),
        ("file", "other-file"),
    ):
        assert by_id[first]["candidate"] == by_id[second]["candidate"]
        assert by_id[first]["applied_metadata"] == by_id[second]["applied_metadata"]
    assert Path(by_id["file"]["candidate"]).read_bytes() == b"durable"
    assert len([row for row in rows if row["publication_unit"]]) == 1
    assert len(dict(plan.restore)) == 6
    assert not destination.exists()


@pytest.mark.parametrize(
    "difference",
    [
        "root-metadata",
        "child-metadata",
        "file-metadata",
        "undeclared",
        "child-group",
        "undeclared-children",
        "child-owner",
        "extra-member",
        "casefold",
    ],
)
def test_shared_root_declaration_cannot_union_different_trees(tmp_path, difference):
    archive = shared_archive(tmp_path, difference=difference)
    with pytest.raises(ValueError, match="destination_collision"):
        plan_restore(
            archive,
            mode="isolated",
            target=None,
            destinations={
                "root": tmp_path / "new",
                "other-root": tmp_path / ("NEW" if difference == "casefold" else "new"),
            },
        )
    assert not (tmp_path / "new").exists()


def test_shared_empty_roots_cannot_be_split(tmp_path):
    archive = shared_archive(tmp_path, empty=True)
    with pytest.raises(ValueError, match="shared_target_split"):
        plan_restore(
            archive,
            mode="isolated",
            target=None,
            destinations={"root": tmp_path / "one", "other-root": tmp_path / "two"},
        )
