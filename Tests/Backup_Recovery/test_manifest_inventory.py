"""Producer coverage remains explicit inert data in recovery archives."""

import json

import pytest

from Tests.Backup_Recovery.test_archive_reader import manifest
from tldw_chatbook.Backup_Recovery.archive_reader import _manifest
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits


def producer_manifest():
    doc = manifest()
    doc["producer_inventory"] = [
        {
            "logical_id": "root",
            "owner_id": "notes",
            "status": "included_directory",
            "dependencies": [],
            "shared_group": None,
        },
        {
            "logical_id": "file",
            "owner_id": "notes",
            "status": "included",
            "dependencies": ["root"],
            "shared_group": None,
        },
        {
            "logical_id": "omitted",
            "owner_id": "notes",
            "status": "unused",
            "dependencies": [],
            "shared_group": None,
        },
    ]
    doc["exclusions"] = [{"logical_id": "omitted", "reason": "unused"}]
    doc["files"][0]["metadata"] = {"version": 1, "mode": 0o600, "mtime_ns": 123}
    return doc


def parse(doc):
    return _manifest(json.dumps(doc).encode(), ArchiveLimits(), False)


def test_explicit_producer_inventory_and_file_metadata_round_trip():
    doc = parse(producer_manifest())
    assert doc.producer_inventory[1].dependencies == ("root",)
    assert doc.files[0].metadata.mode == 0o600
    assert doc.files[0].metadata.mtime_ns == 123


def test_synthetic_container_is_explicit_and_only_allowed_at_root():
    doc = producer_manifest()
    doc["directories"][0]["synthetic"] = True
    assert parse(doc).directories[0].synthetic is True
    doc["directories"].append(
        {
            "logical_id": "child",
            "root_id": "root",
            "parent_id": "root",
            "relative_path": "child",
            "metadata": {"version": 1, "mode": 0o700, "mtime_ns": 0},
            "synthetic": True,
        }
    )
    doc["producer_inventory"].append(
        {
            "logical_id": "child",
            "owner_id": "notes",
            "status": "included_directory",
            "dependencies": ["root"],
        }
    )
    with pytest.raises(ValueError, match="invalid_synthetic_root"):
        parse(doc)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "missing",
        "owner",
        "status",
        "dependency",
        "exclusion",
        "unexplained",
    ],
)
def test_producer_inventory_rejects_conflicting_or_missing_records(mutation):
    doc = producer_manifest()
    items = doc["producer_inventory"]
    if mutation == "duplicate":
        items.append(items[0])
    elif mutation == "missing":
        items.pop(1)
    elif mutation == "owner":
        items[1]["owner_id"] = "unknown"
    elif mutation == "status":
        items[1]["status"] = "included_directory"
    elif mutation == "dependency":
        items[1]["dependencies"] = ["missing"]
    elif mutation == "exclusion":
        items[2]["status"] = "unsupported"
    else:
        items.append(
            {
                "logical_id": "unexplained",
                "owner_id": "notes",
                "status": "included",
                "dependencies": [],
            }
        )
    with pytest.raises(ValueError):
        parse(doc)


def test_legacy_manifest_remains_readable_without_invented_metadata():
    doc = parse(manifest())
    assert doc.producer_inventory == ()
    assert doc.files[0].metadata is None


def test_coherent_manifest_cannot_hide_unsupported_producer_coverage():
    doc = producer_manifest()
    doc["producer_inventory"][2]["status"] = "unsupported"
    doc["exclusions"][0]["reason"] = "unsupported"
    with pytest.raises(ValueError, match="producer_coverage_incomplete"):
        parse(doc)
    doc["consistency"] = "partial"
    assert parse(doc).consistency == "partial"


@pytest.mark.parametrize("mismatch", [False, True])
def test_shared_file_records_require_identical_payload_bytes(mismatch):
    doc = producer_manifest()
    alias = dict(
        doc["files"][0],
        logical_id="alias",
        relative_path="alias",
        payload="payload/alias",
    )
    if mismatch:
        alias["sha256"] = "0" * 64
    doc["files"].append(alias)
    doc["dependency_groups"][0]["members"].append("alias")
    doc["producer_inventory"][1]["shared_group"] = "shared"
    doc["producer_inventory"].append(
        {
            "logical_id": "alias",
            "owner_id": "notes",
            "status": "included",
            "dependencies": ["root"],
            "shared_group": "shared",
        }
    )
    if mismatch:
        with pytest.raises(ValueError, match="shared_payload_mismatch"):
            parse(doc)
    else:
        assert len(parse(doc).files) == 2
