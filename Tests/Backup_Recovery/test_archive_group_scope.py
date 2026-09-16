"""Archive group scope is versioned metadata, never destination authority."""

import json

import pytest

from Tests.Backup_Recovery.test_archive_reader import manifest
from tldw_chatbook.Backup_Recovery.archive_reader import _manifest
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits


def _group_manifest():
    doc = manifest()
    doc["format_version"] = 2
    doc["owners"][0]["owner_id"] = "config"
    doc["files"][0]["owner_id"] = "config"
    doc["directories"][0]["synthetic"] = True
    doc["producer_inventory"] = [
        {
            "logical_id": "root",
            "owner_id": "config",
            "status": "included_directory",
            "dependencies": [],
        },
        {
            "logical_id": "file",
            "owner_id": "config",
            "status": "included",
            "dependencies": [],
        },
    ]
    doc["group_scope"] = {
        "requested_groups": ["settings"],
        "effective_groups": ["settings"],
        "support_ids": [],
    }
    return doc


def _read(doc):
    return _manifest(json.dumps(doc).encode(), ArchiveLimits(), encrypted=False)


def test_version_two_reads_explicit_group_scope():
    doc = _read(_group_manifest())
    assert doc.group_scope.requested_groups == ("settings",)
    assert doc.group_scope.effective_groups == ("settings",)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.pop("group_scope"),
        lambda d: d.update(producer_inventory=[]),
        lambda d: d["group_scope"].update(requested_groups=[]),
        lambda d: d["group_scope"].update(requested_groups=["unknown"]),
        lambda d: d["group_scope"].update(effective_groups=[]),
        lambda d: d["group_scope"].update(support_ids=["file"]),
        lambda d: d["group_scope"].update(extra="untrusted policy"),
    ],
)
def test_version_two_rejects_missing_or_inconsistent_group_scope(mutation):
    doc = _group_manifest()
    mutation(doc)
    with pytest.raises(ValueError):
        _read(doc)


def test_version_one_still_reads_without_new_scope():
    assert _read(manifest()).format_version == 1


def test_version_one_cannot_claim_group_scope():
    doc = _group_manifest()
    doc["format_version"] = 1
    with pytest.raises(ValueError):
        _read(doc)


def test_archive_cannot_attach_an_unrequested_group_payload():
    doc = _group_manifest()
    doc["owners"].append(
        {"owner_id": "db.prompts.primary", "schema_version": 1, "capabilities": []}
    )
    doc["files"].append(
        {
            **doc["files"][0],
            "logical_id": "prompts",
            "relative_path": "prompts.db",
            "payload": "payload/2",
            "owner_id": "db.prompts.primary",
        }
    )
    doc["producer_inventory"].append(
        {
            "logical_id": "prompts",
            "owner_id": "db.prompts.primary",
            "status": "included",
            "dependencies": [],
        }
    )
    doc["dependency_groups"][0]["members"].append("prompts")
    with pytest.raises(ValueError, match="group_scope"):
        _read(doc)
