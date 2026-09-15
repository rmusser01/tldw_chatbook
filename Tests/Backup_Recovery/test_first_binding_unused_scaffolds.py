"""Known empty startup scaffolds remain observed during first profile binding."""

import hashlib
from dataclasses import replace

import pytest

from tldw_chatbook.Backup_Recovery.config_adapter import recovery_adapters
from tldw_chatbook.Backup_Recovery.models import (
    DISCOVERY_CONTEXT_KEY,
    DiscoveryContext,
    Inventory,
    StorageItem,
)
from tldw_chatbook.Backup_Recovery.profile_paths import database_path
from tldw_chatbook.Backup_Recovery.replacement import _first_config_container


@pytest.fixture(params=["research.paste_staging", "collections.archives"])
def scaffold(tmp_path, request):
    parent = tmp_path / "config"
    parent.mkdir(mode=0o700)
    selector = parent / "config.toml"
    selector.write_bytes(b'[general]\nusers_name="default_user"\n')
    selector.chmod(0o600)
    data = parent / "data"
    profile = data / "default_user"
    profile.mkdir(mode=0o700, parents=True)
    config = {
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, "fixture"),
        "paths": {"data_dir": str(data)},
    }
    owner = request.param
    root = profile / (
        "research_paste_staging"
        if owner == "research.paste_staging"
        else "collections_archives"
    )
    root.mkdir(mode=0o700)
    if owner == "research.paste_staging":
        control = root / "index.json"
        control.write_bytes(b'{"operations":{},"schema_version":1}')
    else:
        identity = (
            str(profile.resolve())
            + "\0"
            + str(database_path(config, "library_collections_db_path").resolve())
        )
        authority = root / hashlib.sha256(identity.encode()).hexdigest()[:16]
        authority.mkdir(mode=0o700)
        control = authority / ".lifecycle.lock"
        control.write_bytes(b"\0")
    control.chmod(0o600)
    adapter = next(row for row in recovery_adapters() if row.owner_id == owner)
    rows = adapter.discover(config)
    assert rows and all(row.status == "unused" for row in rows)
    inventory = Inventory(
        (
            StorageItem("config", "profile:fixture:config", selector, "included", ()),
            *rows,
        ),
        True,
        "scope",
        (),
    )
    return selector, root, control, inventory


def prove(scaffold, *, inventory=None, names=("scaffold",)):
    selector, root, _, observed = scaffold
    return _first_config_container(
        selector,
        inventory or observed,
        names,
        {"scaffold": {"roots": [str(root)]}},
        (),
        (selector,),
        (),
    )


def test_first_binding_observes_known_unused_rows_without_payload_coverage(scaffold):
    _, root, control, inventory = scaffold
    _, state = prove(scaffold)
    paths = {row[0] for row in state}
    assert {root, control} <= paths
    assert all(row.status == "unused" for row in inventory.items[1:])


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_first_binding_unused_root_does_not_cover_new_children(scaffold, kind):
    prove(scaffold)
    child = scaffold[1] / "unreviewed"
    if kind == "file":
        child.write_bytes(b"new durable payload")
    else:
        child.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="^replacement_config_container_unverified$"):
        prove(scaffold)


def test_first_binding_detects_changed_unused_control_bytes(scaffold):
    first = prove(scaffold)
    scaffold[2].write_bytes(b"changed control")
    assert prove(scaffold) != first


def test_first_binding_does_not_adopt_foreign_unused_namespace(scaffold):
    with pytest.raises(ValueError, match="^replacement_config_container_unverified$"):
        prove(scaffold, names=())


def test_first_binding_does_not_adopt_other_unused_owner(scaffold):
    observed = scaffold[3]
    changed = replace(
        observed,
        items=(
            observed.items[0],
            *(replace(row, owner="unrecognized") for row in observed.items[1:]),
        ),
    )
    with pytest.raises(ValueError, match="^replacement_config_container_unverified$"):
        prove(scaffold, inventory=changed)
