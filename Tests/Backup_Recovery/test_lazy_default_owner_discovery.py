"""Observable absence of the installed app's six lazily opened SQLite owners."""

import importlib
from pathlib import Path

import pytest

from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY, DiscoveryContext
from tldw_chatbook.Backup_Recovery.profile_paths import user_data_dir


OWNERS = (
    ("Kanban_Interop", "kanban.local", None),
    ("Notifications", "notifications.client", "notifications_db_path"),
    ("Notifications", "runtime.event_state", None),
    ("Sync_Interop", "runtime.sync_state", None),
    ("Research_Interop", "research.local", "research_db_path"),
    ("Writing_Interop", "writing.local", "writing_db_path"),
)


@pytest.fixture(params=OWNERS, ids=[row[1] for row in OWNERS])
def lazy_owner(request, tmp_path):
    package, owner_id, setting_name = request.param
    adapter = next(
        item
        for item in importlib.import_module(
            f"tldw_chatbook.{package}.recovery"
        ).recovery_adapters()
        if item.owner_id == owner_id
    )
    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    config = {
        "paths": {"data_dir": str(data)},
        "general": {"users_name": "fixture"},
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(tmp_path / "config.toml", "fixture"),
    }
    user_data_dir(config).mkdir(mode=0o700)
    (item,) = adapter.discover(config)
    return adapter, config, item.path, setting_name


def test_absent_lazy_default_is_unused_without_creating_store(lazy_owner):
    adapter, config, path, _ = lazy_owner
    (item,) = adapter.discover(config)

    assert item.status == "unused"
    assert not path.exists()
    assert not tuple(path.parent.iterdir())


def test_existing_lazy_default_remains_included(lazy_owner):
    adapter, config, path, _ = lazy_owner
    # Discovery checks presence, while the unchanged schema policy validates
    # every included candidate during capture and restore.
    path.write_bytes(b"existing source must still undergo SQLite validation")
    before = path.read_bytes()

    (item,) = adapter.discover(config)

    assert item.status == "included"
    assert adapter.schema_policy() is not None
    assert path.read_bytes() == before


@pytest.mark.parametrize("suffix", ("-wal", "-shm", "-journal"))
@pytest.mark.parametrize("kind", ("file", "directory"))
def test_missing_lazy_default_with_residue_is_not_unused(lazy_owner, suffix, kind):
    adapter, config, path, _ = lazy_owner
    residue = Path(str(path) + suffix)
    if kind == "file":
        residue.write_bytes(b"retained SQLite residue")
    else:
        residue.mkdir(mode=0o700)

    (item,) = adapter.discover(config)

    assert item.status in {"missing_required", "unavailable"}
    assert residue.exists()
    assert not path.exists()


@pytest.mark.parametrize("selection", ("custom", "explicit_default"))
@pytest.mark.parametrize(
    "lazy_owner",
    [row for row in OWNERS if row[2] is not None],
    ids=[row[1] for row in OWNERS if row[2] is not None],
    indirect=True,
)
def test_explicit_missing_lazy_selector_is_required(lazy_owner, selection):
    adapter, config, path, setting_name = lazy_owner
    selected = path if selection == "explicit_default" else path.parent / "custom.db"
    config["database"] = {setting_name: str(selected)}

    (item,) = adapter.discover(config)

    assert item.path == selected
    assert item.status == "missing_required"
    assert not selected.exists()


def test_directory_at_lazy_default_is_not_a_missing_unused_store(lazy_owner):
    adapter, config, path, _ = lazy_owner
    path.mkdir(mode=0o700)

    (item,) = adapter.discover(config)

    assert item.status == "unsupported"
