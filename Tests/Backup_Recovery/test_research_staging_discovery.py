"""Empty upstream staging is known; new durable payload scope stays explicit."""

import pytest

from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY, DiscoveryContext


@pytest.mark.parametrize(
    "kind",
    [
        "absent",
        "empty",
        "empty_index",
        "pending_index",
        "invalid_index",
        "index_and_payload",
        "payload",
        "file",
        "symlink",
    ],
)
def test_research_staging_does_not_hide_unreviewed_payloads(tmp_path, kind):
    from tldw_chatbook.Backup_Recovery.config_adapter import recovery_adapters

    config = {
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(tmp_path / "config.toml", "fixture"),
        "paths": {"data_dir": str(tmp_path)},
    }
    parent = tmp_path / "default_user"
    parent.mkdir(mode=0o700)
    root = parent / "research_paste_staging"
    if kind in {
        "empty",
        "empty_index",
        "pending_index",
        "invalid_index",
        "index_and_payload",
        "payload",
    }:
        root.mkdir(mode=0o700)
        if kind in {"empty_index", "index_and_payload"}:
            (root / "index.json").write_bytes(b'{"operations":{},"schema_version":1}')
        elif kind == "pending_index":
            (root / "index.json").write_bytes(
                b'{"operations":{"pending":"operation"},"schema_version":1}'
            )
        elif kind == "invalid_index":
            (root / "index.json").write_bytes(b'{"operations":{},"schema_version":2}')
        if kind in {"payload", "index_and_payload"}:
            (root / "pending.txt").write_text("Pending paste must remain visible")
    elif kind == "file":
        root.write_text("not a directory")
    elif kind == "symlink":
        root.symlink_to(tmp_path, target_is_directory=True)
    owner = next(
        a for a in recovery_adapters() if a.owner_id == "research.paste_staging"
    )
    items = owner.discover(config)
    assert all(
        item.status
        == ("unused" if kind in {"absent", "empty", "empty_index"} else "unsupported")
        for item in items
    )
    assert items[0].path == root
