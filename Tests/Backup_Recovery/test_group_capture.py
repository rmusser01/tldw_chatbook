"""Selected coverage remains explicit without hiding ownership defects."""

from dataclasses import replace

import pytest

from tldw_chatbook.Backup_Recovery.capture import _capture_options
from tldw_chatbook.Backup_Recovery.inventory import classify_entries
from tldw_chatbook.Backup_Recovery.models import DiscoverySelections, StorageItem


def test_capture_accepts_an_explicit_group_selection():
    settings, selections, _, _ = _capture_options({"data_groups": ("settings",)})
    assert selections.data_groups == settings["data_groups"] == ("settings",)


@pytest.mark.parametrize("groups", [(), [], ("settings", "settings"), ("unknown",)])
def test_capture_refuses_empty_malformed_or_unknown_groups(groups):
    with pytest.raises(ValueError, match="invalid_backup_groups"):
        DiscoverySelections(data_groups=groups)


def _sources(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("[general]\nusers_name='Ada'\n")
    prompts = tmp_path / "prompts.db"
    prompts.write_bytes(b"owned prompt bytes")
    media = tmp_path / "media.db"
    media.write_bytes(b"owned media bytes")
    return (
        StorageItem("config", "profile:a:config", config, "included", ()),
        StorageItem(
            "db.prompts.primary",
            "profile:a:prompts",
            prompts,
            "included",
            ("profile:a:config",),
        ),
        StorageItem(
            "db.media.primary",
            "profile:a:media",
            media,
            "included",
            ("profile:a:config",),
        ),
    )


def test_selected_coverage_keeps_config_support_and_omits_other_payloads(tmp_path):
    from tldw_chatbook.Backup_Recovery.data_groups import group_for_owner
    from tldw_chatbook.Backup_Recovery.group_selection import select_inventory

    original = classify_entries(_sources(tmp_path))
    selection = (group_for_owner("db.prompts.primary"),)
    selected = select_inventory(original, selection)
    assert {row.owner for row in selected.items if row.status == "included"} == {
        "config",
        "db.prompts.primary",
    }
    assert selected.complete
    assert selected.scope_digest != original.scope_digest
    assert {row.logical_id for row in selected.items} == {
        row.logical_id for row in original.items
    }


def test_unselected_missing_data_does_not_make_selected_coverage_incomplete(tmp_path):
    from tldw_chatbook.Backup_Recovery.data_groups import group_for_owner
    from tldw_chatbook.Backup_Recovery.group_selection import select_inventory

    config, prompts, media = _sources(tmp_path)
    inventory = classify_entries(
        (config, prompts, replace(media, path=None, status="missing_required"))
    )
    selected = select_inventory(inventory, (group_for_owner(prompts.owner),))
    assert selected.complete
    assert (
        next(item for item in selected.items if item.owner == media.owner).status
        == "intentionally_excluded"
    )


def test_selection_does_not_hide_undeclared_aliases(tmp_path):
    from tldw_chatbook.Backup_Recovery.data_groups import group_for_owner
    from tldw_chatbook.Backup_Recovery.group_selection import select_inventory

    config, prompts, media = _sources(tmp_path)
    inventory = classify_entries((config, prompts, replace(media, path=prompts.path)))
    selected = select_inventory(inventory, (group_for_owner(prompts.owner),))
    assert not selected.complete
    assert "undeclared_alias" in selected.issues


def test_everything_preserves_existing_inventory_and_digest(tmp_path):
    from tldw_chatbook.Backup_Recovery.group_selection import select_inventory

    inventory = classify_entries(_sources(tmp_path))
    assert select_inventory(inventory, None) is inventory


def test_selected_capture_manifest_records_exact_scope_and_config_support(tmp_path):
    import json
    import shutil
    from threading import Event

    from tldw_chatbook.Backup_Recovery.capture import _manifest_for
    from tldw_chatbook.Backup_Recovery.group_selection import select_inventory
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits

    inventory = select_inventory(classify_entries(_sources(tmp_path)), ("prompts",))
    payloads = tmp_path / "payload"
    payloads.mkdir()
    staged = []
    for index, item in enumerate(inventory.items):
        if item.status == "included":
            path = payloads / str(index)
            shutil.copyfile(item.path, path)
            staged.append((item, path))
    doc = json.loads(
        _manifest_for(
            inventory,
            staged,
            {},
            {
                "root": tmp_path,
                "cancel": Event(),
                "mode": "exclude",
                "encrypted": False,
                "versions": {},
                "limits": ArchiveLimits(),
                "data_groups": ("prompts",),
            },
            (),
        )
    )
    assert doc["format_version"] == 2
    assert doc["consistency"] == "coherent"
    assert doc["group_scope"] == {
        "requested_groups": ["prompts"],
        "effective_groups": ["prompts"],
        "support_ids": ["profile:a:config"],
    }


def test_public_selected_prompt_capture_uses_native_maintenance(tmp_path):
    """Capture an offline profile after its actual producer releases all leases."""
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    script = r'''
import json
import os
import subprocess
import sys
from pathlib import Path
from threading import Event

from Tests import network_guard
network_guard.install()
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.capture_service import capture, preview_capture
from tldw_chatbook.Backup_Recovery.control_records import admission_authority

source = Path(os.environ['TLDW_CONFIG_PATH'])
data = Path(os.environ['XDG_DATA_HOME']) / 'fixture'
data.mkdir(mode=0o700)
source.write_text('[paths]\ndata_dir=' + json.dumps(str(data)) + '\n')
source.chmod(0o600)
admission_authority(bootstrap.default_bootstrap_root())

# Source services bind process-lived config participants. Let their real process
# exit before offline capture; a mounted live app would supply its own monitor.
producer = """
import os
from pathlib import Path
from Tests import network_guard
network_guard.install()
from Tests.Backup_Recovery.test_capture_service import _populate_required_dependencies
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase

baseline = preview_capture((Path(os.environ['TLDW_CONFIG_PATH']),), options={'allow_partial': True})
_populate_required_dependencies(baseline)
path = next(row.path for row in baseline.items if row.owner == 'db.prompts.primary')
store = PromptsDatabase(path, 'selected-backup-fixture')
try:
    store.add_prompt('Selected prompt', 'fixture', 'Retained prompt', user_prompt='Saved selected prompt bytes')
finally:
    store.close()
"""
prepared = subprocess.run([sys.executable, '-c', producer], capture_output=True, text=True, timeout=30)
assert prepared.returncode == 0, prepared.stderr[-3000:]
assert 'tldw_chatbook.config' not in sys.modules, 'offline capture imported a live config participant'

options = {'data_groups': ('prompts',), 'staging_parent': Path.home()}
preview = preview_capture((source,), options=options)
assert preview.complete, preview.issues
media = next(row.path for row in preview.items if row.owner == 'db.media.primary')
original = {path: (path.read_bytes(), path.stat().st_ino) for path in (media, source)}
result = capture((source,), preview.scope_digest, Path.home() / 'selected.tldw-backup.zip',
                 options=options, cancel=Event())
doc = json.loads(result.manifest_bytes)
assert doc['format_version'] == 2
assert doc['group_scope']['effective_groups'] == ['prompts']
assert {row['owner_id'] for row in doc['files']} == {'config', 'db.prompts.primary'}
prompt = next(row for row in doc['files'] if row['owner_id'] == 'db.prompts.primary')
assert b'Saved selected prompt bytes' in (result.root / prompt['payload']).read_bytes()
assert all((path.read_bytes(), path.stat().st_ino) == saved for path, saved in original.items())
print('retired and reopened')
'''
    _run(tmp_path, "selected-prompts", "success", script=script, timeout=90)
