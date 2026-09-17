"""Existing restore commands review the installed selectable data groups."""

import builtins

import pytest

from tldw_chatbook.Backup_Recovery.launcher import _parser


def test_restore_defaults_to_everything():
    args = _parser().parse_args(["restore", "archive.zip", "--isolated"])
    assert args.group is None


def test_restore_accepts_repeated_installed_groups():
    args = _parser().parse_args(
        [
            "restore",
            "archive.zip",
            "--isolated",
            "--group",
            "prompts",
            "--group",
            "library",
        ]
    )
    assert args.group == ["prompts", "library"]


@pytest.mark.parametrize("value", ["not-installed", ""])
def test_restore_rejects_unknown_or_empty_groups(value):
    with pytest.raises(SystemExit) as error:
        _parser().parse_args(["restore", "archive.zip", "--isolated", "--group", value])
    assert error.value.code == 2


def test_restore_group_help_uses_catalog_without_loading_normal_startup(
    monkeypatch, capsys
):
    from tldw_chatbook.Backup_Recovery.data_groups import BACKUP_GROUPS

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name not in {"tldw_chatbook.app", "tldw_chatbook.config"}, name
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(SystemExit) as error:
        _parser().parse_args(["restore", "--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert all(group.group_id in output for group in BACKUP_GROUPS)
    assert "Everything" in output


def test_manual_extraction_keeps_archive_dependency_group_ids():
    args = _parser().parse_args(
        [
            "extract",
            "archive.zip",
            "--group",
            "legacy-atomic-cohort",
            "--destination",
            "out",
        ]
    )
    assert args.group == ["legacy-atomic-cohort"]


@pytest.mark.parametrize("selection", [("prompts",), None])
def test_selected_replacement_forwards_target_config_and_reviews_required_groups(
    tmp_path, monkeypatch, capsys, selection
):
    from tldw_chatbook.Backup_Recovery.launcher import _restore
    from tldw_chatbook.Backup_Recovery.models import Inventory
    from tldw_chatbook.Backup_Recovery.restore_plan import RestorePlan

    selector = tmp_path / "local.toml"
    destination = tmp_path / "prompts"
    target = Inventory((), True, "target", ())
    seen = {}

    class Service:
        def start_inspection(self, archive, *, password):
            return "inspection"

        def wait(self, operation):
            return {"state": "succeeded"}

        def summary(self, inspection):
            return {
                "profile_ids": ("source",),
                "group_scope": {"effective_groups": ("prompts",)},
            }

        def preview_backup(self, configs, *, options):
            assert configs == (selector,) and options == {}
            return target

        def preview_restore(self, inspection, **choices):
            seen.update(choices)
            return RestorePlan(
                archive_digest="reviewed",
                mode="replace",
                restore=(),
                retire=(),
                preserve=(),
                target_fingerprint="local",
                requested_groups=selection,
                effective_groups=("library", "prompts"),
                required_groups=("library",),
            )

        def replacement_capability(self, plan):
            return True, ""

    args = _parser().parse_args(
        [
            "restore",
            "archive.zip",
            "--replace",
            "--target-config",
            str(selector),
            "--destination",
            "prompt-root=" + str(destination),
        ]
    )
    args.group = list(selection) if selection else None

    def review(prompt):
        shown = capsys.readouterr().out
        assert '"effective_groups": [\n    "library",\n    "prompts"\n  ]' in shown
        assert '"required_groups": [\n    "library"\n  ]' in shown
        if selection is None:
            assert "Everything" in shown
        return "cancel"

    monkeypatch.setattr(builtins, "input", review)
    assert _restore(Service(), args) == 0
    assert seen["data_groups"] == selection
    assert seen["target_configs"] == {"source": selector}
    assert seen["destinations"] == {"prompt-root": destination}
    assert seen["target"] is target


def test_selected_replacement_refuses_ambiguous_target_profile_before_discovery(
    tmp_path,
):
    from tldw_chatbook.Backup_Recovery.launcher import _restore

    class Service:
        def start_inspection(self, archive, *, password):
            return "inspection"

        def wait(self, operation):
            return {"state": "succeeded"}

        def summary(self, inspection):
            return {"profile_ids": ("first", "second"), "group_scope": None}

        def preview_backup(self, *args, **kwargs):
            pytest.fail("Ambiguous target must be refused before local discovery")

    args = _parser().parse_args(
        [
            "restore",
            "archive.zip",
            "--replace",
            "--target-config",
            str(tmp_path / "local.toml"),
        ]
    )
    args.group = ["prompts"]
    with pytest.raises(ValueError, match="target_unverified"):
        _restore(Service(), args)


_ISOLATED = r"""
import builtins, contextlib, io, json, os, sys, tomllib
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
from Tests.Backup_Recovery.test_restore_plan import sealed
from tldw_chatbook.Backup_Recovery.launcher import recovery_main
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

decision, selection = sys.argv[1:]
selector = Path(os.environ['TLDW_CONFIG_PATH'])
selector.write_bytes(b'broken = [')
selector.chmod(0o600)
original_bytes, original_inode = selector.read_bytes(), selector.stat().st_ino
def config_manifest(doc):
    doc['owners'][0]['owner_id'] = 'config'
    doc['files'][0].update(owner_id='config', logical_id='profile:profile:config', relative_path='config.toml')
    doc['directories'][0]['synthetic'] = True
    doc['producer_inventory'] = [
        dict(logical_id='profile:profile:config', owner_id='config', status='included', dependencies=[]),
        dict(logical_id='root', owner_id='config', status='included_directory', dependencies=[]),
    ]
    doc['dependency_groups'][0]['members'] = ['profile:profile:config']
sealed(Path.home(), mutate=config_manifest, data=b'[general]\nusers_name="Original"\n')
parent = Path.home() / 'selected'
parent.mkdir(mode=0o700)
control = Path.home() / 'control'
arguments = ['--control-root', str(control), 'restore', str(Path.home() / 'fixture.zip'),
    '--isolated', '--destination', 'root=' + str(parent / 'config'),
    '--destination', 'profile:profile:paths.data_dir=' + str(parent / 'data'),
    '--profile-name', 'profile=Selected CLI profile']
if selection != 'everything':
    arguments.extend(['--group', 'settings'])
if selection == 'duplicate':
    arguments.extend(['--group', 'settings'])
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    assert name not in {'tldw_chatbook.app', 'tldw_chatbook.config'}, name
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
output = io.StringIO()
def review(prompt):
    shown = output.getvalue()
    expected = 'null' if selection == 'everything' else '[\n    "settings"\n  ]'
    assert '"requested_groups": ' + expected in shown, shown
    assert '"effective_groups": [\n    "settings"\n  ]' in shown, shown
    assert '"required_groups": []' in shown, shown
    return decision
builtins.input = review
with contextlib.redirect_stdout(output):
    result = recovery_main(arguments)
shown = output.getvalue()
if selection == 'duplicate':
    assert result == 1, shown
    assert 'Recovery refused:' in shown, shown
    assert not (parent / 'config').exists()
else:
    assert result == 0, shown
    service = RecoveryService(control)
    try:
        if decision == 'restore':
            assert len(service.profiles()) == 1
            assert tomllib.loads((parent / 'config' / 'config.toml').read_text())['general']['users_name'] == 'Selected CLI profile'
        else:
            assert not service.profiles()
            assert not (parent / 'config').exists()
    finally:
        service.close()
assert (selector.read_bytes(), selector.stat().st_ino) == (original_bytes, original_inode)
assert 'tldw_chatbook.config' not in sys.modules
assert 'tldw_chatbook.app' not in sys.modules
assert not blocked_attempts(), blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "decision,selection",
    [
        ("cancel", "settings"),
        ("restore", "settings"),
        ("cancel", "everything"),
        ("restore", "duplicate"),
    ],
)
def test_real_isolated_restore_reviews_groups_before_confirmation(
    tmp_path, decision, selection
):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(tmp_path, decision, selection, script=_ISOLATED)
