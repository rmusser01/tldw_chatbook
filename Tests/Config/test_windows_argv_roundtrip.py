"""Evidence draft: real config writers preserve exact argv; no command is run."""

from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path

import pytest
import toml

from tldw_chatbook import config
from tldw_chatbook.Agents.hook_permissions import HookPermissions
from tldw_chatbook.Agents.run_hooks import fingerprint_hook


pytestmark = pytest.mark.bootstrap_profile
WINDOWS_CI_EXECUTABLE = r"C:\hostedtoolcache\windows\Python\3.12.10\x64\python.exe"
ARGV = [WINDOWS_CI_EXECUTABLE, "-c", "pass"]


@pytest.fixture
def selected_profile():
    """Declare valid source data independently of the currently broken fixture."""
    root = Path(os.environ["TLDW_TEST_CONFIG_ROOT"]).resolve(strict=True)
    path = Path(os.environ["TLDW_CONFIG_PATH"])
    assert path.resolve(strict=True).is_relative_to(root)
    data = config.get_user_data_dir()
    assert data.resolve(strict=True).is_relative_to(root)
    permissions = data / "hook_permissions.json"
    original = path.read_bytes()
    prior_permissions = permissions.read_bytes() if permissions.exists() else None
    raw = tomllib.loads(original.decode("utf-8"))
    raw["hooks"] = {
        "hook": [
            {
                "id": "exact-argv",
                "event": "PostToolUse",
                "command": ["fixture-command-is-never-launched", "baseline"],
                "timeout_s": 5,
            }
        ]
    }
    # This initial command deliberately has no backslash-x. Both original
    # encoders must accept setup before any tested product call is reached.
    initial = toml.dumps(raw)
    assert tomllib.loads(initial)["hooks"] == raw["hooks"]
    path.write_text(initial, encoding="utf-8")
    permissions.unlink(missing_ok=True)
    try:
        yield path, raw
    finally:
        path.write_bytes(original)
        if prior_permissions is None:
            permissions.unlink(missing_ok=True)
        else:
            permissions.write_bytes(prior_permissions)
        config.refresh_runtime_config_from_cli_config()


def _replacement(section):
    replacement = copy.deepcopy(section)
    replacement["hook"][0]["command"] = list(ARGV)
    return replacement


def test_canonical_config_writer_preserves_literal_windows_argv(selected_profile):
    path, raw = selected_profile
    replacement = copy.deepcopy(raw)
    replacement["hooks"] = _replacement(raw["hooks"])
    loaded = config.replace_cli_config(replacement)
    persisted = tomllib.loads(path.read_text(encoding="utf-8"))
    assert persisted["hooks"]["hook"][0]["command"] == ARGV
    assert loaded["hooks"]["hook"][0]["command"] == ARGV
    assert persisted["hooks"]["hook"][0]["id"] == "exact-argv"


def test_literal_hook_save_preserves_argv_through_expected_raw_preflight(
    selected_profile,
):
    path, _ = selected_profile
    with config.locked_hooks_config_snapshot() as expected:
        replacement = _replacement(expected.section)
    result = config.replace_hooks_config_snapshot(expected, replacement)
    assert result.file_replaced and result.caches_reloaded
    assert result.failure_phase is None
    persisted = tomllib.loads(path.read_text(encoding="utf-8"))
    assert persisted["hooks"] == replacement
    with config.locked_hooks_config_snapshot() as actual:
        assert actual.config_path == expected.config_path
        assert actual.profile_data_dir == expected.profile_data_dir
        assert actual.section == replacement


def test_real_hook_save_keeps_exact_fingerprint_and_requires_changed_command_review(
    selected_profile,
):
    path, _ = selected_profile
    owner = HookPermissions()
    try:
        pending = owner.snapshot()
        assert pending.rows[0].entry is not None
        assert pending.rows[0].entry.spec is not None
        approved = owner.approve(pending, [pending.rows[0].entry.key])
        assert approved.ready
        old_fingerprint = fingerprint_hook(approved.rows[0].entry.spec)
        replacement = _replacement(approved.config.section)
        result, changed = owner.save_configuration(approved, replacement)
        assert result.file_replaced and result.caches_reloaded
        assert not changed.ready
        assert changed.rows[0].entry is not None
        assert changed.rows[0].entry.spec is not None
        spec = changed.rows[0].entry.spec
        assert spec.command == tuple(ARGV)
        changed_fingerprint = fingerprint_hook(spec)
        assert changed_fingerprint != old_fingerprint
        assert tomllib.loads(path.read_text(encoding="utf-8"))["hooks"] == replacement
        reviewed = owner.approve(changed, [changed.rows[0].entry.key])
        assert reviewed.ready
        cosmetic = copy.deepcopy(reviewed.config.section)
        cosmetic["hook"][0]["name"] = "Same command, descriptive name"
        saved, same_definition = owner.save_configuration(reviewed, cosmetic)
        assert saved.file_replaced and saved.caches_reloaded
        assert same_definition.ready
        assert same_definition.rows[0].entry.spec.command == tuple(ARGV)
        assert (
            fingerprint_hook(same_definition.rows[0].entry.spec) == changed_fingerprint
        )
    finally:
        owner.close()
