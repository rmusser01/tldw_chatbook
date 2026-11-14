"""The wizard suites' shared-config restore leaves no test's keys behind.

Qodo (PR #3001): ``_restore_shared_bootstrap_config`` returned early when the
bootstrap ``config.toml`` did not exist before a test, so a file that test
wrote kept its keys for the next test in the same worker. These tests drive
the restore helper directly. Config writes are refused on the per-test
sandbox (``RecoveryRequired: raw_source_selection_changed``), so they run on
the bootstrap profile, and ``shared_config`` puts its file back afterwards
whatever the helper did.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from Tests.Wizards.conftest import put_back_bootstrap_config

pytestmark = pytest.mark.bootstrap_profile

_PROBE = "qodo-leak-probe"


@pytest.fixture
def shared_config():
    """The bootstrap config path; its original text is restored on teardown."""
    from tldw_chatbook import config

    path = Path(os.environ["TLDW_CONFIG_PATH"])
    config.load_cli_config_and_ensure_existence()
    original = path.read_text(encoding="utf-8")
    assert _PROBE not in original
    try:
        yield path
    finally:
        if not path.exists() or path.read_text(encoding="utf-8") != original:
            config.replace_cli_config_serialized(original, create_backup=False)


def _write_probe() -> None:
    from tldw_chatbook import config

    result = config.apply_settings_mutation_to_cli_config(
        {"chat_defaults": {"provider": _PROBE}}
    )
    assert result.file_replaced, result


def _provider_seen() -> object:
    from tldw_chatbook import config

    return config.get_cli_setting("chat_defaults", "provider")


def test_a_config_the_test_created_is_removed_with_its_keys(shared_config) -> None:
    """RED before the fix: the file and its probe key outlived the test."""
    path = shared_config
    path.unlink()  # the fixture records ``before=None`` for an absent file

    _write_probe()
    assert _PROBE in path.read_text(encoding="utf-8")
    assert _provider_seen() == _PROBE

    put_back_bootstrap_config(path, None)

    assert not path.exists()
    assert _provider_seen() != _PROBE


def test_a_config_the_test_deleted_is_written_back(shared_config) -> None:
    """RED before the fix: a deleted file stayed deleted."""
    path = shared_config
    before = path.read_text(encoding="utf-8")
    _write_probe()
    path.unlink()

    put_back_bootstrap_config(path, before)

    assert path.exists()
    assert _provider_seen() != _PROBE


def test_a_config_the_test_changed_is_put_back(shared_config) -> None:
    """Already passed before the fix; pins the restore that was there."""
    path = shared_config
    before = path.read_text(encoding="utf-8")

    _write_probe()
    assert _provider_seen() == _PROBE

    put_back_bootstrap_config(path, before)

    assert _PROBE not in path.read_text(encoding="utf-8")
    assert _provider_seen() != _PROBE
