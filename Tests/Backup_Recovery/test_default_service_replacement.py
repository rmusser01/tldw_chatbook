"""Default config and service locations retain full replacement/rollback safety."""

import json
import sys
from pathlib import Path

import pytest

from Tests.Backup_Recovery.native_package import (
    native_package as native_package,  # noqa: PLC0414
)
from Tests.Backup_Recovery.test_later_rollback_credential_ui import (
    test_f9_later_rollback_requires_explicit_credential_review as _roundtrip,
)


@pytest.mark.timeout(2400 if sys.platform == "win32" else 600)
def test_default_profile_service_replacement_and_later_rollback(
    tmp_path, native_package
):
    _roundtrip(tmp_path, native_package, default_profile=True)
    config_parent = tmp_path / "home" / ".config" / "tldw_cli"
    bootstrap = config_parent / "recovery-bootstrap"
    controls = (bootstrap, config_parent / "recovery")
    profiles = [
        json.loads(path.read_text()) for path in bootstrap.glob("profile-*.json")
    ]
    assert profiles
    for profile in profiles:
        for raw in profile["roots"]:
            root = Path(raw)
            assert root != config_parent
            assert not any(
                root == control or root in control.parents or control in root.parents
                for control in controls
            )
