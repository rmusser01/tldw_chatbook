"""Behavioral boundaries for the disposable native credential qualification."""

import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery import run_platform_product as runner


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {"GITHUB_ACTIONS": "true", "RUNNER_ENVIRONMENT": "self-hosted"},
        {
            "GITHUB_ACTIONS": "true",
            "RUNNER_ENVIRONMENT": "github-hosted",
            "RUNNER_OS": "macOS",
            "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
        },
    ],
)
def test_native_lane_refuses_unsafe_host_before_keyring_access(
    monkeypatch, environment
):
    monkeypatch.setattr(runner.platform, "system", lambda: "Darwin")
    import keyring

    def forbidden():
        pytest.fail("unsafe host accessed the native credential backend")

    monkeypatch.setattr(keyring, "get_keyring", forbidden)
    monkeypatch.setattr(os, "environ", environment)
    with pytest.raises(RuntimeError):
        runner.validate_native_credential_environment()


def test_linux_lane_refuses_callers_session_bus_before_keyring_access(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner.platform, "system", lambda: "Linux")
    root = tmp_path / "credential-session"
    root.mkdir(mode=0o700)
    environment = {
        "TLDW_NATIVE_CREDENTIAL_ROOT": str(root),
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus",
        "PYTHON_KEYRING_BACKEND": "keyring.backends.SecretService.Keyring",
        "KEYRING_PROPERTY_PREFERRED_COLLECTION": "/org/freedesktop/secrets/collection/session",
    }
    import keyring

    monkeypatch.setattr(keyring, "get_keyring", lambda: pytest.fail("caller bus used"))
    monkeypatch.setattr(os, "environ", environment)
    with pytest.raises(RuntimeError):
        runner.validate_native_credential_environment()


def test_linux_lane_refuses_locked_private_session_before_native_reads(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner.platform, "system", lambda: "Linux")
    root = tmp_path / "credential-session"
    root.mkdir(mode=0o700)
    native_lstat = Path.lstat
    socket_info = os.stat_result(
        (stat.S_IFSOCK | 0o600, 0, 0, 1, os.getuid(), os.getgid(), 0, 0, 0, 0)
    )
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda path: socket_info if path == root / "bus" else native_lstat(path),
    )
    monkeypatch.setitem(
        sys.modules,
        "secretstorage",
        SimpleNamespace(
            dbus_init=lambda: SimpleNamespace(close=lambda: None),
            Collection=lambda connection, path: SimpleNamespace(is_locked=lambda: True),
        ),
    )
    import keyring

    monkeypatch.setattr(
        keyring, "get_keyring", lambda: pytest.fail("locked backend read")
    )
    monkeypatch.setattr(
        os,
        "environ",
        {
            "TLDW_NATIVE_CREDENTIAL_ROOT": str(root),
            "DBUS_SESSION_BUS_ADDRESS": f"unix:path={root}/bus,guid={'a' * 32}",
            "PYTHON_KEYRING_BACKEND": "keyring.backends.SecretService.Keyring",
            "KEYRING_PROPERTY_PREFERRED_COLLECTION": "/org/freedesktop/secrets/collection/login",
        },
    )
    with pytest.raises(RuntimeError, match="session_locked"):
        runner.validate_native_credential_environment()


def test_native_child_environment_preserves_backend_and_bus(tmp_path, monkeypatch):
    inherited = {
        "PYTHON_KEYRING_BACKEND": "keyring.backends.SecretService.Keyring",
        "TLDW_NATIVE_CREDENTIAL_ROOT": str(tmp_path / "credential-session"),
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path={tmp_path}/credential-session/bus",
        "KEYRING_PROPERTY_PREFERRED_COLLECTION": "/org/freedesktop/secrets/collection/session",
        "TLDW_CREDENTIAL_TRANSFER_ROOT": str(tmp_path / "transfer"),
    }
    for name, value in inherited.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        runner,
        "validate_native_credential_environment",
        lambda environment=None: inherited["PYTHON_KEYRING_BACKEND"],
    )
    environment = runner._private_environment(
        tmp_path, tmp_path / "private", native_credentials=True
    )
    assert {name: environment[name] for name in inherited} == inherited
    assert (
        runner._private_environment(tmp_path, tmp_path / "ordinary")[
            "PYTHON_KEYRING_BACKEND"
        ]
        == "keyring.backends.null.Keyring"
    )


def _source_archive(root):
    outbound = root / "outbound"
    outbound.mkdir(parents=True)
    archive = outbound / "source-linux.age"
    archive.write_bytes(
        b"age-encryption.org/v1\n-> scrypt dummy\n--- dummy\nencrypted fixture"
    )
    receipt = {
        "schema": 1,
        "system": "Linux",
        "release": "6.8.0",
        "machine": "x86_64",
        "python": "3.12.11",
        "backend": "keyring.backends.SecretService.Keyring",
        "archive": archive.name,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "revision": "a" * 40,
        "wheel_sha256": "b" * 64,
        "status": "passed",
    }
    return outbound, archive, receipt


def test_native_publication_exports_only_encrypted_archive_and_allowlisted_receipt(
    tmp_path,
):
    outbound, archive, receipt = _source_archive(tmp_path / "transfer")
    receipt["credentials"] = [{"value": "positive-native-secret"}]
    (outbound / "source-linux.json").write_text(json.dumps(receipt))
    for name in ("config.toml", "material.json", "keyring.db", "private.log"):
        (outbound / name).write_text("positive-native-secret")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    assert (
        runner._publish_native_credential_artifacts(
            tmp_path / "transfer", artifacts, "native-credentials-source"
        )
        == 1
    )
    assert {path.name for path in artifacts.iterdir()} == {
        "source-linux.age",
        "source-linux.json",
    }
    assert "positive-native-secret" not in (artifacts / "source-linux.json").read_text()
    assert (artifacts / archive.name).read_bytes() == archive.read_bytes()


def test_native_publication_refuses_plaintext_or_unverified_archive(tmp_path):
    outbound, archive, receipt = _source_archive(tmp_path / "transfer")
    archive.write_bytes(b"plaintext positive-native-secret")
    receipt["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    (outbound / "source-linux.json").write_text(json.dumps(receipt))
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    with pytest.raises(RuntimeError):
        runner._publish_native_credential_artifacts(
            tmp_path / "transfer", artifacts, "native-credentials-source"
        )
    assert not list(artifacts.iterdir())


def test_native_lane_refuses_loaded_fallback_backend(monkeypatch):
    monkeypatch.setattr(runner.platform, "system", lambda: "Darwin")
    import keyring
    from keyring.backends.null import Keyring

    monkeypatch.setattr(keyring, "get_keyring", Keyring)
    monkeypatch.setattr(
        os,
        "environ",
        {
            "GITHUB_ACTIONS": "true",
            "RUNNER_ENVIRONMENT": "github-hosted",
            "RUNNER_OS": "macOS",
            "PYTHON_KEYRING_BACKEND": "keyring.backends.macOS.Keyring",
        },
    )
    with pytest.raises(RuntimeError, match="fallback_backend"):
        runner.validate_native_credential_environment()


def test_destination_publication_projects_three_direction_results(tmp_path):
    outbound, _, runtime = _source_archive(tmp_path / "transfer")
    for name in ("archive", "archive_sha256"):
        runtime.pop(name)
    runtime["negative_checks"] = 5
    runtime["credentials"] = "positive-native-secret"
    runtime["results"] = [
        {
            "source_system": system,
            "destination_system": "Linux",
            "archive_sha256": "c" * 64,
            "isolated": True,
            "original_retained": True,
            "replacement": True,
            "rollback": True,
            "captured": 12,
            "manual_required": 1,
            "unavailable": 0,
            "plaintext": "positive-native-secret",
        }
        for system in ("Darwin", "Linux", "Windows")
    ]
    (outbound / "destination-results.json").write_text(json.dumps(runtime))
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    assert (
        runner._publish_native_credential_artifacts(
            tmp_path / "transfer", artifacts, "native-credentials-destination"
        )
        == 3
    )
    published = (artifacts / "destination-results.json").read_text()
    assert "positive-native-secret" not in published
    assert {row["source_system"] for row in json.loads(published)["results"]} == {
        "Darwin",
        "Linux",
        "Windows",
    }


def test_native_pytest_child_keeps_environment_and_publishes_no_raw_logs(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "Tests" / "Backup_Recovery").mkdir(parents=True)
    for package in (workspace / "Tests", workspace / "Tests" / "Backup_Recovery"):
        (package / "__init__.py").touch()
    (workspace / "Tests" / "network_guard.py").write_text("def install(): pass\n")
    # Replace only the native effect boundary; exercise the real subprocess runner.
    (workspace / "Tests" / "Backup_Recovery" / "run_platform_product.py").write_text(
        "import os\n"
        "def validate_native_credential_environment():\n"
        " assert os.environ['PYTHON_KEYRING_BACKEND']=='keyring.backends.SecretService.Keyring'\n"
        " assert os.environ['DBUS_SESSION_BUS_ADDRESS']=='unix:path=/owned/bus'\n"
    )
    test = workspace / "test_child.py"
    test.write_text(
        "import os\n"
        "def test_child():\n"
        " assert os.environ['TLDW_CREDENTIAL_TRANSFER_ROOT']=='/owned/transfer'\n"
        " print('positive-native-secret')\n"
    )
    private = tmp_path / "private"
    private.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    environment = dict(
        os.environ,
        PYTHONPATH=str(workspace),
        PYTHON_KEYRING_BACKEND="keyring.backends.SecretService.Keyring",
        DBUS_SESSION_BUS_ADDRESS="unix:path=/owned/bus",
        TLDW_CREDENTIAL_TRANSFER_ROOT="/owned/transfer",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    result = runner._run_pytest_phase(
        workspace=workspace,
        private_root=private,
        artifacts=artifacts,
        environment=environment,
        phase="product",
        tests=(str(test),),
        noconftest=True,
        timeout_seconds=30,
        native_credentials=True,
    )
    assert result["pytest_returncode"] == 0
    assert result["junit"]["collected"] == 1
    assert not list(artifacts.iterdir())


@pytest.mark.parametrize(
    "field,value", [("replacement", False), ("unavailable", 1), ("captured", 0)]
)
def test_destination_receipt_cannot_report_passed_with_failed_readback(
    tmp_path, field, value
):
    outbound, _, runtime = _source_archive(tmp_path / "transfer")
    runtime["negative_checks"] = 5
    runtime["results"] = [
        {
            "source_system": system,
            "destination_system": "Linux",
            "archive_sha256": "c" * 64,
            "isolated": True,
            "original_retained": True,
            "replacement": True,
            "rollback": True,
            "captured": 12,
            "manual_required": 1,
            "unavailable": 0,
        }
        for system in ("Darwin", "Linux", "Windows")
    ]
    runtime["results"][0][field] = value
    (outbound / "destination-results.json").write_text(json.dumps(runtime))
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    with pytest.raises(RuntimeError):
        runner._publish_native_credential_artifacts(
            tmp_path / "transfer", artifacts, "native-credentials-destination"
        )
    assert not list(artifacts.iterdir())
