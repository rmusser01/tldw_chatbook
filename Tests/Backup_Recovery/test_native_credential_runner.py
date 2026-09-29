"""Behavioral boundaries for the disposable native credential qualification."""

import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery import run_platform_product as runner


def test_native_fixture_installs_owners_before_destination_discovery(
    tmp_path, monkeypatch
):
    import tldw_chatbook
    from Tests.ProductionApp import test_native_credential_recovery as product
    from tldw_chatbook.Backup_Recovery import (
        inventory,
        owner_registry,
        recovery_service,
    )

    class DiscoveryReached(Exception):
        pass

    monkeypatch.setattr(owner_registry, "_adapters", {})
    config = SimpleNamespace(set_encryption_password=lambda password: None)
    monkeypatch.setattr(tldw_chatbook, "config", config, raising=False)
    monkeypatch.setitem(sys.modules, "tldw_chatbook.config", config)
    monkeypatch.setattr(
        recovery_service, "RecoveryService", lambda root: SimpleNamespace()
    )
    monkeypatch.setattr(recovery_service, "default_control_root", lambda: tmp_path)

    def discover(*args, **kwargs):
        assert "db.prompts.primary" in {
            row.owner_id for row in owner_registry.registered()
        }
        raise DiscoveryReached()

    source = tmp_path / "source-linux.age"
    source.with_suffix(".json").write_text(json.dumps({"system": "Linux"}))
    monkeypatch.setattr(inventory, "discover", discover)
    with pytest.raises(DiscoveryReached):
        product._transfer(source)


def test_native_isolated_preview_has_no_replacement_acknowledgments(
    tmp_path, monkeypatch
):
    import tldw_chatbook
    from Tests.ProductionApp import test_native_credential_recovery as product
    from tldw_chatbook.Backup_Recovery import (
        archive_reader,
        inventory,
        recovery_service,
    )

    class PreviewReached(Exception):
        pass

    source = tmp_path / "source-linux.age"
    with product.zipfile.ZipFile(source, "w"):
        pass
    source.with_suffix(".json").write_text(json.dumps({"system": "Linux"}))
    stores = []
    for number in range(2):
        path = tmp_path / f"unselected-{number}.sqlite"
        path.write_bytes(b"synthetic-unselected")
        stores.append(
            SimpleNamespace(path=path, owner="db.prompts.primary", status="included")
        )

    class Service:
        def __init__(self, root):
            pass

        def start_inspection(self, source, *, password):
            return "inspection"

        def wait(self, operation, *, timeout):
            return {"state": "succeeded"}

        def inspection(self, operation):
            return SimpleNamespace(path=source)

        def preview_restore(self, inspection, **options):
            assert options["mode"] == "isolated"
            assert "acknowledged_credential_issues" not in options
            raise PreviewReached()

        def close(self):
            pass

    config = SimpleNamespace(set_encryption_password=lambda password: None)
    monkeypatch.setattr(tldw_chatbook, "config", config, raising=False)
    monkeypatch.setitem(sys.modules, "tldw_chatbook.config", config)
    monkeypatch.setattr(recovery_service, "RecoveryService", Service)
    monkeypatch.setattr(recovery_service, "default_control_root", lambda: tmp_path)
    monkeypatch.setattr(
        inventory, "discover", lambda selectors: SimpleNamespace(items=stores)
    )
    monkeypatch.setattr(
        archive_reader,
        "verify_sealed",
        lambda archive: SimpleNamespace(
            files=(), profile_ids=("default", "retargeted")
        ),
    )
    monkeypatch.setattr(
        product,
        "_material",
        lambda archive: [{"remappable": False, "id": "synthetic-manual"}],
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(PreviewReached):
        product._transfer(source)


@pytest.mark.parametrize(
    ("first_issue", "expected_attempts"),
    (
        ("credential_manual_recovery_required:synthetic-id", 2),
        ("scope_changed:positive-native-secret", 1),
    ),
    ids=("manual-review-retry", "first-scope-refusal"),
)
def test_native_capture_reports_review_category_with_bounded_manual_retry(
    tmp_path, monkeypatch, first_issue, expected_attempts
):
    import tldw_chatbook
    from Tests.ProductionApp import test_native_credential_recovery as product
    from tldw_chatbook.Backup_Recovery import recovery_service, runtime_maintenance

    async def idle(*args):
        pass

    class Service:
        def __init__(self, root):
            self.attempts = 0

        def preview_backup_details(self, selectors, *, options, destination):
            return {
                "inventory": SimpleNamespace(complete=True, scope_digest="synthetic")
            }

        def start_backup(self, *args, **kwargs):
            self.attempts += 1
            return self.attempts

        def wait(self, operation, *, timeout):
            return {
                "state": "failed",
                "issues": ("review_required",),
                "review_issues": (
                    first_issue
                    if operation == 1
                    else "scope_changed:positive-native-secret",
                ),
            }

        def close(self):
            pass

    service = Service(tmp_path)
    config = SimpleNamespace(set_encryption_password=lambda password: None)
    monkeypatch.setattr(tldw_chatbook, "config", config, raising=False)
    monkeypatch.setitem(sys.modules, "tldw_chatbook.config", config)
    monkeypatch.setitem(
        sys.modules,
        "tldw_chatbook.app",
        SimpleNamespace(
            TldwCli=lambda: SimpleNamespace(
                _shutdown_app_owned_lifecycles=idle,
                tts_service=SimpleNamespace(close=idle, wait_closed=idle),
            )
        ),
    )
    monkeypatch.setattr(recovery_service, "RecoveryService", lambda root: service)
    monkeypatch.setattr(recovery_service, "default_control_root", lambda: tmp_path)
    monkeypatch.setattr(runtime_maintenance, "monitor_app", idle)
    monkeypatch.setenv("TLDW_CREDENTIAL_TRANSFER_ROOT", str(tmp_path))
    with pytest.raises(ValueError) as caught:
        product.asyncio.run(product._capture())
    assert service.attempts == expected_attempts
    assert caught.value.args == ("scope_changed",)


def test_native_child_thread_samples_publish_only_safe_late_frames(
    tmp_path, monkeypatch
):
    import tldw_chatbook
    from Tests.Backup_Recovery import thread_diagnostics
    from Tests.ProductionApp import test_native_credential_recovery as product

    private, artifacts = tmp_path / "private", tmp_path / "artifacts"
    failure_root = private / "native-failures"
    failure_root.mkdir(parents=True)
    artifacts.mkdir()
    monkeypatch.setenv("TLDW_NATIVE_FAILURE_ROOT", str(failure_root))
    monkeypatch.setenv(
        "TLDW_TEST_INSTALLED_PACKAGE",
        str(Path(tldw_chatbook.__file__).resolve().parents[1]),
    )
    monkeypatch.setattr(sys, "argv", ["native.py", "setup", "default"])
    monkeypatch.setattr(runner, "validate_native_credential_environment", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "Tests.network_guard",
        SimpleNamespace(install=lambda: None, blocked_attempts=list),
    )
    for name in ("sounddevice", "pyaudio"):
        monkeypatch.setitem(sys.modules, name, None)
    observe = thread_diagnostics.observe_threads
    monkeypatch.setattr(
        thread_diagnostics,
        "observe_threads",
        lambda path, *, interval: observe(path, interval=0.01),
    )
    sampled = Event()
    snapshot = thread_diagnostics._snapshot

    def acknowledged_snapshot():
        rows = snapshot()
        if any(
            frame["function"] == "blocked_setup"
            for row in rows
            for frame in row["frames"]
        ):
            sampled.set()
        return rows

    monkeypatch.setattr(thread_diagnostics, "_snapshot", acknowledged_snapshot)

    async def blocked_setup(role):
        sensitive_fixture = "positive-native-secret"
        assert sampled.wait(timeout=5)
        assert sensitive_fixture

    monkeypatch.setattr(product, "_setup", blocked_setup)
    product._main()
    stacks = next((failure_root / "child-stacks").glob("*.json"))
    samples = json.loads(stacks.read_text())
    for snapshot in samples:
        for thread in snapshot:
            thread["locals"] = "positive-native-secret"
    stacks.write_text(json.dumps(samples))
    (stacks.parent / "raw-fatal.log").write_text("positive-native-secret")
    runner._publish_native_failures(private, artifacts)
    path = artifacts / "native-failures.json"
    projected = path.read_text()
    child = json.loads(projected)["children"][0]
    assert {"route": child["route"], "role": child["role"]} == {
        "route": "setup",
        "role": "default",
    }
    assert 1 <= len(child["samples"]) <= 4
    assert any(
        frame["function"] == "blocked_setup"
        for snapshot in child["samples"]
        for thread in snapshot
        for frame in thread["frames"]
    )
    assert (
        "positive-native-secret" not in projected
        and '"thread"' not in projected
        and '"locals"' not in projected
    )
    assert {item.name for item in artifacts.iterdir()} == {"native-failures.json"}

    samples[-1][0]["frames"][0]["file"] = "../positive-native-secret.log"
    stacks.write_text(json.dumps(samples))
    with pytest.raises(RuntimeError):
        runner._publish_native_failures(private, artifacts)
    assert path.read_text() == projected

    stacks.rename(stacks.with_name("setup--positive-native-secret--123.json"))
    with pytest.raises(RuntimeError):
        runner._publish_native_failures(private, artifacts)
    assert path.read_text() == projected

    monkeypatch.delenv("TLDW_NATIVE_FAILURE_ROOT")
    product._main()
    assert len(list(stacks.parent.glob("*.json"))) == 1


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
        Path(runner.__file__).read_text()
        + "\ndef validate_native_credential_environment():\n"
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


def test_native_failures_publish_only_exception_types_and_code_locations(tmp_path):
    workspace = tmp_path / "workspace"
    helpers = workspace / "Tests" / "Backup_Recovery"
    helpers.mkdir(parents=True)
    for package in (workspace / "Tests", helpers):
        (package / "__init__.py").touch()
    (workspace / "Tests" / "network_guard.py").write_text("def install(): pass\n")
    (helpers / "run_platform_product.py").write_text(
        Path(runner.__file__).read_text()
        + "\ndef validate_native_credential_environment(): return 'keyring.backends.SecretService.Keyring'\n"
    )
    (helpers / "thread_diagnostics.py").write_text(
        Path(runner.__file__).with_name("thread_diagnostics.py").read_text()
    )
    (workspace / "child.py").write_text(
        "def fail_child():\n"
        " secret='positive-native-secret'\n"
        " raise RuntimeError(secret)\n"
        "fail_child()\n"
    )
    test = workspace / "test_child.py"
    test.write_text(
        "import subprocess,sys\n"
        "def test_child_failure():\n"
        " secret='positive-native-secret'\n"
        " result=subprocess.run([sys.executable,'child.py'])\n"
        " assert result.returncode==0,secret\n"
    )
    private, artifacts = tmp_path / "private", tmp_path / "artifacts"
    private.mkdir()
    artifacts.mkdir()
    result = runner._run_pytest_phase(
        workspace=workspace,
        private_root=private,
        artifacts=artifacts,
        environment=dict(
            os.environ,
            PYTHONPATH=os.pathsep.join(
                (str(workspace), str(Path(runner.__file__).resolve().parents[2]))
            ),
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        ),
        phase="product",
        tests=(str(test),),
        noconftest=True,
        timeout_seconds=30,
        native_credentials=True,
    )
    assert result["pytest_returncode"] == 1
    path = artifacts / "native-failures.json"
    receipt = json.loads(path.read_text())
    assert {row["error_class"] for row in receipt["failures"]} == {
        "RuntimeError",
        "AssertionError",
    }
    assert any(
        frame == {"file": "child.py", "function": "fail_child", "line": 3}
        for row in receipt["failures"]
        for frame in row["frames"]
    )
    assert "positive-native-secret" not in path.read_text()
    assert {item.name for item in artifacts.iterdir()} == {"native-failures.json"}


def test_native_failure_publication_refuses_unsafe_frame_fields(tmp_path):
    private, artifacts = tmp_path / "private", tmp_path / "artifacts"
    (private / "native-failures").mkdir(parents=True)
    artifacts.mkdir()
    (private / "native-failures" / "unsafe.json").write_text(
        json.dumps(
            {
                "error_class": "RuntimeError",
                "frames": [
                    {"file": "../private.log", "function": "secret=value", "line": 1}
                ],
                "message": "positive-native-secret",
            }
        )
    )
    with pytest.raises(RuntimeError):
        runner._publish_native_failures(private, artifacts)
    assert not list(artifacts.iterdir())


def test_native_failure_preserves_only_canonical_worker_issue(tmp_path):
    private, artifacts = tmp_path / "private", tmp_path / "artifacts"
    failure_root = private / "native-failures"
    failure_root.mkdir(parents=True)
    artifacts.mkdir()
    runner._record_native_failure(failure_root, ValueError("scope_changed"))
    runner._record_native_failure(failure_root, RuntimeError("positive-native-secret"))
    (failure_root / "untrusted.json").write_text(
        json.dumps(
            {
                "error_class": "ValueError",
                "frames": [],
                "issue": "positive-native-secret",
                "message": "positive-native-secret",
            }
        )
    )
    runner._publish_native_failures(private, artifacts)
    path = artifacts / "native-failures.json"
    receipt = json.loads(path.read_text())
    assert {row["issue"] for row in receipt["failures"]} == {
        "scope_changed",
        "backup_operation_failed",
    }
    assert "positive-native-secret" not in path.read_text()


@pytest.mark.parametrize(
    ("kind", "argument", "expected"),
    (
        ("interrupted", "positive-native-secret", "cancelled"),
        ("capture", ("positive-native-secret",), "review_required"),
        (
            "compression",
            ("positive-native-secret", 1024),
            "compression_review_required",
        ),
        ("crypto", "helper_unavailable", "encryption_unavailable"),
        ("crypto", "positive-native-secret", "encryption_failed"),
    ),
    ids=(
        "interrupted",
        "capture-review",
        "compression-review",
        "crypto-unavailable",
        "crypto-failed",
    ),
)
def test_native_failure_preserves_safe_type_derived_issue(
    tmp_path, kind, argument, expected
):
    from tldw_chatbook.Backup_Recovery.archive_reader import CompressionReviewRequired
    from tldw_chatbook.Backup_Recovery.capture import CaptureReviewRequired
    from tldw_chatbook.Backup_Recovery.crypto import CryptoError

    constructors = {
        "interrupted": InterruptedError,
        "capture": CaptureReviewRequired,
        "crypto": CryptoError,
    }
    error = (
        CompressionReviewRequired(*argument)
        if kind == "compression"
        else constructors[kind](argument)
    )
    private, artifacts = tmp_path / "private", tmp_path / "artifacts"
    failure_root = private / "native-failures"
    failure_root.mkdir(parents=True)
    artifacts.mkdir()
    runner._record_native_failure(failure_root, error)
    runner._publish_native_failures(private, artifacts)
    path = artifacts / "native-failures.json"
    receipt = json.loads(path.read_text())
    assert [row["issue"] for row in receipt["failures"]] == [expected]
    assert "positive-native-secret" not in path.read_text()


def test_native_failure_keeps_import_stack_locations_without_exception_text(tmp_path):
    private, artifacts = tmp_path / "private", tmp_path / "artifacts"
    failure_root = private / "native-failures"
    failure_root.mkdir(parents=True)
    artifacts.mkdir()

    def fail_import():
        raise ImportError("positive-native-secret")

    fail_import.__code__ = fail_import.__code__.replace(
        co_filename="<frozen importlib._bootstrap>"
    )
    try:
        fail_import()
    except ImportError as error:
        runner._record_native_failure(failure_root, error)
    runner._publish_native_failures(private, artifacts)
    path = artifacts / "native-failures.json"
    receipt = json.loads(path.read_text())
    assert receipt["failures"][0]["error_class"] == "ImportError"
    assert {
        "file": "<frozen importlib._bootstrap>",
        "function": "fail_import",
        "line": fail_import.__code__.co_firstlineno + 1,
    } in receipt["failures"][0]["frames"]
    assert "positive-native-secret" not in path.read_text()
