"""Behavioral boundaries for the disposable native credential qualification."""

import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery import run_platform_product as runner


@pytest.mark.parametrize(
    "selection",
    [
        "native-credentials-source",
        "native-credentials-destination",
    ],
)
@pytest.mark.parametrize("value", [None, ""])
def test_native_run_refuses_missing_transfer_root_before_effects(
    tmp_path, monkeypatch, selection, value
):
    if value is None:
        monkeypatch.delenv("TLDW_CREDENTIAL_TRANSFER_ROOT", raising=False)
    else:
        monkeypatch.setenv("TLDW_CREDENTIAL_TRANSFER_ROOT", value)
    effects = []
    monkeypatch.setattr(
        runner,
        "validate_native_credential_environment",
        lambda: effects.append("backend"),
    )
    monkeypatch.setattr(
        runner,
        "_create_private_root",
        lambda *_: effects.append("private-root"),
    )
    monkeypatch.setattr(
        runner,
        "_copy_tracked_source",
        lambda *_: effects.append("source-copy"),
    )
    evidence = tmp_path / "evidence"
    with pytest.raises(RuntimeError, match="native_credential_transfer_root_required"):
        runner.run(tmp_path, evidence, product_selection=selection)
    assert not effects and not evidence.exists()


@pytest.mark.parametrize("system", ["posix", "nt"], ids=["unix", "windows"])
@pytest.mark.parametrize("job", ["source", "destination"])
@pytest.mark.skipif(os.name == "nt", reason="POSIX sandbox models Windows ancestry")
def test_native_workflow_prepares_trusted_transfer_before_download_and_qualification(
    tmp_path, monkeypatch, system, job
):
    import re
    import runpy

    import yaml

    from tldw_chatbook.Backup_Recovery.archive_reader import _regular
    from tldw_chatbook.Utils import windows_files

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/native-credential-backup.yml"
    )
    steps = yaml.safe_load(workflow.read_text())["jobs"][job]["steps"]
    preparations = [step for step in steps if step.get("id") == "credential-transfer"]
    assert len(preparations) == 1, "private_transfer_preparation_required"
    preparation = preparations[0]
    dependencies = next(
        step
        for step in steps
        if step.get("name") == "Install core application and test dependencies"
    )
    assert steps.index(dependencies) < steps.index(preparation)
    home = tmp_path / "home"
    trusted_temp = home / "AppData/Local/Temp"
    trusted_temp.mkdir(parents=True, mode=0o700)
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir(mode=0o700)
    if system == "nt":
        runner_temp.chmod(0o777)
    output = tmp_path / "step-output"
    monkeypatch.setenv("GITHUB_WORKSPACE", str(workflow.parents[2]))
    monkeypatch.syspath_prepend(str(workflow.parents[2]))
    monkeypatch.setenv("RUNNER_TEMP", str(runner_temp))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(runner, "os", SimpleNamespace(name=system, getpid=os.getpid))
    monkeypatch.setattr(
        windows_files,
        "WindowsOS",
        lambda: SimpleNamespace(
            mkdir=lambda path, mode: path.mkdir(mode=mode),
            stat=lambda path: path.lstat(),
            geteuid=os.getuid,
        ),
    )
    assert preparation["shell"] == "python"
    script = tmp_path / "prepare-transfer.py"
    script.write_text(preparation["run"])
    runpy.run_path(str(script))
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    transfer = Path(outputs["path"])
    expected_parent = (
        trusted_temp / f"tldw-backup-platform-{os.getpid()}"
        if system == "nt"
        else runner_temp / f"native-credential-incoming-{os.getpid()}" / "private"
    )
    assert transfer == expected_parent / "transfer"
    assert transfer.lstat().st_uid == os.getuid()
    assert stat.S_IMODE(transfer.lstat().st_mode) == 0o700
    source = transfer / "source-linux.age"
    source.write_bytes(b"synthetic-encrypted-placeholder")
    with _regular(source) as stream:
        assert stream.read() == b"synthetic-encrypted-placeholder"
    consumers = [
        (step, step["env"]["TLDW_CREDENTIAL_TRANSFER_ROOT"])
        for step in steps
        if "TLDW_CREDENTIAL_TRANSFER_ROOT" in step.get("env", {})
    ]
    consumers.extend(
        (step, step["with"]["path"])
        for step in steps
        if step.get("uses", "").startswith("actions/download-artifact@")
    )
    assert len(consumers) == (2 if job == "destination" else 1)
    for consumer, expression in consumers:
        assert steps.index(preparation) < steps.index(consumer)
        selected = re.sub(
            r"\$\{\{\s*steps\.([\w-]+)\.outputs\.(\w+)\s*\}\}",
            lambda match: {preparation["id"]: outputs}[match[1]][match[2]],
            expression,
        )
        assert selected == str(transfer)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        runpy.run_path(str(script))
    assert output.read_bytes() == before
    assert source.read_bytes() == b"synthetic-encrypted-placeholder"
    if system == "nt":
        assert not list(runner_temp.iterdir())


@pytest.mark.skipif(os.name == "nt", reason="entry imports use private POSIX sandbox")
def test_native_workflow_saved_script_imports_checkout_without_pythonpath(
    tmp_path,
):
    import yaml

    workspace = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load(
        (workspace / ".github/workflows/native-credential-backup.yml").read_text()
    )
    preparation = next(
        step
        for step in workflow["jobs"]["source"]["steps"]
        if step.get("id") == "credential-transfer"
    )
    root = tmp_path / "entry-private"
    root.mkdir(mode=0o700)
    environment = runner._private_environment(workspace, root)
    environment.pop("PYTHONPATH", None)
    environment.update(GITHUB_WORKSPACE=str(workspace), RUNNER_TEMP=str(root))
    output = root / "step-output"
    environment["GITHUB_OUTPUT"] = str(output)
    script = root / "prepare-transfer.py"
    script.write_text(preparation["run"])
    result = runner.subprocess.run(  # nosec B603 - fixed interpreter/owned workflow script
        [sys.executable, str(script)],
        cwd=workspace,
        env=environment,
        stdout=runner.subprocess.DEVNULL,
        stderr=runner.subprocess.DEVNULL,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, "workflow_checkout_import_required"
    transfer = Path(output.read_text().strip().split("=", 1)[1])
    assert transfer.is_dir() and transfer.is_relative_to(root)
    assert stat.S_IMODE(transfer.lstat().st_mode) == 0o700


def test_native_worker_observer_is_optional(monkeypatch):
    from Tests.ProductionApp import test_native_credential_recovery as product

    monkeypatch.delenv("TLDW_NATIVE_FAILURE_ROOT", raising=False)
    service = SimpleNamespace()
    assert product._observe_workers(service) is service


@pytest.mark.parametrize(
    "failure",
    (None, "repeat-retention", "unrelated", "missing-journal", "unsafe-abort"),
)
def test_native_review_repreviews_exact_notices_and_aborts_only_linked_copy(failure):
    from Tests.ProductionApp import test_native_credential_recovery as product

    retention = ("credential_isolated_retention_required",)
    manual = ("credential_manual_recovery_required:target-material",)
    notices = [
        {"state": "failed", "review_issues": retention, "result": {}},
        {
            "state": "recovery_required",
            "review_issues": retention if failure == "repeat-retention" else manual,
            "result": {}
            if failure == "missing-journal"
            else {"journal_operation_id": "linked"},
        },
        {"state": "succeeded", "review_issues": (), "result": {}},
    ]
    if failure == "unrelated":
        notices[1]["review_issues"] = ("credential_missing:unavailable",)
    previews, aborts, waits = [], [], []

    class Service:
        def wait(self, operation, *, timeout=None):
            waits.append((operation, timeout))
            if operation == "abort":
                return {"state": "succeeded", "result": {"aborted": True}}
            return notices.pop(0)

        def status(self, operation):
            assert operation == "linked"
            return {"actions": ("finish",) if failure == "unsafe-abort" else ("abort",)}

        def start_recovery(self, operation, *, action):
            assert operation == "linked" and action == "abort"
            aborts.append(operation)
            return "abort"

    def preview(acknowledged):
        previews.append(acknowledged)
        if len(previews) == 3:
            assert aborts == ["linked"]
        return acknowledged

    def run():
        return product._run_reviewed_replacement(
            Service(), preview, lambda plan: "restore"
        )

    if failure is not None:
        with pytest.raises((AssertionError, KeyError)):
            run()
        assert len(previews) == 2
        assert waits == [("restore", None), ("restore", None)]
    else:
        assert run()["state"] == "succeeded"
        assert previews == [(), retention, (*retention, *manual)]
        assert aborts == ["linked"]
        assert waits == [
            ("restore", None),
            ("restore", None),
            ("abort", 120),
            ("restore", None),
        ]


@pytest.mark.parametrize("failed", (False, True), ids=("success", "failure"))
def test_native_transfer_keeps_app_monitor_running_and_closes_lifecycles(
    tmp_path, monkeypatch, failed
):
    import tldw_chatbook
    from Tests.ProductionApp import test_native_credential_recovery as product
    from tldw_chatbook.Backup_Recovery import runtime_maintenance

    entered, monitored = Event(), Event()
    cleanup = []
    thread = get_ident()
    source = tmp_path / "source.age"

    async def closed(name):
        cleanup.append(name)

    app = SimpleNamespace(
        _shutdown_app_owned_lifecycles=lambda: closed("shutdown"),
        tts_service=SimpleNamespace(
            close=lambda: closed("tts-close"),
            wait_closed=lambda: closed("tts-wait"),
        ),
    )

    async def monitor(current):
        assert current is app
        try:
            while not entered.is_set():
                await product.asyncio.sleep(0)
            monitored.set()
            await product.asyncio.Event().wait()
        finally:
            cleanup.append("monitor-stopped")

    def transfer(current):
        assert current == source and get_ident() != thread
        entered.set()
        assert monitored.wait(5), "native_transfer_monitor_not_serviced"
        if failed:
            raise ValueError("scope_changed")

    config = SimpleNamespace(set_encryption_password=lambda password: None)
    monkeypatch.setattr(tldw_chatbook, "config", config, raising=False)
    monkeypatch.setitem(sys.modules, "tldw_chatbook.config", config)
    monkeypatch.setitem(
        sys.modules, "tldw_chatbook.app", SimpleNamespace(TldwCli=lambda: app)
    )
    monkeypatch.setattr(runtime_maintenance, "monitor_app", monitor)
    monkeypatch.setattr(product, "_transfer", transfer)
    if failed:
        with pytest.raises(ValueError, match="scope_changed"):
            product.asyncio.run(product._transfer_with_app(source))
    else:
        product.asyncio.run(product._transfer_with_app(source))
    assert cleanup == ["monitor-stopped", "shutdown", "tts-close", "tts-wait"]


def test_wrong_password_helper_refusal_discards_incomplete_readback(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import archive_reader, crypto
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits

    source = tmp_path / "source.age"
    source.write_bytes(b"age-encryption.org/v1\nsynthetic-encrypted-placeholder")
    work = tmp_path / "wrong-password"

    def reject(source, destination, **options):
        destination.write_bytes(b"synthetic-incomplete-output")
        raise crypto.CryptoError("transform_failed")

    monkeypatch.setattr(crypto, "transform", reject)
    with pytest.raises(crypto.CryptoError, match="transform_failed"):
        archive_reader.acquire(
            source, work, ArchiveLimits(), b"wrong-synthetic-key", Event()
        )
    assert work.is_dir() and not tuple(work.iterdir())


def test_native_worker_failure_is_observed_before_service_maps_it(
    tmp_path, monkeypatch
):
    from Tests.ProductionApp import test_native_credential_recovery as product
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    private, artifacts = tmp_path / "private", tmp_path / "artifacts"
    failure_root = private / "native-failures"
    failure_root.mkdir(parents=True)
    artifacts.mkdir()
    monkeypatch.setenv("TLDW_NATIVE_FAILURE_ROOT", str(failure_root))
    service = RecoveryService(tmp_path / "control")

    def fail_worker(operation, cancel):
        raise RuntimeError("positive-native-secret")

    try:
        assert product._observe_workers(service) is service
        operation = service._start("backup", fail_worker)
        state = service.wait(operation, timeout=5)
        assert state["state"] == "failed"
        assert state["issues"] == ("backup_operation_failed",)
        runner._publish_native_failures(private, artifacts)
        path = artifacts / "native-failures.json"
        receipt = json.loads(path.read_text())
        assert [row["error_class"] for row in receipt["failures"]] == ["RuntimeError"]
        assert any(
            frame["function"] == "fail_worker"
            for row in receipt["failures"]
            for frame in row["frames"]
        )
        assert "positive-native-secret" not in path.read_text()
    finally:
        service.close()


@pytest.mark.parametrize(
    "owner_id", ("db.chachanotes.primary", "positive-native-secret/private.sqlite")
)
@pytest.mark.parametrize(
    "installed_frame", (True, False), ids=("installed", "lookalike")
)
def test_native_worker_sqlite_refusal_keeps_only_installed_owner_and_code(
    tmp_path, monkeypatch, owner_id, installed_frame
):
    from Tests.ProductionApp import test_native_credential_recovery as product
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.Backup_Recovery.sqlite_validation import (
        validated_schema_version as installed_validation,
    )

    private, artifacts = tmp_path / "private", tmp_path / "artifacts"
    failure_root = private / "native-failures"
    failure_root.mkdir(parents=True)
    artifacts.mkdir()
    monkeypatch.setenv("TLDW_NATIVE_FAILURE_ROOT", str(failure_root))
    service = RecoveryService(tmp_path / "control")
    owner = SimpleNamespace(owner_id=owner_id, schema_policy=lambda: None)

    def validated_schema_version(owner, candidate, cancel):
        raise ValueError("unsupported_schema_policy")

    def refused(operation, cancel):
        validate = installed_validation if installed_frame else validated_schema_version
        return validate(owner, tmp_path / "positive-native-secret", cancel)

    try:
        product._observe_workers(service)
        state = service.wait(service._start("backup", refused), timeout=5)
        assert state["issues"] == ("backup_operation_failed",)
        runner._publish_native_failures(private, artifacts)
        path = artifacts / "native-failures.json"
        row = json.loads(path.read_text())["failures"][0]
        if not installed_frame:
            assert "sqlite_owner" not in row and "sqlite_issue" not in row
        elif owner_id == "db.chachanotes.primary":
            assert row["sqlite_owner"] == owner_id
            assert row["sqlite_issue"] == "unsupported_schema_policy"
        else:
            assert "sqlite_owner" not in row
            assert row["sqlite_issue"] == "unsupported_sqlite_owner"
        assert "positive-native-secret" not in path.read_text()
    finally:
        service.close()


@pytest.mark.parametrize(
    "existing_trace", (False, True), ids=("observe", "leave-tracer")
)
def test_native_worker_observes_swallowed_sqlite_failure_and_restores_trace(
    tmp_path, monkeypatch, existing_trace
):
    import sqlite3

    from Tests.ProductionApp import test_native_credential_recovery as product
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validated_schema_version

    private, artifacts = tmp_path / "private", tmp_path / "artifacts"
    failure_root = private / "native-failures"
    failure_root.mkdir(parents=True)
    artifacts.mkdir()
    monkeypatch.setenv("TLDW_NATIVE_FAILURE_ROOT", str(failure_root))
    candidate = tmp_path / "positive-native-secret.sqlite"
    connection = sqlite3.connect(":memory:")
    connection.close()

    def failed_policy():
        connection.execute("SELECT 1")

    owner = SimpleNamespace(
        owner_id="db.chachanotes.primary", schema_policy=failed_policy
    )
    service = RecoveryService(tmp_path / "control")

    def prior_trace(frame, event, argument):
        return prior_trace

    previous = prior_trace if existing_trace else None

    def refused(operation, cancel):
        return validated_schema_version(owner, candidate, cancel)

    try:
        service._executor.submit(sys.settrace, previous).result(timeout=5)
        product._observe_workers(service)
        state = service.wait(service._start("backup", refused), timeout=5)
        assert state["issues"] == ("backup_operation_failed",)
        assert service._executor.submit(sys.gettrace).result(timeout=5) is previous
        runner._publish_native_failures(private, artifacts)
        path = artifacts / "native-failures.json"
        rows = json.loads(path.read_text())["failures"]
        swallowed = [row for row in rows if row["error_class"] == "ProgrammingError"]
        assert bool(swallowed) is not existing_trace
        assert all(row["sqlite_owner"] == "db.chachanotes.primary" for row in swallowed)
        assert all("sqlite_issue" not in row for row in swallowed)
        assert all(
            any(frame["function"] == "failed_policy" for frame in row["frames"])
            for row in swallowed
        )
        assert "positive-native-secret" not in path.read_text()
        assert "Cannot operate on a closed database" not in path.read_text()
    finally:
        service.close()


@pytest.mark.parametrize("diagnostics", ("observe", "leave-tracer", "no-root"))
def test_native_worker_observes_suppressed_archive_io_without_changing_inspection(
    tmp_path, monkeypatch, diagnostics
):
    from contextlib import contextmanager

    from Tests.ProductionApp import test_native_credential_recovery as product
    from tldw_chatbook.Backup_Recovery import archive_reader
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.Backup_Recovery.service_storage import work_root

    private, artifacts = tmp_path / "private", tmp_path / "artifacts"
    failure_root = private / "native-failures"
    failure_root.mkdir(parents=True)
    artifacts.mkdir()
    if diagnostics == "no-root":
        monkeypatch.delenv("TLDW_NATIVE_FAILURE_ROOT", raising=False)
    else:
        monkeypatch.setenv("TLDW_NATIVE_FAILURE_ROOT", str(failure_root))
    source = tmp_path / "positive-native-secret.age"
    source.write_bytes(b"age-encryption.org/v1\nsynthetic-encrypted-placeholder")
    regular = archive_reader._regular

    @contextmanager
    def unavailable_input(path):
        with regular(path) as incoming:

            def failed_read(size):
                if size == 20:
                    return incoming.read(size)
                raise OSError(5, "positive-native-secret", str(source))

            yield SimpleNamespace(
                read=failed_read, seek=incoming.seek, fileno=incoming.fileno
            )

    monkeypatch.setattr(archive_reader, "_regular", unavailable_input)
    service = RecoveryService(tmp_path / "control")

    def prior_trace(frame, event, argument):
        return prior_trace

    previous = prior_trace if diagnostics == "leave-tracer" else None
    try:
        service._executor.submit(sys.settrace, previous).result(timeout=5)
        product._observe_workers(service)
        operation = service.start_inspection(source, password=None)
        state = service.wait(operation, timeout=5)
        assert state["state"] == "failed"
        assert state["issues"] == ("backup_operation_failed",)
        assert service._executor.submit(sys.gettrace).result(timeout=5) is previous
        work = work_root(service.control_root) / ("inspection-" + operation)
        acquired = work / "acquired"
        assert acquired.is_dir() and not any(acquired.iterdir())
        with pytest.raises(ValueError, match="^archive_inspection_required$"):
            service.inspection(operation)
        runner._publish_native_failures(private, artifacts)
        path = artifacts / "native-failures.json"
        if diagnostics == "no-root":
            assert not path.exists() and not any(failure_root.iterdir())
        else:
            text = path.read_text()
            rows = json.loads(text)["failures"]
            original = [row for row in rows if row["error_class"] == "OSError"]
            assert bool(original) is (diagnostics == "observe")
            assert any(row["error_class"] == "ValueError" for row in rows)
            for row in original:
                assert set(row) == {"error_class", "frames", "issue"}
                assert any(
                    frame["file"] == "archive_reader.py"
                    and frame["function"] == "acquire"
                    for frame in row["frames"]
                )
                assert any(
                    frame["function"] == "failed_read" for frame in row["frames"]
                )
            assert "positive-native-secret" not in text and '"errno"' not in text
    finally:
        service.close()
    assert not work.exists()


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        ("shm", "declared_shm"),
        ("wal", "declared_wal"),
        ("main", "declared_main"),
        ("other", "other"),
        ("shared", "declared_shm"),
        ("ordinary-competitor", "ambiguous"),
        ("duplicate", "ambiguous"),
        ("missing-main", "ambiguous"),
        ("excluded-main", "ambiguous"),
        ("raw-payload", "ambiguous"),
        ("classifier-failure", None),
        ("instance", "declared_instance_lock"),
        ("instance-status", "ambiguous"),
        ("instance-competitor", "ambiguous"),
    ),
)
def test_native_failure_classifies_only_exact_fingerprint_read_chain(
    tmp_path, monkeypatch, case, expected
):
    from contextlib import contextmanager
    from dataclasses import replace

    from tldw_chatbook.Backup_Recovery import (
        archive_reader,
        owner_registry,
        restore_plan,
    )
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
        Inventory,
        StorageItem,
    )

    private, artifacts = tmp_path / "private", tmp_path / "artifacts"
    failure_root = private / "native-failures"
    failure_root.mkdir(parents=True)
    artifacts.mkdir()
    main_path = tmp_path / ("payload.bin" if case == "raw-payload" else "main.sqlite")
    main = StorageItem(
        "recovered.media" if case == "raw-payload" else "research.local",
        "positive-native-secret-main",
        main_path,
        "intentionally_excluded" if case == "excluded-main" else "included",
        (),
        shared_group="synthetic-shared" if case == "shared" else None,
    )
    path = (
        main_path
        if case == "main"
        else Path(str(main_path) + ("-wal" if case == "wal" else "-shm"))
    )
    sidecar = StorageItem(
        "sqlite.transient",
        "positive-native-secret-sidecar",
        path,
        "intentionally_excluded",
        (main.logical_id,),
    )
    items = (main,) if case == "main" else (main, sidecar)
    if case == "other":
        items = (
            replace(sidecar, owner="ui.state", status="included", dependencies=()),
        )
    elif case == "ordinary-competitor":
        items += (
            replace(
                sidecar,
                owner="ui.state",
                logical_id="ordinary",
                status="included",
                dependencies=(),
            ),
        )
    elif case == "duplicate":
        items += (sidecar,)
    elif case == "missing-main":
        items = (sidecar,)
    elif case == "shared":
        peer = replace(main, logical_id="peer")
        items += (
            peer,
            replace(
                sidecar, logical_id="peer-sidecar", dependencies=(peer.logical_id,)
            ),
        )
    elif case.startswith("instance"):
        data = tmp_path / "Ada"
        data.mkdir()
        selector = tmp_path / "config.toml"
        selector.write_text("")
        config = {
            "paths": {"data_dir": str(tmp_path)},
            "general": {"users_name": "Ada"},
            DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, "p"),
        }
        path = data / ".instance.lock"
        path.write_bytes(b"positive-native-secret")
        owners = {row.owner_id: row for row in owner_registry.install_adapters()}
        lock = owners["runtime.instance_lock"].discover(config)[0]
        items = (*owners["config"].discover(config), lock)
        if case == "instance-status":
            items = (items[0], replace(lock, status="included"))
        elif case == "instance-competitor":
            items += (replace(lock, owner="ui.state", logical_id="ordinary"),)
    path.write_bytes(b"positive-native-secret")
    original = path.read_bytes()

    @contextmanager
    def write_only(path):
        with path.open("ab") as stream:
            yield stream

    monkeypatch.setattr(archive_reader, "_regular", write_only)
    if case == "classifier-failure":

        def unavailable_declarations():
            raise RuntimeError("positive-native-secret")

        monkeypatch.setattr(
            owner_registry, "install_adapters", unavailable_declarations
        )
    with pytest.raises(OSError) as caught:
        restore_plan._fingerprint((path,), Inventory(items, True, "local", ()))
    runner._record_native_failure(failure_root, caught.value)
    runner._publish_native_failures(private, artifacts)
    receipt_path = artifacts / "native-failures.json"
    rows = json.loads(receipt_path.read_text())["failures"]
    assert [row.get("target_kind") for row in rows] == [expected]
    assert [frame["function"] for frame in rows[0]["frames"]][-3:] == [
        "_fingerprint",
        "_observed",
        "_hash",
    ]
    assert path.read_bytes() == original
    assert "positive-native-secret" not in receipt_path.read_text()


def test_native_failure_does_not_classify_fabricated_fingerprint_names(tmp_path):
    private, artifacts = tmp_path / "private", tmp_path / "artifacts"
    failure_root = private / "native-failures"
    failure_root.mkdir(parents=True)
    artifacts.mkdir()

    def _hash(path):
        raise PermissionError(13, "positive-native-secret", str(path))

    def _observed(path):
        return _hash(path)

    def _fingerprint(path, target):
        return _observed(path)

    with pytest.raises(PermissionError) as caught:
        _fingerprint(tmp_path / "positive-native-secret", object())
    runner._record_native_failure(
        failure_root, caught.value, metadata={"target_kind": "declared_shm"}
    )
    runner._publish_native_failures(private, artifacts)
    receipt_path = artifacts / "native-failures.json"
    rows = json.loads(receipt_path.read_text())["failures"]
    assert rows[0]["error_class"] == "PermissionError" and rows[0]["errno"] == 13
    assert "target_kind" not in rows[0]
    assert "positive-native-secret" not in receipt_path.read_text()


@pytest.mark.parametrize(
    ("errno", "winerror", "expected"),
    (
        (13, 5, {"errno": 13, "winerror": 5}),
        (13, 32, {"errno": 13, "winerror": 32}),
        (13, 33, {"errno": 13, "winerror": 33}),
        (None, None, {}),
        (True, True, {}),
        (14, 34, {}),
        ("13", "33", {}),
    ),
)
def test_native_failure_revalidates_only_fixed_os_codes(
    tmp_path, errno, winerror, expected
):
    private, artifacts = tmp_path / "private", tmp_path / "artifacts"
    failure_root = private / "native-failures"
    failure_root.mkdir(parents=True)
    artifacts.mkdir()
    error = PermissionError("positive-native-secret")
    error.errno, error.winerror = errno, winerror
    runner._record_native_failure(failure_root, error)
    (failure_root / "untrusted.json").write_text(
        json.dumps(
            {
                "error_class": "PermissionError",
                "frames": [],
                "errno": errno,
                "winerror": winerror,
                "target_kind": "positive-native-secret/path",
                "filename": "positive-native-secret",
                "message": "positive-native-secret",
            }
        )
    )
    runner._publish_native_failures(private, artifacts)
    receipt_path = artifacts / "native-failures.json"
    rows = json.loads(receipt_path.read_text())["failures"]
    assert all(
        {key: row[key] for key in ("errno", "winerror") if key in row} == expected
        for row in rows
    )
    assert all("target_kind" not in row for row in rows)
    assert "positive-native-secret" not in receipt_path.read_text()


def test_native_failure_optional_diagnostics_revalidate_fixed_values(tmp_path):
    private, artifacts = tmp_path / "private", tmp_path / "artifacts"
    failure_root = private / "native-failures"
    failure_root.mkdir(parents=True)
    artifacts.mkdir()
    metadata = {
        "sqlite_owner": "db.chachanotes.primary",
        "sqlite_issue": "unsupported_schema",
        "inventory": {
            "issues": ["undeclared_alias"] * 100 + ["positive-native-secret"],
            "blocking": [{"owner": "notes.file_notes", "status": "missing_required"}]
            * 100
            + [
                {"owner": "unknown", "status": "unsupported"},
                {"owner": "sqlite.transient", "status": "unsupported"},
                {
                    "owner": "positive-native-secret/private.sqlite",
                    "status": "unsupported",
                },
                {"owner": "notes.file_notes", "status": "positive-native-secret"},
            ],
            "path": "positive-native-secret",
        },
    }
    runner._record_native_failure(
        failure_root, ValueError("positive-native-secret"), metadata=metadata
    )
    raw = {"error_class": "ValueError", "frames": [], **metadata}
    raw.update(
        sqlite_owner="positive-native-secret/private.sqlite",
        sqlite_issue="positive-native-secret",
        message="positive-native-secret",
    )
    (failure_root / "untrusted.json").write_text(json.dumps(raw))
    runner._publish_native_failures(private, artifacts)
    path = artifacts / "native-failures.json"
    rows = json.loads(path.read_text())["failures"]
    assert [(row.get("sqlite_owner"), row.get("sqlite_issue")) for row in rows] == [
        ("db.chachanotes.primary", "unsupported_schema"),
        (None, None),
    ]
    assert all(
        row["inventory"]
        == {
            "issues": ["undeclared_alias"],
            "blocking": [
                {"owner": "notes.file_notes", "status": "missing_required"},
                {"owner": "sqlite.transient", "status": "unsupported"},
                {"owner": "unknown", "status": "unsupported"},
            ],
        }
        for row in rows
    )
    assert "positive-native-secret" not in path.read_text()


@pytest.mark.parametrize(
    "blocking", (False, True), ids=("issue-only", "blocking-owner")
)
@pytest.mark.parametrize(
    "sqlite_refusal", (False, True), ids=("sqlite-ready", "sqlite-refusal")
)
def test_native_capture_observes_incomplete_inventory_before_refusal(
    tmp_path, monkeypatch, blocking, sqlite_refusal
):
    import sqlite3

    import tldw_chatbook
    from Tests.ProductionApp import test_native_credential_recovery as product
    from tldw_chatbook.Backup_Recovery import recovery_service, runtime_maintenance

    private, artifacts = tmp_path / "private", tmp_path / "artifacts"
    failure_root = private / "native-failures"
    failure_root.mkdir(parents=True)
    artifacts.mkdir()
    monkeypatch.setenv("TLDW_NATIVE_FAILURE_ROOT", str(failure_root))
    monkeypatch.setenv("TLDW_CREDENTIAL_TRANSFER_ROOT", str(tmp_path))

    async def idle(*args):
        pass

    service = recovery_service.RecoveryService(tmp_path / "control")
    items = (
        (
            SimpleNamespace(owner="db.chachanotes.primary", status="unavailable"),
            SimpleNamespace(owner="positive-native-secret", status="unsupported"),
            SimpleNamespace(owner="notes.file_notes", status="included"),
        )
        if blocking
        else ()
    )
    monkeypatch.setattr(
        service,
        "preview_backup_details",
        lambda *args, **kwargs: {
            "inventory": SimpleNamespace(
                complete=False,
                issues=("undeclared_alias", "positive-native-secret"),
                items=items,
            )
        },
    )
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
    connections = []
    connect = sqlite3.connect

    def memory_connection(location):
        assert location == ":memory:"
        connection = connect(location)
        connections.append(connection)
        return SimpleNamespace(close=connection.close) if sqlite_refusal else connection

    monkeypatch.setattr(sqlite3, "connect", memory_connection)
    with pytest.raises(AssertionError, match="native_inventory_incomplete"):
        product.asyncio.run(product._capture())
    assert len(connections) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        connections[0].execute("SELECT 1")
    runner._publish_native_failures(private, artifacts)
    path = artifacts / "native-failures.json"
    rows = json.loads(path.read_text())["failures"]
    assert [row["inventory"] for row in rows if "inventory" in row] == [
        {
            "issues": ["undeclared_alias"],
            "blocking": [{"owner": "db.chachanotes.primary", "status": "unavailable"}]
            if blocking
            else [],
        }
    ]
    assert [row["sqlite_issue"] for row in rows if "sqlite_issue" in row] == (
        ["sqlite_security_unavailable"] if sqlite_refusal else []
    )
    causes = [row for row in rows if row["error_class"] == "AttributeError"]
    assert len(causes) == int(sqlite_refusal)
    assert all(
        any(
            frame["file"] == "sqlite_validation.py"
            and frame["function"] == "_restrict_connection"
            for frame in row["frames"]
        )
        for row in causes
    )
    assert "enable_load_extension" not in path.read_text()
    assert "positive-native-secret" not in path.read_text()


@pytest.mark.parametrize(
    "corrupt_copy,phase_timeout,expected_timeout",
    [
        (None, 4800, 4800),
        ("ciphertext", 4800, 4800),
        ("receipt", 4800, 4800),
        (None, None, 900),
        (None, 0, 900),
        (None, -1, None),
        (None, float("nan"), None),
        (None, float("inf"), None),
        (None, "4800", None),
        (None, False, None),
    ],
)
def test_native_destinations_seed_retargeted_profile_before_negative_checks(
    tmp_path, monkeypatch, corrupt_copy, phase_timeout, expected_timeout
):
    import shutil

    from Tests.ProductionApp import test_native_credential_recovery as product

    transfer = tmp_path / "transfer"
    transfer.mkdir()
    run = tmp_path / "run"
    run.mkdir()
    for system in ("Darwin", "Linux", "Windows"):
        source = transfer / f"source-{system.lower()}.age"
        source.write_bytes(b"synthetic-encrypted-placeholder")
        source.with_suffix(".json").write_text(
            json.dumps(
                {
                    "system": system,
                    "status": "passed",
                    "archive_sha256": product._digest(source),
                }
            )
        )
    originals = {path: path.read_bytes() for path in transfer.iterdir()}
    copyfileobj = shutil.copyfileobj

    def copied(source, destination):
        is_receipt = source.read(1) == b"{"
        source.seek(0)
        result = copyfileobj(source, destination)
        if corrupt_copy == "ciphertext" and not is_receipt:
            destination.seek(0)
            destination.truncate()
            destination.write(b"changed ciphertext")
        elif corrupt_copy == "receipt" and is_receipt:
            source.seek(0)
            receipt = json.loads(source.read())
            receipt["status"] = "failed"
            destination.seek(0)
            destination.truncate()
            destination.write(json.dumps(receipt).encode())
        return result

    monkeypatch.setattr(shutil, "copyfileobj", copied)
    setups, completed_forward, reversed_roots = {}, set(), []
    children, owned_sources = [], {}

    def child(root, installed, route, *, role="retargeted", source=None, timeout=900):
        children.append((route, role))
        assert timeout == (
            expected_timeout if route in {"transfer", "rollback"} else 900
        )
        if source is not None:
            original = transfer / source.name
            assert source != original and source.read_bytes() == originals[original]
            assert (
                source.with_suffix(".json").read_bytes()
                == originals[original.with_suffix(".json")]
            )
            assert (
                product._digest(source)
                == json.loads(source.with_suffix(".json").read_text())["archive_sha256"]
            )
        if route in {"transfer", "rollback"}:
            assert source.parent == root
        if route == "setup":
            setups.setdefault(root, []).append(role)
        elif route == "transfer":
            assert not (root / "direction.json").exists()
            completed_forward.add(root)
            owned_sources[root] = source
        elif route == "rollback":
            assert root in completed_forward
            assert source == owned_sources[root]
            assert not (root / "direction.json").exists()
            reversed_roots.append(root)
            (root / "direction.json").write_text("{}")
        else:
            assert route == "negative"
            assert setups.get(root) == ["retargeted"]
            assert source == owned_sources[run / "Darwin"]
            (root / "negative-results.json").write_text('{"negative_checks":7}')

    monkeypatch.setenv("TLDW_CREDENTIAL_TRANSFER_ROOT", str(transfer))

    def preflight():
        assert expected_timeout is not None, "invalid timeout reached native preflight"

    monkeypatch.setattr(runner, "validate_native_credential_environment", preflight)
    monkeypatch.setattr(product, "_child", child)
    monkeypatch.setattr(product, "_receipt", lambda installed, **extra: extra)
    request = SimpleNamespace(
        config=SimpleNamespace(
            getoption=lambda name: (
                phase_timeout
                if name == "timeout"
                else pytest.fail("unexpected pytest option")
            )
        )
    )
    if expected_timeout is None:
        with pytest.raises(ValueError, match="native_child_timeout_invalid"):
            product.test_native_credential_destinations(
                run, tmp_path / "installed", request
            )
        assert not children
    elif corrupt_copy is not None:
        with pytest.raises(AssertionError):
            product.test_native_credential_destinations(
                run, tmp_path / "installed", request
            )
        assert not children
    else:
        product.test_native_credential_destinations(
            run, tmp_path / "installed", request
        )
        assert reversed_roots == [run / name for name in ("Darwin", "Linux", "Windows")]
        assert (
            json.loads((transfer / "outbound/destination-results.json").read_text())[
                "negative_checks"
            ]
            == 7
        )
    assert all(path.read_bytes() == before for path, before in originals.items())


@pytest.mark.parametrize("linked", ["archive", "receipt", "parent", "root"])
@pytest.mark.skipif(os.name == "nt", reason="POSIX fixture requires symlink privileges")
def test_native_destinations_refuse_transfer_links_before_reads_or_effects(
    tmp_path, monkeypatch, linked
):
    from Tests.ProductionApp import test_native_credential_recovery as product

    parent = tmp_path / "incoming"
    transfer = parent / "transfer"
    transfer.mkdir(parents=True, mode=0o700)
    run = tmp_path / "run"
    run.mkdir(mode=0o700)
    for system in ("Darwin", "Linux", "Windows"):
        source = transfer / f"source-{system.lower()}.age"
        source.write_bytes(b"synthetic-encrypted-placeholder")
        source.with_suffix(".json").write_text(
            json.dumps(
                {
                    "system": system,
                    "status": "passed",
                    "archive_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            )
        )
    outside = tmp_path / "outside"
    if linked in {"archive", "receipt"}:
        source = transfer / (
            "source-windows.age" if linked == "archive" else "source-windows.json"
        )
        source.rename(outside)
        source.symlink_to(outside)
        originals = {outside: outside.read_bytes()}
    else:
        selected = parent if linked == "parent" else transfer
        selected.rename(outside)
        selected.symlink_to(outside, target_is_directory=True)
        originals = {
            path: path.read_bytes() for path in outside.rglob("*") if path.is_file()
        }
    effects = []
    monkeypatch.setenv("TLDW_CREDENTIAL_TRANSFER_ROOT", str(transfer))
    monkeypatch.setattr(
        runner,
        "validate_native_credential_environment",
        lambda: effects.append("native-preflight"),
    )
    monkeypatch.setattr(
        product, "_child", lambda *args, **kwargs: effects.append("native-child")
    )
    request = SimpleNamespace(config=SimpleNamespace(getoption=lambda name: 900))
    with pytest.raises((OSError, ValueError, AssertionError)):
        product.test_native_credential_destinations(
            run, tmp_path / "installed", request
        )
    assert not effects
    assert not list(run.iterdir())
    assert all(path.read_bytes() == before for path, before in originals.items())


@pytest.mark.parametrize("timeout", [None, 4800.0], ids=["default", "phase"])
@pytest.mark.parametrize("outcome", ["success", "nonzero", "timeout", "interrupt"])
def test_native_child_deadline_preserves_failure_and_discards_output(
    tmp_path, monkeypatch, timeout, outcome
):
    from Tests.ProductionApp import test_native_credential_recovery as product

    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    installed, source = tmp_path / "installed", root / "source.age"
    environment = {"HOME": str(root / "home")}
    calls = []
    monkeypatch.setattr(product, "_environment", lambda *args, **kwargs: environment)

    def run(arguments, **kwargs):
        calls.append(kwargs["timeout"])
        assert arguments == [
            sys.executable,
            str(Path(product.__file__).resolve()),
            "transfer",
            "retargeted",
            str(source),
        ]
        assert kwargs["cwd"] == root and kwargs["env"] == environment
        assert kwargs["stderr"] == product.subprocess.STDOUT
        assert kwargs["stdout"] == product.subprocess.DEVNULL
        assert kwargs["check"] is False
        if outcome == "timeout":
            raise product.subprocess.TimeoutExpired(
                arguments, kwargs["timeout"], output="positive-native-secret"
            )
        if outcome == "interrupt":
            raise KeyboardInterrupt
        return SimpleNamespace(returncode=int(outcome == "nonzero"))

    monkeypatch.setattr(product.subprocess, "run", run)

    def child():
        return product._child(
            root,
            installed,
            "transfer",
            source=source,
            **({} if timeout is None else {"timeout": timeout}),
        )

    if outcome == "success":
        assert child() is None
    else:
        failure = {
            "nonzero": AssertionError,
            "timeout": product.subprocess.TimeoutExpired,
            "interrupt": KeyboardInterrupt,
        }[outcome]
        with pytest.raises(failure) as caught:
            child()
        assert "positive-native-secret" not in str(caught.value)
        if outcome == "nonzero":
            assert str(caught.value).startswith("native_child_failed:transfer:1")
    assert calls == [900 if timeout is None else timeout]
    assert not list(root.iterdir())


@pytest.mark.parametrize("outcome", ["success", "nonzero", "timeout"])
def test_native_child_canary_streams_are_never_persisted_or_exported(
    tmp_path, monkeypatch, capfd, outcome
):
    from Tests.ProductionApp import test_native_credential_recovery as product

    script = tmp_path / "emit-canary.py"
    script.write_text(
        "import sys,time\nfrom pathlib import Path\n"
        "print('positive-native-secret',flush=True)\n"
        "print('positive-native-secret',file=sys.stderr,flush=True)\n"
        "Path(sys.argv[3]).touch()\n"
        + ("time.sleep(30)\n" if outcome == "timeout" else "")
        + f"sys.exit({int(outcome == 'nonzero')})\n"
    )
    root = tmp_path / "private" / "product-pytest" / "child"
    root.mkdir(parents=True, mode=0o700)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setattr(product, "__file__", str(script))
    monkeypatch.setattr(
        product, "_environment", lambda *args, **kwargs: os.environ.copy()
    )
    failure = {
        "nonzero": AssertionError,
        "timeout": product.subprocess.TimeoutExpired,
    }.get(outcome)
    ready = tmp_path / "emitted"

    def child():
        product._child(
            root,
            tmp_path / "installed",
            "transfer",
            source=ready,
            timeout=2 if outcome == "timeout" else 30,
        )

    if failure is None:
        child()
    else:
        with pytest.raises(failure):
            child()
    assert ready.is_file()
    captured = capfd.readouterr()
    assert "positive-native-secret" not in captured.out + captured.err
    assert not list(root.iterdir())
    assert runner._collect_safe_logs(tmp_path / "private", artifacts) == 0
    assert not list(artifacts.iterdir())


@pytest.mark.parametrize(
    "timeout", [-1, 0, None, float("nan"), float("inf"), "4800", False]
)
def test_native_child_refuses_invalid_deadline_before_environment(
    tmp_path, monkeypatch, timeout
):
    from Tests.ProductionApp import test_native_credential_recovery as product

    root = tmp_path / "native-child-private"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(
        product,
        "_environment",
        lambda *args, **kwargs: pytest.fail("invalid timeout touched environment"),
    )
    with pytest.raises(ValueError, match="native_child_timeout_invalid"):
        product._child(root, tmp_path / "installed", "transfer", timeout=timeout)
    assert not list(root.iterdir())


def test_native_rollback_child_diagnostics_keep_only_safe_frames(tmp_path):
    private, artifacts = tmp_path / "private", tmp_path / "artifacts"
    stacks = private / "native-failures/child-stacks"
    stacks.mkdir(parents=True)
    artifacts.mkdir()
    (stacks / "rollback--retargeted--123.json").write_text(
        json.dumps(
            [
                [
                    {
                        "frames": [
                            {
                                "file": "test_native_credential_recovery.py",
                                "function": "_rollback",
                                "line": 1,
                            }
                        ],
                        "locals": "positive-native-secret",
                    }
                ]
            ]
        )
    )
    runner._publish_native_failures(private, artifacts)
    published = (artifacts / "native-failures.json").read_text()
    assert json.loads(published)["children"][0]["route"] == "rollback"
    assert "positive-native-secret" not in published and '"locals"' not in published


@pytest.mark.parametrize("damage", ("source", "archive_sha256", "source_system"))
def test_native_rollback_validates_forward_source_before_service_effects(
    tmp_path, monkeypatch, damage
):
    from Tests.ProductionApp import test_native_credential_recovery as product
    from tldw_chatbook.Backup_Recovery import recovery_service

    monkeypatch.chdir(tmp_path)
    for name in ("tldw_chatbook.app", "tldw_chatbook.config"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    source = tmp_path / "source.age"
    source.write_bytes(b"synthetic-encrypted-placeholder")
    source.with_suffix(".json").write_text('{"system":"Linux"}')
    handoff = {
        "source": str(source),
        "result": {
            "archive_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_system": "Linux",
        },
    }
    if damage == "source":
        handoff["source"] = str(tmp_path / "unreviewed.age")
    else:
        handoff["result"][damage] = "unreviewed"
    (tmp_path / "transfer-forward.json").write_text(json.dumps(handoff))
    monkeypatch.setattr(
        recovery_service,
        "RecoveryService",
        lambda root: pytest.fail("rollback_service_before_source_validation"),
    )
    with pytest.raises(AssertionError):
        product._rollback(source)
    assert not (tmp_path / "direction.json").exists()


@pytest.mark.parametrize("role", ("default", "retargeted"))
@pytest.mark.parametrize("rollback", (False, True))
def test_native_fresh_reader_opens_only_its_selected_profile(
    tmp_path, monkeypatch, role, rollback
):
    import keyring

    from Tests.ProductionApp import test_native_credential_recovery as product
    from tldw_chatbook.Chat import citation_trace_identity
    from tldw_chatbook.MCP import server_target_store
    from tldw_chatbook.Utils.config_encryption import ConfigEncryption

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    selector = product._selectors()[role == "retargeted"]
    selector.parent.mkdir(mode=0o700, parents=True)
    selector.write_text(
        f'[general]\nusers_name="native_{role}"\n'
        + ('[API]\nopenai_api_key="synthetic-encrypted"\n' if rollback else "")
    )
    key = b"synthetic-citation-key"
    (tmp_path / f"seed-{role}.json").write_text(
        json.dumps(
            {
                "citation_id": "synthetic",
                "citation_sha256": hashlib.sha256(key).hexdigest(),
            }
        )
    )
    monkeypatch.setattr(
        citation_trace_identity,
        "KeyringCitationFingerprintKeyProvider",
        lambda: SimpleNamespace(load_key=lambda key_id: key),
    )
    monkeypatch.setattr(
        ConfigEncryption,
        "decrypt_value",
        lambda self, value, password: product._value(
            product.platform.system(), role, "encrypted_config"
        ),
    )
    opened = []
    targets = [SimpleNamespace(server_id=purpose) for purpose in product.PURPOSES]

    def store(path):
        assert (
            path.name == "mcp_server_targets.json"
            and path.parent.name == "native_" + role
        )
        opened.append(path)
        return SimpleNamespace(list_targets=lambda: targets)

    def provider(path):
        assert (
            path == selector or role == "retargeted" and path == product._selectors()[0]
        )
        return SimpleNamespace(
            _resolve_auth_token=lambda origin, target, **kwargs: (
                product._value(
                    product.platform.system() if rollback else "source", role, origin
                ),
                "synthetic",
            ),
            _get_credential_secret=lambda origin, purpose: product._value(
                product.platform.system(),
                role if path == selector else "foreign",
                purpose,
            ),
        )

    monkeypatch.setattr(server_target_store, "ConfiguredServerTargetStore", store)
    monkeypatch.setattr(product, "_provider", provider)
    monkeypatch.setattr(
        keyring,
        "get_password",
        lambda service, name: product._value(
            product.platform.system(), "generation", name
        ),
    )
    product._fresh_readback("source", role=role, rollback=rollback)
    assert len(opened) == 1


def test_native_fixture_previews_owner_reads_under_protected_context(
    tmp_path, monkeypatch
):
    import tldw_chatbook
    from Tests.ProductionApp import test_native_credential_recovery as product
    from tldw_chatbook.Backup_Recovery import (
        capture_service,
        inventory,
        owner_registry,
        recovery_service,
        storage_admission,
    )

    class DiscoveryReached(Exception):
        pass

    monkeypatch.setattr(owner_registry, "_adapters", {})
    config = SimpleNamespace(set_encryption_password=lambda password: None)
    monkeypatch.setattr(tldw_chatbook, "config", config, raising=False)
    monkeypatch.setitem(sys.modules, "tldw_chatbook.config", config)
    service = recovery_service.RecoveryService(tmp_path / "control")
    monkeypatch.setattr(recovery_service, "RecoveryService", lambda root: service)
    monkeypatch.setattr(recovery_service, "default_control_root", lambda: tmp_path)
    selector = tmp_path / "config.toml"
    selector.write_bytes(b"[general]\n")
    selector.chmod(0o600)
    monkeypatch.setattr(product, "_selectors", lambda: (selector,))

    def discover(selectors, *, selections):
        assert selectors == (selector,)
        assert "db.prompts.primary" in {
            row.owner_id for row in owner_registry.registered()
        }
        assert storage_admission._local.preview_scope is not None
        assert (
            storage_admission._read_recovery_file("config", selector, max_bytes=1024)
            == b"[general]\n"
        )
        raise DiscoveryReached()

    source = tmp_path / "source-linux.age"
    source.with_suffix(".json").write_text(json.dumps({"system": "Linux"}))
    monkeypatch.setattr(capture_service, "discover", discover)
    monkeypatch.setattr(
        inventory,
        "discover",
        lambda *args, **kwargs: pytest.fail("public_preview_required"),
    )
    try:
        with pytest.raises(DiscoveryReached):
            product._transfer(source)
        assert storage_admission._local.preview_scope is None
    finally:
        service.close()


@pytest.mark.parametrize("profile", ("default", "retargeted"))
def test_native_source_groups_cover_active_evals_and_keep_prompts_unselected(
    tmp_path, profile
):
    from Tests.ProductionApp import test_native_credential_recovery as product
    from tldw_chatbook.Backup_Recovery.data_groups import resolve_inventory_groups
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
        Inventory,
        StorageItem,
        storage_logical_id,
    )
    from tldw_chatbook.Backup_Recovery.restore_groups import required_target_groups
    from tldw_chatbook.Evals.recovery import recovery_adapters

    path = tmp_path / "evals.db"
    path.touch(mode=0o600)
    context = DiscoveryContext(tmp_path / "config.toml", profile)
    config = {
        DISCOVERY_CONTEXT_KEY: context,
        "database": {"evals_db_path": str(path)},
    }
    evals = next(row for row in recovery_adapters() if row.owner_id == "db.evals")
    item = evals.discover(config)[0]
    assert item.status == "included"
    items = tuple(
        StorageItem(owner, storage_logical_id(context, owner), None, "included", ())
        for owner in ("config", "db.chachanotes.primary", "db.prompts.primary")
    )
    target = Inventory((*items, item), True, "fixture", ())
    assert required_target_groups(target, product.GROUPS) == frozenset()
    scope = resolve_inventory_groups(target.items, product.GROUPS)
    assert item.logical_id in scope.member_ids
    assert items[-1].logical_id not in scope.member_ids


@pytest.mark.parametrize(
    "stop_mode", ("isolated", "replace", "dependency-reviewed", "review-retries")
)
def test_native_previews_keep_mode_specific_review_and_private_setup_parent(
    tmp_path, monkeypatch, stop_mode
):
    import tldw_chatbook
    from Tests.Backup_Recovery.test_rollback_dependency_preflight import dependency_plan
    from Tests.ProductionApp import test_native_credential_recovery as product
    from tldw_chatbook.Backup_Recovery import (
        archive_reader,
        inventory,
        isolated_restore,
        profile_catalog,
        recovery_service,
    )

    class PreviewReached(Exception):
        pass

    root, incoming = tmp_path / "child", tmp_path / "incoming"
    root.mkdir(mode=0o700)
    incoming.mkdir(mode=0o700)
    home = root / "home"
    home.mkdir(mode=0o700)
    control = home / "control"
    control.mkdir(mode=0o700)
    monkeypatch.chdir(root)
    source = incoming / "source-linux.age"
    with product.zipfile.ZipFile(source, "w"):
        pass
    source.with_suffix(".json").write_text(json.dumps({"system": "Linux"}))
    stores = []
    for number in range(2):
        path = home / f"unselected-{number}.sqlite"
        path.write_bytes(b"synthetic-unselected")
        stores.append(
            SimpleNamespace(path=path, owner="db.prompts.primary", status="included")
        )
    initial_plan = dependency_plan(root)
    previews, reverse_previews, discoveries = [], [], []
    acknowledgments = (
        (),
        ("credential_isolated_retention_required",),
        (
            "credential_isolated_retention_required",
            "credential_manual_recovery_required:target-material",
        ),
    )

    class Service:
        def __init__(self, root):
            self.control_root = root

        def start_inspection(self, source, *, password):
            return "inspection"

        def wait(self, operation, *, timeout):
            return {"state": "succeeded"}

        def inspection(self, operation):
            return SimpleNamespace(path=source)

        def preview_backup(self, selectors, *, options):
            assert selectors == product._selectors() and options == {}, (
                "local target inventory must include safety-only dependencies"
            )
            discoveries.append(selectors)
            if len(discoveries) > 1:
                return initial_plan.target
            return SimpleNamespace(items=stores, complete=True)

        def preview_restore(self, inspection, **options):
            if options["mode"] == "isolated":
                assert "acknowledged_credential_issues" not in options
                if stop_mode != "isolated":
                    return SimpleNamespace()
            else:
                assert options["mode"] == "replace"
                setup = options["setup_parent"]
                assert setup == root / "files-needing-setup" and setup.is_dir()
                if hasattr(os, "getuid"):
                    assert stat.S_IMODE(setup.stat().st_mode) == 0o700
                    assert setup.stat().st_uid == os.getuid()
                for protected in (home, control, incoming):
                    assert setup != protected and protected not in setup.parents
                    assert setup not in protected.parents
                if stop_mode == "review-retries":
                    from dataclasses import replace

                    assert options["target"] is initial_plan.target
                    assert (
                        options["acknowledged_credential_issues"]
                        == acknowledgments[len(previews) // 2]
                    )
                    previews.append(options)
                    return replace(
                        initial_plan, safety_scope=options.get("safety_scope", ())
                    )
                assert options["acknowledged_credential_issues"] == ()
                if stop_mode == "dependency-reviewed":
                    assert options["target"] is initial_plan.target
                    if not previews:
                        previews.append(options)
                        return initial_plan
                    assert options == {
                        **previews[0],
                        "safety_scope": ("asset", "assets"),
                    }
            raise PreviewReached()

        def preview_rollback(self, operation, **options):
            assert operation == "original" and options["target"] is initial_plan.target
            assert (
                options["acknowledged_credential_issues"]
                == acknowledgments[len(reverse_previews)]
            )
            reverse_previews.append(options)
            return initial_plan

        def recovery_copies(self):
            return [SimpleNamespace(operation_id="original", status="verified")]

        def start_copy_inspection(self, operation, **options):
            return "copy-inspection"

        def start_restore(self, inspection, plan, **options):
            if options:
                raise AssertionError("replacement started before dependency review")
            return "isolated-restore"

        def profiles(self):
            return [{"profile_id": role} for role in ("default", "retargeted")]

        def close(self):
            pass

    config = SimpleNamespace(set_encryption_password=lambda password: None)
    monkeypatch.setattr(tldw_chatbook, "config", config, raising=False)
    monkeypatch.setitem(sys.modules, "tldw_chatbook.config", config)
    monkeypatch.setattr(recovery_service, "RecoveryService", Service)
    monkeypatch.setattr(recovery_service, "default_control_root", lambda: control)
    monkeypatch.setattr(
        inventory,
        "discover",
        lambda *args, **kwargs: pytest.fail("public_preview_required"),
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
    selector = home / "separate.toml"
    selector.write_text('[general]\nusers_name="separate_fixture"\n')
    monkeypatch.setattr(
        profile_catalog,
        "ProfileCatalog",
        lambda root: SimpleNamespace(resolve=lambda profile_id: (selector, None)),
    )
    monkeypatch.setattr(isolated_restore, "_launch_environment", dict)
    monkeypatch.setattr(
        product.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0)
    )
    retained = control / "isolated-fixture"
    retained.mkdir(mode=0o700)
    (retained / "credentials.age").write_bytes(source.read_bytes())
    monkeypatch.setenv("HOME", str(home))
    if stop_mode == "review-retries":
        monkeypatch.setenv("TLDW_TEST_INSTALLED_PACKAGE", str(tmp_path / "installed"))
        rounds = []

        def review(current, preview, start):
            rounds.append(
                tuple(preview(acknowledged) for acknowledged in acknowledgments)
            )
            return {"result": {"journal_operation_id": "original"}}

        monkeypatch.setattr(product, "_run_reviewed_replacement", review)
        children = []
        monkeypatch.setattr(
            product,
            "_child",
            lambda root, installed, route, *, role="retargeted", source: (
                children.append((route, role))
            ),
        )
        product._transfer(source)
        assert children == [("read", "default"), ("read", "retargeted")]
        assert len(rounds) == 1 and not reverse_previews
        assert not (root / "direction.json").exists()
        handoff = json.loads((root / "transfer-forward.json").read_text())
        assert handoff["journal_operation_id"] == "original"
        assert handoff["source"] == str(source.resolve())
        assert handoff["result"]["archive_sha256"] == product._digest(source)
        assert handoff["unselected"] == {
            str(item.path): product._digest(item.path) for item in stores
        }
        for name in ("tldw_chatbook.app", "tldw_chatbook.config"):
            monkeypatch.delitem(sys.modules, name, raising=False)
        product._rollback(source)
        assert json.loads((root / "direction.json").read_text())["rollback"] is True
        assert not {"tldw_chatbook.app", "tldw_chatbook.config"}.intersection(
            sys.modules
        )
    else:
        with pytest.raises(PreviewReached):
            product._transfer(source)
    if stop_mode == "review-retries":
        assert children == [
            ("read", "default"),
            ("read", "retargeted"),
            ("read-rollback", "default"),
            ("read-rollback", "retargeted"),
        ]
        assert (
            len(discoveries) == 7 and len(previews) == 6 and len(reverse_previews) == 3
        )


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


@pytest.mark.skipif(os.name == "nt", reason="POSIX Secret Service session fixture")
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


@pytest.mark.skipif(os.name == "nt", reason="POSIX Secret Service socket fixture")
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
        "TLDW_NATIVE_MAC_KEYCHAIN": str(tmp_path / "native.keychain-db"),
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


def test_native_mac_child_selects_private_keychain_with_exact_home(
    tmp_path, monkeypatch
):
    from Tests.ProductionApp import test_native_credential_recovery as product

    chain = tmp_path / "native.keychain-db"
    chain.write_bytes(b"synthetic-keychain-placeholder")
    if os.name == "nt":
        lstat = Path.lstat

        def mac_lstat(path):
            info = lstat(path)
            mode = stat.S_IFMT(info.st_mode) | (
                0o700 if stat.S_ISDIR(info.st_mode) else 0o600
            )
            return os.stat_result((mode, *info[1:]))

        monkeypatch.setattr(Path, "lstat", mac_lstat)
    root = tmp_path / "child"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(product.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        product.os, "getuid", lambda: chain.stat().st_uid, raising=False
    )
    for name, value in {
        "GITHUB_ACTIONS": "true",
        "RUNNER_ENVIRONMENT": "github-hosted",
        "RUNNER_OS": "macOS",
        "TLDW_NATIVE_MAC_KEYCHAIN": str(chain),
    }.items():
        monkeypatch.setenv(name, value)
    calls = []

    def run(arguments, **options):
        calls.append((arguments, options))
        preferences = Path(options["env"]["HOME"]) / "Library/Preferences"
        assert stat.S_IMODE(preferences.lstat().st_mode) == 0o700
        assert options["stdout"].name == str(root / "mac-keychain-preferences.log")
        assert options["stderr"] == product.subprocess.STDOUT
        assert options["timeout"] == 30 and options["check"] is True
        options["stdout"].write("positive-native-secret")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(product.subprocess, "run", run)
    environment = product._environment(root, tmp_path / "installed", role="retargeted")
    assert [arguments for arguments, _ in calls] == [
        ["/usr/bin/security", command, "-d", "user", "-s", str(chain)]
        for command in ("list-keychains", "default-keychain")
    ]
    assert all(options["env"] == environment for _, options in calls)
    assert environment["HOME"] == str(root / "home")
    assert environment["TLDW_CONFIG_PATH"] == str(root / "home/retargeted/config.toml")
    assert (
        stat.S_IMODE((root / "mac-keychain-preferences.log").lstat().st_mode) == 0o600
    )


@pytest.mark.parametrize(
    "host",
    (
        {},
        {
            "GITHUB_ACTIONS": "true",
            "RUNNER_ENVIRONMENT": "self-hosted",
            "RUNNER_OS": "macOS",
        },
        {
            "GITHUB_ACTIONS": "true",
            "RUNNER_ENVIRONMENT": "github-hosted",
            "RUNNER_OS": "Windows",
        },
    ),
    ids=("personal", "self-hosted", "wrong-os"),
)
def test_native_mac_child_refuses_unsafe_host_before_mutation(
    tmp_path, monkeypatch, host
):
    from Tests.ProductionApp import test_native_credential_recovery as product

    chain = tmp_path / "native.keychain-db"
    chain.write_bytes(b"synthetic-keychain-placeholder")
    monkeypatch.setattr(product.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(os, "environ", {**host, "TLDW_NATIVE_MAC_KEYCHAIN": str(chain)})
    monkeypatch.setattr(
        product.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("unsafe host invoked security"),
    )
    root = tmp_path / "child"
    with pytest.raises(RuntimeError, match="disposable_runner"):
        product._environment(root, tmp_path / "installed", role="default")
    assert not root.exists()


@pytest.mark.parametrize(
    "invalid", ("relative", "missing", "directory", "alias", "foreign-owner")
)
def test_native_mac_child_refuses_unowned_or_indirect_keychain(
    tmp_path, monkeypatch, invalid
):
    from Tests.ProductionApp import test_native_credential_recovery as product

    chain = tmp_path / "native.keychain-db"
    if invalid == "relative":
        chain = Path("native.keychain-db")
    elif invalid == "directory":
        chain.mkdir(mode=0o700)
    elif invalid != "missing":
        chain.write_bytes(b"synthetic-keychain-placeholder")
    if invalid == "alias":
        resolve = Path.resolve
        monkeypatch.setattr(
            Path,
            "resolve",
            lambda path, *args, **kwargs: (
                tmp_path / "actual.keychain-db"
                if path == chain
                else resolve(path, *args, **kwargs)
            ),
        )
    monkeypatch.setattr(product.platform, "system", lambda: "Darwin")
    current_uid = getattr(os, "getuid", lambda: 0)()
    monkeypatch.setattr(
        product.os,
        "getuid",
        lambda: current_uid + (invalid == "foreign-owner"),
        raising=False,
    )
    for name, value in {
        "GITHUB_ACTIONS": "true",
        "RUNNER_ENVIRONMENT": "github-hosted",
        "RUNNER_OS": "macOS",
        "TLDW_NATIVE_MAC_KEYCHAIN": str(chain),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        product.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("invalid keychain invoked security"),
    )
    root = tmp_path / "child"
    with pytest.raises((RuntimeError, OSError)):
        product._environment(root, tmp_path / "installed", role="default")
    assert not root.exists()


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


@pytest.mark.parametrize(
    "selection,expected_timeout",
    [
        ("native-credentials-source", 80 * 60),
        ("native-credentials-destination", 180 * 60),
    ],
)
def test_native_lane_deadline_refuses_publication_after_partial_progress(
    tmp_path, monkeypatch, selection, expected_timeout
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    transfer = tmp_path / "transfer"
    outbound, _, receipt = _source_archive(transfer)
    (outbound / "source-linux.json").write_text(json.dumps(receipt))
    monkeypatch.setenv("TLDW_CREDENTIAL_TRANSFER_ROOT", str(transfer))
    monkeypatch.setattr(runner, "validate_native_credential_environment", lambda: None)
    monkeypatch.setattr(
        runner, "_copy_tracked_source", lambda *_: (workspace, "a" * 64)
    )
    monkeypatch.setattr(runner, "_run_git", lambda *_: "b" * 40)
    monkeypatch.setattr(runner, "_installed_receipts", lambda *_: [receipt])
    deadlines = []

    def timed_out(**arguments):
        deadlines.append(arguments["timeout_seconds"])
        # Even a completed test receipt cannot override the outer deadline.
        return {
            "pytest_returncode": 124,
            "junit": {"collected": 1, "skipped": [], "failed": [], "parse_error": None},
        }

    def publish(*_):
        raise AssertionError("timed_out_native_phase_must_not_publish")

    monkeypatch.setattr(runner, "_run_pytest_phase", timed_out)
    monkeypatch.setattr(runner, "_publish_native_credential_artifacts", publish)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    assert runner.run(workspace, evidence, product_selection=selection) == 1
    assert deadlines == [expected_timeout]
    summary = json.loads((evidence / "artifacts" / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["pytest_returncode"] == 124
    assert summary["installed_package_receipts"] == 1
    assert summary["published_directions"] == 0
    assert {path.name for path in (evidence / "artifacts").iterdir()} == {
        "summary.json",
        "artifact-sha256.json",
    }


@pytest.mark.parametrize(
    "native_credentials,timeout_expired", [(True, False), (False, False), (True, True)]
)
def test_native_pytest_child_keeps_environment_and_publishes_no_raw_logs(
    tmp_path, monkeypatch, native_credentials, timeout_expired
):
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
    subprocess_run = runner.subprocess.run
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs["timeout"]))
        if timeout_expired:
            kwargs["stdout"].write("positive-native-secret\n")
            raise runner.subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess_run(command, **kwargs)

    monkeypatch.setattr(runner.subprocess, "run", run)
    result = runner._run_pytest_phase(
        workspace=workspace,
        private_root=private,
        artifacts=artifacts,
        environment=environment,
        phase="product",
        tests=(str(test),),
        noconftest=True,
        timeout_seconds=30,
        native_credentials=native_credentials,
    )
    assert len(calls) == 1
    command, timeout = calls[0]
    assert f"--timeout={30 if native_credentials else 2400}" in command
    assert timeout == 30
    if timeout_expired:
        assert result["pytest_returncode"] == 124
        assert result["junit"]["collected"] == 0
        assert result["junit"]["parse_error"] == "pytest did not produce JUnit XML"
        assert "positive-native-secret" in (private / "pytest-output.log").read_text()
    else:
        assert result["pytest_returncode"] == 0
        assert result["junit"]["collected"] == 1
        assert result["junit"]["parse_error"] is None
    if native_credentials:
        assert not list(artifacts.iterdir())
    else:
        assert {item.name for item in artifacts.iterdir()} == {
            "pytest-output.log",
            "pytest.xml",
        }


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
