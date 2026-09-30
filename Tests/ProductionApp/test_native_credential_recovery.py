"""Installed-product qualification using disposable native credential stores.

Only encrypted synthetic archives and projected receipts leave the private run.
Every native-writing child repeats the runner's isolation preflight.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import sqlite3
import stat
import subprocess  # nosec B404 - fixed installed-product test children
import sys
import tomllib
import zipfile
from contextlib import closing
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery.native_package import (
    native_package as native_package,  # noqa: PLC0414 - installed product fixture
)

PASSWORD = b"task33422-disposable-synthetic-transfer"
CONFIG_PASSWORD = "task33422-disposable-config-unlock"  # nosec B105 - synthetic fixtures only
PURPOSES = ("api_key", "bearer_token", "access_token", "refresh_token")
GROUPS = ("settings", "tools", "conversations", "evaluations")
pytestmark = pytest.mark.skipif(
    not os.environ.get("TLDW_CREDENTIAL_TRANSFER_ROOT"),
    reason="explicit_native_credential_lane_required",
)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provider(selector):
    from tldw_chatbook.runtime_policy.server_context import RuntimeServerContextProvider
    from tldw_chatbook.runtime_policy.server_credentials import (
        KeyringServerCredentialStore,
    )

    return RuntimeServerContextProvider(
        runtime_context=SimpleNamespace(),
        target_store=SimpleNamespace(),
        credential_store=KeyringServerCredentialStore(),
        app_config={},
        credential_profile_id=None if selector == _selectors()[0] else str(selector),
    )


def _selectors():
    return (
        Path.home() / ".config/tldw_cli/config.toml",
        Path.home() / "retargeted/config.toml",
    )


def _value(system, role, purpose):
    return f"disposable-{system}-{role}-{purpose}-task33422"


def _environment(root, installed, *, role):
    environment = os.environ.copy()
    chain = None
    if platform.system() == "Darwin" and environment.get("TLDW_NATIVE_MAC_KEYCHAIN"):
        if not (
            environment.get("GITHUB_ACTIONS") == "true"
            and environment.get("RUNNER_ENVIRONMENT") == "github-hosted"
            and environment.get("RUNNER_OS") == "macOS"
        ):
            raise RuntimeError("native_credential_disposable_runner_required")
        chain = Path(environment["TLDW_NATIVE_MAC_KEYCHAIN"])
        if not chain.is_absolute() or chain.resolve() != chain:
            raise RuntimeError("native_credential_private_keychain_required")
        info = chain.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise RuntimeError("native_credential_private_keychain_required")
    home = root / "home"
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    environment.update(
        HOME=str(home),
        USERPROFILE=str(home),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        PYTHONNOUSERSITE="1",
        PYTHONPATH=os.pathsep.join(
            (str(installed), str(Path(__file__).resolve().parents[2]))
        ),
        TLDW_TEST_INSTALLED_PACKAGE=str(installed),
    )
    # The native SecretService's XDG roots belong to its private daemon. Keep
    # those roots unchanged; application storage is selected by HOME/config.
    environment.pop("TLDW_CONFIG_PATH", None)
    if role == "retargeted":
        environment["TLDW_CONFIG_PATH"] = str(home / "retargeted/config.toml")
    if chain is not None:
        for directory in (home, home / "Library", home / "Library/Preferences"):
            directory.mkdir(mode=0o700, exist_ok=True)
            info = directory.lstat()
            if (
                directory.resolve() != directory
                or not stat.S_ISDIR(info.st_mode)
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700
            ):
                raise RuntimeError("native_credential_keychain_preferences_not_private")
        log = root / "mac-keychain-preferences.log"
        with log.open("a") as output:
            log.chmod(0o600)
            for command in ("list-keychains", "default-keychain"):
                subprocess.run(  # nosec B603 - fixed native tool and validated disposable chain
                    ["/usr/bin/security", command, "-d", "user", "-s", str(chain)],
                    env=environment,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=30,
                    check=True,
                )
    return environment


def _child(root, installed, route, *, role="retargeted", source=None):
    environment = _environment(root, installed, role=role)
    arguments = [sys.executable, str(Path(__file__).resolve()), route, role]
    if source is not None:
        arguments.append(str(source))
    # Output can contain native backend exception text or credential values.
    # It remains in private storage and never forms an assertion message.
    with (root / f"{route}-{role}.log").open("w") as output:
        result = subprocess.run(  # nosec B603 - fixed interpreter/owned fixture
            arguments,
            cwd=root,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=900,
            check=False,
        )
    assert result.returncode == 0, f"native_child_failed:{route}:{result.returncode}"


def _receipt(installed, **extra):
    wheel = json.loads((installed.parent / "native-package.json").read_text())
    return {
        "schema": 1,
        "status": "passed",
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "backend": os.environ["PYTHON_KEYRING_BACKEND"],
        "revision": os.environ["TLDW_NATIVE_CREDENTIAL_REVISION"],
        "wheel_sha256": wheel["sha256"],
        **extra,
    }


def _write(path, data):
    path.write_text(json.dumps(data, sort_keys=True))
    path.chmod(0o600)


def _material(archive):
    from tldw_chatbook.Backup_Recovery.archive_reader import verify_sealed

    doc = verify_sealed(archive)
    member = next(
        row.payload for row in doc.files if row.owner_id == "recovery.credentials"
    )
    with zipfile.ZipFile(archive.path) as container:
        records = json.loads(container.read(member))["records"]
    assert records and all(row["status"] == "captured" for row in records)
    assert {row["kind"] for row in records} == {
        "server",
        "generation",
        "citation",
        "encrypted_config",
    }
    return records


def _manual(issues):
    assert issues and all(
        issue.startswith("credential_manual_recovery_required:") for issue in issues
    ), "native_capture_has_unavailable_credentials"
    return tuple(issues)


def _run_reviewed_replacement(service, preview, start):
    """Review retained input, then abort and review exact target-copy notices."""
    retention = ("credential_isolated_retention_required",)
    acknowledged = ()
    for attempt in range(3):
        plan = preview(acknowledged)
        operation = start(plan)
        state = service.wait(operation, timeout=300)
        if state["state"] == "succeeded":
            return state
        assert attempt < 2 and state["review_issues"], "native_replacement_failed"
        if state["review_issues"] == retention:
            assert not acknowledged and state["state"] == "failed"
            assert not state["result"].get("journal_operation_id")
            acknowledged = retention
        else:
            assert acknowledged == retention and state["state"] == "recovery_required"
            manual = _manual(state["review_issues"])
            pending = state["result"]["journal_operation_id"]
            assert service.status(pending)["actions"] == ("abort",)
            abort = service.start_recovery(pending, action="abort")
            aborted = service.wait(abort, timeout=120)
            assert aborted["state"] == "succeeded" and aborted["result"]["aborted"]
            acknowledged = (*retention, *manual)
    raise AssertionError("native_replacement_failed")


def _observe_workers(service):
    failure_root = os.environ.get("TLDW_NATIVE_FAILURE_ROOT")
    if not failure_root:
        return service
    from Tests.Backup_Recovery.run_platform_product import _record_native_failure
    from tldw_chatbook.Backup_Recovery import archive_reader
    from tldw_chatbook.Backup_Recovery.sqlite_validation import (
        _validate_candidate,
        validated_schema_version,
    )

    root = Path(failure_root)
    start = service._start

    def observe_sqlite(frame, event, argument):
        if frame.f_code is archive_reader.acquire.__code__:
            if event == "exception" and isinstance(argument[1], OSError):
                _record_native_failure(root, argument[1])
            return observe_sqlite
        if frame.f_code is not _validate_candidate.__code__:
            return None
        if event == "exception":
            error = argument[1]
            try:
                _record_native_failure(
                    root,
                    error,
                    metadata={
                        "sqlite_owner": getattr(
                            frame.f_locals.get("owner"), "owner_id", None
                        ),
                        "sqlite_issue": error.args[0] if error.args else None,
                    },
                )
            except Exception:  # noqa: BLE001 - tracing must preserve validation flow.
                return observe_sqlite
        return observe_sqlite

    def start_observed(kind, function):
        def observed(operation, cancel):
            previous_trace = sys.gettrace()
            if previous_trace is None:
                sys.settrace(observe_sqlite)
            try:
                return function(operation, cancel)
            except Exception as error:
                metadata = {}
                try:
                    trace = error.__traceback__
                    while trace is not None:
                        if trace.tb_frame.f_code is validated_schema_version.__code__:
                            metadata = {
                                "sqlite_owner": getattr(
                                    trace.tb_frame.f_locals.get("owner"),
                                    "owner_id",
                                    None,
                                ),
                                "sqlite_issue": error.args[0] if error.args else None,
                            }
                            break
                        trace = trace.tb_next
                except Exception:  # noqa: BLE001 - preserve the original worker failure.
                    metadata = {}
                _record_native_failure(root, error, metadata=metadata)
                raise
            finally:
                if previous_trace is None:
                    sys.settrace(previous_trace)

        return start(kind, observed)

    service._start = start_observed
    return service


async def _setup(role):
    from tldw_chatbook.Utils.config_encryption import ConfigEncryption

    selector = _selectors()[role == "retargeted"]
    selector.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    encrypted = ConfigEncryption().encrypt_value(
        _value(platform.system(), role, "encrypted_config"),
        CONFIG_PASSWORD,
    )
    selector.write_text(
        f'[general]\nusers_name="native_{role}"\n'
        "[first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n"
        '[tldw_api]\nbase_url=""\n'
        f"[API]\nopenai_api_key={json.dumps(encrypted)}\n"
    )
    selector.chmod(0o600)
    import keyring

    from tldw_chatbook import config
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Backup_Recovery.credential_policies import GENERATION_KEYRINGS
    from tldw_chatbook.Chat.citation_trace_identity import (
        KeyringCitationFingerprintKeyProvider,
    )
    from tldw_chatbook.Chat.citation_trace_repository import (
        load_local_citation_identity_context,
    )
    from tldw_chatbook.MCP.unified_control_models import ConfiguredServerTarget

    config.set_encryption_password(CONFIG_PASSWORD)
    app = TldwCli()
    try:
        provider = app.server_context_provider
        expected_profile = None if role == "default" else str(selector)
        assert provider._credential_profile_id == expected_profile
        targets = []
        for purpose in PURPOSES:
            origin = f"https://native-{role}-{purpose}.invalid"
            provider.store_scoped_credential(
                origin, purpose, _value(platform.system(), role, purpose)
            )
            targets.append(
                ConfiguredServerTarget(
                    server_id=origin,
                    label="Disposable native fixture",
                    base_url=origin,
                    auth_reference="keyring:" + purpose,
                )
            )
            if role == "retargeted":
                _provider(_selectors()[0]).store_scoped_credential(
                    origin,
                    purpose,
                    _value(platform.system(), "foreign", purpose),
                )
        app.unified_mcp_target_store.save_targets(targets)
        for _, service, names in GENERATION_KEYRINGS:
            for name in names:
                keyring.set_password(
                    service, name, _value(platform.system(), "generation", name)
                )
        identity = load_local_citation_identity_context(app.chachanotes_db)
        assert identity is not None
        citation_key = KeyringCitationFingerprintKeyProvider().provision_key(
            identity.fingerprint_key_id
        )
        _write(
            Path.cwd() / f"seed-{role}.json",
            {
                "citation_id": identity.fingerprint_key_id,
                "citation_sha256": hashlib.sha256(citation_key).hexdigest(),
            },
        )
        app.chachanotes_db.add_note(
            "Native qualification", _value(platform.system(), role, "note")
        )
        assert app.prompts_db is not None
        app.prompts_db.add_prompt(
            "Native unselected prompt",
            "disposable",
            "Preserve unselected data",
            user_prompt=_value(platform.system(), role, "unselected_prompt"),
        )
    finally:
        await app._shutdown_app_owned_lifecycles()
        await app.tts_service.close()
        await app.tts_service.wait_closed()


async def _capture():
    from tldw_chatbook import config
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Backup_Recovery import archive_reader, credentials
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
    from tldw_chatbook.Backup_Recovery.recovery_service import (
        RecoveryService,
        default_control_root,
    )
    from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app

    config.set_encryption_password(CONFIG_PASSWORD)
    app = TldwCli()
    monitoring = asyncio.create_task(monitor_app(app))
    service = _observe_workers(RecoveryService(default_control_root()))
    transfer = Path(os.environ["TLDW_CREDENTIAL_TRANSFER_ROOT"])
    outbound = transfer / "outbound"
    outbound.mkdir(mode=0o700, exist_ok=True)
    destination = outbound / f"source-{platform.system().lower()}.age"
    # Public transfer basename is copied only after the app verifies its suffix.
    native_archive = Path.home() / "source.tldw-backup.zip.age"
    options = {"credential_mode": "include", "encrypted": True, "data_groups": GROUPS}
    try:
        for attempt in range(2):
            details = await asyncio.to_thread(
                service.preview_backup_details,
                _selectors(),
                options=options,
                destination=native_archive,
            )
            if not details["inventory"].complete and (
                root := os.environ.get("TLDW_NATIVE_FAILURE_ROOT")
            ):
                from Tests.Backup_Recovery.run_platform_product import (
                    _record_native_failure,
                )
                from tldw_chatbook.Backup_Recovery.inventory import BLOCKING
                from tldw_chatbook.Backup_Recovery.sqlite_validation import (
                    _restrict_connection,
                )

                _record_native_failure(
                    Path(root),
                    AssertionError("native_inventory_incomplete"),
                    metadata={
                        "inventory": {
                            "issues": details["inventory"].issues,
                            "blocking": [
                                {"owner": item.owner, "status": item.status}
                                for item in details["inventory"].items
                                if item.status in BLOCKING
                            ],
                        }
                    },
                )
                try:
                    with closing(sqlite3.connect(":memory:")) as connection:
                        _restrict_connection(connection)
                except Exception as error:  # noqa: BLE001 - preserve the inventory refusal.
                    _record_native_failure(
                        Path(root),
                        error,
                        metadata={
                            "sqlite_issue": error.args[0] if error.args else None
                        },
                    )
                    if error.__cause__ is not None:
                        _record_native_failure(Path(root), error.__cause__)
            assert details["inventory"].complete, "native_inventory_incomplete"
            operation = service.start_backup(
                _selectors(),
                details["inventory"].scope_digest,
                native_archive,
                options=options,
                password=PASSWORD,
            )
            state = await asyncio.to_thread(service.wait, operation, timeout=500)
            if state["state"] == "succeeded":
                break
            if (
                attempt != 0
                or not state["review_issues"]
                or not all(
                    issue.startswith("credential_manual_recovery_required:")
                    for issue in state["review_issues"]
                )
            ):
                issues = state["review_issues"] or state["issues"]
                raise ValueError(
                    issues[0].split(":", 1)[0] if issues else "native_backup_failed"
                )
            options["acknowledged_credential_issues"] = _manual(state["review_issues"])
        assert state["state"] == "succeeded"
        # Captured manual-recovery material is disclosed as partial recovery
        # coverage even though every seeded value must be present/readable.
        assert not state["result"]["complete"]
        archive = archive_reader.acquire(
            native_archive,
            Path.home() / "source-readback",
            ArchiveLimits(),
            PASSWORD,
            Event(),
        )
        records = _material(archive)
        assert sum(row["kind"] == "server" for row in records) == 8
        seeds = [
            json.loads((Path.cwd() / f"seed-{role}.json").read_text())
            for role in ("default", "retargeted")
        ]
        assert {row["username"] for row in records if row["kind"] == "citation"} == {
            seed["citation_id"] for seed in seeds
        }
        from tldw_chatbook.Backup_Recovery.credential_policies import (
            GENERATION_KEYRINGS,
        )

        doc = archive_reader.verify_sealed(archive)
        for row in doc.files:
            if row.owner_id != "config":
                continue
            owned = [record for record in records if record["file"] == row.payload]
            assert {
                (record["service"], record["username"])
                for record in owned
                if record["kind"] == "generation"
            } == {
                (service, name)
                for _, service, names in GENERATION_KEYRINGS
                for name in names
            }
            assert sum(record["kind"] == "encrypted_config" for record in owned) == 1
        for record in records:
            if record["kind"] == "encrypted_config":
                assert record["value"] in {
                    _value(platform.system(), role, "encrypted_config")
                    for role in ("default", "retargeted")
                }
            else:
                assert (
                    credentials._read_scope(record, credentials._credential_store())
                    == record["value"]
                )
        destination.write_bytes(native_archive.read_bytes())
        destination.chmod(0o600)
    finally:
        await asyncio.to_thread(service.close)
        monitoring.cancel()
        await asyncio.gather(monitoring, return_exceptions=True)
        await app._shutdown_app_owned_lifecycles()
        await app.tts_service.close()
        await app.tts_service.wait_closed()


def _fresh_readback(source_system, *, role, rollback=False):
    import keyring

    from tldw_chatbook.Backup_Recovery.credential_policies import GENERATION_KEYRINGS
    from tldw_chatbook.Backup_Recovery.profile_paths import user_data_dir
    from tldw_chatbook.Chat.citation_trace_identity import (
        KeyringCitationFingerprintKeyProvider,
    )
    from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
    from tldw_chatbook.Utils.config_encryption import ConfigEncryption

    selector = _selectors()[role == "retargeted"]
    data = tomllib.loads(selector.read_text())
    seed = json.loads((Path.cwd() / f"seed-{role}.json").read_text())
    assert (
        hashlib.sha256(
            KeyringCitationFingerprintKeyProvider().load_key(seed["citation_id"])
        ).hexdigest()
        == seed["citation_sha256"]
    )
    if rollback:
        assert ConfigEncryption().decrypt_value(
            data["API"]["openai_api_key"], CONFIG_PASSWORD
        ) == _value(platform.system(), role, "encrypted_config")
    else:
        assert "openai_api_key" not in data.get("API", {})
    targets = ConfiguredServerTargetStore(
        user_data_dir(data) / "mcp_server_targets.json"
    ).list_targets()
    assert len(targets) == 4
    provider = _provider(selector)
    for target, purpose in zip(
        sorted(targets, key=lambda row: row.server_id),
        sorted(PURPOSES),
        strict=True,
    ):
        value, _ = provider._resolve_auth_token(
            target.server_id, target, allow_legacy_config=False
        )
        assert value == _value(
            platform.system() if rollback else source_system, role, purpose
        )
        assert provider._get_credential_secret(target.server_id, purpose) == _value(
            platform.system(), role, purpose
        )
        if role == "retargeted":
            assert _provider(_selectors()[0])._get_credential_secret(
                target.server_id, purpose
            ) == _value(platform.system(), "foreign", purpose)
    for _, service, names in GENERATION_KEYRINGS:
        for name in names:
            assert keyring.get_password(service, name) == _value(
                platform.system(), "generation", name
            )


def _transfer(source):
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import archive_reader
    from tldw_chatbook.Backup_Recovery.profile_catalog import ProfileCatalog
    from tldw_chatbook.Backup_Recovery.recovery_service import (
        RecoveryService,
        default_control_root,
    )
    from tldw_chatbook.Backup_Recovery.restore_plan import (
        required_rollback_dependencies,
    )

    config.set_encryption_password(CONFIG_PASSWORD)
    source_system = json.loads(source.with_suffix(".json").read_text())["system"]
    service = _observe_workers(RecoveryService(default_control_root()))
    unselected = {
        item.path: _digest(item.path)
        for item in service.preview_backup(_selectors(), options={}).items
        if item.owner == "db.prompts.primary" and item.status == "included"
    }
    assert len(unselected) == 2
    try:
        inspection = service.start_inspection(source, password=PASSWORD)
        assert service.wait(inspection, timeout=120)["state"] == "succeeded"
        archive = service.inspection(inspection)
        doc = archive_reader.verify_sealed(archive)
        records = _material(archive)
        incoming_manual = tuple(
            "credential_manual_recovery_required:" + row["id"]
            for row in records
            if not row["remappable"]
        )
        with zipfile.ZipFile(archive.path) as container:
            target_configs = {
                row.logical_id.split(":")[1]: _selectors()[
                    tomllib.loads(container.read(row.payload).decode())["general"][
                        "users_name"
                    ]
                    == "native_retargeted"
                ]
                for row in doc.files
                if row.owner_id == "config"
            }
        parent = Path.home() / "isolated"
        parent.mkdir(mode=0o700)
        isolated = service.preview_restore(
            inspection,
            mode="isolated",
            target=None,
            profile_bases={profile: parent / profile for profile in doc.profile_ids},
            external_destinations={},
            profile_names={
                profile: "separate_" + str(index)
                for index, profile in enumerate(doc.profile_ids)
            },
        )
        operation = service.start_restore(inspection, isolated)
        assert service.wait(operation, timeout=180)["state"] == "succeeded", (
            "native_isolated_restore_failed"
        )
        assert len(service.profiles()) == 2
        from tldw_chatbook.Backup_Recovery.isolated_restore import _launch_environment

        for entry in service.profiles():
            selector, _ = ProfileCatalog(service.control_root).resolve(
                entry["profile_id"]
            )
            assert "enc:" not in selector.read_text()
            environment = _launch_environment()
            environment.update(TLDW_TEST_MODE="1", TLDW_DISABLE_CONFIG_WATCH="1")
            with (Path.cwd() / "isolated-open.log").open("a") as output:
                opened = subprocess.run(  # nosec B603 - installed selector, synthetic fixture
                    [
                        sys.executable,
                        "-c",
                        (
                            "import sys,keyring; from pathlib import Path; "
                            "from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile; "
                            "select_profile(sys.argv[1],Path(sys.argv[2])); "
                            "from keyring.backends.null import Keyring; "
                            "assert isinstance(keyring.get_keyring(),Keyring); "
                            "from tldw_chatbook import config; "
                            "assert config.load_cli_config_and_ensure_existence()['general']['users_name'].startswith('separate_')"
                        ),
                        entry["profile_id"],
                        str(service.control_root),
                    ],
                    env=environment,
                    cwd=selector.parent,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=120,
                    check=False,
                )
            assert opened.returncode == 0, "native_isolated_open_failed"
        retained = list(service.control_root.glob("isolated-*/credentials.age"))
        assert len(retained) == 1 and _digest(retained[0]) == _digest(source)
        setup_parent = Path.cwd() / "files-needing-setup"
        setup_parent.mkdir(mode=0o700)

        def preview_replace(acknowledged):
            target = service.preview_backup(_selectors(), options={})
            assert target.complete
            choices = {
                "mode": "replace",
                "target": target,
                "profile_bases": {},
                "external_destinations": {},
                "profile_names": {},
                "target_configs": target_configs,
                "setup_parent": setup_parent,
                "acknowledged_credential_issues": acknowledged,
            }
            plan = service.preview_restore(inspection, **choices)
            safety_scope = required_rollback_dependencies(plan)
            if safety_scope:
                plan = service.preview_restore(
                    inspection, **choices, safety_scope=safety_scope
                )
            return plan

        state = _run_reviewed_replacement(
            service,
            preview_replace,
            lambda plan: service.start_restore(
                inspection, plan, rollback_password=PASSWORD
            ),
        )
        original = state["result"]["journal_operation_id"]
        assert all(_digest(path) == digest for path, digest in unselected.items())
        copy = next(
            row for row in service.recovery_copies() if row.operation_id == original
        )
        assert copy.status == "verified"
        inspect_copy = service.start_copy_inspection(original, password=PASSWORD)
        assert service.wait(inspect_copy, timeout=120)["state"] == "succeeded"
        assert _material(service.inspection(inspect_copy))
        installed = Path(os.environ["TLDW_TEST_INSTALLED_PACKAGE"])
        for role in ("default", "retargeted"):
            _child(Path.cwd(), installed, "read", role=role, source=source)

        result = {
            "source_system": source_system,
            "destination_system": platform.system(),
            "archive_sha256": _digest(source),
            "isolated": True,
            "original_retained": True,
            "replacement": True,
            "rollback": False,
            "captured": len(records),
            "manual_required": len(incoming_manual),
            "unavailable": 0,
        }
        _write(
            Path.cwd() / "transfer-forward.json",
            {
                "source": str(source.resolve()),
                "journal_operation_id": original,
                "unselected": {
                    str(path): digest for path, digest in unselected.items()
                },
                "result": result,
            },
        )
    finally:
        service.close()


def _rollback(source):
    """Review the committed copy in a fresh interpreter without ordinary startup."""
    assert not {"tldw_chatbook.app", "tldw_chatbook.config"}.intersection(sys.modules)
    handoff = json.loads((Path.cwd() / "transfer-forward.json").read_text())
    result = handoff["result"]
    assert handoff["source"] == str(source.resolve())
    assert result["archive_sha256"] == _digest(source)
    assert (
        result["source_system"]
        == json.loads(source.with_suffix(".json").read_text())["system"]
    )
    assert result["destination_system"] == platform.system() and not result["rollback"]
    unselected = {Path(path): digest for path, digest in handoff["unselected"].items()}
    assert len(unselected) == 2
    assert all(_digest(path) == digest for path, digest in unselected.items())
    from tldw_chatbook.Backup_Recovery.recovery_service import (
        RecoveryService,
        default_control_root,
    )

    original = handoff["journal_operation_id"]
    service = _observe_workers(RecoveryService(default_control_root()))
    try:

        def preview_reverse(acknowledged):
            target = service.preview_backup(_selectors(), options={})
            assert target.complete
            return service.preview_rollback(
                original,
                old_password=PASSWORD,
                target=target,
                acknowledged_credential_issues=acknowledged,
            )

        _run_reviewed_replacement(
            service,
            preview_reverse,
            lambda plan: service.start_rollback(
                original, plan, old_password=PASSWORD, new_password=PASSWORD
            ),
        )
        installed = Path(os.environ["TLDW_TEST_INSTALLED_PACKAGE"])
        for role in ("default", "retargeted"):
            _child(Path.cwd(), installed, "read-rollback", role=role, source=source)
        assert all(_digest(path) == digest for path, digest in unselected.items())
        assert result["archive_sha256"] == _digest(source)
        _write(Path.cwd() / "direction.json", {**result, "rollback": True})
    finally:
        service.close()
        assert not {"tldw_chatbook.app", "tldw_chatbook.config"}.intersection(
            sys.modules
        )


async def _transfer_with_app(source):
    from tldw_chatbook import config
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app

    config.set_encryption_password(CONFIG_PASSWORD)
    app = TldwCli()
    monitoring = asyncio.create_task(monitor_app(app))
    try:
        await asyncio.to_thread(_transfer, source)
    finally:
        monitoring.cancel()
        await asyncio.gather(monitoring, return_exceptions=True)
        await app._shutdown_app_owned_lifecycles()
        await app.tts_service.close()
        await app.tts_service.wait_closed()


def _negative_checks(source):
    """Use native reads and refuse collisions/drift before credential writes."""
    import keyring

    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import archive_reader, credentials
    from tldw_chatbook.Backup_Recovery.credential_policies import CITATION_SERVICE
    from tldw_chatbook.Backup_Recovery.crypto import CryptoError
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
    from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem
    from tldw_chatbook.Utils.config_encryption import ConfigEncryption

    provider = _provider(_selectors()[1])
    origin = "https://native-negative-task33422.invalid"
    provider.store_scoped_credential(origin, "api_key", "disposable-negative-original")

    def staged(name):
        root = Path.cwd() / name
        root.mkdir(mode=0o700)
        path = root / "targets.json"
        _write(
            path,
            {
                "targets": [
                    {
                        "server_id": origin,
                        "base_url": origin,
                        "auth_reference": "keyring:api_key",
                    }
                ]
            },
        )
        return root, Inventory(
            (
                StorageItem(
                    "mcp.targets",
                    "profile:negative:mcp.targets",
                    path,
                    "included",
                    (),
                ),
            ),
            True,
            "",
            (),
        )

    root, inventory = staged("negative-exclusion")
    assert not credentials.process_credentials(
        root, inventory, mode="exclude", encrypted=False
    )
    assert not (root / "credential-recovery.json").exists()
    root, inventory = staged("negative-apply")
    assert not credentials.process_credentials(
        root,
        inventory,
        mode="include",
        encrypted=True,
        profile_scopes={"negative": str(_selectors()[1])},
    )
    record = credentials._material(root)[0]
    plan = json.loads(
        credentials.plan_credential_scopes(
            root,
            fresh=True,
            profile_scopes={"targets.json": str(_selectors()[1])},
        )[record["id"]]
    )
    provider.store_scoped_credential(origin, plan["purpose"], "disposable-collision")
    try:
        credentials.check_replacement_credential(record, plan)
    except ValueError as error:
        assert str(error) == "credential_scope_changed"
    else:
        raise AssertionError("native_collision_not_refused")
    provider.delete_scoped_credential(origin, plan["purpose"])
    provider.store_scoped_credential(origin, "api_key", "disposable-external-drift")
    try:
        credentials.check_replacement_credential(record, plan)
    except ValueError as error:
        assert str(error) == "credential_scope_changed"
    else:
        raise AssertionError("native_external_drift_not_refused")
    assert (
        _provider(_selectors()[1])._get_credential_secret(origin, "api_key")
        == "disposable-external-drift"
    )
    material, issues = [], []
    credentials._capture_record(
        {
            "kind": "server",
            "file": "targets.json",
            "server_id": origin,
            "purpose": "refresh_token",
            "profile_id": str(_selectors()[1]),
            "remappable": True,
        },
        material,
        issues,
    )
    assert material[0]["status"] == "missing" and len(issues) == 1
    keyring.set_password(
        CITATION_SERVICE, "native-negative-task33422", "disposable-invalid-base64"
    )
    material, issues = [], []
    credentials._capture_record(
        {
            "kind": "citation",
            "file": "unused.sqlite",
            "service": CITATION_SERVICE,
            "username": "native-negative-task33422",
            "remappable": False,
        },
        material,
        issues,
    )
    assert material[0]["status"] == "unreadable" and len(issues) == 1
    config.clear_encryption_password()
    material, issues = [], []
    credentials._capture_encrypted(
        {
            "API": {
                "api_key": ConfigEncryption().encrypt_value(
                    "disposable-locked-setting",
                    CONFIG_PASSWORD,
                )
            }
        },
        "config.toml",
        material,
        issues,
    )
    assert material[0]["status"] == "locked" and len(issues) == 1
    wrong_password_root = Path.cwd() / "wrong-password"
    try:
        archive_reader.acquire(
            source,
            wrong_password_root,
            ArchiveLimits(),
            b"wrong-disposable-password",
            Event(),
        )
    except CryptoError as error:
        assert error.args == ("transform_failed",)
        assert wrong_password_root.is_dir() and not any(wrong_password_root.iterdir())
    else:
        raise AssertionError("native_wrong_archive_password_accepted")
    _write(Path.cwd() / "negative-results.json", {"negative_checks": 7})


def test_native_credential_source(tmp_path, native_package):
    from Tests.Backup_Recovery.run_platform_product import (
        validate_native_credential_environment,
    )

    validate_native_credential_environment()
    for role in ("default", "retargeted"):
        _child(tmp_path, native_package, "setup", role=role)
    _child(tmp_path, native_package, "capture")
    transfer = Path(os.environ["TLDW_CREDENTIAL_TRANSFER_ROOT"])
    archive = transfer / "outbound" / f"source-{platform.system().lower()}.age"
    _write(
        archive.with_suffix(".json"),
        _receipt(
            native_package,
            archive=archive.name,
            archive_sha256=_digest(archive),
        ),
    )


def test_native_credential_destinations(tmp_path, native_package):
    from Tests.Backup_Recovery.run_platform_product import (
        validate_native_credential_environment,
    )

    validate_native_credential_environment()
    transfer = Path(os.environ["TLDW_CREDENTIAL_TRANSFER_ROOT"])
    sources = sorted(transfer.rglob("source-*.age"))
    assert len(sources) == 3, "three_qualified_sources_required"
    assert {
        json.loads(source.with_suffix(".json").read_text())["system"]
        for source in sources
    } == {"Darwin", "Linux", "Windows"}
    results = []
    for source in sources:
        receipt = json.loads(source.with_suffix(".json").read_text())
        assert receipt["status"] == "passed" and receipt["archive_sha256"] == _digest(
            source
        )
        root = tmp_path / receipt["system"]
        root.mkdir(mode=0o700)
        for role in ("default", "retargeted"):
            _child(root, native_package, "setup", role=role)
        _child(root, native_package, "transfer", source=source)
        _child(root, native_package, "rollback", source=source)
        results.append(json.loads((root / "direction.json").read_text()))
    negative_root = tmp_path / "negative-checks"
    negative_root.mkdir(mode=0o700)
    _child(negative_root, native_package, "setup", role="retargeted")
    _child(negative_root, native_package, "negative", source=sources[0])
    negative_checks = json.loads((negative_root / "negative-results.json").read_text())[
        "negative_checks"
    ]
    (transfer / "outbound").mkdir(mode=0o700, exist_ok=True)
    _write(
        transfer / "outbound" / "destination-results.json",
        _receipt(
            native_package,
            negative_checks=negative_checks,
            results=results,
        ),
    )


def _main():
    """Run one installed native child with bounded private thread observations."""
    from Tests.Backup_Recovery.run_platform_product import (
        validate_native_credential_environment,
    )
    from Tests.Backup_Recovery.thread_diagnostics import (
        observe_threads,
        snapshot_threads,
        stop_observer,
    )
    from Tests.network_guard import blocked_attempts, install

    route, role = sys.argv[1:3]
    if route not in {
        "setup",
        "capture",
        "transfer",
        "rollback",
        "negative",
        "read",
        "read-rollback",
    } or role not in {"default", "retargeted"}:
        raise ValueError("unknown_native_qualification_route")
    stop = lambda: None
    if failure_root := os.environ.get("TLDW_NATIVE_FAILURE_ROOT"):
        stacks = Path(failure_root) / "child-stacks"
        stacks.mkdir(mode=0o700, exist_ok=True)
        path = stacks / f"{route}--{role}--{os.getpid()}.json"
        snapshot_threads(path)
        stop = observe_threads(path, interval=10)
    try:
        validate_native_credential_environment()
        install()
        for name in ("sounddevice", "pyaudio"):
            sys.modules[name] = None
        import tldw_chatbook

        installed = Path(os.environ["TLDW_TEST_INSTALLED_PACKAGE"])
        assert (
            Path(tldw_chatbook.__file__).resolve()
            == installed / "tldw_chatbook/__init__.py"
        )
        if route == "setup":
            asyncio.run(_setup(role))
        elif route == "capture":
            asyncio.run(_capture())
        elif route == "transfer":
            asyncio.run(_transfer_with_app(Path(sys.argv[3])))
        elif route == "rollback":
            _rollback(Path(sys.argv[3]))
        elif route == "negative":
            _negative_checks(Path(sys.argv[3]))
        else:
            source_system = json.loads(
                Path(sys.argv[3]).with_suffix(".json").read_text()
            )["system"]
            _fresh_readback(source_system, role=role, rollback=route == "read-rollback")
        assert not blocked_attempts(), "native_fixture_network_attempt"
    finally:
        stop_observer(stop)


if __name__ == "__main__":
    _main()
