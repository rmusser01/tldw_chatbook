"""Startup-independent commands composing the installed recovery service."""

from __future__ import annotations

import argparse
import getpass
import json
import tomllib
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path


def _secret(prompt: str) -> str:
    """Refuse getpass's echoed fallback when no private terminal is available."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        try:
            return getpass.getpass(prompt)
        except getpass.GetPassWarning:
            raise ValueError("private_password_prompt_unavailable") from None


def startup_preflight() -> tuple[str | None, str | None]:
    """Read the selected input without defaults or normal runtime composition."""
    from . import bootstrap
    from .storage_admission import _read_recovery_file

    selector = bootstrap.effective_config_path()
    allowed, reason = bootstrap.startup_permission(
        selector, bootstrap.default_bootstrap_root()
    )
    if not allowed:
        return reason, None
    try:
        data = _read_recovery_file("config", selector, max_bytes=16 * 1024**2)
    except FileNotFoundError:
        return None, None
    except (OSError, ValueError, RuntimeError):
        return "configuration_unavailable", None
    document = {}
    try:
        document = tomllib.loads(data.decode("utf-8"))
        encryption = document.get("encryption", {})
        if not isinstance(encryption, dict):
            return "configuration_invalid", None
        if not encryption.get("enabled", False):
            return None, None
        from tldw_chatbook.Utils.config_encryption import ConfigEncryption

        verifier = encryption.get("password_verifier")
        if not isinstance(verifier, str) or not verifier:
            return "configuration_unlock_unavailable", None
        password = _secret("Configuration password: ")
        engine = ConfigEncryption()
        if not password or not engine.verify_password(password, verifier):
            return "configuration_unlock_failed", None
        engine.decrypt_config_strict(document, password)
        return None, password
    except (
        OSError,
        ValueError,
        RuntimeError,
        ImportError,
        EOFError,
        KeyboardInterrupt,
    ):
        reason = (
            "configuration_unlock_failed"
            if document.get("encryption")
            else "configuration_invalid"
        )
        return reason, None


def recovery_app(reason: str, *, restart_request=None):
    """Build only a minimal host for the same service and recovery view."""
    import asyncio

    from textual import on, work
    from textual.app import App
    from textual.widgets import Button, Static

    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    from .recovery_service import RecoveryService, default_control_root

    async def settle(function, *args):
        task = asyncio.create_task(asyncio.to_thread(function, *args))
        cancellation = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
        result = task.result()
        if cancellation is not None:
            raise cancellation
        return result

    class RecoveryApp(App):
        def __init__(self):
            super().__init__()
            self.recovery_service = RecoveryService(default_control_root())

        def compose(self):
            yield Static(
                "Recovery mode — review replacement" if reason == "replacement_requested"
                else "Recovery required: " + reason,
                markup=False,
            )
            yield Button("Open Backup & Restore", id="minimal-recovery-open")
            yield Button("Exit", id="minimal-recovery-exit")

        def on_mount(self):
            self.action_backup_restore()

        @on(Button.Pressed, "#minimal-recovery-open")
        def action_backup_restore(self):
            self.push_screen(
                BackupRestoreScreen(
                    self.recovery_service, include_known_profiles=True,
                    restart_request=restart_request,
                )
            )

        @on(Button.Pressed, "#minimal-recovery-exit")
        def leave(self):
            self.exit()

        @work(group="recovery-profile-launch")
        async def open_recovery_profile(self, profile_id):
            current = self.recovery_service.current()
            if current is not None and current["state"] == "running":
                self.notify(
                    "Another recovery operation is running.", severity="warning"
                )
                return
            with self.suspend():
                operation = self.recovery_service.start_open_profile(profile_id)
                await settle(self.recovery_service.wait, operation)

        async def on_unmount(self):
            await settle(self.recovery_service.close)

    return RecoveryApp()


def minimal_recovery(reason: str, *, restart_request=None) -> int:
    """Run a recovery-only UI when ordinary startup cannot safely proceed."""
    recovery_app(reason, restart_request=restart_request).run()
    return 0


def _path(value: str) -> Path:
    return Path(value).expanduser().absolute()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tldw-chatbook recovery", allow_abbrev=False)
    parser.add_argument("--control-root", type=_path)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", allow_abbrev=False)
    inspect.add_argument("archive", type=_path)
    inspect.add_argument("--ask-password", action="store_true")
    extract = commands.add_parser("extract", allow_abbrev=False)
    extract.add_argument("archive", type=_path)
    extract.add_argument("--group", action="append", required=True, metavar="GROUP_ID")
    extract.add_argument("--destination", type=_path, required=True)
    extract.add_argument("--ask-password", action="store_true")
    restore = commands.add_parser("restore", allow_abbrev=False)
    restore.add_argument("archive", type=_path)
    mode = restore.add_mutually_exclusive_group(required=True)
    mode.add_argument("--isolated", dest="mode", action="store_const", const="isolated")
    mode.add_argument("--replace", dest="mode", action="store_const", const="replace")
    restore.add_argument(
        "--destination", action="append", default=[], metavar="ID=PATH"
    )
    restore.add_argument(
        "--profile-name", action="append", default=[], metavar="ID=NAME"
    )
    restore.add_argument("--target-config", type=_path)
    restore.add_argument("--ask-password", action="store_true")
    restore.add_argument(
        "--acknowledge-credential-issue", action="append", default=[], metavar="CODE"
    )
    restore.add_argument(
        "--safety-scope", action="append", default=[], metavar="LOGICAL_ID"
    )
    recover = commands.add_parser("recover", allow_abbrev=False)
    recover.add_argument("operation")
    action = recover.add_mutually_exclusive_group()
    action.add_argument("--finish", dest="action", action="store_const", const="finish")
    action.add_argument(
        "--rollback", dest="action", action="store_const", const="rollback"
    )
    action.add_argument("--abort", dest="action", action="store_const", const="abort")
    recover.add_argument("--ask-password", action="store_true")
    profiles = commands.add_parser("profiles", allow_abbrev=False)
    profiles.add_argument("--open", dest="profile")
    copies = commands.add_parser("copies", allow_abbrev=False)
    copies.add_argument("--inspect", dest="operation")
    return parser


def _plain(value):
    """Escape terminal controls and render only the service's presentation values."""
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _show(value) -> None:
    print(json.dumps(_plain(value), ensure_ascii=True, indent=2))


def _password(prompt: str, *, confirm: bool = False) -> bytes:
    first = _secret(prompt)
    if not first or (confirm and first != _secret("Repeat password: ")):
        raise ValueError("password_confirmation_failed")
    return first.encode("utf-8")


def _pairs(values: Sequence[str], *, paths: bool = False) -> dict:
    result = {}
    for value in values:
        key, separator, selected = value.partition("=")
        if not separator or not key or not selected or key in result:
            raise ValueError("invalid_local_mapping")
        result[key] = _path(selected) if paths else selected
    return result


def _wait(service, operation, *, show=True):
    while True:
        try:
            status = service.wait(operation)
            break
        except KeyboardInterrupt:
            service.cancel(operation)
    if show:
        _show(status)
    return status


def _inspect(service, archive: Path, *, ask_password: bool):
    password = _password("Archive password: ") if ask_password else None
    operation = service.start_inspection(archive, password=password)
    password = None
    status = _wait(service, operation)
    if status["state"] != "succeeded":
        return None
    _show(service.summary(operation))
    return operation


def _restore(service, args) -> int:
    inspection = _inspect(service, args.archive, ask_password=args.ask_password)
    if inspection is None:
        return 1
    target = None
    if args.mode == "replace":
        if args.target_config is None:
            raise ValueError("explicit_target_config_required")
        target = service.preview_backup((args.target_config,), options={})
    elif args.target_config is not None:
        raise ValueError("isolated_target_config_forbidden")
    plan = service.preview_restore(
        inspection,
        mode=args.mode,
        destinations=_pairs(args.destination, paths=True),
        profile_names=_pairs(args.profile_name),
        target=target,
        acknowledged_credential_issues=tuple(args.acknowledge_credential_issue),
        safety_scope=tuple(args.safety_scope),
    )
    _show(
        {
            "mode": plan.mode,
            "restore": plan.restore,
            "retire": plan.retire,
            "preserve": plan.preserve,
            "issues": plan.issues,
            "acknowledged_credential_issues": plan.acknowledged_credential_issues,
            "safety_scope": plan.safety_scope,
        }
    )
    print("Owner review and missing assets remain separate setup requirements.")
    if args.mode == "replace":
        available, reason = service.replacement_capability(plan)
        if not available:
            print("Replacement unavailable: " + reason)
            return 1
    if input("Type restore to apply this reviewed plan: ").strip() != "restore":
        return 0
    password = (
        _password("New encrypted rollback password: ", confirm=True)
        if args.mode == "replace"
        else None
    )
    operation = service.start_restore(inspection, plan, rollback_password=password)
    password = None
    return 0 if _wait(service, operation)["state"] == "succeeded" else 1


def _extract(service, args) -> int:
    inspection = _inspect(service, args.archive, ask_password=args.ask_password)
    if inspection is None:
        return 1
    preview = service.start_extraction_preview(
        inspection, group_ids=tuple(args.group), destination=args.destination
    )
    status = _wait(service, preview, show=False)
    if status["state"] != "succeeded":
        _show(status)
        return 1
    plan = status["result"]["plan"]
    _show(
        {
            "destination": plan.destination,
            "groups": plan.group_ids,
            "payload_bytes": plan.payload_bytes,
            "unselected_dependencies": plan.unselected_dependencies,
        }
    )
    print(
        "Extracted files are inert bytes for manual recovery. This does not restore or open a profile."
    )
    if input("Type extract to copy these reviewed groups: ").strip() != "extract":
        return 0
    operation = service.start_extraction(inspection, plan)
    return 0 if _wait(service, operation)["state"] == "succeeded" else 1


def recovery_main(argv: Sequence[str] | None = None) -> int:
    """Run explicit recovery commands before normal configuration or services."""
    args = _parser().parse_args(argv)
    from .recovery_service import RecoveryService, default_control_root

    service = RecoveryService(args.control_root or default_control_root())
    try:
        if args.command == "inspect":
            return (
                0
                if _inspect(service, args.archive, ask_password=args.ask_password)
                else 1
            )
        if args.command == "restore":
            return _restore(service, args)
        if args.command == "extract":
            return _extract(service, args)
        if args.command == "profiles":
            if args.profile is None:
                _show(service.profiles())
                return 0
            status = _wait(service, service.start_open_profile(args.profile))
            return (
                0
                if status["state"] == "succeeded"
                and status["result"].get("exit_code") == 0
                else 1
            )
        if args.command == "copies":
            if args.operation is None:
                _show(service.recovery_copies())
                return 0
            password = _password("Recovery copy password: ")
            operation = service.start_copy_inspection(args.operation, password=password)
            password = None
            status = _wait(service, operation)
            if status["state"] == "succeeded":
                _show(service.summary(operation))
                return 0
            return 1
        status = service.status(args.operation)
        _show(status)
        if args.action is None:
            return 0
        if args.action not in status["actions"]:
            raise ValueError("recovery_action_unavailable")
        if (
            input("Type " + args.action + " to continue this operation: ").strip()
            != args.action
        ):
            return 0
        password = (
            _password("Rollback archive password: ")
            if args.ask_password and args.action != "abort"
            else None
        )
        operation = service.start_recovery(
            args.operation, action=args.action, rollback_password=password
        )
        password = None
        return 0 if _wait(service, operation)["state"] == "succeeded" else 1
    except (OSError, ValueError, RuntimeError, EOFError) as error:
        print("Recovery refused: " + service.issue_code(error))
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        service.close()
