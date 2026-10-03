"""Startup-independent commands composing the installed recovery service."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import tomllib
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path

#: TASK-34100.4: preflight outcomes that are user choices, not recovery
#: reasons. ``startup_unlock`` maps them; they never reach ``minimal_recovery``.
UNLOCK_QUIT_REASON = "configuration_unlock_quit"
UNLOCK_RESET_REASON = "configuration_unlock_reset"
#: The typed password fails the verifier (or there is none) but reads every
#: saved key: adopt it and repair the verifier (review round 2, R2-F5).
UNLOCK_REKEY_REASON = "configuration_unlock_rekey"
#: Recovery reasons with their own plain sentence (review round 1): a served
#: (browser) child cannot prompt, and "no private terminal" is no longer the
#: copy for every unexpected unlock failure.
UNLOCK_SERVED_REASON = "configuration_unlock_served"
UNLOCK_NO_TERMINAL_REASON = "configuration_unlock_no_terminal"

#: One name for one password everywhere: "master password" (protect-summary-02
#: found "Configuration password", "Unlock Configuration" and "master password"
#: for the same secret).
UNLOCK_INTRO = (
    "Your saved API keys are encrypted. Enter the master password you set "
    "during setup."
)
UNLOCK_PROMPT = "Master password (leave empty if you forgot it): "
UNLOCK_MISMATCH = "That password didn't match. Try again."
#: The password matches the verifier, but a saved key was encrypted under a
#: different one (the stranded state an old second enable left behind). The
#: user typed the RIGHT password; only a reset gets them past it.
UNLOCK_STRANDED = (
    "That password is right, but some saved keys were encrypted with a "
    "different password and can't be read with it. If you set an earlier "
    "master password, enter that one; otherwise leave the prompt empty to "
    "reset the saved keys."
)
UNLOCK_VERIFIER_MISSING = (
    "config.toml says its API keys are encrypted, but the check for the "
    "master password is missing."
)
#: Encrypted keys exist, so the password that encrypted them still reads
#: them: ask for it instead of offering only the reset (R2-F5).
UNLOCK_VERIFIER_MISSING_ASK = (
    UNLOCK_VERIFIER_MISSING + " Enter your master password to unlock the "
    "keys and repair the check, or leave the prompt empty to reset them."
)
UNLOCK_REKEYED = (
    "That password reads every saved key, so it is your master password "
    "again. Use it each time chatbook starts."
)
UNLOCK_REKEY_FAILED = (
    "Repairing the master-password check failed; config.toml was left as it "
    "was."
)
UNLOCK_RESET_EXPLAINED = (
    "Resetting removes the encrypted API keys from config.toml and turns "
    "encryption off. Chats, notes and documents are not affected; you "
    "re-enter your API keys afterwards in Settings."
)
UNLOCK_GIVE_UP = "Forgot your master password? " + UNLOCK_RESET_EXPLAINED
UNLOCK_CHOICE_PROMPT = "[R]eset saved keys or [Q]uit: "
UNLOCK_RESET_DONE = (
    "Saved keys were reset and encryption is off. Re-enter your API keys in "
    "Settings > Providers & Models."
)
UNLOCK_RESET_FAILED = (
    "Resetting the saved keys failed; config.toml was left as it was."
)
#: Holds an outcome sentence on screen; the TUI would cover it within a
#: second (review round 2, F-R2-4).
UNLOCK_CONTINUE_PROMPT = "Press Enter to start chatbook. "

#: The unlock's outcome for ``tldw_chatbook.config``, read once as config
#: first loads (`take_startup_unlock`). Config's import-time load used to run
#: before the password was installed and warned "no password is set" right
#: after the RIGHT password (review round 2, R2-F6). ``(password,)`` installs
#: the master password before that load; ``(None,)`` marks a reset or re-key
#: that finishes right after the import, so the locked load is expected.
_STARTUP_UNLOCK: tuple[str | None] | None = None


def take_startup_unlock() -> tuple[str | None, bool]:
    """Hand ``tldw_chatbook.config`` the startup unlock's outcome, once.

    Returns:
        ``(password, pending)``: the master password typed at startup, or
        None; and whether a reset or re-key finishes the unlock after
        config's import (config then logs its locked load quietly).
    """
    global _STARTUP_UNLOCK
    slot, _STARTUP_UNLOCK = _STARTUP_UNLOCK, None
    if slot is None:
        return None, False
    return slot[0], slot[0] is None

#: Opening a profile hands the terminal to it; a browser session has none.
PROFILE_OPEN_NEEDS_TERMINAL = (
    "Opening a profile needs chatbook running in a terminal; it can't be "
    "done from a browser session."
)

#: Plain sentences for the recovery host instead of a raw reason code.
_UNLOCK_RECOVERY_COPY = {
    "configuration_unlock_failed": (
        "Checking the master password failed unexpectedly, so the encrypted "
        "API keys were not unlocked. Nothing was changed. Quit, then relaunch "
        "chatbook to try again."
    ),
    UNLOCK_NO_TERMINAL_REASON: (
        "The master password could not be asked for: no private terminal was "
        "available. Quit, then relaunch chatbook from a terminal window to "
        "enter it."
    ),
    # Reached only when nothing is encrypted (encrypted keys ask for the
    # password instead, which needs a private terminal of its own).
    "configuration_unlock_unavailable": (
        "config.toml says encryption is on, but the check for its master "
        "password is missing. Quit, then relaunch chatbook from a terminal "
        "window: it offers a reset that turns encryption off. Chats, notes "
        "and documents are not affected."
    ),
    UNLOCK_SERVED_REASON: (
        "This browser session can't ask for the master password, so the "
        "encrypted API keys can't be unlocked here. Run chatbook in a "
        "terminal instead, or turn encryption off there in Settings > "
        "Privacy & Security before serving it."
    ),
}


def _say(message: str) -> None:
    """Write one pre-TUI line where getpass prompts (never stdout data)."""
    print(message, file=sys.stderr, flush=True)


def _open_tty():
    """The controlling terminal, opened the way getpass opens it, or None.

    None when there is no terminal (a service, CI, Windows).
    """
    try:
        return open("/dev/tty", "r+", encoding="utf-8", errors="replace")
    except OSError:
        return None


def _choice(prompt: str) -> str:
    """Ask a non-secret, echoed question before the TUI starts.

    Reads standard input when it is a terminal, else the controlling terminal
    -- where getpass already read the password. With stdin redirected
    (``tldw-cli < /dev/null``, some IDE run configurations) a person is still
    at the keyboard (review round 2, R2-F7). Without any terminal it falls
    back to standard input.

    Raises:
        EOFError: Nothing more can be read.
    """
    terminal = None if _stdin_is_terminal() else _open_tty()
    if terminal is None:
        sys.stderr.write(prompt)
        sys.stderr.flush()
        line = sys.stdin.readline()
    else:
        with terminal:
            terminal.write(prompt)
            terminal.flush()
            line = terminal.readline()
    if not line:
        raise EOFError
    return line


def _stdin_is_terminal() -> bool:
    """Whether standard input is a terminal."""
    try:
        return bool(sys.stdin) and sys.stdin.isatty()
    except (AttributeError, OSError, ValueError):
        return False


def _can_answer() -> bool:
    """Whether a person can answer an echoed question (stdin or the terminal)."""
    if _stdin_is_terminal():
        return True
    terminal = _open_tty()
    if terminal is None:
        return False
    terminal.close()
    return True


def _served_child() -> bool:
    """Whether this process is a textual-serve (browser) session child.

    Its stdin is the web driver's pipe, and getpass would open the SERVER
    operator's controlling terminal (textual-serve spawns without setsid), so
    the browser would hang while the prompt shows on the server's terminal.
    """
    return os.environ.get("CHATBOOK_SERVED_CHILD") == "1" or "web_driver" in (
        os.environ.get("TEXTUAL_DRIVER") or ""
    )


class _PrivatePromptUnavailable(ValueError):
    """No private terminal is available to ask for a password."""


def _secret(prompt: str) -> str:
    """Refuse getpass's echoed fallback when no private terminal is available."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        try:
            return getpass.getpass(prompt)
        except getpass.GetPassWarning:
            raise _PrivatePromptUnavailable(
                "private_password_prompt_unavailable"
            ) from None


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

        if _served_child():
            return UNLOCK_SERVED_REASON, None
        verifier = encryption.get("password_verifier")
        if not isinstance(verifier, str) or not verifier:
            if _has_saved_ciphertext(document):
                # The password that encrypted the keys still reads them:
                # ask for it, and repair the check (review round 2, R2-F5).
                return _unlock_interactively(
                    ConfigEncryption(),
                    document,
                    None,
                    intro=UNLOCK_VERIFIER_MISSING_ASK,
                )
            # Nothing is encrypted, so no password matters and the reset
            # loses nothing: offer it where someone can answer (review round
            # 1; the old copy sent the user to hand-edit config.toml).
            if not _can_answer():
                return "configuration_unlock_unavailable", None
            _say(UNLOCK_VERIFIER_MISSING)
            return _give_up_choice(UNLOCK_RESET_EXPLAINED), None
        return _unlock_interactively(ConfigEncryption(), document, verifier)
    except KeyboardInterrupt:
        # Ctrl+C anywhere before the TUI quits; it is not a recovery problem.
        _say("")
        return UNLOCK_QUIT_REASON, None
    except _PrivatePromptUnavailable:
        return UNLOCK_NO_TERMINAL_REASON, None
    except (
        OSError,
        ValueError,
        RuntimeError,
        ImportError,
        EOFError,
    ):
        reason = (
            "configuration_unlock_failed"
            if document.get("encryption")
            else "configuration_invalid"
        )
        return reason, None


def _unlock_interactively(
    engine, document, verifier: str | None, *, intro: str = UNLOCK_INTRO
) -> tuple[str | None, str | None]:
    """Ask until the master password strict-decrypts, or the user gives up.

    TASK-34100.4 (protect-summary-02): a wrong password re-prompts in place
    instead of dropping the user into recovery mode. An empty entry gives up
    and offers a reset or quit. There is no attempt cap: this guards a local
    file, and each try already costs an scrypt derivation.

    Args:
        engine: The ConfigEncryption engine.
        document: The parsed config.toml.
        verifier: The saved password verifier, or None when it is missing.
        intro: The sentence shown before the first prompt.

    Returns:
        ``(None, password)`` once every saved value decrypts;
        ``(UNLOCK_REKEY_REASON, password)`` when the password fails the
        verifier (or there is none) but reads every saved key; or
        ``(UNLOCK_RESET_REASON | UNLOCK_QUIT_REASON, None)``.

    Raises:
        ValueError: No private terminal is available to ask for the password.
    """
    _say(intro)
    while True:
        try:
            password = _secret(UNLOCK_PROMPT)
        except (EOFError, KeyboardInterrupt):
            _say("")
            return UNLOCK_QUIT_REASON, None
        if not password:
            return _give_up_choice(), None
        try:
            # Ctrl+C after Enter lands here, while scrypt runs: still a quit.
            outcome = _check_password(engine, document, verifier, password)
        except (EOFError, KeyboardInterrupt):
            _say("")
            return UNLOCK_QUIT_REASON, None
        if outcome is None or outcome == UNLOCK_REKEY_REASON:
            return outcome, password
        password = None
        _say(outcome)


def _check_password(engine, document, verifier, password: str) -> str | None:
    """None to unlock, UNLOCK_REKEY_REASON to adopt, else what to tell the user.

    A password that fails the verifier is still tried against the saved keys
    (review round 2, R2-F5): in the stranded state the user's EARLIER
    password reads them all, and refusing it left only a reset that deleted
    keys it could read. AES-GCM authenticates, so a wrong password cannot
    "decrypt" a value.
    """
    verified = verifier is not None and engine.verify_password(password, verifier)
    if verified and _decrypts(engine, document, password):
        return None
    if _reads_every_saved_key(engine, document, password):
        return UNLOCK_REKEY_REASON
    return UNLOCK_STRANDED if verified else UNLOCK_MISMATCH


def _saved_values(document) -> dict:
    """The document without its ``[encryption]`` table (the saved keys)."""
    return {key: value for key, value in document.items() if key != "encryption"}


def _has_saved_ciphertext(value) -> bool:
    """Whether any saved value outside ``[encryption]`` is ``enc:`` ciphertext."""
    from tldw_chatbook.Utils.config_encryption import ConfigEncryption

    pending = [_saved_values(value)]
    while pending:
        current = pending.pop()
        for item in current.values():
            if isinstance(item, dict):
                pending.append(item)
            elif isinstance(item, str) and item.startswith(
                ConfigEncryption.ENCRYPTION_PREFIX
            ):
                return True
    return False


def _reads_every_saved_key(engine, document, password: str) -> bool:
    """Every saved ``enc:`` value decrypts with ``password`` -- and there is at
    least one, or any guess would pass."""
    if not _has_saved_ciphertext(document):
        return False
    return _decrypts(engine, _saved_values(document), password)


def _decrypts(engine, document, password: str) -> bool:
    """Strict decrypt: a password that passes the verifier but cannot read
    every saved key (the stranded state a second enable used to produce) is
    not an unlock -- the app would otherwise start with ciphertext as keys.
    The failure is reported to the user, so it is not logged as an error."""
    try:
        engine.decrypt_config_strict(document, password, log_failure=False)
    except ValueError:
        return False
    return True


def _give_up_choice(intro: str = UNLOCK_GIVE_UP) -> str:
    """Offer the forgotten-password exits until one is chosen."""
    _say(intro)
    while True:
        try:
            answer = _choice(UNLOCK_CHOICE_PROMPT).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return UNLOCK_QUIT_REASON
        if answer in {"r", "reset"}:
            return UNLOCK_RESET_REASON
        if answer in {"q", "quit"}:
            return UNLOCK_QUIT_REASON


def startup_unlock() -> int | None:
    """The one startup unlock, shared by ``tldw-cli`` and ``python -m tldw_chatbook.app``.

    Runs the isolated pre-TUI preflight (strict decrypt), then admits startup
    and installs the master password -- or performs the forgotten-password
    reset -- through the normal config owner.

    Returns:
        None when ordinary startup may continue, else the exit status the
        caller returns (a quit, a failed reset, or the recovery host's).
    """
    global _STARTUP_UNLOCK
    reason, password = startup_preflight()
    if reason == UNLOCK_QUIT_REASON:
        return 0
    if reason is not None and reason not in (UNLOCK_RESET_REASON, UNLOCK_REKEY_REASON):
        return minimal_recovery(reason)
    from .storage_admission import admit_startup

    admit_startup()
    if reason is None and password is None:
        return None  # encryption is off: nothing to unlock
    _STARTUP_UNLOCK = (password if reason is None else None,)
    try:
        if reason == UNLOCK_RESET_REASON:
            from tldw_chatbook.config import reset_encrypted_config_values

            if not reset_encrypted_config_values():
                _say(UNLOCK_RESET_FAILED)
                return 1
            _say(UNLOCK_RESET_DONE)
            return _paused()
        if reason == UNLOCK_REKEY_REASON:
            from tldw_chatbook.config import rekey_encryption_verifier

            # Re-encrypts under this password with a new verifier and
            # installs it for the session, through the normal config writer.
            if not rekey_encryption_verifier(password):
                _say(UNLOCK_REKEY_FAILED)
                return 1
            _say(UNLOCK_REKEYED)
            return _paused()
        from tldw_chatbook.config import (
            get_encryption_password,
            set_encryption_password,
        )

        # Normally config already took the password as it loaded; this
        # covers a config module that was imported earlier.
        if get_encryption_password() != password:
            set_encryption_password(password)
        return None
    finally:
        _STARTUP_UNLOCK = None
        password = None


def _paused() -> int | None:
    """Wait for Enter so an outcome sentence is read before the TUI starts.

    Returns:
        None to start the app, or 0 when the user quits here instead.
    """
    try:
        _choice(UNLOCK_CONTINUE_PROMPT)
    except EOFError:
        pass
    except KeyboardInterrupt:
        _say("")
        return 0
    return None


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

        def _get_default_css(self):
            # Recovery cannot import the normal app/config or its boot bundle.
            # Register only this host's screen at its original default tier.
            return [
                (
                    (__file__, "BackupRestoreScreen.BUNDLED_CSS"),
                    BackupRestoreScreen.BUNDLED_CSS,
                    0,
                    "BackupRestoreScreen",
                ),
                *super()._get_default_css(),
            ]

        def compose(self):
            yield Static(
                "Recovery mode — review replacement" if reason == "replacement_requested"
                else _UNLOCK_RECOVERY_COPY.get(reason, "Recovery required: " + reason),
                id="minimal-recovery-reason",
                markup=False,
            )
            yield Button("Open Backup & Restore", id="minimal-recovery-open")
            yield Button("Exit", id="minimal-recovery-exit")

        def on_mount(self):
            # TASK-34100.4: an unlock problem is not a restore problem. Say
            # what happened and let the user choose; Backup & Restore stays
            # one button away.
            if reason not in _UNLOCK_RECOVERY_COPY:
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
            # Mirrors TldwCli.open_recovery_profile (app_lifecycle.py), which
            # this host cannot import. TASK-34100.4 review round 2: a served
            # (browser) session reaches this host, and the web driver cannot
            # suspend -- an uncaught SuspendNotSupported in this default
            # exit_on_error worker ended the whole session.
            from textual.app import SuspendNotSupported

            service = self.recovery_service
            current = service.current()
            if current is not None and current["state"] == "running":
                self.notify(
                    "Another recovery operation is running.", severity="warning"
                )
                return
            failure = None
            try:
                with self.suspend():
                    try:
                        operation = service.start_open_profile(profile_id)
                        await settle(service.wait, operation)
                    except (OSError, RuntimeError, ValueError) as error:
                        failure = error
            except SuspendNotSupported:
                self.notify(PROFILE_OPEN_NEEDS_TERMINAL, severity="error")
                return
            except (OSError, RuntimeError, ValueError) as error:
                failure = error
            if failure is not None:
                self.notify(
                    "Profile opening failed: " + service.issue_code(failure),
                    severity="error",
                )

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
    from .data_groups import BACKUP_GROUPS

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
    restore.add_argument(
        "--group",
        action="append",
        choices=tuple(group.group_id for group in BACKUP_GROUPS),
        help="Restore one data group; repeat for more. Defaults to Everything in the archive.",
    )
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
    summary = service.summary(inspection)
    data_groups = None if args.group is None else tuple(args.group)
    target = None
    target_choices = {}
    if args.mode == "replace":
        if args.target_config is None:
            raise ValueError("explicit_target_config_required")
        if data_groups is not None or summary.get("group_scope") is not None:
            profiles = summary["profile_ids"]
            if len(profiles) != 1:
                raise ValueError("target_unverified")
            target_choices["target_configs"] = {profiles[0]: args.target_config}
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
        data_groups=data_groups,
        **target_choices,
    )
    effective_groups = plan.effective_groups
    if plan.requested_groups is None:
        print("Requested groups: Everything (all available groups in this archive).")
        if not effective_groups:
            effective_groups = tuple(
                group["group_id"] for group in summary.get("data_groups", ())
            )
    _show(
        {
            "mode": plan.mode,
            "requested_groups": plan.requested_groups,
            "effective_groups": effective_groups,
            "required_groups": plan.required_groups,
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
