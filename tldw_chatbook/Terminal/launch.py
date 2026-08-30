"""Pure launch-policy boundaries for persistent terminal sessions."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import ntpath
import os
from pathlib import Path
import posixpath
import shutil
import tempfile
import unicodedata

from tldw_chatbook.Utils.path_validation import (
    validate_existing_absolute_directory,
)


_POSIX_ACCOUNT_KEYS = ("HOME", "USER", "LOGNAME", "SHELL")
_POSIX_LOCALE_KEYS = (
    "LANG",
    "LC_ALL",
    "LC_ADDRESS",
    "LC_COLLATE",
    "LC_CTYPE",
    "LC_IDENTIFICATION",
    "LC_MEASUREMENT",
    "LC_MESSAGES",
    "LC_MONETARY",
    "LC_NAME",
    "LC_NUMERIC",
    "LC_PAPER",
    "LC_TELEPHONE",
    "LC_TIME",
)
_WINDOWS_ACCOUNT_KEYS = ("USERPROFILE", "HOMEDRIVE", "HOMEPATH", "USERNAME")
_WINDOWS_REQUIRED_SYSTEM_KEYS = (
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
)
_WINDOWS_OPTIONAL_SYSTEM_KEYS = ("PROGRAMFILES(X86)", "PROGRAMW6432")


@dataclass(frozen=True, slots=True)
class ShellChoice:
    """One discovered, code-owned shell picker choice.

    Attributes:
        key: Stable picker identity.
        label: User-visible shell label.
        family: Launch-policy family.
        executable: Validated absolute executable path.
        argv: Complete code-owned interactive launch arguments.
    """

    key: str
    label: str
    family: str
    executable: Path
    argv: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResolvedLaunch:
    """Immutable launch values ready for backend admission.

    Attributes:
        name: Validated display name.
        shell: Discovered code-owned shell choice.
        start_directory: Revalidated absolute existing directory.
        environment: Immutable terminal-specific environment pairs.
    """

    name: str
    shell: ShellChoice
    start_directory: Path
    environment: tuple[tuple[str, str], ...]


def normalize_session_name(
    name: str,
    *,
    existing_names: Iterable[str] = (),
) -> str:
    """Normalize and validate one terminal session display name.

    Args:
        name: Candidate user-visible name.
        existing_names: Names held by live terminal records.

    Returns:
        Trimmed NFC-normalized name.

    Raises:
        TypeError: If the candidate is not text.
        ValueError: If length, control, markup, or uniqueness checks fail.
    """
    if not isinstance(name, str):
        raise TypeError("terminal session name must be text")
    normalized = unicodedata.normalize("NFC", name.strip())
    if not 1 <= len(normalized) <= 64:
        raise ValueError("terminal session name must contain 1 to 64 characters")
    if any(
        unicodedata.category(character) in {"Cc", "Cf", "Cs"}
        for character in normalized
    ):
        raise ValueError("terminal session name must not contain controls")
    if "[" in normalized or "]" in normalized:
        raise ValueError("terminal session name must not contain markup")
    key = session_name_key(normalized)
    if any(session_name_key(existing) == key for existing in existing_names):
        raise ValueError("terminal session name must be unique")
    return normalized


def session_name_key(name: str) -> str:
    """Return the normalized Unicode casefold uniqueness key.

    Args:
        name: Previously validated or existing display name.

    Returns:
        NFC-normalized, trimmed, casefolded key.
    """
    return unicodedata.normalize("NFC", name.strip()).casefold()


def discover_shell_choices(
    *,
    platform_name: str | None = None,
    account_shell: str | Path | None = None,
    executable_lookup: Callable[[str], str | None] = shutil.which,
    executable_is_file: Callable[[Path], bool] | None = None,
) -> tuple[ShellChoice, ...]:
    """Discover the fixed shell picker and resolve its Default choice.

    Args:
        platform_name: ``"posix"`` or ``"nt"``; defaults to ``os.name``.
        account_shell: POSIX account shell path. Ignored on Windows.
        executable_lookup: Injected executable lookup.
        executable_is_file: Injected executable validation predicate.

    Returns:
        Default first, followed by discovered explicit family choices.

    Raises:
        ValueError: If the platform identity is unsupported.
        FileNotFoundError: If no safe Default shell can be resolved.
    """
    platform_name = os.name if platform_name is None else platform_name
    if platform_name not in {"posix", "nt"}:
        raise ValueError(f"unsupported terminal platform: {platform_name!r}")
    is_file = executable_is_file or _is_executable_file
    if platform_name == "nt":
        return _discover_windows_shells(executable_lookup, is_file)
    if account_shell is None:
        account_shell = _read_posix_account_shell()
    return _discover_posix_shells(account_shell, executable_lookup, is_file)


def resolve_shell_choice(
    selector: str,
    choices: Iterable[ShellChoice],
) -> ShellChoice:
    """Resolve an allowlisted picker key from discovered choices.

    Args:
        selector: Stable picker key.
        choices: Result from :func:`discover_shell_choices`.

    Returns:
        Matching code-owned shell choice.

    Raises:
        ValueError: If the selector was not discovered.
    """
    for choice in choices:
        if choice.key == selector:
            return choice
    raise ValueError(f"unsupported terminal shell choice: {selector!r}")


def resolve_start_directory(
    selected_local_root: Path | None,
    *,
    requested_directory: Path | None = None,
    account_home: Path,
) -> Path:
    """Select and revalidate the terminal's initial directory.

    Args:
        selected_local_root: Late-bound selected local Console root, if any.
        requested_directory: Optional explicit user-selected directory.
        account_home: Real current-account home fallback.

    Returns:
        Normalized absolute existing directory.

    Raises:
        ValueError: If the final candidate is not an absolute existing directory.
    """
    candidate = requested_directory or selected_local_root or account_home
    try:
        return validate_existing_absolute_directory(candidate)
    except ValueError:
        raise ValueError(
            "terminal start directory must be an absolute existing directory"
        ) from None


def build_terminal_environment(
    *,
    platform_name: str | None = None,
    ambient: Mapping[str, str] | None = None,
    account_reader: Callable[[], Mapping[str, str]] | None = None,
    system_reader: Callable[[], Mapping[str, str]] | None = None,
    path_is_directory: Callable[[str], bool] | None = None,
    fallback_path: str | None = None,
) -> dict[str, str]:
    """Build the dedicated scrubbed environment for an interactive terminal.

    Args:
        platform_name: ``"posix"`` or ``"nt"``; defaults to ``os.name``.
        ambient: Process environment used only for PATH and locale categories.
        account_reader: Trusted current-account value reader.
        system_reader: Trusted platform-system and temporary-value reader.
        path_is_directory: Injected PATH component validator.
        fallback_path: Platform fallback PATH when ambient PATH is absent.

    Returns:
        New environment mapping containing only approved values.

    Raises:
        ValueError: If the platform or a required approved value is unavailable.
    """
    platform_name = os.name if platform_name is None else platform_name
    if platform_name not in {"posix", "nt"}:
        raise ValueError(f"unsupported terminal platform: {platform_name!r}")
    source = dict(os.environ if ambient is None else ambient)
    if platform_name == "nt":
        source = _uppercase_mapping(source)
    account = _uppercase_mapping(
        (account_reader or (lambda: _read_account_values(platform_name)))()
    )
    system = _uppercase_mapping(
        (system_reader or (lambda: _read_system_values(platform_name)))()
    )
    path_value = source.get("PATH")
    fallback = os.defpath if fallback_path is None else fallback_path
    environment = {
        "PATH": _validated_path(
            path_value,
            platform_name=platform_name,
            is_directory=path_is_directory or _path_is_directory,
            fallback=fallback,
        )
    }
    if platform_name == "nt":
        _copy_required(environment, account, _WINDOWS_ACCOUNT_KEYS, "account")
        _copy_required(
            environment,
            system,
            _WINDOWS_REQUIRED_SYSTEM_KEYS,
            "platform",
        )
        _copy_optional(environment, system, _WINDOWS_OPTIONAL_SYSTEM_KEYS)
        locale_keys = ("LANG", "LC_ALL")
    else:
        _copy_required(environment, account, _POSIX_ACCOUNT_KEYS, "account")
        _copy_optional(environment, system, ("TMPDIR",))
        locale_keys = _POSIX_LOCALE_KEYS
    for key in locale_keys:
        value = _safe_scalar(source.get(key), maximum=256)
        if value is not None:
            environment[key] = value
    environment["TERM"] = "linux"
    return environment


def _discover_posix_shells(
    account_shell: str | Path | None,
    lookup: Callable[[str], str | None],
    is_file: Callable[[Path], bool],
) -> tuple[ShellChoice, ...]:
    discovered = {
        name: _validated_executable(
            lookup(name), platform_name="posix", is_file=is_file
        )
        for name in ("bash", "zsh", "sh")
    }
    account = _validated_executable(
        account_shell,
        platform_name="posix",
        is_file=is_file,
    )
    default_path = account or discovered["bash"] or discovered["sh"]
    if default_path is None:
        raise FileNotFoundError("persistent terminal shell is unavailable")
    default_family = (
        _posix_family(default_path)
        if account is not None
        else ("bash" if discovered["bash"] == default_path else "sh")
    )
    choices = [
        _shell_choice(
            key="default",
            label="Default",
            family=default_family,
            executable=default_path,
        )
    ]
    for family, label in (("bash", "Bash"), ("zsh", "Zsh")):
        executable = discovered[family]
        if executable is not None:
            choices.append(
                _shell_choice(
                    key=family,
                    label=label,
                    family=family,
                    executable=executable,
                )
            )
    return tuple(choices)


def _discover_windows_shells(
    lookup: Callable[[str], str | None],
    is_file: Callable[[Path], bool],
) -> tuple[ShellChoice, ...]:
    discovered = {
        key: _validated_executable(name, platform_name="nt", is_file=is_file)
        for key, name in (
            ("pwsh", lookup("pwsh.exe")),
            ("powershell", lookup("powershell.exe")),
            ("cmd", lookup("cmd.exe")),
        )
    }
    default_key = next(
        (key for key in ("pwsh", "powershell", "cmd") if discovered[key]),
        None,
    )
    if default_key is None:
        raise FileNotFoundError("persistent terminal shell is unavailable")
    family = "powershell" if default_key != "cmd" else "cmd"
    default_path = discovered[default_key]
    assert default_path is not None
    choices = [
        _shell_choice(
            key="default",
            label="Default",
            family=family,
            executable=default_path,
        )
    ]
    for key, label in (
        ("pwsh", "PowerShell 7"),
        ("powershell", "Windows PowerShell"),
        ("cmd", "Command Prompt"),
    ):
        executable = discovered[key]
        if executable is not None:
            choices.append(
                _shell_choice(
                    key=key,
                    label=label,
                    family="powershell" if key != "cmd" else "cmd",
                    executable=executable,
                )
            )
    return tuple(choices)


def _shell_choice(
    *,
    key: str,
    label: str,
    family: str,
    executable: Path,
) -> ShellChoice:
    text = str(executable)
    if family == "bash":
        argv = (text, "--login", "-i")
    elif family in {"zsh", "sh"}:
        argv = (text, "-l", "-i")
    elif family == "account":
        argv = (f"-{executable.name}",)
    elif family == "powershell":
        argv = (text, "-NoLogo")
    elif family == "cmd":
        argv = (text, "/Q")
    else:
        raise ValueError(f"unsupported terminal shell family: {family!r}")
    return ShellChoice(
        key=key,
        label=label,
        family=family,
        executable=executable,
        argv=argv,
    )


def _posix_family(executable: Path) -> str:
    name = executable.name.casefold()
    return name if name in {"bash", "zsh", "sh"} else "account"


def _validated_executable(
    candidate: str | Path | None,
    *,
    platform_name: str,
    is_file: Callable[[Path], bool],
) -> Path | None:
    if candidate is None:
        return None
    text = str(candidate)
    path_module = ntpath if platform_name == "nt" else posixpath
    if "\x00" in text or not path_module.isabs(text):
        return None
    path = Path(text)
    try:
        return path if is_file(path) else None
    except OSError:
        return None


def _is_executable_file(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _read_account_values(platform_name: str) -> Mapping[str, str]:
    if platform_name == "nt":
        values = _read_windows_environment_block()
        return {key: values[key] for key in _WINDOWS_ACCOUNT_KEYS if key in values}
    try:
        import pwd

        account = pwd.getpwuid(os.getuid())
    except (ImportError, KeyError, OSError) as exc:
        raise ValueError("POSIX account values are unavailable") from exc
    return {
        "HOME": account.pw_dir,
        "USER": account.pw_name,
        "LOGNAME": account.pw_name,
        "SHELL": account.pw_shell,
    }


def _read_posix_account_shell() -> str | None:
    try:
        import pwd

        return pwd.getpwuid(os.getuid()).pw_shell
    except (ImportError, KeyError, OSError):
        return None


def _read_system_values(platform_name: str) -> Mapping[str, str]:
    if platform_name == "nt":
        values = _read_windows_environment_block()
        keys = _WINDOWS_REQUIRED_SYSTEM_KEYS + _WINDOWS_OPTIONAL_SYSTEM_KEYS
        return {key: values[key] for key in keys if key in values}
    return {"TMPDIR": tempfile.gettempdir()}


def _read_windows_environment_block() -> dict[str, str]:
    """Read the current account's environment through native Windows APIs."""
    if os.name != "nt":
        raise ValueError("Windows platform values are unavailable")

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    userenv = ctypes.WinDLL("userenv", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    userenv.CreateEnvironmentBlock.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        wintypes.HANDLE,
        wintypes.BOOL,
    ]
    userenv.CreateEnvironmentBlock.restype = wintypes.BOOL
    userenv.DestroyEnvironmentBlock.argtypes = [ctypes.c_void_p]
    userenv.DestroyEnvironmentBlock.restype = wintypes.BOOL

    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(
        kernel32.GetCurrentProcess(),
        0x000A,  # TOKEN_QUERY | TOKEN_DUPLICATE
        ctypes.byref(token),
    ):
        raise ValueError("Windows account values are unavailable")
    block = ctypes.c_void_p()
    try:
        if not userenv.CreateEnvironmentBlock(ctypes.byref(block), token, False):
            raise ValueError("Windows platform values are unavailable")
        return _parse_windows_environment_block(block)
    finally:
        if block.value:
            userenv.DestroyEnvironmentBlock(block)
        kernel32.CloseHandle(token)


def _parse_windows_environment_block(block: object) -> dict[str, str]:
    """Parse one trusted, double-NUL-terminated Windows environment block."""
    import ctypes

    characters = ctypes.cast(block, ctypes.POINTER(ctypes.c_wchar))
    result: dict[str, str] = {}
    offset = 0
    maximum_characters = 131_072
    while offset < maximum_characters:
        start = offset
        while offset < maximum_characters and characters[offset] != "\x00":
            offset += 1
        if offset == maximum_characters:
            break
        if offset == start:
            return result
        entry = "".join(characters[index] for index in range(start, offset))
        offset += 1
        if entry.startswith("="):
            continue
        key, separator, value = entry.partition("=")
        if separator and key:
            result[key.upper()] = value
    raise ValueError("Windows platform environment is malformed")


def _uppercase_mapping(source: Mapping[str, str]) -> dict[str, str]:
    return {str(key).upper(): value for key, value in source.items()}


def _copy_required(
    target: dict[str, str],
    source: Mapping[str, str],
    keys: Iterable[str],
    source_name: str,
) -> None:
    for key in keys:
        value = _safe_scalar(source.get(key))
        if value is None:
            raise ValueError(f"terminal {source_name} value {key} is unavailable")
        target[key] = value


def _copy_optional(
    target: dict[str, str],
    source: Mapping[str, str],
    keys: Iterable[str],
) -> None:
    for key in keys:
        value = _safe_scalar(source.get(key))
        if value is not None:
            target[key] = value


def _safe_scalar(value: object, *, maximum: int = 4096) -> str | None:
    if not isinstance(value, str) or not value or len(value) > maximum:
        return None
    if any(unicodedata.category(character) == "Cc" for character in value):
        return None
    return value


def _validated_path(
    value: str | None,
    *,
    platform_name: str,
    is_directory: Callable[[str], bool],
    fallback: str,
) -> str:
    separator = ";" if platform_name == "nt" else os.pathsep
    path_module = ntpath if platform_name == "nt" else posixpath
    accepted: list[str] = []
    seen: set[str] = set()
    for raw in (value or fallback).split(separator):
        if not raw or "\x00" in raw or not path_module.isabs(raw):
            continue
        try:
            available = is_directory(raw)
        except OSError:
            available = False
        key = raw.casefold() if platform_name == "nt" else raw
        if available and key not in seen:
            accepted.append(raw)
            seen.add(key)
    if not accepted:
        raise ValueError("terminal PATH has no existing absolute directory")
    return separator.join(accepted)


def _path_is_directory(path: str) -> bool:
    return Path(path).is_dir()
