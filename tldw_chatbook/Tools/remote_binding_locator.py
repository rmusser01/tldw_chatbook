"""SSH remote binding locator: parsing, charset validation, argv building.

Security model (see ``Docs/superpowers/specs/2026-09-24-ssh-remote-workspace-bindings-design.md``,
"Binding model & data layer"):

- Locators are ``ssh://[user@]host[:port]/absolute/path`` URIs where the host
  may be an ``~/.ssh/config`` alias. The URI is *parsed* into user / host /
  port / path, and each component is validated against its own charset:
  ``@`` appears in no component ever (it is consumed by parsing), and no
  component may start with ``-`` (option injection — OpenSSH parses options
  appearing after the host, so position alone is not protection).
- Hosts may be bracketed IPv6 literals (``[2001:db8::1]``); brackets are
  stripped during parsing and never reach argv.
- The path must be absolute POSIX, contain no ``..`` component, no ``~``,
  and no whitespace or control characters (these values cross a remote
  shell command line, and the whole raw locator is rejected if it contains
  any).
- argv is always rebuilt from the parsed parts (``-p <port>`` / ``-l <user>``
  / ``--`` / bare host) — never from the raw locator string — so no single
  argv token ever mixes metacharacters.

Canonical form (``locator_string``): ``ssh://[user@]host[:port]/path`` with
the IPv6 host re-bracketed, the path normalized (``.`` collapsed, trailing
slash dropped), and the port omitted when it is ``None`` or the SSH default
``22``. A locator parsed from a canonical string round-trips to an equal
locator (an explicit ``:22`` canonicalizes to ``port=None``).

Fingerprint (``sha256_fingerprint``): sha256 hexdigest of
``<user>@<host>:<port><path>`` where a missing user serializes as the empty
string, a missing port as ``22``, and the path is normalized. Callers
normalize case (e.g. lowercase the ``ssh -G``-resolved host) before hashing;
this helper only hashes the strings it is given.

Canonical resolution (``canonicalize_locator``): ``ssh -G`` prints the
fully resolved *local* configuration for a destination and exits — no
connection is made, nothing is resolved through DNS, and no command runs
remotely. The config-expanded symbolic identity (``HostName`` exactly as
written in the ssh config, ``port``, ``user``) is the ADR-069-style
fingerprint source (via ``canonical_fingerprint``): cosmetic alias edits
do not trip re-consent, while a config change that genuinely moves the
destination does. Resolution runs only at binding add/edit, manual
refresh, and workspace open — never in the send path.

Stdlib only — this module sits below the pinned remote worker's import
closure.
"""

import hashlib
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

__all__ = [
    "CanonicalTarget",
    "RemoteLocator",
    "RemoteLocatorError",
    "build_ssh_argv",
    "canonical_fingerprint",
    "canonicalize_locator",
    "locator_string",
    "parse_remote_locator",
    "sha256_fingerprint",
]

_SCHEME = "ssh://"
_DEFAULT_PORT = 22
_MIN_PORT = 1
_MAX_PORT = 65535

# ``ssh -G`` is a local config dump; 10s is far beyond any legitimate
# runtime and bounds a wedged binary (module-level so tests can tighten
# it for the timeout case).
_SSH_G_TIMEOUT_SECONDS = 10.0

# The only ``ssh -G`` output keys consumed; everything else it prints is
# ignored.
_SSH_G_KEYS = ("hostname", "port", "user")

# Component charsets (spec: "Charset validation, three distinct sets").
# Note ``-`` is allowed inside a component but never as its first character;
# that leading-dash rule is enforced separately in _validate_component.
_USER_RE = re.compile(r"[A-Za-z0-9._-]+")
_HOST_RE = re.compile(r"[A-Za-z0-9._-]+")
_V6_RE = re.compile(r"[A-Za-z0-9:]+")
_PORT_RE = re.compile(r"[0-9]+")

# Whitespace and control characters are rejected anywhere in the raw
# locator — including the path, which is placed on the remote command line
# unquoted.
_UNSAFE_RAW_RE = re.compile(r"[\s\x00-\x1f\x7f]")


class RemoteLocatorError(ValueError):
    """Raised when a locator string fails parsing or component validation."""


@dataclass(frozen=True)
class RemoteLocator:
    """A parsed ``ssh://[user@]host[:port]/absolute/path`` locator.

    Attributes:
        user: Remote login user, or ``None`` when unspecified.
        host: Hostname or ``ssh_config`` alias; for IPv6 the literal
            *without* brackets (``2001:db8::1``).
        port: TCP port, or ``None`` when unspecified.
        ipv6: True when the locator's host was a bracketed IPv6 literal.
        path: Normalized absolute POSIX path (``.`` collapsed, no ``..``,
            no ``~``, no trailing slash except the root ``/``).
    """

    user: str | None
    host: str
    port: int | None
    ipv6: bool
    path: PurePosixPath


def _validate_component(name: str, value: str, pattern: re.Pattern[str]) -> None:
    """Validate one locator component against its charset and dash rule.

    Args:
        name: Component name for error messages ("user"/"host").
        value: Component as split out by the parser (never raw).
        pattern: Allowed-charset regex (fullmatch).

    Raises:
        RemoteLocatorError: If the value is empty, contains characters
            outside its charset, or starts with ``-``.
    """
    if not pattern.fullmatch(value):
        raise RemoteLocatorError(
            f"invalid {name} {value!r}: allowed charset is {pattern.pattern!r}"
        )
    if value.startswith("-"):
        raise RemoteLocatorError(
            f"invalid {name} {value!r}: leading '-' is not allowed"
        )


def parse_remote_locator(raw: str) -> RemoteLocator:
    """Parse an ``ssh://[user@]host[:port]/absolute/path`` locator.

    Args:
        raw: The locator string exactly as entered by the user.

    Returns:
        A validated :class:`RemoteLocator`. IPv6 hosts have their brackets
        stripped and ``ipv6`` set; the path is normalized via
        :class:`pathlib.PurePosixPath`.

    Raises:
        RemoteLocatorError: On any malformed or unsafe input: missing
            scheme, missing absolute path, whitespace/control characters
            anywhere, component charset or leading-dash violations, port
            outside 1..65535, unbracketed IPv6, ``..`` or ``~`` in the
            path.
    """
    if not isinstance(raw, str):
        raise RemoteLocatorError(f"locator must be a str, got {type(raw).__name__}")
    if not raw.startswith(_SCHEME):
        raise RemoteLocatorError(f"locator must start with {_SCHEME!r}: {raw!r}")
    if _UNSAFE_RAW_RE.search(raw):
        raise RemoteLocatorError(
            f"whitespace and control characters are not allowed: {raw!r}"
        )

    rest = raw[len(_SCHEME):]
    slash = rest.find("/")
    if slash == -1:
        raise RemoteLocatorError(f"locator requires an absolute path: {raw!r}")
    authority, path_str = rest[:slash], rest[slash:]

    # Split user at the LAST '@': '@' appears in no component ever, so any
    # embedded '@' lands in the user side and fails the charset check.
    user: str | None = None
    if "@" in authority:
        user, _, host_part = authority.rpartition("@")
    else:
        host_part = authority

    # Host/port. Bracketed IPv6 first (unbracketed IPv6 is ambiguous with
    # the port separator and therefore invalid, per RFC 3986).
    ipv6 = False
    port_str: str | None = None
    if host_part.startswith("["):
        close = host_part.find("]")
        if close == -1:
            raise RemoteLocatorError(f"unterminated IPv6 bracket: {raw!r}")
        host = host_part[1:close]
        tail = host_part[close + 1:]
        if tail:
            if not tail.startswith(":"):
                raise RemoteLocatorError(f"junk after IPv6 bracket: {raw!r}")
            port_str = tail[1:]
        ipv6 = True
    else:
        host, sep, tail = host_part.rpartition(":")
        if sep:
            port_str = tail
        else:
            host = host_part

    if user is not None:
        _validate_component("user", user, _USER_RE)
    _validate_component("host", host, _V6_RE if ipv6 else _HOST_RE)

    port: int | None = None
    if port_str is not None:
        if not _PORT_RE.fullmatch(port_str):
            raise RemoteLocatorError(f"invalid port {port_str!r} in {raw!r}")
        port = int(port_str)
        if not _MIN_PORT <= port <= _MAX_PORT:
            raise RemoteLocatorError(
                f"port {port} outside {_MIN_PORT}..{_MAX_PORT} in {raw!r}"
            )

    if "~" in path_str:
        raise RemoteLocatorError(f"'~' is not allowed in the path: {raw!r}")
    path = PurePosixPath(path_str)
    if not path.is_absolute():
        raise RemoteLocatorError(f"path must be absolute: {raw!r}")
    if ".." in path.parts:
        raise RemoteLocatorError(f"'..' is not allowed in the path: {raw!r}")
    # POSIX leaves a leading '//' implementation-defined and PurePosixPath
    # preserves it; collapse it so the canonical form is unambiguous.
    if str(path).startswith("//"):
        path = PurePosixPath("/" + str(path).lstrip("/"))

    return RemoteLocator(user=user, host=host, port=port, ipv6=ipv6, path=path)


def locator_string(loc: RemoteLocator) -> str:
    """Serialize a locator to its canonical URI form.

    The port is omitted when it is ``None`` or ``22`` (the SSH default):
    both spellings describe the same destination and must produce the same
    canonical string. The IPv6 host is re-bracketed and the path is the
    already-normalized :class:`PurePosixPath`.

    Args:
        loc: A validated locator.

    Returns:
        The canonical ``ssh://[user@]host[:port]/path`` string.
    """
    user = f"{loc.user}@" if loc.user is not None else ""
    host = f"[{loc.host}]" if loc.ipv6 else loc.host
    port = (
        f":{loc.port}"
        if loc.port is not None and loc.port != _DEFAULT_PORT
        else ""
    )
    return f"{_SCHEME}{user}{host}{port}{loc.path}"


def sha256_fingerprint(
    user: str | None,
    host: str,
    port: int | None,
    path: str | PurePosixPath,
) -> str:
    """Hash a resolved locator identity to its retarget-detection fingerprint.

    The fingerprint source string is ``<user>@<host>:<port><path>`` with a
    missing user serialized as the empty string, a missing port as ``22``,
    and the path normalized via :class:`pathlib.PurePosixPath`. Takes plain
    strings (not a :class:`RemoteLocator`) so Task 6 can feed it the
    ``ssh -G``-resolved symbolic identity; callers normalize (e.g. lowercase
    the host) before calling — this function only hashes what it is given.

    Args:
        user: Resolved user, or ``None``/``""`` when unset.
        host: Resolved host (case already normalized by the caller).
        port: Resolved port, or ``None`` for the default.
        path: Absolute POSIX path (string or ``PurePosixPath``).

    Returns:
        The sha256 hexdigest of the fingerprint source string.
    """
    canonical_user = user if user else ""
    canonical_port = str(port) if port is not None else str(_DEFAULT_PORT)
    canonical_path = str(PurePosixPath(path))
    source = f"{canonical_user}@{host}:{canonical_port}{canonical_path}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def build_ssh_argv(
    loc: RemoteLocator,
    options: Sequence[str],
    command: Sequence[str],
) -> list[str]:
    """Build an ``ssh`` argv from validated locator parts.

    Layout is always ``[*options, ("-p", port)?, ("-l", user)?, "--",
    host, *command]`` — rebuilt from the parsed components, never from the
    raw locator string. The ``--`` guard ends option parsing before the
    host; IPv6 brackets were already stripped at parse time.

    Args:
        loc: A validated locator.
        options: Caller-supplied ssh options (e.g. ``["-o",
            "BatchMode=yes"]``), passed through before the locator parts.
        command: Remote command argv, appended verbatim after the host.

    Returns:
        The assembled argv list.
    """
    argv: list[str] = list(options)
    if loc.port is not None:
        argv += ["-p", str(loc.port)]
    if loc.user is not None:
        argv += ["-l", loc.user]
    return argv + ["--", loc.host, *command]


@dataclass(frozen=True)
class CanonicalTarget:
    """The ``ssh -G``-resolved symbolic identity of a locator's destination.

    Attributes:
        hostname: Config-expanded ``HostName`` exactly as written in the
            ssh config — never DNS-resolved to an IP — lowercased (DNS
            names are case-insensitive, so ``DevBox`` and ``devbox`` are
            the same destination and must fingerprint identically).
        port: Config-expanded TCP port. ``ssh -G`` always reports a
            concrete value (defaulting to 22).
        user: Config-expanded login user, verbatim — ssh treats
            usernames case-sensitively, so no case normalization is
            applied. ``None`` only arises from manual construction;
            ``ssh -G`` always reports a user.
    """

    hostname: str
    port: int
    user: str | None


def canonicalize_locator(
    loc: RemoteLocator, *, ssh_bin: str = "ssh"
) -> CanonicalTarget:
    """Resolve a locator to its config-expanded symbolic identity via ``ssh -G``.

    Runs ``ssh -G`` with the argv built from the locator's validated
    parts — ``ssh -G [-p <port>] [-l <user>] -- <host>``, identical
    construction to :func:`build_ssh_argv` — which prints the fully
    resolved local configuration and exits: no connection is made,
    nothing is resolved through DNS, and no command runs remotely
    (``ssh -G`` takes no command). Called only at binding add/edit,
    manual refresh, and workspace open — never in the send path.

    Only the three identity keys (``hostname``, ``port``, ``user``) are
    consumed; every other key ``ssh -G`` prints is ignored. The hostname
    is lowercased; the user is kept verbatim (see
    :class:`CanonicalTarget` for the case rationale).

    Args:
        loc: A validated locator; the host may be an ``ssh_config`` alias.
        ssh_bin: Path to (or name of) the ``ssh`` binary to execute.

    Returns:
        The canonical target: config-expanded hostname (lowercased),
        port, and user.

    Raises:
        RemoteLocatorError: If ``ssh -G`` cannot be executed, exits
            nonzero, times out, or its output lacks a usable
            ``hostname``/``port``/``user`` line. Messages are static
            reasons that name the offending key but never echo command
            output.
    """
    argv = [ssh_bin, *build_ssh_argv(loc, ["-G"], [])]
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_SSH_G_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RemoteLocatorError(
            f"ssh -G timed out after {_SSH_G_TIMEOUT_SECONDS:g} seconds"
        ) from exc
    except OSError as exc:
        raise RemoteLocatorError(f"ssh -G could not be run: {exc}") from exc
    if completed.returncode != 0:
        raise RemoteLocatorError(f"ssh -G exited with status {completed.returncode}")

    values: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0] in _SSH_G_KEYS:
            values[parts[0]] = parts[1].strip()

    missing = [key for key in _SSH_G_KEYS if not values.get(key)]
    if missing:
        raise RemoteLocatorError(
            "ssh -G output missing required key(s): "
            + ", ".join(repr(key) for key in missing)
        )

    try:
        port = int(values["port"])
    except ValueError as exc:
        raise RemoteLocatorError(
            "ssh -G returned a non-numeric 'port' value"
        ) from exc

    return CanonicalTarget(
        hostname=values["hostname"].lower(),
        port=port,
        user=values["user"],
    )


def canonical_fingerprint(target: CanonicalTarget, path: PurePosixPath) -> str:
    """Fingerprint a canonical target plus path for retarget detection.

    Owns case normalization on behalf of :func:`sha256_fingerprint`
    (whose contract is that callers normalize before hashing): the
    hostname is lowercased here — :func:`canonicalize_locator` already
    lowercases it, and the defensive repeat keeps hand-built targets
    honest — while the user is hashed verbatim because ssh treats
    usernames case-sensitively. The port follows ``sha256_fingerprint``'s
    rule (an int as-is; a ``None`` would serialize as the SSH default
    22) and the path is normalized by the underlying hash.

    Args:
        target: The ``ssh -G``-resolved symbolic identity.
        path: The binding's normalized absolute POSIX path.

    Returns:
        The sha256 hexdigest of ``<user>@<hostname>:<port><path>``.
    """
    return sha256_fingerprint(
        target.user, target.hostname.lower(), target.port, path
    )
