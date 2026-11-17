"""Read local recovery evidence before config imports, fallback, or composition.

This module deliberately uses only stdlib and the private native path reader.
There is no cleanup, catalog fallback, config parsing, or native qualification here.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
import stat
import threading
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, ParamSpec, TypeVar, overload

from tldw_chatbook.Utils.platform_files import fcntl, os

from ..Utils.private_paths import _native_close, _open_verified_parent
from .profile_paths import default_config_path, effective_config_path, lexical_path


def inode_token(info: os.stat_result) -> str:
    """Return the physical-identity token for a file: its inode, not its device.

    TASK-34200: macOS can renumber a volume's ``st_dev`` across a reboot. A
    token that pinned the device made the app treat its own unchanged files as
    replaced and refuse every start ("Recovery required:
    recovery_scope_uncertain"). A replaced or copied file still gets a new
    inode, so the replacement fence holds; the path tokens still pin where.

    Args:
        info: ``os.stat`` result of the file.

    Returns:
        ``"inode:<st_ino>"``.
    """
    return inode_token_for(info.st_ino)


def inode_token_for(st_ino: int) -> str:
    """Return the physical-identity token for an inode number (TASK-34200).

    Args:
        st_ino: The file's inode number.

    Returns:
        ``"inode:<st_ino>"``.
    """
    return f"inode:{st_ino}"


#: A token recorded before TASK-34200: ``inode:<st_dev>:<st_ino>``, both numeric.
_LEGACY_INODE_TOKEN = re.compile(r"inode:(\d+):(\d+)")


def identity_view(tokens: Iterable[str]) -> set[str]:
    """Return tokens as identity compares them, whatever format recorded them.

    Registries written before TASK-34200 hold ``"inode:<dev>:<ino>"``; this
    reads them as ``"inode:<ino>"`` so an existing registry keeps matching
    after a device renumbering. Every other token is unchanged.

    Args:
        tokens: Recorded or freshly observed identity tokens.

    Returns:
        The device-free set used for every membership/overlap comparison.
    """
    view = set()
    for token in tokens:
        legacy = _LEGACY_INODE_TOKEN.fullmatch(token)
        # Only a well-formed legacy token is normalized; a malformed one stays
        # as recorded, so it can never match a current identity (Qodo #2994).
        view.add(inode_token_for(int(legacy.group(2))) if legacy else token)
    return view


MAX_RECORD = 1048576
MAX_RECORDS = 4096


class RecoveryRequired(RuntimeError):
    """A bounded reason code, never a local locator or original exception."""


#: PERF-07/08 (ADR-126 amendment, 2026-09-29). Advanced by in-process writers
#: of admission state so reused admission evidence is dropped at once. This is
#: hardening: per-call stamps are what detect every writer, other processes too.
_admission_epoch = 0
_admission_epoch_lock = threading.Lock()


def advance_admission_epoch() -> None:
    """Invalidate every piece of reused admission evidence in this process."""
    global _admission_epoch
    with _admission_epoch_lock:
        _admission_epoch += 1


_P = ParamSpec("_P")
_R = TypeVar("_R")


def advances_admission_epoch(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Advance the admission epoch before and after a writer of admission state.

    Args:
        function: A writer of admission records, the registry, or the profile.

    Returns:
        The writer, wrapped; its arguments, result and exceptions pass through.
    """

    @functools.wraps(function)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        advance_admission_epoch()
        try:
            return function(*args, **kwargs)
        finally:
            advance_admission_epoch()

    return wrapper


def default_bootstrap_root() -> Path:
    return default_config_path().parent / "recovery-bootstrap"


@contextmanager
def pinned_directory(root: Path, *, _close: Callable[[int], None] | None = None):
    parent, _ = _open_verified_parent(
        root / ".bootstrap-reader", missing_leaf_allowed=True, _close=_close
    )
    try:
        yield parent
    finally:
        (_close or _native_close)(parent)


def _read_record_bytes(
    parent: int, name: str, *, max_bytes: int, _observed: dict | None = None
) -> bytes:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o077
        ):
            raise ValueError("unsafe_record")
        if info.st_size > max_bytes:
            raise ValueError("oversized_record")
        data = bytearray()
        while chunk := os.read(fd, min(64 * 1024, max_bytes - len(data) + 1)):
            data.extend(chunk)
            if len(data) > max_bytes:
                raise ValueError("oversized_record")
        if _observed is not None:
            current = _input_identity(os.fstat(fd))
            if current != _input_identity(info):
                raise ValueError("ordinary_input_changed")
            _observed[name] = (current, hashlib.sha256(data).digest())
        return bytes(data)
    finally:
        os.close(fd)


def _read(parent: int, name: str, *, max_bytes: int = MAX_RECORD) -> dict:
    return _decode_record(_read_record_bytes(parent, name, max_bytes=max_bytes))


def _decode_record(raw: bytes) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_key")
            result[key] = value
        return result

    result = json.loads(raw, object_pairs_hook=unique)
    if (
        type(result) is not dict
        or type(result.get("version")) is not int
        or result["version"] != 1
    ):
        raise ValueError("record_version")
    return result


def _strings(values: object) -> bool:
    return (
        type(values) is list
        and bool(values)
        and len(values) <= MAX_RECORDS
        and all(type(v) is str and 0 < len(v) <= 4096 and "\0" not in v for v in values)
        and len(set(values)) == len(values)
    )


def _paths(values: object) -> bool:
    return _strings(values) and all(
        Path(v).is_absolute() and ".." not in Path(v).parts for v in values
    )


def _key(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _fingerprint(selector: Path) -> str:
    """Read a bounded ordinary config without parsing or changing its mode."""
    with pinned_directory(selector.parent) as parent:
        fd = os.open(
            selector.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid():
                raise ValueError("selector_unverified")
            raw = os.read(fd, MAX_RECORD + 1)
            after = os.fstat(fd)
            identity = lambda info: (
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            )
            if len(raw) > MAX_RECORD or identity(after) != identity(before):
                raise ValueError("selector_unverified")
            return hashlib.sha256(raw).hexdigest()
        finally:
            os.close(fd)


def _overlap(left: Path, right: Path) -> bool:
    a, b = left.resolve(), right.resolve()
    if a == b or a in b.parents or b in a.parents:
        return True
    try:
        return a.samefile(b)
    except FileNotFoundError:
        return False


def _activation_witness(record: object) -> None:
    """Validate local generation evidence without importing execution owners."""
    if (
        type(record) is not dict
        or set(record)
        != {"operation_id", "generation", "owners", "namespaces", "store_root"}
        or any(
            type(record[key]) is not str
            or not 0 < len(record[key]) <= 256
            or "\0" in record[key]
            for key in ("operation_id", "generation")
        )
        or not _strings(record["owners"])
        or any(len(owner) > 256 for owner in record["owners"])
        or record["owners"] != sorted(record["owners"])
        or not _strings(record["namespaces"])
        or record["namespaces"] != sorted(record["namespaces"])
        or not _paths([record["store_root"]])
    ):
        raise ValueError("invalid_activation_witness")


def _same_activation_generation(
    left: Mapping[str, object], right: Mapping[str, object]
) -> bool:
    """Compare validated generation identity across per-profile namespace scopes."""
    return all(
        left[key] == right[key]
        for key in ("operation_id", "generation", "owners", "store_root")
    )


def _validate_ancestry_record(root: Path, parent: int, record: dict) -> None:
    """Require a creation receipt bound to this actual local directory."""
    info = os.fstat(parent)
    device = record.get("root_device")
    valid_device = (
        type(device) is int and device == info.st_dev  # noqa: E721 - JSON bool is not an integer
        if os.name == "nt"
        else device is None
    )
    if (
        set(record) != {"version", "root_inode", "root_device", "root_path"}
        or type(record["version"]) is not int  # noqa: E721 - JSON bool is not an integer
        or record["version"] != 1
        or type(record["root_inode"]) is not int  # noqa: E721 - JSON bool is not an integer
        or record["root_inode"] != info.st_ino
        or not valid_device
        or record["root_path"] != os.path.normcase(str(root.resolve(strict=True)))
    ):
        raise ValueError("invalid_bootstrap_ancestry")


@overload
def _control_records(
    root: Path, *, activation: bool = True, ordinary_only: Literal[False] = False
) -> tuple[list[dict], list[dict], list[dict]]: ...


@overload
def _control_records(
    root: Path, *, activation: bool = True, ordinary_only: Literal[True]
) -> tuple[list[dict], list[dict], list[dict]] | None: ...


def _control_records(
    root: Path, *, activation: bool = True, ordinary_only: bool = False
) -> tuple[list[dict], list[dict], list[dict]] | None:
    """Read independent fixed evidence; incomplete paired writes remain fenced."""
    try:
        if stat.S_ISLNK(os.stat(root, follow_symlinks=False).st_mode):
            raise ValueError("bootstrap_linked")
    except FileNotFoundError:
        return [], [], []
    with pinned_directory(root) as parent:
        info = os.fstat(parent)
        if info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError("bootstrap_not_private")
        names = _control_names(parent)
        return _parse_control_records(
            root, parent, names, _read, activation, ordinary_only
        )


def _control_names(parent):
    names = os.listdir(parent)
    if len(names) > MAX_RECORDS:
        raise ValueError("too_many_records")
    return names


def _parse_control_records(root, parent, names, read, activation, ordinary_only):
    pending, profiles, activations = [], [], []
    for name in names:
        if name in ("admission", "unbound-owner", "projection-dependencies"):
            continue
        if name.startswith("activation-") and not name.startswith("activation-update-"):
            digest = name.removeprefix("activation-").removesuffix(".json")
            if (
                len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                or not name.endswith(".json")
            ):
                raise ValueError("unknown_activation_record")
            if ordinary_only:
                # The original route owns activation-only validation and
                # its ordering relative to the startup projection.
                return None
            if not activation:
                # Fixed activation-only evidence cannot authorize execution
                # here. Its unavailability must still allow safe inspection.
                continue
        record = read(parent, name)
        if name.startswith("activation-update-"):
            # This is explicit write-intent evidence, never a record to skip
            # or repair on reads. Even a damaged intent requires recovery.
            raise ValueError("activation_update_pending")
        if name.startswith("activation-"):
            if (
                set(record) != {"version", "selector", "activation"}
                or not _paths([record["selector"]])
                or name != "activation-" + _key(record["selector"]) + ".json"
            ):
                raise ValueError("invalid_activation_association")
            _activation_witness(record["activation"])
            activations.append(record)
        elif name == "ancestry-settled.json":
            _validate_ancestry_record(root, parent, record)
        elif name.startswith("pending-"):
            if (
                set(record)
                != {
                    "version",
                    "operation_id",
                    "namespaces",
                    "control_root",
                    "selectors",
                }
                or type(record["operation_id"]) is not str
                or not 0 < len(record["operation_id"]) <= 256
                or not _strings(record["namespaces"])
                or not _paths(record["selectors"])
                or not _paths([record["control_root"]])
                or name != "pending-" + _key(record["operation_id"]) + ".json"
            ):
                raise ValueError("invalid_pending")
            if ordinary_only:
                return None
            pending.append(record)
        elif name.startswith("profile-"):
            if (
                set(record) - {"activation"}
                != {"version", "selector", "fingerprint", "namespaces", "roots"}
                or not _paths([record["selector"]])
                or not _strings(record["namespaces"])
                or not _paths(record["roots"])
                or type(record["fingerprint"]) is not str
                or len(record["fingerprint"]) != 64
                or any(c not in "0123456789abcdef" for c in record["fingerprint"])
                or name != "profile-" + _key(record["selector"]) + ".json"
            ):
                raise ValueError("invalid_profile")
            if ordinary_only and "activation" in record:
                return None
            if activation and "activation" in record:
                _activation_witness(record["activation"])
                if record["activation"]["namespaces"] != record["namespaces"]:
                    raise ValueError("invalid_profile_activation_scope")
            profiles.append(record)
        else:
            raise ValueError("unknown_record")
    return pending, profiles, activations


def _records(root: Path) -> tuple[list[dict], list[dict]]:
    """Read content admission, leaving activation validation to its read gate."""
    pending, profiles, _ = _control_records(root, activation=False)
    return pending, profiles


@contextmanager
def _registry_read_lock(parent: int, *, access: int = os.O_RDONLY):
    """Observe a finished native publication; never create or repair its lock."""
    fd = os.open("registry.lock", access | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o077
        ):
            raise ValueError("registry_lock_unsafe")
        fcntl.flock(fd, fcntl.LOCK_SH)
        current = os.stat("registry.lock", dir_fd=parent, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError("registry_lock_changed")
        yield
    finally:
        os.close(fd)


def _registry(root: Path) -> dict | None:
    authority = root / "admission"
    try:
        info = os.stat(authority, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(info.st_mode) or info.st_mode & 0o077:
        raise ValueError("authority_unsafe")
    marker = root / "unbound-owner"
    _require_enrollment_marker(os.stat(marker, follow_symlinks=False))
    with pinned_directory(authority) as parent, _registry_read_lock(parent):
        _require_finished_registry(parent)
        return _parse_registry(_read(parent, "registry.json"))


def _require_enrollment_marker(info):
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o077
    ):
        raise ValueError("enrollment_marker_unsafe")


def _require_finished_registry(parent):
    if "registry.pending.json" in os.listdir(parent):
        raise ValueError("registry_pending")


def _parse_registry(result):
    if set(result) != {"version", "entries"} or type(result["entries"]) is not dict:
        raise ValueError("invalid_registry")
    for name, entry in result["entries"].items():
        if (
            not name
            or type(entry) is not dict
            or set(entry) != {"roots", "historical", "pending", "proposed"}
            or not _paths(entry["roots"])
            or type(entry["historical"]) is not list
            or not all(type(v) is str for v in entry["historical"])
            or entry["pending"] is not None
            or entry["proposed"] != []
        ):
            raise ValueError("uncertain_registry")
    return result["entries"]


def _input_identity(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


@contextmanager
def _pinned_admission_child(root_fd):
    admission = os.open(
        "admission", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd
    )
    try:
        info = os.fstat(admission)
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid not in {0, os.geteuid()}
            or info.st_mode & 0o077
        ):
            raise ValueError("authority_unsafe")
        yield admission, (info.st_dev, info.st_ino)
    finally:
        _native_close(admission)


def _pinned_ordinary_inputs(root, selector, require_current):
    """Read one ordinary tuple with explicitly owned root and admission pins."""
    with pinned_directory(root) as root_fd:
        require_current()
        root_before = os.fstat(root_fd)
        require_current()
        if root_before.st_uid != os.geteuid() or root_before.st_mode & 0o077:
            raise ValueError("bootstrap_not_private")
        root_identity = (root_before.st_dev, root_before.st_ino)
        names = _control_names(root_fd)
        require_current()
        controls = {}

        def read_control(parent, name):
            require_current()
            raw = _read_record_bytes(
                parent, name, max_bytes=MAX_RECORD, _observed=controls
            )
            require_current()
            return _decode_record(raw)

        inputs = _parse_control_records(root, root_fd, names, read_control, True, True)
        require_current()
        if inputs is None:
            # Nonordinary initial evidence belongs to the original reader.
            return None
        _, profiles, _ = inputs
        with _pinned_admission_child(root_fd) as (admission, admission_identity):
            require_current()
            lock_before = os.stat(
                "registry.lock", dir_fd=admission, follow_symlinks=False
            )
            require_current()
            marker_before = os.stat(
                "unbound-owner", dir_fd=root_fd, follow_symlinks=False
            )
            require_current()
            _require_enrollment_marker(marker_before)
            registry_receipt = {}
            require_current()
            with _registry_read_lock(admission):
                require_current()
                _require_finished_registry(admission)
                require_current()
                raw = _read_record_bytes(
                    admission,
                    "registry.json",
                    max_bytes=MAX_RECORD,
                    _observed=registry_receipt,
                )
                require_current()
                registry = _parse_registry(_decode_record(raw))
            require_current()
            allowed, reason = _ordinary_startup_permission(root, profiles, registry)
            if not allowed:
                raise RecoveryRequired(reason)
            fingerprint = None
            if any(profile["selector"] == str(selector) for profile in profiles):
                require_current()
                fingerprint = _fingerprint(selector)
            require_current()
            with pinned_directory(root) as current_root:
                require_current()
                root_after = os.fstat(current_root)
                require_current()
                if (root_after.st_dev, root_after.st_ino) != root_identity:
                    raise RecoveryRequired("ordinary_activation_ancestry_changed")
                if root_after.st_uid != os.geteuid() or root_after.st_mode & 0o077:
                    raise ValueError("bootstrap_not_private")
                with _pinned_admission_child(current_root) as (
                    current_admission,
                    current_identity,
                ):
                    require_current()
                    if current_identity != admission_identity:
                        raise RecoveryRequired("ordinary_activation_ancestry_changed")
                    lock_after = os.stat(
                        "registry.lock", dir_fd=current_admission, follow_symlinks=False
                    )
                    require_current()
                    marker_after = os.stat(
                        "unbound-owner", dir_fd=current_root, follow_symlinks=False
                    )
                    require_current()
                    _require_enrollment_marker(marker_after)
                    _require_finished_registry(current_admission)
                    require_current()
                    current_registry = {}
                    _read_record_bytes(
                        current_admission,
                        "registry.json",
                        max_bytes=MAX_RECORD,
                        _observed=current_registry,
                    )
                    require_current()
                    _require_finished_registry(current_admission)
                    require_current()
                    if (
                        _input_identity(lock_before) != _input_identity(lock_after)
                        or _input_identity(marker_before)
                        != _input_identity(marker_after)
                        or current_registry != registry_receipt
                    ):
                        raise RecoveryRequired("ordinary_activation_registry_changed")
                require_current()
                current_controls = {}
                for name in controls:
                    require_current()
                    _read_record_bytes(
                        current_root,
                        name,
                        max_bytes=MAX_RECORD,
                        _observed=current_controls,
                    )
                require_current()
                if current_controls != controls or sorted(
                    _control_names(current_root)
                ) != sorted(names):
                    raise RecoveryRequired("ordinary_activation_controls_changed")
            if fingerprint is not None:
                require_current()
                if _fingerprint(selector) != fingerprint:
                    raise RecoveryRequired("ordinary_activation_selector_changed")
            require_current()
            return (
                profiles,
                registry,
                (lock_after.st_dev, lock_after.st_ino),
                fingerprint,
            )


def effective_roots(
    roots: Iterable[str | Path], entries: Iterable[Mapping[str, object]]
) -> tuple[Path, ...]:
    """Defer absent-alias proof until an enrolled root is actually absent.

    Args:
        roots: Declared paths from the selected namespace set.
        entries: Registry entries from that same namespace set.

    Returns:
        Existing roots unchanged, or roots qualified by the native absence proof.
        Consumers still perform their ordinary strict identity validation.
    """
    roots = tuple(dict.fromkeys(Path(root) for root in roots))
    entries = tuple(entries)
    if all(os.path.lexists(root) for root in roots):
        return roots
    from .effective_roots import effective_roots as prove_absence

    return prove_absence(roots, entries)


def _binding(
    selector: Path, profiles: list[dict], registry: dict | None
) -> dict | None:
    match = next((r for r in profiles if r["selector"] == str(selector)), None)
    if match is None:
        return None
    try:
        if _fingerprint(selector) != match["fingerprint"]:
            return None
    except FileNotFoundError:
        return None
    if registry is None or any(n not in registry for n in match["namespaces"]):
        raise ValueError("binding_authority_missing")
    roots = sorted({p for n in match["namespaces"] for p in registry[n]["roots"]})
    if roots != match["roots"]:
        raise ValueError("binding_mapping_changed")
    # Declared mapping equality stays exact. Omit only absent aliases covered
    # by this profile's already enrolled and physically verified directory.
    entries = tuple(registry[name] for name in match["namespaces"])
    for raw in effective_roots(roots, entries):
        path = Path(raw).resolve(strict=True)
        with pinned_directory(path.parent) as parent:
            info = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                raise ValueError("root_unverified")
    return match


def _ordinary_startup_permission(root, profiles, registry):
    # A normal config edit can require re-enrollment but is not recovery.
    if (profiles or (root / "unbound-owner").exists()) and registry is None:
        return False, "recovery_scope_uncertain"
    return True, "startup_allowed"


def _ordinary_registry(root, profiles):
    """Read the native registry once for a complete ordinary control scan."""
    try:
        registry = _registry(root)
        allowed, reason = _ordinary_startup_permission(root, profiles, registry)
    except RecoveryRequired:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        AttributeError,
    ) as error:
        if _ordinary_close_uncertain(error):
            raise
        return None, False, "recovery_scope_uncertain"
    return registry, allowed, reason


def _ordinary_close_uncertain(error):
    if os.name != "nt":
        return False
    from ..Utils.windows_files import _AdmissionMetadataCloseError

    return isinstance(error, _AdmissionMetadataCloseError)


def startup_permission(config_selector: Path, bootstrap_root: Path) -> tuple[bool, str]:
    """Return admission without loading config or creating any default state."""
    try:
        selector = lexical_path(config_selector)
        pending, profiles = _records(bootstrap_root)
        registry = _registry(bootstrap_root)
        if not pending:
            return _ordinary_startup_permission(bootstrap_root, profiles, registry)
        if any(
            any(_overlap(selector, Path(p)) for p in r["selectors"]) for r in pending
        ):
            return False, "recovery_pending"
        binding = _binding(selector, profiles, registry)
        if binding is None:
            return False, "recovery_scope_uncertain"
        for record in pending:
            if set(binding["namespaces"]) & set(record["namespaces"]):
                return False, "recovery_pending"
            if registry is None or any(n not in registry for n in record["namespaces"]):
                return False, "recovery_scope_uncertain"
            own_tokens = identity_view(
                t for n in binding["namespaces"] for t in registry[n]["historical"]
            )
            affected_tokens = identity_view(
                t for n in record["namespaces"] for t in registry[n]["historical"]
            )
            if own_tokens & affected_tokens:
                return False, "recovery_pending"
            affected = record["selectors"] + [
                p for n in record["namespaces"] for p in registry[n]["roots"]
            ]
            if any(
                _overlap(Path(a), Path(b))
                for a in [binding["selector"]] + binding["roots"]
                for b in affected
            ):
                return False, "recovery_pending"
        return True, "startup_allowed"
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, AttributeError):
        return False, "recovery_scope_uncertain"


def require_startup_permission() -> None:
    """Bounded launcher refusal. Custom config cannot relocate this check."""
    allowed, reason = startup_permission(
        effective_config_path(), default_bootstrap_root()
    )
    if not allowed:
        raise SystemExit("Recovery required: " + reason)
