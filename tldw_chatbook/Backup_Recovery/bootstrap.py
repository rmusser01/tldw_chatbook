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
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from typing import ParamSpec, TypeVar

from tldw_chatbook.Utils.platform_files import fcntl, os

from ..Utils.private_paths import (
    PrivatePathError,
    _native_close,
    _open_directory_component,
    _open_verified_parent,
    _trusted_directory_owner,
)
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


def _read(
    parent: int, name: str, *, max_bytes: int = MAX_RECORD, _observed=None, _path=None
) -> dict:
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
        if _observed is not None:
            _observed[_path] = (_control_metadata_stamp(info), False)
        if info.st_size > max_bytes:
            raise ValueError("oversized_record")
        data = bytearray()
        while chunk := os.read(fd, min(64 * 1024, max_bytes - len(data) + 1)):
            data.extend(chunk)
            if len(data) > max_bytes:
                raise ValueError("oversized_record")

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate_key")
                result[key] = value
            return result

        result = json.loads(data, object_pairs_hook=unique)
        if (
            type(result) is not dict
            or type(result.get("version")) is not int
            or result["version"] != 1
        ):
            raise ValueError("record_version")
        return result
    finally:
        os.close(fd)


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


def _control_records(
    root: Path, *, activation: bool = True, _observed=None, _parent=None
) -> tuple[list[dict], list[dict], list[dict]]:
    """Read independent fixed evidence; incomplete paired writes remain fenced."""
    if _parent is None:
        try:
            if stat.S_ISLNK(os.stat(root, follow_symlinks=False).st_mode):
                raise ValueError("bootstrap_linked")
        except FileNotFoundError:
            if _observed is not None:
                _observed[root] = None
            return [], [], []
    pending, profiles, activations = [], [], []
    with pinned_directory(root) if _parent is None else nullcontext(_parent) as parent:
        info = os.fstat(parent)
        if _observed is not None:
            _observed[root] = (_control_metadata_stamp(info), False)
        if info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError("bootstrap_not_private")
        names = os.listdir(parent)
        if len(names) > MAX_RECORDS:
            raise ValueError("too_many_records")
        for name in names:
            if name in ("admission", "unbound-owner", "projection-dependencies"):
                continue
            if name.startswith("activation-") and not name.startswith(
                "activation-update-"
            ):
                digest = name.removeprefix("activation-").removesuffix(".json")
                if (
                    len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                    or not name.endswith(".json")
                ):
                    raise ValueError("unknown_activation_record")
                if not activation:
                    # Fixed activation-only evidence cannot authorize execution
                    # here. Its unavailability must still allow safe inspection.
                    continue
            record = (
                _read(parent, name)
                if _observed is None
                else _read(parent, name, _observed=_observed, _path=root / name)
            )
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
def _registry_read_lock(parent: int, *, _observed=None, _path=None):
    """Observe a finished native publication; never create or repair its lock."""
    fd = os.open(
        "registry.lock", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
    )
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o077
        ):
            raise ValueError("registry_lock_unsafe")
        if _observed is not None:
            _observed[_path] = (_control_metadata_stamp(info), False)
        fcntl.flock(fd, fcntl.LOCK_SH)
        current = os.stat("registry.lock", dir_fd=parent, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError("registry_lock_changed")
        yield
    finally:
        os.close(fd)


@contextmanager
def _pinned_control_child(parent, name, expected):
    """Own one child of the verified control root for its synchronous read."""
    child = _open_directory_component(parent, name)
    try:
        info = os.fstat(child)
        if (
            not stat.S_ISDIR(info.st_mode)
            or not _trusted_directory_owner(info, os.geteuid())
            or _control_metadata_stamp(info, directory=True)
            != _control_metadata_stamp(expected, directory=True)
        ):
            raise ValueError("authority_unsafe")
        yield child
    finally:
        _native_close(child)


def _registry(root: Path, *, _observed=None, _parent=None) -> dict | None:
    authority = root / "admission"
    if _parent is not None:
        info = os.fstat(_parent)
        if info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError("bootstrap_not_private")
    try:
        info = (
            os.stat(authority, follow_symlinks=False)
            if _parent is None
            else os.stat("admission", dir_fd=_parent, follow_symlinks=False)
        )
    except FileNotFoundError:
        if _observed is not None:
            _observed[authority] = None
        return None
    if _observed is not None:
        # Namespace lease files may change independently; only authority posture
        # and the explicit registry-pending filename affect this observation.
        _observed[authority] = (_control_metadata_stamp(info, directory=True), True)
    if stat.S_ISLNK(info.st_mode) or info.st_mode & 0o077:
        raise ValueError("authority_unsafe")
    marker = root / "unbound-owner"
    marker_info = (
        os.stat(marker, follow_symlinks=False)
        if _parent is None
        else os.stat("unbound-owner", dir_fd=_parent, follow_symlinks=False)
    )
    if _observed is not None:
        _observed[marker] = (_control_metadata_stamp(marker_info), False)
    if (
        not stat.S_ISREG(marker_info.st_mode)
        or marker_info.st_nlink != 1
        or marker_info.st_uid != os.geteuid()
        or marker_info.st_mode & 0o077
    ):
        raise ValueError("enrollment_marker_unsafe")
    parent_scope = (
        pinned_directory(authority)
        if _parent is None
        else _pinned_control_child(_parent, "admission", info)
    )
    with parent_scope as parent:
        lock = (
            _registry_read_lock(parent)
            if _observed is None
            else _registry_read_lock(
                parent, _observed=_observed, _path=authority / "registry.lock"
            )
        )
        with lock:
            return _registry_contents(parent, authority, _observed=_observed)


def _registry_contents(parent, authority, *, _observed=None):
    if "registry.pending.json" in os.listdir(parent):
        raise ValueError("registry_pending")
    if _observed is not None:
        _observed[authority / "registry.pending.json"] = None
    result = (
        _read(parent, "registry.json")
        if _observed is None
        else _read(
            parent,
            "registry.json",
            _observed=_observed,
            _path=authority / "registry.json",
        )
    )
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


def _control_metadata_stamp(info, *, directory=False):
    posture = (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_uid,
        info.st_gid,
    )
    return (
        posture
        if directory
        else (*posture, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    )


def _control_readers_current():
    """Keep pre-existing custom reader signatures outside the borrowed-FD path."""
    for (
        name,
        function,
        code,
        defaults,
        keywords,
        items,
    ) in _ACTIVATION_PREPARATION_CALLBACKS:
        if name not in {"_control_records", "_registry"}:
            continue
        if (
            globals().get(name) is not function
            or function.__code__ is not code
            or function.__defaults__ is not defaults
            or function.__kwdefaults__ is not keywords
            or (
                keywords is not None
                and (
                    len(keywords) != len(items)
                    or any(
                        key not in keywords or keywords[key] is not value
                        for key, value in items
                    )
                )
            )
        ):
            return False
    return True


@contextmanager
def _control_observation(root: Path):
    """Share freshly validated control metadata only through this finite read.

    Completion rechecks named identities and change stamps before any caller can
    return a decision. No record, absence or filesystem observation is cached.
    """
    observed = {}
    if not _control_readers_current():
        records = _control_records(root, _observed=observed)
        registry = _registry(root, _observed=observed)
    else:
        with ExitStack() as stack:
            parent = None
            try:
                # The root may live directly under a trusted sticky directory.
                ancestor, _ = _open_verified_parent(root, missing_leaf_allowed=False)
                stack.callback(_native_close, ancestor)
            except PrivatePathError as error:
                if error.result.reason != "missing_parent":
                    raise
                # Preserve the original missing-control receipts and race checks.
            else:
                try:
                    info = os.stat(root.name, dir_fd=ancestor, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    if stat.S_ISLNK(info.st_mode):
                        raise ValueError("bootstrap_linked")
                    parent = stack.enter_context(
                        _pinned_control_child(ancestor, root.name, info)
                    )
            if not _control_readers_current():
                raise ValueError("projection_control_observation_changed")
            records = _control_records(root, _observed=observed, _parent=parent)
            if not _control_readers_current():
                raise ValueError("projection_control_observation_changed")
            registry = _registry(root, _observed=observed, _parent=parent)
            if not _control_readers_current():
                raise ValueError("projection_control_observation_changed")
    # Initial pins retire before the caller reads dependent activation state.
    yield records, registry
    if os.name == "nt":
        ancestors = {parent for path in observed for parent in path.parents}
        current = os.stat_many_for_admission(tuple(set(observed) | ancestors))
        # The native snapshot already visits every ancestor. Keep its receipts
        # and apply exactly the native private-parent walk's owner/mode rules.
        for path in ancestors:
            value = current[path]
            if value is None:
                continue
            info = value[0]
            mode = stat.S_IMODE(info.st_mode)
            if (
                not stat.S_ISDIR(info.st_mode)
                or not _trusted_directory_owner(info, os.geteuid())
                or mode & 0o022
                and not mode & stat.S_ISVTX
            ):
                raise ValueError("projection_control_observation_changed")
        infos = {
            path: None if current[path] is None else current[path][0]
            for path in observed
        }
    else:
        # Reopen the native named parents before relative metadata checks. A
        # symlink or shared ancestor introduced mid-read cannot inherit proof.
        with ExitStack() as stack:
            parents = {
                path: stack.enter_context(pinned_directory(path))
                for path in (root, root / "admission")
                if observed.get(path) is not None
            }
            infos = {}
            for path in observed:
                try:
                    if path in parents:
                        continue
                    if path.parent in parents:
                        info = os.stat(
                            path.name,
                            dir_fd=parents[path.parent],
                            follow_symlinks=False,
                        )
                    else:
                        info = os.stat(path, follow_symlinks=False)
                    infos[path] = info
                except FileNotFoundError:
                    infos[path] = None
            # Reopen descendants before ancestors while the original pins are
            # live. fstat alone cannot prove those pins still have these names.
            for path in sorted(parents, key=lambda path: len(path.parts), reverse=True):
                with pinned_directory(path) as named:
                    current = os.fstat(named)
                    held = os.fstat(parents[path])
                    if (current.st_dev, current.st_ino) != (held.st_dev, held.st_ino):
                        raise ValueError("projection_control_observation_changed")
                    infos[path] = current
            _check_control_observation(observed, infos)
        return
    _check_control_observation(observed, infos)


def _check_control_observation(observed, infos):
    for path, expected in observed.items():
        info = infos[path]
        if expected is None:
            unchanged = info is None
        else:
            stamp, directory = expected
            unchanged = (
                info is not None
                and _control_metadata_stamp(info, directory=directory) == stamp
            )
        if not unchanged:
            raise ValueError("projection_control_observation_changed")


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
    tree_reader = None
    if os.name == "nt":
        from types import FunctionType, MethodType

        from ..Utils import windows_files

        provenance = getattr(windows_files, "_WINDOWS_BINDING_TREE_ORIGINAL", None)
        if type(provenance) is tuple and len(provenance) == 8:
            tree_class, function, code, namespace, defaults, keywords, closure, name = (
                provenance
            )
            tree_facade = os
            if type(tree_facade) is tree_class:
                tree_reader = object.__getattribute__(tree_facade, "__dict__").get(name)

            def tree_current():
                return (
                    os is tree_facade
                    and type(tree_facade) is tree_class
                    and tree_class is getattr(windows_files, "WindowsOS", None)
                    and tree_class
                    is getattr(windows_files, "_WINDOWS_METADATA_CLASS_ORIGINAL", None)
                    and getattr(windows_files, "_WINDOWS_BINDING_TREE_ORIGINAL", None)
                    is provenance
                    and type(function) is FunctionType
                    and vars(tree_class).get(name) is function
                    and type(tree_reader) is MethodType
                    and tree_reader.__self__ is tree_facade
                    and tree_reader.__func__ is function
                    and object.__getattribute__(tree_facade, "__dict__").get(name)
                    is tree_reader
                    and function.__code__ is code
                    and namespace is vars(windows_files)
                    and function.__globals__ is namespace
                    and function.__defaults__ is defaults
                    and function.__kwdefaults__ is keywords
                    and function.__closure__ is closure
                )

            if not tree_current():
                tree_reader = None
    if tree_reader is not None:
        from ..Utils.private_paths import (
            PrivatePathError,
            PrivatePathResult,
            PrivatePathStatus,
            _describe_stat,
            _offender_mode,
            _offender_path,
            _trusted_directory_owner,
        )

        selected = tuple(
            Path(raw).resolve(strict=True) for raw in effective_roots(roots, entries)
        )
        if not tree_current():
            raise ValueError("binding_tree_reader_changed")
        observed = tree_reader(
            tuple({node for path in selected for node in (*path.parents, path)})
        )
        if not tree_current():
            raise ValueError("binding_tree_reader_changed")
        for path in selected:
            reader = path.parent / ".bootstrap-reader"
            # Match the original missing-leaf parent walk, including its stricter
            # final-parent rule. A sticky ancestor never admits a shared final parent.
            for parent in reversed(path.parents or (path,)):
                value = observed[parent]
                if value is None:
                    raise PrivatePathError(
                        PrivatePathResult(
                            reader,
                            PrivatePathStatus.UNSAFE_PARENT,
                            reason="missing_parent",
                            offender_path=_offender_path(list(parent.parts[1:])),
                        )
                    )
                info = value[0]
                if not stat.S_ISDIR(info.st_mode):
                    raise PrivatePathError(
                        PrivatePathResult(
                            reader,
                            PrivatePathStatus.LINK_OR_NON_REGULAR,
                            reason="non_directory_parent",
                            offender_path=_offender_path(list(parent.parts[1:])),
                        )
                    )
                mode = stat.S_IMODE(info.st_mode)
                reason = (
                    "untrusted_directory_owner"
                    if not _trusted_directory_owner(info, os.geteuid())
                    else "missing_leaf_in_shared_sticky_parent"
                    if mode & 0o022 and mode & stat.S_ISVTX and parent == path.parent
                    else "shared_writable_parent"
                    if mode & 0o022 and not mode & stat.S_ISVTX
                    else None
                )
                if reason is not None:
                    raise PrivatePathError(
                        PrivatePathResult(
                            reader,
                            PrivatePathStatus.UNSAFE_PARENT,
                            reason=reason,
                            offender_path=_offender_path(list(parent.parts[1:])),
                            offender_detail=_describe_stat(info),
                            offender_mode=_offender_mode(info),
                        )
                    )
            if not path.name:
                raise ValueError("invalid_windows_component")
            value = observed[path]
            if value is None:
                raise FileNotFoundError(path)
            info = value[0]
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                raise ValueError("root_unverified")
    else:
        for raw in effective_roots(roots, entries):
            path = Path(raw).resolve(strict=True)
            with pinned_directory(path.parent) as parent:
                info = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
                if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                    raise ValueError("root_unverified")
    if tree_reader is not None and not tree_current():
        raise ValueError("binding_tree_reader_changed")
    return match


def _startup_permission_from_records(
    selector, bootstrap_root, pending, profiles, registry
):
    """Evaluate the existing startup rules against one validated finite read."""
    try:
        if not pending:
            # A normal config edit can require re-enrollment but is not recovery.
            if (
                profiles or (bootstrap_root / "unbound-owner").exists()
            ) and registry is None:
                return False, "recovery_scope_uncertain"
            return True, "startup_allowed"
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


def startup_permission(config_selector: Path, bootstrap_root: Path) -> tuple[bool, str]:
    """Return admission without loading config or creating any default state."""
    try:
        selector = lexical_path(config_selector)
        pending, profiles = _records(bootstrap_root)
        registry = _registry(bootstrap_root)
        return _startup_permission_from_records(
            selector, bootstrap_root, pending, profiles, registry
        )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, AttributeError):
        return False, "recovery_scope_uncertain"


def require_startup_permission() -> None:
    """Bounded launcher refusal. Custom config cannot relocate this check."""
    allowed, reason = startup_permission(
        effective_config_path(), default_bootstrap_root()
    )
    if not allowed:
        raise SystemExit("Recovery required: " + reason)


# Only direct activation-preparation readers are qualified; public readers keep
# their ordinary signatures and source order when any callback input changes.
_ACTIVATION_PREPARATION_CALLBACKS = tuple(
    (
        name,
        function,
        function.__code__,
        function.__defaults__,
        function.__kwdefaults__,
        tuple(dict.items(function.__kwdefaults__ or {})),
    )
    for name in (
        "startup_permission",
        "_records",
        "_control_records",
        "_registry",
        "_startup_permission_from_records",
    )
    for function in (globals()[name],)
)
_CONTROL_OBSERVATION_ORIGINAL = (
    _control_observation,
    _control_observation.__code__,
    _control_observation.__wrapped__,
    _control_observation.__wrapped__.__code__,
    _control_observation.__closure__,
)
