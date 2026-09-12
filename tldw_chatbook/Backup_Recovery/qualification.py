"""Read-only, operation-specific native evidence. No runtime self-qualification."""

from __future__ import annotations

import ctypes
import platform
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


def release_capability(
    *,
    helper: bool,
    owner_coverage: bool,
    admission: bool,
    archive: bool,
    native_publish: bool,
    restore: bool,
    product_flow: bool,
) -> bool:
    """Combine demonstrated release gates; this supplies no native evidence."""
    return all(
        (
            helper,
            owner_coverage,
            admission,
            archive,
            native_publish,
            restore,
            product_flow,
        )
    )


# Darwin's statfs64 ABI from the installed sys/mount.h; no pathname shell probe.
class _StatFS(ctypes.Structure):
    _fields_ = [
        ("bsize", ctypes.c_uint32),
        ("iosize", ctypes.c_int32),
        ("counts", ctypes.c_uint64 * 5),
        ("fsid", ctypes.c_int32 * 2),
        ("owner", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("subtype", ctypes.c_uint32),
        ("fstype", ctypes.c_char * 16),
        ("mount", ctypes.c_char * 1024),
        ("source", ctypes.c_char * 1024),
        ("reserved", ctypes.c_uint32 * 8),
    ]


def native_identity(fd: int) -> dict[str, str | int]:
    """Read current native identity from a pinned directory descriptor."""
    if platform.system() != "Darwin":
        raise OSError("native_platform_unqualified")
    libc = ctypes.CDLL(None, use_errno=True)
    fn = libc.fstatfs
    fn.argtypes = [ctypes.c_int, ctypes.POINTER(_StatFS)]
    fn.restype = ctypes.c_int
    info = _StatFS()
    if fn(fd, ctypes.byref(info)) != 0:
        raise OSError("native_identity_unavailable")
    if not hasattr(libc, "renameatx_np"):
        raise OSError("native_rename_unavailable")
    return {
        "os": platform.system(),
        "release": platform.release(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "filesystem": info.fstype.decode("ascii"),
        "flags": info.flags,
    }


# Protocol 2 includes a full native barrier after metadata publication. Evidence
# for the earlier fsync-only protocol must never authorize the amended operations.
_QUALIFICATION_PROTOCOL = 2
_OPERATIONS = frozenset(
    {"publish_new", "publish_file", "publish_directory", "admission"}
)

NativeCell = tuple[str, str, str, str, str, int, int]
ProductFact = Literal["owner_coverage", "archive", "restore", "product_flow"]
_PRODUCT_FACTS: tuple[ProductFact, ...] = (
    "owner_coverage",
    "archive",
    "restore",
    "product_flow",
)
_COMPLETE_CAPTURE_FACTS: dict[NativeCell, frozenset[ProductFact]] = {
    (
        "Darwin",
        "25.5.0",
        "arm64",
        "3.12.11",
        "apfs",
        76583040,
        2,
    ): frozenset(_PRODUCT_FACTS),
}
_NEW_REPLACEMENT_FACTS: dict[NativeCell, frozenset[ProductFact]] = {
    (
        "Darwin",
        "25.5.0",
        "arm64",
        "3.12.11",
        "apfs",
        76583040,
        2,
    ): frozenset(_PRODUCT_FACTS),
    (
        "Darwin",
        "25.5.0",
        "arm64",
        "3.12.11",
        "apfs",
        76583448,
        2,
    ): frozenset(_PRODUCT_FACTS),
}
_RELEASE_AVAILABLE = "release_capability_available"
_RELEASE_UNAVAILABLE = "release_capability_unavailable"


def _source_product_facts(
    operation: str, identity: Mapping[str, str | int]
) -> frozenset[ProductFact]:
    """Return only facts declared for one exact identity and current protocol."""
    if operation == "complete_capture":
        rows = _COMPLETE_CAPTURE_FACTS
    elif operation == "new_replacement":
        rows = _NEW_REPLACEMENT_FACTS
    else:
        return frozenset()
    try:
        cell = (
            identity["os"],
            identity["release"],
            identity["arch"],
            identity["python"],
            identity["filesystem"],
            identity["flags"],
            _QUALIFICATION_PROTOCOL,
        )
    except (KeyError, TypeError):
        return frozenset()
    if any(type(value) is not str for value in cell[:5]) or type(cell[5]) is not int:
        return frozenset()
    return rows.get(cell, frozenset())


def _source_product_gates(
    operation: str, identities: Sequence[Mapping[str, str | int]]
) -> tuple[bool, bool, bool, bool]:
    """Require every participating native identity to carry every product fact."""
    rows = tuple(identities)
    return tuple(
        bool(rows)
        and all(fact in _source_product_facts(operation, identity) for identity in rows)
        for fact in _PRODUCT_FACTS
    )


class _Identity(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    os: str
    release: str
    arch: str
    python: str
    filesystem: str
    flags: int


class _EvidenceRow(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    identity: _Identity
    operations: list[str] = Field(min_length=1)
    protocol: int
    tests: list[str] = Field(min_length=1)
    date: str
    scope: str


class _Evidence(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    schema_version: int
    evidence: list[_EvidenceRow]


def _qualified_identity(
    operation: str, identity: dict[str, str | int]
) -> tuple[bool, str]:
    try:
        raw = Path(__file__).with_name("native_qualification.json").read_text()
    except OSError:
        return False, "qualification_unavailable"
    try:
        evidence = _Evidence.model_validate_json(raw)
    except ValidationError:
        return False, "qualification_evidence_invalid"
    if evidence.schema_version != 1 or any(
        row.protocol != _QUALIFICATION_PROTOCOL
        or not set(row.operations) <= _OPERATIONS
        or len(set(row.operations)) != len(row.operations)
        for row in evidence.evidence
    ):
        return False, "qualification_evidence_invalid"
    for row in evidence.evidence:
        if row.identity.model_dump() == identity and operation in row.operations:
            return True, "qualified_native_evidence"
    return False, "operation_not_qualified"


def qualified_for(operation: str, root: Path) -> tuple[bool, str]:
    """Match installed test evidence and current identity without writing probes."""
    from .native_files import pinned_directory

    try:
        with pinned_directory(root) as fd:
            return _qualified_identity(operation, native_identity(fd))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False, "qualification_unavailable"


def _release_for_roots(
    operation: str,
    *,
    product_roots: Sequence[Path],
    admission_roots: Sequence[Path],
) -> tuple[bool, str]:
    """Compose installed facts with actual helper, Admission and native checks."""
    from .crypto import helper_capability
    from .native_files import pinned_directory
    from .restore_plan import _ancestor

    try:
        parents = tuple(dict.fromkeys(_ancestor(Path(path)) for path in product_roots))
        authorities = tuple(
            dict.fromkeys(_ancestor(Path(path)) for path in admission_roots)
        )
        identities = []
        for parent in parents:
            with pinned_directory(parent) as fd:
                identities.append(native_identity(fd))
        owner_coverage, archive, restore, product_flow = _source_product_gates(
            operation, identities
        )
        helper = helper_capability()[0]
        admission = bool(authorities) and all(
            qualified_for("admission", root)[0] for root in authorities
        )
        native_publish = bool(parents) and all(
            qualified_for(primitive, root)[0]
            for root in parents
            for primitive in ("publish_new", "publish_file", "publish_directory")
        )
        available = release_capability(
            helper=helper,
            owner_coverage=owner_coverage,
            admission=admission,
            archive=archive,
            native_publish=native_publish,
            restore=restore,
            product_flow=product_flow,
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError):
        available = False
    return (True, _RELEASE_AVAILABLE) if available else (False, _RELEASE_UNAVAILABLE)


def complete_capture_capability(
    *, staging_parent: Path, destination: Path, control_root: Path
) -> tuple[bool, str]:
    """Report Complete capture availability for its actual local write roots."""
    from . import bootstrap
    from .service_storage import work_root

    return _release_for_roots(
        "complete_capture",
        product_roots=(
            Path(staging_parent),
            Path(destination),
            Path(control_root),
            work_root(Path(control_root)),
            bootstrap.default_bootstrap_root().parent,
            bootstrap.default_bootstrap_root(),
            bootstrap.default_bootstrap_root() / "admission",
        ),
        admission_roots=(
            bootstrap.default_bootstrap_root().parent,
            bootstrap.default_bootstrap_root(),
            bootstrap.default_bootstrap_root() / "admission",
        ),
    )


def replacement_capability(plan, *, control_root: Path) -> tuple[bool, str]:
    """Report new replacement availability for one actual reviewed plan."""
    from . import bootstrap
    from .restore_plan import RestorePlan
    from .service_storage import work_root

    if (
        type(plan) is not RestorePlan
        or plan.mode != "replace"
        or plan.target is None
        or plan.local_snapshot is not None
        and plan.local_snapshot.control_root != Path(control_root)
    ):
        return False, _RELEASE_UNAVAILABLE
    paths = tuple(
        path
        for _, path in (
            *plan.destinations,
            *plan.restore,
            *plan.retire,
        )
    )
    return _release_for_roots(
        "new_replacement",
        product_roots=(
            Path(control_root),
            work_root(Path(control_root)),
            bootstrap.default_bootstrap_root().parent,
            bootstrap.default_bootstrap_root(),
            bootstrap.default_bootstrap_root() / "admission",
            *paths,
        ),
        admission_roots=(
            bootstrap.default_bootstrap_root().parent,
            bootstrap.default_bootstrap_root(),
            bootstrap.default_bootstrap_root() / "admission",
        ),
    )
