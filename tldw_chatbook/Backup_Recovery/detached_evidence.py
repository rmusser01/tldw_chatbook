"""Bounded copied recovery evidence inspection, with no native admission (ADR-256).

Only acquired local members are opened. Absolute locators in records are syntax
and historical evidence; none are resolved, stat-ed, followed, or made authoritative.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
from pathlib import Path

from tldw_chatbook.Utils.platform_files import os

from .activation import _Approval, _Requirement
from .archive_reader import _identity, _manifest, _regular
from .bootstrap import _activation_witness, _paths, _strings
from .control_records import _DIRECTORY_CREATION_INTENT, _Profile
from .journal import MAX_EVENTS, _artifact_digest, parse_event_records
from .limits import RECOVERY_RECORD_BYTES, ArchiveLimits
from .native_files import pinned_directory
from .profile_catalog import _Entry

MAX_EVIDENCE_BYTES = 256 * 1024**2
MAX_MEMBERS = 100_000
MAX_DEPTH = 128
MAX_JOURNAL_EVENTS = 4096
_BOOTSTRAP_RELATIVE = Path("home/.config/tldw_cli/recovery-bootstrap")
_CONTROL_RELATIVE = Path("home/.config/tldw_cli/recovery/control")
_HEX = r"[0-9a-f]{64}"
_JOURNAL_COMPANIONS = {"journal.lock", "verified-manifest.json", "restore-plan.json"}
_PENDING = "detached_evidence_pending"
_INVALID = "detached_evidence_invalid"
_UNSETTLED = "detached_evidence_unsettled"
_UNSUPPORTED = "detached_evidence_unsupported"
_LIMIT = "detached_evidence_limit"


def _key(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _creation_pending(bootstrap: Path, control: Path, floor: Path) -> bool:
    """Check only parents where installed _ensure can create its intent.

    Enrollment creates bootstrap ancestors; service storage creates control's
    parent and its sibling control-work directory. Recorded roots are never used.
    """
    parents = {
        parent
        for target in (bootstrap, control.with_name(control.name + "-work"))
        for parent in target.parents
        if parent == floor or floor in parent.parents
    }
    for directory in sorted(parents, key=lambda path: len(path.parts)):
        try:
            with pinned_directory(directory) as parent:
                os.stat(
                    _DIRECTORY_CREATION_INTENT, dir_fd=parent, follow_symlinks=False
                )
                return True
        except FileNotFoundError:
            continue
    return False


def _census(root: Path) -> dict[Path, list[str]]:
    """Enumerate acquired members only, rejecting unsafe or excessive input."""
    found = {}
    pending = [(root, 0)]
    count = 0
    while pending:
        directory, depth = pending.pop()
        if depth > MAX_DEPTH:
            raise ValueError(_LIMIT)
        with pinned_directory(directory) as parent:
            names = sorted(os.listdir(parent))
            count += len(names)
            if count > MAX_MEMBERS:
                raise ValueError(_LIMIT)
            found[directory] = names
            for name in names:
                info = os.stat(name, dir_fd=parent, follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    pending.append((directory / name, depth + 1))
                elif not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError(_INVALID)
    return found


class _Reader:
    def __init__(self):
        self.remaining = MAX_EVIDENCE_BYTES

    def raw(self, directory: Path, name: str) -> bytes:
        with _regular(directory / name) as stream:
            before = os.fstat(stream.fileno())
            if (
                before.st_nlink != 1
                or before.st_uid != os.geteuid()
                or before.st_mode & 0o077
            ):
                raise ValueError(_INVALID)
            limit = min(self.remaining, RECOVERY_RECORD_BYTES)
            if before.st_size > limit:
                raise ValueError(_LIMIT)
            raw = stream.read(limit + 1)
            if len(raw) > limit:
                raise ValueError(_LIMIT)
            if len(raw) != before.st_size or _identity(before) != _identity(
                os.fstat(stream.fileno())
            ):
                raise ValueError(_INVALID)
            self.remaining -= len(raw)
        return raw

    def read(self, directory: Path, name: str) -> dict:
        record = _json(self.raw(directory, name))
        if type(record.get("version")) is not int or record["version"] != 1:
            raise ValueError(_INVALID)
        return record


def _json(raw: bytes) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(_INVALID)
            result[key] = value
        return result

    def constant(_):
        raise ValueError(_INVALID)

    value = json.loads(raw, object_pairs_hook=unique, parse_constant=constant)
    if not isinstance(value, dict):
        raise TypeError(_INVALID)
    return value


def _ancestry(value: dict) -> None:
    if (
        set(value) != {"version", "root_inode", "root_device", "root_path"}
        or type(value["root_inode"]) is not int
        or value["root_inode"] < 0
        or value["root_device"] is not None
        and (type(value["root_device"]) is not int or value["root_device"] < 0)
        or not _paths([value["root_path"]])
    ):
        raise ValueError(_INVALID)


def _registry(value: dict) -> None:
    if set(value) != {"version", "entries"} or not isinstance(value["entries"], dict):
        raise ValueError(_INVALID)
    if len(value["entries"]) > MAX_MEMBERS:
        raise ValueError(_LIMIT)
    for name, entry in value["entries"].items():
        if (
            not name
            or len(name) > 256
            or "\0" in name
            or not isinstance(entry, dict)
            or set(entry) != {"roots", "historical", "pending", "proposed"}
            or not _paths(entry["roots"])
            or not isinstance(entry["historical"], list)
            or any(not isinstance(item, str) for item in entry["historical"])
            or not isinstance(entry["proposed"], list)
        ):
            raise ValueError(_INVALID)
        if entry["pending"] is not None or entry["proposed"]:
            raise ValueError(_PENDING)


def _bootstrap(
    directory: Path, members: dict[Path, list[str]], reader: _Reader
) -> list[tuple[str, dict]]:
    names = members[directory]
    if any(
        name.startswith("pending-")
        and name.endswith(".json")
        or name.startswith("activation-update-")
        for name in names
    ):
        raise ValueError(_PENDING)
    if not {"ancestry-settled.json", "unbound-owner", "admission"} <= set(names):
        raise ValueError(_UNSETTLED)
    profiles, associations = {}, {}
    for name in names:
        if name in {"admission", "unbound-owner"}:
            if (directory / name in members) != (name == "admission"):
                raise ValueError(_INVALID)
            continue
        if name == "ancestry-settled.json":
            _ancestry(reader.read(directory, name))
        elif re.fullmatch("profile-" + _HEX + r"\.json", name):
            value = reader.read(directory, name)
            profile = _Profile.model_validate(value)
            if (
                not _paths([profile.selector])
                or not _paths(profile.roots)
                or not _strings(profile.namespaces)
                or name != "profile-" + _key(profile.selector) + ".json"
                or profile.activation is not None
                and profile.activation["namespaces"] != profile.namespaces
            ):
                raise ValueError(_INVALID)
            profiles[profile.selector] = profile.activation
        elif re.fullmatch("activation-" + _HEX + r"\.json", name):
            value = reader.read(directory, name)
            if (
                set(value) != {"version", "selector", "activation"}
                or not _paths([value["selector"]])
                or name != "activation-" + _key(value["selector"]) + ".json"
            ):
                raise ValueError(_INVALID)
            _activation_witness(value["activation"])
            associations[value["selector"]] = value["activation"]
        else:
            raise ValueError(_UNSUPPORTED)
    admission = directory / "admission"
    if admission not in members or not {"registry.json", "registry.lock"} <= set(
        members[admission]
    ):
        raise ValueError(_UNSETTLED)
    for name in members[admission]:
        if (
            name == "registry.pending.json"
            or name.startswith("registry-")
            and name.endswith(".tmp")
        ):
            raise ValueError(_PENDING)
        if admission / name in members:
            raise ValueError(_INVALID)
        if name == "registry.json":
            _registry(reader.read(admission, name))
        elif name != "registry.lock" and not re.fullmatch(
            _HEX + r"\.(gate|lease|incompatible)", name
        ):
            raise ValueError(_UNSUPPORTED)
    witnesses = {
        selector: value for selector, value in profiles.items() if value is not None
    }
    if witnesses != associations:
        raise ValueError(_UNSETTLED)
    return list(witnesses.items())


def _journal(directory: Path, members: dict[Path, list[str]], reader: _Reader):
    names = members[directory]
    if "journal.lock" not in names:
        raise ValueError(_UNSETTLED)
    if any(directory / name in members for name in names):
        raise ValueError(_INVALID)
    companions = set(names) & _JOURNAL_COMPANIONS
    names = sorted(set(names) - _JOURNAL_COMPANIONS)
    if not names:
        raise ValueError(_UNSETTLED)
    if len(names) > min(MAX_EVENTS, MAX_JOURNAL_EVENTS):
        raise ValueError(_LIMIT)
    first = reader.read(directory, names[0])
    operation = first.get("operation_id")
    if (
        not isinstance(operation, str)
        or not 0 < len(operation) <= 256
        or "\0" in operation
        or directory.name != "operation-" + _key(operation)
    ):
        raise ValueError(_INVALID)
    rows = parse_event_records(
        operation,
        (
            (name, first if index == 0 else reader.read(directory, name))
            for index, name in enumerate(names)
        ),
        historical=True,
    )
    if rows[-1].event not in {"committed", "rolled_back", "prepublication_aborted"}:
        raise ValueError(_UNSETTLED)
    _journal_companions(directory, companions, rows, reader)
    return rows


def _journal_companions(
    directory: Path, names: set[str], rows, reader: _Reader
) -> None:
    candidate = rows[0].evidence if rows[0].event == "candidate_staged" else None
    if "verified-manifest.json" in names:
        raw = reader.raw(directory, "verified-manifest.json")
        _json(raw)
        _manifest(raw, ArchiveLimits(), encrypted=True)
        if (
            candidate is None
            or hashlib.sha256(raw).hexdigest() != candidate["manifest_digest"]
        ):
            raise ValueError(_INVALID)
    plan_receipt = candidate.get("local_plan") if candidate else None
    if (
        plan_receipt is not None
        and not {"restore-plan.json", "verified-manifest.json"} <= names
    ):
        raise ValueError(_UNSETTLED)
    if "restore-plan.json" in names:
        from .plan_records import _PLAN, _encode_plan
        from .publication import _plan_digest

        raw = reader.raw(directory, "restore-plan.json")
        value = _json(raw)
        plan = _PLAN.validate_json(raw, strict=True)
        if (
            plan_receipt is None
            or plan_receipt["sha256"]
            != _artifact_digest(
                [("", "file", len(raw), hashlib.sha256(raw).hexdigest())]
            )
            or plan_receipt["size"] != len(raw)
            or json.loads(_encode_plan(plan)) != value
            or _plan_digest(plan) != candidate["plan_digest"]
        ):
            raise ValueError(_INVALID)


def _activation_generation(directory: Path, names: list[str], reader: _Reader) -> None:
    if "required.json" not in names:
        raise ValueError(_UNSETTLED)
    required = _Requirement.model_validate(reader.read(directory, "required.json"))
    if (
        not _strings(required.owners)
        or required.owners != sorted(required.owners)
        or any(len(owner) > 256 for owner in required.owners)
        or not 0 < len(required.generation) <= 256
        or "\0" in required.generation
        or directory.name != "generation-" + _key(required.generation)
    ):
        raise ValueError(_INVALID)
    for name in names:
        if name == "required.json":
            continue
        if re.fullmatch("approved-" + _HEX + r"\.json", name):
            approval = _Approval.model_validate(reader.read(directory, name))
            if (
                approval.generation != required.generation
                or approval.owner not in required.owners
                or name != "approved-" + _key(approval.owner) + ".json"
            ):
                raise ValueError(_INVALID)
        elif re.fullmatch(r"opened-[0-9a-f]{32}\.json", name):
            from .profile_open import OpenedReceipt

            opened = OpenedReceipt.model_validate(reader.read(directory, name))
            if (
                opened.checks_version != 1
                or opened.generation != required.generation
                or name != "opened-" + opened.attempt + ".json"
                or not _paths([opened.config, opened.data])
            ):
                raise ValueError(_INVALID)
        else:
            raise ValueError(_UNSUPPORTED)


def _catalog(directory: Path, names: list[str], reader: _Reader) -> None:
    for name in names:
        if re.fullmatch(_HEX + r"\.json", name) is None:
            raise ValueError(_UNSUPPORTED)
        entry = _Entry.model_validate(reader.read(directory, name))
        if (
            not 0 < len(entry.profile_id) <= 256
            or "\0" in entry.profile_id
            or name != _key(entry.profile_id) + ".json"
            or not _paths([entry.config, entry.data])
        ):
            raise ValueError(_INVALID)


def _control(directory: Path, members: dict[Path, list[str]], reader: _Reader) -> None:
    for name in members[directory]:
        child = directory / name
        if name == "service.json":
            if reader.read(directory, name) != {
                "version": 1,
                "kind": "backup_recovery_service",
            }:
                raise ValueError(_INVALID)
        elif name == "activation" and child in members:
            for generation in members[child]:
                target = child / generation
                if (
                    not re.fullmatch("generation-" + _HEX, generation)
                    or target not in members
                ):
                    raise ValueError(_UNSUPPORTED)
                _activation_generation(target, members[target], reader)
        elif name == "profile-catalog" and child in members:
            _catalog(child, members[child], reader)
        elif re.fullmatch("operation-" + _HEX, name) and child in members:
            continue  # Journal children are checked separately, never as live work.
        elif name.startswith("isolated-") and child in members:
            operation = name.removeprefix("isolated-")
            if (
                not operation
                or directory / ("operation-" + _key(operation)) not in members
            ):
                raise ValueError(_UNSETTLED)
        else:
            raise ValueError(_UNSUPPORTED)


def _validate_members(
    members: dict[Path, list[str]],
    *,
    bootstrap_roots: tuple[Path, ...],
    control_roots: tuple[Path, ...],
    require_journal: bool = False,
) -> tuple[str, ...]:
    """Share pathless proof validation across complete and explicitly scoped copies."""
    issues = set()
    reader = _Reader()
    journals, witnesses = {}, []
    try:
        for directory in (*bootstrap_roots, *control_roots):
            if directory not in members:
                if directory.name in members.get(directory.parent, []):
                    issues.add(_INVALID)
                continue
            try:
                if directory in bootstrap_roots:
                    witnesses.extend(_bootstrap(directory, members, reader))
                else:
                    _control(directory, members, reader)
                    for name in members[directory]:
                        if re.fullmatch("operation-" + _HEX, name):
                            rows = _journal(directory / name, members, reader)
                            if rows[0].operation_id in journals:
                                raise ValueError(_INVALID)
                            journals[rows[0].operation_id] = rows
            except (
                ValueError,
                OSError,
                TypeError,
                KeyError,
                StopIteration,
                RecursionError,
            ) as error:
                issues.add(
                    str(error)
                    if str(error)
                    in {_PENDING, _INVALID, _UNSETTLED, _UNSUPPORTED, _LIMIT}
                    else _INVALID
                )
        for selector, witness in witnesses:
            rows = journals.get(witness["operation_id"])
            if rows is None or rows[-1].event == "prepublication_aborted":
                issues.add(_UNSETTLED)
                continue
            activation = next(
                (
                    row
                    for row in reversed(rows)
                    if row.event
                    in {"activation_recorded", "rollback_activation_recorded"}
                ),
                None,
            )
            if (
                activation is None
                or rows[-1].evidence["generation"] != witness["generation"]
                or selector not in activation.evidence["selectors"]
                or activation.evidence["owners"] != witness["owners"]
            ):
                issues.add(_UNSETTLED)
        if require_journal and not journals:
            issues.add(_UNSETTLED)
    except (ValueError, OSError, TypeError, RecursionError) as error:
        issues.add(str(error) if str(error) == _LIMIT else _INVALID)
    return tuple(sorted(issues))


def validate_copied_evidence(root: Path) -> tuple[str, ...]:
    """Return sanitized blockers for recovery evidence in an acquired Data copy.

    Only the installed canonical bootstrap/control namespaces select recovery
    grammar. Names in user payload or retained operation staging do not.

    Args:
        root: Absolute private acquired source, never the original live profile.

    Returns:
        Empty tuple for settled evidence, otherwise stable reason codes. Acceptance
        proves only that copied records describe a recognized terminal sequence;
        it grants no native admission, historical authenticity, or activation.
    """
    try:
        bootstrap, control = root / _BOOTSTRAP_RELATIVE, root / _CONTROL_RELATIVE
        if _creation_pending(bootstrap, control, root):
            return (_PENDING,)
        return _validate_members(
            _census(root),
            bootstrap_roots=(bootstrap,),
            control_roots=(control,),
        )
    except (ValueError, OSError, TypeError, RecursionError) as error:
        return (_LIMIT if str(error) == _LIMIT else _INVALID,)


def validate_copied_controls(
    bootstrap_root: Path, control_root: Path
) -> tuple[str, ...]:
    """Inspect a retained attempt's explicit copied controls, without payload IO.

    Args:
        bootstrap_root: Copied enrollment root selected by the installed caller.
        control_root: Copied publication-control root selected by that caller.

    Returns:
        Sanitized blockers, or an empty tuple only when enrollment is settled and
        at least one copied publication journal has complete terminal evidence.
        Recorded paths remain inert; sibling work/source/profiles are not read.
    """
    members = {}
    try:
        floor = Path(os.path.commonpath((bootstrap_root, control_root)))
        if _creation_pending(bootstrap_root, control_root, floor):
            return (_PENDING,)
        for root in (bootstrap_root, control_root):
            acquired = _census(root)
            if members.keys() & acquired.keys():
                raise ValueError(_INVALID)
            members.update(acquired)
        if sum(len(names) for names in members.values()) > MAX_MEMBERS:
            raise ValueError(_LIMIT)
        return _validate_members(
            members,
            bootstrap_roots=(bootstrap_root,),
            control_roots=(control_root,),
            require_journal=True,
        )
    except FileNotFoundError:
        return (_UNSETTLED,)
    except (ValueError, OSError, TypeError, RecursionError) as error:
        return (_LIMIT if str(error) == _LIMIT else _INVALID,)
