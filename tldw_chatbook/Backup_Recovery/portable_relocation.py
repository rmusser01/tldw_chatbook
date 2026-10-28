"""Reviewed detached portable import into independent recovery authority."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from uuid import uuid4

from tldw_chatbook.Utils.path_validation import validate_recovery_relative_path
from tldw_chatbook.Utils.platform_files import fcntl, os

from .bootstrap import _read
from .native_files import (
    create_private_directory,
    create_private_file,
    pinned_directory,
)

WORK_NAME = ".portable-relocation"
TRANSIENT_ROOTS = (
    WORK_NAME,
    "cache",
    "tmp",
    "wezterm",
    "home/.cache",
    "home/AppData/Local",
    "home/.local/share/wezterm",
)
_ATTEMPT = re.compile(r"attempt-[0-9a-f]{32}")
_SELECTION = re.compile(r"selection-[0-9]{6}\.json")


@dataclass(frozen=True)
class PortableSelection:
    version: int
    location: str
    attempt: str
    profile_id: str
    config: str
    data: str
    source_digest: str


def _record(root: Path, name: str) -> dict:
    with pinned_directory(root) as parent:
        return _read(parent, name)


def _write(root: Path, name: str, value: dict) -> None:
    with create_private_file(root / name) as fd:
        view = memoryview(json.dumps(value, sort_keys=True).encode())
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("portable_record_incomplete")
            view = view[written:]


def _directory(path: Path) -> None:
    try:
        create_private_directory(path)
    except FileExistsError:
        with pinned_directory(path) as parent:
            info = os.fstat(parent)
            if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise ValueError("portable_work_not_private")


def _inside(data: Path, relative: str) -> Path:
    validate_recovery_relative_path(relative)
    path = data / relative
    if not relative or path == data or not path.is_relative_to(data):
        raise ValueError("portable_selection_invalid")
    return path


def _checked_selection(data: Path, raw: dict) -> PortableSelection:
    selection = PortableSelection(**raw)
    base = Path(WORK_NAME) / selection.attempt
    if (
        type(selection.version) is not int
        or selection.version != 1
        or not _ATTEMPT.fullmatch(selection.attempt)
        or not re.fullmatch(r"[0-9a-f]{32}", selection.profile_id)
        or not re.fullmatch(r"[0-9a-f]{64}", selection.source_digest)
        or not Path(selection.location).is_absolute()
        or not _inside(data, selection.config).is_relative_to(data / base / "profiles")
        or not _inside(data, selection.data).is_relative_to(data / base / "profiles")
    ):
        raise ValueError("portable_selection_invalid")
    return selection


def read_selection(data: Path) -> PortableSelection | None:
    """Read a confined hint only; no record grants launch or activation authority."""
    work = data / WORK_NAME
    opened = False
    try:
        with pinned_directory(work) as parent:
            opened = True
            names = os.listdir(parent)
            if len(names) > 4096:
                raise ValueError("portable_selection_invalid")
            selections = sorted(name for name in names if _SELECTION.fullmatch(name))
            if not selections:
                return None
            raw = _read(parent, selections[-1])
        return _checked_selection(data, raw)
    except FileNotFoundError:
        if opened:
            raise ValueError("portable_selection_invalid") from None
        return None
    except (TypeError, OSError, ValueError) as error:
        raise ValueError("portable_selection_invalid") from error


@contextmanager
def relocation_lock(data: Path):
    """Serialize review/publication/selection without adopting copied authority."""
    work = data / WORK_NAME
    _directory(work)
    with pinned_directory(work) as parent:
        try:
            with create_private_file(work / "relocation.lock"):
                pass
        except FileExistsError:
            pass
        fd = os.open(
            "relocation.lock", os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_mode & 0o077
                or info.st_uid != os.geteuid()
            ):
                raise ValueError("portable_lock_invalid")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise ValueError("portable_relocation_busy") from error
            named = os.stat("relocation.lock", dir_fd=parent, follow_symlinks=False)
            if (named.st_dev, named.st_ino) != (info.st_dev, info.st_ino):
                raise ValueError("portable_lock_changed")
            yield work
        finally:
            os.close(fd)


def pending_attempt(data: Path) -> Path | None:
    """Locate a local interrupted publication; never blindly allocate another."""
    work = data / WORK_NAME
    with pinned_directory(work) as parent:
        names = os.listdir(parent)
    if len(names) > 4096:
        raise ValueError("portable_relocation_limit")
    pending = []
    selected = read_selection(data)
    for name in sorted(names):
        if not _ATTEMPT.fullmatch(name):
            continue
        attempt = work / name
        with pinned_directory(attempt) as parent:
            children = os.listdir(parent)
        if "publication-started.json" in children:
            try:
                started = _record(attempt, "publication-started.json")
                if (
                    set(started) != {"version", "location", "source_digest"}
                    or type(started["version"]) is not int
                    or started["version"] != 1
                    or not re.fullmatch(r"[0-9a-f]{64}", started["source_digest"])
                    or not Path(started["location"]).is_absolute()
                ):
                    raise ValueError("invalid_started")
                # A moved publication is historical only after pathless terminal
                # proof and its completed outer selection; labels never suffice.
                if started["location"] != str(data.parent):
                    committed = _record(attempt, "committed.json")
                    historical = _checked_selection(data, committed)
                    if (
                        historical.attempt != name
                        or historical.location != started["location"]
                        or historical.source_digest != started["source_digest"]
                    ):
                        raise ValueError("portable_commit_changed")
                    with pinned_directory(work) as records:
                        choices = [
                            row
                            for row in os.listdir(records)
                            if _SELECTION.fullmatch(row)
                        ]
                        if not any(_read(records, row) == committed for row in choices):
                            raise ValueError("portable_selection_incomplete")
                    from .detached_evidence import validate_copied_controls

                    if validate_copied_controls(
                        attempt
                        / "home"
                        / ".config"
                        / "tldw_cli"
                        / "recovery-bootstrap",
                        attempt / "control",
                    ):
                        raise ValueError("portable_copied_publication_unsettled")
                if started["location"] == str(data.parent) and (
                    selected is None or selected.attempt != name
                ):
                    pending.append(attempt)
            except (OSError, ValueError, TypeError):
                raise ValueError("portable_relocation_recovery_required") from None
    if len(pending) > 1:
        raise ValueError("portable_relocation_recovery_required")
    return pending[0] if pending else None


def host_environment(data: Path, attempt: str) -> None:
    """Select only a fresh confined host before config/owner modules are imported."""
    if "tldw_chatbook.config" in sys.modules or not _ATTEMPT.fullmatch(attempt):
        raise ValueError("portable_relocation_requires_fresh_process")
    from .isolated_restore import _launch_environment

    filtered = _launch_environment()
    os.environ.clear()
    os.environ.update(filtered)
    home = data / WORK_NAME / attempt / "home"
    temporary = data / WORK_NAME / attempt / "tmp"
    _directory(temporary)
    os.environ.update(
        HOME=str(home),
        USERPROFILE=str(home),
        TLDW_CONFIG_PATH=str(home / ".config" / "tldw_cli" / "config.toml"),
        XDG_CONFIG_HOME=str(home / ".config"),
        XDG_DATA_HOME=str(home / ".local" / "share"),
        APPDATA=str(home / "AppData" / "Roaming"),
        LOCALAPPDATA=str(home / "AppData" / "Local"),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
        TEMP=str(temporary),
        TMP=str(temporary),
        TMPDIR=str(temporary),
        CHATBOOK_WEZTERM_HOME=str(data / "home"),
        CHATBOOK_WEZTERM_DATA_DIR=str(data / "wezterm" / "data"),
        CHATBOOK_WEZTERM_RUNTIME_DIR=str(data / "wezterm" / "runtime"),
    )
    import tempfile

    tempfile.tempdir = None
    os.environ["HOMEDRIVE"], os.environ["HOMEPATH"] = os.path.splitdrive(str(home))


def verified_selection(data: Path) -> PortableSelection | None:
    """Validate the hint through the committed catalog, journal and activation pair."""
    selection = read_selection(data)
    if selection is None:
        return None
    if selection.location != str(data.parent):
        raise ValueError("portable_profile_moved")
    host_environment(data, selection.attempt)
    from .isolated_restore import _launch_state

    entry, _ = _launch_state(
        selection.profile_id, data / WORK_NAME / selection.attempt / "control"
    )
    if (entry.config, entry.data) != (
        str(_inside(data, selection.config)),
        str(_inside(data, selection.data)),
    ):
        raise ValueError("portable_selection_changed")
    return selection


def _select(data: Path, attempt: Path, profile: str, digest: str) -> PortableSelection:
    from .isolated_restore import _launch_state

    entry, _ = _launch_state(profile, attempt / "control")
    selection = PortableSelection(
        1,
        str(data.parent),
        attempt.name,
        profile,
        Path(entry.config).relative_to(data).as_posix(),
        Path(entry.data).relative_to(data).as_posix(),
        digest,
    )
    from dataclasses import asdict

    value = asdict(selection)
    committed = attempt / "committed.json"
    if committed.exists():
        if _record(attempt, committed.name) != value:
            raise ValueError("portable_commit_changed")
    else:
        _write(attempt, committed.name, value)
    work = data / WORK_NAME
    with pinned_directory(work) as parent:
        numbers = [
            int(name[10:16])
            for name in os.listdir(parent)
            if _SELECTION.fullmatch(name)
        ]
    _write(work, f"selection-{max(numbers, default=0) + 1:06d}.json", value)
    return selection


def _resume(data: Path, attempt: Path) -> PortableSelection:
    """Complete the same local restore operation, with its own recovery executor."""
    host_environment(data, attempt.name)
    from .recovery_service import RecoveryService

    service = RecoveryService(attempt / "control")
    try:
        pending = service.pending_operations()
        if len(pending) > 1:
            raise ValueError("portable_relocation_recovery_required")
        if pending:
            operation = pending[0]["operation_id"]
            state = service.status(operation)
            if "finish" not in state["actions"]:
                raise ValueError("portable_relocation_recovery_required")
            job = service.start_recovery(operation, action="finish")
            if service.wait(job)["state"] != "succeeded":
                raise ValueError("portable_relocation_recovery_required")
        else:
            _resume_staged(attempt)
        # Catalog entries are hints; _select repeats _launch_state for each result.
        catalog = attempt / "control" / "profile-catalog"
        with pinned_directory(catalog) as parent:
            names = os.listdir(parent)
            if len(names) != 1:
                raise ValueError("portable_relocation_recovery_required")
            profile = _read(parent, names[0])["profile_id"]
        started = _record(attempt, "publication-started.json")
        return _select(data, attempt, profile, started["source_digest"])
    except (OSError, KeyError, ValueError) as error:
        raise ValueError("portable_relocation_recovery_required") from error
    finally:
        service.close()


def _resume_staged(attempt: Path) -> None:
    """Resume a receipt-bound local stage that predates the native pending pair."""
    from . import bootstrap
    from .control_records import _existing_admission_authority
    from .destinations import check_isolated_parents
    from .isolated_restore import _finish_isolated
    from .journal import _CandidateReceipt, _IsolatedProfile
    from .plan_records import load_plan
    from .publication import _descriptor
    from .recovery_copies import _journal

    control = attempt / "control"
    with pinned_directory(control) as parent:
        names = os.listdir(parent)
    operations = [
        name for name in names if re.fullmatch(r"operation-[0-9a-f]{64}", name)
    ]
    if len(names) > 4096 or len(operations) != 1:
        raise ValueError("portable_relocation_recovery_required")
    first = _record(control / operations[0], "000000.json")
    operation_id = first.get("operation_id")
    if not isinstance(operation_id, str) or not re.fullmatch(
        r"[0-9a-f]{32}", operation_id
    ):
        raise ValueError("portable_relocation_recovery_required")
    journal = _journal(control, operation_id)
    if journal.root.name != operations[0]:
        raise ValueError("portable_relocation_recovery_required")
    with journal._locked(exclusive=False) as parent:
        rows = journal._records(parent)
    if rows[-1].event == "committed":
        return  # _select independently verifies the complete launch association.
    if len(rows) != 1 or rows[0].event != "candidate_staged":
        raise ValueError("portable_relocation_recovery_required")
    receipt = _CandidateReceipt.model_validate(rows[0].evidence)
    candidate = Path(receipt.stage.path)
    if candidate.parent != control / (
        "isolated-" + operation_id
    ) / "staging" or not re.fullmatch(r"restore-[0-9a-f]{32}", candidate.name):
        raise ValueError("portable_stage_location_changed")
    plan = load_plan(journal)
    targets = (*plan.restore, *plan.containers, *plan.destinations, *plan.selectors)
    if (
        plan.mode != "isolated"
        or plan.target is not None
        or plan.retire
        or any(not path.is_relative_to(attempt / "profiles") for _, path in targets)
    ):
        raise ValueError("portable_stage_plan_changed")
    document = _descriptor(candidate, plan)
    profiles = tuple(
        _IsolatedProfile.model_validate(row) for row in document["isolated_profiles"]
    )
    if len(profiles) != 1 or not Path(profiles[0].config).is_relative_to(
        attempt / "profiles"
    ):
        raise ValueError("portable_stage_profile_changed")
    ancestors = check_isolated_parents(plan, control)
    authority = _existing_admission_authority(bootstrap.default_bootstrap_root())
    namespace = "isolated." + operation_id
    authority.register(namespace, tuple(ancestors))
    _finish_isolated(
        candidate,
        plan,
        journal,
        (namespace,),
        operation_id,
        Event(),
        initial_selectors=(Path(profiles[0].config),),
    )


def acquire_selected_source(
    data: Path, selection: PortableSelection, destination: Path
):
    """Build a staged view of a moved selected generation, preserving old bytes.

    Only confined copied members are acquired. The old location is written to a
    new inert staging marker for lexical path mapping, never used for file IO.
    Retained earlier generations are not recursively imported.
    """
    from .detached_source import acquire_source, review_source

    base = data / WORK_NAME / selection.attempt
    config = _inside(data, selection.config)
    payload = _inside(data, selection.data)
    if (
        not _ATTEMPT.fullmatch(selection.attempt)
        or not config.is_relative_to(base / "profiles")
        or not payload.is_relative_to(base / "profiles")
        or config.name != "config.toml"
        or config.parent == payload
        or config.parent in payload.parents
        or payload in config.parents
    ):
        raise ValueError("portable_selection_invalid")
    canonical = destination / "home" / ".config" / "tldw_cli"
    roots = (
        (config.parent, canonical),
        (payload, destination / selection.data),
        (
            base / "home" / ".config" / "tldw_cli" / "recovery-bootstrap",
            canonical / "recovery-bootstrap",
        ),
        (base / "control", canonical / "recovery" / "control"),
    )
    reviews = tuple(review_source(root) for root, _ in roots)
    create_private_directory(destination)
    for review, (_, target) in zip(reviews, roots, strict=True):
        for parent in reversed(target.parent.parents):
            if parent != destination and parent.is_relative_to(destination):
                _directory(parent)
        if target.parent != destination:
            _directory(target.parent)
        acquire_source(review, target)
    _write(
        destination,
        "portable-location.json",
        {"version": 1, "root": selection.location},
    )
    return destination, reviews


def _produce(source: Path, work: Path) -> tuple[Path, tuple[str, ...]]:
    """Retire owner validation before the parent obtains publication gates."""
    reply = work.parent / "producer-result.json"
    completed = subprocess.run(  # nosec B603 - fixed installed module and argv
        [
            sys.executable,
            "-P",
            "-m",
            "tldw_chatbook.Backup_Recovery.detached_worker",
            str(source),
            str(work),
            str(reply),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    raw = _record(reply.parent, reply.name)
    if raw.get("status") == "refused" and set(raw) == {"version", "status", "issue"}:
        issue = raw["issue"]
        raise ValueError(
            issue
            if isinstance(issue, str)
            and re.fullmatch(r"[a-z][a-z0-9_.:-]{0,255}", issue)
            else "detached_import_failed"
        )
    if (
        completed.returncode != 0
        or type(raw.get("version")) is not int
        or raw["version"] != 1
        or set(raw) != {"version", "status", "archive", "review_lines"}
        or raw["status"] != "succeeded"
        or raw["archive"] != work.name + ".tldw-backup.zip"
        or "tldw_chatbook.config" in sys.modules
    ):
        raise ValueError("detached_worker_incomplete")
    from .archive_models import RecoveryReport

    report = RecoveryReport(version=1, lines=tuple(raw["review_lines"]))
    return work.parent / raw["archive"], report.lines


def relocate(data: Path, confirm) -> PortableSelection | None:
    """Review stable staged bytes and publish only after explicit confirmation.

    Args:
        data: Current bundle Data directory.
        confirm: Callable receiving inert review lines and digest; returns bool.
    """
    from .detached_source import acquire_source, recheck_source, review_source

    with relocation_lock(data):
        pending = pending_attempt(data)
        if pending is not None:
            started = _record(pending, "publication-started.json")
            lines = (
                "Resume the existing previously reviewed sealed snapshot.",
                "Fresh destination: " + str(pending / "profiles" / "restored"),
                "The same receipt-bound operation is revalidated; no new restore is allocated.",
                "Managed credentials need setup again and activation reviews remain required.",
                "The retained copied source stays unchanged.",
            )
            if not confirm(lines, started["source_digest"]):
                return None
            return _resume(data, pending)
        prior_selection = read_selection(data)
        attempt = data / WORK_NAME / ("attempt-" + uuid4().hex)
        _directory(attempt)
        _directory(attempt / "home")
        _directory(attempt / "work")
        _directory(attempt / "profiles")
        host_environment(data, attempt.name)
        if prior_selection is None:
            review = review_source(data, excluded_roots=TRANSIENT_ROOTS)
            acquired = acquire_source(review, attempt / "work" / "source")
            reviews = (review,)
            digest = review.digest
        else:
            acquired, reviews = acquire_selected_source(
                data, prior_selection, attempt / "work" / "source"
            )
            from dataclasses import asdict

            digest = hashlib.sha256(
                json.dumps(
                    {
                        "selection": asdict(prior_selection),
                        "reviews": [row.digest for row in reviews],
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()

        def recheck():
            if read_selection(data) != prior_selection:
                raise ValueError("detached_source_changed")
            for row in reviews:
                recheck_source(row)

        archive_path, review_lines = _produce(acquired, attempt / "work" / "archive")
        from .archive_reader import acquire, verify_sealed
        from .destinations import resolve_destinations
        from .isolated_restore import restore_isolated
        from .limits import ArchiveLimits

        sealed = acquire(
            archive_path,
            attempt / "work" / "inspection",
            ArchiveLimits(),
            None,
            Event(),
        )
        doc = verify_sealed(sealed)
        if len(doc.profile_ids) != 1:
            raise ValueError("portable_single_profile_required")
        profile = doc.profile_ids[0]
        plan = resolve_destinations(
            sealed,
            mode="isolated",
            profile_bases={profile: attempt / "profiles" / "restored"},
            external_destinations={},
            target=None,
            profile_names={profile: "Portable restored"},
        )
        advisory_prefixes = (
            "owner_setup_required:",
            "metadata_normalized:",
            "metadata_unavailable:",
        )
        if any(not issue.startswith(advisory_prefixes) for issue in plan.issues):
            raise ValueError("portable_restore_plan_refused")
        from .recovery_service import owner_label
        from .space import require_capacity

        estimated = sum(row.size for row in doc.files) * 3 + 16 * 1024**2
        require_capacity({attempt: estimated})
        lines = (
            "Included saved data: "
            + ", ".join(sorted({owner_label(row.owner_id) for row in doc.files})),
            f"Validated payload: {len(doc.files)} files; {sum(row.size for row in doc.files)} bytes.",
            f"Estimated additional free space required: {estimated} bytes.",
            "Fresh destination: " + str(attempt / "profiles" / "restored"),
            "Owner setup after restore: "
            + ", ".join(
                sorted(
                    {
                        owner_label(issue.split(":", 1)[1])
                        for issue in plan.issues
                        if issue.startswith("owner_setup_required:")
                    }
                )
            ),
            f"Private destination metadata will be normalized for {sum(issue.startswith('metadata_normalized:') for issue in plan.issues)} members; {sum(issue.startswith('metadata_unavailable:') for issue in plan.issues)} members lack original metadata.",
            "Copied pending publication checks passed; unsupported content blocks this review.",
            "The copied source stays unchanged and retained.",
            "Import uses this sealed snapshot; later external edits remain in the retained source.",
            "Managed provider credentials must be set up again.",
            "Restored executable settings require their existing activation review.",
            "The copied directory cannot prove original multi-store coherence.",
            *review_lines,
        )
        if not confirm(lines, digest):
            _write(attempt, "cancelled.json", {"version": 1, "source_digest": digest})
            return None
        recheck()
        _write(
            attempt,
            "publication-started.json",
            {
                "version": 1,
                "location": str(data.parent),
                "source_digest": digest,
            },
        )
        profile_id = restore_isolated(sealed, plan, attempt / "control", Event())
        return _select(data, attempt, profile_id, digest)


def command_main(data: Path) -> int:
    """Present relocation review within the portable terminal before app startup."""

    def confirm(lines, digest):
        print("Chatbook Portable — relocation review\n")
        for line in lines:
            print(line)
        print("\nReview identity: " + digest)
        try:
            return (
                input("Type relocate to continue, or press Enter to cancel: ").strip()
                == "relocate"
            )
        except EOFError:
            return False

    try:
        selection = relocate(data, confirm)
    except (OSError, ValueError, RuntimeError) as error:
        code = str(error)
        if not re.fullmatch(r"[a-z][a-z0-9_.:-]{0,255}", code):
            code = "portable_relocation_failed"
        print("Relocation could not complete: " + code)
        print(
            "The retained copied source was not changed. Resolve this issue before trying again."
        )
        return 1
    if selection is None:
        return 0
    from .isolated_restore import launch_profile

    return launch_profile(
        selection.profile_id, data / WORK_NAME / selection.attempt / "control"
    )
