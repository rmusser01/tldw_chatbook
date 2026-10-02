"""Reconstructed TASK-33267 probe; September's original scripts were lost.

Run this same file against separately frozen git-archive snapshots. Each source
has .admission-source.json with commit, tree and source_digest(root). The digest
covers every source file except that manifest and bytecode. Receipts contain
only counters, timings and provenance, never application values or child logs.
Boot is a warm second full-app boot, counted from before app import to the
synchronous _ui_ready assignment. Transaction admission includes participant
entry AND retirement; SQLite body and complete transaction are separate.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import re
import statistics
import subprocess  # nosec B404
import sys
import tempfile
import threading
import time
from pathlib import Path

MANIFEST = ".admission-source.json"
CHILD_TIMEOUT = 300  # New outer probe supervision; app/native deadlines unchanged.
CONTAINMENT = (
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook/Notes/git_process_containment.py"
)
CONFIG = (
    '[general]\nusers_name = "probe"\n'
    "[first_run]\nsetup_completed = true\n"
    "[_first_run]\nsetup_completed = true\n"
    "[splash_screen]\nenabled = false\n"
    '[api_settings.openai]\napi_key = "sk-admission-probe-000000000000000000000000000000000000"\n'
)


def source_digest(root: Path) -> str:
    """Hash exact relative filenames and bytes; refuse linked source files."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.name == MANIFEST or "__pycache__" in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError("linked_source_file")
        if path.is_file():
            name = relative.as_posix().encode()
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def select_source(root: Path) -> dict:
    """Reject a changed/unidentified snapshot before application imports."""
    manifest = json.loads((root / MANIFEST).read_text())
    if set(manifest) != {"commit", "tree", "content_sha256"} or not all(
        re.fullmatch(
            r"[0-9a-f]{40}" if key != "content_sha256" else r"[0-9a-f]{64}", value
        )
        for key, value in manifest.items()
    ):
        raise ValueError("invalid_source_manifest")
    if source_digest(root) != manifest["content_sha256"]:
        raise ValueError("source_digest_changed")
    return manifest


def private_environment(profile: Path) -> dict[str, str]:
    """Use the existing private-profile selections and a small OS env allowlist."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "PATH",
            "SYSTEMROOT",
            "SYSTEMDRIVE",
            "WINDIR",
            "COMSPEC",
            "PATHEXT",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "TZ",
            "TERM",
            "OS",
        }
    }
    profile.mkdir(mode=0o700, parents=True, exist_ok=True)
    for key, leaf in (
        ("HOME", "home"),
        ("USERPROFILE", "home"),
        ("XDG_CONFIG_HOME", "config"),
        ("XDG_DATA_HOME", "data"),
    ):
        directory = profile / leaf
        directory.mkdir(mode=0o700, exist_ok=True)
        environment[key] = str(directory)
    config = profile / "config/config.toml"
    if not config.exists():
        config.write_text(CONFIG)
        config.chmod(0o600)
    environment.update(
        TLDW_CONFIG_PATH=str(config),
        TLDW_TEST_CONFIG_ROOT=str(profile),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
        HF_HUB_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        TLDW_TEST_MODE="1",
        TLDW_SCREEN_PREIMPORT="0",
    )
    return environment


def counters() -> dict:
    """Numeric units shared by audit and call-through instrumentation."""
    return {
        "counting": False,
        "entries": 0,
        "entry_attempts": 0,
        "entry_ns": 0,
        "retirements": 0,
        "retirement_attempts": 0,
        "retirement_ns": 0,
        "os_opens": 0,
        "native_handle_opens": 0,
        "native_acl_reads": 0,
        "helper_spawns": 0,
    }


class TimedScope:
    """Time complete context boundaries without altering exceptions/suppression."""

    def __init__(self, factory, counts, clock=time.perf_counter_ns):
        self.factory, self.counts, self.clock = factory, counts, clock

    def __enter__(self):
        self.counts["entry_attempts"] += 1
        start = self.clock()
        try:
            self.scope = self.factory()
            value = self.scope.__enter__()
            self.counts["entries"] += 1
            return value
        finally:
            self.counts["entry_ns"] += self.clock() - start

    def __exit__(self, *error):
        self.counts["retirement_attempts"] += 1
        start = self.clock()
        try:
            result = self.scope.__exit__(*error)
            self.counts["retirements"] += 1
            return result
        finally:
            self.counts["retirement_ns"] += self.clock() - start


_OPEN_COUNTS = None


def install_open_audit(counts):
    """Reuse the census audit pattern; preserve os.open's guard identity."""
    global _OPEN_COUNTS
    if _OPEN_COUNTS is None:

        def audit(event, args):
            active = _OPEN_COUNTS
            if active["counting"] and event == "open" and args[1] is None:
                active["os_opens"] += 1

        sys.addaudithook(audit)
    _OPEN_COUNTS = counts


def instrument(counts, children):
    """Call through installed scopes and native Windows seams, all threads."""
    from tldw_chatbook.Backup_Recovery.participants import _RepositoryParticipant

    original = _RepositoryParticipant.operation
    depth = threading.local()

    def operation(owner):
        # Nested repository scopes reuse the outer boundary; don't double bill.
        class Scope(TimedScope):
            def __enter__(self):
                self.level = getattr(depth, "level", 0)
                depth.level = self.level + 1
                try:
                    if counts["counting"] and self.level == 0:
                        self.measured = True
                        return super().__enter__()
                    self.measured = False
                    self.scope = self.factory()
                    return self.scope.__enter__()
                except BaseException:
                    depth.level = self.level
                    raise

            def __exit__(self, *error):
                try:
                    return (
                        super().__exit__(*error)
                        if self.measured
                        else self.scope.__exit__(*error)
                    )
                finally:
                    depth.level = self.level

        return Scope(lambda: original(owner), counts)

    _RepositoryParticipant.operation = operation
    original_spawn = subprocess.Popen.__init__

    def spawn(child, *args, **kwargs):
        original_spawn(child, *args, **kwargs)
        children.append(child)
        if counts["counting"]:
            counts["helper_spawns"] += 1

    subprocess.Popen.__init__ = spawn
    if os.name == "nt":
        from tldw_chatbook.Utils.windows_files import _Native

        for name, key in (
            ("open_handle", "native_handle_opens"),
            ("security", "native_acl_reads"),
        ):
            original_native = getattr(_Native, name)

            def counted(*args, _original=original_native, _key=key, **kwargs):
                if counts["counting"]:
                    counts[_key] += 1
                return _original(*args, **kwargs)

            setattr(_Native, name, counted)


def prepare_imports(source: Path):
    """Install Tests.network_guard and Null keyring before any app import."""
    if any(name.startswith("tldw_chatbook") for name in sys.modules):
        raise RuntimeError("application_imported_before_isolation")
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location(
        "Tests.network_guard", source / "Tests/network_guard.py"
    )
    guard = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = guard
    spec.loader.exec_module(guard)
    guard.install()
    guard.set_allowed(False)
    import keyring
    from keyring.backends.null import Keyring

    keyring.set_keyring(Keyring())
    return guard


def retired_state(children):
    """Check observable native ownership after public close and startup close."""
    from tldw_chatbook.Backup_Recovery import participants
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    # App teardown and asyncio.run have already settled accepted producers.
    # Existing owner closure is thread-qualified; foreign/unknown caches stay visible.
    pause = storage._begin_local_pause()
    try:
        participants._retire_current_thread_caches(pause)
        # A joined worker's registered connection remains visible. Use only its
        # exact owner's public barrier, which rejects active uses/transactions.
        loaded = sys.modules.get("tldw_chatbook.DB.ChaChaNotes_DB")
        repository_type = getattr(loaded, "CharactersRAGDB", None)
        if repository_type is not None:
            for participant in tuple(participants._installed_repositories):
                repository = participant.repository()
                if (
                    participant.owner_id == "db.chachanotes.primary"
                    and type(repository) is repository_type
                ):
                    with repository_type.quiesce_connections(
                        repository, timeout_seconds=0
                    ):
                        pass
        storage._shutdown()
    finally:
        pause.resume()
    with storage._lock:
        outstanding = {
            name: len(getattr(storage, name, ()))
            for name in (
                "_holds",
                "_retiring_holds",
                "_pending_acquisitions",
                "_live_leases",
                "_operations",
                "_raw_operations",
            )
        }
    live = sum(child.poll() is None for child in children)
    if live:
        # Failed runs are not qualification; own and reap the remaining effects.
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()
    return {
        "retired": not any(outstanding.values()) and live == 0,
        "outstanding_ownership": outstanding,
        "live_children_before_reaping": live,
    }


def transaction(counts, iterations, seed=False):
    """Seed fixed data, then exercise the real CharactersRAGDB transaction."""
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB  # noqa: I001
    from tldw_chatbook.config import get_chachanotes_db_path

    path = get_chachanotes_db_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    db = CharactersRAGDB(path, "admission-probe")
    bodies, totals, admissions = [], [], []
    try:
        if seed:
            for index in range(8):
                db.add_note(
                    "synthetic probe",
                    "fixed synthetic content",
                    note_id=f"probe-{index}",
                )
        else:
            # Warm the actual connection once; recurring operations remain checked.
            with db.transaction() as cursor:
                cursor.execute("SELECT COUNT(*) FROM notes").fetchone()
            counts["counting"] = True
            for _ in range(iterations):
                before = counts["entry_ns"] + counts["retirement_ns"]
                start = time.perf_counter_ns()
                with db.transaction() as cursor:
                    body_start = time.perf_counter_ns()
                    row = cursor.execute(
                        "SELECT COUNT(*) FROM notes WHERE deleted = 0"
                    ).fetchone()
                    if row[0] != 8:
                        raise RuntimeError("synthetic_seed_changed")
                    bodies.append(time.perf_counter_ns() - body_start)
                totals.append(time.perf_counter_ns() - start)
                admissions.append(counts["entry_ns"] + counts["retirement_ns"] - before)
            counts["counting"] = False
    finally:
        counts["counting"] = False
        db.close_connection()
    return {
        "body_median_ns": statistics.median(bodies) if bodies else 0,
        "transaction_median_ns": statistics.median(totals) if totals else 0,
        # Conservative complete boundary: includes SQLite BEGIN/COMMIT and all
        # guarded manager work, so a shortened admission seam cannot pass alone.
        "transaction_boundary_median_ns": statistics.median(
            total - body for total, body in zip(totals, bodies, strict=True)
        )
        if totals
        else 0,
        "admission_median_ns": statistics.median(admissions) if admissions else 0,
        "db_path_depth": len(path.parts),
        "seed_notes": 8,
    }


async def boot(counts, started):
    """Count before full-app import and stop synchronously at actual readiness."""
    from tldw_chatbook.app import TldwCli

    snapshot = {}

    def ready_get(app):
        return app.__dict__.get("_probe_ready", False)

    def ready_set(app, value):
        app.__dict__["_probe_ready"] = value
        if value and not snapshot:
            snapshot.update(counts)
            snapshot["boot_ns"] = time.perf_counter_ns() - started
            counts["counting"] = False

    TldwCli._ui_ready = property(ready_get, ready_set)
    app = TldwCli()
    async with app.run_test(size=(120, 40)):
        while not app._ui_ready:
            await asyncio.sleep(0.005)
    if not snapshot:
        raise RuntimeError("ui_ready_unobserved")
    return snapshot


def child(source, phase, iterations):
    """Project errors to their bounded type; retain all cleanup failures."""
    receipt = {
        "exit_code": 1,
        "error_type": "",
        "retired": False,
        "network_attempts": 0,
        "source_sha256": source_digest(source),
    }
    counts, children, guard = counters(), [], None
    try:
        select_source(source)
        guard = prepare_imports(source)
        install_open_audit(counts)
        started = time.perf_counter_ns()
        counts["counting"] = phase == "boot"
        instrument(counts, children)
        measured = (
            asyncio.run(boot(counts, started))
            if phase == "boot"
            else transaction(counts, iterations, seed=phase == "seed")
        )
        receipt.update(measured)
        receipt["exit_code"] = 0
    except BaseException as error:  # noqa: BLE001 - failures remain in the receipt
        receipt["error_type"] = type(error).__name__[:80]
        if isinstance(error, SystemExit) and isinstance(error.code, int) and error.code:
            receipt["exit_code"] = error.code
    finally:
        counts["counting"] = False
        try:
            if "tldw_chatbook.Backup_Recovery.storage_admission" in sys.modules:
                receipt.update(retired_state(children))
            else:
                receipt["retired"] = not children
        except BaseException as error:  # noqa: BLE001 - cleanup failure is qualification failure
            receipt["cleanup_error_type"] = type(error).__name__[:80]
        receipt["network_attempts"] = len(guard.blocked_attempts()) if guard else 0
        leaked = sum(
            not Path(module.__file__).resolve().is_relative_to(source)
            for name, module in tuple(sys.modules.items())
            if name.startswith("tldw_chatbook") and getattr(module, "__file__", None)
        )
        receipt["foreign_source_modules"] = leaked
        if not receipt["retired"] or receipt["network_attempts"] or leaked:
            receipt["exit_code"] = receipt["exit_code"] or 1
        for key, value in counts.items():
            receipt.setdefault(key, value)
    return receipt


async def supervise(command, source, environment, log):
    """Reuse native suspended Job/group admission and positive empty-tree proof."""
    spec = importlib.util.spec_from_file_location("admission_containment", CONTAINMENT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    controller, tree, pumps = module.ProcessTreeController(), None, []
    result = {"exit_code": 1, "supervisor_retired": False}

    async def drain(reader, output):
        while chunk := await reader.read(8192):
            output.write(chunk)

    with log.open("wb") as output:
        try:
            try:
                tree = await controller.spawn(
                    *command, cwd=str(source), environment=environment, stdin=False
                )
            except module.ProcessTreeAdmissionError as error:
                tree = error.tree
                raise
            pumps = [
                asyncio.create_task(drain(reader, output))
                for reader in (tree.process.stdout, tree.process.stderr)
            ]
            result["exit_code"] = await asyncio.wait_for(
                tree.process.wait(), CHILD_TIMEOUT
            )
        except BaseException as error:  # noqa: BLE001 - project failure without raw logs
            result["supervisor_error_type"] = type(error).__name__[:80]
        finally:
            if tree is not None:
                empty = await controller.wait(tree, timeout=0)
                if not empty:
                    controller.kill(tree)
                    empty = await controller.wait(tree, timeout=2)
                result["supervisor_retired"] = empty
                if empty:
                    controller.close(tree)
                for pump in pumps:
                    if not empty:
                        pump.cancel()
                await asyncio.gather(*pumps, return_exceptions=False)
    return result


def run_child(source, profile, phase, iterations):
    """Create isolated child selections before interpreter/application startup."""
    source = source.resolve(strict=True)
    manifest = select_source(source)
    environment = private_environment(profile.resolve())
    environment["TLDW_PROBE_SOURCE"] = str(source)
    result_file = profile / f"{phase}-child.json"
    result_file.unlink(missing_ok=True)
    command = [
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        "--child",
        "--source",
        str(source),
        "--phase",
        phase,
        "--iterations",
        str(iterations),
        "--receipt",
        str(result_file),
    ]
    supervised = asyncio.run(
        supervise(command, source, environment, profile / f"{phase}-child.log")
    )
    if not result_file.exists():
        return {
            **supervised,
            "exit_code": supervised["exit_code"] or 1,
            "error_type": "MissingReceipt",
            "retired": False,
            "source_sha256": manifest["content_sha256"],
        }
    receipt = json.loads(result_file.read_text())
    receipt["exit_code"] = supervised["exit_code"] or receipt["exit_code"]
    receipt.update(
        {key: value for key, value in supervised.items() if key != "exit_code"}
    )
    if not supervised["supervisor_retired"] or supervised.get("supervisor_error_type"):
        receipt["exit_code"] = receipt["exit_code"] or 1
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("transaction", "boot", "seed"), required=True
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 10000 or (args.phase == "seed" and not args.child):
        parser.error("invalid_probe_iterations_or_phase")
    if args.child:
        receipt = child(args.source, args.phase, args.iterations)
    else:
        manifest = select_source(args.source)
        args.receipt.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        work = Path(
            tempfile.mkdtemp(prefix="admission-profile-", dir=args.receipt.parent)
        )
        # Same fixed component depth on both sources. Profiles and raw logs stay private.
        profile = work / "profile"
        seed = run_child(args.source, profile, "seed", 1)
        warmup = (
            run_child(args.source, profile, "boot", 1)
            if seed["exit_code"] == 0 and args.phase == "boot"
            else None
        )
        prerequisite = warmup or seed
        runs = (
            [run_child(args.source, profile, args.phase, args.iterations)]
            if args.phase == "transaction" and prerequisite["exit_code"] == 0
            else [
                run_child(args.source, profile, "boot", 1)
                for _ in range(args.iterations)
            ]
            if prerequisite["exit_code"] == 0
            else []
        )
        receipt = {
            "protocol": "reconstructed-v1",
            "source": manifest,
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "containment_sha256": hashlib.sha256(CONTAINMENT.read_bytes()).hexdigest(),
            "platform": sys.platform,
            "native_windows_measured": os.name == "nt",
            "iterations": args.iterations,
            "phase": args.phase,
            "limits": {
                "admission_ns": 500000,
                "boot_open_reduction_percent": 80,
                "child_timeout_seconds": CHILD_TIMEOUT,
            },
            "private_profile_depth": len(profile.parts),
            "seed": seed,
            "warmup": warmup,
            "runs": runs,
            "exit_code": max(
                [prerequisite["exit_code"], *[r["exit_code"] for r in runs]]
            ),
        }
    args.receipt.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    args.receipt.chmod(0o600)
    if not args.child:
        print(json.dumps(receipt, sort_keys=True))
    return receipt["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
