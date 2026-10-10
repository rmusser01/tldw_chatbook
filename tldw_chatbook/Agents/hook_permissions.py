"""Local exact-definition consent for standalone Console hooks (ADR-197)."""

from __future__ import annotations

import copy
import fnmatch
import json
import os
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from uuid import UUID, uuid4

import portalocker
from loguru import logger

from tldw_chatbook import config
from tldw_chatbook.Agents.run_hooks import (
    BLOCKING_EVENTS,
    HOOK_EVENTS,
    HookInventory,
    HookInventoryRow,
    HookLaunchRefused,
    HookSpec,
    HookTarget,
    fingerprint_hook,
    inspect_hooks_config,
)
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Utils.path_validation import validate_path_simple
from tldw_chatbook.Utils.private_paths import (
    PrivateFileWritePrecondition,
    atomic_private_write_text,
    open_private_binary,
    open_private_lock_stream,
)

_STORE_MAX_BYTES = 4 * 1024 * 1024
_FINGERPRINT = re.compile(r"[0-9a-f]{64}")


class HookReviewConflict(ValueError):
    """The saved definition or consent revision changed after review."""


@dataclass(frozen=True, slots=True)
class HookReviewRow:
    """A current row or a non-selectable global recovery state."""

    entry: HookInventoryRow | None
    state: str
    change: str = "Existing"


@dataclass(frozen=True, slots=True)
class HookReviewSnapshot:
    """Detached review presentation; no authority is derived from this object."""

    config: config.HookConfigSnapshot
    rows: tuple[HookReviewRow, ...]
    store_path: Path
    store_revision: tuple[str, int]
    blocked_reason: str | None
    notice: str | None = None

    @property
    def ready(self) -> bool:
        return self.blocked_reason is None

    @property
    def pending_count(self) -> int:
        return sum(row.state in {"pending", "invalid", "recovery"} for row in self.rows)


@dataclass(frozen=True, slots=True, eq=False)
class HookAuthorityRead:
    """One full reconciled read, kept only by the Send attempt that made it.

    ADR-225 decision 3: one attempt's pre-commit preparation consumers share
    one immutable authority observation instead of each repeating the config
    write lock, config read, store lock and raw admission. The owner never
    stores or reuses this object; the attempt passes it explicitly to its own
    later consumers, which re-validate it in memory
    (``attempt_read_current``) and fall back to their fresh read on any
    mismatch. It authorizes no effect: every hook launch still enters
    ``launch_guard`` (a fresh read) immediately before process creation, and
    the final pre-dispatch admission stays a fresh read.

    It only ever answers "nothing to do": admission that it does not refuse,
    and hook selections (legacy targets, v2 grants) that are empty. Anything
    that would select, build or refuse from it is decided by a fresh read,
    because a consent change made by another process after this read is
    invisible in memory -- a target selected from it could reach its launch
    guard stale and be refused (blocking the Send) where a fresh selection
    would have omitted it.

    Attributes:
        owner: The exact consent owner that performed the read.
        snapshot: The review state the read reconciled and published.
        targets: The grant targets published together with ``snapshot``, or
            ``None`` when another read published in between (unusable).
        config_identity: ``config.current_config_identity()`` taken before
            the read, or ``None`` when it could not be observed (unusable).
    """

    owner: HookPermissions = field(repr=False)
    snapshot: HookReviewSnapshot = field(repr=False)
    targets: tuple[HookTarget, ...] | None = field(repr=False)
    config_identity: tuple[int, str] | None = field(repr=False)


def default_hook_permissions_path() -> Path:
    """Resolve the live canonical profile directory, never a startup constant."""
    return Path(config.get_user_data_dir()) / "hook_permissions.json"


#: A visit snapshot is kept only once both hook files' last change is this
#: old (storage_admission's _EVIDENCE_SETTLE_NS): a same-size edit inside one
#: coarse timestamp tick leaves an identical stamp.
_VISIT_SETTLE_NS = 1_000_000_000


def _file_stamp(path: Path) -> tuple[int, int, int, int, int, int] | None:
    """A file's type, identity, size and change times, without following links.

    Args:
        path: The file to observe.

    Returns:
        ``(st_mode, st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns)`` of the
        entry itself (so a symlink swapped in is a different stamp), or
        ``None`` when it is absent.

    Raises:
        OSError: The file could not be observed for another reason.
    """
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    return (
        info.st_mode,
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _empty_state() -> dict:
    return {"schema_version": 1, "store_id": str(uuid4()), "revision": 0, "configs": {}}


def _valid_uuid(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return str(UUID(value)) == value
    except ValueError:
        return False


def _validate_state(state: object) -> dict:
    if (
        not isinstance(state, dict)
        or type(state.get("schema_version")) is not int
        or state["schema_version"] != 1
        or not _valid_uuid(state.get("store_id"))
        or type(state.get("revision")) is not int
        or state["revision"] < 0
        or not isinstance(state.get("configs"), dict)
    ):
        raise ValueError("Hook permission state is invalid or unsupported.")
    for scope, record in state["configs"].items():
        if not isinstance(scope, str) or not isinstance(record, dict):
            raise TypeError("Invalid hook permission scope.")
        for kind in ("observed", "grants"):
            entries = record.get(kind)
            if not isinstance(entries, dict):
                raise TypeError("Invalid hook permission records.")
            for key, entry in entries.items():
                if (
                    not isinstance(key, str)
                    or not isinstance(entry, dict)
                    or not isinstance(entry.get("fingerprint"), str)
                    or _FINGERPRINT.fullmatch(entry["fingerprint"]) is None
                ):
                    raise ValueError("Invalid hook permission identity.")
                if kind == "grants" and not _valid_uuid(entry.get("token")):
                    raise ValueError("Invalid hook permission grant.")
                if kind == "observed" and (
                    type(entry.get("enabled")) is not bool
                    or entry.get("change") not in {"Existing", "New", "Modified"}
                ):
                    raise ValueError("Invalid hook permission observation.")
    return state


class HookPermissions:
    """One app-owned consent owner; callers offload its I/O from Textual."""

    def __init__(self) -> None:
        self._authority_lock = threading.RLock()
        self._cache_lock = threading.Lock()
        self._sealed: set[tuple[str, str, str]] = set()
        self._refresh_pending: set[tuple[str, str]] = set()
        self._closed = threading.Event()
        self._published: HookReviewSnapshot | None = None
        self._published_targets: tuple[HookTarget, ...] = ()
        #: TASK-33642: ``(snapshot, stamp)`` a Console visit may reuse.
        self._visit_reuse: tuple[HookReviewSnapshot, tuple] | None = None

    @contextmanager
    def _store_lock(self, path: Path) -> Iterator[None]:
        from tldw_chatbook.Backup_Recovery import raw_participants

        # Retain config's lease and writer locks while independently admitting
        # this concrete store; config helpers keep their original narrow scope.
        with raw_participants._scope(
            self, "hook_permissions", writing=True, selected_read=path
        ):
            lock_path = path.with_name(path.name + ".lock")
            stream = open_private_lock_stream(
                lock_path, application_owned_directory=path.parent
            )
            try:
                portalocker.lock(stream, portalocker.LockFlags.EXCLUSIVE)
                try:
                    yield
                finally:
                    portalocker.unlock(stream)
            finally:
                stream.close()

    def _read_state(
        self, path: Path
    ) -> tuple[dict | None, PrivateFileWritePrecondition, str | None]:
        precondition = PrivateFileWritePrecondition.missing()
        try:
            with open_private_binary(path) as opened:
                precondition = PrivateFileWritePrecondition.from_opened(opened)
                encoded = opened.stream.read(_STORE_MAX_BYTES + 1)
            if len(encoded) > _STORE_MAX_BYTES:
                raise ValueError("Hook permission state exceeds its limit.")
            return _validate_state(json.loads(encoded)), precondition, None
        except FileNotFoundError:
            return _empty_state(), precondition, None
        except (
            OSError,
            ValueError,
            TypeError,
            UnicodeError,
            RecursionError,
            RecoveryRequired,
        ):
            return (
                None,
                precondition,
                "Hook permission state unavailable; retry or reset invalid state.",
            )

    def _write_state(
        self, path: Path, state: dict, precondition: PrivateFileWritePrecondition
    ) -> None:
        state["revision"] += 1
        encoded = json.dumps(
            state, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        if len(encoded.encode("utf-8")) > _STORE_MAX_BYTES:
            raise ValueError("Hook permission state exceeds its limit.")
        atomic_private_write_text(
            path,
            encoded,
            application_owned_directory=path.parent,
            target_precondition=precondition,
        )

    @staticmethod
    def _inventory(snapshot: config.HookConfigSnapshot) -> HookInventory:
        return inspect_hooks_config(
            {"hooks": snapshot.section} if snapshot.section_present else {}
        )

    def _reconcile(self, state: dict, cfg: config.HookConfigSnapshot) -> bool:
        before = copy.deepcopy(state["configs"])
        scope = str(cfg.config_path)
        first = scope not in state["configs"]
        record = state["configs"].setdefault(scope, {"observed": {}, "grants": {}})
        previous = record["observed"]
        inventory = self._inventory(cfg)
        observed = {}
        for row in inventory.review_rows:
            if row.spec is None:
                continue
            fingerprint = fingerprint_hook(row.spec)
            prior = previous.get(row.key)
            change = "Existing" if first else "New"
            if prior:
                change = prior["change"]
                if prior["fingerprint"] != fingerprint:
                    change = (
                        "Modified" if row.key.startswith(("id:", "v2:id:")) else "New"
                    )
                    record["grants"].pop(row.key, None)
            eligible = inventory.master_enabled is True and row.enabled is True
            if (
                prior
                and prior["enabled"]
                and not eligible
                and row.key in record["grants"]
            ):
                record["grants"][row.key]["token"] = str(uuid4())
            observed[row.key] = {
                "fingerprint": fingerprint,
                "enabled": eligible,
                "change": change,
            }
        old_groups = Counter(
            value["fingerprint"]
            for key, value in previous.items()
            if key.startswith("legacy:")
        )
        new_groups = Counter(
            value["fingerprint"]
            for key, value in observed.items()
            if key.startswith("legacy:")
        )
        changed_groups = {
            fp
            for fp in old_groups.keys() | new_groups.keys()
            if old_groups[fp] != new_groups[fp]
        }
        for key in tuple(record["grants"]):
            if key not in observed or (
                key.startswith("legacy:")
                and record["grants"][key]["fingerprint"] in changed_groups
            ):
                record["grants"].pop(key, None)
        record["observed"] = observed
        return before != state["configs"]

    def _is_sealed(self, path: Path, scope: str, key: str) -> bool:
        with self._cache_lock:
            return (str(path), scope, key) in self._sealed or (
                str(path),
                scope,
            ) in self._refresh_pending

    def _seal(self, expected: HookReviewSnapshot, keys: Collection[str]) -> None:
        with self._cache_lock:
            self._sealed.update(
                (str(expected.store_path), str(expected.config.config_path), key)
                for key in keys
            )

    def _unseal(self, path: Path, scope: str, keys: Collection[str]) -> None:
        with self._cache_lock:
            self._sealed.difference_update((str(path), scope, key) for key in keys)

    def _make_snapshot(
        self,
        cfg: config.HookConfigSnapshot,
        path: Path,
        state: dict | None,
        error: str | None = None,
        notice: str | None = None,
    ) -> HookReviewSnapshot:
        inventory = self._inventory(cfg)
        scope = str(cfg.config_path)
        record = (
            state["configs"].get(scope, {"observed": {}, "grants": {}})
            if state
            else {"observed": {}, "grants": {}}
        )
        rows = []
        blocked = inventory.container_error
        with self._cache_lock:
            refresh_pending = (str(path), scope) in self._refresh_pending
        if refresh_pending and inventory.requires_authority:
            error = error or "Saved configuration needs runtime refresh; retry refresh."
        if blocked:
            rows.append(HookReviewRow(None, "invalid"))
        if error:
            rows.append(HookReviewRow(None, "recovery"))
            blocked = error
        targets = []
        for row in inventory.review_rows:
            change = record["observed"].get(row.key, {}).get("change", "Existing")
            grant = record["grants"].get(row.key)
            if row.spec is None:
                status = (
                    "invalid"
                    if row.enabled is not False
                    and inventory.master_enabled is not False
                    else "disabled"
                )
            elif row.enabled is False or inventory.master_enabled is False:
                status = "disabled"
            elif inventory.container_error:
                status = "invalid"
            elif (
                grant
                and grant["fingerprint"] == fingerprint_hook(row.spec)
                and not self._is_sealed(path, scope, row.key)
            ):
                status = "approved"
                targets.append(
                    HookTarget(
                        scope,
                        row.key,
                        grant["fingerprint"],
                        row.spec,
                        state["store_id"] + ":" + grant["token"],
                    )
                )
            else:
                status = "pending"
            rows.append(HookReviewRow(row, status, change))
            if status in {"pending", "invalid"} and not blocked:
                blocked = "Review enabled hooks before sending."
        if not inventory.requires_authority:
            blocked = None
        if self._closed.is_set() and inventory.requires_authority:
            blocked = "Hook permission owner is closed."
            targets = []
        result = HookReviewSnapshot(
            cfg,
            tuple(rows),
            path,
            (state["store_id"], state["revision"]) if state else ("", 0),
            blocked,
            notice,
        )
        with self._cache_lock:
            self._published = result
            self._published_targets = tuple(targets) if not error else ()
        return result

    @contextmanager
    def _current(
        self,
        *,
        reconcile: bool = True,
    ) -> Iterator[tuple[HookReviewSnapshot, dict | None, PrivateFileWritePrecondition]]:
        with (
            config.locked_hooks_config_snapshot() as cfg,
            self._authority_lock,
            ExitStack() as stack,
        ):
            path = cfg.config_path.parent / "hook_permissions.json"
            try:
                if cfg.profile_data_dir is None:
                    raise RecoveryRequired("raw_source_selection_changed")
                path = cfg.profile_data_dir / "hook_permissions.json"
                # Raw admission also checks this exact selected_read against the
                # bound cache; a throttled previous profile must never supply grants.
                # Not derivable from ``cfg``: ``profile_data_dir`` comes from the
                # raw file read under this lock, while this follows the runtime's
                # cached selection (and its posture checks). Comparing the two
                # IS the selection check, so the second lookup stays.
                if default_hook_permissions_path() != path:
                    raise RecoveryRequired("raw_source_selection_changed")
                stack.enter_context(self._store_lock(path))
            except (OSError, ValueError, TypeError, RecoveryRequired):
                yield (
                    self._make_snapshot(
                        cfg, path, None, "Hook permission state unavailable; retry."
                    ),
                    None,
                    PrivateFileWritePrecondition.missing(),
                )
                return
            state, precondition, error = self._read_state(path)
            if state is not None and reconcile and self._reconcile(state, cfg):
                try:
                    self._write_state(path, state, precondition)
                    state, precondition, error = self._read_state(path)
                except (OSError, ValueError, RecoveryRequired):
                    state = None
                    error = "Hook permission state could not be saved; retry."
            yield self._make_snapshot(cfg, path, state, error), state, precondition

    def _visit_stamp(self, config_path: Path, store_path: Path) -> tuple:
        """Everything a visit snapshot depends on, read without opening a file.

        Args:
            config_path: The config file holding the hook section.
            store_path: The hook permission store.

        Returns:
            First, both files' type, identity and change times taken
            without following links (``None`` for a missing file); the
            config selection (the effective config path as well as the
            loaded source), generation and the environment that selects the
            data root; the no-follow posture of both lock files and of every
            component of both parent directories (the full read refuses an
            unsafe one); the storage admission epoch and the serving state of
            its native holds; and last, the in-memory sealing, refresh and
            closed state the snapshot also reflects (``_memory_state()``).
            Admission records other processes keep on disk are not mirrored:
            no authority derives from a visit snapshot, and Send and review
            always run the full read.
        """
        from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission

        with self._cache_lock:
            memory = self._memory_state()
        with storage_admission._lock:
            holds = tuple(
                sorted(
                    (str(key), storage_admission._hold_serving(hold))
                    for key, hold in storage_admission._holds.items()
                )
            )
        chain, posture = storage_admission._chain, storage_admission._posture
        return (
            (_file_stamp(config_path), _file_stamp(store_path)),
            str(config.get_cli_config_path()),
            config._CONFIG_CACHE_SOURCE,
            config._CONFIG_GENERATION,
            # The data root (and so the store) follows HOME and the XDG
            # variables as well as config; a retarget forces a full read.
            os.environ.copy(),
            posture(config_path.with_name(config_path.name + ".lock")),
            posture(store_path.with_name(store_path.name + ".lock")),
            tuple(posture(part) for part in chain(config_path.parent)),
            tuple(posture(part) for part in chain(store_path.parent)),
            bootstrap._admission_epoch,
            holds,
            memory,
        )

    def _memory_state(self) -> tuple:
        """Sealing, refresh and closed state; the caller holds ``_cache_lock``."""
        return (
            frozenset(self._sealed),
            frozenset(self._refresh_pending),
            self._closed.is_set(),
        )

    def visit_snapshot(self) -> HookReviewSnapshot:
        """``snapshot()`` for a Console visit, reused while nothing it read moved.

        TASK-33642: every Console visit re-read the hook section and the
        permission store under their locks. A snapshot is reused only when
        stamps taken before and after the read that produced it are equal and
        still equal now, so a write at any time -- including during that read
        -- forces a full read on the next visit. It is kept only once both
        files' last change is ``_VISIT_SETTLE_NS`` old, so a same-size write
        inside one timestamp tick cannot hide behind an equal stamp. Sending
        and reviewing keep using ``snapshot()``.

        Returns:
            The current review state.
        """
        from tldw_chatbook.Backup_Recovery import storage_admission

        with self._cache_lock:
            reusable = self._visit_reuse
            published = self._published
        # A maintenance pause refuses the full read; show what it reports.
        paused = storage_admission._pause is not None
        if reusable is not None and reusable[0] is published and not paused:
            try:
                stamp = self._visit_stamp(
                    published.config.config_path, published.store_path
                )
            except OSError:
                stamp = None
            if stamp is not None and stamp == reusable[1]:
                # Re-read under the lock sealing takes: a hook sealed while
                # the probes ran is never served as approved.
                with self._cache_lock:
                    if (
                        self._visit_reuse is reusable
                        and self._memory_state() == stamp[-1]
                    ):
                        return published
        try:
            # Validated as locked_hooks_config_snapshot validates it, before any
            # metadata probe touches the environment-selected path.
            selected = validate_path_simple(
                config.get_cli_config_path(),
                require_exists=False,
                probe_existing=False,
                reject_shell_metacharacters=False,
            )
            before = self._visit_stamp(selected, default_hook_permissions_path())
        except (OSError, ValueError, RecoveryRequired):
            before = None
        snapshot = self.snapshot()
        try:
            after = self._visit_stamp(snapshot.config.config_path, snapshot.store_path)
        except OSError:
            after = None
        settled = time.time_ns() - _VISIT_SETTLE_NS
        keep = (
            after is not None
            and after == before
            and all(
                stamp is None or max(stamp[4], stamp[5]) <= settled
                for stamp in after[0]
            )
            and not any(row.state == "recovery" for row in snapshot.rows)
        )
        with self._cache_lock:
            self._visit_reuse = (snapshot, after) if keep else None
        return snapshot

    def snapshot(self) -> HookReviewSnapshot:
        """Reconcile the current saved section and return review state."""
        try:
            with self._current() as (snapshot, _state, _precondition):
                return snapshot
        except (OSError, ValueError, RecoveryRequired):
            with self._cache_lock:
                previous = self._published
            cfg = config.HookConfigSnapshot(
                previous.config.config_path
                if previous
                else Path(config.get_cli_config_path()).resolve(),
                True,
                None,
                "unavailable",
            )
            path = (
                previous.store_path
                if previous
                else Path(config.get_cli_config_path()).parent / "hook_permissions.json"
            )
            return self._make_snapshot(
                cfg,
                path,
                None,
                "Hook configuration or permission state unavailable; retry.",
            )

    def authority_read(self) -> HookAuthorityRead:
        """``snapshot()`` plus the exact grant targets published with it.

        One Send attempt's first full read (ADR-225 decision 3). It runs the
        unchanged ``snapshot()`` -- the same reconciliation, locks, errors and
        review state -- and additionally keeps the targets ``_make_snapshot``
        published atomically with that snapshot. When another read published
        in between, the targets are unknown and the result is unusable for
        sharing; the caller still gets the snapshot it read.

        Returns:
            The attempt-bound read; never stored by this owner.
        """
        try:
            # Taken before the read: any later reload or retarget, including
            # one racing the read itself, makes the result unusable.
            identity = config.current_config_identity()
        except Exception:  # noqa: BLE001 -- unobservable identity only disables sharing
            identity = None
        snapshot = self.snapshot()
        with self._cache_lock:
            targets = self._published_targets if self._published is snapshot else None
        return HookAuthorityRead(self, snapshot, targets, identity)

    def attempt_read_current(self, read: HookAuthorityRead) -> bool:
        """Whether an attempt's earlier full read still stands for that attempt.

        In memory only (no I/O, no lock wait beyond ``_cache_lock``). A read is
        usable only when it was ready, error-free and read this owner's store,
        and nothing this process has since observed or done moved it: the
        latest published read must have the same config file, section stamp,
        profile and store revision; no runtime refresh may be pending; no
        definition of that scope may be sealed by an in-flight decision; the
        owner must be open; and the effective config must be the same
        generation and file. Any mismatch returns False and the consumer
        performs its own fresh read. This is a narrowing check, never a grant:
        a change only another process has made is invisible here, which is
        why a standing read still answers only "nothing to do"
        (``attempt_targets``, ``attempt_v2_configuration``) and why the final
        admission and every launch's ``launch_guard`` read fresh.

        Args:
            read: A read this owner produced for the asking attempt.

        Returns:
            Whether the attempt may use ``read`` instead of a fresh read.
        """
        if type(read) is not HookAuthorityRead or read.owner is not self:
            return False
        snapshot = read.snapshot
        try:
            identity_current = (
                read.config_identity is not None
                and config.current_config_identity() == read.config_identity
            )
        except Exception:  # noqa: BLE001 -- unknown currency falls back to a fresh read
            return False
        if (
            not identity_current
            or read.targets is None
            or not snapshot.ready
            or snapshot.store_revision == ("", 0)
            or any(row.state == "recovery" for row in snapshot.rows)
        ):
            return False
        scope = (str(snapshot.store_path), str(snapshot.config.config_path))
        with self._cache_lock:
            published = self._published
            return (
                not self._closed.is_set()
                and published is not None
                and published.config.config_path == snapshot.config.config_path
                and published.config.section_stamp == snapshot.config.section_stamp
                and published.config.profile_data_dir
                == snapshot.config.profile_data_dir
                and published.store_path == snapshot.store_path
                and published.store_revision == snapshot.store_revision
                and scope not in self._refresh_pending
                and not any(sealed[:2] == scope for sealed in self._sealed)
            )

    def attempt_v2_configuration(
        self, read: HookAuthorityRead
    ) -> tuple[HookReviewSnapshot, tuple[()]] | None:
        """``v2_configuration()``'s answer from the attempt's read, if it is "none".

        Like ``attempt_targets``, an earlier read may only answer that there is
        nothing to authorize. A v2 grant captured from it could be stale --
        another process may have disabled, removed or revoked the handler since
        -- and an engine built around it would have that handler refused by its
        fresh authority check (blocking the Send when the handler is required)
        where a fresh read would simply not configure it. So any read that
        grants a v2 handler is left to the fresh ``v2_configuration()``.

        Args:
            read: The asking attempt's earlier full read.

        Returns:
            The read's review state with its (empty) v2 grant selection, or
            ``None`` when the read no longer stands (``attempt_read_current``)
            or grants any v2 handler, and a fresh read is required.
        """
        if not self.attempt_read_current(read) or any(
            not isinstance(target.spec, HookSpec) for target in read.targets
        ):
            return None
        return read.snapshot, ()

    def attempt_targets(
        self, read: HookAuthorityRead, event: str, tool_name: str | None
    ) -> tuple[()] | None:
        """``targets()``'s answer from the attempt's read, if it selects nothing.

        An earlier read may only answer "no hook to launch". A target selected
        from it could be stale: if another process disabled, removed or
        revoked that hook since, the target still reaches its fresh
        ``launch_guard``, whose refusal blocks a blocking event such as
        UserPromptSubmit -- where a fresh selection would simply have omitted
        the hook and the Send proceeded. Likewise a refusal derived from the
        earlier read is never issued from it. Both go to the fresh
        ``targets()`` read, which reproduces the existing selection and
        refusals exactly; only a firing with nothing to launch saves the read.

        Args:
            read: The asking attempt's earlier full read.
            event: Lifecycle event being fired.
            tool_name: Tool name for tool events, else ``None``.

        Returns:
            ``()`` when the standing read selects no target and refuses
            nothing; ``None`` when it no longer stands, selects any target or
            would refuse, and a fresh read is required.
        """
        if not self.attempt_read_current(read):
            return None
        try:
            selected = self._select(read.snapshot, read.targets, event, tool_name)
        except HookLaunchRefused:
            return None
        return None if selected else ()

    @staticmethod
    def _expect_current(
        expected: HookReviewSnapshot, current: HookReviewSnapshot
    ) -> None:
        if (
            expected.config.config_path != current.config.config_path
            or expected.config.section_stamp != current.config.section_stamp
            or expected.store_path != current.store_path
            or expected.store_revision != current.store_revision
        ):
            raise HookReviewConflict(
                "Hooks or permissions changed; review the current definitions."
            )

    def _decision(
        self, expected: HookReviewSnapshot, keys: Collection[str], approve: bool
    ) -> HookReviewSnapshot:
        owned_keys = frozenset(keys)
        with self._current() as (current, state, precondition):
            self._expect_current(expected, current)
            if state is None:
                return current
            record = state["configs"][str(current.config.config_path)]
            entries = {row.entry.key: row.entry for row in current.rows if row.entry}
            for key in owned_keys:
                row = entries.get(key)
                if row is None or (
                    approve
                    and (
                        row.spec is None
                        or row.enabled is not True
                        or self._inventory(current.config).master_enabled is not True
                    )
                ):
                    raise HookReviewConflict(
                        "Only current enabled valid definitions can be approved."
                    )
                if approve:
                    record["grants"][key] = {
                        "fingerprint": fingerprint_hook(row.spec),
                        "token": str(uuid4()),
                    }
                else:
                    record["grants"].pop(key, None)
            self._seal(current, owned_keys)
            try:
                self._write_state(current.store_path, state, precondition)
            except (OSError, ValueError, RecoveryRequired):
                return self._make_snapshot(
                    current.config,
                    current.store_path,
                    state,
                    notice="Permission save failed; affected hooks remain blocked in this app.",
                )
            self._unseal(
                current.store_path, str(current.config.config_path), owned_keys
            )
            return self._make_snapshot(
                current.config,
                current.store_path,
                state,
                notice="Hook permissions saved.",
            )

    def approve(
        self, expected: HookReviewSnapshot, keys: Collection[str]
    ) -> HookReviewSnapshot:
        """Approve only still-current selected definitions and epochs.

        Args:
            expected: Saved definitions and consent revision shown in review.
            keys: Current enabled valid definition keys explicitly selected.

        Returns:
            Updated review state, including any persistence failure fence.

        Raises:
            HookReviewConflict: The snapshot or selected definitions changed.
        """
        return self._decision(expected, keys, True)

    def revoke(self, expected: HookReviewSnapshot, key: str) -> HookReviewSnapshot:
        """Validate the review, then fence launches before persisting revoke.

        Args:
            expected: Saved definitions and consent revision shown in review.
            key: Current definition whose grant is being revoked.

        Returns:
            Updated review state; a failed write remains locally fenced.

        Raises:
            HookReviewConflict: The snapshot or selected definition changed.
        """
        return self._decision(expected, [key], False)

    def disable(self, expected: HookReviewSnapshot, key: str) -> HookReviewSnapshot:
        """Immediately disable a saved row through the canonical config writer."""
        row = next(
            (row.entry for row in expected.rows if row.entry and row.entry.key == key),
            None,
        )
        section = copy.deepcopy(expected.config.section)
        if (
            row is None
            or row.source != "hook"
            or not isinstance(section, dict)
            or not isinstance(section.get("hook"), list)
        ):
            raise HookReviewConflict(
                "Hook cannot be disabled here; repair in Settings."
            )
        raw = section["hook"][row.index]
        if not isinstance(raw, dict):
            raise HookReviewConflict("Repair the malformed entry in Settings.")
        raw["enabled"] = False
        with self._current() as (current, _state, _precondition):
            self._expect_current(expected, current)
            self._seal(current, [key])
        # The writer owns its config lock; never reacquire it under consent locks.
        result = config.replace_hooks_config_snapshot(expected.config, section)
        current = self.snapshot()
        if result.caches_reloaded:
            self._unseal(expected.store_path, str(expected.config.config_path), [key])
            notice = "Hook disabled in saved configuration."
        elif result.file_replaced:
            notice = "Hook disabled in saved configuration; runtime refresh pending. Retry refresh."
        else:
            notice = "Disable was not saved; hook remains blocked. Retry Disable or explicitly review to allow again."
        return replace(current, notice=notice)

    def save_configuration(
        self,
        expected: HookReviewSnapshot,
        replacement: Mapping[str, object],
        *,
        legacy_ids: Mapping[str, str] | None = None,
    ) -> tuple[config.LiteralConfigMutationResult, HookReviewSnapshot]:
        """Save a staged section; transfer consent only for complete unchanged legacy groups."""
        with self._current() as (current, state, _precondition):
            if (
                expected.config.config_path != current.config.config_path
                or expected.config.section_stamp != current.config.section_stamp
                or expected.store_path != current.store_path
            ):
                raise HookReviewConflict(
                    "Hook configuration changed; reload before saving."
                )
            pinned_revision = current.store_revision
            previous = (
                copy.deepcopy(state["configs"].get(str(current.config.config_path), {}))
                if state
                else {}
            )
        scope = (str(expected.store_path), str(expected.config.config_path))
        with self._cache_lock:
            self._refresh_pending.add(scope)
        # The canonical writer takes the config lock itself. Never acquire it
        # while holding the consent lock.
        result = config.replace_hooks_config_snapshot(expected.config, replacement)
        if not result.file_replaced:
            return result, replace(
                self.snapshot(), notice="Hooks were not saved. Reload before retrying."
            )
        with self._current(reconcile=False) as (current, state, precondition):
            if state is not None:
                same_owner = (
                    current.config.config_path == expected.config.config_path
                    and current.store_path == expected.store_path
                    and current.store_revision == pinned_revision
                    and current.config.section == dict(replacement)
                )
                changed = self._reconcile(state, current.config)
                if same_owner and legacy_ids:
                    old = previous.get("observed", {})
                    grants = previous.get("grants", {})
                    old_counts = Counter(
                        v["fingerprint"]
                        for k, v in old.items()
                        if k.startswith("legacy:")
                    )
                    entries = {
                        r.entry.key: r.entry
                        for r in current.rows
                        if r.entry and r.entry.spec
                    }
                    eligible = {}
                    for old_key, new_id in legacy_ids.items():
                        new_key = "id:" + new_id
                        entry = entries.get(new_key)
                        if (
                            old_key.startswith("legacy:")
                            and old_key in old
                            and new_key not in old
                            and new_key not in eligible.values()
                            and entry
                            and old[old_key]["fingerprint"]
                            == fingerprint_hook(entry.spec)
                        ):
                            eligible[old_key] = new_key
                    transferred = Counter(old[k]["fingerprint"] for k in eligible)
                    record = state["configs"][str(current.config.config_path)]
                    for old_key, new_key in eligible.items():
                        fp = old[old_key]["fingerprint"]
                        if transferred[fp] == old_counts[fp] and old_key in grants:
                            record["grants"][new_key] = {
                                "fingerprint": fp,
                                "token": str(uuid4()),
                            }
                            changed = True
                if changed:
                    try:
                        self._write_state(current.store_path, state, precondition)
                        state, _precondition, _error = self._read_state(
                            current.store_path
                        )
                    except (OSError, ValueError, RecoveryRequired):
                        state = None
                if result.caches_reloaded and state is not None:
                    with self._cache_lock:
                        self._refresh_pending.discard(scope)
            notice = (
                "Hooks saved. Enabled new or changed definitions need review before Send."
                if result.caches_reloaded and state is not None
                else "Hooks saved; runtime or permission refresh pending. Retry refresh."
            )
            return result, self._make_snapshot(
                current.config, current.store_path, state, notice=notice
            )

    def recover(self) -> HookReviewSnapshot:
        """Refresh without replaying a failed write or silently undoing a revoke."""
        result = config.refresh_runtime_config_from_cli_config()
        with self._current() as (current, state, _precondition):
            if result.caches_reloaded and state is not None:
                with self._cache_lock:
                    self._refresh_pending.discard(
                        (str(current.store_path), str(current.config.config_path))
                    )
            inventory = self._inventory(current.config)
            grants = (
                state["configs"]
                .get(str(current.config.config_path), {})
                .get("grants", {})
                if state
                else {}
            )
            safe = [
                row.entry.key
                for row in current.rows
                if row.entry
                and (
                    row.entry.enabled is False
                    or inventory.master_enabled is False
                    or row.entry.key not in grants
                )
            ]
            self._unseal(current.store_path, str(current.config.config_path), safe)
            return self._make_snapshot(current.config, current.store_path, state)

    def reset_invalid_state(self, expected: HookReviewSnapshot) -> HookReviewSnapshot:
        """Explicitly replace invalid state with zero grants and a fresh identity."""
        with self._current() as (current, state, precondition):
            self._expect_current(expected, current)
            if state is not None:
                raise HookReviewConflict(
                    "Permission state is valid; refresh instead of resetting."
                )
            if precondition.target_identity is None:
                raise HookReviewConflict(
                    "Permission file cannot be safely reset; repair its location and retry."
                )
            state = _empty_state()
            self._reconcile(state, current.config)
            self._write_state(current.store_path, state, precondition)
            return self._make_snapshot(
                current.config,
                current.store_path,
                state,
                notice="Permission state reset; enabled hooks need review.",
            )

    def _guard_reason(
        self, snapshot: HookReviewSnapshot, event: str, tool_name: str | None
    ) -> str | None:
        inventory = self._inventory(snapshot.config)
        if not inventory.requires_authority:
            return None
        if event == "UserPromptSubmit":
            return snapshot.blocked_reason
        if event != "PreToolUse":
            return None
        if (
            inventory.container_error
            or snapshot.store_revision == ("", 0)
            or self._closed.is_set()
        ):
            return "Hooks unavailable; tool dispatch refused."
        raw_rows = (
            snapshot.config.section.get("hook", [])
            if isinstance(snapshot.config.section, dict)
            else []
        )
        for row in snapshot.rows:
            entry = row.entry
            if entry is None or entry.source != "hook" or entry.enabled is False:
                continue
            if entry.spec:
                affects = entry.spec.event == event and (
                    entry.spec.matcher is None
                    or tool_name is not None
                    and entry.spec.matches_tool(tool_name)
                )
            else:
                raw = raw_rows[entry.index]
                raw_event = raw.get("event") if isinstance(raw, dict) else None
                matcher = raw.get("matcher") if isinstance(raw, dict) else None
                affects = (
                    not isinstance(raw_event, str)
                    or raw_event not in HOOK_EVENTS
                    or raw_event == event
                    and (
                        not isinstance(matcher, str)
                        or not matcher
                        or tool_name is not None
                        and fnmatch.fnmatchcase(tool_name, matcher)
                    )
                )
            if affects and (
                row.state != "approved"
                or self._is_sealed(
                    snapshot.store_path, str(snapshot.config.config_path), entry.key
                )
            ):
                return "Review or disable the affected hook before tool dispatch."
        return None

    def _select(
        self,
        snapshot: HookReviewSnapshot,
        targets: tuple[HookTarget, ...],
        event: str,
        tool_name: str | None,
    ) -> tuple[HookTarget, ...]:
        reason = self._guard_reason(snapshot, event, tool_name)
        if reason:
            raise HookLaunchRefused(reason)
        return tuple(
            target
            for target in targets
            if isinstance(target.spec, HookSpec)
            and target.spec.event == event
            and (
                target.spec.matcher is None
                or tool_name is not None
                and target.spec.matches_tool(tool_name)
            )
            and not self._is_sealed(
                snapshot.store_path, target.config_scope, target.key
            )
            and not self._closed.is_set()
        )

    def targets(self, event: str, tool_name: str | None) -> tuple[HookTarget, ...]:
        """Read authority off-thread; never drop a pending guard silently."""
        try:
            with self._current() as (snapshot, _state, _precondition):
                with self._cache_lock:
                    targets = self._published_targets
                return self._select(snapshot, targets, event, tool_name)
        except HookLaunchRefused:
            raise
        except (OSError, ValueError, RecoveryRequired):
            raise HookLaunchRefused(
                "Hook authority unavailable; dispatch refused."
            ) from None

    def v2_configuration(self) -> tuple[HookReviewSnapshot, tuple[HookTarget, ...]]:
        """Capture saved v2 definitions and exact grants in one authority read."""
        with self._current() as (snapshot, _state, _precondition):
            with self._cache_lock:
                targets = tuple(
                    target
                    for target in self._published_targets
                    if not isinstance(target.spec, HookSpec)
                )
            return snapshot, targets

    def configuration_current(self, expected: HookReviewSnapshot) -> bool:
        """Check the last worker-refreshed source without Textual-thread I/O."""
        with self._cache_lock:
            current = self._published
            return (
                not self._closed.is_set()
                and current is not None
                and current.config.config_path == expected.config.config_path
                and current.config.section_stamp == expected.config.section_stamp
                and current.store_path == expected.store_path
                and (str(current.store_path), str(current.config.config_path))
                not in self._refresh_pending
            )

    def target_current(self, target: HookTarget, *, refresh: bool = True) -> bool:
        """Check the captured epoch; cached acceptance performs no loop I/O."""
        if refresh:
            try:
                with self.launch_guard(target, tool_name=None):
                    return True
            except HookLaunchRefused:
                return False
        with self._cache_lock:
            return (
                not self._closed.is_set()
                and target in self._published_targets
                and (str(self._published.store_path), target.config_scope, target.key)
                not in self._sealed
                and (str(self._published.store_path), target.config_scope)
                not in self._refresh_pending
            )

    def notification_targets(
        self, event: str, tool_name: str | None
    ) -> tuple[HookTarget, ...]:
        """Pin current targets without I/O on the emitting thread."""
        with self._cache_lock:
            snapshot, targets = self._published, self._published_targets
        if snapshot is None:
            logger.warning("run-hooks: notification omitted (authority not published)")
            return ()
        if any(
            row.state in {"pending", "invalid", "recovery"} for row in snapshot.rows
        ):
            logger.warning(
                "run-hooks: notification omits unapproved or unavailable hooks"
            )
        return self._select(snapshot, targets, event, tool_name)

    @contextmanager
    def launch_guard(
        self, target: HookTarget, *, tool_name: str | None
    ) -> Iterator[None]:
        """Hold current config and consent authority through process creation."""
        entered = False
        try:
            with self._current() as (current, _state, _precondition):
                reason = self._guard_reason(current, target.spec.event, tool_name)
                if reason:
                    raise HookLaunchRefused(reason)
                with self._cache_lock:
                    targets = self._published_targets
                if target not in targets or self._is_sealed(
                    current.store_path, target.config_scope, target.key
                ):
                    raise HookLaunchRefused(
                        "Captured hook is disabled, changed or unapproved.",
                        skip=target.spec.event not in BLOCKING_EVENTS,
                    )
                entered = True
                yield
        except HookLaunchRefused:
            raise
        except Exception:
            if entered:
                raise
            raise HookLaunchRefused("Hook authority unavailable at launch.") from None

    def close(self) -> None:
        """Fence every later launch without waiting for existing processes."""
        self._closed.set()

    def _session_end_current(
        self,
        expected: HookReviewSnapshot,
        target: HookTarget,
        current: Callable[[], bool],
    ) -> bool:
        """Re-read the original grant for an authentic bounded host delivery."""
        try:
            with self._session_end_launch_guard(expected, target, current):
                return True
        except HookLaunchRefused:
            return False

    @contextmanager
    def _session_end_launch_guard(
        self,
        expected: HookReviewSnapshot,
        target: HookTarget,
        current: Callable[[], bool],
    ) -> Iterator[None]:
        """Validate teardown under the existing locks without reopening targets."""
        entered = False
        try:
            if (
                not current()
                or isinstance(target.spec, HookSpec)
                or target.spec.event != "SessionEnd"
                or target.spec.type != "command"
                or target.spec.effects
                or target.spec.required
            ):
                raise HookLaunchRefused("Host teardown delivery is unavailable.")
            with self._current() as (snapshot, state, _precondition):
                row = next(
                    (
                        row.entry
                        for row in snapshot.rows
                        if row.entry
                        and row.entry.key == target.key
                        and row.state == "approved"
                    ),
                    None,
                )
                grant = (
                    state["configs"]
                    .get(target.config_scope, {})
                    .get("grants", {})
                    .get(target.key)
                    if state
                    else None
                )
                if (
                    snapshot.config.config_path != expected.config.config_path
                    or snapshot.config.profile_data_dir
                    != expected.config.profile_data_dir
                    or snapshot.config.section_stamp != expected.config.section_stamp
                    or snapshot.store_path != expected.store_path
                    or target.config_scope != str(snapshot.config.config_path)
                    or row is None
                    or row.spec != target.spec
                    or grant is None
                    or grant["fingerprint"] != target.fingerprint
                    or state["store_id"] + ":" + grant["token"] != target.approval_token
                    or self._is_sealed(
                        snapshot.store_path, target.config_scope, target.key
                    )
                    or not current()
                ):
                    raise HookLaunchRefused("Captured teardown hook authority changed.")
                entered = True
                yield
        except HookLaunchRefused:
            raise
        except Exception:
            if entered:
                raise
            raise HookLaunchRefused(
                "Teardown hook authority unavailable at launch."
            ) from None
