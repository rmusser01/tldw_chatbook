"""Local exact-definition consent for standalone Console hooks (ADR-197)."""

from __future__ import annotations

import copy
import fnmatch
import json
import re
import threading
from collections import Counter
from collections.abc import Collection, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
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
    HookTarget,
    fingerprint_hook,
    inspect_hooks_config,
)
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Utils.private_paths import (
    PrivateFileWritePrecondition,
    atomic_private_write_text,
    create_private_text,
    open_private_binary,
    open_private_text_append_stream,
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


def default_hook_permissions_path() -> Path:
    """Resolve the live canonical profile directory, never a startup constant."""
    return Path(config.get_user_data_dir()) / "hook_permissions.json"


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

    @contextmanager
    def _store_lock(self, path: Path) -> Iterator[None]:
        from tldw_chatbook.Backup_Recovery import raw_participants

        # Retain config's lease and writer locks while independently admitting
        # this concrete store; config helpers keep their original narrow scope.
        with raw_participants._scope(
            self, "hook_permissions", writing=True, selected_read=path
        ):
            lock_path = path.with_name(path.name + ".lock")
            try:
                create_private_text(
                    lock_path, "", application_owned_directory=path.parent
                )
            except FileExistsError:
                pass
            stream = open_private_text_append_stream(
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
        for row in inventory.rows:
            if row.spec is None:
                continue
            fingerprint = fingerprint_hook(row.spec)
            prior = previous.get(row.key)
            change = "Existing" if first else "New"
            if prior:
                change = prior["change"]
                if prior["fingerprint"] != fingerprint:
                    change = "Modified" if row.key.startswith("id:") else "New"
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
        for row in inventory.rows:
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
                path = default_hook_permissions_path()
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
        if not approve:
            self._seal(expected, owned_keys)
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
        """Approve only still-current selected definitions and epochs."""
        return self._decision(expected, keys, True)

    def revoke(self, expected: HookReviewSnapshot, key: str) -> HookReviewSnapshot:
        """Seal immediately; do not report durable revoke before saving."""
        return self._decision(expected, [key], False)

    def disable(self, expected: HookReviewSnapshot, key: str) -> HookReviewSnapshot:
        """Immediately disable a saved row through the canonical config writer."""
        self._seal(expected, [key])
        row = next(
            (row.entry for row in expected.rows if row.entry and row.entry.key == key),
            None,
        )
        section = copy.deepcopy(expected.config.section)
        if (
            row is None
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
        result = config.replace_hooks_config_snapshot(expected.config, section)
        current = self.snapshot()
        if result.caches_reloaded:
            self._unseal(expected.store_path, str(expected.config.config_path), [key])
            notice = "Hook disabled in saved configuration."
        elif result.file_replaced:
            notice = "Hook disabled in saved configuration; runtime refresh pending. Retry refresh."
        else:
            notice = "Disable was not saved; hook remains blocked in this app. Reload and retry."
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
            if entry is None or entry.enabled is False:
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
            if target.spec.event == event
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
