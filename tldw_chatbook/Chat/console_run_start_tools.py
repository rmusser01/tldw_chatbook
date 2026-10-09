"""Reuse a Console run's composed MCP catalog while its inputs hold (TASK-33620.15.1).

Measured on 46c3959526 (live, Anthropic haiku, 160x45, ten sends): every run
start composed the MCP catalog on the UI loop, a 63-130 ms stretch between a
send's acceptance and its provider call. 180 of its 265 main-thread samples
were the MCP stores' storage-admission scopes working out, inside every read,
which store generation is selected and whether it is admitted; the kill switch
alone was read three times. Moving the composition to a worker hop freed the loop but made first
sends reach the provider 97-249 ms later (the hop then competed for the GIL
with the UI work the freed loop started), so it runs on the loop again.

Run start now reads the kill switch once, for the MCP and the local tools,
and keeps the composed catalog. A later run start reuses it only after
``run_catalog_fingerprint`` confirms that nothing it was composed from has
changed. That check reads each MCP store once, through the same admission
scope, path fence and recovery-readability check as ``load()`` (so a refusal
or a changed store selection fails the check as it fails a read), and
compares exact content hashes. It also compares the governance checks, the
built-in manifest's source, the composed servers' connection state, and the
run's permission profile and tool and definition maxima. Any failed check
falls through to the full composition, which meets and logs the same failure
as before. So the catalog a run is offered is the one a fresh composition
would build at the same point.

A composition is kept only when it provably read what its fingerprint
describes: the store files keep their stamps across it, each was last changed
at least ``SETTLE_NS`` before the fingerprint (storage admission's own rule
before it trusts file stamps), the admission epoch did not move, and the
servers it saw are still in the connection state it saw.
"""

from __future__ import annotations

import os
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Any

from loguru import logger

#: A store file changed less than this long before the fingerprint is not
#: trusted to keep its stamp (storage admission waits 1 s; timestamps can be
#: coarser than a write).
SETTLE_NS = 2_000_000_000


@dataclass
class LocalKillSwitchRead:
    """The run-start kill-switch read the local tools use.

    ``read`` stays False when no MCP composition ran (no MCP service, or a
    caller that composes differently); the local provider then reads the
    switch itself.
    """

    read: bool = False
    value: bool = False
    error: Exception | None = None


@dataclass(frozen=True)
class _Entry:
    service: Any  # weakref.ref to the MCP service
    identity: tuple  # store and manifest identities, stamps left out
    key: tuple  # the provider's own composition inputs
    servers: tuple  # ((profile_id, plugin_owned), ...) the catalog used
    connected: tuple  # their connection state when composed
    engaged: bool  # the kill switch was on: no MCP, no local tools
    composition: tuple | None


_lock = threading.Lock()
_entry: _Entry | None = None


def forget() -> None:
    """Drop the kept composition (tests; a new service never matches)."""
    global _entry
    with _lock:
        _entry = None


def _identity(fingerprint: tuple) -> tuple:
    """The fingerprint without file stamps and connection states."""
    permission, (local_store, manifest, _connected) = fingerprint
    return (
        None if permission is None else permission[:2],
        local_store[:2],
        manifest,
    )


def _stamps(fingerprint: tuple, service: Any) -> tuple:
    """``(path, stamp)`` per store file the fingerprint read (None: no file)."""
    permission, (local_store, _manifest, _connected) = fingerprint
    pairs = []
    for store, identity in (
        (getattr(service, "permission_store", None), permission),
        (getattr(service.local_service, "store", None), local_store),
    ):
        if identity is None:
            continue
        if identity[0] not in {"file", "missing", "fresh"} or store is None:
            raise ValueError("unstamped_store_state")
        pairs.append((store.path, identity[2] if identity[0] == "file" else None))
    return tuple(pairs)


def _stamp(path) -> tuple | None:
    try:
        info = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _admission_epoch() -> Any:
    from tldw_chatbook.Backup_Recovery import bootstrap

    return getattr(bootstrap, "_admission_epoch", None)


def _fill(local_read: LocalKillSwitchRead | None, engaged: bool, error=None) -> None:
    if local_read is not None:
        local_read.value, local_read.error, local_read.read = engaged, error, True


async def compose_run_mcp_provider(
    service: Any,
    provider: Any,
    local_read: LocalKillSwitchRead | None = None,
) -> Any | None:
    """Compose (or reuse) this run's MCP catalog, on the UI loop.

    Args:
        service: The app's unified MCP service.
        provider: The run's uncomposed provider (built with the UI loop).
        local_read: Filled with the kill switch this run start decided on,
            for the local tools (one read decides MCP and local tools).

    Returns:
        The composed provider, or None when MCP is not offered this run: the
        kill switch is on, or a read failed (fail closed, logged as before).
    """
    with _lock:
        entry = _entry
    if entry is not None and entry.service() is not service:
        entry = None
    servers = entry.servers if entry is not None else ()
    observed_at = time.time_ns()
    epoch = _admission_epoch()
    fingerprint = None
    try:
        fingerprint = await service.run_catalog_fingerprint(servers)
    except Exception as caught:  # noqa: BLE001 -- the reads below meet any refusal
        logger.debug(
            "Console MCP catalog check failed ({}); composing from the stores",
            type(caught).__name__,
        )
    key = provider.composition_key()
    if (
        entry is not None
        and fingerprint is not None
        and _identity(fingerprint) == entry.identity
        and fingerprint[1][2] == entry.connected
        and key == entry.key
    ):
        _fill(local_read, entry.engaged)
        if entry.engaged:
            return None
        provider.install_composition(entry.composition)
        return provider

    error = None
    try:
        engaged = bool(service.get_kill_switch())
    except Exception as caught:  # noqa: BLE001 -- fail closed to "no MCP this run"
        logger.opt(exception=caught).warning(
            "ConsoleChatController: get_kill_switch failed; skipping MCP this run"
        )
        engaged, error = True, caught
    _fill(local_read, engaged, error)
    if not engaged:
        try:
            await provider.compose_catalog(kill_switch_engaged=False)
        except Exception:  # noqa: BLE001 -- a composition failure must not abort the send
            logger.opt(exception=True).warning(
                "ConsoleChatController: MCP compose_catalog failed; skipping MCP this run"
            )
            return None
    if error is None and fingerprint is not None:
        try:
            _keep(service, provider, fingerprint, key, engaged, observed_at, epoch)
        except Exception as caught:  # noqa: BLE001 -- not keeping is always safe
            logger.debug("Console MCP catalog not kept ({})", type(caught).__name__)
    return None if engaged else provider


def _keep(service, provider, fingerprint, key, engaged, observed_at, epoch) -> None:
    """Keep the composition if it provably read what ``fingerprint`` describes."""
    global _entry
    stamps = _stamps(fingerprint, service)
    settled = all(
        stamp is None or stamp[4] <= observed_at - SETTLE_NS for _path, stamp in stamps
    )
    if not settled or any(_stamp(path) != stamp for path, stamp in stamps):
        return
    if _admission_epoch() != epoch:
        return
    composed = () if engaged else getattr(provider, "composed_servers", None)
    if composed is None:
        return
    servers = tuple((profile_id, owned) for profile_id, owned, _seen in composed)
    seen = tuple(connected for _id, _owned, connected in composed)
    if service.local_service.connection_states(servers) != seen:
        return
    with _lock:
        _entry = _Entry(
            service=weakref.ref(service),
            identity=_identity(fingerprint),
            key=key,
            servers=servers,
            connected=seen,
            engaged=engaged,
            composition=None if engaged else provider.composition(),
        )


def local_tools_kill_switch(
    service: Any, local_read: LocalKillSwitchRead | None
) -> bool:
    """Whether the local tools are switched off this run (fail closed).

    Uses the MCP composition's decision when it ran; otherwise reads the
    switch here, as the local provider always did.
    """
    error = None
    if local_read is not None and local_read.read:
        error, engaged = local_read.error, local_read.value
    else:
        try:
            engaged = bool(service.get_kill_switch())
        except Exception as caught:  # noqa: BLE001
            error = caught
    if error is not None:
        logger.opt(exception=error).warning(
            "ConsoleChatController: get_kill_switch failed; skipping local tools this run"
        )
        return True
    return engaged
