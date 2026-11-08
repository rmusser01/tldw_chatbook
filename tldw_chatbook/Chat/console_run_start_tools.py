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
before it trusts file stamps), the admission epoch did not move, the servers
it saw are still in the connection state it saw, and its key is still the one
it read after its await. Only a plain ``MCPToolProvider`` is reused or kept:
a subclass (a plugin provider) composes from more than the key covers.

A run start with nothing to reuse (no kept composition, or a run that
freezes its definition ceiling at composition and so never reuses) takes no
fingerprint before it composes (TASK-33620.15.38): there it decides nothing
for the run, and a live Console send always composed after it. Such a run
over stock sources (the installed stores and the default manifest provider)
takes an anchor instead -- the store files' stamps, the admission epoch and
the selected sources, before the composition reads -- and keeps the
composition as pending; a run over custom readers or a custom manifest
provider still takes the fingerprint before it composes. The next run start's own fingerprint proves it: the
same sources, every file still at its anchored stamp and settled before the
anchor, the epoch unmoved. Only then does that fingerprint's identity become
the kept identity; anything else refuses the pending composition and the run
composes from the stores.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
import threading
import time
import weakref
from dataclasses import dataclass, replace
from typing import Any

from loguru import logger

from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider

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
    _include_mcp: bool = True
    _need_local: bool = True
    _share: bool = True
    _current: Any = None


def _require_current(local_read):
    if local_read is not None and local_read._current is not None:
        local_read._current()


def capture_run_tool_read(controller, session_id, turn_context, project_guard, *, plugins):
    """Attach demand/currentness to the existing run-local switch observation."""
    result = LocalKillSwitchRead(_share=False)
    if plugins:
        return result
    from tldw_chatbook import config
    from tldw_chatbook.Agents import mcp_tool_provider as providers
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.MCP import console_snapshot as snapshot

    from . import console_chat_controller as owner

    methods = owner._CONSOLE_TOOL_COMPOSITION_METHODS

    def original_methods():
        for name, original, code, namespace, defaults, keywords, items, closure in methods:
            if (
                inspect.getattr_static(controller, name, None) is not original
                or vars(owner.ConsoleChatController).get(name) is not original
                or original.__code__ is not code
                or original.__globals__ is not namespace
                or namespace is not vars(owner)
                or original.__closure__ is not closure
                or original.__defaults__ is not defaults
                or original.__kwdefaults__ is not keywords
                or len(keywords or {}) != len(items)
                or any((keywords or {}).get(key) is not value for key, value in items)
            ):
                return False
        return True

    app = controller.app
    service = getattr(app, "unified_mcp_service", None)
    if (
        not original_methods()
        or not providers.controller_factory_current(owner.MCPToolProvider)
        or not snapshot.standard_console_catalog_sources(service)
    ):
        return result
    app_config = getattr(app, "app_config", None)
    store = controller.store
    session = next((item for item in store.sessions() if item.id == session_id), None)
    revision = store.session_settings_revision(session_id) if session is not None else None
    scratch = controller._scratch_spaces
    identity = config.current_config_identity()

    def current():
        if (
            not original_methods()
            or not providers.controller_factory_current(owner.MCPToolProvider)
            or controller.app is not app
            or getattr(app, "app_config", None) is not app_config
            or controller.store is not store
            or controller._scratch_spaces is not scratch
            or getattr(app, "unified_mcp_service", None) is not service
            or sys.modules.get("tldw_chatbook.config") is not config
            or config.current_config_identity() != identity
            or next((item for item in store.sessions() if item.id == session_id), None) is not session
            or (session is not None and store.session_settings_revision(session_id) != revision)
            or (project_guard is not None and not project_guard())
        ):
            raise RecoveryRequired("console_snapshot_owner_changed")

    maximum = turn_context.mcp_tool_maximum if turn_context is not None else None
    enabled = (
        turn_context.tool_configuration.get("local_tools_enabled", owner.LOCAL_TOOLS_DEFAULT_ENABLED)
        if turn_context is not None else
        owner.get_cli_setting("console", "local_tools_enabled", owner.LOCAL_TOOLS_DEFAULT_ENABLED)
    )
    result._include_mcp = not (type(maximum) is frozenset and not maximum)
    result._need_local = owner.coerce_bool_setting(enabled, owner.LOCAL_TOOLS_DEFAULT_ENABLED)
    result._share = True
    result._current = current
    current()
    return result


@dataclass(frozen=True)
class _Entry:
    service: Any  # weakref.ref to the MCP service
    identity: tuple | None  # store and manifest identities, stamps left out
    key: tuple  # the provider's own composition inputs
    servers: tuple  # ((profile_id, plugin_owned), ...) the catalog used
    connected: tuple  # their connection state when composed
    engaged: bool  # the kill switch was on: no MCP, no local tools
    composition: tuple | None
    anchor: "_Anchor | None" = None  # pending (identity None) until proven


@dataclass(frozen=True)
class _Anchor:
    """What a composition with no fingerprint saw before it read (TASK-33620.15.38).

    The next run start's fingerprint proves the composition only if it read
    through the same sources, every store file still has the stamp anchored
    here and was last changed ``SETTLE_NS`` before ``observed_at``, and the
    admission epoch has not moved: then nothing the composition read changed
    between its read and that fingerprint's.
    """

    stamps: tuple  # ((path, stamp or None), ...) as ``_stamps`` orders them
    observed_at: int
    epoch: Any
    sources: tuple  # weak refs, as ``_sources`` orders them


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


def _settled(stamps: tuple, observed_at: int) -> bool:
    """Every file was last changed ``SETTLE_NS`` before ``observed_at``."""
    return all(
        stamp is None or stamp[4] <= observed_at - SETTLE_NS for _path, stamp in stamps
    )


def _sources(service: Any) -> tuple:
    """The receivers a composition and a fingerprint read through, now.

    The manifest provider is one of them: the fingerprint names the built-in
    manifest by the default provider's source, so a pending composition is
    proven only while that provider stays selected.
    """
    local = service.local_service
    return (
        service.permission_store,
        local,
        getattr(local, "store", None),
        getattr(local, "manifest_provider", None),
    )


def _same_sources(refs: tuple, sources: tuple) -> bool:
    return len(refs) == len(sources) and all(
        (ref is None and source is None)
        or (ref is not None and source is not None and ref() is source)
        for ref, source in zip(refs, sources, strict=True)
    )


def _take_anchor(service: Any, observed_at: int, epoch: Any) -> _Anchor | None:
    """Anchor a composition that runs without a fingerprint (TASK-33620.15.38).

    Only stock sources are anchored: a run with custom readers or a custom
    manifest provider keeps the fingerprint before its composition. The
    governance checks the fingerprint makes first are made here too, so a
    refused listing anchors nothing (the fingerprint route then meets the
    same refusal). The stamps are two ``stat`` calls, not admission scopes.

    Returns:
        The anchor, or None to take the fingerprint instead.
    """
    from tldw_chatbook.MCP import console_snapshot as snapshot

    try:
        permission, local, store, _manifest = sources = _sources(service)
        if not snapshot._stock_fingerprint_sources(service, local, store, permission):
            return None
        snapshot._require_listing(local)
        stamps = tuple(
            (source.path, _stamp(source.path)) for source in (permission, store)
        )
    except Exception:  # noqa: BLE001 -- the fingerprint route decides then
        return None
    return _Anchor(
        stamps=stamps,
        observed_at=observed_at,
        epoch=epoch,
        sources=tuple(weakref.ref(source) for source in sources),
    )


def _proven(entry: _Entry, fingerprint: tuple, service: Any) -> _Entry | None:
    """The pending ``entry`` with ``fingerprint``'s identity, if it proves it.

    ``fingerprint`` was read after the pending composition; it describes what
    that composition read only if the same sources were read, every store
    file kept the stamp anchored before the composition, and no admission or
    activation record was written in between. (``_keep`` kept the pending
    composition only if those stamps were settled at the anchor.)
    """
    anchor = entry.anchor
    try:
        stamps = _stamps(fingerprint, service)
        sources = _sources(service)
    except Exception:  # noqa: BLE001 -- an unstamped store proves nothing
        return None
    if (
        stamps != anchor.stamps
        or _admission_epoch() != anchor.epoch
        or not _same_sources(anchor.sources, sources)
    ):
        return None
    return replace(entry, identity=_identity(fingerprint), anchor=None)


def _settle_pending(entry: _Entry, proven: _Entry | None) -> None:
    """Replace the pending entry with its proof, or drop it; never a newer one."""
    global _entry
    with _lock:
        if _entry is entry:
            _entry = proven


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
    _require_current(local_read)
    if local_read is not None and not local_read._include_mcp:
        if not local_read._need_local:
            _fill(local_read, False)
            return None
        from tldw_chatbook.MCP.console_snapshot import read_console_kill_switch

        try:
            engaged = await read_console_kill_switch(service)
        except Exception as caught:  # noqa: BLE001 - preserve optional catalog refusal and the shared permission result.
            _require_current(local_read)
            _fill(local_read, True, caught)
        else:
            _require_current(local_read)
            _fill(local_read, bool(engaged))
        return None
    with _lock:
        entry = _entry
    if entry is not None and entry.service() is not service:
        entry = None
    deferred = getattr(provider, "_composition_definition_ceiling", False)
    if deferred:
        entry = None  # Each execution consumer establishes its own fresh ceiling.
    servers = entry.servers if entry is not None else ()
    observed_at = time.time_ns()
    epoch = _admission_epoch()
    plain = type(provider) is MCPToolProvider  # Anything else always composes.
    fingerprint = anchor = None
    if plain and entry is None:
        # TASK-33620.15.38: nothing to reuse, so no fingerprint decides this
        # run; the next run start's fingerprint proves the composition.
        anchor = _take_anchor(service, observed_at, epoch)
    try:
        if plain and anchor is None:
            fingerprint = await service.run_catalog_fingerprint(servers)
    except Exception as caught:  # noqa: BLE001 -- the reads below meet any refusal
        logger.debug(
            "Console MCP catalog check failed ({}); composing from the stores",
            type(caught).__name__,
        )
    _require_current(local_read)
    key = provider.composition_key()
    if plain and entry is not None and entry.anchor is not None:
        pending = entry
        entry = None if fingerprint is None else _proven(pending, fingerprint, service)
        _settle_pending(pending, entry)
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

    # Keep the current fingerprint/reuse owner. On a qualified stock cold
    # composition, one detached payload supplies MCP and the local switch.
    # Custom producers continue through their original public call shape.
    prepared = None
    if plain and (local_read is None or local_read._share):
        from tldw_chatbook.Agents import mcp_tool_provider as provider_source
        from tldw_chatbook.MCP import console_tool_preparation as preparation_source

        if (
            provider_source.controller_factory_current(type(provider))
            and provider_source.console_preparation_pipeline_current(preparation_source)
        ):
            try:
                prepared = await preparation_source.prepare_console_tools(
                    service,
                    profile_id=provider._profile_id(),
                    include_mcp_catalog=True,
                    need_local_switch=local_read is not None,
                    builtin_raw_name_exclusions=provider._builtin_raw_name_exclusions,
                    owned_profile_ids=provider._owned_profile_ids,
                    maximum_tool_ids=provider._maximum_tool_ids,
                )
                _require_current(local_read)
                if prepared is not None and not prepared.kill_switch:
                    composition = provider_source.capture_standard_controller_composition(
                        type(provider), service
                    )
                    if composition is None:
                        raise PermissionError("mcp_catalog_source_changed")
                    provider.adopt_console_preparation(
                        prepared, _controller_composition=composition
                    )
            except Exception as caught:  # noqa: BLE001 - preserve optional catalog refusal and the shared permission result. same optional catalog failure policy
                _require_current(local_read)
                if deferred:
                    raise
                _fill(local_read, True, caught)
                logger.opt(exception=caught).warning(
                    "ConsoleChatController: MCP preparation failed; skipping MCP this run"
                )
                return None

    if prepared is None:
        # TASK-33620.15.5: the kill switch below is a store read on the loop,
        # like the fingerprint's two; give the UI loop a turn between them.
        await asyncio.sleep(0)
        _require_current(local_read)
    error = None
    try:
        engaged = prepared.kill_switch if prepared is not None else bool(service.get_kill_switch())
    except Exception as caught:  # noqa: BLE001 -- fail closed to "no MCP this run"
        logger.opt(exception=caught).warning(
            "ConsoleChatController: get_kill_switch failed; skipping MCP this run"
        )
        engaged, error = True, caught
    _fill(local_read, engaged, error)
    if not engaged and prepared is None:
        await asyncio.sleep(0)  # and between it and the composition's reads
        _require_current(local_read)
        try:
            compose = provider.compose_catalog
            await (compose(kill_switch_engaged=False) if plain else compose())
        except Exception:  # noqa: BLE001 -- a composition failure must not abort the send
            if deferred:
                raise
            logger.opt(exception=True).warning(
                "ConsoleChatController: MCP compose_catalog failed; skipping MCP this run"
            )
            return None
    # The composition read its key after an await: keep it only under that key.
    if (
        error is None
        and (fingerprint is not None or anchor is not None)
        and provider.composition_key() == key
    ):
        try:
            _keep(
                service, provider, fingerprint, key, engaged, observed_at, epoch, anchor
            )
        except Exception as caught:  # noqa: BLE001 -- not keeping is always safe
            logger.debug("Console MCP catalog not kept ({})", type(caught).__name__)
    return None if engaged else provider


def _keep(
    service, provider, fingerprint, key, engaged, observed_at, epoch, anchor=None
) -> None:
    """Keep the composition if it provably read what ``fingerprint`` describes.

    With ``anchor`` (no fingerprint before the composition, TASK-33620.15.38)
    the composition is kept pending: the next run start's fingerprint proves
    it (``_proven``) or refuses it. What can already refuse it is checked now:
    a store file not settled at the anchor, a moved admission epoch, another
    source selected, a refused listing.
    """
    global _entry
    if anchor is None:
        stamps = _stamps(fingerprint, service)
        if not _settled(stamps, observed_at) or any(
            _stamp(path) != stamp for path, stamp in stamps
        ):
            return
    else:
        from tldw_chatbook.MCP import console_snapshot as snapshot

        if not _settled(anchor.stamps, observed_at) or not _same_sources(
            anchor.sources, _sources(service)
        ):
            return
        snapshot._require_listing(service.local_service)
    if _admission_epoch() != epoch:
        return
    composed = () if engaged else getattr(provider, "composed_servers", None)
    if composed is None:
        return
    servers = tuple((profile_id, owned) for profile_id, owned, _seen in composed)
    seen = tuple(connected for _id, _owned, connected in composed)
    # Defense in depth: reuse compares these too, but never keep a stale view.
    if service.local_service.connection_states(servers) != seen:
        return
    with _lock:
        _entry = _Entry(
            service=weakref.ref(service),
            identity=None if anchor is not None else _identity(fingerprint),
            key=key,
            servers=servers,
            connected=seen,
            engaged=engaged,
            composition=None if engaged else provider.composition(),
            anchor=anchor,
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
