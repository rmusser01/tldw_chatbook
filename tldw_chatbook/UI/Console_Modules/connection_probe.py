"""Probe one provider connection and settle the result (TASK-33005.2).

The single "test this connection" path the Console shares: Chat settings'
connection test (through ``ChatScreen._test_console_connection``), the
Console's one-action ``retry_connection`` recovery and the switcher's local
probe (TASK-33005.5). A settled result goes to the app's shared evidence
owner, so every surface reading that connection sees it.

Imported lazily: nothing here runs before the first probe, so it stays out
of the boot (``_ui_ready``) module census.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from ipaddress import ip_address
from typing import Any
from urllib.parse import urlsplit
from weakref import WeakKeyDictionary

from loguru import logger

from tldw_chatbook.Chat.console_session_settings import (
    _endpoint_failure_blocker,
    build_target_default_console_session_settings,
    console_send_connection,
)
from tldw_chatbook.Chat.provider_endpoint_contract import (
    URL_BASED_PROVIDER_KEYS,
    ConnectionProbeAvailability,
    canonical_connection_identity,
    connection_probe_availability,
)
from tldw_chatbook.Chat.provider_readiness import (
    KEYLESS_PROVIDER_KEYS,
    get_provider_readiness,
)
from tldw_chatbook.Chat.provider_test_evidence import (
    ProviderConnectionEvidence,
    ProviderDraftIdentity,
    ProviderProbeResult,
    ProviderTestEvidence,
    ProviderTestEvidenceStore,
    connection_credential_revision,
    shared_connection_evidence,
)

#: TASK-33005.5 (spec §5, "start it; rechecked on open"): a switcher probe, or
#: any test, younger than this is reused. It outlasts one probe (two 2.5 s
#: requests at most) and a double Alt+M, and is short enough that a server
#: the user has just started reads its new state on the next open.
SWITCHER_PROBE_CACHE_SECONDS = 10.0
#: At most this many switcher probes in flight for one open (AC#2).
SWITCHER_PROBES_IN_FLIGHT = 3
#: The only providers a switcher open may contact: keyless and URL-based.
#: Not qwencloud (a keyed cloud) nor the in-process local providers.
_SWITCHER_PROBE_PROVIDER_KEYS = KEYLESS_PROVIDER_KEYS & URL_BASED_PROVIDER_KEYS
#: Per evidence owner: when each connection's last switcher probe was queued.
_SWITCHER_PROBE_STARTS: WeakKeyDictionary[
    ProviderConnectionEvidence, dict[ProviderDraftIdentity, float]
] = WeakKeyDictionary()


async def probe_console_connection(
    identity: ProviderDraftIdentity,
    *,
    app_config: Mapping[str, object] | None = None,
) -> ProviderProbeResult:
    """Run the bounded, non-generating model-catalog probe for one connection.

    The probe carries the credential a send would use, the one whose digest
    ``identity.credential_revision`` is (ADR-012: same destination, same
    credential). A keyless probe of a keyed server reads 401, which the
    Console would show as "key rejected" for a key that was never sent.

    Args:
        identity: The exact connection to probe.
        app_config: Config holding the connection's saved credential.

    Returns:
        The bounded probe result.
    """
    from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
        SettingsEndpointProbePurpose,
        probe_settings_endpoint,
        provider_probe_result_from_settings_outcome,
    )

    probe_kwargs = {}
    if identity.custom_endpoint_id is None:
        api_key = get_provider_readiness(
            identity.provider_key, app_config or {}, background_credentials=True
        ).api_key
        if connection_credential_revision(api_key) != identity.credential_revision:
            # The saved credential changed since this identity was keyed.
            return ProviderProbeResult("unreachable", (), "connection_error")
        if api_key:
            probe_kwargs["api_key"] = api_key
    else:
        from tldw_chatbook.Chat.custom_endpoint_registry import (
            entry_for,
            family_execution_key,
            resolve_entry_credential,
        )

        entry = entry_for(app_config or {}, identity.custom_endpoint_id)
        if (
            entry is None
            or canonical_connection_identity(
                family_execution_key(entry.family), entry.base_url
            )
            != identity.connection_identity
        ):
            return ProviderProbeResult("unreachable", (), "connection_error")
        probe_kwargs["api_key"] = resolve_entry_credential(entry)[0]
    outcome = await probe_settings_endpoint(
        identity.connection_identity[1],
        provider=identity.provider_key,
        purpose=SettingsEndpointProbePurpose.CHAT_CATALOG,
        **probe_kwargs,
    )
    return provider_probe_result_from_settings_outcome(outcome)


async def settle_connection_probe(
    app: object,
    identity: ProviderDraftIdentity,
    *,
    app_config: Mapping[str, object] | None = None,
) -> ProviderTestEvidence | None:
    """Probe ``identity`` and settle the result into ``app``'s shared owner.

    A probe begun later always wins over one begun earlier, whichever settles
    first (the owner orders by begin, TASK-33005.1).

    Args:
        app: The running app the shared evidence owner is attached to.
        identity: The exact connection to probe.
        app_config: Passed to :func:`probe_console_connection`.

    Returns:
        The settled evidence, or ``None`` if it could not be settled.
    """
    store = ProviderTestEvidenceStore(lambda: app)
    token = store.begin(identity)
    store.settle(token, await probe_console_connection(identity, app_config=app_config))
    return store.evidence_for(identity)


async def retry_console_connection(screen: Any) -> None:
    """Re-test the active chat's failed connection: one action, no settings.

    The ``retry_connection`` recovery means the server did not answer, which
    the user fixes outside the app; opening Chat settings made it three
    steps. The probe runs in a worker and settles into the shared owner,
    whose version the Console's idle poll watches, so readiness refreshes
    once. A cloud key check (no models route, not URL-based) is re-run only
    by Settings 't' (D2), so it opens Providers & Models there; any other
    connection with no models route still opens Chat settings.

    Args:
        screen: The Console ``ChatScreen``.
    """
    settings, readiness = screen._active_console_settings_readiness()
    identity = readiness.connection
    if identity is None or (
        connection_probe_availability(
            identity.provider_key, identity.connection_identity[1]
        )
        is not ConnectionProbeAvailability.MODELS_ROUTE
    ):
        if (
            identity is not None
            and identity.custom_endpoint_id is None
            and identity.provider_key not in URL_BASED_PROVIDER_KEYS
        ):
            from .model_switcher import open_provider_setup

            open_provider_setup(screen, identity.provider_key, settings.model)
            screen.app.notify(
                f"Press t to test {readiness.provider_display_name or 'it'} again."
            )
            return
        await screen._open_console_settings(focus_model=False)
        return
    screen.run_worker(
        _retry(
            screen.app,
            identity,
            readiness.provider_display_name or "The provider",
            screen._provider_readiness_app_config(),
        ),
        group="console-connection-retry",
        exclusive=True,
    )


async def _retry(
    app: Any,
    identity: ProviderDraftIdentity,
    provider: str,
    app_config: Mapping[str, object],
) -> None:
    evidence = await settle_connection_probe(app, identity, app_config=app_config)
    # Only a still-down server: a rejected key or a bad route moves the
    # Console to its own recovery ("Configure API key", "Review settings").
    if (
        evidence is not None
        and evidence.endpoint == "unreachable"
        and _endpoint_failure_blocker(evidence.category)[1] == "retry_connection"
    ):
        app.notify(
            f"{provider} is still unreachable. Start it, then retry.",
            severity="warning",
        )


def switcher_probe_target(identity: ProviderDraftIdentity | None) -> bool:
    """Whether opening Switch model may probe ``identity`` on its own (AC#4).

    Only a keyless, URL-based endpoint with a models route that sends no
    credential, on a loopback or private-network address (D2: no cloud or
    public host is contacted automatically, and no key is ever sent).

    Args:
        identity: The connection a send uses, or ``None``.

    Returns:
        Whether an automatic probe is allowed.
    """
    if (
        identity is None
        or identity.provider_key not in _SWITCHER_PROBE_PROVIDER_KEYS
        or identity.credential_revision
    ):
        return False
    endpoint = identity.connection_identity[1]
    if (
        connection_probe_availability(identity.provider_key, endpoint)
        is not ConnectionProbeAvailability.MODELS_ROUTE
    ):
        return False
    host = urlsplit(endpoint).hostname or ""
    if host == "localhost":
        return True
    try:
        address = ip_address(host)
    except ValueError:
        # ponytail: host names are never resolved here, so a LAN name is not
        # probed; resolve it off-thread first if that is ever wanted.
        return False
    address = getattr(address, "ipv4_mapped", None) or address
    return address.is_loopback or (address.is_private and not address.is_link_local)


def switcher_probe_plan(
    app_config: Mapping[str, object], targets: Mapping[str, str | None]
) -> dict[ProviderDraftIdentity, list[str]]:
    """Map each listed provider to the connection a probe may check.

    Blocking (config and credential reads): run it in a worker thread.

    Args:
        app_config: The live configuration snapshot.
        targets: Provider key -> the model its switcher row resolves.

    Returns:
        Probe-eligible connection -> the providers whose rows it backs.
    """
    plan: dict[ProviderDraftIdentity, list[str]] = {}
    for provider, model in targets.items():
        if provider not in _SWITCHER_PROBE_PROVIDER_KEYS and not provider.startswith(
            "custom-ep:"
        ):
            continue
        try:
            identity = console_send_connection(
                build_target_default_console_session_settings(
                    app_config, provider, model
                ),
                app_config=app_config,
            )
        except Exception as exc:  # noqa: BLE001 - one bad provider must not stop the rest
            logger.debug("Switcher probe skipped {}: {}", provider, type(exc).__name__)
            continue
        if switcher_probe_target(identity):
            plan.setdefault(identity, []).append(provider)
    return plan


def _recently_tested(evidence: ProviderTestEvidence | None) -> bool:
    observed = evidence.observed_at if evidence is not None else None
    # A clock stepped back must not make old evidence look fresh forever.
    return observed is not None and (
        0
        <= (datetime.now().astimezone() - observed).total_seconds()
        < SWITCHER_PROBE_CACHE_SECONDS
    )


def switcher_connection_prober(
    app: object, app_config: Mapping[str, object]
) -> Callable[[Mapping[str, str | None], Callable[[str], None]], Awaitable[None]]:
    """Return one switcher open's ``connection_prober`` (TASK-33005.5).

    The switcher hands it its listed providers; it probes the local ones
    (:func:`switcher_probe_target`) that no result younger than
    ``SWITCHER_PROBE_CACHE_SECONDS`` covers, at most
    ``SWITCHER_PROBES_IN_FLIGHT`` at a time, each in a worker thread with
    the bounded probe, settles each into the shared owner (a probe begun
    earlier never replaces a newer result) and calls ``settled(provider)``
    for each row to re-read. Cancelled with the switcher, a running probe
    still settles.

    Args:
        app: The running app the shared evidence owner is attached to.
        app_config: The configuration snapshot the switcher opened with.

    Returns:
        The seam the switcher runs in its worker.
    """
    gate = asyncio.Semaphore(SWITCHER_PROBES_IN_FLIGHT)

    async def probe(
        targets: Mapping[str, str | None], settled: Callable[[str], None]
    ) -> None:
        owner = shared_connection_evidence(lambda: app)
        if owner is None:
            return
        plan = await asyncio.to_thread(switcher_probe_plan, app_config, targets)
        started = _SWITCHER_PROBE_STARTS.setdefault(owner, {})
        now = time.monotonic()
        due = {
            identity: providers
            for identity, providers in plan.items()
            if now - started.get(identity, float("-inf")) >= SWITCHER_PROBE_CACHE_SECONDS
            and not _recently_tested(owner.evidence_for(identity))
        }
        # Marked when queued, so a reopen never sends a second probe for one
        # still waiting or in flight (AC#3). ponytail: one cancelled while
        # queued (switcher closed) waits out the window; mark at start if
        # more than SWITCHER_PROBES_IN_FLIGHT local servers become common.
        started.update(dict.fromkeys(due, now))

        async def one(identity: ProviderDraftIdentity, providers: list[str]) -> None:
            async with gate:
                try:
                    await asyncio.to_thread(
                        asyncio.run,
                        settle_connection_probe(app, identity, app_config=app_config),
                    )
                except Exception as exc:  # noqa: BLE001 - the row keeps its word
                    logger.debug(
                        "Switcher probe of {} failed: {}",
                        identity.provider_key,
                        type(exc).__name__,
                    )
                    return
            for provider in providers:
                settled(provider)

        await asyncio.gather(*(one(identity, due[identity]) for identity in due))

    return probe
