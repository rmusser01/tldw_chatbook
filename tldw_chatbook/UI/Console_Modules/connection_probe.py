"""Probe one provider connection and settle the result (TASK-33005.2).

The single "test this connection" path the Console shares: Chat settings'
connection test (through ``ChatScreen._test_console_connection``), the
Console's one-action ``retry_connection`` recovery and, later, the switcher's
local probe. A settled result goes to the app's shared evidence owner, so
every surface reading that connection sees it.

Imported lazily: nothing here runs before the first probe, so it stays out
of the boot (``_ui_ready``) module census.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tldw_chatbook.Chat.provider_endpoint_contract import (
    ConnectionProbeAvailability,
    canonical_connection_identity,
    connection_probe_availability,
)
from tldw_chatbook.Chat.provider_test_evidence import (
    ProviderDraftIdentity,
    ProviderProbeResult,
    ProviderTestEvidence,
    ProviderTestEvidenceStore,
)


async def probe_console_connection(
    identity: ProviderDraftIdentity,
    *,
    app_config: Mapping[str, object] | None = None,
) -> ProviderProbeResult:
    """Run the bounded, non-generating model-catalog probe for one connection.

    Args:
        identity: The exact connection to probe.
        app_config: Config holding custom endpoint entries (their credential
            is the one a send would use).

    Returns:
        The bounded probe result.
    """
    from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
        SettingsEndpointProbePurpose,
        probe_settings_endpoint,
        provider_probe_result_from_settings_outcome,
    )

    probe_kwargs = {}
    if identity.custom_endpoint_id is not None:
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
    once. A connection with no models route (nothing to re-test) still
    opens Chat settings.

    Args:
        screen: The Console ``ChatScreen``.
    """
    _settings, readiness = screen._active_console_settings_readiness()
    identity = readiness.connection
    if identity is None or (
        connection_probe_availability(
            identity.provider_key, identity.connection_identity[1]
        )
        is not ConnectionProbeAvailability.MODELS_ROUTE
    ):
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
    if evidence is not None and evidence.endpoint == "unreachable":
        app.notify(
            f"{provider} is still unreachable. Start it, then retry.",
            severity="warning",
        )
