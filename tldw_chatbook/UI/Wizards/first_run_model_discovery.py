"""Model discovery helpers the first-run Provider and Model steps share.

Moved out of ``FirstRunSetupWizard.py`` (TASK-34100.1) with the two steps
that use them. Patch this module, not the wizard, to change a discovery
timeout or outcome for both steps.
"""

from __future__ import annotations

from typing import Mapping

from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state


def _model_ids_from_discovery_result(result: object) -> tuple[str, ...]:
    """Extract exact typed catalog IDs without accepting duck-typed payloads."""

    from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
        DiscoveredModel,
        ModelDiscoveryResult,
    )
    from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
        DISCOVERED_MODEL_MAX_COUNT,
    )

    if type(result) is not ModelDiscoveryResult:
        raise ValueError("Model discovery result is invalid.")
    if result.status != "success":
        return ()
    if type(result.models) is not tuple:
        raise ValueError("Model discovery result is invalid.")
    # Bound this typed path by the *discovery* limit, not the probe's
    # MODEL_IDS_MAX_COUNT sample. Two incidents, one line: bounding at 100
    # rejected a successful 128-model discovery outright, which the caller
    # folds into a failed discovery ("Couldn't reach the server"), and
    # truncating to 100 instead silently dropped the newest 28 --
    # api.openai.com returns models in roughly chronological order, so a
    # 100-cap hides exactly the flagship models a user came for (gpt-5.4,
    # gpt-5.4-pro, gpt-5.3-chat-latest were all lost).
    #
    # Reject rather than truncate above the ceiling, and validate every
    # entry: discovery itself fails closed above DISCOVERED_MODEL_MAX_COUNT,
    # so an over-ceiling typed result did not come from that path and is
    # genuinely anomalous. Truncating instead would leave the tail
    # unvalidated and quietly break this helper's reject-malformed contract.
    # The legacy/local probe seam (_legacy_model_ids) keeps the smaller
    # sample bound.
    if len(result.models) > DISCOVERED_MODEL_MAX_COUNT:
        raise ValueError("Model discovery result is invalid.")
    model_ids: list[str] = []
    seen: set[str] = set()
    for discovered in result.models:
        if type(discovered) is not DiscoveredModel:
            raise ValueError("Model discovery result is invalid.")
        try:
            model_id = wizard_state.validate_first_run_model_id(discovered.model_id)
        except ValueError as exc:
            raise ValueError("Model discovery result is invalid.") from exc
        if model_id in seen:
            continue
        seen.add(model_id)
        model_ids.append(model_id)
    return tuple(model_ids)


def _legacy_model_ids(values: object) -> tuple[str, ...]:
    """Validate the intentionally retained injected string-list test seam."""

    from tldw_chatbook.Chat.local_server_discovery import MODEL_IDS_MAX_COUNT

    if type(values) not in {list, tuple} or len(values) > MODEL_IDS_MAX_COUNT:
        raise ValueError("Legacy model discovery result is invalid.")
    model_ids: list[str] = []
    seen: set[str] = set()
    for value in values:
        model_id = wizard_state.validate_first_run_model_id(value)
        if model_id in seen:
            continue
        seen.add(model_id)
        model_ids.append(model_id)
    return tuple(model_ids)


# The category a failed discovery falls back to when nothing more specific is
# known. It drives user-visible copy (see classify_discovery_failure), so the
# fallback branches must not drift apart from each other.
GENERIC_DISCOVERY_FAILURE_CATEGORY = "request failed"


def _model_discovery_ui_outcome(result: object) -> tuple[list[str], str, str]:
    """Interpret one typed discovery result into bounded Model-step state."""

    from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
        ModelDiscoveryResult,
    )

    if type(result) is not ModelDiscoveryResult:
        raise ValueError("Model discovery result is invalid.")
    if result.status == "success":
        models = list(_model_ids_from_discovery_result(result))
        return models, "available" if models else "empty", ""
    if result.status == "unsupported" or (
        result.error is not None and result.error.kind == "unsupported_endpoint"
    ):
        return [], "listing_unavailable", ""
    error_kind = result.error.kind if result.error is not None else ""
    category = {
        "invalid_response": "invalid response",
        "missing_credentials": "authentication",
    }.get(error_kind, GENERIC_DISCOVERY_FAILURE_CATEGORY)
    return [], "connection_failed", category


def _handed_off_failure_category(owner: object, discovery_key: object) -> str:
    """Recover the real failure category from the owner's recorded outcome.

    ProviderStep records the typed ``ModelDiscoveryResult`` for a selection
    even on the handoff paths where ModelStep never receives one directly.
    Without this, those paths reported a flat "request failed", so an
    authentication rejection rendered as "Couldn't reach the server. Check
    it's running" -- telling the user to check a server when the problem was
    their key, and defeating the provider-aware copy added for UAT M-4. That
    wording masked the true cause for most of the TASK-23089 investigation.

    Args:
        owner: The ProviderStep that owned the discovery, if any.
        discovery_key: The exact discovery identity ModelStep is rendering.

    Returns:
        The category derived from the recorded outcome, or "request failed"
        when no typed outcome is available to be more specific than that.
    """

    outcomes = getattr(owner, "_selected_provider_outcomes", None)
    if not isinstance(outcomes, Mapping) or discovery_key not in outcomes:
        return GENERIC_DISCOVERY_FAILURE_CATEGORY
    try:
        _models, _state, category = _model_discovery_ui_outcome(outcomes[discovery_key])
    except ValueError:
        return GENERIC_DISCOVERY_FAILURE_CATEGORY
    return category or GENERIC_DISCOVERY_FAILURE_CATEGORY


def _first_run_discovery_staged_settings(
    provider_draft: wizard_state.FirstRunProviderDraft,
    discovery_key: wizard_state.FirstRunModelDiscoveryKey,
) -> dict[str, dict[str, dict[str, str]]]:
    """Build transient settings for one exact typed discovery boundary."""

    from tldw_chatbook.Chat.provider_setup_persistence import provider_endpoint_key

    endpoint = (
        provider_draft.discovery_endpoint
        or provider_draft.endpoint
        or discovery_key.connection_identity[1]
    )
    settings = {provider_endpoint_key(discovery_key.provider_key): endpoint}
    credential = provider_draft.credential
    credential_value = wizard_state._credential_value_for_boundary(credential)
    boundary_source = credential.source
    if boundary_source == "draft" and not credential_value:
        boundary_source = "none"
    settings["credential_source"] = boundary_source
    if credential.source == "draft":
        settings["api_key"] = credential_value
    elif credential.source == "environment":
        settings["api_key_env_var"] = credential_value
    elif credential.source == "none":
        settings["api_key"] = ""
    return {"api_settings": {discovery_key.provider_key: settings}}


MODEL_DISCOVERY_TIMEOUT_SECONDS = 8.0
