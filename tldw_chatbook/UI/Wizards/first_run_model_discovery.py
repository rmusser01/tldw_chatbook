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

# TASK-34100.1 review round 3: the one reuse rule for a provider identity's
# selected discovery. Only a discovery still running, or one whose model list
# arrived, stands for that identity. A failed one (the local server was not
# running yet) or a cancelled one is asked again by whoever needs it next:
# Provider's show and Next, Model's show and Retry.
REUSABLE_DISCOVERY_STATES = frozenset({"in_progress", "complete"})


def discovery_is_reusable(owner: object, discovery_key: object) -> bool:
    """Whether ``owner``'s selected discovery can stand for ``discovery_key``.

    Args:
        owner: The ProviderStep that runs the selected-provider discovery.
        discovery_key: The exact provider identity a step needs models for.

    Returns:
        True only for the same identity, with the discovery in progress or
        complete; never for a failed, cancelled or idle one.
    """

    return (
        discovery_key is not None
        and getattr(owner, "_selected_discovery_key", None) == discovery_key
        and getattr(owner, "_selected_discovery_state", "")
        in REUSABLE_DISCOVERY_STATES
    )


def discovery_failed_for(owner: object, discovery_key: object) -> bool:
    """Whether ``owner``'s last discovery for ``discovery_key`` failed."""

    return (
        getattr(owner, "_selected_discovery_key", None) == discovery_key
        and getattr(owner, "_selected_discovery_state", "") == "failed"
    )


def ask_again_on_return(model_step: object, discovery_key: object) -> None:
    """Ask the server again when the user comes back to Model after a failure.

    Model calls this once per visit, as it shows. It asks only when the
    Provider step's discovery is still the one Model showed as the user left
    it, so a Next from Provider, which has just asked again itself, is not
    asked twice. It never asks while a discovery runs or after a list arrived.

    Args:
        model_step: The Model step. Supplies the wizard (whose
            ``_first_run_provider_discovery_owner`` is the Provider step), the
            staged provider draft, and ``_left_generation``: the Provider
            discovery it showed when it was last hidden.
        discovery_key: The exact discovery identity Model is showing.
    """

    wizard = getattr(model_step, "wizard", None)
    owner = getattr(wizard, "_first_run_provider_discovery_owner", None)
    begin = getattr(owner, "_begin_selected_provider_discovery", None)
    left = getattr(model_step, "_left_generation", None)
    if (
        discovery_key is None
        or left is None
        or not callable(begin)
        or getattr(owner, "is_attached", False) is not True
        or getattr(owner, "_selected_discovery_generation", None) != left
        or discovery_is_reusable(owner, discovery_key)
    ):
        return
    draft = model_step._current_provider_draft()
    if draft is not None:
        begin(draft, sync_live_credential=False)
