"""Model discovery helpers the first-run Provider and Model steps share.

Moved out of ``FirstRunSetupWizard.py`` (TASK-34100.1) with the two steps
that use them. Patch this module, not the wizard, to change a discovery
timeout or outcome for both steps.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

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
    # A 403 means this key may not list models: only a 401 rejects a key
    # (owner ruling 2026-10-02, PR #2958 review; as Test and Settings read it).
    # It keeps "forbidden" for the keyless hint (owner decision 2026-10-17, D2).
    forbidden = result.error is not None and result.error.category == "forbidden"
    code = answered_http_status(result.error)
    if (
        forbidden
        or result.status == "unsupported"
        or (
            result.error is not None
            # A listing route that isn't there (404, 405) is a listing that
            # isn't available, never 'couldn't reach the server' (TASK-34100.6
            # review round 1, live F7). Any other HTTP answer (429, 5xx) is a
            # failure that names its status (review round 2, R2-02).
            and (result.error.kind == "unsupported_endpoint" or code in {404, 405})
        )
    ):
        return [], "listing_unavailable", "forbidden" if forbidden else ""
    if code is not None:
        return [], "connection_failed", f"HTTP {code}" if code else "HTTP error"
    error_kind = result.error.kind if result.error is not None else ""
    category = {
        "invalid_response": "invalid response",
        "missing_credentials": "authentication",
    }.get(error_kind, GENERIC_DISCOVERY_FAILURE_CATEGORY)
    transport = result.error.category if result.error is not None else None
    category = _TRANSPORT_CATEGORIES.get(str(transport), category)
    return [], "connection_failed", category


_HTTP_STATUS = re.compile(r"HTTP (\d{3})")


def answered_http_status(error: object) -> int | None:
    """The HTTP status a discovery error was answered with, or None."""
    if getattr(error, "category", None) != "http_status":
        return None
    match = _HTTP_STATUS.search(str(getattr(error, "message", "") or ""))
    return int(match.group(1)) if match else 0


def http_error_hint(code: int) -> tuple[str, str]:
    """(what the status means, what to do) for an HTTP error answer.

    llama-server answers 503 on every route while its model loads, a common
    first-run moment (review round 2, R2-02).
    """
    if code == 429:
        return "too many requests", "wait a moment"
    if code == 503:
        return "it may still be loading or busy", "wait"
    if code >= 500:
        return "the server hit an error", "check its logs"
    return "", ""


#: TASK-34100.6 (review coverage-19, provider-15): transport causes that
#: change what the user should do, named in the Model step's copy.
_TRANSPORT_CATEGORIES = {
    "tls_untrusted": "untrusted certificate",
    "tls_plain_http": "plain HTTP",
}


def evidence_failure_category(evidence: object, owner: object = None) -> str:
    """Model's UI category for an unreachable Test-connection result.

    TASK-34100.6 review round 1 (live F1): the evidence category was shown
    with its underscores replaced ('tls untrusted'), which never matched the
    'untrusted certificate' / 'plain HTTP' categories the Model row's copy
    keys on, so Test connection then Next lost the Settings ▸ Network pointer.
    Review round 2 (R2-02): an HTTP error answer names its status, read from
    the Provider step (``owner``) that ran the test. TASK-33008 review: a
    Test's 401 ("unauthorized") reads as "authentication", the one category
    ``classify_discovery_failure`` reads as a key problem; it said "not
    reachable" on every surface after Provider.
    """
    category = str(getattr(evidence, "category", "") or "connection_error")
    if category == "unauthorized":
        return "authentication"
    if category == "http_status":
        code = getattr(owner, "_last_probe_status_code", None)
        return f"HTTP {code}" if isinstance(code, int) and code else "HTTP error"
    return _TRANSPORT_CATEGORIES.get(category, category.replace("_", " "))


def _origin(endpoint: str) -> str:
    """'https://127.0.0.1:8443' for an endpoint, or "" when it has none."""
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(endpoint)
    except ValueError:
        return ""
    return f"{parts.scheme}://{parts.netloc}" if parts.scheme and parts.netloc else ""


def connection_failed_text(provider_key: str, category: str, endpoint: str) -> str:
    """The Model step's failed-discovery row (TASK-21143 UAT M-1/M-4).

    Connection failures name the server the user has to start; a certificate
    the computer doesn't trust, or https against a plain-HTTP server, says so
    (TASK-34100.6). A refused or missing key has its own row
    (``ProbeFailureCopy.model_line``, TASK-33008).

    Args:
        provider_key: The provider whose models were listed.
        category: The UI failure category (``_model_discovery_ui_outcome``).
        endpoint: The address that was asked, or "".

    Returns:
        The row's text.
    """
    at_endpoint = f" at {endpoint}" if endpoint else ""
    origin = _origin(endpoint) or "The server"  # one line: no chat route
    if category == "untrusted certificate":
        return (
            f"{origin} answered, but its certificate isn't trusted — add your CA "
            "in Settings ▸ Network."  # Retry is the button below the row
        )
    if category == "plain HTTP":
        return f"{origin} speaks plain HTTP — go Back and use http:// instead of https://."
    if category.startswith("HTTP "):  # it answered (review round 2, R2-02)
        code = int(category[5:]) if category[5:].isdigit() else 0
        what, _todo = http_error_hint(code)
        meaning = f" — {what}" if what else ""
        answered = category if code else "with an HTTP error"
        return f"{origin} answered {answered}{meaning}. Retry, or enter a model ID below."
    if provider_key in ("ollama", "local_ollama"):
        return (
            f"Ollama isn't running{at_endpoint}. Start it (ollama serve), then "
            "Retry — or enter a model ID below."
        )
    if provider_key in ("llama_cpp", "local_llamacpp"):
        return (
            f"The llama.cpp server isn't reachable{at_endpoint}. Start it, then "
            "Retry — or enter a model ID below."
        )
    return (
        f"Couldn't reach the server ({category}). Check it's running, then "
        "Retry — or enter a model ID below."
    )


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
        _models, state, category = _model_discovery_ui_outcome(outcomes[discovery_key])
    except ValueError:
        return GENERIC_DISCOVERY_FAILURE_CATEGORY
    # Only a failure's category: a 403's "forbidden" names no failure.
    failed = state == "connection_failed"
    return category if failed else GENERIC_DISCOVERY_FAILURE_CATEGORY


def classify_render(step: Any, state: str, category: str) -> tuple[str, str]:
    """The Model step's discovery state and PROVIDER_PROBE_* value to render."""

    draft = step._current_provider_draft()
    if subscription_cannot_list(step, draft, state, category):
        state = "listing_unavailable"
    key_sent = wizard_state.discovery_sends_a_key(draft)
    return state, wizard_state.classify_discovery_failure(
        state, category, key_sent=key_sent
    )


def show_provider_line(step: Any, models: object = (), state: str = "") -> None:
    """Paint the Model step's subtitle: whose models these are and, while the
    rows came from a server's listing, that a pick is not added to the saved
    model list (TASK-35578). "saved" marks the one source that is not a
    listing, the saved ``[providers]`` list; rows a Test listed can come with
    any other state ("empty", "connection_failed"). Setup never offers a
    named custom endpoint, so the note is always the saved list's."""

    try:
        from textual.widgets import Static

        from tldw_chatbook.Chat.provider_catalog import (
            listed_pick_note,
            provider_display_name,
        )

        # TASK-1503: display-case the provider; raw keys are not UI language.
        provider_key = step._current_provider()[0]
        display = provider_display_name(provider_key) if provider_key else ""
        line = f"Models for {display or 'your provider'}."
        if models and state != "saved":
            line = f"{line} {listed_pick_note()}"
        step.query_one("#setup-model-provider-line", Static).update(line)
    except Exception:
        pass


def subscription_cannot_list(
    step: Any, provider_draft: object, state: str, category: str
) -> bool:
    """Whether a refused listing is a Claude subscription's, not a missing key.

    TASK-33008 review: a subscription signs chat in, but discovery never
    carries it, so a keyless Anthropic model list is refused. The copy said
    the server "needs an API key and none was set", steering the user to a
    key they do not need. Such a listing reads as unavailable instead.

    The table is found the way readiness finds it (``provider_settings_for_key``,
    the first normalised match), so ``[api_settings.Anthropic]`` reads as a
    subscription here too, not only in readiness (review round 2).
    """

    return (
        state == "connection_failed"
        and category == "authentication"
        and not wizard_state.discovery_sends_a_key(provider_draft)
        and _signs_in_with_subscription(step, getattr(provider_draft, "provider", ""))
    )


def _signs_in_with_subscription(step: Any, provider: str) -> bool:
    """Whether ``provider`` is Anthropic signed in with a Claude subscription."""

    wizard = getattr(step, "wizard", None)
    app_config = getattr(getattr(wizard, "app_instance", None), "app_config", None)
    if provider != "anthropic" or not isinstance(app_config, Mapping):
        return False
    from tldw_chatbook.config import provider_settings_for_key
    from tldw_chatbook.LLM_Calls.anthropic_subscription import anthropic_auth_source

    settings = provider_settings_for_key(app_config.get("api_settings"), "anthropic")
    return anthropic_auth_source(settings) == "claude_subscription"


def _with_keyless_hint(
    step: Any, line: str, hint: str, category: object, *, key_sent: bool, provider: str
) -> str:
    """``line`` for a listing that cannot list, plus ``hint`` after a keyless 403.

    The owner's decision of 2026-10-17 (PR #3144, D2): a 403 rejects no key
    (only a 401 does), but a server that answered one to a check that sent no
    key may want a key. A Claude subscription signs in without one, so it
    never gets the hint (D3 confirmed its copy).
    """

    if (
        category != "forbidden"
        or key_sent
        or _signs_in_with_subscription(step, provider)
    ):
        return line
    return f"{line} {hint}"


def model_unavailable_line(step: Any, category: object) -> str:
    """The Model step's line for a listing that cannot list (``category`` as
    ``_model_discovery_ui_outcome`` or a Test's evidence names it). A key is
    judged sent as ``classify_render`` judges it, by the step's draft."""

    draft = step._current_provider_draft()
    return _with_keyless_hint(
        step,
        wizard_state.MODEL_LISTING_UNAVAILABLE_LINE,
        wizard_state.MODEL_KEYLESS_REFUSAL_HINT,
        category,
        key_sent=wizard_state.discovery_sends_a_key(draft),
        provider=getattr(draft, "provider", ""),
    )


def failed_listing(
    owner: Any, display: str, outcome: object, provider_draft: object
) -> tuple[str, str]:
    """The Provider step's line for a failed listing, told as the Model step
    tells it, and the ``PROVIDER_PROBE_*`` failure it names. It changes
    nothing: the caller marks a refused key (``note_probe_failure``).

    TASK-33008 (capture 06a): the two steps told one 401 two ways. Review
    round 2: a listing Model reads as "Model listing unavailable" (no model
    list, or a Claude subscription's) read "Couldn't discover models for
    {display}. You can continue anyway." here; it names Model's cause now,
    with this step's own action, while Next is allowed. A missing or untyped
    outcome reads as a connection failure.
    """

    try:
        _models, state, category = _model_discovery_ui_outcome(outcome)
    except ValueError:
        state, category = "connection_failed", ""
    if subscription_cannot_list(owner, provider_draft, state, category):
        state = "listing_unavailable"
    failure = wizard_state.classify_discovery_failure(
        state, category, key_sent=wizard_state.discovery_sends_a_key(provider_draft)
    )
    try:  # Model's cause only while Next is allowed
        unavailable = state == "listing_unavailable" and bool(
            owner._current_provider_readiness().ready
        )
    except Exception:  # noqa: BLE001 - unreadable readiness: the step's own copy
        unavailable = False
    if unavailable:
        return _with_keyless_hint(
            owner,
            wizard_state.PROVIDER_LISTING_UNAVAILABLE_LINE,
            wizard_state.PROVIDER_KEYLESS_REFUSAL_HINT,
            category,
            key_sent=wizard_state.discovery_sends_a_key(provider_draft),
            provider=getattr(provider_draft, "provider", ""),
        ), failure
    return owner._discovery_failure_status(display, failure), failure


def note_probe_failure(owner: Any, failure: str, checked: Any) -> None:
    """Remember a check that was refused for its key (capture 06b).

    ``checked`` is the discovery key or Test identity the check ran under.
    From then on the Provider step's Authentication section is not called
    optional while it shows that connection (see ``auth_title``), and the
    section is scrolled into view: the line says "Add one under
    Authentication", and at 211x44 it sat below the step's fold. Review
    round 2 (capture 06d): a server that refused a key that was sent asks
    for a key too, so that refusal marks the section as well. Review round
    4: every failed check redraws the key line, so a 403 or a timeout after
    a 401 to the same key does not leave "the server rejected it" standing.
    """

    refused = failure in {
        wizard_state.PROVIDER_PROBE_KEY_REQUIRED,
        wizard_state.PROVIDER_PROBE_AUTH,
    }
    if refused:  # before the refresh: ``auth_title`` reads the mark
        owner._key_required_at = checked.connection_identity
    owner._refresh_auth_readiness()
    if refused:
        auth = owner.query_one("#setup-provider-auth-toggle")
        # After the refresh, so the line's new height is laid out first.
        # ponytail: a check settled before the step's first layout (the app
        # mid-batch) finds no size and does not scroll; checks are network
        # round trips, so retry on the step's Resize only if that is seen.
        owner.call_after_refresh(
            lambda: auth.is_attached and auth.scroll_visible(animate=False)
        )


def note_probe_reached(owner: Any, checked: Any, endpoint: str = "reachable") -> None:
    """Forget ``note_probe_failure``'s mark once a check of that connection
    sent no key and got through: the server no longer asks for one (it was
    restarted without --api-key), so the section is optional again.

    "Sent no key" is the rule that set the mark (``discovery_sends_a_key``):
    a stored key being cleared is source "draft" with no text, and reading
    the source alone kept the mark for good on the listing path.

    Every settled Test, and every listing that answered, comes here; a
    failed listing goes to ``note_probe_failure``. Both refresh the key line
    (``staged_key_line``, review rounds 2 and 4).
    """

    key_sent = checked.credential_source != "none"
    if checked.credential_source == "draft":
        draft = owner._effective_provider_draft()
        key_sent = wizard_state.discovery_sends_a_key(draft)
    if (
        endpoint == "reachable"
        and not key_sent
        and owner._key_required_at == checked.connection_identity
    ):
        owner._key_required_at = None
    owner._refresh_auth_readiness()


def tested_evidence_detail(
    owner: Any, identity: Any, evidence: Any, detail: str
) -> str:
    """The Provider line for a settled Test, moving the key mark as it reads.

    TASK-33008: a 401 to a Test that sent no key asks for one, as the
    listing does, instead of saying the server rejected a key. Review round
    2 (capture 06d): a 401 to a key that was sent marks the section too.
    Round 3: that 401 is told as the listing tells it as well, not in the
    shared verdict's own words. Round 4: so is a listing Test could not get
    (a 403, a 404), which the shared verdict calls "chat endpoint not tested".
    """

    if evidence.category == "unauthorized":
        from tldw_chatbook.Chat.provider_catalog import provider_display_name

        failure = (
            wizard_state.PROVIDER_PROBE_KEY_REQUIRED
            if identity.credential_source == "none"
            else wizard_state.PROVIDER_PROBE_AUTH
        )
        detail = owner._discovery_failure_status(
            provider_display_name(identity.provider_key), failure
        )
        note_probe_failure(owner, failure, identity)
    elif evidence.endpoint == "model_listing_unavailable":
        # A key sent is read as the 401 above reads it (D2, 2026-10-17).
        detail = _with_keyless_hint(
            owner,
            wizard_state.PROVIDER_LISTING_UNAVAILABLE_LINE,
            wizard_state.PROVIDER_KEYLESS_REFUSAL_HINT,
            evidence.category,
            key_sent=identity.credential_source != "none",
            provider=identity.provider_key,
        )
    note_probe_reached(owner, identity, evidence.endpoint)
    return detail


#: The line under a typed key that no check has answered.
STAGED_KEY_LINE = "Key staged — it will be checked when you continue."


def staged_key_line(owner: Any) -> str:
    """The line under a typed key, from the last check of it (capture 06d).

    It said "it will be checked when you continue" under a key a Test had
    just seen refused. The Test of this key and address, or failing that its
    model listing, now says what it found: a 401 rejected the key, an answer
    did not. A check that is running, could not reach the server, or got no
    usable list keeps the line: Next still checks the key.
    """

    key = owner._model_discovery_key(owner._effective_provider_draft())
    if key is None:
        return STAGED_KEY_LINE
    evidence = owner._test_evidence_for_discovery_key(key)
    if evidence is not None:
        refused = evidence.category == "unauthorized"
        answered = evidence.endpoint == "reachable"
    else:
        try:
            outcome = owner._selected_provider_outcomes.get(key)
            _models, state, category = _model_discovery_ui_outcome(outcome)
        except ValueError:  # no listing of this key yet
            return STAGED_KEY_LINE
        refused = state == "connection_failed" and category == "authentication"
        answered = state in {"available", "empty"}
    if refused:
        return "Key staged — the server rejected it; update it above."
    if answered:
        return "Key staged — the server did not refuse it."
    return STAGED_KEY_LINE


def auth_title(owner: Any, requires_api_key: bool) -> str:
    """The Provider step's Authentication title.

    TASK-33008 (capture 06b): a keyless provider's section said "(optional)"
    under a line saying its server needs a key. It drops the word when the
    provider needs a key, or when a check against the connection on screen
    was refused for want of one (``note_probe_failure``).
    """

    if not requires_api_key and owner._key_required_at is not None:
        key = owner._model_discovery_key(owner._effective_provider_draft())
        requires_api_key = getattr(key, "connection_identity", None) == (
            owner._key_required_at
        )
    return "Authentication" if requires_api_key else "Authentication (optional)"


def handed_off_probe_failure(model_step: object) -> str:
    """Provider's settled failure for the staged identity, before Model renders.

    TASK-34100.6 (review provider-03): the tracker showed ✓ on Provider for
    the whole Model step after a failed check, flipping to '!' only after
    Model's 'Continue anyway?'. The failure is known as soon as Provider's
    background check settles, so the tracker reads it from there.

    Args:
        model_step: The ModelStep asking.

    Returns:
        A ``PROVIDER_PROBE_*`` classification, or "" while nothing failed.
    """
    owner = getattr(getattr(model_step, "wizard", None), "_first_run_provider_discovery_owner", None)
    try:
        discovery_key = model_step._current_discovery_key()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - no identity yet means nothing failed.
        return ""
    outcomes = getattr(owner, "_selected_provider_outcomes", None)
    if discovery_key is None or not isinstance(outcomes, Mapping) or discovery_key not in outcomes:
        return ""
    try:
        _models, state, category = _model_discovery_ui_outcome(outcomes[discovery_key])
    except ValueError:
        return ""
    return classify_render(model_step, state, category)[1]  # as Model will (TASK-33008)


#: Providers whose saved chat route is a native one discovery refuses to list
#: from (``openai_compatible_model_discovery._NATIVE_ENDPOINT_PATHS_BY_PROVIDER``)
#: while the server also serves the OpenAI listing the endpoint contract names
#: as ``models_url``. KoboldCpp saves /api/v1/generate (review new-provider-02);
#: listing from it made no request, so nothing answering had no named cause
#: (TASK-34100.6 review round 1, live F11).
_LISTS_FROM_MODELS_URL = frozenset({"koboldcpp"})


def _resolves_through_hosted_engine(provider_key: str) -> bool:
    """Whether catalog discovery resolves ``provider_key`` like a hosted send."""
    from tldw_chatbook.provider_registry import RECORDS_BY_KEY

    record = RECORDS_BY_KEY.get(provider_key)
    return bool(record is not None and record.engine_driven)


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
    if _resolves_through_hosted_engine(discovery_key.provider_key):
        # Review round 1 (live F7's real cause): a hosted preset's discovery
        # resolves its base URL like a send, and refused the chat URL ('must
        # not include a terminal /chat/completions endpoint path') before any
        # request. Stage the base Next saves.
        from tldw_chatbook.Chat.provider_setup_persistence import api_base_from_chat_url

        endpoint = api_base_from_chat_url(endpoint) or endpoint
    if discovery_key.provider_key in _LISTS_FROM_MODELS_URL:
        from tldw_chatbook.Chat.provider_endpoint_contract import (
            resolve_provider_endpoint,
        )

        endpoint = resolve_provider_endpoint(
            discovery_key.provider_key, endpoint
        ).models_url or endpoint
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


async def discover_within_timeout(
    discover_models: Any, provider_key: str, staged_settings: Mapping[str, Any]
) -> object:
    """The selected-provider listing under the wizard's wait, never untyped.

    Review round 2 (R2-07): httpx's own ConnectTimeout (10 s) -- where a
    stalled loopback https handshake is confirmed as plain HTTP -- never
    fired before the wizard's 8 s wait, so the failure reached the step with
    no cause. The listing now gets the wait minus the confirmation's own
    budget; when it runs out, the result is a typed 'timeout', or
    'tls_plain_http' when one plain GET to the same loopback host:port is
    answered. The whole call ends inside MODEL_DISCOVERY_TIMEOUT_SECONDS, so
    Model's own wait for this result never fires first.

    Args:
        discover_models: The catalog scope service's ``discover_models``.
        provider_key: The provider listed.
        staged_settings: The transient settings the listing uses.

    Returns:
        The listing's ``ModelDiscoveryResult``, or a typed error result.
    """
    import asyncio

    from tldw_chatbook.Chat import local_server_discovery as discovery
    from tldw_chatbook.Chat.provider_setup_persistence import provider_endpoint_key
    from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
        ModelDiscoveryError,
        ModelDiscoveryResult,
    )

    total = MODEL_DISCOVERY_TIMEOUT_SECONDS
    confirm = discovery.PLAIN_HTTP_CONFIRM_TIMEOUT_SECONDS
    try:
        return await asyncio.wait_for(
            discover_models(
                mode="local",
                provider=provider_key,
                staged_settings=staged_settings,
                use_shared_cache=False,
            ),
            timeout=max(total - confirm - 0.5, total / 2),
        )
    except TimeoutError:
        pass
    settings = (staged_settings.get("api_settings") or {}).get(provider_key) or {}
    endpoint = str(settings.get(provider_endpoint_key(provider_key)) or "")
    plain = await discovery.loopback_answers_plain_http(endpoint)
    return ModelDiscoveryResult(
        provider=provider_key,
        provider_list_key=None,
        endpoint_fingerprint=None,
        status="error",
        error=ModelDiscoveryError(
            kind="request_failed",
            message="Model discovery timed out.",
            recovery_hint="Check the server is running.",
            category="tls_plain_http" if plain else "timeout",
        ),
    )

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
        and getattr(owner, "_selected_discovery_state", "") in REUSABLE_DISCOVERY_STATES
    )


def drop_unreusable_handoff(owner: Any) -> None:
    """Drop ``owner``'s handed-over discovery result unless it is reusable.

    Qodo (PR #3001): Model's hide used to drop every result, including a
    model list that had arrived, while ``discovery_is_reusable`` still let
    that discovery stand for its identity. A Back before Model drew the list
    then left nobody asking again, and Model fell back to curated models. So
    a list that arrived is kept, under the same rule that reuses it; any
    other result is dropped. The caller has already fenced a running
    discovery, which therefore reads as cancelled here.

    Args:
        owner: The ProviderStep whose selected discovery Model no longer
            consumes. Its result maps and the wizard's handoff copies are
            cleared in place.
    """

    if getattr(owner, "_selected_discovery_state", "") in REUSABLE_DISCOVERY_STATES:
        return
    owner._selected_provider_models.clear()
    owner._selected_provider_outcomes.clear()
    owner.wizard._first_run_selected_provider_models = {}
    owner.wizard._first_run_selected_provider_outcomes = {}


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
