"""Setup's key checks, in the provider's own words (TASK-34100.2 AC#3, AC#5).

The Provider step's Enter/Test ran Settings' OpenAI-compatible probe. That
probe cannot fail a key a model list answers without -- OpenRouter, Hugging
Face, NVIDIA NIM and Novita list models for any key, so a wrong OpenRouter key
read '✓ Reached the server' -- and it has no target at all for Gemini,
Anthropic or Hugging Face, which read 'Enter a valid endpoint before testing.'
beside 'Key staged -- it will be checked when you continue. Connection testing
is unavailable for this provider.' Those providers' keys are checked by the
shared discovery the Model step already uses
(``LocalLLMProviderCatalogService.discover_models``): Google's native
models.list, OpenRouter's /api/v1/key, Hugging Face's whoami-v2 and
Anthropic's authenticated listing. This module says which providers use it
and words its verdict, so the Provider step, the Model step and the Summary
speak with one voice:

* '✓ Google accepted this key -- found 15 model(s).'
* '✗ OpenRouter rejected this key: User not found.'
* 'Found 81 model(s) for NVIDIA NIM -- key not checked: its model list answers
  without a valid key.' (never ✓)

The provider's words are text, never markup: the Provider step's status
parses Textual markup, so its line is escaped here (a masked key's
'[redacted]' vanished as a style tag, and a '[/docs]' raised MarkupError
inside the handler). The Model step's status line (TASK-33008.5) and the
'Continue anyway?' body render with markup off, so ``rejected_text`` and
``continue_prompt`` stay plain.

One mechanism with TASK-33008's wizard copy (``first_run_setup_state``'s
``PROBE_FAILURE_COPY``, ``first_run_model_discovery``'s ``failed_listing``,
``note_probe_*`` and ``staged_key_line``): those classify a failed check and
move the Authentication mark for every provider; this module only puts the
provider's own words on a rejected key (a 401 -- a 403 rejects no key) and
says which keys a model list can and cannot prove.

The Provider and Model step modules are pinned at their exact size
(``Tests/Architecture/test_module_size_ratchet.py``); their hooks are one-line
calls into here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Any

from textual.markup import escape as escape_markup

#: Providers whose key only the shared discovery can check in setup: their
#: model list needs the key (Google's native list, Anthropic's x-api-key
#: listing). Public-listing providers join them (``uses_shared_key_check``).
_KEYED_LISTING_PROVIDERS = frozenset({"google", "anthropic"})
_UNCHECKED_SUFFIX = "(key not checked)"
#: The name a key verdict uses where it differs from the display name: the
#: shared discovery's own Gemini messages say 'Google rejected this key'.
_KEY_CHECK_NAMES = {"google": "Google"}


def _key(provider_key: object) -> str:
    from tldw_chatbook.Chat.provider_readiness import provider_config_key

    return provider_config_key(str(provider_key or ""))


def _public_listing(provider_key: object) -> bool:
    from tldw_chatbook.Chat.provider_test_evidence import (
        PUBLIC_MODEL_LISTING_PROVIDER_KEYS,
    )

    return _key(provider_key) in PUBLIC_MODEL_LISTING_PROVIDER_KEYS


def uses_shared_key_check(provider_key: object) -> bool:
    """Whether setup checks this provider's key through the shared discovery.

    Args:
        provider_key: The selected provider.

    Returns:
        True for Google, Anthropic and every public-listing provider.
    """
    return _key(provider_key) in _KEYED_LISTING_PROVIDERS or _public_listing(
        provider_key
    )


def key_can_be_checked(provider_key: object) -> bool:
    """Whether the shared discovery can prove or reject this provider's key.

    A public listing proves nothing unless the record names a key endpoint
    (OpenRouter, Hugging Face); NVIDIA NIM and Novita have none.
    """
    from tldw_chatbook.provider_registry import RECORDS_BY_KEY

    key = _key(provider_key)
    if key in _KEYED_LISTING_PROVIDERS:
        return True
    record = RECORDS_BY_KEY.get(key)
    return bool(record is not None and record.key_check_url)


def unchecked_key_line(provider_key: object) -> str:
    """The Provider step's line for a typed key no check has answered yet.

    Args:
        provider_key: The selected provider.

    Returns:
        For a provider the shared discovery checks, the prompt to check it
        now (Enter runs that check) or, when its model list cannot prove a
        key, that it is not checked; otherwise ``""`` (the step keeps
        TASK-33008's 'it will be checked when you continue').
    """
    if not uses_shared_key_check(provider_key):
        return ""
    from tldw_chatbook.Chat.provider_catalog import provider_display_name

    display = provider_display_name(_key(provider_key))
    if key_can_be_checked(provider_key):
        return (
            f"Key staged — press Enter to check it with {display} now "
            "(it is also checked when you continue)."
        )
    return (
        f"Key staged — key not checked: {display}'s model list answers "
        "without a valid key, so your first message checks it."
    )


def answered_key_line(provider_key: object, outcome: Any) -> str:
    """The line for a typed key whose listing answered, or ``""``.

    A public listing answers any key, so its success says nothing about the
    key unless the provider's key endpoint passed it: then the line keeps
    'key not checked', never TASK-33008's 'the server did not refuse it'.
    """
    if not _public_listing(provider_key):
        return ""
    if getattr(outcome, "key_verified", False) is True:
        return ""
    from tldw_chatbook.Chat.provider_catalog import provider_display_name

    display = provider_display_name(_key(provider_key))
    return (
        f"Key staged — key not checked: {display}'s model list answers "
        "without a valid key, so your first message checks it."
    )


def key_verified(provider_key: object, result: Any, *, key_sent: bool) -> bool:
    """Whether a discovery result proves the key it was sent.

    ``ModelDiscoveryResult.key_verified`` is set by a key endpoint or Google's
    native list. An authenticated listing (Anthropic) that succeeded with a
    key proves it too; a public listing never does.
    """
    if getattr(result, "status", None) != "success" or not key_sent:
        return False
    if getattr(result, "key_verified", False) is True:
        return True
    return _key(provider_key) in _KEYED_LISTING_PROVIDERS


def _refused_key(result: Any) -> bool:
    """Whether a discovery result is the provider's 401 to the key it was sent.

    Only a 401 rejects a key: a 403 means this key may not list models
    (owner ruling 2026-10-02; TASK-33008 reads it as "listing unavailable").
    """
    error = getattr(result, "error", None)
    return (
        getattr(error, "kind", None) == "missing_credentials"
        and getattr(error, "category", None) == "unauthorized"
    )


def rejected_key_message(display: str, result: Any) -> str:
    """The provider's own words refusing a key, or ``""``.

    A refusal that carries no reason of the provider's own also returns
    ``""``: every surface then reads TASK-33008's ``PROBE_FAILURE_COPY`` line
    for it, as it does for OpenAI's check (resume after the dev merge: one
    401 read '✗ Anthropic rejected this key.' beside '✗ OpenAI rejected this
    API key. Update it, or continue anyway.').

    Args:
        display: The provider's display name (the discovery message names it).
        result: A ``ModelDiscoveryResult`` (or anything else).

    Returns:
        For example 'OpenRouter rejected this key: User not found.'
    """
    del display  # the discovery's own message already names the provider
    if not _refused_key(result):
        return ""
    error = getattr(result, "error", None)
    message = str(getattr(error, "message", "") or "").strip()
    reason = str(getattr(error, "provider_reason", "") or "").strip()
    return message if reason else ""


def discovery_status_copy(
    provider_key: object,
    display: str,
    result: Any,
    *,
    models: Iterable[str],
    failed: bool,
    attempted: bool,
    key_sent: bool,
    failure_copy: Callable[[], str],
) -> str:
    """The Provider step's line once its discovery settles.

    Args:
        provider_key: The selected provider.
        display: Its display name.
        result: The typed discovery outcome, or ``None``.
        models: The model ids found.
        failed: Whether discovery failed.
        attempted: Whether any discovery ran.
        key_sent: Whether a key went with it.
        failure_copy: The step's own wording for a failure that is not a
            key verdict ('Couldn't discover models for X. ...').

    Returns:
        The status line ("" when nothing ran), escaped for a markup Static.
    """
    return escape_markup(
        _discovery_status_text(
            provider_key,
            display,
            result,
            models=models,
            failed=failed,
            attempted=attempted,
            key_sent=key_sent,
            failure_copy=failure_copy,
        )
    )


def _discovery_status_text(
    provider_key: object,
    display: str,
    result: Any,
    *,
    models: Iterable[str],
    failed: bool,
    attempted: bool,
    key_sent: bool,
    failure_copy: Callable[[], str],
) -> str:
    """``discovery_status_copy`` before escaping."""
    count = len(tuple(models))
    if count:
        if key_verified(provider_key, result, key_sent=key_sent):
            name = _KEY_CHECK_NAMES.get(_key(provider_key), display)
            return f"✓ {name} accepted this key — found {count} model(s)."
        if key_sent and _public_listing(provider_key):
            return (
                f"Found {count} model(s) for {display} — key not checked: its "
                "model list answers without a valid key."
            )
        return f"Found {count} model(s) for {display}."
    if failed:
        # Review round 1 (live F1): a keyed provider's listing answers 401
        # before any key is typed. That is no verdict on a key -- only a key
        # that was sent can be rejected.
        if key_sent and _refused_key(result):
            # The provider's own reason, or the step's TASK-33008 line.
            return f"✗ {rejected_key_message(display, result) or failure_copy()}"
        return failure_copy()
    if attempted:
        return f"Checked {display}; no models were reported."
    return ""


def recorded_outcome(wizard: Any, discovery_key: object) -> Any:
    """The typed discovery outcome the Provider step recorded for this key."""
    for source in (
        getattr(wizard, "_first_run_selected_provider_outcomes", None),
        getattr(
            getattr(wizard, "_first_run_provider_discovery_owner", None),
            "_selected_provider_outcomes",
            None,
        ),
    ):
        if isinstance(source, dict) and discovery_key in source:
            return source[discovery_key]
    return None


def _display(discovery_key: object) -> str:
    from tldw_chatbook.Chat.provider_catalog import provider_display_name

    return provider_display_name(str(getattr(discovery_key, "provider_key", "")))


def rejected_text(wizard: Any, discovery_key: object) -> str:
    """The Model step's line for a rejected key, in the provider's words.

    Plain text: the Model step's status line renders with markup off
    (TASK-33008.5). Without a reason it is ``PROBE_FAILURE_COPY``'s own line.
    """
    from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state

    reason = rejected_key_message(
        _display(discovery_key), recorded_outcome(wizard, discovery_key)
    )
    if not reason:
        return wizard_state.probe_failure_copy(wizard_state.PROVIDER_PROBE_AUTH).model_line
    return f"{reason} Go Back to fix it, or enter a model ID below."


def continue_prompt(wizard: Any, discovery_key: object) -> str:
    """The Model step's 'Continue anyway?' body for a rejected key."""
    from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state

    reason = rejected_key_message(
        _display(discovery_key), recorded_outcome(wizard, discovery_key)
    )
    if not reason:
        return wizard_state.probe_failure_copy(wizard_state.PROVIDER_PROBE_AUTH).confirm
    return f"{reason} This model setup is unverified. Continue anyway?"


def confirm_prompt(wizard: Any, discovery_key: object, failure: str) -> str | None:
    """The Model step's question before Next, for its last failed check.

    Args:
        wizard: The setup wizard.
        discovery_key: The discovery identity the Model step rendered.
        failure: Its ``PROVIDER_PROBE_*`` classification.

    Returns:
        ``PROBE_FAILURE_COPY``'s question (TASK-33008), with a rejected key's
        reason in the provider's words when it gave one; None when no check
        failed.
    """
    from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state

    if failure == wizard_state.PROVIDER_PROBE_AUTH:
        return continue_prompt(wizard, discovery_key)
    copy = wizard_state.PROBE_FAILURE_COPY.get(failure)
    return copy.confirm if copy else None


def model_failure_line(wizard: Any, discovery_key: object, failure: str) -> str:
    """The Model step's status line for a key problem, or ``""`` for the
    step's own unreachable-server copy.

    ``PROBE_FAILURE_COPY``'s ``model_line`` (TASK-33008: a refusal with no
    key sent asks for one), with a rejected key's reason when it gave one.
    """
    from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state

    if failure == wizard_state.PROVIDER_PROBE_AUTH:
        return rejected_text(wizard, discovery_key)
    line = wizard_state.probe_failure_copy(failure).model_line
    return line.format(display=_display(discovery_key)) if line else ""


def listing_key_verified(wizard: Any, provider_key: object) -> bool:
    """Whether the key setup holds for this provider passed its key check.

    Read from the discovery outcome recorded for the staged provider draft
    (the identity the Model step uses: provider, endpoint, credential
    revision), never from the provider record: a key endpoint that answered
    403/429/5xx, a network failure or another host verifies nothing
    (review round 1, F5).

    Args:
        wizard: The setup wizard (or anything with its attributes).
        provider_key: The provider the Summary row names.

    Returns:
        True only for a recorded successful discovery with ``key_verified``.
    """
    from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state

    draft = getattr(wizard, "staged_provider_draft", None)
    if type(draft) is not wizard_state.FirstRunProviderDraft:
        return False
    try:
        discovery_key = wizard_state.build_first_run_model_discovery_key(draft)
    except ValueError:
        return False
    if _key(getattr(discovery_key, "provider_key", "")) != _key(provider_key):
        return False
    outcome = recorded_outcome(wizard, discovery_key)
    return (
        getattr(outcome, "status", None) == "success"
        and getattr(outcome, "key_verified", False) is True
    )


def mark_unchecked_listing_keys(
    rows: tuple[Any, ...],
    *,
    key_verified_for: Callable[[str], bool] | None = None,
) -> tuple[Any, ...]:
    """The Summary's Provider row never reads ✓ for a key nothing checked.

    A configured provider whose model list answers without a valid key reads
    '– Provider -- nvidia (key not checked)', matching Settings' 'models
    listed; key not checked': always when it has no key endpoint (NVIDIA
    NIM, Novita), and for OpenRouter and Hugging Face unless
    ``key_verified_for`` says their key check actually passed.

    Args:
        rows: The Summary rows.
        key_verified_for: The recorded verdict for a provider
            (``listing_key_verified``); None means nothing was recorded.

    Returns:
        The rows, with an unchecked Provider row marked.
    """
    from tldw_chatbook.UI.Wizards.first_run_setup_state import (
        ROW_CONFIGURED,
        ROW_DEFAULT,
    )

    def unchecked(row: Any) -> bool:
        provider = getattr(row, "detail", "")
        if not (
            getattr(row, "label", "") == "Provider"
            and getattr(row, "state", "") == ROW_CONFIGURED
            and _public_listing(provider)
        ):
            return False
        if not key_can_be_checked(provider):
            return True
        return not (key_verified_for is not None and key_verified_for(provider))

    return tuple(
        replace(row, state=ROW_DEFAULT, detail=f"{row.detail} {_UNCHECKED_SUFFIX}")
        if unchecked(row)
        else row
        for row in rows
    )


def settle_discovery_status(
    step: Any,
    provider_key: str,
    discovery_key: Any,
    result: Any,
    provider_draft: Any,
    *,
    models: Iterable[str],
    failed: bool,
    attempted: bool,
) -> None:
    """Write the Provider step's line once its discovery settles.

    TASK-33008's flow -- ``failed_listing`` classifies a failure as the Model
    step does, ``note_probe_failure``/``note_probe_reached`` move the
    Authentication mark -- with this module's words for the providers the
    shared discovery checks (``discovery_status_copy``). A key is "sent" by
    TASK-33008's one rule (``discovery_sends_a_key``).
    """
    from tldw_chatbook.Chat.provider_catalog import provider_display_name
    from tldw_chatbook.UI.Wizards import first_run_model_discovery as model_discovery
    from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state

    display = provider_display_name(provider_key)
    status = step.query_one("#setup-provider-probe-status")
    models = tuple(models)
    if failed and not models:
        line, failure = model_discovery.failed_listing(
            step, display, result, provider_draft
        )
    else:
        line, failure = "", wizard_state.PROVIDER_PROBE_NONE
    if uses_shared_key_check(provider_key):
        # Only a 401 to a key that was sent is the provider's verdict on it.
        rejected = failure == wizard_state.PROVIDER_PROBE_AUTH
        line = discovery_status_copy(
            provider_key,
            display,
            result if rejected or not failed else None,
            models=models,
            failed=failed,
            attempted=attempted,
            key_sent=wizard_state.discovery_sends_a_key(provider_draft),
            failure_copy=lambda: line,
        )
    elif models:
        line = f"Found {len(models)} model(s) for {display}."
    elif not failed:
        line = f"Checked {display}; no models were reported." if attempted else ""
    status.update(line)
    if models or (attempted and not failed and result is not None):
        model_discovery.note_probe_reached(step, discovery_key)
    elif failed:
        model_discovery.note_probe_failure(step, failure, discovery_key)
