"""Setup's key verdicts in the provider's own words (TASK-34100.2 AC#3, AC#5).

The Provider step's 'Check key' (TASK-34100.6, ``first_run_provider_check``)
runs the shared discovery the Model step already uses
(``LocalLLMProviderCatalogService.discover_models``): Google's native
models.list, Anthropic's authenticated listing, and -- for a provider whose
model list answers without a valid key -- the provider record's key check
(OpenRouter's /api/v1/key, Hugging Face's whoami-v2), which sets
``ModelDiscoveryResult.key_verified``. That step words the verdict
(``key_check_line``); this module adds what TASK-34100.2 owns on top of it:

* a rejected key's line carries the provider's own reason
  ('✗ OpenRouter rejected this key: User not found.', ``with_provider_reason``);
* the Model step's line and 'Continue anyway?' body quote that reason
  (``model_failure_line``, ``confirm_prompt``);
* the Summary never ticks a key nothing could check (``mark_unchecked_listing_keys``):
  NVIDIA NIM and Novita have no key check, and OpenRouter and Hugging Face
  only tick when theirs passed.

The provider's words are text, never markup: the Provider step's status line
parses Textual markup, so its reason is escaped here (a masked key's
'[redacted]' vanished as a style tag, and a '[/docs]' raised MarkupError
inside the handler). The Model step's status line (TASK-33008.5) and the
'Continue anyway?' body render with markup off, so ``rejected_text`` and
``continue_prompt`` stay plain.

One mechanism with TASK-33008's wizard copy (``first_run_setup_state``'s
``PROBE_FAILURE_COPY``, ``first_run_model_discovery``'s ``failed_listing`` and
``note_probe_*``): those classify a failed check and move the Authentication
mark for every provider; this module only puts the provider's own words on a
rejected key (a 401 -- a 403 rejects no key) and says which keys a model list
can and cannot prove.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from textual.markup import escape as escape_markup

_UNCHECKED_SUFFIX = "(key not checked)"


def _key(provider_key: object) -> str:
    from tldw_chatbook.Chat.provider_readiness import provider_config_key

    return provider_config_key(str(provider_key or ""))


def _public_listing(provider_key: object) -> bool:
    from tldw_chatbook.Chat.provider_test_evidence import (
        PUBLIC_MODEL_LISTING_PROVIDER_KEYS,
    )

    return _key(provider_key) in PUBLIC_MODEL_LISTING_PROVIDER_KEYS


def key_can_be_checked(provider_key: object) -> bool:
    """Whether a public listing's key can be proved: the record names a key check.

    OpenRouter and Hugging Face have one; NVIDIA NIM and Novita have none.
    """
    from tldw_chatbook.provider_registry import RECORDS_BY_KEY

    record = RECORDS_BY_KEY.get(_key(provider_key))
    return bool(record is not None and record.key_check_url)


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


def with_provider_reason(line: str, result: Any) -> str:
    """The Provider step's rejected-key line, with the provider's own reason.

    ``first_run_provider_check`` words a rejected key for where it came from
    ('✗ OpenRouter rejected this key.', '... rejected the key in
    OPENROUTER_API_KEY.'); AC#3 adds the reason the provider gave, escaped
    for the step's markup status line.

    Args:
        line: The step's own rejected-key line.
        result: The settled ``ModelDiscoveryResult`` (or anything else).

    Returns:
        ``line`` with ': <reason>' in place of its final period, or ``line``
        unchanged when the provider gave no reason or rejected no key.
    """
    if not _refused_key(result):
        return line
    reason = str(getattr(getattr(result, "error", None), "provider_reason", "") or "").strip()
    if not reason:
        return line
    return f"{line.rstrip().removesuffix('.')}: {escape_markup(reason)}"


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


