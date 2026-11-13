"""Settings' provider Test rows: key verdicts and model capacity (TASK-34100.2).

``SettingsScreen._provider_test_rows`` states one fact per row. TASK-34100.2
added two facts to it -- a rejected key's reason in the provider's own words
(review F3) and the Console's model-level capacity blocker with its fix
(AC#2) -- and ``settings_screen.py`` sits at its size-ratchet pin
(``Tests/Architecture/test_module_size_ratchet.py``), so the Key row's
verdict, the Model row's capacity note and the bookkeeping behind them live
here. The screen makes one call per seam.

Nothing here reads widgets. ``reply_max_tokens`` and the rejection record
read the screen's staged draft config and its per-session ``__dict__`` slot.
"""

from __future__ import annotations

from typing import Any

#: Where a Test's rejected-key reason is kept: provider key -> (credential
#: revision, reason). Per screen instance; a later Test of the provider
#: replaces or clears it.
_REJECTIONS_SLOT = "_provider_key_rejections"


def key_verdict(
    readiness: Any,
    *,
    evidence: object,
    listing: str,
    listed: str,
    rejection_reason: str = "",
) -> str:
    """The Key row's suffix: what the last listing or test proved of the key.

    Args:
        readiness: ``ProviderReadiness`` of the draft-overlaid config.
        evidence: Stored evidence, or a settled probe, for this draft.
        listing: The Endpoint row's listing state ("reachable", ...).
        listed: The listed-models phrase for an accepted key.
        rejection_reason: A rejected key's reason in the provider's words.

    Returns:
        ``" · key accepted (...)"`` and the like, or ``""`` with no key.
    """
    from tldw_chatbook.Chat.provider_test_evidence import (
        PUBLIC_MODEL_LISTING_PROVIDER_KEYS,
    )

    if not (readiness.api_key_source or ""):
        return ""
    credential = getattr(evidence, "credential", "")
    if credential == "listing_accepted":
        return f" · key accepted ({listed})"
    if credential == "authenticated":
        return " · accepted by a generation test"
    if (
        listing == "reachable"
        and readiness.provider_key in PUBLIC_MODEL_LISTING_PROVIDER_KEYS
    ):
        # ADR-020: this listing is public, so it proves nothing (AC#6).
        return " · models listed; key not checked"
    if listing == "unreachable" and getattr(evidence, "category", None) == "unauthorized":
        # Only a 401 rejects a key (Qodo #2958, owner ruling).
        return f" · key rejected: {rejection_reason}" if rejection_reason else " · key rejected"
    return " · present, not verified"


def model_capacity(
    provider_key: str,
    model: str,
    max_tokens: int | None,
    *,
    from_model_default: bool = False,
) -> tuple[str, bool]:
    """The Model row's capacity note and whether the model is blocked.

    TASK-34100.2 AC#2: the Console's model-level blocker (the reply
    reservation leaves no input capacity), named with its fix. Review round
    1 F1: when the model's own saved default sets ``max_tokens``, the fix
    names that default, not the provider's Max tokens field.

    Args:
        provider_key: The provider under test.
        model: The model under test.
        max_tokens: The reply maximum a new chat of this pair reserves.
        from_model_default: The model's saved default sets ``max_tokens``.

    Returns:
        (suffix for the Model row, whether a model-level blocker applies).
    """
    from tldw_chatbook.Chat.console_session_settings import (
        model_level_not_ready_words,
    )

    blocked = model_level_not_ready_words(provider_key, model, max_tokens)
    if not (blocked and blocked.endswith("window too small")):
        return "", bool(blocked)
    lower = (
        f"lower {model}'s saved Max tokens default in Chat settings"
        if from_model_default
        else "lower Max tokens"
    )
    return f" · replies fill its whole context window — {lower} or choose another model", True


def rejected_key_copy(display_name: str, reason: str) -> str:
    """The Test summary for a key the provider rejected, in its own words."""
    return f"{display_name} rejected the API key: {reason} Enter a valid key, then Test again"


def reply_max_tokens(screen: Any, provider: str, model: str) -> tuple[int | None, bool]:
    """The reply maximum a new chat of this draft pair reserves.

    TASK-34100.2 review round 1 F1: the Console's default chain, never the
    provider table alone.

    Args:
        screen: The Settings screen (its staged draft config).
        provider: The provider under test.
        model: The model under test.

    Returns:
        (the value or None, whether the model's own saved default sets it).
    """
    from tldw_chatbook.Chat.console_session_settings import default_reply_max_tokens
    from tldw_chatbook.Chat.provider_readiness import provider_config_key

    try:
        staged = screen._provider_test_staged_config(provider)
    except Exception:
        return None, False
    value, from_model_default = default_reply_max_tokens(
        staged, provider_config_key(provider), model.strip() or None
    )
    if type(value) is not int or value <= 0:
        return None, False
    return value, from_model_default


def remember_key_rejection(
    screen: Any, provider: str, api_key: str, result: object, probe: object
) -> None:
    """Keep the provider's own reason for a rejected key (review F3).

    Only for the key that was sent: the Key row and the toast quote it for
    THIS key, and any later Test of the provider replaces or clears it.
    """
    from tldw_chatbook.Chat.provider_readiness import provider_config_key
    from tldw_chatbook.Chat.provider_test_evidence import (
        connection_credential_revision,
    )

    reason = getattr(getattr(result, "error", None), "provider_reason", "")
    rejections = screen.__dict__.setdefault(_REJECTIONS_SLOT, {})
    rejections.pop(provider_config_key(provider), None)
    if reason and api_key and getattr(probe, "category", None) == "unauthorized":
        rejections[provider_config_key(provider)] = (
            connection_credential_revision(api_key),
            str(reason),
        )


def test_row_extras(
    screen: Any, provider: str, model: str, evidence: object
) -> dict[str, Any]:
    """The extra ``_provider_test_rows`` arguments TASK-34100.2 added.

    Args:
        screen: The Settings screen.
        provider: The provider under test.
        model: The model under test.
        evidence: Stored evidence, or a settled probe, for this draft.

    Returns:
        ``max_tokens``, ``max_tokens_from_model_default`` and
        ``rejection_reason`` (the remembered reason, only while the
        evidence is for the key that was rejected).
    """
    from tldw_chatbook.Chat.provider_readiness import provider_config_key

    rejection = screen.__dict__.get(_REJECTIONS_SLOT, {}).get(
        provider_config_key(provider) if provider else ""
    )
    identity = getattr(evidence, "identity", None)
    reason = (
        rejection[1]
        if rejection is not None
        and (
            identity is None
            or getattr(identity, "credential_revision", None) == rejection[0]
        )
        else ""
    )
    max_tokens, from_model_default = reply_max_tokens(screen, provider, model)
    return {
        "max_tokens": max_tokens,
        "max_tokens_from_model_default": from_model_default,
        "rejection_reason": reason,
    }
