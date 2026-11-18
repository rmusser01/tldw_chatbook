"""Which OpenAI key the first-run Voice step would test and save with.

Moved out of ``first_run_voice_step.py`` (TASK-34100.8 review round 1, F9) so
the step stays under its size budget. Every candidate goes through
``resolve_provider_api_key``, the validity check CLAUDE.md requires of code
that reads a key from config directly: the shipped ``<API_KEY_HERE>``
placeholder, a blank or padded value, or still-encrypted ``enc:`` ciphertext
must never read as "key found" or travel as a Bearer token.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from tldw_chatbook.config import resolve_provider_api_key
from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state

#: Where Settings ▸ Speech & TTS and the OpenAI backend look, in order.
_SAVED_KEY_LOCATIONS = (
    ("api_settings", "openai", "api_key"),
    ("openai_api", "api_key"),
    ("API", "openai_api_key"),
)


def _lookup(source: Mapping[str, object], path: tuple[str, ...]) -> object:
    current: object = source
    for part in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def find_openai_credential(
    app_config: object,
    *,
    staged_key: wizard_state.ProviderCredentialDraft | None,
    staged_provider_draft: object = None,
    provider_setup_committed: bool = False,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, bool] | None:
    """The OpenAI key a Voice test or save would use, and whether Next writes it.

    Order: a key pasted in the Voice step; then the saved and environment
    locations Settings reads; then the key the Provider step staged for
    OpenAI but has not written (Model was skipped). The first and last are
    written by the Voice step's save.

    Args:
        app_config: The app's loaded settings (carrying
            ``COMPREHENSIVE_CONFIG_RAW``).
        staged_key: A key pasted in the Voice step, if any.
        staged_provider_draft: The Provider step's staged draft, if any.
        provider_setup_committed: Whether the Provider step already wrote it.
        environ: The process environment (``os.environ`` by default).

    Returns:
        ``(key, written_by_this_step)``, or None when no usable key exists.
    """
    env = os.environ if environ is None else environ
    if staged_key is not None:
        value = resolve_provider_api_key(
            wizard_state._credential_value_for_boundary(staged_key)
        )
        if value:
            return value, True
    if isinstance(app_config, Mapping):
        persisted = app_config.get("COMPREHENSIVE_CONFIG_RAW")
        source = persisted if isinstance(persisted, Mapping) else app_config
        for location in _SAVED_KEY_LOCATIONS:
            value = resolve_provider_api_key(_lookup(source, location))
            if value:
                return value, False
        environment_name = _lookup(
            source, ("api_settings", "openai", "api_key_env_var")
        )
        if isinstance(environment_name, str) and environment_name:
            value = resolve_provider_api_key(env.get(environment_name))
            if value:
                return value, False
        value = resolve_provider_api_key(app_config.get("OPENAI_API_KEY"))
        if value:
            return value, False
    value = resolve_provider_api_key(env.get("OPENAI_API_KEY"))
    if value:
        return value, False
    if (
        isinstance(staged_provider_draft, wizard_state.FirstRunProviderDraft)
        and staged_provider_draft.provider == "openai"
        and staged_provider_draft.credential.source == "draft"
        and not provider_setup_committed
    ):
        value = resolve_provider_api_key(
            wizard_state._credential_value_for_boundary(
                staged_provider_draft.credential
            )
        )
        if value:
            return value, True
    return None


__all__ = ["find_openai_credential"]
