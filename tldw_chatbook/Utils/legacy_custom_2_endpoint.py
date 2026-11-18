"""Keep an explicit legacy Custom OpenAI API 2 endpoint in the merged config.

Kernel code: ``config._merge_user_config_over_shipped_defaults`` calls it, so it
lives below Chat (Docs/Architecture/layer-map.md); ``Chat.console_provider_endpoints``
imports it from here.
"""

from __future__ import annotations

from collections.abc import Mapping

#: The code default of the legacy ``[API] custom_openai2_api_ip``
#: (``config.load_settings``), which is not an explicit endpoint.
LEGACY_CUSTOM_2_API_IP_DEFAULT = "http://127.0.0.1:5000/v1/chat/completions"

#: Where ``keep_legacy_custom_2_endpoint`` records a bridge in the merged
#: view (a provenance key, like ``legacy_api_key_providers``; never written).
LEGACY_CUSTOM_2_BRIDGE = "legacy_custom_2_endpoint_bridge"


def keep_legacy_custom_2_endpoint(
    merged: dict, legacy: object, template: Mapping[str, object]
) -> None:
    """Keep an explicit legacy Custom OpenAI API 2 endpoint in the merged view.

    The adapter reads ``[api_settings.custom_2] api_url`` before the legacy
    ``[API] custom_openai2_api_ip`` (owner decision 2026-11-16). An explicit
    legacy endpoint must not be silently redirected (ruling 2026-11-17), so
    while the slot's ``api_url`` is absent, blank or still the shipped
    template's placeholder, and the legacy value is anything but its default,
    the view's ``api_url`` takes the legacy value. In memory only, like
    ``config._normalize_legacy_provider_api_key``: every reader (the adapter,
    ``adapter_own_endpoint``, Console, Settings) sees one value, and the
    user's file is never written.

    Args:
        merged: The user's config merged over the template; changed in place.
        legacy: The user's ``[API] custom_openai2_api_ip``.
        template: The shipped template (``DEFAULT_CONFIG_FROM_TOML``).
    """
    if not isinstance(legacy, str) or legacy.strip() in {"", LEGACY_CUSTOM_2_API_IP_DEFAULT}:
        return
    api_settings = merged.get("api_settings")
    slot = api_settings.get("custom_2") if isinstance(api_settings, dict) else None
    shipped = ((template.get("api_settings") or {}).get("custom_2") or {}).get("api_url")
    if isinstance(slot, dict) and str(slot.get("api_url") or "").strip() in {"", shipped}:
        # What the slot read before, so a save that leaves the Endpoint alone
        # writes that, not the legacy value (review r15ep-rev2 I-1).
        merged[LEGACY_CUSTOM_2_BRIDGE] = {
            "legacy": legacy.strip(),
            "before": slot.get("api_url") or "",
        }
        slot["api_url"] = legacy.strip()
