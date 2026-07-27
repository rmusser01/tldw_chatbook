# tldw_chatbook/Utils/sensitive_config_keys.py
"""Single source of truth for "is this config key a secret".

Before this module existed, four call sites each carried their own
hand-rolled notion of a sensitive config key, and they disagreed with each
other in both directions:

* ``config.py`` (``encrypt_api_keys_in_config``'s ``encrypt_sensitive_fields``
  closure, and the near-duplicate ``_is_sensitive_setting_key``) matched via
  bare substring containment, so ``max_tokens`` (contains ``_token``) and
  ``api_key_env_var`` (contains ``api_key``) were both wrongly flagged as
  secrets -- the former got redacted in logs for no reason, the latter risked
  encrypting an *environment variable name* rather than the secret it points
  to.
* ``Utils/config_encryption.py``'s ``detect_api_keys`` matched only six exact
  literal names (``api_key``, ``apikey``, ``api-key``, ``secret``, ``token``,
  ``password``). Every real secret-bearing key in this app is prefixed or
  suffixed instead (``openai_api_key``, ``bing_search_api_key``,
  ``api_token``, ``OPENAI_API_KEY_fallback``, ...), so none of them were ever
  detected and a config full of live plaintext secrets reported nothing
  worth encrypting.
* ``UI/Screens/settings_privacy_security.py``'s ``_is_sensitive_config_key``
  used ``endswith`` (safer than substring) plus an ``_env_var`` guard, but
  being ``endswith``-only it missed keys where the sensitive fragment is not
  the suffix, e.g. ``search_engine_api_key_baidu`` or the ``_fallback``
  family (``OPENAI_API_KEY_fallback``), undercounting what Settings > Privacy
  & Security reports as protected.

This module fixes all three failure modes in one predicate, used by every
one of those call sites so they cannot drift from each other again.
"""

from __future__ import annotations

# Keys that are secrets outright, regardless of prefix/suffix.
SENSITIVE_CONFIG_EXACT_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "api-key",
        "secret",
        "token",
        "password",
        "auth_token",
        "api_token",
        "access_token",
        "secret_key",
        "refresh_token",
        "client_secret",
    }
)

# Fragments that mark a key as a secret no matter where they appear in it.
# Unlike the suffix list below, these are matched by containment: real
# provider keys embed "api_key" mid-name in both directions -- as a prefix
# (``search_engine_api_key_baidu``) and as a suffix-of-a-suffix
# (``OPENAI_API_KEY_fallback``, ``ELEVENLABS_API_KEY_fallback``) -- so an
# endswith-only rule misses them.
SENSITIVE_CONFIG_KEY_CONTAINS_PATTERNS = (
    "api_key",
    "apikey",
    "api-key",
)

# Suffixes that mark a key as a secret. These are endswith-only (not
# containment) on purpose: ``_token`` as a containment check would also
# match ``max_tokens`` (which merely *limits* a token count, it does not
# hold one), turning a numeric setting into a false positive.
SENSITIVE_CONFIG_KEY_SUFFIXES = (
    "_key",
    "_token",
    "_secret",
    "_password",
)

# Keys ending in this suffix name an *environment variable* that holds the
# secret (e.g. ``api_key_env_var = "OPENAI_API_KEY"``); the env var name
# itself is not a secret and must never be encrypted or redacted as one.
_ENV_VAR_NAME_SUFFIX = "_env_var"


def is_sensitive_config_key(key: object) -> bool:
    """Return whether a config key name is expected to hold a secret value.

    Args:
        key: The config key name to check. Coerced to ``str`` so callers may
            pass non-string mapping keys without raising.

    Returns:
        True if the key should be treated as secret-bearing (encrypted when
        config encryption is enabled, redacted in logs, counted as a
        protected field in the Privacy & Security posture).
    """
    key_text = str(key).strip().lower()
    if not key_text or key_text.endswith(_ENV_VAR_NAME_SUFFIX):
        return False
    if key_text in SENSITIVE_CONFIG_EXACT_KEYS:
        return True
    if any(
        pattern in key_text for pattern in SENSITIVE_CONFIG_KEY_CONTAINS_PATTERNS
    ):
        return True
    return any(key_text.endswith(suffix) for suffix in SENSITIVE_CONFIG_KEY_SUFFIXES)
