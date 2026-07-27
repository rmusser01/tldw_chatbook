"""Tests for the shared sensitive-config-key predicate (task-852).

Before ``Utils/sensitive_config_keys.py`` existed, four independent copies of
"is this config key a secret" disagreed with each other: an exact-6-literal
match in ``Utils/config_encryption.py`` never matched any real provider key
name in this app (they are all prefixed/suffixed, e.g. ``openai_api_key``,
``bing_search_api_key``, ``OPENAI_API_KEY_fallback``); a substring match in
``config.py`` over-flagged ``max_tokens`` and ``api_key_env_var``; an
endswith match in ``settings_privacy_security.py`` under-counted keys where
the sensitive fragment isn't a suffix (``search_engine_api_key_baidu``,
``OPENAI_API_KEY_fallback``).

The real-key-name tests below deliberately read key names out of
``tldw_chatbook.config.CONFIG_TOML_CONTENT`` (the app's actual shipped
default config) and ``DEFAULT_APP_TTS_CONFIG`` (the other real default
source for the ``_fallback`` key family), rather than re-typing a
hand-picked literal list -- a hand-typed list would silently go vacuous the
next time the real config drifted, which is exactly the failure mode that
produced this bug in the first place.
"""

import sys

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib

from tldw_chatbook.config import CONFIG_TOML_CONTENT, DEFAULT_APP_TTS_CONFIG
from tldw_chatbook.Utils.sensitive_config_keys import is_sensitive_config_key


DEFAULT_CONFIG = tomllib.loads(CONFIG_TOML_CONTENT)


def _iter_leaf_key_names(mapping):
    """Yield every leaf key name (not path) found anywhere in ``mapping``."""
    for key, value in mapping.items():
        if isinstance(value, dict):
            yield from _iter_leaf_key_names(value)
        else:
            yield key


class TestRealConfigKeyNamesAreDetected:
    """AC #1/#4: real provider-key config names, sourced from the app's own
    default config, must be detected -- not just synthetic literals."""

    def test_search_engine_provider_keys_are_detected(self):
        # Real TOML keys under [SearchEngines], read from the shipped
        # default config rather than typed by hand.
        search_engines = DEFAULT_CONFIG["SearchEngines"]
        provider_key_names = [
            key for key in search_engines if key.endswith("_api_key")
        ]
        # Sanity: the default config still actually ships this family.
        assert len(provider_key_names) >= 5
        for key_name in provider_key_names:
            assert is_sensitive_config_key(key_name), (
                f"expected {key_name!r} (a real [SearchEngines] key) to be "
                "detected as sensitive"
            )

    def test_provider_api_settings_key_is_detected(self):
        # [api_settings.google] is the one provider section that ships an
        # uncommented literal "api_key" in the real default config (every
        # other provider defaults to *_api_key_env_var instead); read the
        # key name from there rather than typing "api_key" disconnected
        # from the source.
        google_section = DEFAULT_CONFIG["api_settings"]["google"]
        (real_key_name,) = [key for key in google_section if key == "api_key"]
        assert is_sensitive_config_key(real_key_name)

    def test_tldw_api_auth_token_is_detected(self):
        assert "auth_token" in DEFAULT_CONFIG["tldw_api"]
        assert is_sensitive_config_key("auth_token")

    def test_github_api_token_is_detected(self):
        assert "api_token" in DEFAULT_CONFIG["github"]
        assert is_sensitive_config_key("api_token")

    def test_tts_fallback_api_key_family_is_detected(self):
        # Real keys from the other live default source referenced by
        # task-852 (config.py:66,70), written by the TTS settings form.
        fallback_key_names = [
            key
            for key in DEFAULT_APP_TTS_CONFIG
            if "api_key" in key.lower() or "apikey" in key.lower()
        ]
        assert len(fallback_key_names) >= 2
        for key_name in fallback_key_names:
            assert is_sensitive_config_key(key_name), (
                f"expected {key_name!r} (a real DEFAULT_APP_TTS_CONFIG key) "
                "to be detected as sensitive"
            )


class TestRealNonSecretKeyNamesAreExcluded:
    """AC #3: keys the old substring-matching copy over-flagged must not be
    treated as secrets, using the real key names that triggered the bug."""

    def test_env_var_name_keys_are_excluded(self):
        api_settings = DEFAULT_CONFIG["api_settings"]
        env_var_key_names = {
            key
            for section in api_settings.values()
            for key in section
            if key.endswith("_env_var")
        }
        assert "api_key_env_var" in env_var_key_names
        for key_name in env_var_key_names:
            assert not is_sensitive_config_key(key_name), (
                f"{key_name!r} names an environment variable, not a secret"
            )

        assert "api_token_env_var" in DEFAULT_CONFIG["github"]
        assert not is_sensitive_config_key("api_token_env_var")

    def test_max_tokens_is_excluded(self):
        # config.py's old substring-matching copy flagged this because it
        # contains "_token" as a bare substring.
        assert "max_tokens" in DEFAULT_CONFIG["api_settings"]["openai"]
        assert not is_sensitive_config_key("max_tokens")

    def test_keyword_and_keypress_settings_are_not_flagged(self):
        """Real non-secret keys from the default config that a naive bare
        "_key" substring match (the old config.py behavior) wrongly flagged,
        because each embeds "_key" as part of an unrelated word
        (keyword/keypress), not as a suffix."""
        keyword_settings = {
            "default_keyword_filter",
            "extract_keywords",
            "max_keywords",
            "skip_on_keypress",
            "auto_save_on_every_key",
        }
        for key_name in keyword_settings:
            assert key_name in set(_iter_leaf_key_names(DEFAULT_CONFIG)), (
                f"expected {key_name!r} to still be a real default config key"
            )
        # "auto_save_on_every_key" genuinely ends with "_key" (it is about
        # keystrokes, not API keys); it is a boolean setting so it is never
        # encrypted regardless, but it is not asserted here as excluded.
        for key_name in keyword_settings - {"auto_save_on_every_key"}:
            assert not is_sensitive_config_key(key_name), (
                f"expected {key_name!r} to NOT be treated as a secret"
            )


class TestSensitiveConfigKeyPredicateSemantics:
    """Direct unit coverage of the predicate's documented rules."""

    def test_exact_literal_names(self):
        for key_name in (
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
        ):
            assert is_sensitive_config_key(key_name)

    def test_case_insensitive(self):
        assert is_sensitive_config_key("API_KEY")
        assert is_sensitive_config_key("Auth_Token")

    def test_prefixed_and_suffixed_api_key_variants(self):
        assert is_sensitive_config_key("openai_api_key")
        assert is_sensitive_config_key("bing_search_api_key")
        assert is_sensitive_config_key("search_engine_api_key_baidu")
        assert is_sensitive_config_key("OPENAI_API_KEY_fallback")
        assert is_sensitive_config_key("ELEVENLABS_API_KEY_fallback")

    def test_env_var_guard_overrides_everything(self):
        assert not is_sensitive_config_key("api_key_env_var")
        assert not is_sensitive_config_key("ANYTHING_API_KEY_ENV_VAR")

    def test_max_tokens_style_false_positive_excluded(self):
        assert not is_sensitive_config_key("max_tokens")
        assert not is_sensitive_config_key("chat_default_max_tokens")

    def test_non_secret_settings_excluded(self):
        assert not is_sensitive_config_key("default_tab")
        assert not is_sensitive_config_key("timeout")
        assert not is_sensitive_config_key("model")

    def test_empty_and_non_string_keys(self):
        assert not is_sensitive_config_key("")
        assert not is_sensitive_config_key(None)
        assert is_sensitive_config_key(123) is False
