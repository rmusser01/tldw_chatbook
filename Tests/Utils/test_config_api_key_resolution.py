"""`get_api_key()` must read the config shape the app actually loads.

Sibling bug to TASK-229 (`test_config_nested_settings.py`, same directory,
same root cause one accessor over): `get_api_key` tested membership of a
FLAT DOTTED key -- `"api_settings.openai" in settings` -- while
`load_settings()` returns a NESTED dict (`{"api_settings": {"openai":
{...}}}`). No real config has ever produced that flat key, so the primary
branch was dead code and every key entered through the Settings screen --
which writes `api_settings.<provider>.api_key`, the correct modern
location -- was invisible to every consumer of this function.

Found at the realtime engine's live gate: the user's config had a real
OpenAI key stored by the Settings screen, and the pre-connect check
refused with "no OpenAI API key is configured" -- honestly reporting a
lookup that could never have succeeded.

Every test here drives the REAL loader via `TLDW_CONFIG_PATH` +
force_reload rather than monkeypatching `load_settings`, for the reason
the sibling file records: the shape returned by the real loader IS the
thing under test, and an accessor mock would have agreed with whatever
shape the test author imagined (which is exactly how this bug survived).
The one exception is the flat-dotted fallback, which no TOML file can
produce -- see that test's own comment.
"""

import os
from contextlib import contextmanager

import tldw_chatbook.config as config_mod
from tldw_chatbook.config import get_api_key

#: Names an env var this suite guarantees is unset, so the nested branch's
#: `api_key_env_var` override cannot fire and mask the stored-key path.
UNSET_ENV_VAR = "TLDW_TEST_UNSET_OPENAI_KEY_VAR"

NESTED_STORED_KEY_TOML = f"""
[api_settings.openai]
api_key = "sk-test-nested-stored-key"
api_key_env_var = "{UNSET_ENV_VAR}"
"""

NESTED_PLACEHOLDER_TOML = f"""
[api_settings.openai]
api_key = "<API_KEY_HERE>"
api_key_env_var = "{UNSET_ENV_VAR}"
"""

LEGACY_API_SECTION_TOML = f"""
[api_settings.openai]
api_key = "<API_KEY_HERE>"
api_key_env_var = "{UNSET_ENV_VAR}"

[API]
openai_api_key = "sk-test-legacy-section-key"
"""


@contextmanager
def _real_config(tmp_path, monkeypatch, toml_text: str):
    """Point the real loader at a scratch TOML; restore + reload afterwards.

    Copied deliberately from `test_config_nested_settings.py` -- same
    isolation contract, same teardown, so the two suites cannot drift on
    how "the real loader" is driven.
    """
    config_path = tmp_path / "scratch-api-key-config.toml"
    config_path.write_text(toml_text, encoding="utf-8")
    original_env = os.environ.get("TLDW_CONFIG_PATH")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    config_mod.load_cli_config_and_ensure_existence(force_reload=True)
    config_mod.load_settings(force_reload=True)
    try:
        yield
    finally:
        if original_env is not None:
            monkeypatch.setenv("TLDW_CONFIG_PATH", original_env)
        else:
            monkeypatch.delenv("TLDW_CONFIG_PATH", raising=False)
        config_mod.load_cli_config_and_ensure_existence(force_reload=True)
        config_mod.load_settings(force_reload=True)


def _clear_provider_env(monkeypatch) -> None:
    """Remove every env var that could satisfy the lookup by accident.

    Without this a developer machine with `OPENAI_API_KEY` exported would
    pass these tests through the LAST fallback while the branch under test
    stayed broken -- the failure mode this whole file exists to catch.
    """
    for name in ("OPENAI_API_KEY", UNSET_ENV_VAR):
        monkeypatch.delenv(name, raising=False)


def test_nested_api_settings_api_key_resolves(tmp_path, monkeypatch):
    """The shape the Settings screen actually writes must resolve.

    RED against the pre-fix code: `"api_settings.openai" in settings` is
    False for every real config, so this returned None.
    """
    _clear_provider_env(monkeypatch)
    with _real_config(tmp_path, monkeypatch, NESTED_STORED_KEY_TOML):
        assert get_api_key("openai") == "sk-test-nested-stored-key"


def test_provider_name_is_case_insensitive(tmp_path, monkeypatch):
    """`api_name.lower()` is the documented normalization -- it has to keep
    applying to the nested lookup, not just the dead flat one."""
    _clear_provider_env(monkeypatch)
    with _real_config(tmp_path, monkeypatch, NESTED_STORED_KEY_TOML):
        assert get_api_key("OpenAI") == "sk-test-nested-stored-key"


def test_env_var_named_by_config_wins_over_the_stored_key(tmp_path, monkeypatch):
    """Precedence WITHIN the branch is unchanged: `api_key_env_var` first,
    stored `api_key` second."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv(UNSET_ENV_VAR, "sk-test-from-env-var")
    with _real_config(tmp_path, monkeypatch, NESTED_STORED_KEY_TOML):
        assert get_api_key("openai") == "sk-test-from-env-var"


def test_placeholder_api_key_is_never_returned(tmp_path, monkeypatch):
    """`<API_KEY_HERE>` is the shipped placeholder -- returning it would
    send a literal placeholder to a provider as a bearer token."""
    _clear_provider_env(monkeypatch)
    with _real_config(tmp_path, monkeypatch, NESTED_PLACEHOLDER_TOML):
        assert get_api_key("openai") is None


def test_legacy_api_section_still_resolves(tmp_path, monkeypatch):
    """The legacy `[API] <provider>_api_key` fallback is untouched."""
    _clear_provider_env(monkeypatch)
    with _real_config(tmp_path, monkeypatch, LEGACY_API_SECTION_TOML):
        assert get_api_key("openai") == "sk-test-legacy-section-key"


def test_direct_environment_variable_still_resolves(tmp_path, monkeypatch):
    """The last fallback (`<PROVIDER>_API_KEY`) is untouched."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-direct-env")
    with _real_config(tmp_path, monkeypatch, NESTED_PLACEHOLDER_TOML):
        assert get_api_key("openai") == "sk-test-direct-env"


def test_unknown_provider_returns_none(tmp_path, monkeypatch):
    _clear_provider_env(monkeypatch)
    with _real_config(tmp_path, monkeypatch, NESTED_STORED_KEY_TOML):
        assert get_api_key("not-a-provider") is None


def test_flat_dotted_settings_shape_is_still_supported(monkeypatch):
    """The ONE test here that monkeypatches `load_settings`, on purpose.

    A flat `"api_settings.openai"` key cannot come out of a TOML file --
    `[api_settings.openai]` always loads nested -- so a real-loader test
    cannot cover the flat fallback that was kept alongside the nested
    lookup. Pinned here so the fallback cannot be dropped silently by a
    later cleanup, and marked clearly as the exception it is.
    """
    for name in ("OPENAI_API_KEY", UNSET_ENV_VAR):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        config_mod,
        "load_settings",
        lambda *args, **kwargs: {
            "api_settings.openai": {
                "api_key": "sk-test-flat-shape",
                "api_key_env_var": UNSET_ENV_VAR,
            }
        },
    )
    assert get_api_key("openai") == "sk-test-flat-shape"
