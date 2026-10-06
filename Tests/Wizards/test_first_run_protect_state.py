"""Pure rules behind the first-run Protect step (TASK-34100.4).

``first_run_protect_state`` turns the persisted config into one of five
states and the copy for each; ``first_run_setup_state`` turns the step's
recorded state into the tracker glyph and the Summary row. No Textual here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tldw_chatbook.UI.Screens.settings_encryption import ENCRYPTION_CARD_PATH
from tldw_chatbook.UI.Wizards.first_run_protect_state import (
    NOTHING_TO_PROTECT,
    PROTECT_ENCRYPTED,
    PROTECT_ENV_ONLY,
    PROTECT_NOTHING,
    PROTECT_PLAINTEXT,
    PROTECT_UNREADABLE,
    PROTECT_UNSAVED,
    ProtectState,
    protect_copy,
    read_protect_state,
    typed_provider_key,
)
from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    STEP_PROTECT,
    STEP_PROVIDER,
    build_summary_rows,
    progress_glyph,
    setup_progress_items,
)
from tldw_chatbook.Utils.config_encryption import ConfigEncryption

_TRACK = ("welcome", "provider", "model", "voice", "protect-keys", "summary")
_WIZARDS = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "UI" / "Wizards"


def test_card_path_names_the_settings_encryption_card():
    assert ENCRYPTION_CARD_PATH == "Settings ▸ Privacy & Security ▸ Encryption"


def test_encryption_on_wins_even_over_decrypted_keys():
    state = read_protect_state(
        {
            "encryption": {"enabled": True},
            "api_settings": {"openai": {"api_key": "sk-decrypted"}},
        },
        {},
    )
    assert state.kind == PROTECT_ENCRYPTED


def test_plaintext_keys_are_named_in_config_order_and_placeholders_skipped():
    state = read_protect_state(
        {
            "api_settings": {
                "anthropic": {"api_key": "sk-ant-x"},
                "google": {"api_key": "<API_KEY_HERE>"},
                "openai": {"api_key": "sk-openai-x"},
                "llama_cpp": {"api_url": "http://127.0.0.1:8080"},
            }
        },
        {},
    )
    assert state == ProtectState(PROTECT_PLAINTEXT, providers=("Anthropic", "OpenAI"))


def test_a_stranded_encrypted_key_is_not_offered_as_plain_text():
    ciphertext = ConfigEncryption().encrypt_value("sk-real", "earlier-pw")
    state = read_protect_state(
        {"api_settings": {"openai": {"api_key": ciphertext}}}, {}
    )
    assert state.kind == PROTECT_NOTHING


def test_a_legacy_api_table_key_is_offered_and_named():
    # Review round 1 (R1-F1/TE-1): `[API] openai_api_key` is a supported
    # source that encrypting does encrypt, but Protect looked only at
    # api_settings and said "Nothing to protect".
    state = read_protect_state({"API": {"openai_api_key": "sk-legacy-plain"}}, {})

    assert state == ProtectState(PROTECT_PLAINTEXT, providers=("OpenAI",))


def test_a_legacy_alias_key_is_named_without_the_catalog_aside():
    # `[API] mistral_api_key` bridges to the `mistral` alias, whose catalog
    # name is "Mistral AI (legacy alias)" -- an aside for the provider
    # picker, not copy for "Your … key is saved as plain text".
    state = read_protect_state({"API": {"mistral_api_key": "mistral-real-key"}}, {})

    assert state == ProtectState(PROTECT_PLAINTEXT, providers=("Mistral AI",))


def test_a_custom_endpoint_key_is_named_by_its_endpoint():
    state = read_protect_state(
        {
            "custom_endpoints": {
                "home": {
                    "display_name": "Home llama",
                    "family": "llama_cpp",
                    "base_url": "http://127.0.0.1:8181",
                    "api_key": "local-server-key",
                }
            }
        },
        {},
    )

    assert state == ProtectState(PROTECT_PLAINTEXT, providers=("Home llama",))


def test_other_saved_secrets_are_offered_and_named_by_their_config_key():
    state = read_protect_state(
        {
            "api_settings": {"openai": {"api_key": "sk-openai-x"}},
            "SearchEngines": {"serper_search_api_key": "serper-real-key"},
        },
        {},
    )

    assert state == ProtectState(
        PROTECT_PLAINTEXT,
        providers=("OpenAI",),
        other_secrets=("SearchEngines.serper_search_api_key",),
    )
    body = protect_copy(state, config_path="/p/config.toml").body
    assert (
        "Your OpenAI key and 1 other secret (SearchEngines.serper_search_api_key) "
        "are saved as plain text" in body
    )


def test_only_other_secrets_still_get_the_offer():
    state = read_protect_state(
        {"SearchEngines": {"serper_search_api_key": "serper-real-key"}}, {}
    )
    copy = protect_copy(state, config_path="/p/config.toml")

    assert state.kind == PROTECT_PLAINTEXT
    assert copy.primary_label == "Encrypt with a password…"
    assert "1 secret (SearchEngines.serper_search_api_key) is saved" in copy.body


def test_the_shipped_template_values_are_not_secrets():
    # The merged read-back always carries the template's placeholders
    # (`YOUR_OPENAI_API_KEY…`, `<API_KEY_HERE>`, the tldw_api placeholder
    # token); none of them is the user's key.
    from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML

    assert read_protect_state(DEFAULT_CONFIG_FROM_TOML, {}).kind == PROTECT_NOTHING


def test_a_typed_key_that_is_not_on_disk_is_unsaved():
    state = read_protect_state({}, {}, typed_provider_key="anthropic")
    assert state == ProtectState(PROTECT_UNSAVED, unsaved_provider="Anthropic")


def test_an_unsaved_typed_key_outranks_env_only_keys():
    # Review round 1 (TE-6): the typed key is the one the user expects to be
    # saved, so it is called out even when an env var also supplies a key.
    state = read_protect_state(
        {"api_settings": {"openai": {"api_key_env_var": "OPENAI_API_KEY"}}},
        {"OPENAI_API_KEY": "sk-env-value"},
        typed_provider_key="anthropic",
    )

    assert state == ProtectState(PROTECT_UNSAVED, unsaved_provider="Anthropic")


def test_an_unsaved_typed_key_is_still_called_out_beside_a_stored_one():
    # Review round 1 (TE-6): a stored OpenAI key gets the offer, and the
    # Anthropic key typed this run but never saved is not silently dropped.
    state = read_protect_state(
        {"api_settings": {"openai": {"api_key": "sk-openai-x"}}},
        {},
        typed_provider_key="anthropic",
    )
    copy = protect_copy(state, config_path="/p/config.toml")

    assert state == ProtectState(
        PROTECT_PLAINTEXT, providers=("OpenAI",), unsaved_provider="Anthropic"
    )
    assert "The Anthropic key you typed isn't saved" in copy.body
    assert copy.show_keep_plain and copy.show_back_to_model


def test_a_typed_key_that_is_on_disk_is_not_called_out():
    state = read_protect_state(
        {"API": {"openai_api_key": "sk-legacy-plain"}}, {}, typed_provider_key="openai"
    )

    assert state == ProtectState(PROTECT_PLAINTEXT, providers=("OpenAI",))


# --- the real config read-back (TE-1: no hand-built dict) ---------------------


@pytest.fixture
def real_config_file(tmp_path, monkeypatch):
    """Write config.toml and select it as the real config module's source."""
    from Tests.Backup_Recovery.config_test_support import install_config_source

    def _select(text: str | None):
        path = tmp_path / "profile" / "config.toml"
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if text is not None:
            path.write_text(text)
            path.chmod(0o600)
        monkeypatch.setenv("TLDW_CONFIG_PATH", str(path))
        module = install_config_source(monkeypatch)
        monkeypatch.setattr(
            module, "DEFAULT_CONFIG_PATH", tmp_path / "decoy" / "config.toml"
        )
        module.clear_encryption_password()
        module._SETTINGS_CACHE = None
        module._CONFIG_CACHE = None
        return path

    return _select


def _protect_state_from_disk() -> ProtectState:
    from tldw_chatbook.UI.Wizards.first_run_protect_step import (
        _load_persisted_config,
    )

    config, _path = _load_persisted_config()
    return read_protect_state(config, {})


def test_a_legacy_key_in_a_real_config_file_gets_the_offer(real_config_file):
    real_config_file('[API]\nopenai_api_key = "sk-legacy-plain-protect-probe"\n')

    assert _protect_state_from_disk() == ProtectState(
        PROTECT_PLAINTEXT, providers=("OpenAI",)
    )


def test_a_fresh_real_config_file_has_nothing_to_protect(real_config_file):
    real_config_file(None)  # the loader writes the shipped template

    assert _protect_state_from_disk().kind == PROTECT_NOTHING


def test_env_keys_are_named_by_variable():
    state = read_protect_state(
        {
            "api_settings": {
                "openai": {"api_key_env_var": "OPENAI_API_KEY"},
                "groq": {"api_key_env_var": "GROQ_API_KEY"},
            }
        },
        {"OPENAI_API_KEY": "sk-env-value", "GROQ_API_KEY": ""},
    )
    assert state == ProtectState(PROTECT_ENV_ONLY, env_vars=("OPENAI_API_KEY",))


def test_no_key_anywhere_is_nothing():
    assert read_protect_state({"api_settings": {}}, {}).kind == PROTECT_NOTHING


def test_typed_provider_key_reads_the_provider_steps_record():
    assert (
        typed_provider_key(
            {STEP_PROVIDER: {"provider_key": "openai", "entered_key": True}}
        )
        == "openai"
    )
    assert (
        typed_provider_key(
            {STEP_PROVIDER: {"provider_key": "openai", "entered_key": False}}
        )
        == ""
    )
    assert typed_provider_key(None) == ""


_STATES = (
    ProtectState(PROTECT_ENCRYPTED),
    ProtectState(PROTECT_PLAINTEXT, providers=("OpenAI",)),
    ProtectState(PROTECT_PLAINTEXT, providers=("OpenAI", "Anthropic", "Groq")),
    ProtectState(PROTECT_UNSAVED, unsaved_provider="Anthropic"),
    ProtectState(PROTECT_ENV_ONLY, env_vars=("OPENAI_API_KEY",)),
    ProtectState(PROTECT_NOTHING),
    ProtectState(PROTECT_PLAINTEXT, other_secrets=("SearchEngines.serper_api_key",)),
    ProtectState(PROTECT_UNREADABLE),
)


@pytest.mark.parametrize("state", _STATES, ids=lambda s: f"{s.kind}-{len(s.providers)}")
def test_every_state_names_the_card_and_no_skip_control(state):
    copy = protect_copy(state, config_path="/home/u/.config/tldw_cli/config.toml")
    lines = (copy.title, copy.body, copy.path_line, copy.later_line, copy.primary_label)

    assert ENCRYPTION_CARD_PATH in copy.later_line
    assert not any("Skip" in line for line in lines)


@pytest.mark.parametrize("state", _STATES, ids=lambda s: f"{s.kind}-{len(s.providers)}")
def test_only_a_stored_plaintext_key_gets_the_offer(state):
    copy = protect_copy(state, config_path="/p/config.toml")

    offered = copy.primary_label == "Encrypt with a password…"
    assert offered is (state.kind == PROTECT_PLAINTEXT)
    assert copy.show_keep_plain is offered
    assert copy.show_back_to_model is (state.kind == PROTECT_UNSAVED)
    if state.kind in NOTHING_TO_PROTECT:
        assert "master password" not in copy.body  # no pitch
        assert copy.primary_label == ""


def test_plaintext_copy_names_every_provider_and_the_file():
    copy = protect_copy(
        ProtectState(PROTECT_PLAINTEXT, providers=("OpenAI", "Anthropic", "Groq")),
        config_path="/home/u/.config/tldw_cli/config.toml",
    )
    assert "OpenAI, Anthropic and Groq keys are saved as plain text" in copy.body
    assert copy.path_line == "Config file: /home/u/.config/tldw_cli/config.toml"
    assert copy.title == "Protect your API keys"


def test_encrypted_copy_offers_change_password():
    copy = protect_copy(ProtectState(PROTECT_ENCRYPTED), config_path="/p")
    assert copy.primary_label == "Change password…"
    assert "Your saved API keys are encrypted" in copy.body


# --- tracker -------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(NOTHING_TO_PROTECT))
def test_tracker_shows_a_dash_when_there_was_nothing_to_protect(kind):
    items = setup_progress_items(
        _TRACK,
        _TRACK.index("summary"),
        {STEP_PROTECT: {"encryption_enabled": False, "protect_state": kind}},
    )
    protect = next(item for item in items if item.step_id == STEP_PROTECT)

    assert protect.state == "not_needed"
    assert progress_glyph(protect.state, _TRACK.index(STEP_PROTECT)) == "–"


def test_tracker_ticks_protect_only_when_the_keys_are_encrypted():
    items = setup_progress_items(
        _TRACK,
        _TRACK.index("summary"),
        {
            STEP_PROTECT: {
                "encryption_enabled": True,
                "protect_state": PROTECT_ENCRYPTED,
            }
        },
    )
    protect = next(item for item in items if item.step_id == STEP_PROTECT)

    assert protect.state == "complete"
    assert progress_glyph(protect.state, 4) == "✓"


@pytest.mark.parametrize(
    "record",
    [
        # Review round 1 (R1-F7): "Keep as plain text" ticked ✓ while
        # Summary flagged the plain-text keys.
        {"encryption_enabled": False, "protect_state": PROTECT_PLAINTEXT},
        {"encryption_enabled": False, "protect_state": PROTECT_UNREADABLE},
        # A session resumed past Protect: the checkpoint keeps only this.
        {"encryption_enabled": False},
    ],
    ids=["kept-plain-text", "unreadable", "resumed"],
)
def test_tracker_never_ticks_protect_when_the_keys_were_not_encrypted(record):
    items = setup_progress_items(
        _TRACK, _TRACK.index("summary"), {STEP_PROTECT: record}
    )
    protect = next(item for item in items if item.step_id == STEP_PROTECT)

    assert progress_glyph(protect.state, 4) == "–"


def test_tracker_ticks_a_resumed_protect_that_encrypted():
    items = setup_progress_items(
        _TRACK, _TRACK.index("summary"), {STEP_PROTECT: {"encryption_enabled": True}}
    )
    protect = next(item for item in items if item.step_id == STEP_PROTECT)

    assert progress_glyph(protect.state, 4) == "✓"


def test_progress_glyphs():
    assert progress_glyph("attention", 1) == "!"
    assert progress_glyph("active", 2) == "3"
    assert progress_glyph("upcoming", 9) == "10"


# --- Summary --------------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        {
            "encryption": {"enabled": True},
            "api_settings": {"openai": {"api_key": "sk-x"}},
        },
        {"api_settings": {"openai": {"api_key": "sk-x"}}},
    ],
    ids=["encrypted", "plaintext"],
)
def test_summary_key_encryption_row_names_the_card(config):
    rows = {
        row.label: row
        for row in build_summary_rows(config, {}, rag_deps_installed=False)
    }
    assert ENCRYPTION_CARD_PATH in rows["Key encryption"].detail


def test_summary_flags_the_same_plain_text_secrets_protect_offers():
    # Review round 1 (R1-F1): Summary and Protect read the same file and must
    # agree; a legacy `[API]` key read as "off" on Summary.
    from tldw_chatbook.UI.Wizards.first_run_setup_state import ROW_ATTENTION

    rows = {
        row.label: row
        for row in build_summary_rows(
            {"API": {"openai_api_key": "sk-legacy-plain"}}, {}, rag_deps_installed=False
        )
    }
    assert rows["Key encryption"].state == ROW_ATTENTION
    assert "plain text" in rows["Key encryption"].detail


def test_an_unreadable_config_never_offers_or_claims_plain_text():
    copy = protect_copy(ProtectState(PROTECT_UNREADABLE), config_path="/p")

    assert copy.primary_label == ""
    assert not copy.show_keep_plain
    assert "plain text" not in copy.body
    assert "couldn't read config.toml" in copy.body


# --- every wizard 'change it later' line names the card --------------------------


def _wizard_string_literals() -> list[tuple[str, int, str]]:
    found = []
    for path in sorted(_WIZARDS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                found.append((path.name, node.lineno, node.value))
    return found


def test_no_wizard_copy_points_at_privacy_and_security_without_the_card():
    bare = [
        f"{name}:{line}"
        for name, line, text in _wizard_string_literals()
        if "Privacy & Security" in text
        and "Privacy & Security ▸ Encryption" not in text
    ]
    assert bare == []
