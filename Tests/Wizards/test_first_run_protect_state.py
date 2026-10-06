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


def test_a_typed_key_that_is_not_on_disk_is_unsaved():
    state = read_protect_state({}, {}, typed_provider_key="anthropic")
    assert state == ProtectState(PROTECT_UNSAVED, unsaved_provider="Anthropic")


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


@pytest.mark.parametrize("kind", [PROTECT_PLAINTEXT, PROTECT_ENCRYPTED])
def test_tracker_ticks_protect_when_a_key_was_there(kind):
    items = setup_progress_items(
        _TRACK,
        _TRACK.index("summary"),
        {
            STEP_PROTECT: {
                "encryption_enabled": kind == PROTECT_ENCRYPTED,
                "protect_state": kind,
            }
        },
    )
    protect = next(item for item in items if item.step_id == STEP_PROTECT)

    assert protect.state == "complete"
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
