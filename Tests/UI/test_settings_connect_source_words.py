"""Settings ▸ Providers & Models ▸ Connect: one word for one state (review of
PR #3057, R1 #9, #18, #21 and #24; controller ruling C3).

The Source words are read straight from the card's functions with a stand-in
for the screen, so each case names exactly the screen answers it depends on.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_session_settings import (
    CONSOLE_VALUE_SOURCE_WORDS,
    ConsoleValueLayer,
)
from tldw_chatbook.UI.Settings_Modules import providers_models_card as card

# Importing the card imports the Settings screen, whose config-participant
# admission the per-test sandbox refuses; keep the collection-time profile.
pytestmark = pytest.mark.bootstrap_profile

_EDITED = CONSOLE_VALUE_SOURCE_WORDS[ConsoleValueLayer.EDITED_DRAFT]
_BUILT_IN = CONSOLE_VALUE_SOURCE_WORDS[ConsoleValueLayer.BUILT_IN]


def _draft(*dirty: str):
    return SimpleNamespace(dirty_keys=frozenset(dirty), values={})


def _screen(
    *,
    dirty=(),
    saved_auth_source=False,
    endpoint_registry=None,
    auth_source="api_key",
):
    return SimpleNamespace(
        _provider_draft=lambda: _draft(*dirty) if dirty else None,
        _provider_auth_source_draft_key=lambda provider: "provider_auth_source:x",
        _provider_config=lambda provider: (
            {"auth_source": "api_key"} if saved_auth_source else {}
        ),
        _provider_auth_source_value=lambda provider: auth_source,
        _provider_registry_endpoint=lambda provider: endpoint_registry,
        _provider_is_registry_id=lambda provider: False,
    )


@pytest.mark.parametrize(
    ("source", "value", "word"),
    [
        ("settings_draft", "openai", _EDITED),
        ("settings_draft", "", _EDITED),  # an emptied model is still an edit
        ("chat_defaults", "openai", "new-chat default"),
        # R1 #18: nothing saved is not a "new-chat default".
        ("chat_defaults", "", "not set"),
        ("chat_defaults", None, "not set"),
    ],
)
def test_provider_and_model_rows_say_where_the_shown_value_comes_from(
    source, value, word
):
    assert card.selection_source_word(source, value) == word


def test_the_selection_words_are_the_two_sources_settings_hands_the_rows():
    """R1 #18: "this chat" and "built-in" could never be shown here."""
    assert set(card.SELECTION_SOURCE_WORDS) == {"settings_draft", "chat_defaults"}
    assert card.SELECTION_SOURCE_WORDS["settings_draft"] == _EDITED


def test_a_value_read_from_config_reads_saved_in_config_on_every_row():
    """C3: Endpoint and Sign in with said "config"; API key said "saved in
    config". The phrase fits the 16-cell Source column at 211x44."""
    assert card.endpoint_row_copy(_screen(), "openai", "https://x.test/v1")[0] == (
        card.SAVED_IN_CONFIG
    )
    assert card.auth_source_row_copy(_screen(saved_auth_source=True), "anthropic")[
        0
    ] == (card.SAVED_IN_CONFIG)
    assert card.SAVED_IN_CONFIG == "saved in config"
    assert len(card.SAVED_IN_CONFIG) <= 16


def test_the_other_endpoint_and_sign_in_words_are_the_shared_ones():
    """R1 #21: "edited *" and "built-in" are the rows' own words, not copies."""
    assert card.endpoint_row_copy(_screen(dirty=("endpoint",)), "openai", "x")[0] == (
        _EDITED
    )
    assert card.endpoint_row_copy(_screen(), "openai", "")[0] == _BUILT_IN
    assert card.auth_source_row_copy(_screen(), "anthropic")[0] == _BUILT_IN
    assert card.auth_source_row_copy(
        _screen(dirty=("provider_auth_source:x",)), "anthropic"
    )[0] == (_EDITED)


@pytest.mark.parametrize(
    ("value", "word"),
    [
        ("sk-proj-abcdefghijklmnop1234", "set in shell"),
        ("  sk-proj-abcdefghijklmnop1234  ", "set in shell"),
        # R1 #9: readiness refuses these, so the row must not say "set".
        ("<API_KEY_HERE>", "not set"),
        ("   ", "not set"),
        (None, "not set"),
    ],
)
def test_env_var_row_says_set_only_for_a_value_readiness_accepts(
    monkeypatch, value, word
):
    if value is None:
        monkeypatch.delenv("A1_ENV_VAR", raising=False)
    else:
        monkeypatch.setenv("A1_ENV_VAR", value)

    assert card.env_var_source_word(_screen(), "openai", "A1_ENV_VAR") == word


def test_env_var_row_reads_not_used_under_a_claude_subscription(monkeypatch):
    """R1 #9: the row is disabled then, and the variable is never read."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abcdefghijklmnop1234")
    subscription = _screen(auth_source="claude_subscription")

    assert card.env_var_source_word(subscription, "anthropic", "") == "not used"
    assert card.env_var_source_word(_screen(), "anthropic", "") == "set in shell"
    # An unsaved edit is still an unsaved edit.
    edited = _screen(auth_source="claude_subscription", dirty=("credential_env_var",))
    assert card.env_var_source_word(edited, "anthropic", "X") == _EDITED
