"""Settings ▸ Providers & Models ▸ Connect: the API key row's fixed sentences
fit the row (review of PR #3057, R1 #11).

The row is one line, cut with an ellipsis. The review measured its help
column at 39 cells at 211x44, and four sentences written for the old
multi-line status block were cut there ("Claude subscription credential
expired; log in with Claude Code" is 63). Readiness is a stand-in, so no
case reads a real credential.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from rich.cells import cell_len

from tldw_chatbook.UI.Screens.settings_screen import ENDPOINT_NOT_FOUND_SETTINGS_COPY
from tldw_chatbook.UI.Settings_Modules import providers_models_card as card

# Importing the card imports the Settings screen, whose config-participant
# admission the per-test sandbox refuses; keep the collection-time profile.
pytestmark = pytest.mark.bootstrap_profile

#: The API key row's help column at 211x44, as the review measured it.
_HELP_CELLS = 39


def _row(monkeypatch, *, registry=None, subscription=None, reason="") -> tuple:
    readiness = SimpleNamespace(
        subscription_status=subscription,
        reason=reason,
        requires_api_key=True,
        env_var="OPENAI_API_KEY",
    )
    monkeypatch.setattr(card, "get_provider_readiness", lambda *a, **k: readiness)
    screen = SimpleNamespace(
        _provider_auth_readiness_config=lambda provider: {},
        _provider_registry_credential_text=lambda provider, readiness: registry,
    )
    return card.api_key_row_copy(screen, "openai")


@pytest.mark.parametrize("status", ["pending", "ready", "expired", "missing"])
def test_a_subscription_state_fits_the_row(monkeypatch, status):
    word, help_line = _row(monkeypatch, subscription=status)

    assert word == "subscription"
    assert cell_len(help_line) <= _HELP_CELLS, help_line
    if status in ("expired", "missing"):
        assert help_line == f"{status} · log in with Claude Code"


def test_invalid_provider_settings_fit_the_row(monkeypatch):
    word, help_line = _row(monkeypatch, reason="Invalid provider settings")

    assert (word, help_line) == ("invalid", "repair it in Advanced Config")
    assert cell_len(help_line) <= _HELP_CELLS


@pytest.mark.parametrize(
    ("screen_text", "help_line"),
    [
        (ENDPOINT_NOT_FOUND_SETTINGS_COPY, "not found · choose another provider"),
        ("API key source: saved in this endpoint", "saved in this endpoint"),
        (
            "API key source: none required by this endpoint",
            "none required by this endpoint",
        ),
        # The Source word already says "this endpoint".
        ("API key source: env var GPU_KEY (this endpoint)", "env var GPU_KEY"),
    ],
)
def test_a_custom_endpoint_s_key_line_fits_the_row(monkeypatch, screen_text, help_line):
    assert _row(monkeypatch, registry=screen_text) == ("this endpoint", help_line)
    assert cell_len(help_line) <= _HELP_CELLS


def test_the_rows_other_fixed_sentences_fit_too():
    for help_line in (
        card.API_KEY_SAVED_HELP,
        card.API_KEY_CLEARED_HELP,
        f"{card.API_KEY_CLEARED_HELP} · Env var used",
        card.ENV_VAR_HELP_COPY,
        card.AUTH_SOURCE_API_KEY_HELP,
        card.AUTH_SOURCE_SUBSCRIPTION_HELP,
        card.MODEL_PICKER_HELP_COPY,
    ):
        assert cell_len(help_line) <= _HELP_CELLS, help_line
