"""Settings ▸ Providers & Models ▸ Connect: the API key row's fixed sentences
fit the row (review of PR #3057, R1 #11).

The row is one line, cut with an ellipsis. The review measured its help
column at 39 cells at 211x44, and four sentences written for the old
multi-line status block were cut there ("Claude subscription credential
expired; log in with Claude Code" is 63). Readiness is a stand-in, so no
case reads a real credential.

Fix round 1 (findings 2, 9 and 26): a custom endpoint's line is the screen's
own sentence, printed as it is returned, and an endpoint that is not ready
has a short one too ("Not ready · Set GPU_KEY or update the stored api_key
for the 'GPU box' endpoint." is 72 cells; the invalid-name recovery is 200).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from rich.cells import cell_len

from tldw_chatbook.Chat.custom_endpoint_registry import entry_for
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
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
        _provider_draft=lambda: None,
        _validate_credential_env_var=SettingsScreen._validate_credential_env_var,
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


_STORED = {"api_key": "endpoint-key-abcdefgh1234"}
_ENV = {"api_key_env": "GPU_KEY"}
_NOT_READY = {
    "ready": False,
    "reason": "Missing API key",
    "recovery": "Set GPU_KEY or update the stored api_key for the 'GPU box' endpoint.",
}


def _endpoint(fields):
    """The registry's entry for ``fields``; None stands for a deleted one."""
    if fields is None:
        return None
    box = {
        "display_name": "GPU box",
        "family": "openai_compatible",
        "base_url": "http://192.168.1.6:9000/v1",
        "models": ["model-a"],
        **fields,
    }
    return entry_for({"custom_endpoints": {"box": box}}, "custom-ep:box")


@pytest.mark.parametrize(
    ("entry_fields", "readiness", "help_line"),
    [
        (None, _NOT_READY, "not found · choose another provider"),
        (_STORED, {}, "saved in this endpoint"),
        ({}, {}, "none required by this endpoint"),
        # The Source word already says "this endpoint".
        (_ENV, {}, "env var GPU_KEY"),
        # Not ready: the variable is unset, the endpoint's family is not
        # ready, or the name is no name. The Test result's Key row says why.
        (_ENV, _NOT_READY, "not ready · Test (t) says why"),
        ({}, _NOT_READY, "not ready · Test (t) says why"),
        (
            {"api_key_env": "gpu-key"},
            _NOT_READY,
            "invalid Env var · edit this endpoint",
        ),
    ],
)
def test_a_custom_endpoint_s_key_line_fits_the_row(
    monkeypatch, entry_fields, readiness, help_line
):
    """The screen's own sentence for the endpoint, as the card prints it."""
    monkeypatch.setenv("GPU_KEY", "endpoint-key-abcdefgh1234")
    entry = _endpoint(entry_fields)
    readiness = SimpleNamespace(
        **{"ready": True, "reason": "", "recovery": None, **readiness}
    )
    screen = SimpleNamespace(
        _provider_is_registry_id=lambda provider: True,
        _provider_registry_entry=lambda provider: entry,
        _validate_credential_env_var=SettingsScreen._validate_credential_env_var,
        _provider_auth_readiness_config=lambda provider: {},
    )
    screen._provider_registry_credential_text = lambda provider, readiness: (
        SettingsScreen._provider_registry_credential_text(screen, provider, readiness)
    )
    monkeypatch.setattr(card, "get_provider_readiness", lambda *a, **k: readiness)

    assert card.api_key_row_copy(screen, "custom-ep:box") == (
        "this endpoint",
        help_line,
    )
    assert cell_len(help_line) <= _HELP_CELLS


def test_the_env_var_help_says_the_field_holds_a_name():
    """C4: the field is not masked, so its help says a key never goes there;
    ADR-012 still calls the env var the safer path."""
    assert card.ENV_VAR_HELP_COPY == "safer: a variable's name, never the key"


def test_the_rows_other_fixed_sentences_fit_too():
    for help_line in (
        card.API_KEY_SAVED_HELP,
        card.API_KEY_CLEARED_HELP,
        f"{card.API_KEY_CLEARED_HELP} · Env var used",
        card.ENV_VAR_HELP_COPY,
        card.ENV_VAR_OPTIONAL_HELP,
        card.AUTH_SOURCE_API_KEY_HELP,
        card.AUTH_SOURCE_SUBSCRIPTION_HELP,
        card.MODEL_PICKER_HELP_COPY,
    ):
        assert cell_len(help_line) <= _HELP_CELLS, help_line
