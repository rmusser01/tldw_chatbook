"""Direct characterization tests for the shared console selection core.

TASK-32859 review follow-up: the resolver became the single implementation
of the selection algorithm; these pin its contract directly, independent of
the (partially pre-existing-red) console suites.
"""

from types import SimpleNamespace

from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleSelectionCore,
    resolve_console_selection_core,
)


def _settings(**overrides):
    values = {"provider": "openai", "model": None, "base_url": None}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_core_shape_is_frozen() -> None:
    core = resolve_console_selection_core(_settings(), app_config={})
    assert isinstance(core, ConsoleSelectionCore)
    assert core.provider == "openai"
    assert core.explicit_model is None
    assert core.configured_model is None
    assert core.base_url is None


def test_provider_identity_falls_back_to_llama_cpp() -> None:
    core = resolve_console_selection_core(_settings(provider=None), app_config={})
    assert core.provider == "llama_cpp"


def test_configured_model_reads_model_api_model_default_chain() -> None:
    config = {"api_settings": {"openai": {"api_model": "gpt-x"}}}
    core = resolve_console_selection_core(_settings(), app_config=config)
    assert core.configured_model == "gpt-x"


def test_explicit_equals_configured_dedup_clears_explicit() -> None:
    config = {"api_settings": {"openai": {"model": "same-model"}}}
    core = resolve_console_selection_core(_settings(model="same-model"), app_config=config)
    assert core.explicit_model is None
    assert core.configured_model == "same-model"


def test_explicit_model_survives_when_it_differs() -> None:
    config = {"api_settings": {"openai": {"model": "configured"}}}
    core = resolve_console_selection_core(_settings(model="explicit"), app_config=config)
    assert core.explicit_model == "explicit"
    assert core.configured_model == "configured"


def test_legacy_model_blocks_the_explicit_dedup() -> None:
    config = {"api_settings": {"openai": {"model": "same-model"}}}
    core = resolve_console_selection_core(
        _settings(model="same-model"), app_config=config, legacy_model="legacy"
    )
    assert core.explicit_model == "same-model"


def test_blankish_models_are_treated_as_unset() -> None:
    config = {"api_settings": {"openai": {"model": "none"}}}
    core = resolve_console_selection_core(_settings(model=" null "), app_config=config)
    assert core.explicit_model is None
    assert core.configured_model is None
