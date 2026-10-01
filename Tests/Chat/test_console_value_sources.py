"""Source words: which layer of the Console parameter stack a value came from.

TASK-33004.5 AC#3. Spec §6 prints one of six words next to every value; the
ADR-147 stack has eight layers. ``resolve_console_value_layers`` names the
layer, and ``CONSOLE_VALUE_SOURCE_WORDS`` maps each layer to exactly one word.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.console_session_settings import (
    CONSOLE_VALUE_SOURCE_WORDS,
    ConsoleSessionSettings,
    ConsoleValueLayer,
    build_default_console_session_settings,
    resolve_console_value_layers,
)

SPEC_WORDS = {
    "edited *",
    "this chat",
    "model default",
    "Console Behavior",
    "provider",
    "built-in",
}
PROVIDER, MODEL = "vllm", "vendor/model:b"


def _config(**layers: dict) -> dict:
    """A config where each named layer sets its own value for every field."""
    vllm: dict = {"api_url": "http://127.0.0.1:9098", "model": MODEL}
    vllm.update(layers.get("provider_scalars", {}))
    if "model_default" in layers:
        vllm["model_defaults"] = {MODEL: layers["model_default"]}
    config: dict = {
        "chat_defaults": {"provider": "llama_cpp", "model": "model-a"},
        "api_settings": {"vllm": vllm},
    }
    config["chat_defaults"].update(layers.get("chat_defaults", {}))
    if "console_provider_default" in layers:
        config["console"] = {
            "provider_defaults": {PROVIDER: layers["console_provider_default"]}
        }
    return config


def _layers(config: dict, **kwargs) -> dict[str, ConsoleValueLayer]:
    return resolve_console_value_layers(
        config, PROVIDER, MODEL, ("temperature", "max_tokens", "streaming"), **kwargs
    )


def _values(temperature: float, max_tokens: int, streaming: bool) -> dict:
    return {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "streaming": streaming,
    }


def test_every_layer_maps_to_exactly_one_spec_word() -> None:
    assert set(CONSOLE_VALUE_SOURCE_WORDS) == set(ConsoleValueLayer)
    assert set(CONSOLE_VALUE_SOURCE_WORDS.values()) == SPEC_WORDS


def test_an_edited_draft_value_says_edited() -> None:
    layers = _layers(_config(), edited=frozenset({"temperature", "streaming"}))
    assert layers["temperature"] is ConsoleValueLayer.EDITED_DRAFT
    assert layers["streaming"] is ConsoleValueLayer.EDITED_DRAFT
    assert CONSOLE_VALUE_SOURCE_WORDS[layers["temperature"]] == "edited *"
    assert layers["max_tokens"] is not ConsoleValueLayer.EDITED_DRAFT


def test_a_chat_value_that_differs_from_the_defaults_says_this_chat() -> None:
    config = _config(model_default=_values(0.3, 1024, False))
    chat = ConsoleSessionSettings(
        provider=PROVIDER,
        model=MODEL,
        temperature=1.3,
        max_tokens=1024,
        streaming=False,
    )
    layers = _layers(config, chat_settings=chat)
    assert layers["temperature"] is ConsoleValueLayer.THIS_CHAT
    assert CONSOLE_VALUE_SOURCE_WORDS[layers["temperature"]] == "this chat"
    # A chat value that equals the default inherits that default's word.
    assert layers["max_tokens"] is ConsoleValueLayer.MODEL_DEFAULT
    assert layers["streaming"] is ConsoleValueLayer.MODEL_DEFAULT


@pytest.mark.parametrize(
    ("layer", "expected", "word"),
    [
        ("model_default", ConsoleValueLayer.MODEL_DEFAULT, "model default"),
        (
            "console_provider_default",
            ConsoleValueLayer.CONSOLE_PROVIDER_DEFAULT,
            "provider",
        ),
        ("chat_defaults", ConsoleValueLayer.CHAT_DEFAULTS, "Console Behavior"),
        ("provider_scalars", ConsoleValueLayer.PROVIDER_SCALARS, "provider"),
    ],
)
def test_each_config_layer_names_itself(layer, expected, word) -> None:
    config = _config(**{layer: _values(0.4, 2048, False)})
    layers = _layers(config)
    for name in ("temperature", "max_tokens", "streaming"):
        assert layers[name] is expected, name
        assert CONSOLE_VALUE_SOURCE_WORDS[layers[name]] == word


def test_custom_endpoint_params_name_their_layer() -> None:
    """ADR-147: registry entry params ride the ``extra_sources`` seam."""
    layers = _layers(
        _config(chat_defaults=_values(0.6, 512, True)),
        extra_sources=({"temperature": 0.5, "max_tokens": 4096},),
    )
    assert layers["temperature"] is ConsoleValueLayer.CUSTOM_ENDPOINT_PARAMS
    assert layers["max_tokens"] is ConsoleValueLayer.CUSTOM_ENDPOINT_PARAMS
    assert CONSOLE_VALUE_SOURCE_WORDS[layers["temperature"]] == "provider"
    # Streaming is transport, never an endpoint param: chat_defaults supplies it.
    assert layers["streaming"] is ConsoleValueLayer.CHAT_DEFAULTS


def test_a_value_no_layer_sets_is_built_in() -> None:
    layers = _layers(_config())
    assert set(layers.values()) == {ConsoleValueLayer.BUILT_IN}
    assert CONSOLE_VALUE_SOURCE_WORDS[ConsoleValueLayer.BUILT_IN] == "built-in"


def test_the_highest_layer_that_sets_a_value_wins_in_build_order() -> None:
    """The walk is the builder's own: the named layer supplies the built value."""
    config = _config(
        model_default={"temperature": 0.3},
        console_provider_default={"temperature": 0.4, "max_tokens": 777},
        chat_defaults=_values(0.6, 512, False),
        provider_scalars=_values(0.8, 9999, True),
    )
    layers = _layers(config)
    assert layers == {
        "temperature": ConsoleValueLayer.MODEL_DEFAULT,
        "max_tokens": ConsoleValueLayer.CONSOLE_PROVIDER_DEFAULT,
        "streaming": ConsoleValueLayer.CHAT_DEFAULTS,
    }
    built = build_default_console_session_settings(config, PROVIDER, MODEL)
    assert (built.temperature, built.max_tokens, built.streaming) == (0.3, 777, False)


def test_blank_and_unparsable_values_fall_through_like_the_builder() -> None:
    config = _config(
        model_default={"temperature": "", "max_tokens": "lots", "streaming": "maybe"},
        provider_scalars=_values(0.8, 9999, True),
    )
    layers = _layers(config)
    assert set(layers.values()) == {ConsoleValueLayer.PROVIDER_SCALARS}
