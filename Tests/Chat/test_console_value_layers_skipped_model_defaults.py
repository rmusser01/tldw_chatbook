"""Source words when the model default is skipped AND a chat's values are given.

#3057 review R3 #15: ``resolve_console_value_layers`` skipped
``excluded_model_profile_fields`` in the layers it names, yet compared the
chat against defaults built WITH them, so the two halves of one answer read
different default chains.
"""

from __future__ import annotations

from Tests.Chat.test_console_value_sources import MODEL, PROVIDER, _config, _values
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleValueLayer,
    resolve_console_value_layers,
)

_NAMES = ("temperature", "max_tokens", "streaming")


def test_a_chat_holding_the_inherited_value_names_the_layer_it_inherits_from() -> None:
    """The model default says 0.3; skipped, the chain gives Console Behavior's
    0.6, which is what the chat holds: that is not the chat's own value."""
    config = _config(
        model_default=_values(0.3, 1024, False),
        chat_defaults=_values(0.6, 512, True),
    )
    chat = ConsoleSessionSettings(
        provider=PROVIDER, model=MODEL, temperature=0.6, max_tokens=512, streaming=True
    )

    layers = resolve_console_value_layers(
        config,
        PROVIDER,
        MODEL,
        _NAMES,
        chat_settings=chat,
        excluded_model_profile_fields=frozenset(_NAMES),
    )

    assert layers == dict.fromkeys(_NAMES, ConsoleValueLayer.CHAT_DEFAULTS)


def test_a_chat_value_off_the_skipped_chain_is_still_this_chat() -> None:
    config = _config(
        model_default=_values(0.3, 1024, False),
        chat_defaults=_values(0.6, 512, True),
    )
    chat = ConsoleSessionSettings(
        provider=PROVIDER, model=MODEL, temperature=0.3, max_tokens=512, streaming=True
    )

    layers = resolve_console_value_layers(
        config,
        PROVIDER,
        MODEL,
        _NAMES,
        chat_settings=chat,
        excluded_model_profile_fields=frozenset({"temperature"}),
    )

    # 0.3 equals the skipped model default, not the chain the caller asked for.
    assert layers["temperature"] is ConsoleValueLayer.THIS_CHAT
    # max_tokens was not skipped: the chat's 512 differs from the model's 1024.
    assert layers["max_tokens"] is ConsoleValueLayer.THIS_CHAT
