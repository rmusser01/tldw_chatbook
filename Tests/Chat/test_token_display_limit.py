from tldw_chatbook.Event_Handlers.Chat_Events.chat_token_events import (
    _resolve_token_display_limit,
)


def test_display_limit_is_input_window_by_default():
    # AC#3: the gauge denominator is the model input window, not the output budget.
    assert _resolve_token_display_limit(total_limit=128000, custom_limit=0) == 128000


def test_display_limit_honors_custom_override():
    assert _resolve_token_display_limit(total_limit=128000, custom_limit=5000) == 5000


def test_display_limit_ignores_nonpositive_custom():
    assert _resolve_token_display_limit(total_limit=200000, custom_limit=0) == 200000
    assert _resolve_token_display_limit(total_limit=200000, custom_limit=-3) == 200000
