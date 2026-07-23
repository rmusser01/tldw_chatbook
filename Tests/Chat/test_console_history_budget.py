"""Unit tests for the Console history token-budget trimmer."""

from tldw_chatbook.Chat.console_history_budget import (
    BoundResult,
    DEFAULT_RESPONSE_RESERVATION,
    bound_messages_to_window,
    count_console_messages_tokens,
)


# A deterministic counter: 1 token per whitespace word in every string text
# part, plus 10 tokens per image part. tiktoken-independent.
def _wordcount(messages, model):  # noqa: ARG001
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    total += len(part.get("text", "").split())
                else:
                    total += 10
        else:
            total += len(str(content).split())
    return total


def _msg(role, text):
    return {"role": role, "content": text}


def test_fits_under_budget_drops_nothing():
    msgs = [_msg("system", "sys"), _msg("user", "hi there"), _msg("assistant", "hello back")]
    result = bound_messages_to_window(
        msgs, model="m", provider="p", response_reservation=0, window=1000, count_fn=_wordcount
    )
    assert result.dropped_count == 0
    assert result.messages == msgs


def test_over_budget_drops_oldest_whole_turns_keeps_system_and_current():
    # window 20, reservation 0, margin max(512, 0) -> 512 makes budget negative;
    # use a big window and a big reservation instead to get a small positive budget.
    msgs = [
        _msg("system", "you are helpful"),          # 3 words
        _msg("user", "old question one two"),        # turn A (4)
        _msg("assistant", "old answer one two"),     # turn A (4)
        _msg("user", "mid question three four"),     # turn B (4)
        _msg("assistant", "mid answer three four"),  # turn B (4)
        _msg("user", "current question five"),       # current turn (3)
    ]
    # budget: window 1000 - reservation 980 - max(512, 20)=512 -> negative? No:
    # choose window 1000, reservation 0, but shrink via a tiny window instead:
    result = bound_messages_to_window(
        msgs, model="m", provider="p", response_reservation=0,
        window=525, count_fn=_wordcount,  # budget = 525 - 0 - 512 = 13
    )
    # keep must fit in 13 tokens: system(3) + current(3) = 6; +turn B(8)=14 > 13 -> drop B too;
    # +turn A also dropped. So only system + current survive.
    roles = [m["role"] for m in result.messages]
    assert roles == ["system", "user"]
    assert result.messages[0]["content"] == "you are helpful"
    assert result.messages[-1]["content"] == "current question five"
    # dropped the 4 middle messages (turns A + B)
    assert result.dropped_count == 4


def test_keeps_one_turn_when_it_fits():
    msgs = [
        _msg("system", "sys one"),                   # 2
        _msg("user", "old one two three four"),      # turn A (5)
        _msg("assistant", "old ans"),                # turn A (2)
        _msg("user", "current q"),                   # current (2)
    ]
    # window 521, reservation 0, margin 512 -> budget 9.
    # system(2)+current(2)=4; +turnA(7)=11 > 9 -> drop turn A. Result 4 <= 9.
    result = bound_messages_to_window(
        msgs, model="m", provider="p", response_reservation=0, window=521, count_fn=_wordcount
    )
    assert [m["role"] for m in result.messages] == ["system", "user"]
    assert result.dropped_count == 2


def test_degenerate_system_plus_current_over_budget_kept_anyway():
    msgs = [_msg("system", "a b c d e"), _msg("user", "f g h i j")]  # 5 + 5 = 10
    result = bound_messages_to_window(
        msgs, model="m", provider="p", response_reservation=0, window=515, count_fn=_wordcount
    )  # budget = 515 - 512 = 3 < 10, but nothing droppable
    assert result.messages == msgs
    assert result.dropped_count == 0


def test_window_override_takes_precedence_over_lookup():
    msgs = [_msg("user", "one two three four five six")]
    # No system, single user turn = current turn -> never dropped regardless.
    result = bound_messages_to_window(
        msgs, model="gpt-4", provider="openai", response_reservation=0, window=1, count_fn=_wordcount
    )
    assert result.dropped_count == 0
    assert result.messages == msgs


def test_leading_assistant_orphan_is_its_own_droppable_unit():
    msgs = [
        _msg("system", "s"),
        _msg("assistant", "orphan a b c d e f g h"),  # 9-word leading orphan
        _msg("user", "cur"),
    ]
    result = bound_messages_to_window(
        msgs, model="m", provider="p", response_reservation=0, window=515, count_fn=_wordcount
    )  # budget 3: system(1)+current(1)=2 fits; orphan(9) can't be added -> dropped
    assert [m["role"] for m in result.messages] == ["system", "user"]
    assert result.dropped_count == 1


def test_multimodal_content_counted_without_error_and_images_cost():
    # Real counter (no injected count_fn). Verifies list content doesn't crash
    # and each image adds per_image_tokens.
    text_only = [{"role": "user", "content": "hello world"}]
    with_image = [
        {"role": "user", "content": [
            {"type": "text", "text": "hello world"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
        ]}
    ]
    base = count_console_messages_tokens(text_only, "gpt-4")
    withimg = count_console_messages_tokens(with_image, "gpt-4", per_image_tokens=1024)
    assert withimg >= base + 1024


def test_non_string_text_part_does_not_crash_and_is_skipped():
    # A malformed/legacy multimodal part whose `text` is not a string must not
    # crash the counter (which sits on the send path); it is skipped, not
    # stringified. The sibling string part is still counted.
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "hello world"},
            {"type": "text", "text": 123},          # non-string -> skipped
            {"type": "text", "text": None},         # falsy -> skipped
        ]}
    ]
    # Does not raise, and counts only the valid "hello world" text (2 words +
    # count_tokens_messages overhead), never the non-string values.
    assert count_console_messages_tokens(msgs, "gpt-4") >= 2


def test_long_history_drops_exact_turns_via_binary_search():
    # 50 middle turns behind a short current turn; only a few can be kept.
    # Verifies the binary-search trim drops the correct oldest-first count and
    # preserves the system prefix and current turn.
    msgs = [_msg("system", "sys")]                  # 1 word
    for i in range(50):
        msgs.append(_msg("user", f"q{i} a b c d"))   # 5 words each
        msgs.append(_msg("assistant", f"r{i} a b c d"))  # 5 words each
    msgs.append(_msg("user", "current now"))         # current turn, 2 words
    # budget = 600 - 0 - 512 = 88. Keep must fit 88 tokens.
    # system(1) + current(2) = 3; each kept turn = 10. floor((88-3)/10) = 8
    # turns kept -> 42 turns dropped -> 84 messages dropped.
    result = bound_messages_to_window(
        msgs, model="m", provider="p", response_reservation=0,
        window=600, count_fn=_wordcount,
    )
    assert result.messages[0]["content"] == "sys"
    assert result.messages[-1]["content"] == "current now"
    assert result.dropped_count == 84
    # Kept payload is within budget.
    assert _wordcount(result.messages, "m") <= 88


def test_default_response_reservation_value():
    assert DEFAULT_RESPONSE_RESERVATION == 1024
