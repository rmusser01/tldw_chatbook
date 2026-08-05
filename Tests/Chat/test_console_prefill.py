"""Tests for the pure /prefill command parsing + metadata helpers."""

import json

from tldw_chatbook.Chat.console_prefill import (
    ACTION_CLEAR,
    ACTION_ERROR,
    ACTION_ONE_SHOT,
    ACTION_PIN,
    ACTION_STATUS,
    PINNED_PREFILL_METADATA_KEY,
    PREFILL_MAX_CHARS,
    PrefillCommandAction,
    describe_prefill_preview,
    parse_prefill_args,
    pinned_prefill_from_conversation_metadata,
)


class TestParsePrefillArgs:
    def test_bare_args_is_status(self):
        assert parse_prefill_args("") == PrefillCommandAction(kind=ACTION_STATUS)
        assert parse_prefill_args("   ") == PrefillCommandAction(kind=ACTION_STATUS)

    def test_clear_matches_only_as_entire_args(self):
        assert parse_prefill_args("clear").kind == ACTION_CLEAR
        assert parse_prefill_args("  CLEAR  ").kind == ACTION_CLEAR
        # 'clear' with trailing text is a one-shot whose text starts with 'clear'
        result = parse_prefill_args("clear the table, then")
        assert result == PrefillCommandAction(
            kind=ACTION_ONE_SHOT, text="clear the table, then"
        )

    def test_pin_requires_trailing_text(self):
        result = parse_prefill_args("pin *She pauses*")
        assert result == PrefillCommandAction(kind=ACTION_PIN, text="*She pauses*")
        bare_pin = parse_prefill_args("pin")
        assert bare_pin.kind == ACTION_ERROR
        assert "pin" in bare_pin.error

    def test_pin_word_prefix_is_one_shot(self):
        result = parse_prefill_args("pinch of salt")
        assert result == PrefillCommandAction(kind=ACTION_ONE_SHOT, text="pinch of salt")

    def test_plain_text_is_one_shot_stripped(self):
        result = parse_prefill_args("  {\"answer\":  ")
        assert result == PrefillCommandAction(kind=ACTION_ONE_SHOT, text="{\"answer\":")

    def test_over_length_is_error(self):
        result = parse_prefill_args("x" * (PREFILL_MAX_CHARS + 1))
        assert result.kind == ACTION_ERROR
        assert str(PREFILL_MAX_CHARS) in result.error

    def test_pin_over_length_is_error(self):
        result = parse_prefill_args("pin " + "x" * (PREFILL_MAX_CHARS + 1))
        assert result.kind == ACTION_ERROR

    def test_max_length_exactly_is_accepted(self):
        result = parse_prefill_args("x" * PREFILL_MAX_CHARS)
        assert result.kind == ACTION_ONE_SHOT
        assert len(result.text) == PREFILL_MAX_CHARS


class TestDescribePrefillPreview:
    def test_short_text_verbatim(self):
        assert describe_prefill_preview("Sure thing:") == "Sure thing:"

    def test_long_text_truncated_with_ellipsis(self):
        text = "a" * 100
        preview = describe_prefill_preview(text)
        assert len(preview) == 60
        assert preview.endswith("…")

    def test_newlines_flattened_to_spaces(self):
        assert describe_prefill_preview("line one\nline two") == "line one line two"


class TestPinnedPrefillFromConversationMetadata:
    def test_reads_key_from_json(self):
        raw = json.dumps({PINNED_PREFILL_METADATA_KEY: "*She pauses*"})
        assert pinned_prefill_from_conversation_metadata(raw) == "*She pauses*"

    def test_none_metadata_returns_none(self):
        assert pinned_prefill_from_conversation_metadata(None) is None

    def test_invalid_json_returns_none(self):
        assert pinned_prefill_from_conversation_metadata("{not json") is None

    def test_non_dict_json_returns_none(self):
        assert pinned_prefill_from_conversation_metadata("[1, 2]") is None

    def test_missing_key_returns_none(self):
        assert pinned_prefill_from_conversation_metadata("{}") is None

    def test_non_string_value_returns_none(self):
        raw = json.dumps({PINNED_PREFILL_METADATA_KEY: 42})
        assert pinned_prefill_from_conversation_metadata(raw) is None

    def test_blank_string_value_returns_none(self):
        raw = json.dumps({PINNED_PREFILL_METADATA_KEY: "   "})
        assert pinned_prefill_from_conversation_metadata(raw) is None

    def test_sibling_keys_ignored(self):
        raw = json.dumps(
            {"active_dictionaries": [1, 2], PINNED_PREFILL_METADATA_KEY: "Voice:"}
        )
        assert pinned_prefill_from_conversation_metadata(raw) == "Voice:"
