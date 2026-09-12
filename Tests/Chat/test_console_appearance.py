"""Tests for Console conversation appearance metadata (task-31207).

Mirrors the ``console_speech`` metadata contract tests: versioned,
namespaced ``conversations.metadata`` JSON that fails closed on malformed
input and merges without disturbing sibling keys.
"""

from __future__ import annotations

import json

import pytest

from tldw_chatbook.Chat.console_appearance import (
    CONSOLE_APPEARANCE_COLOR_HEXES,
    CONSOLE_APPEARANCE_METADATA_KEY,
    CONSOLE_APPEARANCE_PALETTE,
    ConsoleConversationAppearance,
    is_valid_console_appearance_color,
    is_valid_console_appearance_icon,
    merge_console_conversation_appearance,
    parse_console_conversation_appearance,
)
from tldw_chatbook.Chat.console_glyphs import (
    GLYPH_APPEARANCE_ASCII_SET,
    GLYPH_APPEARANCE_ASCII_UNSET,
)


def _metadata_with(appearance: object, **siblings: object) -> str:
    payload: dict[str, object] = {CONSOLE_APPEARANCE_METADATA_KEY: appearance}
    payload.update(siblings)
    return json.dumps(payload)


class TestIconValidation:
    def test_single_code_point_emoji_is_valid(self) -> None:
        assert is_valid_console_appearance_icon("🧪")

    def test_variation_selector_emoji_is_valid(self) -> None:
        # "❤️" is U+2764 U+FE0F: one glyph plus a variation selector.
        assert is_valid_console_appearance_icon("❤️")

    def test_wide_emoji_pair_is_invalid(self) -> None:
        assert not is_valid_console_appearance_icon("🧪🧪")

    def test_ascii_pair_is_invalid(self) -> None:
        assert not is_valid_console_appearance_icon("ab")

    def test_zwj_family_is_invalid(self) -> None:
        assert not is_valid_console_appearance_icon("👨‍👩‍👧")

    def test_empty_and_non_string_are_invalid(self) -> None:
        assert not is_valid_console_appearance_icon("")
        assert not is_valid_console_appearance_icon(None)  # type: ignore[arg-type]
        assert not is_valid_console_appearance_icon(7)  # type: ignore[arg-type]


class TestColorValidation:
    def test_six_digit_hex_is_valid(self) -> None:
        assert "#f87171" in CONSOLE_APPEARANCE_COLOR_HEXES

    def test_named_colors_are_invalid(self) -> None:
        assert not is_valid_console_appearance_color("red")

    def test_short_and_long_hex_are_invalid(self) -> None:
        assert not is_valid_console_appearance_color("#f87")
        assert not is_valid_console_appearance_color("#f871711")
        assert not is_valid_console_appearance_color("f87171")

    def test_non_string_is_invalid(self) -> None:
        assert not is_valid_console_appearance_color(0xF87171)  # type: ignore[arg-type]

    def test_palette_entries_are_unique_canonical_hex(self) -> None:
        hexes = [value for _, value in CONSOLE_APPEARANCE_PALETTE]
        assert len(hexes) == len(set(hexes))
        assert set(hexes) == CONSOLE_APPEARANCE_COLOR_HEXES
        assert len(hexes) >= 12
        for label, value in CONSOLE_APPEARANCE_PALETTE:
            assert isinstance(label, str) and label
            assert is_valid_console_appearance_color(value)

    def test_ascii_glyphs_are_pure_ascii(self) -> None:
        assert GLYPH_APPEARANCE_ASCII_SET.isascii()
        assert GLYPH_APPEARANCE_ASCII_UNSET.isascii()
        assert GLYPH_APPEARANCE_ASCII_SET != GLYPH_APPEARANCE_ASCII_UNSET


class TestDataclassValidation:
    def test_defaults_are_unset(self) -> None:
        appearance = ConsoleConversationAppearance()
        assert appearance.icon is None
        assert appearance.color is None

    def test_invalid_icon_raises(self) -> None:
        with pytest.raises(ValueError):
            ConsoleConversationAppearance(icon="nope")

    def test_invalid_color_raises(self) -> None:
        with pytest.raises(ValueError):
            ConsoleConversationAppearance(color="blue")


class TestParse:
    def test_missing_metadata_fails_closed(self) -> None:
        assert parse_console_conversation_appearance(None) == (
            ConsoleConversationAppearance()
        )
        assert parse_console_conversation_appearance("") == (
            ConsoleConversationAppearance()
        )
        assert parse_console_conversation_appearance("not json") == (
            ConsoleConversationAppearance()
        )
        assert parse_console_conversation_appearance(json.dumps([])) == (
            ConsoleConversationAppearance()
        )

    def test_missing_key_fails_closed(self) -> None:
        metadata = json.dumps({"console_speech": {"paused": False}})
        assert parse_console_conversation_appearance(metadata) == (
            ConsoleConversationAppearance()
        )

    def test_valid_payload_round_trips(self) -> None:
        metadata = _metadata_with({"icon": "🧪", "color": "#f87171"})
        assert parse_console_conversation_appearance(metadata) == (
            ConsoleConversationAppearance(icon="🧪", color="#f87171")
        )

    def test_unset_members_round_trip_as_none(self) -> None:
        metadata = _metadata_with({"icon": None, "color": None})
        assert parse_console_conversation_appearance(metadata) == (
            ConsoleConversationAppearance()
        )

    def test_icon_without_color_is_kept(self) -> None:
        metadata = _metadata_with({"icon": "🧪", "color": None})
        assert parse_console_conversation_appearance(metadata).icon == "🧪"

    def test_unknown_keys_drop_the_whole_entry(self) -> None:
        # Strict key set, like ``console_speech``: a payload from a newer
        # shape must not be partially interpreted by this version.
        metadata = _metadata_with({"icon": "🧪", "color": None, "scale": 2})
        assert parse_console_conversation_appearance(metadata) == (
            ConsoleConversationAppearance()
        )

    def test_malformed_icon_is_sanitized_per_member(self) -> None:
        # Per-member sanitization: a bad icon drops only the icon.
        metadata = _metadata_with({"icon": "🧪🧪", "color": "#f87171"})
        assert parse_console_conversation_appearance(metadata) == (
            ConsoleConversationAppearance(color="#f87171")
        )

    def test_malformed_color_is_sanitized_per_member(self) -> None:
        metadata = _metadata_with({"icon": "🧪", "color": "cerulean"})
        assert parse_console_conversation_appearance(metadata) == (
            ConsoleConversationAppearance(icon="🧪")
        )

    def test_non_object_entry_fails_closed(self) -> None:
        metadata = _metadata_with(["🧪"])
        assert parse_console_conversation_appearance(metadata) == (
            ConsoleConversationAppearance()
        )

    def test_non_string_members_are_sanitized_per_member(self) -> None:
        metadata = _metadata_with({"icon": 3, "color": "#f87171"})
        assert parse_console_conversation_appearance(metadata) == (
            ConsoleConversationAppearance(color="#f87171")
        )


class TestMerge:
    def test_merge_replaces_only_the_appearance_key(self) -> None:
        metadata = _metadata_with(
            {"icon": "🧪", "color": None}, console_speech={"paused": True}
        )
        merged = merge_console_conversation_appearance(
            metadata, ConsoleConversationAppearance(icon="🎨", color="#22d3ee")
        )
        assert merged["console_speech"] == {"paused": True}
        assert merged[CONSOLE_APPEARANCE_METADATA_KEY] == {
            "icon": "🎨",
            "color": "#22d3ee",
        }

    def test_merge_clears_both_members(self) -> None:
        metadata = _metadata_with({"icon": "🧪", "color": "#f87171"})
        merged = merge_console_conversation_appearance(
            metadata, ConsoleConversationAppearance()
        )
        assert merged[CONSOLE_APPEARANCE_METADATA_KEY] == {
            "icon": None,
            "color": None,
        }

    def test_merge_accepts_mapping_metadata(self) -> None:
        merged = merge_console_conversation_appearance(
            {"console_speech": {"paused": True}},
            ConsoleConversationAppearance(icon="🧪"),
        )
        assert merged["console_speech"] == {"paused": True}
        assert merged[CONSOLE_APPEARANCE_METADATA_KEY]["icon"] == "🧪"

    def test_merge_rejects_non_appearance_objects(self) -> None:
        with pytest.raises(TypeError):
            merge_console_conversation_appearance("{}", {"icon": "🧪"})  # type: ignore[arg-type]

    def test_merge_rejects_invalid_values(self) -> None:
        with pytest.raises(ValueError):
            merge_console_conversation_appearance(
                "{}", ConsoleConversationAppearance(icon="bad", color=None)
            )

    def test_merge_result_is_json_serializable_and_stable(self) -> None:
        merged = merge_console_conversation_appearance(
            "{}", ConsoleConversationAppearance(icon="🧪", color="#f87171")
        )
        encoded = json.dumps(merged, sort_keys=True)
        assert parse_console_conversation_appearance(encoded) == (
            ConsoleConversationAppearance(icon="🧪", color="#f87171")
        )


class TestHexInputNormalization:
    def test_optional_hash_and_case(self) -> None:
        from tldw_chatbook.Chat.console_appearance import (
            normalize_console_appearance_hex_input,
        )

        assert normalize_console_appearance_hex_input("c0ffee") == "#c0ffee"
        assert normalize_console_appearance_hex_input("#C0FFEE") == "#c0ffee"
        assert normalize_console_appearance_hex_input(" #f87171 ") == "#f87171"

    def test_partial_and_junk_return_none(self) -> None:
        from tldw_chatbook.Chat.console_appearance import (
            normalize_console_appearance_hex_input,
        )

        assert normalize_console_appearance_hex_input("") is None
        assert normalize_console_appearance_hex_input("#f87") is None
        assert normalize_console_appearance_hex_input("#f871711") is None
        assert normalize_console_appearance_hex_input("red") is None
        assert normalize_console_appearance_hex_input("##f87171") is None
        assert normalize_console_appearance_hex_input(31) is None  # type: ignore[arg-type]

    def test_palette_extended_beyond_the_initial_set(self) -> None:
        # task-31209: the shipped palette grew from 13 to 21 swatches.
        assert len(CONSOLE_APPEARANCE_PALETTE) >= 21
