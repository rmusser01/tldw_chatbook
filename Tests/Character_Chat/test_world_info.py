"""
Tests for World Info/Lorebook functionality.
"""

import copy
import re

import pytest
from typing import Dict, Any

from tldw_chatbook.Character_Chat.world_info_processor import WorldInfoProcessor


class TestWorldInfoProcessor:
    """Test cases for the WorldInfoProcessor class."""

    @pytest.fixture
    def sample_character_data(self) -> Dict[str, Any]:
        """Create sample character data with world info."""
        return {
            "name": "Test Character",
            "extensions": {
                "character_book": {
                    "scan_depth": 3,
                    "token_budget": 500,
                    "recursive_scanning": False,
                    "entries": [
                        {
                            "keys": ["Eldoria", "eldoria"],
                            "content": "Eldoria is a magical kingdom in the north.",
                            "enabled": True,
                            "position": "before_char",
                            "insertion_order": 1,
                            "case_sensitive": False,
                            "selective": False,
                        },
                        {
                            "keys": ["dragon", "Dragon", "dragons", "Dragons"],
                            "content": "Dragons are ancient creatures of immense power.",
                            "enabled": True,
                            "position": "after_char",
                            "insertion_order": 2,
                            "case_sensitive": False,
                            "selective": False,
                        },
                        {
                            "keys": ["magic sword"],
                            "content": "The magic sword can only be wielded by the chosen one.",
                            "enabled": True,
                            "position": "at_start",
                            "insertion_order": 3,
                            "case_sensitive": False,
                            "selective": True,
                            "secondary_keys": ["chosen one", "prophecy"],
                        },
                        {
                            "keys": ["disabled entry"],
                            "content": "This should not appear.",
                            "enabled": False,
                            "position": "before_char",
                            "insertion_order": 4,
                            "case_sensitive": False,
                            "selective": False,
                        },
                    ],
                }
            },
        }

    @pytest.fixture
    def processor(self, sample_character_data) -> WorldInfoProcessor:
        """Create a WorldInfoProcessor instance."""
        return WorldInfoProcessor(sample_character_data)

    def test_entry_token_estimate_weights_cjk_over_len_over_four(self, processor):
        """task-32808.10: the entry token estimate must go through the shared
        CJK-aware estimator, not a plain len//4. A run of CJK characters costs
        far more than one token per four characters; the old heuristic
        under-counted it and could overflow _apply_token_budget."""
        cjk_entry = {"content": "\u4e2d\u6587" * 100}  # 200 CJK characters
        ascii_entry = {"content": "a" * 200}
        cjk_tokens = processor._estimate_entry_tokens(cjk_entry)
        ascii_tokens = processor._estimate_entry_tokens(ascii_entry)
        # plain len//4 would give 50 for both; CJK must now exceed that floor
        assert cjk_tokens > 200 // 4
        assert cjk_tokens > ascii_tokens
        # empty content still floors at 1 (a non-empty entry never costs zero)
        assert processor._estimate_entry_tokens({"content": ""}) == 1

    def test_initialization(self, processor):
        """Test processor initialization."""
        assert processor is not None
        assert len(processor.entries) == 3  # Only enabled entries
        assert processor.scan_depth == 3
        assert processor.token_budget == 500
        assert processor.recursive_scanning is False

    def test_simple_keyword_matching(self, processor):
        """Test basic keyword matching."""
        result = processor.process_messages("I traveled to Eldoria yesterday.", [])

        assert len(result["matched_entries"]) == 1
        assert (
            result["matched_entries"][0]["content"]
            == "Eldoria is a magical kingdom in the north."
        )
        assert "before_char" in result["injections"]

    def test_case_insensitive_matching(self, processor):
        """Test case-insensitive keyword matching."""
        result = processor.process_messages("I saw a DRAGON flying overhead!", [])

        assert len(result["matched_entries"]) == 1
        assert (
            result["matched_entries"][0]["content"]
            == "Dragons are ancient creatures of immense power."
        )

    def test_word_boundary_matching(self, processor):
        """Test that keywords match on word boundaries."""
        # Should not match "dragon" in "dragonfly"
        result = processor.process_messages("A dragonfly landed on my hand.", [])

        assert len(result["matched_entries"]) == 0
        assert "tokens_used" in result

    def test_multiple_matches(self, processor):
        """Test multiple keyword matches in one message."""
        result = processor.process_messages(
            "In Eldoria, I encountered a dragon near the castle.", []
        )

        assert len(result["matched_entries"]) == 2
        contents = [entry["content"] for entry in result["matched_entries"]]
        assert "Eldoria is a magical kingdom in the north." in contents
        assert "Dragons are ancient creatures of immense power." in contents

    def test_selective_activation(self, processor):
        """Test selective activation with secondary keys."""
        # Primary key alone should not trigger selective entry
        result = processor.process_messages("I found a magic sword in the cave.", [])
        assert len(result["matched_entries"]) == 0

        # Primary + secondary key should trigger
        result = processor.process_messages(
            "The magic sword awaits the chosen one.", []
        )
        assert len(result["matched_entries"]) == 1
        assert (
            result["matched_entries"][0]["content"]
            == "The magic sword can only be wielded by the chosen one."
        )

    def test_conversation_history_scanning(self, processor):
        """Test scanning conversation history."""
        history = [
            {"role": "user", "content": "Tell me about your homeland."},
            {
                "role": "assistant",
                "content": "I come from Eldoria, a beautiful kingdom.",
            },
            {"role": "user", "content": "What creatures live there?"},
        ]

        result = processor.process_messages(
            "Are there any dangerous ones?", history, scan_depth=3
        )

        assert len(result["matched_entries"]) == 1
        assert (
            result["matched_entries"][0]["content"]
            == "Eldoria is a magical kingdom in the north."
        )

    def test_position_organization(self, processor):
        """Test organization of entries by position."""
        result = processor.process_messages(
            "In Eldoria, a dragon guards the magic sword of the chosen one.", []
        )

        injections = result["injections"]

        # Debug: print what we actually got
        print(f"Matched entries: {len(result['matched_entries'])}")
        for entry in result["matched_entries"]:
            print(
                f"  - {entry.get('content', '')[:50]}... at position: {entry.get('position', 'unknown')}"
            )
        print(f"Injections: {injections}")

        # Verify we have the expected entries
        assert len(result["matched_entries"]) == 3  # Eldoria, dragon, and magic sword

        # Check that injections are organized by position
        assert "before_char" in injections
        assert len(injections["before_char"]) > 0
        assert any("Eldoria" in content for content in injections["before_char"])

        assert "after_char" in injections
        assert len(injections["after_char"]) > 0
        assert any("Dragons" in content for content in injections["after_char"])

        assert "at_start" in injections
        assert len(injections["at_start"]) > 0
        assert any("magic sword" in content for content in injections["at_start"])

    def test_format_injections(self, processor):
        """Test formatting of injections."""
        result = processor.process_messages(
            "Tell me about Eldoria and its dragons.", []
        )

        formatted = processor.format_injections(result["injections"])

        # Only positions with content should be in the formatted result
        assert "before_char" in formatted  # Eldoria entry
        assert "after_char" in formatted  # Dragon entry
        assert isinstance(formatted["before_char"], str)
        assert isinstance(formatted["after_char"], str)

        # These positions had no matching entries, so they shouldn't be in formatted
        assert "at_start" not in formatted
        assert "at_end" not in formatted

    def test_empty_character_book(self):
        """Test handling of character without world info."""
        char_data = {"name": "Test", "extensions": {}}
        processor = WorldInfoProcessor(char_data)

        result = processor.process_messages("Any text here", [])
        assert len(result["matched_entries"]) == 0
        assert len(result["injections"]) == 0

    def test_malformed_entries(self):
        """Test handling of malformed world info entries."""
        char_data = {
            "name": "Test",
            "extensions": {
                "character_book": {
                    "entries": [
                        {
                            # Missing keys
                            "content": "This entry has no keys",
                            "enabled": True,
                        },
                        {
                            "keys": [],  # Empty keys
                            "content": "This entry has empty keys",
                            "enabled": True,
                        },
                        {
                            "keys": ["valid"],
                            # Missing content
                            "enabled": True,
                        },
                    ]
                }
            },
        }

        processor = WorldInfoProcessor(char_data)
        # Should handle gracefully without crashing
        assert len(processor.entries) == 1  # Only the valid entry

    def test_recursive_scanning(self):
        """Test recursive scanning feature."""
        char_data = {
            "name": "Test",
            "extensions": {
                "character_book": {
                    "recursive_scanning": True,
                    "entries": [
                        {
                            "keys": ["castle"],
                            "content": "The castle is protected by a dragon.",
                            "enabled": True,
                            "position": "before_char",
                        },
                        {
                            "keys": ["dragon"],
                            "content": "Dragons breathe fire.",
                            "enabled": True,
                            "position": "before_char",
                        },
                    ],
                }
            },
        }

        processor = WorldInfoProcessor(char_data)
        result = processor.process_messages("Tell me about the castle.", [])

        # With recursive scanning, mentioning castle should also trigger dragon
        assert len(result["matched_entries"]) == 2

    def test_case_sensitive_matching(self):
        """Test case-sensitive keyword matching."""
        char_data = {
            "name": "Test",
            "extensions": {
                "character_book": {
                    "entries": [
                        {
                            "keys": ["HTML"],
                            "content": "HTML is a markup language.",
                            "enabled": True,
                            "case_sensitive": True,
                        }
                    ]
                }
            },
        }

        processor = WorldInfoProcessor(char_data)

        # Should match exact case
        result = processor.process_messages("I'm learning HTML.", [])
        assert len(result["matched_entries"]) == 1

        # Should not match different case
        result = processor.process_messages("I'm learning html.", [])
        assert len(result["matched_entries"]) == 0

    def test_token_budget_management(self):
        """Test token budget limiting."""
        char_data = {
            "name": "Test",
            "extensions": {
                "character_book": {
                    "token_budget": 50,  # Very small budget
                    "entries": [
                        {
                            "keys": ["test"],
                            "content": "A" * 200,  # ~50 tokens
                            "enabled": True,
                            "position": "before_char",
                            "insertion_order": 1,
                        },
                        {
                            "keys": ["test"],
                            "content": "B" * 200,  # ~50 tokens
                            "enabled": True,
                            "position": "before_char",
                            "insertion_order": 2,
                        },
                    ],
                }
            },
        }

        processor = WorldInfoProcessor(char_data)
        result = processor.process_messages(
            "This is a test.", [], apply_token_budget=True
        )

        # Only first entry should be included due to budget
        assert len(result["matched_entries"]) == 1
        assert result["matched_entries"][0]["content"] == "A" * 200
        assert result["tokens_used"] <= 50


class TestWorldInfoIntegration:
    """Integration tests for world info with the chat system."""

    def test_message_injection_order(self):
        """Test the order of content injection."""
        processor = WorldInfoProcessor(
            {
                "extensions": {
                    "character_book": {
                        "entries": [
                            {
                                "keys": ["test"],
                                "content": "START_CONTENT",
                                "enabled": True,
                                "position": "at_start",
                            },
                            {
                                "keys": ["test"],
                                "content": "BEFORE_CONTENT",
                                "enabled": True,
                                "position": "before_char",
                            },
                            {
                                "keys": ["test"],
                                "content": "AFTER_CONTENT",
                                "enabled": True,
                                "position": "after_char",
                            },
                            {
                                "keys": ["test"],
                                "content": "END_CONTENT",
                                "enabled": True,
                                "position": "at_end",
                            },
                        ]
                    }
                }
            }
        )

        result = processor.process_messages("This is a test message.", [])
        formatted = processor.format_injections(result["injections"])

        # Build the final message as the chat system would
        parts = []
        if formatted.get("at_start"):
            parts.append(formatted["at_start"])
        if formatted.get("before_char"):
            parts.append(formatted["before_char"])
        parts.append("This is a test message.")
        if formatted.get("after_char"):
            parts.append(formatted["after_char"])
        if formatted.get("at_end"):
            parts.append(formatted["at_end"])

        final_message = "\n\n".join(parts)

        # Check order
        assert final_message.index("START_CONTENT") < final_message.index(
            "BEFORE_CONTENT"
        )
        assert final_message.index("BEFORE_CONTENT") < final_message.index(
            "This is a test message."
        )
        assert final_message.index("This is a test message.") < final_message.index(
            "AFTER_CONTENT"
        )
        assert final_message.index("AFTER_CONTENT") < final_message.index("END_CONTENT")


class TestPrecompiledKeywordPatterns:
    """task-34393: literal keyword patterns compile once per entry/key at load."""

    @pytest.fixture
    def rich_book(self) -> Dict[str, Any]:
        """A fixture spanning every literal-matching branch: case-insensitive,
        case-sensitive, selective+secondary, metacharacter keys, unicode keys,
        multi-word keys, a word-boundary trap, and recursion-only entries."""
        return {
            "name": "Golden book",
            "scan_depth": 3,
            "token_budget": 5000,
            "recursive_scanning": True,
            "entries": [
                {
                    "keys": ["dragon", "Dragon", "DRAGON"],
                    "content": "Dragons are ancient creatures.",
                    "enabled": True,
                    "position": "before_char",
                    "insertion_order": 1,
                    "case_sensitive": False,
                },
                {
                    "keys": ["dragonfly"],
                    "content": "Dragonflies are not dragons.",
                    "enabled": True,
                    "position": "before_char",
                    "insertion_order": 2,
                    "case_sensitive": False,
                },
                {
                    "keys": ["HTML"],
                    "content": "HTML is a markup language.",
                    "enabled": True,
                    "position": "before_char",
                    "insertion_order": 3,
                    "case_sensitive": True,
                },
                {
                    "keys": ["c++", "r.x?", "a|b"],
                    "content": "Metacharacter keywords match literally.",
                    "enabled": True,
                    "position": "before_char",
                    "insertion_order": 4,
                    "case_sensitive": False,
                },
                {
                    "keys": ["Él Dorado", "smörgåsbord"],
                    "content": "Unicode keywords match.",
                    "enabled": True,
                    "position": "before_char",
                    "insertion_order": 5,
                    "case_sensitive": False,
                },
                {
                    "keys": ["magic sword", "magic sword"],
                    "content": "The magic sword glows.",
                    "enabled": True,
                    "position": "at_start",
                    "insertion_order": 6,
                    "selective": True,
                    "secondary_keys": ["chosen one", "prophecy"],
                    "case_sensitive": False,
                },
                {
                    "keys": ["castle"],
                    "content": "The castle is guarded by a dragon.",
                    "enabled": True,
                    "position": "before_char",
                    "insertion_order": 7,
                    "case_sensitive": False,
                },
                {
                    "keys": ["treasure"],
                    "content": "The treasure is hidden in the keep.",
                    "enabled": True,
                    "position": "at_end",
                    "insertion_order": 8,
                    "case_sensitive": False,
                },
            ],
        }

    @pytest.fixture
    def char_data(self, rich_book) -> Dict[str, Any]:
        return {"name": "Golden", "extensions": {"character_book": rich_book}}

    def test_processed_entries_carry_compiled_patterns(self, char_data):
        """Entries processed by _process_entry store a compiled word-boundary
        pattern per primary and secondary key (both case variants)."""
        processor = WorldInfoProcessor(char_data)

        dragon = next(e for e in processor.entries if e["keys"][0] == "dragon")
        pat = dragon["_key_patterns"]["dragon"]
        assert isinstance(pat, re.Pattern)
        assert pat.pattern == r"\b" + re.escape("dragon") + r"\b"
        # Case-insensitive variant is keyed by the lowercased key and built
        # from the lowercased key (mirrors _keyword_in_text(key.lower(), ...)).
        lowered = dragon["_key_patterns_lower"]["dragon"]
        assert isinstance(lowered, re.Pattern)
        assert lowered.pattern == r"\b" + re.escape("dragon".lower()) + r"\b"
        # Uppercase alias folds into the same lowered key with the same pattern.
        assert dragon["_key_patterns_lower"]["DRAGON".lower()].pattern == lowered.pattern
        # The case-sensitive dict compiles each alias as written (match-time
        # case handling is unchanged — see _key_hits).
        assert dragon["_key_patterns"]["Dragon"].pattern == (
            r"\b" + re.escape("Dragon") + r"\b"
        )

        # Secondary keys of selective entries are compiled too.
        sword = next(e for e in processor.entries if "magic sword" in e["keys"])
        assert sword["_key_patterns"]["magic sword"].pattern == (
            r"\b" + re.escape("magic sword") + r"\b"
        )
        assert sword["_key_patterns_lower"]["chosen one"].pattern == (
            r"\b" + re.escape("chosen one".lower()) + r"\b"
        )

        # Metacharacters are escaped exactly as re.escape did per call.
        meta = next(e for e in processor.entries if "c++" in e["keys"])
        assert meta["_key_patterns_lower"]["c++"].pattern == (
            r"\b" + re.escape("c++") + r"\b"
        )

        # Regex-flagged entries compile nothing (they go through regex_search).
        regex_book = {
            "name": "R",
            "extensions": {
                "character_book": {
                    "entries": [
                        {
                            "keys": ["w[ao]rden"],
                            "content": "c",
                            "enabled": True,
                            "regex": True,
                        }
                    ]
                }
            },
        }
        regex_processor = WorldInfoProcessor(regex_book)
        assert regex_processor.entries[0]["_key_patterns"] == {}
        assert regex_processor.entries[0]["_key_patterns_lower"] == {}

    def test_patterns_compiled_once_not_per_message(self, char_data, monkeypatch):
        """After entry load, sends must not recompile keyword patterns nor fall
        back to the per-call rebuild in _keyword_in_text."""
        from tldw_chatbook.Character_Chat import world_info_processor as wip

        compiles = {"n": 0}
        real_compile = wip._compile_keyword_pattern

        def counting_compile(keyword):
            compiles["n"] += 1
            return real_compile(keyword)

        monkeypatch.setattr(wip, "_compile_keyword_pattern", counting_compile)

        keyword_checks = {"n": 0}
        real_check = wip.WorldInfoProcessor._keyword_in_text

        def counting_check(self, keyword, text):
            keyword_checks["n"] += 1
            return real_check(self, keyword, text)

        monkeypatch.setattr(wip.WorldInfoProcessor, "_keyword_in_text", counting_check)

        processor = WorldInfoProcessor(char_data)
        compiles_at_load = compiles["n"]
        assert compiles_at_load > 0  # compiled at entry-load time

        message = (
            "In Eldoria the HTML codex mentions a DRAGON, a dragonfly, c++, "
            "Él Dorado, and the magic sword of the chosen one."
        )
        history = [
            {"role": "user", "content": "Tell me about the castle."},
            {"role": "assistant", "content": "The keep hides a treasure."},
        ]
        results = [processor.process_messages(message, history) for _ in range(5)]

        # Matching still works...
        matched_contents = [e["content"] for e in results[-1]["matched_entries"]]
        assert "Dragons are ancient creatures." in matched_contents
        assert "The magic sword glows." in matched_contents
        # ...and nothing was recompiled (or rebuilt per call) across sends.
        assert compiles["n"] == compiles_at_load
        assert keyword_checks["n"] == 0

    def test_precompiled_matching_equals_fallback_golden(self, char_data):
        """Golden: activation decisions with precompiled patterns are identical
        to the previous per-call implementation (entries with the precompiled
        patterns stripped fall back to _keyword_in_text — the old path)."""
        precompiled = WorldInfoProcessor(copy.deepcopy(char_data))
        fallback = WorldInfoProcessor(copy.deepcopy(char_data))
        for entry in fallback.entries + fallback._candidate_entries:
            entry.pop("_key_patterns", None)
            entry.pop("_key_patterns_lower", None)
        assert precompiled.entries

        history = [
            {"role": "user", "content": "Tell me about the castle."},
            {"role": "assistant", "content": "The keep hides a treasure."},
            {"role": "user", "content": "Anyone speaking smörgåsbord there?"},
        ]
        probes = [
            "A DRAGON flies overhead.",
            "A dragonfly landed on my hand.",
            "I'm learning HTML.",
            "I'm learning html.",
            "Something about c++ and r.x? and a|b here.",
            "Welcome to Él Dorado.",
            "The magic sword awaits the chosen one.",
            "The magic sword alone.",
            "Tell me about the castle.",
            "dragon castle treasure html c++",
            "nothing relevant at all",
        ]

        def project(result: Dict[str, Any]) -> Dict[str, Any]:
            """Activation decisions on stable fields (entry dicts carry
            non-comparable compiled pattern objects)."""
            return {
                "matched": [
                    (e["content"], e["position"], e["insertion_order"])
                    for e in result["matched_entries"]
                ],
                "injections": result["injections"],
                "tokens_used": result["tokens_used"],
            }

        for probe in probes:
            got = precompiled.process_messages(probe, history)
            want = fallback.process_messages(probe, history)
            assert project(got) == project(want), f"divergence on probe: {probe!r}"
