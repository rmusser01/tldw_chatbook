"""Efficiency guards on the chat() send path (task-20b/20c).

20b: the DEBUG-only payload-summary loop must not perform any summary work
(per-message part scans + joins) when the root logger is below DEBUG.
20c: the post-generation replacement dictionary must be parsed once per
(size, mtime_ns) file signature instead of once per non-streaming response.
"""

import logging
from contextlib import contextmanager
from unittest.mock import patch

import pytest

import tldw_chatbook.Chat.Chat_Functions as chat_functions_module

GOLDEN_PAYLOAD_SUMMARY_LINES = [
    "Debug - Chat Function - Final LLM payload structure:",
    "  Msg 0: content_type=list; parts=[text(length=11)]",
    "  Msg 1: content_type=list; parts=[text(length=18), image_url(length=26)]",
    "  Msg 2: content_type=list; parts=[text(length=6)]",
]


def _golden_history():
    return [
        {"role": "user", "content": [{"type": "text", "text": "hello there"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "an assistant reply"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,QUJD"},
                },
            ],
        },
    ]


def _large_history(message_count: int):
    return [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": [{"type": "text", "text": f"turn {index}"}],
        }
        for index in range(message_count)
    ]


class TestDebugPayloadSummaryGuard:
    @pytest.mark.unit
    def test_payload_summary_performs_zero_work_at_info_level(self, caplog):
        summary_calls = []
        real_summary = chat_functions_module._debug_dump_llm_payload_summary

        def spy(payload):
            summary_calls.append(payload)
            return real_summary(payload)

        with (
            patch.object(chat_functions_module, "_debug_dump_llm_payload_summary", spy),
            patch.object(
                chat_functions_module, "chat_api_call", return_value="ok"
            ) as mock_call,
            patch.object(chat_functions_module, "load_settings", return_value={}),
            caplog.at_level(logging.INFO),
        ):
            response = chat_functions_module.chat(
                message="latest",
                history=_large_history(499),
                media_content=None,
                selected_parts=[],
                api_endpoint="openai",
                api_key="k",
                custom_prompt=None,
                temperature=0.7,
            )

        assert response == "ok"
        assert mock_call.call_count == 1
        # 499 history turns + the current user message reach the payload build.
        assert len(mock_call.call_args.kwargs["messages_payload"]) == 500
        assert summary_calls == [], (
            "at INFO level the per-message summary loop (part scans + joins) "
            "must not run at all"
        )

    @pytest.mark.unit
    def test_payload_summary_output_is_byte_identical_at_debug_level(self, caplog):
        with (
            patch.object(chat_functions_module, "chat_api_call", return_value="ok"),
            patch.object(chat_functions_module, "load_settings", return_value={}),
            caplog.at_level(logging.DEBUG),
        ):
            response = chat_functions_module.chat(
                message="latest",
                history=_golden_history(),
                media_content=None,
                selected_parts=[],
                api_endpoint="openai",
                api_key="k",
                custom_prompt=None,
                temperature=0.7,
                image_history_mode="send_all",
                strip_thinking_tags=False,
            )

        assert response == "ok"
        emitted = [
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith(
                "Debug - Chat Function - Final LLM payload structure:"
            )
            or record.getMessage().startswith("  Msg ")
        ]
        assert emitted == GOLDEN_PAYLOAD_SUMMARY_LINES


import os


@pytest.fixture
def isolated_post_gen_dict_cache(monkeypatch, tmp_path):
    """Count real admitted parser file reads with a fresh cache per test."""
    from Tests.Backup_Recovery.test_compound_chat_source_lifetimes import (
        fresh_dictionary_library,
    )

    library = fresh_dictionary_library(monkeypatch)
    monkeypatch.setattr(
        library, "_default_dictionary_import_directory", lambda: tmp_path
    )
    monkeypatch.setattr(
        chat_functions_module,
        "parse_user_dict_markdown_file",
        library.parse_user_dict_markdown_file,
    )
    reads = []
    original = library._dictionary_files.opened

    @contextmanager
    def observed_open(path, *args, **kwargs):
        reads.append(str(path))
        with original(path, *args, **kwargs) as source:
            yield source

    monkeypatch.setattr(library._dictionary_files, "opened", observed_open)
    return reads


def _write_dict_file(path, content: str):
    path.write_text(content, encoding="utf-8")
    stat = path.stat()
    # Ensure a later rewrite always differs in mtime_ns, even on
    # coarse-grained filesystems.
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))


@pytest.mark.bootstrap_profile
class TestPostGenDictionaryCache:
    @pytest.mark.unit
    def test_two_responses_parse_dictionary_once_with_identical_replacement(
        self, tmp_path, isolated_post_gen_dict_cache
    ):
        dict_file = tmp_path / "post_gen.md"
        _write_dict_file(dict_file, "llm: large language model\n")
        parse_calls = isolated_post_gen_dict_cache

        settings = {
            "chat_dictionaries": {
                "post_gen_replacement": True,
                "post_gen_replacement_dict": str(dict_file),
            }
        }
        with (
            patch.object(
                chat_functions_module, "chat_api_call", return_value="the llm says hi"
            ),
            patch.object(chat_functions_module, "load_settings", return_value=settings),
        ):
            first = chat_functions_module.chat(
                message="hi",
                history=[],
                media_content=None,
                selected_parts=[],
                api_endpoint="openai",
                api_key="k",
                custom_prompt=None,
                temperature=0.7,
                strip_thinking_tags=False,
            )
            second = chat_functions_module.chat(
                message="hi",
                history=[],
                media_content=None,
                selected_parts=[],
                api_endpoint="openai",
                api_key="k",
                custom_prompt=None,
                temperature=0.7,
                strip_thinking_tags=False,
            )

        assert len(parse_calls) == 1, (
            "two non-streaming responses over an unchanged dictionary file must "
            f"parse it exactly once, parsed {len(parse_calls)} times"
        )
        assert first == "the large language model says hi"
        assert second == first

    @pytest.mark.unit
    def test_dictionary_reparse_after_file_change_applies_new_entries(
        self, tmp_path, isolated_post_gen_dict_cache
    ):
        dict_file = tmp_path / "post_gen.md"
        _write_dict_file(dict_file, "llm: large language model\n")
        parse_calls = isolated_post_gen_dict_cache

        settings = {
            "chat_dictionaries": {
                "post_gen_replacement": True,
                "post_gen_replacement_dict": str(dict_file),
            }
        }
        with (
            patch.object(
                chat_functions_module, "chat_api_call", return_value="the llm says hi"
            ),
            patch.object(chat_functions_module, "load_settings", return_value=settings),
        ):
            first = chat_functions_module.chat(
                message="hi",
                history=[],
                media_content=None,
                selected_parts=[],
                api_endpoint="openai",
                api_key="k",
                custom_prompt=None,
                temperature=0.7,
                strip_thinking_tags=False,
            )
            _write_dict_file(dict_file, "llm: rewritten replacement\n")
            second = chat_functions_module.chat(
                message="hi",
                history=[],
                media_content=None,
                selected_parts=[],
                api_endpoint="openai",
                api_key="k",
                custom_prompt=None,
                temperature=0.7,
                strip_thinking_tags=False,
            )

        assert len(parse_calls) == 2, (
            "a touched dictionary file must trigger a re-parse"
        )
        assert first == "the large language model says hi"
        assert second == "the rewritten replacement says hi"

    @pytest.mark.unit
    def test_dictionary_cache_is_keyed_by_configured_path(
        self, tmp_path, isolated_post_gen_dict_cache
    ):
        first_file = tmp_path / "post_gen_a.md"
        second_file = tmp_path / "post_gen_b.md"
        _write_dict_file(first_file, "llm: large language model\n")
        _write_dict_file(second_file, "llm: other dictionary\n")
        parse_calls = isolated_post_gen_dict_cache

        def settings_for(path):
            return {
                "chat_dictionaries": {
                    "post_gen_replacement": True,
                    "post_gen_replacement_dict": str(path),
                }
            }

        current_settings = settings_for(first_file)
        with (
            patch.object(
                chat_functions_module, "chat_api_call", return_value="the llm says hi"
            ),
            patch.object(
                chat_functions_module,
                "load_settings",
                side_effect=lambda: current_settings,
            ),
        ):
            chat_functions_module.chat(
                message="hi",
                history=[],
                media_content=None,
                selected_parts=[],
                api_endpoint="openai",
                api_key="k",
                custom_prompt=None,
                temperature=0.7,
                strip_thinking_tags=False,
            )
            current_settings = settings_for(second_file)
            chat_functions_module.chat(
                message="hi",
                history=[],
                media_content=None,
                selected_parts=[],
                api_endpoint="openai",
                api_key="k",
                custom_prompt=None,
                temperature=0.7,
                strip_thinking_tags=False,
            )

        assert len(parse_calls) == 2, (
            "switching the configured dictionary path must not serve the old "
            "path's cached entries"
        )
        assert parse_calls == [str(first_file), str(second_file)]
