"""Nested-table resolution for get_cli_setting (TASK-229).

TOML ``[chat.images]`` loads as ``config["chat"]["images"]`` — but
``get_cli_setting(section, key, default)`` did a FLAT ``config.get(section)``,
so every caller passing a dotted section silently got its default forever
(``show_attach_button``, ``save_location``, ``show_mic_button``,
``[mcp.hub_state] advanced_open``, and the three-level
``prompts.document_generation.*`` document configs). TASK-222 hit the same
bug (C1) and worked around it locally; this fixes the accessor itself.

Every test here drives the REAL loader via TLDW_CONFIG_PATH + force_reload —
zero monkeypatching of the accessor (the C1 lesson: accessor mocks hid an
inert feature through five review gates).
"""

import os
from contextlib import contextmanager

import pytest

import tldw_chatbook.config as config_mod
from tldw_chatbook.config import get_cli_setting


SCRATCH_TOML = """
[general]
default_tab = "chat"

[chat.images]
show_attach_button = false
save_location = "~/Pictures/tldw"

[chat.voice]
show_mic_button = false

[mcp.hub_state]
advanced_open = true

[prompts.document_generation.timeline]
prompt = "Configured timeline prompt."
temperature = 0.9
max_tokens = 123

[splash_screen]
custom_image_path = "/tmp/custom.png"
"""


@contextmanager
def _real_config(tmp_path, monkeypatch, toml_text: str):
    """Point the real loader at a scratch TOML; restore + reload afterwards."""
    config_path = tmp_path / "scratch-nested-config.toml"
    config_path.write_text(toml_text, encoding="utf-8")
    original_env = os.environ.get("TLDW_CONFIG_PATH")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    config_mod.load_cli_config_and_ensure_existence(force_reload=True)
    try:
        yield
    finally:
        if original_env is not None:
            monkeypatch.setenv("TLDW_CONFIG_PATH", original_env)
        else:
            monkeypatch.delenv("TLDW_CONFIG_PATH", raising=False)
        config_mod.load_cli_config_and_ensure_existence(force_reload=True)


class TestRepairedReaderTuples:
    """The exact (section, key, default) tuples of every production caller
    the audit found broken — each must now read the real nested table."""

    def test_chat_images_show_attach_button(self, tmp_path, monkeypatch):
        with _real_config(tmp_path, monkeypatch, SCRATCH_TOML):
            assert get_cli_setting("chat.images", "show_attach_button", True) is False

    def test_chat_images_save_location(self, tmp_path, monkeypatch):
        with _real_config(tmp_path, monkeypatch, SCRATCH_TOML):
            assert (
                get_cli_setting("chat.images", "save_location", "~/Downloads")
                == "~/Pictures/tldw"
            )

    def test_chat_voice_show_mic_button(self, tmp_path, monkeypatch):
        with _real_config(tmp_path, monkeypatch, SCRATCH_TOML):
            assert get_cli_setting("chat.voice", "show_mic_button", True) is False

    def test_mcp_hub_state_advanced_open(self, tmp_path, monkeypatch):
        with _real_config(tmp_path, monkeypatch, SCRATCH_TOML):
            assert get_cli_setting("mcp.hub_state", "advanced_open", False) is True

    def test_three_level_document_generation_dict_form(self, tmp_path, monkeypatch):
        """document_generator.py's exact shape: dotted three-level section
        with the default dict as the second positional argument."""
        default = {"prompt": "fallback", "temperature": 0.3, "max_tokens": 2000}
        with _real_config(tmp_path, monkeypatch, SCRATCH_TOML):
            value = get_cli_setting("prompts.document_generation.timeline", default)
            assert value == {
                "prompt": "Configured timeline prompt.",
                "temperature": 0.9,
                "max_tokens": 123,
            }

    def test_missing_nested_key_returns_default(self, tmp_path, monkeypatch):
        with _real_config(tmp_path, monkeypatch, SCRATCH_TOML):
            assert get_cli_setting("chat.images", "nope", "fallback") == "fallback"

    def test_missing_nested_section_returns_default(self, tmp_path, monkeypatch):
        with _real_config(tmp_path, monkeypatch, SCRATCH_TOML):
            assert get_cli_setting("chat.absent", "key", 7) == 7
            default = {"prompt": "fallback"}
            assert (
                get_cli_setting("prompts.document_generation.absent", default)
                is default
            )


class TestWorkingShapesUnchanged:
    """The shapes that already worked must keep their exact semantics."""

    def test_traditional_flat_section(self, tmp_path, monkeypatch):
        with _real_config(tmp_path, monkeypatch, SCRATCH_TOML):
            assert get_cli_setting("general", "default_tab", "notes") == "chat"
            assert get_cli_setting("general", "absent", "d") == "d"
            assert get_cli_setting("absent_section", "key", "d") == "d"

    def test_one_arg_single_dot(self, tmp_path, monkeypatch):
        with _real_config(tmp_path, monkeypatch, SCRATCH_TOML):
            assert get_cli_setting("general.default_tab") == "chat"
            assert (
                get_cli_setting("splash_screen.custom_image_path", None, "")
                == "/tmp/custom.png"
            )

    def test_dotted_with_non_string_default_second_arg(self, tmp_path, monkeypatch):
        with _real_config(tmp_path, monkeypatch, SCRATCH_TOML):
            assert get_cli_setting("prompts.document_generation.timeline", 500) == {
                "prompt": "Configured timeline prompt.",
                "temperature": 0.9,
                "max_tokens": 123,
            }
            assert get_cli_setting("general.absent_number", 500) == 500

    def test_flat_key_shadows_nested(self, tmp_path, monkeypatch):
        """A literal flat "chat.images" top-level key must win over nested
        navigation (impossible from TOML, but code-merged dicts can carry it;
        flat-first preserves every previously-working lookup bit-for-bit)."""
        with _real_config(tmp_path, monkeypatch, SCRATCH_TOML):
            config = config_mod.load_cli_config_and_ensure_existence()
            config["chat.images"] = {"show_attach_button": "flat-wins"}
            try:
                assert (
                    get_cli_setting("chat.images", "show_attach_button", True)
                    == "flat-wins"
                )
            finally:
                config.pop("chat.images", None)


class TestConsumerPath:
    """One repaired reader exercised through its real consumer (AC #3):
    DocumentGenerator's ctor reads the three-level document configs."""

    def test_document_generator_reads_configured_prompt(self, tmp_path, monkeypatch):
        from tldw_chatbook.Chat.document_generator import DocumentGenerator

        with _real_config(tmp_path, monkeypatch, SCRATCH_TOML):
            generator = DocumentGenerator(
                str(tmp_path / "docgen-test.db"), client_id="task-229-test"
            )
            assert generator.timeline_config["prompt"] == "Configured timeline prompt."
            assert generator.timeline_config["max_tokens"] == 123
            # briefing has no override in the scratch TOML -> ctor default
            assert generator.briefing_config["max_tokens"] == 2500
