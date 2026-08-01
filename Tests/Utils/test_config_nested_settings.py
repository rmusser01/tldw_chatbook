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


TRANSCRIPTION_SCRATCH_TOML = """
[transcription]
default_provider = "parakeet-mlx"
"""


class TestDottedFormStringDefaultRegression:
    """TASK-1754: the dotted 2-arg form dropped the caller's default (and
    the real configured value) whenever the default happened to be a
    ``str`` -- the common case for provider/model/language/device names.

    Root cause: the old heuristic (``not isinstance(key, str) and default
    is None``) only recaptured ``key`` as the real default when it was a
    non-string type. For a string default, ``key`` was left untouched and
    misinterpreted as one more path segment to walk, so the lookup always
    fell off the tree and ``default`` (never reassigned, still ``None``)
    won -- silently discarding both the configured value AND the caller's
    fallback. Measured directly (see the task):

        get_cli_setting("transcription.default_provider", "FALLBACK")    -> None
        get_cli_setting("transcription", "default_provider", "FALLBACK") -> "parakeet-mlx"
    """

    def test_dotted_form_with_string_default_reads_the_configured_value(
        self, tmp_path, monkeypatch
    ):
        """The exact call shape `_LegacyTranscriptionBackend.__init__` uses:
        a dotted path plus a string fallback. Must resolve to the real
        configured provider, not silently fall back to "FALLBACK"."""
        with _real_config(tmp_path, monkeypatch, TRANSCRIPTION_SCRATCH_TOML):
            assert (
                get_cli_setting("transcription.default_provider", "FALLBACK")
                == "parakeet-mlx"
            )

    def test_dotted_form_with_string_default_honours_the_default_when_unset(
        self, tmp_path, monkeypatch
    ):
        """When the key genuinely isn't configured, the caller's own
        string default must come back -- not ``None``.

        Deliberately not ``transcription.default_provider``: task-867 bakes
        a platform-preference default for that specific key into
        ``CONFIG_TOML_CONTENT`` itself, so it is never actually "unset" --
        ``parakeet_precision`` has no such baked-in default and is absent
        from ``TRANSCRIPTION_SCRATCH_TOML``.
        """
        with _real_config(tmp_path, monkeypatch, TRANSCRIPTION_SCRATCH_TOML):
            assert (
                get_cli_setting("transcription.parakeet_precision", "FALLBACK")
                == "FALLBACK"
            )

    def test_matches_the_traditional_three_arg_form(self, tmp_path, monkeypatch):
        """The dotted 2-arg form and the traditional 3-arg form must agree
        on the same underlying setting (task's measured-evidence pair)."""
        with _real_config(tmp_path, monkeypatch, TRANSCRIPTION_SCRATCH_TOML):
            dotted = get_cli_setting("transcription.default_provider", "FALLBACK")
            traditional = get_cli_setting(
                "transcription", "default_provider", "FALLBACK"
            )
            assert dotted == traditional == "parakeet-mlx"

    def test_library_ingest_browse_location_audit_fix(self, tmp_path, monkeypatch):
        """TASK-1754 audit fallout: ``library_screen._library_ingest_browse_location``
        called ``get_cli_setting("library.ingest", "last_directory")`` -- a 2-arg
        dotted-section call with a literal (non-default) second argument and no
        explicit default. Under the fixed accessor, an omitted third argument on
        a dotted `section` means "the whole path is in `section`, `key` is the
        default" -- so this exact shape would have started returning the whole
        ``[library.ingest]`` table instead of just ``last_directory``. Fixed by
        adding an explicit ``None`` default at the call site (now the
        unambiguous 3-arg traditional form).

        Drives the real method (not a re-derivation of its call shape): the
        method touches no other screen/app state, so it can be invoked
        unbound with a throwaway `self`.
        """
        from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

        remembered_dir = tmp_path / "last-imported"
        remembered_dir.mkdir()
        with _real_config(
            tmp_path,
            monkeypatch,
            f'[library.ingest]\nlast_directory = "{remembered_dir}"\nbackend = "local"\n',
        ):
            result = LibraryScreen._library_ingest_browse_location(None)

        assert result == str(remembered_dir)
