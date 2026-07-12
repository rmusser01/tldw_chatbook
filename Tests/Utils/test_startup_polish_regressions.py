"""Regression coverage for first-run startup polish issues."""

from __future__ import annotations

import builtins
import importlib
import sys
from types import SimpleNamespace

import pytest
from loguru import logger


def test_splash_effect_modules_import_without_annotation_name_errors() -> None:
    """Splash effects should not fail import due to runtime-only typing names."""
    modules = [
        "tldw_chatbook.Utils.Splash_Screens.classic.scrolling_credits",
        "tldw_chatbook.Utils.Splash_Screens.tech.terminal_boot",
    ]

    for module_name in modules:
        sys.modules.pop(module_name, None)
        importlib.import_module(module_name)


def test_code_scroll_splash_effect_renders_without_missing_escape_constant() -> None:
    """The first-run random splash card should not fail if code_scroll is selected."""
    from tldw_chatbook.Utils.Splash_Screens.tech.code_scroll import CodeScrollEffect

    frame = CodeScrollEffect(
        parent_widget=object(),
        width=40,
        height=12,
        title="tldw",
        subtitle="Ready",
    ).update()

    assert isinstance(frame, str)
    assert "tldw" in frame


def test_game_of_life_splash_effect_renders_without_missing_escape_constant() -> None:
    """The game_of_life splash card should render a frame without a NameError.

    Regression: game_of_life.py used ESCAPED_OPEN_BRACKET without importing it,
    crashing every animation frame ("name 'ESCAPED_OPEN_BRACKET' is not defined").
    Grid dimensions equal to the display height force the title overlay onto grid
    rows so both escape call sites are exercised.
    """
    from tldw_chatbook.Utils.Splash_Screens.gaming.game_of_life import GameOfLifeEffect

    effect = GameOfLifeEffect(
        parent_widget=object(),
        title="GoL",
        width=10,
        height=10,
        display_width=20,
        display_height=10,
        update_interval=0.0,
    )

    frame = effect.update()

    assert isinstance(frame, str)
    assert len(frame.splitlines()) == 10


@pytest.mark.parametrize(
    ("module_name", "class_name", "effect_kwargs"),
    [
        ("tldw_chatbook.Utils.Splash_Screens.gaming.tetris_block", "TetrisBlockEffect", {}),
        ("tldw_chatbook.Utils.Splash_Screens.gaming.maze_generator", "MazeGeneratorEffect", {}),
        ("tldw_chatbook.Utils.Splash_Screens.classic.pixel_dissolve", "PixelDissolveEffect", {}),
        ("tldw_chatbook.Utils.Splash_Screens.environmental.morphing_shape", "MorphingShapeEffect", {}),
        ("tldw_chatbook.Utils.Splash_Screens.environmental.dna_helix", "DNAHelixEffect", {}),
        ("tldw_chatbook.Utils.Splash_Screens.environmental.wave_ripple", "WaveRippleEffect", {}),
        ("tldw_chatbook.Utils.Splash_Screens.environmental.fireworks", "FireworksEffect", {}),
        ("tldw_chatbook.Utils.Splash_Screens.environmental.ascii_kaleidoscope", "ASCIIKaleidoscopeEffect", {}),
        ("tldw_chatbook.Utils.Splash_Screens.environmental.particle_swarm", "ParticleSwarmEffect", {}),
        ("tldw_chatbook.Utils.Splash_Screens.tech.mining", "MiningEffect", {"content": "tldw chatbook"}),
        ("tldw_chatbook.Utils.Splash_Screens.tech.circuit_board", "CircuitBoardEffect", {}),
        ("tldw_chatbook.Utils.Splash_Screens.tech.digital_rain", "DigitalRainEffect", {}),
    ],
)
def test_sibling_splash_effects_render_without_missing_escape_constant(
    module_name: str, class_name: str, effect_kwargs: dict
) -> None:
    """Sibling splash effects must have ESCAPED_OPEN_BRACKET bound and render frames.

    Regression: these effects used ESCAPED_OPEN_BRACKET in their render paths
    without importing it from base_effect (same bug class as game_of_life),
    crashing whichever animation frame first reached an escape call site. The
    module-level assertion makes that NameError class impossible even for
    effects whose escape path only fires once time-based content appears; the
    frame advances exercise the escape sites for effects that render styled
    cells immediately.
    """
    module = importlib.import_module(module_name)

    assert getattr(module, "ESCAPED_OPEN_BRACKET", None) == r"\["

    effect = getattr(module, class_name)(parent_widget=object(), **effect_kwargs)
    frame = None
    for _ in range(3):
        frame = effect.update()

    assert isinstance(frame, str)


def test_random_splash_selection_skips_missing_active_card_definitions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default active cards can outpace implemented card definitions."""
    from tldw_chatbook.Widgets import splash_screen

    def fake_get_cli_setting(setting: str, default=None, *args, **kwargs):
        if setting == "splash_screen":
            return {
                "card_selection": "random",
                "active_cards": ["neon_sign", "default"],
            }
        return default

    choices: list[list[str]] = []

    def fake_choice(options):
        choices.append(list(options))
        assert "neon_sign" not in options
        return options[0]

    monkeypatch.setattr(splash_screen, "get_cli_setting", fake_get_cli_setting)
    monkeypatch.setattr(splash_screen.random, "choice", fake_choice)

    screen = splash_screen.SplashScreen(duration=0)

    assert screen.card_name == "default"
    assert choices == [["default"]]


def test_nltk_download_false_is_not_logged_as_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_chatbook.Chunking import Chunk_Lib

    messages: list[tuple[str, str]] = []
    sink_id = logger.add(lambda message: messages.append((message.record["level"].name, message.record["message"])))
    try:
        def mock_find(_path):
            raise LookupError("missing")

        fake_nltk = SimpleNamespace(
            data=SimpleNamespace(find=mock_find),
            download=lambda _package: False,
        )

        monkeypatch.setattr(Chunk_Lib, "NLTK_AVAILABLE", True)
        monkeypatch.setattr(Chunk_Lib, "nltk", fake_nltk)
        # ensure_nltk_data() is now idempotent (first successful run sets
        # _nltk_data_ready); reset it so this test always exercises the real
        # find/download path regardless of whether an earlier test already
        # warmed punkt in this process.
        monkeypatch.setattr(Chunk_Lib, "_nltk_data_ready", False)

        Chunk_Lib.ensure_nltk_data()
    finally:
        logger.remove(sink_id)

    assert not any("downloaded successfully" in message for _level, message in messages)
    assert any(level in {"WARNING", "ERROR"} and "punkt" in message for level, message in messages)


def test_missing_openai_tts_mapping_falls_back_without_error_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib import resources as importlib_resources
    from tldw_chatbook import config

    messages: list[tuple[str, str]] = []
    sink_id = logger.add(lambda message: messages.append((message.record["level"].name, message.record["message"])))
    original_open = builtins.open
    try:
        monkeypatch.setattr(
            importlib_resources,
            "files",
            lambda _package: (_ for _ in ()).throw(FileNotFoundError("packaged mapping missing")),
        )

        def fake_open(path, *args, **kwargs):
            if str(path).endswith("openai_tts_mappings.json"):
                raise FileNotFoundError("filesystem mapping missing")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", fake_open)

        mappings = config.load_openai_mappings()
    finally:
        logger.remove(sink_id)

    assert mappings["models"]["tts-1"] == "openai_official_tts-1"
    assert mappings["voices"]["alloy"] == "alloy"
    assert not any(level == "ERROR" for level, _message in messages)
