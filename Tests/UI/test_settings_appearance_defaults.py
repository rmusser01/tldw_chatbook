import inspect

from tldw_chatbook.UI.Screens.settings_appearance_defaults import (
    SettingsAppearanceDefaults,
    build_appearance_save_sections,
    load_appearance_defaults,
    validate_appearance_defaults,
)


def test_load_appearance_defaults_uses_safe_defaults():
    defaults = load_appearance_defaults({})

    assert defaults.default_theme == "textual-dark"
    assert defaults.palette_theme_limit == 1
    assert defaults.font_size == 12
    assert defaults.density == "normal"
    assert defaults.animations_enabled is True
    assert defaults.smooth_scrolling is True
    assert defaults.console_transcript_style == "role_accents"


def test_load_appearance_defaults_reads_general_web_and_appearance_sections():
    defaults = load_appearance_defaults(
        {
            "general": {
                "default_theme": "monokai",
                "palette_theme_limit": "3",
            },
            "web_server": {
                "font_size": "14",
            },
            "appearance": {
                "density": "comfortable",
                "animations_enabled": "false",
                "smooth_scrolling": "yes",
                "console_transcript_style": "immersive_rp",
            },
        }
    )

    assert defaults == SettingsAppearanceDefaults(
        default_theme="monokai",
        palette_theme_limit=3,
        font_size=14,
        density="comfortable",
        animations_enabled=False,
        smooth_scrolling=True,
        console_transcript_style="immersive_rp",
    )


def test_load_appearance_defaults_coerces_integer_boolean_values():
    defaults = load_appearance_defaults(
        {
            "appearance": {
                "animations_enabled": 0,
                "smooth_scrolling": 1,
            },
        }
    )

    assert defaults.animations_enabled is False
    assert defaults.smooth_scrolling is True


def test_load_appearance_defaults_falls_back_for_malformed_values():
    defaults = load_appearance_defaults(
        {
            "general": {
                "default_theme": "",
                "palette_theme_limit": "not-an-int",
            },
            "web_server": {
                "font_size": "huge",
            },
            "appearance": {
                "density": "spacious",
                "animations_enabled": "unknown",
                "smooth_scrolling": object(),
            },
        }
    )

    assert defaults == SettingsAppearanceDefaults()


def test_validate_appearance_defaults_accepts_valid_values():
    result = validate_appearance_defaults(
        SettingsAppearanceDefaults(
            default_theme="textual-light",
            palette_theme_limit=0,
            font_size=16,
            density="compact",
            animations_enabled=False,
            smooth_scrolling=True,
        )
    )

    assert result.valid is True
    assert "valid" in result.message.lower()


def test_validate_appearance_defaults_accepts_web_runtime_minimum_font_size():
    result = validate_appearance_defaults(SettingsAppearanceDefaults(font_size=6))

    assert result.valid is True


def test_validate_appearance_defaults_rejects_invalid_values():
    invalid_values = (
        ({"default_theme": ""}, "Theme"),
        ({"palette_theme_limit": -1}, "Palette theme limit"),
        ({"palette_theme_limit": 101}, "Palette theme limit"),
        ({"font_size": 5}, "Font size"),
        ({"font_size": 33}, "Font size"),
        ({"density": "spacious"}, "Density"),
        ({"animations_enabled": "yes"}, "Animations"),
        ({"smooth_scrolling": "yes"}, "Smooth scrolling"),
        ({"reduce_motion": "yes"}, "Reduce motion"),
        ({"ascii_glyphs": "yes"}, "ASCII glyphs"),
        ({"console_transcript_style": "rainbow"}, "Transcript style"),
    )

    for overrides, expected_message in invalid_values:
        values = SettingsAppearanceDefaults(
            **{**SettingsAppearanceDefaults().__dict__, **overrides}
        )
        result = validate_appearance_defaults(values)

        assert result.valid is False
        assert expected_message in result.message


def test_build_appearance_save_sections_preserves_unrelated_config():
    sections = build_appearance_save_sections(
        {
            "general": {"default_tab": "settings", "log_level": "INFO"},
            "web_server": {"enabled": True, "port": 8000},
            "appearance": {"accent_color": "#00ffaa"},
            "chat_defaults": {"provider": "openai"},
        },
        SettingsAppearanceDefaults(
            default_theme="textual-light",
            palette_theme_limit=5,
            font_size=14,
            density="comfortable",
            animations_enabled=False,
            smooth_scrolling=False,
        ),
    )

    assert sections == {
        "general": {
            "default_tab": "settings",
            "log_level": "INFO",
            "default_theme": "textual-light",
            "palette_theme_limit": 5,
        },
        "web_server": {
            "enabled": True,
            "port": 8000,
            "font_size": 14,
        },
        "appearance": {
            "accent_color": "#00ffaa",
            "density": "comfortable",
            "animations_enabled": False,
            "smooth_scrolling": False,
            "reduce_motion": False,
            "ascii_glyphs": False,
            "console_transcript_style": "role_accents",
        },
    }


def test_appearance_defaults_public_functions_use_google_style_docstrings():
    for function in (
        load_appearance_defaults,
        validate_appearance_defaults,
        build_appearance_save_sections,
    ):
        doc = inspect.getdoc(function)
        assert doc is not None
        assert "Args:" in doc
        assert "Returns:" in doc


def test_load_appearance_defaults_reduce_motion_defaults_off_and_coerces():
    # Absent key (older config files): animations keep their legacy behavior.
    assert load_appearance_defaults({}).reduce_motion is False

    assert (
        load_appearance_defaults({"appearance": {"reduce_motion": "true"}}).reduce_motion
        is True
    )
    assert (
        load_appearance_defaults({"appearance": {"reduce_motion": 1}}).reduce_motion
        is True
    )
    assert (
        load_appearance_defaults(
            {"appearance": {"reduce_motion": object()}}
        ).reduce_motion
        is False
    )


def test_load_appearance_defaults_ascii_glyphs_defaults_off_and_coerces():
    # Absent key (older config files): unicode glyph set stays the default.
    assert load_appearance_defaults({}).ascii_glyphs is False

    assert (
        load_appearance_defaults({"appearance": {"ascii_glyphs": "true"}}).ascii_glyphs
        is True
    )
    assert (
        load_appearance_defaults({"appearance": {"ascii_glyphs": 1}}).ascii_glyphs
        is True
    )
    assert (
        load_appearance_defaults(
            {"appearance": {"ascii_glyphs": object()}}
        ).ascii_glyphs
        is False
    )
