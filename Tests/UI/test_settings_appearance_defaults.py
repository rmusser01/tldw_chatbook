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
    assert defaults.library_reader_library_width == 31


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
        "library": {
            "reader": {
                "library_open": True,
                "custom_widths_enabled": False,
                "library_width": 31,
            },
            "media_reader": {
                "items_open": True,
                "items_width": 40,
            },
            "conversations_reader": {"items_open": True, "items_width": 40},
            "notes_reader": {"items_open": True, "items_width": 40},
            "prompts_reader": {"items_open": True, "items_width": 40},
            "skills_reader": {"items_open": True, "items_width": 40},
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


def test_load_appearance_defaults_reads_shared_and_destination_preferences():
    defaults = load_appearance_defaults(
        {
            "library": {
                "reader": {
                    "library_open": "false",
                    "custom_widths_enabled": "yes",
                    "library_width": "36",
                },
                "media_reader": {
                    "items_open": True,
                    "items_width": 54,
                },
                "conversations_reader": {"items_open": False, "items_width": 48},
                "notes_reader": {"items_open": True, "items_width": 52},
                "prompts_reader": {"items_open": False, "items_width": 60},
                "skills_reader": {"items_open": True, "items_width": 68},
            }
        }
    )

    assert defaults.library_reader_library_open is False
    assert defaults.library_reader_custom_widths_enabled is True
    assert defaults.library_reader_library_width == 36
    assert defaults.library_media_items_open is True
    assert defaults.library_media_items_width == 54
    assert defaults.library_conversations_items_open is False
    assert defaults.library_conversations_items_width == 48
    assert defaults.library_notes_items_open is True
    assert defaults.library_notes_items_width == 52
    assert defaults.library_prompts_items_open is False
    assert defaults.library_prompts_items_width == 60
    assert defaults.library_skills_items_open is True
    assert defaults.library_skills_items_width == 68


def test_load_appearance_defaults_falls_back_to_legacy_media_per_shared_key():
    defaults = load_appearance_defaults(
        {
            "library": {
                "reader": {"library_open": False},
                "media_reader": {
                    "library_open": True,
                    "items_open": "not-bool",
                    "custom_widths_enabled": True,
                    "library_width": -500,
                    "items_width": 500,
                }
            }
        }
    )

    assert defaults.library_reader_library_open is False
    assert defaults.library_reader_custom_widths_enabled is True
    assert defaults.library_reader_library_width == 24
    assert defaults.library_media_items_open is True
    assert defaults.library_media_items_width == 72


def test_custom_width_toggle_keeps_saved_width_dormant_and_unchanged():
    dormant = load_appearance_defaults(
        {
            "library": {
                "reader": {
                    "custom_widths_enabled": False,
                    "library_width": 48,
                }
            }
        }
    )
    enabled = load_appearance_defaults(
        {
            "library": {
                "reader": {
                    "custom_widths_enabled": True,
                    "library_width": dormant.library_reader_library_width,
                }
            }
        }
    )

    assert dormant.library_reader_custom_widths_enabled is False
    assert dormant.library_reader_library_width == 48
    assert enabled.library_reader_custom_widths_enabled is True
    assert enabled.library_reader_library_width == 48


def test_validate_appearance_defaults_keeps_custom_range_24_through_48():
    for width in (24, 34, 35, 48):
        result = validate_appearance_defaults(
            SettingsAppearanceDefaults(library_reader_library_width=width)
        )

        assert result.valid is True


def test_validate_appearance_defaults_rejects_media_reader_types_and_widths():
    invalid_values = (
        ({"library_reader_library_open": "yes"}, "Library pane"),
        ({"library_media_items_open": 1}, "Items pane"),
        ({"library_reader_custom_widths_enabled": "yes"}, "Custom widths"),
        ({"library_reader_library_width": 23}, "Library width"),
        ({"library_reader_library_width": 49}, "Library width"),
        ({"library_media_items_width": 31}, "Items width"),
        ({"library_media_items_width": 73}, "Items width"),
        ({"library_notes_items_open": 1}, "Notes Items pane"),
        ({"library_skills_items_width": 73}, "Skills Items width"),
    )

    for overrides, expected_message in invalid_values:
        values = SettingsAppearanceDefaults(
            **{**SettingsAppearanceDefaults().__dict__, **overrides}
        )
        result = validate_appearance_defaults(values)

        assert result.valid is False
        assert expected_message in result.message


def test_build_appearance_save_sections_deep_merges_shared_and_destinations():
    sections = build_appearance_save_sections(
        {
            "library": {
                "search": {"history": ["alpha"]},
                "reader": {"future_shared": "preserved"},
                "media_reader": {
                    "future_media": "preserved",
                    "library_open": False,
                },
                "notes_reader": {"future_notes": "preserved"},
            }
        },
        SettingsAppearanceDefaults(
            library_reader_library_open=False,
            library_reader_custom_widths_enabled=True,
            library_reader_library_width=32,
            library_media_items_open=True,
            library_media_items_width=56,
            library_conversations_items_open=False,
            library_conversations_items_width=48,
            library_notes_items_open=True,
            library_notes_items_width=52,
            library_prompts_items_open=False,
            library_prompts_items_width=60,
            library_skills_items_open=True,
            library_skills_items_width=68,
        ),
    )

    assert sections["library"] == {
        "search": {"history": ["alpha"]},
        "reader": {
            "future_shared": "preserved",
            "library_open": False,
            "custom_widths_enabled": True,
            "library_width": 32,
        },
        "media_reader": {
            "future_media": "preserved",
            "library_open": False,
            "items_open": True,
            "items_width": 56,
        },
        "conversations_reader": {"items_open": False, "items_width": 48},
        "notes_reader": {
            "future_notes": "preserved",
            "items_open": True,
            "items_width": 52,
        },
        "prompts_reader": {"items_open": False, "items_width": 60},
        "skills_reader": {"items_open": True, "items_width": 68},
    }
