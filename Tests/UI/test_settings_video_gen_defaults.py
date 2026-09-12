"""Video Gen settings data layer (task-3401.12, AC1/AC3)."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Screens import settings_video_gen_defaults as d


def _cfg(**overrides):
    base = {
        "default_backend": "minimax",
        "enabled_backends": ["minimax"],
        "minimax_video_api_key": "sk-test",
        "comfyui_base_url": "http://127.0.0.1:8188",
        "sd_cpp_binary_path": "",
        "key_sources": {"minimax": "env:MINIMAX_API_KEY", "comfyui": "missing", "stable_diffusion_cpp": "missing"},
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_build_backend_rows_status():
    rows = d.build_backend_rows(_cfg())
    by_id = {row.backend_id: row for row in rows}
    assert by_id["minimax"].configured is True
    assert by_id["minimax"].enabled is True
    assert by_id["minimax"].is_default is True
    assert by_id["minimax"].key_source == "env:MINIMAX_API_KEY"
    assert by_id["comfyui"].configured is True
    assert by_id["comfyui"].enabled is False
    assert by_id["stable_diffusion_cpp"].configured is False  # no binary path


def test_effective_placeholder_reads_flat_field():
    cfg = _cfg()
    assert d.effective_placeholder(cfg, "comfyui", "base_url") == "http://127.0.0.1:8188"


def test_comfyui_curated_fields_keep_model_workflow_owned():
    assert [spec.toml_key for spec in d.FIELD_SCHEMA["comfyui"]] == [
        "base_url",
        "default_workflow",
        "timeout_seconds",
    ]


def test_canonical_backend_order():
    assert d.canonical_backend_order(["stable_diffusion_cpp", "minimax", "bogus"]) == [
        "minimax",
        "stable_diffusion_cpp",
    ]
    assert d.canonical_backend_order(None) == []


def test_playback_tool_rows_shape():
    rows = d.playback_tool_rows()
    assert [tool for tool, _ in rows] == ["ffmpeg", "ffplay", "yt-dlp"]
    assert all(isinstance(found, bool) for _, found in rows)


# -- diff_to_sections -----------------------------------------------------------


def test_diff_global_and_backend_fields():
    draft = d.VideoGenDraftValues(
        default_backend="comfyui",
        retention_ttl_hours="48",
        backend_fields={"minimax": {"default_model": "MiniMax-H3", "poll_interval_seconds": "5"}},
    )
    raw = {
        "video_generation": {
            "default_backend": "minimax",
            "minimax": {"default_model": "MiniMax-H3", "poll_interval_seconds": 10},
        }
    }
    sections, deletions = d.diff_to_sections(draft, raw)
    assert sections["video_generation"] == {
        "default_backend": "comfyui",
        "retention_ttl_hours": "48",
    }
    # int-kind field coerced before compare; unchanged model not re-emitted.
    assert sections["video_generation.minimax"] == {"poll_interval_seconds": 5}
    assert deletions == {}


def test_diff_clear_and_empty_become_deletions():
    draft = d.VideoGenDraftValues(
        backend_fields={"minimax": {"base_url": "   "}},
        cleared_fields={"minimax": ["api_key"]},
    )
    raw = {"video_generation": {"minimax": {"base_url": "https://x", "api_key": "sk"}}}
    sections, deletions = d.diff_to_sections(draft, raw)
    assert sections == {}
    assert sorted(deletions["video_generation.minimax"]) == ["api_key", "base_url"]


def test_diff_enabled_backends_normalized_order():
    draft = d.VideoGenDraftValues(
        enabled_backends=["stable_diffusion_cpp", "minimax"]
    )
    raw = {"video_generation": {"enabled_backends": ["minimax", "stable_diffusion_cpp"]}}
    sections, _ = d.diff_to_sections(draft, raw)
    # Same set, different file order: NOT a diff (would otherwise rewrite on
    # every save and never clear the dirty marker).
    assert "video_generation" not in sections
    draft2 = d.VideoGenDraftValues(enabled_backends=["minimax"])
    sections2, _ = d.diff_to_sections(draft2, raw)
    assert sections2["video_generation"]["enabled_backends"] == ["minimax"]


def test_diff_bool_field_coercion():
    draft = d.VideoGenDraftValues(
        backend_fields={"minimax": {"allow_uploads": True}}
    )
    raw = {"video_generation": {"minimax": {"allow_uploads": False}}}
    sections, _ = d.diff_to_sections(draft, raw)
    assert sections["video_generation.minimax"] == {"allow_uploads": True}


# -- validate_draft ------------------------------------------------------------------


def test_validate_default_backend_must_be_enabled():
    draft = d.VideoGenDraftValues(
        default_backend="comfyui", enabled_backends=["minimax"]
    )
    errors, _ = d.validate_draft(draft)
    assert any("default backend" in e for e in errors)


def test_validate_int_minimums_and_retention_choice():
    draft = d.VideoGenDraftValues(retention="forever", max_store_mb="0")
    errors, _ = d.validate_draft(draft)
    assert any("Retention" in e for e in errors)
    assert any("Store cap" in e for e in errors)


def test_validate_disabled_backends_warns_not_blocks():
    draft = d.VideoGenDraftValues(enabled_backends=[], default_backend="minimax")
    errors, warnings = d.validate_draft(draft)
    assert errors == []
    assert warnings


# -- panel compose smoke ----------------------------------------------------------


@pytest.mark.asyncio
async def test_panel_compose_covers_all_sections():
    from textual.screen import Screen
    from textual.widgets import Checkbox, Collapsible, Select, Static

    from tldw_chatbook.Widgets.settings_video_gen_panel import VideoGenSettingsPanel

    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    async with app.run_test() as pilot:
        screen = Screen()
        await app.push_screen(screen)
        panel = VideoGenSettingsPanel(id="settings-videogen-panel")
        await screen.mount(panel)
        await pilot.pause()
        assert panel.query(Select)  # default backend + retention
        assert panel.query(Checkbox)  # enable toggles + confirm cost + allow_uploads
        assert panel.query(Collapsible)  # per-backend editors
        section_texts = [
            str(child.renderable) for child in panel.query(Static)
        ]
        assert any("Backends" in text for text in section_texts)
        assert any("Diagnostics" in text for text in section_texts)
        assert any("ffmpeg" in text for text in section_texts)
        assert any("Style templates" in text for text in section_texts)


# -- fresh-profile dirty state (TASK-23191) ----------------------------------------


@pytest.mark.asyncio
async def test_video_gen_opens_clean_on_a_fresh_profile(monkeypatch):
    """Opening Video Gen with no persisted ``[video_generation]`` is not dirty.

    Regression pin for TASK-23191. Both of the panel's ``Select``s re-post
    ``Changed`` from Textual's ``_on_mount``, and this category diffs
    against the RAW config table, so an unguarded echo of a value that is
    only the effective default reads as an edit. The shipped config
    template writes no ``[video_generation]`` table at all, so every fresh
    profile hits exactly that state and the banner read "Unsaved changes"
    on a page nobody had touched.

    The load patch pins that premise rather than creating it: a fresh
    profile genuinely has no such table, but this suite shares one config
    file with the rest of the run, and a sibling that saves Video Gen
    defaults would otherwise quietly retire the regression this covers.
    """
    from textual.widgets import Select, Static

    from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
    from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId

    from Tests.UI.test_destination_shells import (
        DestinationHarness,
        _active_destination_screen,
        _build_test_app,
        _wait_for_selector,
    )

    real_load = SettingsConfigAdapter.load

    def _load_without_video_generation(self):
        loaded = dict(real_load(self))
        loaded.pop("video_generation", None)
        return loaded

    monkeypatch.setattr(
        SettingsConfigAdapter, "load", _load_without_video_generation
    )

    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        screen._select_category(SettingsCategoryId.VIDEO_GENERATION.value)
        await _wait_for_selector(screen, pilot, "#settings-videogen-panel")
        await pilot.pause()

        assert not screen._video_gen_raw_section(), "premise: no persisted table"

        def _dirty_keys():
            draft = screen._settings_drafts.get(SettingsCategoryId.VIDEO_GENERATION)
            return getattr(draft, "dirty_keys", None)

        def _banner():
            return str(
                screen.query_one("#settings-category-state-banner", Static).renderable
            )

        # AC1: nothing staged, and the banner does not ask for a Save.
        assert not screen._category_has_unsaved_changes(
            SettingsCategoryId.VIDEO_GENERATION
        ), f"fresh profile staged: {_dirty_keys()}"
        assert "Unsaved changes" not in _banner(), _banner()

        # AC2: a real edit -- through the widget, not the staging helper --
        # still turns it dirty...
        retention = screen.query_one("#settings-videogen-retention", Select)
        retention.value = "ttl"
        await pilot.pause()
        assert screen._category_has_unsaved_changes(
            SettingsCategoryId.VIDEO_GENERATION
        ), "a genuine retention edit must stage"
        assert "Unsaved changes" in _banner(), _banner()

        # ...and Revert clears it without the recomposed Selects re-dirtying.
        await screen._handle_video_gen_revert()
        await pilot.pause()
        assert not screen._category_has_unsaved_changes(
            SettingsCategoryId.VIDEO_GENERATION
        ), f"revert left staged: {_dirty_keys()}"
        assert "Unsaved changes" not in _banner(), _banner()
