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
