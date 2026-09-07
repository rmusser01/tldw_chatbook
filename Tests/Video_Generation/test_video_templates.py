"""Video style templates + @style grammar (task-3401.12, AC2)."""

import pytest

from tldw_chatbook.Chat.console_generate_video import parse_generate_video_args
from tldw_chatbook.Video_Generation import video_templates as vt


@pytest.fixture(autouse=True)
def _no_user_styles(monkeypatch):
    monkeypatch.setattr(vt, "_user_style_tables", lambda: {})


# -- templates ------------------------------------------------------------------


def test_builtin_templates_resolve_case_insensitively():
    template = vt.get_video_template("Cinematic")
    assert template is not None and template.id == "cinematic"
    assert "cinematic" in template.prompt_suffix
    assert template.default_params["duration_seconds"] == 6


def test_unknown_style_is_none():
    assert vt.get_video_template("does-not-exist") is None


def test_apply_template_prompt_leads_style_follows():
    template = vt.get_video_template("drone")
    prompt, negative = vt.apply_video_template(template, "a lighthouse on cliffs")
    assert prompt.startswith("a lighthouse on cliffs")
    assert "drone" in prompt or "aerial FPV" in prompt
    assert "flicker" in negative


def test_apply_template_empty_prompt_and_negative():
    template = vt.get_video_template("anime")
    prompt, negative = vt.apply_video_template(template, "  ")
    assert prompt == template.prompt_suffix
    assert negative == template.negative_prompt_suffix


def test_user_styles_overlay_and_override(monkeypatch):
    monkeypatch.setattr(
        vt,
        "_user_style_tables",
        lambda: {
            "noir": {
                "name": "Noir",
                "prompt_suffix": "black and white noir, hard shadows",
                "default_params": {"duration_seconds": 7},
            },
            "cinematic": {  # user override of a builtin id wins
                "prompt_suffix": "USER cinematic override",
            },
            "broken": {"name": "no suffix"},  # missing prompt_suffix: skipped
        },
    )
    templates = vt.get_all_video_templates()
    assert templates["noir"].default_params["duration_seconds"] == 7
    assert templates["cinematic"].prompt_suffix == "USER cinematic override"
    assert "broken" not in templates


# -- grammar ---------------------------------------------------------------------


def test_parse_style_token():
    args = parse_generate_video_args("@cinematic a kite")
    assert args.style == "cinematic"
    assert args.backend is None
    assert args.prompt == "a kite"


def test_parse_backend_and_style_any_order():
    first = parse_generate_video_args(":minimax @drone a kite")
    second = parse_generate_video_args("@drone :minimax a kite")
    for args in (first, second):
        assert args.backend == "minimax"
        assert args.style == "drone"
        assert args.prompt == "a kite"


def test_parse_bare_at_stays_prompt():
    args = parse_generate_video_args("@ a kite")
    assert args.style is None
    assert args.prompt == "@ a kite"


def test_parse_style_without_prompt():
    args = parse_generate_video_args("@cinematic")
    assert args.style == "cinematic"
    assert args.prompt == ""
