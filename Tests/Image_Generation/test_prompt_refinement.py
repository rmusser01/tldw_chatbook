import pytest


@pytest.fixture
def pr():
    from tldw_chatbook.Image_Generation import prompt_refinement as m
    return m


def test_off_returns_prompt_unchanged(pr):
    assert pr.refine_image_prompt("a cat", mode="off") == "a cat"


def test_basic_always_appends_suffix(pr):
    out = pr.refine_image_prompt("a cat", mode="basic")
    assert pr.DEFAULT_QUALITY_SUFFIX in out


def test_auto_skips_when_prompt_already_detailed(pr):
    detailed = "a cat, highly detailed, cinematic lighting, sharp focus, 8k, masterpiece composition"
    assert pr.refine_image_prompt(detailed, mode="auto") == detailed  # has quality cues -> no append


def test_auto_appends_for_short_sparse_prompt(pr):
    out = pr.refine_image_prompt("a cat", mode="auto")
    assert pr.DEFAULT_QUALITY_SUFFIX in out


def test_normalize_mode_aliases(pr):
    assert pr.normalize_prompt_refinement_mode(True) == "basic"
    assert pr.normalize_prompt_refinement_mode(False) == "off"
    assert pr.normalize_prompt_refinement_mode("none") == "off"
    assert pr.normalize_prompt_refinement_mode("garbage") == "auto"  # default
