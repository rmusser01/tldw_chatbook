"""Tests for console_generate_image pure helpers."""

import dataclasses

import pytest
from tldw_chatbook.Chat.console_generate_image import (
    clamp_initial_batch,
    compose_styled_request,
    parse_generate_image_args,
    generation_content_marker,
    resolve_style_token,
    run_generation_batch,
)
from tldw_chatbook.Media_Creation.generation_templates import get_template


@pytest.mark.parametrize(
    "args,backend,style,prompt",
    [
        ("a red dragon", None, None, "a red dragon"),
        (":swarmui a red dragon", "swarmui", None, "a red dragon"),
        (":openrouter   spaced  prompt ", "openrouter", None, "spaced  prompt"),
        (":swarmui", "swarmui", None, ""),
        ("", None, None, ""),
        ("   ", None, None, ""),
        (": lonely colon", None, None, ": lonely colon"),  # bare ':' is not a backend token
        ("@anime dragon", None, "anime", "dragon"),
        (":swarmui @anime dragon", "swarmui", "anime", "dragon"),
        ("@anime :swarmui dragon", "swarmui", "anime", "dragon"),
        ("@ lonely at", None, None, "@ lonely at"),  # bare '@' is not a style token
        ("@unknown x", None, "unknown", "x"),  # unresolved style token passes through raw
    ],
)
def test_parse_table(args, backend, style, prompt):
    """Test parse_generate_image_args against spec table."""
    parsed = parse_generate_image_args(args)
    assert (parsed.backend, parsed.style, parsed.prompt) == (backend, style, prompt)


def test_resolve_style_exact_id():
    """An exact (case-insensitive) template id resolves directly."""
    res = resolve_style_token("STYLE_anime")
    assert res.template is not None
    assert res.template.id == "style_anime"
    assert res.ambiguous == ()


def test_resolve_style_exact_name_spaces_and_underscores():
    """Exact name matching treats spaces and underscores as interchangeable."""
    assert resolve_style_token("anime style").template.id == "style_anime"
    assert resolve_style_token("Anime_Style").template.id == "style_anime"


def test_resolve_style_unique_prefix():
    """A prefix unique across ids+names resolves to the one match."""
    res = resolve_style_token("waterc")
    assert res.template is not None
    assert res.template.id == "style_watercolor"
    assert res.ambiguous == ()


def test_resolve_style_ambiguous_prefix_lists_matched_ids():
    """A prefix matching several templates is ambiguous and lists the ids."""
    res = resolve_style_token("style_")
    assert res.template is None
    assert res.ambiguous == ("style_anime", "style_cyberpunk", "style_watercolor")


def test_resolve_style_unknown_token():
    """A token matching nothing resolves to no template and no ambiguity."""
    res = resolve_style_token("totally_not_a_style")
    assert res.template is None
    assert res.ambiguous == ()


def test_compose_styled_request_substitutes_placeholder():
    """The user prompt fills every context-mapped placeholder."""
    template = get_template("quick_simple")
    prompt, negative, params = compose_styled_request("a red dragon", template)
    assert prompt == "a red dragon"
    assert negative == "low quality"
    assert params == {"width": 512, "height": 512, "steps": 15, "cfg_scale": 7.0}


def test_compose_styled_request_appends_when_prompt_unconsumed():
    """When the template doesn't consume the prompt, it is appended (invariant)."""
    base_template = get_template("style_anime")
    template = dataclasses.replace(base_template, context_mappings={})
    prompt, negative, params = compose_styled_request("a dragon", template)
    assert prompt == (
        "anime style, detailed, vibrant colors, high quality anime art, a dragon"
    )
    assert prompt.endswith("a dragon")
    assert negative == base_template.negative_prompt
    assert params == base_template.default_params


@pytest.mark.parametrize(
    "default_batch,max_variants,expected",
    [
        (4, 8, 4),  # default is smaller
        (12, 8, 8),  # cap is smaller
        (5, 5, 5),  # equal
        (1, 1, 1),  # both minimum
    ],
)
def test_clamp_initial_batch(default_batch, max_variants, expected):
    """Test clamp_initial_batch returns minimum of the two values."""
    result = clamp_initial_batch(default_batch, max_variants)
    assert result == expected


def test_content_marker_no_trim():
    """Test generation_content_marker with short prompt."""
    assert generation_content_marker("a red dragon") == "[image] a red dragon"


def test_content_marker_trims():
    """Test generation_content_marker with long prompt."""
    long = "x" * 200
    marker = generation_content_marker(long)
    assert marker.startswith("[image] ")
    assert len(marker) <= 8 + 80 + 1  # [image] (8) + limit (80) + ellipsis (1)
    assert marker.endswith("…")


def test_content_marker_default_limit():
    """Test that default limit is 80."""
    # Prompt of exactly 81 chars should be trimmed
    prompt = "a" * 81
    marker = generation_content_marker(prompt)
    assert marker.endswith("…")


def test_content_marker_custom_limit():
    """Test generation_content_marker with custom limit."""
    prompt = "a" * 50
    marker = generation_content_marker(prompt, limit=40)
    assert marker == "[image] " + "a" * 39 + "…"


def test_content_marker_whitespace_normalize():
    """Test that newlines are collapsed to spaces."""
    prompt = "a\n\nb\n\nc"
    marker = generation_content_marker(prompt)
    assert marker == "[image] a b c"


class _Res:
    """Minimal ImageGenResult-shaped fake for run_generation_batch tests."""

    def __init__(self, b):
        self.content = b
        self.content_type = "image/png"
        self.bytes_len = len(b)


def test_batch_all_succeed():
    """All variants generate successfully -> no errors, N successes."""
    calls = []

    def gen(req):
        calls.append(req)
        return _Res(b"img")

    out = run_generation_batch(
        backend="swarmui", prompt="p", negative_prompt=None,
        seed=None, count=2, generate=gen,
    )
    assert len(out.successes) == 2 and out.errors == []


def test_batch_partial_failure_keeps_successes():
    """One variant raises -> the rest still succeed and are kept."""
    n = {"i": 0}

    def gen(req):
        n["i"] += 1
        if n["i"] == 2:
            raise RuntimeError("boom")
        return _Res(b"img")

    out = run_generation_batch(
        backend="swarmui", prompt="p", negative_prompt=None,
        seed=None, count=3, generate=gen,
    )
    assert len(out.successes) == 2 and len(out.errors) == 1


def test_batch_explicit_seed_only_first_variant():
    """An explicit seed only applies to variant 0; later variants force -1."""
    seeds = []

    def gen(req):
        seeds.append(req.seed)
        return _Res(b"img")

    run_generation_batch(
        backend="swarmui", prompt="p", negative_prompt=None,
        seed=1234, count=3, generate=gen,
    )
    assert seeds == [1234, -1, -1]  # identical-image guard


def test_batch_build_exception_collected():
    """When build raises on a variant, the error is collected and batch continues."""
    n = {"i": 0}

    def build_fn(*args, **kwargs):
        n["i"] += 1
        if n["i"] == 2:
            raise RuntimeError("build boom")
        return {"seed": kwargs.get("seed")}

    def gen(req):
        return _Res(b"img")

    out = run_generation_batch(
        backend="swarmui", prompt="p", negative_prompt=None,
        seed=None, count=3, generate=gen, build=build_fn,
    )
    # First and third variants succeed (build calls 1 and 3).
    # Second fails during build (call 2).
    assert len(out.successes) == 2 and len(out.errors) == 1
    assert "build boom" in out.errors[0]


def test_batch_threads_style_and_template_params():
    """style_name/width/height/steps/cfg_scale reach build() and the variant meta."""
    captured = []

    def build_fn(**kwargs):
        captured.append(kwargs)
        return {"seed": kwargs.get("seed")}

    def gen(req):
        return _Res(b"img")

    template = get_template("style_anime")
    out = run_generation_batch(
        backend="swarmui", prompt="p", negative_prompt=None,
        seed=None, count=1, style_name=template.name,
        width=template.default_params["width"],
        height=template.default_params["height"],
        steps=template.default_params["steps"],
        cfg_scale=template.default_params["cfg_scale"],
        generate=gen, build=build_fn,
    )
    assert captured[0]["width"] == template.default_params["width"]
    assert captured[0]["height"] == template.default_params["height"]
    assert captured[0]["steps"] == template.default_params["steps"]
    assert captured[0]["cfg_scale"] == template.default_params["cfg_scale"]
    assert len(out.successes) == 1
    meta = out.successes[0][2]
    assert meta.style == template.name


def test_batch_default_style_and_dims_are_none():
    """Without style/dims arguments, meta.style is None and build() receives None dims."""
    captured = []

    def build_fn(**kwargs):
        captured.append(kwargs)
        return {"seed": kwargs.get("seed")}

    def gen(req):
        return _Res(b"img")

    out = run_generation_batch(
        backend="swarmui", prompt="p", negative_prompt=None,
        seed=None, count=1, generate=gen, build=build_fn,
    )
    assert captured[0]["width"] is None
    assert captured[0]["height"] is None
    assert captured[0]["steps"] is None
    assert captured[0]["cfg_scale"] is None
    assert out.successes[0][2].style is None
