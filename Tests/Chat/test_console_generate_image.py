"""Tests for console_generate_image pure helpers."""

import pytest
from tldw_chatbook.Chat.console_generate_image import (
    clamp_initial_batch,
    parse_generate_image_args,
    generation_content_marker,
    run_generation_batch,
)


@pytest.mark.parametrize(
    "args,backend,prompt",
    [
        ("a red dragon", None, "a red dragon"),
        (":swarmui a red dragon", "swarmui", "a red dragon"),
        (":openrouter   spaced  prompt ", "openrouter", "spaced  prompt"),
        (":swarmui", "swarmui", ""),
        ("", None, ""),
        ("   ", None, ""),
        (": lonely colon", None, ": lonely colon"),  # bare ':' is not a backend token
    ],
)
def test_parse_table(args, backend, prompt):
    """Test parse_generate_image_args against spec table."""
    parsed = parse_generate_image_args(args)
    assert (parsed.backend, parsed.prompt) == (backend, prompt)


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
