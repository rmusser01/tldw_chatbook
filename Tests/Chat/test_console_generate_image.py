"""Tests for console_generate_image pure helpers."""

import pytest
from tldw_chatbook.Chat.console_generate_image import (
    parse_generate_image_args,
    generation_content_marker,
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
