"""Regression tests: first-run orientation empty-state must not crash markup.

The Console empty state (``#chat-empty-state`` Static) renders
``ChatWindowEnhanced.build_first_run_orientation_text`` with Rich markup
enabled. ``ProviderReadiness.user_message`` embeds the literal TOML config
path ``[api_settings.<provider>]`` -- square brackets that markup parsing
treated as a style tag, crashing first-time rendering with
``textual.markup.MarkupError`` exactly when a first-time user had no API key
configured (or an unknown provider selected).

These tests build the orientation text for both not-ready states and push it
through ``Content.from_markup`` -- the exact parse Static.update() performs.
"""

from types import SimpleNamespace

from textual.content import Content

from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced


def _orientation_text(provider: str, app_config: dict) -> str:
    """Build the orientation text without mounting the full chat window."""
    fake_self = SimpleNamespace(app_instance=SimpleNamespace(app_config=app_config))
    return ChatWindowEnhanced.build_first_run_orientation_text(
        fake_self, provider=provider
    )


def test_orientation_markup_safe_when_api_key_missing():
    """Missing-key recovery mentions [api_settings.openai] literally."""
    text = _orientation_text("OpenAI", {"api_settings": {"openai": {}}})
    assert "[api_settings.openai]" in text or r"\[api_settings.openai]" in text
    content = Content.from_markup(text)  # must not raise MarkupError
    assert "[api_settings.openai]" in content.plain


def test_orientation_markup_safe_for_unknown_provider():
    """Unknown-provider recovery also carries the bracketed TOML path."""
    text = _orientation_text("OpenAI-typo", {"api_settings": {}})
    content = Content.from_markup(text)  # must not raise MarkupError
    assert "[api_settings.openai_typo]" in content.plain


def test_orientation_markup_safe_when_ready():
    """A configured provider keeps rendering (no brackets, still parses)."""
    text = _orientation_text(
        "OpenAI", {"api_settings": {"openai": {"api_key": "sk-test"}}}
    )
    content = Content.from_markup(text)
    assert "OpenAI is ready" in content.plain
