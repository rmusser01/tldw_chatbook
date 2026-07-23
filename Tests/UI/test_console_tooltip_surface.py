"""Contract tests for the app tooltip surface and static-guidance polish.

Covers Console UX review tasks 382 (tab tooltip garbles the header) and 386
(hover tooltips occlude the content they describe; essential guidance is
hover-only). The framework tooltip is borderless and its ``$panel`` fill reads
as bare text bleeding into adjacent widgets; the app must give tooltips an
opaque, bordered surface so they cover what they overlap instead of interleaving.
"""

import re
from pathlib import Path

import pytest

from Tests.UI.test_settings_configuration_hub import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _open_settings_category,
    _visible_text,
)

ROOT = Path(__file__).resolve().parents[2]
OVERRIDES = ROOT / "tldw_chatbook/css/utilities/_overrides.tcss"
BUNDLE = ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss"


def _tooltip_block(css_text: str) -> str:
    """Return the declaration body of the top-level ``Tooltip { ... }`` rule."""
    match = re.search(r"(?<![\w.#-])Tooltip\s*\{([^}]*)\}", css_text)
    return match.group(1) if match else ""


def test_overrides_give_tooltip_an_opaque_bordered_surface():
    """TASK-382/386-AC#1: the app stylesheet must style ``Tooltip`` with an
    opaque background and a visible border so it fully covers underlying text."""
    body = _tooltip_block(OVERRIDES.read_text(encoding="utf-8"))
    assert body, "no top-level `Tooltip {}` rule in utilities/_overrides.tcss"

    # A border delimits the surface (the framework default is borderless).
    border_match = re.search(r"\bborder\s*:\s*([^;]+);", body)
    assert border_match, "Tooltip rule must declare a border"
    assert border_match.group(1).strip().split()[0] not in ("none", "hidden")

    # An explicit opaque fill (no alpha component) so text underneath is covered.
    bg_match = re.search(r"\bbackground\s*:\s*([^;]+);", body)
    assert bg_match, "Tooltip rule must declare a background"
    bg_value = bg_match.group(1).strip()
    assert bg_value.split()[0] != "transparent"
    # Reject a fractional alpha suffix like `$panel 60%` that would show through.
    assert not re.search(r"\b\d{1,2}%\s*$", bg_value), (
        f"tooltip background '{bg_value}' is translucent; it must be opaque"
    )


def test_generated_bundle_carries_the_tooltip_surface():
    """The rebuilt bundle (what the app actually loads) must include the rule."""
    body = _tooltip_block(BUNDLE.read_text(encoding="utf-8"))
    assert body, "generated bundle is missing the Tooltip surface rule -- rebuild CSS"
    assert re.search(r"\bborder\s*:", body)


@pytest.mark.asyncio
async def test_provider_test_live_probe_guidance_is_static_not_hover_only():
    """TASK-386-AC#2: the readiness/live-probe explanation must exist as visible
    static text, not only inside the Test Provider hover tooltip."""
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama.cpp", "model": ""}
    app.app_config["api_settings"] = {"llama_cpp": {"api_url": "http://localhost:8080"}}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 60)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        await pilot.pause()
        text = _visible_text(screen)

    # The live-probe guidance that was hover-only is now on screen.
    assert "live endpoint probe" in text
