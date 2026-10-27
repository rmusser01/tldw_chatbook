"""TASK-34100.6 slice B: entering an API key on the Provider step.

Review findings gap-04 (a pasted key with an invisible character crashed the
step; one in smart quotes read as 'couldn't reach the server'), provider-17
(no reveal, no echo, no Clear hint), provider-19 (an exported key hid the key
field), provider-23 (a pick did not reach the key field; the Authentication
header was an extra Tab stop) and E11 (a key pasted under the wrong provider).
"""

from __future__ import annotations

import string

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from textual import events
from textual.widgets import Button, Static

from Tests.Wizards.provider_step_harness import (
    SIZE,
    WizardHost,
    current_step,
    key_input,
    pick_provider,
    provider_list,
    resumed_on_provider,
    static_text,
    wait_for_step,
)
from tldw_chatbook.UI.Wizards.first_run_provider_key_entry import (
    clean_api_key,
    paste_echo,
    suggest_provider_for_key,
)
from tldw_chatbook.UI.Wizards.first_run_setup_state import STEP_PROVIDER

pytestmark = pytest.mark.bootstrap_profile

_KEY_ALPHABET = string.ascii_letters + string.digits + "-_."
_INVISIBLES = ("​", "﻿", "‎", "⁠", "\n", "\r", "\t", " ")
_QUOTES = (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"))
_PREFIXES = ("", "Bearer ", "OPENAI_API_KEY=", "export ANTHROPIC_API_KEY=")
_STRIPPED = {"Cf", "Cc", "Zs", "Zl", "Zp"}

keys = st.text(alphabet=_KEY_ALPHABET, min_size=8, max_size=80)


@st.composite
def dirty_keys(draw):
    key = draw(keys)
    prefix = draw(st.sampled_from(_PREFIXES))
    quote = draw(st.one_of(st.none(), st.sampled_from(_QUOTES)))
    body = key if quote is None else f"{quote[0]}{key}{quote[1]}"
    text = list(prefix + body)
    for _ in range(draw(st.integers(min_value=0, max_value=4))):
        position = draw(st.integers(min_value=0, max_value=len(text)))
        text.insert(position, draw(st.sampled_from(_INVISIBLES)))
    return key, "".join(text)


@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
@given(dirty_keys())
def test_cleaning_recovers_the_key_from_any_realistic_paste(case):
    key, pasted = case
    cleaned = clean_api_key(pasted)
    assert cleaned.value == key
    assert cleaned.non_ascii == 0
    assert (cleaned.note == "") == (pasted == key)
    if pasted != key:
        assert cleaned.note.startswith("Removed ")


@given(st.text(max_size=200))
def test_cleaning_is_idempotent_and_leaves_no_invisible_characters(text):
    import unicodedata

    once = clean_api_key(text)
    assert clean_api_key(once.value).value == once.value
    assert not any(unicodedata.category(c) in _STRIPPED for c in once.value)
    assert key_repr_hides_value(once)


def key_repr_hides_value(cleaned) -> bool:
    return len(cleaned.value) < 6 or cleaned.value not in repr(cleaned)


def test_the_note_names_what_was_removed():
    cleaned = clean_api_key("“sk-abc123​def”")
    assert cleaned.value == "sk-abc123def"
    assert cleaned.note == (
        "Removed 1 invisible character and surrounding quotes from the pasted key."
    )
    assert "Bearer" in clean_api_key("Bearer sk-abc123def").note
    assert "OPENAI_API_KEY=" in clean_api_key("OPENAI_API_KEY=sk-abc123def").note


def test_the_paste_echo_shows_only_length_and_last_four():
    assert paste_echo("sk-" + "x" * 44 + "abcd") == (
        "Pasted 51 characters, ending …abcd."
    )
    assert paste_echo("short") == "Pasted 5 characters."


@pytest.mark.parametrize(
    ("value", "selected", "expected"),
    [
        ("sk-ant-api03-xyz", "openai", "anthropic"),
        ("sk-or-v1-xyz", "openai", "openrouter"),
        ("gsk_xyz", "openai", "groq"),
        ("AIzaSyXYZ", "openai", "google"),
        ("sk-proj-xyz", "anthropic", "openai"),
        ("sk-proj-xyz", "openai", ""),
        ("sk-plain", "anthropic", ""),
    ],
)
def test_a_known_prefix_suggests_its_provider(value, selected, expected):
    assert suggest_provider_for_key(value, selected) == expected


async def _mounted_on_openai(pilot, wizard):
    await wait_for_step(pilot, wizard, STEP_PROVIDER)
    _container, provider = current_step(wizard)
    await pick_provider(pilot, provider, "openai")
    field = key_input(provider)
    field.focus()
    await pilot.pause()
    return provider, field


@pytest.mark.asyncio
async def test_a_bracketed_paste_is_cleaned_whole_and_echoed(monkeypatch):
    """gap-04 + provider-17: the Paste event the terminal sends, end to end."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        provider, field = await _mounted_on_openai(pilot, wizard)
        app.post_message(events.Paste("“sk-abc123​def\nghij1234”"))
        await pilot.pause()
        assert field.value == "sk-abc123defghij1234"
        note = static_text(provider, "#setup-provider-key-note")
        assert "Pasted 20 characters, ending …1234." in note
        assert "invisible character" in note and "quotes" in note
        provider_list(provider).focus()  # blur clears the echo
        await pilot.pause()
        assert static_text(provider, "#setup-provider-key-note") == ""


@pytest.mark.asyncio
async def test_a_zero_width_key_never_crashes_the_step(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        provider, field = await _mounted_on_openai(pilot, wizard)
        field.value = "sk-test-key-000000000000000000000000000000​"
        await pilot.pause()
        assert field.value == "sk-test-key-000000000000000000000000000000"
        ok, error = await provider.commit()
        assert ok, error
        assert app.is_running and app._exception is None


@pytest.mark.asyncio
async def test_non_ascii_left_in_a_key_is_a_field_message_not_a_crash(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        provider, field = await _mounted_on_openai(pilot, wizard)
        app.post_message(events.Paste("sk-abcdé1234567890"))
        await pilot.pause()
        note = provider.query_one("#setup-provider-key-note", Static)
        assert note.has_class("-error")
        assert "1 character that API keys never use" in str(note.render())
        ok, error = await provider.commit()
        assert not ok
        assert "API keys never use" in error
        assert "reach" not in error
        assert wizard.query_one("SetupWizardContainer").staged_provider_draft is None
        assert app.is_running and app._exception is None


@pytest.mark.asyncio
async def test_show_reveals_the_key_and_the_help_line_names_ctrl_u(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        provider, field = await _mounted_on_openai(pilot, wizard)
        assert field.password
        assert "Ctrl+U" in static_text(provider, "#setup-provider-key-status")
        await pilot.click("#setup-provider-key-reveal")
        await pilot.pause()
        assert not field.password
        assert str(provider.query_one("#setup-provider-key-reveal", Button).label) == (
            "Hide"
        )


@pytest.mark.asyncio
async def test_a_key_pasted_under_the_wrong_provider_offers_the_switch(monkeypatch):
    """E11: the prefix suggests its provider; the switch carries the key along."""
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        provider, field = await _mounted_on_openai(pilot, wizard)
        app.post_message(events.Paste("sk-ant-api03-abcdefghijklmnop"))
        await pilot.pause()
        hint = static_text(provider, "#setup-provider-key-hint-text")
        assert hint == "This looks like an Anthropic key — switch to Anthropic?"
        await pilot.click("#setup-provider-key-switch")
        await pilot.pause()
        assert provider.selected_provider_key == "anthropic"
        assert key_input(provider).value == "sk-ant-api03-abcdefghijklmnop"
        await pick_provider(pilot, provider, "openai")
        assert key_input(provider).value == ""


@pytest.mark.asyncio
async def test_an_explicit_pick_moves_focus_to_the_key_but_arrows_do_not(monkeypatch):
    """provider-23: Enter/Space/click on a keyed row lands in its key field."""
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        await pick_provider(pilot, provider, "openai")
        await pilot.pause()
        assert app.focused is key_input(provider)
        # The required section's header is not an extra Tab stop.
        titles = list(provider.query("#setup-provider-auth-toggle CollapsibleTitle"))
        assert titles and all(t not in app.screen.focus_chain for t in titles)
        choices = provider_list(provider)
        choices.focus()
        await pilot.pause()
        await pilot.press("down")  # arrow-highlight onto Anthropic
        await pilot.pause()
        assert provider.selected_provider_key == "anthropic"
        assert app.focused is choices


@pytest.mark.asyncio
async def test_an_exported_key_offers_use_it_or_store_a_different_one(monkeypatch):
    """provider-19: the env key is the default; 'Store a different key' types one."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-0000000000000000000000000000")
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        await pick_provider(pilot, provider, "openai")
        env = provider.query_one("#setup-provider-key-source-env")
        assert provider.query_one("#setup-provider-key-source").display
        # Live F12: the label fits 120 columns; the shell rule is on the
        # wrapped line under the key, where it is never truncated.
        assert str(env.label) == (
            "Use OPENAI_API_KEY from your environment (found; not stored)"
        )
        assert "shell that starts chatbook" in static_text(
            provider, "#setup-provider-key-status"
        )
        assert env.value and not key_input(provider).display
        await pilot.click("#setup-provider-key-source-store")
        await pilot.pause()
        field = key_input(provider)
        assert field.display
        field.value = "sk-app-only-0000000000000000000000000"
        await pilot.pause()
        ok, error = await provider.commit()
        assert ok, error
        staged = wizard.query_one("SetupWizardContainer").staged_provider_draft
        assert staged.credential.source == "draft"
        # TE-11(b): what is staged is the typed key, never the environment's.
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            _credential_value_for_boundary,
        )

        staged_value = _credential_value_for_boundary(staged.credential)
        assert staged_value == "sk-app-only-0000000000000000000000000"
        assert staged_value != "sk-env-0000000000000000000000000000"


@pytest.mark.asyncio
async def test_the_env_choice_label_is_whole_at_120_columns(monkeypatch):
    """Live F12: at 120 columns the default option read 'Use OPENAI_API_KEY
    from your environment (found; not stored — it must be set in the shell
    that…', hiding the instruction AC#21 requires."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-0000000000000000000000000000")
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        await pick_provider(pilot, provider, "openai")
        env = provider.query_one("#setup-provider-key-source-env")
        env.scroll_visible(animate=False)
        await pilot.pause(0.2)
        label = str(env.label)
        # The button's content width holds the whole label (no ellipsis).
        assert len(label) + 4 <= env.content_region.width, (label, env.content_region)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (80, 24)], ids=["120x40", "80x24"])
async def test_the_env_choice_label_is_never_cut_short(monkeypatch, size):
    """Pass 2 live (g6b-env80/02): at 80x24, a supported size, the default
    option read 'Use ANTHROPIC_API_KEY from your environment (found; no…' --
    the '(found; not stored)' AC#21 requires was ellipsized. The option wraps
    instead of clipping."""
    from textual.geometry import Region

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-00000000000000000000000000")
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=size) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        await pick_provider(pilot, provider, "anthropic")
        env = provider.query_one("#setup-provider-key-source-env")
        env.scroll_visible(animate=False)
        await pilot.pause(0.2)
        rows = env.render_lines(Region(0, 0, env.size.width, env.size.height))
        shown = " ".join(" ".join(strip.text.split()) for strip in rows)
        assert "…" not in shown, shown
        assert "(found; not stored)" in shown, shown


@pytest.mark.asyncio
async def test_a_typed_key_cleaned_on_next_says_what_was_removed(monkeypatch):
    """Review round 1, F10: Next cleaned a typed 'Bearer …' key and at once
    cleared the note that said so, so the cleaning was silent; the note also
    said 'pasted' for a typed key. It matters when Next stays (here: Azure
    with no resource URL); leaving the step blurs the field, which clears it."""
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        await pick_provider(pilot, provider, "azure")
        field = key_input(provider)
        field.focus()
        field.value = "Bearer azure-key-" + "a" * 30
        await pilot.pause()
        await pilot.press("ctrl+n")
        await pilot.pause()
        assert provider.selected_provider_key == "azure"  # refused: no URL
        assert field.value == "azure-key-" + "a" * 30
        note = static_text(provider, "#setup-provider-key-note")
        assert note == "Removed a 'Bearer ' prefix from the key."


@pytest.mark.asyncio
async def test_the_paste_echo_clears_on_next_and_is_never_logged(monkeypatch):
    """AC#23; review round 1, TE-03: only the blur clear was tested. The
    echo (length and last four) clears on a refused Next and on one that
    advances, and no log line carries it or the key's tail."""
    from loguru import logger

    from tldw_chatbook.UI.Wizards.first_run_setup_state import STEP_MODEL

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    logged: list[str] = []
    sink = logger.add(lambda message: logged.append(str(message)), level="TRACE")
    tail = "Q7zW"
    try:
        wizard = resumed_on_provider()
        app = WizardHost(wizard)
        async with app.run_test(size=SIZE) as pilot:
            await wait_for_step(pilot, wizard, STEP_PROVIDER)
            _container, provider = current_step(wizard)
            # Refused: Azure with no resource URL.
            await pick_provider(pilot, provider, "azure")
            key_input(provider).focus()
            app.post_message(events.Paste("azure-key-0123456789abcdef" + tail))
            await pilot.pause()
            note = "#setup-provider-key-note"
            assert static_text(provider, note).startswith("Pasted 30 characters")
            await pilot.press("ctrl+n")
            await pilot.pause()
            assert provider.selected_provider_key == "azure"  # refused, still here
            assert static_text(provider, note) == ""

            # Advanced: OpenAI, then Back.
            provider_field = key_input(provider)
            await pick_provider(pilot, provider, "openai")
            provider_field.focus()
            app.post_message(events.Paste("sk-proj-" + "b" * 36 + tail))
            await pilot.pause()
            assert tail in static_text(provider, note)
            await pilot.press("ctrl+n")
            await wait_for_step(pilot, wizard, STEP_MODEL)
            await pilot.press("ctrl+b")
            await wait_for_step(pilot, wizard, STEP_PROVIDER)
            assert static_text(provider, note) == ""
    finally:
        logger.remove(sink)
    assert not [line for line in logged if tail in line or "Pasted " in line]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (80, 24)], ids=["120x40", "80x24"])
async def test_a_clicked_keyed_row_brings_its_key_field_on_screen(monkeypatch, size):
    """Review round 1, live F5: at 120x40 with the full list ('Ready on this
    machine' above Popular), a click on OpenAI moved focus to the key input
    but left it below the fold -- the user typed into an invisible field.

    Measured live (temporary logging): a click highlights and picks in one
    go, and the pick's after-refresh focus ran before the expanded
    Authentication section was laid out -- the field's region was still
    empty, so Textual's focus scroll aimed at nothing. Pilot lays out first,
    so the test runs that first scheduled callback at once, as live."""
    from Tests.Wizards.provider_step_harness import click_provider_row, wait_until
    from tldw_chatbook.Chat import local_server_discovery
    from tldw_chatbook.Chat.local_server_discovery import DiscoveredLocalServer

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-" + "a" * 40)
    servers = (
        DiscoveredLocalServer("llama_cpp", "http://127.0.0.1:9099", ("a",)),
        DiscoveredLocalServer(
            "custom", "http://127.0.0.1:1234", ("q",), engine="LM Studio"
        ),
    )

    async def scan(_app_config, **_kwargs):
        return servers

    monkeypatch.setattr(local_server_discovery, "discover_local_servers", scan)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=size) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        choices = provider_list(provider)
        await wait_until(pilot, lambda: choices.get_option_index("found-1") >= 0)
        scheduled = provider.call_after_refresh
        armed = []

        def first_runs_before_layout(callback, *args, **kwargs):
            if armed:
                armed.clear()
                callback(*args, **kwargs)
                return True
            return scheduled(callback, *args, **kwargs)

        monkeypatch.setattr(provider, "call_after_refresh", first_runs_before_layout)
        picked = provider._focus_key_after_pick

        def pick_then_focus():
            armed.append(1)
            picked()

        monkeypatch.setattr(provider, "_focus_key_after_pick", pick_then_focus)
        await click_provider_row(pilot, provider, "provider-openai")
        field = key_input(provider)
        await wait_until(pilot, lambda: field.has_focus)
        await pilot.pause(0.3)
        assert provider.selected_provider_key == "openai"
        assert wizard.can_view_entire(field), (
            f"the key field {field.region} is outside the step viewport {provider.region}"
        )


@pytest.mark.asyncio
async def test_an_explicit_pick_of_an_env_keyed_row_shows_its_choice(monkeypatch):
    """Live F5 (second half): picking a provider whose key is exported, with
    Enter, left the 'Use OPENAI_API_KEY from your environment' choice below
    the fold at 120x40. Focus stays on the list (the key is already there);
    the choice is scrolled on screen."""
    from Tests.Wizards.provider_step_harness import wait_until
    from tldw_chatbook.Chat import local_server_discovery
    from tldw_chatbook.Chat.local_server_discovery import DiscoveredLocalServer

    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-0000000000000000000000000000")
    servers = (DiscoveredLocalServer("llama_cpp", "http://127.0.0.1:9099", ("a",)),)

    async def scan(_app_config, **_kwargs):
        return servers

    monkeypatch.setattr(local_server_discovery, "discover_local_servers", scan)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        choices = provider_list(provider)
        await wait_until(pilot, lambda: choices.get_option_index("found-0") >= 0)
        await pick_provider(pilot, provider, "openai")
        await pilot.pause(0.3)
        choice = provider.query_one("#setup-provider-key-source")
        assert choice.display
        assert app.focused is choices
        assert wizard.can_view_entire(choice), (choice.region, provider.region)


@pytest.mark.asyncio
async def test_a_switch_that_replaces_a_typed_draft_says_so(monkeypatch):
    """Review round 1, live F14: 'Switch to Anthropic' moved the pasted key
    over a key already typed for Anthropic without notice, and Keep current /
    Replace / Clear then showed with unclear referents."""
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        await pick_provider(pilot, provider, "anthropic")
        key_input(provider).value = "sk-ant-api03-an-earlier-draft"
        await pilot.pause()
        await pick_provider(pilot, provider, "openai")
        key_input(provider).focus()
        app.post_message(events.Paste("sk-ant-api03-the-pasted-one"))
        await pilot.pause()
        await pilot.click("#setup-provider-key-switch")
        await pilot.pause()
        assert provider.selected_provider_key == "anthropic"
        assert key_input(provider).value == "sk-ant-api03-the-pasted-one"
        assert static_text(provider, "#setup-provider-key-note") == (
            "Moved the key to Anthropic, replacing the key you had typed there."
        )
        assert not provider.query_one("#setup-provider-key-actions").display or (
            provider.query_one("#setup-provider-key-actions").has_class("hidden")
        ), "no Keep/Replace/Clear for a typed key"


@pytest.mark.asyncio
async def test_switching_between_exported_keys_keeps_one_source_chosen(monkeypatch):
    """Review round 2, G6-VR2-04: after 'Store a different key' on OpenAI, a
    switch to Anthropic (also exported) showed both radios on, the click on
    'Store a different key' did nothing, and Enter advanced. The sync wrote
    both buttons' values directly; RadioSet reverts a pressed button turned
    off that way."""
    from textual.widgets import RadioButton

    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-0000000000000000000000000000")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-00000000000000000000000000")
    wizard = resumed_on_provider()
    app = WizardHost(wizard)

    def pressed(provider) -> list[str]:
        choice = provider.query_one("#setup-provider-key-source")
        return [str(b.id) for b in choice.query(RadioButton) if b.value]

    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        await pick_provider(pilot, provider, "openai")
        await pilot.click("#setup-provider-key-source-store")
        await pilot.pause()
        assert pressed(provider) == ["setup-provider-key-source-store"]
        await pick_provider(pilot, provider, "anthropic")
        await pilot.pause(0.2)
        assert pressed(provider) == ["setup-provider-key-source-env"]
        assert not key_input(provider).display
        await pilot.click("#setup-provider-key-source-store")
        await pilot.pause(0.2)
        assert pressed(provider) == ["setup-provider-key-source-store"]
        assert key_input(provider).display, "'Store a different key' revealed nothing"
        await pick_provider(pilot, provider, "openai")
        await pilot.pause(0.2)
        assert pressed(provider) == ["setup-provider-key-source-store"]
        choice = provider.query_one("#setup-provider-key-source")
        assert choice.pressed_button is provider.query_one("#setup-provider-key-source-store")


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [SIZE, (120, 40)], ids=["160x52", "120x40"])
async def test_a_switch_shows_the_provider_it_moved_to(monkeypatch, size):
    """Review round 2, G6-VR2-08: after a filtered pick ('openai', Enter),
    'Switch to Anthropic' changed the provider but the list kept the
    'openai' filter (no Anthropic row) and OpenAI highlighted, so nothing on
    screen named Anthropic until Check key did."""
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=size) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        provider.query_one("#setup-provider-filter").focus()
        await pilot.press(*"openai")
        await pilot.press("enter")
        await pilot.pause()
        assert provider.selected_provider_key == "openai"
        key_input(provider).focus()
        app.post_message(events.Paste("sk-ant-api03-" + "z" * 40))
        await pilot.pause()
        await pilot.click("#setup-provider-key-switch")
        await pilot.pause(0.3)
        assert provider.selected_provider_key == "anthropic"
        assert provider.query_one("#setup-provider-filter").value == ""
        assert provider_list(provider).highlighted_option.id == "provider-anthropic"
        assert static_text(provider, "#setup-provider-key-note").startswith(
            "Moved the key to Anthropic."
        )
        # Live (g6-r2fix/06): the cleared filter grew the list and pushed the
        # focused key field below the fold.
        field = key_input(provider)
        assert app.focused is field
        assert wizard.can_view_entire(field), (field.region, provider.region)


@pytest.mark.asyncio
async def test_ctrl_u_clears_the_whole_key_wherever_the_cursor_is(monkeypatch):
    """Review round 2, G6-VR2-10: the help line says 'Ctrl+U clears the
    field', but Textual's Ctrl+U deletes only what is left of the cursor. A
    click into the masked field puts the cursor mid-key, so Ctrl+U then a
    paste produced a 239-character mixed key the user could not see."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        provider, field = await _mounted_on_openai(pilot, wizard)
        field.value = "sk-proj-" + "m" * 40
        field.cursor_position = 12  # a click lands mid-key
        await pilot.pause()
        await pilot.press("ctrl+u")
        await pilot.pause()
        assert field.value == ""
        assert "Ctrl+U clears the field" in static_text(provider, "#setup-provider-key-status")
