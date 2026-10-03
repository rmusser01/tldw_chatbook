"""Connection, Request estimate and the name fold into one-row disclosures (TASK-33006.3).

Closed, each disclosure is one row whose summary already carries its useful
value (spec §7 mock (b)): Connection names the endpoint host, where the key
comes from and where to change it; Request estimate shows the estimate; the
name shows the name this chat uses. Every test mounts the real modal under
the production stylesheets and reads what the compositor painted.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.widgets import Button, Collapsible, Input, Static

from Tests.UI.test_console_settings_core_first import (
    FULL_SCREEN_SIZES,
    CoreFirstHarness,
    _modal,
    _open,
    _painted,
    _settings,
)
from tldw_chatbook.Chat.console_provider_support import MODEL_FIELD_LABELS
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Utils.token_counter import ContextWindowResolution
from tldw_chatbook.Widgets.Console.console_settings_field_row import (
    CONNECTION_DISCLOSURE_ID,
    DISCLOSURE_TITLE_CELLS,
    FIELD_ROW_FIELDS,
    INVALID_ENDPOINT_HOST,
    NAME_DISCLOSURE_ID,
    NAME_INPUT_ID,
    REQUEST_ESTIMATE_DISCLOSURE_ID,
    SAMPLING_DISCLOSURE_ID,
    SETTINGS_POINTER,
    connection_summary,
    endpoint_host,
    field_control_id,
    key_source_phrase,
)
from tldw_chatbook.Widgets.Console.console_settings_modal import (
    BASE_URL_ENTRY_HINT_COPY,
    GENERATION_TEST_BUTTON_ID,
    GENERATION_TEST_CONSENT_COPY,
    MODEL_DISCOVER_BUTTON_ID,
)

# Census-gated (scripts/ui_pr_gate_census.txt): Tests/UI/conftest.py imports
# tldw_chatbook.app per test, which fails closed with
# RecoveryRequired("raw_source_selection_changed") under the per-test sandbox.
pytestmark = pytest.mark.bootstrap_profile

#: Every closed disclosure, one row each (owner ruling 2026-10-02): this
#: task's three, and Sampling, which counts the fields it cannot name.
FOLDED = (SAMPLING_DISCLOSURE_ID, CONNECTION_DISCLOSURE_ID, REQUEST_ESTIMATE_DISCLOSURE_ID, NAME_DISCLOSURE_ID)
SECRET = "sk-test-0123456789abcdefXYZ"
LONG_HOST = "a-very-long-internal-inference-hostname.corp.example.com"
LONG_ENV = "MY_COMPANY_LLM_GATEWAY_API_KEY"


def _anthropic_from_env(app: CoreFirstHarness, monkeypatch) -> None:
    app.app_config["api_settings"]["anthropic"] = {}
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)


def _openai_missing(app: CoreFirstHarness, monkeypatch) -> None:
    app.app_config["api_settings"]["openai"] = {}
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _long_registry_entry(app: CoreFirstHarness, monkeypatch) -> None:
    app.app_config["custom_endpoints"]["gateway"] = {
        "display_name": "Gateway",
        "family": "llama_cpp",
        "base_url": f"https://{LONG_HOST}:8443/v1",
        "api_key_env": LONG_ENV,
        "models": ["model-a"],
    }
    monkeypatch.setenv(LONG_ENV, SECRET)


def _together_blank_endpoint(app: CoreFirstHarness, monkeypatch) -> None:
    # An engine preset whose endpoint setting is blank: the send falls back
    # to the registry record's default URL, so the summary names that host.
    app.app_config["api_settings"]["together"] = {"api_base_url": ""}
    monkeypatch.setenv("TOGETHER_API_KEY", SECRET)


#: (setup, chat, expected summary). The summary never names the key itself.
_CONNECTION_CHATS = (
    pytest.param(
        _anthropic_from_env,
        _settings("anthropic", "claude-opus-4-8", temperature=0.7),
        f"Connection · api.anthropic.com · key from env ANTHROPIC_API_KEY · {SETTINGS_POINTER}",
        id="env-key",
    ),
    pytest.param(
        None,
        _settings("openai", "gpt-5", temperature=0.7),
        f"Connection · api.openai.com · key saved · {SETTINGS_POINTER}",
        id="saved-key",
    ),
    pytest.param(
        _openai_missing,
        _settings("openai", "gpt-5", temperature=0.7),
        f"Connection · api.openai.com · key missing · {SETTINGS_POINTER}",
        id="missing-key",
    ),
    pytest.param(
        None,
        _settings(),
        f"Connection · 127.0.0.1:9099 · no key needed · {SETTINGS_POINTER}",
        id="keyless-local",
    ),
    pytest.param(
        None,
        _settings("custom-ep:gpu-box", "model-a", temperature=0.7),
        f"Connection · 192.168.1.9:8080 · no key needed · {SETTINGS_POINTER}",
        id="registry-entry",
    ),
    pytest.param(
        _together_blank_endpoint,
        _settings("together", "model-t", temperature=0.7),
        f"Connection · api.together.xyz · key from env TOGETHER_API_KEY · {SETTINGS_POINTER}",
        id="engine-preset-default",
    ),
)


@pytest.mark.parametrize(
    ("url", "host"),
    [
        ("https://api.anthropic.com/v1", "api.anthropic.com"),
        ("http://user:hunter2@10.0.0.5:8080/v1", "10.0.0.5:8080"),
        ("http://[::1]:8080/v1", "[::1]:8080"),
        ("localhost:11434", "localhost:11434"),
        ("http://box:notaport/v1", "box"),
        ("http://[not-a-valid-url", INVALID_ENDPOINT_HOST),
        (None, ""),
        ("   ", ""),
    ],
)
def test_endpoint_host_shows_host_and_port_never_userinfo(url, host) -> None:
    """AC#1: the summary's host part; credentials written into a URL never
    reach the title."""
    assert endpoint_host(url) == host


def _readiness(**facts):
    values = {
        "subscription_status": None,
        "credential": "present_unverified",
        "credential_source": "none",
        "configuration_issue": None,
    }
    values.update(facts)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("facts", "env_var", "phrase"),
    [
        ({"credential_source": "environment"}, "OPENAI_API_KEY", "key from env OPENAI_API_KEY"),
        ({"credential_source": "environment"}, None, "key from env"),
        ({"credential_source": "stored"}, None, "key saved"),
        # T3 review item 1: a key typed in a draft is not saved yet.
        ({"credential_source": "draft"}, None, "unsaved key"),
        ({"credential": "missing", "configuration_issue": "credential_missing"}, "OPENAI_API_KEY", "key missing"),
        ({"credential": "missing", "configuration_issue": "endpoint_missing"}, None, "key not checked"),
        ({"credential": "not_required"}, None, "no key needed"),
        ({"subscription_status": "ready"}, None, "Claude subscription"),
    ],
)
def test_key_source_phrase_names_where_the_key_comes_from(facts, env_var, phrase) -> None:
    """AC#1: saved, the env var's name, or missing; the phrase is built from
    readiness facts only, so it has no way to carry the key."""
    assert key_source_phrase(_readiness(**facts), env_var) == phrase


def test_connection_summary_shortens_only_the_host_to_stay_one_row() -> None:
    """AC#1/#6: a long host ends in "…" and the key source and the Settings
    pointer survive whole."""
    assert connection_summary("api.openai.com", "key saved") == (
        f"Connection · api.openai.com · key saved · {SETTINGS_POINTER}"
    )
    line = connection_summary("h" * 200, "key from env GATEWAY_KEY")
    assert len(line) == DISCLOSURE_TITLE_CELLS
    assert line.endswith(f"… · key from env GATEWAY_KEY · {SETTINGS_POINTER}")


def _title_line(app, modal, disclosure_id: str) -> str:
    """The painted row of one disclosure's title."""
    return _painted(app.screen)[modal.query_one(f"#{disclosure_id}").region.y]


@pytest.mark.parametrize(("setup", "settings", "summary"), _CONNECTION_CHATS)
@pytest.mark.asyncio
async def test_connection_summarises_host_key_source_and_where_to_change_it(
    setup, settings, summary, monkeypatch
) -> None:
    """AC#1: one row naming the endpoint host, where the key comes from
    (saved, the env var's name, or missing) and where to change it; the key
    itself is never painted."""
    app = CoreFirstHarness()
    if setup is not None:
        setup(app, monkeypatch)
    modal = _modal(app, settings)
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        disclosure = modal.query_one(f"#{CONNECTION_DISCLOSURE_ID}", Collapsible)
        if disclosure.collapsed:
            assert disclosure.region.height == 1
        line = _title_line(app, modal, CONNECTION_DISCLOSURE_ID)
        assert summary in line, line
        assert all(SECRET not in row for row in _painted(app.screen))


@pytest.mark.asyncio
async def test_connection_summary_follows_a_typed_endpoint() -> None:
    """AC#1: typing a base URL repaints the host in the summary at once."""
    app = CoreFirstHarness()
    modal = _modal(app, _settings())
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        modal.query_one(f"#{CONNECTION_DISCLOSURE_ID}", Collapsible).collapsed = False
        for _ in range(3):
            await pilot.pause()
        modal.query_one("#console-settings-base-url", Input).focus()
        await pilot.press("ctrl+a", *"http://10.0.0.5:8080")
        for _ in range(3):
            await pilot.pause()
        line = _title_line(app, modal, CONNECTION_DISCLOSURE_ID)
        assert f"Connection · 10.0.0.5:8080 · no key needed · {SETTINGS_POINTER}" in line, line
        # A half-typed IPv6 host makes urlsplit raise; the title says so and
        # the modal keeps running (it crashed the open modal before the guard).
        await pilot.press("ctrl+a", *"http://[fe80")
        for _ in range(3):
            await pilot.pause()
        assert app.screen is modal
        line = _title_line(app, modal, CONNECTION_DISCLOSURE_ID)
        assert f"Connection · {INVALID_ENDPOINT_HOST} · no key needed" in line, line


@pytest.mark.parametrize("size", FULL_SCREEN_SIZES)
@pytest.mark.parametrize(
    ("setup", "settings"),
    [
        pytest.param(_anthropic_from_env, _settings("anthropic", "claude-opus-4-8", temperature=0.7), id="anthropic"),
        pytest.param(_long_registry_entry, _settings("custom-ep:gateway", "model-a", temperature=0.7), id="long-host-and-env"),
    ],
)
@pytest.mark.asyncio
async def test_every_closed_disclosure_measures_one_row(
    size, setup, settings, monkeypatch
) -> None:
    """AC#6 and the owner ruling of 2026-10-02: under the production
    stylesheets every closed disclosure (Sampling, Connection, Request
    estimate, name) is one row, a long custom host included (it is
    shortened, the rest stays) and Anthropic's seven hidden fields too
    (Sampling counts them)."""
    app = CoreFirstHarness()
    setup(app, monkeypatch)
    modal = _modal(app, settings)
    async with app.run_test(size=size) as pilot:
        await _open(pilot, app, modal)
        for disclosure_id in FOLDED:
            disclosure = modal.query_one(f"#{disclosure_id}", Collapsible)
            assert disclosure.collapsed is True, disclosure_id
            assert disclosure.region.height == 1, (disclosure_id, disclosure.region)
        line = _title_line(app, modal, CONNECTION_DISCLOSURE_ID)
        assert SETTINGS_POINTER in line, line
        if settings.provider == "custom-ep:gateway":
            assert f"key from env {LONG_ENV}" in line and "…" in line, line
            assert LONG_HOST[:20] in line, line


@pytest.mark.asyncio
async def test_expanded_connection_offers_what_it_offered_for_a_url_provider() -> None:
    """AC#2: a URL-based provider shows its Endpoint label with its input,
    Test connection, the paid test behind its consent step, and the
    readiness detail."""
    app = CoreFirstHarness()
    modal = _modal(app, _settings())
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        modal.query_one(f"#{CONNECTION_DISCLOSURE_ID}", Collapsible).collapsed = False
        for _ in range(3):
            await pilot.pause()
        base_url = modal.query_one("#console-settings-base-url", Input)
        label = base_url.parent.children[0]
        assert base_url.display and label.display
        assert str(label.content) == MODEL_FIELD_LABELS["endpoint"]
        assert modal.query_one(f"#{MODEL_DISCOVER_BUTTON_ID}", Button).display
        assert str(modal.query_one("#console-settings-readiness", Static).content)
        generation = modal.query_one(f"#{GENERATION_TEST_BUTTON_ID}", Button)
        assert generation.display
        generation.press()
        for _ in range(3):
            await pilot.pause()
        confirmation = modal.query_one("#console-settings-generation-confirmation")
        assert confirmation.display
        consent = modal.query_one("#console-settings-generation-consent-copy", Static)
        assert str(consent.content) == GENERATION_TEST_CONSENT_COPY


@pytest.mark.parametrize("entries", [True, False], ids=["entries", "no-entries"])
@pytest.mark.asyncio
async def test_expanded_connection_has_no_endpoint_label_without_an_input(entries) -> None:
    """AC#2: a cloud provider takes no base URL, so neither the Endpoint
    label nor an empty row shows; New endpoint… stays while entries exist."""
    app = CoreFirstHarness()
    if not entries:
        app.app_config.pop("custom_endpoints")
    modal = _modal(app, _settings("openai", "gpt-5", temperature=0.7))
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        modal.query_one(f"#{CONNECTION_DISCLOSURE_ID}", Collapsible).collapsed = False
        for _ in range(3):
            await pilot.pause()
        base_url = modal.query_one("#console-settings-base-url", Input)
        row = base_url.parent
        label = row.children[0]
        assert not base_url.display and not label.display
        assert row.display is entries
        if entries:
            new_endpoint = modal.query_one("#console-settings-endpoint-new", Button)
            body = modal.query_one("#console-settings-body")
            body.scroll_to_center(new_endpoint, animate=False, immediate=True)
            for _ in range(3):
                await pilot.pause()
            line = _painted(app.screen)[new_endpoint.region.y]
            assert "New endpoint" in line
            assert MODEL_FIELD_LABELS["endpoint"] not in line, line


@pytest.mark.parametrize(
    ("setup", "settings", "recovery"),
    [
        pytest.param(
            _openai_missing,
            _settings("openai", "gpt-5", temperature=0.7),
            "#console-settings-configure-credential",
            id="missing-key",
        ),
        pytest.param(
            None,
            ConsoleSessionSettings(
                provider="llama_cpp",
                model="model-a",
                base_url="ftp://127.0.0.1:9099",
                temperature=0.4,
                max_tokens=2048,
            ),
            "#console-settings-base-url",
            id="invalid-endpoint",
        ),
    ],
)
@pytest.mark.parametrize("size", FULL_SCREEN_SIZES)
@pytest.mark.asyncio
async def test_not_ready_chat_opens_connection_on_its_recovery_action(
    size, setup, settings, recovery, monkeypatch
) -> None:
    """AC#3: a Not-ready chat opens with Connection expanded and focus on
    its recovery action, painted in view; no tuning field has focus."""
    app = CoreFirstHarness()
    if setup is not None:
        setup(app, monkeypatch)
    modal = _modal(app, settings)
    async with app.run_test(size=size) as pilot:
        await _open(pilot, app, modal)
        assert modal.query_one(f"#{CONNECTION_DISCLOSURE_ID}", Collapsible).collapsed is False
        control = modal.query_one(recovery)
        assert app.focused is control
        assert not any(
            app.focused is modal.query_one(f"#{field_control_id(name)}")
            for name in FIELD_ROW_FIELDS
        )
        body = modal.query_one("#console-settings-body")
        assert body.region.contains_region(control.region), (control.region, body.region)


@pytest.mark.asyncio
async def test_request_estimate_summary_shows_the_estimate_and_follows_it() -> None:
    """AC#5: closed, Request estimate shows the current estimate, and a
    resolved context window repaints it."""

    async def resolver(_draft) -> ContextWindowResolution:
        return ContextWindowResolution(200_000, "model metadata", True)

    app = CoreFirstHarness()
    modal = _modal(app, _settings())
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        assert "Request estimate · 10 / 4k tokens" in _title_line(app, modal, REQUEST_ESTIMATE_DISCLOSURE_ID)

    app = CoreFirstHarness()
    modal = _modal(app, _settings(), context_window_resolver=resolver)
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        for _ in range(4):
            await pilot.pause()
        line = _title_line(app, modal, REQUEST_ESTIMATE_DISCLOSURE_ID)
        assert "Request estimate · 10 / 200,000 tokens" in line, line
        assert modal.query_one(f"#{REQUEST_ESTIMATE_DISCLOSURE_ID}", Collapsible).region.height == 1


@pytest.mark.asyncio
async def test_name_summary_shows_the_name_this_chat_uses_and_follows_edits() -> None:
    """AC#5: closed, the name shows this chat's name or the global default it
    inherits, literally; typing repaints it."""
    app = CoreFirstHarness()
    modal = _modal(app, _settings())
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        assert "Your name in this chat · User (global default)" in _title_line(
            app, modal, NAME_DISCLOSURE_ID
        )

    app = CoreFirstHarness()
    modal = _modal(
        app,
        _settings(),
        user_display_name_override="Lab [gpu]",
        global_user_display_name="Default Name",
    )
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        assert "Your name in this chat · Lab [gpu]" in _title_line(app, modal, NAME_DISCLOSURE_ID)
        name = modal.query_one(f"#{NAME_INPUT_ID}", Input)
        name.focus()  # opens the disclosure
        await pilot.pause()
        await pilot.press("ctrl+a", "A", "d", "a")
        for _ in range(3):
            await pilot.pause()
        assert "Your name in this chat · Ada" in _title_line(app, modal, NAME_DISCLOSURE_ID)
        await pilot.press("ctrl+a", "backspace")
        for _ in range(3):
            await pilot.pause()
        assert "Your name in this chat · Default Name (global default)" in _title_line(
            app, modal, NAME_DISCLOSURE_ID
        )


@pytest.mark.asyncio
async def test_settings_pointers_advertise_the_settings_key() -> None:
    """Parent AC#9 (R11): the Connection copy names F4, the key that opens
    Settings, not F9."""
    app = CoreFirstHarness()
    app.app_config["api_settings"]["openai"] = {}
    modal = _modal(app, _settings("openai", "gpt-5", temperature=0.7))
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        tooltip = str(modal.query_one("#console-settings-configure-credential").tooltip)
        assert "F4 Settings" in tooltip and "F9" not in tooltip
        assert "F4 Settings" in BASE_URL_ENTRY_HINT_COPY
        assert "F9" not in BASE_URL_ENTRY_HINT_COPY
