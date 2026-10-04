"""Settings ▸ Providers & Models ▸ Advanced: one-row disclosures, one frame level (TASK-33007.6).

Mounted with the real application stylesheet at 211x44, the size the spec's
mockup (c) is drawn at. The rarely used controls fold into closed one-row
disclosures under Model defaults; each title says its state, each row of the
opened Saved model list and Catalog refresh says its state in words, and the
card draws no frame inside the detail pane.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Collapsible, Input, SelectionList, Static
from textual.widgets._collapsible import CollapsibleTitle

import tldw_chatbook.UI.Screens.settings_screen as settings_screen_module
from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import _active_destination_screen
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_category_sweep import (
    _click_settings_category,
    _settle_settings,
)
from Tests.UI.test_settings_configuration_hub import (
    FakeSettingsModelDiscoveryScope,
    ModelDiscoveryResult,
    _discovered_model,
)
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import (
    AUTO_REFRESH_PROVIDER_LIST_KEYS,
)

_SIZE = (211, 44)
#: AC#1: the Advanced disclosures in the card's order.
_ADVANCED = (
    ("settings-advanced-context-window", "Context window"),
    ("settings-advanced-saved-models", "Saved model list"),
    ("settings-advanced-catalog-refresh", "Catalog refresh"),
    ("settings-advanced-custom-endpoints", "Custom endpoints"),
    ("settings-snapshot-controls", "Prompt-cache snapshots"),
)
_SAVED = ("gpt-saved-a", "gpt-saved-b")
_DISCOVERED = ("gpt-saved-a", "gpt-new-1", "gpt-new-2")


def _app(*, discovered=()):
    """Build an OpenAI app with two saved models and a fake discovery scope.

    Args:
        discovered: Model ids the fake endpoint lists.

    Returns:
        The app.
    """
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {"api_base_url": "https://first.invalid/v1"},
    }
    app.providers_models = {"OpenAI": list(_SAVED)}
    app.llm_provider_catalog_scope_service = FakeSettingsModelDiscoveryScope(
        result=ModelDiscoveryResult(
            provider="openai",
            provider_list_key="OpenAI",
            endpoint_fingerprint="fixture",
            status="success",
            models=tuple(_discovered_model(name) for name in discovered),
        )
    )
    return app


async def _open(host, pilot):
    await _settle_settings(pilot)
    await _click_settings_category(pilot, "providers-models")
    await host.workers.wait_for_complete()
    await pilot.pause()
    return _active_destination_screen(host)


def _framed(widget: Widget) -> bool:
    """Return True when the widget draws any border edge."""
    styles = widget.styles
    edges = (styles.border_top, styles.border_right, styles.border_bottom)
    return any(edge[0] not in ("", "none") for edge in (*edges, styles.border_left))


def _title(screen, disclosure_id: str) -> str:
    return str(screen.query_one(f"#{disclosure_id}", Collapsible).title)


def _prompts(selection: SelectionList) -> list[str]:
    return [
        str(selection.get_option_at_index(index).prompt)
        for index in range(selection.option_count)
    ]


@pytest.mark.asyncio
@private_profile_test
async def test_advanced_follows_model_defaults_as_closed_one_row_disclosures(request):
    """AC#1, AC#2, AC#8: after Model defaults, the Advanced header, then five
    closed one-row disclosures in order, each titled with a summary; Prompt-
    cache snapshots is no longer above Connect; the card and its groups draw
    no frame while other categories keep theirs."""
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        card = screen.query_one("#settings-providers-models-card")
        children = list(card.children)
        defaults = screen.query_one("#settings-generation-defaults")
        header = screen.query_one("#settings-advanced-title", Static)
        at = children.index(defaults)
        assert children[at + 1] is header
        assert str(header.render()) == "Advanced"
        assert [child.id for child in children[at + 2 : at + 7]] == [
            disclosure_id for disclosure_id, _name in _ADVANCED
        ]
        assert children[at + 7 :] == []
        connect = screen.query_one("#settings-provider-connect-title")
        assert children.index(connect) == 0
        for disclosure_id, name in _ADVANCED:
            disclosure = screen.query_one(f"#{disclosure_id}", Collapsible)
            assert disclosure.collapsed, disclosure_id
            title = str(disclosure.title)
            assert title.startswith(f"{name} · ") and title != f"{name} · ", title
            assert disclosure.query_one(CollapsibleTitle).region.height == 1
            assert disclosure.region.height == 1, disclosure_id
            assert not _framed(disclosure), disclosure_id
        for section in (
            "#settings-provider-connect-title",
            "#settings-default-model-title",
        ):
            assert screen.query_one(section).region.height == 1
        assert header.region.height == 1

        assert not _framed(card)
        for group in ("#settings-model-catalog-group", "#settings-custom-endpoints"):
            widget = screen.query_one(group)
            assert widget.has_class("settings-instant-apply-group")
            assert not _framed(widget), group

        await _click_settings_category(pilot, "appearance")
        assert _framed(screen.query_one("#settings-appearance-card"))


@pytest.mark.asyncio
@private_profile_test
async def test_saved_model_list_counts_and_each_row_says_selected_and_saved(request):
    """AC#1, AC#3, AC#4: the title counts saved against discovered models;
    Discover and Save selected still work inside the disclosure; each row says
    in words whether it is selected and whether it is already saved."""
    app = _app(discovered=_DISCOVERED)
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        disclosure_id = "settings-advanced-saved-models"
        assert _title(screen, disclosure_id) == (
            "Saved model list · 2 saved in config · none discovered"
        )
        screen.query_one(f"#{disclosure_id}", Collapsible).collapsed = False
        await pilot.pause()
        screen.query_one("#settings-discover-provider-models", Button).press()
        await host.workers.wait_for_complete()
        await pilot.pause()

        selection = screen.query_one("#settings-discovered-models-list", SelectionList)
        assert _title(screen, disclosure_id) == (
            "Saved model list · 2 saved in config · 3 discovered, 2 not saved"
        )
        assert [prompt.split(" · ")[:3] for prompt in _prompts(selection)] == [
            ["not selected", "gpt-saved-a", "saved"],
            ["not selected", "gpt-new-1", "not saved"],
            ["not selected", "gpt-new-2", "not saved"],
        ]

        selection.focus()
        await pilot.pause()
        selection.highlighted = 1
        await pilot.press("space")
        await pilot.pause()
        assert selection.selected == ["gpt-new-1"]
        assert [prompt.split(" · ")[0] for prompt in _prompts(selection)] == [
            "not selected",
            "selected",
            "not selected",
        ]
        assert _title(screen, disclosure_id).endswith(" · 1 selected")

        screen.query_one("#settings-save-discovered-provider-models", Button).press()
        await host.workers.wait_for_complete()
        await pilot.pause()
        scope = app.llm_provider_catalog_scope_service
        assert [call["model_ids"] for call in scope.persist_calls] == [["gpt-new-1"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("startup", [True, False])
async def test_catalog_refresh_rows_say_on_off_and_whether_choices_apply(startup):
    """AC#5, AC#6: one row per provider with On/Off words, the instant-apply
    label, and -- while startup refresh is Off -- a line and a title saying the
    per-provider choices are not in effect."""
    saved = {
        "model_catalog": {
            "auto_refresh_enabled": startup,
            "stale_after_hours": 6,
            "auto_refresh_disabled": ["openai"],
            "write_to_config": ["openrouter"],
        }
    }
    host = _SettingsCssHarness(_app(), "settings")

    def save(section_values):
        # The written section is what the next load reads, as on disk.
        saved.update(section_values)
        return True

    with (
        patch.object(settings_screen_module, "load_settings", return_value=saved),
        patch.object(
            settings_screen_module, "save_settings_to_cli_config", side_effect=save
        ) as save_mock,
    ):
        async with host.run_test(size=_SIZE) as pilot:
            screen = await _open(host, pilot)
            disclosure_id = "settings-advanced-catalog-refresh"
            total = len(AUTO_REFRESH_PROVIDER_LIST_KEYS)
            title = _title(screen, disclosure_id)
            if startup:
                assert title == (
                    "Catalog refresh · applies immediately · startup refresh On · "
                    f"every 6 h · {total - 1} of {total} providers"
                )
            else:
                assert title == (
                    "Catalog refresh · applies immediately · startup refresh Off · "
                    "per-provider choices not in effect"
                )
            disclosure = screen.query_one(f"#{disclosure_id}", Collapsible)
            disclosure.collapsed = False
            await pilot.pause()

            group = screen.query_one("#settings-model-catalog-group")
            hint = screen.query_one("#settings-model-catalog-instant-hint", Static)
            assert "applies immediately" in str(hint.render())
            assert hint in group.query(Static)
            startup_off = screen.query_one(
                "#settings-model-catalog-startup-off", Static
            )
            assert startup_off.display is (not startup)
            master = screen.query_one("#settings-model-catalog-auto-refresh", Checkbox)
            assert (
                master.label.plain == f"Refresh on startup {'On' if startup else 'Off'}"
            )
            openai = screen.query_one("#settings-mc-auto-openai", Checkbox)
            assert openai.label.plain == "OpenAI: refresh Off"
            write = screen.query_one("#settings-mc-write-openrouter", Checkbox)
            assert write.label.plain == "save to config On"
            for provider in AUTO_REFRESH_PROVIDER_LIST_KEYS:
                row = screen.query_one(f"#settings-mc-auto-{provider.lower()}").parent
                assert row.region.height == 1, provider
                assert len(row.query(Checkbox)) == 2, provider

            master.toggle()
            await host.workers.wait_for_complete()
            await pilot.pause()
            assert (
                master.label.plain == f"Refresh on startup {'Off' if startup else 'On'}"
            )
            assert startup_off.display is startup
            assert ("startup refresh Off" in _title(screen, disclosure_id)) is startup
            openai.toggle()
            await pilot.pause()
            assert openai.label.plain == "OpenAI: refresh On"
            # Instant apply: written straight away, never staged.
            assert save_mock.called
            assert screen._provider_draft() is None or not (
                screen._provider_draft().dirty_keys
            )


@pytest.mark.asyncio
@private_profile_test
async def test_advanced_titles_follow_the_context_window_and_snapshot_drafts(request):
    """AC#1: titles are live -- a typed context-window override and a ticked
    snapshot box re-say their titles; an opened disclosure stays open across
    a card rebuild."""
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        context_id = "settings-advanced-context-window"
        before = _title(screen, context_id)
        assert before.startswith("Context window · ") and "tokens" in before
        assert "no override" in before
        assert _title(screen, "settings-snapshot-controls") == (
            "Prompt-cache snapshots · llama.cpp only · Off"
        )
        assert _title(screen, "settings-advanced-custom-endpoints") == (
            "Custom endpoints · no named endpoints · applies immediately"
        )

        screen.query_one(f"#{context_id}", Collapsible).collapsed = False
        await pilot.pause()
        field = screen.query_one("#settings-model-context-window", Input)
        field.value = "4242"
        await pilot.pause()
        assert _title(screen, context_id) == "Context window · 4,242 tokens · edited *"

        screen.query_one("#settings-snapshot-controls", Collapsible).collapsed = False
        await pilot.pause()
        screen.query_one("#settings-snapshot-enabled", Checkbox).value = True
        await pilot.pause()
        assert _title(screen, "settings-snapshot-controls").startswith(
            "Prompt-cache snapshots · llama.cpp only · On"
        )

        await _click_settings_category(pilot, "appearance")
        await _click_settings_category(pilot, "providers-models")
        assert not screen.query_one(f"#{context_id}", Collapsible).collapsed
        assert not screen.query_one(
            "#settings-snapshot-controls", Collapsible
        ).collapsed
        assert screen.query_one(
            "#settings-advanced-saved-models", Collapsible
        ).collapsed


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(
    ("query", "field_id", "disclosure_id"),
    [
        (
            "context window",
            "settings-model-context-window",
            "settings-advanced-context-window",
        ),
        (
            "refresh after",
            "settings-model-catalog-stale-hours",
            "settings-advanced-catalog-refresh",
        ),
        ("snapshot keep", "settings-snapshot-keep", "settings-snapshot-controls"),
    ],
)
async def test_field_search_opens_the_closed_disclosure_it_lands_in(
    request, query, field_id, disclosure_id
):
    """AC#10: '/' reaches a field inside a closed Advanced disclosure, opens
    that disclosure and focuses the field at a real region."""
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        assert screen.query_one(f"#{disclosure_id}", Collapsible).collapsed
        screen._submit_category_search(query)
        for _ in range(12):
            await pilot.pause()
        focused = host.focused
        assert focused is not None and focused.id == field_id, f"focused={focused!r}"
        assert not screen.query_one(f"#{disclosure_id}", Collapsible).collapsed
        assert focused.region.height > 0
