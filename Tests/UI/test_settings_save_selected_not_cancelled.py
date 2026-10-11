"""Settings ▸ Providers & Models: Save selected is never cancelled (review of
#3057, R2 "outside the diff", ruling C9).

Discover, Save selected and Clear shared one exclusive worker group, so a
press on Discover or Clear while Save selected was writing cancelled a
durable write.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Button, SelectionList

from Tests.private_profile import private_profile_test
from Tests.UI.test_settings_default_model_picker import (
    _app,
    _CatalogScope,
    _open_providers,
    _settle,
)
from Tests.UI.test_settings_discovery_listing_dropped import (
    _SERVED,
    _SIZE,
    _discover,
)
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    PersistenceResult,
)


class _HeldSaveScope(_CatalogScope):
    """Save selected waits until the test lets it finish."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.saved: tuple[str, ...] = ()

    async def persist_discovered_models_to_settings(self, **kwargs):
        self.persist_calls.append(kwargs)
        self.started.set()
        await self.release.wait()
        self.saved = tuple(kwargs["model_ids"])
        return PersistenceResult(
            provider="openai",
            provider_list_key="OpenAI",
            status="saved",
            saved_model_ids=self.saved,
        )


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(
    "interrupt",
    [
        "#settings-discover-provider-models",
        "#settings-clear-discovered-provider-models",
    ],
)
async def test_save_selected_is_not_cancelled_by_discover_or_clear(request, interrupt):
    app = _app("openai", "gpt-4o", catalog=("gpt-4o",), served=(_SERVED,))
    scope = app.llm_provider_catalog_scope_service = _HeldSaveScope(
        "openai", catalog=("gpt-4o",), served=(_SERVED,)
    )
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        await _discover(host, pilot, screen)
        screen.query_one("#settings-discovered-models-list", SelectionList).select_all()
        await pilot.pause()

        screen.query_one("#settings-save-discovered-provider-models", Button).press()
        await asyncio.wait_for(scope.started.wait(), timeout=10)
        screen.query_one(interrupt, Button).press()
        await pilot.pause()
        await pilot.pause()
        scope.release.set()
        await _settle(host, pilot)

        assert scope.saved == (_SERVED,)
        assert _SERVED in app.providers_models["OpenAI"]
