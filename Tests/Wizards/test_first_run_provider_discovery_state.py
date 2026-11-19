"""TASK-34100.6 folded defect (a): a Next before discovery settles.

Live evidence setup-wizard-ux-qa/evidence/g1-fm80/14-17 (base 3c439d606e):
Full track, llama.cpp at an unreachable endpoint, Next on Provider then Next on
Model before the check settled. Model then offered 'None (recommended)' -- the
shipped ``[providers] Llama_cpp = ["None"]`` placeholder rendered as a model --
and Provider kept saying 'Checking the selected provider…' for a check that
Model's hide had already cancelled.
"""

from __future__ import annotations

import asyncio

import pytest

from Tests.private_profile import private_profile_test
from Tests.Wizards.provider_step_harness import (
    SIZE,
    WizardHost,
    current_step,
    pick_provider,
    resumed_on_provider,
    static_text,
    wait_for_step,
)
from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    STEP_MODEL,
    STEP_PROVIDER,
    STEP_VOICE,
    curated_models_for_provider,
)

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize("placeholder", ["None", "none", "null", "", "  "])
def test_curated_placeholder_models_are_never_offered(placeholder):
    catalog = {"Llama_cpp": [placeholder], "OpenAI": ["gpt-4o-mini", placeholder]}

    assert curated_models_for_provider(catalog, "llama_cpp") == []
    assert curated_models_for_provider(catalog, "openai") == ["gpt-4o-mini"]


@pytest.mark.asyncio
@private_profile_test
async def test_a_cancelled_handoff_never_leaves_provider_checking(request):
    """Model's hide cancels the check it borrowed; Provider must not claim it runs."""
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    never = asyncio.Event()

    async def hanging_discovery(_provider_key):
        await never.wait()
        return ()

    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        provider._discover = hanging_discovery
        await pick_provider(pilot, provider, "llama_cpp")
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        assert provider._selected_discovery_state == "in_progress"
        await pilot.press("ctrl+n")  # before the check settles
        await wait_for_step(pilot, wizard, STEP_VOICE)
        await pilot.pause()

        assert provider._selected_discovery_state != "in_progress"
        status = static_text(provider, "#setup-provider-probe-status")
        assert "Checking" not in status, status
