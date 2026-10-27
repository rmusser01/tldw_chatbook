"""TASK-34100.2 AC#9 (entry-exit-handoff-23): template samples are not Ready.

The switcher listed TabbyAPI's ``tabby-model`` and vLLM's ``vllm-model-x/y/z``
-- placeholder values the shipped template carries -- as 'Ready · not tested',
because something answered on their shipped ports without listing models
(a Next.js dev server on :8080 in the review). The template now marks those
sections ``template_sample = true``; a marked provider the user never saved,
used or reached is listed under the existing NOT RUNNING group until the user
saves it or a probe lists its models. In an already-loaded configuration the
marker alone decides whether a provider is a sample (the loader infers it for
a table written before the marker existed:
``Tests/Chat/test_template_sample_upgrade.py``); for a refused sample, only the
endpoint a send would use and its model (not the whole table) decide whether
the user edited it by hand.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from Tests.UI.test_console_model_switcher import _settle, list_lines
from Tests.UI.test_console_switcher_local_probe import (  # noqa: F401 - fixture
    CONFIG,
    PROVIDERS_MODELS,
    _group,
    _harness,
    build,
    servers,
)
from tldw_chatbook.Chat.console_session_settings import provider_is_template_sample
from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML

# Review round 1 (TE-2): mounted switcher cases read app config inside the
# test, so they keep the collection-time profile; without it the file passed
# only beside modules that import the app at collection (RecoveryRequired).
pytestmark = pytest.mark.bootstrap_profile

TABBY = 8080
MODELS = {**PROVIDERS_MODELS, "tabbyapi": ["tabby-model", "tabby-model-2"]}


def _with_tabby(**overrides) -> dict:
    config = deepcopy(CONFIG)
    table = deepcopy(DEFAULT_CONFIG_FROM_TOML["api_settings"]["tabbyapi"])
    table.update(overrides)
    config["api_settings"]["tabbyapi"] = table
    return config


def test_the_shipped_placeholders_carry_an_explicit_marker() -> None:
    api = DEFAULT_CONFIG_FROM_TOML["api_settings"]
    for section in ("tabbyapi", "vllm", "local_mlx_lm"):
        assert api[section]["template_sample"] is True, section
    assert "template_sample" not in api["llama_cpp"]
    assert "template_sample" not in api["openai"]


def test_the_marker_not_the_values_decides() -> None:
    marked = _with_tabby()
    saved = _with_tabby(template_sample=False)  # what Save writes

    assert provider_is_template_sample(marked, "tabbyapi") is True
    assert provider_is_template_sample(saved, "tabbyapi") is False
    # No marker in an already-merged mapping reads as not-a-sample. (The
    # loader infers the marker for a pre-marker table that still holds the
    # sample endpoint and model, before this mapping exists.)
    unmarked = _with_tabby()
    del unmarked["api_settings"]["tabbyapi"]["template_sample"]
    assert provider_is_template_sample(unmarked, "tabbyapi") is False


@pytest.mark.asyncio
async def test_a_sample_whose_port_answers_without_models_is_not_ready(servers):
    servers.status[TABBY] = 404  # something else listens on :8080
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build(app, config=_with_tabby(), providers_models=MODELS)
        await app.push_screen(switcher)
        await _settle(app, pilot)
        quiet = _group(switcher, "NOT RUNNING")
        ready = _group(switcher, "READY PROVIDERS")
        lines = list_lines(app, switcher)

    assert any("TabbyAPI" in row.text for row in quiet), quiet
    assert not any(row.provider == "tabbyapi" for row in ready), ready
    assert not any("tabby-model" in line and "Ready" in line for line in lines)


@pytest.mark.asyncio
async def test_a_marked_sample_whose_values_drifted_is_still_not_ready(servers):
    """The marker, not a value comparison, keeps a sample out of Ready: a
    marked table hand-edited without Save no longer equals the template, and
    still never reads Ready. (A table written before the marker existed has
    no marker in the user's file; the loader infers one while it still holds
    the sample endpoint and model, and writes ``false`` otherwise --
    ``Tests/Chat/test_template_sample_upgrade.py``.)"""
    servers.status[TABBY] = 404
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build(
            app,
            config=_with_tabby(model="tabby-model-from-an-older-template"),
            providers_models=MODELS,
        )
        await app.push_screen(switcher)
        await _settle(app, pilot)
        quiet = _group(switcher, "NOT RUNNING")
        ready = _group(switcher, "READY PROVIDERS")

    assert any("TabbyAPI" in row.text for row in quiet), quiet
    assert not any(row.provider == "tabbyapi" for row in ready), ready


@pytest.mark.asyncio
async def test_a_sample_that_lists_models_is_ready(servers):
    servers.up.add(TABBY)  # a real TabbyAPI answers with its models
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build(app, config=_with_tabby(), providers_models=MODELS)
        await app.push_screen(switcher)
        await _settle(app, pilot)
        await pilot.pause(0.2)
        ready = _group(switcher, "READY PROVIDERS")

    assert any(row.provider == "tabbyapi" for row in ready), ready


@pytest.mark.asyncio
async def test_a_saved_sample_keeps_its_ready_row(servers):
    servers.status[TABBY] = 404
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build(
            app,
            config=_with_tabby(template_sample=False),
            providers_models=MODELS,
        )
        await app.push_screen(switcher)
        await _settle(app, pilot)
        quiet = _group(switcher, "NOT RUNNING")

    assert not any("TabbyAPI" in row.text for row in quiet), quiet


@pytest.mark.asyncio
async def test_a_refused_sample_whose_other_keys_drifted_is_not_running(servers):
    """Pass 3 (AC#9 on old installs): a sample table from a profile written
    before the marker keeps the shipped endpoint and model but differs from
    today's template elsewhere (Custom shipped streaming off; a tuned
    temperature). Refused on its shipped port, it is listed under NOT RUNNING,
    never as a 'refused' NEEDS SETUP row: the sample's endpoint and model, not
    the whole table, say whether the user changed it."""
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build(
            app,
            config=_with_tabby(streaming=False, temperature=0.2),
            providers_models=MODELS,
        )
        await app.push_screen(switcher)
        await _settle(app, pilot)
        quiet = _group(switcher, "NOT RUNNING")
        setup = {row.provider for row in _group(switcher, "NEEDS SETUP")}
        lines = list_lines(app, switcher)

    assert any("TabbyAPI" in row.text for row in quiet), quiet
    assert "tabbyapi" not in setup, setup
    assert not any(f"refused :{TABBY}" in line for line in lines), lines


@pytest.mark.asyncio
async def test_a_refused_marked_sample_with_its_own_endpoint_keeps_its_row(servers):
    """The counterpart: a marked table whose endpoint the user edited by hand
    is theirs, so its refusal stays a NEEDS SETUP row (TASK-33005.6 AC#3)."""
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build(
            app,
            config=_with_tabby(api_url="http://127.0.0.1:5999/v1/chat/completions"),
            providers_models=MODELS,
        )
        await app.push_screen(switcher)
        await _settle(app, pilot)
        setup = {row.provider for row in _group(switcher, "NEEDS SETUP")}

    assert "tabbyapi" in setup, setup


@pytest.mark.asyncio
async def test_a_refused_marked_sample_sent_to_its_own_api_base_url_keeps_its_row(
    servers,
):
    """Review round 1 (R1-2): the Console sends to ``api_base_url`` (then
    ``api_base``, ``base_url``) before ``api_url``, so a marked table pointed
    at the user's own server through one of those keys -- the template's
    ``api_url`` left behind -- is the user's. Its refusal stays a NEEDS SETUP
    row, as the counterpart above."""
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build(
            app,
            config=_with_tabby(api_base_url="http://127.0.0.1:5999/v1"),
            providers_models=MODELS,
        )
        await app.push_screen(switcher)
        await _settle(app, pilot)
        setup = {row.provider for row in _group(switcher, "NEEDS SETUP")}
        quiet = _group(switcher, "NOT RUNNING")

    assert "tabbyapi" in setup, setup
    assert not any("TabbyAPI" in row.text for row in quiet), quiet
