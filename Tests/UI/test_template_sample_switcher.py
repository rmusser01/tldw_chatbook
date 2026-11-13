"""TASK-34100.2 AC#9 (entry-exit-handoff-23): template samples are not Ready.

The switcher listed TabbyAPI's ``tabby-model`` and vLLM's ``vllm-model-x/y/z``
-- placeholder values the shipped template carries -- as 'Ready · not tested',
because something answered on their shipped ports without listing models
(a Next.js dev server on :8080 in the review). The template now marks those
sections ``template_sample = true``; a marked provider the user never saved,
used or reached is listed under the existing NOT RUNNING group until the user
saves it or a probe lists its models.

Owner ruling 2026-11-05 ("Values decide"): a section counts as a sample only
while it carries the marker AND still holds the shipped endpoint and model
(the endpoint a send would use, not the whole table). A table written before
the marker existed counts as one while it holds the frozen shipped endpoint
and model (the loader infers its marker:
``Tests/Chat/test_template_sample_upgrade.py``). Once the endpoint or model
differs -- a hand edit, a Settings save, Chat settings' 'Save endpoint & use
model' -- it is the user's and is listed normally, never under NOT RUNNING,
whether its server was reached, refused or never contacted.
"""

from __future__ import annotations

from copy import deepcopy

import pytest
from textual.widgets import Input

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


def test_the_marker_reader_sees_a_saved_or_unmarked_table_as_no_sample() -> None:
    """``provider_is_template_sample`` reads the marker only; the switcher
    asks ``provider_holds_template_sample_values`` as well (Values decide)."""
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
async def test_a_marked_sample_whose_model_the_user_edited_is_theirs(servers):
    """Owner ruling 2026-11-05 ("Values decide", G2C-R2-1): a marked table
    whose model the user edited by hand, with no Save, is the user's. Its
    shipped port answers without listing models (here a 404), which kept it
    under NOT RUNNING while only a refusal consulted the values; it is now
    listed as the user's own row, in whichever group its readiness says."""
    servers.status[TABBY] = 404
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build(
            app,
            config=_with_tabby(model="my-own-tabby-model"),
            providers_models={**MODELS, "tabbyapi": ["my-own-tabby-model"]},
        )
        await app.push_screen(switcher)
        await _settle(app, pilot)
        quiet = _group(switcher, "NOT RUNNING")
        listed = {
            row.provider
            for header in ("READY PROVIDERS", "NEEDS SETUP")
            for row in _group(switcher, header)
        }

    assert not any("TabbyAPI" in row.text for row in quiet), quiet
    assert "tabbyapi" in listed, listed


def _with_template(section: str, **overrides) -> dict:
    """CONFIG plus the shipped template's ``section`` table, marker and all,
    with the user's own hand edits on top (no Save: the marker stays true)."""
    config = deepcopy(CONFIG)
    table = deepcopy(DEFAULT_CONFIG_FROM_TOML["api_settings"][section])
    table.update(overrides)
    assert table["template_sample"] is True  # a hand edit, never a Save
    config["api_settings"][section] = table
    return config


@pytest.mark.asyncio
async def test_a_hand_edited_marked_mlx_lm_is_ready_and_pickable(servers):
    """G2C-R2-1: MLX-LM has no models route, so no listing can ever answer
    for it. A marked MLX-LM whose model the user set by hand in config.toml
    was hidden under NOT RUNNING for good. Values decide: it is the user's,
    a READY row that Find reaches and Enter picks."""
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build(
            app,
            config=_with_template("local_mlx_lm", model="my-mlx-model"),
            providers_models={**MODELS, "local_mlx_lm": ["my-mlx-model"]},
        )
        await app.push_screen(switcher)
        await _settle(app, pilot)
        quiet = " ".join(row.text for row in _group(switcher, "NOT RUNNING"))
        ready = [
            (row.provider, row.model) for row in _group(switcher, "READY PROVIDERS")
        ]
        await pilot.press(*"my-mlx")
        await _settle(app, pilot)
        highlighted = switcher.highlighted_row()

    assert "MLX-LM" not in quiet, quiet
    assert ("local_mlx_lm", "my-mlx-model") in ready, ready
    assert highlighted is not None
    assert (highlighted.provider, highlighted.model) == ("local_mlx_lm", "my-mlx-model")
    assert 5678 not in servers.ports  # its shipped port is never contacted


@pytest.mark.asyncio
async def test_a_keyed_marked_vllm_pointed_at_the_users_server_is_ready(servers):
    """G2C-R2-1: a keyed vLLM is never contacted by the switcher (AC#4), so
    while only a refusal consulted the values, a marked vLLM the user pointed
    at their own running server and model by hand read NOT RUNNING. Values
    decide: its endpoint and model are the user's, so it is a READY row
    that Find reaches and Enter picks."""
    own = 9098
    servers.up.add(own)
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build(
            app,
            config=_with_template(
                "vllm",
                api_url=f"http://127.0.0.1:{own}/v1/chat/completions",
                model="my-own-model",
                api_key="sk-local",
            ),
            providers_models={**MODELS, "vllm": ["my-own-model"]},
        )
        await app.push_screen(switcher)
        await _settle(app, pilot)
        quiet = " ".join(row.text for row in _group(switcher, "NOT RUNNING"))
        ready = [
            (row.provider, row.model) for row in _group(switcher, "READY PROVIDERS")
        ]
        await pilot.press(*"my-own")
        await _settle(app, pilot)
        highlighted = switcher.highlighted_row()

    assert "vLLM" not in quiet, quiet
    assert ("vllm", "my-own-model") in ready, ready
    assert highlighted is not None
    assert (highlighted.provider, highlighted.model) == ("vllm", "my-own-model")
    assert own not in servers.ports  # keyed: listed, never contacted (AC#4)


@pytest.mark.asyncio
async def test_a_marked_sample_at_its_shipped_values_stays_not_running(servers):
    """The counterpart: the same marked vLLM and MLX-LM tables still at the
    shipped endpoint and model are samples, under NOT RUNNING, never READY."""
    config = _with_template("vllm")
    config["api_settings"]["local_mlx_lm"] = deepcopy(
        DEFAULT_CONFIG_FROM_TOML["api_settings"]["local_mlx_lm"]
    )
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build(
            app,
            config=config,
            providers_models={**MODELS, "vllm": [], "local_mlx_lm": []},
        )
        await app.push_screen(switcher)
        await _settle(app, pilot)
        quiet = " ".join(row.text for row in _group(switcher, "NOT RUNNING"))
        ready = {row.provider for row in _group(switcher, "READY PROVIDERS")}

    assert "vLLM" in quiet and "MLX-LM" in quiet, quiet
    assert not {"vllm", "local_mlx_lm"} & ready, ready


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


@pytest.mark.asyncio
async def test_a_marked_mlx_lm_pointed_at_the_users_port_is_theirs(servers):
    """Review v5r2 G2C-R3-3, in the shape the live app builds (G2C-R3-6).

    ``chat_with_mlx_lm`` posts to its ``host``/``port`` (default
    127.0.0.1:8080), never to ``api_url``. A user who points MLX-LM at their
    own server with ``port`` and keeps the shipped model was hidden as
    'MLX-LM (Apple silicon) (not checked)' under NOT RUNNING, because the
    sample identity read only the URL keys. It is the user's: READY
    PROVIDERS carries its 'no models reported' line (a hand edit adds no
    catalog row; the template's list is ``["None"]``), and the name and an
    id typed in Find pick it.
    """
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build(
            app,
            config=_with_template("local_mlx_lm", port=5679),
            providers_models={**MODELS, "local_mlx_lm": ["None"]},
        )
        await app.push_screen(switcher)
        await _settle(app, pilot)
        quiet = " ".join(row.text for row in _group(switcher, "NOT RUNNING"))
        ready = " ".join(row.text for row in _group(switcher, "READY PROVIDERS"))
        # The provider's whole name, then the id (the line's own advice).
        switcher.query_one("#console-popover-find", Input).value = (
            "MLX-LM (Apple silicon) custom-model-gamma"
        )
        await _settle(app, pilot)
        row = switcher.highlighted_row()

    assert "MLX-LM" not in quiet, quiet
    assert "MLX-LM (Apple silicon) · no models reported" in ready, ready
    assert row is not None
    assert (row.kind, row.provider, row.model) == (
        "typed",
        "local_mlx_lm",
        "custom-model-gamma",
    )
    assert not {5678, 5679} & set(servers.ports)  # never contacted
