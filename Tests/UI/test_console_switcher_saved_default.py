"""Switch model lists each provider's saved default among its first rows (TASK-35012).

With Find empty, READY PROVIDERS shows three models a provider. A model saved
last of a provider's list sat under "… N more" and looked unsaved, though
TASK-35011 had just made it the provider's default for new chats. The rows
are now ordered at display time: that default leads its provider's group, or
follows the surface's current model when that leads. Nothing is added, the
saved list on disk keeps its order, and typed Find ranks as before.

The real popover on the switcher harness with the production stylesheets;
the local-server cases run the real prober over a stubbed network.
"""

from __future__ import annotations

import threading
from copy import deepcopy
from datetime import timedelta

import pytest

from Tests.private_profile import private_profile_test
from Tests.UI.test_console_model_switcher import (
    HIGHLIGHT_GLYPH,
    Recorder,
    SwitcherHarness,
    _draft,
    _row_keys,
    _settle,
    _Use,
    build_switcher,
    line_with,
    list_lines,
    open_switcher,
)
from Tests.UI.test_console_switcher_local_probe import (
    CONFIG,
    OLLAMA,
    PROVIDERS_MODELS,
    _group,
    _harness,
    _wait,
    build,
    servers,  # noqa: F401 - the fixture the probe cases take
)
from tldw_chatbook.Widgets.Console.console_model_popover import (
    CURRENT_MARK,
    ConsoleModelPopover,
)

pytestmark = pytest.mark.asyncio

#: Anthropic's saved list, with the default for new chats saved last of five.
ANTHROPIC = [
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "claude-opus-4-1",
    "claude-3-7-sonnet",
    "claude-3-5-haiku",
]
DEFAULT = "claude-3-5-haiku"
LISTS = {
    "Llama_cpp": ["model-a", "model-b"],
    "Anthropic": ANTHROPIC,
    "OpenRouter": ["openai/gpt-y", "meta/llama-z"],
}
CONFIG_WITH_DEFAULT = {"api_settings": {"anthropic": {"model": DEFAULT}}}
READY_ALL = frozenset({"*"})


def _ready(switcher: ConsoleModelPopover, provider: str) -> tuple[list, str | None]:
    """A provider's READY PROVIDERS models, in order, and its "… N more" line."""
    rows = [
        row for row in _group(switcher, "READY PROVIDERS") if row.provider == provider
    ]
    more = next((row.text.strip() for row in rows if row.kind == "more"), None)
    return [row.model for row in rows if row.kind == "pair"], more


@private_profile_test
@pytest.mark.parametrize(
    "case",
    ["first", "new-chat-default", "behind-current", "is-current", "under-recent"],
)
async def test_the_saved_default_leads_its_group_or_follows_the_current_model(
    request, case
):
    """AC#1: with Find empty, the provider's saved default for new chats
    (P8's resolver: ``chat_defaults`` for its own provider, else the
    provider's table) leads its group, though it is saved last of five; it is
    second behind this chat's model when that leads the group; nothing moves
    when it is this chat's model; and a pair already listed under RECENT is
    not listed again (TASK-35012's position rule)."""
    config = deepcopy(CONFIG_WITH_DEFAULT)
    draft = ("llama_cpp", "model-a")
    recent = None
    if case == "new-chat-default":
        # The table names list[0]; new chats on Anthropic start with DEFAULT.
        config["api_settings"]["anthropic"]["model"] = ANTHROPIC[0]
        config["chat_defaults"] = {"provider": "anthropic", "model": DEFAULT}
    elif case == "behind-current":
        draft = ("anthropic", "claude-opus-4-1")
    elif case == "is-current":
        draft = ("anthropic", DEFAULT)
    elif case == "under-recent":
        recent = (
            _Use("openrouter", "openai/gpt-y", timedelta(hours=1)),  # PREVIOUS
            _Use("anthropic", DEFAULT, timedelta(hours=2)),
        )
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app,
            pilot,
            build_switcher(
                Recorder(ready=READY_ALL),
                draft=_draft(*draft),
                providers_models=LISTS,
                app_config=config,
                recent=recent,
            ),
        )
        models, more = _ready(switcher, "anthropic")
        lines = list_lines(app, switcher)
        recent_rows = [(row.provider, row.model) for row in _group(switcher, "RECENT")]

    if case in {"first", "new-chat-default"}:
        assert models == [DEFAULT, *ANTHROPIC[:2]], models
        assert more == "… 2 more Anthropic models", more
    elif case == "behind-current":
        assert models == ["claude-opus-4-1", DEFAULT, ANTHROPIC[0]], models
        assert CURRENT_MARK in line_with(lines, " claude-opus-4-1 ")
    elif case == "is-current":
        assert models == [DEFAULT, *ANTHROPIC[:2]], models
        assert CURRENT_MARK in line_with(lines, f" {DEFAULT} ")
    else:
        assert recent_rows == [("anthropic", DEFAULT)], recent_rows
        assert models == ANTHROPIC[:3], models
        assert more == "… 1 more Anthropic model", more


@private_profile_test
async def test_a_lifted_default_lists_the_same_models_and_the_more_count(request):
    """AC#2: the group holds the same models as before, only reordered: the
    three rows and the models behind "… N more" are the saved list, the count
    is the rest of it, and the provider's models (what Find searches and the
    readiness check names) keep the saved order."""
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app,
            pilot,
            build_switcher(
                Recorder(ready=READY_ALL),
                providers_models=LISTS,
                app_config=CONFIG_WITH_DEFAULT,
            ),
        )
        models, more = _ready(switcher, "anthropic")
        saved_order = switcher._models_for("anthropic")
        line = line_with(list_lines(app, switcher), "more Anthropic")

    assert models == [DEFAULT, ANTHROPIC[0], ANTHROPIC[1]], models
    hidden = [model for model in ANTHROPIC if model not in models]
    assert sorted(models + hidden) == sorted(ANTHROPIC)
    assert more == f"… {len(hidden)} more Anthropic models", more
    assert line.strip() == more
    assert saved_order == tuple(ANTHROPIC)


@private_profile_test
@pytest.mark.parametrize(
    ("typed", "expected"),
    [("anthropic ", ANTHROPIC), ("claude", ANTHROPIC)],
    ids=["exact-name", "prefix-tie"],
)
async def test_find_ranks_its_matches_as_before(request, typed, expected):
    """AC#4: typed Find keeps its results and their order. Find that is the
    provider's name ranks all its rows -1, and a prefix every id shares ranks
    them 1, so the saved order decides both the order and the highlight:
    the lifted default stays where its list put it (last)."""
    recorder = Recorder(ready=READY_ALL)
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app,
            pilot,
            build_switcher(
                recorder, providers_models=LISTS, app_config=CONFIG_WITH_DEFAULT
            ),
        )
        untyped, _more = _ready(switcher, "anthropic")
        for key in typed:
            await pilot.press("space" if key == " " else key)
        await _settle(app, pilot)
        typed_rows = [
            row.model
            for row in _group(switcher, "READY PROVIDERS")
            if row.kind == "pair" and row.provider == "anthropic"
        ]
        highlighted = switcher.highlighted_row()

    assert untyped[0] == DEFAULT, untyped  # the lift runs in this open
    assert typed_rows == expected, typed_rows
    assert highlighted is not None
    assert (highlighted.provider, highlighted.model) == ("anthropic", expected[0])


#: TabbyAPI's shipped table names the sample id "tabby-model" (config.py).
SAMPLE = "tabby-model"
TABBY = ["tabby-one", "tabby-two", "tabby-three", SAMPLE]


@private_profile_test
@pytest.mark.parametrize("seeded", ["saved-list", "served"])
async def test_a_shipped_sample_id_default_is_never_lifted(request, seeded):
    """AC#3: a provider whose saved default is a shipped sample id keeps its
    rows in list order, the sample behind "… 1 more", whether the id came
    from a saved list (seeded past the loader, which drops it) or a listing;
    Anthropic's real default beside it is lifted in the same open."""
    config = deepcopy(CONFIG_WITH_DEFAULT)
    config["api_settings"]["tabbyapi"] = {"model": SAMPLE}
    lists = {**LISTS, "TabbyAPI": TABBY if seeded == "saved-list" else []}
    served = {"tabbyapi": TABBY} if seeded == "served" else None
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app,
            pilot,
            build_switcher(
                Recorder(ready=READY_ALL),
                providers_models=lists,
                app_config=config,
                served_models=served,
            ),
        )
        anthropic, _more = _ready(switcher, "anthropic")
        tabby, more = _ready(switcher, "tabbyapi")

    assert anthropic[0] == DEFAULT, anthropic
    assert tabby == TABBY[:3], tabby
    assert more == "… 1 more TabbyAPI model", more


@private_profile_test
async def test_chat_settings_model_picker_lists_the_same_order(request):
    """NO-3, AC#1: Chat settings' model picker (Switch model's pick mode)
    lists the rows Alt+M lists, in the same order; its current model is the
    draft's pair, so the default is second behind it."""
    keys, groups = {}, {}
    for pick_only in (False, True):
        app = SwitcherHarness()
        async with app.run_test(size=(211, 44)) as pilot:
            switcher = await open_switcher(
                app,
                pilot,
                build_switcher(
                    Recorder(ready=READY_ALL),
                    draft=_draft("anthropic", "claude-opus-4-1"),
                    providers_models=LISTS,
                    app_config=CONFIG_WITH_DEFAULT,
                    pick_only=pick_only,
                ),
            )
            keys[pick_only] = _row_keys(switcher)
            groups[pick_only] = _ready(switcher, "anthropic")

    assert groups[True] == (
        ["claude-opus-4-1", DEFAULT, ANTHROPIC[0]],
        "… 2 more Anthropic models",
    ), groups[True]
    assert keys[True] == keys[False]


@private_profile_test
async def test_empty_find_with_no_previous_or_current_row_highlights_the_lifted_default(
    request,
):
    """D-c: with no PREVIOUS row and no CURRENT row (this chat names no model
    yet, and there are no recent pairs), the highlight falls back to the
    first pair row, which is now the lifted default, so Enter applies it."""
    recorder = Recorder(ready=READY_ALL)
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app,
            pilot,
            build_switcher(
                recorder,
                draft=_draft("anthropic", None),
                providers_models=LISTS,
                app_config=CONFIG_WITH_DEFAULT,
            ),
        )
        assert not _group(switcher, "PREVIOUS") and not _group(switcher, "RECENT")
        row = switcher.highlighted_row()
        assert row is not None and (row.provider, row.model) == ("anthropic", DEFAULT)
        assert line_with(list_lines(app, switcher), f" {DEFAULT} ").startswith(
            HIGHLIGHT_GLYPH
        )
        await pilot.press("enter")
        await pilot.pause()

    settings = recorder.submissions[0].draft.settings
    assert (settings.provider, settings.model) == ("anthropic", DEFAULT)


#: Ollama's saved list, its saved default last of four.
OLLAMA_SAVED = ["m-one", "m-two", "m-three", "fake-beta"]


@private_profile_test
@pytest.mark.parametrize("listing", ["lists-the-default-4th", "leaves-it-out"])
async def test_a_local_servers_listing_orders_the_default_and_never_adds_it(
    servers,  # noqa: F811 - the probe fixture, imported above
    request,
    listing,
):
    """The probe rule (TASK-33005.12's rows, ordered only): until the local
    probe answers, Ollama's rows are its saved models and the default leads
    them; once the server's listing answers, the rows are what it listed, so
    a default it lists 4th leads within the listing, and one it does not list
    is gone, not added back."""
    listed = ["fake-alpha", "fake-gamma", "fake-delta"]
    listed.append("fake-beta" if listing == "lists-the-default-4th" else "fake-eps")
    servers.up = {OLLAMA}
    servers.listing = {OLLAMA: listed}
    servers.hold = threading.Event()
    config = deepcopy(CONFIG)
    config["api_settings"]["ollama"]["model"] = "fake-beta"
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build(
            app,
            config=config,
            providers_models={**PROVIDERS_MODELS, "ollama": OLLAMA_SAVED},
            pair=("llama_cpp", "model-a"),
        )
        await app.push_screen(switcher)
        await _wait(pilot, lambda: len(servers.requests) >= 2)
        await _wait(pilot, lambda: bool(_ready(switcher, "ollama")[0]))
        before = _ready(switcher, "ollama")
        servers.hold.set()
        await _settle(app, pilot)
        after = _ready(switcher, "ollama")
        models = switcher._models_for("ollama")

    assert before == (["fake-beta", "m-one", "m-two"], "… 1 more Ollama model")
    if listing == "lists-the-default-4th":
        assert after == (["fake-beta", *listed[:2]], "… 1 more Ollama model")
        assert models == tuple(listed)
    else:
        assert after == (listed[:3], "… 1 more Ollama model")
        assert "fake-beta" not in models
