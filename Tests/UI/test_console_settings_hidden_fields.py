"""Chat settings hides the fields the provider does not accept (TASK-33006.2).

Spec rule 2: supported fields come from the one shared support decision
(``supported_generation_fields`` for the samplers, and
``console_generation_control_support`` for the reasoning and thinking
controls, whose "unknown" stays visible, TASK-30012 AC#3). Hidden fields are
named on the Sampling disclosure's one summary line. Every test mounts the
real modal under the production stylesheets and reads what was painted.

Owner ruling (2026-10-02): every closed disclosure title stays one row. The
Sampling title names the hidden fields only when the whole title fits
``DISCLOSURE_TITLE_CELLS``; otherwise it counts them, and the opened
disclosure lists every one by its field-table label.
"""

from __future__ import annotations

import time

import pytest
from rich.cells import cell_len
from textual.containers import ScrollableContainer
from textual.widgets import Button, Collapsible, Select, Static
from textual.widgets._collapsible import CollapsibleTitle

from Tests.private_profile import private_profile_test
from Tests.UI.test_console_settings_core_first import (
    CoreFirstHarness,
    _modal,
    _open,
    _painted,
    _settings,
)
from tldw_chatbook.Chat.console_provider_support import (
    MODEL_CONFIG_FIELDS,
    MODEL_FIELD_LABELS,
    console_generation_control_support,
    reasoning_effort_values_sent,
    supported_generation_fields,
)
from tldw_chatbook.Chat.custom_endpoint_registry import ENDPOINT_FAMILIES
from tldw_chatbook.Chat.provider_catalog import provider_display_name
from tldw_chatbook.Widgets.Console.console_settings_field_row import (
    CORE_FIELDS,
    DISCLOSURE_TITLE_CELLS,
    FIELD_ROW_FIELDS,
    GENERATION_CONTROL_UNKNOWN_COPY,
    SAMPLING_DISCLOSURE_ID,
    SAMPLING_FIELDS,
    SAMPLING_HIDDEN_LIST_ID,
    UNSENT_CHOICE_COPY,
    UNSENT_CHOICE_SUFFIX,
    field_control_id,
    generation_field_support,
    hidden_fields_line,
)

# Census-gated (scripts/ui_pr_gate_census.txt): Tests/UI/conftest.py imports
# tldw_chatbook.app per test, which fails closed with
# RecoveryRequired("raw_source_selection_changed") under the per-test sandbox.
pytestmark = pytest.mark.bootstrap_profile

#: The reasoning and thinking controls; every other row is a sampler.
_CONTROLS = frozenset(
    {
        "reasoning_effort",
        "reasoning_summary",
        "verbosity",
        "thinking_effort",
        "thinking_budget_tokens",
    }
)
_ANTHROPIC_HIDDEN = (
    "min_p",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "reasoning_effort",
    "reasoning_summary",
    "verbosity",
)


def _expected_line(
    provider: str, hidden: tuple[str, ...], name: str | None = None
) -> str:
    """The one-row Sampling title for ``hidden``, under ``name`` if given.

    The form itself is pinned by literals in
    ``test_the_sampling_title_names_what_fits_and_counts_the_rest``.
    """
    return hidden_fields_line(name or provider_display_name(provider), hidden)


def _shared_hidden(app_config, provider: str, model: str) -> tuple[str, ...]:
    """The fields the shared support functions reject, in line order."""
    supported = supported_generation_fields(provider, model, app_config)
    return tuple(
        name
        for name in (*SAMPLING_FIELDS, *CORE_FIELDS)
        if (
            console_generation_control_support(provider, model, name, app_config)
            == "unsupported"
            if name in _CONTROLS
            else name not in supported
        )
    )


def _hidden_rows(modal) -> tuple[str, ...]:
    return tuple(
        name
        for name in (*SAMPLING_FIELDS, *CORE_FIELDS)
        if not modal.query_one(f"#{field_control_id(name)}-row").display
    )


def _sampling_title(modal) -> str:
    return str(modal.query_one(f"#{SAMPLING_DISCLOSURE_ID}", Collapsible).title)


def _painted_title(app, modal) -> str:
    """The Sampling title as painted; it must be one row (owner ruling)."""
    title = modal.query_one(f"#{SAMPLING_DISCLOSURE_ID}").query_one(CollapsibleTitle)
    region = title.content_region
    assert region.height == 1, region
    row = _painted(app.screen)[region.y]
    return " ".join(row[region.x : region.x + region.width].split())


def _real_rebase(state, **kwargs):
    """The controller's rebaser, the seam the switcher's pick mode lands on."""
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    return ConsoleChatController.rebase_console_settings_draft(
        object(), state, **kwargs
    )


@pytest.mark.asyncio
async def test_anthropic_hides_unaccepted_fields_behind_the_sampling_line() -> None:
    """AC#1/#2 as amended by the owner ruling: Min P, Seed, the penalties and
    the reasoning controls Anthropic does not accept are not rendered and
    never take focus; the one-row Sampling title counts them, because their
    labels do not fit one row, and the opened disclosure names them all."""
    app = CoreFirstHarness()
    modal = _modal(app, _settings("anthropic", "claude-sonnet-4-5", temperature=0.7))
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        assert _hidden_rows(modal) == _ANTHROPIC_HIDDEN
        expected = "Sampling · Anthropic does not accept 7 fields (open to list them)"
        assert _expected_line("anthropic", _ANTHROPIC_HIDDEN) == expected
        assert _sampling_title(modal) == expected
        assert _painted_title(app, modal).endswith(expected)
        # The line does not cost the footer its place (T1 AC#9).
        body = modal.query_one("#console-settings-body", ScrollableContainer)
        assert body.max_scroll_y == 0

        # Open Sampling from its title with real keys, then Tab to Apply.
        modal.query_one(f"#{SAMPLING_DISCLOSURE_ID}").query_one(
            CollapsibleTitle
        ).focus()
        await pilot.press("enter")
        for _ in range(3):
            await pilot.pause()
        modal.query_one("#console-settings-temperature").focus()
        await pilot.pause()
        visited: list[str | None] = []
        for _ in range(40):
            await pilot.press("tab")
            visited.append(app.focused.id if app.focused else None)
            if app.focused is modal.query_one("#console-settings-save", Button):
                break
        else:
            raise AssertionError(f"Tab never reached Apply: {visited}")
        assert {"console-settings-top-p", "console-settings-top-k"} <= set(visited)
        assert not {field_control_id(name) for name in _ANTHROPIC_HIDDEN} & set(
            visited
        )
        painted = "\n".join(_painted(app.screen))
        for name in ("min_p", "seed", "presence_penalty", "frequency_penalty"):
            row = modal.query_one(f"#{field_control_id(name)}-row")
            assert not row.region.area, name
        assert "Top K" in painted


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("anthropic", "claude-sonnet-4-5"),
        ("anthropic", "claude-opus-4-7"),
        ("openai", "gpt-4o"),
        ("google", "gemini-2.5-pro"),
        ("llama_cpp", "model-a"),
        ("ollama", "qwen3"),
        ("custom-ep:gpu-box", "model-a"),
    ],
)
@pytest.mark.asyncio
async def test_hidden_set_and_line_come_from_the_shared_support_functions(
    provider, model
) -> None:
    """AC#3: the modal hides exactly what the shared functions reject and
    holds no support table of its own."""
    import tldw_chatbook.Widgets.Console.console_settings_modal as modal_module

    for name in (
        "_GENERATION_CONTROL_INPUTS",
        "PROVIDER_CHOICE_NO_EFFECT_SUFFIX",
        "GENERATION_CONTROL_UNKNOWN_COPY",
        # Its domain-joined placeholders outlived their reader (TASK-33002.7).
        "PROVIDER_CHOICE_INPUTS",
    ):
        assert not hasattr(modal_module, name), name
    app = CoreFirstHarness()
    modal = _modal(app, _settings(provider, model, temperature=0.7))
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        hidden = _shared_hidden(app.app_config, provider, model)
        assert _hidden_rows(modal) == hidden
        expected = _expected_line(
            provider, hidden, provider_display_name(provider, app.app_config)
        )
        assert _sampling_title(modal) == expected
        assert _painted_title(app, modal).endswith(expected)
        # Invariant, not a mirror: a shown row is one Apply keeps, unless its
        # support is unknown (TASK-30012 AC#3).
        supported = supported_generation_fields(provider, model, app.app_config)
        for name in set(FIELD_ROW_FIELDS) - set(hidden) - supported:
            assert (
                console_generation_control_support(
                    provider, model, name, app.app_config
                )
                == "unknown"
            ), name


#: The provider-choice rows (Reasoning effort, Reasoning summary, Verbosity,
#: Thinking) each registry family shows (TASK-33003.13).
_CHOICE_ROWS = ("reasoning_effort", "reasoning_summary", "verbosity", "thinking_effort")


#: One registry entry per family: (slug, name, family, family_key, shown_choices).
_FAMILY_ROWS = (
    ("gpu-box", "GPU box", "llama_cpp", "llama_cpp", {"reasoning_effort"}),
    ("lab", "Lab", "ollama", "ollama", set()),
    ("gateway", "Gateway", "openai_compatible", "custom", {"reasoning_effort"}),
)


@pytest.mark.parametrize(
    ("slug", "name", "family", "family_key", "shown_choices"), _FAMILY_ROWS
)
@pytest.mark.asyncio
async def test_a_registry_endpoint_hides_what_its_family_hides(
    slug, name, family, family_key, shown_choices
) -> None:
    """AC#3 (review fix): a ``custom-ep`` entry is decided as its family, so
    a llama.cpp entry shows no control that Apply would then clear.

    TASK-33003.13, owner ruling 2026-10-09: a custom endpoint shows exactly
    the provider-choice rows its registry family supports, the rule the
    request follows; pinned over every family (``ENDPOINT_FAMILIES``)."""
    assert {row[2] for row in _FAMILY_ROWS} == ENDPOINT_FAMILIES
    app = CoreFirstHarness()
    app.app_config["custom_endpoints"].setdefault(
        slug,
        {
            "display_name": name,
            "family": family,
            "base_url": "http://192.168.1.9:8080",
            "models": ["model-a"],
        },
    )
    assert app.app_config["custom_endpoints"][slug]["family"] == family
    provider = f"custom-ep:{slug}"
    modal = _modal(app, _settings(provider, "model-a", temperature=0.7))
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        family_hidden = _shared_hidden(app.app_config, family_key, "model-a")
        assert {"reasoning_summary", "verbosity", "thinking_effort"} <= set(
            family_hidden
        )
        assert _hidden_rows(modal) == family_hidden
        assert set(_CHOICE_ROWS) - set(_hidden_rows(modal)) == shown_choices
        shown = set(FIELD_ROW_FIELDS) - set(family_hidden)
        assert shown <= supported_generation_fields(provider, "model-a", app.app_config)
        assert _sampling_title(modal) == _expected_line(family_key, family_hidden, name)


@pytest.mark.parametrize("name", ["Lab [gpu]", "Lab [/b] box"])
@pytest.mark.asyncio
async def test_a_bracketed_endpoint_name_paints_literally(name) -> None:
    """Review fix: a registry display name is user text, never markup."""
    app = CoreFirstHarness()
    app.app_config["custom_endpoints"]["lab"] = {
        "display_name": name,
        "family": "ollama",
        "base_url": "http://192.168.1.9:11434",
        "models": ["qwen3"],
    }
    modal = _modal(app, _settings("custom-ep:lab", "qwen3", temperature=0.7))
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        hidden = _hidden_rows(modal)
        assert hidden, "an ollama entry hides something to name"
        expected = _expected_line("ollama", hidden, name)
        assert name in expected
        assert _sampling_title(modal) == expected
        assert _painted_title(app, modal).endswith(expected)


#: Long enough to need the count form, short enough to name one field.
_ONE_FIELD = ("min_p",)


def test_the_sampling_title_names_what_fits_and_counts_the_rest() -> None:
    """Owner ruling (2026-10-02), amending AC#1-2: the closed title names the
    hidden fields only when the whole title fits one row; otherwise it
    counts them; a name too wide for even that is shortened."""
    assert hidden_fields_line("Ollama", ()) == "Sampling"
    assert hidden_fields_line("Ollama", _ONE_FIELD) == (
        "Sampling · hidden for Ollama: Min P (this provider does not accept them)"
    )
    assert hidden_fields_line("Together", SAMPLING_FIELDS + CORE_FIELDS[3:]) == (
        "Sampling · Together does not accept 11 fields (open to list them)"
    )
    wide = "実験" * 40  # 80 characters, the registry's display-name cap, 160 cells
    title = hidden_fields_line(wide, _ONE_FIELD)
    assert cell_len(title) == DISCLOSURE_TITLE_CELLS
    assert title.startswith("Sampling · 実験") and "…" in title
    assert title.endswith(" does not accept 1 field (open to list them)")


@pytest.mark.parametrize("cells", [90, 70, 66, 60, 50, 30])
@pytest.mark.parametrize("state", ["", "all inherit", "Top P 0.95 · others inherit"])
def test_a_narrow_sampling_title_never_overruns_its_budget(
    cells: int, state: str
) -> None:
    """Final review finding 8: when the count form's fixed part fills the
    budget, the title gives up the state, then the hidden-field part, rather
    than cut the provider name from its end and overrun (79 cells at 70)."""
    title = hidden_fields_line("Anthropic", _ANTHROPIC_HIDDEN, cells=cells, state=state)

    assert cell_len(title) <= cells, title
    assert title.startswith("Sampling")
    if " does not accept " in title:
        assert title.endswith(" does not accept 7 fields (open to list them)")
        assert " · Anthropic does" in title or " · Ant…" in title, title
    if cells in (70, 66):
        # The widths the review measured: the state goes, the name stays whole.
        assert title == (
            "Sampling · Anthropic does not accept 7 fields (open to list them)"
        )


def test_every_closed_sampling_title_fits_one_row() -> None:
    """Owner ruling census: for every PROVIDER_PARAM_MAP provider, with its
    shipped default model and with every field hidden (any model's worst
    case), the closed Sampling title measures at most 141 cells."""
    from tldw_chatbook.Chat.Chat_Functions import PROVIDER_PARAM_MAP
    from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML as config

    over: list[tuple[str, int]] = []
    for provider in PROVIDER_PARAM_MAP:
        model = (config.get("api_settings", {}).get(provider) or {}).get("model")
        name = provider_display_name(provider, config)
        hidden, _unknown = generation_field_support(provider, model, config)
        for fields in (hidden, FIELD_ROW_FIELDS):
            width = cell_len(hidden_fields_line(name, fields))
            if width > DISCLOSURE_TITLE_CELLS:
                over.append((provider, width))
    assert len(PROVIDER_PARAM_MAP) > 50
    assert over == []


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("anthropic", "claude-sonnet-4-5"),
        ("google", "gemini-2.5-pro"),
        ("custom-ep:gpu-box", "model-a"),
    ],
)
@pytest.mark.asyncio
async def test_the_opened_sampling_disclosure_names_every_hidden_field(
    provider, model
) -> None:
    """Owner ruling: opened with real keys, Sampling lists every hidden field
    by its field-table label, the ones its closed title only counts too."""
    app = CoreFirstHarness()
    modal = _modal(app, _settings(provider, model, temperature=0.7))
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        hidden = _hidden_rows(modal)
        assert hidden
        listing = modal.query_one(f"#{SAMPLING_HIDDEN_LIST_ID}", Static)
        assert not listing.region.area  # closed: nothing below the title
        modal.query_one(f"#{SAMPLING_DISCLOSURE_ID}").query_one(
            CollapsibleTitle
        ).focus()
        await pilot.press("enter")
        for _ in range(3):
            await pilot.pause()
        region = listing.region
        assert region.area
        painted = " ".join(
            " ".join(row[region.x : region.right].split())
            for row in _painted(app.screen)[region.y : region.bottom]
        )
        name = provider_display_name(provider, app.app_config)
        assert painted.startswith(f"{name} does not accept: ")
        for field in hidden:
            assert MODEL_FIELD_LABELS[field] in painted, field


@pytest.mark.asyncio
async def test_apply_succeeds_when_a_required_field_is_hidden() -> None:
    """Review fix (Critical 1): Custom OpenAI 2 does not accept Top P, so its
    row is hidden and the rebase commits it blank. Apply must not then demand
    a Top P the user cannot reach."""
    app = CoreFirstHarness()
    results: list[object] = []
    modal = _modal(app, _settings(), draft_rebaser=_real_rebase)
    async with app.run_test(size=(211, 44)) as pilot:
        await app.push_screen(modal, callback=results.append)
        for _ in range(4):
            await pilot.pause()
        assert "top_p" not in supported_generation_fields(
            "custom_2", "model-a", app.app_config
        )
        assert modal._rebase_to("custom_2", "model-a") is True
        for _ in range(3):
            await pilot.pause()
        assert "top_p" in _hidden_rows(modal)
        assert modal.query_one("#console-settings-top-p").value == ""
        modal.query_one("#console-settings-save", Button).focus()
        await pilot.press("enter")
        for _ in range(4):
            await pilot.pause()
        if app.screen is modal:
            error = modal.query_one("#console-settings-error", Static)
            raise AssertionError(f"Apply stayed open: {error.content}")
    assert len(results) == 1
    committed = results[0].live_commit.settings
    assert committed.top_p is None
    assert committed.temperature is not None


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("openai", "future-custom-model"),
        # No PROVIDER_PARAM_MAP entry: every field stays (TASK-30012 AC#3).
        ("unmapped-provider", "model-x"),
    ],
)
@pytest.mark.asyncio
async def test_unknown_support_stays_visible_with_neutral_copy_in_its_help_line(
    provider, model
) -> None:
    """AC#4: an unknown control stays visible and says so in its own help
    line; the per-row "Support not verified" statics are gone."""
    from tldw_chatbook.Chat.Chat_Functions import PROVIDER_PARAM_MAP

    app = CoreFirstHarness()
    modal = _modal(app, _settings(provider, model, temperature=0.7))
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        assert not modal.query(".console-settings-control-support")
        for name in FIELD_ROW_FIELDS:
            assert not modal.query(f"#{field_control_id(name)}-support"), name
        unknown = [
            name
            for name in _CONTROLS
            if console_generation_control_support(provider, model, name, app.app_config)
            == "unknown"
        ]
        assert unknown, "a new or unmapped model's controls are unknown"
        if provider not in PROVIDER_PARAM_MAP:
            assert _hidden_rows(modal) == ()
            assert _sampling_title(modal) == "Sampling"
            assert not modal.query_one(f"#{SAMPLING_HIDDEN_LIST_ID}").display
        painted = _painted(app.screen)
        for name in unknown:
            row = modal.query_one(f"#{field_control_id(name)}-row")
            assert row.display, name
            help_line = modal.query_one(f"#{field_control_id(name)}-help", Static)
            assert str(help_line.content).startswith(GENERATION_CONTROL_UNKNOWN_COPY)
            assert GENERATION_CONTROL_UNKNOWN_COPY in painted[row.region.y], name
            # Capture fix 3, finding 2 (08): the sentence after the full stop
            # starts with a capital ("… for this model. The provider decides").
            rest = str(help_line.content)[len(GENERATION_CONTROL_UNKNOWN_COPY) :]
            assert rest[1:2].isupper(), (name, rest)
            # TASK-33006.12: at 211x44 the marker leaves the field's own help
            # painted whole beside it, not cut off by the row's ellipsis.
            assert MODEL_CONFIG_FIELDS[name].help in painted[row.region.y], name
        # Known support keeps the field table's plain help. Rewritten on
        # purpose by TASK-33006.11: a provider with no request map has no
        # known support, so its samplers carry the neutral copy too.
        temperature_help = modal.query_one("#console-settings-temperature-help", Static)
        assert (GENERATION_CONTROL_UNKNOWN_COPY in str(temperature_help.content)) is (
            provider not in PROVIDER_PARAM_MAP
        )


@pytest.mark.parametrize("provider", ["local_onnx", "local_transformers"])
@pytest.mark.asyncio
async def test_a_provider_without_a_request_map_says_its_samplers_are_unverified(
    provider,
) -> None:
    """TASK-33006.11: local_onnx and local_transformers have no
    ``PROVIDER_PARAM_MAP`` entry (and no chat handler), so no request decides
    which samplers they take. Every row stays shown, and each one's help line
    leads with the neutral copy instead of reading as supported; at 211x44
    the field's own help still paints whole beside it."""
    from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS, PROVIDER_PARAM_MAP

    assert provider not in PROVIDER_PARAM_MAP
    assert provider not in API_CALL_HANDLERS
    app = CoreFirstHarness()
    modal = _modal(app, _settings(provider, "model-x", temperature=0.7))
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        assert _hidden_rows(modal) == ()
        assert generation_field_support(provider, "model-x", app.app_config) == (
            (),
            frozenset(FIELD_ROW_FIELDS),
        )
        painted = _painted(app.screen)
        for name in ("temperature", "max_tokens", "streaming"):
            row = modal.query_one(f"#{field_control_id(name)}-row")
            assert GENERATION_CONTROL_UNKNOWN_COPY in painted[row.region.y], name
            assert MODEL_CONFIG_FIELDS[name].help in painted[row.region.y], name
        for name in FIELD_ROW_FIELDS:
            help_line = modal.query_one(f"#{field_control_id(name)}-help", Static)
            text = str(help_line.content)
            assert text.startswith(GENERATION_CONTROL_UNKNOWN_COPY), (name, text)
            assert MODEL_CONFIG_FIELDS[name].help in text, name


#: Every Reasoning effort level, in display order.
_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh")


def _reasoning_options(modal) -> list[tuple[str, str]]:
    select = modal.query_one(f"#{field_control_id('reasoning_effort')}", Select)
    return [(str(label), value) for label, value in select._options[1:]]


@pytest.mark.parametrize(
    ("provider", "offered"),
    [
        ("llama_cpp", ("none", "low", "medium", "high", "xhigh")),
        ("custom-ep:gpu-box", ("none", "low", "medium", "high", "xhigh")),
        ("openai", _LEVELS),
    ],
)
@pytest.mark.asyncio
async def test_reasoning_effort_offers_only_the_levels_the_request_sends(
    provider, offered
) -> None:
    """TASK-33002.7 AC#1: a strict local template's request drops "minimal"
    (``build_local_thinking_payload_fields``), so Chat settings does not
    offer it, decided by ``reasoning_effort_values_sent`` as Settings'
    model-default list is; a llama.cpp registry entry follows its family."""
    app = CoreFirstHarness()
    modal = _modal(app, _settings(provider, "model-a", temperature=0.7))
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        assert offered == reasoning_effort_values_sent(
            provider, _LEVELS, app.app_config
        )
        assert [value for _label, value in _reasoning_options(modal)] == list(offered)


@pytest.mark.asyncio
async def test_a_saved_level_the_request_drops_stays_selected_and_says_so() -> None:
    """TASK-33002.7 AC#2: a "minimal" saved before stays selected, listed as
    not supported, and its row says why, rather than being cleared; Apply
    keeps it. A deliberate pick of a sent level clears the note."""
    app = CoreFirstHarness()
    settings = _settings("llama_cpp", "model-a", reasoning_effort="minimal")
    modal = _modal(app, settings)
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        select = modal.query_one("#console-settings-reasoning-effort", Select)
        assert select.value == "minimal"
        assert _reasoning_options(modal)[-1] == (
            f"Minimal {UNSENT_CHOICE_SUFFIX}",
            "minimal",
        )
        note = UNSENT_CHOICE_COPY.format(level="Minimal")
        help_line = modal.query_one("#console-settings-reasoning-effort-help", Static)
        assert str(help_line.content) == note
        row = modal.query_one("#console-settings-reasoning-effort-row")
        assert note in _painted(app.screen)[row.region.y]
        assert modal._build_draft().reasoning_effort == "minimal"

        select.value = "low"
        for _ in range(3):
            await pilot.pause()
        assert str(help_line.content) == MODEL_CONFIG_FIELDS["reasoning_effort"].help


@pytest.mark.asyncio
async def test_a_pick_re_lists_reasoning_effort_for_the_new_provider() -> None:
    """TASK-33002.7: the list follows the pair a pick lands on. A "minimal"
    picked on OpenAI and carried to llama.cpp stays, labelled; back on OpenAI
    it is an ordinary level again."""
    app = CoreFirstHarness()
    modal = _modal(app, _settings("openai", "gpt-4o"), draft_rebaser=_real_rebase)
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        select = modal.query_one("#console-settings-reasoning-effort", Select)
        select.value = "minimal"
        for _ in range(3):
            await pilot.pause()

        modal._model_picked(("llama_cpp", "model-a"))
        for _ in range(3):
            await pilot.pause()
        assert select.value == "minimal"
        assert [value for _label, value in _reasoning_options(modal)] == [
            "none",
            "low",
            "medium",
            "high",
            "xhigh",
            "minimal",
        ]
        assert str(
            modal.query_one("#console-settings-reasoning-effort-help", Static).content
        ) == UNSENT_CHOICE_COPY.format(level="Minimal")

        modal._model_picked(("openai", "gpt-4o"))
        for _ in range(3):
            await pilot.pause()
        assert [value for _label, value in _reasoning_options(modal)] == list(_LEVELS)
        assert UNSENT_CHOICE_SUFFIX not in str(_reasoning_options(modal))


@pytest.mark.asyncio
async def test_a_restored_draft_keeps_a_carried_level_the_request_drops() -> None:
    """TASK-33002.7: reopening after a credential round trip restores a
    "minimal" the draft carried onto llama.cpp; it stays selected and
    labelled, rather than raising InvalidSelectValueError on a list that no
    longer offers it."""
    from tldw_chatbook.Chat.console_context_policy import (
        ConsoleContextPolicyOverrides,
    )
    from tldw_chatbook.Widgets.Console.console_settings_modal import (
        ConsoleSettingsDraftSnapshot,
    )

    app = CoreFirstHarness()
    settings = _settings("llama_cpp", "model-a")
    snapshot = ConsoleSettingsDraftSnapshot(
        settings=settings,
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={"console-settings-reasoning-effort": "minimal"},
        provider_model_drafts={"llama_cpp": "model-a"},
        provider_base_url_drafts={},
        active_view="model",
        scroll_anchor=0,
        focus_control_id=None,
        disclosure_state={"advanced_generation": False, "connection_details": False},
    )
    modal = _modal(app, settings, suspended_draft=snapshot)
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        select = modal.query_one("#console-settings-reasoning-effort", Select)
        assert select.value == "minimal"
        assert _reasoning_options(modal)[-1] == (
            f"Minimal {UNSENT_CHOICE_SUFFIX}",
            "minimal",
        )
        help_line = modal.query_one("#console-settings-reasoning-effort-help", Static)
        assert str(help_line.content) == UNSENT_CHOICE_COPY.format(level="Minimal")


@pytest.mark.asyncio
async def test_changing_the_model_updates_the_hidden_set_and_line_at_once() -> None:
    """AC#6: the pick Change lands re-decides the hidden rows and the
    Sampling line in the same call, both ways.

    Re-pointed by TASK-33006.4 (AC#7) at the Change gesture: real keys open
    pick mode, and the pick lands through the controller's rebaser.
    """
    from Tests.UI.test_console_settings_model_change import pick, pick_modal

    app = CoreFirstHarness()
    modal = pick_modal(app, _settings())
    landed: list[tuple[tuple[str, ...], str]] = []
    model_picked = modal._model_picked

    def record_landing(pair) -> None:
        model_picked(pair)
        # No pause: the call that landed the pick already re-decided both.
        landed.append((_hidden_rows(modal), _sampling_title(modal)))

    modal._model_picked = record_landing
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        llama_hidden = _shared_hidden(app.app_config, "llama_cpp", "model-a")
        assert _hidden_rows(modal) == llama_hidden
        assert _sampling_title(modal) == _expected_line("llama_cpp", llama_hidden)

        await pick(pilot, app, modal, "claude-sonnet")
        assert landed[-1] == (
            _ANTHROPIC_HIDDEN,
            _expected_line("anthropic", _ANTHROPIC_HIDDEN),
        )
        assert _painted_title(app, modal).endswith(
            _expected_line("anthropic", _ANTHROPIC_HIDDEN)
        )

        await pick(pilot, app, modal, "model-a")
        assert landed[-1] == (llama_hidden, _expected_line("llama_cpp", llama_hidden))


@pytest.mark.asyncio
async def test_a_focused_field_that_becomes_hidden_hands_focus_on() -> None:
    """AC#1: a field hidden under focus leaves the Tab order at once, and
    focus goes to where the view opens (R13), not nowhere."""
    app = CoreFirstHarness()
    modal = _modal(app, _settings(), draft_rebaser=_real_rebase)
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        modal.query_one(f"#{SAMPLING_DISCLOSURE_ID}", Collapsible).collapsed = False
        for _ in range(3):
            await pilot.pause()
        seed = modal.query_one("#console-settings-seed")
        seed.focus()
        await pilot.pause()
        assert app.focused is seed
        assert modal._rebase_to("anthropic", "claude-sonnet-4-5") is True
        for _ in range(4):
            await pilot.pause()
        assert app.focused is modal.query_one("#console-settings-temperature")


#: Every generic request key a hidden Anthropic field could ride on.
_HIDDEN_REQUEST_KEYS = frozenset(
    {
        "minp",
        "min_p",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "reasoning_effort",
        "reasoning_summary",
        "verbosity",
    }
)


@pytest.mark.asyncio
@private_profile_test
async def test_applying_on_anthropic_submits_none_of_the_hidden_fields(
    request, tmp_path, monkeypatch
) -> None:
    """AC#7: the real Console opens Chat settings on an Anthropic chat whose
    defaults carry Min P, Seed, the penalties and reasoning values. Apply
    commits through the controller with none of them, and the next send hands
    the Anthropic handler none of them, while a shown field (Temperature)
    still goes."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_console_provider_apply_defaults_flow import (
        _ConsoleFlowHarness,
        _drain_settings_tasks,
    )
    from Tests.UI.test_destination_shells import _wait_for_selector
    from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.UI.Screens.settings_config_adapter import (
        SettingsConfigAdapter,
    )
    from tldw_chatbook.Widgets.Console import ConsoleComposerBar
    from tldw_chatbook.Widgets.Console.console_settings_modal import (
        ConsoleSettingsModal,
    )

    assert SettingsConfigAdapter().save_sections(
        {
            "first_run": {"setup_completed": True},
            "chat_defaults": {
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "temperature": 0.6,
                "min_p": 0.05,
                "seed": 7,
                "presence_penalty": 0.5,
                "frequency_penalty": 0.3,
                "reasoning_effort": "high",
                "verbosity": "high",
            },
            "api_settings.anthropic": {
                "api_key": "sk-ant-test-hidden-fields-0000",
                "model": "claude-sonnet-4-5",
            },
        }
    )
    app = _build_test_app()
    # File-backed: a send commits its trace from a worker thread, and each
    # thread's ":memory:" connection is an empty database of its own.
    app.chachanotes_db = CharactersRAGDB(
        tmp_path / "chachanotes.sqlite", client_id="hidden-fields"
    )
    app.chat_api_provider_value = "anthropic"
    app.chat_api_model_value = "claude-sonnet-4-5"
    app.providers_models = {"anthropic": ["claude-sonnet-4-5"]}
    captured: list[dict] = []

    def anthropic_handler(**kwargs):
        captured.append(kwargs)
        return "hidden fields reply"

    monkeypatch.setitem(API_CALL_HANDLERS, "anthropic", anthropic_handler)
    harness = _ConsoleFlowHarness(app)
    async with harness.run_test(size=(211, 44)) as pilot:
        console = harness.screen
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        before = store.session_settings(session_id)
        assert (before.provider, before.min_p, before.seed) == ("anthropic", 0.05, 7)

        assert await console._open_console_settings() is True
        modal = None
        for _ in range(80):
            if isinstance(harness.screen, ConsoleSettingsModal):
                modal = harness.screen
                break
            await pilot.pause(0.05)
        assert modal is not None
        for _ in range(4):
            await pilot.pause()
        assert _hidden_rows(modal) == _ANTHROPIC_HIDDEN
        assert _sampling_title(modal) == _expected_line("anthropic", _ANTHROPIC_HIDDEN)
        modal.query_one("#console-settings-save", Button).press()
        for _ in range(80):
            if harness.screen is console:
                break
            await pilot.pause(0.05)
        assert harness.screen is console
        await _drain_settings_tasks(app)
        # What Apply committed for the chat: none of the hidden fields.
        after = store.session_settings(session_id)
        assert after.provider == "anthropic"
        for name in _ANTHROPIC_HIDDEN:
            assert getattr(after, name) is None, name

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")
        await pilot.pause(0.1)
        console.query_one("#console-send-message", Button).press()
        deadline = time.monotonic() + 30.0
        while not captured and time.monotonic() < deadline:
            await pilot.pause(0.05)

    assert captured, "the Anthropic handler was never called"
    sent = captured[-1]
    assert not _HIDDEN_REQUEST_KEYS & set(sent), sorted(sent)
    assert sent["temp"] == pytest.approx(0.6)
    assert sent["model"] == "claude-sonnet-4-5"
