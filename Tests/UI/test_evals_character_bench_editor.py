"""Character bench editor (task-1691 phase 2, Task 4).

``CharacterBenchEditor`` edits a stored character-probe bench: name,
description, character selection (via ``card_picker.CardPicker``), and the
four sampler fields. The probe set and target list are read-only in this
task -- see ``character_bench_editor.py``'s own module docstring for why.

Task 5 (not yet implemented) is what wires ``EvalsScreen``'s selection
routing to mount this widget for a real ``"character_bench"`` selection.
Until then, ``select_bench`` below mounts it directly into the real
``EvalsScreen``'s ``#evals-detail-pane`` -- the real screen, the real
bundled stylesheet, the real bounded-height ancestor chain
``#evals-cb-save``'s own reachability depends on -- rather than routing
through ``EvalsScreen.select()``'s kind dispatch, which has no branch for
this bench type yet. ``EvalsHarness``/``_FakeAppInstance`` are imported
from ``test_evals_screen.py`` rather than re-implemented, per this phase's
own "reuse the existing harness" convention (see ``test_evals_bench_
editor.py``'s identical import for the word-bench editor's own tests).

**The nested-scroll risk.** ``CardPicker``'s own row list
(``#evals-card-picker-rows``) is bounded (``max-height: 10``) and
independently scrollable -- proven reachable on its own in
``test_evals_card_picker.py``, mounted directly in a bare host. Task 4's
own open risk is whether that still holds once the picker sits INSIDE this
editor's own scrolling pane (``#evals-character-bench-editor``, itself
``overflow-y: auto``) rather than a bare host with nothing above it.
``test_a_card_past_the_row_cap_is_reachable_inside_the_editors_own_scroll``
answers this empirically, with more cards than the cap, exactly as
`test_evals_card_picker.py`'s own
``test_a_card_beyond_the_bounded_row_list_is_reachable_by_scrolling`` does
for the standalone case -- and it passes with NO structural change beyond
the plain ``overflow-y: auto`` on ``#evals-character-bench-editor`` itself
(see that selector's own CSS comment in ``_evals.tcss`` for why: this
widget is the sole selection-kind-specific child of ``#evals-detail-pane``,
a REAL bounded height, with nothing after it to starve -- the same shape
``#evals-bench-editor`` already has, not ``#evals-inspector-bench``'s
ambiguous-``auto``-ancestor shape, which is the one that needs `height:
auto` too).

**Real widgets, not `.value=` assignment.** Every test below that edits a
field types through ``pilot.press`` after a real ``pilot.click`` -- setting
``Input.value`` directly is used ONLY to clear a field as setup before
typing the actual edit under test, per this phase's own rule (a checkbox
no user could toggle, and steering that never reached the model, both
shipped past tests that asserted a widget's internal state rather than
driving it).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from textual.widgets import Button, Input

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.character_probe.models import (
    CharacterProbeConfig,
    Probe,
    ProbeSet,
)
from tldw_chatbook.Evals.character_probe.storage import (
    load_character_bench,
    save_character_bench,
    save_probe_set,
)
from tldw_chatbook.Evals.character_probe.tags import Tag
from tldw_chatbook.UI.Evals.character_bench_editor import (
    MAX_TOKENS_ERROR_TEXT,
    SAMPLES_ERROR_TEXT,
    SEED_ERROR_TEXT,
    TEMPERATURE_ERROR_TEXT,
    CharacterBenchEditor,
)
from Tests.UI.test_evals_screen import EvalsHarness, _FakeAppInstance

#: Mirrors test_evals_card_picker.py's own CARDS fixture exactly -- row
#: index 1 ("#evals-card-row-1") is Marlow, id 7, so
#: test_selecting_cards_in_the_picker_persists_on_save's own click target
#: and asserted id agree with that established convention rather than
#: inventing a second one.
CARDS: list[dict[str, Any]] = [
    {"id": 3, "name": "Vex"},
    {"id": 7, "name": "Marlow"},
    {"id": 9, "name": "vexing puzzle"},
]

_REALISTIC_SIZE = (160, 45)
_WIDE_SIZE = (235, 52)


async def select_bench(
    pilot, bench_id: str, cards: Sequence[Mapping[str, Any]] = CARDS
) -> None:
    """Mounts ``CharacterBenchEditor`` directly into the real
    ``EvalsScreen``'s ``#evals-detail-pane`` -- see the module docstring
    for why this bypasses ``EvalsScreen.select()``'s kind dispatch (Task 5
    is what adds a real branch for this bench type).

    ``await pilot.pause()`` first: ``LabScreen`` mounts its body from
    ``call_after_refresh`` so first paint is not blocked (see
    ``lab_frame.py``'s own comment on ``build_lab_body``) -- ``#evals-
    detail-pane`` does not exist in the DOM until that deferred mount has
    run at least one event-loop turn.
    """
    await pilot.pause()
    screen = pilot.app.screen
    detail_pane = screen.query_one("#evals-detail-pane")
    await detail_pane.remove_children()
    await detail_pane.mount(
        CharacterBenchEditor(
            screen._view_model, bench_id, cards, id="evals-character-bench-editor"
        )
    )


async def _click_after_scroll(pilot, selector: str) -> None:
    """Scrolls ``selector``'s own widget into ``#evals-character-bench-
    editor``'s viewport before clicking it, then clicks.

    This pane's fields do not all fit unscrolled at a realistic 160x45
    terminal even in the base (one target, one probe, one card) case --
    the ``CardPicker`` alone (its own bordered search ``Input`` plus row
    list) is real vertical weight ``bench_editor.py``'s own word-bench form
    never had to budget for. Mirrors ``test_evals_bench_editor.py``'s own
    established ``scroll_visible``-before-click convention for its
    (rarer, many-targets-only) tall-form case, applied here to every field
    below the first couple, not only Save.
    """
    widget = pilot.app.screen.query_one(selector)
    widget.scroll_visible(animate=False)
    await pilot.pause()
    await pilot.click(selector)


@pytest.fixture
def evals_db() -> EvalsDB:
    return EvalsDB(db_path=":memory:", client_id="test")


@pytest.fixture
def character_app(evals_db: EvalsDB) -> EvalsHarness:
    return EvalsHarness(_FakeAppInstance(evals_db))


def _make_probe_set(evals_db: EvalsDB, name: str = "villain probe set") -> str:
    """A probe set with one probe whose turn carries a LEADING space --
    this is what ``test_the_probe_listing_shows_whitespace_markers`` needs
    to find. A byte-exact prompt's leading whitespace is meaningful (it
    changes what is literally sent to the model), so this is a realistic
    probe, not merely a test fixture contrivance."""
    return save_probe_set(
        evals_db,
        name,
        ProbeSet(probes=(Probe(turns=(" Sure, I can help with that.",)),)),
    )


@pytest.fixture
def saved_bench_id(evals_db: EvalsDB) -> str:
    target_id = evals_db.create_model(
        name="local-target", provider="llama_cpp", model_id="m"
    )
    probe_set_id = _make_probe_set(evals_db)
    config = CharacterProbeConfig(
        name="villain probes",
        description="",
        probe_set_id=probe_set_id,
        character_ids=(3,),
        target_ids=(target_id,),
        samples_per_cell=1,
    )
    return save_character_bench(evals_db, config)


# ---------------------------------------------------------------------------
# Read: the stored bench renders into the form
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_editor_renders_the_stored_bench(character_app, saved_bench_id):
    async with character_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await select_bench(pilot, saved_bench_id)
        await pilot.pause()
        screen = pilot.app.screen
        assert screen.query_one("#evals-cb-name", Input).value == "villain probes"
        assert screen.query_one("#evals-cb-samples", Input).value == "1"
        assert screen.query_one("#evals-cb-seed", Input).value == ""
        assert screen.query_one("#evals-cb-temperature", Input).value == "0.8"
        assert screen.query_one("#evals-cb-max-tokens", Input).value == "512"


@pytest.mark.asyncio
async def test_probe_set_line_and_target_listing_render_read_only_state(
    character_app, saved_bench_id
):
    """Neither field has an edit control in this task -- both must still
    be visible so a user can see what a Save will carry through verbatim
    (see the module docstring's read-only-in-this-task paragraph)."""
    async with character_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await select_bench(pilot, saved_bench_id)
        await pilot.pause()
        screen = pilot.app.screen
        probe_set_line = screen.query_one("#evals-cb-probe-set")
        assert "villain probe set" in str(probe_set_line.render())
        assert "1 probes" in str(probe_set_line.render())

        target_row = screen.query_one("#evals-cb-target-0")
        assert "local-target" in str(target_row.render())
        assert "llama_cpp" in str(target_row.render())


# ---------------------------------------------------------------------------
# Save: real typed edits persist
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_saving_persists_every_edited_field(
    character_app, saved_bench_id, evals_db
):
    async with character_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await select_bench(pilot, saved_bench_id)
        await pilot.pause()
        screen = pilot.app.screen

        name = screen.query_one("#evals-cb-name", Input)
        await _click_after_scroll(pilot, "#evals-cb-name")
        name.value = ""  # setup: clear before typing the real edit
        await pilot.press(*"renamed")

        samples = screen.query_one("#evals-cb-samples", Input)
        await _click_after_scroll(pilot, "#evals-cb-samples")
        samples.value = ""
        await pilot.press("3")

        # seed starts blank (saved_bench_id's own config has seed=None) --
        # no setup-clear needed before typing.
        await _click_after_scroll(pilot, "#evals-cb-seed")
        await pilot.press("-", "1")

        temperature = screen.query_one("#evals-cb-temperature", Input)
        await _click_after_scroll(pilot, "#evals-cb-temperature")
        temperature.value = ""
        await pilot.press("1", ".", "2")

        max_tokens = screen.query_one("#evals-cb-max-tokens", Input)
        await _click_after_scroll(pilot, "#evals-cb-max-tokens")
        max_tokens.value = ""
        await pilot.press("6", "4")

        await _click_after_scroll(pilot, "#evals-cb-save")
        await pilot.pause()

        assert not screen.query_one("#evals-cb-form-error").display

        stored = load_character_bench(evals_db, saved_bench_id)
        assert stored.name == "renamed"
        assert stored.samples_per_cell == 3
        assert stored.seed == -1
        assert stored.temperature == pytest.approx(1.2)
        assert stored.max_tokens == 64


@pytest.mark.asyncio
async def test_an_invalid_samples_value_renders_the_error_and_keeps_typed_state(
    character_app, saved_bench_id
):
    async with character_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await select_bench(pilot, saved_bench_id)
        await pilot.pause()
        screen = pilot.app.screen

        samples = screen.query_one("#evals-cb-samples", Input)
        await _click_after_scroll(pilot, "#evals-cb-samples")
        samples.value = ""
        await pilot.press("0")

        name = screen.query_one("#evals-cb-name", Input)
        await _click_after_scroll(pilot, "#evals-cb-name")
        name.value = ""
        await pilot.press(*"typed-but-not-saved")

        await _click_after_scroll(pilot, "#evals-cb-save")
        await pilot.pause()

        error = screen.query_one("#evals-cb-form-error")
        assert error.display
        assert SAMPLES_ERROR_TEXT in str(error.render())
        assert screen.query_one("#evals-cb-name", Input).value == "typed-but-not-saved"


@pytest.mark.asyncio
async def test_an_invalid_seed_value_renders_the_pinned_error(
    character_app, saved_bench_id
):
    async with character_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await select_bench(pilot, saved_bench_id)
        await pilot.pause()
        screen = pilot.app.screen

        seed = screen.query_one("#evals-cb-seed", Input)
        await _click_after_scroll(pilot, "#evals-cb-seed")
        seed.value = ""
        await pilot.press(*"not-a-number")

        await _click_after_scroll(pilot, "#evals-cb-save")
        await pilot.pause()

        error = screen.query_one("#evals-cb-form-error")
        assert error.display
        assert SEED_ERROR_TEXT in str(error.render())


@pytest.mark.asyncio
async def test_a_negative_temperature_renders_the_pinned_error(
    character_app, saved_bench_id
):
    async with character_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await select_bench(pilot, saved_bench_id)
        await pilot.pause()
        screen = pilot.app.screen

        temperature = screen.query_one("#evals-cb-temperature", Input)
        await _click_after_scroll(pilot, "#evals-cb-temperature")
        temperature.value = ""
        await pilot.press("-", "1")

        await _click_after_scroll(pilot, "#evals-cb-save")
        await pilot.pause()

        error = screen.query_one("#evals-cb-form-error")
        assert error.display
        assert TEMPERATURE_ERROR_TEXT in str(error.render())


@pytest.mark.asyncio
async def test_a_negative_max_tokens_renders_the_pinned_error(
    character_app, saved_bench_id
):
    async with character_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await select_bench(pilot, saved_bench_id)
        await pilot.pause()
        screen = pilot.app.screen

        max_tokens = screen.query_one("#evals-cb-max-tokens", Input)
        await _click_after_scroll(pilot, "#evals-cb-max-tokens")
        max_tokens.value = ""
        await pilot.press("-", "1")

        await _click_after_scroll(pilot, "#evals-cb-save")
        await pilot.pause()

        error = screen.query_one("#evals-cb-form-error")
        assert error.display
        assert MAX_TOKENS_ERROR_TEXT in str(error.render())


@pytest.mark.asyncio
async def test_saving_with_a_taken_name_renders_the_conflict_callout(
    character_app, saved_bench_id, evals_db
):
    target_id = evals_db.create_model(
        name="other-target", provider="llama_cpp", model_id="m2"
    )
    other_probe_set_id = _make_probe_set(evals_db, name="other probe set")
    save_character_bench(
        evals_db,
        CharacterProbeConfig(
            name="taken-name",
            probe_set_id=other_probe_set_id,
            character_ids=(9,),
            target_ids=(target_id,),
        ),
    )
    async with character_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await select_bench(pilot, saved_bench_id)
        await pilot.pause()
        screen = pilot.app.screen

        name = screen.query_one("#evals-cb-name", Input)
        await _click_after_scroll(pilot, "#evals-cb-name")
        name.value = ""
        await pilot.press(*"taken-name")

        await _click_after_scroll(pilot, "#evals-cb-save")
        await pilot.pause()

        error = screen.query_one("#evals-cb-form-error")
        assert error.display
        assert "already exists" in str(error.render())

        # Unsaved: the original bench must still hold its original name.
        stored = load_character_bench(evals_db, saved_bench_id)
        assert stored.name == "villain probes"


@pytest.mark.asyncio
async def test_saving_carries_concurrency_and_extra_tags_through_verbatim(
    character_app, evals_db
):
    """Neither field has a UI control in this task -- a save must not
    silently reset either back to a dataclass default, the exact
    ``capture_continuations``-reset defect ``bench_editor.py``'s own
    ``_on_save_pressed`` comment records having shipped once for a word
    bench's ``concurrency``."""
    target_id = evals_db.create_model(
        name="local-target", provider="llama_cpp", model_id="m"
    )
    probe_set_id = _make_probe_set(evals_db)
    bench_id = save_character_bench(
        evals_db,
        CharacterProbeConfig(
            name="tagged bench",
            probe_set_id=probe_set_id,
            character_ids=(3,),
            target_ids=(target_id,),
            concurrency=4,
            extra_tags=({"slug": "villain", "kind": "notable"},),
        ),
    )
    async with character_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await select_bench(pilot, bench_id)
        await pilot.pause()
        await _click_after_scroll(pilot, "#evals-cb-save")
        await pilot.pause()

        stored = load_character_bench(evals_db, bench_id)
        assert stored.concurrency == 4
        assert stored.extra_tags == (Tag("villain", "villain", "notable"),)


# ---------------------------------------------------------------------------
# The CardPicker: selection persists on Save
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_selecting_cards_in_the_picker_persists_on_save(
    character_app, saved_bench_id, evals_db
):
    async with character_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await select_bench(pilot, saved_bench_id)
        await pilot.pause()
        await _click_after_scroll(pilot, "#evals-card-row-1")
        await _click_after_scroll(pilot, "#evals-cb-save")
        await pilot.pause()
        stored = load_character_bench(evals_db, saved_bench_id)
        assert 7 in stored.character_ids
        assert all(isinstance(cid, int) for cid in stored.character_ids)


@pytest.mark.asyncio
async def test_deselecting_every_card_renders_the_construction_error(
    character_app, saved_bench_id
):
    """``CharacterProbeConfig.__post_init__`` rejects an empty
    ``character_ids`` -- Save must surface that as the same in-place
    callout every other validation failure uses, not an uncaught crash."""
    async with character_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await select_bench(pilot, saved_bench_id)
        await pilot.pause()
        # saved_bench_id starts with character_ids=(3,) -- row 0 is Vex
        # (id 3, see CARDS); clicking it deselects the bench's only card.
        await _click_after_scroll(pilot, "#evals-card-row-0")
        await _click_after_scroll(pilot, "#evals-cb-save")
        await pilot.pause()
        error = pilot.app.screen.query_one("#evals-cb-form-error")
        assert error.display
        assert "at least one character" in str(error.render())


# ---------------------------------------------------------------------------
# The probe listing: whitespace markers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_probe_listing_shows_whitespace_markers(character_app, saved_bench_id):
    """Probe turns are byte-exact prompts; leading spaces must be visible."""
    async with character_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await select_bench(pilot, saved_bench_id)
        await pilot.pause()
        listing = pilot.app.screen.query_one("#evals-cb-probes").render()
        assert "␣" in str(listing)


@pytest.mark.asyncio
async def test_a_missing_probe_set_degrades_inline_without_crashing(
    character_app, evals_db
):
    """A probe set deleted after a bench was created must not take the
    whole editor down with it -- the bench's OTHER fields (name, sampler)
    stay editable around the dangling reference, mirroring how a deleted
    TARGET degrades inline elsewhere in this same widget."""
    target_id = evals_db.create_model(
        name="local-target", provider="llama_cpp", model_id="m"
    )
    bench_id = save_character_bench(
        evals_db,
        CharacterProbeConfig(
            name="orphaned bench",
            probe_set_id="does-not-exist",
            character_ids=(3,),
            target_ids=(target_id,),
        ),
    )
    async with character_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await select_bench(pilot, bench_id)
        await pilot.pause()
        screen = pilot.app.screen
        assert "not found" in str(screen.query_one("#evals-cb-probe-set").render())
        assert "unavailable" in str(screen.query_one("#evals-cb-probes").render())
        # The rest of the form is still there and still editable.
        assert screen.query_one("#evals-cb-name", Input).value == "orphaned bench"
        assert screen.query_one("#evals-cb-save") is not None


@pytest.mark.asyncio
async def test_a_deleted_target_renders_unresolvable_without_crashing(
    character_app, evals_db
):
    bench_id = save_character_bench(
        evals_db,
        CharacterProbeConfig(
            name="dangling target bench",
            probe_set_id=_make_probe_set(evals_db),
            character_ids=(3,),
            target_ids=("does-not-exist",),
        ),
    )
    async with character_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await select_bench(pilot, bench_id)
        await pilot.pause()
        row = pilot.app.screen.query_one("#evals-cb-target-0")
        assert "unresolvable" in str(row.render())


# ---------------------------------------------------------------------------
# Whole-branch review Critical 1 (fix round): a steered target's row is
# unreachable through this bench type's normal creation path now (see
# test_evals_screen.py's own steered-target coverage), but a pre-existing
# or hand-authored bench can still carry one -- this listing must surface
# that steering, mirroring bench_editor.py's `_build_target_row`, rather
# than rendering the same bare "name (provider)" it did for every target
# regardless of steering.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_prefix_steered_target_renders_as_unusable(character_app, evals_db):
    target_id = evals_db.create_model(
        name="steered-target",
        provider="llama_cpp",
        model_id="m",
        config={"prefix": " Sure"},
    )
    bench_id = save_character_bench(
        evals_db,
        CharacterProbeConfig(
            name="prefix steered bench",
            probe_set_id=_make_probe_set(evals_db),
            character_ids=(3,),
            target_ids=(target_id,),
        ),
    )
    async with character_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await select_bench(pilot, bench_id)
        await pilot.pause()
        rendered = str(pilot.app.screen.query_one("#evals-cb-target-0").render())
        assert "steered-target" in rendered
        assert "prefix" in rendered.lower()
        assert "unusable" in rendered.lower()


@pytest.mark.asyncio
async def test_a_system_prompt_steered_target_renders_its_steering(
    character_app, evals_db
):
    target_id = evals_db.create_model(
        name="steered-target",
        provider="llama_cpp",
        model_id="m",
        config={"system_prompt": "Be extra dramatic at all times."},
    )
    bench_id = save_character_bench(
        evals_db,
        CharacterProbeConfig(
            name="system prompt steered bench",
            probe_set_id=_make_probe_set(evals_db),
            character_ids=(3,),
            target_ids=(target_id,),
        ),
    )
    async with character_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await select_bench(pilot, bench_id)
        await pilot.pause()
        rendered = str(pilot.app.screen.query_one("#evals-cb-target-0").render())
        assert "steered-target" in rendered
        assert "system prompt" in rendered.lower()


@pytest.mark.asyncio
async def test_an_unsteered_target_renders_with_no_steering_suffix(
    character_app, saved_bench_id
):
    """`saved_bench_id`'s own target (`local-target`, no `config`) must
    keep rendering the bare `name (provider)` label -- no steering suffix
    of any kind for a genuinely unsteered row."""
    async with character_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await select_bench(pilot, saved_bench_id)
        await pilot.pause()
        rendered = str(pilot.app.screen.query_one("#evals-cb-target-0").render())
        assert "local-target (llama_cpp)" in rendered
        assert "·" not in rendered


# ---------------------------------------------------------------------------
# Revert
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_revert_asks_the_screen_to_reselect_this_bench(
    character_app, saved_bench_id
):
    """``#evals-cb-revert`` must call the screen's own ``select()`` with
    this bench's id -- Task 5's own eventual job is making that call
    actually reload the form; this only pins that the button reaches for
    the right lever."""
    async with character_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await select_bench(pilot, saved_bench_id)
        await pilot.pause()
        screen = pilot.app.screen

        calls: list[tuple[str, str | None]] = []
        original_select = screen.select

        def _spy(*, kind, id=None):  # noqa: A002 -- mirrors select()'s own param name
            calls.append((kind, id))
            original_select(kind=kind, id=id)

        screen.select = _spy
        try:
            await _click_after_scroll(pilot, "#evals-cb-revert")
            await pilot.pause()
        finally:
            del screen.select

        assert ("character_bench", saved_bench_id) in calls


# ---------------------------------------------------------------------------
# Painted geometry -- Save must stay hit-testable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [_REALISTIC_SIZE, _WIDE_SIZE])
async def test_save_stays_hit_testable_at_realistic_sizes(character_app, saved_bench_id, size):
    """This pane has pushed a control out of reach three times (task-1764,
    in the sibling word-bench editor) -- proven here, not merely assumed,
    for this new bench type's own editor."""
    async with character_app.run_test(size=size) as pilot:
        await select_bench(pilot, saved_bench_id)
        await pilot.pause()
        editor = pilot.app.screen.query_one("#evals-character-bench-editor")
        editor.scroll_end(animate=False)
        await pilot.pause()
        save = pilot.app.screen.query_one("#evals-cb-save")
        hit, _ = pilot.app.screen.get_widget_at(*save.region.center)
        assert hit is save or save in hit.ancestors


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [_REALISTIC_SIZE, _WIDE_SIZE])
async def test_a_card_past_the_row_cap_is_reachable_inside_the_editors_own_scroll(
    character_app, saved_bench_id, size
):
    """The open risk this task's brief flags directly: ``CardPicker``'s own
    row list (bounded ``max-height: 10``) nested inside THIS widget's own
    ``overflow-y: auto`` scroll must not strand a row past the fold. 60
    cards (well over the cap) mirrors ``test_evals_card_picker.py``'s own
    identical proof for the standalone picker; this is the same proof one
    level deeper, inside a real detail pane."""
    many_cards = [{"id": i, "name": f"Card {i}"} for i in range(60)]
    async with character_app.run_test(size=size) as pilot:
        await select_bench(pilot, saved_bench_id, cards=many_cards)
        await pilot.pause()
        screen = pilot.app.screen

        editor = screen.query_one("#evals-character-bench-editor")
        rows_container = screen.query_one("#evals-card-picker-rows")
        last_row = screen.query_one("#evals-card-row-59", Button)

        # First bring the picker's own row list into the editor's visible
        # viewport (it may start below the fold at a short terminal), then
        # scroll the row list's OWN bounded container to its end -- two
        # independent scrolls, the nested-container risk this test exists
        # to prove actually compose correctly together.
        editor.scroll_to_widget(rows_container, animate=False)
        await pilot.pause()
        rows_container.scroll_end(animate=False, force=True)
        await pilot.pause()

        try:
            hit, _ = screen.get_widget_at(*last_row.region.center)
        except Exception:  # NoWidget -- treated as "not reachable" below
            hit = None
        assert hit is last_row, (
            f"card 59 not hit-testable inside the bench editor's own scroll "
            f"at {size} -- landed on {hit!r} instead"
        )


# ---------------------------------------------------------------------------
# is_dirty() -- task-1691 phase 2 Task 6 review round 1 (Important finding):
# EvalsScreen._selection_unmoved_since_launch queried ONLY BenchEditor's own
# is_dirty(), leaving a completing character-bench run free to silently
# discard an unsaved edit in THIS editor. Mirrors BenchEditor.is_dirty()'s
# own contract/tests, per the reviewer's explicit instruction not to invent
# a second one.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_dirty_flips_true_independently_for_each_scalar_field(
    character_app, saved_bench_id
):
    """Name/Description/Samples/Seed/Temperature/Max-tokens each flip
    ``is_dirty()`` to True on their own -- exercised one field at a time
    against a freshly re-selected (clean) editor, so an earlier field's
    edit can never mask a later one's assertion. Real typing throughout
    (this module's own "Real widgets, not `.value=` assignment" rule);
    ``.value = ""`` appears only as the established setup-clear."""
    async with character_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await select_bench(pilot, saved_bench_id)
        await pilot.pause()
        await _click_after_scroll(pilot, "#evals-cb-name")
        pilot.app.screen.query_one("#evals-cb-name", Input).value = ""
        await pilot.press(*"renamed")
        assert pilot.app.screen.query_one(CharacterBenchEditor).is_dirty() is True

        await select_bench(pilot, saved_bench_id)
        await pilot.pause()
        await _click_after_scroll(pilot, "#evals-cb-description")
        await pilot.press(*"new description")
        assert pilot.app.screen.query_one(CharacterBenchEditor).is_dirty() is True

        await select_bench(pilot, saved_bench_id)
        await pilot.pause()
        await _click_after_scroll(pilot, "#evals-cb-samples")
        pilot.app.screen.query_one("#evals-cb-samples", Input).value = ""
        await pilot.press("3")
        assert pilot.app.screen.query_one(CharacterBenchEditor).is_dirty() is True

        await select_bench(pilot, saved_bench_id)
        await pilot.pause()
        await _click_after_scroll(pilot, "#evals-cb-seed")
        await pilot.press("-", "1")
        assert pilot.app.screen.query_one(CharacterBenchEditor).is_dirty() is True

        await select_bench(pilot, saved_bench_id)
        await pilot.pause()
        await _click_after_scroll(pilot, "#evals-cb-temperature")
        pilot.app.screen.query_one("#evals-cb-temperature", Input).value = ""
        await pilot.press("1", ".", "2")
        assert pilot.app.screen.query_one(CharacterBenchEditor).is_dirty() is True

        await select_bench(pilot, saved_bench_id)
        await pilot.pause()
        await _click_after_scroll(pilot, "#evals-cb-max-tokens")
        pilot.app.screen.query_one("#evals-cb-max-tokens", Input).value = ""
        await pilot.press("6", "4")
        assert pilot.app.screen.query_one(CharacterBenchEditor).is_dirty() is True

        # An unparseable numeric value counts as dirty too -- matches
        # Save's own treatment of that value as a real, if invalid, edit.
        await select_bench(pilot, saved_bench_id)
        await pilot.pause()
        await _click_after_scroll(pilot, "#evals-cb-samples")
        pilot.app.screen.query_one("#evals-cb-samples", Input).value = ""
        await pilot.press(*"nan")
        assert pilot.app.screen.query_one(CharacterBenchEditor).is_dirty() is True


@pytest.mark.asyncio
async def test_is_dirty_flips_true_when_a_card_selection_changes(
    character_app, saved_bench_id
):
    """``saved_bench_id`` starts with only Vex (id 3, row 0) selected --
    clicking Marlow's row (id 7, row 1) is a real character-selection edit,
    not just a form-field edit, and ``is_dirty()`` must catch it too."""
    async with character_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await select_bench(pilot, saved_bench_id)
        await pilot.pause()
        assert pilot.app.screen.query_one(CharacterBenchEditor).is_dirty() is False

        await _click_after_scroll(pilot, "#evals-card-row-1")
        await pilot.pause()
        assert pilot.app.screen.query_one(CharacterBenchEditor).is_dirty() is True


@pytest.mark.asyncio
async def test_is_dirty_treats_a_reordered_but_unchanged_character_selection_as_clean(
    character_app, evals_db
):
    """``CardPicker.selected_ids()`` returns ids in CARD-LIST order (``
    CARDS``' own order here), not the order they happen to be stored in
    ``config.character_ids`` -- a bench saved with ``character_ids=(9, 3)``
    (id 9, "vexing puzzle", stored BEFORE id 3, "Vex") reads back through
    the picker as ``(3, 9)`` (``CARDS``' own order: Vex is index 0, "vexing
    puzzle" is index 2) even though nothing was ever edited. ``is_dirty()``
    must not manufacture a false positive out of this reordering -- see its
    own docstring for why the comparison is a SET, not a tuple, unlike
    ``BenchEditor.is_dirty()``'s order-sensitive staged-target comparison.
    A naive tuple comparison here would report a freshly-selected, never-
    touched bench as dirty forever, permanently blocking auto-navigate for
    every run of it."""
    probe_set_id = _make_probe_set(evals_db, name="reorder probe set")
    target_id = evals_db.create_model(name="t", provider="llama_cpp", model_id="m")
    config = CharacterProbeConfig(
        name="reorder-check bench",
        probe_set_id=probe_set_id,
        character_ids=(9, 3),
        target_ids=(target_id,),
    )
    bench_id = save_character_bench(evals_db, config)

    async with character_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await select_bench(pilot, bench_id)
        await pilot.pause()
        editor = pilot.app.screen.query_one(CharacterBenchEditor)
        assert editor.is_dirty() is False


@pytest.mark.asyncio
async def test_is_dirty_is_false_again_after_save(character_app, saved_bench_id):
    """Save -> ``Saved`` -> the screen's own re-selection (via ``select_
    bench``, which mounts a brand-new editor instance from what was
    actually persisted) reads clean again -- the round-trip this whole
    feature exists to protect: an in-flight worker completing AFTER a Save
    must still be free to auto-navigate."""
    async with character_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await select_bench(pilot, saved_bench_id)
        await pilot.pause()
        await _click_after_scroll(pilot, "#evals-cb-name")
        pilot.app.screen.query_one("#evals-cb-name", Input).value = ""
        await pilot.press(*"renamed-and-saved")
        assert pilot.app.screen.query_one(CharacterBenchEditor).is_dirty() is True

        await _click_after_scroll(pilot, "#evals-cb-save")
        await pilot.pause()

        # `select_bench` bypasses `EvalsScreen.select()`'s own recompose
        # (see its own docstring), so re-select explicitly to rebuild the
        # editor from the freshly persisted row, mirroring what a real
        # `CharacterBenchEditor.Saved` -> `EvalsScreen.select()` round trip
        # produces.
        await select_bench(pilot, saved_bench_id)
        await pilot.pause()
        editor = pilot.app.screen.query_one(CharacterBenchEditor)
        assert editor.is_dirty() is False
