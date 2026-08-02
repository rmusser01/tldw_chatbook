"""End-to-end: running a character-probe bench through the real UI.

task-1691 phase 2's final task -- everything up to here (rail markers,
probe-set import, the card picker, the character-bench editor, routing)
built the ability to CREATE a character bench through the UI; this file
drives the last missing step, pressing Run, through real widgets against a
real (``:memory:``) ``EvalsDB`` and a real (``:memory:``) ``CharactersRAGDB``
for the card snapshot. Mirrors this suite's other e2e files
(``test_evals_authoring_e2e.py``, ``test_evals_continuation_e2e.py``):
``EvalsHarness``/``_FakeAppInstance`` imported from ``test_evals_screen.py``
rather than redefined, plus this file's own ``evals_db``/``evals_app``
fixtures.

The chat callable ``CharacterProbeRunner`` (and therefore
``EvalsScreen._run_character_bench_worker``) dispatches is SYNCHRONOUS --
every fake chat function below matches ``character_probe.runner.
ChatCallable``'s own shape (``chat_fn(*, messages, model, temperature,
max_tokens, seed) -> str``), a plain ``def``, never a coroutine function.
The runner already threads it through ``asyncio.to_thread``; nothing in
these tests (or in the worker itself) ever ``await``s it directly.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.character_probe.models import CharacterProbeConfig, Probe, ProbeSet
from tldw_chatbook.Evals.character_probe.storage import save_character_bench, save_probe_set

from .test_evals_screen import EvalsHarness, _FakeAppInstance

_REALISTIC_SIZE = (160, 45)


@pytest.fixture
def evals_db() -> EvalsDB:
    return EvalsDB(db_path=":memory:", client_id="test")


@pytest.fixture
def chachanotes_db():
    """A real, in-memory ``CharactersRAGDB`` -- character cards live in a
    different database from ``evals_db`` (see ``EvalsViewModel.
    character_cards``'s own docstring, and ``EvalsScreen._resolve_chacha_
    db``), so a real Run (which snapshots real cards via ``cards.
    snapshot_cards``) needs a real handle here, not a bare dict."""
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    return CharactersRAGDB(":memory:", "test-client")


@pytest.fixture
def evals_app(evals_db: EvalsDB, chachanotes_db) -> EvalsHarness:
    """Mirrors ``test_evals_screen.py``'s own ``character_bench_app``: a
    real ``chachanotes_db`` wired the same way the real app's
    ``TldwCli.chachanotes_db`` is, needed by every test in this file (a
    real Run always snapshots real cards)."""
    return EvalsHarness(_FakeAppInstance(evals_db, chachanotes_db=chachanotes_db))


@pytest.fixture
def runnable_character_bench(evals_db: EvalsDB, chachanotes_db) -> str:
    """2 cards x 2 probes (turns 2, 1) x 1 target x 1 sample -- 4
    conversations, ``sum((2, 1)) * 2 * 1 * 1 == 6`` calls (the design
    spec's own "Total calls = cards x probes x targets x samples x
    turns-per-probe" formula, collapsed into "turns summed across probes"
    since the two probes here have different turn counts)."""
    card_a = chachanotes_db.add_character_card({"name": "Vex"})
    card_b = chachanotes_db.add_character_card({"name": "Lyra"})
    probe_set_id = save_probe_set(
        evals_db,
        "run-e2e probe set",
        ProbeSet(
            probes=(
                Probe(turns=("Hello there.", "And then what happened?")),
                Probe(turns=("One more thing.",)),
            )
        ),
    )
    target_id = evals_db.create_model(
        name="run-e2e target", provider="llama_cpp", model_id="m"
    )
    config = CharacterProbeConfig(
        name="run-e2e bench",
        probe_set_id=probe_set_id,
        character_ids=(card_a, card_b),
        target_ids=(target_id,),
    )
    return save_character_bench(evals_db, config)


def _succeeding_chat(*, messages, model, temperature, max_tokens, seed) -> str:
    """A trivial, always-successful fake chat callable -- matches
    ``ChatCallable``'s exact keyword shape (mirrors ``runner.py``'s own
    ``asyncio.to_thread(self._chat, messages=..., model=..., temperature=
    ..., max_tokens=..., seed=seed)`` call)."""
    return "In character, always."


@pytest.fixture
def succeeding_chat():
    return _succeeding_chat


@pytest.fixture
def failing_once_chat():
    """Fails its FIRST call (any conversation, any turn), succeeds on
    every call after that -- ``CharacterProbeConfig.concurrency`` defaults
    to 1, so with a single target and a single sample the four
    conversations in ``runnable_character_bench`` run strictly one at a
    time (``CharacterProbeRunner.run``'s own semaphore), making this
    deterministic: only the very first provider call in the whole run
    ever fails."""
    calls = {"n": 0}

    def _chat(*, messages, model, temperature, max_tokens, seed) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated provider failure")
        return "In character, mostly."

    return _chat


async def _wait_until(pilot, predicate, *, tries: int = 300, interval: float = 0.02) -> None:
    for _ in range(tries):
        if predicate():
            return
        await pilot.pause(interval)
    raise AssertionError("condition never became true")


@pytest.mark.asyncio
async def test_the_estimate_counts_cards_probes_targets_samples_and_turns(
    evals_app, runnable_character_bench
):
    """2 cards x 2 probes (2 turns, 1 turn) x 1 target x 1 sample = 6 calls.

    Corrected against the real source: the estimate widget's id is
    ``#evals-inspector-estimate-calls`` (``inspector.py``'s
    ``EvalsInspector``/``CharacterBenchEstimate`` both use this exact id,
    never mounted at once since the two selection kinds are mutually
    exclusive), not ``#evals-estimate-calls``. The selection kind for a
    character bench is ``"character_bench"`` -- a distinct
    ``SelectionKind`` Task 5 added (``evals_state.py``), never ``"bench"``
    (that kind only ever resolves a WORD bench, via ``bench_by_id``).
    """
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        pilot.app.screen.select(kind="character_bench", id=runnable_character_bench)
        await pilot.pause()
        estimate = pilot.app.screen.query_one("#evals-inspector-estimate-calls").render()
        assert str(estimate).startswith("6 calls")


@pytest.mark.asyncio
async def test_running_a_character_bench_persists_conversations(
    evals_app, runnable_character_bench, evals_db, succeeding_chat
):
    """Corrected against the real source in three ways from the plan's own
    draft: the ``kind="character_bench"``/id fixes above, PLUS a real
    ``_character_probe_chat_factory`` override -- the plan's draft never
    set one, which would exercise the PRODUCTION default (a real
    ``chat_api_call`` against whatever llama.cpp endpoint happens to be
    configured, or none at all) and could never satisfy `all(c.turns for
    c in conversations)` against a real or absent server in a unit test.
    """
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        pilot.app.screen._character_probe_chat_factory = lambda cfg: succeeding_chat
        pilot.app.screen.select(kind="character_bench", id=runnable_character_bench)
        await pilot.pause()
        await pilot.click("#evals-primary-action")
        await _wait_until(pilot, lambda: pilot.app.screen._selection.kind == "run_group")
        from tldw_chatbook.Evals.character_probe.storage import load_conversations

        conversations = load_conversations(evals_db, pilot.app.screen._selection.id)
        assert len(conversations) == 4
        assert all(c.turns for c in conversations)
        assert all(not c.error for c in conversations)


@pytest.mark.asyncio
async def test_a_failing_provider_leaves_the_rest_of_the_grid_intact(
    evals_app, runnable_character_bench, evals_db, failing_once_chat
):
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        pilot.app.screen._character_probe_chat_factory = lambda cfg: failing_once_chat
        pilot.app.screen.select(kind="character_bench", id=runnable_character_bench)
        await pilot.pause()
        await pilot.click("#evals-primary-action")
        await _wait_until(pilot, lambda: pilot.app.screen._selection.kind == "run_group")
        from tldw_chatbook.Evals.character_probe.storage import load_conversations

        conversations = load_conversations(evals_db, pilot.app.screen._selection.id)
        assert len(conversations) == 4
        assert any(c.error for c in conversations)
        assert any(not c.error and c.turns for c in conversations)


@pytest.mark.asyncio
async def test_the_run_snapshot_records_card_text_and_sampler(
    evals_app, runnable_character_bench, evals_db, succeeding_chat
):
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        pilot.app.screen._character_probe_chat_factory = lambda cfg: succeeding_chat
        pilot.app.screen.select(kind="character_bench", id=runnable_character_bench)
        await pilot.pause()
        await pilot.click("#evals-primary-action")
        await _wait_until(pilot, lambda: pilot.app.screen._selection.kind == "run_group")
        from tldw_chatbook.Evals.character_probe.storage import load_probe_run_snapshot

        snapshot = load_probe_run_snapshot(evals_db, pilot.app.screen._selection.id)
        assert snapshot["cards"]
        assert len(snapshot["cards"]) == 2
        assert {card["name"] for card in snapshot["cards"]} == {"Vex", "Lyra"}
        assert snapshot["sampler"]["samples_per_cell"] == 1
        assert snapshot["targets"]
        assert snapshot["composed_system_prompts"]


@pytest.mark.asyncio
async def test_a_bench_with_no_resolvable_target_row_fails_loudly_instead_of_silently(
    evals_app, evals_db, chachanotes_db, succeeding_chat
):
    """Two things earlier tasks left in place, both pinned here: a
    character-probe bench's engine layer is independently backstopped
    (``cards.py``/``targets.py`` raise on empty ids even though
    ``CharacterProbeConfig`` itself was relaxed for drafts), and this
    worker must surface those failures to the user rather than silently
    reporting success. ``EvalsDB`` has no way to delete an ``eval_models``
    row at all (only ``create_model``/``get_model``/``list_models``
    exist), so a target id that never resolved to a live row in the first
    place -- persisted directly, since there is no UI path that could
    produce this today (``CharacterBenchEditor`` carries the target list
    verbatim with no Add/Remove control) -- is the real, reachable shape
    this test exercises: a hand-authored or migrated row, not a UI-driven
    one."""
    card_id = chachanotes_db.add_character_card({"name": "Vex"})
    probe_set_id = save_probe_set(
        evals_db, "probes", ProbeSet(probes=(Probe(turns=("Hi.",)),))
    )
    config = CharacterProbeConfig(
        name="dangling-target bench",
        probe_set_id=probe_set_id,
        character_ids=(card_id,),
        target_ids=("nonexistent-target-id",),
    )
    bench_id = save_character_bench(evals_db, config)

    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        pilot.app.screen._character_probe_chat_factory = lambda cfg: succeeding_chat
        pilot.app.screen.select(kind="character_bench", id=bench_id)
        await pilot.pause()
        # `_primary_action_state` reads the bench's own persisted
        # `target_ids` (still non-empty -- only the `eval_models` ROW was
        # deleted), so the button is enabled; this is the real,
        # reachable shape.
        action = pilot.app.screen.query_one("#evals-primary-action", Button)
        assert not action.disabled
        await pilot.click("#evals-primary-action")

        def _toasted() -> bool:
            return any(
                "could not be resolved" in message
                for message, _severity in pilot.app.screen.app_instance.notifications
            )

        await _wait_until(pilot, _toasted)
        # Fails loudly: the selection must NOT have moved to a run group
        # that (per `save_conversations`'s own docstring) would otherwise
        # never even be reached here -- `create_probe_run_group` raises
        # before writing anything for this target.
        assert pilot.app.screen._selection.kind == "character_bench"
