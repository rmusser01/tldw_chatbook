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

import threading

import pytest
from textual.widgets import Button, Input

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

    Review round 2 (verification gap noted by the re-reviewer): also
    asserts the run group reads ``"completed"`` through the REAL
    ``EvalsViewModel.run_groups()`` classification (the same pivot the
    rail itself renders from), not the raw ``eval_runs.status`` column --
    the property this whole fix round exists to protect is what the RAIL
    shows, and ``run_groups()`` is the one function that decides that.
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

        run_groups = pilot.app.screen._view_model.run_groups()
        assert len(run_groups) == 1
        assert run_groups[0]["id"] == pilot.app.screen._selection.id
        assert run_groups[0]["status"] == "completed"


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


@pytest.mark.asyncio
async def test_the_worker_names_the_missing_character_database_if_reached_by_any_path(
    evals_db, chachanotes_db, runnable_character_bench, succeeding_chat
):
    """Qodo review (task-1691 phase 2 fix wave), Finding 1's second half:
    ``_primary_action_state`` (pinned in ``test_evals_screen.py``'s
    ``test_a_character_bench_reopened_without_a_character_database_cannot_
    be_run``) now keeps Run disabled whenever ``self._chacha_db is None``,
    so the real button can no longer dispatch this worker in that state --
    but the worker itself must still fail loudly, not with a bare
    ``AttributeError``, for any OTHER path that could still reach it
    (``run_worker`` dispatch racing a late ``chachanotes_db`` teardown, a
    future caller that skips the button). Deliberately does NOT use this
    module's own ``evals_app`` fixture -- that fixture wires a real
    ``chachanotes_db`` (see its own docstring) precisely so a normal Run
    can snapshot real cards; this test instead builds a bare app with no
    ``chachanotes_db=`` kwarg, mirroring ``test_evals_screen.py``'s own
    bare ``evals_app``, so ``EvalsScreen._chacha_db`` resolves to ``None``
    (``_resolve_chacha_db``'s own docstring) while ``runnable_character_
    bench`` still carries real, non-empty ``character_ids`` against the
    SEPARATE ``chachanotes_db`` fixture instance that built it.
    """
    from .test_evals_screen import EvalsHarness, _FakeAppInstance

    bare_app = EvalsHarness(_FakeAppInstance(evals_db))
    async with bare_app.run_test(size=_REALISTIC_SIZE) as pilot:
        screen = pilot.app.screen
        assert screen._chacha_db is None
        screen._character_probe_chat_factory = lambda cfg: succeeding_chat
        screen.select(kind="character_bench", id=runnable_character_bench)
        await pilot.pause()

        # The button itself is disabled by Finding 1's other half -- reach
        # the worker directly, as any OTHER dispatch path would, rather
        # than through a click the real UI would never deliver here.
        action = screen.query_one("#evals-primary-action", Button)
        assert action.disabled
        screen._character_bench_run_task_id = runnable_character_bench
        await screen._run_character_bench_worker()

        message, severity = screen.app_instance.notifications[-1]
        assert severity == "error"
        assert "character card database" in message
        assert "AttributeError" not in message
        assert "NoneType" not in message

        # Fails loudly, no run group to navigate to -- the new guard fires
        # BEFORE `create_probe_run_group` ever runs (it sits right after
        # `load_probe_set`, ahead of `snapshot_cards`), so unlike an
        # ordinary mid-run failure this path never even creates the
        # `eval_runs` rows in the first place.
        assert screen._selection.kind == "character_bench"
        run_groups = screen._view_model.run_groups()
        assert len(run_groups) == 0


# ---------------------------------------------------------------------------
# Review round 1 (Important findings): the dirty-editor guard and the run
# status transition, both newly reachable because this task is the first
# code to ever call select() after a character-bench run completes.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_running_a_character_bench_does_not_yank_a_dirty_editor(
    evals_app, runnable_character_bench, evals_db
):
    """Before ``CharacterBenchEditor.is_dirty()`` existed and
    ``EvalsScreen._selection_unmoved_since_launch`` learned to consult it,
    editing this bench's Name field without saving, then pressing Run (a
    SEPARATE button -- the editor stays mounted the whole time), let the
    completing worker's own ``select(kind="run_group", ...)`` silently
    discard the unsaved edit -- exactly the class of bug task-1610's
    original ``BenchEditor``-only dirty guard was built to prevent for
    word benches, newly reachable here because this task is the first
    code to ever call ``select()`` after a character-bench run completes.

    A ``threading.Event``-paused chat callable (not ``asyncio.Event``):
    ``ChatCallable`` is a plain synchronous ``def`` running inside
    ``asyncio.to_thread``'s own worker thread, so a blocking
    ``threading.Event.wait()`` is what actually holds the run open
    without blocking the event loop -- the pilot stays fully responsive
    to real clicks/keystrokes while the first provider call is paused.
    """
    release = threading.Event()

    def _pausable(*, messages, model, temperature, max_tokens, seed) -> str:
        release.wait()
        return "In character, while paused."

    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        screen = pilot.app.screen
        screen._character_probe_chat_factory = lambda cfg: _pausable
        screen.select(kind="character_bench", id=runnable_character_bench)
        await pilot.pause()

        await pilot.click("#evals-primary-action")
        await _wait_until(pilot, lambda: screen._character_bench_run_running)
        await pilot.pause()

        name_input = screen.query_one("#evals-cb-name", Input)
        name_input.scroll_visible(animate=False)
        await pilot.pause()
        await pilot.click("#evals-cb-name")
        name_input.value = ""  # setup: clear before typing the real edit
        await pilot.press(*"typed-while-running")

        release.set()
        await _wait_until(pilot, lambda: not screen._character_bench_run_running)
        await pilot.pause()

        assert screen._selection.kind == "character_bench"
        assert screen._selection.id == runnable_character_bench
        # The same widget instance, still carrying the typed value -- proof
        # this is a SKIPPED recompose, not merely a value that happens to
        # match after a rebuild.
        assert screen.query_one("#evals-cb-name", Input) is name_input
        assert screen.query_one("#evals-cb-name", Input).value == "typed-while-running"
        message, severity = screen.app_instance.notifications[-1]
        assert severity == "information"
        assert message == "Bench run finished — see the Runs section."

        # The run itself is real -- the DB write is not lost, only the
        # auto-navigate is skipped.
        from tldw_chatbook.Evals.character_probe.storage import load_conversations

        run_groups = screen._view_model.run_groups()
        assert len(run_groups) == 1
        assert len(load_conversations(evals_db, run_groups[0]["id"])) == 4


@pytest.mark.asyncio
async def test_an_in_flight_character_bench_run_reads_as_running_not_completed(
    evals_app, runnable_character_bench
):
    """Whole-branch review, deferred-minor-promoted-to-must-fix: one
    ``_mark_character_run_ids(db, run_ids, "running")`` call was missing
    right after ``create_probe_run_group`` -- before it, every run row
    this worker created sat at its ``'pending'`` DB default for the
    ENTIRE run, not merely the hard-cancellation window the test below
    covers. ``EvalsViewModel.run_groups()``'s own pivot falls a "pending,
    nothing running/cancelled/failed" group through to "completed" (see
    that method's own docstring), so a run genuinely IN PROGRESS read
    exactly like one that had already finished successfully with real
    results, from the very first provider call onward -- inconsistent
    with this same worker's own terminal-status remediation for every
    OTHER outcome (completed/cancelled/failed all already stamped
    correctly)."""
    release = threading.Event()

    def _pausable(*, messages, model, temperature, max_tokens, seed) -> str:
        release.wait()
        return "In character, while paused."

    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        screen = pilot.app.screen
        screen._character_probe_chat_factory = lambda cfg: _pausable
        screen.select(kind="character_bench", id=runnable_character_bench)
        await pilot.pause()

        await pilot.click("#evals-primary-action")
        await _wait_until(pilot, lambda: screen._character_bench_run_running)
        await pilot.pause()

        try:
            run_groups = screen._view_model.run_groups()
            assert len(run_groups) == 1
            assert run_groups[0]["status"] == "running"
        finally:
            # Unblock the paused provider thread NO MATTER what the
            # assertions above did -- `to_thread` survives Task
            # cancellation (see `_run_character_bench_worker`'s own
            # module docstring), so an assertion failure here must still
            # release it or the worker thread leaks past this test and
            # hangs the whole run's teardown waiting to join it.
            release.set()
        await _wait_until(pilot, lambda: not screen._character_bench_run_running)
        await pilot.pause()


@pytest.mark.asyncio
async def test_a_hard_cancelled_character_bench_run_does_not_read_as_completed(
    evals_app, runnable_character_bench
):
    """``EvalsViewModel.run_groups()``'s own pivot falls a "pending,
    nothing running/cancelled/failed" group through to "completed", and
    nothing in ``character_probe/storage.py``/``runner.py`` (Task 1's
    phase-1 engine) ever transitions ``eval_runs.status`` itself, unlike
    ``WordBenchRunner``. Before this fix, a HARD cancellation (Textual's
    own ``exclusive=True`` worker mechanism -- the same mechanism a
    second, superseding Run press already uses in production) left every
    run row this worker created stuck at its 'pending' DB default, which
    the rail's own pivot then read as "completed": indistinguishable from
    a genuinely finished run with real results.

    Cancels the worker's own Task directly via ``pilot.app.workers.
    cancel_group`` -- the sharpest available reproduction of "cancelled
    after work started" without a Cancel button to press yet.
    """
    release = threading.Event()

    def _pausable(*, messages, model, temperature, max_tokens, seed) -> str:
        release.wait()
        return "never returns during this test"

    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        screen = pilot.app.screen
        screen._character_probe_chat_factory = lambda cfg: _pausable
        screen.select(kind="character_bench", id=runnable_character_bench)
        await pilot.pause()

        await pilot.click("#evals-primary-action")
        await _wait_until(pilot, lambda: screen._character_bench_run_running)
        await pilot.pause()

        cancelled = pilot.app.workers.cancel_group(
            screen, "evals-run-character-bench"
        )
        assert cancelled, "expected an in-flight character-bench run worker to cancel"

        await _wait_until(pilot, lambda: not screen._character_bench_run_running)
        await pilot.pause()

        run_groups = screen._view_model.run_groups()
        assert len(run_groups) == 1
        assert run_groups[0]["status"] == "cancelled"

        # Unblock the still-paused provider thread -- it survives Task
        # cancellation per `to_thread`'s own contract (see the module
        # docstring), so it must be released rather than left to leak
        # past this test.
        release.set()


@pytest.mark.asyncio
async def test_a_character_bench_run_that_errors_ordinarily_does_not_read_as_completed(
    evals_app, runnable_character_bench
):
    """Review round 2 (Important finding): the general ``except
    Exception:`` branch of ``_run_character_bench_worker`` is reachable
    with ``run_ids`` already populated -- ``create_probe_run_group`` runs
    (and writes real ``eval_runs`` rows) BEFORE the chat factory is ever
    called -- whenever an ORDINARY exception fires afterward, not only a
    cancellation. Here the injected ``_character_probe_chat_factory``
    itself raises building the chat callable, the same class of failure a
    real ``chat_api_call`` configuration error would produce. Before this
    fix, the created run rows stayed stuck at their ``'pending'`` DB
    default forever, which ``run_groups()``'s own pivot fell through to
    "completed" -- indistinguishable from a genuinely finished run with
    real results.
    """

    def _broken_factory(cfg):
        raise RuntimeError("simulated: could not build a chat client")

    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        screen = pilot.app.screen
        screen._character_probe_chat_factory = _broken_factory
        screen.select(kind="character_bench", id=runnable_character_bench)
        await pilot.pause()

        await pilot.click("#evals-primary-action")

        def _toasted() -> bool:
            return any(
                "Could not run the bench" in message
                for message, _severity in screen.app_instance.notifications
            )

        await _wait_until(pilot, _toasted)
        await pilot.pause()

        # The run genuinely failed -- the selection must not have moved to
        # a run group (there is nothing worth navigating to).
        assert screen._selection.kind == "character_bench"

        run_groups = screen._view_model.run_groups()
        assert len(run_groups) == 1
        assert run_groups[0]["status"] != "completed"
        # The concrete classification, not merely "not completed" --
        # `run_groups()`'s own pivot has no separate group-level "failed"
        # bucket (a run-level "failed" status buckets into the SAME
        # group-level "cancelled" label a "cancelled" run-level status
        # gets -- see `EvalsScreen._mark_character_run_ids`'s own
        # docstring), so this is the real, deterministic value, not a
        # guess.
        assert run_groups[0]["status"] == "cancelled"
