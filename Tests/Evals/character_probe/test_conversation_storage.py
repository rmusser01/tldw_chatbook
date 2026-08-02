import pytest

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.character_probe.models import (
    CardSnapshot,
    CharacterProbeConfig,
    Conversation,
    ConversationTurn,
    Probe,
    ProbeSet,
)
from tldw_chatbook.Evals.character_probe.storage import (
    annotate_turn,
    conversation_sample_id,
    create_probe_run_group,
    load_character_bench,
    load_conversations,
    load_probe_run_snapshot,
    load_review_state,
    load_turn_annotations,
    mark_conversation_reviewed,
    run_group_vocabulary,
    save_character_bench,
    save_conversations,
)
from tldw_chatbook.Evals.character_probe.tags import BUILTIN_TAGS, Tag


@pytest.fixture
def db():
    return EvalsDB(db_path=":memory:", client_id="test")


def _conversation(card_id=1, probe_index=0, sample_index=0, target_id="t-1"):
    return Conversation(
        card_id=card_id,
        probe_index=probe_index,
        sample_index=sample_index,
        target_id=target_id,
        turns=(
            ConversationTurn(user="One", reply="Reply one"),
            ConversationTurn(user="Two", reply="Reply two"),
        ),
    )


def _seed_run(db):
    task_id = db.create_task(
        name="probe bench", description="", task_type="generation",
        config_format="custom", config_data={"bench_type": "character_probe"},
    )
    model_id = db.create_model(name="m", provider="llama_cpp", model_id="m")
    run_id = db.create_run(name="r", task_id=task_id, model_id=model_id)
    return run_id, model_id


def test_sample_id_composes_card_probe_and_sample():
    assert conversation_sample_id(3, 1, 2) == "3:1:2"


def test_conversations_round_trip(db):
    run_id, target_id = _seed_run(db)
    original = _conversation(target_id=target_id)
    save_conversations(db, "rg-1", {target_id: run_id}, [original])
    (loaded,) = load_conversations(db, "rg-1")
    assert loaded.turns == original.turns
    assert loaded.card_id == original.card_id


def test_save_conversations_rejects_a_stale_run_id(db):
    """update_run silently no-ops on an unmatched id; save_conversations must
    not let a deleted/nonexistent run look like a successful stamp."""
    with pytest.raises(ValueError):
        save_conversations(
            db, "rg-1", {"t-1": "no-such-run-id"}, [_conversation(target_id="t-1")]
        )


def test_turns_are_stored_in_metadata_not_actual_output(db):
    """actual_output is shaped for a single answer; a conversation is not one."""
    run_id, target_id = _seed_run(db)
    save_conversations(db, "rg-1", {target_id: run_id}, [_conversation(target_id=target_id)])
    row = db.get_run_results(run_id)[0]
    assert "Reply one" in str(row.get("metadata"))


def test_a_turn_annotation_persists_with_its_tags_and_note(db):
    run_id, target_id = _seed_run(db)
    save_conversations(db, "rg-1", {target_id: run_id}, [_conversation(target_id=target_id)])
    annotate_turn(db, "rg-1", 1, 0, 0, target_id, 1, ["broke-character"], "drifted here")
    stored = load_turn_annotations(db, "rg-1")[(1, 0, 0, target_id, 1)]
    assert stored["tags"] == ["broke-character"]
    assert stored["note"] == "drifted here"


def test_re_annotating_the_same_turn_replaces_it(db, probe_run_group):
    annotate_turn(db, probe_run_group, 1, 0, 0, "t-1", 0, ["refused"], "")
    annotate_turn(
        db, probe_run_group, 1, 0, 0, "t-1", 0, ["in-character"], "fine actually"
    )
    stored = load_turn_annotations(db, probe_run_group)[(1, 0, 0, "t-1", 0)]
    assert stored["tags"] == ["in-character"]


def test_a_conversation_can_be_reviewed_with_no_annotations(db):
    """'Nothing notable' is a real verdict and needs its own home."""
    mark_conversation_reviewed(db, "rg-1", 1, 0, 0, "t-1")
    state = load_review_state(db, "rg-1")[(1, 0, 0, "t-1")]
    assert state["reviewed_at"]
    assert load_turn_annotations(db, "rg-1") == {}


def test_review_state_is_scoped_to_its_run_group(db):
    mark_conversation_reviewed(db, "rg-1", 1, 0, 0, "t-1")
    assert load_review_state(db, "rg-2") == {}


def test_save_conversations_writes_nothing_when_a_target_is_unknown(db):
    """The unknown-target check used to run INSIDE the write loop, so earlier
    conversations were already committed when it raised -- a half-written
    group that loads back as if it were complete, since nothing
    distinguishes a missing conversation from one never meant to exist."""
    run_id, target_id = _seed_run(db)
    conversations = [
        _conversation(card_id=1, target_id=target_id),
        _conversation(card_id=2, target_id="not-a-target"),
    ]
    with pytest.raises(ValueError, match="not-a-target"):
        save_conversations(db, "rg-1", {target_id: run_id}, conversations)
    assert db.get_run_results(run_id) == []
    assert load_conversations(db, "rg-1") == []


def test_save_conversations_does_not_stamp_runs_when_validation_fails(db):
    """The run_group_id stamp is a write too: it must not land either."""
    run_id, target_id = _seed_run(db)
    with pytest.raises(ValueError):
        save_conversations(
            db,
            "rg-1",
            {target_id: run_id},
            [_conversation(target_id="not-a-target")],
        )
    assert db.get_run(run_id)["run_group_id"] is None


# --- Run group + snapshot (whole-branch review I4) ----------------------


def _bench_config(**overrides):
    base = dict(
        name="villain probes",
        probe_set_id="ps-1",
        character_ids=(1,),
        target_ids=("t-1",),
        temperature=0.3,
        max_tokens=256,
        seed=1234,
        samples_per_cell=2,
    )
    base.update(overrides)
    return CharacterProbeConfig(**base)


def _cards():
    return [
        CardSnapshot(
            id=1,
            name="Vex",
            description="A dock-side fixer.",
            system_prompt="You are {{char}}.",
            personality="sardonic",
        )
    ]


@pytest.fixture
def bench(db):
    config = _bench_config()
    return config, save_character_bench(db, config)


def _target_row(db, name="steered", config=None):
    row_id = db.create_model(
        name=name, provider="llama_cpp", model_id="m", config=config or {}
    )
    return db.get_model(row_id)


@pytest.fixture
def probe_run_group(db, bench):
    """A run group opened under a bench with no extra tags."""
    config, task_id = bench
    group_id, _ = create_probe_run_group(
        db,
        task_id,
        config,
        _cards(),
        ProbeSet(probes=(Probe(turns=("One",)),)),
        [_target_row(db)],
    )
    return group_id


@pytest.fixture
def bench_id_of_that_run(db):
    """A bench carrying an extra tag, saved before any run group opens."""
    config = _bench_config(
        name="villain probes with extras",
        extra_tags=(Tag("meta-commentary", "Meta commentary", "failure"),),
    )
    return save_character_bench(db, config)


@pytest.fixture
def probe_run_group_with_extra_tags(db, bench_id_of_that_run):
    """A run group opened under a bench that carries an extra tag."""
    config = load_character_bench(db, bench_id_of_that_run)
    group_id, _ = create_probe_run_group(
        db,
        bench_id_of_that_run,
        config,
        _cards(),
        ProbeSet(probes=(Probe(turns=("One",)),)),
        [_target_row(db)],
    )
    return group_id


def test_create_probe_run_group_returns_a_run_per_target(db, bench):
    config, task_id = bench
    targets = [_target_row(db, "base"), _target_row(db, "steered")]
    group_id, run_ids = create_probe_run_group(
        db, task_id, config, _cards(), ProbeSet(probes=(Probe(turns=("One",)),)), targets
    )
    assert set(run_ids) == {t["id"] for t in targets}
    assert len(db.list_runs(run_group_id=group_id)) == 2


def test_the_run_snapshot_carries_the_card_text(db, bench):
    """CardSnapshot's whole provenance purpose -- copying card text across a
    boundary with no foreign keys -- was defeated the moment a run ended,
    because nothing about the cards was persisted."""
    config, task_id = bench
    group_id, _ = create_probe_run_group(
        db,
        task_id,
        config,
        _cards(),
        ProbeSet(probes=(Probe(turns=("One",)),)),
        [_target_row(db)],
    )
    snapshot = load_probe_run_snapshot(db, group_id)
    (card,) = snapshot["cards"]
    assert card["description"] == "A dock-side fixer."
    assert card["system_prompt"] == "You are {{char}}."
    assert card["name"] == "Vex"


def test_the_run_snapshot_carries_the_sampler_settings(db, bench):
    config, task_id = bench
    group_id, _ = create_probe_run_group(
        db,
        task_id,
        config,
        _cards(),
        ProbeSet(probes=(Probe(turns=("One",)),)),
        [_target_row(db)],
    )
    assert load_probe_run_snapshot(db, group_id)["sampler"] == {
        "temperature": 0.3,
        "max_tokens": 256,
        "seed": 1234,
        "samples_per_cell": 2,
        "concurrency": 1,
    }


def test_the_run_snapshot_carries_the_composed_system_prompt(db, bench):
    """What the model was actually told is not derivable from the parts
    later: field order, labelling and macro resolution all live in prompt.py,
    which is free to change."""
    config, task_id = bench
    target = _target_row(db, config={"system_prompt": "Be terse."})
    group_id, _ = create_probe_run_group(
        db,
        task_id,
        config,
        _cards(),
        ProbeSet(probes=(Probe(turns=("One",)),)),
        [target],
    )
    composed = load_probe_run_snapshot(db, group_id)["composed_system_prompts"]
    text = composed["1"][target["id"]]
    assert text.startswith("Be terse.")
    assert "You are Vex." in text  # steering kept AND macros resolved
    assert "A dock-side fixer." in text


def test_the_run_snapshot_carries_the_target_list_with_its_steering(db, bench):
    config, task_id = bench
    target = _target_row(db, name="terse", config={"system_prompt": "Be terse."})
    group_id, _ = create_probe_run_group(
        db,
        task_id,
        config,
        _cards(),
        ProbeSet(probes=(Probe(turns=("One",)),)),
        [target],
    )
    (stored,) = load_probe_run_snapshot(db, group_id)["targets"]
    assert stored == {
        "id": target["id"],
        "name": "terse",
        "provider": "llama_cpp",
        "model_id": "m",
        "system_prompt": "Be terse.",
    }


def test_the_snapshot_survives_the_bench_being_edited_afterwards(db, bench):
    """A run must be self-describing: eval_tasks is mutable, so rendering
    from the live bench would let a later edit rewrite history."""
    config, task_id = bench
    group_id, _ = create_probe_run_group(
        db,
        task_id,
        config,
        _cards(),
        ProbeSet(probes=(Probe(turns=("One",)),)),
        [_target_row(db)],
    )
    save_character_bench(
        db, _bench_config(name="renamed", temperature=1.9), task_id=task_id
    )
    snapshot = load_probe_run_snapshot(db, group_id)
    assert snapshot["bench_name"] == "villain probes"
    assert snapshot["sampler"]["temperature"] == 0.3


def test_each_run_reports_its_own_cell_count(db, bench):
    """The target axis IS the run, so it is not part of one run's count."""
    config, task_id = bench  # samples_per_cell=2
    probe_set = ProbeSet(probes=(Probe(turns=("One",)), Probe(turns=("Two",))))
    group_id, run_ids = create_probe_run_group(
        db, task_id, config, _cards(), probe_set, [_target_row(db)]
    )
    (run_id,) = run_ids.values()
    assert db.get_run(run_id)["total_samples"] == 1 * 2 * 2


def test_loading_the_snapshot_of_an_unknown_run_group_raises(db):
    with pytest.raises(ValueError, match="No runs found"):
        load_probe_run_snapshot(db, "rg-nope")


def test_a_prefix_steered_target_never_opens_a_run_group(db, bench):
    config, task_id = bench
    prefixed = _target_row(db, config={"prefix": "Be careful. "})
    with pytest.raises(ValueError, match="prefix"):
        create_probe_run_group(
            db,
            task_id,
            config,
            _cards(),
            ProbeSet(probes=(Probe(turns=("One",)),)),
            [prefixed],
        )
    assert db.list_runs(task_id=task_id) == []


def test_character_probe_never_imports_the_word_bench_measurement_stack():
    """This eval reads generated text only. Importing the capture client,
    normalizer, or canary code (httpx-backed) would let distribution
    vocabulary leak into a surface that has none -- pinned the way
    Tests/UI/test_evals_bench_editor.py pins the same rule for the editor.

    A source-token grep alone cannot catch this: ``character_probe/targets.py``
    imports ``model_steering`` (task-1611's steering reader), and until
    task-1754 relocated that reader out of ``word_bench.storage`` into a
    shared, stdlib-only module (``Evals/steering.py``), importing it pulled
    ``word_bench.capture_client`` -> ``normalizer`` -> ``httpx`` in
    transitively -- with no forbidden token ever appearing in ANY
    character_probe source file, so the old version of this test passed
    throughout. The subprocess check below is what actually pins the rule;
    the token grep after it is a cheap secondary check, not a substitute.

    Runs in a subprocess so an already-populated ``sys.modules`` from other
    tests in this same session (e.g. ``Tests/Evals/word_bench/*``, which
    legitimately import word_bench) can never produce a false pass -- a
    fresh interpreter is the only way to observe what importing
    character_probe *by itself* actually loads. ``cwd`` is computed from
    this file's own resolved path, not the process's ambient working
    directory, so the check cannot silently no-op when pytest is invoked
    from somewhere other than the repo root -- the previous version's own
    bug: a relative ``pathlib.Path("tldw_chatbook/...")`` glob that matched
    nothing (and therefore asserted nothing) from any other cwd.

    The probe script reports one of three OUTCOMES on stdout, not merely a
    process exit code, and this function asserts on which one it is rather
    than on the exit code alone: a non-zero exit is ambiguous by itself --
    it is what BOTH "a forbidden module loaded" (the thing this test
    exists to catch) AND "some unrelated import in the walked package
    raised" (a syntax error, a missing optional dependency, anything else)
    produce, and conflating the two would send someone hunting a
    forbidden-import violation that does not exist. ``OK`` is success;
    ``VIOLATION:<modules>`` names exactly which forbidden modules loaded
    (this is the failure this test is FOR); ``CRASH`` means the probe
    itself failed to even finish walking the package, which is a different
    problem entirely and is reported as one, with the subprocess's own
    traceback surfaced so it is debuggable as what it actually is.
    """
    import pathlib
    import subprocess
    import sys

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    package_dir = repo_root / "tldw_chatbook" / "Evals" / "character_probe"
    assert package_dir.is_dir(), (
        f"computed repo root {repo_root!r} does not contain the package "
        "under test; this test's parents[N] no longer matches this file's "
        "location on disk"
    )

    # word_bench.storage is deliberately not listed here even though it is
    # the module that USED to carry model_steering: it unconditionally
    # imports capture_client itself, so any path that still reaches
    # word_bench.storage necessarily also reaches capture_client, which is
    # listed below. Checking the leaf modules covers the whole chain.
    #
    # The import loop is wrapped so an unrelated import failure anywhere in
    # the walked package reports as CRASH (with a full traceback on
    # stderr), never silently reads as "no forbidden module loaded" (a
    # false pass) or gets conflated with VIOLATION (a false accusation of
    # the wrong failure) -- see this function's own docstring.
    probe_script = (
        "import importlib, pkgutil, sys, traceback\n"
        "try:\n"
        "    import tldw_chatbook.Evals.character_probe as pkg\n"
        "    for info in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + '.'):\n"
        "        importlib.import_module(info.name)\n"
        "except BaseException:\n"
        "    traceback.print_exc()\n"
        "    sys.stdout.write('CRASH\\n')\n"
        "    sys.exit(2)\n"
        "forbidden = [\n"
        "    'tldw_chatbook.Evals.word_bench.capture_client',\n"
        "    'tldw_chatbook.Evals.word_bench.normalizer',\n"
        "    'httpx',\n"
        "]\n"
        "present = sorted(m for m in forbidden if m in sys.modules)\n"
        "if present:\n"
        "    sys.stdout.write('VIOLATION:' + ','.join(present) + '\\n')\n"
        "    sys.exit(1)\n"
        "sys.stdout.write('OK\\n')\n"
        "sys.exit(0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe_script],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=60,
    )
    stdout = result.stdout
    stderr_tail = result.stderr[-4000:]

    if stdout.startswith("VIOLATION:"):
        loaded = stdout[len("VIOLATION:"):].strip()
        pytest.fail(
            "importing tldw_chatbook.Evals.character_probe (every module of "
            f"it) loaded word_bench's measurement stack: {loaded}"
        )
    if stdout.startswith("CRASH"):
        pytest.fail(
            "the import-graph probe crashed while walking "
            "tldw_chatbook.Evals.character_probe -- this is NOT evidence of "
            "a forbidden import (see VIOLATION above for that); it means "
            "the probe itself, or an unrelated import in the package, "
            f"failed. Traceback from the subprocess:\n{stderr_tail}"
        )
    assert result.returncode == 0 and stdout.strip() == "OK", (
        "the import-graph probe subprocess ended in an unrecognized state "
        f"-- returncode={result.returncode} stdout={stdout!r} "
        f"stderr(tail)={stderr_tail!r}"
    )

    # Secondary, cheap check kept alongside the graph assertion above (not
    # instead of it -- see the docstring): no forbidden token appears in
    # this package's own source either. package_dir is computed from
    # repo_root above, not a relative literal, for the same cwd-independence
    # reason as the subprocess check.
    forbidden_tokens = ("capture_client", "normalize_logprobs", "CANARY", "top_k", "logprobs")
    for module in package_dir.glob("*.py"):
        source = module.read_text()
        for token in forbidden_tokens:
            assert token not in source, f"{module.name} mentions {token}"


# --- run_group_vocabulary (whole-branch review I4, task-4) ---------------


def test_a_run_with_no_extra_tags_has_exactly_the_builtins(db, probe_run_group):
    assert run_group_vocabulary(db, probe_run_group) == BUILTIN_TAGS


def test_a_runs_vocabulary_includes_the_benchs_extras_as_of_the_run(
    db, probe_run_group_with_extra_tags
):
    vocab = run_group_vocabulary(db, probe_run_group_with_extra_tags)
    assert Tag("meta-commentary", "Meta commentary", "failure") in vocab


def test_editing_the_bench_after_the_run_does_not_change_the_runs_vocabulary(
    db, probe_run_group_with_extra_tags, bench_id_of_that_run
):
    """Snapshot provenance: the run is annotated with what it captured."""
    config = load_character_bench(db, bench_id_of_that_run)
    save_character_bench(
        db,
        type(config)(
            name=config.name,
            probe_set_id=config.probe_set_id,
            character_ids=config.character_ids,
            target_ids=config.target_ids,
            extra_tags=(),
        ),
        bench_id_of_that_run,
    )
    vocab = run_group_vocabulary(db, probe_run_group_with_extra_tags)
    assert any(t.slug == "meta-commentary" for t in vocab)


def test_an_unknown_run_group_raises_naming_it(db):
    with pytest.raises(Exception) as exc:
        run_group_vocabulary(db, "no-such-group")
    assert "no-such-group" in str(exc.value)


# --- annotate_turn tag validation (whole-branch review I4, task-5) -------


def test_a_known_tag_is_stored(db, probe_run_group):
    annotate_turn(
        db, probe_run_group, 1, 0, 0, "t-1", 0,
        tags=["broke-character"], note="third turn",
    )
    stored = load_turn_annotations(db, probe_run_group)
    assert stored[(1, 0, 0, "t-1", 0)]["tags"] == ["broke-character"]


def test_an_unknown_tag_is_rejected_naming_it(db, probe_run_group):
    with pytest.raises(ValueError) as exc:
        annotate_turn(
            db, probe_run_group, 1, 0, 0, "t-1", 0,
            tags=["brok-charcter"], note="",
        )
    assert "brok-charcter" in str(exc.value)


def test_nothing_is_written_when_one_tag_of_several_is_unknown(
    db, probe_run_group
):
    with pytest.raises(ValueError):
        annotate_turn(
            db, probe_run_group, 1, 0, 0, "t-1", 0,
            tags=["broke-character", "no-such-tag"], note="",
        )
    assert load_turn_annotations(db, probe_run_group) == {}


def test_a_non_canonical_tag_is_canonicalised_rather_than_rejected(
    db, probe_run_group
):
    annotate_turn(
        db, probe_run_group, 1, 0, 0, "t-1", 0,
        tags=["Broke Character"], note="",
    )
    stored = load_turn_annotations(db, probe_run_group)
    assert stored[(1, 0, 0, "t-1", 0)]["tags"] == ["broke-character"]


def test_duplicate_tags_are_stored_once(db, probe_run_group):
    annotate_turn(
        db, probe_run_group, 1, 0, 0, "t-1", 0,
        tags=["broke-character", "broke-character"], note="",
    )
    stored = load_turn_annotations(db, probe_run_group)
    assert stored[(1, 0, 0, "t-1", 0)]["tags"] == ["broke-character"]


def test_an_annotation_with_no_tags_but_a_note_is_allowed(
    db, probe_run_group
):
    """A note without a tag is a real observation, not an empty write."""
    annotate_turn(
        db, probe_run_group, 1, 0, 0, "t-1", 0, tags=[], note="odd phrasing",
    )
    stored = load_turn_annotations(db, probe_run_group)
    assert stored[(1, 0, 0, "t-1", 0)]["note"] == "odd phrasing"


def test_a_benchs_extra_tag_is_accepted(db, probe_run_group_with_extra_tags):
    annotate_turn(
        db, probe_run_group_with_extra_tags, 1, 0, 0, "t-1", 0,
        tags=["meta-commentary"], note="",
    )
    stored = load_turn_annotations(db, probe_run_group_with_extra_tags)
    assert stored[(1, 0, 0, "t-1", 0)]["tags"] == ["meta-commentary"]


# --- annotate_turn's optional `vocabulary` param (final review fix wave,
# findings 1+2) --------------------------------------------------------
#
# Phase 3b's recommended flow creates a tag mid-review: written to the
# bench's `extra_tags` and added to the review pane's live vocabulary, never
# to the already-captured run snapshot. Without a way to pass that live
# vocabulary in, such a tag would render as selectable and then raise
# ValueError on `annotate_turn`, which only ever consulted
# `run_group_vocabulary` (the snapshot-derived vocabulary). These tests
# cover the new `vocabulary` parameter that closes that gap.


def test_an_explicit_vocabulary_accepts_a_tag_not_in_the_runs_snapshot(
    db, probe_run_group
):
    """`probe_run_group`'s snapshot carries no extra tags, so
    `run_group_vocabulary` alone would reject `meta-commentary`. Passing it
    explicitly -- standing in for a review pane's live, session-extended
    vocabulary -- accepts and persists it anyway."""
    live_vocabulary = run_group_vocabulary(db, probe_run_group) + (
        Tag("meta-commentary", "Meta commentary", "failure"),
    )
    annotate_turn(
        db, probe_run_group, 1, 0, 0, "t-1", 0,
        tags=["meta-commentary"], note="",
        vocabulary=live_vocabulary,
    )
    stored = load_turn_annotations(db, probe_run_group)
    assert stored[(1, 0, 0, "t-1", 0)]["tags"] == ["meta-commentary"]


def test_an_explicit_vocabulary_still_rejects_a_slug_outside_it(db, probe_run_group):
    """An explicit vocabulary is a real allowlist, not a bypass: a slug
    outside it is rejected exactly as the default (snapshot-derived) path
    rejects a slug outside the run's captured vocabulary, and nothing is
    written."""
    live_vocabulary = run_group_vocabulary(db, probe_run_group) + (
        Tag("meta-commentary", "Meta commentary", "failure"),
    )
    with pytest.raises(ValueError, match="no-such-tag"):
        annotate_turn(
            db, probe_run_group, 1, 0, 0, "t-1", 0,
            tags=["no-such-tag"], note="",
            vocabulary=live_vocabulary,
        )
    assert load_turn_annotations(db, probe_run_group) == {}


def test_omitting_vocabulary_still_validates_against_the_runs_captured_vocabulary(
    db, probe_run_group
):
    """Omitting `vocabulary` must behave exactly as before this parameter
    existed: the same `meta-commentary` slug that an explicit live
    vocabulary accepts (see above) is still rejected here, because
    `probe_run_group`'s own captured snapshot has no such extra tag."""
    with pytest.raises(ValueError, match="meta-commentary"):
        annotate_turn(
            db, probe_run_group, 1, 0, 0, "t-1", 0,
            tags=["meta-commentary"], note="",
        )
    assert load_turn_annotations(db, probe_run_group) == {}


# --- delete_task cascades probe annotations (whole-branch review I4, task-6) --
#
# `probe_run_group` is opened under the `bench` fixture's task_id (see the
# `probe_run_group` fixture above: it destructures `bench` for its
# `task_id`). Requesting `bench` directly here, alongside `probe_run_group`,
# yields that SAME cached task_id -- pytest caches a function-scoped fixture
# once per test, however many other fixtures also depend on it -- so `bench`
# is the right handle for "the bench `probe_run_group` runs under", not
# `bench_id_of_that_run` (that fixture is a deliberately different bench,
# carrying an extra tag, used by `probe_run_group_with_extra_tags`).


@pytest.fixture
def second_probe_run_group(db):
    """A run group under a bench distinct from `probe_run_group`'s, so the
    cascade-isolation test can prove deleting one bench does not touch
    another's annotations.

    Uses its own target name -- eval_models is UNIQUE on
    (name, provider, model_id), and this fixture is deliberately combined
    with `probe_run_group` (via `bench`) in the same test, which already
    creates a target named "steered" (`_target_row`'s default).
    """
    config = _bench_config(name="a different bench")
    task_id = save_character_bench(db, config)
    group_id, _ = create_probe_run_group(
        db,
        task_id,
        config,
        _cards(),
        ProbeSet(probes=(Probe(turns=("One",)),)),
        [_target_row(db, name="steered-2")],
    )
    return group_id


@pytest.fixture
def seeded_word_bench_id(db):
    """A minimal word-bench `eval_tasks` row, written directly against
    `create_task` the same way
    `Tests/Evals/character_probe/test_bench_storage.py::
    test_loading_a_word_bench_as_a_character_bench_raises` does. All this
    test needs is a bench `delete_task` can target that is NOT a character
    probe bench, to prove the cascade is scoped to that task's own run
    groups rather than to every probe annotation row in the database.

    Carries a real run group (an `eval_runs` row stamped with
    `run_group_id`), not just the bare task row: without one,
    `delete_task`'s own `run_group_ids` lookup for this task always returns
    `[]`, and `delete_probe_annotations_for_run_groups` no-ops on an empty
    id list -- the DELETE against `eval_probe_turn_annotations`/
    `eval_probe_review_state` is never actually executed, so a scoping bug
    in that DELETE's `WHERE run_group_id IN (...)` would pass unnoticed."""
    task_id = db.create_task(
        name="word bench",
        description="",
        task_type="logprob",
        config_format="custom",
        config_data={"bench_type": "word_bench"},
    )
    model_id = db.create_model(name="word-target", provider="llama_cpp", model_id="m")
    run_id = db.create_run(name="r", task_id=task_id, model_id=model_id)
    db.update_run(run_id, {"run_group_id": "word-bench-rg"})
    return task_id


def test_deleting_a_bench_removes_its_turn_annotations(db, probe_run_group, bench):
    _config, task_id = bench
    annotate_turn(
        db, probe_run_group, 1, 0, 0, "t-1", 0,
        tags=["broke-character"], note="",
    )
    assert load_turn_annotations(db, probe_run_group)

    db.delete_task(task_id)

    assert db.list_probe_turn_annotations(probe_run_group) == []


def test_deleting_a_bench_removes_its_review_state(db, probe_run_group, bench):
    _config, task_id = bench
    mark_conversation_reviewed(db, probe_run_group, 1, 0, 0, "t-1", note="fine")
    assert db.list_probe_review_state(probe_run_group)

    db.delete_task(task_id)

    assert db.list_probe_review_state(probe_run_group) == []


def test_deleting_a_bench_leaves_another_benchs_annotations_alone(
    db, probe_run_group, bench, second_probe_run_group
):
    _config, task_id = bench
    annotate_turn(
        db, probe_run_group, 1, 0, 0, "t-1", 0, tags=["refused"], note="",
    )
    annotate_turn(
        db, second_probe_run_group, 1, 0, 0, "t-1", 0,
        tags=["refused"], note="",
    )

    db.delete_task(task_id)

    assert db.list_probe_turn_annotations(second_probe_run_group)


def test_deleting_a_word_bench_touches_no_probe_annotation_rows(
    db, probe_run_group, seeded_word_bench_id
):
    annotate_turn(
        db, probe_run_group, 1, 0, 0, "t-1", 0, tags=["refused"], note="",
    )
    db.delete_task(seeded_word_bench_id)
    assert db.list_probe_turn_annotations(probe_run_group)
