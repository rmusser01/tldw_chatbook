"""Real local research/writing/study/evaluation recovery evidence."""


def test_optional_domain_discovery_does_not_require_engines():
    from tldw_chatbook.Research_Interop.recovery import recovery_adapters

    adapters = recovery_adapters()
    assert adapters
    assert all(adapter.schema_policy() is not None for adapter in adapters)


from contextlib import closing
from dataclasses import replace
import importlib
import json
from pathlib import Path
import sqlite3
from threading import Event
import pytest

from Tests.Backup_Recovery.test_core_owners import application_authority
from tldw_chatbook.Backup_Recovery.models import (
    DiscoveryContext,
    DISCOVERY_CONTEXT_KEY,
    StorageItem,
)


PACKAGES = {
    "research": "Research_Interop",
    "writing": "Writing_Interop",
    "evals": "Evals",
    "study": "Study_Interop",
}


def adapter(name):
    return importlib.import_module(
        "tldw_chatbook." + PACKAGES[name] + ".recovery"
    ).recovery_adapters()[0]


def context(source, key):
    return {
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(
            source.parent / "profile.toml", "fixture"
        ),
        "database": {key: str(source)},
    }


def records(source):
    with closing(sqlite3.connect(source)) as conn:
        return tuple(conn.iterdump())


@pytest.fixture(params=["research", "writing", "evals"])
def domain_store(request, tmp_path):
    name = request.param
    source = tmp_path / (name + ".db")
    if name == "research":
        from tldw_chatbook.Research_Interop.local_research_service import (
            LocalResearchService,
        )

        service = LocalResearchService(source)
        session = service.create_session(
            title="Keep historical session", query="Question"
        )
        run = service.create_run(query="Lossless run")
        service.save_artifact(
            run["id"],
            artifact_name="Report",
            content_type="application/json",
            content={
                "text": "Retained research",
                "sources": ["https://example.invalid/source"],
            },
        )
        service.create_checkpoint(
            run["id"],
            checkpoint_type="sources_review",
            proposed_payload={"sources": []},
        )
        service.delete_session(session["id"])
    elif name == "writing":
        from tldw_chatbook.Writing_Interop.local_writing_service import (
            LocalWritingService,
        )

        service = LocalWritingService(source)
        project = service.create_project(title="Novel")
        manuscript = service.create_manuscript(project["id"], title="Book")
        chapter = service.create_chapter(
            project["id"], title="Chapter", manuscript_id=manuscript["id"]
        )
        scene = service.create_scene(
            chapter["id"], title="Scene", content_markdown="Original prose"
        )
        service.update_scene(
            scene["id"], expected_version=1, content_markdown="Revised prose"
        )
        service.delete_scene(scene["id"], expected_version=2)
    else:
        from tldw_chatbook.DB.Evals_DB import EvalsDB
        from tldw_chatbook.Evals.word_bench.models import BenchConfig
        from tldw_chatbook.Evals.word_bench.storage import save_bench
        from tldw_chatbook.Evals.character_probe.models import (
            Probe,
            ProbeSet,
            CharacterProbeConfig,
        )
        from tldw_chatbook.Evals.character_probe.storage import (
            save_probe_set,
            save_character_bench,
        )

        service = EvalsDB(source, "recovery-fixture")
        dataset = service.create_dataset(
            "Durable inline dataset",
            "custom",
            "inline:snippets",
            metadata={"samples": [{"id": "stable", "text": "Prompt"}]},
        )
        model = service.create_model("Inert model", "local", "no-execution")
        task = save_bench(
            service,
            BenchConfig(
                name="Word bench",
                prompt_mode="raw",
                top_k=5,
                dataset_id=dataset,
                target_ids=(model,),
            ),
        )
        probes = save_probe_set(
            service,
            "Character probes",
            ProbeSet(probes=(Probe(turns=("Question one", "Question two")),)),
        )
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

        character_db = CharactersRAGDB(tmp_path / "characters.db", "eval-fixture")
        character_id = character_db.add_character_card(
            {
                "name": "Retained character",
                "description": "Owned character",
                "image": b"\x00card image",
            }
        )
        character_db.close()
        char_config = CharacterProbeConfig(
            name="Character bench",
            probe_set_id=probes,
            character_ids=(character_id,),
            target_ids=(model,),
        )
        char_task = save_character_bench(service, char_config)
        from tldw_chatbook.Evals.character_probe.models import (
            CardSnapshot,
            Conversation,
            ConversationTurn,
        )
        from tldw_chatbook.Evals.character_probe.storage import (
            create_probe_run_group,
            save_conversations,
            annotate_turn,
            mark_conversation_reviewed,
        )

        group, run_ids = create_probe_run_group(
            service,
            char_task,
            char_config,
            [
                CardSnapshot(
                    id=character_id, name="Retained character", description="Snapshot"
                )
            ],
            ProbeSet(probes=(Probe(turns=("Question one", "Question two")),)),
            [service.get_model(model)],
        )
        save_conversations(
            service,
            group,
            run_ids,
            [
                Conversation(
                    card_id=character_id,
                    probe_index=0,
                    sample_index=0,
                    target_id=model,
                    turns=(
                        ConversationTurn(user="Question one", reply="Retained reply"),
                    ),
                )
            ],
        )
        annotate_turn(
            service,
            group,
            character_id,
            0,
            0,
            model,
            0,
            ["broke-character"],
            "Retained annotation",
        )
        mark_conversation_reviewed(service, group, character_id, 0, 0, model)

        run = service.create_run("Saved run", task, model)
        service.store_result(
            run,
            "stable",
            {"prompt": "Prompt"},
            "Persisted output",
            metadata={"continuation": "All tokens retained"},
        )
        service.store_run_metrics(run, {"accuracy": (1.0, "accuracy")})
    try:
        yield name, source, service
    finally:
        service.close()


def test_real_domain_complete_capture_and_current_schema(
    domain_store, tmp_path, monkeypatch
):
    name, source, service = domain_store
    a = adapter(name)
    with closing(sqlite3.connect(source)) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        schema = tuple(
            r[0]
            for r in conn.execute(
                "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
            )
        )
    assert dict(a.schema_policy().schema_sql)[version] == schema
    expected = records(source)
    service.close()
    authority = application_authority(tmp_path, source, monkeypatch)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    item = a.discover(context(source, name + "_db_path"))[0]
    assert item.status == "included"
    before = source.read_bytes()
    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        with session.capture_scope((source,), stage):
            a.capture(item, stage / "snapshot.db", Event())
            assert a.validate(stage / "snapshot.db") == ()
            a.relocate(stage / "snapshot.db", {"external": tmp_path / "external"})
    assert records(stage / "snapshot.db") == expected
    assert source.read_bytes() == before


@pytest.mark.parametrize("kind", ["future", "schema"])
def test_domain_refuses_unqualified_schemas(domain_store, kind):
    name, source, service = domain_store
    service.close()
    with closing(sqlite3.connect(source)) as conn:
        conn.execute(
            "PRAGMA user_version=999"
            if kind == "future"
            else "CREATE TABLE injected(payload)"
        )
        conn.commit()
    assert adapter(name).validate(source) == (
        ("unsupported_schema_version" if kind == "future" else "unsupported_schema"),
    )


def test_real_historical_research_v0_capture_without_migration(tmp_path, monkeypatch):
    from tldw_chatbook.Research_Interop.recovery import recovery_adapters

    # Exact pre-lease _init_schema SQL extracted from b6ba7d013^; independent
    # of today's evolving constructor and version stamp.
    script = "\n                CREATE TABLE IF NOT EXISTS research_sessions (\n                    id TEXT PRIMARY KEY,\n                    title TEXT NOT NULL,\n                    query TEXT NOT NULL,\n                    status TEXT NOT NULL DEFAULT 'active',\n                    notes TEXT,\n                    created_at TEXT NOT NULL,\n                    updated_at TEXT NOT NULL,\n                    deleted INTEGER NOT NULL DEFAULT 0,\n                    client_id TEXT NOT NULL DEFAULT 'local',\n                    version INTEGER NOT NULL DEFAULT 1\n                );\n                CREATE TABLE IF NOT EXISTS research_runs (\n                    id TEXT PRIMARY KEY,\n                    session_id TEXT,\n                    query TEXT NOT NULL,\n                    status TEXT NOT NULL DEFAULT 'running',\n                    phase TEXT NOT NULL DEFAULT 'local_planning',\n                    control_state TEXT NOT NULL DEFAULT 'running',\n                    progress_percent REAL,\n                    progress_message TEXT,\n                    source_policy TEXT NOT NULL DEFAULT 'balanced',\n                    autonomy_mode TEXT NOT NULL DEFAULT 'checkpointed',\n                    limits_json TEXT NOT NULL DEFAULT '{}',\n                    provider_overrides_json TEXT NOT NULL DEFAULT '{}',\n                    chat_handoff_json TEXT NOT NULL DEFAULT '{}',\n                    follow_up_json TEXT NOT NULL DEFAULT '{}',\n                    created_at TEXT NOT NULL,\n                    updated_at TEXT NOT NULL,\n                    deleted INTEGER NOT NULL DEFAULT 0,\n                    client_id TEXT NOT NULL DEFAULT 'local',\n                    version INTEGER NOT NULL DEFAULT 1,\n                    FOREIGN KEY(session_id) REFERENCES research_sessions(id)\n                );\n                CREATE TABLE IF NOT EXISTS research_run_events (\n                    id INTEGER PRIMARY KEY AUTOINCREMENT,\n                    run_id TEXT NOT NULL,\n                    event TEXT NOT NULL,\n                    data_json TEXT NOT NULL DEFAULT '{}',\n                    created_at TEXT NOT NULL,\n                    FOREIGN KEY(run_id) REFERENCES research_runs(id)\n                );\n                CREATE TABLE IF NOT EXISTS research_checkpoints (\n                    id TEXT PRIMARY KEY,\n                    run_id TEXT NOT NULL,\n                    checkpoint_type TEXT NOT NULL,\n                    status TEXT NOT NULL DEFAULT 'pending',\n                    resolution TEXT,\n                    proposed_payload_json TEXT NOT NULL DEFAULT '{}',\n                    user_patch_payload_json TEXT,\n                    created_at TEXT NOT NULL,\n                    updated_at TEXT NOT NULL,\n                    deleted INTEGER NOT NULL DEFAULT 0,\n                    version INTEGER NOT NULL DEFAULT 1,\n                    FOREIGN KEY(run_id) REFERENCES research_runs(id)\n                );\n                CREATE TABLE IF NOT EXISTS research_artifacts (\n                    id INTEGER PRIMARY KEY AUTOINCREMENT,\n                    run_id TEXT NOT NULL,\n                    artifact_name TEXT NOT NULL,\n                    content_type TEXT NOT NULL,\n                    content_json TEXT,\n                    content_text TEXT,\n                    created_at TEXT NOT NULL,\n                    UNIQUE(run_id, artifact_name),\n                    FOREIGN KEY(run_id) REFERENCES research_runs(id)\n                );\n                "
    source = tmp_path / "historical.db"
    with closing(sqlite3.connect(source)) as conn:
        conn.executescript(script)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
        assert "lease_owner" not in {
            r[1] for r in conn.execute("PRAGMA table_info(research_runs)")
        }
        assert (
            len(
                conn.execute(
                    "SELECT name FROM sqlite_schema WHERE type='table' AND name LIKE 'research_%'"
                ).fetchall()
            )
            == 5
        )
        conn.execute(
            "INSERT INTO research_runs(id,query,created_at,updated_at) VALUES ('old-run','Historical paid work','2026-01-01','2026-01-01')"
        )
        conn.commit()
    a = recovery_adapters()[0]
    authority = application_authority(tmp_path, source, monkeypatch)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    before = records(source)
    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        with session.capture_scope((source,), stage):
            a.capture(
                a.discover(context(source, "research_db_path"))[0],
                stage / "old.db",
                Event(),
            )
    assert records(stage / "old.db") == before
    assert records(source) == before
    # Execute only installed migration SQL against the known staged historical
    # layout to prove the declared transition, never an ordinary constructor.
    with closing(sqlite3.connect(stage / "old.db")) as conn:
        for statement in a.schema_policy().migration_steps[0][2]:
            conn.execute(statement)
        conn.commit()
    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        with session.capture_scope((source,), stage):
            assert a.validate(stage / "old.db") == ()


@pytest.fixture
def study_store(tmp_path):
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Study_Interop.local_study_service import LocalStudyService
    from tldw_chatbook.Study_Interop.local_quiz_service import LocalQuizService

    source = tmp_path / "study.db"
    db = CharactersRAGDB(source, "recovery-study")
    study, quiz = LocalStudyService(db), LocalQuizService(db)
    deck = study.create_deck(name="Retained deck")
    uploaded = tmp_path / "asset.bin"
    uploaded.write_bytes(b"\x00retained study bytes\xff")
    asset = study.upload_flashcard_asset(uploaded)
    study.create_flashcard(
        deck_id=deck["id"], front=asset["markdown_snippet"], back="Answer"
    )
    q = quiz.create_quiz(name="Retained quiz")
    question = quiz.create_question(
        q["id"],
        question_type="multiple_choice",
        question_text="Question",
        options=["A", "B"],
        correct_answer="A",
    )
    attempt = quiz.start_attempt(q["id"])
    quiz.submit_attempt(
        attempt["id"], answers=[{"question_id": question["id"], "answer": "A"}]
    )
    db.close()
    yield source, asset
    db.close()


def test_shared_study_quiz_lossless_bytes_and_semantics(
    study_store, tmp_path, monkeypatch
):
    from tldw_chatbook.Study_Interop.recovery import recovery_adapters
    from tldw_chatbook.DB.recovery_core import core_adapters

    source, asset = study_store
    config = context(source, "chachanotes_db_path")
    owners = (core_adapters()[0],) + recovery_adapters()
    items = [a.discover(config)[0] for a in owners]
    assert len({i.shared_group for i in items}) == 1
    authority = application_authority(tmp_path, source, monkeypatch)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    expected = records(source)
    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        with session.capture_scope((source,), stage):
            owners[1].capture(items[1], stage / "shared.db", Event())
            for a, i in zip(owners[1:], items[1:]):
                assert (
                    a.validate_dependencies(
                        i,
                        stage / "shared.db",
                        {items[0].logical_id: stage / "shared.db"},
                    )
                    == ()
                )
                a.relocate(stage / "shared.db", {})
    assert list(stage.glob("*.db")) == [stage / "shared.db"]
    assert records(stage / "shared.db") == expected
    with closing(sqlite3.connect(stage / "shared.db")) as conn:
        assert (
            conn.execute(
                "SELECT content FROM flashcard_assets WHERE asset_uuid=?",
                (asset["asset_uuid"],),
            ).fetchone()[0]
            == b"\x00retained study bytes\xff"
        )


def test_missing_shared_asset_and_quiz_snapshot_references(study_store):
    from tldw_chatbook.Study_Interop.recovery import recovery_adapters

    source, asset = study_store
    study, quiz = recovery_adapters()
    with closing(sqlite3.connect(source)) as conn:
        conn.execute(
            "DELETE FROM flashcard_assets WHERE asset_uuid=?", (asset["asset_uuid"],)
        )
        conn.execute('UPDATE quiz_attempts SET answers=\'[{"question_id":"absent"}]\'')
        conn.commit()
    assert study.validate(source) == ("missing_required_asset",)
    assert quiz.validate(source) == ("invalid_domain_reference",)


def shared_items(paths, profiles):
    result = []
    for source, profile in zip(paths, profiles):
        for owner in ("db.chachanotes.primary", "study.local", "quiz.local"):
            result.append(
                StorageItem(
                    owner,
                    f"profile:{profile}:{owner}",
                    source,
                    "included",
                    (),
                    shared_group="shared:chachanotes:profile:" + profile,
                )
            )
    return tuple(result)


@pytest.mark.parametrize("alias", ["same", "hardlink", "distinct"])
def test_shared_cohort_profiles_and_stable_scope(tmp_path, alias):
    import os
    from tldw_chatbook.Backup_Recovery.inventory import (
        _merge_chachanotes_cohort,
        classify_entries,
    )

    source = tmp_path / "first.db"
    source.write_bytes(b"owned bytes")
    other = tmp_path / "second.db"
    if alias == "hardlink":
        os.link(source, other)
    elif alias == "distinct":
        other.write_bytes(b"other owned bytes")
    else:
        other = source
    items = shared_items((source, other), ("one", "two"))
    merged, issues = _merge_chachanotes_cohort(items)
    assert not issues
    assert len({i.shared_group for i in merged}) == (2 if alias == "distinct" else 1)
    before = classify_entries(merged)
    assert before.complete
    if alias == "same":
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"new ordinary content")
        replacement.replace(source)
        after, _ = _merge_chachanotes_cohort(items)
        assert classify_entries(after).scope_digest == before.scope_digest


@pytest.mark.parametrize("fault", ["mismatch", "forged"])
def test_shared_cohort_original_bad_declarations_not_laundered(tmp_path, fault):
    from tldw_chatbook.Backup_Recovery.inventory import (
        _merge_chachanotes_cohort,
        classify_entries,
    )

    source = tmp_path / "first"
    source.write_bytes(b"a")
    other = tmp_path / "other"
    other.write_bytes(b"b")
    items = list(shared_items((source,), ("one",)))
    if fault == "mismatch":
        items[1] = replace(items[1], path=other)
    else:
        items = [replace(i, shared_group="forged") for i in items]
    result, issues = _merge_chachanotes_cohort(tuple(items))
    assert result == tuple(items)
    assert issues == (
        (
            "shared_identity_mismatch"
            if fault == "mismatch"
            else "invalid_shared_declaration"
        ),
    )
    if fault == "mismatch":
        assert not classify_entries(result).complete


@pytest.fixture
def raw_scope(tmp_path, monkeypatch):
    from contextlib import contextmanager

    source = tmp_path / "definition.yaml"
    source.write_bytes(b"task_types: [generation]\nprovider_configs: {}\n")
    authority = application_authority(tmp_path, source, monkeypatch)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)

    @contextmanager
    def enter():
        with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
            with session.capture_scope((source,), stage):
                yield

    return source, stage, enter


def test_owned_definition_capture_exact_bytes_private_mode_and_native_retirement(
    raw_scope,
):
    import os
    from tldw_chatbook.Evals.recovery import recovery_adapters
    from tldw_chatbook.Backup_Recovery.admission import _local

    source, stage, enter = raw_scope
    a = recovery_adapters()[1]
    with enter():
        a.capture(
            StorageItem(
                a.owner_id, "profile:fixture:eval.definitions", source, "included", ()
            ),
            stage / "saved.yaml",
            Event(),
        )
        assert a.validate(stage / "saved.yaml") == ()
        assert _local.capture_scope.resources == []
    assert (stage / "saved.yaml").read_bytes() == source.read_bytes()
    assert (stage / "saved.yaml").stat().st_mode & 0o777 == 0o600
    assert a.schema_policy() is not None


@pytest.mark.parametrize(
    "fault",
    [
        "no_authority",
        "owner",
        "source",
        "source_symlink",
        "stage",
        "existing",
        "cancel",
        "limit",
        "bool_limit",
    ],
)
def test_raw_capture_refusals_preserve_authority_and_existing_data(
    raw_scope, fault, tmp_path
):
    from tldw_chatbook.Backup_Recovery.storage_admission import copy_capture_file
    from tldw_chatbook.Backup_Recovery.admission import _local

    source, stage, enter = raw_scope
    destination = stage / "saved.yaml"
    cancel = Event()
    owner = "eval.definitions"
    budget = 4096
    if fault == "owner":
        owner = "ordinary.owner"
    if fault == "existing":
        destination.write_bytes(b"existing important bytes")
    if fault == "cancel":
        cancel.set()
    if fault == "limit":
        budget = 3
    if fault == "bool_limit":
        budget = True
    if fault == "no_authority":
        with pytest.raises(Exception, match="capture_requires_maintenance"):
            copy_capture_file(owner, source, destination, cancel, max_bytes=budget)
        assert not destination.exists()
        return
    with enter():
        if fault == "source":
            retained = tmp_path / "old"
            source.rename(retained)
            source.write_bytes(b"substitution")
        if fault == "source_symlink":
            alias = tmp_path / "alias"
            alias.symlink_to(source)
            source = alias
        if fault == "stage":
            stage.rename(tmp_path / "oldstage")
            stage.mkdir(mode=0o700)
        with pytest.raises((OSError, ValueError, RuntimeError)):
            copy_capture_file(owner, source, destination, cancel, max_bytes=budget)
        assert _local.capture_scope.resources == []
    if fault == "existing":
        assert destination.read_bytes() == b"existing important bytes"


@pytest.mark.parametrize(
    "data",
    [
        b"!!python/object/apply:os.system [echo forbidden]",
        b"[not, a, mapping]",
        b"a: &loop [*loop]",
    ],
)
def test_yaml_definition_schema_refuses_unsafe_or_nonportable_values(tmp_path, data):
    from tldw_chatbook.Evals.recovery import recovery_adapters

    candidate = tmp_path / "bad.yaml"
    candidate.write_bytes(data)
    assert recovery_adapters()[1].validate(candidate)


def test_disabled_optional_features_still_discover_custom_durable_stores(domain_store):
    name, source, _ = domain_store
    config = context(source, name + "_db_path")
    config[name] = {"enabled": False}
    item = adapter(name).discover(config)[0]
    assert item.path == source and item.status == "included"


def test_discovery_factories_import_no_engines_or_store_constructors(tmp_path):
    import os
    import subprocess
    import sys

    script = r"""
import importlib, importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('tldw_chatbook.config','tldw_chatbook.DB.Evals_DB','tldw_chatbook.Research_Interop.local_research_service','tldw_chatbook.Writing_Interop.local_writing_service') or any(x in fullname for x in ('eval_runner','local_research_engine','torch','transformers','openai','anthropic')):
            raise AssertionError('forbidden runtime import: '+fullname)
sys.meta_path.insert(0,Block())
from pathlib import Path
from tldw_chatbook.Backup_Recovery.models import DiscoveryContext,DISCOVERY_CONTEXT_KEY
config={DISCOVERY_CONTEXT_KEY:DiscoveryContext(Path(sys.argv[1]),'isolated')}
for package in ('Research_Interop','Writing_Interop','Study_Interop','Evals'):
    for a in importlib.import_module('tldw_chatbook.'+package+'.recovery').recovery_adapters():
        assert a.schema_policy() is not None
        assert a.discover(config)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "config.toml")],
        env=dict(os.environ),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("package", list(PACKAGES.values()))
def test_lazy_public_exports_preserve_original_object_identity(package):
    module = importlib.import_module("tldw_chatbook." + package)
    assert set(module.__all__) <= set(dir(module))
    for name in module.__all__:
        original = importlib.import_module(
            "tldw_chatbook." + package + "." + module._EXPORTS[name]
        )
        assert getattr(module, name) is getattr(original, name)
    with pytest.raises(AttributeError):
        getattr(module, "missing_public_export")


def test_real_config_save_process_excluded_until_capture_retires(raw_scope, tmp_path):
    import os
    import subprocess
    import sys

    source, stage, enter = raw_scope
    script = r"""
import sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery import bootstrap
bootstrap.default_bootstrap_root=lambda:Path(sys.argv[1])
from tldw_chatbook.Evals.config_loader import EvalConfigLoader
loader=EvalConfigLoader(sys.argv[2])
loader.update({'saved_by_child':True})
print('READY',flush=True)
loader.save()
print('SAVED',flush=True)
"""
    proc = None
    try:
        with enter():
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(tmp_path / "bootstrap"),
                    str(source),
                ],
                env=dict(os.environ),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert proc.stdout.readline().strip() == "READY"
            with pytest.raises(subprocess.TimeoutExpired):
                proc.communicate(timeout=0.25)
            assert b"saved_by_child" not in source.read_bytes()
        stdout, stderr = proc.communicate(timeout=15)
        assert proc.returncode == 0, stderr
        assert "SAVED" in stdout and b"saved_by_child: true" in source.read_bytes()
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.communicate()


@pytest.mark.parametrize("name", ["research", "writing"])
@pytest.mark.parametrize("failed", [False, True])
def test_real_file_owner_operations_retire_native_connections(tmp_path, name, failed):
    from tldw_chatbook.Research_Interop.local_research_service import (
        LocalResearchService,
    )
    from tldw_chatbook.Writing_Interop.local_writing_service import LocalWritingService

    service = {"research": LocalResearchService, "writing": LocalWritingService}[name](
        tmp_path / (name + ".db")
    )
    connection = service._connect()
    try:
        with connection:
            connection.execute("CREATE TABLE recovery_retirement_marker(value)")
            if failed:
                raise ValueError("fixture_abort")
    except ValueError:
        assert failed
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")
    service.close()


@pytest.mark.parametrize("name", ["research", "writing"])
def test_domain_committed_wal_and_source_bytes_preserved(tmp_path, monkeypatch, name):
    from tldw_chatbook.Research_Interop.local_research_service import (
        LocalResearchService,
    )
    from tldw_chatbook.Writing_Interop.local_writing_service import LocalWritingService

    source = tmp_path / (name + ".db")
    service = {"research": LocalResearchService, "writing": LocalWritingService}[name](
        source
    )
    with closing(sqlite3.connect(source)) as writer:
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    observer = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
    try:
        observer.execute("BEGIN")
        observer.execute("SELECT count(*) FROM sqlite_schema").fetchone()
        if name == "research":
            service.create_run(query="Committed WAL history")
        else:
            service.create_project(title="Committed WAL novel")
        service.close()
        expected = records(source)
        wal = Path(str(source) + "-wal")
        assert wal.stat().st_size > 32
        before = (source.read_bytes(), wal.read_bytes(), source.stat().st_mtime_ns)
        authority = application_authority(tmp_path, source, monkeypatch)
        stage = tmp_path / "stage"
        stage.mkdir(mode=0o700)
        a = adapter(name)
        with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
            with session.capture_scope((source,), stage):
                a.capture(
                    a.discover(context(source, name + "_db_path"))[0],
                    stage / "snapshot.db",
                    Event(),
                )
        assert (
            source.read_bytes(),
            wal.read_bytes(),
            source.stat().st_mtime_ns,
        ) == before
        assert records(stage / "snapshot.db") == expected
    finally:
        observer.close()
        service.close()


def test_raw_copy_cancels_during_actual_streaming_and_retires_descriptors(raw_scope):
    import threading
    from tldw_chatbook.Backup_Recovery.storage_admission import copy_capture_file
    from tldw_chatbook.Backup_Recovery.admission import _local

    source, stage, enter = raw_scope
    source.write_bytes(b"x" * (64 * 1024**2))
    destination = stage / "large"
    cancel = Event()
    stop = Event()

    def cancel_after_write():
        while not stop.wait(0.001):
            if destination.exists() and destination.stat().st_size >= 1024**2:
                cancel.set()
                return

    worker = threading.Thread(target=cancel_after_write)
    worker.start()
    try:
        with enter():
            with pytest.raises(InterruptedError, match="cancelled"):
                copy_capture_file(
                    "eval.definitions",
                    source,
                    destination,
                    cancel,
                    max_bytes=64 * 1024**2,
                )
            assert _local.capture_scope.resources == []
        assert 0 < destination.stat().st_size < source.stat().st_size
    finally:
        stop.set()
        worker.join(3)
    assert not worker.is_alive()


def test_eval_current_character_references_require_exact_profile_peer(tmp_path):
    from tldw_chatbook.DB.Evals_DB import EvalsDB
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Evals.character_probe.models import CharacterProbeConfig
    from tldw_chatbook.Evals.character_probe.storage import save_character_bench

    source = tmp_path / "eval.db"
    peer = tmp_path / "characters.db"
    db = EvalsDB(source, "fixture")
    chars = CharactersRAGDB(peer, "fixture")
    char = chars.add_character_card({"name": "Owned local character"})
    model = db.create_model("Model", "local", "inert")
    save_character_bench(
        db,
        CharacterProbeConfig(
            name="Bench",
            probe_set_id="external-probe-set",
            character_ids=(char,),
            target_ids=(model,),
        ),
    )
    db.close()
    chars.close()
    a = adapter("evals")
    item = a.discover(context(source, "evals_db_path"))[0]
    key = "profile:fixture:db.chachanotes.primary"
    assert a.validate_dependencies(item, source, {key: peer}) == ()
    assert a.validate_dependencies(
        item, source, {"profile:other:db.chachanotes.primary": peer}
    ) == ("dependency_unavailable",)
    with closing(sqlite3.connect(source)) as conn:
        config = json.loads(
            conn.execute("SELECT config_data FROM eval_tasks").fetchone()[0]
        )
        config["character_ids"] = [char + 1234]
        conn.execute("UPDATE eval_tasks SET config_data=?", (json.dumps(config),))
        conn.commit()
    assert a.validate_dependencies(item, source, {key: peer}) == (
        "invalid_domain_reference",
    )


@pytest.mark.parametrize("fault", ["outside", "symlink", "hardlink", "budget", "owner"])
def test_definition_reader_uses_native_capture_authority(raw_scope, tmp_path, fault):
    import os
    from tldw_chatbook.Backup_Recovery.storage_admission import _read_recovery_file
    from tldw_chatbook.Backup_Recovery.admission import _local

    source, stage, enter = raw_scope
    selected = source
    owner = "eval.definitions"
    limit = 4096
    if fault == "outside":
        selected = tmp_path / "outside"
        selected.write_bytes(b"key: value")
    if fault == "symlink":
        selected = stage / "linked"
        selected.symlink_to(source)
    if fault == "hardlink":
        selected = stage / "hard"
        os.link(source, selected)
    if fault == "budget":
        limit = 2
    if fault == "owner":
        owner = "unregistered"
    with enter():
        with pytest.raises((OSError, ValueError, RuntimeError)):
            _read_recovery_file(owner, selected, max_bytes=limit)
        assert _local.capture_scope.resources == []


def test_definition_reader_accepts_only_bounded_bytes_and_retires_before_return(
    raw_scope,
):
    from tldw_chatbook.Backup_Recovery.storage_admission import _read_recovery_file
    from tldw_chatbook.Backup_Recovery.admission import _local

    source, stage, enter = raw_scope
    with enter():
        assert (
            _read_recovery_file("eval.definitions", source, max_bytes=4096)
            == source.read_bytes()
        )
        assert _local.capture_scope.resources == []
        staged = stage / "candidate"
        staged.write_bytes(b"key: preserved bytes")
        assert (
            _read_recovery_file("eval.definitions", staged, max_bytes=4096)
            == b"key: preserved bytes"
        )
        assert _local.capture_scope.resources == []
    assert (
        _read_recovery_file("eval.definitions", source, max_bytes=4096)
        == source.read_bytes()
    )


def test_registered_cohort_composition_discovers_one_shared_physical_payload(
    study_store, tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import inventory, owner_registry
    from tldw_chatbook.DB.recovery_core import core_adapters
    from tldw_chatbook.Study_Interop.recovery import recovery_adapters

    source, _ = study_store
    selector = tmp_path / "profile.toml"
    selector.write_text(
        '[general]\nusers_name="fixture"\n[database]\nchachanotes_db_path='
        + json.dumps(str(source))
        + "\n"
    )
    monkeypatch.setattr(owner_registry, "_adapters", {})
    for a in (core_adapters()[0],) + recovery_adapters():
        owner_registry.register(a)
    found = inventory.discover((selector,))
    cohort = [
        i
        for i in found.items
        if i.owner in {"db.chachanotes.primary", "study.local", "quiz.local"}
    ]
    assert len(cohort) == 3
    assert all(i.path == source and i.status == "included" for i in cohort)
    assert len({i.shared_group for i in cohort}) == 1
    assert not (
        {
            "duplicate_physical_owner",
            "shared_identity_mismatch",
            "invalid_shared_declaration",
        }
        & set(found.issues)
    )


def test_eval_default_definition_selector_is_shared_and_import_light():
    from tldw_chatbook.Evals import _default_config_path
    from tldw_chatbook.Evals.recovery import recovery_adapters
    from tldw_chatbook.Evals.config_loader import EvalConfigLoader

    loader = EvalConfigLoader()
    item = recovery_adapters()[1].discover(
        {
            DISCOVERY_CONTEXT_KEY: DiscoveryContext(
                Path("/isolated/profile.toml"), "fixture"
            )
        }
    )[0]
    assert item.path == loader.config_path == _default_config_path()


def test_asset_reference_cannot_resolve_a_valid_identifier_prefix(study_store):
    source, asset = study_store
    with closing(sqlite3.connect(source)) as connection:
        connection.execute(
            "UPDATE flashcards SET front=?",
            ("![missing](flashcard-asset://" + asset["asset_uuid"] + ".missing)",),
        )
        connection.commit()
    assert adapter("study").validate(source) == ("missing_required_asset",)


@pytest.mark.parametrize(
    "operation",
    [
        "copy_source",
        "copy_destination",
        "capture_read",
        "ordinary_read",
        "traversal_parent",
        "symlink_parent",
    ],
)
def test_simulated_parent_close_failure_retains_real_exclusion_until_process_exit(
    tmp_path, monkeypatch, operation
):
    """A child-only close fault leaves a usable FD; an independent process waits."""
    import os
    import subprocess
    import sys

    source_dir = tmp_path / "owned"
    source_dir.mkdir(mode=0o700)
    source = source_dir / "definition.yaml"
    source.write_bytes(b"provider_configs: {}\n")
    if operation == "symlink_parent":
        alias = tmp_path / "alias"
        alias.symlink_to(source_dir, target_is_directory=True)
        source = alias / source.name
    application_authority(tmp_path, source, monkeypatch)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    child_script = r"""
import os, sys
from pathlib import Path
from threading import Event
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
root, source, stage = map(Path, sys.argv[1:4])
operation = sys.argv[4]
bootstrap.default_bootstrap_root = lambda: root
real_os = os
wanted_path = stage if operation == "copy_destination" else source.parent
if operation in ("traversal_parent", "symlink_parent"):
    wanted_path = source.parent.parent
wanted = wanted_path.stat()
state = {"active": False, "fd": None, "calls": 0}
class FaultOS:
    def __getattr__(self, name):
        return getattr(real_os, name)
    def close(self, fd):
        info = real_os.fstat(fd)
        if state["active"] and (info.st_dev, info.st_ino) == (wanted.st_dev, wanted.st_ino):
            state["calls"] += 1
            state["fd"] = fd
            raise OSError("simulated ambiguous parent close; descriptor remains open")
        real_os.close(fd)
# Module-local proxy only: never mutate shared os.close or the parent environment.
bootstrap.os = FaultOS()
storage.os = FaultOS()
original_init = storage._CaptureFileDescriptors.__init__
def activate(self, scope):
    original_init(self, scope)
    state["active"] = True
storage._CaptureFileDescriptors.__init__ = activate
try:
    if operation == "ordinary_read":
        storage._read_recovery_file("eval.definitions", source, max_bytes=4096)
    else:
        authority = admission_authority(root)
        with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
            with session.capture_scope((source,), stage):
                if operation == "capture_read":
                    storage._read_recovery_file("eval.definitions", source, max_bytes=4096)
                else:
                    storage.copy_capture_file("eval.definitions", source, stage / "copy.yaml", Event(), max_bytes=4096)
except (OSError, RuntimeError):
    pass
assert state["fd"] is not None and state["calls"] == 1, state
real_os.listdir(state["fd"])
print("LIVE_PARENT_ONCE", flush=True)
sys.stdin.readline()
real_os.listdir(state["fd"])
assert state["calls"] == 1, state
print("STILL_LIVE_ONCE", flush=True)
os._exit(0)
"""
    observer_script = r"""
import sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
from tldw_chatbook.Backup_Recovery import bootstrap
root = Path(sys.argv[1])
bootstrap.default_bootstrap_root = lambda: root
authority = admission_authority(root)
print("OBSERVER_READY", flush=True)
with authority.maintenance(("core", "bootstrap.unbound"), 10):
    print("ACQUIRED", flush=True)
"""
    processes = []
    try:
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                child_script,
                str(tmp_path / "bootstrap"),
                str(source),
                str(stage),
                operation,
            ],
            env=dict(os.environ),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(child)
        assert child.stdout.readline().strip() == "LIVE_PARENT_ONCE"
        observer = subprocess.Popen(
            [sys.executable, "-c", observer_script, str(tmp_path / "bootstrap")],
            env=dict(os.environ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(observer)
        assert observer.stdout.readline().strip() == "OBSERVER_READY"
        try:
            stdout, stderr = observer.communicate(timeout=0.3)
        except subprocess.TimeoutExpired:
            pass
        else:
            assert observer.returncode == 0 and "ACQUIRED" in stdout, stderr
            pytest.fail(
                "independent maintenance acquired while parent FD remained usable"
            )
        stdout, stderr = child.communicate(input="exit\n", timeout=5)
        assert child.returncode == 0 and "STILL_LIVE_ONCE" in stdout, stderr
        stdout, stderr = observer.communicate(timeout=15)
        assert observer.returncode == 0 and "ACQUIRED" in stdout, stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate()


@pytest.mark.parametrize(
    "failure", ["missing_candidate", "missing_parent", "refused_parent"]
)
def test_failed_ordinary_definition_read_releases_admission(
    tmp_path, monkeypatch, failure
):
    import os
    import subprocess
    import sys
    from tldw_chatbook.Backup_Recovery import storage_admission

    owned = tmp_path / "owned"
    owned.mkdir(mode=0o700)
    application_authority(tmp_path, owned, monkeypatch)
    candidate = owned / "missing.yaml"
    if failure == "missing_parent":
        candidate = owned / "missing" / "definition.yaml"
    if failure == "refused_parent":
        parent = owned / "unsafe"
        parent.mkdir(mode=0o700)
        candidate = parent / "definition.yaml"
        candidate.write_bytes(b"provider_configs: {}\n")
        parent.chmod(0o777)
    before = dict(storage_admission._holds)
    try:
        with pytest.raises(OSError):
            storage_admission._read_recovery_file(
                "eval.definitions", candidate, max_bytes=4096
            )
    finally:
        if failure == "refused_parent":
            candidate.parent.chmod(0o700)
    assert storage_admission._holds == before
    script = r"""
import sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
with admission_authority(Path(sys.argv[1])).maintenance(("core", "bootstrap.unbound"), 1):
    print("ACQUIRED")
"""
    observer = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "bootstrap")],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert observer.returncode == 0 and "ACQUIRED" in observer.stdout, observer.stderr


@pytest.mark.parametrize(
    "failure",
    ["no_authority", "invalid_item", "cancel", "existing", "invalid_snapshot"],
)
def test_domain_sqlite_capture_safety_guards_preserved(
    domain_store, tmp_path, monkeypatch, failure
):
    name, source, _ = domain_store
    a = adapter(name)
    item = a.discover(context(source, name + "_db_path"))[0]
    stage = tmp_path / "guard-stage"
    stage.mkdir(mode=0o700)
    destination = stage / "snapshot.db"
    cancel = Event()
    if failure == "no_authority":
        with pytest.raises(ValueError, match="capture_requires_maintenance"):
            a.capture(item, destination, cancel)
        assert not destination.exists()
        return
    authority = application_authority(tmp_path, source, monkeypatch)
    expected = {
        "invalid_item": "invalid_capture_item",
        "cancel": "cancelled",
        "existing": "capture_destination_exists",
        "invalid_snapshot": "unsupported_schema",
    }[failure]
    if failure == "invalid_item":
        item = replace(item, owner="unrelated.owner")
    if failure == "cancel":
        cancel.set()
    if failure == "existing":
        destination.write_bytes(b"preserve existing content")
    if failure == "invalid_snapshot":
        with closing(sqlite3.connect(source)) as conn:
            conn.execute("CREATE TABLE unqualified_schema(value)")
            conn.commit()
    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        with session.capture_scope((source,), stage):
            with pytest.raises((ValueError, OSError), match=expected):
                a.capture(item, destination, cancel)
    if failure == "existing":
        assert destination.read_bytes() == b"preserve existing content"
    elif failure != "invalid_snapshot":
        assert not destination.exists()
