"""Safety and independent eligibility oracles for fresh qualification fixtures."""

import subprocess
from pathlib import Path

import pytest

from Tests.Benchmarks import character_qualification_fixture as fixture


@pytest.mark.parametrize("kind", ["directory", "file", "symlink", "relative", "real"])
def test_reservation_refuses_existing_or_non_disposable_targets(tmp_path, kind):
    target = tmp_path / "owned"
    if kind == "directory":
        target.mkdir()
    elif kind == "file":
        target.write_text("Keep me")
    elif kind == "symlink":
        target.symlink_to(tmp_path, target_is_directory=True)
    elif kind == "relative":
        target = Path("relative-fixture")
    else:
        target = Path("/Users/qualification-must-not-create-this")
    with pytest.raises((ValueError, FileExistsError)):
        fixture.reserve_root(target)
    if kind == "file":
        assert target.read_text() == "Keep me"


def test_reservation_does_not_follow_an_outside_parent_link(tmp_path):
    link = tmp_path / "outside"
    link.symlink_to("/Users", target_is_directory=True)
    with pytest.raises(ValueError):
        fixture.reserve_root(link / "qualification-must-not-create-this")


def test_private_environment_is_complete_without_inherited_credentials(
    tmp_path, monkeypatch
):
    root = fixture.reserve_root(tmp_path / "fresh")
    monkeypatch.setenv("OPENAI_API_KEY", "not-for-the-fixture")
    monkeypatch.setenv("TLDW_CONFIG_PATH", "/some/ambient/config.toml")
    env = fixture.isolated_environment(root)
    for name in (
        "HOME",
        "USERPROFILE",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "TIKTOKEN_CACHE_DIR",
        "TMPDIR",
        "TLDW_CONFIG_PATH",
    ):
        assert Path(env[name]).resolve().is_relative_to(root)
    assert env["PYTHON_KEYRING_BACKEND"] == "keyring.backends.null.Keyring"
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert "OPENAI_API_KEY" not in env
    assert "PYTHONPATH" not in env


def test_source_guard_refuses_wrong_head_and_dirty_tracked_or_untracked_source(
    tmp_path,
):
    repo = tmp_path / "source"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True
        ).strip()

    git("init", "-q")
    (repo / "source.txt").write_text("Frozen")
    git("add", "source.txt")
    git(
        "-c",
        "user.name=Qualification",
        "-c",
        "user.email=qualification@example.invalid",
        "commit",
        "-qm",
        "freeze",
    )
    head = git("rev-parse", "HEAD")
    assert fixture.verify_source(repo, head) == head
    with pytest.raises(ValueError, match="head"):
        fixture.verify_source(repo, "0" * 40)
    (repo / "untracked.txt").write_text("Dirty")
    with pytest.raises(ValueError, match="clean"):
        fixture.verify_source(repo, head)
    (repo / "untracked.txt").unlink()
    (repo / "source.txt").write_text("Changed")
    with pytest.raises(ValueError, match="clean"):
        fixture.verify_source(repo, head)


def test_manifest_is_independent_complete_and_not_mutated_by_a_consumer():
    rows = fixture.load_manifest()
    assert len(rows) == len({row["id"] for row in rows}) == 30
    assert rows[0]["query"] == "querytoken0"
    assert rows[0]["expected"] == ["perf-00000"]
    assert rows[8]["expected"] == ["perf-00008"]
    assert rows[16]["expected"] == rows[24]["expected"] == []
    assert rows[27]["query"] == "海辺の物語"
    assert rows[28]["query"] == "caféclair"
    rows[0]["expected"].clear()
    assert fixture.load_manifest()[0]["expected"] == ["perf-00000"]


def test_navigation_fixture_config_does_not_require_provider_setup():
    import tomllib

    from tldw_chatbook.Chat.console_onboarding_state import console_setup_is_blocking
    from tldw_chatbook.Chat.console_session_settings import ConsoleSettingsReadiness

    config = tomllib.loads(
        Path(fixture.__file__)
        .with_name("character_qualification_config.toml")
        .read_text()
    )
    onboarding = config.get("console", {}).get("onboarding", {})
    # Navigation qualification represents a user with saved chats, not first-send
    # onboarding or send readiness. No fake credential or provider-ready result.
    readiness = ConsoleSettingsReadiness("Blocked", "No key", False)
    assert not console_setup_is_blocking(
        readiness=readiness,
        has_model=False,
        first_send_completed=onboarding.get("first_send_completed", False),
    )
    assert readiness.native_send_supported is False


def test_tiny_real_corpus_matches_independent_queries_and_excludes_nonselected_data(
    tmp_path,
):
    from tldw_chatbook.Character_Chat.character_conversation_navigation import (
        CharacterConversationNavigationService,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.character_conversation_search import (
        SelectedBranchEligibilityProjector,
    )

    root = fixture.reserve_root(tmp_path / "tiny")
    receipt = fixture.build_corpus(root, conversations=30, messages_per_chat=3)
    assert receipt["counts"] == [30, 94, 30]
    assert receipt["eligible_messages"] == 90
    assert receipt["registered_handles_after_cleanup"] == 0
    assert receipt["status"] == "built"
    database = CharactersRAGDB(root / "corpus.sqlite", client_id="fixture-reader")
    try:
        service = CharacterConversationNavigationService(database)
        assert service.keyword_index_status().value == "ready"
        for query in fixture.load_manifest():
            assert [
                row.target.conversation_id
                for row in service.keyword_search(query["query"]).rows
            ] == query["expected"]
        # These literals are an independent negative oracle, not builder output.
        for canary in (
            "SYSTEM_CANARY",
            "TOOL_CANARY",
            "NON_SELECTED_CANARY",
            "DELETED_CANARY",
            "THINKING_CANARY",
            "ATTACHMENT_CANARY",
        ):
            assert service.keyword_search(canary).rows == ()
        document = SelectedBranchEligibilityProjector(database).project("perf-00000")
        assert document is not None
        assert "Visible message 2" in document.body
        assert len(document.body.split("\n\n")) == 3
        assert [
            row.target.conversation_id
            for row in service.keyword_search("Fixture", limit=3).rows
        ] == ["perf-00029", "perf-00028", "perf-00027"]
        assert (
            database.get_connection().execute("PRAGMA quick_check").fetchone()[0]
            == "ok"
        )
        timestamps = dict(
            database.get_connection()
            .execute(
                "SELECT id, CAST(timestamp AS TEXT) FROM messages WHERE conversation_id = 'perf-00000'"
            )
            .fetchall()
        )
        assert timestamps["perf-00000-m00"] == "2026-01-01T00:00:00+00:00"
        assert timestamps["perf-00000-m02"] == "2026-01-01T00:00:00.002000+00:00"
        assert timestamps["fixture-system"] == "2026-01-01T00:00:00.001500+00:00"
        assert timestamps["fixture-deleted"] == "2026-01-01T00:00:00.000700+00:00"
    finally:
        database.close()
    assert database.registered_connection_count() == 0
    assert not Path(str(root / "corpus.sqlite") + "-wal").exists()


def test_failed_seed_retires_exact_database_and_keeps_failure_receipt(
    tmp_path, monkeypatch
):
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    original = CharactersRAGDB.add_message
    captured = []

    def fail_after_real_write(self, data):
        original(self, data)
        captured.append(self)
        raise RuntimeError("controlled seed failure")

    monkeypatch.setattr(CharactersRAGDB, "add_message", fail_after_real_write)
    root = fixture.reserve_root(tmp_path / "failed")
    with pytest.raises(RuntimeError, match="controlled seed failure"):
        fixture.build_corpus(root, conversations=30, messages_per_chat=3)
    assert captured and captured[0].registered_connection_count() == 0
    import json

    assert json.loads((root / "build-receipt.json").read_text())["status"] == "failed"
    # Failure preserves the owned database for diagnosis, never deletes it.
    reader = CharactersRAGDB(root / "corpus.sqlite", client_id="failure-reader")
    try:
        assert (
            reader.get_connection().execute("PRAGMA quick_check").fetchone()[0] == "ok"
        )
    finally:
        reader.close()


def test_keyword_receipt_uses_real_queries_preserves_source_and_retires_worker(
    tmp_path,
):
    import asyncio

    root = fixture.reserve_root(tmp_path / "source-corpus")
    build = fixture.build_corpus(root, conversations=30, messages_per_chat=3)
    output = fixture.reserve_root(tmp_path / "query-receipt")
    receipt = asyncio.run(
        fixture.measure_keyword(
            output,
            root / "corpus.sqlite",
            root / "build-receipt.json",
            expected_head="tiny-test",
            repetitions=1,
            warmups=0,
        )
    )
    assert receipt["status"] == "smoke"
    assert receipt["correctness_failures"] == []
    assert len(receipt["timings"]) == 30
    assert receipt["source_unchanged"] is True
    assert receipt["corpus_digest"] == build["corpus_digest"]
    assert receipt["registered_handles_after_cleanup"] == 0
    assert receipt["owned_database_descriptors_after_cleanup"] == []


def test_native_fixture_has_paged_cards_exact_markers_and_unavailable_recovery(
    tmp_path,
):
    from Tests.Benchmarks.character_native_fixture import build_native
    from tldw_chatbook.Character_Chat.character_conversation_navigation import (
        CharacterConversationNavigationService,
        ResolvedLocalCharacterKey,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    root = fixture.reserve_root(tmp_path / "native")
    receipt = build_native(root)
    assert receipt["status"] == "prepared-not-qualified"
    assert receipt["counts"] == [30, 60, 28]
    assert receipt["registered_handles_after_cleanup"] == 0
    database = CharactersRAGDB(
        root / "native.sqlite", client_id="native-fixture-reader"
    )
    try:
        service = CharacterConversationNavigationService(database)
        cards = receipt["cards"]
        authority = database.get_local_authority_id()
        for name in ("Amber", "Indigo", "Cedar", "Copper"):
            page = service.page_for_character(
                ResolvedLocalCharacterKey(authority, cards[name])
            )
            assert page.total == 7
            assert [row.target.conversation_id for row in page.rows] == [
                f"native-{name.lower()}-{i:02d}" for i in range(7, 0, -1)
            ]
        assert (
            service.page_for_character(
                ResolvedLocalCharacterKey(authority, cards["Empty Atlas"])
            ).total
            == 0
        )
        assert service.unavailable_page().total == 2
        assert [
            row.target.conversation_id
            for row in service.keyword_search("NATIVE_MARKER_AMBER_01").rows
        ] == ["native-amber-01"]
        assert service.keyword_search("NATIVE_MARKER_UNAVAILABLE_01").rows == ()
        assert (
            database.get_connection()
            .execute(
                "SELECT CAST(timestamp AS TEXT) FROM messages WHERE id = 'native-amber-01-user'"
            )
            .fetchone()[0]
            == "2026-01-01T00:00:00+00:00"
        )
        assert (
            database.get_connection()
            .execute(
                "SELECT CAST(timestamp AS TEXT) FROM messages WHERE id = 'native-amber-01-assistant'"
            )
            .fetchone()[0]
            == "2026-01-01T00:00:01+00:00"
        )
    finally:
        database.close()


def test_keyword_receipt_rejects_changed_corpus_before_copying(tmp_path):
    import asyncio

    root = fixture.reserve_root(tmp_path / "source-corpus")
    fixture.build_corpus(root, conversations=30, messages_per_chat=3)
    with (root / "corpus.sqlite").open("ab") as stream:
        stream.write(b"changed")
    output = fixture.reserve_root(tmp_path / "query-receipt")
    with pytest.raises(ValueError, match="digest"):
        asyncio.run(
            fixture.measure_keyword(
                output,
                root / "corpus.sqlite",
                root / "build-receipt.json",
                expected_head="tiny-test",
                repetitions=1,
                warmups=0,
            )
        )
    assert not (output / "measurement.sqlite").exists()


def test_measurement_refuses_a_non_disposable_source_before_reading_it(tmp_path):
    import asyncio

    output = fixture.reserve_root(tmp_path / "measurement")
    with pytest.raises(ValueError, match="disposable"):
        asyncio.run(
            fixture.measure_keyword(
                output,
                Path("/Users/not-a-fixture/corpus.sqlite"),
                Path("/Users/not-a-fixture/receipt.json"),
                expected_head="tiny-test",
            )
        )
    assert not (output / "measurement.sqlite").exists()


def test_unbound_build_cannot_be_used_as_current_head_scale_evidence(tmp_path):
    import asyncio
    import json

    root = fixture.reserve_root(tmp_path / "source")
    receipt = fixture.build_corpus(root, conversations=30, messages_per_chat=3)
    receipt.update(
        counts=[10000, 250004, 10000], conversations=10000, eligible_messages=250000
    )
    (root / "build-receipt.json").write_text(json.dumps(receipt))
    output = fixture.reserve_root(tmp_path / "measurement")
    with pytest.raises(ValueError, match="head"):
        asyncio.run(
            fixture.measure_keyword(
                output,
                root / "corpus.sqlite",
                root / "build-receipt.json",
                expected_head="later-head",
                repetitions=1,
                warmups=0,
            )
        )
    assert not (output / "measurement.sqlite").exists()


def test_changed_source_at_measurement_end_retains_failed_samples(
    tmp_path, monkeypatch
):
    import asyncio
    import json

    root = fixture.reserve_root(tmp_path / "source")
    receipt = fixture.build_corpus(root, conversations=30, messages_per_chat=3)
    receipt["head"] = "frozen-head"
    (root / "build-receipt.json").write_text(json.dumps(receipt))
    output = fixture.reserve_root(tmp_path / "measurement")

    def changed_source(*args):
        raise ValueError("source no longer clean")

    monkeypatch.setattr(fixture, "verify_source", changed_source)
    with pytest.raises(RuntimeError, match="qualification failed"):
        asyncio.run(
            fixture.measure_keyword(
                output,
                root / "corpus.sqlite",
                root / "build-receipt.json",
                expected_head="frozen-head",
                repetitions=1,
                warmups=0,
            )
        )
    result = json.loads((output / "keyword-receipt.json").read_text())
    assert result["status"] == "failed"
    assert len(result["timings"]) == 30
    assert any("Source" in reason for reason in result["failures"])


@pytest.mark.parametrize(
    "artifact",
    [
        "build-receipt.json",
        "keyword-receipt.json",
        "native-return.json",
        "ui-evidence/ui-latency-evidence.json",
    ],
)
@pytest.mark.parametrize("mutation", ["dirty", "head"])
def test_cli_final_source_guard_invalidates_receipt_and_preserves_samples(
    tmp_path, artifact, mutation
):
    import json

    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(repository), *args], text=True
        ).strip()

    git("init", "-q")
    tracked = repository / "source.txt"
    tracked.write_text("Original")
    git("add", "source.txt")
    git(
        "-c",
        "user.name=Qualification",
        "-c",
        "user.email=qualification@example.invalid",
        "commit",
        "-qm",
        "freeze",
    )
    head = git("rev-parse", "HEAD")
    root = fixture.reserve_root(tmp_path / "artifacts")
    path = root / artifact
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"status": "passed", "samples": [{"raw_ns": 12345}]}))
    fixture._final_source_guard(root, repository, head)
    assert json.loads(path.read_text())["source_clean_and_exact_after"] is True
    tracked.write_text("Changed")
    if mutation == "head":
        git("add", "source.txt")
        git(
            "-c",
            "user.name=Qualification",
            "-c",
            "user.email=qualification@example.invalid",
            "commit",
            "-qm",
            "change head",
        )
    with pytest.raises(ValueError):
        fixture._final_source_guard(root, repository, head)
    receipt = json.loads(path.read_text())
    assert receipt["status"] == "failed"
    assert receipt["samples"] == [{"raw_ns": 12345}]
    assert receipt["source_clean_and_exact_after"] is False


def test_launcher_rejects_wrong_head_before_production_import_or_profile_creation(
    tmp_path,
):
    import sys

    root = tmp_path / "must-not-be-created"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy,sys; sys.addaudithook(lambda e,a: (_ for _ in ()).throw(RuntimeError('PRODUCTION IMPORT BEFORE GUARD')) if e=='import' and a[0].startswith('tldw_chatbook') else None); sys.argv=['fixture','build','--root',sys.argv[1],'--expected-head','0000000000000000000000000000000000000000','--size','tiny']; runpy.run_module('Tests.Benchmarks.character_qualification_fixture',run_name='__main__')",
            str(root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "head differs" in result.stderr
    assert "PRODUCTION IMPORT BEFORE GUARD" not in result.stderr
    assert not root.exists()


@pytest.mark.asyncio
async def test_cancelled_receipt_drains_its_real_worker_before_terminal_proof(
    tmp_path, monkeypatch
):
    import asyncio
    import json
    import threading

    from tldw_chatbook.Character_Chat.character_conversation_navigation import (
        CharacterConversationNavigationService,
    )

    root = fixture.reserve_root(tmp_path / "source")
    fixture.build_corpus(root, conversations=30, messages_per_chat=3)
    output = fixture.reserve_root(tmp_path / "measurement")
    entered, release = threading.Event(), threading.Event()
    original = CharacterConversationNavigationService.keyword_search

    def held_search(self, *args, **kwargs):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test release was not delivered")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        CharacterConversationNavigationService, "keyword_search", held_search
    )
    task = asyncio.create_task(
        fixture.measure_keyword(
            output,
            root / "corpus.sqlite",
            root / "build-receipt.json",
            expected_head="tiny-test",
            repetitions=1,
            warmups=0,
        )
    )
    try:
        async with asyncio.timeout(5):
            while not entered.is_set():
                await asyncio.sleep(0.005)
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done(), (
            "Caller must not publish terminal proof while its worker still owns SQLite"
        )
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    receipt = json.loads((output / "keyword-receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["registered_handles_after_cleanup"] == 0
    assert receipt["owned_database_descriptors_after_cleanup"] == []
