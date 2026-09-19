"""Actual Shared Visual Identity generation and native maintenance boundaries."""

import hashlib

import pytest

from Tests.Backup_Recovery.test_admission import launch  # noqa: F401 -- fixture
from Tests.Backup_Recovery.test_participant_lifetimes import local_root  # noqa: F401 -- fixture
from Tests.Backup_Recovery.test_compound_chat_source_lifetimes import configured_db  # noqa: F401 -- fixture
from Tests.Character_Chat.test_visual_identity_publication import _png_bytes
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Character_Chat import visual_identity as visual
from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository


@pytest.fixture
def visual_environment(configured_db):
    db, config = configured_db
    profile = config.get_user_data_dir()
    data = _png_bytes((20, 30, 40))
    relative = (
        "packs/profile-00000000000000000000000000000000/versions/original/neutral.png"
    )
    path = profile / "visual_identities" / relative
    path.parent.mkdir(mode=0o700, parents=True)
    path.write_bytes(data)
    actor_id = db.add_character_card({"name": "Shared Visual fixture"})
    graph = VisualIdentityRepository(db).activate_pack(
        pack={
            "title": "Fixture reactions",
            "description": "Synthetic local bytes",
            "default_expression_key": "neutral",
            "source_kind": "manual",
            "source_context": {
                "profile_pack_id": "profile-00000000000000000000000000000000"
            },
        },
        manifest={"fixture": True},
        assets=[
            {
                "expression_key": "neutral",
                "original_expression_key": "neutral",
                "display_label": "Neutral",
                "source_filename": "neutral.png",
                "storage_relpath": relative,
                "content_type": "image/png",
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "width": 16,
                "height": 16,
                "source_context": {},
                "is_animated": False,
                "frame_count": 1,
                "duration_ms": None,
            }
        ],
        actor_kind="character",
        actor_id=actor_id,
    )
    candidate = visual.create_visual_identity_candidate(
        db, actor_kind="character", actor_id=actor_id
    )
    candidate.stage_replacement("neutral", _png_bytes((1, 2, 3)), source="upload")
    return db, profile, actor_id, graph, candidate


def test_shared_visual_publication_finishes_core_activation_after_rename_pause(
    visual_environment, monkeypatch
):
    db, profile, actor_id, graph, candidate = visual_environment
    original = visual._sync_publication_directory
    pauses = []

    def sync_then_pause(directory):
        original(directory)
        versions = (
            profile
            / "visual_identities/packs/profile-00000000000000000000000000000000/versions"
        )
        if (
            any(
                p.name != "original" and not p.name.startswith(".")
                for p in versions.iterdir()
            )
            and not pauses
        ):
            pauses.append(storage._begin_local_pause())

    monkeypatch.setattr(visual, "_sync_publication_directory", sync_then_pause)
    result = None
    error = None
    try:
        try:
            result = visual.publish_visual_identity_candidate(
                db, candidate, user_data_dir=profile
            )
        except Exception as caught:
            error = caught
        assert pauses, "fixture never reached real post-rename boundary"
        assert error is None, (
            f"file renamed but actual core activation newly refused: {error!r}"
        )
        assert result is not None
        assert (
            db._local.conn.execute(
                "SELECT active_version_id FROM visual_identity_bindings WHERE actor_id = ?",
                (str(actor_id),),
            ).fetchone()[0]
            != graph["version"]["id"]
        )
    finally:
        for pause in pauses:
            pause.resume()


def test_runtime_asset_pause_refuses_before_file_read(visual_environment, monkeypatch):
    db, profile, _actor, graph, candidate = visual_environment
    seen = []
    original = visual._read_user_asset

    def observe(*args, **kwargs):
        seen.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(visual, "_read_user_asset", observe)
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(Exception):
            visual.load_visual_identity_asset(
                visual._manifest_asset_from_row(graph["assets"][0]),
                source_kind="manual",
                user_data_dir=profile,
            )
        assert not seen
        assert candidate.replaced_expression_keys == ("neutral",)
    finally:
        pause.resume()


def test_source_candidate_copy_retarget_and_dirty_pause(visual_environment):
    import copy
    from tldw_chatbook.Backup_Recovery import visual_identity_participants as life

    db, profile, _actor, _graph, candidate = visual_environment
    assert life.safe_point(profile) == "needs-user-save/discard"
    copied = copy.copy(candidate)
    with pytest.raises(visual.VisualIdentityPublicationError):
        visual.publish_visual_identity_candidate(db, copied, user_data_dir=profile)
    with pytest.raises(visual.VisualIdentityPublicationError):
        visual.publish_visual_identity_candidate(
            db, candidate, user_data_dir=profile.parent / "other"
        )
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(visual.VisualIdentityPublicationError):
            visual.publish_visual_identity_candidate(
                db, candidate, user_data_dir=profile
            )
        assert life.safe_point(profile) == "needs-user-save/discard"
        assert candidate.replaced_expression_keys == ("neutral",)
        assert not candidate._publishing and not candidate._cancelled
    finally:
        pause.resume()


@pytest.mark.parametrize("when", ["before", "after", "partial"])
def test_shared_native_uncertainty_holds_independent_maintainer(
    visual_environment, local_root, monkeypatch, when, launch
):
    import os
    import select
    import subprocess
    import sys
    import time
    from pathlib import Path
    from tldw_chatbook.Backup_Recovery import visual_identity_participants as life

    if os.environ.get("TASK10_SHARED_VISUAL_CHILD") != when:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                f"{__file__}::test_shared_native_uncertainty_holds_independent_maintainer[{when}]",
                "-q",
                "-p",
                "no:cacheprovider",
                "--tb=short",
            ],
            cwd=Path(__file__).resolve().parents[2],
            env={
                **os.environ,
                "TASK10_SHARED_VISUAL_CHILD": when,
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            capture_output=True,
            text=True,
            timeout=25,
        )
        log = Path(
            f"/private/tmp/chatbook-backup-execution-hgp11i7t/phase13-native-{when}.log"
        )
        log.write_text(result.stdout + result.stderr)
        assert result.returncode == 0, str(log)
        return
    db, profile, _actor, _graph, candidate = visual_environment
    close = life._close_native
    opened = life._open_native
    injected = []

    def close_fault(fd):
        state = getattr(life._local, "state", None)
        if state is not None and state.result is not None and not injected:
            injected.append(fd)
            if when == "after":
                close(fd)
            raise OSError("synthetic close uncertainty")
        close(fd)

    def open_fault(*args, **kwargs):
        state = getattr(life._local, "state", None)
        if state is not None and len(state.descriptors) >= 2 and not injected:
            fd = opened(*args, **kwargs)
            injected.append(fd)
            raise OSError("synthetic partial native constructor")
        return opened(*args, **kwargs)

    monkeypatch.setattr(
        life,
        "_open_native" if when == "partial" else "_close_native",
        open_fault if when == "partial" else close_fault,
    )
    with pytest.raises(visual.VisualIdentityPublicationError) as error:
        visual.publish_visual_identity_candidate(db, candidate, user_data_dir=profile)
    assert injected
    assert any(state.uncertain for state in life._states)
    if when != "partial":
        assert candidate._published
        assert error.value.result.new_version_id != candidate.old_version_id
    db.close_connection()
    # Private child diagnostic only; production startup is never released.
    storage._startups[(os.getpid(), str(local_root))].close()
    observer = launch(local_root / "admission", "maintenance", ("bootstrap.unbound",))
    pause = storage._begin_local_pause()
    try:
        assert not select.select([observer.stdout], [], [], 0.05)[0]
        assert not pause.drain(time.monotonic() + 0.02)
        source = life.source_for(profile)
        source.close_admission()
        assert not source.drain(time.monotonic() + 0.02)
    finally:
        pause.resume()
        observer.kill()
        observer.wait(timeout=5)


def test_actual_canonical_ui_restoration_replacement_save(visual_environment):
    from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen

    db, profile, actor, _graph, candidate = visual_environment
    canonical = PersonasScreen._canonical_visual_identity_assets()
    PersonasScreen._restore_candidate_reaction_rows(candidate, canonical)
    for asset in canonical:
        candidate.stage_replacement(
            asset.expression_key, _png_bytes((3, 4, 5)), source="generated"
        )
    result = visual.publish_visual_identity_candidate(
        db, candidate, user_data_dir=profile
    )
    assert result.new_version_id != candidate.old_version_id
    active = VisualIdentityRepository(db).get_active_actor_pack("character", actor)
    assert len(active["assets"]) == len(canonical)


def test_shared_snapshot_pause_precedes_editor_selector(configured_db, monkeypatch):
    from types import SimpleNamespace
    from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen

    db, _config = configured_db
    screen = PersonasScreen(SimpleNamespace(chachanotes_db=db))
    seen = []
    monkeypatch.setattr(screen, "_editor_or_none", lambda: seen.append(True))
    pause = storage._begin_local_pause()
    try:
        try:
            screen._visual_identity_author_snapshot()
        except RuntimeError:
            pass
        assert not seen
    finally:
        pause.resume()


def test_rejected_duplicate_does_not_release_original_candidate_publish_flag(
    visual_environment,
):
    import os

    db, profile, _actor, _graph, candidate = visual_environment

    def replace_with_duplicate(*args, **kwargs):
        with pytest.raises(
            visual.VisualIdentityPublicationError, match="candidate_publishing"
        ):
            visual.publish_visual_identity_candidate(
                db, candidate, user_data_dir=profile
            )
        assert candidate._publishing, (
            "rejected duplicate erased actual active publication flag"
        )
        os.replace(*args, **kwargs)

    result = visual.publish_visual_identity_candidate(
        db, candidate, user_data_dir=profile, atomic_replace=replace_with_duplicate
    )
    assert result.new_version_id != candidate.old_version_id


@pytest.mark.asyncio
async def test_shared_ui_worker_preserves_existing_and_retires_new_borrower(
    visual_environment, monkeypatch
):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen

    db, _profile, actor, _graph, _candidate = visual_environment
    screen = PersonasScreen(SimpleNamespace(chachanotes_db=db))
    loop = asyncio.get_running_loop()
    execute = loop.run_in_executor
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(
            loop,
            "run_in_executor",
            lambda selected, function, *args: execute(
                executor if selected is None else selected, function, *args
            ),
        )
        for borrowed in (False, True):
            previous = await execute(executor, db.get_connection) if borrowed else None
            outcome = await screen._visual_identity_thread(
                visual.create_visual_identity_candidate,
                db,
                actor_kind="character",
                actor_id=actor,
                task_name="phase13-borrower",
            )
            assert outcome.completed and outcome.error is None

            def observe():
                assert getattr(db._local, "conn", None) is previous
                if previous is not None:
                    assert previous.execute("SELECT 1").fetchone()[0] == 1
                    db.close_connection()

            await execute(executor, observe)
            outcome.value.cancel()
        assert not screen._visual_identity_pending


@pytest.mark.asyncio
async def test_shared_ui_native_retirement_precedes_independent_maintenance(
    visual_environment, local_root, monkeypatch, launch
):
    import asyncio
    import os
    from pathlib import Path
    import select
    import subprocess
    import sys
    import threading
    from types import SimpleNamespace
    from Tests.Backup_Recovery.test_admission import line, release
    from tldw_chatbook.Backup_Recovery import visual_identity_participants as life
    from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen

    if os.environ.get("TASK10_SHARED_RETIRE_CHILD") != "1":
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                f"{__file__}::test_shared_ui_native_retirement_precedes_independent_maintenance",
                "-q",
                "-p",
                "no:cacheprovider",
                "--tb=short",
            ],
            cwd=Path(__file__).resolve().parents[2],
            env={
                **os.environ,
                "TASK10_SHARED_RETIRE_CHILD": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            capture_output=True,
            text=True,
            timeout=25,
        )
        log = Path(
            "/private/tmp/chatbook-backup-execution-hgp11i7t/phase13-retire-child.log"
        )
        log.write_text(result.stdout + result.stderr)
        assert result.returncode == 0, str(log)
        return
    db, profile, _actor, _graph, candidate = visual_environment
    screen = PersonasScreen(SimpleNamespace(chachanotes_db=db))
    entered, finish = threading.Event(), threading.Event()
    original = life._close_native

    def close(fd):
        state = getattr(life._local, "state", None)
        if state is not None and state.result is not None and not entered.is_set():
            entered.set()
            assert finish.wait(8)
        original(fd)

    monkeypatch.setattr(life, "_close_native", close)
    db.close_connection()
    # Private diagnostic startup only, after this child's known quiescent setup.
    storage._startups[(os.getpid(), str(local_root))].close()
    task = asyncio.create_task(
        screen._visual_identity_thread(
            visual.publish_visual_identity_candidate,
            db,
            candidate,
            user_data_dir=profile,
            task_name="phase13-retire",
        )
    )
    observer = None
    try:
        while not entered.is_set():
            if task.done():
                pytest.fail(
                    f"actual source stopped before native close: {task.result()}"
                )
            await asyncio.sleep(0.01)
        observer = launch(
            local_root / "admission", "maintenance", ("bootstrap.unbound",)
        )
        assert not select.select([observer.stdout], [], [], 0.05)[0]
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        assert screen.visual_identity_maintenance_state() == "pending"
        finish.set()
        outcome = await task
        assert (
            outcome.error is None
            and outcome.completed
            and outcome.cancellation is not None
        )
        assert await asyncio.to_thread(line, observer) == "entered"
        assert not screen._visual_identity_pending
        check = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sqlite3,sys,hashlib; from pathlib import Path; c=sqlite3.connect(sys.argv[1]); rows=c.execute('SELECT storage_relpath,sha256 FROM visual_identity_assets').fetchall(); assert rows; assert all(hashlib.sha256((Path(sys.argv[2])/\"visual_identities\"/p).read_bytes()).hexdigest()==h for p,h in rows); c.close()",
                str(db.db_path),
                str(profile),
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert check.returncode == 0, check.stderr
        release(observer)
        observer = None
    finally:
        finish.set()
        await task
        if observer is not None:
            observer.kill()
            observer.wait(timeout=5)


@pytest.mark.asyncio
async def test_shared_queued_ui_cancel_never_enters_source(
    visual_environment, monkeypatch
):
    import asyncio
    from types import SimpleNamespace
    from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen

    db, profile, _actor, _graph, candidate = visual_environment
    screen = PersonasScreen(SimpleNamespace(chachanotes_db=db))
    loop = asyncio.get_running_loop()
    queued = loop.create_future()
    callbacks = []
    monkeypatch.setattr(
        loop,
        "run_in_executor",
        lambda _, function: callbacks.append(function) or queued,
    )
    task = asyncio.create_task(
        screen._visual_identity_thread(
            visual.publish_visual_identity_candidate,
            db,
            candidate,
            user_data_dir=profile,
            task_name="phase13-queued",
        )
    )
    await asyncio.sleep(0)
    task.cancel()
    outcome = await task
    callbacks[0]()
    assert outcome.cancellation is not None
    assert not candidate._published and not candidate._publishing
    assert not screen._visual_identity_pending
    assert candidate.replaced_expression_keys == ("neutral",)


def test_actual_restore_only_dirty_and_copied_candidate_refused(visual_environment):
    import copy
    from tldw_chatbook.Backup_Recovery import visual_identity_participants as life
    from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen

    db, profile, actor, _graph, old = visual_environment
    old.cancel()
    candidate = visual.create_visual_identity_candidate(
        db, actor_kind="character", actor_id=actor
    )
    copied = copy.copy(candidate)
    with pytest.raises(Exception):
        PersonasScreen._restore_candidate_reaction_rows(
            copied, PersonasScreen._canonical_visual_identity_assets()
        )
    PersonasScreen._restore_candidate_reaction_rows(
        candidate, PersonasScreen._canonical_visual_identity_assets()
    )
    assert not candidate._replacements and not candidate._cleared
    assert life.safe_point(profile) == "needs-user-save/discard"
    pause = storage._begin_local_pause()
    try:
        assert life.safe_point(profile) == "needs-user-save/discard"
        assert not candidate._cancelled
    finally:
        pause.resume()
    candidate.cancel()
    assert life.safe_point(profile) == "source-idle"


def test_issued_cleanup_rejects_same_path_other_database_source(visual_environment):
    import os
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db, profile, _actor, _graph, candidate = visual_environment

    def replace_then_fail(*args, **kwargs):
        os.replace(*args, **kwargs)
        raise OSError("synthetic post-rename error")

    with pytest.raises(visual.VisualIdentityPublicationError) as error:
        visual.publish_visual_identity_candidate(
            db, candidate, user_data_dir=profile, atomic_replace=replace_then_fail
        )
    token = error.value.cleanup_candidate_relpath
    assert token is not None
    other = CharactersRAGDB(db.db_path, "same-path-other-source")
    try:
        with pytest.raises(visual.VisualIdentityPublicationError):
            visual.cleanup_visual_identity_publication_candidate(
                other, token, user_data_dir=profile
            )
        assert (profile / "visual_identities" / token).is_dir()
    finally:
        other.close_connection()


@pytest.mark.parametrize("kind", ["positive", "copied", "foreign"])
def test_issued_cleanup_exact_retirement_and_ordinary_copy(visual_environment, kind):
    import os
    from tldw_chatbook.Backup_Recovery import visual_identity_participants as life

    db, profile, _actor, _graph, candidate = visual_environment

    def replace_then_fail(*args, **kwargs):
        os.replace(*args, **kwargs)
        raise OSError("synthetic post-rename error")

    with pytest.raises(visual.VisualIdentityPublicationError) as error:
        visual.publish_visual_identity_candidate(
            db, candidate, user_data_dir=profile, atomic_replace=replace_then_fail
        )
    token = error.value.cleanup_candidate_relpath
    assert token is not None
    root = profile / "visual_identities" / token
    source = life.db_source(db)
    assert source.failures and not any(s.source is source for s in life._states)
    if kind == "foreign":
        member = root / "manifest.json"
        original = member.read_bytes()
        member.rename(root.parent / "preserved-original-manifest")
        member.write_bytes(b"foreign data")
        with pytest.raises(visual.VisualIdentityPublicationError):
            visual.cleanup_visual_identity_publication_candidate(
                db, token, user_data_dir=profile
            )
        assert member.read_bytes() == b"foreign data"
        assert (root.parent / "preserved-original-manifest").read_bytes() == original
        assert source.failures
    else:
        supplied = (" " + token)[1:] if kind == "copied" else token
        assert (supplied is token) == (kind == "positive")
        pause = storage._begin_local_pause()
        try:
            with pytest.raises(visual.VisualIdentityPublicationError):
                visual.cleanup_visual_identity_publication_candidate(
                    db, supplied, user_data_dir=profile
                )
            assert root.is_dir()
        finally:
            pause.resume()
        assert visual.cleanup_visual_identity_publication_candidate(
            db, supplied, user_data_dir=profile
        )
        assert not root.exists()
        assert bool(source.failures) == (kind == "copied")
        candidate.cancel()
        assert life.safe_point(profile) == (
            "incomplete" if kind == "copied" else "source-idle"
        )


@pytest.mark.parametrize("kind", ["forged", "old-row", "cancelled", "publishing"])
def test_canonical_restoration_refuses_changed_or_unstageable_generation(
    visual_environment, kind
):
    from dataclasses import replace
    from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen

    db, profile, _actor, _graph, candidate = visual_environment
    canonical = PersonasScreen._canonical_visual_identity_assets()
    before = len(candidate.assets)
    if kind == "forged":
        canonical = (replace(canonical[1], asset_id=77),)
    elif kind == "old-row":
        candidate.assets[0]["display_label"] = "forged old row"
    elif kind == "cancelled":
        candidate.cancel()
    if kind == "publishing":
        import os

        def restore_during_replace(*args, **kwargs):
            with pytest.raises(ValueError, match="candidate_publishing"):
                PersonasScreen._restore_candidate_reaction_rows(candidate, canonical)
            assert candidate._publishing and len(candidate.assets) == before
            os.replace(*args, **kwargs)

        visual.publish_visual_identity_candidate(
            db, candidate, user_data_dir=profile, atomic_replace=restore_during_replace
        )
    else:
        with pytest.raises(Exception):
            PersonasScreen._restore_candidate_reaction_rows(candidate, canonical)
        assert len(candidate.assets) == before


@pytest.mark.parametrize("pack_failure", [False, True])
def test_actual_config_seed_keeps_card_and_bounded_pack_lifetime(
    configured_db, monkeypatch, pack_failure
):
    from tldw_chatbook.Backup_Recovery import visual_identity_participants as life

    db, config = configured_db
    before = (
        db.get_connection()
        .execute("SELECT count(*) FROM character_cards")
        .fetchone()[0]
    )
    original = visual._load_samira_pack
    observed = []
    pauses = []

    def pack_after_card(*args, **kwargs):
        state = life.current()
        assert state.source is life.db_source(db)
        assert state.repository is db and state.native and state.active
        assert visual._find_builtin_samira_card(db) is not None
        observed.append(len(state.files))
        pauses.append(storage._begin_local_pause())
        if pack_failure:
            raise ValueError("synthetic pack failure after real card commit")
        return original(*args, **kwargs)

    monkeypatch.setattr(visual, "_load_samira_pack", pack_after_card)
    try:
        assert config.seed_builtin_content(db) is db
        assert observed and observed[0] > 30
        native = db._local.conn
        assert (
            native.execute("SELECT count(*) FROM character_cards").fetchone()[0]
            == before + 1
        )
        assert native.execute("SELECT count(*) FROM visual_identity_packs").fetchone()[
            0
        ] == (0 if pack_failure else 1)
        assert not any(s.source is life.db_source(db) for s in life._states)
    finally:
        for pause in pauses:
            pause.resume()


@pytest.mark.parametrize("pause_inside", [True, False])
def test_public_atomic_replace_callback_has_no_inherited_core_authority(
    visual_environment, pause_inside
):
    import os
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    db, profile, _actor, _graph, candidate = visual_environment
    pauses = []
    disposed = []

    class IgnoredResult:
        def __del__(self):
            try:
                db.add_character_card({"name": "ignored replacement result"})
            except RecoveryRequired:
                disposed.append("refused")
            else:
                disposed.append("ordinary")

    def replace_then_attempt_other_work(*args, **kwargs):
        os.replace(*args, **kwargs)
        if pause_inside:
            pauses.append(storage._begin_local_pause())
            with pytest.raises(RecoveryRequired):
                db.add_character_card({"name": "unrelated callback write"})
        else:
            db.add_character_card({"name": "unrelated callback write"})
        return IgnoredResult()

    try:
        visual.publish_visual_identity_candidate(
            db,
            candidate,
            user_data_dir=profile,
            atomic_replace=replace_then_attempt_other_work,
        )
        assert candidate._published
        assert disposed == (["refused"] if pause_inside else ["ordinary"])
    finally:
        for pause in pauses:
            pause.resume()


@pytest.mark.parametrize("method", ["activate_pack", "publish_version"])
@pytest.mark.parametrize("pause_inside", [True, False])
def test_public_repository_guard_has_no_inherited_core_authority(
    visual_environment, method, pause_inside
):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    db, _profile, actor, graph, _candidate = visual_environment
    assets = [
        {key: value for key, value in row.items() if key != "pack_version_id"}
        for row in graph["assets"]
    ]
    pauses = []

    def guard():
        if pause_inside:
            pauses.append(storage._begin_local_pause())
            with pytest.raises(RecoveryRequired):
                db.add_character_card({"name": "unrelated public guard write"})
            with pytest.raises(RecoveryRequired):
                VisualIdentityRepository(db).get_active_actor_pack("character", actor)
        else:
            db.add_character_card({"name": "unrelated public guard write"})
        return True

    try:
        repository = VisualIdentityRepository(db)
        args = (graph["pack"]["id"],) if method == "publish_version" else ()
        kw = (
            {}
            if method == "publish_version"
            else {
                "pack": {
                    "title": "Callback fixture",
                    "default_expression_key": "neutral",
                    "source_kind": "manual",
                    "source_context": {},
                }
            }
        )
        if method == "activate_pack":
            assets = [
                {key: value for key, value in row.items() if key != "pack_id"}
                for row in assets
            ]
        result = getattr(repository, method)(
            *args,
            **kw,
            manifest={"fixture": True},
            assets=assets,
            actor_kind="character",
            actor_id=actor,
            publication_guard=guard,
        )
        assert result["version"]["id"] != graph["version"]["id"]
    finally:
        for pause in pauses:
            pause.resume()


def test_public_repository_guard_truth_check_has_no_inherited_authority(
    visual_environment,
):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    db, _profile, actor, graph, _candidate = visual_environment
    pauses = []

    class GuardResult:
        def __bool__(self):
            with pytest.raises(RecoveryRequired):
                db.add_character_card({"name": "unrelated guard truth check"})
            return True

    def guard():
        pauses.append(storage._begin_local_pause())
        return GuardResult()

    assets = [
        {key: value for key, value in row.items() if key != "pack_version_id"}
        for row in graph["assets"]
    ]
    try:
        result = VisualIdentityRepository(db).publish_version(
            graph["pack"]["id"],
            manifest={"fixture": True},
            assets=assets,
            actor_kind="character",
            actor_id=actor,
            publication_guard=guard,
        )
        assert result["version"]["id"] != graph["version"]["id"]
    finally:
        for pause in pauses:
            pause.resume()


@pytest.mark.asyncio
@pytest.mark.parametrize("when", ["before", "after", "unrelated"])
async def test_ui_retains_publication_cleanup_when_borrower_close_also_raises(
    visual_environment, monkeypatch, when
):
    import os
    import subprocess
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen

    if when == "before" and os.environ.get("TASK10_VISUAL_DOUBLE_ERROR") != "before":
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                f"{__file__}::test_ui_retains_publication_cleanup_when_borrower_close_also_raises[before]",
                "-q",
                "-p",
                "no:cacheprovider",
                "--tb=short",
            ],
            cwd=Path(__file__).resolve().parents[2],
            env={
                **os.environ,
                "TASK10_VISUAL_DOUBLE_ERROR": "before",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            capture_output=True,
            text=True,
            timeout=25,
        )
        log = Path(
            "/private/tmp/chatbook-backup-execution-hgp11i7t/phase13-double-error-before-child.log"
        )
        log.write_text(result.stdout + result.stderr)
        assert result.returncode == 0, str(log)
        return
    db, profile, _actor, _graph, candidate = visual_environment
    screen = PersonasScreen(SimpleNamespace(chachanotes_db=db))
    original = db.close_connection

    def close_then_raise():
        if when != "before":
            original()
        raise OSError("synthetic source-thread close error")

    def replace_then_fail(*args, **kwargs):
        os.replace(*args, **kwargs)
        raise OSError("synthetic post-rename source failure")

    monkeypatch.setattr(db, "close_connection", close_then_raise)
    if when == "unrelated":
        candidate.assets[0]["display_label"] = "changed source"
    try:
        outcome = await screen._visual_identity_thread(
            visual.publish_visual_identity_candidate,
            db,
            candidate,
            user_data_dir=profile,
            atomic_replace=replace_then_fail,
            task_name="phase13-double-error",
        )
    finally:
        monkeypatch.setattr(db, "close_connection", original)
    assert outcome.error is not None
    token = getattr(outcome.error, "cleanup_candidate_relpath", None)
    assert outcome.error.result is None
    assert isinstance(outcome.error.source_error, visual.VisualIdentityPublicationError)
    if when == "unrelated":
        assert (
            token is None
            and outcome.error.source_error.cleanup_candidate_relpath is None
        )
    else:
        assert token is not None, (
            "native retirement error lost the issued cleanup association"
        )
        assert token is outcome.error.source_error.cleanup_candidate_relpath
        assert (profile / "visual_identities" / token).is_dir()
    assert outcome.error in screen._visual_identity_retained_errors
    assert screen.visual_identity_maintenance_state() == "incomplete"
