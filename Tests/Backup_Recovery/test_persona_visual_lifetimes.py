"""Actual Persona Visual file/core and native worker maintenance boundaries."""

import pytest

from Tests.Backup_Recovery.test_participant_lifetimes import local_root  # noqa: F401 -- fixture
from Tests.Backup_Recovery.test_admission import launch  # noqa: F401 -- fixture
from Tests.Backup_Recovery.test_compound_chat_source_lifetimes import configured_db  # noqa: F401 -- fixture
from Tests.Persona_Visual.test_persona_visual_publication import _snapshot, _png_bytes
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Persona_Visual.authoring_workspace import (
    create_persona_visual_authoring_workspace,
    stage_persona_visual_authoring_asset,
)
from tldw_chatbook.Persona_Visual.publication import publish_persona_visual
from tldw_chatbook.Persona_Visual.repository import PersonaVisualRepository


def test_pause_precedes_workspace_directory_mutation(tmp_path, local_root):
    profile = tmp_path / "profile"
    profile.mkdir(mode=0o700)
    workspace = create_persona_visual_authoring_workspace(profile)
    pause = storage._begin_local_pause()
    try:
        try:
            stage_persona_visual_authoring_asset(workspace, _png_bytes(), state="idle")
        except Exception:
            pass
        assert list((profile / workspace.relative_root / "assets").iterdir()) == []
    finally:
        pause.resume()


def test_publication_activation_finishes_after_file_publish_pause(
    configured_db, monkeypatch
):
    from tldw_chatbook.Persona_Visual import publication

    db, config = configured_db
    profile = config.get_user_data_dir()
    source = profile.parent / "source"
    source.mkdir(mode=0o700)
    snapshot = _snapshot(source)
    repository = PersonaVisualRepository(db)
    original_sync = publication._sync_directory
    pauses = []

    def sync_then_pause(fd):
        original_sync(fd)
        if tuple(profile.glob("persona_visual/packs/*/versions/[!.]*")) and not pauses:
            pauses.append(storage._begin_local_pause())

    monkeypatch.setattr(publication, "_sync_directory", sync_then_pause)
    result = None
    error = None
    try:
        try:
            result = publish_persona_visual(
                repository,
                snapshot,
                source_root=source,
                profile_root=profile,
                authority_guard=lambda: True,
            )
        except Exception as exc:
            error = exc
        assert pauses
        assert error is None, f"file published but activation newly refused: {error!r}"
        assert result is not None
        native = db._local.conn
        assert (
            native.execute("SELECT count(*) FROM persona_visual_bindings").fetchone()[0]
            == 1
        )
    finally:
        for pause in pauses:
            pause.resume()


@pytest.mark.asyncio
async def test_running_visual_thread_does_not_retire_on_executor_task_cancellation():
    import asyncio
    import threading
    from tldw_chatbook.UI.Screens.personas_screen import _drain_to_thread

    entered = threading.Event()
    released = threading.Event()
    finished = threading.Event()

    def native_job():
        entered.set()
        released.wait(5)
        finished.set()
        return "materialized"

    outer = asyncio.create_task(
        _drain_to_thread(native_job, task_name="phase12-native")
    )
    try:
        while not entered.is_set():
            await asyncio.sleep(0.01)
        child = next(t for t in asyncio.all_tasks() if t.get_name() == "phase12-native")
        child.cancel()
        await asyncio.sleep(0.03)
        assert not outer.done(), "cancelled Task was treated as native completion"
        released.set()
        outcome = await outer
        assert finished.is_set()
        assert outcome.value == "materialized"
        assert outcome.cancellation is not None
    finally:
        released.set()
        await outer


@pytest.mark.asyncio
async def test_queued_visual_executor_cancel_and_rejection_never_enter_source(
    monkeypatch,
):
    import asyncio
    from tldw_chatbook.UI.Screens.personas_screen import _drain_to_thread

    loop = asyncio.get_running_loop()
    queued = loop.create_future()
    callbacks = []
    effects = []
    monkeypatch.setattr(
        loop,
        "run_in_executor",
        lambda _, function: callbacks.append(function) or queued,
    )
    outer = asyncio.create_task(
        _drain_to_thread(lambda: effects.append("entered"), task_name="phase12-queued")
    )
    await asyncio.sleep(0)
    outer.cancel()
    outcome = await outer
    callbacks[0]()  # A late executor delivery cannot resurrect cancelled work.
    assert outcome.cancellation is not None
    assert effects == []

    def reject(*args):
        raise RuntimeError("synthetic executor rejection")

    monkeypatch.setattr(loop, "run_in_executor", reject)
    with pytest.raises(RuntimeError, match="synthetic executor rejection"):
        await _drain_to_thread(
            lambda: effects.append("entered"), task_name="phase12-rejected"
        )
    assert effects == []


def test_snapshot_pause_refuses_before_editor_selector(configured_db):
    from types import SimpleNamespace
    from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen

    db, _ = configured_db
    screen = PersonasScreen(SimpleNamespace(chachanotes_db=db))
    screen._edit_mode = "edit"
    observed = []
    screen.persona_handler.current_mode = lambda: observed.append("selector") or "local"
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(Exception):
            screen._persona_visual_snapshot()
        assert not observed
    finally:
        pause.resume()


def test_installed_workspace_generation_rejects_copies_stale_and_retarget(
    configured_db,
):
    from dataclasses import replace
    from tldw_chatbook.Backup_Recovery import persona_visual_participants as visual
    from tldw_chatbook.Persona_Visual.authoring_workspace import (
        cleanup_persona_visual_authoring_workspace,
    )

    _, config = configured_db
    profile = config.get_user_data_dir()
    workspace = create_persona_visual_authoring_workspace(profile)
    forged = replace(workspace)
    with pytest.raises(Exception):
        stage_persona_visual_authoring_asset(forged, _png_bytes(), state="idle")
    newer, _ = stage_persona_visual_authoring_asset(
        workspace, _png_bytes(), state="idle"
    )
    assert visual.issued(workspace) is None
    with pytest.raises(Exception):
        stage_persona_visual_authoring_asset(workspace, _png_bytes(), state="idle")
    moved = replace(newer, profile_root=profile.parent)
    with pytest.raises(Exception):
        stage_persona_visual_authoring_asset(moved, _png_bytes(), state="idle")
    assert cleanup_persona_visual_authoring_workspace(newer)


def test_installed_workspace_cleanup_preserves_foreign_asset(configured_db):
    from tldw_chatbook.Persona_Visual.authoring_workspace import (
        cleanup_persona_visual_authoring_workspace,
    )

    _, config = configured_db
    profile = config.get_user_data_dir()
    workspace, _ = stage_persona_visual_authoring_asset(
        create_persona_visual_authoring_workspace(profile), _png_bytes(), state="idle"
    )
    leaf = profile / workspace.relative_root / "assets" / workspace.asset_names[0]
    old = leaf.with_suffix(".old")
    leaf.rename(old)
    leaf.write_bytes(b"foreign")
    try:
        assert not cleanup_persona_visual_authoring_workspace(workspace)
        assert leaf.read_bytes() == b"foreign"
        assert old.exists()
    finally:
        leaf.unlink()
        old.rename(leaf)
        # Rename changes ctime. The original full asset snapshot remains stale;
        # restoring a pathname alone is deliberately not cleanup authorization.
        assert not cleanup_persona_visual_authoring_workspace(workspace)


def test_import_pause_refuses_before_external_archive_read(configured_db, monkeypatch):
    from tldw_chatbook.Persona_Visual import importer

    _, config = configured_db
    profile = config.get_user_data_dir()
    read = []
    monkeypatch.setattr(importer, "_pin_source", lambda path: read.append(path))
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(Exception):
            importer.import_persona_visual_pack(
                profile.parent / "external.zip",
                staging_root=profile / "persona_visual/imports",
                persona_id="p1",
                persona_revision=1,
                expected_identity=None,
            )
        assert not read
        assert not (profile / "persona_visual/imports").exists()
    finally:
        pause.resume()


def test_canonical_import_overlap_publishes_only_issued_assets(configured_db):
    from dataclasses import replace
    from Tests.Persona_Visual.test_persona_visual_importer import _write_archive
    from tldw_chatbook.Persona_Visual import importer
    from tldw_chatbook.Persona_Visual.authoring import (
        persona_visual_draft_publication_snapshot,
    )

    db, config = configured_db
    profile = config.get_user_data_dir()
    archive = _write_archive(profile.parent / "visual.tldw-persona-vpack")
    review = importer.import_persona_visual_pack(
        archive,
        staging_root=profile / "persona_visual/imports",
        persona_id="p1",
        persona_revision=1,
        expected_identity=None,
    )
    source = importer.persona_visual_import_source_root(
        review, staging_root=profile / "persona_visual/imports"
    )
    with pytest.raises(Exception):
        importer.persona_visual_import_source_root(
            replace(review), staging_root=profile / "persona_visual/imports"
        )
    result = publish_persona_visual(
        PersonaVisualRepository(db),
        persona_visual_draft_publication_snapshot(review.draft),
        source_root=source,
        profile_root=profile,
        authority_guard=lambda: True,
    )
    assert result.new_identity.persona_id == "p1"
    assert importer.cleanup_persona_visual_import_review(
        review, staging_root=profile / "persona_visual/imports"
    )


@pytest.mark.parametrize("when", ["before", "after", "partial"])
def test_native_uncertainty_retains_source_exclusion(
    configured_db, local_root, monkeypatch, when, launch
):
    import os
    import select
    import subprocess
    import sys
    import time
    from pathlib import Path
    from tldw_chatbook.Backup_Recovery import persona_visual_participants as visual

    if os.environ.get("TASK10_VISUAL_CHILD") != when:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                f"{__file__}::test_native_uncertainty_retains_source_exclusion[{when}]",
                "-q",
                "-p",
                "no:cacheprovider",
                "--tb=short",
            ],
            cwd=Path(__file__).resolve().parents[2],
            env={
                **os.environ,
                "TASK10_VISUAL_CHILD": when,
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            capture_output=True,
            text=True,
        )
        log = Path(
            f"/private/tmp/chatbook-backup-execution-hgp11i7t/phase12-native-{when}.log"
        )
        log.write_text(result.stdout + result.stderr)
        assert result.returncode == 0, str(log)
        return
    db, config = configured_db
    profile = config.get_user_data_dir()
    source = profile.parent / "source"
    source.mkdir(mode=0o700)
    snapshot = _snapshot(source)
    original = visual._close_native
    opened = visual._open_native
    injected = []

    def close_fault(fd):
        state = getattr(visual._local, "state", None)
        if state is not None and state.result is not None and not injected:
            injected.append(fd)
            if when == "after":
                original(fd)
            raise OSError("synthetic close uncertainty")
        original(fd)

    def open_fault(*args, **kwargs):
        state = getattr(visual._local, "state", None)
        if state is not None and len(state.descriptors) >= 2 and not injected:
            fd = opened(*args, **kwargs)
            injected.append(fd)
            raise OSError("synthetic partial native constructor")
        return opened(*args, **kwargs)

    monkeypatch.setattr(
        visual, "_open_native", open_fault
    ) if when == "partial" else monkeypatch.setattr(
        visual, "_close_native", close_fault
    )
    with pytest.raises(Exception) as error:
        publish_persona_visual(
            PersonaVisualRepository(db),
            snapshot,
            source_root=source,
            profile_root=profile,
            authority_guard=lambda: True,
        )
    assert injected
    assert any(state.uncertain for state in visual._states)
    if when != "partial":
        assert error.value.result.new_identity.persona_id == snapshot.persona_id
    db.close_connection()
    # Diagnostic child only: known quiescent startup, not application composition.
    storage._startups[(os.getpid(), str(local_root))].close()
    observer = launch(local_root / "admission", "maintenance", ("bootstrap.unbound",))
    pause = storage._begin_local_pause()
    try:
        assert not select.select([observer.stdout], [], [], 0.05)[0]
        assert not pause.drain(time.monotonic() + 0.02)
        participant = visual.source_for(profile)
        participant.close_admission()
        assert not participant.drain(time.monotonic() + 0.02)
    finally:
        pause.resume()
        observer.kill()
        observer.wait(timeout=5)


@pytest.mark.asyncio
async def test_dirty_inspection_keeps_actual_editor_draft(configured_db):
    from types import SimpleNamespace
    from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen

    db, _ = configured_db
    screen = PersonasScreen(SimpleNamespace(chachanotes_db=db))
    draft = SimpleNamespace(dirty=True)
    screen._persona_visual_authoring = draft
    assert screen.persona_visual_maintenance_state() == "needs-user-save/discard"
    assert screen._persona_visual_authoring is draft


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["after-publication", "before-visual"])
async def test_actual_ui_actor_guard_finishes_after_publication_pause(
    configured_db, monkeypatch, boundary
):
    from dataclasses import replace
    from types import SimpleNamespace
    import threading
    import weakref
    from tldw_chatbook.Backup_Recovery.chat_source_participants import (
        build_persona_service,
    )
    from tldw_chatbook.Persona_Visual import publication
    from tldw_chatbook.Persona_Visual.authoring import create_persona_visual_draft
    from tldw_chatbook.UI.Screens.personas_screen import (
        PersonasScreen,
        _PersonaVisualAuthorSnapshot,
        _PersonaVisualAuthoringState,
    )

    db, config = configured_db
    profile = config.get_user_data_dir()
    service = build_persona_service(db)
    persona = service.create_persona_profile({"name": "Actual local actor"})
    source = profile.parent / "source"
    source.mkdir(mode=0o700)
    request = replace(
        _snapshot(source, persona_revision=persona["version"]), persona_id=persona["id"]
    )
    screen = PersonasScreen(SimpleNamespace(chachanotes_db=db))
    snapshot = _PersonaVisualAuthorSnapshot(
        weakref.ref(screen),
        weakref.ref(screen),
        db,
        service,
        persona["id"],
        persona["version"],
        0,
        0,
    )
    screen._persona_visual_authoring = _PersonaVisualAuthoringState(
        snapshot,
        create_persona_visual_draft(
            persona_id=persona["id"],
            persona_revision=persona["version"],
            title="Visual",
        ),
        source,
    )
    pauses = []
    original_sync = publication._sync_directory
    original_close = db.close_connection

    def sync_then_pause(fd):
        original_sync(fd)
        if tuple(profile.glob("persona_visual/packs/*/versions/[!.]*")) and not pauses:
            pauses.append(storage._begin_local_pause())

    def close_then_resume():
        try:
            original_close()
        finally:
            if pauses and pauses[0].thread is threading.current_thread():
                pauses.pop().resume()

    monkeypatch.setattr(publication, "_sync_directory", sync_then_pause)
    if boundary == "before-visual":
        original_validate = publication._validate_snapshot

        def validate_then_pause(value):
            result = original_validate(value)
            pauses.append(storage._begin_local_pause())
            return result

        monkeypatch.setattr(publication, "_validate_snapshot", validate_then_pause)
    monkeypatch.setattr(db, "close_connection", close_then_resume)
    outcome = await screen._persona_visual_thread(
        publication.publish_persona_visual,
        PersonaVisualRepository(db),
        request,
        source_root=source,
        profile_root=profile,
        authority_guard=lambda: screen._persona_visual_authority_guard(snapshot),
        task_name="phase12-real-guard",
    )
    if boundary == "before-visual":
        assert outcome.error is not None
        assert not (profile / "persona_visual/packs").exists()
    else:
        assert outcome.error is None, (
            f"actual Persona guard was newly refused: {outcome.error!r}"
        )
        assert outcome.value.new_identity.persona_id == persona["id"]


@pytest.mark.asyncio
async def test_public_guard_and_actual_ui_publish_do_not_invert_locks(
    configured_db, monkeypatch
):
    import asyncio
    from contextlib import contextmanager
    from dataclasses import replace
    import os
    from pathlib import Path
    import subprocess
    import sys
    import threading
    from types import SimpleNamespace
    import weakref
    from tldw_chatbook.Backup_Recovery import chat_source_participants as chat
    from tldw_chatbook.Persona_Visual import publication
    from tldw_chatbook.Persona_Visual.authoring import create_persona_visual_draft
    from tldw_chatbook.UI.Screens.personas_screen import (
        PersonasScreen,
        _PersonaVisualAuthorSnapshot,
        _PersonaVisualAuthoringState,
    )

    if os.environ.get("TASK10_VISUAL_LOCK_CHILD") != "1":
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "pytest",
                f"{__file__}::test_public_guard_and_actual_ui_publish_do_not_invert_locks",
                "-q",
                "-s",
                "-p",
                "no:cacheprovider",
                "--tb=short",
            ],
            cwd=Path(__file__).resolve().parents[2],
            env={
                **os.environ,
                "TASK10_VISUAL_LOCK_CHILD": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        timed_out = False
        try:
            output, _ = process.communicate(timeout=12)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            output, _ = process.communicate(timeout=5)
        Path(
            "/private/tmp/chatbook-backup-execution-hgp11i7t/phase12-lock-child.log"
        ).write_text(output)
        assert not timed_out, (
            "actual public guard/Persona UI source lock inversion; see phase12-lock-child.log"
        )
        assert process.returncode == 0, output[-3000:]
        return
    db, config = configured_db
    profile = config.get_user_data_dir()
    actor = chat.build_persona_service(db)
    persona = actor.create_persona_profile({"name": "Concurrent actor"})
    source = profile.parent / "source"
    source.mkdir(mode=0o700)
    request = replace(
        _snapshot(source, persona_revision=persona["version"]), persona_id=persona["id"]
    )
    screen = PersonasScreen(SimpleNamespace(chachanotes_db=db))
    snapshot = _PersonaVisualAuthorSnapshot(
        weakref.ref(screen),
        weakref.ref(screen),
        db,
        actor,
        persona["id"],
        persona["version"],
        0,
        0,
    )
    screen._persona_visual_authoring = _PersonaVisualAuthoringState(
        snapshot,
        create_persona_visual_draft(
            persona_id=persona["id"],
            persona_revision=persona["version"],
            title="Visual",
        ),
        source,
    )
    public_guard = threading.Event()
    ui_actor = threading.Event()
    public_finished = threading.Event()
    original_operation = chat.operation

    @contextmanager
    def observed_operation(service):
        with original_operation(service) as token:
            if threading.current_thread().name != "phase12-public":
                ui_actor.set()
                print("UI_ACTOR_HELD", flush=True)
            yield token

    monkeypatch.setattr(chat, "operation", observed_operation)

    def guard():
        public_guard.set()
        print("PUBLIC_AT_GUARD", flush=True)
        assert ui_actor.wait(5)
        return actor.get_persona_profile(persona["id"])["version"] == persona["version"]

    def public_call():
        try:
            publication.publish_persona_visual(
                PersonaVisualRepository(db),
                request,
                source_root=source,
                profile_root=profile,
                authority_guard=guard,
            )
        except publication.PersonaVisualPublicationError:
            pass  # Optimistic stale claim is legitimate; deadlock is not.
        finally:
            db.close_connection()
            public_finished.set()

    thread = threading.Thread(target=public_call, name="phase12-public", daemon=True)
    thread.start()
    while not public_guard.is_set():
        await asyncio.sleep(0.01)
    outcome = await asyncio.wait_for(
        screen._persona_visual_thread(
            publication.publish_persona_visual,
            PersonaVisualRepository(db),
            request,
            source_root=source,
            profile_root=profile,
            authority_guard=lambda: screen._persona_visual_authority_guard(snapshot),
            task_name="phase12-competing-ui",
        ),
        5,
    )
    assert outcome.error is None
    assert await asyncio.to_thread(public_finished.wait, 3)


def test_publication_cleanup_requires_actual_issued_capability(configured_db):
    from tldw_chatbook.Backup_Recovery import persona_visual_participants as visual
    from tldw_chatbook.Persona_Visual import publication

    db, config = configured_db
    profile = config.get_user_data_dir()
    source = profile.parent / "source"
    source.mkdir(mode=0o700)
    repository = PersonaVisualRepository(db)
    calls = []

    def guard():
        calls.append(None)
        return len(calls) == 1

    with pytest.raises(publication.PersonaVisualPublicationError) as error:
        publication.publish_persona_visual(
            repository,
            _snapshot(source),
            source_root=source,
            profile_root=profile,
            authority_guard=guard,
        )
    capability = error.value.cleanup_candidate
    assert capability is not None
    copied = capability.encode().decode()
    assert copied == capability and copied is not capability
    with pytest.raises(publication.PersonaVisualPublicationError):
        publication.cleanup_persona_visual_publication_candidate(
            repository, copied, profile_root=profile
        )
    assert visual.safe_point(profile) == "incomplete"
    assert publication.cleanup_persona_visual_publication_candidate(
        repository, capability, profile_root=profile
    )
    assert visual.safe_point(profile) == "source-idle"


def test_concurrent_publications_keep_one_optimistic_winner_and_owned_cleanup(
    configured_db,
):
    import threading
    from tldw_chatbook.Backup_Recovery import persona_visual_participants as visual
    from tldw_chatbook.Persona_Visual import publication

    db, config = configured_db
    profile = config.get_user_data_dir()
    source = profile.parent / "source"
    source.mkdir(mode=0o700)
    request = _snapshot(source)
    # Existing shared ancestors avoid making their initial creation the race.
    (profile / "persona_visual/packs").mkdir(mode=0o700, parents=True)
    barrier = threading.Barrier(2)
    outcomes = []

    def publish():
        first = True

        def guard():
            nonlocal first
            if first:
                first = False
                barrier.wait(5)
            return True

        try:
            outcomes.append(
                publication.publish_persona_visual(
                    PersonaVisualRepository(db),
                    request,
                    source_root=source,
                    profile_root=profile,
                    authority_guard=guard,
                )
            )
        except publication.PersonaVisualPublicationError as error:
            outcomes.append(error)
        finally:
            db.close_connection()

    workers = [threading.Thread(target=publish) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(8)
        assert not worker.is_alive()
    winners = [
        value
        for value in outcomes
        if isinstance(value, publication.PersonaVisualPublicationResult)
    ]
    losers = [
        value
        for value in outcomes
        if isinstance(value, publication.PersonaVisualPublicationError)
    ]
    assert len(winners) == len(losers) == 1
    repository = PersonaVisualRepository(db)
    assert (
        repository.get_active_persona_pack(request.persona_id).identity
        == winners[0].new_identity
    )
    assert losers[0].cleanup_candidate
    assert visual.safe_point(profile) == "incomplete"
    assert publication.cleanup_persona_visual_publication_candidate(
        repository, losers[0].cleanup_candidate, profile_root=profile
    )
    assert visual.safe_point(profile) == "source-idle"


@pytest.mark.asyncio
async def test_actual_ui_native_retirement_precedes_independent_maintenance(
    configured_db, local_root, monkeypatch, launch
):
    import asyncio
    import os
    from pathlib import Path
    import select
    import subprocess
    import sys
    import threading
    from types import SimpleNamespace
    import weakref
    from Tests.Backup_Recovery.test_admission import line, release
    from tldw_chatbook.Backup_Recovery import persona_visual_participants as visual
    from tldw_chatbook.Persona_Visual import publication
    from tldw_chatbook.Persona_Visual.authoring import create_persona_visual_draft
    from tldw_chatbook.UI.Screens.personas_screen import (
        PersonasScreen,
        _PersonaVisualAuthorSnapshot,
        _PersonaVisualAuthoringState,
    )

    if os.environ.get("TASK10_VISUAL_RETIRE_CHILD") != "1":
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                f"{__file__}::test_actual_ui_native_retirement_precedes_independent_maintenance",
                "-q",
                "-p",
                "no:cacheprovider",
                "--tb=short",
            ],
            cwd=Path(__file__).resolve().parents[2],
            env={
                **os.environ,
                "TASK10_VISUAL_RETIRE_CHILD": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            capture_output=True,
            text=True,
            timeout=25,
        )
        log = Path(
            "/private/tmp/chatbook-backup-execution-hgp11i7t/phase12-retire-child.log"
        )
        log.write_text(result.stdout + result.stderr)
        assert result.returncode == 0, str(log)
        return
    db, config = configured_db
    profile = config.get_user_data_dir()
    source = profile.parent / "source"
    source.mkdir(mode=0o700)
    request = _snapshot(source)
    screen = PersonasScreen(SimpleNamespace(chachanotes_db=db))
    snapshot = _PersonaVisualAuthorSnapshot(
        weakref.ref(screen),
        weakref.ref(screen),
        db,
        object(),
        request.persona_id,
        1,
        0,
        0,
    )
    screen._persona_visual_authoring = _PersonaVisualAuthoringState(
        snapshot,
        create_persona_visual_draft(
            persona_id=request.persona_id, persona_revision=1, title="Visual"
        ),
        source,
    )
    entered, finish = threading.Event(), threading.Event()
    original = visual._close_native

    def close(fd):
        state = getattr(visual._local, "state", None)
        if state is not None and state.result is not None and not entered.is_set():
            entered.set()
            assert finish.wait(8)
        original(fd)

    monkeypatch.setattr(visual, "_close_native", close)
    db.close_connection()
    # Isolated diagnostic child: explicitly retire its known quiescent startup.
    storage._startups[(os.getpid(), str(local_root))].close()
    task = asyncio.create_task(
        screen._persona_visual_thread(
            publication.publish_persona_visual,
            PersonaVisualRepository(db),
            request,
            source_root=source,
            profile_root=profile,
            authority_guard=lambda: True,
            task_name="phase12-retire",
        )
    )
    try:
        while not entered.is_set():
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
        assert screen.persona_visual_maintenance_state() == "pending"
        finish.set()
        outcome = await task
        assert (
            outcome.error is None
            and outcome.completed
            and outcome.cancellation is not None
        )
        assert await asyncio.to_thread(line, observer) == "entered"
        assert not screen._persona_visual_pending
        # This diagnostic read is made under the independent maintainer's held
        # gate, after the real worker retired both native files and its DB.
        check = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sqlite3,sys,hashlib; from pathlib import Path; c=sqlite3.connect(sys.argv[1]); rows=c.execute('SELECT storage_relpath,sha256 FROM persona_visual_assets').fetchall(); assert rows; assert all(hashlib.sha256((Path(sys.argv[2])/p).read_bytes()).hexdigest()==h for p,h in rows); c.close()",
                str(db.db_path),
                str(profile),
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert check.returncode == 0, check.stderr
        release(observer)
    finally:
        finish.set()
        await task


@pytest.mark.asyncio
async def test_actual_ui_preserves_preexisting_source_thread_database_borrower(
    configured_db, monkeypatch
):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    import weakref
    from tldw_chatbook.UI.Screens.personas_screen import (
        PersonasScreen,
        _PersonaVisualAuthorSnapshot,
    )

    db, _ = configured_db
    screen = PersonasScreen(SimpleNamespace(chachanotes_db=db))
    snapshot = _PersonaVisualAuthorSnapshot(
        weakref.ref(screen), weakref.ref(screen), db, object(), "p1", 1, 0, 0
    )
    loop = asyncio.get_running_loop()
    original_executor = loop.run_in_executor
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(
            loop,
            "run_in_executor",
            lambda selected, function, *args: original_executor(
                executor if selected is None else selected, function, *args
            ),
        )
        previous = await original_executor(executor, db.get_connection)
        try:
            outcome = await screen._persona_visual_thread(
                screen._load_persona_visual_authoring_draft,
                snapshot,
                task_name="phase12-preexisting-borrower",
            )
            assert outcome.completed and outcome.error is None

            def observed():
                assert db._local.conn is previous
                return previous.execute("SELECT 1").fetchone()[0]

            assert await original_executor(executor, observed) == 1
        finally:
            await original_executor(executor, db.close_connection)


@pytest.mark.asyncio
async def test_actual_ui_actor_retarget_refuses_before_publication(configured_db):
    from types import SimpleNamespace
    import weakref
    from tldw_chatbook.Backup_Recovery import chat_source_participants as chat
    from tldw_chatbook.Persona_Visual import publication
    from tldw_chatbook.Persona_Visual.authoring import create_persona_visual_draft
    from tldw_chatbook.UI.Screens.personas_screen import (
        PersonasScreen,
        _PersonaVisualAuthorSnapshot,
        _PersonaVisualAuthoringState,
    )

    db, config = configured_db
    profile = config.get_user_data_dir()
    actor = chat.build_persona_service(db)
    source = profile.parent / "source"
    source.mkdir(mode=0o700)
    request = _snapshot(source)
    screen = PersonasScreen(SimpleNamespace(chachanotes_db=db))
    snapshot = _PersonaVisualAuthorSnapshot(
        weakref.ref(screen), weakref.ref(screen), db, actor, "p1", 1, 0, 0
    )
    screen._persona_visual_authoring = _PersonaVisualAuthoringState(
        snapshot,
        create_persona_visual_draft(
            persona_id="p1", persona_revision=1, title="Visual"
        ),
        source,
    )
    actor.db = object()
    outcome = await screen._persona_visual_thread(
        publication.publish_persona_visual,
        PersonaVisualRepository(db),
        request,
        source_root=source,
        profile_root=profile,
        authority_guard=lambda: True,
        task_name="phase12-retarget",
    )
    assert outcome.error is not None
    assert not (profile / "persona_visual/packs").exists()


def test_concurrent_import_cleanup_cannot_publish_stale_source(configured_db):
    import threading
    from Tests.Persona_Visual.test_persona_visual_importer import _write_archive
    from tldw_chatbook.Backup_Recovery import persona_visual_participants as visual
    from tldw_chatbook.Persona_Visual import importer, publication
    from tldw_chatbook.Persona_Visual.authoring import (
        persona_visual_draft_publication_snapshot,
    )

    db, config = configured_db
    profile = config.get_user_data_dir()
    staging = profile / "persona_visual/imports"
    archive = _write_archive(profile.parent / "source.tldw-persona-vpack")
    review = importer.import_persona_visual_pack(
        archive,
        staging_root=staging,
        persona_id="p1",
        persona_revision=1,
        expected_identity=None,
    )
    source = importer.persona_visual_import_source_root(review, staging_root=staging)
    ready, cleaned = threading.Event(), threading.Event()
    outcomes = []

    def cleanup():
        assert ready.wait(5)
        try:
            outcomes.append(
                importer.cleanup_persona_visual_import_review(
                    review, staging_root=staging
                )
            )
        finally:
            cleaned.set()

    worker = threading.Thread(target=cleanup)
    worker.start()

    def guard():
        ready.set()
        assert cleaned.wait(5)
        return True

    repository = PersonaVisualRepository(db)
    try:
        with pytest.raises(publication.PersonaVisualPublicationError) as error:
            publication.publish_persona_visual(
                repository,
                persona_visual_draft_publication_snapshot(review.draft),
                source_root=source,
                profile_root=profile,
                authority_guard=guard,
            )
        assert outcomes == [True]
        assert repository.get_active_persona_pack("p1") is None
        assert visual.safe_point(profile) == "incomplete"
        assert publication.cleanup_persona_visual_publication_candidate(
            repository, error.value.cleanup_candidate, profile_root=profile
        )
        assert visual.safe_point(profile) == "source-idle"
    finally:
        ready.set()
        worker.join(5)
        assert not worker.is_alive()


def test_existing_publication_destination_refused_before_candidate_effects(
    configured_db, monkeypatch
):
    from dataclasses import replace
    from types import SimpleNamespace
    from tldw_chatbook.Backup_Recovery import persona_visual_participants as visual
    from tldw_chatbook.Persona_Visual import publication

    db, config = configured_db
    profile = config.get_user_data_dir()
    source = profile.parent / "source"
    source.mkdir(mode=0o700)
    request = _snapshot(source)
    repository = PersonaVisualRepository(db)
    original = publication.publish_persona_visual(
        repository,
        request,
        source_root=source,
        profile_root=profile,
        authority_guard=lambda: True,
    )
    graph = repository.get_active_persona_pack(request.persona_id)
    key = repository._get_active_asset_storage_key(graph.identity, graph.assets[0])
    tokens = iter([key.split("/")[2], key.split("/")[4]])
    monkeypatch.setattr(publication, "uuid4", lambda: SimpleNamespace(hex=next(tokens)))
    effects = []
    mkdir = visual.mkdir
    monkeypatch.setattr(
        visual, "mkdir", lambda *a, **k: effects.append(a) or mkdir(*a, **k)
    )
    with pytest.raises(publication.PersonaVisualPublicationError):
        publication.publish_persona_visual(
            repository,
            replace(request, expected_identity=original.new_identity),
            source_root=source,
            profile_root=profile,
            authority_guard=lambda: True,
        )
    assert effects == []
    assert (
        repository.get_active_persona_pack(request.persona_id).identity
        == original.new_identity
    )


@pytest.mark.asyncio
async def test_ui_cleanup_reconciles_only_actual_matching_error(configured_db):
    from types import SimpleNamespace
    from tldw_chatbook.Persona_Visual import publication
    from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen

    db, config = configured_db
    profile = config.get_user_data_dir()
    source = profile.parent / "source"
    source.mkdir(mode=0o700)
    request = _snapshot(source)
    repository = PersonaVisualRepository(db)
    screen = PersonasScreen(SimpleNamespace(chachanotes_db=db))
    unrelated = object()
    screen._persona_visual_retained_cleanup.append(unrelated)
    outcome = await screen._persona_visual_thread(
        publication.publish_persona_visual,
        repository,
        request,
        source_root=source,
        profile_root=profile,
        authority_guard=lambda: False,
        task_name="phase12-ui-owned-error",
    )
    assert isinstance(outcome.error, publication.PersonaVisualPublicationError)
    assert outcome.error in screen._persona_visual_retained_cleanup
    cleanup = await screen._persona_visual_thread(
        publication.cleanup_persona_visual_publication_candidate,
        repository,
        outcome.error.cleanup_candidate,
        profile_root=profile,
        task_name="phase12-ui-owned-cleanup",
    )
    assert cleanup.error is None and cleanup.value is True
    assert screen._persona_visual_retained_cleanup == [unrelated]
