"""Private native ownership evidence for original bundle public callers."""

import pytest

from Tests.TTS.test_profile_repository_maintenance import _run_private_child


async def _native_close_child(root, phase, after):
    import os
    import time
    import types

    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_voice_bundle_service import _bundle, _service
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import voice_bundle_service as module

    service, repository, dependency = _service(root)
    payload = module.encode_clone_voice_bundle(_bundle())
    source = root / "source.bundle"
    source.write_bytes(payload)
    source.chmod(0o600)
    ordinary = storage.acquire_storage(service._root)
    hold = next(iter(storage._holds.values()))
    ordinary.close()
    await service._ensure_root()
    original = os.close
    calls = []
    selected_inode = source.stat().st_ino
    module.os = types.SimpleNamespace(**vars(os))

    target_fd = []

    def close(fd):
        if target_fd and fd == target_fd[0]:
            calls.append(fd)
            raise OSError("repeated ambiguous close attempt")
        info = os.fstat(fd)
        target = info.st_ino == selected_inode
        if phase == "operation":
            # Exact opened directory identity is selected by the original creator.
            target = fd == operation_fd[0] if operation_fd else False
        if target and not calls:
            target_fd.append(fd)
            calls.append(fd)
            if after:
                original(fd)
            raise OSError("injected exact native close")
        original(fd)

    operation_fd = []
    create = module._create_operation

    def created(*args, **kwargs):
        result = create(*args, **kwargs)
        operation_fd.append(result.operation_fd)
        return result

    module._create_operation = created
    module.os.close = close
    try:
        await service.inspect(source)
    except module.TTSVoiceBundleError:
        pass
    assert len(calls) == 1, "intended native resource was not reached exactly once"
    assert source.read_bytes() == payload
    pause = storage._begin_local_pause()
    drained = pause.drain(time.monotonic() + 0.05)
    observed = _probe(hold.authority.control_root, hold.names)
    assert observed == "blocked", (phase, after, drained, observed)
    assert not drained


@pytest.mark.parametrize("phase", ["source", "operation"])
@pytest.mark.parametrize("after", [False, True])
def test_uncertain_bundle_close_retains_native_exclusion(tmp_path, phase, after):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_voice_bundle_maintenance import _native_close_child
asyncio.run(_native_close_child(Path(sys.argv[1]), sys.argv[2], sys.argv[3] == 'True'))
""",
        phase,
        str(after),
    )


def _export_profile(repository):
    from types import SimpleNamespace
    from Tests.TTS.test_voice_bundle_service import (
        _bundle,
        command_to_profile,
        PROFILE_ID,
    )
    from tldw_chatbook.TTS.profile_reference_types import TTSCloneReference
    from tldw_chatbook.TTS.profile_types import ProfileStoreResult

    bundle = _bundle()
    profile = command_to_profile(
        SimpleNamespace(
            choice="create",
            source_profile_id=PROFILE_ID,
            copy_profile_id=None,
            copy_display_name=None,
            source_draft=bundle.profile.draft,
            canonical_reference=bundle.reference,
            recipe_requirement=bundle.recipe_requirement,
        )
    )
    reference = TTSCloneReference(
        summary=profile.reference,
        wav_bytes=bundle.reference.wav_bytes,
        reference_text=bundle.reference.reference_text,
        sha256=bundle.reference.sha256,
        recipe_requirement=bundle.recipe_requirement,
    )

    async def get_profile(_):
        return ProfileStoreResult(repository.generation, profile)

    async def get_reference(*args, **kwargs):
        return ProfileStoreResult(repository.generation, reference)

    repository.get_profile = get_profile
    repository.get_reference = get_reference
    return profile, bundle


async def _export_close_child(root, after):
    import os
    import time
    import types
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_voice_bundle_service import _service
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import voice_bundle_service as module

    service, repository, _ = _service(root)
    profile, bundle = _export_profile(repository)
    destination = root / "output.bundle"
    ordinary = storage.acquire_storage(service._root)
    hold = next(iter(storage._holds.values()))
    ordinary.close()
    original = os.close
    module.os = types.SimpleNamespace(**vars(os))
    selected = []
    calls = []

    def boundary(name):
        if name == "destination_pre_publish":
            temp = next(root.glob(".output.bundle.*.tmp"))
            selected.append(temp.stat().st_ino)

    def close(fd):
        if calls and fd == calls[0]:
            calls.append(fd)
            raise OSError("repeated publication close attempt")
        if (
            selected
            and os.fstat(fd).st_ino == selected[0]
            and destination.exists()
            and not calls
        ):
            # Final temp descriptor uses WRONLY; convergence readers use RDONLY.
            import fcntl

            if fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_WRONLY:
                calls.append(fd)
                if after:
                    original(fd)
                raise OSError("late native publication close")
        original(fd)

    module.os.close = close
    module._test_boundary = boundary
    await service.export(
        profile.profile_id,
        destination,
        expected_generation=repository.generation,
        expected_revision=profile.revision,
        acknowledged=True,
    )
    assert calls
    assert destination.read_bytes() == module.encode_clone_voice_bundle(bundle)
    assert not list(root.glob(".output.bundle.*.tmp"))
    pause = storage._begin_local_pause()
    drained = pause.drain(time.monotonic() + 0.05)
    assert _probe(hold.authority.control_root, hold.names) == "blocked", drained
    assert not drained
    with pytest.raises(module.TTSVoiceBundleError):
        await service.close()
    assert len(calls) == 1


@pytest.mark.parametrize("after", [False, True])
def test_successful_publication_retains_uncertain_native_close(tmp_path, after):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_voice_bundle_maintenance import _export_close_child
asyncio.run(_export_close_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(after),
    )


async def _app_source_child(root, selector):
    from Tests.UI.app_factory import _build_test_app
    from Tests.TTS.test_voice_bundle_service import _bundle
    from tldw_chatbook import config
    from tldw_chatbook.TTS import voice_bundle_service as module

    app = _build_test_app()
    service = await app._ensure_tts_voice_bundle_service()
    assert type(service) is module.TTSVoiceBundlePortabilityService
    assert service._repository is app._tts_profile_repository
    assert service._dependency_service is app.tts_service
    assert (
        service._artifact_lease_coordinator is app._audio_cpp_artifact_lease_coordinator
    )
    selected = service._root
    assert not selected.exists()
    source = root / "input.bundle"
    source.write_bytes(module.encode_clone_voice_bundle(_bundle()))
    source.chmod(0o600)
    if selector == "profile":
        config._CONFIG_CACHE["general"]["users_name"] = "changed-profile"
    else:
        original = config.get_user_data_dir
        config.get_user_data_dir = lambda: original()
    try:
        await service.inspect(source)
    except module.TTSVoiceBundleError:
        pass
    assert not selected.exists(), "stale actual app source created operation root"


@pytest.mark.parametrize("selector", ["profile", "callback"])
def test_actual_app_bundle_rejects_configured_source_drift(tmp_path, selector):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_voice_bundle_maintenance import _app_source_child
asyncio.run(_app_source_child(Path(sys.argv[1]), sys.argv[2]))
""",
        selector,
    )


async def _allocation_child(root, phase, outcome):
    import os
    import time
    import types
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_voice_bundle_service import _bundle, _service
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import voice_bundle_service as module

    service, repository, _ = _service(root)
    payload = module.encode_clone_voice_bundle(_bundle())
    source = root / "source.bundle"
    source.write_bytes(payload)
    source.chmod(0o600)
    ordinary = storage.acquire_storage(service._root)
    hold = next(iter(storage._holds.values()))
    ordinary.close()
    await service._ensure_root()
    original_open = module._open_bundle_descriptor
    calls = []
    original_os_open = os.open
    module.os = types.SimpleNamespace(**vars(os))
    target = {
        "source": source.name,
        "parent": str(source.parent),
        "member": "reference.wav",
        "operation": None,
        "temp": None,
    }[phase]

    def matches(args):
        value = str(args[0])
        return (
            value.startswith("operation-")
            if phase == "operation"
            else value.startswith(".output.bundle.")
            if phase == "temp"
            else value == target
        )

    def opening(*args, _outcome, **kwargs):
        if not calls and matches(args):
            calls.append(args[0])
            if outcome == "rejected":
                # The actual original OS primitive rejects the exact selected open.
                return (
                    original_open(
                        args[0],
                        args[1] | os.O_EXCL | os.O_CREAT,
                        0o600,
                        _outcome=_outcome,
                        **kwargs,
                    )
                    if phase in ("source", "parent", "operation")
                    else original_open(
                        args[0], os.O_RDONLY, _outcome=_outcome, **kwargs
                    )
                )
            if outcome == "returned":
                original_open(*args, _outcome=_outcome, **kwargs)
            else:
                original_os_open(*args, **kwargs)
            raise OSError("after actual native open")
        return original_open(*args, _outcome=_outcome, **kwargs)

    module._open_bundle_descriptor = opening
    if phase == "temp":
        profile, bundle = _export_profile(repository)
        try:
            await service.export(
                profile.profile_id,
                root / "output.bundle",
                expected_generation=repository.generation,
                expected_revision=profile.revision,
                acknowledged=True,
            )
        except module.TTSVoiceBundleError:
            pass
    else:
        try:
            await service.inspect(source)
        except module.TTSVoiceBundleError:
            pass
    assert len(calls) == 1
    assert source.read_bytes() == payload
    blocked = outcome == "unknown" or (phase == "operation" and outcome == "rejected")
    pause = storage._begin_local_pause()
    drained = pause.drain(time.monotonic() + 0.05)
    assert _probe(hold.authority.control_root, hold.names) == (
        "blocked" if blocked else "entered"
    ), (phase, outcome, drained)
    assert drained is not blocked


@pytest.mark.parametrize("phase", ["source", "parent", "member", "operation", "temp"])
@pytest.mark.parametrize("outcome", ["rejected", "returned", "unknown"])
def test_distinct_native_allocation_outcomes(tmp_path, phase, outcome):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_voice_bundle_maintenance import _allocation_child
asyncio.run(_allocation_child(Path(sys.argv[1]), sys.argv[2], sys.argv[3]))
""",
        phase,
        outcome,
    )


async def _session_child(root):
    import time
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_voice_bundle_service import _bundle, _service
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import voice_bundle_service as module

    service, repository, _ = _service(root)
    source = root / "input.bundle"
    payload = module.encode_clone_voice_bundle(_bundle())
    source.write_bytes(payload)
    source.chmod(0o600)
    ordinary = storage.acquire_storage(service._root)
    hold = next(iter(storage._holds.values()))
    ordinary.close()
    review = await service.inspect(source)
    session = service._sessions[review.handle]
    assert not list(service._root.iterdir())
    assert await service._maintenance_drain(time.monotonic() + 0.2)
    assert service._sessions[review.handle] is session
    assert _probe(hold.authority.control_root, hold.names) == "entered"
    with pytest.raises(module.TTSVoiceBundleError):
        await service.commit(
            review.handle, module.TTSVoiceBundleImportChoice("create", False)
        )
    assert service._sessions[review.handle] is session
    await service._maintenance_resume()
    result = await service.commit(
        review.handle, module.TTSVoiceBundleImportChoice("create", False)
    )
    assert result.status == "created"
    assert source.read_bytes() == payload
    assert review.handle not in service._sessions
    assert await service._maintenance_drain(time.monotonic() + 0.2)
    assert _probe(hold.authority.control_root, hold.names) == "entered"
    await service._maintenance_resume()
    await service.close()


def test_reversible_pause_preserves_review_and_allows_exact_commit(tmp_path):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_voice_bundle_maintenance import _session_child
asyncio.run(_session_child(Path(sys.argv[1])))
""",
    )


async def _create_substitution_child(root):
    import os
    import types
    from Tests.TTS.test_voice_bundle_service import _bundle, _service
    from tldw_chatbook.TTS import voice_bundle_service as module

    service, _, _ = _service(root)
    source = root / "input.bundle"
    source.write_bytes(module.encode_clone_voice_bundle(_bundle()))
    source.chmod(0o600)
    await service._ensure_root()
    module.os = types.SimpleNamespace(**vars(os))
    original = os.open
    selected = []

    def opening(name, *args, **kwargs):
        if str(name).startswith("operation-") and not selected:
            target = service._root / name
            moved = service._root / "original-retained"
            target.rename(moved)
            target.mkdir(mode=0o700)
            selected.extend((target, moved))
            raise OSError("creator rejected after namespace substitution")
        return original(name, *args, **kwargs)

    module.os.open = opening
    with pytest.raises(module.TTSVoiceBundleError):
        await service.inspect(source)
    assert selected
    assert selected[0].is_dir(), "failed operation creation deleted foreign replacement"
    assert selected[1].is_dir()


def test_failed_operation_creation_preserves_foreign_replacement(tmp_path):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_voice_bundle_maintenance import _create_substitution_child
asyncio.run(_create_substitution_child(Path(sys.argv[1])))
""",
    )


async def _enrolled_child(root, allowed, action):
    import os
    from pathlib import Path
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_voice_bundle_service import _bundle, _service
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        bind_profile,
    )
    from tldw_chatbook.TTS import voice_bundle_service as module

    selector = Path(os.environ["TLDW_CONFIG_PATH"])
    service, repository, _ = _service(root / "profile")
    external = root / "external"
    external.mkdir(mode=0o700)
    source = external / "input.bundle"
    payload = module.encode_clone_voice_bundle(_bundle())
    source.write_bytes(payload)
    source.chmod(0o644)
    authority = admission_authority(bootstrap.default_bootstrap_root())
    authority.register("bundle.profile", (root / "profile", selector))
    names = ["bundle.profile"]
    if allowed:
        authority.register("bundle.external", (external,))
        names.append("bundle.external")
    bind_profile(
        bootstrap.default_bootstrap_root(),
        selector,
        tuple(names),
        authority.control_root,
    )
    review = None
    observed = []

    def boundary(name):
        if allowed and name in ("source_post_inspection", "destination_pre_publish"):
            observed.append(_probe(authority.control_root, ("bundle.external",)))

    module._test_boundary = boundary
    try:
        if action == "export":
            profile, bundle = _export_profile(repository)
            await service.export(
                profile.profile_id,
                external / "output.bundle",
                expected_generation=repository.generation,
                expected_revision=profile.revision,
                acknowledged=True,
            )
            review = True
        else:
            review = await service.inspect(source)
    except module.TTSVoiceBundleError:
        pass
    if allowed:
        assert review is not None
        assert observed == ["blocked"]
        if action == "export":
            assert (
                external / "output.bundle"
            ).read_bytes() == module.encode_clone_voice_bundle(bundle)
        assert _probe(authority.control_root, tuple(names)) == "entered"
    else:
        assert review is None, "runtime namespace authorized unenrolled external source"
        assert source.stat().st_mode & 0o777 == 0o644, "unenrolled source was hardened"
        assert not (external / "output.bundle").exists()
        assert not list(external.glob(".*.tmp"))
    assert source.read_bytes() == payload


@pytest.mark.parametrize("allowed", [False, True])
@pytest.mark.parametrize("action", ["inspect", "export"])
def test_selected_external_path_has_ordinary_declared_admission(
    tmp_path, allowed, action
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_voice_bundle_maintenance import _enrolled_child
asyncio.run(_enrolled_child(Path(sys.argv[1]), sys.argv[2] == 'True', sys.argv[3]))
""",
        str(allowed),
        action,
    )


async def _stream_child(root, mode, after):
    import os
    import time
    import types
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_voice_bundle_service import _bundle, _service
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import voice_bundle_service as module

    service, _, _ = _service(root)
    source = root / "input.bundle"
    payload = module.encode_clone_voice_bundle(_bundle())
    source.write_bytes(payload)
    source.chmod(0o600)
    lease = storage.acquire_storage(service._root)
    hold = next(iter(storage._holds.values()))
    lease.close()
    reached = []
    streams = []
    original = module._open_bundle_stream
    module.os = types.SimpleNamespace(**vars(os))
    native_fdopen = os.fdopen

    class Stream:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def close(self):
            reached.append("close")
            if after:
                self.stream.close()
            raise OSError("native buffered stream close")

    if mode == "close":

        def fdopen(fd, *args, **kwargs):
            stream = native_fdopen(fd, *args, **kwargs)
            if not streams:
                stream = Stream(stream)
            streams.append(stream)
            return stream

        module.os.fdopen = fdopen
    elif mode == "returned":

        def opening(fd, **kwargs):
            stream = original(fd, **kwargs)
            if not reached:
                reached.append(fd)
                streams.append(stream)
                raise OSError("returned stream outer failure")
            return stream

        module._open_bundle_stream = opening
    else:

        def fdopen(fd, *args, **kwargs):
            stream = native_fdopen(fd, *args, **kwargs)
            if not reached:
                reached.append(fd)
                streams.append(stream)
                raise OSError("unreturned native stream")
            return stream

        module.os.fdopen = fdopen
    with pytest.raises(module.TTSVoiceBundleError):
        await service.inspect(source)
    assert len(reached) == 1
    assert source.read_bytes() == payload
    blocked = mode != "returned"
    assert await service._maintenance_drain(time.monotonic() + 0.05) is not blocked
    assert _probe(hold.authority.control_root, hold.names) == (
        "blocked" if blocked else "entered"
    )
    if blocked:
        with pytest.raises(module.TTSVoiceBundleError):
            await service.close()
        assert len(reached) == 1
    else:
        assert streams[0].closed
        await service.close()


@pytest.mark.parametrize(
    "mode,after",
    [("close", False), ("close", True), ("returned", False), ("unknown", False)],
)
def test_native_stream_outcome_has_one_descriptor_closer(tmp_path, mode, after):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_voice_bundle_maintenance import _stream_child
asyncio.run(_stream_child(Path(sys.argv[1]), sys.argv[2], sys.argv[3] == 'True'))
""",
        mode,
        str(after),
    )


async def _final_release_child(root, phase, after, control):
    import asyncio
    import time
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_voice_bundle_service import _bundle, _service
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import voice_bundle_service as module

    service, repository, _ = _service(root)
    source = root / "input.bundle"
    source.write_bytes(module.encode_clone_voice_bundle(_bundle()))
    source.chmod(0o600)
    calls, holds = [], []
    if phase != "publish":
        review = await service.inspect(source)
    function = (
        module._publish_sync if phase == "publish" else module._fingerprint_source_sync
    )

    def finish_then_fault(*args, **kwargs):
        if phase == "fingerprint_body":
            source.write_bytes(b"changed selected source")
        result = function(*args, **kwargs)
        native = kwargs["_native"]
        lease = native.leases[-1]
        holds.append(storage._holds[lease._key])
        close = lease.close

        def closing():
            calls.append(True)
            if after:
                close()
            raise (
                asyncio.CancelledError("private close control")
                if control
                else OSError("private lease failure")
            )

        lease.close = closing
        return result

    if phase == "publish":
        module._publish_sync = finish_then_fault
        profile, bundle = _export_profile(repository)
        destination = root / "output.bundle"
        await service.export(
            profile.profile_id,
            destination,
            expected_generation=repository.generation,
            expected_revision=profile.revision,
            acknowledged=True,
        )
        assert destination.read_bytes() == module.encode_clone_voice_bundle(bundle)
    else:
        module._fingerprint_source_sync = finish_then_fault
        with pytest.raises(
            asyncio.CancelledError
            if control and phase != "fingerprint_body"
            else module.TTSVoiceBundleError,
            match="source_changed" if phase == "fingerprint_body" else None,
        ):
            await service.commit(
                review.handle, module.TTSVoiceBundleImportChoice("create", False)
            )
        assert repository.commits == []
    assert calls == [True]
    assert _probe(holds[0].authority.control_root, holds[0].names) == "blocked"
    assert not await service._maintenance_drain(time.monotonic() + 0.05)
    with pytest.raises(module.TTSVoiceBundleError):
        await service.close()
    assert calls == [True]


@pytest.mark.parametrize("phase", ["fingerprint", "publish", "fingerprint_body"])
@pytest.mark.parametrize("after", [False, True])
@pytest.mark.parametrize("control", [False, True])
def test_final_native_lease_failure_distinguishes_publication(
    tmp_path, phase, after, control
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_voice_bundle_maintenance import _final_release_child
asyncio.run(_final_release_child(Path(sys.argv[1]), sys.argv[2], sys.argv[3] == 'True', sys.argv[4] == 'True'))
""",
        phase,
        str(after),
        str(control),
    )


async def _lifetime_child(root, phase, cancel):
    import asyncio
    import concurrent.futures
    import threading
    import time
    from Tests.TTS.test_voice_bundle_service import _bundle, _service
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import voice_bundle_service as module

    service, repository, dependency = _service(root)
    source = root / "input.bundle"
    payload = module.encode_clone_voice_bundle(_bundle())
    source.write_bytes(payload)
    source.chmod(0o600)
    entered, release = threading.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    blocker = None
    if phase == "queued":
        loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1))

        def occupy():
            entered.set()
            assert release.wait(5)

        blocker = asyncio.create_task(asyncio.to_thread(occupy))
        while not entered.is_set():
            await asyncio.sleep(0.001)
    elif phase == "native":

        def boundary(name):
            if name == "source_post_inspection":
                entered.set()
                assert release.wait(5)

        module._test_boundary = boundary
    else:
        original = dependency.audio_cpp_guided_dependency_snapshot

        async def pending(requirement):
            entered.set()
            while not release.is_set():
                await asyncio.sleep(0.001)
            return await original(requirement)

        dependency.audio_cpp_guided_dependency_snapshot = pending
    task = asyncio.create_task(service.inspect(source))
    if phase == "queued":
        while not service._workers:
            await asyncio.sleep(0.001)
    else:
        while not entered.is_set():
            await asyncio.sleep(0.001)
    if cancel:
        task.cancel("first")
        task.cancel("second")
    assert not await service._maintenance_drain(time.monotonic() + 0.03)
    assert not task.done()
    pause = storage._begin_local_pause() if phase == "queued" else None
    release.set()
    if blocker is not None:
        await blocker
    if cancel:
        expected_failure = (
            (asyncio.CancelledError, module.TTSVoiceBundleError)
            if phase == "queued"
            else asyncio.CancelledError
        )
        with pytest.raises(expected_failure):
            await task
    elif phase == "queued":
        with pytest.raises(module.TTSVoiceBundleError):
            await task
    else:
        review = await task
        assert service._sessions[review.handle].source == source
    assert await service._maintenance_drain(time.monotonic() + 0.2)
    if phase == "queued":
        assert not service._root.exists()
        pause.resume()
    assert source.read_bytes() == payload
    await service._maintenance_resume()
    review = await service.inspect(source)
    await service.invalidate(review.handle)
    await service.close()


@pytest.mark.parametrize(
    "phase,cancel",
    [
        ("queued", False),
        ("queued", True),
        ("native", False),
        ("native", True),
        ("review", False),
    ],
)
def test_pause_waits_actual_queued_native_and_late_review_lifetimes(
    tmp_path, phase, cancel
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_voice_bundle_maintenance import _lifetime_child
asyncio.run(_lifetime_child(Path(sys.argv[1]), sys.argv[2], sys.argv[3] == 'True'))
""",
        phase,
        str(cancel),
    )


async def _traversal_child(root, after):
    import os
    import time
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_voice_bundle_service import _bundle, _service
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import voice_bundle_service as module
    from tldw_chatbook.Utils import private_paths

    service, _, _ = _service(root)
    service._root.mkdir(mode=0o700)
    selected = service._root.stat()
    source = root / "input.bundle"
    source.write_bytes(module.encode_clone_voice_bundle(_bundle()))
    source.chmod(0o600)
    lease = storage.acquire_storage(service._root)
    hold = next(iter(storage._holds.values()))
    lease.close()
    close = private_paths._native_close
    calls = []

    def closing(fd):
        info = os.fstat(fd)
        if (info.st_dev, info.st_ino) == (selected.st_dev, selected.st_ino):
            calls.append(fd)
            if after:
                close(fd)
            raise OSError("actual selected traversal descriptor")
        close(fd)

    private_paths._native_close = closing
    with pytest.raises(module.TTSVoiceBundleError):
        await service.inspect(source)
    assert len(calls) == 1
    assert not await service._maintenance_drain(time.monotonic() + 0.05)
    assert _probe(hold.authority.control_root, hold.names) == "blocked"
    assert not list(service._root.iterdir())


@pytest.mark.parametrize("after", [False, True])
def test_prepare_root_tracks_exact_private_traversal_native_close(tmp_path, after):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_voice_bundle_maintenance import _traversal_child
asyncio.run(_traversal_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(after),
    )


async def _close_edge_child(root, edge, after):
    import os
    import time
    import types
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_voice_bundle_service import _bundle, _service
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import voice_bundle_service as module

    service, repository, _ = _service(root)
    source = root / "input.bundle"
    source.write_bytes(module.encode_clone_voice_bundle(_bundle()))
    source.chmod(0o600)
    lease = storage.acquire_storage(service._root)
    hold = next(iter(storage._holds.values()))
    lease.close()
    if edge == "fingerprint":
        review = await service.inspect(source)
        original_fingerprint = module._fingerprint_source_sync
        armed = []

        def fingerprint(*args, **kwargs):
            armed.append(True)
            return original_fingerprint(*args, **kwargs)

        module._fingerprint_source_sync = fingerprint
    else:
        armed = [True]
    original_open = module._open_bundle_descriptor
    targets, calls = [], []

    def opening(*args, **kwargs):
        fd = original_open(*args, **kwargs)
        name = str(args[0])
        match = (
            edge == "root"
            and name == str(service._root)
            or edge in ("parent", "export_parent")
            and name == str(source.parent)
            or edge == "copy"
            and name == "source.bundle"
            or edge == "member"
            and name == "reference.txt"
            or edge == "fingerprint"
            and name == source.name
            or edge == "published_read"
            and name == "output.bundle"
        )
        if match and armed and not targets:
            targets.append(fd)
        return fd

    module._open_bundle_descriptor = opening
    module.os = types.SimpleNamespace(**vars(os))
    original_close = os.close

    def close(fd):
        if targets and fd == targets[0]:
            calls.append(fd)
            if after:
                original_close(fd)
            raise OSError("exact native phase descriptor")
        original_close(fd)

    module.os.close = close
    if edge in ("published_read", "export_parent"):
        profile, bundle = _export_profile(repository)
        try:
            await service.export(
                profile.profile_id,
                root / "output.bundle",
                expected_generation=repository.generation,
                expected_revision=profile.revision,
                acknowledged=True,
            )
        except module.TTSVoiceBundleError:
            pass
        assert (
            root / "output.bundle"
        ).read_bytes() == module.encode_clone_voice_bundle(bundle)
    else:
        with pytest.raises(module.TTSVoiceBundleError):
            if edge == "fingerprint":
                await service.commit(
                    review.handle, module.TTSVoiceBundleImportChoice("create", False)
                )
            else:
                await service.inspect(source)
    assert calls == targets and len(calls) == 1
    assert not await service._maintenance_drain(time.monotonic() + 0.05)
    assert _probe(hold.authority.control_root, hold.names) == "blocked"
    with pytest.raises(module.TTSVoiceBundleError):
        await service.close()
    assert calls == targets


@pytest.mark.parametrize(
    "edge",
    [
        "root",
        "parent",
        "copy",
        "member",
        "fingerprint",
        "published_read",
        "export_parent",
    ],
)
@pytest.mark.parametrize("after", [False, True])
def test_each_concrete_descriptor_close_retains_native_uncertainty(
    tmp_path, edge, after
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_voice_bundle_maintenance import _close_edge_child
asyncio.run(_close_edge_child(Path(sys.argv[1]), sys.argv[2], sys.argv[3] == 'True'))
""",
        edge,
        str(after),
    )


async def _source_matrix_child(root, mode):
    import time
    from Tests.UI.app_factory import _build_test_app
    from Tests.TTS.test_voice_bundle_service import _bundle
    from tldw_chatbook import config
    from tldw_chatbook.TTS import voice_bundle_service as module
    from tldw_chatbook.TTS.profile_source import _ConfiguredBundleSource

    app = _build_test_app()
    if mode == "custom":
        import tldw_chatbook.app as app_module

        calls = []

        def custom():
            calls.append(True)
            return root / "custom"

        app_module.get_user_data_dir = custom
    service = await app._ensure_tts_voice_bundle_service()
    assert type(service) is module.TTSVoiceBundlePortabilityService
    selected = service._root
    assert not selected.exists()
    source = root / "input.bundle"
    payload = module.encode_clone_voice_bundle(_bundle())
    source.write_bytes(payload)
    source.chmod(0o600)
    if mode == "custom":
        assert service._configured_source is None
        assert calls == [True]
        await service.inspect(source)
        assert selected.exists()
        await service.close()
        return
    assert type(service._configured_source) is _ConfiguredBundleSource
    if mode == "normal":
        first = await service.inspect(source)
        assert service is await app._ensure_tts_voice_bundle_service()
        session = service._sessions[first.handle]
        assert await service._maintenance_drain(time.monotonic() + 0.2)
        assert service._sessions[first.handle] is session

        class PathSelector:
            def __fspath__(self):
                raise AssertionError("paused path callback ran")

        with pytest.raises(module.TTSVoiceBundleError):
            await service.inspect(PathSelector())
        await service._maintenance_resume()
        imported = await service.commit(
            first.handle, module.TTSVoiceBundleImportChoice("create", True)
        )
        assert imported.status == "created"
        stored = await service._repository.get_reference(
            imported.profile.profile_id,
            expected_generation=service._repository.generation,
            expected_revision=imported.profile.revision,
        )
        assert stored.value.wav_bytes == _bundle().reference.wav_bytes
        assert stored.value.reference_text == _bundle().reference.reference_text
        exported = root / "actual-app-output.bundle"
        await service.export(
            imported.profile.profile_id,
            exported,
            expected_generation=service._repository.generation,
            expected_revision=imported.profile.revision,
            acknowledged=True,
        )
        assert exported.read_bytes() == payload
    elif mode == "cleanup_drift":
        reached = []

        def boundary(name):
            if name == "source_post_inspection":
                reached.append(name)
                config._CONFIG_CACHE["general"]["users_name"] = "changed-profile"

        module._test_boundary = boundary
        with pytest.raises(module.TTSVoiceBundleError):
            await service.inspect(source)
        assert reached
        assert not list(selected.iterdir())
        assert not service._native_operations
        assert not service._sessions
        with pytest.raises(module.TTSVoiceBundleError):
            await service._maintenance_resume()
    else:
        if mode == "repository":
            app._tts_profile_repository = object()
        elif mode == "dependency":
            app.tts_service = object()
        elif mode == "coordinator":
            app._audio_cpp_artifact_lease_coordinator = object()
        elif mode == "fence":
            original = service._profile_mutation_fence
            service._profile_mutation_fence = lambda: original()
        elif mode == "receiver":
            app._tts_voice_bundle_service = object()
        with pytest.raises(module.TTSVoiceBundleError):
            await service.inspect(source)
        assert not selected.exists()
    assert source.read_bytes() == payload
    await service.close()
    # Real profile SQLite was opened by the original factory. Close original owner.
    await service._repository.close()
    await service._dependency_service.close()
    await service._dependency_service.wait_closed()


@pytest.mark.parametrize(
    "mode",
    [
        "normal",
        "repository",
        "dependency",
        "coordinator",
        "fence",
        "receiver",
        "cleanup_drift",
        "custom",
    ],
)
def test_exact_app_source_relationship_and_cleanup(tmp_path, mode):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_voice_bundle_maintenance import _source_matrix_child
asyncio.run(_source_matrix_child(Path(sys.argv[1]), sys.argv[2]))
""",
        mode,
    )


async def _membership_child(root, action):
    from dataclasses import replace
    from Tests.TTS.test_voice_bundle_service import _bundle, _service
    from tldw_chatbook.TTS import voice_bundle_service as module

    service, repository, _ = _service(root)
    source = root / "input.bundle"
    payload = module.encode_clone_voice_bundle(_bundle())
    source.write_bytes(payload)
    source.chmod(0o600)
    foreign = root / "foreign"
    foreign.write_bytes(b"foreign bytes")
    reached = []

    def boundary(name):
        if name not in ("source_post_inspection", "destination_pre_publish"):
            return
        native = next(n for n in service._native_operations if n.thread is not None)
        parent_fd = (
            native.operation.root_fd
            if native.operation is not None
            else next(fd for fd, p in native.parents.items() if p == root)
        )
        fake_name = (
            "operation-" + "f" * 32
            if action == "inspect"
            else ".output.bundle." + "f" * 32 + ".tmp"
        )
        with pytest.raises(module.TTSVoiceBundleError):
            module._bundle_select(native, parent_fd, fake_name)
        with pytest.raises(module.TTSVoiceBundleError):
            native.select(parent_fd, fake_name)
        with pytest.raises(module.TTSVoiceBundleError):
            module._bundle_open(native, foreign, module._SOURCE_FLAGS)
        if native.operation is not None:
            with pytest.raises(module.TTSVoiceBundleError):
                module._cleanup_operation(replace(native.operation), _native=native)
            with pytest.raises(module.TTSVoiceBundleError):
                module._create_operation_file(
                    replace(native.operation), "reference.txt", _native=native
                )
        reached.append(True)

    module._test_boundary = boundary
    if action == "inspect":
        await service.inspect(source)
    else:
        profile, bundle = _export_profile(repository)
        await service.export(
            profile.profile_id,
            root / "output.bundle",
            expected_generation=repository.generation,
            expected_revision=profile.revision,
            acknowledged=True,
        )
        assert (
            root / "output.bundle"
        ).read_bytes() == module.encode_clone_voice_bundle(bundle)
    assert reached == [True]
    assert foreign.read_bytes() == b"foreign bytes"
    assert source.read_bytes() == payload
    assert not service._native_operations
    await service.close()


@pytest.mark.parametrize("action", ["inspect", "export"])
def test_callbacks_cannot_select_regex_sibling_or_replace_native_operation(
    tmp_path, action
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_voice_bundle_maintenance import _membership_child
asyncio.run(_membership_child(Path(sys.argv[1]), sys.argv[2]))
""",
        action,
    )


def _flock_probe(path):
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print('entered')
except BlockingIOError:
    print('blocked')
finally:
    os.close(fd)
""",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


async def _flock_child(root):
    import asyncio
    import threading
    from Tests.TTS.test_voice_bundle_service import _bundle, _service
    from tldw_chatbook.TTS import voice_bundle_service as module

    service, _, _ = _service(root)
    source = root / "input.bundle"
    payload = module.encode_clone_voice_bundle(_bundle())
    source.write_bytes(payload)
    source.chmod(0o600)
    entered, release = threading.Event(), threading.Event()

    def boundary(name):
        if name == "source_post_inspection":
            entered.set()
            assert release.wait(5)

    module._test_boundary = boundary
    task = asyncio.create_task(service.inspect(source))
    while not entered.is_set():
        await asyncio.sleep(0.001)
    assert _flock_probe(service._root) == "blocked"
    operation = next(iter(service._native_operations)).operation
    assert (
        service._root / operation.operation_leaf / "source.bundle"
    ).read_bytes() == payload
    release.set()
    await task
    assert _flock_probe(service._root) == "entered"
    assert not list(service._root.iterdir())
    assert source.read_bytes() == payload
    await service.close()


def test_original_operation_native_root_flock_lifetime(tmp_path):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_voice_bundle_maintenance import _flock_child
asyncio.run(_flock_child(Path(sys.argv[1])))
""",
    )


async def _custom_fence_child(root, kind):
    from Tests.UI.app_factory import _build_test_app
    from Tests.TTS.test_voice_bundle_service import _bundle
    from tldw_chatbook.TTS import voice_bundle_service as module

    app = _build_test_app()
    profile_service = await app._ensure_tts_profile_service()
    fence = profile_service.consumer_mutation_fence
    calls = []
    if kind == "proxy":

        class CustomProfile:
            def __getattr__(self, name):
                return getattr(profile_service, name)

            @property
            def consumer_mutation_fence(self):
                calls.append("access")
                return fence

        app._tts_profile_service = CustomProfile()
    else:

        def custom_fence():
            calls.append("call")
            return fence()

        profile_service.consumer_mutation_fence = custom_fence
    service = await app._ensure_tts_voice_bundle_service()
    assert type(service) is module.TTSVoiceBundlePortabilityService
    assert service._configured_source is None, (
        "custom fence/profile service gained configured authority"
    )
    assert calls == (["access"] if kind == "proxy" else [])
    source = root / "input.bundle"
    source.write_bytes(module.encode_clone_voice_bundle(_bundle()))
    source.chmod(0o600)
    await service.inspect(source)
    assert calls == (["access"] if kind == "proxy" else [])
    await service.close()
    await service._repository.close()
    await service._dependency_service.close()
    await service._dependency_service.wait_closed()


@pytest.mark.parametrize("kind", ["proxy", "shadow"])
def test_prebinding_custom_profile_fence_remains_ordinary_without_accessor_replay(
    tmp_path, kind
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_voice_bundle_maintenance import _custom_fence_child
asyncio.run(_custom_fence_child(Path(sys.argv[1]), sys.argv[2]))
""",
        kind,
    )


async def _original_app_compatibility_child(root):
    from Tests.TTS.test_tts_app_ownership import (
        test_voice_bundle_service_is_lazy_singleton_and_closes_before_repository,
    )

    with pytest.MonkeyPatch.context() as patch:
        await test_voice_bundle_service_is_lazy_singleton_and_closes_before_repository(
            root, patch
        )


def test_original_app_bundle_factory_and_terminal_order_with_private_source(tmp_path):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_voice_bundle_maintenance import _original_app_compatibility_child
asyncio.run(_original_app_compatibility_child(Path(sys.argv[1])))
""",
    )
