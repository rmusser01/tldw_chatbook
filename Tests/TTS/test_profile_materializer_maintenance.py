"""Private process and native filesystem evidence for clone owners."""

import pytest

from Tests.TTS.test_profile_repository_maintenance import _run_private_child


async def _live_child(root, mode, after):
    import hashlib
    import io
    import os
    import time
    import types
    import wave
    from dataclasses import replace

    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_profile_reference_materialization import _reference
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import profile_reference_materialization as module

    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x01\x00" * 240)
    reference = replace(
        _reference(),
        wav_bytes=output.getvalue(),
        sha256=hashlib.sha256(output.getvalue()).hexdigest(),
        summary=replace(
            _reference().summary, byte_length=len(output.getvalue()), duration_ms=10
        ),
    )
    selected = root / "runtime"
    # Observe the native namespace independently of a future materializer API.
    ordinary = storage.acquire_storage(selected)
    hold = next(iter(storage._holds.values()))
    ordinary.close()
    materializer = module.TTSCloneReferenceMaterializer(selected)
    handle = await materializer.materialize(reference)
    assert handle.voice_ref.read_bytes() == reference.wav_bytes
    original_close = os.close
    calls = []
    if mode != "live":
        target = handle._record.lock_fd
        module.os = types.SimpleNamespace(**vars(os))

        def close(fd):
            if fd == target:
                calls.append(fd)
                if after:
                    original_close(fd)
                raise OSError("injected native close outcome")
            original_close(fd)

        module.os.close = close
        failure = None
        try:
            await (handle.aclose() if mode == "handle" else materializer.close())
        except module.TTSCloneMaterializationError as error:
            failure = error
        # Native close uncertainty must remain visible even if unlink succeeded.
        assert calls
    pause = storage._begin_local_pause()
    drained = pause.drain(time.monotonic() + 0.05)
    native = _probe(hold.authority.control_root, hold.names)
    assert native == "blocked", (mode, after, drained, native)
    assert not drained
    if mode == "live":
        assert handle.voice_ref.read_bytes() == reference.wav_bytes
        pause.resume()
        await handle.aclose()
        await materializer.close()
        assert _probe(hold.authority.control_root, hold.names) == "entered"
    else:
        assert failure is not None
        assert materializer.owns(handle)
        assert len(calls) == 1


@pytest.mark.parametrize("mode", ["live", "handle", "terminal"])
@pytest.mark.parametrize("after", [False, True])
def test_live_and_uncertain_clone_owners_retain_native_exclusion(tmp_path, mode, after):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_materializer_maintenance import _live_child
asyncio.run(_live_child(Path(sys.argv[1]), sys.argv[2], sys.argv[3] == 'True'))
""",
        mode,
        str(after),
    )


async def _paused_first_child(root):
    from Tests.TTS.test_profile_reference_materialization import _reference
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS.profile_reference_materialization import (
        TTSCloneMaterializationError,
        TTSCloneReferenceMaterializer,
    )

    selected = root / "not-created"
    materializer = TTSCloneReferenceMaterializer(selected)
    pause = storage._begin_local_pause()
    handle = None
    try:
        try:
            handle = await materializer.materialize(_reference())
        except TTSCloneMaterializationError:
            pass
        assert not selected.exists(), "first sweep created native paths during pause"
        assert handle is None
    finally:
        pause.resume()
        await materializer.close()


def test_paused_first_materialization_has_no_filesystem_effects(tmp_path):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_materializer_maintenance import _paused_first_child
asyncio.run(_paused_first_child(Path(sys.argv[1])))
""",
    )


async def _prepare_close_child(root, after):
    import os
    import time

    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_profile_reference_materialization import _reference
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import profile_reference_materialization as module
    from tldw_chatbook.Utils import private_paths

    selected = root / "runtime"
    ordinary = storage.acquire_storage(selected)
    hold = next(iter(storage._holds.values()))
    ordinary.close()
    selected.mkdir(mode=0o700)
    selected_identity = (selected.stat().st_dev, selected.stat().st_ino)
    close = private_paths._native_close
    calls = []

    def uncertain(fd):
        info = os.fstat(fd)
        if (info.st_dev, info.st_ino) == selected_identity and not calls:
            calls.append(fd)
            if after:
                close(fd)
            raise OSError("native traversal close outcome")
        close(fd)

    private_paths._native_close = uncertain
    materializer = module.TTSCloneReferenceMaterializer(selected)
    try:
        await materializer.materialize(_reference())
    except module.TTSCloneMaterializationError:
        pass
    assert calls, "traversal native seam not reached"
    pause = storage._begin_local_pause()
    drained = pause.drain(time.monotonic() + 0.05)
    assert _probe(hold.authority.control_root, hold.names) == "blocked", drained
    assert not drained


@pytest.mark.parametrize("after", [False, True])
def test_prepare_traversal_uncertainty_retains_native_ownership(tmp_path, after):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_materializer_maintenance import _prepare_close_child
asyncio.run(_prepare_close_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(after),
    )


async def _source_child(root):
    from Tests.TTS.test_profile_reference_materialization import _reference
    from tldw_chatbook import config
    from tldw_chatbook.TTS.adapter_bootstrap import build_default_tts_service
    from tldw_chatbook.TTS.profile_reference_materialization import (
        TTSCloneMaterializationError,
    )

    config.load_cli_config_and_ensure_existence()
    service = build_default_tts_service({})
    materializer = service._clone_materializer
    selected = materializer._root
    assert not selected.exists(), "factory eagerly materialized private data"
    config._CONFIG_CACHE["general"]["users_name"] = "remapped-profile"
    handle = None
    try:
        try:
            handle = await materializer.materialize(_reference())
        except TTSCloneMaterializationError:
            pass
        assert handle is None, "stale configured materializer published a reference"
        assert not selected.exists(), (
            "stale configured source performed native creation"
        )
    finally:
        await service.close()
        await service.wait_closed()


def test_actual_factory_materializer_rejects_configured_source_remapping(tmp_path):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_materializer_maintenance import _source_child
asyncio.run(_source_child(Path(sys.argv[1])))
""",
    )


async def _allocation_child(root, phase, outcome):
    import os
    import time

    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_profile_reference_materialization import _reference
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import profile_reference_materialization as module
    from tldw_chatbook.Utils import private_paths

    selected = root / "runtime"
    ordinary = storage.acquire_storage(selected)
    hold = next(iter(storage._holds.values()))
    ordinary.close()
    materializer = module.TTSCloneReferenceMaterializer(selected)
    handle = None
    if phase == "traversal":
        selected.mkdir(mode=0o700)
    if phase == "validate":
        handle = await materializer.materialize(_reference())
    elif phase == "create":
        await materializer._ensure_swept()
    elif phase == "sweep":
        selected.mkdir(mode=0o700)
        orphan = selected / ("clone-v1-" + "a" * 32)
        orphan.mkdir(mode=0o700)
        (orphan / "owner.lock").write_bytes(b"")
        (orphan / "owner.lock").chmod(0o600)
    module_to_patch = private_paths if phase == "traversal" else module
    name = "_native_open" if phase == "traversal" else "_open_clone_descriptor"
    original = getattr(module_to_patch, name)
    reached = []
    returned = []

    def opening(*args, **kwargs):
        matches = (
            (args[0] == selected.name)
            if phase == "traversal"
            else phase != "sweep" or args[0] == "owner.lock"
        )
        if matches and not reached:
            reached.append(args[0])
            if outcome == "unreturned":
                native_kwargs = {
                    key: value for key, value in kwargs.items() if key != "_outcome"
                }
                returned.append(os.open(*args, **native_kwargs))
                raise OSError("unknown allocating provider")
            if outcome == "returned":
                returned.append(original(*args, **kwargs))
                raise OSError("failure after known native return")
            return original(selected / "missing-native-leaf", args[1], **kwargs)
        return original(*args, **kwargs)

    setattr(module_to_patch, name, opening)
    failure = None
    try:
        if handle is not None:
            await handle.validated_voice_ref()
        else:
            handle = await materializer.materialize(_reference())
    except module.TTSCloneMaterializationError as error:
        failure = error
    assert reached
    setattr(module_to_patch, name, original)
    if handle is not None:
        await handle.aclose()
    pause = storage._begin_local_pause()
    drained = pause.drain(time.monotonic() + 0.05)
    expected = "blocked" if outcome == "unreturned" else "entered"
    observed = _probe(hold.authority.control_root, hold.names)
    assert observed == expected, (phase, outcome, drained, observed)
    assert drained == (outcome != "unreturned")
    if outcome == "unreturned":
        assert returned and os.fstat(returned[0])
    else:
        for fd in returned:
            with pytest.raises(OSError):
                os.fstat(fd)
    if phase != "sweep":
        assert failure is not None


@pytest.mark.parametrize(
    "phase", ["traversal", "prepare", "create", "sweep", "validate"]
)
@pytest.mark.parametrize("outcome", ["rejected", "returned", "unreturned"])
def test_native_allocation_outcomes_are_distinct(tmp_path, phase, outcome):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_materializer_maintenance import _allocation_child
asyncio.run(_allocation_child(Path(sys.argv[1]), sys.argv[2], sys.argv[3]))
""",
        phase,
        outcome,
    )


async def _lifecycle_child(root, pause_native, cancel):
    import asyncio
    import time

    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_profile_reference_materialization import _reference
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import profile_reference_materialization as module

    materializer = module.TTSCloneReferenceMaterializer(root / "runtime")
    entered, release = asyncio.Event(), asyncio.Event()
    loop = asyncio.get_running_loop()
    original = module._create_materialization_sync

    def creating(*args, **kwargs):
        loop.call_soon_threadsafe(entered.set)
        asyncio.run_coroutine_threadsafe(release.wait(), loop).result(4)
        return original(*args, **kwargs)

    module._create_materialization_sync = creating
    task = asyncio.create_task(materializer.materialize(_reference()))
    await entered.wait()
    hold = next(iter(storage._holds.values()))
    materializer._maintenance_close_admission()
    assert not await materializer._maintenance_drain(time.monotonic() + 0.03)
    assert _probe(hold.authority.control_root, hold.names) == "blocked"
    pause = storage._begin_local_pause() if pause_native else None
    if cancel:
        task.cancel("first")
        await asyncio.sleep(0)
        task.cancel("second")
    release.set()
    with pytest.raises((module.TTSCloneMaterializationError, asyncio.CancelledError)):
        await task
    assert not tuple(materializer._root.iterdir())
    assert await materializer._maintenance_drain(time.monotonic() + 1)
    if pause is not None:
        assert pause.drain(time.monotonic() + 0.03)
        pause.resume()
    await materializer._maintenance_resume()
    module._create_materialization_sync = original
    handle = await materializer.materialize(_reference())
    materializer._maintenance_close_admission()
    before = handle.voice_ref.read_bytes()
    assert not await materializer._maintenance_drain(time.monotonic() + 0.03)
    assert handle.voice_ref.read_bytes() == before
    await handle.aclose()
    assert await materializer._maintenance_drain(time.monotonic() + 1)
    await materializer.close()
    with pytest.raises(module.TTSCloneMaterializationError):
        await materializer._maintenance_resume()


@pytest.mark.parametrize("pause_native", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
def test_reversible_pause_waits_for_actual_workers_and_preserves_consumers(
    tmp_path, pause_native, cancel
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_materializer_maintenance import _lifecycle_child
asyncio.run(_lifecycle_child(Path(sys.argv[1]), sys.argv[2] == 'True', sys.argv[3] == 'True'))
""",
        str(pause_native),
        str(cancel),
    )


async def _source_variants_child(root, drift):
    import time

    from Tests.TTS.test_profile_reference_materialization import _reference
    from tldw_chatbook import config
    from tldw_chatbook.TTS import adapter_bootstrap as factory
    from tldw_chatbook.TTS.profile_reference_materialization import (
        TTSCloneMaterializationError,
        TTSCloneReferenceMaterializer,
    )

    config.load_cli_config_and_ensure_existence()
    service = factory.build_default_tts_service({})
    materializer = service._clone_materializer
    handle = await materializer.materialize(_reference())
    before = handle.voice_ref.read_bytes()
    old = None
    calls = []
    if drift == "profile":
        config._CONFIG_CACHE["general"]["users_name"] = "different"
    elif drift == "config_path":
        config._CONFIG_CACHE_SOURCE = root / "other-config.toml"
    elif drift == "root":
        materializer._root = root / "different"
    elif drift == "selector":
        old = factory.get_user_data_dir

        def substituted():
            calls.append(True)
            return old()

        factory.get_user_data_dir = substituted
    elif drift == "config_selector":
        old = config.get_user_data_dir

        def substituted():
            calls.append(True)
            return old()

        config.get_user_data_dir = substituted
    else:
        raise AssertionError(drift)
    materializer._maintenance_close_admission()
    with pytest.raises(TTSCloneMaterializationError):
        await materializer._maintenance_resume()
    with pytest.raises(TTSCloneMaterializationError):
        await materializer.materialize(_reference())
    assert handle.voice_ref.read_bytes() == before
    assert calls == [], "changed selector was invoked before identity refusal"
    await handle.aclose()
    assert not handle.voice_ref.exists()
    await service.close()
    await service.wait_closed()
    # Explicit custom constructor still works ordinarily under the same config.
    custom = TTSCloneReferenceMaterializer(root / "ordinary-custom")
    assert custom._configured_source is None
    custom_handle = await custom.materialize(_reference())
    await custom_handle.aclose()
    assert await custom._maintenance_drain(time.monotonic() + 0.1)
    await custom._maintenance_resume()
    await custom.close()


@pytest.mark.parametrize(
    "drift", ["profile", "config_path", "root", "selector", "config_selector"]
)
def test_source_drift_refuses_new_work_but_preserves_original_cleanup(tmp_path, drift):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_materializer_maintenance import _source_variants_child
asyncio.run(_source_variants_child(Path(sys.argv[1]), sys.argv[2]))
""",
        drift,
    )


async def _response_child(root, after):
    import os
    import time
    import types

    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS import test_tts_request_admission as cases
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import profile_reference_materialization as module
    from tldw_chatbook.TTS.adapter_types import TTSAudioResponse

    original = TTSAudioResponse.add_cleanup
    observed = []
    holds = []
    close_calls = []
    owned_before = []

    def adding(response, callback):
        original(response, callback)
        handle = getattr(callback, "__self__", None)
        if isinstance(handle, module.TTSCloneReferenceMaterialization):
            materializer = handle._materializer
            observed.append((materializer, handle))
            native = handle._record.native
            owned_before.append((native, frozenset(native.leases)))
            assert native in materializer._native_operations
            assert (
                frozenset(
                    lease
                    for lease in storage._live_leases
                    if getattr(lease, "native_owner", None) is native
                )
                == owned_before[-1][1]
            )
            assert handle.voice_ref.read_bytes() == cases._clone_reference().wav_bytes
            hold = next(iter(storage._holds.values()))
            holds.append(hold)
            assert _probe(hold.authority.control_root, hold.names) == "blocked"

            async def checked_cleanup():
                materializer._maintenance_close_admission()
                assert not await materializer._maintenance_drain(
                    time.monotonic() + 0.03
                )
                if after != "success":
                    target = handle._record.lock_fd
                    close = os.close
                    module.os = types.SimpleNamespace(**vars(os))

                    def failing(fd):
                        if fd == target:
                            close_calls.append(fd)
                            if after == "after":
                                close(fd)
                            raise OSError("response materialization cleanup outcome")
                        close(fd)

                    module.os.close = failing
                try:
                    await callback()
                finally:
                    observed.append(_probe(hold.authority.control_root, hold.names))

            response._cleanup_callbacks[-1] = checked_cleanup

    TTSAudioResponse.add_cleanup = adding
    if after == "success":
        await cases.test_character_clone_materialization_lives_through_response_cleanup(
            root
        )
        assert not observed[0][0]._native_operations
    else:
        with pytest.raises(module.TTSCloneMaterializationError):
            await cases.test_character_clone_materialization_lives_through_response_cleanup(
                root
            )
        assert observed[-1] == "blocked"
        materializer, handle = observed[0]
        assert materializer.owns(handle)
        assert not await materializer._maintenance_drain(time.monotonic() + 0.03)
    assert observed
    materializer, handle = observed[0]
    native, leases_before = owned_before[0]
    retained = frozenset(
        lease
        for lease in storage._live_leases
        if getattr(lease, "native_owner", None) is native
    )
    if after == "success":
        assert not retained and not close_calls
        assert native not in materializer._native_operations
    else:
        assert retained == leases_before
        assert native in materializer._native_operations
        assert close_calls == [handle._record.lock_fd]
        with pytest.raises(module.TTSCloneMaterializationError):
            await materializer.close()
        assert close_calls == [handle._record.lock_fd]
    startups = tuple(storage._startups.values())
    assert len(startups) == 1
    assert all(
        lease in startups
        or getattr(lease, "native_owner", None) in materializer._native_operations
        for lease in storage._live_leases
    )
    # Actual fixture service has closed; now retire its positively identified
    # process startup through the existing terminal API, never registry clearing.
    storage._shutdown()
    assert _probe(holds[0].authority.control_root, holds[0].names) == (
        "entered" if after == "success" else "blocked"
    )


@pytest.mark.parametrize("after", ["success", "before", "after"])
def test_actual_generation_response_owns_reference_through_native_cleanup(
    tmp_path, after
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_materializer_maintenance import _response_child
asyncio.run(_response_child(Path(sys.argv[1]), sys.argv[2]))
""",
        after,
    )


async def _close_edge_child(root, edge, after):
    import os
    import time
    import types

    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_profile_reference_materialization import _reference
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import profile_reference_materialization as module

    selected = root / "runtime"
    ordinary = storage.acquire_storage(selected)
    hold = next(iter(storage._holds.values()))
    ordinary.close()
    materializer = module.TTSCloneReferenceMaterializer(selected)
    handle = None
    targets, calls = [], []
    if edge in ("validate_asset", "cleanup_root", "cleanup_owner"):
        handle = await materializer.materialize(_reference())
        if edge.startswith("cleanup"):
            targets.append(
                handle._record.root_fd
                if edge == "cleanup_root"
                else handle._record.owner_fd
            )
    elif edge.startswith("sweep"):
        selected.mkdir(mode=0o700)
        orphan = selected / ("clone-v1-" + "b" * 32)
        orphan.mkdir(mode=0o700)
        for name, data in (
            ("owner.lock", b""),
            ("asset-" + "c" * 32 + ".wav", _reference().wav_bytes),
        ):
            (orphan / name).write_bytes(data)
            (orphan / name).chmod(0o600)
    original_open = module._open_clone_descriptor

    def opening(*args, **kwargs):
        fd = original_open(*args, **kwargs)
        leaf = str(args[0])
        target = (
            edge in ("create_asset", "validate_asset", "sweep_asset")
            and leaf.startswith("asset-")
            or edge == "prepare_root"
            and leaf == str(selected)
            or edge == "sweep_lock"
            and leaf == "owner.lock"
            or edge == "sweep_owner"
            and leaf.startswith("clone-v1-")
        )
        if target and not targets:
            targets.append(fd)
        return fd

    module._open_clone_descriptor = opening
    original_close = os.close
    module.os = types.SimpleNamespace(**vars(os))

    def closing(fd):
        if targets and fd == targets[0]:
            calls.append(fd)
            if after:
                original_close(fd)
            raise OSError("selected native close outcome")
        original_close(fd)

    module.os.close = closing
    with pytest.raises(module.TTSCloneMaterializationError):
        if edge.startswith("cleanup"):
            await handle.aclose()
        elif edge == "validate_asset":
            await handle.validated_voice_ref()
        else:
            await materializer.materialize(_reference())
    assert targets and calls
    if handle is not None and not edge.startswith("cleanup"):
        await handle.aclose()
    try:
        await materializer.close()
    except module.TTSCloneMaterializationError:
        pass
    pause = storage._begin_local_pause()
    assert not pause.drain(time.monotonic() + 0.03)
    assert _probe(hold.authority.control_root, hold.names) == "blocked"
    assert len(calls) == 1, "uncertain numeric descriptor was retried"


@pytest.mark.parametrize(
    "edge",
    [
        "prepare_root",
        "create_asset",
        "validate_asset",
        "sweep_lock",
        "sweep_owner",
        "sweep_asset",
        "cleanup_root",
        "cleanup_owner",
    ],
)
@pytest.mark.parametrize("after", [False, True])
def test_each_native_cleanup_edge_retains_uncertainty(tmp_path, edge, after):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_materializer_maintenance import _close_edge_child
asyncio.run(_close_edge_child(Path(sys.argv[1]), sys.argv[2], sys.argv[3] == 'True'))
""",
        edge,
        str(after),
    )


async def _late_constructor_child(root, close_outcome):
    from Tests.TTS.test_profile_reference_materialization import _reference
    from tldw_chatbook.TTS import profile_reference_materialization as module

    materializer = module.TTSCloneReferenceMaterializer(root / "runtime")
    original = module._create_materialization_sync
    records = []
    import os
    import types

    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    hold = []

    def creating(*args, **kwargs):
        record = original(*args, **kwargs)
        records.append(record)
        hold.extend(storage._holds.values())
        if close_outcome != "success":
            original_close = os.close
            module.os = types.SimpleNamespace(**vars(os))

            def closing(fd):
                if fd == record.lock_fd:
                    if close_outcome == "after":
                        original_close(fd)
                    raise OSError("late constructor cleanup outcome")
                original_close(fd)

            module.os.close = closing
        raise OSError("failure after actual record construction")

    module._create_materialization_sync = creating
    with pytest.raises(module.TTSCloneMaterializationError):
        await materializer.materialize(_reference())
    assert records
    assert not records[0].asset_path.exists(), "unpublished owned WAV escaped cleanup"
    assert not records[0].asset_path.parent.exists()
    if close_outcome == "success":
        await materializer.close()
        assert _probe(hold[0].authority.control_root, hold[0].names) == "entered"
    else:
        with pytest.raises(module.TTSCloneMaterializationError):
            await materializer.close()
        assert _probe(hold[0].authority.control_root, hold[0].names) == "blocked"


@pytest.mark.parametrize("close_outcome", ["success", "before", "after"])
def test_known_materialization_record_is_cleaned_after_outer_constructor_failure(
    tmp_path,
    close_outcome,
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_materializer_maintenance import _late_constructor_child
asyncio.run(_late_constructor_child(Path(sys.argv[1]), sys.argv[2]))
""",
        close_outcome,
    )


async def _creation_substitution_child(root, kind):
    import os
    import types

    from Tests.TTS.test_profile_reference_materialization import _reference
    from tldw_chatbook.TTS import profile_reference_materialization as module

    selected = root / "runtime"
    original_open, original_write = module._open_clone_descriptor, os.write
    asset = []
    owner = []

    def opening(*args, **kwargs):
        fd = original_open(*args, **kwargs)
        if str(args[0]).startswith("asset-"):
            asset.append((fd, args[0], kwargs["dir_fd"]))
        if str(args[0]).startswith("clone-v1-"):
            owner.append((args[0], kwargs["dir_fd"]))
        return fd

    module._open_clone_descriptor = opening
    module.os = types.SimpleNamespace(**vars(os))

    def writing(fd, data):
        if asset and fd == asset[0][0]:
            _, leaf, parent = asset[0]
            original_write(fd, data)
            if kind == "asset":
                os.rename(
                    leaf, "original-kept.wav", src_dir_fd=parent, dst_dir_fd=parent
                )
                replaced = leaf
            elif kind == "lock":
                os.rename(
                    "owner.lock",
                    "original-owner.lock",
                    src_dir_fd=parent,
                    dst_dir_fd=parent,
                )
                replaced = "owner.lock"
            elif kind == "owner":
                name, parent = owner[0]
                os.rename(name, "renamed-owner", src_dir_fd=parent, dst_dir_fd=parent)
                os.mkdir(name, 0o700, dir_fd=parent)
                parent = os.open(name, module._DIRECTORY_FLAGS, dir_fd=parent)
                replaced = "foreign-sibling"
            else:
                replaced = "foreign-sibling"
            other = os.open(
                replaced, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600, dir_fd=parent
            )
            os.write(other, b"foreign replacement")
            os.close(other)
            if kind == "owner":
                os.close(parent)
            raise OSError("creation failed after namespace substitution")
        return original_write(fd, data)

    module.os.write = writing
    materializer = module.TTSCloneReferenceMaterializer(selected)
    with pytest.raises(module.TTSCloneMaterializationError):
        await materializer.materialize(_reference())
    assert asset
    patterns = {
        "asset": "clone-v1-*/asset-*.wav",
        "lock": "clone-v1-*/owner.lock",
        "owner": "clone-v1-*/foreign-sibling",
        "sibling": "clone-v1-*/foreign-sibling",
    }
    foreign = list(selected.glob(patterns[kind]))
    assert len(foreign) == 1, "failed creation deleted substituted foreign asset"
    assert foreign[0].read_bytes() == b"foreign replacement"
    if kind == "asset":
        assert (
            next(selected.glob("*/original-kept.wav")).read_bytes()
            == _reference().wav_bytes
        )
    if kind == "owner":
        assert (
            next(selected.glob("renamed-owner/asset-*.wav")).read_bytes()
            == _reference().wav_bytes
        )
    import time

    assert not await materializer._maintenance_drain(time.monotonic() + 0.03)


@pytest.mark.parametrize("kind", ["asset", "lock", "owner", "sibling"])
def test_failed_creation_preserves_substituted_foreign_asset(tmp_path, kind):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_materializer_maintenance import _creation_substitution_child
asyncio.run(_creation_substitution_child(Path(sys.argv[1]), sys.argv[2]))
""",
        kind,
    )


async def _configured_native_child(root):
    import os
    import subprocess
    import sys
    from pathlib import Path

    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_profile_reference_materialization import _reference
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        bind_profile,
    )

    selector = Path(os.environ["TLDW_CONFIG_PATH"])
    profile = Path.home() / ".local/share/tldw_cli/default_user"
    profile.mkdir(mode=0o700, parents=True)
    control = bootstrap.default_bootstrap_root()
    from tldw_chatbook import config

    config.load_cli_config_and_ensure_existence()
    from tldw_chatbook.TTS.adapter_bootstrap import build_default_tts_service

    service = build_default_tts_service({})
    assert all(lease in storage._startups.values() for lease in storage._live_leases)
    storage._shutdown()
    authority = admission_authority(control)
    authority.register("clone.profile", (profile, selector))
    bind_profile(control, selector, ("clone.profile",), authority.control_root)
    storage.admit_startup()
    materializer = service._clone_materializer
    assert materializer._configured_source is not None
    assert materializer._root == profile / "tts_clone_materializations"
    materializer._root.mkdir(mode=0o700)
    foreign = materializer._root / "foreign-sibling"
    foreign.write_bytes(b"keep this namespace")
    handle = await materializer.materialize(_reference())
    leases = handle._record.native.leases
    hold = storage._holds[leases[0]._key]
    assert "clone.profile" in hold.names
    observer = subprocess.run(  # noqa: ASYNC221 - bounded independent native observer
        [
            sys.executable,
            "-c",
            """
import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR | os.O_NOFOLLOW)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print('entered')
except BlockingIOError:
    print('blocked')
finally:
    os.close(fd)
""",
            str(handle.voice_ref.parent / "owner.lock"),
        ],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    assert observer.returncode == 0 and observer.stdout.strip() == "blocked"
    assert handle.voice_ref.read_bytes() == _reference().wav_bytes
    assert _probe(hold.authority.control_root, hold.names) == "blocked"
    await handle.aclose()
    await service.close()
    await service.wait_closed()
    assert foreign.read_bytes() == b"keep this namespace"
    assert all(lease in storage._startups.values() for lease in storage._live_leases)
    storage._shutdown()
    assert _probe(hold.authority.control_root, hold.names) == "entered"


def test_configured_factory_uses_declared_profile_namespace_and_real_owner_lock(
    tmp_path,
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_materializer_maintenance import _configured_native_child
asyncio.run(_configured_native_child(Path(sys.argv[1])))
""",
    )


async def _queued_child(root):
    import asyncio
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from Tests.TTS.test_profile_reference_materialization import _reference
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import profile_reference_materialization as module

    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(pool)
    entered, release = threading.Event(), threading.Event()

    def blocking():
        entered.set()
        assert release.wait(4)

    blocked = loop.run_in_executor(None, blocking)
    while not entered.is_set():
        await asyncio.sleep(0.001)
    materializer = module.TTSCloneReferenceMaterializer(root / "never-created")
    task = asyncio.create_task(materializer.materialize(_reference()))
    while not materializer._worker_tasks:
        await asyncio.sleep(0.001)
    materializer._maintenance_close_admission()
    task.cancel("one")
    await asyncio.sleep(0)
    task.cancel("two")
    assert not await materializer._maintenance_drain(time.monotonic() + 0.03)
    assert not materializer._root.exists()
    pause = storage._begin_local_pause()
    release.set()
    await blocked
    with pytest.raises((asyncio.CancelledError, module.TTSCloneMaterializationError)):
        await task
    assert not materializer._root.exists()
    assert await materializer._maintenance_drain(time.monotonic() + 0.3)
    pause.resume()
    await materializer._maintenance_resume()
    handle = await materializer.materialize(_reference())
    await handle.aclose()
    await materializer.close()


def test_queued_to_thread_repeated_cancellation_has_no_late_creation(tmp_path):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_materializer_maintenance import _queued_child
asyncio.run(_queued_child(Path(sys.argv[1])))
""",
    )


async def _release_child(root, after):
    import time

    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_profile_reference_materialization import _reference
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import profile_reference_materialization as module

    materializer = module.TTSCloneReferenceMaterializer(root / "runtime")
    handle = await materializer.materialize(_reference())
    native = handle._record.native
    target = native.leases[-1]
    hold = storage._holds[target._key]
    original = target.close
    calls = []

    def close():
        calls.append(True)
        if after:
            original()
        raise OSError("uncertain lease release")

    target.close = close
    with pytest.raises(module.TTSCloneMaterializationError):
        await handle.aclose()
    assert not handle.voice_ref.exists()
    with pytest.raises(module.TTSCloneMaterializationError):
        await materializer.close()
    assert len(calls) == 1
    assert _probe(hold.authority.control_root, hold.names) == "blocked"
    assert not await materializer._maintenance_drain(time.monotonic() + 0.03)


@pytest.mark.parametrize("after", [False, True])
def test_first_uncertain_admission_release_retains_remaining_native_holds(
    tmp_path, after
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_materializer_maintenance import _release_child
asyncio.run(_release_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(after),
    )


async def _validation_pause_child(root):
    from Tests.TTS.test_profile_reference_materialization import _reference
    from tldw_chatbook.TTS import profile_reference_materialization as module

    materializer = module.TTSCloneReferenceMaterializer(root / "runtime")
    handle = await materializer.materialize(_reference())
    materializer._maintenance_close_admission()
    with pytest.raises(module.TTSCloneMaterializationError):
        await handle.validated_voice_ref()
    await materializer._maintenance_resume()
    assert await handle.validated_voice_ref() == handle.voice_ref
    await handle.aclose()
    await materializer.close()


def test_local_admission_pause_refuses_new_handle_validation(tmp_path):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_materializer_maintenance import _validation_pause_child
asyncio.run(_validation_pause_child(Path(sys.argv[1])))
""",
    )


async def _final_control_child(root):
    import asyncio

    from Tests.TTS.test_profile_reference_materialization import _reference
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import profile_reference_materialization as module

    signal = asyncio.CancelledError("native final close")
    original = storage.StorageLease.close
    calls = []

    def closing(lease):
        native = getattr(lease, "native_owner", None)
        if isinstance(native, module._CloneNativeOperation) and not calls:
            calls.append(lease)
            original(lease)
            raise signal
        original(lease)

    storage.StorageLease.close = closing
    materializer = module.TTSCloneReferenceMaterializer(root / "runtime")
    caught = None
    try:
        await materializer.materialize(_reference())
    except BaseException as error:  # noqa: BLE001 - assert exact native control signal
        caught = error
    assert calls
    assert caught is signal, ("original control replaced", type(caught))


def test_final_native_release_preserves_original_control_identity(tmp_path):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_materializer_maintenance import _final_control_child
asyncio.run(_final_control_child(Path(sys.argv[1])))
""",
    )


async def _foreign_parent_child(root, foreign_root):
    import os

    from Tests.TTS.test_profile_reference_materialization import _reference
    from tldw_chatbook.TTS import profile_reference_materialization as module

    materializer = module.TTSCloneReferenceMaterializer(root / "runtime")
    foreign = root / "foreign"
    foreign.mkdir(mode=0o700)
    (foreign / "keep").write_bytes(b"foreign bytes")
    original = module._create_materialization_sync
    reached = []

    def creating(selected, identity, wav, *, _native):
        if foreign_root:
            info = foreign.stat()
            reached.append(True)
            return original(foreign, (info.st_dev, info.st_ino), wav, _native=_native)
        parent = os.open(foreign, module._DIRECTORY_FLAGS)
        try:
            reached.append(True)
            fd = module._clone_open(_native, "keep", os.O_RDONLY, dir_fd=parent)
            module._clone_close(_native, fd)
        finally:
            os.close(parent)
        return original(selected, identity, wav, _native=_native)

    module._create_materialization_sync = creating
    with pytest.raises(module.TTSCloneMaterializationError):
        await materializer.materialize(_reference())
    assert reached
    assert list(foreign.iterdir()) == [foreign / "keep"]
    assert (foreign / "keep").read_bytes() == b"foreign bytes"


@pytest.mark.parametrize("foreign_root", [False, True])
def test_native_helper_rejects_foreign_root_and_unknown_parent(tmp_path, foreign_root):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_materializer_maintenance import _foreign_parent_child
asyncio.run(_foreign_parent_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(foreign_root),
    )


async def _cancelled_cleanup_child(root, after):
    import asyncio
    import os
    import time
    import types

    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_profile_reference_materialization import _reference
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import profile_reference_materialization as module

    loop = asyncio.get_running_loop()
    entered, release = asyncio.Event(), asyncio.Event()
    original = module._create_materialization_sync
    records = []

    def creating(*args, **kwargs):
        record = original(*args, **kwargs)
        records.append(record)
        loop.call_soon_threadsafe(entered.set)
        asyncio.run_coroutine_threadsafe(release.wait(), loop).result(4)
        return record

    module._create_materialization_sync = creating
    materializer = module.TTSCloneReferenceMaterializer(root / "runtime")
    task = asyncio.create_task(materializer.materialize(_reference()))
    await entered.wait()
    hold = next(iter(storage._holds.values()))
    target = records[0].lock_fd
    original_close = os.close
    calls = []
    module.os = types.SimpleNamespace(**vars(os))

    def closing(fd):
        if fd == target:
            calls.append(fd)
            if after:
                original_close(fd)
            raise OSError("cancelled unpublished native cleanup")
        original_close(fd)

    module.os.close = closing
    task.cancel("first")
    await asyncio.sleep(0)
    task.cancel("second")
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(module.TTSCloneMaterializationError):
        await materializer.close()
    assert calls == [target]
    assert not await materializer._maintenance_drain(time.monotonic() + 0.03)
    assert _probe(hold.authority.control_root, hold.names) == "blocked"


@pytest.mark.parametrize("after", [False, True])
def test_cancelled_unpublished_record_retains_failed_native_cleanup(tmp_path, after):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_materializer_maintenance import _cancelled_cleanup_child
asyncio.run(_cancelled_cleanup_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(after),
    )


async def _factory_compatibility_child(root, case):
    import inspect

    from Tests.TTS import test_tts_app_ownership as app_cases
    from Tests.TTS import test_tts_registry_service as registry_cases

    with pytest.MonkeyPatch.context() as patch:
        if case == "app":
            app_cases.test_app_constructs_one_tts_service(patch, root)
            return
        function = getattr(registry_cases, case)
        result = (
            function(patch)
            if "monkeypatch" in inspect.signature(function).parameters
            else function()
        )
        if inspect.isawaitable(result):
            await result


@pytest.mark.parametrize(
    "case",
    [
        "app",
        "test_default_bootstrap_prepends_audio_cpp_without_changing_legacy_specs",
        "test_default_service_constructs_one_supervisor_without_launch",
        "test_default_service_runtime_observation_does_not_materialize_adapter",
        "test_default_bootstrap_parses_supplied_preferences_exactly_once",
        "test_default_bootstrap_wires_lazy_nonmigrating_studio_reader",
    ],
)
def test_original_factory_and_app_cases_with_coherent_private_source(tmp_path, case):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_materializer_maintenance import _factory_compatibility_child
asyncio.run(_factory_compatibility_child(Path(sys.argv[1]), sys.argv[2]))
""",
        case,
    )


async def _custom_factory_child(root):
    from Tests.TTS.test_profile_reference_materialization import _reference
    from tldw_chatbook import config
    from tldw_chatbook.TTS import adapter_bootstrap

    config.load_cli_config_and_ensure_existence()
    selected = root / "custom-factory"
    calls = []

    def custom_selector():
        calls.append(selected)
        return selected

    adapter_bootstrap.get_user_data_dir = custom_selector
    service = adapter_bootstrap.build_default_tts_service({})
    materializer = service._clone_materializer
    assert materializer._configured_source is None
    assert materializer._root == selected / "tts_clone_materializations"
    assert calls == [selected]
    assert not selected.exists()
    config._CONFIG_CACHE["general"]["users_name"] = "other-profile"
    handle = await materializer.materialize(_reference())
    assert handle.voice_ref.is_relative_to(selected)
    await handle.aclose()
    await service.close()
    await service.wait_closed()


def test_prebinding_custom_factory_selector_stays_ordinary(tmp_path):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_materializer_maintenance import _custom_factory_child
asyncio.run(_custom_factory_child(Path(sys.argv[1])))
""",
    )


async def _final_descriptor_control_child(root, after, body_failure):
    import asyncio
    import os
    import time
    import types

    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_profile_reference_materialization import _reference
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import profile_reference_materialization as module

    signal = asyncio.CancelledError("final descriptor control")
    original_prepare = module._prepare_runtime_root_sync
    original_close = os.close
    module.os = types.SimpleNamespace(**vars(os))
    target = []
    calls = []
    body = module.TTSCloneMaterializationError("unavailable")

    def preparing(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        native = kwargs["_native"]
        target.append(native.open(args[0], module._DIRECTORY_FLAGS))
        if body_failure:
            raise body
        return result

    def closing(fd):
        if fd in target:
            calls.append(fd)
            if after:
                original_close(fd)
            raise signal
        original_close(fd)

    module._prepare_runtime_root_sync = preparing
    module.os.close = closing
    materializer = module.TTSCloneReferenceMaterializer(root / "runtime")
    caught = None
    try:
        await materializer.materialize(_reference())
    except BaseException as error:  # noqa: BLE001 - exact body/control precedence
        caught = error
    if body_failure:
        assert isinstance(caught, module.TTSCloneMaterializationError)
        assert caught.code == "unavailable"
    else:
        assert caught is signal
    assert calls == target and len(target) == 1
    hold = next(iter(storage._holds.values()))
    assert not await materializer._maintenance_drain(time.monotonic() + 0.01)
    assert _probe(hold.authority.control_root, hold.names) == "blocked"
    with pytest.raises(module.TTSCloneMaterializationError):
        await materializer.close()
    assert calls == target


@pytest.mark.parametrize("after", [False, True])
@pytest.mark.parametrize("body_failure", [False, True])
def test_final_descriptor_control_preserves_body_precedence_and_retention(
    tmp_path, after, body_failure
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_materializer_maintenance import _final_descriptor_control_child
asyncio.run(_final_descriptor_control_child(Path(sys.argv[1]), sys.argv[2] == 'True', sys.argv[3] == 'True'))
""",
        str(after),
        str(body_failure),
    )
