"""Profile Library caller lifetime and non-destructive maintenance evidence."""

import pytest

from Tests.TTS.test_profile_repository_maintenance import _run_private_child


async def _until(predicate):
    import asyncio
    import time

    deadline = time.monotonic() + 4
    while not predicate():
        assert time.monotonic() < deadline, "condition did not settle"
        await asyncio.sleep(0.01)


async def _export_child(root, queued=False):
    import asyncio
    import threading
    import pytest
    from textual.app import App, ComposeResult
    from textual.widgets import DataTable

    from Tests.TTS.test_profile_service import _FakeTTSService
    from Tests.TTS.test_profile_repository_lifecycle import _draft
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
    from tldw_chatbook.TTS.profile_service import TTSProfileService
    from tldw_chatbook.UI.stts_profile_library import STTSProfileLibrary

    class Host(App):
        async def load(self):
            return service

        def compose(self) -> ComposeResult:
            yield STTSProfileLibrary(self.load)

    repository = TTSProfileRepository(root / "profiles.sqlite")
    await repository.open()
    await repository.create_profile(_draft("Exported"))
    service = TTSProfileService(repository, _FakeTTSService())
    app = Host()
    entered, release = threading.Event(), threading.Event()
    destination = root / "export.json"
    destination.write_text("existing destination")
    original_write = STTSProfileLibrary._write_profile_export

    expected = []

    def write(target, content, **kwargs):
        expected.append(content.encode("utf-8"))
        entered.set()
        assert release.wait(4)
        original_write(target, content, **kwargs)

    STTSProfileLibrary._write_profile_export = staticmethod(write)
    try:
        async with app.run_test() as pilot:
            library = app.query_one(STTSProfileLibrary)
            await _until(lambda: bool(library._loaded_rows))
            table = library.query_one(DataTable)
            table.move_cursor(row=0)
            table.action_select_cursor()
            await _until(lambda: library._selected_profile is not None)

            async def choose():
                return destination

            library._choose_profile_export_path = choose
            executor = None
            queue_release = threading.Event()
            if queued:
                from concurrent.futures import ThreadPoolExecutor

                executor = ThreadPoolExecutor(max_workers=1)
                asyncio.get_running_loop().set_default_executor(executor)
                occupied = threading.Event()

                def occupy():
                    occupied.set()
                    assert queue_release.wait(4)

                executor.submit(occupy)
                await _until(occupied.is_set)
            action = asyncio.create_task(library.export_selected_profile())
            if queued:
                await _until(lambda: bool(library._export_operations))
                action.cancel()
                await asyncio.sleep(0.02)
                action.cancel()
                assert not entered.is_set() and not action.done()
                assert library._export_workers and library._export_operations
                queue_release.set()
            await _until(entered.is_set)
            action.cancel()
            await asyncio.sleep(0.03)
            action.cancel()
            await asyncio.sleep(0.02)
            try:
                assert not action.done(), (
                    "caller retired before original native write completed"
                )
            finally:
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await action
            await _until(lambda: destination.read_text() != "existing destination")
            assert destination.read_bytes() == expected[0]
    finally:
        release.set()
        await repository.close()


@pytest.mark.parametrize("queued", [False, True])
def test_cancelled_sanitized_export_waits_for_actual_write_result(tmp_path, queued):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_library_maintenance import _export_child
asyncio.run(_export_child(Path(sys.argv[1]), sys.argv[2] == "True"))
""",
        str(queued),
    )


async def _export_close_child(root, after):
    import os
    import time
    from textual.app import App, ComposeResult
    from textual.widgets import DataTable
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_profile_service import _FakeTTSService
    from Tests.TTS.test_profile_repository_lifecycle import _draft
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
    from tldw_chatbook.TTS.profile_service import TTSProfileService
    from tldw_chatbook.UI.stts_profile_library import STTSProfileLibrary
    from tldw_chatbook.UI import stts_profile_library as module

    repository = TTSProfileRepository(root / "profiles.sqlite")
    await repository.open()
    await repository.create_profile(_draft("Exported"))
    service = TTSProfileService(repository, _FakeTTSService())
    hold = next(iter(storage._holds.values()))
    destination = root / "export.json"
    destination.write_text("existing destination")
    streams, attempts = [], []

    class Stream:
        def __init__(self, native):
            self.native = native
            self.fd = native.fileno()

        def __getattr__(self, name):
            return getattr(self.native, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def close(self):
            attempts.append(self.fd)
            assert os.fstat(self.fd).st_ino == destination.stat().st_ino
            self.native.flush()
            if after is not False:
                self.native.close()
            if type(after) is bool:
                raise OSError("injected selected export close")

    original_creator = module._open_profile_export_stream

    def open_stream(path, *, _outcome):
        native = original_creator(path, _outcome=_outcome)
        stream = Stream(native)
        streams.append(stream)
        _outcome.stream = stream
        if after == "late":
            raise OSError("late observed export stream")
        return stream

    module._open_profile_export_stream = open_stream
    expected = []
    original_writer = STTSProfileLibrary._write_profile_export

    def writer(target, content, **kwargs):
        expected.append(content.encode("utf-8"))
        return original_writer(target, content, **kwargs)

    STTSProfileLibrary._write_profile_export = staticmethod(writer)
    if after in {"unreturned", "rejected"}:
        import io
        import types

        module.io = types.SimpleNamespace(**vars(io))
        if after == "unreturned":

            def leaked(*args, **kwargs):
                stream = io.open(*args, **kwargs)
                streams.append(stream)
                raise OSError("unreturned actual export stream")

            module.io.open = leaked
        else:
            destination = root / "missing" / "export.json"
    descriptor_attempts = []
    if after == "portable":
        import types

        module.os = types.SimpleNamespace(
            **{name: value for name, value in vars(os).items() if name != "O_DIRECTORY"}
        )

        def unsupported(*args, **kwargs):
            raise NotImplementedError("simulated directory/dir_fd unavailable")

        module.os.open = unsupported
    if type(after) is str and "_fd_" in after:
        import types

        module.os = types.SimpleNamespace(**vars(os))
        target_inode = (
            (destination.parent if after.startswith("parent") else destination)
            .stat()
            .st_ino
        )

        def close_descriptor(fd):
            if os.fstat(fd).st_ino == target_inode:
                descriptor_attempts.append(fd)
                if after.endswith("after"):
                    os.close(fd)
                raise OSError("selected export descriptor close")
            os.close(fd)

        module.os.close = close_descriptor
    if after == "lease":
        original_close = storage.StorageLease.close
        failed = []

        def close_lease(lease):
            original_close(lease)
            if getattr(lease, "native_owner", None) is not None and not failed:
                failed.append(lease)
                raise OSError("after actual export lease release")

        storage.StorageLease.close = close_lease

    class Host(App):
        async def load(self):
            return service

        def compose(self) -> ComposeResult:
            yield STTSProfileLibrary(self.load)

    app = Host()
    async with app.run_test():
        library = app.query_one(STTSProfileLibrary)
        await _until(lambda: bool(library._loaded_rows))
        table = library.query_one(DataTable)
        table.move_cursor(row=0)
        table.action_select_cursor()
        await _until(lambda: library._selected_profile is not None)

        async def choose():
            return destination

        library._choose_profile_export_path = choose
        assert await library.export_selected_profile() is (
            after in {"normal", "portable"}
        )
        if after == "portable":
            assert not await library._maintenance_drain(time.monotonic() + 0.05)
            assert library.profile_maintenance_state() == "unqualified"
        if after not in {"unreturned", "rejected"}:
            assert len(attempts) == 1 and len(streams) == 1
            assert destination.read_bytes() == (b"" if after == "late" else expected[0])
        elif after == "unreturned":
            assert len(streams) == 1 and not attempts
        else:
            assert not streams and not attempts and not destination.exists()
        await repository.close()
        # This bare pilot owns no runtime services. Only its known startup and
        # actual export holds may remain; retire that fixture startup explicitly.
        assert all(
            lease in storage._startups.values()
            or getattr(lease, "native_owner", None)
            in getattr(library, "_export_operations", ())
            for lease in storage._live_leases
        )
        storage._shutdown()
        pause = storage._begin_local_pause()
        drained = pause.drain(time.monotonic() + 0.03)
        observed = _probe(hold.authority.control_root, hold.names)
        uncertain = (
            type(after) is bool or after in {"unreturned", "lease"} or "_fd_" in after
        )
        if type(after) is str and "_fd_" in after:
            assert len(descriptor_attempts) == 1
        assert observed == ("blocked" if uncertain else "entered"), (
            after,
            drained,
            observed,
        )
        assert drained is not uncertain


@pytest.mark.parametrize("after", [False, True])
def test_export_uncertain_native_close_is_not_retirement(tmp_path, after):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_library_maintenance import _export_close_child
asyncio.run(_export_close_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(after),
    )


async def _dirty_child(root, conflict):
    import time
    from textual.app import App, ComposeResult
    from textual.widgets import Button, DataTable, Input
    from Tests.TTS.test_profile_service import _FakeTTSService
    from Tests.TTS.test_profile_repository_lifecycle import _draft
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
    from tldw_chatbook.TTS.profile_service import TTSProfileService
    from tldw_chatbook.UI.stts_profile_library import (
        STTSProfileLibrary,
        TTSProfileEditorModal,
    )

    repository = TTSProfileRepository(root / "profiles.sqlite")
    await repository.open()
    created = await repository.create_profile(_draft("Original"))
    service = TTSProfileService(repository, _FakeTTSService())

    class Host(App):
        async def load(self):
            return service

        def compose(self) -> ComposeResult:
            yield STTSProfileLibrary(self.load)

    app = Host()
    try:
        async with app.run_test(size=(100, 45)) as pilot:
            library = app.query_one(STTSProfileLibrary)
            await _until(lambda: bool(library._loaded_rows))
            table = library.query_one(DataTable)
            table.move_cursor(row=0)
            table.action_select_cursor()
            await _until(lambda: library._selected_profile is not None)
            selected = library._selected_profile
            worker = library.run_worker(
                library.edit_selected_profile(), exit_on_error=False
            )
            await _until(
                lambda: (
                    isinstance(app.screen, TTSProfileEditorModal)
                    and app.screen.is_mounted
                )
            )
            modal = app.screen
            field = modal.query_one("#stts-profile-editor-name", Input)
            field.value = "Uncommitted private draft"
            await _until(lambda: field.value == "Uncommitted private draft")
            if conflict is True:
                await repository.update_profile(
                    created.value.profile_id,
                    1,
                    _draft("Competing writer"),
                    expected_generation=created.generation,
                )
                modal.query_one("#stts-profile-editor-save", Button).press()
                await worker.wait()
                assert library._retained_editor_draft[1].display_name == field.value
            else:
                assert library._active_modal is modal
            await _until(lambda: library._active_page_task is None)
            # BASE has no Library participation: demonstrate the real lower-layer
            # composition gap, not an absent API or a Complete-backup capability.
            local_boundary = getattr(library, "_maintenance_drain", None)
            if local_boundary is None:
                repository._maintenance_close_admission()
                assert await repository._maintenance_drain(time.monotonic() + 2)
                pause = storage._begin_local_pause()
                try:
                    ready = pause.drain(time.monotonic() + 0.05)
                finally:
                    pause.resume()
            else:
                ready = await local_boundary(time.monotonic() + 0.05)
            assert not ready, (
                "existing repository/global drains overlook unresolved editor"
            )
            assert library._selected_profile is selected
            if conflict is True:
                assert (
                    library._retained_editor_draft[1].display_name
                    == "Uncommitted private draft"
                )
            else:
                assert library._active_modal is modal and modal.is_mounted
                assert field.value == "Uncommitted private draft"
            if conflict in {"service_paused", "global_paused"}:
                pause = None
                if conflict == "service_paused":
                    service._maintenance_close_admission()
                else:
                    pause = storage._begin_local_pause()
                modal.query_one("#stts-profile-editor-save", Button).press()
                try:
                    await worker.wait()
                    assert library._active_modal is None
                    assert library._retained_editor_draft is not None, (
                        "submitted draft lost after maintenance refused Save"
                    )
                    assert (
                        library._retained_editor_draft[1].display_name
                        == "Uncommitted private draft"
                    )
                finally:
                    if pause is not None:
                        pause.resume()
                    if conflict == "service_paused":
                        await service._maintenance_resume()
                assert (
                    await repository.get_profile(created.value.profile_id)
                ).value.display_name == "Original"
                await library._maintenance_resume()
                retry = library.run_worker(
                    library.edit_selected_profile(), exit_on_error=False
                )
                await _until(
                    lambda: (
                        isinstance(app.screen, TTSProfileEditorModal)
                        and app.screen.is_mounted
                    )
                )
                assert (
                    app.screen.query_one("#stts-profile-editor-name", Input).value
                    == "Uncommitted private draft"
                )
                app.screen.query_one("#stts-profile-editor-save", Button).press()
                await retry.wait()
                assert library._retained_editor_draft is None
                assert (
                    await repository.get_profile(created.value.profile_id)
                ).value.display_name == "Uncommitted private draft"
                return
            await library._maintenance_resume()
            if conflict is not True:
                modal.query_one("#stts-profile-editor-cancel", Button).press()
                await worker.wait()
                assert library._active_modal is None
                assert (
                    await repository.get_profile(created.value.profile_id)
                ).value.display_name == "Original"
            else:
                # Cancel the reopened editor preserves the submitted conflict draft.
                retained = library._retained_editor_draft
                retry = library.run_worker(
                    library.edit_selected_profile(), exit_on_error=False
                )
                await _until(
                    lambda: (
                        isinstance(app.screen, TTSProfileEditorModal)
                        and app.screen.is_mounted
                    )
                )
                reopened = app.screen
                assert (
                    reopened.query_one("#stts-profile-editor-name", Input).value
                    == retained[1].display_name
                )
                reopened.query_one("#stts-profile-editor-cancel", Button).press()
                await retry.wait()
                assert library._retained_editor_draft is retained
                library.query_one("#stts-profile-refresh-btn", Button).press()
                await _until(
                    lambda: any(
                        row.profile.display_name == "Competing writer"
                        for row in library._loaded_rows.values()
                    )
                )
                table.move_cursor(row=0)
                table.action_select_cursor()
                await _until(
                    lambda: (
                        library._selected_profile is not None
                        and library._selected_profile.profile.display_name
                        == "Competing writer"
                    )
                )
                retry = library.run_worker(
                    library.edit_selected_profile(), exit_on_error=False
                )
                await _until(
                    lambda: (
                        isinstance(app.screen, TTSProfileEditorModal)
                        and app.screen.is_mounted
                    )
                )
                assert (
                    app.screen.query_one("#stts-profile-editor-name", Input).value
                    == retained[1].display_name
                )
                app.screen.query_one("#stts-profile-editor-save", Button).press()
                await retry.wait()
                assert library._retained_editor_draft is None
                assert (
                    await repository.get_profile(created.value.profile_id)
                ).value.display_name == "Uncommitted private draft"
            await _until(lambda: library._active_page_task is None)
            assert await library._maintenance_drain(time.monotonic() + 1)
            assert library.profile_maintenance_state() == "ready"
            assert await library.export_selected_profile() is False
            assert await library.import_voice_bundle() is False
            assert await library.delete_selected_profile() is False
            snapshot = tuple(library._loaded_rows)
            library._queue_page_request("deferred search", 0)
            assert tuple(library._loaded_rows) == snapshot
            assert library._active_page_task is None
            await library._maintenance_resume()
            await _until(
                lambda: library._active_page_task is None and not library._loaded_rows
            )
    finally:
        await repository.close()


@pytest.mark.parametrize("conflict", [False, True])
def test_library_drain_preserves_live_or_conflict_editor(tmp_path, conflict):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_library_maintenance import _dirty_child
asyncio.run(_dirty_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(conflict),
    )


@pytest.mark.parametrize("mode", ["normal", "late", "unreturned", "rejected", "lease"])
def test_export_allocation_and_positive_retirement_outcomes(tmp_path, mode):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_library_maintenance import _export_close_child
asyncio.run(_export_close_child(Path(sys.argv[1]), sys.argv[2]))
""",
        mode,
    )


async def _review_child(root, cleanup_failure=False):
    import time
    from textual.widgets import Button
    from Tests.UI.test_stts_profile_library import (
        _ActionProfileService,
        _BundleActionHost,
        _profile,
        _select_action_profile,
        _acknowledge_bundle_warning,
        _wait_until,
    )
    from Tests.TTS.test_voice_bundle_service import _bundle, _service
    from tldw_chatbook.TTS.voice_bundle_codec import encode_clone_voice_bundle

    bundle, _, _ = _service(root)
    source = root / "selected.bundle"
    payload = encode_clone_voice_bundle(_bundle())
    source.write_bytes(payload)
    source.chmod(0o600)
    app = _BundleActionHost(_ActionProfileService(_profile(0)), bundle)
    try:
        async with app.run_test(size=(80, 24)) as pilot:
            library, selected = await _select_action_profile(app, pilot)

            async def choose():
                return source

            library._choose_voice_bundle_import_path = choose
            library.query_one("#stts-profile-import-btn", Button).press()
            await _wait_until(
                pilot, lambda: len(app.screen.query("#bundle-warning-ack")) == 1
            )
            await _acknowledge_bundle_warning(app, pilot)
            await _wait_until(
                pilot, lambda: len(app.screen.query("#stts-bundle-review-cancel")) == 1
            )
            await _wait_until(
                pilot,
                lambda: (
                    library._active_modal is not None
                    and library._active_modal.is_mounted
                ),
            )
            modal = library._active_modal
            handle = library._active_bundle_handle
            session = bundle._sessions[handle]
            assert not await library._maintenance_drain(time.monotonic() + 0.05)
            assert await bundle._maintenance_drain(time.monotonic() + 0.2)
            assert library._active_modal is modal and modal.is_mounted
            assert library._active_bundle_handle is handle
            assert bundle._sessions[handle] is session
            assert library._selected_profile is selected
            await bundle._maintenance_resume()
            await library._maintenance_resume()
            if cleanup_failure:
                original_invalidate = bundle.invalidate

                async def invalidate(value):
                    await original_invalidate(value)
                    raise RuntimeError("after actual original session invalidation")

                bundle.invalidate = invalidate
            modal.query_one("#stts-bundle-review-cancel", Button).press()
            await _wait_until(
                pilot,
                lambda: (
                    library._active_bundle_handle is None and not library._action_calls
                ),
            )
            assert handle not in bundle._sessions
            if cleanup_failure:
                task = library._bundle_invalidation_tasks[handle]
                assert task.done() and isinstance(task.exception(), RuntimeError)
                assert not await library._maintenance_drain(time.monotonic() + 0.05)
                assert library._bundle_invalidation_tasks[handle] is task
            else:
                assert not library._bundle_invalidation_tasks
                assert await library._maintenance_drain(time.monotonic() + 1)
            await library._maintenance_resume()
            assert source.read_bytes() == payload
    finally:
        await bundle.close()


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_library_review_pause_preserves_original_bundle_session_until_cancel(
    tmp_path, cleanup_failure
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_library_maintenance import _review_child
asyncio.run(_review_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(cleanup_failure),
    )


@pytest.mark.parametrize("mode", ["service_paused", "global_paused"])
def test_save_after_maintenance_refusal_retains_exact_submitted_draft(tmp_path, mode):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_library_maintenance import _dirty_child
asyncio.run(_dirty_child(Path(sys.argv[1]), sys.argv[2]))
""",
        mode,
    )


async def _namespace_child(root, mode="validation_parent"):
    import os
    import types
    from tldw_chatbook.UI import stts_profile_library as module
    from tldw_chatbook.UI.stts_profile_library import STTSProfileLibrary
    from tldw_chatbook.Utils import path_validation

    selected = root / "output"
    selected.mkdir()
    destination = selected / "export.json"
    if mode != "new":
        destination.write_text("selected original output")
    moved = root / "retained-original-output"
    reached = []

    def substitute():
        if mode.endswith("parent"):
            selected.rename(moved)
            selected.mkdir()
        else:
            moved.mkdir()
            destination.rename(moved / "export.json")
        destination.write_text("foreign replacement")
        reached.append(True)

    if mode == "validation_parent":
        original = path_validation.validate_path

        def validate(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path == destination:
                substitute()
            return result

        path_validation.validate_path = validate
    elif mode.startswith("prevalidation"):
        writer = STTSProfileLibrary._write_profile_export

        def write(*args, **kwargs):
            substitute()
            return writer(*args, **kwargs)

        STTSProfileLibrary._write_profile_export = staticmethod(write)
    elif mode.startswith("pretruncate"):
        module.os = types.SimpleNamespace(**vars(os))

        def truncate(fd, size):
            assert os.fstat(fd).st_ino == destination.stat().st_ino
            substitute()
            return os.ftruncate(fd, size)

        module.os.ftruncate = truncate
    library = STTSProfileLibrary(lambda: None)
    if mode == "new":
        await library._run_profile_export(destination, "exported private data")
        assert destination.read_text() == "exported private data"
    else:
        with pytest.raises(ValueError):
            await library._run_profile_export(destination, "exported private data")
        assert reached == [True]
        assert destination.read_text() == "foreign replacement"
        expected = (
            "exported private data"
            if mode.startswith("pretruncate")
            else "selected original output"
        )
        assert (moved / "export.json").read_text() == expected
    assert not library._export_operations


@pytest.mark.parametrize(
    "mode",
    [
        "validation_parent",
        "prevalidation_parent",
        "prevalidation_target",
        "pretruncate_parent",
        "pretruncate_target",
        "new",
    ],
)
def test_export_refuses_original_namespace_substitution_before_truncation(
    tmp_path, mode
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_library_maintenance import _namespace_child
asyncio.run(_namespace_child(Path(sys.argv[1]), sys.argv[2]))
""",
        mode,
    )


async def _late_page_child(root, failed, callback=False):
    import asyncio
    import time
    from textual.app import App
    from textual.widgets import DataTable
    from Tests.TTS.test_profile_service import _FakeTTSService
    from Tests.TTS.test_profile_repository_lifecycle import _draft
    from tldw_chatbook.TTS.profile_repository import (
        TTSProfileRepository,
        ProfileRepositoryError,
    )
    from tldw_chatbook.TTS.profile_service import TTSProfileService
    from tldw_chatbook.UI.stts_profile_library import STTSProfileLibrary

    repository = TTSProfileRepository(root / "profiles.sqlite")
    await repository.open()
    await repository.create_profile(_draft("Existing selection"))
    service = TTSProfileService(repository, _FakeTTSService())

    class Host(App):
        async def load(self):
            return service

        def compose(self):
            yield STTSProfileLibrary(self.load)

    entered, release = asyncio.Event(), asyncio.Event()
    original = repository.list_profiles

    async def page(*args, **kwargs):
        result = await original(*args, **kwargs)
        entered.set()
        await release.wait()
        if failed:
            raise ProfileRepositoryError("unavailable")
        return result

    app = Host()
    try:
        async with app.run_test():
            library = app.query_one(STTSProfileLibrary)
            await _until(
                lambda: bool(library._loaded_rows) and library._active_page_task is None
            )
            table = library.query_one(DataTable)
            table.move_cursor(row=0)
            table.action_select_cursor()
            await _until(lambda: library._selected_profile is not None)
            selected = library._selected_profile
            rows = tuple(library._loaded_rows.items())
            repository.list_profiles = page
            library._queue_page_request(None, 0)
            await asyncio.wait_for(entered.wait(), 2)
            assert not await library._maintenance_drain(time.monotonic() + 0.02)
            if callback:
                from tldw_chatbook.TTS.profile_service import TTSProfileAvailability

                previous = dict(library._row_availability)
                library.publish_profile_test_availability(
                    selected,
                    TTSProfileAvailability(
                        profile_id=selected.profile.profile_id,
                        state="unavailable",
                        recovery_action="edit",
                    ),
                )
                assert library._row_availability == previous, (
                    "paused Library accepted late sample UI publication"
                )
            release.set()
            await _until(lambda: library._active_page_task is None)
            assert library._selected_profile is selected
            assert tuple(library._loaded_rows.items()) == rows
            repository.list_profiles = original
            assert await library._maintenance_drain(time.monotonic() + 0.2)
            await library._maintenance_resume()
            await _until(lambda: library._active_page_task is None)
            assert library._loaded_rows
    finally:
        release.set()
        await repository.close()


@pytest.mark.parametrize("failed", [False, True])
def test_late_original_page_result_cannot_mutate_paused_selection(tmp_path, failed):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_library_maintenance import _late_page_child
asyncio.run(_late_page_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(failed),
    )


@pytest.mark.parametrize(
    "mode",
    ["parent_fd_before", "parent_fd_after", "target_fd_before", "target_fd_after"],
)
def test_export_selected_descriptor_close_retains_native_exclusion(tmp_path, mode):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_library_maintenance import _export_close_child
asyncio.run(_export_close_child(Path(sys.argv[1]), sys.argv[2]))
""",
        mode,
    )


async def _export_fd_allocation_child(root, resource, late):
    import os
    import time
    import types
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.UI import stts_profile_library as module

    selected = root / "output"
    selected.mkdir()
    destination = selected / "export.json"
    destination.write_bytes(b"original output")
    ordinary = storage.acquire_storage(destination)
    hold = next(iter(storage._holds.values()))
    ordinary.close()
    reached = []

    def matches(path):
        return path == selected if resource == "parent" else path == destination.name

    if late:
        original = module._open_profile_export_descriptor

        def opened(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if matches(path):
                reached.append(result)
                raise OSError("after observed selected descriptor allocation")
            return result

        module._open_profile_export_descriptor = opened
    else:
        module.os = types.SimpleNamespace(**vars(os))

        def opened(path, *args, **kwargs):
            result = os.open(path, *args, **kwargs)
            if matches(path):
                reached.append(result)
                raise OSError("unreturned selected descriptor allocation")
            return result

        module.os.open = opened
    library = module.STTSProfileLibrary(lambda: None)
    with pytest.raises(OSError):
        await library._run_profile_export(destination, "new output")
    assert len(reached) == 1
    assert destination.read_bytes() == b"original output"
    assert bool(library._export_operations) is not late
    assert all(
        lease in storage._startups.values()
        or getattr(lease, "native_owner", None) in library._export_operations
        for lease in storage._live_leases
    )
    storage._shutdown()
    pause = storage._begin_local_pause()
    assert pause.drain(time.monotonic() + 0.02) is late
    assert _probe(hold.authority.control_root, hold.names) == (
        "entered" if late else "blocked"
    )


@pytest.mark.parametrize("resource", ["parent", "target"])
@pytest.mark.parametrize("late", [False, True])
def test_export_new_descriptor_allocation_outcomes_are_retained(
    tmp_path, resource, late
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_library_maintenance import _export_fd_allocation_child
asyncio.run(_export_fd_allocation_child(Path(sys.argv[1]), sys.argv[2], sys.argv[3] == 'True'))
""",
        resource,
        str(late),
    )


async def _fifo_child(root):
    import asyncio
    import os
    import stat
    import threading
    from tldw_chatbook.UI import stts_profile_library as module

    destination = root / "export.json"
    destination.write_bytes(b"original output")
    original_file = root / "retained.json"
    original = module._open_profile_export_descriptor
    entered = threading.Event()

    def opened(path, *args, **kwargs):
        if path == destination.name:
            destination.rename(original_file)
            os.mkfifo(destination)
            entered.set()
        return original(path, *args, **kwargs)

    module._open_profile_export_descriptor = opened
    library = module.STTSProfileLibrary(lambda: None)
    action = asyncio.create_task(
        library._run_profile_export(destination, "private bytes")
    )
    await _until(entered.is_set)
    await asyncio.sleep(0.03)
    try:
        assert action.done(), (
            "replacement FIFO blocked selected target open without a reader"
        )
    finally:
        # Unblock the unfixed writer only; this descriptor belongs to the fixture.
        reader = os.open(destination, os.O_RDONLY | os.O_NONBLOCK)
        try:
            with pytest.raises((OSError, ValueError)):
                await asyncio.wait_for(action, 2)
            assert os.read(reader, 32) == b""
        finally:
            os.close(reader)
    assert stat.S_ISFIFO(destination.stat().st_mode)
    assert original_file.read_bytes() == b"original output"
    assert not library._export_operations


def test_export_target_fifo_substitution_refuses_without_waiting_for_reader(tmp_path):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_library_maintenance import _fifo_child
asyncio.run(_fifo_child(Path(sys.argv[1])))
""",
    )


def test_paused_library_defers_late_sample_availability_publication(tmp_path):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_library_maintenance import _late_page_child
asyncio.run(_late_page_child(Path(sys.argv[1]), False, True))
""",
    )


def test_ordinary_export_preserves_unqualified_path_stream_compatibility(tmp_path):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_library_maintenance import _export_close_child
asyncio.run(_export_close_child(Path(sys.argv[1]), "portable"))
""",
    )
