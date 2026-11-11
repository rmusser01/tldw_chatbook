"""TASK-33620.15.1.5: one helper owns a session's consent-admitted Change Review roots.

Three readers ask for a session's Change Review roots: the turn
configuration builder, the hook authority check (which compares its answer
with the frozen configuration) and the Environment panel / Change Review
accessor. These drive a mounted Console with original consent and verify
that passive capture and live admission agree on the owning workspace's
ready roots as strings.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from contextlib import asynccontextmanager, contextmanager
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

import tldw_chatbook.Chat.console_turn_context as turn_context
from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.config import get_user_data_dir

# Real bindings and consent writes belong to one privately selected profile.
pytestmark = pytest.mark.ui

_WORKSPACE = "ws-admitted-roots"


@contextmanager
def _original_consent_reads(consent, observe_read=None):
    """Observe original readers without replacing their guarded methods."""
    methods = {
        consent.capture_turn.__func__.__code__: "capture_turn",
        consent.admit_turn.__func__.__code__: "admit_turn",
        consent._admit.__func__.__code__: "_admit",
    }
    reads = []
    previous_main, previous_threads = sys.getprofile(), threading.getprofile()

    def observe(frame, event, result):
        if (
            event == "return"
            and frame.f_code in methods
            and frame.f_locals.get("self") is consent
        ):
            read = (
                methods[frame.f_code],
                frame.f_locals["workspace_id"],
                result,
                threading.current_thread(),
            )
            reads.append(read)
            if observe_read is not None:
                observe_read(*read)

    threading.setprofile_all_threads(observe)
    try:
        yield reads
    finally:
        threading.setprofile_all_threads(previous_threads)
        sys.setprofile(previous_main)


@asynccontextmanager
async def _console(roots, monkeypatch):
    monkeypatch.setenv("TLDW_CHANGE_REVIEW_ENABLED", "1")
    app = _build_test_app(user_data_dir=get_user_data_dir())
    _configure_native_ready_console(app)
    registry = app.workspace_registry_service
    consent = app.change_review_consent_service
    registry.create_workspace(workspace_id=_WORKSPACE, name="Admitted roots")
    registry.set_change_review_enabled(_WORKSPACE, True)
    registry.set_active_workspace(_WORKSPACE)
    initialized = []
    async with asyncio.timeout(5):
        for root in roots:
            root.mkdir(parents=True, exist_ok=True)
            registry.add_folder_binding(_WORKSPACE, root)
            consent.admit_turn(_WORKSPACE)
            # task_done follows the native ticket's physical return; do not let
            # the next opaque initializer contend with this original job.
            await asyncio.to_thread(consent._queue.join)
            initialized.append(str(root.resolve()))
            expected = tuple(
                binding.locator for binding in registry.list_folder_bindings(_WORKSPACE)
            )
            assert set(expected) == set(initialized)
            assert consent.capture_turn(_WORKSPACE).ready_roots == expected
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 45)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        await pilot.pause(0.2)
        controller = screen._ensure_console_chat_controller()
        assert controller._turn_context_provider is None, (
            "harness precondition: the runtime-owned controller"
        )
        session_id = controller.store.active_session_id
        assert session_id, "harness precondition: an active session"
        assert controller.store.session_workspace_id(session_id) == _WORKSPACE
        assert consent.capture_turn(_WORKSPACE).ready_roots == expected
        yield pilot, screen, controller, session_id


def _three_readers(screen, controller, session_id) -> dict[str, tuple]:
    return {
        "snapshot builder": controller.resolve_turn_configuration_snapshot(
            session_id
        ).workspace_roots,
        "hook authority": controller._hook_authority_values(session_id)[
            "workspace_roots"
        ],
        "Environment / Change Review accessor": (
            screen._review_selection._console_change_review_workspace_roots()
        ),
    }


@pytest.mark.asyncio
@private_profile_test
async def test_the_three_readers_return_the_same_string_roots(
    request, monkeypatch, tmp_path
):
    """Metadata and narrow admission report the same ready roots as strings."""
    roots = (tmp_path / "first", tmp_path / "second")
    async with _console(roots, monkeypatch) as (_pilot, screen, controller, session_id):
        readers = _three_readers(screen, controller, session_id)
        expected = tuple(
            binding.locator
            for binding in controller.app.workspace_registry_service.list_folder_bindings(
                controller.store.session_workspace_id(session_id)
            )
        )
    assert set(expected) == {str(root.resolve()) for root in roots}
    assert readers == dict.fromkeys(readers, expected)
    assert all(type(root) is str for values in readers.values() for root in values)


@pytest.mark.asyncio
@private_profile_test
async def test_the_three_readers_answer_from_one_helper(request, monkeypatch, tmp_path):
    """Each reader reaches the real consent owner with the owning workspace."""
    async with _console((tmp_path / "repo",), monkeypatch) as (
        _pilot,
        screen,
        controller,
        session_id,
    ):
        own_workspace = controller.store.session_workspace_id(session_id)
        consent = controller.app.change_review_consent_service
        with _original_consent_reads(consent) as reads:
            readers = _three_readers(screen, controller, session_id)
        shared_reads = [read for read in reads if read[0] == "_admit"]
        assert [workspace for _, workspace, *_ in shared_reads] == [own_workspace] * 3
        readers_called = [name for name, *_ in reads if name != "_admit"]
        assert readers_called == [
            "capture_turn",
            "admit_turn",
            "admit_turn",
        ]
        expected = (str((tmp_path / "repo").resolve()),)
        assert readers == dict.fromkeys(readers, expected)
        assert all(result.ready_roots == expected for _, _, result, _ in reads)


def test_narrow_admission_normalizes_paths_without_granting_metadata_authority(
    tmp_path,
):
    """Custom narrow-reader results cannot become original capture authority."""
    root = PurePosixPath(tmp_path)
    consent = SimpleNamespace(
        admit_turn=lambda _workspace: SimpleNamespace(
            ready_roots=(root, root / "second"),
            ready_aliases=("one", "two"),
            skipped_roots=(),
        )
    )
    app = SimpleNamespace(change_review_consent_service=consent)
    assert turn_context.capture_change_review_admission(app, "workspace") == (
        (str(root), str(root / "second")),
        ("one", "two"),
        (),
    )
    assert turn_context.capture_change_review_metadata(app, "workspace") == ((), (), ())
