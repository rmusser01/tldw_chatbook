"""Tests for the TASK-18310 resume-time reconcile seam.

Every IN-Console workspace-activation path (the Alt+W switcher, the shared
create modal's Console handler, and conversation-browser row-open) calls
``set_active_workspace`` and ``_activate_console_session_for_workspace``
together, so the registry and the Console chat store's active session can
never drift apart from any of those paths. Cross-screen activation --
Settings' create-modal ``_done``, Library's ``create_local_workspace``
``_done``, and Settings' "Set active" button (Qodo finding 5 on PR #1809) --
only calls ``set_active_workspace`` on the registry, so a stale Console
session can only arise from a cross-screen change. ``_reconcile_console_
session_with_registry`` repairs exactly that gap on every Console resume.
"""

from types import SimpleNamespace

from tldw_chatbook.Chat.console_chat_models import CONSOLE_GLOBAL_WORKSPACE_ID
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession, ConsoleChatStore
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController
from tldw_chatbook.Workspaces.models import DEFAULT_WORKSPACE_ID
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


class _FakeStore:
    """Exposes only the seams the reconcile reads.

    Mirrors `ConsoleChatStore`'s contract for the active-session lookup:
    `ensure_session()` is a pure dict hit when `active_session_id` is set
    (never creates in that case) and raises `KeyError` on a stale id --
    the reconcile treats that as divergent, same as the old linear scan.
    """

    def __init__(self, active_session_id, sessions):
        self.active_session_id = active_session_id
        self._sessions = sessions

    def sessions(self):
        return self._sessions

    def ensure_session(self):
        for session in self._sessions:
            if session.id == self.active_session_id:
                return session
        raise KeyError(self.active_session_id)


class _Stub:
    """House stub pattern (see test_console_workspace_create_handler.py):
    an unbound method is invoked with this as `self`, exposing exactly the
    seams the method under test reads. Never name a harness attr `_registry`.
    """

    def __init__(self, registry, store):
        self.calls = []
        self.app_instance = SimpleNamespace(workspace_registry_service=registry)
        self._store = store

    def _ensure_console_chat_store(self):
        return self._store

    def _sync_console_chat_core_state(self):
        self.calls.append("core")

    def _activate_console_session_for_workspace(self, workspace_id):
        self.calls.append(f"activate:{workspace_id}")

    def _sync_console_workspace_context(self):
        self.calls.append("context")

    def _sync_native_console_chat_ui(self):
        return "ui-sync-sentinel"

    def run_worker(self, work, **kw):
        self.calls.append(f"worker:{work}")


def _registry(tmp_path, *, active_id=None):
    db = WorkspaceDB(tmp_path / "ws.sqlite", client_id="console-reconcile-tests")
    service = LocalWorkspaceRegistryService(db)
    service.create_workspace(workspace_id="workspace-a", name="Workspace A")
    service.create_workspace(workspace_id="workspace-b", name="Workspace B")
    if active_id:
        service.set_active_workspace(active_id)
    return service


def test_aligned_session_is_a_cheap_noop(tmp_path):
    """Session already belongs to the registry's active workspace -> no sync calls."""
    registry = _registry(tmp_path, active_id="workspace-a")
    store = _FakeStore(
        active_session_id="session-a",
        sessions=[ConsoleChatSession(id="session-a", workspace_id="workspace-a")],
    )
    stub = _Stub(registry, store)
    ConsoleWorkspaceController._reconcile_console_session_with_registry(stub)
    assert stub.calls == []


def test_cross_screen_change_runs_full_sequence_in_order(tmp_path):
    """A registry-only activation (simulating Settings/Library) is repaired
    by running the same four-step sequence the create handler uses, in order.
    """
    registry = _registry(tmp_path, active_id="workspace-a")
    # Simulate the cross-screen change: another surface activates workspace B
    # via the registry alone (no Console-side session sync).
    registry.set_active_workspace("workspace-b")
    store = _FakeStore(
        active_session_id="session-a",
        sessions=[ConsoleChatSession(id="session-a", workspace_id="workspace-a")],
    )
    stub = _Stub(registry, store)
    ConsoleWorkspaceController._reconcile_console_session_with_registry(stub)
    assert stub.calls == [
        "core",
        "activate:workspace-b",
        "context",
        "worker:ui-sync-sentinel",
    ]


def test_no_active_session_activates_registry_workspace(tmp_path):
    """A fresh Console uses the registry-active workspace for its first tab."""
    registry = _registry(tmp_path, active_id="workspace-a")
    store = _FakeStore(active_session_id=None, sessions=[])
    stub = _Stub(registry, store)
    ConsoleWorkspaceController._reconcile_console_session_with_registry(stub)
    assert stub.calls == [
        "core",
        "activate:workspace-a",
        "context",
        "worker:ui-sync-sentinel",
    ]


def test_no_registry_service_is_a_noop(tmp_path):
    """No registry service wired yet -- return quietly, no raise."""
    store = _FakeStore(
        active_session_id="session-a",
        sessions=[ConsoleChatSession(id="session-a", workspace_id="workspace-a")],
    )
    stub = _Stub(None, store)
    ConsoleWorkspaceController._reconcile_console_session_with_registry(stub)
    assert stub.calls == []


def test_registry_raising_get_active_workspace_is_a_noop(tmp_path):
    """A raising registry must never break screen resume."""

    class _RaisingRegistry:
        def get_active_workspace(self):
            raise RuntimeError("registry unavailable")

    store = _FakeStore(
        active_session_id="session-a",
        sessions=[ConsoleChatSession(id="session-a", workspace_id="workspace-a")],
    )
    stub = _Stub(_RaisingRegistry(), store)
    ConsoleWorkspaceController._reconcile_console_session_with_registry(stub)
    assert stub.calls == []


def test_global_session_aligned_with_registry_default_is_a_noop(tmp_path):
    """Regression (found live by Tests/UI/test_console_session_settings.py):
    a session's default/unset `workspace_id` (`CONSOLE_GLOBAL_WORKSPACE_ID`,
    or "") and the registry's built-in Default workspace row
    (`DEFAULT_WORKSPACE_ID`) are THE SAME state on two layers (task-15120),
    not a divergence -- comparing them raw tore down every ordinary mounted
    session (whose workspace_id defaults to "global") the instant the
    registry's active workspace was the ordinary resting Default row.
    """
    db = WorkspaceDB(tmp_path / "ws.sqlite", client_id="console-reconcile-tests")
    registry = LocalWorkspaceRegistryService(db)
    registry.ensure_default_workspace()
    assert registry.get_active_workspace().workspace_id == DEFAULT_WORKSPACE_ID

    store = _FakeStore(
        active_session_id="session-global",
        sessions=[
            ConsoleChatSession(
                id="session-global", workspace_id=CONSOLE_GLOBAL_WORKSPACE_ID
            )
        ],
    )
    stub = _Stub(registry, store)
    ConsoleWorkspaceController._reconcile_console_session_with_registry(stub)
    assert stub.calls == []


def test_no_active_workspace_is_a_noop(tmp_path):
    """Registry present but nothing active (e.g. a global conversation) -- skip."""
    registry = _registry(tmp_path, active_id=None)
    store = _FakeStore(
        active_session_id="session-a",
        sessions=[ConsoleChatSession(id="session-a", workspace_id="workspace-a")],
    )
    stub = _Stub(registry, store)
    ConsoleWorkspaceController._reconcile_console_session_with_registry(stub)
    assert stub.calls == []


class _RealActivateStub(_Stub):
    """Runs the REAL `_activate_console_session_for_workspace` body against a
    real `ConsoleChatStore`, stubbing only the screen-sync seams (the parts
    that touch the mounted UI, which this unit test has none of).
    """

    def __init__(self, registry, store):
        super().__init__(registry, store)

    def _activate_console_session_for_workspace(self, workspace_id):
        ConsoleWorkspaceController._activate_console_session_for_workspace(
            self, workspace_id
        )

    def _capture_console_draft_switch_snapshot(self):
        pass

    def _sync_console_temporary_chip(self):
        pass

    def _console_workspace_session_title(self, workspace_id):
        return f"{workspace_id} Chat"

    def _default_console_session_settings(self):
        return None


class _RealStoreCoreStub(_RealActivateStub):
    """Make core sync exercise the real store's active-session invariant."""

    def _sync_console_chat_core_state(self):
        self._store.ensure_session()
        self.calls.append("core")


def test_stale_active_session_id_is_repaired_before_core_sync(tmp_path):
    """A stale identity cannot abort registry reconciliation before repair."""
    registry = _registry(tmp_path, active_id="workspace-b")
    store = ConsoleChatStore()
    store.create_session(title="Workspace A", workspace_id="workspace-a")
    store.active_session_id = "missing-session"
    stub = _RealStoreCoreStub(registry, store)

    ConsoleWorkspaceController._reconcile_console_session_with_registry(stub)

    active_session = store.ensure_session()
    assert active_session.workspace_id == "workspace-b"
    assert stub.calls == ["core", "context", "worker:ui-sync-sentinel"]


def test_end_to_end_library_style_cross_screen_activation_switches_session(tmp_path):
    """AC#4: activating a workspace from a non-Console surface (a plain
    registry call, exactly what Library's `create_local_workspace` `_done`
    and Settings' create-modal/"Set active" do) leaves the Console session on
    the OLD workspace until this reconcile runs -- as resume would run it --
    at which point the store's active session belongs to the new workspace.
    """
    registry = _registry(tmp_path, active_id="workspace-a")
    store = ConsoleChatStore()
    store.create_session(
        title="Workspace A Chat", workspace_id="workspace-a", settings=None
    )
    assert store.active_session_id is not None
    original_active_session = next(
        s for s in store.sessions() if s.id == store.active_session_id
    )
    assert original_active_session.workspace_id == "workspace-a"

    # "Another surface" (Library/Settings) activates workspace B via the
    # registry alone -- no Console involvement at all.
    registry.set_active_workspace("workspace-b")

    stub = _RealActivateStub(registry, store)
    ConsoleWorkspaceController._reconcile_console_session_with_registry(stub)

    active_session = next(
        s for s in store.sessions() if s.id == store.active_session_id
    )
    assert active_session.workspace_id == "workspace-b"
