"""Ownership and LIFETIME pins for the app-owned Console runtime (task-15860).

`test_console_runtime_is_the_single_construction_site` pins the ownership
move: if a later change re-adds a `ConsoleChatStore(...)` or
`ConsoleChatController(...)` anywhere but `Chat/console_runtime.py`, the app
stops being the owner and Design A quietly loses its premise.

The rest pin the lifetime landing. `test_second_console_visit_reuses_the_
runtime` **replaces** Task 1's `test_second_console_visit_gets_a_new_
runtime`, which asserted the opposite and whose docstring said it must be
rewritten here rather than deleted -- that is the whole reason it existed.

`test_a_terminal_run_state_after_leaving_does_not_reach_the_dead_screen` is
this landing's defect red for the attach/detach seam. Before the seam
existed, every screen-owned hook slot stayed bound to the unmounted
`ChatScreen` (Task 0's P3 measured five of them still bound and none
raising), so a run settling after the navigation called straight into a
dead view.
"""

from __future__ import annotations

import ast
import asyncio
from dataclasses import fields
import gc
import inspect
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
from textual.events import Key

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _wait_for_selector
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleProviderSelection,
    ConsoleRunState,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_runtime import (
    CONSOLE_RUNTIME_ATTR,
    CONSOLE_VIEW_HOOK_SLOTS,
    ConsoleRuntime,
)
from tldw_chatbook.Chat.console_turn_context import (
    ConsoleTurnConfigurationSnapshot,
    ConsoleTurnCustodyRequest,
)
from tldw_chatbook.Chat.console_scratch_space import ConsoleScratchSpaceManager
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.Persona_Buddy.console_adapter import PersonaBuddyConsoleAdapter
from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController
from tldw_chatbook.UI.Console_Modules.fleet import (
    ConsoleFleetLifecycleController,
)
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_composer_bar import (
    ConsoleComposerBar,
    classify_console_raw_draft,
)
from tldw_chatbook.Widgets.Console import ProjectInstructionSetupResult

#: Constructor calls that must exist in exactly one place: the runtime.
#: `ConsoleProviderGateway(` is deliberately NOT here -- the Personas
#: preview controller builds its own, unrelated to the Console runtime
#: (`UI/Persona_Modules/personas_preview_controller.py`).
_RUNTIME_OWNED_CONSTRUCTIONS = ("ConsoleChatStore", "ConsoleChatController")

_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "tldw_chatbook"
_RUNTIME_MODULE = "tldw_chatbook/Chat/console_runtime.py"

_VIEW_HOOK_OWNERSHIP = {
    "_chat_dictionary_applier": "app-owned-domain",
    "_world_info_applier": "app-owned-domain",
    "_rag_capture_provider": "runtime-custody",
    "_default_session_settings": "app-owned-domain",
    "_library_provider_factory": "app-owned-domain",
    "_global_user_display_name": "app-owned-domain",
    "_turn_context_provider": "frozen-request",
    "on_submission_accepted": "runtime-custody",
    "on_queued_submission_accepted": "app-owned-domain",
    "prompt_history": "app-owned-domain",
    "set_pending_approval": "view-projection",
    "set_pending_decision": "runtime-projection-router",
    "park_pending_approval": "view-projection",
    "notify_run_outcome": "view-projection",
    "notify_run_failure": "view-projection",
    "follow_watchlists_operations": "view-projection",
    "set_task_panel": "view-projection",
    "set_pending_question": "view-projection",
    "update_pending_approval_summary": "view-projection",
    "set_pending_skill_install": "runtime-projection-router",
    "set_pending_skill_script": "runtime-projection-router",
    "wake_user_priority_probe": "view-projection",
    "wake_conversation_in_view": "view-projection",
    "on_scope_flushed": "view-projection",
    "delivery_ui_hook": "view-projection",
    "project_project_instruction_binding": "runtime-projection-router",
    "project_project_instruction_dispatch": "runtime-projection-router",
    "dismiss_project_instruction_decision": "runtime-projection-router",
}


@pytest.mark.unit
def test_console_runtime_owns_one_screen_free_persona_buddy_sink():
    """The app-owned runtime, not a screen, retains the trusted sink."""
    app = type("App", (), {})()
    app.persona_buddy_controller = PersonaBuddyController()
    runtime = ConsoleRuntime(app)

    assert isinstance(runtime.persona_buddy_sink, PersonaBuddyConsoleAdapter)
    assert runtime.persona_buddy_sink is runtime.persona_buddy_sink
    assert "view" not in vars(runtime.persona_buddy_sink)


@pytest.mark.unit
def test_console_runtime_reuses_one_scratch_manager_across_console_visits():
    runtime = ConsoleRuntime(type("App", (), {})())
    first = runtime.scratch_spaces

    runtime.detach_view(None)

    assert runtime.scratch_spaces is first


@pytest.mark.unit
def test_runtime_owned_custody_tracks_only_lifetime_handles():
    """Custody is not a second controller run-state or queue store."""
    runtime = ConsoleRuntime(type("App", (), {})())

    request = ConsoleTurnCustodyRequest(
        turn_id="turn-ownership",
        session_id="session-ownership",
        draft="draft",
        configuration=ConsoleTurnConfigurationSnapshot.capture(
            session_id="session-ownership",
            provider_selection=ConsoleProviderSelection(provider="openai"),
        ),
    )
    record = runtime._register_custody(request)

    assert {field.name for field in fields(ConsoleTurnCustodyRequest)} == {
        "turn_id",
        "session_id",
        "draft",
        "configuration",
        "attachment_ids",
        "staged_evidence_launch",
        "one_shot_prefill",
        "one_shot_prefill_revision",
    }
    assert {field.name for field in fields(type(record))} == {
        "turn_id",
        "session_id",
        "request",
        "inputs",
        "task",
    }
    assert {field.name for field in fields(type(record.inputs))} == {
        "attachments",
        "staged_evidence_revision",
        "durable_accepted",
    }
    assert not {
        "run_state",
        "queue_state",
        "terminal_state",
        "stream_state",
    } & set(vars(runtime))


def test_runtime_staged_evidence_release_is_exactly_revision_fenced() -> None:
    """A durable A cannot clear a newer launch B staged during capture."""
    runtime = ConsoleRuntime(type("App", (), {})())
    launch_a = ConsoleLiveWorkLaunch.from_values(
        source="A", title="A", payload={"source": "A"}
    )
    launch_b = ConsoleLiveWorkLaunch.from_values(
        source="B", title="B", payload={"source": "B"}
    )

    revision_a = runtime.stage_console_staged_evidence(launch_a)
    captured, captured_revision, _notice = runtime.snapshot_console_staged_evidence()
    assert captured is launch_a
    assert captured_revision == revision_a

    runtime.stage_console_staged_evidence(launch_b)
    released = runtime.release_console_staged_evidence(
        launch_a,
        revision=revision_a,
        result=SimpleNamespace(context="captured context"),
    )

    current, _revision, _notice = runtime.snapshot_console_staged_evidence()
    assert released is False
    assert current is launch_b


def test_runtime_restore_staged_evidence_never_overwrites_a_newer_launch() -> None:
    """A navigation payload restores only when its revision is still current."""
    runtime = ConsoleRuntime(type("App", (), {})())
    launch_a = ConsoleLiveWorkLaunch.from_values(
        source="A", title="A", payload={"source": "A"}
    )
    launch_b = ConsoleLiveWorkLaunch.from_values(
        source="B", title="B", payload={"source": "B"}
    )

    revision_a = runtime.stage_console_staged_evidence(launch_a)
    runtime.stage_console_staged_evidence(launch_b)

    assert (
        runtime.restore_console_staged_evidence(
            launch_a, revision=revision_a, sent_source_count=None
        )
        is False
    )
    assert runtime.snapshot_console_staged_evidence()[0] is launch_b


@pytest.mark.asyncio
async def test_successful_viewless_manual_turn_records_first_send(monkeypatch) -> None:
    """Terminal success records onboarding without retaining a ChatScreen."""
    saved: list[tuple[str, str, bool]] = []
    monkeypatch.setattr(
        "tldw_chatbook.config.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    app = type("App", (), {"app_config": {}})()
    runtime = ConsoleRuntime(app)
    store = ConsoleChatStore()
    session = store.ensure_session()
    runtime.set_chat_store(store)

    class SuccessfulController:
        async def run_prompt_chain(self, *, session_id, initial_turn):
            assert session_id == session.id
            return await initial_turn()

        async def submit_draft(self, draft, **kwargs):
            assert draft == "first successful turn"
            kwargs["custody_acceptance_hook"]()
            return SimpleNamespace(accepted=True)

        def run_state_for(self, session_id):
            assert session_id == session.id
            return ConsoleRunState(ConsoleRunStatus.COMPLETED, "done")

    runtime.set_chat_controller(SuccessfulController())
    request = ConsoleTurnCustodyRequest(
        turn_id="turn-first-send",
        session_id=session.id,
        draft="first successful turn",
        configuration=ConsoleTurnConfigurationSnapshot.capture(
            session_id=session.id,
            provider_selection=ConsoleProviderSelection(provider="openai"),
        ),
    )

    turn_id = runtime.accept_turn(request)
    await runtime.wait_for_turn(turn_id)

    assert runtime.view is None
    assert app.app_config["console"]["onboarding"]["first_send_completed"] is True
    assert saved == [("console.onboarding", "first_send_completed", True)]


@pytest.mark.asyncio
async def test_runtime_injects_its_scratch_manager_into_chat_controller():
    runtime = ConsoleRuntime(type("App", (), {})())

    controller = runtime.ensure_chat_controller(
        store=ConsoleChatStore(),
        provider_gateway=object(),
    )

    assert controller._scratch_spaces is runtime.scratch_spaces
    assert controller._owns_scratch_spaces is False
    await runtime.dispose()


@pytest.mark.asyncio
async def test_runtime_owns_one_receipt_service_and_coalesces_hydration(
    tmp_path, monkeypatch
):
    app = SimpleNamespace(
        chachanotes_db=SimpleNamespace(db_path=tmp_path / "chat.db"),
        conversation_local_marks_service=None,
    )
    runtime = ConsoleRuntime(app)
    bridge = runtime.ensure_agent_bridge(
        store_factory=ConsoleChatStore,
        provider_gateway_factory=object,
    )

    assert bridge is not None
    assert runtime.activity_receipts is not None
    assert bridge.runs_db is runtime._agent_runs_db
    assert runtime.profile_authority == str((tmp_path / "chat.db").resolve())
    assert runtime.authority_token

    entered = threading.Event()
    release = threading.Event()
    calls = {"count": 0}

    def blocked_hydration():
        calls["count"] += 1
        entered.set()
        assert release.wait(5)
        return 0

    monkeypatch.setattr(
        runtime.activity_receipts, "hydrate_from_storage", blocked_hydration
    )
    first = runtime.ensure_activity_hydration()
    second = runtime.ensure_activity_hydration()

    assert first is second
    assert await asyncio.to_thread(entered.wait, 5)
    release.set()
    assert await first == 0
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_future_session_defaults_read_fresh_app_owned_provider_and_model():
    app = SimpleNamespace(
        app_config={
            "chat_defaults": {"provider": "anthropic", "model": "old-model"},
            "api_settings": {"anthropic": {}},
        }
    )
    runtime = ConsoleRuntime(app)
    controller = runtime.ensure_chat_controller(
        store=ConsoleChatStore(),
        provider_gateway=object(),
        provider="anthropic",
        model="old-model",
    )

    app.app_config = {
        "chat_defaults": {"provider": "openai", "model": "new-model"},
        "api_settings": {"openai": {}},
    }

    defaults = controller._default_session_settings()
    assert (defaults.provider, defaults.model) == ("openai", "new-model")
    await runtime.dispose()


@pytest.mark.asyncio
async def test_runtime_disposal_invalidates_inflight_receipt_hydration(
    tmp_path, monkeypatch
):
    app = SimpleNamespace(
        chachanotes_db=SimpleNamespace(db_path=tmp_path / "chat.db"),
        conversation_local_marks_service=None,
    )
    runtime = ConsoleRuntime(app)
    runtime.ensure_agent_bridge(
        store_factory=ConsoleChatStore,
        provider_gateway_factory=object,
    )
    entered = threading.Event()
    release = threading.Event()

    def blocked_hydration():
        entered.set()
        assert release.wait(5)
        return 4

    monkeypatch.setattr(
        runtime.activity_receipts, "hydrate_from_storage", blocked_hydration
    )
    task = runtime.ensure_activity_hydration()
    assert await asyncio.to_thread(entered.wait, 5)

    dispose = asyncio.create_task(runtime.dispose())
    release.set()
    await dispose

    assert task.cancelled() or await task == 0
    assert runtime.ensure_activity_hydration() is None


@pytest.mark.asyncio
async def test_leaving_console_preserves_live_session_scratch(tmp_path):
    runtime = ConsoleRuntime(type("App", (), {})())
    runtime._scratch_spaces = ConsoleScratchSpaceManager(temp_parent=tmp_path)
    snapshot = runtime.scratch_spaces.snapshot("session-a")

    assert await runtime.leave_console() is True

    assert runtime.scratch_spaces.is_live(snapshot)
    assert snapshot.root.is_dir()
    await runtime.dispose()
    assert not snapshot.root.exists()


@pytest.mark.asyncio
async def test_raw_cli_refusal_bank_survives_leave_and_clears_on_dispose():
    runtime = ConsoleRuntime(type("App", (), {})())
    stash = object()
    bank = runtime.raw_cli_refusal_stash_bank
    bank["session-a"] = [stash]

    assert runtime.accepts_raw_cli_refusal_callbacks is True
    assert await runtime.leave_console() is True
    assert runtime.accepts_raw_cli_refusal_callbacks is True
    assert runtime.raw_cli_refusal_stash_bank is bank
    assert bank == {"session-a": [stash]}

    await runtime.dispose()
    assert runtime.accepts_raw_cli_refusal_callbacks is False
    assert bank == {}


@pytest.mark.asyncio
async def test_runtime_tombstones_before_shutdown_and_disposes_via_to_thread(
    monkeypatch,
):
    events: list[str] = []

    class ScratchSpaces:
        def tombstone_all(self) -> None:
            events.append("scratch-tombstone")

        def dispose(self) -> bool:
            events.append("scratch-dispose")
            return True

    class Controller:
        async def shutdown(self) -> None:
            events.append("controller-shutdown")

    class Gateway:
        async def aclose(self) -> None:
            events.append("gateway-close")

    async def fake_to_thread(function, *args, **kwargs):
        events.append("to-thread")
        return function(*args, **kwargs)

    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_runtime.asyncio.to_thread",
        fake_to_thread,
    )
    runtime = ConsoleRuntime(type("App", (), {})())
    runtime._scratch_spaces = ScratchSpaces()
    runtime._chat_controller = Controller()
    runtime._provider_gateway = Gateway()

    await runtime.dispose()

    assert events == [
        "scratch-tombstone",
        "controller-shutdown",
        "to-thread",
        "scratch-dispose",
        "gateway-close",
    ]


@pytest.mark.asyncio
async def test_persona_buddy_release_follows_controller_wake_disposal():
    """Runtime shutdown terminally fences wake producers before sink release."""
    events: list[str] = []

    class Sink:
        def dispose(self) -> None:
            events.append("sink-release")

    class Controller:
        async def shutdown(self) -> None:
            events.append("wake-dispose")

    runtime = ConsoleRuntime(type("App", (), {})())
    runtime._persona_buddy_sink = Sink()
    runtime._chat_controller = Controller()
    runtime._provider_gateway = None

    await runtime.dispose()

    assert events == ["wake-dispose", "sink-release"]


def _construction_sites(class_name: str) -> list[str]:
    """Every `<path>:<line>` in the shipped package CALLING `class_name(...)`.

    AST-based on purpose (PR #1648 follow-up): a raw substring scan counts
    innocuous mentions in comments, docstrings, and string literals as
    construction sites and false-fails the pin. Only actual call nodes
    count here.
    """
    sites: list[str] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, UnicodeDecodeError, SyntaxError):  # pragma: no cover
            continue
        rel = path.relative_to(_PACKAGE_ROOT.parent).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            called = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else None
            )
            if called == class_name:
                line = source.splitlines()[node.lineno - 1].strip()
                sites.append(f"{rel}:{node.lineno}: {line}")
    return sites


@pytest.mark.unit
def test_console_runtime_is_the_single_construction_site():
    """PIN (characterization): only the runtime builds the store/controller."""
    for token in _RUNTIME_OWNED_CONSTRUCTIONS:
        sites = _construction_sites(token)
        assert sites, f"{token} vanished entirely -- the pin is stale."
        foreign = [site for site in sites if not site.startswith(_RUNTIME_MODULE)]
        assert not foreign, (
            f"{token} is constructed outside {_RUNTIME_MODULE}; the app is no "
            "longer the Console runtime's owner:\n  " + "\n  ".join(foreign)
        )


@pytest.mark.unit
def test_attach_and_detach_cover_exactly_the_same_slot_set():
    """The ONE enumerated list runs in both directions.

    `ChatScreen.console_view_hooks()` is what `attach_view` sets and
    `CONSOLE_VIEW_HOOK_SLOTS` is what `detach_view` clears. A slot present
    in one and not the other is either bound and never cleared (a dead
    screen kept alive, answering questions about a view that is gone) or
    cleared and never bound (a live Console with a silently dead hook).
    """
    screen = ChatScreen.__new__(ChatScreen)

    def no_op(*_args, **_kwargs):
        return None

    screen._fleet = ConsoleFleetLifecycleController(
        **{
            name: no_op
            for name in inspect.signature(ConsoleFleetLifecycleController).parameters
        }
    )
    screen._library_activity = SimpleNamespace(build_provider=no_op)
    screen._session = SimpleNamespace(
        _project_project_instruction_binding=no_op,
        _project_project_instruction_dispatch=no_op,
        _dismiss_project_instruction_decision_projection=no_op,
    )
    declared = {slot.name for slot in CONSOLE_VIEW_HOOK_SLOTS}
    provided = set(ChatScreen.console_view_hooks(screen))

    runtime_routed = {
        name
        for name, ownership in _VIEW_HOOK_OWNERSHIP.items()
        if ownership == "runtime-projection-router"
    }
    assert provided == declared | runtime_routed, (
        "view projections must be either directly detachable or reached "
        "through a stable runtime router: "
        f"only in the view: {sorted(provided - declared - runtime_routed)}; "
        f"missing from the view: {sorted((declared | runtime_routed) - provided)}"
    )
    assert len(CONSOLE_VIEW_HOOK_SLOTS) == len(declared), "duplicate slot name"


def test_view_hook_inventory_contains_only_disposable_projections():
    """Every former hook is classified before navigation becomes detach-only."""
    declared = {slot.name for slot in CONSOLE_VIEW_HOOK_SLOTS}
    expected = {
        name
        for name, ownership in _VIEW_HOOK_OWNERSHIP.items()
        if ownership == "view-projection"
    }

    assert set(_VIEW_HOOK_OWNERSHIP) == {
        "_chat_dictionary_applier",
        "_world_info_applier",
        "_rag_capture_provider",
        "_default_session_settings",
        "_library_provider_factory",
        "_global_user_display_name",
        "_turn_context_provider",
        "on_submission_accepted",
        "on_queued_submission_accepted",
        "prompt_history",
        "set_pending_approval",
        "set_pending_decision",
        "park_pending_approval",
        "notify_run_outcome",
        "notify_run_failure",
        "follow_watchlists_operations",
        "set_task_panel",
        "set_pending_question",
        "update_pending_approval_summary",
        "set_pending_skill_install",
        "set_pending_skill_script",
        "wake_user_priority_probe",
        "wake_conversation_in_view",
        "on_scope_flushed",
        "delivery_ui_hook",
        "project_project_instruction_binding",
        "project_project_instruction_dispatch",
        "dismiss_project_instruction_decision",
    }
    assert declared == expected


def test_runtime_first_send_completion_has_no_direct_view_reachback():
    """First-send persistence is domain work; a later view reconciles it."""
    source = inspect.getsource(ConsoleRuntime._record_successful_first_send)

    assert "self.view" not in source
    assert "_sync_console_transcript_guidance" not in source


@pytest.mark.asyncio
async def test_unmount_detaches_before_later_view_cleanup_can_fail():
    """A broken view cleanup cannot leave its runtime projections attached."""
    runtime = ConsoleRuntime(None)
    screen = ChatScreen.__new__(ChatScreen)
    generation = 41
    runtime.view = screen
    runtime._attached_generation = generation
    screen._console_runtime = lambda: runtime
    screen._console_runtime_attachment_generation = generation
    screen._release_claimed_conversation_settings_return = lambda: None

    async def fail_cleanup() -> None:
        raise RuntimeError("injected view cleanup failure")

    screen._flush_sidebar_state_now = fail_cleanup

    with pytest.raises(RuntimeError, match="injected view cleanup failure"):
        await ChatScreen.on_unmount(screen)

    assert runtime.view is None
    assert runtime._attached_generation is None
    assert screen._console_runtime_attachment_generation == generation
    assert screen._console_runtime_attachment_retired is True


def test_detach_requires_the_exact_monotonic_attachment_generation():
    runtime = ConsoleRuntime(type("App", (), {})())
    first_view = SimpleNamespace(console_view_hooks=lambda: {})
    successor = SimpleNamespace(console_view_hooks=lambda: {})

    first_generation = runtime.attach_view(first_view)
    successor_generation = runtime.attach_view(successor)
    refreshed_generation = runtime.attach_view(
        successor, prior_generation=successor_generation
    )

    assert first_generation < successor_generation
    assert refreshed_generation == successor_generation
    assert runtime.detach_view(first_view, first_generation) is False
    assert runtime.detach_view(successor, successor_generation) is True


@pytest.mark.asyncio
async def test_late_outgoing_ensure_cannot_reclaim_a_successor_attachment():
    """A refused stale claim cannot run any outgoing screen-derived wiring."""
    runtime = ConsoleRuntime(SimpleNamespace())
    wire_calls: list[object] = []
    wake = SimpleNamespace(wire=lambda **kwargs: wire_calls.append(kwargs["app"]))
    runtime._chat_controller = SimpleNamespace(
        fleet_wake=wake,
        provider="successor-provider",
    )
    outgoing = ChatScreen.__new__(ChatScreen)
    successor = ChatScreen.__new__(ChatScreen)
    for screen in (outgoing, successor):
        screen._release_claimed_conversation_settings_return = lambda: None
        screen._console_runtime_ref = runtime
        screen.console_view_hooks = lambda: {}
        screen.app_instance = SimpleNamespace()
        screen._sync_console_chat_core_state = lambda: setattr(
            runtime._chat_controller, "provider", "outgoing-provider"
        )

    outgoing_generation = runtime.attach_view(outgoing)
    successor_generation = runtime.attach_view(successor)
    assert runtime.view is successor

    # A late callback on the outgoing screen reaches this ordinary wiring
    # seam after the successor has already claimed the runtime.
    assert outgoing._ensure_console_chat_controller() is runtime.chat_controller
    assert outgoing._console_runtime_attachment_generation == outgoing_generation
    assert runtime.view is successor
    assert runtime.chat_controller.provider == "successor-provider"
    assert wire_calls == []

    async def fail_cleanup() -> None:
        raise RuntimeError("stop after detach")

    outgoing._flush_sidebar_state_now = fail_cleanup
    with pytest.raises(RuntimeError, match="stop after detach"):
        await ChatScreen.on_unmount(outgoing)

    assert outgoing._console_runtime_attachment_generation == outgoing_generation
    assert outgoing._console_runtime_attachment_retired is True
    assert runtime.view is successor
    assert runtime._attached_generation == successor_generation

    # A callback after the outgoing view's token was retired is not a fresh
    # claim and still cannot run any screen-derived mutation.
    assert outgoing._ensure_console_chat_controller() is runtime.chat_controller
    assert runtime.view is successor
    assert runtime.chat_controller.provider == "successor-provider"
    assert wire_calls == []


@pytest.mark.asyncio
async def test_unmount_releases_settings_claim_before_blocked_sidebar_flush():
    released = []
    flushing = asyncio.Event()
    release_flush = asyncio.Event()

    async def flush():
        flushing.set()
        await release_flush.wait()
        raise RuntimeError("stop after flush")

    screen = SimpleNamespace(
        _release_claimed_conversation_settings_return=lambda: released.append("claim"),
        _console_runtime=lambda: SimpleNamespace(detach_view=lambda *_: None),
        _flush_sidebar_state_now=flush,
    )
    task = asyncio.create_task(ChatScreen.on_unmount(screen))
    try:
        await asyncio.wait_for(flushing.wait(), timeout=1)
        assert released == ["claim"]
    finally:
        release_flush.set()
        with pytest.raises(RuntimeError, match="stop after flush"):
            await task


def _project_selection(binding_id: str = "binding-a") -> tuple[SimpleNamespace, ...]:
    return (
        SimpleNamespace(
            binding=SimpleNamespace(
                binding_id=binding_id,
                display_name=f"Workspace {binding_id}",
            )
        ),
    )


class _WeakView:
    def __init__(self, hooks=None):
        self._hooks = hooks or {}

    def console_view_hooks(self):
        return self._hooks


@pytest.mark.asyncio
async def test_project_binding_selection_waits_runtime_owned_until_console_attaches():
    presented = asyncio.Event()
    decisions: list[tuple[str, tuple]] = []
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime._chat_store = SimpleNamespace(
        active_session_id="session-a",
        sessions=lambda: [SimpleNamespace(id="session-a")],
    )
    task = asyncio.create_task(
        runtime._select_project_instruction_binding(
            "session-a", _project_selection(), "binding_unavailable"
        )
    )
    await asyncio.sleep(0)
    assert not task.done(), "a detached Console must retain the decision"

    def project(decision_id, options):
        decisions.append((decision_id, options))
        presented.set()
        return True

    view = _WeakView({"project_project_instruction_binding": project})
    generation = runtime.attach_view(view)
    assert decisions == [], "decision projection must wait for full reconciliation"
    assert runtime.finish_view_reconciliation(view, generation)
    await asyncio.wait_for(presented.wait(), timeout=1)
    decision_id, options = decisions[0]
    assert options[0].binding_id == "binding-a"
    assert runtime.resolve_project_instruction_binding(
        decision_id, "select", "binding-a"
    )

    assert await asyncio.wait_for(task, timeout=1) == ("select", "binding-a")


@pytest.mark.asyncio
async def test_project_binding_selection_remounts_after_detach_without_retaining_view():
    first_presented = asyncio.Event()
    second_presented = asyncio.Event()
    decision_ids: list[str] = []
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime._chat_store = SimpleNamespace(
        active_session_id="session-a",
        sessions=lambda: [SimpleNamespace(id="session-a")],
    )
    view = _WeakView(
        {
            "project_project_instruction_binding": lambda decision_id, _options: (
                decision_ids.append(decision_id),
                first_presented.set(),
            )
            and True
        }
    )
    generation = runtime.attach_view(view)
    assert runtime.finish_view_reconciliation(view, generation)
    dead_view = weakref.ref(view)

    task = asyncio.create_task(
        runtime._select_project_instruction_binding(
            "session-a", _project_selection(), "binding_unavailable"
        )
    )
    await asyncio.wait_for(first_presented.wait(), timeout=1)
    assert runtime.detach_view(view, generation)
    del view
    gc.collect()
    assert dead_view() is None

    successor = _WeakView(
        {
            "project_project_instruction_binding": lambda decision_id, _options: (
                decision_ids.append(decision_id),
                second_presented.set(),
            )
            and True
        }
    )
    successor_generation = runtime.attach_view(successor)
    assert runtime.finish_view_reconciliation(successor, successor_generation)
    await asyncio.wait_for(second_presented.wait(), timeout=1)
    assert decision_ids[0] == decision_ids[1]
    assert runtime.resolve_project_instruction_binding(
        decision_ids[1], "select", "binding-a"
    )
    assert await asyncio.wait_for(task, timeout=1) == ("select", "binding-a")


@pytest.mark.asyncio
async def test_project_binding_selection_closes_when_owning_session_closes(
):
    sessions = [SimpleNamespace(id="session-a")]
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime._chat_store = SimpleNamespace(
        active_session_id="session-a", sessions=lambda: sessions
    )

    task = asyncio.create_task(
        runtime._select_project_instruction_binding(
            "session-a", _project_selection(), "binding_unavailable"
        )
    )
    await asyncio.sleep(0)
    sessions.clear()

    assert await asyncio.wait_for(task, timeout=1) == ("cancel", None)


@pytest.mark.asyncio
async def test_project_binding_selection_fails_closed_on_app_dispose():
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime._chat_store = SimpleNamespace(
        active_session_id="session-a",
        sessions=lambda: [SimpleNamespace(id="session-a")],
    )
    task = asyncio.create_task(
        runtime._select_project_instruction_binding(
            "session-a", _project_selection(), "binding_unavailable"
        )
    )
    await asyncio.sleep(0)
    await runtime.dispose()
    assert await asyncio.wait_for(task, timeout=1) == ("cancel", None)


def _project_notice_runtime() -> ConsoleRuntime:
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime._chat_store = SimpleNamespace(
        active_session_id="session-a",
        sessions=lambda: [SimpleNamespace(id="session-a")],
    )
    runtime._chat_controller = SimpleNamespace(_active_cancel_events={})
    return runtime


def test_project_dispatch_confirmation_waits_app_owned_when_detached_before_request():
    runtime = _project_notice_runtime()
    projected = threading.Event()
    decision_ids: list[str] = []
    result: list[str] = []
    worker = threading.Thread(
        target=lambda: result.append(
            runtime._confirm_project_instruction_dispatch(
                SimpleNamespace(session_id="session-a")
            )
        ),
        daemon=True,
    )
    worker.start()
    assert not projected.wait(0.05), "Library/Settings must not receive a Console modal"

    def project(decision_id, _notice):
        decision_ids.append(decision_id)
        projected.set()
        return True

    view = _WeakView({"project_project_instruction_dispatch": project})
    generation = runtime.attach_view(view)
    assert runtime.finish_view_reconciliation(view, generation)
    assert projected.wait(1)
    assert runtime.resolve_project_instruction_dispatch(decision_ids[0], "proceed")
    worker.join(1)

    assert not worker.is_alive()
    assert result == ["proceed"]


def test_project_dispatch_confirmation_does_not_retain_view_detached_during_wait():
    runtime = _project_notice_runtime()
    first_projected = threading.Event()
    second_projected = threading.Event()
    decision_ids: list[str] = []
    view = _WeakView(
        {
            "project_project_instruction_dispatch": lambda decision_id, _notice: (
                decision_ids.append(decision_id),
                first_projected.set(),
            )
            and True
        }
    )
    generation = runtime.attach_view(view)
    assert runtime.finish_view_reconciliation(view, generation)
    dead_view = weakref.ref(view)
    result: list[str] = []
    worker = threading.Thread(
        target=lambda: result.append(
            runtime._confirm_project_instruction_dispatch(
                SimpleNamespace(session_id="session-a")
            )
        ),
        daemon=True,
    )
    worker.start()
    assert first_projected.wait(1)

    assert runtime.detach_view(view, generation)
    del view
    gc.collect()
    assert dead_view() is None

    successor = _WeakView(
        {
            "project_project_instruction_dispatch": lambda decision_id, _notice: (
                decision_ids.append(decision_id),
                second_projected.set(),
            )
            and True
        }
    )
    successor_generation = runtime.attach_view(successor)
    assert runtime.finish_view_reconciliation(successor, successor_generation)
    assert second_projected.wait(1)
    assert decision_ids[0] == decision_ids[1]
    assert runtime.resolve_project_instruction_dispatch(decision_ids[1], "proceed")
    worker.join(1)
    assert not worker.is_alive()
    assert result == ["proceed"]


def test_project_dispatch_timeout_counts_only_successfully_projected_time():
    runtime = _project_notice_runtime()
    runtime._project_instruction_notice_timeout_seconds = 0.04
    first_projected = threading.Event()
    second_projected = threading.Event()
    decision_ids: list[str] = []

    def first_project(decision_id, _notice):
        decision_ids.append(decision_id)
        first_projected.set()
        return True

    first = _WeakView({"project_project_instruction_dispatch": first_project})
    first_generation = runtime.attach_view(first)
    assert runtime.finish_view_reconciliation(first, first_generation)
    result: list[str] = []
    worker = threading.Thread(
        target=lambda: result.append(
            runtime._confirm_project_instruction_dispatch(
                SimpleNamespace(session_id="session-a")
            )
        ),
        daemon=True,
    )
    worker.start()
    assert first_projected.wait(1)

    assert runtime.detach_view(first, first_generation)
    time.sleep(0.08)
    assert worker.is_alive(), "detached time consumed the answerable-time budget"

    def second_project(decision_id, _notice):
        decision_ids.append(decision_id)
        second_projected.set()
        return True

    second = _WeakView({"project_project_instruction_dispatch": second_project})
    second_generation = runtime.attach_view(second)
    assert runtime.finish_view_reconciliation(second, second_generation)
    assert second_projected.wait(1)
    assert decision_ids == [decision_ids[0], decision_ids[0]]
    assert runtime.resolve_project_instruction_dispatch(decision_ids[0], "proceed")
    worker.join(1)

    assert not worker.is_alive()
    assert result == ["proceed"]


@pytest.mark.asyncio
async def test_project_decisions_project_one_ordered_head_for_the_active_session():
    sessions = [SimpleNamespace(id="session-a"), SimpleNamespace(id="session-b")]
    store = SimpleNamespace(
        active_session_id="session-b",
        sessions=lambda: sessions,
        on_active_session_changed=None,
    )
    controller = SimpleNamespace(_active_cancel_events={})
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)

    binding_a = asyncio.create_task(
        runtime._select_project_instruction_binding(
            "session-a", _project_selection("binding-a"), "unavailable-a"
        )
    )
    await asyncio.sleep(0)
    dispatch_result: list[str] = []
    dispatch_b = threading.Thread(
        target=lambda: dispatch_result.append(
            runtime._confirm_project_instruction_dispatch(
                SimpleNamespace(session_id="session-b")
            )
        ),
        daemon=True,
    )
    dispatch_b.start()
    await asyncio.sleep(0)
    binding_b = asyncio.create_task(
        runtime._select_project_instruction_binding(
            "session-b", _project_selection("binding-b"), "unavailable-b"
        )
    )
    await asyncio.sleep(0)

    projected = threading.Event()
    projections: list[tuple[str, str, str]] = []
    dismissed: list[str] = []

    def project_binding(decision_id, options):
        projections.append(("binding", decision_id, options[0].binding_id))
        projected.set()
        return True

    def project_dispatch(decision_id, notice):
        projections.append(("dispatch", decision_id, notice.session_id))
        projected.set()
        return True

    view = _WeakView(
        {
            "project_project_instruction_binding": project_binding,
            "project_project_instruction_dispatch": project_dispatch,
            "dismiss_project_instruction_decision": (
                lambda decision_id: dismissed.append(decision_id) or True
            ),
        }
    )
    generation = runtime.attach_view(view)
    assert runtime.finish_view_reconciliation(view, generation)
    assert projected.wait(1)
    assert [item[0] for item in projections] == ["dispatch"]
    dispatch_id = projections[0][1]
    projected.clear()
    assert runtime.resolve_project_instruction_dispatch(dispatch_id, "proceed")
    dispatch_b.join(1)
    assert dispatch_result == ["proceed"]

    assert await asyncio.to_thread(projected.wait, 1)
    assert [item[0] for item in projections] == ["dispatch", "binding"]
    assert projections[1][2] == "binding-b"
    binding_b_id = projections[1][1]

    projected.clear()
    store.active_session_id = "session-a"
    store.on_active_session_changed()
    assert await asyncio.to_thread(projected.wait, 1)
    assert projections[-1][0::2] == ("binding", "binding-a")
    assert dismissed == [binding_b_id]
    binding_a_id = projections[-1][1]
    assert runtime.resolve_project_instruction_binding(
        binding_a_id, "select", "binding-a"
    )
    assert await asyncio.wait_for(binding_a, timeout=1) == ("select", "binding-a")

    projected.clear()
    store.active_session_id = "session-b"
    store.on_active_session_changed()
    assert await asyncio.to_thread(projected.wait, 1)
    assert projections[-1] == ("binding", binding_b_id, "binding-b")
    assert runtime.resolve_project_instruction_binding(
        binding_b_id, "select", "binding-b"
    )
    assert await asyncio.wait_for(binding_b, timeout=1) == ("select", "binding-b")


@pytest.mark.asyncio
async def test_new_session_activation_dismisses_the_previous_project_decision():
    store = ConsoleChatStore()
    first = store.create_session(session_id="session-a")
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime.set_chat_store(store)
    runtime.set_chat_controller(SimpleNamespace(_active_cancel_events={}))
    projected = threading.Event()
    decision_ids: list[str] = []
    dismissed: list[str] = []

    view = _WeakView(
        {
            "project_project_instruction_binding": lambda decision_id, _options: (
                decision_ids.append(decision_id),
                projected.set(),
            )
            and True,
            "dismiss_project_instruction_decision": (
                lambda decision_id: dismissed.append(decision_id) or True
            ),
        }
    )
    generation = runtime.attach_view(view)
    assert runtime.finish_view_reconciliation(view, generation)
    pending = asyncio.create_task(
        runtime._select_project_instruction_binding(
            first.id, _project_selection(), "binding_unavailable"
        )
    )
    assert await asyncio.to_thread(projected.wait, 1)
    decision_id = decision_ids[0]

    store.create_session(session_id="session-new")

    assert dismissed == [decision_id]
    assert runtime._project_binding_decisions[decision_id].projected_generation is None
    assert runtime.resolve_project_instruction_binding(
        decision_id, "select", "binding-a"
    )
    assert await asyncio.wait_for(pending, timeout=1) == ("select", "binding-a")


@pytest.mark.asyncio
async def test_direct_workspace_session_activation_rederives_the_ordered_head():
    store = ConsoleChatStore()
    first = store.create_session(session_id="session-a", workspace_id="workspace-a")
    second = store.create_session(
        session_id="session-b", workspace_id="workspace-b", activate=False
    )
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime.set_chat_store(store)
    runtime.set_chat_controller(SimpleNamespace(_active_cancel_events={}))
    projected = threading.Event()
    projections: list[tuple[str, str]] = []
    dismissed: list[str] = []

    view = _WeakView(
        {
            "project_project_instruction_binding": lambda decision_id, options: (
                projections.append((decision_id, options[0].binding_id)),
                projected.set(),
            )
            and True,
            "dismiss_project_instruction_decision": (
                lambda decision_id: dismissed.append(decision_id) or True
            ),
        }
    )
    generation = runtime.attach_view(view)
    assert runtime.finish_view_reconciliation(view, generation)
    first_pending = asyncio.create_task(
        runtime._select_project_instruction_binding(
            first.id, _project_selection("binding-a"), "unavailable-a"
        )
    )
    assert await asyncio.to_thread(projected.wait, 1)
    first_decision_id = projections[0][0]
    second_pending = asyncio.create_task(
        runtime._select_project_instruction_binding(
            second.id, _project_selection("binding-b"), "unavailable-b"
        )
    )
    await asyncio.sleep(0)
    projected.clear()

    # Workspace and hydration controllers intentionally activate through the
    # store directly rather than calling ConsoleChatController.switch_session.
    store.switch_session(second.id)

    assert await asyncio.to_thread(projected.wait, 1)
    assert dismissed == [first_decision_id]
    assert projections[-1][1] == "binding-b"
    second_decision_id = projections[-1][0]
    assert runtime.resolve_project_instruction_binding(
        second_decision_id, "select", "binding-b"
    )
    assert await asyncio.wait_for(second_pending, timeout=1) == (
        "select",
        "binding-b",
    )
    assert runtime.resolve_project_instruction_binding(
        first_decision_id, "select", "binding-a"
    )
    assert await asyncio.wait_for(first_pending, timeout=1) == (
        "select",
        "binding-a",
    )


@pytest.mark.asyncio
async def test_failed_modal_mount_retries_same_decision_after_consecutive_failures():
    scheduled: list[tuple[float, object]] = []
    mounted_callbacks: list[object] = []

    class App:
        attempts = 0

        def set_timer(self, delay, callback):
            scheduled.append((delay, callback))

        def call_later(self, callback):
            scheduled.append((0.0, callback))

        def push_screen(self, _modal, callback):
            self.attempts += 1
            if self.attempts < 3:
                raise RuntimeError("mount failed")
            mounted_callbacks.append(callback)

    class Screen:
        _console_runtime_attachment_retired = False

        def console_view_hooks(self):
            return {
                "project_project_instruction_binding": (
                    session_controller._project_project_instruction_binding
                )
            }

        def _console_runtime(self):
            return runtime

    app = App()
    runtime = ConsoleRuntime(app)
    runtime._chat_store = SimpleNamespace(
        active_session_id="session-a",
        sessions=lambda: [SimpleNamespace(id="session-a")],
    )
    screen = Screen()
    screen.app = app
    session_controller = ConsoleSessionController.__new__(ConsoleSessionController)
    session_controller._screen = screen
    session_controller._project_instruction_decision_modals = {}
    generation = runtime.attach_view(screen)
    screen._console_runtime_attachment_generation = generation
    assert runtime.finish_view_reconciliation(screen, generation)

    task = asyncio.create_task(
        runtime._select_project_instruction_binding(
            "session-a", _project_selection(), "binding_unavailable"
        )
    )
    await asyncio.sleep(0)
    decision_id = next(iter(runtime._project_binding_decisions))
    assert app.attempts == 1
    assert len(scheduled) == 1
    assert session_controller._project_instruction_decision_modals == {}

    delay, retry = scheduled.pop()
    assert delay == 0.05
    retry()
    assert app.attempts == 2
    assert len(scheduled) == 1
    assert session_controller._project_instruction_decision_modals == {}

    delay, retry = scheduled.pop()
    assert delay == 0.1
    retry()
    assert app.attempts == 3
    assert list(session_controller._project_instruction_decision_modals) == [
        decision_id
    ]
    mounted_callbacks[0](ProjectInstructionSetupResult("select", "binding-a"))

    assert await asyncio.wait_for(task, timeout=1) == ("select", "binding-a")


@pytest.mark.asyncio
async def test_persistent_modal_mount_failure_exhausts_until_session_rederivation():
    scheduled: list[tuple[float, object]] = []
    mounted_callbacks: list[object] = []

    class App:
        attempts = 0
        failing = True

        def set_timer(self, delay, callback):
            scheduled.append((delay, callback))

        def call_later(self, callback):
            scheduled.append((0.0, callback))

        def push_screen(self, _modal, callback):
            self.attempts += 1
            if self.failing:
                raise RuntimeError("persistent mount failure")
            mounted_callbacks.append(callback)

    class Screen:
        _console_runtime_attachment_retired = False

        def console_view_hooks(self):
            return {
                "project_project_instruction_binding": (
                    session_controller._project_project_instruction_binding
                )
            }

        def _console_runtime(self):
            return runtime

    app = App()
    store = ConsoleChatStore()
    session = store.create_session(session_id="session-a")
    other_session = store.create_session(session_id="session-b")
    store.switch_session(session.id)
    runtime = ConsoleRuntime(app)
    runtime.set_chat_store(store)
    screen = Screen()
    screen.app = app
    session_controller = ConsoleSessionController.__new__(ConsoleSessionController)
    session_controller._screen = screen
    session_controller._project_instruction_decision_modals = {}
    generation = runtime.attach_view(screen)
    screen._console_runtime_attachment_generation = generation
    assert runtime.finish_view_reconciliation(screen, generation)

    task = asyncio.create_task(
        runtime._select_project_instruction_binding(
            session.id, _project_selection(), "binding_unavailable"
        )
    )
    await asyncio.sleep(0)
    decision_id = next(iter(runtime._project_binding_decisions))
    try:
        delays: list[float] = []
        for _ in range(3):
            assert len(scheduled) == 1
            delay, retry = scheduled.pop()
            delays.append(delay)
            retry()

        assert delays == [0.05, 0.1, 0.2]
        assert scheduled == []
        assert app.attempts == 4
        assert not task.done()
        assert decision_id in runtime._project_binding_decisions

        app.failing = False
        store.switch_session(other_session.id)
        assert app.attempts == 4
        assert scheduled == []
        store.switch_session(session.id)

        assert app.attempts == 5
        assert list(session_controller._project_instruction_decision_modals) == [
            decision_id
        ]
        mounted_callbacks[0](ProjectInstructionSetupResult("select", "binding-a"))
        assert await asyncio.wait_for(task, timeout=1) == ("select", "binding-a")
    finally:
        if not task.done():
            runtime.resolve_project_instruction_binding(decision_id, "cancel", None)
            await task


def _attach_reconciliation_screen(sync, *, start=None, resume_pending=False):
    scheduled: list[tuple[float, object]] = []
    runtime = SimpleNamespace(finish_view_reconciliation=lambda *_args: True)
    screen = ChatScreen.__new__(ChatScreen)
    screen._closing = False
    screen._closed = False
    screen._console_attach_reconciled = False
    screen._console_attach_reconcile_running = False
    screen._console_resume_after_reconcile = resume_pending
    screen._console_runtime_attachment_generation = 1
    screen._sync_native_console_chat_ui = sync
    screen._console_runtime = lambda: runtime
    screen._start_console_view_after_reconciliation = start or (lambda: None)
    screen.set_timer = lambda delay, callback: scheduled.append((delay, callback))
    return screen, runtime, scheduled


@pytest.mark.asyncio
async def test_persistent_attach_sync_failure_has_bounded_backoff_and_resume_retry():
    sync_calls = 0

    async def fail_sync():
        nonlocal sync_calls
        sync_calls += 1
        raise RuntimeError("persistent sync failure")

    screen, _runtime, scheduled = _attach_reconciliation_screen(fail_sync)

    await screen._reconcile_console_after_attach()
    delays: list[float] = []
    for _ in range(3):
        assert len(scheduled) == 1
        delay, retry = scheduled.pop()
        delays.append(delay)
        await retry()

    assert delays == [0.05, 0.1, 0.2]
    assert scheduled == []
    assert sync_calls == 4
    assert screen._console_attach_reconciled is False

    screen.call_after_refresh = lambda callback: scheduled.append((0.0, callback))
    screen.on_screen_resume()
    assert len(scheduled) == 1
    _delay, retry = scheduled.pop()
    await retry()
    assert sync_calls == 5


@pytest.mark.asyncio
async def test_attach_view_start_failure_retries_without_repeating_core_reconciliation():
    sync_calls = 0
    finish_calls = 0
    start_calls = 0

    async def sync():
        nonlocal sync_calls
        sync_calls += 1

    def start():
        nonlocal start_calls
        start_calls += 1
        if start_calls == 1:
            raise RuntimeError("first view start failed")

    screen, runtime, scheduled = _attach_reconciliation_screen(sync, start=start)

    def finish(*_args):
        nonlocal finish_calls
        finish_calls += 1
        return True

    runtime.finish_view_reconciliation = finish

    await screen._reconcile_console_after_attach()
    assert screen._console_attach_reconciled is False
    assert len(scheduled) == 1

    _delay, retry = scheduled.pop()
    await retry()
    assert (sync_calls, finish_calls, start_calls) == (1, 1, 2)
    assert screen._console_attach_reconciled is True


@pytest.mark.asyncio
async def test_attach_deferred_resume_failure_retries_after_view_start_once():
    sync_calls = 0
    start_calls = 0
    resume_calls = 0

    async def sync():
        nonlocal sync_calls
        sync_calls += 1

    def start():
        nonlocal start_calls
        start_calls += 1

    screen, _runtime, scheduled = _attach_reconciliation_screen(
        sync, start=start, resume_pending=True
    )

    def resume():
        nonlocal resume_calls
        resume_calls += 1
        if resume_calls == 1:
            raise RuntimeError("first deferred resume failed")

    screen.on_screen_resume = resume

    await screen._reconcile_console_after_attach()
    assert screen._console_attach_reconciled is False
    assert screen._console_resume_after_reconcile is True
    assert len(scheduled) == 1

    _delay, retry = scheduled.pop()
    await retry()
    assert (sync_calls, start_calls, resume_calls) == (1, 1, 2)
    assert screen._console_resume_after_reconcile is False
    assert screen._console_attach_reconciled is True


def _post_reconciliation_admission_screen(monkeypatch, live_reason=None):
    from contextlib import nullcontext
    from tldw_chatbook.UI.Screens import chat_screen

    def no_op(*_args, **_kwargs):
        return None

    async def async_no_op(*_args, **_kwargs):
        return None

    screen, runtime, _scheduled = _attach_reconciliation_screen(async_no_op)
    screen._start_console_view_after_reconciliation = (
        ChatScreen._start_console_view_after_reconciliation.__get__(screen)
    )
    screen._sync_native_console_chat_ui = (
        ChatScreen._sync_native_console_chat_ui.__get__(screen)
    )
    screen._console_sync_in_progress = False
    screen._console_sync_requested = False
    screen._console_transcript_sync_timer = None
    runtime.chat_store = None
    runtime.chat_controller = SimpleNamespace(
        run_state=SimpleNamespace(
            status=(
                ConsoleRunStatus.STREAMING
                if live_reason == "viewed"
                else ConsoleRunStatus.IDLE
            )
        ),
        in_flight_run_count=lambda: int(live_reason == "other"),
        fleet_wake=SimpleNamespace(
            delivering_conversation_id=lambda: (
                "wake-owner" if live_reason == "wake" else None
            )
        ),
    )
    runtime.change_review_coordinator = SimpleNamespace(
        publication_signal=SimpleNamespace(
            snapshot=lambda: SimpleNamespace(pending=int(live_reason == "review"))
        )
    )
    screen._task_resume_state = SimpleNamespace(followed_watchlists_operations=())
    screen._resume_navigation_startup_in_progress = False
    screen._pending_character_return_focus_id = None
    screen._fleet = SimpleNamespace(
        consume_pending_console_fleet_completion=no_op,
        _maybe_start_console_fleet_survivor_tick=no_op,
    )
    screen._image = SimpleNamespace(_reconcile_h3_image_edit_completions=no_op)
    screen._skill = SimpleNamespace(_refresh_console_skill_candidates=async_no_op)
    screen._message = SimpleNamespace(reconcile_console_speech_context=no_op)
    screen._session = SimpleNamespace(_sync_console_session_draft=no_op)
    screen._retrieval = SimpleNamespace(
        _warm_console_effective_scope_cache_if_stale=async_no_op,
        _refresh_active_dictionaries_summary_if_scope_changed=async_no_op,
        _refresh_active_world_books_summary_if_scope_changed=async_no_op,
    )
    screen._character = SimpleNamespace(
        _refresh_active_character_avatar_if_scope_changed=async_no_op
    )
    screen._character_context = SimpleNamespace(refresh_if_scope_changed=async_no_op)
    screen._workspace = SimpleNamespace(
        tick_workspace_build_scope=nullcontext,
        _invalidate_console_persisted_rows_cache=no_op,
    )
    for name in (
        "_record_ui_timer_created",
        "_record_ui_timer_stopped",
        "_record_ui_worker_started",
        "_record_ui_worker_finished",
        "_sync_console_chat_core_state",
        "_current_console_rail_state",
        "_sync_console_settings_summary",
        "_sync_console_control_bar",
        "_sync_console_settings_recovery_surfaces",
        "_sync_console_live_work_readiness_rows",
        "_sync_console_mode_bar",
        "_dispatch_active_console_roleplay_refresh",
        "_sync_console_workspace_context",
        "_sync_console_rail_visibility_if_changed",
        "_dispatch_console_rail_preference_prune",
    ):
        setattr(screen, name, no_op)
    screen._sync_console_native_session_tabs = async_no_op
    screen._sync_native_console_transcript = async_no_op
    monkeypatch.setattr(
        chat_screen.project_instruction_ui,
        "sync_project_instruction_status_for_screen",
        no_op,
    )
    intervals, refreshed, workers = [], [], []

    def interval(seconds, callback):
        intervals.append((seconds, callback))
        return SimpleNamespace(stop=no_op)

    def run_worker(coroutine, **kwargs):
        workers.append((coroutine.cr_code.co_name, kwargs.get("group")))
        coroutine.close()

    screen.set_interval = interval
    screen.call_after_refresh = refreshed.append
    screen.run_worker = run_worker
    return screen, intervals, refreshed, workers


def test_idle_reconciled_view_does_not_admit_transcript_poll(monkeypatch):
    screen, intervals, _refreshed, _workers = _post_reconciliation_admission_screen(
        monkeypatch
    )
    screen._start_console_view_after_reconciliation()
    assert not [
        callback for _, callback in intervals if callback.__name__ == "_poll_transcript"
    ]


@pytest.mark.parametrize("live_reason", ("viewed", "other", "wake", "review"))
def test_reconciled_view_keeps_each_live_poll_reason_and_one_timer(
    monkeypatch, live_reason
):
    screen, intervals, _refreshed, _workers = _post_reconciliation_admission_screen(
        monkeypatch, live_reason
    )
    screen._start_console_view_after_reconciliation()
    screen._start_console_transcript_sync_timer()
    assert (
        len(
            [
                callback
                for _, callback in intervals
                if callback.__name__ == "_poll_transcript"
            ]
        )
        == 1
    )


@pytest.mark.asyncio
async def test_captured_attach_timer_overlap_rearms_real_sync_worker(monkeypatch):
    screen, intervals, refreshed, workers = _post_reconciliation_admission_screen(
        monkeypatch, "other"
    )
    screen._start_console_view_after_reconciliation()
    poll = next(
        callback for _, callback in intervals if callback.__name__ == "_poll_transcript"
    )
    after_refresh = next(
        callback
        for callback in refreshed
        if callback.__name__ == "_sync_native_console_chat_ui"
    )
    entered, release = asyncio.Event(), asyncio.Event()

    async def hold_scope_warmup():
        entered.set()
        await release.wait()

    screen._retrieval._warm_console_effective_scope_cache_if_stale = hold_scope_warmup
    first = asyncio.create_task(after_refresh())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert screen._console_sync_requested is False
        await poll()
        assert screen._console_sync_requested is True
    finally:
        release.set()
        await asyncio.wait_for(first, 1)
    assert ("_sync_native_console_chat_ui", "console-sync") in workers


@pytest.mark.asyncio
@pytest.mark.parametrize("before_sibling", (False, True))
async def test_legacy_alias_skips_queryable_but_detached_tray(before_sibling):
    from textual.widget import Widget

    screen = ChatScreen.__new__(ChatScreen)
    screen._closing = screen._closed = False
    tray = SimpleNamespace(
        is_attached=False,
        is_mounted=True,
        is_running=True,
        _closing=False,
        _closed=False,
        _pruning=False,
    )
    mounts, allocations = [], []

    def mount(*children, **kwargs):
        mounts.append(children)
        return Widget.mount(tray, *children, **kwargs)

    tray.mount = mount
    screen.query_one = lambda *_args: tray
    screen.query = lambda selector: (
        [object()]
        if before_sibling and selector == "#console-workspace-conversations"
        else []
    )
    screen._workspace = SimpleNamespace(
        _build_console_workspace_context_state=lambda: SimpleNamespace(
            new_conversation_enabled=True
        )
    )
    screen._request_console_context_allocation_reconcile = lambda: allocations.append(
        True
    )
    await screen._sync_console_legacy_workspace_context_aliases()
    assert mounts == allocations == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transition", ("detach", "replace", "close", "live_error", "success")
)
async def test_legacy_alias_allocation_requires_current_live_tray(transition):
    from textual.widget import MountError

    screen = ChatScreen.__new__(ChatScreen)
    screen._closing = screen._closed = False
    tray = SimpleNamespace(is_attached=True)
    current = [tray]
    allocations, children = [], []
    entered, release = asyncio.Event(), asyncio.Event()

    async def mount(child, **_kwargs):
        if transition == "live_error":
            raise MountError("sentinel attached-tray failure")
        children.append(child)
        entered.set()
        await release.wait()

    tray.mount = mount
    screen.query_one = lambda *_args: current[0]
    screen.query = lambda _selector: []
    screen._workspace = SimpleNamespace(
        _build_console_workspace_context_state=lambda: SimpleNamespace(
            new_conversation_enabled=True
        )
    )
    screen._request_console_context_allocation_reconcile = lambda: allocations.append(
        True
    )
    if transition == "live_error":
        with pytest.raises(MountError, match="sentinel attached-tray failure"):
            await screen._sync_console_legacy_workspace_context_aliases()
        assert allocations == []
        return
    pending = asyncio.create_task(
        screen._sync_console_legacy_workspace_context_aliases()
    )
    try:
        await asyncio.wait_for(entered.wait(), 1)
        if transition == "detach":
            tray.is_attached = False
        elif transition == "replace":
            current[0] = SimpleNamespace(is_attached=True)
        elif transition == "close":
            screen._closing = True
    finally:
        release.set()
        await asyncio.wait_for(pending, 1)
    assert len(children) == 1
    assert children[0].id == "console-new-workspace-conversation"
    assert allocations == ([True] if transition == "success" else [])


def test_runtime_project_decision_owner_never_constructs_textual_widgets():
    source = (_PACKAGE_ROOT.parent / _RUNTIME_MODULE).read_text(encoding="utf-8")
    assert "ProjectInstructionSetupModal" not in source
    assert "ProjectInstructionNoticeModal" not in source


def test_detach_releases_the_old_chat_screen_from_every_runtime_hook():
    runtime = ConsoleRuntime(None)
    screen = ChatScreen.__new__(ChatScreen)

    def no_op(*_args, **_kwargs):
        return None

    screen._fleet = ConsoleFleetLifecycleController(
        **{
            name: no_op
            for name in inspect.signature(ConsoleFleetLifecycleController).parameters
        }
    )
    screen._session = SimpleNamespace(
        _project_project_instruction_binding=no_op,
        _project_project_instruction_dispatch=no_op,
        _dismiss_project_instruction_decision_projection=no_op,
    )
    wake = SimpleNamespace(delivering_session_id=lambda: None)
    controller = SimpleNamespace(
        fleet_wake=wake,
        prompt_queue_coordinator=SimpleNamespace(),
    )
    store = SimpleNamespace()
    runtime.set_chat_controller(controller)
    runtime.set_chat_store(store)

    generation = runtime.attach_view(screen)
    assert any(
        inspect.ismethod(getattr(runtime._hook_target(slot.target), slot.name))
        for slot in CONSOLE_VIEW_HOOK_SLOTS
    )

    dead_screen = weakref.ref(screen)
    assert runtime.detach_view(screen, generation)
    del screen
    gc.collect()

    assert dead_screen() is None


@pytest.mark.asyncio
async def test_active_runtime_custody_does_not_retain_the_detached_chat_screen():
    """The surviving task owns inputs and services, never its old projection."""
    runtime = ConsoleRuntime(None)
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingController:
        fleet_wake = SimpleNamespace(delivering_session_id=lambda: None)
        prompt_queue_coordinator = SimpleNamespace()

        async def run_prompt_chain(self, *, session_id, initial_turn):
            return await initial_turn()

        async def submit_draft(self, _draft, **kwargs):
            started.set()
            await release.wait()
            kwargs["custody_acceptance_hook"]()
            return SimpleNamespace(accepted=True)

    runtime.set_chat_store(store)
    runtime.set_chat_controller(BlockingController())
    screen = ChatScreen.__new__(ChatScreen)

    def no_op(*_args, **_kwargs):
        return None

    screen._fleet = ConsoleFleetLifecycleController(
        **{
            name: no_op
            for name in inspect.signature(ConsoleFleetLifecycleController).parameters
        }
    )
    screen._session = SimpleNamespace(
        _project_project_instruction_binding=no_op,
        _project_project_instruction_dispatch=no_op,
        _dismiss_project_instruction_decision_projection=no_op,
    )
    generation = runtime.attach_view(screen)
    turn_id = runtime.accept_turn(
        ConsoleTurnCustodyRequest(
            turn_id="screen-free-custody",
            session_id=session.id,
            draft="private draft",
            configuration=ConsoleTurnConfigurationSnapshot.capture(
                session_id=session.id,
                provider_selection=ConsoleProviderSelection(provider="openai"),
            ),
        )
    )
    await started.wait()
    dead_screen = weakref.ref(screen)

    assert runtime.detach_view(screen, generation)
    del screen
    gc.collect()

    assert dead_screen() is None
    assert tuple(runtime._turn_custody) == (turn_id,)
    release.set()
    result = await runtime.wait_for_turn(turn_id)
    assert result.accepted is True


@pytest.mark.asyncio
async def test_second_console_visit_reuses_the_runtime(tmp_path):
    """The runtime SURVIVES leaving Console -- this landing's central change.

    Replaces Task 1's `test_second_console_visit_gets_a_new_runtime`, which
    pinned the opposite (dispose-at-unmount) and said in its own docstring
    that it must be rewritten here.
    """
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    terminal_manager = app.terminal_session_manager

    async with app.run_test(size=(160, 48)) as pilot:
        chat = ChatScreen(app)
        await app.push_screen(chat)
        app._initial_screen_pushed = True
        app.current_tab = "chat"
        await pilot.pause()
        await _wait_for_selector(chat, pilot, "#console-native-composer")

        controller_one = chat._ensure_console_chat_controller()
        runtime_one = getattr(app, CONSOLE_RUNTIME_ATTR, None)
        assert isinstance(runtime_one, ConsoleRuntime), type(runtime_one).__name__
        # The app -- not the screen -- is what built these.
        assert runtime_one.chat_controller is controller_one
        assert runtime_one.chat_store is chat._console_chat_store
        assert runtime_one.provider_gateway is chat._console_provider_gateway
        bridge_one = runtime_one.agent_bridge
        assert bridge_one is not None, (
            "the real-on-disk-DB rig must actually build an agent bridge, "
            "or this test cannot say anything about bridge identity"
        )
        assert bridge_one is chat._console_agent_bridge
        store_one = runtime_one.chat_store
        visit_one_event = controller_one._shutdown_requested
        assert runtime_one.generation == 0
        retained_domain_callbacks = (
            runtime_one.provider_gateway._config_provider,
            controller_one._provider_config,
            controller_one._select_project_instruction_binding,
            controller_one._confirm_project_instruction_dispatch,
            bridge_one._native_tools_enabled,
        )
        for callback in retained_domain_callbacks:
            assert getattr(callback, "__self__", None) is not chat
            assert chat not in getattr(callback, "args", ()), (
                "a provider/project/native-tools callback retained ChatScreen"
            )

        # ---- leave Console through the real navigation API ---------------
        await app.handle_screen_navigation(NavigateToScreen("library"))
        await pilot.pause()
        assert chat not in app.screen_stack, "Console must actually unmount"
        # Navigation is a view-only detach; domain cancellation is unchanged.
        assert controller_one._shutdown_requested is visit_one_event
        assert not visit_one_event.is_set()
        # ...every screen-owned slot is back at its viewless default...
        assert controller_one.notify_run_outcome is None
        assert controller_one.fleet_wake.delivery_ui_hook is None
        # task-15860 Task 4: the view probe's viewless default is NOT None
        # (its read site reads an unwired probe as IN VIEW). Asserted here
        # through the DECISION the production path makes, after a real
        # navigation -- an unwatched delivery must not be able to report
        # itself as watched and clear the ◈ mark.
        assert (
            controller_one.fleet_wake._conversation_in_view(
                "conv-anything", "sess-anything"
            )
            is False
        ), (
            "with Console genuinely unmounted the runtime still reported the "
            "conversation as being watched"
        )
        assert runtime_one.view is None
        # ...and the runtime itself is untouched and still the app's.
        assert runtime_one.generation == 0, "leaving Console must NOT dispose"
        assert getattr(app, CONSOLE_RUNTIME_ATTR, None) is runtime_one
        assert runtime_one.chat_controller is controller_one
        assert runtime_one.provider_gateway is not None, (
            "the gateway is app-owned now and must not be closed/dropped "
            "on a navigation"
        )

        # ---- return to Console -------------------------------------------
        await app.handle_screen_navigation(NavigateToScreen("chat"))
        await pilot.pause()
        chat_two = app.screen
        assert isinstance(chat_two, ChatScreen), type(chat_two).__name__
        assert chat_two is not chat, "screens are never cached"
        await _wait_for_selector(chat_two, pilot, "#console-native-composer")

        controller_two = chat_two._ensure_console_chat_controller()
        runtime_two = getattr(app, CONSOLE_RUNTIME_ATTR, None)
        assert runtime_two is runtime_one, "the SAME runtime serves visit two"
        assert app.terminal_session_manager is terminal_manager
        assert controller_two is controller_one
        assert runtime_two.chat_store is store_one
        assert runtime_two.agent_bridge is bridge_one
        # Reattaching does not replace or signal the domain cancellation gate.
        assert controller_two._shutdown_requested is visit_one_event
        assert not visit_one_event.is_set()
        # The hooks now answer for the LIVE screen, not the dead one.
        assert controller_two.notify_run_outcome is not None
        assert controller_two.notify_run_outcome.__self__ is chat_two, (
            "a hook is still bound to the previous, unmounted screen"
        )


@pytest.mark.asyncio
async def test_post_unmount_raw_refusal_restores_on_second_console_visit(tmp_path):
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)

    async with app.run_test(size=(160, 48)) as pilot:
        chat = ChatScreen(app)
        await app.push_screen(chat)
        app._initial_screen_pushed = True
        app.current_tab = "chat"
        await pilot.pause()
        await _wait_for_selector(chat, pilot, "#console-native-composer")

        store = chat._ensure_console_chat_store()
        origin_session_id = store.active_session_id
        assert origin_session_id is not None
        runtime = chat._console_runtime()
        controller_a = chat._raw_cli

        source = ConsoleComposerBar()
        assert source.handle_console_key(Key("exclamation_mark", "!")) is True
        assert source.handle_console_key(Key("space", " ")) is True
        source.insert_pasted_text("pwd")
        stash = source.stash_draft_for_send()
        assert stash is not None

        await app.handle_screen_navigation(NavigateToScreen("library"))
        await pilot.pause()
        assert chat not in app.screen_stack

        controller_a._append_local_error = lambda _session_id, _text: None
        controller_a._refuse(origin_session_id, stash, "test refusal")
        assert runtime.raw_cli_refusal_stash_bank[origin_session_id][0] is stash

        await app.handle_screen_navigation(NavigateToScreen("chat"))
        await pilot.pause()
        chat_two = app.screen
        assert isinstance(chat_two, ChatScreen)
        await _wait_for_selector(chat_two, pilot, "#console-native-composer")

        assert chat_two._console_runtime() is runtime
        assert chat_two._raw_cli is not controller_a
        composer = chat_two.query_one("#console-native-composer", ConsoleComposerBar)
        restored = composer.stash_draft_for_send()
        assert restored is not None
        assert restored.segments == stash.segments
        assert restored.raw_cli_prefix_typed is True
        assert restored.has_paste is True
        classified = classify_console_raw_draft(restored)
        assert classified.kind == "raw"
        assert classified.text == "pwd"
        assert runtime.raw_cli_refusal_stash_bank == {}


@pytest.mark.asyncio
async def test_a_terminal_run_state_after_leaving_does_not_reach_the_dead_screen(
    tmp_path,
):
    """RED (defect): a run settling post-unmount called into a dead view.

    Drives the production path -- `_set_run_state` into a terminal status
    for a NON-active session -- on the controller that SURVIVES the
    navigation, and asserts the unmounted screen's toast never fires.
    Without `detach_view` the slot is still bound to that screen and it
    does.
    """
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)

    async with app.run_test(size=(160, 48)) as pilot:
        chat = ChatScreen(app)
        await app.push_screen(chat)
        app._initial_screen_pushed = True
        app.current_tab = "chat"
        await pilot.pause()
        await _wait_for_selector(chat, pilot, "#console-native-composer")

        controller = chat._ensure_console_chat_controller()
        store = chat._ensure_console_chat_store()
        background = store.create_session(title="background")
        # `create_session` activates what it creates, so this second one
        # leaves a DIFFERENT session viewed -- which is what makes the
        # terminal transition below take the non-active branch that owns
        # `notify_run_outcome`.
        store.create_session(title="viewed")
        assert store.active_session_id != background.id

        reached: list[tuple[str, ConsoleRunStatus]] = []
        original = chat._notify_console_run_outcome
        def broken_dead_view_notification(session_id, status):
            reached.append((session_id, status))
            raise RuntimeError("injected dead-view notification failure")

        chat._notify_console_run_outcome = broken_dead_view_notification
        # Re-bind so the recorder is what the runtime holds for THIS visit.
        chat._console_runtime().attach_view(chat)
        assert controller.notify_run_outcome is not None

        await app.handle_screen_navigation(NavigateToScreen("library"))
        await pilot.pause()
        assert chat not in app.screen_stack, "Console must actually unmount"

        controller._set_run_state(
            ConsoleRunState(status=ConsoleRunStatus.COMPLETED),
            session_id=background.id,
        )
        await pilot.pause()

        assert reached == [], (
            "a run that settled after the navigation reached the UNMOUNTED "
            f"screen's toast: {reached}"
        )
        chat._notify_console_run_outcome = original


@pytest.mark.asyncio
async def test_a_superseded_screen_never_detaches_the_successors_runtime(tmp_path):
    """The restore-before-unmount order, made explicit.

    `_complete_screen_navigation` constructs the incoming screen and calls
    `restore_state` (which reaches `ensure_chat_store`) BEFORE
    `switch_screen` unmounts the outgoing one, so on a SAME-TARGET
    navigation -- reachable through the `coding` route, which aliases to
    Chat -- the incoming screen attaches FIRST and the outgoing screen
    detaches SECOND. The successor's claim must win: its hooks stay bound
    and its visit Event stays unset.
    """
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)

    async with app.run_test(size=(160, 48)) as pilot:
        chat = ChatScreen(app)
        await app.push_screen(chat)
        app._initial_screen_pushed = True
        app.current_tab = "chat"
        await pilot.pause()
        await _wait_for_selector(chat, pilot, "#console-native-composer")

        controller = chat._ensure_console_chat_controller()
        runtime = getattr(app, CONSOLE_RUNTIME_ATTR, None)
        assert runtime.view is chat

        await app.handle_screen_navigation(NavigateToScreen("coding"))
        await pilot.pause()
        chat_two = app.screen
        assert isinstance(chat_two, ChatScreen), type(chat_two).__name__
        assert chat_two is not chat
        await _wait_for_selector(chat_two, pilot, "#console-native-composer")

        assert runtime.view is chat_two, (
            "the outgoing screen's detach ran anyway and dropped the claim "
            "its successor had already made"
        )
        assert controller.notify_run_outcome is not None, (
            "the outgoing screen cleared a hook its successor had bound"
        )
        assert not controller._shutdown_requested.is_set(), (
            "the outgoing screen's leave_console poisoned the incoming "
            "visit -- a dead Console after a same-target navigation"
        )


@pytest.mark.asyncio
async def test_opening_console_during_a_headless_delivery_arms_the_poll(tmp_path):
    """task-15860 Task 4: the mid-delivery freeze, at the REAL mount.

    `delivery_ui_hook` fires exactly once, in `_attempt`, at delivery
    start -- and it is the only thing that arms the 0.2s transcript poll
    for a wake turn (a wake bypasses the user-send worker that normally
    arms it). With the runtime outliving the screen, "delivery start" and
    "view attach" are independent events, so a Console opened DURING a
    delivery that began with no view must be re-armed by the attach
    itself. PR 3a-2 Task 7 measured the cost of not doing it live: a 4+
    minute frozen transcript.

    Driven through the production navigation API, with a control leg
    (return with nothing delivering -> no poll) so the assertion cannot
    pass for the wrong reason.
    """
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)

    async with app.run_test(size=(160, 48)) as pilot:
        chat = ChatScreen(app)
        await app.push_screen(chat)
        app._initial_screen_pushed = True
        app.current_tab = "chat"
        await pilot.pause()
        await _wait_for_selector(chat, pilot, "#console-native-composer")

        controller = chat._ensure_console_chat_controller()
        store = chat._ensure_console_chat_store()
        session_id = store.active_session_id
        assert session_id, "the rig must have an active session"
        wake = controller.fleet_wake

        # -- control leg: leave and return with NOTHING delivering -------
        await app.handle_screen_navigation(NavigateToScreen("library"))
        await pilot.pause()
        await app.handle_screen_navigation(NavigateToScreen("chat"))
        await pilot.pause()
        control = app.screen
        await _wait_for_selector(control, pilot, "#console-native-composer")
        await pilot.pause()
        assert control._console_transcript_sync_timer is None, (
            "returning to Console with nothing in flight armed the poll "
            "anyway -- the delivery leg below would prove nothing"
        )

        # -- the real leg: a wake begins while Console is unmounted ------
        await app.handle_screen_navigation(NavigateToScreen("library"))
        await pilot.pause()
        assert control not in app.screen_stack, "Console must actually unmount"
        # Harness precondition ONLY: stand in for `_attempt` having marked
        # a delivery in flight. Everything after this line is production.
        wake._delivering = session_id
        wake._delivering_session = session_id
        try:
            await app.handle_screen_navigation(NavigateToScreen("chat"))
            await pilot.pause()
            reopened = app.screen
            assert isinstance(reopened, ChatScreen), type(reopened).__name__
            await _wait_for_selector(reopened, pilot, "#console-native-composer")
            await pilot.pause()

            assert reopened._console_transcript_sync_timer is not None, (
                "Console opened during a wake delivery with no transcript "
                f"poll armed: reconciled={reopened._console_attach_reconciled}, "
                f"delivering={wake.delivering_session_id()}, "
                f"same_controller={reopened._console_chat_controller is controller}"
            )
        finally:
            wake._delivering = None
            wake._delivering_session = None


@pytest.mark.unit
def test_the_runtime_is_disposed_by_the_apps_shutdown_lifecycles():
    """`dispose()` is app-exit work and must be registered as such.

    The runtime survives every navigation now, so nothing else ends it.
    """
    import inspect

    from tldw_chatbook.app import TldwCli

    source = inspect.getsource(TldwCli._shutdown_app_owned_lifecycles)
    assert "_shutdown_console_runtime" in source, source
    disposer = inspect.getsource(TldwCli._shutdown_console_runtime)
    assert "dispose_console_runtime" in disposer, disposer


@pytest.mark.unit
def test_sync_constructed_app_starts_canvas_policy_watch_in_running_lifecycle(
    monkeypatch,
):
    """The real app loop observes disable even when no preview was opened."""
    from tldw_chatbook import config as config_module

    canvas_policy = {"enabled": True}
    monkeypatch.setattr(
        config_module,
        "get_canvas_execution_enabled",
        lambda: canvas_policy["enabled"],
    )
    # Shipping CLI construction happens before Textual creates its loop.
    app = _build_test_app()
    runtime = app.console_runtime
    assert isinstance(runtime, ConsoleRuntime)
    assert runtime._canvas_policy_watch_task is None

    async def exercise_app_lifecycle() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            watcher = runtime._canvas_policy_watch_task
            assert watcher is not None
            assert not watcher.done()
            assert runtime.canvas_gateway is None
            assert runtime.canvas_controller is None

            # Starting the lifecycle again must retain the sole watcher.
            runtime.start_async_lifecycles()
            runtime.start_async_lifecycles()
            assert runtime._canvas_policy_watch_task is watcher

            canvas_policy["enabled"] = False
            # Await only the watcher: no gate read is allowed to latch the
            # disabled interval on behalf of the lifecycle under test.
            await asyncio.wait_for(asyncio.shield(watcher), timeout=1.0)
            canvas_policy["enabled"] = True

            assert runtime.canvas_enabled() is False

        assert runtime._disposed is True
        assert runtime._canvas_policy_watch_task is None
        runtime.start_async_lifecycles()
        assert runtime._canvas_policy_watch_task is None
        await runtime.dispose()
        assert runtime._canvas_policy_watch_task is None

    asyncio.run(exercise_app_lifecycle())


@pytest.mark.unit
def test_raw_cli_runtime_is_app_owned_unarmed_and_reads_config_replacements():
    """The app owns one launch-local arm bit over its latest config object."""
    from tldw_chatbook.app import TldwCli

    initializer = inspect.getsource(TldwCli.__init__)
    config_load = initializer.index("self.app_config = load_settings()")
    raw_runtime = initializer.index("self.raw_cli_runtime")
    next_owner = initializer.index("self.library_new_profile_admission")
    assert config_load < raw_runtime < next_owner, initializer

    app = _build_test_app(config_overrides={"console": {"raw_cli_permitted": True}})
    runtime = app.raw_cli_runtime
    assert runtime.permitted is True
    assert runtime.armed is False

    app.app_config = {"console": {"raw_cli_permitted": "true"}}
    assert runtime.arm().armed is False
    app.app_config = {"console": {"raw_cli_permitted": 1}}
    assert runtime.arm().armed is False
    app.app_config = {"console": {"raw_cli_permitted": True}}
    assert runtime.arm().armed is True
    assert app.raw_cli_runtime is runtime
    runtime.shutdown()


@pytest.mark.unit
def test_terminal_manager_is_app_owned_unarmed_and_reads_config_replacements():
    """The app lazily owns one launch-local Terminal arm over latest config."""
    from tldw_chatbook.app import TldwCli

    initializer = inspect.getsource(TldwCli.__init__)
    config_load = initializer.index("self.app_config = load_settings()")
    terminal_manager = initializer.index("self._terminal_session_manager")
    console_runtime = initializer.index("self.console_runtime")
    assert config_load < terminal_manager < console_runtime, initializer

    app = _build_test_app(config_overrides={"console": {"raw_cli_permitted": True}})
    assert app._terminal_session_manager is None

    from tldw_chatbook.Terminal.session_manager import TerminalSessionManager

    manager = app.terminal_session_manager
    assert isinstance(manager, TerminalSessionManager)
    assert manager.permitted is True
    assert manager.armed is False

    app.app_config = {"console": {"raw_cli_permitted": "true"}}
    assert manager.arm(acknowledge_disclosure=True).armed is False
    app.app_config = {"console": {"raw_cli_permitted": 1}}
    assert manager.arm(acknowledge_disclosure=True).armed is False
    app.app_config = {"console": {"raw_cli_permitted": True}}
    assert manager.arm(acknowledge_disclosure=True).armed is True
    assert app.terminal_session_manager is manager
    manager.disarm()


@pytest.mark.asyncio
async def test_raw_cli_runtime_shutdown_is_once_and_precedes_console_shutdown():
    """Both Textual shutdown paths share one raw-runtime shutdown task."""
    from tldw_chatbook.app import TldwCli

    calls: list[str] = []

    class Runtime:
        def shutdown(self) -> object:
            calls.append("raw")
            return object()

    app = object.__new__(TldwCli)
    app.raw_cli_runtime = Runtime()
    app._raw_cli_runtime_shutdown_task = None

    await TldwCli._shutdown_raw_cli_runtime(app)
    await TldwCli._shutdown_raw_cli_runtime(app)

    assert calls == ["raw"]
    source = inspect.getsource(TldwCli._shutdown_app_owned_lifecycles)
    raw = source.index("_shutdown_raw_cli_runtime")
    console = source.index("_shutdown_console_runtime")
    assert raw < console, source


@pytest.mark.asyncio
async def test_terminal_manager_shutdown_is_shared_and_shielded_from_waiter_cancel():
    """One app task owns the five-second drain and final handle closure."""
    from tldw_chatbook.app import TldwCli

    entered = asyncio.Event()
    release = asyncio.Event()
    calls: list[object] = []

    class Manager:
        async def shutdown(self, *, deadline_seconds: float) -> bool:
            calls.append(("shutdown", deadline_seconds))
            entered.set()
            await release.wait()
            return False

        def finalize_shutdown(self) -> None:
            calls.append("finalize")

    app = object.__new__(TldwCli)
    app.terminal_session_manager = Manager()
    app._terminal_session_manager_shutdown_task = None

    first = asyncio.create_task(TldwCli._shutdown_terminal_session_manager(app))
    await asyncio.wait_for(entered.wait(), 1)
    second = asyncio.create_task(TldwCli._shutdown_terminal_session_manager(app))
    await asyncio.sleep(0)

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert calls == [("shutdown", 5.0)]

    release.set()
    await asyncio.wait_for(second, 1)
    assert calls == [("shutdown", 5.0), "finalize"]

    source = inspect.getsource(TldwCli._shutdown_app_owned_lifecycles)
    terminal = source.index("_shutdown_terminal_session_manager")
    console = source.index("_shutdown_console_runtime")
    buddy = source.index("_shutdown_persona_buddy")
    assert terminal < console < buddy, source


@pytest.mark.asyncio
async def test_app_shutdown_drains_and_finalizes_a_real_terminal_manager():
    """The app boundary drives real manager cleanup through finalization."""
    from tldw_chatbook.Terminal.contracts import (
        AdmissionGate,
        BackendIdentity,
        CleanupAttempt,
        CleanupProof,
        TerminalLaunchRequest,
    )
    from tldw_chatbook.Terminal.session_manager import TerminalSessionManager
    from tldw_chatbook.app import TldwCli

    cleanup_entered = threading.Event()
    cleanup_release = threading.Event()

    class Backend:
        def __init__(self) -> None:
            self.finalize_calls = 0

        def start(
            self,
            _request: TerminalLaunchRequest,
            admission: AdmissionGate,
        ) -> BackendIdentity:
            return BackendIdentity(session_id=admission.token)

        def read(self, _maximum: int = 64 * 1024) -> bytes | None:
            return None

        def write(self, _data: bytes) -> None:
            return None

        def resize(self, _columns: int, _rows: int) -> None:
            return None

        def request_priority_close(self) -> None:
            return None

        def cleanup(self, _attempt: CleanupAttempt) -> CleanupProof:
            cleanup_entered.set()
            assert cleanup_release.wait(1)
            return CleanupProof()

        def finalize_shutdown(self) -> None:
            self.finalize_calls += 1

    backend = Backend()
    manager = TerminalSessionManager(lambda: True, lambda: backend)
    manager.arm(acknowledge_disclosure=True)
    created = manager.create_session(
        TerminalLaunchRequest(
            name="app-shutdown-integration",
            shell="default",
            start_directory=str(Path.cwd()),
            columns=80,
            rows=24,
        )
    )
    assert created.admitted is True

    app = object.__new__(TldwCli)
    app.terminal_session_manager = manager
    app._terminal_session_manager_shutdown_task = None

    first = asyncio.create_task(TldwCli._shutdown_terminal_session_manager(app))
    assert await asyncio.to_thread(cleanup_entered.wait, 1)
    second = asyncio.create_task(TldwCli._shutdown_terminal_session_manager(app))

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert backend.finalize_calls == 0

    cleanup_release.set()
    await asyncio.wait_for(second, 1)
    assert backend.finalize_calls == 1


@pytest.mark.unit
def test_persona_buddy_is_app_owned_and_shutdown_after_console_producers():
    """Console producers stop before Buddy drains, which precedes profiles.

    TASK-21103 rewrote what the construction half of this pin means. The
    controller is no longer built inside ``__init__`` — importing it drags
    Persona_Visual and PIL (1.28 s cold) onto the boot path, so the eager
    wiring became the lazy ``persona_buddy_controller`` property over
    ``_build_persona_buddy_controller``. The construction SEMANTICS the old
    pin protected (portrait loader partial over the local persona service)
    moved there intact, and ``__init__`` must stay construction-free. The
    shutdown half is unchanged in meaning: Console producers stop first,
    then Buddy drains — but the disposer must now PEEK the slot rather than
    read the property, or a never-built controller would be constructed
    (importing PIL) purely to be shut down.
    """
    import inspect

    from tldw_chatbook.app import TldwCli

    initializer = inspect.getsource(TldwCli.__init__)
    assert "PersonaBuddyController(" not in initializer, (
        "eager Buddy construction is back in __init__ (TASK-21103 regression)"
    )
    slot = initializer.index("self._persona_buddy_controller")
    console_runtime = initializer.index("= ConsoleRuntime(self)")
    assert slot < console_runtime, (
        "the controller slot must exist before ConsoleRuntime reads the "
        "persona_buddy_controller property"
    )

    assert isinstance(
        inspect.getattr_static(TldwCli, "persona_buddy_controller"), property
    )
    builder = inspect.getsource(TldwCli._build_persona_buddy_controller)
    assert "PersonaBuddyController(" in builder, builder
    assert "portrait_loader=partial(" in builder, builder
    assert "load_local_persona_portrait" in builder, builder
    guard = builder.index('"local_character_persona_service"')
    construction = builder.index("PersonaBuddyController(")
    assert guard < construction, builder

    source = inspect.getsource(TldwCli._shutdown_app_owned_lifecycles)
    buddy = source.index("_shutdown_persona_buddy")
    console = source.index("_shutdown_console_runtime")
    assert console < buddy, source
    disposer = inspect.getsource(TldwCli._shutdown_persona_buddy)
    assert "self._persona_buddy_controller" in disposer, disposer
    assert "controller.shutdown()" in disposer, disposer
    assert "self.persona_buddy_controller.shutdown" not in disposer, (
        "shutdown must peek the slot, never the constructing property"
    )


@pytest.mark.unit
def test_lazy_persona_buddy_property_defers_and_ensure_constructs():
    """The lazy controller property's three states behave as designed.

    TASK-21103 behavior pins, on a skeletal ``TldwCli``:

    - disabled preferences: the passive property returns None WITHOUT
      constructing (the every-screen-mount reconcile early-out stays free of
      the Persona_Visual/PIL import);
    - disabled preferences, explicit feature use:
      ``ensure_persona_buddy_controller()`` constructs anyway (this is what
      lets Workbench "Use for Buddy" enable from a disabled state), and the
      passive property then returns the same cached instance;
    - enabled preferences: the first passive read constructs, and the
      construction is cached (same object on the second read).
    - the setter installs a test double the property returns verbatim.
    """
    from tldw_chatbook.app import TldwCli

    def skeleton(enabled: bool) -> TldwCli:
        app = object.__new__(TldwCli)
        app._persona_buddy_controller = None
        app._persona_buddy_controller_lock = threading.Lock()
        app.app_config = {"persona_buddy": {"enabled": enabled}}
        app.local_character_persona_service = object()
        app.chachanotes_db = object()
        app.call_after_refresh = lambda *args, **kwargs: None
        return app

    disabled = skeleton(enabled=False)
    assert disabled.persona_buddy_controller is None
    assert disabled._persona_buddy_controller is None, (
        "the passive property constructed a controller for a disabled profile"
    )

    ensured = disabled.ensure_persona_buddy_controller()
    assert ensured is not None
    assert disabled.persona_buddy_controller is ensured

    enabled = skeleton(enabled=True)
    first = enabled.persona_buddy_controller
    assert first is not None
    assert enabled.persona_buddy_controller is first

    injected = object()
    enabled.persona_buddy_controller = injected
    assert enabled.persona_buddy_controller is injected


@pytest.mark.unit
def test_actor_pack_recovery_precedes_character_persona_surfaces():
    """Cross-store recovery is gated ahead of every affected surface.

    task-21106 rewrote what this pin means. Recovery no longer runs inside
    ``_wire_character_persona_services`` — synchronous SQLite during
    ``__init__`` cost every boot and crashed the test app factory — so the
    old ``local_service < coordinator < recover() < scope`` source ordering
    is gone by design. The guarantee it protected now has three seams, and
    this test pins all of them:

    - the deferred-startup worker kicks ``ensure_actor_pack_recovery`` on a
      thread right after first paint (ahead of any user-driven Console/Buddy
      persona read);
    - the Personas surface awaits the same idempotent gate before its first
      library read (behavioral proof in test_actor_pack_recovery_seam.py);
    - the coordinator itself runs ``ensure_recovered`` before admitting a
      ``create_persona`` mutation, so no caller ordering can bypass it.
    """
    import inspect

    from tldw_chatbook.Actor_Packs.persona_coordinator import (
        PersonaActorPackCoordinator,
    )
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen

    wiring = inspect.getsource(TldwCli._wire_character_persona_services)
    local_service = wiring.index("LocalCharacterPersonaService(")
    coordinator = wiring.index("PersonaActorPackCoordinator(")
    scope = wiring.index("CharacterPersonaScopeService(")
    assert local_service < coordinator < scope, wiring
    assert ".recover()" not in wiring, (
        "recovery is back on the construction path (task-21106 regression)"
    )

    # TASK-22215 moved the deferred-startup kick behind the boot-worker
    # stagger policy: `_schedule_deferred_startup_work` opens the gate, the
    # policy names the row, and the app's starter table calls the gate's
    # `ensure_actor_pack_recovery`. Recovery is still FIRST in that order --
    # it is a prefetch for a surface that would otherwise run SQLite recovery
    # on the event loop -- and the behavioral proof that it actually runs
    # after first paint lives in Tests/UI/test_actor_pack_recovery_seam.py
    # and Tests/App/test_boot_worker_stagger_policy.py.
    from tldw_chatbook.Utils.boot_worker_policy import STAGGERED_BOOT_WORKER_KEYS

    deferred = inspect.getsource(TldwCli._schedule_deferred_startup_work)
    assert "_start_staggered_boot_workers" in deferred, deferred
    assert STAGGERED_BOOT_WORKER_KEYS[0] == "actor_pack_recovery", (
        f"recovery must stay first in the staggered boot order; observed "
        f"{STAGGERED_BOOT_WORKER_KEYS!r}"
    )
    starters = inspect.getsource(TldwCli.boot_worker_starters)
    assert "ensure_actor_pack_recovery" in starters, starters

    personas_load = inspect.getsource(PersonasScreen._load_after_mount)
    assert "ensure_actor_pack_recovery" in personas_load, personas_load

    create = inspect.getsource(PersonaActorPackCoordinator.create_persona)
    assert create.index("self.ensure_recovered()") < create.index(
        "self._blocked_intent_ids"
    ), create

    # TASK-21103 moved Buddy construction out of ``__init__`` into the lazy
    # ``_build_persona_buddy_controller``. The ordering guarantee this stanza
    # pinned — Buddy is only ever wired to a fully constructed local persona
    # service — is now enforced by the builder itself: it reads the service
    # defensively and defers (returns None, retried on next access) until
    # ``_wire_character_persona_services`` has run.
    builder = inspect.getsource(TldwCli._build_persona_buddy_controller)
    guard = builder.index('"local_character_persona_service"')
    buddy = builder.index("PersonaBuddyController(")
    assert guard < buddy, builder
    assert "PersonaBuddyController(" not in inspect.getsource(TldwCli.__init__)


@pytest.mark.asyncio
async def test_app_fences_console_then_drains_buddy_before_profile_teardown(
    monkeypatch: pytest.MonkeyPatch,
):
    """Repeated cancellation cannot skip either ordered app-owned drain."""
    from textual.app import App

    from tldw_chatbook.app import TldwCli

    terminal_entered = asyncio.Event()
    terminal_release = asyncio.Event()
    console_entered = asyncio.Event()
    console_release = asyncio.Event()
    buddy_entered = asyncio.Event()
    buddy_release = asyncio.Event()
    events: list[str] = []

    class Buddy:
        async def shutdown(self) -> None:
            events.append("buddy-start")
            buddy_entered.set()
            await buddy_release.wait()
            events.append("buddy-finished")

    class ProfileService:
        def teardown(self) -> None:
            events.append("profile-teardown")

    class AsyncOwner:
        async def shutdown(self) -> None:
            events.append("later-owner")

    async def later_lifecycle() -> None:
        events.append("later-lifecycle")

    async def no_op_lifecycle() -> None:
        return None

    terminal_task: asyncio.Task[None] | None = None

    async def terminal_runner() -> None:
        events.append("terminal-start")
        terminal_entered.set()
        await terminal_release.wait()
        events.append("terminal-finished")

    async def shutdown_terminal_manager() -> None:
        nonlocal terminal_task
        if terminal_task is None:
            terminal_task = asyncio.create_task(terminal_runner())
        await asyncio.shield(terminal_task)

    console_task: asyncio.Task[None] | None = None

    async def console_runner() -> None:
        events.append("console-start")
        console_entered.set()
        await console_release.wait()
        events.append("console-finished")

    async def shutdown_console_runtime() -> None:
        nonlocal console_task
        if console_task is None:
            console_task = asyncio.create_task(console_runner())
        await asyncio.shield(console_task)

    app = object.__new__(TldwCli)
    app.persona_buddy_controller = Buddy()
    app._persona_buddy_shutdown_task = None
    app._audio_cpp_artifact_lease_coordinator = None
    app.audio_cpp_model_install_owner = AsyncOwner()
    app._meeting_session_owner = None
    app._shutdown_notes_sync_runtime = no_op_lifecycle
    app._shutdown_raw_cli_runtime = no_op_lifecycle
    app._shutdown_terminal_session_manager = shutdown_terminal_manager
    app._shutdown_console_image_edits = later_lifecycle
    app._shutdown_console_runtime = shutdown_console_runtime
    app._shutdown_file_notes_session_owner = later_lifecycle
    profile_service = ProfileService()

    async def profile_teardown(_app: App[None]) -> None:
        profile_service.teardown()

    monkeypatch.setattr(App, "_shutdown", profile_teardown)
    draining = asyncio.create_task(TldwCli._shutdown(app))
    await asyncio.wait_for(terminal_entered.wait(), 2)
    assert events == ["terminal-start"]
    assert not draining.done()

    draining.cancel()
    await asyncio.sleep(0)
    draining.cancel()
    await asyncio.sleep(0)
    assert not draining.done()
    assert not console_entered.is_set()
    assert not buddy_entered.is_set()

    terminal_release.set()
    await asyncio.wait_for(console_entered.wait(), 2)
    assert events[:3] == [
        "terminal-start",
        "terminal-finished",
        "console-start",
    ]

    console_release.set()
    await asyncio.wait_for(buddy_entered.wait(), 2)
    assert events[2:5] == ["console-start", "console-finished", "buddy-start"]
    assert "profile-teardown" not in events

    buddy_release.set()
    with pytest.raises(asyncio.CancelledError):
        await draining

    assert events.index("terminal-finished") < events.index("console-start")
    assert events.index("console-finished") < events.index("buddy-start")
    assert events.index("buddy-finished") < events.index("profile-teardown")
    assert events[-1] == "profile-teardown"
