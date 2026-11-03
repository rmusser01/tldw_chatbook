"""Ordinary Console admission covers the actual originating physical tree."""

import asyncio
import threading

import pytest

from Tests.Chat.test_console_reference_custody import (
    reference_host as reference_host,  # noqa: PLC0414 - expose pytest fixture
)
from Tests.Chat.test_console_reference_custody import (
    until,
)
from tldw_chatbook.Agents.execution_capacity import current_execution_owner
from tldw_chatbook.Chat import console_references
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.fixture
async def ordinary_host(reference_host):
    h = reference_host
    database = AgentRunsDB(h.tmp_path / "agent-runs.sqlite", client_id="ordinary-tree")
    bridge = ConsoleAgentBridge(
        agent_runs_db=database,
        store=h.store,
        provider_gateway=h.gateway,
        runtime_capacity=h.runtime.execution_capacity,
        get_hooks_v2=h.runtime.get_hooks_v2,
    )
    h.runtime.set_agent_bridge(bridge)
    h.controller._agent_bridge = bridge
    h.controller._agent_runtime_enabled = True
    h.database, h.bridge = database, bridge
    try:
        yield h
    finally:
        await asyncio.gather(*h.tasks, return_exceptions=True)
        database.close()


async def test_conflicting_actual_reference_refuses_other_turn_before_row_or_model(
    ordinary_host, monkeypatch
):
    h = ordinary_host
    entered, release = threading.Event(), threading.Event()
    owners = []
    expand = console_references._expand_path

    def held_read(*args, **kwargs):
        owners.append(current_execution_owner())
        entered.set()
        assert release.wait(20)
        return expand(*args, **kwargs)

    monkeypatch.setattr(console_references, "_expand_path", held_read)
    first = asyncio.create_task(
        h.controller.submit_draft("read @visible.txt", session_id=h.session.id)
    )
    h.tasks.append(first)
    other = h.store.create_session(workspace_id=h.session.workspace_id)
    try:
        await until(lambda: entered.is_set() or first.done())
        assert entered.is_set(), first.result() if first.done() else "no original read"
        assert owners[0] is not None and owners[0].is_admitted
        result = await h.controller.submit_draft(
            "original second draft", session_id=other.id
        )
        assert not result.accepted, "conflicting ordinary turn reached acceptance"
        assert h.gateway.payloads == [], "conflicting ordinary turn reached MODEL"
        assert other.persisted_conversation_id is None
        assert all(
            not entry.child
            for entry in h.runtime.execution_capacity.snapshot().executions
        )
    finally:
        release.set()
        await first


async def test_prefill_excludes_unreachable_agent_effects_from_original_maximum(
    ordinary_host,
):
    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree

    h = ordinary_host
    seen = []

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            seen.append(current_ordinary_tree())
            self.payloads.append(messages)
            yield "bounded reply"

    gateway = Gateway()
    h.controller.provider_gateway = gateway
    h.bridge._gateway = gateway
    h.store.set_session_one_shot_prefill(h.session.id, "Start")
    result = await h.controller.submit_draft(
        "bounded prefill send",
        session_id=h.session.id,
    )
    assert result.accepted
    assert len(seen) == 1 and seen[0] is not None
    assert not seen[0].footprint.opaque, (
        "unreachable child tools widened prefill maximum"
    )


@pytest.mark.parametrize(
    "workers,write_mode",
    [
        (1, "normal"),
        (2, "normal"),
        (1, "failed"),
        (1, "committed_error"),
        (1, "already_canonical"),
    ],
)
async def test_actual_child_joins_original_claim_before_its_row(
    ordinary_host, monkeypatch, workers, write_mode
):
    from Tests.Agents.test_agent_service import fence
    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from tldw_chatbook.Agents.agent_models import SPAWN_TOOL_NAME
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree

    async def exercise():
        h = ordinary_host
        if write_mode == "failed":
            import json
            import os
            import subprocess
            import sys
            from pathlib import Path

            if os.environ.get("TASK8_ORDINARY_REQUIRED_PEER") != "failed":
                receipt = h.tmp_path / "required-retention.json"
                result = await asyncio.to_thread(
                    subprocess.run,
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        "-s",
                        f"{__file__}::test_actual_child_joins_original_claim_before_its_row[1-failed]",
                        "--basetemp=" + str(h.tmp_path / "required-child"),
                    ],
                    env=dict(
                        os.environ,
                        TASK8_ORDINARY_REQUIRED_PEER="failed",
                        TASK8_ORDINARY_REQUIRED_RECEIPT=str(receipt),
                    ),
                    capture_output=True,
                    text=True,
                    timeout=35,
                )
                (h.tmp_path / "required-child.log").write_text(
                    result.stdout + result.stderr
                )
                assert result.returncode == 0, result.stdout + result.stderr
                assert json.loads(receipt.read_text()) == {
                    "canonical_missing": True,
                    "completion_pending": True,
                    "disposal_pending": True,
                    "lease_retained": True,
                }
                return
        monkeypatch.setenv("TLDW_AGENTS_MAX_LIVE_SUBAGENTS", str(workers))
        monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "false")
        h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
        observations = []
        trees = []
        create_run = h.database.create_run

        def observed_create(**kwargs):
            tree = current_ordinary_tree()
            trees.append(tree)
            if kwargs["agent_kind"] == "subagent":
                children = (
                    [owner for owner in tree.members if owner.child] if tree else []
                )
                observations.append(
                    bool(children) and all(owner.is_admitted for owner in children)
                )
            return create_run(**kwargs)

        monkeypatch.setattr(h.database, "create_run", observed_create)
        terminal_write = h.database.set_terminal_with_step

        def terminal(run_id, *args, **kwargs):
            child = h.database.get_run(run_id)["agent_kind"] == "subagent"
            if child and write_mode == "failed":
                raise OSError("controlled missing canonical child terminal")
            result = terminal_write(run_id, *args, **kwargs)
            if child and write_mode == "committed_error":
                raise OSError("controlled error after original canonical commit")
            if child and write_mode == "already_canonical":
                return False
            return result

        monkeypatch.setattr(h.database, "set_terminal_with_step", terminal)

        class Gateway(_RecordingGateway):
            async def stream_chat(self, resolution, messages, **kwargs):
                self.payloads.append(messages)
                if len(self.payloads) == 1:
                    yield fence(
                        SPAWN_TOOL_NAME, {"task": "Read the original instructions"}
                    )
                elif len(self.payloads) == 2:
                    yield "child completed"
                else:
                    yield "parent completed"

        gateway = Gateway()
        h.controller.provider_gateway = gateway
        h.bridge._gateway = gateway
        result = await h.controller.submit_draft(
            "spawn a child", session_id=h.session.id
        )
        assert result.accepted
        await until(lambda: not h.runtime.execution_capacity.snapshot().executions)
        rows = h.database.list_runs(h.session.persisted_conversation_id)
        assert len(gateway.payloads) == 3, rows
        assert observations == [True], (
            "actual child row was created outside the original claim"
        )
        assert all(tree is trees[0] for tree in trees)
        from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

        with h.runtime.worktree_activity.prelaunch(
            footprint=WorktreeFootprint(opaque=True)
        ) as claim:
            assert (claim is None) is (write_mode == "failed"), (
                "canonical write proof did not control original claim release"
            )

        if write_mode == "failed":
            tree = trees[0]
            child = next(
                member for owner, member in tree.members.items() if owner.child
            )
            row = h.database.get_run(child.run_id)
            from tldw_chatbook.Chat.console_ordinary_tree import _json

            stored_steps = {step["index"]: _json(step) for step in row["steps"]}
            assert any(
                stored_steps.get(index) != value for index, value in child.steps.items()
            )
            assert child.host_returned and child.drained and not child.durable
            grace_returned = asyncio.Event()
            original_wait = h.runtime._bounded_wait

            async def observe_grace(tasks, *, timeout_seconds):
                pending = await original_wait(tasks, timeout_seconds=timeout_seconds)
                if timeout_seconds == 0:
                    grace_returned.set()
                return pending

            monkeypatch.setattr(h.runtime, "_bounded_wait", observe_grace)
            disposal = asyncio.create_task(h.runtime.dispose(timeout_seconds=0))
            await asyncio.wait_for(grace_returned.wait(), 5)
            done, _pending = await asyncio.wait({disposal}, timeout=0.1)
            assert not done and not tree.retired.done()
            assert tree.scratch_lease is not None
            Path(os.environ["TASK8_ORDINARY_REQUIRED_RECEIPT"]).write_text(
                json.dumps(
                    {
                        "canonical_missing": True,
                        "completion_pending": True,
                        "disposal_pending": True,
                        "lease_retained": True,
                    }
                )
            )
            # This intentional uncertainty has no cleanup proof. The isolated peer
            # exits after recording the actual native retention, never resetting it.
            os._exit(0)

    try:
        await exercise()
    except BaseException:
        import os
        import sys
        import traceback

        if (
            write_mode == "failed"
            and os.environ.get("TASK8_ORDINARY_REQUIRED_PEER") == "failed"
        ):
            traceback.print_exc(file=sys.__stderr__)
            sys.__stderr__.flush()
            os._exit(1)
        raise


@pytest.mark.parametrize("dispose", [False, True])
async def test_personal_context_deadline_retains_original_preparation_worker(
    ordinary_host, monkeypatch, dispose
):
    from tldw_chatbook.Chat import console_chat_controller as host
    from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

    h = ordinary_host
    h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    entered, release = threading.Event(), threading.Event()
    workers = []

    def original_getter():
        workers.append(current_execution_owner())
        entered.set()
        assert release.wait(20)

    monkeypatch.setattr(
        h.controller.app, "get_personal_context_service", original_getter, raising=False
    )
    monkeypatch.setattr(host, "CONSOLE_PRE_PROVIDER_SETUP_BUDGET_SECONDS", 0.05)
    task = asyncio.create_task(
        h.controller.submit_draft("bounded setup deadline", session_id=h.session.id)
    )
    h.tasks.append(task)
    disposal = None
    try:
        await until(lambda: entered.is_set() or task.done())
        assert entered.is_set(), task.result() if task.done() else "no original getter"
        assert (await task).accepted
        assert h.gateway.payloads, "the original setup deadline no longer permits MODEL"
        snapshot = h.runtime.execution_capacity.snapshot()
        assert snapshot.tool_workers == 1, (
            "deadline discarded the original getter operation"
        )
        assert workers[0] is not None and workers[0].is_admitted
        with h.runtime.worktree_activity.prelaunch(
            footprint=WorktreeFootprint(opaque=True)
        ) as claim:
            assert claim is None, "the getter lost its covering tree before return"
        if dispose:
            disposal = asyncio.create_task(h.runtime.dispose())
            await until(lambda: h.runtime._disposed)
            assert not disposal.done(), "disposal discarded the original getter"
    finally:
        release.set()
        await task
        await until(lambda: not host._PERSONAL_CONTEXT_BOOTSTRAP_IN_FLIGHT.is_set())
        if disposal is not None:
            await disposal
    await until(lambda: not h.runtime.execution_capacity.snapshot().executions)
    with h.runtime.worktree_activity.prelaunch(
        footprint=WorktreeFootprint(opaque=True)
    ) as claim:
        assert claim is not None


@pytest.mark.parametrize(
    "library_kind", ["unknown", "notes_read", "notes_save", "conversation_read"]
)
async def test_original_unknown_library_factory_is_opaque_before_construction(
    ordinary_host, library_kind
):
    from dataclasses import replace

    from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider
    from tldw_chatbook.Chat.console_library_policy import (
        ConsoleAssistantLibraryAccess,
        ConsoleAutoRetrieve,
        ConsoleLibraryPolicyDefaults,
    )
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree
    from tldw_chatbook.Library.local_library_tool_service import LocalLibraryToolService

    h = ordinary_host
    h.store.set_library_policy_defaults(
        ConsoleLibraryPolicyDefaults(
            auto_retrieve=ConsoleAutoRetrieve.NEVER,
            assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
        )
    )
    session = h.store.create_session(workspace_id=h.session.workspace_id)
    h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    constructions = []

    def factory(context):
        tree = current_ordinary_tree()
        constructions.append((tree, tree.footprint.opaque if tree else None))
        return LibraryToolProvider(LocalLibraryToolService())

    if library_kind == "unknown":
        h.controller._library_provider_factory = factory
    else:
        from functools import partial

        from Tests.Agents.test_agent_service import fence
        from Tests.Chat.test_console_skill_substitution import _RecordingGateway
        from tldw_chatbook.Chat.console_runtime import _library_provider_for_app
        from tldw_chatbook.Notes.Notes_Library import NotesInteropService

        h.app.notes_service = NotesInteropService(
            h.tmp_path, "ordinary-library", global_db_to_use=h.store.persistence.db
        )
        h.controller._library_provider_factory = partial(
            _library_provider_for_app, h.app
        )

        class Gateway(_RecordingGateway):
            async def stream_chat(self, resolution, messages, **kwargs):
                self.payloads.append(messages)
                tree = current_ordinary_tree()
                constructions.append((tree, tree.footprint.opaque))
                selected = (
                    "library_list_conversations"
                    if library_kind == "conversation_read"
                    else "library_list_notes"
                )
                if len(self.payloads) == 1:
                    yield fence("load_tools", {"ids": ["library:" + selected]})
                elif len(self.payloads) == 2:
                    yield fence(selected, {})
                else:
                    yield "Library read complete"

        h.gateway = Gateway()
        h.controller.provider_gateway = h.gateway
        h.bridge._gateway = h.gateway
    configuration = await h.controller.capture_turn_configuration_snapshot(session.id)
    tools = dict(configuration.tool_configuration)
    tools["agent_run_budget_maximum"] = replace(
        tools["agent_run_budget_maximum"], max_subagents=0
    )
    configuration = replace(configuration, tool_configuration=tools)
    if library_kind in {"notes_read", "conversation_read"}:
        selected = (
            "library_list_conversations"
            if library_kind == "conversation_read"
            else "library_list_notes"
        )
        configuration = replace(
            configuration,
            persona_policy_rules=(
                {"rule_kind": "mcp_tool", "rule_name": selected, "allowed": True},
            ),
        )
    result = await h.controller.submit_draft(
        "use Library", session_id=session.id, configuration=configuration
    )
    assert result.accepted and h.gateway.payloads
    expected_opaque = library_kind in {"unknown", "notes_save"}
    assert constructions and all(
        tree is not None and opaque is expected_opaque for tree, opaque in constructions
    ), constructions
    if library_kind != "unknown":
        assert len(h.gateway.payloads) == 3
        assert constructions[0][0].library_providers
        assert '"items"' in str(h.gateway.payloads[-1]), h.gateway.payloads[-1]


async def test_custom_change_review_reader_never_runs_before_ordinary_admission(
    ordinary_host,
):
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree
    from tldw_chatbook.Workspaces.change_review_consent import (
        ChangeReviewCapability,
        ChangeReviewConsentService,
        ChangeReviewState,
    )

    h = ordinary_host
    reads = []

    def capability():
        reads.append(current_ordinary_tree())
        return ChangeReviewCapability(ChangeReviewState.DISABLED)

    consent = ChangeReviewConsentService(h.registry, capability_reader=capability)
    h.app.change_review_consent_service = consent
    h.registry.attach_change_review_consent_service(consent)
    h.runtime.bind_change_review_consent_service(consent)
    h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    assert (
        await h.controller.submit_draft("original metadata", session_id=h.session.id)
    ).accepted
    assert reads and all(
        tree is not None and tree.footprint.opaque for tree in reads
    ), reads


async def test_native_personal_getter_does_not_classify_custom_cached_service_as_bounded(
    ordinary_host,
):
    from dataclasses import replace

    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree

    h = ordinary_host
    from textual.app import App

    app = object.__new__(TldwCli)
    App.__init__(app)
    app.__dict__.update(vars(h.app))
    app._loop = asyncio.get_running_loop()
    app._thread_id = threading.get_ident()
    from tldw_chatbook.Personal_Context.service import PersonalContextService

    class CustomProfileService(PersonalContextService):
        def authorized_context_view(self, **kwargs):
            return super().authorized_context_view(**kwargs)

    app._personal_context_service = CustomProfileService.locked("unavailable")
    h.runtime._app = app
    h.controller.app = app
    h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    seen = []

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            seen.append(current_ordinary_tree())
            self.payloads.append(messages)
            yield "personal context unavailable"

    gateway = Gateway()
    h.controller.provider_gateway = gateway
    h.bridge._gateway = gateway
    configuration = await h.controller.capture_turn_configuration_snapshot(h.session.id)
    tools = dict(configuration.tool_configuration)
    tools["agent_run_budget_maximum"] = replace(
        tools["agent_run_budget_maximum"], max_subagents=0
    )
    result = await h.controller.submit_draft(
        "original cached source",
        session_id=h.session.id,
        configuration=replace(configuration, tool_configuration=tools),
    )
    assert result.accepted and seen, [
        row.content for row in h.store.messages_for_session(h.session.id)
    ]
    assert all(tree.footprint.opaque for tree in seen), (
        "native getter hid its custom cached service"
    )


@pytest.mark.parametrize("closure", ["cancel", "dispose"])
async def test_cancelled_original_admission_returns_its_unpublished_tree(
    ordinary_host, monkeypatch, closure
):
    import json
    import os
    import subprocess
    import sys
    import traceback
    from pathlib import Path

    from tldw_chatbook.Chat.console_ordinary_tree import _OrdinaryTurn

    h = ordinary_host
    if os.environ.get("TASK8_ORDINARY_ADMISSION_PEER") != closure:
        receipt = h.tmp_path / "admission-return.json"
        result = await asyncio.to_thread(
            subprocess.run,
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-s",
                f"{__file__}::test_cancelled_original_admission_returns_its_unpublished_tree[{closure}]",
                "--basetemp=" + str(h.tmp_path / "admission-child"),
            ],
            env=dict(
                os.environ,
                TASK8_ORDINARY_ADMISSION_PEER=closure,
                TASK8_ORDINARY_ADMISSION_RECEIPT=str(receipt),
            ),
            capture_output=True,
            text=True,
            timeout=35,
        )
        (h.tmp_path / "admission-child.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(receipt.read_text()) == {"retired": True, "models": 0}
        return

    entered, release = threading.Event(), threading.Event()
    trees = []
    admit = _OrdinaryTurn.admit

    def held(tree, footprint):
        admit(tree, footprint)
        trees.append(tree)
        entered.set()
        assert release.wait(20)

    monkeypatch.setattr(_OrdinaryTurn, "admit", held)
    task = asyncio.create_task(
        h.controller.submit_draft(
            "cancel before admission publication", session_id=h.session.id
        )
    )
    h.tasks.append(task)
    try:
        await until(lambda: entered.is_set() or task.done())
        assert entered.is_set()
        disposal = None
        if closure == "dispose":
            disposal = asyncio.create_task(h.runtime.dispose(timeout_seconds=0))
            await until(lambda: h.runtime._disposed)
            assert not disposal.done()
        else:
            task.cancel()
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        tree = trees[0]
        assert tree.retired.done(), (
            "cancelled actual admission lost its finish publisher"
        )
        assert not h.gateway.payloads
        assert tree not in h.runtime._ordinary_turns
        assert not h.runtime.execution_capacity.snapshot().executions
        if disposal is not None:
            await disposal
        else:
            await h.runtime.dispose(timeout_seconds=0)
        Path(os.environ["TASK8_ORDINARY_ADMISSION_RECEIPT"]).write_text(
            json.dumps({"retired": True, "models": 0})
        )
    except BaseException:  # noqa: BLE001 - isolated peer reports cancellation and exits
        release.set()
        traceback.print_exc(file=sys.__stderr__)
        sys.__stderr__.flush()
        os._exit(1)


async def test_actual_received_turn_without_mcp_is_bounded(ordinary_host):
    from dataclasses import replace

    from Tests.Chat.test_console_received_intent_custody import _intent
    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree
    from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest

    h = ordinary_host
    trees = []

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            trees.append(current_ordinary_tree())
            self.payloads.append(messages)
            yield "received reply"

    gateway = Gateway()
    h.controller.provider_gateway = gateway
    h.bridge._gateway = gateway
    h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    h.store.set_session_draft(h.session.id, "received original draft")
    configuration = await h.controller.capture_turn_configuration_snapshot(h.session.id)
    request = ConsoleTurnCustodyRequest(
        turn_id="original-received",
        session_id=h.session.id,
        draft="received original draft",
        configuration=configuration,
    )
    intent = _intent(
        (h.runtime, h.store, h.session, request), turn_id="original-received"
    )
    tools = dict(configuration.tool_configuration)
    tools["agent_run_budget_maximum"] = replace(
        tools["agent_run_budget_maximum"], max_subagents=0
    )
    intent = replace(
        intent,
        selection=replace(
            intent.selection,
            agent_runtime_enabled=True,
            tool_configuration=tools,
        ),
    )
    turn_id = h.runtime.accept_received_intent(intent)
    task = h.runtime._turn_custody[turn_id].task
    h.tasks.append(task)
    assert (await task).accepted
    assert trees and all(
        tree is trees[0] and not tree.footprint.opaque for tree in trees
    )


async def test_skill_tool_child_joins_the_original_ordinary_claim(
    ordinary_host, monkeypatch
):
    from dataclasses import replace

    from Tests.Agents.test_agent_service import fence
    from Tests.Chat.test_console_agent_bridge import _FakeSkillsService
    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree

    h = ordinary_host
    monkeypatch.setenv("TLDW_AGENTS_MAX_LIVE_SUBAGENTS", "1")
    monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "false")
    h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    configuration = await h.controller.capture_turn_configuration_snapshot(h.session.id)
    skills = _FakeSkillsService()
    maximum = await skills.get_context()
    maximum["backend"] = "local"
    configuration = replace(configuration, skill_context_maximum=maximum)
    h.bridge._skills_service = skills
    h.controller._skills_service = skills
    trees, child_owned = [], []
    create_run = h.database.create_run

    def observed_create(**kwargs):
        tree = current_ordinary_tree()
        if kwargs["agent_kind"] == "subagent":
            child_owned.append(
                bool(
                    tree
                    and any(owner.child and owner.is_admitted for owner in tree.members)
                )
            )
            trees.append(tree)
        return create_run(**kwargs)

    monkeypatch.setattr(h.database, "create_run", observed_create)

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            trees.append(current_ordinary_tree())
            self.payloads.append(messages)
            if len(self.payloads) == 1:
                yield fence("code-review", {"args": "the diff"})
            elif len(self.payloads) == 2:
                yield "skill child completed"
            else:
                yield "parent completed"

    gateway = Gateway()
    h.controller.provider_gateway = gateway
    h.bridge._gateway = gateway
    result = await h.controller.submit_draft(
        "review a diff",
        session_id=h.session.id,
        configuration=configuration,
    )
    assert result.accepted
    await until(lambda: not h.runtime.execution_capacity.snapshot().executions)
    assert len(gateway.payloads) == 3 and skills.execute_calls == ["the diff"]
    assert child_owned == [True]
    assert trees and trees[0] is not None and all(tree is trees[0] for tree in trees)
    children = [
        row
        for row in h.database.list_runs(h.session.persisted_conversation_id)
        if row["agent_kind"] == "subagent"
    ]
    assert len(children) == 1 and children[0]["status"] == "done"


async def test_resumed_fleet_child_joins_the_original_ordinary_claim(
    ordinary_host, monkeypatch
):
    from Tests.Agents.test_agent_service import fence
    from Tests.Chat.test_console_agent_bridge import SUBAGENT_PROMPT_PREFIX
    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from tldw_chatbook.Agents.agent_models import (
        SEND_TO_AGENT_TOOL_NAME,
        SPAWN_TOOL_NAME,
        WAIT_AGENTS_TOOL_NAME,
    )
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree

    h = ordinary_host
    monkeypatch.setenv("TLDW_AGENTS_MAX_LIVE_SUBAGENTS", "2")
    monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "false")
    h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    trees, children, child_model_owners = [], [], []
    original_create = h.database.create_run

    def observed_create(**kwargs):
        tree = current_ordinary_tree()
        if kwargs["agent_kind"] == "subagent":
            children.append(
                bool(
                    tree
                    and any(owner.child and owner.is_admitted for owner in tree.members)
                )
            )
            trees.append(("child-row", tree))
        return original_create(**kwargs)

    monkeypatch.setattr(h.database, "create_run", observed_create)

    class Gateway(_RecordingGateway):
        def __init__(self):
            super().__init__()
            self.parent_calls = self.child_calls = 0

        async def stream_chat(self, resolution, messages, **kwargs):
            system = str(messages[0].get("content", "")) if messages else ""
            trees.append(
                (
                    "child-model"
                    if system.startswith(SUBAGENT_PROMPT_PREFIX)
                    else "primary-model",
                    current_ordinary_tree(),
                )
            )
            self.payloads.append(messages)
            if system.startswith(SUBAGENT_PROMPT_PREFIX):
                owner = current_execution_owner()
                root = trees[0][1]
                with root.lock:
                    original_member = owner in root.members
                with h.runtime.execution_capacity._lock:
                    model_owned = bool(
                        owner and any(op.kind == "model" for op in owner._operations)
                    )
                child_model_owners.append(
                    (bool(owner and owner.child), original_member, model_owned)
                )
                self.child_calls += 1
                yield "child completed" if self.child_calls == 1 else "child resumed"
                return
            self.parent_calls += 1
            if self.parent_calls == 1:
                yield fence(SPAWN_TOOL_NAME, {"task": "inspect the original"})
            elif self.parent_calls in (2, 4):
                yield fence(WAIT_AGENTS_TOOL_NAME, {})
            elif self.parent_calls == 3:
                coordinator = h.bridge._fleet_coordinators[
                    h.session.persisted_conversation_id
                ]
                handle = next(item for item in coordinator.snapshot() if item.run_id)
                yield fence(
                    SEND_TO_AGENT_TOOL_NAME,
                    {"id": handle.handle_id, "message": "check again"},
                )
            else:
                yield "parent completed"

    gateway = Gateway()
    h.controller.provider_gateway = gateway
    h.bridge._gateway = gateway
    result = await h.controller.submit_draft(
        "inspect and recheck", session_id=h.session.id
    )
    assert result.accepted
    await until(lambda: not h.runtime.execution_capacity.snapshot().executions)
    rows = [
        row
        for row in h.database.list_runs(h.session.persisted_conversation_id)
        if row["agent_kind"] == "subagent"
    ]
    resumed = [row for row in rows if row.get("resumed_from_run_id")]
    assert len(rows) == 2 and len(resumed) == 1
    assert resumed[0]["resumed_from_run_id"] in {row["id"] for row in rows}
    assert gateway.child_calls == 2 and children == [True, True]
    assert child_model_owners == [(True, True, True), (True, True, True)]
    assert trees and trees[0][1] is not None
    assert all(tree is trees[0][1] for kind, tree in trees if kind != "child-model")


async def test_surviving_child_retains_original_claim_across_next_send(
    ordinary_host, monkeypatch
):
    from Tests.Agents.test_agent_service import fence
    from Tests.Chat.test_console_agent_bridge import SUBAGENT_PROMPT_PREFIX
    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from tldw_chatbook.Agents.agent_models import SPAWN_TOOL_NAME
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree

    h = ordinary_host
    monkeypatch.setenv("TLDW_AGENTS_MAX_LIVE_SUBAGENTS", "1")
    monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "false")
    h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    entered, release = threading.Event(), threading.Event()
    trees, child_owners = [], []

    class Gateway(_RecordingGateway):
        def __init__(self):
            super().__init__()
            self.parent_calls = 0

        async def stream_chat(self, resolution, messages, **kwargs):
            system = str(messages[0].get("content", "")) if messages else ""
            self.payloads.append(messages)
            if system.startswith(SUBAGENT_PROMPT_PREFIX):
                child_owners.append(current_execution_owner())
                entered.set()
                if not await asyncio.to_thread(release.wait, 20):
                    raise RuntimeError("surviving child was not released")
                yield "survivor completed"
                return
            trees.append(current_ordinary_tree())
            self.parent_calls += 1
            yield (
                fence(SPAWN_TOOL_NAME, {"task": "survive the parent"})
                if self.parent_calls == 1
                else "parent completed"
            )

    gateway = Gateway()
    h.controller.provider_gateway = gateway
    h.bridge._gateway = gateway
    first = asyncio.create_task(
        h.controller.submit_draft("start survivor", session_id=h.session.id)
    )
    h.tasks.append(first)
    try:
        await until(lambda: entered.is_set() or first.done())
        assert entered.is_set(), (
            first.result() if first.done() else "child did not enter"
        )
        assert (await asyncio.wait_for(first, 10)).accepted
        tree = trees[0]
        assert tree is not None and not tree.retired.done() and child_owners
        with tree.lock:
            assert child_owners[0] in tree.members and child_owners[0].child
        calls = len(gateway.payloads)
        h.store.set_session_draft(h.session.id, "conflicting next draft")
        second = await h.controller.submit_draft(
            "conflicting next draft", session_id=h.session.id
        )
        assert not second.accepted
        assert len(gateway.payloads) == calls
        assert h.store.session_draft(h.session.id) == "conflicting next draft"
    finally:
        release.set()
    await until(lambda: tree.retired.done())
    rows = [
        row
        for row in h.database.list_runs(h.session.persisted_conversation_id)
        if row["agent_kind"] == "subagent"
    ]
    assert len(rows) == 1 and rows[0]["status"] == "done"
    assert not h.runtime.execution_capacity.snapshot().executions


async def test_late_subagent_stop_retains_original_tree_until_physical_return(
    ordinary_host, monkeypatch
):
    import os
    import subprocess
    import sys
    import traceback

    if os.environ.get("TASK8_SUBAGENT_STOP_PEER") != "1":
        env = dict(os.environ, TASK8_SUBAGENT_STOP_PEER="1")
        result = await asyncio.to_thread(
            subprocess.run,
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-s",
                f"{__file__}::test_late_subagent_stop_retains_original_tree_until_physical_return",
                "--basetemp=" + str(ordinary_host.tmp_path / "late-subagent-peer"),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=35,
        )
        (ordinary_host.tmp_path / "late-subagent-peer.log").write_text(
            result.stdout + result.stderr
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "1 passed" in result.stdout
        return

    try:
        from Tests.Agents.test_agent_service import fence
        from Tests.Agents.test_hooks_v2_execution import command
        from Tests.Chat.test_console_agent_bridge import SUBAGENT_PROMPT_PREFIX
        from Tests.Chat.test_console_skill_substitution import _RecordingGateway
        from tldw_chatbook.Agents.agent_models import SPAWN_TOOL_NAME
        from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree
        from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

        h = ordinary_host
        monkeypatch.setenv("TLDW_AGENTS_MAX_LIVE_SUBAGENTS", "1")
        monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "false")
        h.runtime.execution_capacity.set_tool_limits(2, 0)
        h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
        child_entered, child_release = threading.Event(), threading.Event()
        hook_entered, hook_release = (
            h.tmp_path / "late-child-stop-entered",
            h.tmp_path / "late-child-stop-release",
        )
        code = (
            "import time;from pathlib import Path;"
            f"Path({str(hook_entered)!r}).touch();"
            f"exec({f'while not Path({str(hook_release)!r}).exists(): time.sleep(0.01)'!r})"
        )
        engine = h.runtime.ensure_hooks_v2(
            h.session.id,
            (command(code, name="SubagentStop", timeout_seconds=30),),
            lambda *_: True,
        )
        trees = []

        class Gateway(_RecordingGateway):
            def __init__(self):
                super().__init__()
                self.parent_calls = 0

            async def stream_chat(self, resolution, messages, **kwargs):
                system = str(messages[0].get("content", "")) if messages else ""
                self.payloads.append(messages)
                if system.startswith(SUBAGENT_PROMPT_PREFIX):
                    child_entered.set()
                    if not await asyncio.to_thread(child_release.wait, 20):
                        raise RuntimeError("late child was not released")
                    yield "child completed"
                    return
                trees.append(current_ordinary_tree())
                self.parent_calls += 1
                yield (
                    fence(SPAWN_TOOL_NAME, {"task": "late stop child"})
                    if self.parent_calls == 1
                    else "parent completed"
                )

        gateway = Gateway()
        h.controller.provider_gateway = gateway
        h.bridge._gateway = gateway
        task = asyncio.create_task(
            h.controller.submit_draft("start late child", session_id=h.session.id)
        )
        h.tasks.append(task)
        try:
            await until(lambda: child_entered.is_set() or task.done())
            assert child_entered.is_set(), (
                task.result() if task.done() else "child did not enter"
            )
            assert (await asyncio.wait_for(task, 10)).accepted
            child_release.set()
            await until(hook_entered.exists)
            tree = trees[0]
            assert tree is not None and not tree.retired.done()
            with h.runtime.worktree_activity.prelaunch(
                footprint=WorktreeFootprint(opaque=True)
            ) as claim:
                assert claim is None
            calls = len(gateway.payloads)
            h.store.set_session_draft(h.session.id, "late stop next draft")
            result = await h.controller.submit_draft(
                "late stop next draft", session_id=h.session.id
            )
            assert not result.accepted and len(gateway.payloads) == calls
            assert h.store.session_draft(h.session.id) == "late stop next draft"
        finally:
            child_release.set()
            hook_release.touch()
        await until(lambda: tree.retired.done())
        await asyncio.wait_for(h.runtime.close_hooks_v2(h.session.id), timeout=10)
        assert not engine._native_commands.cleanup_pending
        rows = [
            row
            for row in h.database.list_runs(h.session.persisted_conversation_id)
            if row["agent_kind"] == "subagent"
        ]
        assert len(rows) == 1 and rows[0]["status"] == "done"
        assert not h.runtime.execution_capacity.snapshot().executions
    except BaseException:  # noqa: BLE001 - isolated peer reports cancellation and exits
        traceback.print_exc(file=sys.__stderr__)
        sys.__stderr__.flush()
        os._exit(1)


@pytest.mark.parametrize("onboarded", [False, True])
async def test_original_stop_hook_holds_same_ordinary_tree(ordinary_host, onboarded):
    from Tests.Agents.test_hooks_v2_execution import command
    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from tldw_chatbook import config
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree
    from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest
    from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

    h = ordinary_host
    if onboarded:
        h.app.app_config["console"] = {"onboarding": {"first_send_completed": True}}
    entered, release = h.tmp_path / "stop-entered", h.tmp_path / "stop-release"
    code = (
        "import time;from pathlib import Path;"
        f"Path({str(entered)!r}).touch();"
        f"exec({f'while not Path({str(release)!r}).exists(): time.sleep(0.01)'!r})"
    )
    engine = h.runtime.ensure_hooks_v2(
        h.session.id,
        (command(code, name="Stop", timeout_seconds=30),),
        lambda *_: True,
    )
    trees = []

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            trees.append(current_ordinary_tree())
            self.payloads.append(messages)
            yield "original reply"

    h.controller.provider_gateway = h.bridge._gateway = Gateway()
    h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    configuration = await h.controller.capture_turn_configuration_snapshot(h.session.id)
    identity = config.current_config_identity()
    request = ConsoleTurnCustodyRequest(
        turn_id="held-original-stop",
        session_id=h.session.id,
        draft="hold the original Stop event",
        configuration=configuration,
    )
    turn_id = h.runtime.accept_turn(request)
    task = h.runtime._turn_custody[turn_id].task
    h.tasks.append(task)
    try:
        await until(lambda: entered.exists() or task.done())
        assert entered.exists(), task.result() if task.done() else "Stop did not enter"
        assert trees and trees[0] is not None and not trees[0].retired.done()
        assert config.current_config_identity() == identity
        with h.runtime.worktree_activity.prelaunch(
            footprint=WorktreeFootprint(opaque=True)
        ) as claim:
            assert claim is None
    finally:
        release.touch()
    assert (await asyncio.wait_for(asyncio.shield(task), 10)).accepted
    await asyncio.wait_for(h.runtime.close_hooks_v2(h.session.id), 12)
    await until(lambda: trees[0].retired.done())
    assert not engine._native_commands.cleanup_pending
    if not onboarded:
        await asyncio.wait_for(asyncio.shield(h.runtime._first_send_task), 10)
        assert h.app.app_config["console"]["onboarding"]["first_send_completed"] is True


from Tests.Chat import test_console_async_mcp_snapshot as snapshot_controls

catalog_store = snapshot_controls.catalog_store
mcp_sources = snapshot_controls.mcp_sources
local_root = snapshot_controls.local_root
snapshot_case = snapshot_controls.snapshot_case


@pytest.fixture
async def native_ordinary_host(snapshot_case, tmp_path_factory, monkeypatch):
    from types import SimpleNamespace

    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from Tests.console_provider_doubles import persisted_console_store
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Tools import workspace_file_roots
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    root = tmp_path_factory.mktemp("task8-native-project")
    project = root / "project"
    project.mkdir()
    registry = LocalWorkspaceRegistryService(WorkspaceDB(root / "workspace.db"))
    registry.ensure_default_workspace()
    registry.create_workspace(workspace_id="native-received", name="Native")
    registry.add_folder_binding("native-received", project)
    registry.set_active_workspace("native-received")
    monkeypatch.setattr(workspace_file_roots, "_registry_factory", lambda: registry)
    monkeypatch.setattr(workspace_file_roots, "_LAUNCH_CWD", str(project))
    store = persisted_console_store(
        db_path=root / "chat.db", workspace_registry=registry
    )
    session = store.create_session(workspace_id="native-received")
    gateway = _RecordingGateway()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, provider="llama_cpp", model="m"
    )
    app = SimpleNamespace(
        workspace_registry_service=registry,
        app_config={},
        chachanotes_db=store.persistence.db,
        unified_mcp_service=snapshot_case.service,
    )
    app.local_chat_conversation_service = ChatConversationService(store.persistence.db)
    runtime = app.console_runtime = ConsoleRuntime(app)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    database = AgentRunsDB(root / "runs.db", client_id="received-native")
    bridge = ConsoleAgentBridge(
        agent_runs_db=database,
        store=store,
        provider_gateway=gateway,
        runtime_capacity=runtime.execution_capacity,
        get_hooks_v2=runtime.get_hooks_v2,
    )
    runtime.set_agent_bridge(bridge)
    controller._agent_bridge, controller._agent_runtime_enabled = bridge, True
    tasks = []
    host = SimpleNamespace(**locals())
    try:
        with workspace_file_roots.run_file_sandbox(project):
            yield host
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        await runtime.dispose()
        database.close()
        store.persistence.db.close_connection()
        registry.db.close()


@pytest.mark.parametrize("replace_metadata", [False, True])
async def test_actual_received_turn_with_native_killed_mcp_is_bounded(
    snapshot_case, native_ordinary_host, monkeypatch, replace_metadata
):
    from dataclasses import replace

    import tldw_chatbook.MCP.server  # noqa: F401 - load the original closed manifest source
    from Tests.Chat.test_console_received_intent_custody import _intent
    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree
    from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest
    from tldw_chatbook.MCP.local_control_service import _default_manifest_provider

    snapshot_case.local.policy_enforcer = None
    snapshot_case.local.manifest_provider = _default_manifest_provider
    snapshot_case.permissions.set_kill_switch(True)
    h = native_ordinary_host
    h.app.unified_mcp_service = snapshot_case.service
    trees = []

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            trees.append(current_ordinary_tree())
            self.payloads.append(messages)
            yield "received reply"

    gateway = Gateway()
    h.controller.provider_gateway = gateway
    h.bridge._gateway = gateway
    h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    h.store.set_session_draft(h.session.id, "received original draft")
    configuration = await h.controller.capture_turn_configuration_snapshot(h.session.id)
    request = ConsoleTurnCustodyRequest(
        turn_id="native-received",
        session_id=h.session.id,
        draft="received original draft",
        configuration=configuration,
    )
    intent = _intent(
        (h.runtime, h.store, h.session, request), turn_id="native-received"
    )
    tools = dict(configuration.tool_configuration)
    tools["agent_run_budget_maximum"] = replace(
        tools["agent_run_budget_maximum"], max_subagents=0
    )
    intent = replace(
        intent,
        selection=replace(
            intent.selection,
            agent_runtime_enabled=True,
            tool_configuration=tools,
        ),
    )
    if replace_metadata:
        original_run = h.runtime._run_custodied_turn

        async def changed(record, **kwargs):
            record.request = replace(
                record.request,
                configuration=replace(
                    record.request.configuration,
                    mcp_definition_complete=False,
                ),
            )
            return await original_run(record, **kwargs)

        monkeypatch.setattr(h.runtime, "_run_custodied_turn", changed)
    turn_id = h.runtime.accept_received_intent(intent)
    task = h.runtime._turn_custody[turn_id].task
    h.tasks.append(task)
    if replace_metadata:
        with pytest.raises(RuntimeError, match="Hook review source or session changed"):
            await task
        assert (
            not trees
            and h.store.session_draft(h.session.id) == "received original draft"
        )
    else:
        assert (await task).accepted
        assert trees and all(
            tree is trees[0] and not tree.footprint.opaque for tree in trees
        )


async def test_busy_received_turn_preserves_original_draft_and_attachment(
    ordinary_host,
):
    from dataclasses import replace

    from Tests.Chat.test_console_received_intent_custody import _intent
    from tldw_chatbook.Chat.attachment_core import PendingAttachment
    from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest
    from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

    h = ordinary_host
    draft = "original received input"
    h.store.set_session_draft(h.session.id, draft)
    h.store.set_session_one_shot_prefill(h.session.id, "original prefill")
    attachment = PendingAttachment(
        file_path="original.txt",
        display_name="original.txt",
        file_type="text",
        insert_mode="inline",
        data=b"original bytes",
        attachment_id="original-attachment",
    )
    assert h.store.add_pending_attachment(h.session.id, attachment)
    configuration = await h.controller.capture_turn_configuration_snapshot(h.session.id)
    request = ConsoleTurnCustodyRequest(
        turn_id="busy-received",
        session_id=h.session.id,
        draft=draft,
        configuration=configuration,
    )
    intent = _intent((h.runtime, h.store, h.session, request), turn_id="busy-received")
    intent = replace(
        intent, selection=replace(intent.selection, agent_runtime_enabled=True)
    )
    with h.runtime.worktree_activity.prelaunch(
        footprint=WorktreeFootprint(opaque=True)
    ) as blocker:
        assert blocker is not None
        turn_id = h.runtime.accept_received_intent(intent)
        task = h.runtime._turn_custody[turn_id].task
        h.tasks.append(task)
        with pytest.raises(RuntimeError, match="Console turn was refused"):
            await task
        assert h.store.session_draft(h.session.id) == draft
        assert h.store.session_one_shot_prefill(h.session.id) == "original prefill"
        pending = h.store.pending_attachments(h.session.id)
        assert len(pending) == 1 and pending[0].attachment_id == "original-attachment"
        assert pending[0].data == b"original bytes"
        assert not h.gateway.payloads and h.session.persisted_conversation_id is None


async def test_original_native_captured_direct_turn_with_killed_mcp_is_bounded(
    snapshot_case, native_ordinary_host
):
    from dataclasses import replace

    import tldw_chatbook.MCP.server  # noqa: F401 - load the original closed manifest source
    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree
    from tldw_chatbook.MCP.local_control_service import _default_manifest_provider

    snapshot_case.local.policy_enforcer = None
    snapshot_case.local.manifest_provider = _default_manifest_provider
    snapshot_case.permissions.set_kill_switch(True)
    h = native_ordinary_host
    seen = []

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            seen.append(current_ordinary_tree())
            self.payloads.append(messages)
            yield "direct native reply"

    gateway = Gateway()
    h.controller.provider_gateway = gateway
    h.bridge._gateway = gateway
    h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    configuration = await h.controller.capture_turn_configuration_snapshot(h.session.id)
    tools = dict(configuration.tool_configuration)
    tools["agent_run_budget_maximum"] = replace(
        tools["agent_run_budget_maximum"], max_subagents=0
    )
    configuration = replace(configuration, tool_configuration=tools)
    task = asyncio.create_task(
        h.controller.submit_draft(
            "direct original", session_id=h.session.id, configuration=configuration
        )
    )
    h.tasks.append(task)
    assert (await task).accepted
    assert seen and all(tree is seen[0] and not tree.footprint.opaque for tree in seen)


async def test_custom_mcp_manifest_waits_for_original_ordinary_admission(
    snapshot_case, native_ordinary_host, monkeypatch
):
    from dataclasses import replace

    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree

    snapshot_controls._loop_projection(snapshot_case)
    h = native_ordinary_host
    calls, trees, projection_reads = [], [], []
    original_load = snapshot_case.permissions.load

    def observed_load():
        import inspect

        if inspect.currentframe().f_back.f_code.co_name == "_permission_payload":
            projection_reads.append(
                (
                    threading.current_thread(),
                    current_execution_owner(),
                    current_ordinary_tree(),
                )
            )
        return original_load()

    monkeypatch.setattr(snapshot_case.permissions, "load", observed_load)

    def original_manifest():
        calls.append((current_execution_owner(), current_ordinary_tree()))
        return {"tools": []}

    snapshot_case.local.manifest_provider = original_manifest

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            trees.append(current_ordinary_tree())
            self.payloads.append(messages)
            yield "original custom manifest reply"

    gateway = Gateway()
    h.controller.provider_gateway = gateway
    h.bridge._gateway = gateway
    h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    configuration = await h.controller.capture_turn_configuration_snapshot(h.session.id)
    assert not calls, "custom manifest ran before ordinary admission"
    tools = dict(configuration.tool_configuration)
    tools["agent_run_budget_maximum"] = replace(
        tools["agent_run_budget_maximum"], max_subagents=0
    )
    task = asyncio.create_task(
        h.controller.submit_draft(
            "use original custom source",
            session_id=h.session.id,
            configuration=replace(configuration, tool_configuration=tools),
        )
    )
    h.tasks.append(task)
    assert (await task).accepted
    assert trees and calls
    assert all(owner is trees[0].owner and tree is trees[0] for owner, tree in calls), (
        calls
    )
    assert trees[0].footprint.opaque
    assert projection_reads and all(
        thread is not threading.main_thread()
        and owner is trees[0].owner
        and tree is trees[0]
        for thread, owner, tree in projection_reads
    ), projection_reads


async def test_deferred_mcp_worker_and_finish_survive_zero_grace_disposal(
    snapshot_case, native_ordinary_host, monkeypatch
):
    from dataclasses import replace

    from tldw_chatbook.Agents.execution_capacity import ExecutionOwner
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree
    from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

    snapshot_controls._loop_projection(snapshot_case)
    h = native_ordinary_host
    h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    configuration = await h.controller.capture_turn_configuration_snapshot(h.session.id)
    tools = dict(configuration.tool_configuration)
    tools["agent_run_budget_maximum"] = replace(
        tools["agent_run_budget_maximum"], max_subagents=0
    )
    finishes, held_return = [], []
    finish_entered, finish_release = threading.Event(), threading.Event()
    disposal_ref = []
    reserve = ExecutionOwner.reserve_tool

    def observed_reserve(owner):
        operation = reserve(owner)
        finish = operation.finish

        def observed_finish():
            tree = current_ordinary_tree()
            if tree is not None:
                with h.runtime.worktree_activity.prelaunch(
                    footprint=WorktreeFootprint(opaque=True)
                ) as other:
                    finishes.append((tree, current_execution_owner(), other))
            result = finish()
            if tree is not None and not finish_entered.is_set():
                finish_entered.set()
                assert finish_release.wait(10)
            return result

        operation.finish = observed_finish
        return operation

    monkeypatch.setattr(ExecutionOwner, "reserve_tool", observed_reserve)

    def inspect_held_return():
        try:
            if finish_entered.wait(8):
                tree = finishes[-1][0]
                with h.runtime.worktree_activity.prelaunch(
                    footprint=WorktreeFootprint(opaque=True)
                ) as other:
                    held_return.append(
                        (
                            not disposal_ref[0].done(),
                            other,
                            not tree.retired.done(),
                            tree.source_scopes > 0,
                        )
                    )
        finally:
            finish_release.set()

    observer = threading.Thread(target=inspect_held_return)
    observer.start()
    probe = snapshot_controls._MaximumProbe(
        snapshot_case.source, snapshot_case.permissions, hold=True
    )
    with probe.installed():
        task = asyncio.create_task(
            h.controller.submit_draft(
                "hold original metadata",
                session_id=h.session.id,
                configuration=replace(configuration, tool_configuration=tools),
            )
        )
        h.tasks.append(task)
        disposal = None
        try:
            await until(lambda: probe.entered.is_set() or task.done())
            assert probe.entered.is_set(), (
                task.result() if task.done() else "no original metadata read"
            )
            task.cancel()
            disposal = asyncio.create_task(h.runtime.dispose(timeout_seconds=0))
            disposal_ref.append(disposal)
            await until(lambda: h.runtime._disposed)
            assert not disposal.done() and not task.done()
            with h.runtime.worktree_activity.prelaunch(
                footprint=WorktreeFootprint(opaque=True)
            ) as claim:
                assert claim is None
        finally:
            probe.release.set()
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.to_thread(observer.join, 10)
            if disposal is not None:
                await disposal
    assert finishes and all(
        owner is tree.owner and claim is None for tree, owner, claim in finishes
    )
    assert held_return == [(True, None, True, True)]


async def _exercise_pending_mcp_queue(
    snapshot_case, native_ordinary_host, monkeypatch, busy
):
    from dataclasses import replace

    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree

    snapshot_controls._loop_projection(snapshot_case)
    h = native_ordinary_host
    h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    calls, models = [], []
    first_started, release = threading.Event(), threading.Event()

    def original_manifest():
        calls.append((current_execution_owner(), current_ordinary_tree()))
        return {"tools": []}

    snapshot_case.local.manifest_provider = original_manifest

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            models.append((current_ordinary_tree(), messages))
            self.payloads.append(messages)
            if len(models) == 1:
                first_started.set()
                await asyncio.to_thread(release.wait)
            yield "original queue reply"

    gateway = Gateway()
    h.controller.provider_gateway = gateway
    h.bridge._gateway = gateway
    witness_seen = []
    if busy:
        from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

        submit_queued = h.controller.prompt_queue_coordinator._submit_queued

        async def refuse_once(prompt, **kwargs):
            witness_seen.append(prompt.mcp_capture_witness)
            if len(witness_seen) == 1:
                with h.runtime.worktree_activity.prelaunch(
                    footprint=WorktreeFootprint(opaque=True)
                ) as blocker:
                    assert blocker is not None
                    return await submit_queued(prompt, **kwargs)
            return await submit_queued(prompt, **kwargs)

        monkeypatch.setattr(
            h.controller.prompt_queue_coordinator, "_submit_queued", refuse_once
        )

    async def initial():
        configuration = await h.controller.capture_turn_configuration_snapshot(
            h.session.id
        )
        tools = dict(configuration.tool_configuration)
        tools["agent_run_budget_maximum"] = replace(
            tools["agent_run_budget_maximum"], max_subagents=0
        )
        return await h.controller.submit_draft(
            "first",
            session_id=h.session.id,
            configuration=replace(configuration, tool_configuration=tools),
        )

    task = asyncio.create_task(
        h.controller.run_prompt_chain(session_id=h.session.id, initial_turn=initial)
    )
    h.tasks.append(task)
    try:
        await until(first_started.is_set)
        snapshot = h.controller.prompt_queue_registry.snapshot(h.session.id)
        admitted = await h.controller.queue_prompt(
            h.session.id, text="queued original", expected_revision=snapshot.revision
        )
        assert admitted.applied
        edited = h.controller.prompt_queue_registry.edit(
            h.session.id,
            entry_id=admitted.entry_id,
            text="queued edited",
            expected_revision=admitted.snapshot.revision,
        )
        assert edited.applied
        release.set()
        done, _ = await asyncio.wait({task}, timeout=6)
        if not done:
            for pending in asyncio.all_tasks():
                print(
                    "QUEUE_PENDING", pending.get_name(), pending.get_coro(), flush=True
                )
                pending.print_stack()
        assert task in done, "original prompt chain did not return after MODEL"
        await task
        snapshot = h.controller.prompt_queue_registry.snapshot(h.session.id)
        if busy:
            assert (
                len(models) == 1
                and snapshot.waiting_count == 1
                and snapshot.claimed_count == 0
            )
            await h.controller.prompt_queue_coordinator.resume_and_drain(h.session.id)
            assert (
                len(witness_seen) == 2
                and witness_seen[0] is witness_seen[1]
                and witness_seen[0] is not None
            )
            snapshot = h.controller.prompt_queue_registry.snapshot(h.session.id)
        if len(models) != 2:
            print(
                "QUEUE_DIAGNOSTIC",
                snapshot.mode,
                snapshot.pause_reason,
                snapshot.waiting_count,
                snapshot.claimed_count,
                h.controller._run_states.get(h.session.id),
                flush=True,
            )
        assert len(models) == 2 and "queued edited" in str(models[1][1])
        assert calls and all(
            tree is not None and owner is tree.owner for owner, tree in calls
        )
        assert all(tree.footprint.opaque for tree, _messages in models)
    finally:
        release.set()


@pytest.mark.parametrize("busy", [False, True])
async def test_pending_mcp_queue_edit_retains_original_source_until_dispatch(
    snapshot_case, native_ordinary_host, monkeypatch, busy
):
    import os
    import traceback

    try:
        await _exercise_pending_mcp_queue(
            snapshot_case, native_ordinary_host, monkeypatch, busy
        )
    except BaseException:
        if os.environ.get("TASK8_QUEUE_DIAGNOSTIC_PEER") == "1":
            traceback.print_exc()
            import sys

            sys.stderr.flush()
            os._exit(1)
        raise


async def test_pending_mcp_capture_is_opaque_even_when_prefill_skips_agent(
    snapshot_case, native_ordinary_host
):
    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree

    snapshot_controls._loop_projection(snapshot_case)
    h = native_ordinary_host
    h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    h.store.set_session_one_shot_prefill(h.session.id, "original prefill")
    models = []

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            models.append(current_ordinary_tree())
            self.payloads.append(messages)
            yield "prefill reply"

    gateway = Gateway()
    h.controller.provider_gateway = gateway
    h.bridge._gateway = gateway
    configuration = await h.controller.capture_turn_configuration_snapshot(h.session.id)
    assert configuration.mcp_capture_pending
    prefill, revision = h.store.session_one_shot_prefill_snapshot(h.session.id)
    result = await h.controller.submit_draft(
        "prefill turn",
        session_id=h.session.id,
        configuration=configuration,
        captured_one_shot_prefill=prefill,
        captured_one_shot_prefill_revision=revision,
    )
    assert result.accepted
    assert models and all(
        tree is models[0] and tree.footprint.opaque for tree in models
    )


async def test_deferred_native_mcp_permission_lazy_publication_keeps_original_source(
    snapshot_case, native_ordinary_host
):
    import os
    import sys
    import traceback
    from dataclasses import replace

    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )

    try:
        snapshot_controls._loop_projection(snapshot_case)
        h = native_ordinary_host
        original_service = UnifiedMCPControlPlaneService(
            target_store=None,
            context_store=None,
            local_service=snapshot_case.local,
            server_service=None,
        )
        assert original_service._permission_store is None
        h.app.unified_mcp_service = original_service
        h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
        trees = []

        class Gateway(_RecordingGateway):
            async def stream_chat(self, resolution, messages, **kwargs):
                trees.append(current_ordinary_tree())
                self.payloads.append(messages)
                yield "cold original source reply"

        gateway = Gateway()
        h.controller.provider_gateway = gateway
        h.bridge._gateway = gateway
        configuration = await h.controller.capture_turn_configuration_snapshot(
            h.session.id
        )
        assert (
            configuration.mcp_capture_pending
            and original_service._permission_store is None
        )
        tools = dict(configuration.tool_configuration)
        tools["agent_run_budget_maximum"] = replace(
            tools["agent_run_budget_maximum"], max_subagents=0
        )
        result = await h.controller.submit_draft(
            "cold original source",
            session_id=h.session.id,
            configuration=replace(configuration, tool_configuration=tools),
        )
        assert result.accepted and trees
        assert original_service._permission_store is trees[0].source.permission
    except BaseException:
        if os.environ.get("TASK8_COLD_DIAGNOSTIC_PEER") == "1":
            traceback.print_exc()
            sys.stderr.flush()
            os._exit(1)
        raise


async def test_pending_mcp_without_original_task_context_refuses_before_effects(
    snapshot_case, native_ordinary_host
):
    from contextvars import Context

    snapshot_controls._loop_projection(snapshot_case)
    h = native_ordinary_host
    h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    h.store.set_session_draft(h.session.id, "original unsent draft")
    calls = []

    def original_manifest():
        calls.append(True)
        return {"tools": []}

    snapshot_case.local.manifest_provider = original_manifest
    configuration = await h.controller.capture_turn_configuration_snapshot(h.session.id)
    assert configuration.mcp_capture_pending and not calls
    task = asyncio.create_task(
        h.controller.submit_draft(
            "original unsent draft",
            session_id=h.session.id,
            configuration=configuration,
        ),
        context=Context(),
    )
    h.tasks.append(task)
    with pytest.raises(
        RuntimeError, match="Original MCP capture source is unavailable"
    ):
        await task
    assert not calls and not h.gateway.payloads
    assert h.store.session_draft(h.session.id) == "original unsent draft"
    assert h.session.persisted_conversation_id is None


async def test_pending_cold_mcp_receiver_replacement_refuses_before_callback(
    snapshot_case, native_ordinary_host
):
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )

    snapshot_controls._loop_projection(snapshot_case)
    h = native_ordinary_host
    service = UnifiedMCPControlPlaneService(
        target_store=None,
        context_store=None,
        local_service=snapshot_case.local,
        server_service=None,
    )
    assert service._permission_store is None
    h.app.unified_mcp_service = service
    h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    h.store.set_session_draft(h.session.id, "original cold draft")
    calls = []

    def original_manifest():
        calls.append(True)
        return {"tools": []}

    snapshot_case.local.manifest_provider = original_manifest
    configuration = await h.controller.capture_turn_configuration_snapshot(h.session.id)
    assert configuration.mcp_capture_pending and not calls
    service._permission_store = object()
    with pytest.raises(
        RuntimeError, match="Original MCP capture source is unavailable"
    ):
        await h.controller.submit_draft(
            "original cold draft", session_id=h.session.id, configuration=configuration
        )
    assert not calls and not h.gateway.payloads
    assert h.store.session_draft(h.session.id) == "original cold draft"
    assert h.session.persisted_conversation_id is None


@pytest.mark.parametrize("replacement", ["path", "manifest"])
async def test_pending_mcp_original_path_or_factory_replacement_refuses(
    snapshot_case, native_ordinary_host, replacement
):
    import os
    import sys
    import traceback

    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )

    try:
        snapshot_controls._loop_projection(snapshot_case)
        h = native_ordinary_host
        service = UnifiedMCPControlPlaneService(
            target_store=None,
            context_store=None,
            local_service=snapshot_case.local,
            server_service=None,
        )
        h.app.unified_mcp_service = service
        h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
        h.store.set_session_draft(h.session.id, "original path draft")
        calls = []

        def manifest():
            calls.append(True)
            return {"tools": []}

        snapshot_case.local.manifest_provider = manifest
        configuration = await h.controller.capture_turn_configuration_snapshot(
            h.session.id
        )
        assert configuration.mcp_capture_pending and not calls
        if replacement == "path":
            snapshot_case.source.path = snapshot_case.source.path.with_name(
                "other-local-mcp.json"
            )
        else:
            snapshot_case.local.manifest_provider = lambda: calls.append("replacement")
        try:
            result = await h.controller.submit_draft(
                "original path draft",
                session_id=h.session.id,
                configuration=configuration,
            )
        except (RuntimeError, PermissionError):
            pass
        else:
            assert not result.accepted
        assert not calls and not h.gateway.payloads
        assert h.store.session_draft(h.session.id) == "original path draft"
        assert h.session.persisted_conversation_id is None
        if os.environ.get("TASK8_REPLACEMENT_PEER") == "1":
            print("REPLACEMENT_OK", replacement, flush=True)
            os._exit(0)
    except BaseException:
        if os.environ.get("TASK8_REPLACEMENT_PEER") == "1":
            traceback.print_exc()
            sys.stderr.flush()
            os._exit(1)
        raise


async def test_pending_mcp_manifest_changed_defaults_refuses_before_callback(
    snapshot_case, native_ordinary_host
):
    snapshot_controls._loop_projection(snapshot_case)
    h = native_ordinary_host
    h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    h.store.set_session_draft(h.session.id, "original callback draft")
    calls = []

    def manifest(value="original"):
        calls.append(value)
        return {"tools": []}

    snapshot_case.local.manifest_provider = manifest
    configuration = await h.controller.capture_turn_configuration_snapshot(h.session.id)
    assert configuration.mcp_capture_pending and not calls
    manifest.__defaults__ = ("replacement",)
    with pytest.raises(
        RuntimeError, match="Original MCP capture source is unavailable"
    ):
        await h.controller.submit_draft(
            "original callback draft",
            session_id=h.session.id,
            configuration=configuration,
        )
    assert not calls and not h.gateway.payloads
    assert h.store.session_draft(h.session.id) == "original callback draft"
    assert h.session.persisted_conversation_id is None


async def test_accepted_voice_uses_original_pending_mcp_source(
    snapshot_case, native_ordinary_host
):
    from dataclasses import replace

    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from Tests.Chat.test_console_voice_effect_barrier import _context
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree

    snapshot_controls._loop_projection(snapshot_case)
    h = native_ordinary_host
    h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    calls, models = [], []

    def original_manifest():
        calls.append((current_execution_owner(), current_ordinary_tree()))
        return {"tools": []}

    snapshot_case.local.manifest_provider = original_manifest

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            models.append((current_ordinary_tree(), messages))
            self.payloads.append(messages)
            yield "voice reply"

    gateway = Gateway()
    h.controller.provider_gateway = gateway
    h.bridge._gateway = gateway
    configuration = await h.controller.capture_turn_configuration_snapshot(h.session.id)
    tools = dict(configuration.tool_configuration)
    tools["agent_run_budget_maximum"] = replace(
        tools["agent_run_budget_maximum"], max_subagents=0
    )
    context = replace(
        _context(), configuration=replace(configuration, tool_configuration=tools)
    )
    turn_id = h.controller.submit_accepted_voice_turn("exact accepted voice", context)
    task = h.runtime._turn_custody[turn_id].task
    h.tasks.append(task)
    assert (await task).accepted
    assert models and "exact accepted voice" in str(models[0][1])
    assert calls and all(
        tree is models[0][0] and owner is tree.owner for owner, tree in calls
    )
    assert models[0][0].footprint.opaque


async def test_unwitnessed_permission_reader_never_runs_before_ordinary_claim(
    ordinary_host,
):
    from dataclasses import replace

    from Tests.Chat.test_console_local_review_hook import _FakeService
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree

    h = ordinary_host
    h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    observed = []

    class CustomPermission:
        def load(self):
            owner = current_execution_owner()
            tree = current_ordinary_tree()
            observed.append((owner, tree, tree.claim if tree else None))
            if owner is None:
                raise RuntimeError("custom permission read before admission")
            h.runtime.worktree_activity.validate_claim(
                tree.claim, owner=owner, footprint=tree.footprint
            )
            return {"kill_switch": False, "profiles": {}}

    service = _FakeService()
    service._permission_store = CustomPermission()
    h.app.unified_mcp_service = service
    configuration = await h.controller.capture_turn_configuration_snapshot(h.session.id)
    configuration = replace(configuration, mcp_capture_pending=False)
    await h.controller.submit_draft(
        "original permission", session_id=h.session.id, configuration=configuration
    )
    assert observed
    first_owner, first_tree, first_claim = observed[0]
    assert all(
        owner is first_owner
        and tree is first_tree
        and claim is first_claim
        and owner is tree.owner
        and tree.footprint.opaque
        for owner, tree, claim in observed
    ), observed


async def test_public_mcp_maximum_still_returns_original_custom_hashes(snapshot_case):
    from tldw_chatbook.MCP.console_snapshot import capture_console_definition_maximum

    snapshot_controls._loop_projection(snapshot_case)
    result = await capture_console_definition_maximum(
        snapshot_case.service, frozenset()
    )
    assert "local:one::first" in result
    assert "builtin:tldw_chatbook::sample_builtin" in result


async def test_stale_managed_mcp_runtime_never_falls_back_to_standalone(
    snapshot_case, native_ordinary_host
):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    snapshot_controls._loop_projection(snapshot_case)
    h = native_ordinary_host
    calls = []
    snapshot_case.local.manifest_provider = lambda: calls.append(True) or {"tools": []}
    h.controller._hooks_v2_runtime = None
    with pytest.raises(RecoveryRequired, match="console_snapshot_owner_changed"):
        await h.controller.capture_turn_configuration_snapshot(h.session.id)
    assert not calls


async def test_standalone_mcp_capture_refuses_runtime_attachment_during_original_read(
    snapshot_case,
):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    snapshot_controls._loop_projection(snapshot_case)
    calls = []

    def attach():
        calls.append(True)
        snapshot_case.app.console_runtime = object()
        return {"tools": []}

    snapshot_case.local.manifest_provider = attach
    with pytest.raises(RecoveryRequired, match="console_snapshot_owner_changed"):
        await snapshot_case.controller.capture_turn_configuration_snapshot(
            snapshot_case.session.id
        )
    assert calls == [True]


async def test_actual_change_review_initializer_borrows_original_ordinary_tree(
    ordinary_host,
):
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree
    from tldw_chatbook.Workspaces.change_review_consent import (
        ChangeReviewCapability,
        ChangeReviewConsentService,
        ChangeReviewState,
        _default_root_initializer,
    )
    from tldw_chatbook.Workspaces.change_review_ownership import _current_snapshot
    from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

    h = ordinary_host
    entered, release = threading.Event(), threading.Event()
    observed, models = [], []

    def initialize(root):
        source = _current_snapshot.get()
        observed.append((current_execution_owner(), source.ordinary_tree, source))
        entered.set()
        assert release.wait(10)
        _default_root_initializer(root, shadow_service=source.shadow)

    service = ChangeReviewConsentService(
        h.registry,
        initialize_root=initialize,
        capability_reader=lambda: ChangeReviewCapability(ChangeReviewState.ENABLED),
        worker_count=1,
    )
    h.registry.set_change_review_enabled("references", True)
    h.app.change_review_consent_service = service
    h.registry.attach_change_review_consent_service(service)
    h.runtime.bind_change_review_consent_service(service)
    h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    original = h.gateway.stream_chat

    async def model(*args, **kwargs):
        models.append(current_ordinary_tree())
        async for item in original(*args, **kwargs):
            yield item

    h.gateway.stream_chat = model
    task = asyncio.create_task(
        h.controller.submit_draft("observe initializer", session_id=h.session.id)
    )
    h.tasks.append(task)
    try:
        await until(lambda: entered.is_set() or task.done())
        assert entered.is_set(), (
            task.result() if task.done() else "initializer not entered"
        )
        owner, tree, source = observed[0]
        assert tree is not None and owner is tree.owner and tree.footprint.opaque
        assert source.ordinary_tree is tree
        with h.runtime.worktree_activity.prelaunch(
            footprint=WorktreeFootprint(opaque=True)
        ) as contender:
            assert contender is None
    finally:
        release.set()
    assert (await task).accepted
    assert models and models[0] is observed[0][1]
    await until(lambda: not h.runtime.execution_capacity.snapshot().executions)


@pytest.mark.parametrize("custom_capability", [False, True])
async def test_actual_ordinary_change_review_baseline_and_end_borrow_same_tree(
    reference_host, monkeypatch, custom_capability
):
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree
    from tldw_chatbook.Workspaces.change_review_consent import (
        ChangeReviewCapability,
        ChangeReviewConsentService,
        ChangeReviewState,
        RootReadinessState,
    )
    from tldw_chatbook.Workspaces.change_review_ownership import _current_snapshot

    h = reference_host
    monkeypatch.setenv("TLDW_CHANGE_REVIEW_ENABLED", "true")
    service = ChangeReviewConsentService(
        h.registry,
        **(
            {
                "capability_reader": lambda: ChangeReviewCapability(
                    ChangeReviewState.ENABLED
                )
            }
            if custom_capability
            else {}
        ),
        worker_count=1,
    )
    h.registry.set_change_review_enabled("references", True)
    h.app.change_review_consent_service = service
    h.registry.attach_change_review_consent_service(service)
    h.runtime.bind_change_review_consent_service(service)
    bridge = h.runtime.ensure_agent_bridge(
        store_factory=lambda: h.store,
        provider_gateway_factory=lambda: h.gateway,
    )
    assert bridge is not None and h.runtime.change_review_coordinator is not None
    h.controller._agent_bridge = bridge
    h.controller._agent_runtime_enabled = True
    h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    assert (
        await h.controller.submit_draft(
            "prepare change review", session_id=h.session.id
        )
    ).accepted
    await until(
        lambda: any(
            item.state is RootReadinessState.READY
            for item in service.status("references").roots
        )
    )
    coordinator = h.runtime.change_review_coordinator
    tracker = coordinator._tracker
    observations, models = [], []
    for method in ("populate_prepared_baseline", "finish_turn"):
        original = getattr(tracker, method)

        def observe(*args, _original=original, _method=method, **kwargs):
            source = _current_snapshot.get()
            observations.append(
                (_method, current_execution_owner(), source.ordinary_tree)
            )
            return _original(*args, **kwargs)

        monkeypatch.setattr(tracker, method, observe)
    original_model = h.gateway.stream_chat

    async def model(*args, **kwargs):
        models.append(current_ordinary_tree())
        async for item in original_model(*args, **kwargs):
            yield item

    h.gateway.stream_chat = model
    configuration = await h.controller.capture_turn_configuration_snapshot(h.session.id)
    if custom_capability:
        assert (
            configuration.change_review_capture_pending
            and not configuration.workspace_roots
        )
    else:
        assert configuration.workspace_roots, service.status("references")
    assert (
        await h.controller.submit_draft(
            "tracked change review",
            session_id=h.session.id,
            configuration=configuration,
        )
    ).accepted
    await until(lambda: coordinator.wait_idle(0))
    assert models and {row[0] for row in observations} == {
        "populate_prepared_baseline",
        "finish_turn",
    }
    assert all(
        owner is models[0].owner and tree is models[0]
        for _, owner, tree in observations
    )


async def test_mounted_authority_defers_custom_sources_until_ordinary_owner(
    snapshot_case, native_ordinary_host
):
    from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
    from tldw_chatbook.UI.Console_Modules.turn_admission import (
        authority_inputs,
        capture_authority,
    )
    from tldw_chatbook.Workspaces.change_review_consent import (
        ChangeReviewCapability,
        ChangeReviewConsentService,
        ChangeReviewState,
    )

    snapshot_controls._loop_projection(snapshot_case)
    h = native_ordinary_host
    calls = []
    snapshot_case.local.manifest_provider = lambda: (
        calls.append("manifest") or {"tools": []}
    )
    consent = ChangeReviewConsentService(
        h.registry,
        capability_reader=lambda: (
            calls.append("consent")
            or ChangeReviewCapability(ChangeReviewState.DISABLED)
        ),
    )
    h.app.change_review_consent_service = consent
    h.registry.attach_change_review_consent_service(consent)
    h.runtime.bind_change_review_consent_service(consent)
    screen = ConsoleSessionController.__new__(ConsoleSessionController)
    screen.app_instance = h.app
    screen._current_chat_store_accessor = lambda: h.store
    screen._chat_store_accessor = lambda: h.store
    screen._provider_readiness_app_config_fn = dict
    screen._build_provider_selection_fn = lambda session_id: snapshot_case.selection
    screen._rag_source_types_accessor = list
    screen._rag_top_k_accessor = lambda: 4
    screen._scratch_snapshot_provider = lambda session_id: None
    try:
        with pytest.raises(
            RuntimeError, match="Console preparation source unavailable"
        ):
            capture_authority(screen, authority_inputs(screen, h.session.id))
        assert not calls, calls
    finally:
        consent.shutdown(timeout=1)


def _mounted_test_screen(h):
    from functools import partial
    from types import SimpleNamespace

    from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
    from tldw_chatbook.UI.Console_Modules.wiring import _stock_console_scratch_snapshot

    h.controller._scratch_spaces = h.runtime._scratch_spaces
    owner = ConsoleSessionController.__new__(ConsoleSessionController)
    owner.app_instance = h.app
    owner._provider_readiness_app_config_fn = dict
    owner._build_provider_selection_fn = h.controller._provider_selection_for_session
    owner._current_chat_store_accessor = lambda: h.store
    owner._chat_store_accessor = lambda: h.store
    owner._rag_source_types_accessor = list
    owner._rag_top_k_accessor = lambda: 4
    screen = SimpleNamespace(
        _session=owner, app_instance=h.app, _console_runtime=lambda: h.runtime
    )
    owner._scratch_snapshot_provider = partial(_stock_console_scratch_snapshot, screen)
    screen._console_received_live_adapters = tuple(
        (owner, name, getattr(owner, name))
        for name in (
            "_provider_readiness_app_config_fn",
            "_build_provider_selection_fn",
            "_scratch_snapshot_provider",
            "_rag_source_types_accessor",
            "_rag_top_k_accessor",
        )
    )
    from tldw_chatbook.Chat.console_configuration_preparation import (
        standard_console_configuration_sources,
    )

    assert standard_console_configuration_sources(
        h.app, h.store, h.controller, session_id=h.session.id
    )
    assert h.runtime._scratch_spaces is h.controller._scratch_spaces
    return screen, owner


@pytest.mark.parametrize("replace_source", [False, True])
async def test_mounted_owned_precapture_reuses_original_configuration_and_source(
    ordinary_host, replace_source
):
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree
    from tldw_chatbook.UI.Console_Modules.wiring import _precapture_turn_authority
    from tldw_chatbook.Workspaces.change_review_consent import (
        ChangeReviewCapability,
        ChangeReviewConsentService,
        ChangeReviewState,
    )

    h = ordinary_host
    calls = []
    service = ChangeReviewConsentService(
        h.registry,
        capability_reader=lambda: (
            calls.append(current_ordinary_tree())
            or ChangeReviewCapability(ChangeReviewState.DISABLED)
        ),
    )
    h.app.change_review_consent_service = service
    h.registry.attach_change_review_consent_service(service)
    h.runtime.bind_change_review_consent_service(service)
    h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    h.store.set_session_draft(h.session.id, "original mounted draft")
    screen, owner = _mounted_test_screen(h)
    prepared = await _precapture_turn_authority(screen, h.session.id)
    assert not calls
    if replace_source:
        service._capability_reader = lambda: ChangeReviewCapability(
            ChangeReviewState.DISABLED
        )
        with prepared(), pytest.raises(RuntimeError, match="source changed"):
            owner._build_console_turn_execution_context(h.session.id)
        assert h.store.session_draft(h.session.id) == "original mounted draft"
        assert not h.gateway.payloads and not calls
        return
    with prepared():
        configuration = owner._build_console_turn_execution_context(h.session.id)
        assert (
            owner._build_console_turn_execution_context(h.session.id) is configuration
        )
        assert configuration.change_review_capture_pending
        result = await h.controller.submit_draft(
            "original mounted draft",
            session_id=h.session.id,
            configuration=configuration,
        )
    assert result.accepted and h.gateway.payloads
    assert calls and all(tree is not None and tree.footprint.opaque for tree in calls)


async def test_mounted_owned_precapture_defers_original_custom_mcp(
    snapshot_case, native_ordinary_host
):
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree
    from tldw_chatbook.UI.Console_Modules.wiring import _precapture_turn_authority

    h = native_ordinary_host
    snapshot_controls._loop_projection(snapshot_case)
    observed = []
    snapshot_case.local.manifest_provider = lambda: (
        observed.append(current_ordinary_tree()) or {"tools": []}
    )
    h.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    screen, owner = _mounted_test_screen(h)
    prepared = await _precapture_turn_authority(screen, h.session.id)
    assert not observed
    with prepared():
        configuration = owner._build_console_turn_execution_context(h.session.id)
        assert configuration.mcp_capture_pending
        result = await h.controller.submit_draft(
            "mounted custom MCP", session_id=h.session.id, configuration=configuration
        )
    assert result.accepted and h.gateway.payloads
    assert observed and all(
        tree is observed[0] and tree.footprint.opaque for tree in observed
    )


async def test_mounted_missing_original_runtime_refuses_before_callbacks(
    ordinary_host, monkeypatch
):
    from tldw_chatbook.UI.Console_Modules.wiring import _precapture_turn_authority
    from tldw_chatbook.Workspaces.change_review_consent import (
        ChangeReviewCapability,
        ChangeReviewConsentService,
        ChangeReviewState,
    )

    h = ordinary_host
    calls = []
    service = ChangeReviewConsentService(
        h.registry,
        capability_reader=lambda: (
            calls.append(True) or ChangeReviewCapability(ChangeReviewState.DISABLED)
        ),
    )
    h.app.change_review_consent_service = service
    h.registry.attach_change_review_consent_service(service)
    h.runtime.bind_change_review_consent_service(service)
    screen, _owner = _mounted_test_screen(h)
    h.store.set_session_draft(h.session.id, "original draft after source loss")
    monkeypatch.setattr(h.app, "console_runtime", None)
    with pytest.raises(RuntimeError, match="Console preparation source unavailable"):
        await _precapture_turn_authority(screen, h.session.id)
    assert not calls and not h.gateway.payloads
    assert h.store.session_draft(h.session.id) == "original draft after source loss"


async def test_mounted_queued_precursor_refuses_replaced_controller_without_eager_mcp(
    snapshot_case, native_ordinary_host, monkeypatch
):
    from tldw_chatbook.UI.Console_Modules.wiring import _precapture_turn_authority

    h = native_ordinary_host
    snapshot_controls._loop_projection(snapshot_case)
    calls = []
    snapshot_case.local.manifest_provider = lambda: calls.append(True) or {"tools": []}
    h.store.set_session_draft(h.session.id, "unchanged queued draft")
    before = h.controller.prompt_queue_registry.snapshot(h.session.id)
    screen, owner = _mounted_test_screen(h)
    prepared = await _precapture_turn_authority(screen, h.session.id)
    assert not calls
    with monkeypatch.context() as patch:
        patch.setattr(h.runtime, "_chat_controller", object())
        with prepared(), pytest.raises(RuntimeError, match="source changed"):
            owner._build_console_turn_execution_context(h.session.id)
    assert not calls and not h.gateway.payloads
    assert h.controller.prompt_queue_registry.snapshot(h.session.id) == before
    assert h.store.session_draft(h.session.id) == "unchanged queued draft"


async def test_wired_chatscreen_unqualified_cold_source_refuses(
    snapshot_case, monkeypatch
):
    from Tests.UI import app_factory
    from tldw_chatbook import config
    from tldw_chatbook.UI.Console_Modules.wiring import _precapture_turn_authority
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    monkeypatch.setattr(app_factory, "load_settings", config.load_settings)
    app = app_factory._build_test_app(
        config_overrides={"console": {"agent_runtime": True}}
    )
    screen = ChatScreen(app)
    runtime = app.console_runtime
    try:
        store = screen._ensure_console_chat_store()
        session = store.ensure_session()
        controller = screen._ensure_console_chat_controller()
        assert runtime._chat_store is store and runtime._chat_controller is controller
        from tldw_chatbook.Chat.console_configuration_preparation import (
            capture_console_received_configuration_preparation,
        )

        diagnostic = capture_console_received_configuration_preparation(
            app, store, controller, session_id=session.id
        )
        assert diagnostic is None
        with pytest.raises(RuntimeError, match="source unavailable"):
            await _precapture_turn_authority(screen, session.id)
    finally:
        await runtime.dispose(timeout_seconds=3)


async def test_queue_pending_capture_without_original_witness_refuses_before_draft(
    snapshot_case, native_ordinary_host
):
    import contextvars

    h = native_ordinary_host
    snapshot_controls._loop_projection(snapshot_case)
    calls = []
    snapshot_case.local.manifest_provider = lambda: calls.append(True) or {"tools": []}
    h.store.set_session_draft(h.session.id, "original unqueued draft")
    configuration = await h.controller.capture_turn_configuration_snapshot(h.session.id)
    assert configuration.mcp_capture_pending and not calls
    before = h.controller.prompt_queue_registry.snapshot(h.session.id)
    context = contextvars.Context()
    task = context.run(
        asyncio.create_task,
        h.controller.queue_prompt(
            h.session.id,
            text="original unqueued draft",
            expected_revision=before.revision,
            configuration=configuration,
        ),
    )
    with pytest.raises(
        RuntimeError, match="Original queued capture source is unavailable"
    ):
        await task
    assert not calls and not h.gateway.payloads
    assert h.store.session_draft(h.session.id) == "original unqueued draft"
    assert h.controller.prompt_queue_registry.snapshot(h.session.id) == before
