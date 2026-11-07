"""Explicit references retain their original roots, worker and native custody."""

import asyncio
import subprocess
import threading
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_skill_substitution import _RecordingGateway
from Tests.console_provider_doubles import persisted_console_store
from tldw_chatbook.Agents.execution_capacity import current_execution_owner
from tldw_chatbook.Chat import console_references as refs
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Tools import workspace_file_roots as roots_api
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


async def until(probe):
    async with asyncio.timeout(10):
        while not probe():
            await asyncio.sleep(0.005)


@pytest.fixture
async def reference_host(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "visible.txt").write_text("VISIBLE-REFERENCE\n")
    private = root / "private"
    private.mkdir()
    (private / "hidden.txt").write_text("EXCLUDED-CONTENT\n")
    registry = LocalWorkspaceRegistryService(WorkspaceDB(tmp_path / "workspace.db"))
    registry.ensure_default_workspace()
    registry.create_workspace(workspace_id="references", name="References")
    binding = registry.add_folder_binding("references", root)
    registry.add_binding_exclusion("references", binding.binding_id, "private")
    registry.set_active_workspace("references")
    monkeypatch.setattr(roots_api, "_registry_factory", lambda: registry)
    monkeypatch.setattr(roots_api, "_LAUNCH_CWD", str(root))
    store = persisted_console_store(
        db_path=tmp_path / "chat.db", workspace_registry=registry
    )
    session = store.create_session(workspace_id="references")
    gateway = _RecordingGateway()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, provider="llama_cpp", model="m"
    )
    app = SimpleNamespace(
        workspace_registry_service=registry,
        app_config={},
        chachanotes_db=store.persistence.db,
    )
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService

    app.local_chat_conversation_service = ChatConversationService(store.persistence.db)
    runtime = app.console_runtime = ConsoleRuntime(app)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    tasks = []
    host = SimpleNamespace(**locals())
    try:
        with roots_api.run_file_sandbox(root):
            yield host
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        await runtime.dispose()
        store.persistence.db.close_connection()
        registry.db.close()


async def test_real_expansion_worker_is_native_before_file_read(
    reference_host, monkeypatch
):
    h = reference_host
    entered, release = threading.Event(), threading.Event()
    observed = []
    original = refs._expand_path

    def held(*args, **kwargs):
        observed.append(current_execution_owner())
        entered.set()
        assert release.wait(15)
        return original(*args, **kwargs)

    monkeypatch.setattr(refs, "_expand_path", held)
    task = asyncio.create_task(
        h.controller.submit_draft("read @visible.txt", session_id=h.session.id)
    )
    h.tasks.append(task)
    try:
        await until(lambda: entered.is_set() or task.done())
        assert entered.is_set(), task.result() if task.done() else "no worker"
        assert observed[0] is not None, "file body worker has no native owner"
        snapshot = h.runtime.execution_capacity.snapshot()
        assert snapshot.tool_workers == 1
        assert snapshot.model_lifelines == snapshot.child_executions == 0
    finally:
        release.set()
    assert (await task).accepted
    assert h.runtime.execution_capacity.snapshot().executions == ()


@pytest.mark.parametrize("refuse", [False, True])
async def test_custom_parser_reads_only_under_original_opaque_custody(
    reference_host, monkeypatch, refuse
):
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
    from tldw_chatbook.Workspaces.worktree_activity import (
        WorktreeFootprint,
        WorktreeScope,
    )

    h = reference_host
    original = refs.find_reference_candidates
    other = h.tmp_path / "unrelated"
    other.mkdir()
    scope = WorktreeScope(other, True, capture_directory_chain(other))
    observations = []

    def custom_parser(text):
        body = (h.root / "visible.txt").read_text()
        with h.runtime.worktree_activity.prelaunch(
            footprint=WorktreeFootprint(reads=(scope,))
        ) as claim:
            observations.append(
                (
                    current_execution_owner() is not None,
                    threading.current_thread() is not threading.main_thread(),
                    bool(h.runtime._preparation_reads),
                    h.runtime.execution_capacity.snapshot().tool_workers,
                    claim is None,
                    body.strip(),
                )
            )
        if refuse:
            raise RuntimeError("parser refused after its read")
        return original(text)

    monkeypatch.setattr(refs, "find_reference_candidates", custom_parser)
    try:
        result = await h.controller.submit_draft(
            "read @visible.txt", session_id=h.session.id
        )
    except RuntimeError as error:
        result = error
    assert observations == [(True, True, True, 1, True, "VISIBLE-REFERENCE")]
    assert result.accepted
    assert ("VISIBLE-REFERENCE" in str(h.gateway.payloads)) is not refuse
    if refuse:
        assert h.gateway.payloads[-1][-1]["content"] == "read @visible.txt"
        assert any(
            "refused" in row.content.lower()
            for row in h.store.messages_for_session(h.session.id)
        )
    assert h.runtime.execution_capacity.snapshot().executions == ()
    assert not h.runtime._preparation_reads


@pytest.mark.parametrize("blocked", ["no-roots", "activity"])
async def test_unadmitted_reference_never_calls_custom_parser(
    reference_host, monkeypatch, blocked
):
    from contextlib import ExitStack

    from tldw_chatbook.Tools import file_operation_tools
    from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

    h = reference_host
    original = refs.find_reference_candidates
    reads = []

    def custom_parser(text):
        reads.append((h.root / "visible.txt").read_text())
        return original(text)

    monkeypatch.setattr(refs, "find_reference_candidates", custom_parser)
    with ExitStack() as stack:
        if blocked == "no-roots":
            missing = h.tmp_path / "missing-reference-sandbox"
            monkeypatch.setattr(
                file_operation_tools, "_sandbox_config_path", lambda settings: missing
            )
            stack.enter_context(roots_api.run_workspace("workspace-default"))
            stack.enter_context(roots_api.run_file_sandbox(None))
        else:
            claim = stack.enter_context(
                h.runtime.worktree_activity.prelaunch(
                    footprint=WorktreeFootprint(opaque=True)
                )
            )
            assert claim is not None
        result = await h.controller.submit_draft(
            "read @visible.txt", session_id=h.session.id
        )
    assert reads == [], "failed admission still invoked the effectful parser"
    if blocked == "activity":
        # Original ordinary admission refuses the whole conflicting turn before
        # any reference worker, durable user row or provider request exists.
        assert not result.accepted
        assert "busy" in result.visible_copy.lower()
        assert h.gateway.payloads == []
        assert h.session.persisted_conversation_id is None
    else:
        assert result.accepted
        assert h.gateway.payloads[-1][-1]["content"] == "read @visible.txt"
        assert any(
            "refused" in row.content.lower()
            for row in h.store.messages_for_session(h.session.id)
        )
    assert h.runtime.execution_capacity.snapshot().executions == ()
    assert not h.runtime._preparation_reads


@pytest.mark.parametrize("draft", ["plain text", "mail bob@example.com", "literal @."])
async def test_native_no_candidate_preflight_never_calls_custom_parser(
    reference_host, monkeypatch, draft
):
    h = reference_host
    original = refs.find_reference_candidates
    reads = []

    def custom_parser(text):
        reads.append((h.root / "visible.txt").read_text())
        return original(text)

    monkeypatch.setattr(refs, "find_reference_candidates", custom_parser)
    result = await h.controller.submit_draft(draft, session_id=h.session.id)
    assert reads == [], "literal/email text invoked the effectful parser"
    assert result.accepted
    assert h.gateway.payloads[-1][-1]["content"] == draft
    assert not h.runtime._preparation_reads


async def test_folder_excluded_child_name_is_not_exposed(reference_host):
    h = reference_host
    result = refs.build_console_reference_resolver()(str(h.root))
    assert result[0] == "folder"
    assert "visible.txt" in result[1]
    assert "private" not in result[1], "folder enumeration leaked excluded child name"


async def test_git_repo_above_readable_root_is_refused(reference_host, monkeypatch):
    h = reference_host
    git(h.root, "init", "-q")
    nested = h.root / "nested"
    nested.mkdir()
    git(h.root, "add", "visible.txt")
    monkeypatch.setattr(roots_api, "_LAUNCH_CWD", str(nested))
    with pytest.raises(RuntimeError, match="outside"):
        refs.run_git_reference("staged", roots=(nested,))


async def test_stock_read_claim_allows_unrelated_root_work(reference_host):
    from Tests.Chat.test_console_received_native_preparation import _HeldOriginal
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
    from tldw_chatbook.Workspaces.worktree_activity import (
        WorktreeFootprint,
        WorktreeScope,
    )

    h = reference_host
    other = h.tmp_path / "unrelated"
    other.mkdir()
    chain = capture_directory_chain(other)
    held = _HeldOriginal(refs._expand_path)
    with held.installed():
        task = asyncio.create_task(
            h.controller.submit_draft("read @visible.txt", session_id=h.session.id)
        )
        h.tasks.append(task)
        await until(lambda: held.entered.is_set() or task.done())
        assert held.entered.is_set()
        with h.runtime.worktree_activity.prelaunch(
            footprint=WorktreeFootprint(writes=(WorktreeScope(other, True, chain),))
        ) as claim:
            assert claim is not None, "stock file references were classified opaque"
        root_chain = capture_directory_chain(h.root)
        with h.runtime.worktree_activity.prelaunch(
            footprint=WorktreeFootprint(
                writes=(WorktreeScope(h.root, True, root_chain),)
            )
        ) as claim:
            assert claim is None
        held.release.set()
        assert (await task).accepted


@pytest.mark.parametrize("close", [False, True])
async def test_repeated_cancellation_retains_actual_worker(reference_host, close):
    from Tests.Chat.test_console_received_native_preparation import _HeldOriginal

    h = reference_host
    held = _HeldOriginal(refs._expand_path)
    with held.installed():
        task = asyncio.create_task(
            h.controller.submit_draft("read @visible.txt", session_id=h.session.id)
        )
        h.tasks.append(task)
        await until(lambda: held.entered.is_set() or task.done())
        assert held.entered.is_set()
        closing = asyncio.create_task(h.runtime.dispose()) if close else task
        if close:
            h.tasks.append(closing)
            await until(lambda: h.runtime._disposed)
        for _ in range(2):
            closing.cancel()
            await asyncio.sleep(0.03)
            assert not task.done()
            assert h.runtime.execution_capacity.snapshot().tool_workers == 1
            assert h.runtime._preparation_reads
        held.release.set()
        await asyncio.gather(task, closing, return_exceptions=True)
    assert not h.gateway.payloads
    assert not h.runtime._preparation_reads
    assert h.runtime.execution_capacity.snapshot().executions == ()


@pytest.mark.parametrize(
    "change",
    ["workspace", "cwd", "root", "session", "registry", "runtime", "exclusion"],
)
async def test_changed_original_source_discards_read_body(
    reference_host, monkeypatch, change
):
    from Tests.Chat.test_console_received_native_preparation import _HeldOriginal

    h = reference_host
    held = _HeldOriginal(refs._expand_path)
    with held.installed():
        task = asyncio.create_task(
            h.controller.submit_draft("read @visible.txt", session_id=h.session.id)
        )
        h.tasks.append(task)
        await until(lambda: held.entered.is_set() or task.done())
        assert held.entered.is_set()
        if change == "workspace":
            h.registry.create_workspace(workspace_id="other", name="Other")
            h.registry.set_active_workspace("other")
        elif change == "cwd":
            monkeypatch.setattr(roots_api, "_LAUNCH_CWD", str(h.tmp_path))
        elif change == "root":
            h.root.rename(h.tmp_path / "original")
            h.root.mkdir()
            (h.root / "visible.txt").write_text("REPLACEMENT-CONTENT")
        elif change == "session":
            h.session.settings_revision += 1
        elif change == "registry":
            monkeypatch.setattr(h.app, "workspace_registry_service", object())
        elif change == "runtime":
            monkeypatch.setattr(h.app, "console_runtime", object())
        else:
            h.registry.add_binding_exclusion(
                "references", h.binding.binding_id, "visible.txt"
            )
        held.release.set()
        await task
    assert all(
        "VISIBLE-REFERENCE" not in str(p) and "REPLACEMENT-CONTENT" not in str(p)
        for p in h.gateway.payloads
    )
    rows = h.store.messages_for_session(h.session.id)
    assert any("reference source unavailable" in row.content for row in rows)
    assert h.runtime.execution_capacity.snapshot().executions == ()


async def test_exclusion_removal_does_not_widen_frozen_resolver(reference_host):
    h = reference_host
    resolve = refs.build_console_reference_resolver()
    h.registry.remove_binding_exclusion("references", h.binding.binding_id, "private")
    assert resolve(str(h.root / "private" / "hidden.txt"))[0] == "refused"
    assert "private" not in resolve(str(h.root))[1]


async def test_missing_sandbox_is_not_created_and_bound_roots_remain(
    reference_host, monkeypatch
):
    from tldw_chatbook.Tools import file_operation_tools

    h = reference_host
    missing = h.tmp_path / "missing" / "sandbox"
    monkeypatch.setattr(
        file_operation_tools, "_sandbox_config_path", lambda settings: missing
    )
    with roots_api.run_file_sandbox(None):
        resolver = refs.build_console_reference_resolver()
        assert resolver(str(h.root / "visible.txt"))[1] == "VISIBLE-REFERENCE\n"
        assert not missing.exists() and not missing.parent.exists()


async def test_zero_roots_has_no_reference_operation_or_git_launch(
    reference_host, monkeypatch
):
    from tldw_chatbook.Chat.console_reference_preparation import _ReferenceRead
    from tldw_chatbook.Tools import file_operation_tools

    h = reference_host
    missing = h.tmp_path / "missing-sandbox"
    monkeypatch.setattr(
        file_operation_tools, "_sandbox_config_path", lambda settings: missing
    )
    observed = []
    original_prepare = _ReferenceRead.prepare

    def prepared(request):
        result = original_prepare(request)
        observed.append((request.owner, request.operation, request.authority.roots))
        return result

    monkeypatch.setattr(_ReferenceRead, "prepare", prepared)
    monkeypatch.setattr(
        refs,
        "run_git_reference",
        lambda *a, **k: pytest.fail("Git launched without roots"),
    )
    with roots_api.run_workspace("workspace-default"), roots_api.run_file_sandbox(None):
        assert (
            await h.controller.submit_draft(
                "read @diff @/unavailable", session_id=h.session.id
            )
        ).accepted
    assert observed == [(None, None, ())]
    assert not h.runtime.execution_capacity.snapshot().executions
    assert not missing.exists()
    assert "git-diff" not in str(h.gateway.payloads)


def git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout


async def test_whole_repo_git_exclusions_magic_paths_and_no_mutation(reference_host):
    h = reference_host
    git(h.root, "init", "-q")
    (h.root / "safe*.txt").write_text("SAFE-MAGIC-BODY\n")
    excluded = h.root / "private[1].txt"
    excluded.write_text("EXCLUDED-MAGIC-BODY\n")
    h.registry.add_binding_exclusion("references", h.binding.binding_id, excluded.name)
    git(h.root, "add", ".")
    config = (h.root / ".git" / "config").read_bytes()
    index = (h.root / ".git" / "index").read_bytes()
    head = (h.root / ".git" / "HEAD").read_bytes()
    nested = h.root / "nested"
    nested.mkdir()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(roots_api, "_LAUNCH_CWD", str(nested))
        assert (
            await h.controller.submit_draft("read @staged", session_id=h.session.id)
        ).accepted
    payload = str(h.gateway.payloads)
    assert "VISIBLE-REFERENCE" in payload and "SAFE-MAGIC-BODY" in payload
    assert "EXCLUDED-CONTENT" not in payload and "EXCLUDED-MAGIC-BODY" not in payload
    assert (h.root / ".git" / "config").read_bytes() == config
    assert (h.root / ".git" / "index").read_bytes() == index
    assert (h.root / ".git" / "HEAD").read_bytes() == head


@pytest.mark.parametrize(
    "key,value",
    [
        ("filter.fixture.clean", "never-start-this"),
        ("filter.fixture.process", "never-start-this"),
        ("filter.fixture.smudge", "never-start-this"),
        ("remote.origin.promisor", "true"),
        ("extensions.partialclone", "origin"),
    ],
)
async def test_git_helpers_refuse_without_body(reference_host, key, value):
    h = reference_host
    git(h.root, "init", "-q")
    git(h.root, "add", "visible.txt")
    git(h.root, "config", key, value)
    config = (h.root / ".git" / "config").read_bytes()
    assert (
        await h.controller.submit_draft("read @staged", session_id=h.session.id)
    ).accepted
    assert "VISIBLE-REFERENCE" not in str(h.gateway.payloads)
    assert (h.root / ".git" / "config").read_bytes() == config
    assert any(
        "configured" in row.content
        for row in h.store.messages_for_session(h.session.id)
    )


@pytest.fixture
async def reference_git_child(reference_host, monkeypatch, request):
    import os
    import shutil
    import sys

    h = reference_host
    git(h.root, "init", "-q")
    git(h.root, "add", "visible.txt")
    real_git = shutil.which("git")
    wrapper_dir = h.tmp_path / "git-wrapper"
    wrapper_dir.mkdir()
    entered, release, returned = (
        wrapper_dir / n for n in ("entered", "release", "returned")
    )
    selected = request.param
    child = wrapper_dir / "child.py"
    child.write_text(
        "import os,time\nfrom pathlib import Path\n"
        f"Path({str(entered)!r}).write_text(str(os.getpid()))\n"
        f"while not Path({str(release)!r}).exists(): time.sleep(.01)\n"
    )
    wrapper = wrapper_dir / "git"
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import os,subprocess,sys\nfrom pathlib import Path\nargs=sys.argv[1:]\n"
        f"selected=({selected!r} == 'version' and args == ['--version']) or "
        f"({selected!r} == 'config' and '--name-only' in args) or "
        f"({selected!r} == 'diff' and 'diff' in args)\n"
        f"armed=Path({str(wrapper_dir / 'once')!r})\nheld=False\n"
        "if selected:\n try:\n  armed.mkdir()\n  held=True\n except FileExistsError: pass\n"
        "if held:\n"
        f" subprocess.Popen([{sys.executable!r},{str(child)!r}],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
        f" while not Path({str(entered)!r}).exists(): pass\n"
        f"result=subprocess.run([{real_git!r},*args],capture_output=True)\n"
        "sys.stdout.buffer.write(result.stdout)\nsys.stderr.buffer.write(result.stderr)\n"
        f"if held: Path({str(returned)!r}).write_text(str(os.getpid()))\n"
        "sys.exit(result.returncode)\n"
    )
    wrapper.chmod(0o700)
    monkeypatch.setenv("PATH", str(wrapper_dir) + os.pathsep + os.environ["PATH"])
    state = SimpleNamespace(**locals())
    try:
        yield state
    finally:
        release.touch()


@pytest.mark.parametrize(
    "reference_git_child", ["version", "config", "diff"], indirect=True
)
async def test_real_managed_git_child_keeps_original_operation(reference_git_child):
    from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

    w = reference_git_child
    h = w.h
    task = asyncio.create_task(
        h.controller.submit_draft("read @staged", session_id=h.session.id)
    )
    h.tasks.append(task)
    try:
        await until(lambda: w.returned.exists() or task.done())
        assert w.entered.exists() and w.returned.exists()
        await asyncio.sleep(0.15)
        assert not task.done(), "direct Git exit abandoned its managed descendant"
        assert h.runtime.execution_capacity.snapshot().tool_workers == 1
        with h.runtime.worktree_activity.prelaunch(
            footprint=WorktreeFootprint(opaque=True)
        ) as claim:
            assert claim is None
        task.cancel()
        await asyncio.sleep(0.03)
        assert not task.done()
        assert h.runtime._preparation_reads
    finally:
        w.release.touch()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not h.gateway.payloads
    assert h.runtime.execution_capacity.snapshot().executions == ()


async def test_original_finish_return_precedes_root_and_read_retirement(
    reference_host, monkeypatch
):
    from tldw_chatbook.Agents.execution_capacity import OwnedOperation

    h = reference_host
    entered, release = threading.Event(), threading.Event()
    original = OwnedOperation.finish
    owners = []

    def held_finish(operation):
        owner = current_execution_owner()
        result = original(operation)
        if owner is not None and owner.capacity is h.runtime._execution_capacity:
            owners.append(owner)
            entered.set()
            assert release.wait(15)
        return result

    monkeypatch.setattr(OwnedOperation, "finish", held_finish)
    task = asyncio.create_task(
        h.controller.submit_draft("read @visible.txt", session_id=h.session.id)
    )
    h.tasks.append(task)
    try:
        await until(lambda: entered.is_set() or task.done())
        assert entered.is_set()
        assert owners[0].is_admitted
        assert h.runtime._preparation_reads and not task.done()
        task.cancel()
        await asyncio.sleep(0.03)
        assert not task.done() and owners[0].is_admitted
    finally:
        release.set()
    await asyncio.gather(task, return_exceptions=True)
    assert not owners[0].is_admitted
    assert h.runtime.execution_capacity.snapshot().executions == ()


async def test_actual_controller_result_assignment_retains_operation(
    reference_host, monkeypatch
):
    h = reference_host
    original = refs.ReferenceExpansion.__getattribute__
    observed = []

    def observe(expansion, key):
        if (
            key == "expanded_text"
            and threading.current_thread() is threading.main_thread()
        ):
            observed.append(
                (
                    h.runtime.execution_capacity.snapshot().tool_workers,
                    bool(h.runtime._preparation_reads),
                )
            )
        return original(expansion, key)

    monkeypatch.setattr(refs.ReferenceExpansion, "__getattribute__", observe)
    assert (
        await h.controller.submit_draft("read @visible.txt", session_id=h.session.id)
    ).accepted
    assert observed == [(1, True)]
    assert "VISIBLE-REFERENCE" in str(h.gateway.payloads)
    rows = h.store.messages_for_session(h.session.id)
    assert [r.content for r in rows if r.role.value == "user"] == ["read @visible.txt"]


@pytest.mark.parametrize("route", ["received", "voice"])
async def test_original_received_and_voice_consumers_expand_once(reference_host, route):
    from dataclasses import replace

    from Tests.Chat.test_console_received_intent_custody import _intent
    from Tests.Chat.test_console_voice_effect_barrier import _context
    from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest

    h = reference_host
    draft = "read @visible.txt"
    configuration = h.controller.resolve_turn_configuration_snapshot(h.session.id)
    h.store.set_session_draft(h.session.id, draft)
    if route == "voice":
        context = replace(_context(), configuration=configuration)
        turn = h.controller.submit_accepted_voice_turn(draft, context)
    else:
        request = ConsoleTurnCustodyRequest(
            turn_id="reference-received",
            session_id=h.session.id,
            draft=draft,
            configuration=configuration,
        )
        intent = _intent((h.runtime, h.store, h.session, request))
        turn = h.runtime.accept_received_intent(intent)
    task = h.runtime._turn_custody[turn].task
    h.tasks.append(task)
    result = await task
    assert result.accepted
    assert str(h.gateway.payloads).count("VISIBLE-REFERENCE") == 1
    assert [
        r.content
        for r in h.store.messages_for_session(h.session.id)
        if r.role.value == "user"
    ] == [draft]
    assert not h.runtime._preparation_reads


async def test_actual_queued_head_expands_with_original_text(reference_host):
    from Tests.Chat.test_console_prompt_queue_coordinator import (
        SequencedGateway,
        _queue,
    )

    h = reference_host
    gateway = SequencedGateway()
    h.controller.provider_gateway = gateway
    task = asyncio.create_task(
        h.controller.run_prompt_chain("first", session_id=h.session.id)
    )
    h.tasks.append(task)
    try:
        await until(lambda: gateway.started[0].is_set() or task.done())
        assert gateway.started[0].is_set()
        await _queue(h.controller, h.session.id, "next @visible.txt")
        gateway.release[0].set()
        await until(lambda: gateway.started[1].is_set() or task.done())
        assert gateway.started[1].is_set(), (
            task.result() if task.done() else "queued worker missing"
        )
        gateway.release[1].set()
        assert (await task).accepted
    finally:
        for event in gateway.release:
            event.set()
    assert gateway.user_turns[0] == "first"
    assert gateway.user_turns[1].count("VISIBLE-REFERENCE") == 1
    assert h.controller.prompt_queue_registry.snapshot(h.session.id).total_count == 0
    assert [
        r.content
        for r in h.store.messages_for_session(h.session.id)
        if r.role.value == "user"
    ] == ["first", "next @visible.txt"]


@pytest.mark.parametrize("reference_git_child", ["diff"], indirect=True)
async def test_reference_timeout_retains_same_worker_for_late_tree_proof(
    reference_git_child, monkeypatch
):
    from tldw_chatbook.Notes.git_process_containment import ProcessTreeController

    w = reference_git_child
    h = w.h
    entered = threading.Event()
    original_run, original_wait = refs.run_git_reference, ProcessTreeController.wait

    def short_run(*args, **kwargs):
        return original_run(*args, **{**kwargs, "timeout": 1.0})

    def stop_unavailable(controller, tree):
        raise OSError("controlled stop unavailable")

    async def wait(controller, tree, *, timeout):
        if timeout == float("inf"):
            entered.set()
        return await original_wait(controller, tree, timeout=timeout)

    monkeypatch.setattr(refs, "run_git_reference", short_run)
    monkeypatch.setattr(ProcessTreeController, "terminate", stop_unavailable)
    monkeypatch.setattr(ProcessTreeController, "kill", stop_unavailable)
    monkeypatch.setattr(ProcessTreeController, "wait", wait)
    task = asyncio.create_task(
        h.controller.submit_draft("read @staged", session_id=h.session.id)
    )
    h.tasks.append(task)
    try:
        await until(lambda: entered.is_set() or task.done())
        assert entered.is_set()
        assert not task.done() and h.runtime._preparation_reads
        assert h.runtime.execution_capacity.snapshot().tool_workers == 1
        task.cancel()
        await asyncio.sleep(0.03)
        assert not task.done()
    finally:
        w.release.touch()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert h.runtime.execution_capacity.snapshot().executions == ()
