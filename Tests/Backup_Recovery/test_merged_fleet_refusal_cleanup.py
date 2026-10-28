"""A fleet worker refused before execution still settles its reserved launch."""

import pytest

from Tests.Backup_Recovery.test_activation_agents import _SCRIPT
from Tests.Backup_Recovery.test_home_citation_retirement import _run


_REFUSED_CHILD = _SCRIPT.split("effects=[]")[0] + r'''
import sqlite3, threading, time
from contextlib import contextmanager
from Tests.Agents.conftest import join_fleet_children
from Tests.Agents.test_automatic_child_scope import accepted_context, make_service, run
from Tests.Agents.test_agent_service import fence
from tldw_chatbook.Agents import activation
from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator

context = accepted_context(db, child_launches=1)
refused, retired, settled, connections, callback_errors = [], [], [], [], []
parent_thread = threading.current_thread()
caller = db._held_connection()
native_pause = None
original_start = threading.Thread.start
original_execution = activation.execution

@contextmanager
def refuse_child(service=None, *, sources=()):
    if threading.current_thread().name.startswith('fleet-'):
        refused.append(threading.current_thread())
        if route == 'fleet-refused':
            raise activation.AgentActivationRequired()
    with original_execution(service, sources=sources):
        yield

activation.execution = refuse_child

def start(thread):
    global native_pause
    if thread.name.startswith('fleet-') and route == 'thread-start':
        raise RuntimeError('native thread start refused')
    if thread.name.startswith('fleet-') and route == 'fleet-paused':
        native_pause = storage._begin_local_pause()
    return original_start(thread)

threading.Thread.start = start
fleet = FleetCoordinator(max_live=3, clock=time.monotonic)
provider = None
worktree_authority = None
spawn_args = {'task': 'refused child'}
worktree_paths = []
if route == 'fleet-paused':
    from Tests.Agents.test_fleet_runtime import _git, _fs_local_provider
    repo = base/'disposable-child-repo'
    repo.mkdir()
    _git(repo, 'init', '-b', 'main')
    _git(repo, 'config', 'user.email', 'test@example.invalid')
    _git(repo, 'config', 'user.name', 'Backup test')
    (repo/'seed.txt').write_text('parent content\n')
    _git(repo, 'add', 'seed.txt')
    _git(repo, 'commit', '-m', 'disposable baseline')
    provider = _fs_local_provider(repo)
    from tldw_chatbook.Agents.agent_worktree import _worktree_root_identity
    from tldw_chatbook.Agents.local_tool_provider import RunAdmittedWorkspaceRoot
    worktree_authority = RunAdmittedWorkspaceRoot(
        workspace_id='disposable-workspace', binding_id='selected', alias='selected',
        root=repo, locator_fingerprint='f' * 64,
        root_identity=_worktree_root_identity(repo), allow_write=True,
        guard=lambda write: _worktree_root_identity(repo) == worktree_authority.root_identity,
    )
    spawn_args['isolation'] = 'worktree'

def on_settled(run_id, status):
    try:
        assert threading.current_thread() is parent_thread
        if route == 'fleet-paused':
            assert storage._pause is native_pause
            assert not native_pause.drain(time.monotonic() + .01)
        settled.append((run_id, status, db.get_run(run_id)['status']))
        connections.append(db._thread_local.conn)
    except BaseException as error:
        callback_errors.append(error)
    finally:
        if native_pause is not None and threading.current_thread() is parent_thread:
            native_pause.resume()
service, chat = make_service(
    db, context,
    [fence('spawn_subagent', spawn_args), 'parent complete'],
    {'refused child': ['must not execute']},
    fleet_coordinator=fleet,
    provider=provider,
    worktree_repo_authority=worktree_authority,
    on_child_settled=on_settled,
)
original_retire = service._retire_agent_worktree
def retire(run_id, handle_id, **kwargs):
    retired.append((run_id, handle_id, kwargs.get('discard', False)))
    if route == 'fleet-paused':
        assert threading.current_thread() is parent_thread
        assert storage._pause is native_pause
        worktree_paths.append(service._agent_worktrees[handle_id].worktree_path)
    return original_retire(run_id, handle_id, **kwargs)
service._retire_agent_worktree = retire
try:
    try:
        run_id, outcome = run(service)
    finally:
        if native_pause is not None and storage._pause is native_pause:
            native_pause.resume()
    join_fleet_children(service)
    assert outcome.status == RUN_DONE, outcome
    assert not callback_errors, callback_errors
    assert not chat.child_calls
    assert len(refused) == (0 if route == 'thread-start' else 1)
    snapshot = db.automatic_work.snapshot(context.chain_id)
    assert snapshot.reserved['child_launch'] == 0, snapshot
    assert snapshot.used['child_launch'] == 0, snapshot
    children = [row for row in db.list_runs('conversation') if row['task']]
    assert len(children) == 1 and children[0]['status'] == 'error', children
    child_id = children[0]['id']
    assert settled == ([] if route == 'thread-start' else [(child_id, 'error', 'error')]), settled
    assert len(retired) == 1 and retired[0][0] == child_id, retired
    assert retired[0][2] is True, retired
    assert service.runtime_capacity.snapshot().executions == ()
    assert fleet.live_count() == 0
    if route == 'fleet-paused':
        from tldw_chatbook.DB.agent_worktrees import AgentWorktreeRepository
        assert len(worktree_paths) == 1 and worktree_paths[0].is_dir()
        created = next(iter(service._agent_worktrees.values()))
        assert created.worktree_path == worktree_paths[0]
        assert not provider._agent_roots
        assert _git(repo, 'branch', '--list', created.branch).strip()
        record = AgentWorktreeRepository(db).get_for_conversation(child_id, 'conversation')
        assert record is not None and record['writer_state'] == 'drained'
        assert record['base_sha'] == created.base_sha
        assert (repo/'seed.txt').read_text() == 'parent content\n'
    assert connections == ([] if route == 'thread-start' else [caller])
    assert caller.execute('SELECT 1').fetchone()[0] == 1

finally:
    activation.execution = original_execution
    threading.Thread.start = original_start
    if native_pause is not None and storage._pause is native_pause:
        native_pause.resume()
    join_fleet_children(service)
    if route == 'fleet-paused':
        from tldw_chatbook.Agents.agent_worktree import discard_agent_worktree
        for created in service._agent_worktrees.values():
            assert discard_agent_worktree(repo, created) is None
    db.close()
try:
    sqlite3.Connection.in_transaction.__get__(caller)
except sqlite3.ProgrammingError as error:
    assert 'closed' in str(error)
else:
    raise AssertionError('parent native handle did not retire on owner close')
assert not blocked_attempts()
print('retired and reopened')
'''


@pytest.mark.parametrize("route", ["fleet-refused", "fleet-paused", "thread-start"])
def test_refused_fleet_worker_refunds_and_runs_complete_upstream_teardown(
    tmp_path, route
):
    _run(tmp_path, route, "approved", script=_REFUSED_CHILD)


_STORAGE_REFUSED_CHILD = (
    _SCRIPT.split("effects=[]")[0]
    + r"""
import sqlite3, time
from Tests.Agents.conftest import join_fleet_children
from Tests.Agents.test_automatic_child_scope import accepted_context, make_service, run
from Tests.Agents.test_agent_service import fence
from Tests.Agents.test_fleet_runtime import _git, _fs_local_provider
from tldw_chatbook.Agents import agent_worktree
from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
from tldw_chatbook.Agents.local_tool_provider import RunAdmittedWorkspaceRoot

repo = base/'disposable-source'
repo.mkdir()
_git(repo, 'init', '-b', 'main')
_git(repo, 'config', 'user.email', 'test@example.invalid')
_git(repo, 'config', 'user.name', 'Backup test')
(repo/'seed.txt').write_text('unchanged source\n')
_git(repo, 'add', 'seed.txt')
_git(repo, 'commit', '-m', 'baseline')
source_sha = _git(repo, 'rev-parse', 'HEAD')
provider = _fs_local_provider(repo)
authority = RunAdmittedWorkspaceRoot(
    workspace_id='disposable', binding_id='selected', alias='selected',
    root=repo, locator_fingerprint='f' * 64,
    root_identity=agent_worktree._worktree_root_identity(repo), allow_write=True,
    guard=lambda write: agent_worktree._worktree_root_identity(repo) == authority.root_identity,
)
worktrees = agent_worktree._worktrees_base()
saved = worktrees.with_name('agent-worktrees-symlink-control')
worktrees.rename(saved)
outside = base/'outside-sensitive-path-sentinel'
outside.mkdir(mode=0o700)
worktrees.symlink_to(outside, target_is_directory=True)
context = accepted_context(db, child_launches=1)
fleet = FleetCoordinator(max_live=3, clock=time.monotonic)
service, chat = make_service(
    db, context,
    [fence('spawn_subagent', {'task':'refused child', 'isolation':'worktree'}), 'parent complete'],
    {'refused child':['must not execute']}, fleet_coordinator=fleet,
    provider=provider, worktree_repo_authority=authority,
)
caller = db._held_connection()
prior = getattr(storage._operation_local, 'operation', None)
original_create = agent_worktree.create_agent_worktree
restored = []
def observe_create(*args, **kwargs):
    core = storage._operation_local.operation
    assert core is not None and core in storage._operations
    try:
        return original_create(*args, **kwargs)
    finally:
        assert storage._operation_local.operation is core
        storage._check_operation(core, core.path)
        assert core.lease in storage._live_leases
        restored.append(core)
agent_worktree.create_agent_worktree = observe_create
try:
    run_id, outcome = run(service)
    join_fleet_children(service)
    assert outcome.status == RUN_DONE, outcome
    assert not chat.child_calls
    assert len(restored) == 1
    assert getattr(storage._operation_local, 'operation', None) is prior
    snapshot = db.automatic_work.snapshot(context.chain_id)
    assert snapshot.reserved['child_launch'] == 0, snapshot
    assert snapshot.used['child_launch'] == 0, snapshot
    children = [row for row in db.list_runs('conversation') if row['task']]
    assert len(children) == 1 and children[0]['status'] == 'error', children
    refusal = 'worktree isolation refused [storage_unavailable]: Worktree storage is unavailable; local review may be required.'
    assert children[0]['result'] == refusal, children
    assert service.runtime_capacity.snapshot().executions == ()
    assert fleet.live_count() == 0
    assert not service._agent_worktrees
    assert not service._fleet_cancels
    assert not provider._agent_roots
    assert tuple(outside.iterdir()) == ()
    assert _git(repo, 'rev-parse', 'HEAD') == source_sha
    assert _git(repo, 'status', '--porcelain') == ''
    assert _git(repo, 'branch', '--list', 'agent/*') == ''
    assert (repo/'seed.txt').read_text() == 'unchanged source\n'
    assert caller.execute('SELECT 1').fetchone()[0] == 1
finally:
    agent_worktree.create_agent_worktree = original_create
    join_fleet_children(service)
    worktrees.unlink()
    saved.rename(worktrees)
    db.close()
try:
    sqlite3.Connection.in_transaction.__get__(caller)
except sqlite3.ProgrammingError as error:
    assert 'closed' in str(error)
else:
    raise AssertionError('parent native handle did not retire on owner close')
assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_native_storage_symlink_refusal_settles_durable_child_and_refunds(tmp_path):
    _run(tmp_path, "storage-symlink", "approved", script=_STORAGE_REFUSED_CHILD)
