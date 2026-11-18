"""Private extraction delegates preserve the existing approval owners and seams."""

from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.bootstrap_profile
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("fails", [False, True])
def test_local_persistence_keeps_writer_objects_and_best_effort_failure(
    monkeypatch, fails
):
    from tldw_chatbook.Agents import local_approval_state as helper
    from tldw_chatbook.Agents import local_tool_provider as module

    hub = SimpleNamespace(name="fs_read")
    decision = "approve_session"
    writer_calls = []
    delegated = []
    failure = RuntimeError("synthetic writer failure")

    def writer(actual_hub, actual_decision):
        writer_calls.append((actual_hub, actual_decision))
        if fails:
            raise failure
        return object()

    owner = SimpleNamespace(_persist_approval=writer)
    original = helper.persist_local_approval

    def delegate(provider, actual_hub, actual_decision, *, log):
        delegated.append((provider, actual_hub, actual_decision, log))
        return original(provider, actual_hub, actual_decision, log=log)

    monkeypatch.setattr(helper, "persist_local_approval", delegate)
    assert module.LocalToolProvider._persist_approval_safe(owner, hub, decision) is None
    assert len(delegated) == len(writer_calls) == 1
    assert delegated[0][0] is owner and delegated[0][1] is hub
    assert delegated[0][2] is decision and delegated[0][3] is module.logger
    assert writer_calls[0][0] is hub and writer_calls[0][1] is decision


def test_local_scope_restores_exact_run_after_original_body_error(monkeypatch):
    from tldw_chatbook.Agents import local_approval_state as helper
    from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider

    run, other_run = "run", "other"
    original_stamp, sibling_stamp, index_stamp = object(), object(), object()
    owner = SimpleNamespace(
        _stamps_lock=threading.Lock(),
        _stamps={(run, "tool"): original_stamp, (other_run, "tool"): sibling_stamp},
        _index_call_stamps={(run, "call"): object(), (other_run, "call"): index_stamp},
    )
    original = helper.local_stamp_scope
    seen = []

    @contextmanager
    def delegate(provider, actual_run):
        seen.append((provider, actual_run))
        with original(provider, actual_run):
            yield

    monkeypatch.setattr(helper, "local_stamp_scope", delegate)
    failure = RuntimeError("original nested body")
    outer_stamp, inner_stamp = object(), object()
    with pytest.raises(RuntimeError) as caught:
        with LocalToolProvider.stamp_scope(owner, run):
            assert (run, "tool") not in owner._stamps
            owner._stamps[(run, "tool")] = outer_stamp
            with pytest.raises(RuntimeError) as inner:
                with LocalToolProvider.stamp_scope(owner, run):
                    assert (run, "tool") not in owner._stamps
                    assert owner._stamps[(other_run, "tool")] is sibling_stamp
                    owner._stamps[(run, "tool")] = inner_stamp
                    raise failure
            assert inner.value is failure
            assert owner._stamps[(run, "tool")] is outer_stamp
            raise inner.value
    assert caught.value is failure
    assert seen == [(owner, run), (owner, run)]
    assert owner._stamps[(run, "tool")] is original_stamp
    assert owner._stamps[(other_run, "tool")] is sibling_stamp
    assert (run, "call") not in owner._index_call_stamps
    assert owner._index_call_stamps[(other_run, "call")] is index_stamp


@pytest.mark.parametrize("stale", [False, True])
def test_session_grant_keeps_live_dependencies_and_original_error(monkeypatch, stale):
    from tldw_chatbook.MCP import session_approval as helper
    from tldw_chatbook.MCP import unified_control_plane_service as module

    owner = object.__new__(module.UnifiedMCPControlPlaneService)
    owner._session_approvals_lock = threading.Lock()
    owner._session_approvals = set()
    owner._permission_store = None
    owner.local_service = SimpleNamespace(store=None)
    seen, observations = [], []
    lifecycle, digest = object(), object()
    failure = RuntimeError("original mutation refusal")

    def mutation_error(code):
        assert code == "stale_profile"
        return failure

    def publish(*args, **kwargs):
        if args == ("applied",):
            assert ("profile", "server", "tool") in owner._session_approvals
        observations.append((args, kwargs))

    monkeypatch.setattr(module, "profile_lifecycle_disposition", lifecycle)
    monkeypatch.setattr(module, "profile_policy_digest", digest)
    monkeypatch.setattr(module, "ProfileMutationError", mutation_error)
    monkeypatch.setattr(module, "_publish_grant_application", publish)
    original = helper.apply_session_approval

    def delegate(service, server_key, tool_name, **kwargs):
        seen.append((service, server_key, tool_name, kwargs))
        return original(service, server_key, tool_name, **kwargs)

    monkeypatch.setattr(helper, "apply_session_approval", delegate)
    kwargs = dict(
        profile_id="profile",
        expected_profile_digest="captured" if stale else None,
        expected_revision=None,
    )
    if stale:
        with pytest.raises(RuntimeError) as caught:
            owner.approve_for_session("server", "tool", **kwargs)
        assert caught.value is failure
        assert not owner._session_approvals
        assert observations == [(("failed",), {"error_code": "writer_failed"})]
    else:
        assert owner.approve_for_session("server", "tool", **kwargs) is None
        assert observations == [(("applied",), {"actual_scope": "approve_session"})]
    assert len(seen) == 1 and seen[0][0] is owner
    dependencies = seen[0][3]
    assert dependencies["publish"] is publish
    assert (
        dependencies["lifecycle"] is lifecycle
        and dependencies["policy_digest"] is digest
    )
    assert dependencies["mutation_error"] is mutation_error


def test_bridge_projection_keeps_fact_callkey_and_replace_objects(monkeypatch):
    from tldw_chatbook.Chat import console_agent_bridge as module
    from tldw_chatbook.Chat import console_tool_activity as helper

    facts = (object(),)
    def keys(identity):
        return ("call",)
    received = []
    activity = SimpleNamespace(
        session_id="session",
        feedback=lambda actual, **kwargs: received.append((actual, kwargs)),
    )
    owner = SimpleNamespace(
        approval_feedback_store=SimpleNamespace(
            snapshot=lambda session, run: facts, projection_call_keys=keys
        ),
        _tool_activity_runs={"run": activity},
        _raw_shell_marker_lock=threading.Lock(),
        _raw_shell_markers={},
    )
    original = helper.project_approval_feedback
    seen = []

    def delegate(bridge, session_id, run_id, *, replace_presentation):
        seen.append((bridge, session_id, run_id, replace_presentation))
        return original(
            bridge, session_id, run_id, replace_presentation=replace_presentation
        )

    monkeypatch.setattr(helper, "project_approval_feedback", delegate)
    assert (
        module.ConsoleAgentBridge.project_approval_feedback(owner, "session", "run")
        is None
    )
    assert (
        len(seen) == 1
        and seen[0][0] is owner
        and seen[0][3] is module.dataclass_replace
    )
    assert received[0][0] is facts and received[0][1]["call_keys"] is keys


def test_bridge_step_keeps_original_step_and_public_projection_seam(monkeypatch):
    from tldw_chatbook.Chat import console_agent_bridge as module
    from tldw_chatbook.Chat import console_tool_activity as helper

    projected, seen = [], []
    owner = SimpleNamespace(
        approval_feedback_store=SimpleNamespace(
            context_for_call=lambda *args, **kw: None, snapshot=lambda *args: ()
        ),
        project_approval_feedback=lambda *args: projected.append(args),
    )
    step = SimpleNamespace(
        kind="model_request_started", call_id="", tool_name="", tool_outcome=""
    )
    original = helper.observe_approval_step

    def delegate(bridge, session_id, run_id, actual_step):
        seen.append((bridge, actual_step))
        return original(bridge, session_id, run_id, actual_step)

    monkeypatch.setattr(helper, "observe_approval_step", delegate)
    assert (
        module.ConsoleAgentBridge._observe_approval_step(owner, "session", "run", step)
        is None
    )
    assert len(seen) == 1 and seen[0][0] is owner and seen[0][1] is step
    assert projected == [("session", "run")]


def test_tool_detail_keeps_literal_text_and_conditional_feedback(monkeypatch):
    from tldw_chatbook.Chat import console_tool_activity as helper
    from tldw_chatbook.Chat import console_approval_feedback as feedback

    fact = object()
    calls = []

    def format_feedback(actual, *, tool_status):
        calls.append((actual, tool_status))
        return "Decision accepted"

    monkeypatch.setattr(feedback, "format_approval_feedback", format_feedback)
    activity = SimpleNamespace(
        raw_cli_presentation=None, tool_output_full="literal [output]"
    )
    presentation = SimpleNamespace(
        approval_feedback=None,
        arguments="literal [args]",
        status="running",
        result_preview="preview",
    )
    assert (
        helper.format_tool_activity_detail(activity, presentation)
        == "Arguments\nliteral [args]\n\nLive output\nliteral [output]"
    )
    assert calls == []
    presentation.approval_feedback = fact
    assert (
        helper.format_tool_activity_detail(activity, presentation)
        == "Decision accepted\n\nArguments\nliteral [args]\n\nLive output\nliteral [output]"
    )
    assert calls == [(fact, "running")]


_COLD_HELPERS = r"""
import importlib.abc, importlib.util, os, sys
from pathlib import Path
repo = Path(sys.argv[1]).resolve()
assert not any(n == p or n.startswith(p+'.') for n in sys.modules for p in ('tldw_chatbook','tldw_profile_core'))
sys.path.insert(0,str(repo));sys.path.insert(1,str(repo/'packages/tldw_profile_core/src'))
for name,expected in {'Tests':repo/'Tests/__init__.py','tldw_chatbook':repo/'tldw_chatbook/__init__.py','tldw_profile_core':repo/'packages/tldw_profile_core/src/tldw_profile_core/__init__.py'}.items():
    assert Path(importlib.util.find_spec(name).origin).resolve()==expected.resolve()
os.environ['TLDW_TEST_REAL_HOME']=sys.argv[2]
from Tests import real_profile_guard
real_profile_guard.install()
assert Path.home().resolve()!=real_profile_guard._real_home().resolve()
network=[]
def audit(event,args):
    if event in {'socket.connect','socket.getaddrinfo','socket.bind','socket.sendto'}:
        network.append(event);raise RuntimeError('test_network_refused')
sys.addaudithook(audit)
from tldw_chatbook.Backup_Recovery import bootstrap
assert bootstrap.startup_permission(bootstrap.effective_config_path(),bootstrap.default_bootstrap_root())[0]
blocked=('tldw_chatbook.Agents.approval_observation','tldw_chatbook.Chat.console_approval_feedback','numpy','torch','tldw_chatbook.RAG_Search.simplified')
assert not any(n==p or n.startswith(p+'.') for n in sys.modules for p in blocked)
loader = sys.modules['importlib._bootstrap']
find_code, load_code, query_code = loader._find_spec.__code__, loader._find_and_load_unlocked.__code__, importlib.util.find_spec.__code__
bootstrap_globals, query_globals = vars(loader), vars(importlib.util)
attempts=[]
class RefuseEagerObserver(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if any(fullname==p or fullname.startswith(p+'.') for p in blocked):
            caller=sys._getframe(1)
            if (caller.f_code is find_code and caller.f_globals is bootstrap_globals
                    and caller.f_back.f_code is query_code and caller.f_back.f_globals is query_globals):
                return None
            assert caller.f_code is find_code and caller.f_globals is bootstrap_globals
            assert caller.f_back.f_code is load_code and caller.f_back.f_globals is bootstrap_globals
            attempts.append(fullname)
            raise RuntimeError('unexpected_eager_import')
sys.meta_path.insert(0,RefuseEagerObserver())
assert importlib.util.find_spec('tldw_chatbook.Agents.approval_observation') is not None
assert 'tldw_chatbook.Agents.approval_observation' not in sys.modules
print('REAL_GUARDED_HELPER_IMPORT_ENTER',flush=True)
import tldw_chatbook.Agents.local_approval_state
import tldw_chatbook.MCP.session_approval
import tldw_chatbook.Chat.console_tool_activity
assert not network and not real_profile_guard.take_violations() and not attempts
assert not any(n==p or n.startswith(p+'.') for n in sys.modules for p in blocked)
try:
    __import__('tldw_chatbook.Agents.approval_observation')
except RuntimeError:
    assert attempts==['tldw_chatbook.Agents.approval_observation']
else:
    raise AssertionError('Import blocker did not observe actual initialization')
assert 'tldw_chatbook.Agents.approval_observation' not in sys.modules
print('REAL_GUARDED_HELPER_IMPORT_OK',flush=True)
"""


def test_real_guarded_helpers_leave_optional_observers_lazy(tmp_path):
    from Tests import real_profile_guard

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-X",
            "utf8",
            "-c",
            _COLD_HELPERS,
            str(ROOT),
            str(real_profile_guard._real_home()),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    (tmp_path / "helper-import.stdout.txt").write_text(
        completed.stdout, encoding="utf-8"
    )
    (tmp_path / "helper-import.stderr.txt").write_text(
        completed.stderr, encoding="utf-8"
    )
    assert "REAL_GUARDED_HELPER_IMPORT_ENTER" in completed.stdout, completed.stderr
    assert completed.returncode == 0, completed.stderr
    assert "REAL_GUARDED_HELPER_IMPORT_OK" in completed.stdout
