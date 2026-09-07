"""Both permission gates must key verdicts per RUN, not per tool name.

Pre-fix, `BuiltinToolGate.begin_turn()` cleared a dict keyed by tool name
on a SHARED gate instance, and `MCPToolProvider.apply_batch_decisions`
REPLACED its dict wholesale -- so with concurrent children, one child's
turn wipes or overwrites a verdict a sibling (or the parent) already
decided and has not yet consumed.
"""

import threading

from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate
from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider


def test_builtin_gate_stamps_do_not_leak_across_runs():
    gate = BuiltinToolGate(service=None)
    gate.begin_turn("run-parent")
    gate.stamp("run-parent", "calculator", "proceed")
    # A concurrent child starts its own turn on the SAME gate instance.
    gate.begin_turn("run-child")
    gate.stamp("run-child", "calculator", "deny")
    # The parent's verdict must survive the child's turn untouched.
    assert gate.stamped("run-parent", "calculator") == "proceed"
    assert gate.stamped("run-child", "calculator") == "deny"


def test_builtin_gate_begin_turn_clears_only_its_own_run():
    gate = BuiltinToolGate(service=None)
    gate.begin_turn("run-a")
    gate.stamp("run-a", "calculator", "proceed")
    gate.begin_turn("run-b")
    assert gate.stamped("run-a", "calculator") == "proceed"
    gate.begin_turn("run-a")  # a's NEXT turn clears a's stamps
    assert gate.stamped("run-a", "calculator") is None


def test_mcp_decisions_do_not_clobber_across_runs():
    provider = MCPToolProvider.__new__(MCPToolProvider)
    provider._init_decision_state()  # helper added by this task
    provider.apply_batch_decisions("run-parent", {"srv__tool": "proceed"})
    provider.apply_batch_decisions("run-child", {"srv__tool": "deny"})
    assert provider.stamped_decision("run-parent", "srv__tool") == "proceed"
    assert provider.stamped_decision("run-child", "srv__tool") == "deny"


def test_local_decisions_do_not_clobber_across_runs(tmp_path):
    """The THIRD gate with this exact shape, and the one with no other
    coverage of its per-run keying.

    ``LocalToolProvider`` shares one instance across a parent run and every
    sub-agent it spawns, exactly like the two gates above, and its stamps
    were name-keyed and REPLACED wholesale for the same reasons. Review of
    the first cut found both of its per-run mutations survived the entire
    suite (1012 passed): reverting ``apply_batch_decisions`` to a
    whole-dict replace, and making ``stamped`` ignore ``run_id`` -- the
    latter a genuine FAIL-OPEN, since a sibling's ``approve_once`` would
    then permit this run's call. Both are red against this test.
    """
    provider = LocalToolProvider(workspace_root=tmp_path)
    provider.apply_batch_decisions("run-parent", {"fs_list": "approve_once"})
    provider.apply_batch_decisions("run-child", {"fs_list": "deny"})

    # The child's turn must not have replaced the parent's slice.
    assert provider.stamped("run-parent", "fs_list") == "approve_once"
    assert provider.stamped("run-child", "fs_list") == "deny"
    # The fail-open direction, stated explicitly: a run that was never
    # granted anything must never READ a sibling's verdict, permitting or
    # otherwise. A name-keyed lookup would hand this run one of the two
    # above.
    assert provider.stamped("run-other", "fs_list") is None


def test_concurrent_runs_keep_their_own_verdicts():
    gate = BuiltinToolGate(service=None)
    errors = []

    def worker(i):
        run = f"run-{i}"
        # try/except, not a bare body: a raise inside a Thread target is
        # swallowed by threading (pytest only warns), so without this the
        # whole test passes vacuously whenever the gate's signature is
        # wrong -- exactly the pre-fix state this test is meant to catch.
        try:
            gate.begin_turn(run)
            gate.stamp(run, "calculator", f"verdict-{i}")
            for _ in range(200):
                if gate.stamped(run, "calculator") != f"verdict-{i}":
                    errors.append(run)
                    return
        except Exception as exc:  # noqa: BLE001 -- surfaced as a failure below
            errors.append(f"{run}: {exc!r}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
