import threading
import time
import json
import Tests.Chat.test_console_agent_bridge as T


def test_probe_two_children(tmp_path):
    gate = threading.Event()
    gateway = T._FleetTwoChildGateway(
        parent_script=[
            [T._fence("spawn_subagent", {"task": "task A"})],
            [T._fence("spawn_subagent", {"task": "task B"})],
            ["parent final"],
        ],
        child_result=["child answer"],
        gate=gate,
    )
    db = T.AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = T.ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=T.ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=T.ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = T.ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=gateway
    )
    captured = {}
    bridge._teardown_fleet_service = lambda cid, svc: captured.setdefault("svc", svc)
    result = {}
    marks = []

    def do_run():
        result["outcome"] = T._run(
            bridge, store, session, assistant.id, conversation_id="conv-two-children"
        )
        marks.append(("run_returned", time.monotonic()))

    runner = threading.Thread(target=do_run)
    t_start = time.monotonic()
    runner.start()
    assert gateway.entered_event.wait(5)
    marks.append(("both_children_gated", time.monotonic()))
    marks.append(("runner_alive_at_peek", runner.is_alive()))
    gate.set()
    marks.append(("gate_set", time.monotonic()))
    runner.join(10)
    marks.append(("joined", time.monotonic()))
    svc = captured.get("svc")
    if svc is not None:
        marks.append(("statuses_at_join", [h.status for h in svc.fleet_snapshot()]))
        t0 = time.monotonic()
        for _ in range(5000):
            snap = svc.fleet_snapshot()
            if snap and all(h.status != "running" for h in snap):
                break
            time.sleep(0.0005)
        marks.append(
            ("ms_until_all_terminal", round((time.monotonic() - t0) * 1000, 2))
        )
    out = [
        (m[0], (round((m[1] - t_start) * 1000, 2) if isinstance(m[1], float) else m[1]))
        for m in marks
    ]
    print("PROBE2", json.dumps(out))
