import threading, time, json
from Tests.Chat.test_console_agent_bridge import _bridge, _run, _fence


def test_probe_child_completion_timing(tmp_path, monkeypatch):
    results = []
    for i in range(30):
        scripts = [
            [_fence("spawn_subagent", {"task": "compute 1+1"})],
            ["2"],
            ["Done: ", "2."],
        ]
        d = tmp_path / f"i{i}"
        d.mkdir()
        bridge, db, store, session, aid = _bridge(d, scripts)
        captured = {}
        real = bridge._teardown_fleet_service

        def fake(cid, svc, _c=captured):
            _c["svc"] = svc
        bridge._teardown_fleet_service = fake
        outcome = _run(bridge, store, session, aid)
        svc = captured["svc"]
        t0 = time.monotonic()
        first = [h.status for h in svc.fleet_snapshot()]
        waited = None
        for _ in range(2000):
            snap = svc.fleet_snapshot()
            if snap and all(h.status != "running" for h in snap):
                waited = time.monotonic() - t0
                break
            time.sleep(0.001)
        results.append((first, round((waited or -1) * 1000, 2)))
    print("PROBE", json.dumps(results))
    assert True
