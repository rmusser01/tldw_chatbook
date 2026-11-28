"""Eight source-frozen disposable first sends through the existing CDP driver."""
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys

E = Path(__file__).resolve().parent
OUT = E / "paired-postsave-current31-rag-path-firstsend-r1"
ROOT = Path(os.environ["FIRSTSEND_SOURCE"])
PYTHON = Path(sys.executable)
BASE_HEAD = "cf788f548edd2ba3eb865776bf8568b27219a447"
CANDIDATE_HEAD = "cf788f548edd2ba3eb865776bf8568b27219a447"
HEAD = None
ROLE = None
PROMPTS = ["Remember the marker CEDAR. Reply exactly CEDAR READY."]

def checked_helper(name, filename, digest):
    path = E / filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def sources():
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == HEAD
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    return {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names if n.endswith(".py") and (ROOT / n).is_file()}

def owned_processes(roots):
    rows = {}
    for line in subprocess.check_output(["ps", "-axo", "pid=,ppid=,comm="], text=True).splitlines():
        pid, parent, comm = line.split(None, 2)
        rows[int(pid)] = {"ppid": int(parent), "comm": comm}
    found = set(roots)
    for _ in range(20):
        extra = {pid for pid, row in rows.items() if row["ppid"] in found}
        if extra <= found:
            break
        found |= extra
    return {str(pid): rows[pid] for pid in found if pid in rows}

async def wait_for(predicate, seconds, message):
    async with asyncio.timeout(seconds):
        while not predicate():
            await asyncio.sleep(0.05)

async def run(number, plan, native):
    out = OUT / f"run{number}"
    out.mkdir()
    prepared = subprocess.run([str(PYTHON), str(E / "pr3023-postsave-current31-rag-path-firstsend-live-launch-r1.py"), "--role", ROLE, "--prepare"], cwd=ROOT, check=True, capture_output=True, text=True)
    row = json.loads(prepared.stdout)
    profile = Path(row["profile"])
    port = 9520 + number
    with socket.socket() as check:
        assert check.connect_ex(("127.0.0.1", port)) != 0, "port already occupied"
    row.update(number=number, root=str(ROOT), port=port, host_load=os.getloadavg(), logical_cpus=os.cpu_count())
    plan["runs"].append(row)
    (OUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    preflight = subprocess.run([str(PYTHON), str(E / "pr3023-postsave-current31-rag-path-firstsend-live-launch-r1.py"), "--role", ROLE, "--profile", str(profile), "--app", "--check"], cwd=ROOT, env=native.child_environment(ROOT, profile, profile / "config.toml"), capture_output=True, text=True)
    (out / "preflight.log").write_text(preflight.stdout + preflight.stderr)
    assert preflight.returncode == 0, "source/settings preflight failed before key access"
    server_log = (out / "server.log").open("w")
    driver_log = (out / "driver.log").open("w")
    server = driver = None
    observations = {}
    exits = {}
    async def command(action, **fields):
        driver.stdin.write((json.dumps({"action": action, **fields}) + "\n").encode())
        await driver.stdin.drain()
        response = await asyncio.wait_for(driver.stdout.readline(), 240)
        assert response, "browser driver exited"
        data = json.loads(response)
        driver_log.write(response.decode()); driver_log.flush()
        assert data.get("ok"), data
        return data
    try:
        server = await asyncio.create_subprocess_exec(str(PYTHON), "-u", str(E / "pr3023-postsave-current31-rag-path-firstsend-live-launch-r1.py"), "--role", ROLE, "--profile", str(profile), "--port", str(port), cwd=ROOT, stdout=server_log, stderr=server_log)
        async def listening():
            for _ in range(1200):
                assert server.returncode is None, "server exited during startup"
                try:
                    reader, writer = await asyncio.open_connection("127.0.0.1", port)
                    writer.close(); await writer.wait_closed(); return
                except OSError:
                    await asyncio.sleep(0.05)
            raise TimeoutError("server did not listen")
        await listening()
        driver = await asyncio.create_subprocess_exec(str(PYTHON), "-u", str(E / "pr3023-cdp-driver-preparation-r3.py"), "--out", str(out / "browser"), "--width", "1084", "--height", "738", stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=driver_log)
        ready = json.loads(await asyncio.wait_for(driver.stdout.readline(), 60))
        assert ready.get("ready")
        await command("navigate", url=f"http://127.0.0.1:{port}")
        await command("wait_text", text="Send", seconds=120)
        await command("capture", name="ready")
        session = json.loads((out / "browser/ready.json").read_text())
        assert [session["columns"], session["rows"]] == [120, 35]
        observations = owned_processes((server.pid, driver.pid))
        (out / "owned-processes-before.json").write_text(json.dumps({"owned_observed": observations}, indent=2) + "\n")
        await command("click_text", text="Ask, command, or paste task...")
        for index, prompt in enumerate(PROMPTS, 1):
            await command("wait_text", text="Send", seconds=60)
            if index > 1:
                await command("click_text", text="Ask, command, or paste task...")
            await command("type", text=prompt)
            await command("wait_text", text="READY.", seconds=5)
            await command("capture", name=f"draft-{index}")
            await command("key", key="Enter")
            await command("wait_http_send", path=str(profile / "http.jsonl"), request=index, seconds=60)
            def completed():
                records = [json.loads(x) for x in (profile / "http.jsonl").read_text().splitlines()]
                return any(x["event"] == "response_closed" and x["request"] == index for x in records)
            await wait_for(completed, 190, "HTTP completion")
            await asyncio.sleep(2)
            await command("wait_text", text="Send", seconds=60)
            await command("capture", name=f"response-{index}")
            print(json.dumps({"run": number, "request": index, "http_closed": True}), flush=True)
        await command("key", key="Control+q")
        await wait_for(lambda: (profile / "app-returned.json").exists(), 60, "normal app return")
    finally:
        cleanup_errors = []
        if driver is not None:
            if not (profile / "app-returned.json").exists():
                try:
                    await command("key", key="Control+q")
                    await wait_for(lambda: (profile / "app-returned.json").exists(), 60, "error-path app return")
                except Exception as error:
                    print(json.dumps({"cleanup_error": type(error).__name__}), flush=True)
            try:
                await command("close")
            except Exception:
                pass
            driver.stdin.close()
            try:
                exits["driver"] = await asyncio.wait_for(driver.wait(), 60)
            except Exception as error:
                cleanup_errors.append(error)
        if server is not None:
            try:
                if server.returncode is None:
                    server.send_signal(signal.SIGINT)
                exits["server"] = await asyncio.wait_for(server.wait(), 60)
            except Exception as error:
                cleanup_errors.append(error)
        server_log.close(); driver_log.close()
        (out / "exit-codes.json").write_text(json.dumps(exits) + "\n")
        assert sources() == json.loads((OUT / f"{ROLE}-sources-before.json").read_text())
        if cleanup_errors:
            raise ExceptionGroup("Owned measurement cleanup failed", cleanup_errors)
    assert exits == {"driver": 0, "server": 0}, exits
    subprocess.run([str(PYTHON), str(E / "pr3023-postsave-current31-rag-path-firstsend-live-audit-r1.py"), "--run", str(number)], cwd=ROOT, check=True)

async def main():
    global HEAD, ROLE
    assert not OUT.exists(), "never overwrite a measurement packet"
    assert CANDIDATE_HEAD == "cf788f548edd2ba3eb865776bf8568b27219a447"
    assert subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() == "codex/console-postsave-current31"
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == CANDIDATE_HEAD
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    OUT.mkdir(mode=0o700)
    driver = E / "pr3023-cdp-driver-preparation-r3.py"
    assert hashlib.sha256(driver.read_bytes()).hexdigest() == "26f307e0c408d860cb5215851b0b86703969eb8287068f6b682c0eed8aa4979e"
    native = checked_helper("native_environment", "fixture_environment.py", "19da2f2d27b5ae6e9933a6bb587d8564af596e75ccca318e85ce63dced3a4975")
    order = ("baseline",)
    plan = {"heads": {"baseline": BASE_HEAD, "candidate": CANDIDATE_HEAD}, "order": order, "runs": [], "prompts": PROMPTS, "max_posts": 3, "planned_posts": 1, "posts_per_process": 1, "http_observer_hard_cap_per_process": 3, "profiler_enabled": True, "think_time_seconds": 2, "submit_input": "Enter for every send; no Send button clicking", "draft_input": "atomic CDP Input.insertText; rendered READY. suffix then full persisted prompt audit; actual Send readiness before drafts and after HTTP close", "scope": "one profiled first send on the frozen dev source; explicit composer focus and exact-input audits; served CDP, no native paint or population p95"}
    for number, ROLE in enumerate(order, 1):
        HEAD = plan["heads"][ROLE]
        assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        # The preceding app, driver and server must be audited closed before switching.
        subprocess.run(["git", "switch", "--detach", HEAD], cwd=ROOT, check=True)
        source_path = OUT / f"{ROLE}-sources-before.json"
        current = sources()
        if source_path.exists():
            assert current == json.loads(source_path.read_text())
        else:
            source_path.write_text(json.dumps(current, sort_keys=True) + "\n")
        await run(number, plan, native)
    subprocess.run(["git", "switch", "codex/console-postsave-current31"], cwd=ROOT, check=True)
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == CANDIDATE_HEAD
    plan["status"] = "one closed audited profile and one instrumented first send"
    (OUT / "plan-closed.json").write_text(json.dumps(plan, indent=2) + "\n")

if __name__ == "__main__":
    asyncio.run(main())
