"""Tests for the Console run-hooks engine (spec: 2026-09-11-console-run-hooks-design)."""

import asyncio
import json
import os
import signal
import sys
import time

import pytest
from loguru import logger as _loguru_logger

from tldw_chatbook.Agents import run_hooks
from tldw_chatbook.Agents.agent_models import ToolCall
from tldw_chatbook.Agents.run_hooks import (
    HOOK_DEFAULT_TIMEOUT_S,
    HOOK_IO_BUDGET_CHARS,
    HookOutcome,
    HookSpec,
    RunHooksConfig,
    RunHooksEngine,
    load_hooks_config,
)


@pytest.fixture
def caplog(caplog):
    """Bridge loguru records into pytest's caplog (loguru's documented recipe).

    loguru does not propagate to stdlib logging in this repo, so tests that
    assert on engine log output attach caplog.handler as a loguru sink.
    """
    handler_id = _loguru_logger.add(caplog.handler, format="{message}")
    yield caplog
    _loguru_logger.remove(handler_id)


@pytest.fixture(autouse=True)
def close_engines(monkeypatch):
    """Each test owns its live engines until all process cleanup completes."""
    engines = []
    original = RunHooksEngine.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        engines.append(self)

    monkeypatch.setattr(RunHooksEngine, "__init__", initialize)
    yield
    for engine in engines:
        engine.close()
        for pool in (engine._notify_worker, engine._notification_pool, engine._pool):
            pool.shutdown(wait=True, cancel_futures=True)


def _cfg(enabled=True, **hook_kwargs):
    base = {"event": "PreToolUse", "command": ["/bin/true"]}
    base.update(hook_kwargs)
    return {"hooks": {"enabled": enabled, "hook": [base]}}


class TestLoadHooksConfig:
    def test_empty_section_is_no_op(self):
        result = load_hooks_config({})
        assert result == RunHooksConfig(enabled=True, hooks=())

    def test_master_switch_off(self):
        result = load_hooks_config({"hooks": {"enabled": False}})
        assert result.enabled is False

    def test_valid_hook_parsed(self):
        result = load_hooks_config(
            {
                "hooks": {
                    "hook": [
                        {
                            "event": "PreToolUse",
                            "matcher": "fs_*",
                            "command": ["/bin/guard", "--strict"],
                            "timeout_s": 5,
                        }
                    ]
                }
            }
        )
        assert result.hooks == (
            HookSpec("PreToolUse", ("/bin/guard", "--strict"), "fs_*", 5.0),
        )

    def test_unknown_event_disables_that_hook(self):
        result = load_hooks_config(
            {"hooks": {"hook": [{"event": "Nope", "command": ["/bin/true"]}]}}
        )
        assert result.hooks == ()

    def test_matcher_rejected_on_non_tool_event(self):
        result = load_hooks_config(
            {
                "hooks": {
                    "hook": [
                        {"event": "Stop", "matcher": "fs_*", "command": ["/bin/true"]}
                    ]
                }
            }
        )
        assert result.hooks == ()

    def test_empty_command_rejected(self):
        result = load_hooks_config(
            {"hooks": {"hook": [{"event": "Stop", "command": []}]}}
        )
        assert result.hooks == ()

    def test_non_positive_timeout_rejected(self):
        result = load_hooks_config(
            {
                "hooks": {
                    "hook": [
                        {"event": "Stop", "command": ["/bin/true"], "timeout_s": 0}
                    ]
                }
            }
        )
        assert result.hooks == ()

    def test_string_command_rejected(self):
        result = load_hooks_config(
            {"hooks": {"hook": [{"event": "Stop", "command": "/bin/true"}]}}
        )
        assert result.hooks == ()

    def test_default_timeout_applied(self):
        result = load_hooks_config(
            {"hooks": {"hook": [{"event": "Stop", "command": ["/bin/true"]}]}}
        )
        assert result.hooks[0].timeout_s == HOOK_DEFAULT_TIMEOUT_S

    def test_hook_key_not_a_list_disables_hooks(self):
        result = load_hooks_config({"hooks": {"enabled": True, "hook": None}})
        assert result == RunHooksConfig(enabled=True, hooks=())

    def test_unhashable_event_disables_that_hook(self):
        result = load_hooks_config(
            {"hooks": {"hook": [{"event": ["Stop"], "command": ["/bin/true"]}]}}
        )
        assert result.hooks == ()

    def test_nan_timeout_rejected(self):
        result = load_hooks_config(
            {
                "hooks": {
                    "hook": [
                        {
                            "event": "Stop",
                            "command": ["/bin/true"],
                            "timeout_s": float("nan"),
                        }
                    ]
                }
            }
        )
        assert result.hooks == ()

    def test_inf_timeout_rejected(self):
        result = load_hooks_config(
            {
                "hooks": {
                    "hook": [
                        {
                            "event": "Stop",
                            "command": ["/bin/true"],
                            "timeout_s": float("inf"),
                        }
                    ]
                }
            }
        )
        assert result.hooks == ()

    def test_bool_timeout_rejected(self):
        result = load_hooks_config(
            {
                "hooks": {
                    "hook": [
                        {"event": "Stop", "command": ["/bin/true"], "timeout_s": True}
                    ]
                }
            }
        )
        assert result.hooks == ()


# ---------------------------------------------------------------------------
# Task 2: engine execution core
# ---------------------------------------------------------------------------


def _engine(*hooks, enabled=True):
    cfg = RunHooksConfig(enabled=enabled, hooks=tuple(hooks))
    return RunHooksEngine(lambda: cfg, lambda: os.getcwd())


class TestFire:
    def test_exit0_clean_pass_logs_stdout(self, caplog):
        eng = _engine(HookSpec("Stop", (sys.executable, "-c", "print('done')")))
        out = eng.fire("Stop", session_id="s", run_id="r", data={})
        assert out == HookOutcome(blocked=False, reason="", context="")
        # R13: non-blocking events log captured stdout/stderr at INFO carrying
        # session_id/run_id/hook event.
        info_records = [
            r
            for r in caplog.records
            if r.levelname == "INFO" and "event=Stop" in r.getMessage()
        ]
        assert info_records, (
            f"no INFO record for the Stop hook: {[r.getMessage() for r in caplog.records]}"
        )
        msg = info_records[0].getMessage()
        assert "done" in msg and "session_id=s" in msg and "run_id=r" in msg

    def test_json_decision_beats_exit_code(self):
        code = "import sys; print(__import__('json').dumps({'decision': 'deny', 'reason': 'nope'}))"
        eng = _engine(HookSpec("PreToolUse", (sys.executable, "-c", code)))
        out = eng.fire("PreToolUse", session_id="s", data={"tool_name": "fs_write"})
        assert out.blocked is True and out.reason == "nope"

    def test_exit2_is_deny_shorthand(self):
        eng = _engine(
            HookSpec(
                "PreToolUse",
                (
                    sys.executable,
                    "-c",
                    "import sys; sys.stderr.write('blocked'); sys.exit(2)",
                ),
            )
        )
        out = eng.fire("PreToolUse", session_id="s", data={"tool_name": "t"})
        assert out.blocked is True and "blocked" in out.reason

    def test_allow_decision_ignored(self):
        code = "print(__import__('json').dumps({'decision': 'allow'}))"
        eng = _engine(HookSpec("PreToolUse", (sys.executable, "-c", code)))
        out = eng.fire("PreToolUse", session_id="s", data={"tool_name": "t"})
        assert out.blocked is False

    def test_pretooluse_crash_fails_closed(self):
        eng = _engine(HookSpec("PreToolUse", (sys.executable, "-c", "sys.exit(1)")))
        out = eng.fire("PreToolUse", session_id="s", data={"tool_name": "t"})
        assert out.blocked is True and "failed" in out.reason

    def test_userpromptsubmit_crash_fails_open(self, caplog):
        eng = _engine(
            HookSpec("UserPromptSubmit", (sys.executable, "-c", "sys.exit(1)"))
        )
        out = eng.fire("UserPromptSubmit", session_id="s", data={"prompt": "hi"})
        assert out.blocked is False
        # Fail-open paths log at WARNING (spec §5), not silently at INFO.
        assert any(
            r.levelname == "WARNING"
            and "UserPromptSubmit" in r.getMessage()
            and "failing open" in r.getMessage()
            for r in caplog.records
        ), (
            f"expected WARNING for UPS crash fail-open, got: "
            f"{[(r.levelname, r.getMessage()) for r in caplog.records]}"
        )

    def test_userpromptsubmit_stdout_is_context(self):
        eng = _engine(
            HookSpec("UserPromptSubmit", (sys.executable, "-c", "print('extra ctx')"))
        )
        out = eng.fire("UserPromptSubmit", session_id="s", data={"prompt": "hi"})
        assert out.blocked is False and out.context == "extra ctx"

    def test_userpromptsubmit_crash_stdout_not_injected(self):
        # Ruling R31: injection requires exit 0 AND no parsed decision key
        # (spec §5: "exit 0, no JSON: stdout is prepended as context"). A
        # crashed hook's parting words are not context — fail open, inject
        # nothing.
        eng = _engine(
            HookSpec(
                "UserPromptSubmit",
                (sys.executable, "-c", "import sys; print('boom'); sys.exit(1)"),
            )
        )
        out = eng.fire("UserPromptSubmit", session_id="s", data={"prompt": "hi"})
        assert out == HookOutcome(blocked=False, reason="", context="")

    def test_userpromptsubmit_ignored_decision_json_not_injected(self):
        # Ruling R31: an ignored decision key means the hook spoke the
        # protocol and offered no opinion — its raw JSON must not leak in as
        # context. Injection is exit-0-and-no-decision only.
        code = "import json; print(json.dumps({'decision': 'allow'}))"
        eng = _engine(HookSpec("UserPromptSubmit", (sys.executable, "-c", code)))
        out = eng.fire("UserPromptSubmit", session_id="s", data={"prompt": "hi"})
        assert out == HookOutcome(blocked=False, reason="", context="")

    def test_userpromptsubmit_block(self):
        eng = _engine(
            HookSpec(
                "UserPromptSubmit",
                (
                    sys.executable,
                    "-c",
                    "print(__import__('json').dumps({'decision': 'block', 'reason': 'no'}))",
                ),
            )
        )
        out = eng.fire("UserPromptSubmit", session_id="s", data={"prompt": "hi"})
        assert out.blocked is True and out.reason == "no"

    def test_userpromptsubmit_timeout_fails_open_warns(self, caplog):
        eng = _engine(
            HookSpec(
                "UserPromptSubmit",
                (sys.executable, "-c", "import time; time.sleep(2)"),
                timeout_s=0.5,
            )
        )
        out = eng.fire("UserPromptSubmit", session_id="s", data={"prompt": "hi"})
        assert out.blocked is False  # timeout fails open
        assert any(
            r.levelname == "WARNING"
            and "UserPromptSubmit" in r.getMessage()
            and "timed out" in r.getMessage()
            and "failing open" in r.getMessage()
            for r in caplog.records
        ), (
            f"expected WARNING for UPS timeout fail-open, got: "
            f"{[(r.levelname, r.getMessage()) for r in caplog.records]}"
        )

    def test_nonblocking_logged_output_truncated(self, caplog):
        eng = _engine(HookSpec("Stop", (sys.executable, "-c", "print('z' * 999999)")))
        eng.fire("Stop", session_id="s", run_id="r", data={})
        stop_records = [
            r
            for r in caplog.records
            if r.levelname == "INFO" and "event=Stop" in r.getMessage()
        ]
        assert stop_records, (
            f"no INFO record for the Stop hook: {[r.getMessage() for r in caplog.records]}"
        )
        assert "…[truncated]" in stop_records[0].getMessage()

    def test_blocking_fire_info_record_carries_ids(self, caplog):
        # Ruling R27: blocking fires' INFO records carry session_id/run_id so
        # a user-visible refusal is correlatable in the logs the same way the
        # non-blocking branch's records are.
        eng = _engine(HookSpec("PreToolUse", (sys.executable, "-c", "pass")))
        out = eng.fire(
            "PreToolUse",
            session_id="s-blocking",
            run_id="r-blocking",
            data={"tool_name": "t"},
        )
        assert out.blocked is False
        info_records = [
            r
            for r in caplog.records
            if r.levelname == "INFO" and "event=PreToolUse" in r.getMessage()
        ]
        assert info_records, (
            f"no INFO record for the PreToolUse hook: "
            f"{[r.getMessage() for r in caplog.records]}"
        )
        msg = info_records[0].getMessage()
        assert "session_id=s-blocking" in msg and "run_id=r-blocking" in msg

    # --- Ruling R12: a JSON object with a "decision" key suppresses exit codes ---

    def test_pretooluse_allow_with_exit2_not_blocked(self, caplog):
        code = (
            "import sys; print(__import__('json').dumps({'decision': 'allow'})); "
            "sys.exit(2)"
        )
        eng = _engine(HookSpec("PreToolUse", (sys.executable, "-c", code)))
        out = eng.fire("PreToolUse", session_id="s", data={"tool_name": "t"})
        assert (
            out.blocked is False
        )  # explicit allow suppresses the exit-2 deny shorthand
        assert any(
            r.levelname == "WARNING"
            and "ignored" in r.getMessage()
            and "allow" in r.getMessage()
            for r in caplog.records
        ), (
            f"expected WARNING for ignored allow decision, got: "
            f"{[(r.levelname, r.getMessage()) for r in caplog.records]}"
        )

    def test_pretooluse_allow_with_crash_exit_fails_closed(self):
        # R12 refined: a parsed decision key suppresses only the exit-2
        # shorthand — a crash exit (non-zero, non-2) still fails closed.
        code = (
            "import sys; print(__import__('json').dumps({'decision': 'allow'})); "
            "sys.exit(1)"
        )
        eng = _engine(HookSpec("PreToolUse", (sys.executable, "-c", code)))
        out = eng.fire("PreToolUse", session_id="s", data={"tool_name": "t"})
        assert out.blocked is True and "failing closed" in out.reason

    def test_userpromptsubmit_deny_word_is_not_block(self):
        code = "print(__import__('json').dumps({'decision': 'deny'}))"
        eng = _engine(HookSpec("UserPromptSubmit", (sys.executable, "-c", code)))
        out = eng.fire("UserPromptSubmit", session_id="s", data={"prompt": "hi"})
        assert out.blocked is False  # "deny" is not a UserPromptSubmit decision word

    def test_userpromptsubmit_block_json_reason_wins_over_exit2(self):
        code = (
            "import sys; print(__import__('json').dumps("
            "{'decision': 'block', 'reason': 'json wins'})); sys.exit(2)"
        )
        eng = _engine(HookSpec("UserPromptSubmit", (sys.executable, "-c", code)))
        out = eng.fire("UserPromptSubmit", session_id="s", data={"prompt": "hi"})
        assert out.blocked is True and out.reason == "json wins"

    def test_pretooluse_junk_stdout_warns(self, caplog):
        eng = _engine(
            HookSpec("PreToolUse", (sys.executable, "-c", "print('not json')"))
        )
        out = eng.fire("PreToolUse", session_id="s", data={"tool_name": "t"})
        assert out.blocked is False  # exit 0 = pass
        assert any(
            r.levelname == "WARNING"
            and "unrecognized stdout" in r.getMessage()
            and "not json" not in r.getMessage()
            for r in caplog.records
        ), (
            f"expected WARNING for unrecognized stdout on clean pass, got: "
            f"{[(r.levelname, r.getMessage()) for r in caplog.records]}"
        )

    def test_payload_envelope_on_stdin(self, tmp_path):
        payload_file = tmp_path / "hook_payload.json"
        code = (
            "import json, sys; p = json.load(sys.stdin); "
            f"open({str(payload_file)!r}, 'w').write(json.dumps(p))"
        )
        eng = _engine(HookSpec("Stop", (sys.executable, "-c", code)))
        eng.fire("Stop", session_id="s1", run_id="r1", data={"status": "completed"})
        seen = json.loads(payload_file.read_text())
        assert seen["hook_event"] == "Stop" and seen["session_id"] == "s1"
        assert seen["run_id"] == "r1" and seen["data"] == {"status": "completed"}
        assert "timestamp" in seen and "cwd" in seen

    def test_truncation_to_budget(self):
        eng = _engine(
            HookSpec("UserPromptSubmit", (sys.executable, "-c", "print('x' * 999999)"))
        )
        out = eng.fire("UserPromptSubmit", session_id="s", data={"prompt": "hi"})
        assert len(out.context) <= HOOK_IO_BUDGET_CHARS

    def test_first_deny_wins_across_concurrent_hooks(self):
        eng = _engine(
            HookSpec(
                "PreToolUse",
                (sys.executable, "-c", "import time; time.sleep(0.5); print('late')"),
            ),
            HookSpec("PreToolUse", (sys.executable, "-c", "import sys; sys.exit(2)")),
        )
        start = time.monotonic()
        out = eng.fire("PreToolUse", session_id="s", data={"tool_name": "t"})
        assert out.blocked is True and time.monotonic() - start < 0.45

    def test_master_switch_off_fires_nothing(self):
        eng = _engine(
            HookSpec("PreToolUse", (sys.executable, "-c", "import sys; sys.exit(2)")),
            enabled=False,
        )
        assert eng.fire("PreToolUse", session_id="s", data={}).blocked is False

    def test_engine_never_raises(self):
        eng = _engine(HookSpec("Stop", ("/nonexistent/hook-binary", "--x")))
        assert eng.fire("Stop", session_id="s", data={}) == HookOutcome()


class TestTimeoutKill:
    def test_timeout_kills_process_group(self, tmp_path):
        if sys.platform == "win32":
            pytest.skip("POSIX process-group test")
        pid_file = tmp_path / "hook_child_pid.txt"
        # Grandchild writes its own pid, then sleeps with the hook parent.
        child_code = f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(30)"
        parent_code = (
            "import subprocess, sys, time; "
            f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
            "time.sleep(30)"
        )
        eng = _engine(
            HookSpec("PreToolUse", (sys.executable, "-c", parent_code), timeout_s=1.0)
        )
        out = eng.fire("PreToolUse", session_id="s", data={})
        assert out.blocked is True  # fail-closed on timeout
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not pid_file.exists():
            time.sleep(0.05)
        assert pid_file.exists(), "hook's grandchild never wrote its pid file"
        pid = int(pid_file.read_text().strip())
        # The group kill is asynchronous: poll up to ~2s for the child to die.
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break  # child is gone — the process group was killed
            time.sleep(0.05)
        else:
            pytest.fail(
                f"hook's grandchild survived the process-group kill (pid {pid} alive)"
            )

    def test_timeout_reap_abandoned_when_escaped_descendant_holds_pipe(
        self, tmp_path, monkeypatch, caplog
    ):
        # Ruling R32: a grandchild that setsid's out of the hook's process
        # group survives the group kill while holding the stdout pipe's
        # write-end, so EOF never comes. The post-kill reap must give up
        # after HOOK_POST_KILL_REAP_TIMEOUT_S (shrunk here to keep the test
        # fast) instead of wedging the pool worker until the escaper exits.
        if sys.platform == "win32":
            pytest.skip("POSIX process-group test")
        monkeypatch.setattr(run_hooks, "HOOK_POST_KILL_REAP_TIMEOUT_S", 0.5)
        pid_file = tmp_path / "escaper_pid.txt"
        ready_file = tmp_path / "escaper_ready.txt"
        grandchild_code = (
            "import os, time; "
            f"open({str(pid_file)!r}, 'w').write(str(os.getpid())); "
            "os.setsid(); "
            f"open({str(ready_file)!r}, 'w').write('ready'); "
            "time.sleep(20)"
        )
        parent_code = (
            "import os, subprocess, sys, time\n"
            f"subprocess.Popen([sys.executable, '-c', {grandchild_code!r}])\n"
            f"while not os.path.exists({str(ready_file)!r}):\n"
            "    time.sleep(0.01)\n"
            "time.sleep(20)"
        )
        eng = _engine(
            HookSpec("PreToolUse", (sys.executable, "-c", parent_code), timeout_s=1.0)
        )
        start = time.monotonic()
        out = eng.fire("PreToolUse", session_id="s", data={})
        elapsed = time.monotonic() - start
        assert out.blocked is True  # fail-closed verdict still delivered
        # 1s hook timeout + 0.5s reap ceiling + slack; without the bounded
        # reap this fire waits out the escaper's full 20s pipe hold.
        assert elapsed < 5.0, f"fire wedged {elapsed:.1f}s on the escaped pipe-holder"
        assert any(
            r.levelname == "WARNING" and "reap" in r.getMessage()
            for r in caplog.records
        ), (
            f"expected WARNING for the abandoned reap, got: "
            f"{[(r.levelname, r.getMessage()) for r in caplog.records]}"
        )
        # The abandoned escaper was deliberately left alive — clean it up.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not pid_file.exists():
            time.sleep(0.05)
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text().strip()), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass


class TestFireAsync:
    def test_loop_stays_responsive_while_hook_sleeps(self):
        eng = _engine(
            HookSpec(
                "UserPromptSubmit", (sys.executable, "-c", "import time; time.sleep(1)")
            )
        )

        async def main():
            ticks = 0

            async def ticker():
                nonlocal ticks
                while True:
                    await asyncio.sleep(0.05)
                    ticks += 1

            task = asyncio.create_task(ticker())
            await eng.fire_async(
                "UserPromptSubmit", session_id="s", data={"prompt": "x"}
            )
            task.cancel()
            return ticks

        ticks = asyncio.run(main())
        assert ticks >= 8  # loop kept ticking through the 1s hook


class TestNotify:
    def test_notify_returns_immediately_and_eventually_runs(self, tmp_path):
        marker = tmp_path / "hook_notify_marker.txt"
        eng = _engine(
            HookSpec(
                "PostToolUse",
                (sys.executable, "-c", f"open({str(marker)!r}, 'w').write('x')"),
            )
        )
        eng.notify("PostToolUse", session_id="s", data={"tool_name": "t"})
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(0.05)
        assert marker.exists()


class TestCwdOverride:
    """Ruling R18: a fire-site ``cwd`` beats the engine's provider fallback.

    Tasks 6-8 pass the session's bound workspace root here; the provider's
    app-level ``[console] workspace_root``-or-app-cwd answer is only the
    fallback for fires with no session root to name.
    """

    def _payload_dumping_engine(self, tmp_path, provider_cwd):
        # Both cwds must be REAL directories: _run_hook spawns the hook
        # process with cwd=payload["cwd"], and Popen ENOENTs on a fake path.
        payload_file = tmp_path / "hook_payload.json"
        code = (
            "import json, sys; p = json.load(sys.stdin); "
            f"open({str(payload_file)!r}, 'w').write(json.dumps(p))"
        )
        cfg = RunHooksConfig(
            enabled=True, hooks=(HookSpec("Stop", (sys.executable, "-c", code)),)
        )
        eng = RunHooksEngine(lambda: cfg, lambda: provider_cwd)
        return eng, payload_file

    def test_fire_with_cwd_override_carries_it_in_the_payload(self, tmp_path):
        session_root = tmp_path / "session-root"
        session_root.mkdir()
        eng, payload_file = self._payload_dumping_engine(
            tmp_path, str(tmp_path / "provider-root")
        )
        eng.fire("Stop", session_id="s", cwd=str(session_root))
        assert json.loads(payload_file.read_text())["cwd"] == str(session_root)

    def test_fire_without_override_falls_back_to_the_provider(self, tmp_path):
        provider_root = tmp_path / "provider-root"
        provider_root.mkdir()
        eng, payload_file = self._payload_dumping_engine(tmp_path, str(provider_root))
        eng.fire("Stop", session_id="s")
        assert json.loads(payload_file.read_text())["cwd"] == str(provider_root)

    def test_fire_async_cwd_override(self, tmp_path):
        session_root = tmp_path / "async-root"
        session_root.mkdir()
        eng, payload_file = self._payload_dumping_engine(
            tmp_path, str(tmp_path / "provider-root")
        )
        asyncio.run(eng.fire_async("Stop", session_id="s", cwd=str(session_root)))
        assert json.loads(payload_file.read_text())["cwd"] == str(session_root)

    def test_notify_cwd_override(self, tmp_path):
        session_root = tmp_path / "notify-root"
        session_root.mkdir()
        eng, payload_file = self._payload_dumping_engine(
            tmp_path, str(tmp_path / "provider-root")
        )
        eng.notify("Stop", session_id="s", cwd=str(session_root))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not payload_file.exists():
            time.sleep(0.05)
        assert json.loads(payload_file.read_text())["cwd"] == str(session_root)


class TestPoolIsolation:
    def test_blocking_fire_not_delayed_by_queued_notify_hooks(self):
        # Ruling R14: blocking-event hook executions must never queue behind
        # non-blocking (notify-driven) ones. Queue 4 slow notify hooks, then
        # check a PreToolUse fire completes within its normal time. If notify
        # hooks shared the blocking pool, all 4 workers would be busy ~1s and
        # the fire would wait behind them.
        eng = _engine(
            *[
                HookSpec(
                    "PostToolUse",
                    (sys.executable, "-c", "import time; time.sleep(1.0)"),
                )
                for _ in range(4)
            ],
            HookSpec("PreToolUse", (sys.executable, "-c", "pass")),
        )
        eng.notify("PostToolUse", session_id="s", data={"tool_name": "t"})
        time.sleep(0.05)  # let the notify work start consuming capacity
        start = time.monotonic()
        out = eng.fire("PreToolUse", session_id="s", data={"tool_name": "t"})
        elapsed = time.monotonic() - start
        assert out.blocked is False
        assert elapsed < 0.6, (
            f"blocking fire queued behind notify hooks: {elapsed:.2f}s"
        )


# ---------------------------------------------------------------------------
# Task 3: review-hook integration (wrap_review)
# ---------------------------------------------------------------------------


class TestWrapReview:
    def _engine_with(self, spec):
        cfg = RunHooksConfig(enabled=True, hooks=(spec,))
        return RunHooksEngine(lambda: cfg, lambda: os.getcwd())

    def test_deny_short_circuits_before_inner(self):
        eng = self._engine_with(
            HookSpec(
                "PreToolUse", (sys.executable, "-c", "sys.exit(2)"), matcher="fs_*"
            )
        )
        called = []

        def inner(calls, run_id):
            called.append([c.name for c in calls])
            return {c.name: "proceed" for c in calls}

        wrapped = eng.wrap_review(inner, session_id="s")
        verdicts = wrapped(
            [
                ToolCall(name="fs_write", args={"path": "x"}),
                ToolCall(name="calculator", args={}),
            ],
            "run-1",
        )
        assert verdicts["fs_write"] not in ("proceed",)  # refusal string per protocol
        assert called == [
            ["calculator"]
        ]  # denied call never reaches the permission store

    def test_hook_deny_reason_proceed_is_namespaced(self):
        # R16: a hook emitting reason "proceed" must not serialize the dispatch
        # sentinel into the verdict map — the refusal is namespaced with a
        # hook: prefix inside wrap_review (fire()/HookOutcome keep raw .reason).
        code = (
            "print(__import__('json').dumps({'decision': 'deny', 'reason': 'proceed'}))"
        )
        eng = self._engine_with(HookSpec("PreToolUse", (sys.executable, "-c", code)))
        called = []

        def inner(calls, run_id):
            called.append([c.name for c in calls])
            return {c.name: "proceed" for c in calls}

        wrapped = eng.wrap_review(inner, session_id="s")
        verdicts = wrapped([ToolCall(name="fs_write", args={})], "run-1")
        assert verdicts["fs_write"] != "proceed"
        assert verdicts["fs_write"].startswith("hook: ")
        assert called == []  # still a deny end-to-end: never reaches inner review

    def test_clean_pass_delegates_to_inner(self):
        eng = self._engine_with(HookSpec("PreToolUse", (sys.executable, "-c", "pass")))

        def inner(calls, run_id):
            return {c.name: "proceed" for c in calls}

        wrapped = eng.wrap_review(inner, session_id="s")
        assert wrapped([ToolCall(name="fs_write", args={})], "run-1") == {
            "fs_write": "proceed"
        }

    def test_no_hooks_configured_preserves_behavior(self):
        eng = _engine()

        def inner(calls, run_id):
            return {c.name: "proceed" for c in calls}

        calls = [ToolCall(name="fs_read", args={})]
        assert eng.wrap_review(inner, session_id="s")(calls, "r") == inner(calls, "r")


# ---------------------------------------------------------------------------
# Task 5: PostToolUse runtime dep (post_tool_dep)
# ---------------------------------------------------------------------------


class TestPostToolDep:
    """The runtime dep the dispatch loop fires at its capture point.

    The dep receives FULL (uncapped) content — the engine truncates here, to
    the payload budget — and speaks notify(), so a slow hook never stalls a
    dispatch. It is built per session (session_id closes over) and handed to
    ``AgentService(post_tool_call=...)`` only when an engine exists.
    """

    def test_dep_fires_notify_with_full_payload(self):
        fired = []
        eng = _engine(HookSpec("PostToolUse", (sys.executable, "-c", "pass")))
        dep = eng.post_tool_dep(session_id="s")
        # intercept notify to observe without racing the executor
        eng.notify = lambda event, **kw: fired.append((event, kw))
        dep(
            "fs_write",
            "call-1",
            {"path": "x"},
            "full result " * 1000,
            True,
            run_id="run-7",
        )
        event, kw = fired[0]
        assert event == "PostToolUse"
        assert kw["session_id"] == "s"
        # R20: the envelope's run_id must carry the FIRING run's id (bound
        # per-run by the service), not stay null like PreToolUse's counterpart
        # never would.
        assert kw["run_id"] == "run-7"
        assert kw["data"]["tool_name"] == "fs_write"
        assert kw["data"]["tool_args"] == {"path": "x"}
        assert len(kw["data"]["tool_result"]) <= HOOK_IO_BUDGET_CHARS
        assert kw["data"]["is_error"] is False

    def test_dep_inverts_ok_into_is_error(self):
        fired = []
        eng = _engine()
        dep = eng.post_tool_dep(session_id="s")
        eng.notify = lambda event, **kw: fired.append((event, kw))
        dep("fs_read", "call-2", {}, "ERROR: boom", False)
        _event, kw = fired[0]
        assert kw["data"]["is_error"] is True
        assert kw["data"]["tool_result"] == "ERROR: boom"


class TestEngineHardening:
    @pytest.mark.parametrize("value", ["false", 1, [], None])
    def test_master_switch_rejects_non_boolean_values(self, value):
        assert load_hooks_config(_cfg(enabled=value)).enabled is False

    @pytest.mark.parametrize("command", [[""], ["\x00bad"], ["/bin/true", "\x00"]])
    def test_invalid_argv_is_disabled_at_load(self, command):
        assert not load_hooks_config(_cfg(command=command)).hooks

    def test_refusals_are_per_call_and_survive_inner_failure(self):
        eng = _engine(
            HookSpec(
                "PreToolUse",
                (
                    sys.executable,
                    "-c",
                    "import json,sys;sys.exit(2 if json.load(sys.stdin)['data']['tool_args'].get('deny') else 0)",
                ),
            )
        )
        calls = [
            ToolCall(name="fs_read", args={"deny": True}, call_id="denied"),
            ToolCall(name="fs_read", args={}, call_id="allowed"),
        ]
        wrapped = eng.wrap_review(
            lambda cs, r: {c.call_id: "proceed" for c in cs}, session_id="s"
        )
        verdicts = wrapped(calls, "r")
        assert verdicts["denied"].startswith("hook: ")
        assert verdicts["allowed"] == "proceed"

        def broken(cs, run_id):
            raise RuntimeError("private-error-canary")

        verdicts = eng.wrap_review(broken, session_id="s")(calls, "r")
        assert verdicts["denied"].startswith("hook: ")
        assert verdicts["allowed"] != "proceed"

    def test_new_guard_is_seen_by_existing_wrapper(self):
        cfg = [RunHooksConfig()]
        eng = RunHooksEngine(lambda: cfg[0], lambda: os.getcwd())
        wrapped = eng.wrap_review(lambda cs, r: {}, session_id="s")
        cfg[0] = RunHooksConfig(
            hooks=(HookSpec("PreToolUse", (sys.executable, "-c", "exit(2)")),)
        )
        assert (
            wrapped([ToolCall(name="fs_write", args={}, call_id="c")], "r")["c"]
            != "proceed"
        )

    def test_oversized_decision_output_fails_closed(self):
        eng = _engine(
            HookSpec(
                "PreToolUse",
                (
                    sys.executable,
                    "-c",
                    "print(' ' * 2000000 + '{\"decision\":\"allow\"}')",
                ),
            )
        )
        out = eng.fire("PreToolUse", session_id="s")
        assert out.blocked

    def test_nonblocking_hooks_for_one_firing_run_concurrently(self, tmp_path):
        left, right = tmp_path / "left", tmp_path / "right"

        def script(own, other):
            return (
                f"import pathlib,time;pathlib.Path({str(own)!r}).touch()\n"
                f"while not pathlib.Path({str(other)!r}).exists(): time.sleep(.01)"
            )

        eng = _engine(
            *(
                HookSpec("Stop", (sys.executable, "-c", script(a, b)), timeout_s=0.7)
                for a, b in [(left, right), (right, left)]
            )
        )
        start = time.monotonic()
        eng.fire("Stop", session_id="s")
        assert time.monotonic() - start < 0.6

    def test_close_stops_live_process_and_refuses_new_guards(self, tmp_path):
        from concurrent.futures import ThreadPoolExecutor

        marker = tmp_path / "pid"
        code = f"import pathlib,os,time;pathlib.Path({str(marker)!r}).write_text(str(os.getpid()));time.sleep(30)"
        eng = _engine(HookSpec("PreToolUse", (sys.executable, "-c", code), timeout_s=2))
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(eng.fire, "PreToolUse", session_id="s")
            deadline = time.monotonic() + 2
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert marker.exists()
            close = getattr(eng, "close", None)
            assert callable(close), "engine must own its executor and process cleanup"
            start = time.monotonic()
            close()
            assert time.monotonic() - start < 0.2
            assert future.result(timeout=1).blocked
            assert eng.fire("PreToolUse", session_id="s").blocked
            close()
        with pytest.raises(ProcessLookupError):
            os.kill(int(marker.read_text()), 0)

    def test_notify_queue_admission_is_bounded(self, monkeypatch, caplog):
        import threading

        release, started = threading.Event(), threading.Event()
        eng = _engine()

        def blocked(*args, **kwargs):
            started.set()
            release.wait(2)

        monkeypatch.setattr(eng, "fire", blocked)
        eng.notify("Stop", session_id="s")
        assert started.wait(1)
        try:
            for _ in range(1000):
                eng.notify("Stop", session_id="s")
            assert eng._notify_worker._work_queue.qsize() < 100
            assert any("dropped" in r.getMessage() for r in caplog.records)
        finally:
            release.set()

    def test_notify_drops_oversize_whole_event(self, tmp_path, caplog):
        marker = tmp_path / "ran"
        eng = _engine(
            HookSpec(
                "PostToolUse",
                (sys.executable, "-c", f"open({str(marker)!r},'w').write('ran')"),
            )
        )
        eng.notify(
            "PostToolUse",
            session_id="s",
            data={"tool_args": {"content": "x" * 2000000}},
        )
        eng._notify_worker.shutdown(wait=True)
        assert not marker.exists()
        assert any("dropped" in r.getMessage() for r in caplog.records)

    def test_notify_rejects_large_scalar_before_allocating_encoded_chunk(self):
        import tracemalloc

        eng = _engine()
        # Allocate the source before measuring: only notify's additional
        # allocation counts, including the encoder's escaped scalar temporary.
        arguments = {"content": "\\" * (16 * 1024 * 1024)}
        tracemalloc.start()
        try:
            eng.notify("PostToolUse", session_id="s", data={"tool_args": arguments})
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 2 * run_hooks.HOOK_NOTIFY_PAYLOAD_BYTES

    def test_approval_summary_preserves_small_json_and_omits_unsupported_values(self):
        assert run_hooks.summarize_hook_arguments({"path": "notes.txt"}) == (
            '{"path": "notes.txt"}'
        )

        class Unsupported:
            def __str__(self):
                raise AssertionError(
                    "approval summary must never coerce arbitrary objects"
                )

        for arguments in ({"object": Unsupported()}, {"text": "😀" * 1000}):
            assert run_hooks.summarize_hook_arguments(arguments) == (
                "Arguments omitted (too large or unsupported)."
            )

    @pytest.mark.parametrize("shape", ["aggregate", "wide", "deep", "cycle"])
    def test_notify_bounds_structure_before_encoding(self, monkeypatch, shape):
        eng = _engine()
        if shape == "aggregate":
            payload = ["x" * 600000] * 2
        elif shape == "wide":
            payload = [None] * 20000
        else:
            payload = []
            if shape == "cycle":
                payload.append(payload)
            else:
                for _ in range(100):
                    payload = [payload]
        encoded = []
        original = json.JSONEncoder.iterencode

        def encode(self, value, *args, **kwargs):
            encoded.append(True)
            return original(self, value, *args, **kwargs)

        monkeypatch.setattr(json.JSONEncoder, "iterencode", encode)
        eng.notify("PostToolUse", session_id="s", data={"tool_args": payload})
        assert encoded == []

    def test_invalid_relative_cwd_never_launches(self, tmp_path):
        marker = tmp_path / "ran"
        eng = _engine(
            HookSpec(
                "PreToolUse",
                (sys.executable, "-c", f"open({str(marker)!r},'w').write('ran')"),
            )
        )
        assert eng.fire("PreToolUse", session_id="s", cwd=".").blocked
        assert not marker.exists()

    def test_cancelled_pending_hooks_finish_the_waiter(self):
        import threading
        from concurrent.futures import Future

        future = Future()
        future.cancel()
        spec = HookSpec("PreToolUse", (sys.executable, "-c", "pass"))
        outcomes = []
        waiter = threading.Thread(
            target=lambda: outcomes.extend(
                RunHooksEngine._results_from_futures({future: spec})
            ),
            daemon=True,
        )
        waiter.start()
        waiter.join(0.5)
        assert not waiter.is_alive(), "shutdown must release pending firing waiters"
        assert outcomes[0][1].denied

    def test_both_streams_are_drained_with_bounded_retention(self):
        spec = HookSpec(
            "Stop",
            (
                sys.executable,
                "-c",
                "import os\nfor _ in range(1000):\n os.write(1,b'x'*8192);os.write(2,b'y'*8192)",
            ),
        )
        code, captured, timed_out = asyncio.run(
            run_hooks._capture_hook(spec, {"cwd": os.getcwd()}, None)
        )
        assert code == 0 and not timed_out
        for stream in (1, 2):
            assert len(captured.output[stream]) == run_hooks.HOOK_IO_BUDGET_BYTES
            assert captured.overflow[stream]
            assert len(captured.text(stream)) <= HOOK_IO_BUDGET_CHARS
            assert captured.text(stream).endswith("…[truncated]")

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX process group contract")
    def test_timeout_kills_children_after_group_leader_exits(self, tmp_path):
        marker = tmp_path / "child-pid"
        child = (
            f"import pathlib,os,time;pathlib.Path({str(marker)!r}).write_text(str(os.getpid()));"
            "time.sleep(10)"
        )
        parent = (
            f"import subprocess,sys;subprocess.Popen([sys.executable,'-c',{child!r}])"
        )
        eng = _engine(
            HookSpec("PreToolUse", (sys.executable, "-c", parent), timeout_s=0.3)
        )
        try:
            assert eng.fire("PreToolUse", session_id="s").blocked
            assert marker.exists()
            pid = int(marker.read_text())
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.01)
            else:
                pytest.fail("descendant survived after its group leader exited")
        finally:
            if marker.exists():
                try:
                    os.kill(int(marker.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_config_and_failure_diagnostics_omit_private_values(self, caplog):
        canary = "private-content-zqmarker"
        load_hooks_config({"hooks": {"hook": [canary, {"event": canary}]}})
        eng = RunHooksEngine(
            lambda: (_ for _ in ()).throw(ValueError(canary)), lambda: os.getcwd()
        )
        assert eng.fire("PreToolUse", session_id="s").blocked
        assert caplog.records
        assert all(canary not in r.getMessage() for r in caplog.records)

    def test_execution_fingerprints_distinguish_hooks_without_private_argv(
        self, caplog
    ):
        import hashlib

        commands = (
            (sys.executable, "-c", "pass", "zqprivate-first-argument"),
            ("/nonexistent/zqprivate-executable", "zqprivate-second-argument"),
        )
        eng = _engine(*(HookSpec("Stop", command) for command in commands))
        eng.fire("Stop", session_id="fingerprint-session", run_id="fingerprint-run")
        messages = [record.getMessage() for record in caplog.records]
        for command in commands:
            fingerprint = hashlib.sha256(
                json.dumps(command).encode("utf-8")
            ).hexdigest()[:12]
            matches = [
                message for message in messages if f"hook={fingerprint}" in message
            ]
            assert matches, "both completed and failed-start hooks must be attributable"
            assert all(
                "session_id=fingerprint-session" in message
                and "run_id=fingerprint-run" in message
                for message in matches
            )
        assert all("zqprivate" not in message for message in messages)
        eng.close()
        eng.fire("PreToolUse", session_id="closed-session", run_id="closed-run")
        assert any(
            "session_id=closed-session" in r.getMessage()
            and "run_id=closed-run" in r.getMessage()
            for r in caplog.records
        )
