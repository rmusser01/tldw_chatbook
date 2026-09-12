"""Tests for the Console run-hooks engine (spec: 2026-09-11-console-run-hooks-design)."""

import asyncio
import json
import os
import sys
import time

import pytest
from loguru import logger as _loguru_logger

from tldw_chatbook.Agents.agent_models import ToolCall
from tldw_chatbook.Agents.run_hooks import (
    HOOK_DEFAULT_TIMEOUT_S,
    HOOK_EVENTS,
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
            {"hooks": {"hook": [{"event": "PreToolUse", "matcher": "fs_*",
                                  "command": ["/bin/guard", "--strict"], "timeout_s": 5}]}}
        )
        assert result.hooks == (HookSpec("PreToolUse", ("/bin/guard", "--strict"), "fs_*", 5.0),)

    def test_unknown_event_disables_that_hook(self):
        result = load_hooks_config({"hooks": {"hook": [{"event": "Nope", "command": ["/bin/true"]}]}})
        assert result.hooks == ()

    def test_matcher_rejected_on_non_tool_event(self):
        result = load_hooks_config({"hooks": {"hook": [{"event": "Stop", "matcher": "fs_*",
                                                         "command": ["/bin/true"]}]}})
        assert result.hooks == ()

    def test_empty_command_rejected(self):
        result = load_hooks_config({"hooks": {"hook": [{"event": "Stop", "command": []}]}})
        assert result.hooks == ()

    def test_non_positive_timeout_rejected(self):
        result = load_hooks_config({"hooks": {"hook": [{"event": "Stop", "command": ["/bin/true"],
                                                         "timeout_s": 0}]}})
        assert result.hooks == ()

    def test_string_command_rejected(self):
        result = load_hooks_config({"hooks": {"hook": [{"event": "Stop", "command": "/bin/true"}]}})
        assert result.hooks == ()

    def test_default_timeout_applied(self):
        result = load_hooks_config({"hooks": {"hook": [{"event": "Stop", "command": ["/bin/true"]}]}})
        assert result.hooks[0].timeout_s == HOOK_DEFAULT_TIMEOUT_S

    def test_hook_key_not_a_list_disables_hooks(self):
        result = load_hooks_config({"hooks": {"enabled": True, "hook": None}})
        assert result == RunHooksConfig(enabled=True, hooks=())

    def test_unhashable_event_disables_that_hook(self):
        result = load_hooks_config({"hooks": {"hook": [{"event": ["Stop"], "command": ["/bin/true"]}]}})
        assert result.hooks == ()

    def test_nan_timeout_rejected(self):
        result = load_hooks_config({"hooks": {"hook": [{"event": "Stop", "command": ["/bin/true"],
                                                         "timeout_s": float("nan")}]}})
        assert result.hooks == ()

    def test_inf_timeout_rejected(self):
        result = load_hooks_config({"hooks": {"hook": [{"event": "Stop", "command": ["/bin/true"],
                                                         "timeout_s": float("inf")}]}})
        assert result.hooks == ()

    def test_bool_timeout_rejected(self):
        result = load_hooks_config({"hooks": {"hook": [{"event": "Stop", "command": ["/bin/true"],
                                                         "timeout_s": True}]}})
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
        info_records = [r for r in caplog.records
                        if r.levelname == "INFO" and "event=Stop" in r.getMessage()]
        assert info_records, f"no INFO record for the Stop hook: {[r.getMessage() for r in caplog.records]}"
        msg = info_records[0].getMessage()
        assert "done" in msg and "session_id=s" in msg and "run_id=r" in msg

    def test_json_decision_beats_exit_code(self):
        code = "import sys; print(__import__('json').dumps({'decision': 'deny', 'reason': 'nope'}))"
        eng = _engine(HookSpec("PreToolUse", (sys.executable, "-c", code)))
        out = eng.fire("PreToolUse", session_id="s", data={"tool_name": "fs_write"})
        assert out.blocked is True and out.reason == "nope"

    def test_exit2_is_deny_shorthand(self):
        eng = _engine(HookSpec("PreToolUse", (sys.executable, "-c",
                                              "import sys; sys.stderr.write('blocked'); sys.exit(2)")))
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
        eng = _engine(HookSpec("UserPromptSubmit", (sys.executable, "-c", "sys.exit(1)")))
        out = eng.fire("UserPromptSubmit", session_id="s", data={"prompt": "hi"})
        assert out.blocked is False
        # Fail-open paths log at WARNING (spec §5), not silently at INFO.
        assert any(r.levelname == "WARNING" and "UserPromptSubmit" in r.getMessage()
                   and "failing open" in r.getMessage() for r in caplog.records), (
            f"expected WARNING for UPS crash fail-open, got: "
            f"{[(r.levelname, r.getMessage()) for r in caplog.records]}")

    def test_userpromptsubmit_stdout_is_context(self):
        eng = _engine(HookSpec("UserPromptSubmit", (sys.executable, "-c", "print('extra ctx')")))
        out = eng.fire("UserPromptSubmit", session_id="s", data={"prompt": "hi"})
        assert out.blocked is False and out.context == "extra ctx"

    def test_userpromptsubmit_block(self):
        eng = _engine(HookSpec("UserPromptSubmit", (sys.executable, "-c",
            "print(__import__('json').dumps({'decision': 'block', 'reason': 'no'}))")))
        out = eng.fire("UserPromptSubmit", session_id="s", data={"prompt": "hi"})
        assert out.blocked is True and out.reason == "no"

    def test_userpromptsubmit_timeout_fails_open_warns(self, caplog):
        eng = _engine(HookSpec("UserPromptSubmit", (sys.executable, "-c",
                                                   "import time; time.sleep(2)"),
                               timeout_s=0.5))
        out = eng.fire("UserPromptSubmit", session_id="s", data={"prompt": "hi"})
        assert out.blocked is False  # timeout fails open
        assert any(r.levelname == "WARNING" and "UserPromptSubmit" in r.getMessage()
                   and "timed out" in r.getMessage() and "failing open" in r.getMessage()
                   for r in caplog.records), (
            f"expected WARNING for UPS timeout fail-open, got: "
            f"{[(r.levelname, r.getMessage()) for r in caplog.records]}")

    def test_nonblocking_logged_output_truncated(self, caplog):
        eng = _engine(HookSpec("Stop", (sys.executable, "-c", "print('z' * 999999)")))
        eng.fire("Stop", session_id="s", run_id="r", data={})
        stop_records = [r for r in caplog.records
                        if r.levelname == "INFO" and "event=Stop" in r.getMessage()]
        assert stop_records, f"no INFO record for the Stop hook: {[r.getMessage() for r in caplog.records]}"
        assert "…[truncated]" in stop_records[0].getMessage()

    # --- Ruling R12: a JSON object with a "decision" key suppresses exit codes ---

    def test_pretooluse_allow_with_exit2_not_blocked(self, caplog):
        code = ("import sys; print(__import__('json').dumps({'decision': 'allow'})); "
                "sys.exit(2)")
        eng = _engine(HookSpec("PreToolUse", (sys.executable, "-c", code)))
        out = eng.fire("PreToolUse", session_id="s", data={"tool_name": "t"})
        assert out.blocked is False  # explicit allow suppresses the exit-2 deny shorthand
        assert any(r.levelname == "WARNING" and "ignored" in r.getMessage()
                   and "allow" in r.getMessage() for r in caplog.records), (
            f"expected WARNING for ignored allow decision, got: "
            f"{[(r.levelname, r.getMessage()) for r in caplog.records]}")

    def test_pretooluse_allow_with_crash_exit_fails_closed(self):
        # R12 refined: a parsed decision key suppresses only the exit-2
        # shorthand — a crash exit (non-zero, non-2) still fails closed.
        code = ("import sys; print(__import__('json').dumps({'decision': 'allow'})); "
                "sys.exit(1)")
        eng = _engine(HookSpec("PreToolUse", (sys.executable, "-c", code)))
        out = eng.fire("PreToolUse", session_id="s", data={"tool_name": "t"})
        assert out.blocked is True and "failing closed" in out.reason

    def test_userpromptsubmit_deny_word_is_not_block(self):
        code = "print(__import__('json').dumps({'decision': 'deny'}))"
        eng = _engine(HookSpec("UserPromptSubmit", (sys.executable, "-c", code)))
        out = eng.fire("UserPromptSubmit", session_id="s", data={"prompt": "hi"})
        assert out.blocked is False  # "deny" is not a UserPromptSubmit decision word

    def test_userpromptsubmit_block_json_reason_wins_over_exit2(self):
        code = ("import sys; print(__import__('json').dumps("
                "{'decision': 'block', 'reason': 'json wins'})); sys.exit(2)")
        eng = _engine(HookSpec("UserPromptSubmit", (sys.executable, "-c", code)))
        out = eng.fire("UserPromptSubmit", session_id="s", data={"prompt": "hi"})
        assert out.blocked is True and out.reason == "json wins"

    def test_pretooluse_junk_stdout_warns(self, caplog):
        eng = _engine(HookSpec("PreToolUse", (sys.executable, "-c", "print('not json')")))
        out = eng.fire("PreToolUse", session_id="s", data={"tool_name": "t"})
        assert out.blocked is False  # exit 0 = pass
        assert any(r.levelname == "WARNING" and "unrecognized stdout" in r.getMessage()
                   and "not json" in r.getMessage() for r in caplog.records), (
            f"expected WARNING for unrecognized stdout on clean pass, got: "
            f"{[(r.levelname, r.getMessage()) for r in caplog.records]}")

    def test_payload_envelope_on_stdin(self, tmp_path):
        payload_file = tmp_path / "hook_payload.json"
        code = ("import json, sys; p = json.load(sys.stdin); "
                "open(%r, 'w').write(json.dumps(p))" % str(payload_file))
        eng = _engine(HookSpec("Stop", (sys.executable, "-c", code)))
        eng.fire("Stop", session_id="s1", run_id="r1", data={"status": "completed"})
        seen = json.loads(payload_file.read_text())
        assert seen["hook_event"] == "Stop" and seen["session_id"] == "s1"
        assert seen["run_id"] == "r1" and seen["data"] == {"status": "completed"}
        assert "timestamp" in seen and "cwd" in seen

    def test_truncation_to_budget(self):
        eng = _engine(HookSpec("UserPromptSubmit", (sys.executable, "-c", "print('x' * 999999)")))
        out = eng.fire("UserPromptSubmit", session_id="s", data={"prompt": "hi"})
        assert len(out.context) <= HOOK_IO_BUDGET_CHARS

    def test_first_deny_wins_across_concurrent_hooks(self):
        eng = _engine(
            HookSpec("PreToolUse", (sys.executable, "-c",
                                    "import time; time.sleep(0.5); print('late')")),
            HookSpec("PreToolUse", (sys.executable, "-c", "import sys; sys.exit(2)")),
        )
        start = time.monotonic()
        out = eng.fire("PreToolUse", session_id="s", data={"tool_name": "t"})
        assert out.blocked is True and time.monotonic() - start < 0.45

    def test_master_switch_off_fires_nothing(self):
        eng = _engine(HookSpec("PreToolUse", (sys.executable, "-c", "import sys; sys.exit(2)")),
                      enabled=False)
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
        child_code = ("import os, time; open(%r, 'w').write(str(os.getpid())); time.sleep(30)"
                      % str(pid_file))
        parent_code = ("import subprocess, sys, time; "
                       "subprocess.Popen([sys.executable, '-c', %r]); "
                       "time.sleep(30)" % child_code)
        eng = _engine(HookSpec("PreToolUse", (sys.executable, "-c", parent_code), timeout_s=1.0))
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
            pytest.fail("hook's grandchild survived the process-group kill (pid %d alive)" % pid)


class TestFireAsync:
    def test_loop_stays_responsive_while_hook_sleeps(self):
        eng = _engine(HookSpec("UserPromptSubmit", (sys.executable, "-c",
                                                    "import time; time.sleep(1)")))

        async def main():
            ticks = 0

            async def ticker():
                nonlocal ticks
                while True:
                    await asyncio.sleep(0.05)
                    ticks += 1

            task = asyncio.create_task(ticker())
            await eng.fire_async("UserPromptSubmit", session_id="s", data={"prompt": "x"})
            task.cancel()
            return ticks

        ticks = asyncio.run(main())
        assert ticks >= 8  # loop kept ticking through the 1s hook


class TestNotify:
    def test_notify_returns_immediately_and_eventually_runs(self, tmp_path):
        marker = tmp_path / "hook_notify_marker.txt"
        eng = _engine(HookSpec("PostToolUse", (sys.executable, "-c",
                                               "open(%r, 'w').write('x')" % str(marker))))
        eng.notify("PostToolUse", session_id="s", data={"tool_name": "t"})
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(0.05)
        assert marker.exists()


class TestPoolIsolation:
    def test_blocking_fire_not_delayed_by_queued_notify_hooks(self):
        # Ruling R14: blocking-event hook executions must never queue behind
        # non-blocking (notify-driven) ones. Queue 4 slow notify hooks, then
        # check a PreToolUse fire completes within its normal time. If notify
        # hooks shared the blocking pool, all 4 workers would be busy ~1s and
        # the fire would wait behind them.
        eng = _engine(
            *[HookSpec("PostToolUse", (sys.executable, "-c",
                                       "import time; time.sleep(1.0)")) for _ in range(4)],
            HookSpec("PreToolUse", (sys.executable, "-c", "pass")),
        )
        eng.notify("PostToolUse", session_id="s", data={"tool_name": "t"})
        time.sleep(0.05)  # let the notify work start consuming capacity
        start = time.monotonic()
        out = eng.fire("PreToolUse", session_id="s", data={"tool_name": "t"})
        elapsed = time.monotonic() - start
        assert out.blocked is False
        assert elapsed < 0.6, f"blocking fire queued behind notify hooks: {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Task 3: review-hook integration (wrap_review)
# ---------------------------------------------------------------------------


class TestWrapReview:
    def _engine_with(self, spec):
        cfg = RunHooksConfig(enabled=True, hooks=(spec,))
        return RunHooksEngine(lambda: cfg, lambda: os.getcwd())

    def test_deny_short_circuits_before_inner(self):
        eng = self._engine_with(HookSpec("PreToolUse", (sys.executable, "-c", "sys.exit(2)"),
                                         matcher="fs_*"))
        called = []

        def inner(calls, run_id):
            called.append([c.name for c in calls])
            return {c.name: "proceed" for c in calls}

        wrapped = eng.wrap_review(inner, session_id="s")
        verdicts = wrapped([ToolCall(name="fs_write", args={"path": "x"}),
                            ToolCall(name="calculator", args={})], "run-1")
        assert verdicts["fs_write"] not in ("proceed",)  # refusal string per protocol
        assert called == [["calculator"]]  # denied call never reaches the permission store

    def test_hook_deny_reason_proceed_is_namespaced(self):
        # R16: a hook emitting reason "proceed" must not serialize the dispatch
        # sentinel into the verdict map — the refusal is namespaced with a
        # hook: prefix inside wrap_review (fire()/HookOutcome keep raw .reason).
        code = ("print(__import__('json').dumps("
                "{'decision': 'deny', 'reason': 'proceed'}))")
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
        assert wrapped([ToolCall(name="fs_write", args={})], "run-1") == {"fs_write": "proceed"}

    def test_no_hooks_configured_is_identity(self):
        eng = _engine()

        def inner(calls, run_id):
            return {c.name: "proceed" for c in calls}

        assert eng.wrap_review(inner, session_id="s") is inner
