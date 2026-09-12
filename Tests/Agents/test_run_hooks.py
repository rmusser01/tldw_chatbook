"""Tests for the Console run-hooks engine (spec: 2026-09-11-console-run-hooks-design)."""

import asyncio
import json
import os
import sys
import time

import pytest

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
    def test_exit0_clean_pass_logs_stdout(self):
        eng = _engine(HookSpec("Stop", (sys.executable, "-c", "print('done')")))
        out = eng.fire("Stop", session_id="s", run_id="r", data={})
        assert out == HookOutcome(blocked=False, reason="", context="")

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

    def test_userpromptsubmit_crash_fails_open(self):
        eng = _engine(HookSpec("UserPromptSubmit", (sys.executable, "-c", "sys.exit(1)")))
        out = eng.fire("UserPromptSubmit", session_id="s", data={"prompt": "hi"})
        assert out.blocked is False

    def test_userpromptsubmit_stdout_is_context(self):
        eng = _engine(HookSpec("UserPromptSubmit", (sys.executable, "-c", "print('extra ctx')")))
        out = eng.fire("UserPromptSubmit", session_id="s", data={"prompt": "hi"})
        assert out.blocked is False and out.context == "extra ctx"

    def test_userpromptsubmit_block(self):
        eng = _engine(HookSpec("UserPromptSubmit", (sys.executable, "-c",
            "print(__import__('json').dumps({'decision': 'block', 'reason': 'no'}))")))
        out = eng.fire("UserPromptSubmit", session_id="s", data={"prompt": "hi"})
        assert out.blocked is True and out.reason == "no"

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
