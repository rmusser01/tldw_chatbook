"""Raw command outcomes distinguish empty searches and preserve diagnostics."""

import sys

import pytest

from Tests.Agents.test_agent_runtime import fence, make_deps
from Tests.Agents.test_raw_shell_integration import _provider
from tldw_chatbook.Agents.agent_models import AgentConfig, ModelTurn, RunBudget
from tldw_chatbook.Agents.agent_runtime import run_agent_loop

pytestmark = pytest.mark.bootstrap_profile


def test_truncated_stderr_cannot_turn_a_search_error_into_success(tmp_path):
    from io import BytesIO

    from tldw_chatbook.Agents.raw_shell_tool_provider import RawShellToolProvider
    from tldw_chatbook.Tools.raw_cli_executor import (
        MAX_RAW_PREVIEW_BYTES,
        RawCliResult,
        _OutputAccumulator,
    )

    output = _OutputAccumulator(BytesIO(), 65536)
    output.consume("stdout", b"x" * MAX_RAW_PREVIEW_BYTES, lambda _: None)
    output.consume("stderr", b"rg: unreadable file: Permission denied", lambda _: None)
    assert output.preview("stderr") == ""
    assert "Permission denied" in output.record_output()
    receipt = RawCliResult(
        invocation_id="truncated-search",
        caller="model",
        resolved_shell="powershell.exe",
        initial_directory=tmp_path,
        elapsed_seconds=0.1,
        stdout_preview=output.preview("stdout"),
        stderr_preview=output.preview("stderr"),
        record_output=output.record_output(),
        exit_code=1,
        terminal_state="exited",
        truncated=output.truncated,
        cleanup_proven=True,
    )
    result = RawShellToolProvider._tool_result(receipt, command="rg pattern .")
    assert not result.ok
    assert result.outcome == "failed"


@pytest.mark.parametrize(
    "command,expected",
    [
        ('findstr /C:"a|b" sample.txt', True),
        ("findstr 'a|b' sample.txt", False),
        ("findstr %PATTERN% sample.txt", False),
        ('findstr ^"absent& echo COMPOUND_RAN & exit /b 1 & rem ^"', False),
        ("findstr /C:a^b sample.txt", False),
        ('"C:\\Program Files\\Git\\usr\\bin\\grep.exe" missing sample.txt', True),
    ],
)
def test_empty_search_respects_cmd_quote_and_expansion_rules(command, expected):
    from tldw_chatbook.Tools.shell_search_outcomes import is_empty_search

    assert is_empty_search(command, 1, "", shell="cmd.exe") is expected


@pytest.mark.parametrize(
    "shell",
    [
        "powershell.exe",
        "pwsh.exe",
        r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
    ],
)
@pytest.mark.parametrize(
    "command,expected",
    [
        (r"C:\Windows\System32\findstr.exe /C:absent NUL", True),
        (
            r'findstr "absent\"; Write-Output COMPOUND_RAN; cmd /c exit 1; # \""',
            False,
        ),
    ],
)
def test_empty_search_preserves_powershell_backslashes(command, expected, shell):
    from tldw_chatbook.Tools.shell_search_outcomes import is_empty_search

    assert is_empty_search(command, 1, "", shell=shell) is expected


@pytest.mark.skipif(sys.platform != "win32", reason="native PowerShell quoting")
@pytest.mark.parametrize("compound", [False, True])
def test_native_powershell_search_distinguishes_compound_failure(tmp_path, compound):
    import shutil
    import subprocess

    from tldw_chatbook.Tools.shell_search_outcomes import is_empty_search

    shell = shutil.which("powershell.exe")
    search = shutil.which("findstr.exe")
    if shell is None or search is None:
        pytest.skip("PowerShell and findstr are required")
    command = (
        r'findstr "absent\"; Write-Output COMPOUND_RAN; cmd /c exit 1; # \""'
        if compound
        else f"{search} /C:cli_recovery_absent NUL"
    )
    result = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", command],
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 1
    assert result.stderr == ""
    assert result.stdout.strip() == ("COMPOUND_RAN" if compound else "")
    assert is_empty_search(command, result.returncode, result.stderr, shell=shell) is (
        not compound
    )


def test_shell_failure_keeps_stdout_and_stderr_beyond_old_model_cap(tmp_path):
    provider, runtime, _ = _provider(tmp_path)
    runtime.grants.add("console-session")
    runtime.exit_code = 2
    runtime.stdout = "stdout marker\n" + "x" * 10_000
    runtime.stderr = "specific diagnostic after large stdout"
    result = provider.invoke("shell_exec", {"command": "python failing.py"})
    assert not result.ok
    assert runtime.stdout in result.error
    assert runtime.stderr in result.error
    assert "exit_code: 2" in result.error


@pytest.mark.parametrize(
    "command,stderr,expected",
    [
        ("grep missing sample.txt", "", True),
        ("rg --glob '*.py' 'x|y' .", "", True),
        ("findstr /C:missing sample.txt", "", True),
        ("/usr/bin/grep missing sample.txt", "", True),
        ('"C:\\Program Files\\Git\\usr\\bin\\grep.exe" missing sample.txt', "", True),
        ("grep missing absent.txt", "grep: absent.txt: No such file", False),
        ("python failure.py", "", False),
        ("grep missing sample.txt; false", "", False),
        ("grep missing sample.txt && false", "", False),
        ("grep missing sample.txt | false", "", False),
        ("grep $(false) sample.txt", "", False),
        ("grep `false` sample.txt", "", False),
        ("grep missing sample.txt\nfalse", "", False),
        ("echo grep missing sample.txt", "", False),
        ("grep missing sample.txt > output.txt", "", False),
        ("grep 'unterminated sample.txt", "", False),
    ],
)
def test_only_simple_empty_searches_are_successful(tmp_path, command, stderr, expected):
    provider, runtime, _ = _provider(tmp_path)
    runtime.grants.add("console-session")
    runtime.exit_code = 1
    runtime.stdout = ""
    runtime.stderr = stderr
    result = provider.invoke("shell_exec", {"command": command})
    assert result.ok is expected
    assert "exit_code: 1" in (result.content if result.ok else result.error)


@pytest.mark.parametrize("command", ["grep missing", "rg missing", "findstr missing"])
def test_three_empty_searches_continue_to_next_model_turn(tmp_path, command):
    provider, runtime, _ = _provider(tmp_path)
    runtime.grants.add("console-session")
    runtime.exit_code = 1
    runtime.stdout = ""
    turns = [
        ModelTurn(text=fence("shell_exec", {"command": f"{command}_{i} sample.txt"}))
        for i in range(3)
    ] + [ModelTurn(text="No matches; continue with the task.")]
    outcome = run_agent_loop(
        AgentConfig(
            model="fixture",
            system_prompt="",
            allowed_tools=("shell_exec",),
            budget=RunBudget(max_steps=50),
        ),
        [],
        [provider.load_schema("shell_exec")],
        make_deps(turns, invoke=lambda call: provider.invoke(call.name, call.args)),
    )
    assert outcome.status == "done"
    assert len(runtime.execute_calls) == 3
    assert all(
        step.tool_outcome == "success"
        for step in outcome.steps
        if step.kind == "tool_result"
    )


def test_empty_search_keeps_posix_caret_regex():
    from tldw_chatbook.Tools.shell_search_outcomes import is_empty_search

    assert is_empty_search("grep '^absent' sample.txt", 1, "", shell="/bin/bash")


@pytest.mark.skipif(sys.platform != "win32", reason="native cmd caret quoting")
@pytest.mark.parametrize("compound", [False, True])
def test_native_cmd_search_distinguishes_caret_compound_failure(tmp_path, compound):
    import shutil

    from tldw_chatbook.Agents.raw_shell_tool_provider import RawShellToolProvider
    from tldw_chatbook.Tools.raw_cli_executor import (
        RawCliRequest,
        RawCliResult,
        _launch_shell,
    )

    shell = shutil.which("cmd.exe")
    if shell is None or shutil.which("findstr.exe") is None:
        pytest.skip("cmd and findstr are required")
    command = (
        'findstr ^"absent& echo COMPOUND_RAN & exit /b 1 & rem ^"'
        if compound
        else "findstr /C:cli_recovery_absent NUL"
    )
    request = RawCliRequest(
        invocation_id="caret-search",
        caller="model",
        command=command,
        shell="cmd",
        initial_directory=tmp_path,
        timeout_seconds=15,
        console_session_id="caret-session",
    )
    process = _launch_shell((shell, "/D", "/S", "/C", command), request)
    try:
        stdout, stderr = process.communicate(timeout=15)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    assert process.returncode == 1
    assert stderr == b""
    assert stdout.strip() == (b"COMPOUND_RAN" if compound else b"")
    result = RawShellToolProvider._tool_result(
        RawCliResult(
            invocation_id=request.invocation_id,
            caller="model",
            resolved_shell=shell,
            initial_directory=tmp_path,
            elapsed_seconds=0.1,
            stdout_preview=stdout.decode("utf-8"),
            stderr_preview="",
            record_output=stdout.decode("utf-8"),
            exit_code=process.returncode,
            terminal_state="exited",
            truncated=False,
            cleanup_proven=True,
        ),
        command=command,
    )
    assert result.ok is (not compound)
    if compound:
        assert result.outcome == "failed"
        assert "COMPOUND_RAN" in result.error
