"""Two-tier worker watchdog (Phase 2c, Task 12).

The worker gets ONE budget per exchange (the request's
``timeout_seconds`` — over ssh the transport writes the remaining budget
into that field; the worker just consumes it) and enforces it itself with
two tiers:

* **Tier 1 (graceful)**: a ``threading.Timer`` that sweeps registered
  temp files, writes the fixed ``tldw-worker-watchdog`` stderr line, and
  ``os._exit(75)``.
* **Tier 2 (OS backstop)**: ``signal.alarm(budget + 2)`` with the
  DEFAULT action. The catastrophic case is ``fs_grep`` with a pattern
  like ``(a+)+$`` over a near-miss line: the C regex engine spins
  without releasing the GIL, so the Timer thread starves — but the
  kernel still delivers SIGALRM and the default action kills the
  process. No Python handler is installed on purpose: a handler would
  need the GIL and starve exactly like the Timer.

Everything that arms or fires the watchdog runs in a SPAWNED process —
arming inside pytest would schedule a real ``signal.alarm`` in the test
runner. In-process tests only touch the temp-registry data plane.
"""

from __future__ import annotations

import importlib
import json
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from tldw_chatbook.Tools import remote_workspace_executor as executor_module
from tldw_chatbook.Tools.build_remote_worker_bundle import BUNDLE_MODULES
from tldw_chatbook.Tools.worker_watchdog import (
    WATCHDOG_EXIT_CODE,
    WATCHDOG_STDERR_MARKER,
)
from tldw_chatbook.Tools.workspace_tool_protocol import MAX_RESPONSE_BYTES

pytestmark = pytest.mark.skipif(
    not hasattr(signal, "SIGALRM"),
    reason="the alarm backstop is POSIX-only; remotes are POSIX hosts",
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_BUNDLE_PATH = (
    _REPOSITORY_ROOT / "tldw_chatbook" / "Tools" / "remote_worker_bundle.py"
)

#: Outer spawn budget for exchanges whose INNER (request) budget is the
#: thing under test: the harness must outlive the worker's own death
#: (alarm at budget+2) so the returncode we assert is the WORKER's.
_OUTER_SPAWN_BUDGET = 10.0

#: The worker must be dead within this window however it dies: budget
#: (1s, tier 1) .. budget+2 (3s, tier 2) plus interpreter startup.
_DEATH_BOUND_SECONDS = 4.5

_PING_PROBE_IDENTITY = {"device": 0, "inode": 0, "mode": 0, "reparse": False}


def _workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "alpha.txt").write_text("alpha body\n", encoding="utf-8")
    return root


def _request(
    root: Path,
    operation: str,
    arguments: dict[str, Any],
    *,
    chain: dict[str, Any] | None = None,
    timeout_seconds: int,
) -> dict[str, Any]:
    """One wire-legal request dict with an EXPLICIT inner budget."""
    if chain is None:
        identities = [_PING_PROBE_IDENTITY]
        locator = str(root)
    else:
        identities = [
            {
                "device": entry[1],
                "inode": entry[2],
                "mode": entry[3],
                "reparse": False,
            }
            for entry in chain["identity_chain"]
        ]
        locator = chain["canonical_path"]
    return {
        "version": 1,
        "operation_id": uuid.uuid4().hex,
        "operation": operation,
        "intent": "read",
        "root_locator": locator,
        "root_identity": identities[0],
        "ancestor_identities": identities,
        "arguments": arguments,
        "timeout_seconds": timeout_seconds,
        "output_max_bytes": MAX_RESPONSE_BYTES,
    }


def _ping_chain(root: Path) -> dict[str, Any]:
    result = executor_module.run_bundle_loopback(
        root, _request(root, "ping", {}, timeout_seconds=30)
    )
    assert result["outcome"] == "success", result
    return json.loads(result["result"])


def _spawn(request: dict[str, Any]) -> subprocess.CompletedProcess[bytes]:
    """Run one loopback exchange with the outer budget the test controls."""
    return executor_module._spawn_loopback_worker(
        executor_module._encode_request(request),
        budget_seconds=_OUTER_SPAWN_BUDGET,
    )


# ---------------------------------------------------------------------------
# Tier 2 backstop: the catastrophic fs_grep (GIL starvation)
# ---------------------------------------------------------------------------


def test_catastrophic_grep_dies_within_bound_via_either_tier(
    tmp_path: Path,
) -> None:
    """budget 1s + ``(a+)+$`` over a 60KB near-miss line of ``a`` chars.

    The C regex engine holds the GIL through the backtracking, so tier 1
    (the Timer) may starve; the kernel alarm at budget+2 must then kill
    the process regardless. Either death is acceptable; hanging is not.
    The subprocess is reaped by ``subprocess.run`` — an int returncode
    IS the reaped proof (a zombie would leave ``returncode`` unset).
    """
    root = _workspace(tmp_path)
    (root / "poison.txt").write_text("a" * 60000 + "b\n", encoding="utf-8")
    chain = _ping_chain(root)

    started = time.monotonic()
    completed = _spawn(
        _request(
            root,
            "fs_grep",
            {
                "pattern": "(a+)+$",
                "sensitive_exclusions": [],
                "content_exclusions": [],
            },
            chain=chain,
            timeout_seconds=1,
        )
    )
    elapsed = time.monotonic() - started

    assert isinstance(completed.returncode, int), "worker was not reaped"
    assert elapsed <= _DEATH_BOUND_SECONDS, (
        f"worker outran both watchdog tiers ({elapsed:.2f}s)"
    )
    # Three attributable deaths, all watchdog-caused: tier 1 (75 with
    # the stderr marker), tier 2's default-action alarm (-SIGALRM at
    # budget+2 wall), or the optional RLIMIT_CPU backstop (-SIGXCPU at
    # ceil(budget*2) CPU — for a pure-CPU regex spin this fires FIRST:
    # 2s CPU < 3s alarm). A stall or a healthy exit is the failure.
    accepted = {
        WATCHDOG_EXIT_CODE,
        -int(signal.SIGALRM),
        -int(getattr(signal, "SIGXCPU", signal.SIGALRM)),
    }
    assert completed.returncode in accepted, (
        f"returncode {completed.returncode}; "
        f"stderr: {completed.stderr.decode(errors='replace')}"
    )
    if completed.returncode == WATCHDOG_EXIT_CODE:
        assert WATCHDOG_STDERR_MARKER in completed.stderr
    # The root was pinned and the admitted marker emitted before the op
    # hung — the taxonomy the transport buckets as OP_TIMEOUT.
    assert executor_module.RESPONSE_MAGIC in completed.stdout
    assert b'"outcome":"admitted"' in completed.stdout


def test_healthy_grep_over_the_same_input_is_instant(tmp_path: Path) -> None:
    """The poison file is tiny; a sane pattern proves the death above is
    the regex's fault, not the input size."""
    root = _workspace(tmp_path)
    (root / "poison.txt").write_text("a" * 60000 + "b\n", encoding="utf-8")
    chain = _ping_chain(root)

    started = time.monotonic()
    completed = _spawn(
        _request(
            root,
            "fs_grep",
            {
                "pattern": "definitely-not-present",
                "sensitive_exclusions": [],
                "content_exclusions": [],
            },
            chain=chain,
            timeout_seconds=1,
        )
    )
    elapsed = time.monotonic() - started

    assert completed.returncode == 0
    assert elapsed < 4.0
    frames = executor_module._magic_frame_segments(completed.stdout)
    terminal = json.loads(frames[-1])
    assert terminal["outcome"] == "success"
    assert "no matches" in (terminal["result"] or "")


# ---------------------------------------------------------------------------
# Tier 1: the graceful Timer (temp sweep + marker + exit 75)
# ---------------------------------------------------------------------------


def _timer_harness(temp_path: Path, loader: str) -> str:
    """Spawn a process that registers a temp, arms the watchdog, sleeps.

    ``loader`` selects what runs: the real source module, or the
    COMMITTED bundle artifact (proving the generated artifact carries
    the implementation, not the Task-8 stub).
    """
    if loader == "source":
        loader_code = (
            "from tldw_chatbook.Tools.worker_watchdog import ("
            "TEMP_REGISTRY, arm_watchdog, register_temp)"
        )
    elif loader == "bundle":
        loader_code = (
            "import types\n"
            "module = types.ModuleType('remote_worker_bundle')\n"
            "sys.modules['remote_worker_bundle'] = module\n"
            "exec(compile(open(sys.argv[1]).read(), "
            "'remote_worker_bundle.py', 'exec'), module.__dict__)\n"
            "TEMP_REGISTRY = module.TEMP_REGISTRY\n"
            "arm_watchdog = module.arm_watchdog\n"
            "register_temp = module.register_temp"
        )
    else:  # pragma: no cover - test plumbing
        raise AssertionError(loader)
    source = f"""
import sys, time
{loader_code}
temp_path = sys.argv[-1]
with open(temp_path, 'w') as handle:
    handle.write('partial')
register_temp(temp_path)
arm_watchdog(1, TEMP_REGISTRY)
time.sleep(3)
print('SURVIVED', flush=True)
"""
    return source


@pytest.mark.parametrize("loader", ["source", "bundle"])
def test_timer_tier_sweeps_temps_writes_marker_and_exits_75(
    tmp_path: Path, loader: str
) -> None:
    temp_path = tmp_path / "stale-write.tmp"
    argv = [sys.executable, "-c", _timer_harness(temp_path, loader)]
    if loader == "bundle":
        argv.append(str(_BUNDLE_PATH))
    argv.append(str(temp_path))

    completed = subprocess.run(
        argv, capture_output=True, timeout=20, cwd=str(_REPOSITORY_ROOT)
    )

    assert completed.returncode == WATCHDOG_EXIT_CODE, (
        f"stderr: {completed.stderr.decode(errors='replace')}"
    )
    assert b"SURVIVED" not in completed.stdout
    assert WATCHDOG_STDERR_MARKER in completed.stderr
    assert not temp_path.exists(), "watchdog left a registered temp behind"


def test_marker_line_is_the_transport_marker() -> None:
    """The fixed stderr line must match what Task 11's transport hunts."""
    from tldw_chatbook.Tools import remote_workspace_transport as transport

    assert WATCHDOG_STDERR_MARKER == b"tldw-worker-watchdog\n"
    assert transport._WATCHDOG_STDERR_MARKER == (b"tldw-worker-watchdog",)
    assert WATCHDOG_EXIT_CODE == 75


# ---------------------------------------------------------------------------
# Disarm: an op completing in time must not die late
# ---------------------------------------------------------------------------


def test_disarm_cancels_timer_and_zeroes_alarm() -> None:
    """Spawned harness: arm, disarm, then outlive the budget cleanly.

    ``signal.alarm(0)`` cancels and RETURNS the remaining seconds — a
    disarm that forgot the alarm leaves a positive remainder here.
    """
    source = (
        "import signal, sys, time\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from tldw_chatbook.Tools import worker_watchdog as wd\n"
        "wd.arm_watchdog(1, wd.TEMP_REGISTRY)\n"
        "wd.disarm_watchdog()\n"
        "remaining = signal.alarm(0)\n"
        "assert remaining == 0, f'alarm not zeroed: {remaining}'\n"
        "assert wd._armed_timer is None, 'timer not cancelled'\n"
        "time.sleep(2.5)\n"
        "print('CLEAN', flush=True)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source, str(_REPOSITORY_ROOT)],
        capture_output=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert b"CLEAN" in completed.stdout
    assert b"tldw-worker-watchdog" not in completed.stderr


def test_op_completing_within_budget_exits_zero_not_late_75(
    tmp_path: Path,
) -> None:
    """Loopback integration: a fast fs_read with budget 1s must return
    BOTH frames, exit 0, and the process must be gone — a missed disarm
    would surface as a late exit 75 (live timer) or a stall."""
    root = _workspace(tmp_path)
    chain = _ping_chain(root)

    started = time.monotonic()
    completed = _spawn(
        _request(
            root,
            "fs_read",
            {"path": "alpha.txt", "sensitive_exclusions": []},
            chain=chain,
            timeout_seconds=1,
        )
    )
    elapsed = time.monotonic() - started

    assert completed.returncode == 0
    assert elapsed < _DEATH_BOUND_SECONDS
    assert WATCHDOG_STDERR_MARKER not in completed.stderr
    frames = executor_module._magic_frame_segments(completed.stdout)
    assert len(frames) == 2
    terminal = json.loads(frames[-1])
    assert terminal["outcome"] == "success"
    assert "alpha body" in (terminal["result"] or "")


# ---------------------------------------------------------------------------
# Budget edge: <= 0 never arms (defense in depth)
# ---------------------------------------------------------------------------


def test_nonpositive_budget_skips_arming() -> None:
    source = (
        "import signal, sys, time\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from tldw_chatbook.Tools import worker_watchdog as wd\n"
        "wd.arm_watchdog(0, wd.TEMP_REGISTRY)\n"
        "wd.arm_watchdog(-3, wd.TEMP_REGISTRY)\n"
        "remaining = signal.alarm(0)\n"
        "assert remaining == 0, f'alarm armed for budget<=0: {remaining}'\n"
        "time.sleep(0.6)\n"
        "print('UNTOUCHED', flush=True)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source, str(_REPOSITORY_ROOT)],
        capture_output=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert b"UNTOUCHED" in completed.stdout


# ---------------------------------------------------------------------------
# Exit-code reservation: only the watchdog path may exit 75
# ---------------------------------------------------------------------------


def test_exit_75_is_reserved_to_the_watchdog() -> None:
    """AST-level pin: docstrings may MENTION exit 75; only the watchdog
    may CALL a hard exit, and its single call site uses the named
    constant (never a bare literal at a non-watchdog call site)."""
    import ast

    def exit_calls(source: str) -> list[str]:
        calls = []
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if (
                isinstance(function, ast.Attribute)
                and function.attr == "_exit"
                and isinstance(function.value, ast.Name)
                and function.value.id in {"os", "sys"}
            ):
                calls.append(ast.unparse(node))
        return calls

    for module_name in BUNDLE_MODULES:
        source = Path(importlib.import_module(module_name).__file__).read_text(
            encoding="utf-8"
        )
        calls = exit_calls(source)
        if module_name == "tldw_chatbook.Tools.worker_watchdog":
            assert calls == ["os._exit(WATCHDOG_EXIT_CODE)"], calls
        else:
            assert not calls, f"{module_name} hard-exits: {calls}"

    # The committed artifact flattens the same closure: exactly one
    # hard-exit call site, the watchdog's.
    artifact = _BUNDLE_PATH.read_text(encoding="utf-8")
    assert exit_calls(artifact) == ["os._exit(WATCHDOG_EXIT_CODE)"]


# ---------------------------------------------------------------------------
# Temp-registry wiring on the atomic-write path (in-process data plane)
# ---------------------------------------------------------------------------


def _dispatch_write(root: Path, arguments: dict[str, Any]) -> str:
    from types import SimpleNamespace

    from tldw_chatbook.Tools.workspace_root_pin import pin_workspace_root
    from tldw_chatbook.Tools.workspace_tool_dispatch import (
        execute_pinned_operation,
    )
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain

    chain = capture_directory_chain(root)
    request = SimpleNamespace(operation="fs_write", arguments=arguments)
    with pin_workspace_root(chain.canonical_root, chain) as pinned:
        return execute_pinned_operation(request, pinned)


def test_atomic_write_registers_then_unregisters_the_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tldw_chatbook.Tools import local_tool_impls

    events: list[tuple[str, str]] = []
    monkeypatch.setattr(
        local_tool_impls, "register_temp", lambda p: events.append(("add", p))
    )
    monkeypatch.setattr(
        local_tool_impls,
        "unregister_temp",
        lambda p: events.append(("drop", p)),
    )

    _dispatch_write(
        tmp_path,
        {
            "path": "out.txt",
            "content": "payload",
            "sensitive_exclusions": [],
        },
    )

    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "payload"
    added = [path for kind, path in events if kind == "add"]
    dropped = [path for kind, path in events if kind == "drop"]
    assert len(added) == 1, "temp must be registered exactly once"
    assert added == dropped, "temp must be unregistered on success"
    assert re.search(r"\.chatbook-write-[0-9a-f]{32}\.tmp$", added[0])
    leftovers = [
        entry.name
        for entry in tmp_path.iterdir()
        if entry.name.startswith(".chatbook-write-")
    ]
    assert not leftovers


def test_failed_precondition_registers_no_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tldw_chatbook.Tools import local_tool_impls
    from tldw_chatbook.Tools.local_tool_impls import LocalToolError

    events: list[str] = []
    monkeypatch.setattr(
        local_tool_impls, "register_temp", lambda p: events.append(p)
    )
    (tmp_path / "cas.txt").write_text("actual", encoding="utf-8")
    stale_digest = "0" * 64

    with pytest.raises(LocalToolError):
        _dispatch_write(
            tmp_path,
            {
                "path": "cas.txt",
                "content": "replacement",
                "expected_sha256": stale_digest,
                "sensitive_exclusions": [],
            },
        )
    assert not events


# ---------------------------------------------------------------------------
# The worker flow arms and disarms (source-level pin)
# ---------------------------------------------------------------------------


def test_worker_flow_arms_after_decode_and_disarms_on_completion() -> None:
    """``run_workspace_worker`` must consume the request's budget."""
    import inspect

    from tldw_chatbook.Tools import workspace_tool_worker

    source = inspect.getsource(workspace_tool_worker.run_workspace_worker)
    assert "arm_watchdog(request.timeout_seconds" in source
    assert "disarm_watchdog()" in source
    # _DecodedRequest carries the budget off the wire.
    request_fields = {
        name
        for name in workspace_tool_worker._DecodedRequest.__dataclass_fields__
    }
    assert "timeout_seconds" in request_fields
