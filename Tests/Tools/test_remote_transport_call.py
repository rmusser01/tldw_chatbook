"""RemoteWorkspaceTransport.call: the per-call ssh path (Phase 2b, Task 11).

Every test drives ``call`` against a fake ``ssh`` (a bash script) — no
network, no real ssh. The fake logs argv, answers the master-lifecycle
ops the manager drives (``-O check`` / ``-MNf``), and plays one
``FAKE_SSH_MODE`` scenario per call, covering every row of the failure
taxonomy:

* no admitted marker + exit 255 / 127 / 76 / other → UNREACHABLE /
  INTERPRETER_MISSING / PYTHON_TOO_OLD / WORKER_FAILED_TO_START;
* no marker + deadline kill → UNREACHABLE "handshake stalled" (a
  stalled handshake is transport-class, never OP_TIMEOUT);
* admitted + deadline kill or watchdog exit 75 → OP_TIMEOUT;
* admitted + other exits → REMOTE_OP_FAILED (reason from the
  ``tldw-worker-watchdog`` stderr marker when present);
* >4KB pre-magic stdout → STDOUT_NOISE;
* exit 255 + a mux error on stderr → MUX_ERROR plus a
  failure-triggered master restart (asserted through the fake's ssh
  invocation log) — and NO retry of the failed call.

Also pinned: the spawn argv layout (manager client options, then argv
rebuilt from parsed parts, then ``python -I -c <bootstrap>``),
``start_new_session=True`` (the deadline kill's own process group), the
bootstrap's N matching the compressed bundle length on stdin, the
completion deadline anchored to the marker's ARRIVAL (not spawn), and
the process-group kill actually reaping the fake (its PID is gone
afterwards).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.Tools.remote_binding_locator import parse_remote_locator
from tldw_chatbook.Tools.remote_workspace_executor import _bundle_payload
from tldw_chatbook.Tools.remote_workspace_transport import (
    RemoteWorkspaceTransport,
    SshMasterManager,
    TransportFailureKind,
)

# The fake ssh, verbatim (bash). ``%C`` substitution uses a fixed
# 40-character token, mirroring the master-manager fake. Scenario knobs
# arrive via the environment so the transport's Popen inherits them.
_FAKE_SSH_BASH = r"""#!/bin/bash
# Fake ssh for RemoteWorkspaceTransport.call tests (Task 11).
log="${FAKE_SSH_LOG:-}"
if [ -n "$log" ]; then
  {
    echo "=== argv ==="
    for arg in "$@"; do
      echo "$arg"
    done
  } >> "$log"
fi

# -- master lifecycle (for the mux-restart assertion) ----------------------
prev=""
for arg in "$@"; do
  if [ "$prev" = "-O" ]; then
    case "$arg" in
      check) exit 1 ;;
      exit)  exit 0 ;;
    esac
  fi
  prev="$arg"
done
for arg in "$@"; do
  if [ "$arg" = "-MNf" ]; then
    for opt in "$@"; do
      case "$opt" in
        ControlPath=*)
          sock="${opt#ControlPath=}"
          sock="${sock//%C/cccccccccccccccccccccccccccccccccccc}"
          mkdir -p "$(dirname "$sock")"
          touch "$sock"
          ;;
      esac
    done
    exit 0
  fi
done

# -- per-call scenarios -----------------------------------------------------
MAGIC="TLDW-REMOTE-0001"
mode="${FAKE_SSH_MODE:-success}"
capture="${FAKE_SSH_STDIN_CAPTURE:-}"
admitted="${FAKE_SSH_ADMITTED_FRAME:-}"
final="${FAKE_SSH_FINAL_FRAME:-}"
stderr_line="${FAKE_SSH_STDERR_LINE:-}"
exit_code="${FAKE_SSH_EXIT:-0}"

drain_stdin() {
  if [ -n "$capture" ]; then
    cat > "$capture"
  else
    cat > /dev/null
  fi
}

emit_admitted() {
  printf '%s%s\n' "$MAGIC" "$admitted"
}

write_pid() {
  if [ -n "${FAKE_SSH_PID_FILE:-}" ]; then
    echo $$ > "$FAKE_SSH_PID_FILE"
  fi
}

case "$mode" in
  success)
    if [ "${FAKE_SSH_NOISE:-0}" = "1" ]; then
      printf 'Welcome to build-box 12.3\nLast login: from the tests\n'
    fi
    drain_stdin
    emit_admitted
    printf '%s%s\n' "$MAGIC" "$final"
    exit 0
    ;;
  single-frame)
    drain_stdin
    printf '%s%s\n' "$MAGIC" "$final"
    exit 2
    ;;
  exit-only)
    if [ -n "$stderr_line" ]; then
      printf '%s\n' "$stderr_line" >&2
    fi
    exit "$exit_code"
    ;;
  admitted-then-exit)
    drain_stdin
    emit_admitted
    if [ -n "$stderr_line" ]; then
      printf '%s\n' "$stderr_line" >&2
    fi
    exit "$exit_code"
    ;;
  admitted-then-slow-exit)
    # The worker's own alarm/rlimit tier killed the remote python
    # mid-op: ssh then exits on its own (no final frame) BEFORE the
    # laptop's completion deadline would fire.
    write_pid
    drain_stdin
    emit_admitted
    if [ -n "$stderr_line" ]; then
      printf '%s\n' "$stderr_line" >&2
    fi
    sleep "${FAKE_SSH_SLOW_SECONDS:-1.0}"
    exit "$exit_code"
    ;;
  stall)
    write_pid
    drain_stdin
    sleep 30
    ;;
  admitted-then-stall)
    write_pid
    drain_stdin
    emit_admitted
    sleep 30
    ;;
  late-admitted-then-stall)
    write_pid
    drain_stdin
    sleep "${FAKE_SSH_ADMIT_DELAY:-0.6}"
    emit_admitted
    sleep 30
    ;;
  noise-capped)
    write_pid
    dd if=/dev/zero bs=5120 count=1 2>/dev/null | tr '\0' 'x'
    drain_stdin
    sleep 30
    ;;
esac
exit 0
"""

#: One admitted-marker frame and one terminal frame, wire-shaped. Defined
#: once here; the fake prints them verbatim after the magic.
_ADMITTED_FRAME = json.dumps(
    {
        "version": 1,
        "operation_id": "cafe1234",
        "outcome": "admitted",
        "code": "root_pinned",
        "result": None,
        "error": None,
        "elapsed_ms": 1,
        "truncated": False,
        "cleanup_proven": True,
    },
    separators=(",", ":"),
).encode("utf-8")
_FINAL_FRAME = json.dumps(
    {
        "version": 1,
        "operation_id": "cafe1234",
        "outcome": "success",
        "code": "ok",
        "result": "remote-answer",
        "error": None,
        "elapsed_ms": 5,
        "truncated": False,
        "cleanup_proven": True,
    },
    separators=(",", ":"),
).encode("utf-8")
_REFUSAL_FRAME = json.dumps(
    {
        "version": 1,
        "operation_id": "cafe1234",
        "outcome": "failure",
        "code": "root_pin_failed",
        "result": None,
        "error": "identity mismatch",
        "elapsed_ms": 2,
        "truncated": False,
        "cleanup_proven": True,
    },
    separators=(",", ":"),
).encode("utf-8")

#: The request payload is opaque to the transport; any bytes will do.
_REQUEST = (
    b'{"version":1,"operation_id":"cafe1234","operation":"ping"}'
)

_LOCATOR = "ssh://ops@build-box.internal:2222/srv/work/workspace"


def _short_state_dir() -> Path:
    """Unique SHORT state dir: pytest's ``tmp_path`` alone exceeds the
    ControlPath sun_path budget, which would route these managers onto
    the manager's /tmp fallback instead of the primary layout."""
    return Path(f"/tmp/tldw-cs-test-{os.getpid()}-{uuid.uuid4().hex[:8]}")
_LOC = parse_remote_locator(_LOCATOR)

#: Marker-arrival timing shared by the fast deadline tests (budget +
#: grace), kept small for suite speed; the anchoring test uses its own.
_BUDGET = 0.3
_GRACE = 0.3


class FakeSsh:
    """Installs the bash fake binary and parses its invocation log."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.bin = tmp_path / "fake-ssh"
        self.bin.write_text(_FAKE_SSH_BASH, encoding="utf-8")
        self.bin.chmod(0o755)
        self.log_path = tmp_path / "ssh-invocations.log"
        self.stdin_capture = tmp_path / "call-stdin.bin"
        self.pid_file = tmp_path / "fake-ssh.pid"
        self.state_dir = _short_state_dir()
        monkeypatch.setenv("FAKE_SSH_LOG", str(self.log_path))
        monkeypatch.setenv("FAKE_SSH_STDIN_CAPTURE", str(self.stdin_capture))
        monkeypatch.setenv("FAKE_SSH_PID_FILE", str(self.pid_file))
        monkeypatch.setenv("FAKE_SSH_ADMITTED_FRAME", _ADMITTED_FRAME.decode())
        monkeypatch.setenv("FAKE_SSH_FINAL_FRAME", _FINAL_FRAME.decode())
        for stale in (
            "FAKE_SSH_MODE",
            "FAKE_SSH_NOISE",
            "FAKE_SSH_STDERR_LINE",
            "FAKE_SSH_EXIT",
            "FAKE_SSH_ADMIT_DELAY",
            "FAKE_SSH_SLOW_SECONDS",
        ):
            monkeypatch.delenv(stale, raising=False)

    def invocations(self) -> list[list[str]]:
        """One argv list per logged invocation (argv[0] never logged)."""
        if not self.log_path.exists():
            return []
        argvs: list[list[str]] = []
        current: list[str] = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            if line == "=== argv ===":
                if current:
                    argvs.append(current)
                current = []
            else:
                current.append(line)
        if current:
            argvs.append(current)
        return argvs

    def count(self, token: str) -> int:
        return sum(argv.count(token) for argv in self.invocations())

    def call_invocations(self) -> list[list[str]]:
        """Per-call client invocations (the ones carrying ``-I``)."""
        return [argv for argv in self.invocations() if "-I" in argv]

    def assert_pid_gone(self) -> None:
        pid = int(self.pid_file.read_text(encoding="utf-8").strip())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.fixture()
def env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake = FakeSsh(tmp_path, monkeypatch)
    manager = SshMasterManager(
        ssh_bin=str(fake.bin),
        control_persist="10m",
        enabled=True,
        connect_timeout_s=3,
        state_dir=fake.state_dir,
    )
    transport = RemoteWorkspaceTransport(manager, grace_seconds=_GRACE)

    def call(**kwargs: Any) -> Any:
        merged = {"budget": _BUDGET}
        merged.update(kwargs)
        return transport.call(_LOC, _REQUEST, **merged)

    yield SimpleNamespace(
        fake=fake, manager=manager, transport=transport, call=call
    )
    shutil.rmtree(fake.state_dir, ignore_errors=True)


def _call_argv(fake: FakeSsh) -> list[str]:
    calls = fake.call_invocations()
    assert len(calls) == 1, fake.invocations()
    return calls[0]


# ---------------------------------------------------------------------------
# success: noise-tolerant framing
# ---------------------------------------------------------------------------


def test_success_strips_newline_noise_and_returns_final_frame(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SSH_MODE", "success")
    monkeypatch.setenv("FAKE_SSH_NOISE", "1")

    result = env.call()

    assert result.failure is None
    assert result.admitted is True
    # The final frame verbatim — no magic, no newline, and none of the
    # newline-containing noise (byte-level scan, never line-based).
    assert result.response == _FINAL_FRAME
    assert b"Welcome to build-box" not in result.response
    assert b"Last login" not in result.response
    assert json.loads(result.response)["code"] == "ok"


def test_magic_and_frames_may_straddle_read_chunk_boundaries(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """7-byte reads split the 16-byte magic and both frames mid-way.

    The scan must keep a possible magic prefix pending across reads
    instead of classifying it as garbage, and a frame must complete only
    at its newline.
    """
    monkeypatch.setenv("FAKE_SSH_MODE", "success")
    monkeypatch.setattr(
        "tldw_chatbook.Tools.remote_workspace_transport._READ_CHUNK_BYTES",
        7,
    )

    result = env.call()

    assert result.failure is None
    assert result.admitted is True
    assert result.response == _FINAL_FRAME


def test_single_refusal_frame_is_a_response_not_a_transport_failure(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker-refused op frames its failure; the transport delivered it."""
    monkeypatch.setenv("FAKE_SSH_MODE", "single-frame")
    monkeypatch.setenv("FAKE_SSH_FINAL_FRAME", _REFUSAL_FRAME.decode())

    result = env.call()

    assert result.failure is None
    assert result.admitted is False
    assert result.response == _REFUSAL_FRAME


# ---------------------------------------------------------------------------
# spawn: argv layout, bootstrap N, own session
# ---------------------------------------------------------------------------


def test_call_argv_layout_and_bootstrap_match_compressed_bundle(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SSH_MODE", "success")

    env.call(python="python3.11")

    argv = _call_argv(env.fake)
    manager = env.manager
    expected_options = [
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=3",
        "-o",
        "ControlMaster=no",
        "-o",
        f"ControlPath={manager.control_path_for(_LOC).as_posix()}/%C",
    ]
    assert argv[: len(expected_options)] == expected_options, argv
    # argv rebuilt from parsed parts, then the interpreter triple.
    tail = argv[len(expected_options):]
    assert tail[:5] == ["-p", "2222", "-l", "ops", "--"], tail
    assert tail[5:8] == ["build-box.internal", "python3.11", "-I"], tail
    assert tail[8] == "-c"
    bootstrap = tail[9]
    assert len(tail) == 10, tail

    # N in the bootstrap is exactly the compressed bundle byte count,
    # and stdin carried compressed-bundle || request.
    import re

    sizes = re.findall(r"sys\.stdin\.buffer\.read\((\d+)\)", bootstrap)
    assert len(sizes) == 1, bootstrap
    n = int(sizes[0])
    captured = env.fake.stdin_capture.read_bytes()
    assert len(captured) == n + len(_REQUEST)
    assert captured[:2] == b"\x78\x9c"  # zlib header
    bundle, _compressed, _bootstrap = _bundle_payload()
    assert zlib.decompress(captured[:n]) == bundle
    assert captured[n:] == _REQUEST


def test_call_spawns_in_own_session_with_piped_std_streams(
    env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_SSH_MODE", "exit-only")
    monkeypatch.setenv("FAKE_SSH_EXIT", "255")
    captured: dict[str, Any] = {}
    real_popen = subprocess.Popen

    def spying_popen(argv: list[str], **kwargs: Any) -> Any:
        captured["argv"] = argv
        captured.update(kwargs)
        return real_popen(argv, **kwargs)

    monkeypatch.setattr(
        "tldw_chatbook.Tools.remote_workspace_transport.subprocess.Popen",
        spying_popen,
    )

    env.call()

    assert captured["argv"][0] == str(env.fake.bin)
    assert captured["start_new_session"] is True
    assert captured["stdin"] is subprocess.PIPE
    assert captured["stdout"] is subprocess.PIPE
    assert captured["stderr"] is subprocess.PIPE
    assert captured.get("shell") in (None, False)


def test_call_logs_bundle_stamp_audit_line_per_call(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec audit rule: every spawn logs the bundle hash at debug level.

    One line per ``call`` carrying the identity-free endpoint label
    (the host key tuple — user/host/port; no credentials exist in a
    locator), the committed bundle's stamp prefix derived through the
    SAME single-source rule the wire contract verifies, and the
    request's operation name; never request content.
    """
    import tldw_chatbook.Tools.remote_workspace_transport as transport_module
    from tldw_chatbook.Tools.build_remote_worker_bundle import (
        expected_bundle_stamp,
    )

    monkeypatch.setenv("FAKE_SSH_MODE", "success")
    artifact, _compressed, _bootstrap = _bundle_payload()
    stamp_prefix = expected_bundle_stamp(artifact)[:12]

    records: list[Any] = []
    sink_id = transport_module.logger.add(
        lambda message: records.append(message.record), level="DEBUG"
    )
    try:
        assert env.call().failure is None
    finally:
        transport_module.logger.remove(sink_id)

    audit = [
        record
        for record in records
        if record["message"].startswith("remote workspace call")
    ]
    assert len(audit) == 1, [record["message"] for record in records]
    message = audit[0]["message"]
    assert audit[0]["level"].name == "DEBUG"
    assert repr(transport_module._host_key(_LOC)) in message
    assert f"bundle={stamp_prefix}" in message
    assert "op=ping" in message
    # Audit label only — no request content rides the line.
    assert "operation_id" not in message


# ---------------------------------------------------------------------------
# taxonomy: no admitted marker
# ---------------------------------------------------------------------------


def test_no_marker_exit_255_is_unreachable(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SSH_MODE", "exit-only")
    monkeypatch.setenv("FAKE_SSH_EXIT", "255")

    result = env.call()

    assert result.admitted is False
    assert result.response is None
    kind, reason = result.failure.kind, result.failure.reason
    assert kind is TransportFailureKind.UNREACHABLE
    assert reason == "unreachable or auth failed"
    assert result.failure.exit_code == 255


def test_no_marker_exit_127_is_interpreter_missing(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SSH_MODE", "exit-only")
    monkeypatch.setenv("FAKE_SSH_EXIT", "127")

    result = env.call()

    assert result.admitted is False
    failure = result.failure
    assert failure.kind is TransportFailureKind.INTERPRETER_MISSING
    assert failure.reason == "host lacks python3"
    assert failure.exit_code == 127


def test_no_marker_exit_76_parses_found_python_version(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SSH_MODE", "exit-only")
    monkeypatch.setenv("FAKE_SSH_EXIT", "76")
    monkeypatch.setenv(
        "FAKE_SSH_STDERR_LINE", "tldw-worker:python3.10+:found:3.9.7"
    )

    result = env.call()

    failure = result.failure
    assert failure.kind is TransportFailureKind.PYTHON_TOO_OLD
    assert failure.reason == "python ≥ 3.10 required (found 3.9.7)"
    assert failure.exit_code == 76


def test_no_marker_exit_76_without_version_uses_fallback_reason(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit 76 is reserved for the version gate even if stderr is odd."""
    monkeypatch.setenv("FAKE_SSH_MODE", "exit-only")
    monkeypatch.setenv("FAKE_SSH_EXIT", "76")
    monkeypatch.setenv("FAKE_SSH_STDERR_LINE", "something else entirely")

    result = env.call()

    failure = result.failure
    assert failure.kind is TransportFailureKind.PYTHON_TOO_OLD
    assert failure.reason == "python ≥ 3.10 required"
    assert failure.exit_code == 76


def test_no_marker_silent_exit_1_is_worker_failed_to_start(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SSH_MODE", "exit-only")
    monkeypatch.setenv("FAKE_SSH_EXIT", "1")

    result = env.call()

    failure = result.failure
    assert failure.kind is TransportFailureKind.WORKER_FAILED_TO_START
    assert failure.reason == "remote worker failed to start"
    assert failure.exit_code == 1


def test_pre_magic_garbage_cap_is_stdout_noise(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """5KB of magicless stdout trips the 4KB cap — mid-read, not on exit."""
    monkeypatch.setenv("FAKE_SSH_MODE", "noise-capped")

    started = time.monotonic()
    result = env.call()
    elapsed = time.monotonic() - started

    assert result.admitted is False
    failure = result.failure
    assert failure.kind is TransportFailureKind.STDOUT_NOISE
    assert elapsed < 5.0, "the cap must fire while reading, not on exit"
    env.fake.assert_pid_gone()


def test_no_marker_deadline_kill_is_unreachable_handshake_stalled(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The marker-arrival deadline kill is transport-class, never OP_TIMEOUT."""
    monkeypatch.setenv("FAKE_SSH_MODE", "stall")

    started = time.monotonic()
    result = env.call()
    elapsed = time.monotonic() - started

    failure = result.failure
    assert failure.kind is TransportFailureKind.UNREACHABLE
    assert failure.reason == "handshake stalled"
    assert result.admitted is False
    # Killed at budget + grace, not the fake's 30s sleep.
    assert elapsed < 5.0
    env.fake.assert_pid_gone()


# ---------------------------------------------------------------------------
# taxonomy: admitted marker seen
# ---------------------------------------------------------------------------


def test_admitted_deadline_kill_is_op_timeout_and_reaps_the_group(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SSH_MODE", "admitted-then-stall")

    started = time.monotonic()
    result = env.call()
    elapsed = time.monotonic() - started

    assert result.admitted is True
    failure = result.failure
    assert failure.kind is TransportFailureKind.OP_TIMEOUT
    assert failure.reason == "operation timed out"
    assert failure.exit_code == -9  # SIGKILL, the group kill
    assert elapsed < 5.0
    env.fake.assert_pid_gone()


def test_completion_deadline_is_anchored_to_marker_arrival(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The completion kill counts from admitted_at, not from spawn.

    Marker delayed 0.6s, budget 0.5 + grace 0.7: a spawn-anchored kill
    would fire at 1.2s, the specified anchor fires at 0.6 + 1.2 = 1.8s.
    """
    monkeypatch.setenv("FAKE_SSH_MODE", "late-admitted-then-stall")
    monkeypatch.setenv("FAKE_SSH_ADMIT_DELAY", "0.6")
    transport = RemoteWorkspaceTransport(env.manager, grace_seconds=0.7)

    started = time.monotonic()
    result = transport.call(_LOC, _REQUEST, budget=0.5)
    elapsed = time.monotonic() - started

    assert result.admitted is True
    assert result.failure.kind is TransportFailureKind.OP_TIMEOUT
    assert 1.45 <= elapsed <= 3.0, (
        f"kill at {elapsed:.2f}s: not anchored to marker arrival"
    )
    env.fake.assert_pid_gone()


def test_admitted_exit_75_is_op_timeout(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SSH_MODE", "admitted-then-exit")
    monkeypatch.setenv("FAKE_SSH_EXIT", "75")
    monkeypatch.setenv(
        "FAKE_SSH_STDERR_LINE",
        "tldw-worker-watchdog: budget exhausted, temp files swept",
    )

    result = env.call()

    assert result.admitted is True
    failure = result.failure
    assert failure.kind is TransportFailureKind.OP_TIMEOUT
    assert failure.reason == "operation timed out"
    assert failure.exit_code == 75


def test_admitted_exit_255_is_remote_op_failed(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SSH_MODE", "admitted-then-exit")
    monkeypatch.setenv("FAKE_SSH_EXIT", "255")

    result = env.call()

    assert result.admitted is True
    failure = result.failure
    assert failure.kind is TransportFailureKind.REMOTE_OP_FAILED
    assert failure.reason == "remote op failed"
    assert failure.exit_code == 255


def test_admitted_fast_death_without_marker_stays_remote_op_failed(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fast post-admission death (elapsed far under budget): a mid-op
    network/process death is an operation failure, not a timeout."""
    monkeypatch.setenv("FAKE_SSH_MODE", "admitted-then-exit")
    monkeypatch.setenv("FAKE_SSH_EXIT", "2")

    result = env.call()

    assert result.admitted is True
    failure = result.failure
    assert failure.kind is TransportFailureKind.REMOTE_OP_FAILED
    assert failure.exit_code == 2


def test_admitted_watchdog_stderr_marker_is_op_timeout(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task-12 ruling: the worker's graceful tier writes the marker and
    exits (here a non-75 code, e.g. a marker line surviving a different
    exit) — the marker alone types the failure OP_TIMEOUT even when the
    death was fast."""
    monkeypatch.setenv("FAKE_SSH_MODE", "admitted-then-exit")
    monkeypatch.setenv("FAKE_SSH_EXIT", "2")
    monkeypatch.setenv(
        "FAKE_SSH_STDERR_LINE",
        "tldw-worker-watchdog: budget exhausted, temp files swept",
    )

    result = env.call()

    assert result.admitted is True
    failure = result.failure
    assert failure.kind is TransportFailureKind.OP_TIMEOUT
    assert failure.reason == "operation timed out"
    assert failure.exit_code == 2


def test_admitted_full_budget_death_by_signal_is_op_timeout(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task-12 ruling clock rule: the worker's own alarm/RLIMIT_CPU tier
    kills the remote python mid-op; ssh dies on its own (exit 255, no
    final frame, no marker) BEFORE the laptop's completion deadline.
    ssh does not decode remote signal deaths, so the classifier uses the
    clock: the op ran its full budget — a timeout by definition,
    regardless of which tier fired. Elapsed < budget stays
    REMOTE_OP_FAILED (tests above).
    """
    monkeypatch.setenv("FAKE_SSH_MODE", "admitted-then-slow-exit")
    monkeypatch.setenv("FAKE_SSH_EXIT", "255")
    monkeypatch.setenv("FAKE_SSH_SLOW_SECONDS", "1.0")
    # Grace 2.0s keeps the laptop kill (admitted_at + budget + grace)
    # strictly BEHIND the fake's own 1.0s exit: the clock rule, not the
    # kill bit, must type this failure.
    transport = RemoteWorkspaceTransport(env.manager, grace_seconds=2.0)

    started = time.monotonic()
    result = transport.call(_LOC, _REQUEST, budget=0.5)
    elapsed = time.monotonic() - started

    assert result.admitted is True
    failure = result.failure
    assert failure.kind is TransportFailureKind.OP_TIMEOUT
    assert failure.reason == "operation timed out"
    assert failure.exit_code == 255  # died on its own; we did not kill
    assert elapsed < 3.0
    env.fake.assert_pid_gone()


# ---------------------------------------------------------------------------
# mux errors: typed failure + failure-triggered restart, no retry
# ---------------------------------------------------------------------------


def test_mux_error_is_typed_and_restarts_the_master_without_retry(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SSH_MODE", "exit-only")
    monkeypatch.setenv("FAKE_SSH_EXIT", "255")
    monkeypatch.setenv(
        "FAKE_SSH_STDERR_LINE",
        "ssh: ControlSocket /state/cs/deadbeef: Connection refused",
    )

    result = env.call()

    assert result.admitted is False
    failure = result.failure
    assert failure.kind is TransportFailureKind.MUX_ERROR
    assert failure.exit_code == 255
    # Failure-triggered master restart ran (fake -O check says dead, so
    # one -MNf spawn) — asserted through the fake's ssh invocation log.
    assert env.fake.count("-MNf") == 1, env.fake.invocations()
    # No retry: exactly one per-call invocation happened for this call.
    assert len(env.fake.call_invocations()) == 1


# ---------------------------------------------------------------------------
# local errors: the call contract holds and nothing leaks
# ---------------------------------------------------------------------------


def test_watch_exchange_error_is_typed_failure_and_reaps_the_child(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local exception in the watch loop must not escape or leak ssh.

    ``call`` promises a typed result for every failed exchange; a raise
    from ``_watch_exchange`` (only ValueError is caught at the parse
    site, so e.g. RecursionError escapes it) must become a typed
    failure, and the still-running ssh must be killed and reaped —
    never orphaned.
    """
    monkeypatch.setenv("FAKE_SSH_MODE", "stall")

    pid_path = env.fake.pid_file

    def exploding_watch(
        proc: object, *, budget: float, grace: float
    ) -> object:
        # Let the fake publish its PID first, so the reaped-process
        # assertion below cannot race the bash startup.
        deadline = time.monotonic() + 5.0
        while not pid_path.exists():
            if time.monotonic() >= deadline:
                raise AssertionError("fake ssh never wrote its pid file")
            time.sleep(0.01)
        raise RecursionError("deeply nested frame")

    monkeypatch.setattr(
        "tldw_chatbook.Tools.remote_workspace_transport._watch_exchange",
        exploding_watch,
    )

    result = env.call()  # must not raise

    assert result.admitted is False
    failure = result.failure
    assert failure.kind is TransportFailureKind.WORKER_FAILED_TO_START
    assert "RecursionError" in failure.reason
    assert failure.exit_code == -9  # the reap-everything path group-killed it
    env.fake.assert_pid_gone()


# ---------------------------------------------------------------------------
# knobs
# ---------------------------------------------------------------------------


def test_grace_defaults_to_five_seconds(env: SimpleNamespace) -> None:
    default = RemoteWorkspaceTransport(env.manager)
    assert default.grace_seconds == 5.0
