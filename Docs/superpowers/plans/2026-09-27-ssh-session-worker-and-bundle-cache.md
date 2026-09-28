# SSH Session Worker and Bundle Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut a warm SSH-binding tool call from ~0.7 s to about one network round trip with a per-run fork-server session on the host, plus a host-side bundle cache in `$XDG_RUNTIME_DIR`.

**Architecture:** A stage-1 loader (sent through the unchanged bootstrap) checks a sha256-verified cache, loads the bundle outside `__main__`, and enters `serve`: a single-threaded parent that forks one child per request. Each child runs today's `run_workspace_worker` unchanged. Frames (`u32 len | u32 id | u8 kind | body`) multiplex requests over one ssh channel. On the laptop, a `RemoteSessionWorker` per `(run key, binding)` returns the same `RemoteCallResult` as the one-shot transport, so the executor, the status cache and the taxonomy are unchanged; the one-shot path stays as the fallback.

**Tech Stack:** Python ≥3.12 laptop, stdlib-only host code (floor 3.10), OpenSSH ControlMaster, `selectors`, `os.fork`, pytest.

**Spec:** `Docs/superpowers/specs/2026-09-27-ssh-session-worker-and-bundle-cache-design.md` — read it alongside this plan; where they disagree, the spec wins and the plan gets fixed.

## Global Constraints

- **Branch:** the implementation branch is cut from `origin/dev`, not from the spec branch; cherry-pick the spec + plan commits onto it first.
- **Host code** (frames, serve, loader, bundle IO adapter): stdlib-only, Python ≥ 3.10, no threads in the serve parent, never `signal.alarm` in the parent.
- **Frame format:** `struct ">IIB"` header — length of body (u32), request id (u32), kind (u8) — then body. Kinds: `HELLO=1, REQUEST=2, CANCEL=3, LINE=16, STATUS=17, BUSY=18`.
- **Size caps, enforced before buffering:** host inbound `MAX_REQUEST_BYTES`; laptop inbound and per-child output `MAX_RESPONSE_BYTES`; loader bundle ≤ 8 MiB (`8 << 20`).
- **Magic:** `RESPONSE_MAGIC = b"TLDW-REMOTE-0001"` prefixes the loader's `NEED`/`READY` lines only; frames after `READY` have no magic.
- **Loader stdin contract:** `<N zlib bytes of loader><header JSON line: {"hash": <sha256 hex of compressed bundle>, "cache": bool}>`, then (only after `NEED`) `u32 length + compressed bundle`, then (only after `READY`) frames. **Lockstep:** the laptop never sends bytes the loader has not asked for — this keeps `sys.stdin.buffer`'s read-ahead empty when `serve` switches to `os.read(0)`.
- **Cache:** `$XDG_RUNTIME_DIR/tldw-worker/<hash>`, used only when base and dir are owned by the uid with no group/other bits, file mode 0600 and sha256 matches; atomic write; deletes other entries; skipped when `cache` is false or the base is missing/not private. Never `/tmp`, never `~/.cache`.
- **Config (`[console_ssh]`):** `session_worker = true`, `session_idle_s = 60`, `bundle_cache = true`.
- **Exit codes unchanged:** 75 watchdog, 76 version gate, 127 interpreter missing, 2 worker framed failure, 255 ssh-level. Child crash inside serve → `os._exit(70)`.
- **Boot ratchet (ADR-097):** no new module resident at UI-ready; new laptop modules are imported only from the SSH execution path.
- **Repo gates:** `PYTHON=<venv python> ./scripts/preflight.sh` passes before push; regenerate the bundle with `python -m tldw_chatbook.Tools.build_remote_worker_bundle` after touching any bundled module; review diagnostic statements before `--write`.
- **Tests:** targeted runs only; the known local `RecoveryRequired` failures are environmental — compare against base, don't chase them.

## Review Focus

1. **Stdin read-ahead at the loader→serve handoff:** a laptop that pipelines `HELLO`/`REQUEST` before `READY` would leave bytes in `sys.stdin.buffer` that `os.read(0)` never sees and the session hangs. Pinned by a Task 6 test that the laptop sends nothing between the header and `NEED`/`READY`.
2. **A child that inherits the channel fds:** a child still holding fd 1 keeps the ssh channel open after the parent exits, and stray writes corrupt framing. Pinned by a Task 3 test (child `/proc/self/fd` or `os.fstat` probe on fd 1 after fork).
3. **Two sessions for one binding in one run:** the tool executor and the AGENTS.md reader both call `for_ssh`. Pinned by a Task 7 registry test and a Task 8 controller test (one ssh spawn per binding per run).
4. **A laptop-side deadline on a hung parent:** `CANCEL` sent, no `STATUS` — must kill the session and fail the call, never block the run. Pinned in Task 6.
5. **App shutdown mid-run:** live sessions must close (stdin EOF) on app exit, not leave remote parents until idle. Pinned by a Task 8 test that `close_all_remote_sessions()` is called from the existing SSH master shutdown hook.

---

### Task 1: Latency spike (throwaway; go/no-go)

**Files:** scratch only (session scratchpad); nothing committed except a results paragraph in this plan.

**Interfaces:** Produces the measured floor (`echo_median_ms`, `echo_p90_ms`, `fork_pin_op_ms`) used by Task 9's success check.

- [ ] **Step 1: Write the echo server + client**

```python
# scratch/echo_spike.py — run from the laptop
import os, statistics, struct, subprocess, time, uuid
HOST = "ml-user@192.168.5.84"; CP = f"/tmp/tes-{uuid.uuid4().hex[:6]}"
SERVER = r'''
import os, struct, sys
r, w = 0, 1
buf = b""
while True:
    d = os.read(r, 65536)
    if not d: break
    buf += d
    while len(buf) >= 9:
        n, rid, kind = struct.unpack(">IIB", buf[:9])
        if len(buf) < 9 + n: break
        body = buf[9:9 + n]; buf = buf[9 + n:]
        if kind == 9:   # fork+exit probe
            pid = os.fork()
            if pid == 0: os._exit(0)
            os.waitpid(pid, 0)
        out = struct.pack(">IIB", len(body), rid, 16) + body + struct.pack(">IIB", 0, rid, 17)
        os.write(w, out)
'''
base = ["ssh", "-o", "BatchMode=yes", "-o", f"ControlPath={CP}"]
subprocess.run(["ssh", "-o", "BatchMode=yes", "-MNf", "-o", f"ControlPath={CP}", "-o", "ControlPersist=120", HOST], check=True)
p = subprocess.Popen(base + [HOST, "python3", "-I", "-c", repr(SERVER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
def rt(kind):
    s = time.perf_counter()
    os.write(p.stdin.fileno(), struct.pack(">IIB", 64, 1, kind) + b"x" * 64)
    got = b""
    while len(got) < 9 + 64 + 9:  # LINE frame + empty STATUS frame
        got += os.read(p.stdout.fileno(), 65536)
    return (time.perf_counter() - s) * 1000
echo = [rt(2) for _ in range(60)][10:]
fork = [rt(9) for _ in range(60)][10:]
pings = subprocess.run(["ping", "-c", "50", "-i", "0.2", "-q", HOST.split("@")[1]], capture_output=True, text=True).stdout
print("echo median/p90", statistics.median(echo), sorted(echo)[int(len(echo) * .9)])
print("fork median", statistics.median(fork)); print(pings.splitlines()[-1])
p.stdin.close(); p.wait(); subprocess.run(["ssh", "-o", f"ControlPath={CP}", "-O", "exit", HOST], capture_output=True)
```

- [ ] **Step 2: Run it three times at different moments** and note echo median/p90, fork median, and ping median.

- [ ] **Step 3: Decide.** Go if echo median ≤ ping median + 15 ms. If echo is consistently ≥ 40 ms above ping (delayed-ACK stall), try `-o IPQoS=lowdelay` on the master in the spike; record which setting wins and carry it into Task 6's master options. Record results under "Spike results" at the end of this plan and commit the plan edit:

```bash
git add Docs/superpowers/plans/2026-09-27-ssh-session-worker-and-bundle-cache.md
git commit -m "docs: record SSH session latency spike results"
```

---

### Task 2: Frame codec (shared, stdlib)

**Files:**
- Create: `tldw_chatbook/Tools/remote_session_frames.py`
- Test: `Tests/Tools/test_remote_session_frames.py`

**Interfaces:**
- Produces:
  - constants `HELLO, REQUEST, CANCEL, LINE, STATUS, BUSY` (ints), `HEADER = struct.Struct(">IIB")`
  - `encode_frame(kind: int, request_id: int, body: bytes) -> bytes`
  - `class FrameError(ValueError)`
  - `class FrameReader: __init__(self, max_body: int); feed(self, data: bytes) -> list[tuple[int, int, bytes]]` returning `(kind, request_id, body)`; raises `FrameError` as soon as a header announces `length > max_body`
  - `encode_status(exit_code: int | None, signal_no: int | None) -> bytes`, `decode_status(body: bytes) -> tuple[int | None, int | None]`

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Tools/test_remote_session_frames.py
import pytest
from tldw_chatbook.Tools.remote_session_frames import (
    LINE, REQUEST, STATUS, FrameError, FrameReader, decode_status,
    encode_frame, encode_status,
)

def test_roundtrip_split_across_arbitrary_chunks():
    data = encode_frame(REQUEST, 7, b"hello") + encode_frame(LINE, 8, b"") + encode_frame(STATUS, 7, encode_status(0, None))
    reader = FrameReader(max_body=1024)
    got = []
    for i in range(len(data)):
        got += reader.feed(data[i:i + 1])
    assert got == [(REQUEST, 7, b"hello"), (LINE, 8, b""), (STATUS, 7, encode_status(0, None))]

def test_oversize_header_rejected_before_body_arrives():
    reader = FrameReader(max_body=10)
    with pytest.raises(FrameError):
        reader.feed(encode_frame(REQUEST, 1, b"x" * 11)[:9])

def test_status_codec():
    assert decode_status(encode_status(75, None)) == (75, None)
    assert decode_status(encode_status(None, 9)) == (None, 9)

def test_request_id_range_checked():
    with pytest.raises(ValueError):
        encode_frame(REQUEST, 2**32, b"")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest Tests/Tools/test_remote_session_frames.py -q`
Expected: FAIL — `ModuleNotFoundError: remote_session_frames`.

- [ ] **Step 3: Implement**

```python
# tldw_chatbook/Tools/remote_session_frames.py
"""Binary frames for the SSH session worker (stdlib-only; bundled).

``u32 length | u32 request_id | u8 kind | body``. Shared by the host
``serve`` loop and the laptop ``RemoteSessionWorker``; both reject an
oversize frame from its header, before buffering its body.
"""

from __future__ import annotations

import json
import struct

HELLO, REQUEST, CANCEL = 1, 2, 3
LINE, STATUS, BUSY = 16, 17, 18
HEADER = struct.Struct(">IIB")
_U32_MAX = 2**32 - 1


class FrameError(ValueError):
    """A frame violated the size cap or the header format."""


def encode_frame(kind: int, request_id: int, body: bytes) -> bytes:
    """Serialize one frame.

    Args:
        kind: One of the kind constants.
        request_id: Per-session request id (0 for session-level frames).
        body: Frame payload.

    Returns:
        Header plus body.

    Raises:
        ValueError: If the id or body length does not fit a u32.
    """
    if not 0 <= request_id <= _U32_MAX or len(body) > _U32_MAX:
        raise ValueError("frame field out of range")
    return HEADER.pack(len(body), request_id, kind) + body


class FrameReader:
    """Incremental frame parser with a per-frame body cap."""

    def __init__(self, max_body: int) -> None:
        self._max = max_body
        self._buf = bytearray()

    def feed(self, data: bytes) -> list[tuple[int, int, bytes]]:
        """Consume bytes and return every complete frame.

        Raises:
            FrameError: When a header announces a body over the cap.
        """
        self._buf += data
        frames: list[tuple[int, int, bytes]] = []
        while len(self._buf) >= HEADER.size:
            length, request_id, kind = HEADER.unpack_from(self._buf)
            if length > self._max:
                raise FrameError(f"frame body {length} exceeds cap {self._max}")
            end = HEADER.size + length
            if len(self._buf) < end:
                break
            frames.append((kind, request_id, bytes(self._buf[HEADER.size:end])))
            del self._buf[:end]
        return frames


def encode_status(exit_code: int | None, signal_no: int | None) -> bytes:
    """Encode a child's termination for a STATUS frame."""
    return json.dumps({"exit": exit_code, "signal": signal_no}).encode()


def decode_status(body: bytes) -> tuple[int | None, int | None]:
    """Decode a STATUS body into ``(exit_code, signal_no)``."""
    payload = json.loads(body)
    return payload["exit"], payload["signal"]
```

- [ ] **Step 4: Run tests, verify pass.** Same command → 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Tools/remote_session_frames.py Tests/Tools/test_remote_session_frames.py
git commit -m "feat: SSH session frame codec (stdlib, bundled)"
```

---

### Task 3: Host `serve` fork server (stdlib)

**Files:**
- Create: `tldw_chatbook/Tools/remote_session_serve.py`
- Test: `Tests/Tools/test_remote_session_serve.py`

**Interfaces:**
- Consumes: Task 2 codec.
- Produces: `serve(in_fd: int, out_fd: int, *, run_request: Callable[[bytes, BinaryIO], int], max_request_bytes: int, max_response_bytes: int, clock: Callable[[], float] = time.monotonic) -> int`. The first inbound frame must be `HELLO` with JSON `{"max_children": int, "idle_s": float}`. `run_request(request_bytes, out)` runs **in the child**; its return value is the child's exit code.

- [ ] **Step 1: Write the failing tests** — drive `serve` in a real subprocess so fork semantics are real:

```python
# Tests/Tools/test_remote_session_serve.py
import json, os, subprocess, sys, textwrap, time
from pathlib import Path
from tldw_chatbook.Tools.remote_session_frames import (
    BUSY, CANCEL, HELLO, LINE, REQUEST, STATUS, FrameReader, decode_status, encode_frame,
)

REPO = Path(__file__).resolve().parents[2]
HANDLER = textwrap.dedent('''
    import os, sys, time
    sys.path.insert(0, %r)
    from tldw_chatbook.Tools.remote_session_serve import serve
    def run_request(raw, out):
        cmd = raw.decode()
        if cmd.startswith("sleep:"):
            time.sleep(float(cmd.split(":")[1]))
        if cmd == "flood":
            while True:
                out.write(b"x" * 65536 + b"\\n"); out.flush()
        if cmd == "fdprobe":
            try:
                os.fstat(3); leaked = "fd3"
            except OSError:
                leaked = "clean"
            out.write(leaked.encode() + b"\\n")
            return 0
        out.write(b"echo:" + raw + b"\\n"); out.flush()
        return 0
    sys.exit(serve(0, 1, run_request=run_request, max_request_bytes=1 << 20, max_response_bytes=1 << 16))
''' % str(REPO))

def _session(max_children=4, idle_s=30.0):
    proc = subprocess.Popen([sys.executable, "-c", HANDLER], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    os.write(proc.stdin.fileno(), encode_frame(HELLO, 0, json.dumps({"max_children": max_children, "idle_s": idle_s}).encode()))
    return proc, FrameReader(max_body=1 << 20)

def _collect(proc, reader, want_status, timeout=10.0):
    done, frames, end = set(), [], time.monotonic() + timeout
    while len(done) < want_status and time.monotonic() < end:
        for kind, rid, body in reader.feed(os.read(proc.stdout.fileno(), 65536)):
            frames.append((kind, rid, body))
            if kind == STATUS:
                done.add(rid)
    return frames

def test_concurrent_requests_complete_out_of_order_lines_before_status():
    proc, reader = _session()
    os.write(proc.stdin.fileno(), encode_frame(REQUEST, 1, b"sleep:0.5") + encode_frame(REQUEST, 2, b"fast"))
    frames = _collect(proc, reader, 2)
    order = [rid for kind, rid, _ in frames if kind == STATUS]
    assert order == [2, 1]
    for rid in (1, 2):
        kinds = [k for k, r, _ in frames if r == rid]
        assert kinds[-1] == STATUS and LINE in kinds and kinds.index(LINE) < kinds.index(STATUS)
    proc.stdin.close(); assert proc.wait(5) == 0

def test_cancel_kills_only_that_child():
    proc, reader = _session()
    os.write(proc.stdin.fileno(), encode_frame(REQUEST, 1, b"sleep:30") + encode_frame(REQUEST, 2, b"fast"))
    time.sleep(0.3)
    os.write(proc.stdin.fileno(), encode_frame(CANCEL, 1, b""))
    frames = _collect(proc, reader, 2)
    statuses = {rid: decode_status(body) for kind, rid, body in frames if kind == STATUS}
    assert statuses[1][1] == 9 and statuses[2] == (0, None)
    proc.stdin.close(); proc.wait(5)

def test_output_cap_kills_the_flooding_child_only():
    proc, reader = _session()
    os.write(proc.stdin.fileno(), encode_frame(REQUEST, 1, b"flood") + encode_frame(REQUEST, 2, b"fast"))
    frames = _collect(proc, reader, 2)
    statuses = {rid: decode_status(body) for kind, rid, body in frames if kind == STATUS}
    assert statuses[1][1] == 9 and statuses[2] == (0, None)
    assert sum(len(b) for k, r, b in frames if r == 1 and k == LINE) <= (1 << 16) + 65537
    proc.stdin.close(); proc.wait(5)

def test_child_cap_queues_and_reports_busy():
    proc, reader = _session(max_children=1)
    os.write(proc.stdin.fileno(), encode_frame(REQUEST, 1, b"sleep:0.3") + encode_frame(REQUEST, 2, b"fast"))
    frames = _collect(proc, reader, 2)
    assert (BUSY, 2, b"") in frames
    proc.stdin.close(); proc.wait(5)

def test_child_holds_no_channel_fds():
    proc, reader = _session()
    os.write(proc.stdin.fileno(), encode_frame(REQUEST, 1, b"fdprobe"))
    frames = _collect(proc, reader, 1)
    assert any(k == LINE and b"clean" in b for k, _, b in frames)
    proc.stdin.close(); proc.wait(5)

def test_eof_kills_live_children_and_exits():
    proc, reader = _session()
    os.write(proc.stdin.fileno(), encode_frame(REQUEST, 1, b"sleep:30"))
    time.sleep(0.3); proc.stdin.close()
    assert proc.wait(5) == 0

def test_idle_exit_ignores_a_long_running_child():
    proc, reader = _session(idle_s=0.5)
    os.write(proc.stdin.fileno(), encode_frame(REQUEST, 1, b"sleep:1.5"))
    frames = _collect(proc, reader, 1, timeout=5)
    assert any(k == STATUS for k, _, _ in frames), "idle timer fired under a live child"
    assert proc.wait(5) == 0  # then idles out

def test_no_zombies_after_many_requests():
    proc, reader = _session()
    for i in range(1, 51):
        os.write(proc.stdin.fileno(), encode_frame(REQUEST, i, b"fast"))
    _collect(proc, reader, 50)
    out = subprocess.run(["ps", "-o", "stat=", "--ppid", str(proc.pid)] if sys.platform.startswith("linux")
                         else ["ps", "-o", "stat=", "-g", str(proc.pid)], capture_output=True, text=True).stdout
    assert "Z" not in out
    proc.stdin.close(); proc.wait(5)
```

`fdprobe` checks fd 3: the child's own pipe is `dup2`'d onto fd 1 and every descriptor above 2 is closed, so any open fd ≥ 3 in a child is a leak (the parent's stdin/stdout copies or another child's pipe).

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest Tests/Tools/test_remote_session_serve.py -q` → FAIL, module missing.

- [ ] **Step 3: Implement**

```python
# tldw_chatbook/Tools/remote_session_serve.py
"""Fork-server parent for the SSH session worker (stdlib-only; bundled).

One parent per session: reads frames from ``in_fd``, forks a child per
REQUEST (the child runs ``run_request`` with stdin=/dev/null and stdout
= a pipe back here), relays the child's lines as LINE frames, reaps it,
then sends STATUS. Single-threaded by design: forking a threaded
process is unsafe, and the per-request watchdog lives in the child.
"""

from __future__ import annotations

import json
import os
import selectors
import signal
import time
from collections import deque
from typing import BinaryIO, Callable

from tldw_chatbook.Tools.remote_session_frames import (
    BUSY, CANCEL, HELLO, LINE, REQUEST, STATUS, FrameError, FrameReader,
    encode_frame, encode_status,
)

_CHILD_CRASH_EXIT = 70


class _Child:
    __slots__ = ("pid", "fd", "request_id", "partial", "sent", "capped")

    def __init__(self, pid: int, fd: int, request_id: int) -> None:
        self.pid, self.fd, self.request_id = pid, fd, request_id
        self.partial = bytearray()
        self.sent = 0
        self.capped = False


def _close_fds_above_2(keep: int) -> None:
    try:
        max_fd = os.sysconf("SC_OPEN_MAX")
    except (AttributeError, ValueError, OSError):
        max_fd = 1024
    os.closerange(3, keep)
    os.closerange(keep + 1, max(max_fd, keep + 2))


def _spawn(raw: bytes, request_id: int, run_request: Callable[[bytes, BinaryIO], int]) -> _Child:
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        code = _CHILD_CRASH_EXIT
        try:
            devnull = os.open(os.devnull, os.O_RDONLY)
            os.dup2(devnull, 0)
            os.dup2(write_fd, 1)
            _close_fds_above_2(keep=2)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            with os.fdopen(1, "wb", closefd=False) as out:
                code = run_request(raw, out)
                out.flush()
        except BaseException:  # noqa: BLE001 - a child must never return into the parent loop
            code = _CHILD_CRASH_EXIT
        os._exit(code if isinstance(code, int) else _CHILD_CRASH_EXIT)
    os.close(write_fd)
    os.set_blocking(read_fd, False)
    return _Child(pid, read_fd, request_id)


def serve(
    in_fd: int,
    out_fd: int,
    *,
    run_request: Callable[[bytes, BinaryIO], int],
    max_request_bytes: int,
    max_response_bytes: int,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Run the session loop until stdin EOF or idle; return an exit code."""
    reader = FrameReader(max_body=max_request_bytes)
    selector = selectors.DefaultSelector()
    selector.register(in_fd, selectors.EVENT_READ, None)
    children: dict[int, _Child] = {}          # fd -> child
    by_request: dict[int, _Child] = {}
    queue: deque[tuple[int, bytes]] = deque()
    max_children, idle_s, hello = 8, 60.0, False
    last_activity = clock()
    outbox = bytearray()

    def flush() -> None:
        view = memoryview(outbox)
        while view:
            written = os.write(out_fd, view)
            view = view[written:]
        outbox.clear()

    def start(request_id: int, raw: bytes) -> None:
        child = _spawn(raw, request_id, run_request)
        children[child.fd] = child
        by_request[request_id] = child
        selector.register(child.fd, selectors.EVENT_READ, child)

    def finish(child: _Child) -> None:
        selector.unregister(child.fd)
        os.close(child.fd)
        if child.partial and not child.capped:
            outbox.extend(encode_frame(LINE, child.request_id, bytes(child.partial)))
        _, status = os.waitpid(child.pid, 0)
        exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else None
        signal_no = os.WTERMSIG(status) if os.WIFSIGNALED(status) else None
        outbox.extend(encode_frame(STATUS, child.request_id, encode_status(exit_code, signal_no)))
        del children[child.fd]
        by_request.pop(child.request_id, None)
        while queue and len(children) < max_children:
            start(*queue.popleft())

    def kill_all() -> None:
        for child in list(children.values()):
            try:
                os.kill(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            selector.unregister(child.fd)
            os.close(child.fd)
            os.waitpid(child.pid, 0)
        children.clear()

    try:
        while True:
            busy = bool(children or queue)
            timeout = None if busy else max(0.0, idle_s - (clock() - last_activity))
            events = selector.select(timeout)
            if not events and not busy and clock() - last_activity >= idle_s:
                return 0
            for key, _mask in events:
                if key.data is None:  # stdin
                    data = os.read(in_fd, 65536)
                    if not data:
                        return 0
                    last_activity = clock()
                    for kind, request_id, body in reader.feed(data):
                        if kind == HELLO:
                            limits = json.loads(body)
                            max_children = max(1, int(limits["max_children"]))
                            idle_s = max(1.0, float(limits["idle_s"]))
                            hello = True
                        elif kind == REQUEST and hello:
                            if len(children) < max_children:
                                start(request_id, body)
                            else:
                                queue.append((request_id, body))
                                outbox.extend(encode_frame(BUSY, request_id, b""))
                        elif kind == CANCEL:
                            child = by_request.get(request_id)
                            if child is not None:
                                try:
                                    os.kill(child.pid, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass
                            remaining = [item for item in queue if item[0] != request_id]
                            queue.clear()
                            queue.extend(remaining)
                else:
                    child: _Child = key.data
                    try:
                        chunk = os.read(child.fd, 65536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        finish(child)
                        last_activity = clock()
                        continue
                    if child.capped:
                        continue
                    child.sent += len(chunk)
                    if child.sent > max_response_bytes:
                        child.capped = True
                        os.kill(child.pid, signal.SIGKILL)
                        continue
                    child.partial += chunk
                    while b"\n" in child.partial:
                        line, _, rest = bytes(child.partial).partition(b"\n")
                        outbox.extend(encode_frame(LINE, child.request_id, line + b"\n"))
                        child.partial = bytearray(rest)
            if outbox:
                flush()
    except FrameError:
        return 3
    finally:
        kill_all()
        selector.close()
```

`queue` is mutated in place (never rebound) because `finish()` closes over it.

- [ ] **Step 4: Run tests, verify pass.** Linux and macOS both (CI covers macOS).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Tools/remote_session_serve.py Tests/Tools/test_remote_session_serve.py
git commit -m "feat: host fork-server session loop (stdlib, bundled)"
```

---

### Task 4: Stage-1 loader + host cache (generated by the builder)

**Files:**
- Modify: `tldw_chatbook/Tools/build_remote_worker_bundle.py` (add `_LOADER_SOURCE`, `build_loader_text() -> str`, `loader_payload() -> bytes` returning zlib bytes)
- Test: `Tests/Tools/test_remote_session_loader.py`

**Interfaces:**
- Produces: `build_loader_text() -> str`; `loader_payload() -> bytes` (zlib-compressed loader); the loader's stdin/stdout contract from Global Constraints. The loader calls `mod.serve_session(sys.stdin.buffer, sys.stdout.buffer)` (Task 5 defines it) after printing `READY <BUNDLE_SHA256>`.

- [ ] **Step 1: Write the failing tests** — run the loader as the bootstrap would, with a stub bundle that defines `BUNDLE_SHA256` and a `serve_session` that just exits 0:

```python
# Tests/Tools/test_remote_session_loader.py
import hashlib, json, os, stat, subprocess, sys, zlib
from pathlib import Path
import pytest
from tldw_chatbook.Tools.build_remote_worker_bundle import build_loader_text, loader_payload
from tldw_chatbook.Tools.remote_workspace_executor import bootstrap_source

MAGIC = b"TLDW-REMOTE-0001"
STUB = b'BUNDLE_SHA256 = "stub-stamp"\nimport sys\ndef serve_session(i, o):\n    assert __name__ != "__main__"\n    return 0\n'
STUB_Z = zlib.compress(STUB)
STUB_HASH = hashlib.sha256(STUB_Z).hexdigest()

def _run(env_runtime, cache=True, hash_=STUB_HASH, send_bundle=True):
    env = dict(os.environ)
    env.pop("XDG_RUNTIME_DIR", None)
    if env_runtime is not None:
        env["XDG_RUNTIME_DIR"] = str(env_runtime)
    loader = loader_payload()
    proc = subprocess.Popen([sys.executable, "-I", "-c", bootstrap_source(len(loader))],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
    proc.stdin.write(loader + json.dumps({"hash": hash_, "cache": cache}).encode() + b"\n"); proc.stdin.flush()
    first = proc.stdout.readline()
    if first.startswith(MAGIC + b"NEED") and send_bundle:
        proc.stdin.write(len(STUB_Z).to_bytes(4, "big") + STUB_Z); proc.stdin.flush()
        second = proc.stdout.readline()
    else:
        second = b""
    proc.stdin.close(); code = proc.wait(10)
    return first, second, code

def _runtime(tmp_path):
    base = tmp_path / "run"; base.mkdir(mode=0o700); return base

def test_miss_then_hit(tmp_path):
    base = _runtime(tmp_path)
    first, second, code = _run(base)
    assert first == MAGIC + b"NEED " + STUB_HASH.encode() + b"\n"
    assert second == MAGIC + b"READY stub-stamp\n" and code == 0
    cached = base / "tldw-worker" / STUB_HASH
    assert stat.S_IMODE(cached.stat().st_mode) == 0o600
    first, _, code = _run(base)
    assert first == MAGIC + b"READY stub-stamp\n" and code == 0

def test_tampered_or_loose_cache_is_a_miss(tmp_path):
    base = _runtime(tmp_path); _run(base)
    cached = base / "tldw-worker" / STUB_HASH
    cached.write_bytes(b"tampered")
    assert _run(base)[0].startswith(MAGIC + b"NEED")
    cached.chmod(0o644)
    assert _run(base)[0].startswith(MAGIC + b"NEED")

def test_no_runtime_dir_or_not_private_means_no_cache(tmp_path):
    first, second, code = _run(None)
    assert first.startswith(MAGIC + b"NEED") and code == 0
    loose = tmp_path / "loose"; loose.mkdir(mode=0o755)
    _run(loose)
    assert not (loose / "tldw-worker").exists()

def test_cache_disabled_by_header(tmp_path):
    base = _runtime(tmp_path)
    _run(base, cache=False)
    assert not (base / "tldw-worker").exists()

def test_other_hashes_are_cleaned_up(tmp_path):
    base = _runtime(tmp_path)
    (base / "tldw-worker").mkdir(mode=0o700)
    stale = base / "tldw-worker" / ("0" * 64); stale.write_bytes(b"old"); stale.chmod(0o600)
    _run(base)
    assert not stale.exists()

def test_bundle_hash_mismatch_refused(tmp_path):
    first, second, code = _run(_runtime(tmp_path), hash_="f" * 64)
    assert second == b"" and code != 0

def test_loader_is_stdlib_only_and_310_compatible():
    import ast
    tree = ast.parse(build_loader_text())
    roots = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert roots <= set(sys.stdlib_module_names)
```

- [ ] **Step 2: Run to verify failure** → ImportError on `build_loader_text`.

- [ ] **Step 3: Implement** — add to `build_remote_worker_bundle.py`:

```python
_LOADER_SOURCE = r'''
import hashlib, json, os, stat, sys, tempfile, types, zlib
_in = sys.stdin.buffer
_out = sys.stdout.buffer
_MAGIC = b"TLDW-REMOTE-0001"
_MAX_BUNDLE = 8 << 20
_header = json.loads(_in.readline(4096))
_hash = str(_header["hash"])
if len(_hash) != 64 or any(c not in "0123456789abcdef" for c in _hash):
    sys.exit(3)

def _private_dir(path):
    try:
        info = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid() and not info.st_mode & 0o077

def _cache_dir():
    base = os.environ.get("XDG_RUNTIME_DIR")
    if not _header.get("cache") or not base or not _private_dir(base):
        return None
    path = os.path.join(base, "tldw-worker")
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    except OSError:
        return None
    return path if _private_dir(path) else None

def _read_cached(directory):
    try:
        fd = os.open(os.path.join(directory, _hash), os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return None
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            return None
        data = handle.read(_MAX_BUNDLE + 1)
    return data if hashlib.sha256(data).hexdigest() == _hash else None

def _store(directory, data):
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.chmod(tmp, 0o600)
        os.replace(tmp, os.path.join(directory, _hash))
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return
    for name in os.listdir(directory):
        if name != _hash and not name.startswith(".tmp-"):
            try:
                os.unlink(os.path.join(directory, name))
            except OSError:
                pass

_dir = _cache_dir()
_data = _read_cached(_dir) if _dir else None
if _data is None:
    _out.write(_MAGIC + b"NEED " + _hash.encode() + b"\n"); _out.flush()
    _size = int.from_bytes(_in.read(4), "big")
    if _size > _MAX_BUNDLE:
        sys.exit(3)
    _data = _in.read(_size)
    if hashlib.sha256(_data).hexdigest() != _hash:
        sys.exit(3)
    if _dir:
        _store(_dir, _data)
_module = types.ModuleType("tldw_remote_worker")
sys.modules[_module.__name__] = _module
exec(compile(zlib.decompress(_data), "tldw_remote_worker", "exec"), _module.__dict__)
_out.write(_MAGIC + b"READY " + _module.BUNDLE_SHA256.encode() + b"\n"); _out.flush()
sys.exit(_module.serve_session(_in, _out))
'''


def build_loader_text() -> str:
    """Return the stage-1 loader source (stdlib-only, Python >= 3.10)."""
    return _LOADER_SOURCE.lstrip("\n")


def loader_payload() -> bytes:
    """Return the zlib-compressed loader the bootstrap decompresses."""
    return zlib.compress(build_loader_text().encode("utf-8"), 9)
```

(`zlib` import at the top of the builder if not already present.)

- [ ] **Step 4: Run tests, verify pass.**

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Tools/build_remote_worker_bundle.py Tests/Tools/test_remote_session_loader.py
git commit -m "feat: stage-1 loader with sha256-verified runtime-dir bundle cache"
```

---

### Task 5: Bundle integration — `serve_session` entry, loopback harness, 3.10 CI

**Files:**
- Modify: `tldw_chatbook/Tools/build_remote_worker_bundle.py` — include `remote_session_frames` and `remote_session_serve` in the bundle (extend the closure walk in `_worker_closure_modules()` to start from both `_WORKER_MODULE` and `"tldw_chatbook.Tools.remote_session_serve"`, and add both to `BUNDLE_MODULES` in dependency order before the worker); add `serve_session` to `_IO_ADAPTER_SOURCE` **above** `_enter_worker_exchange` (inside the stamped region).
- Modify: `tldw_chatbook/Tools/remote_worker_bundle.py` (regenerated, never hand-edited)
- Modify: `tldw_chatbook/Tools/remote_workspace_executor.py` — add `run_session_loopback(requests, *, spawn_env=None, cache=False)` harness
- Modify: `.github/workflows/test.yml` — the existing 3.10 floor job also runs `Tests/Tools/test_remote_session_loopback.py`
- Test: `Tests/Tools/test_remote_session_loopback.py`

**Interfaces:**
- Consumes: Tasks 2–4; `run_workspace_worker(stdin, stdout, stderr, *, bundle_sha256)` from the worker section.
- Produces: bundle function `serve_session(in_stream, out_stream) -> int`; `run_session_loopback(root: Path, requests: list[dict], *, cache_dir: Path | None = None) -> dict[int, list[dict]]` mapping request index → parsed LINE dicts plus `{"status": (exit, signal)}` as the last element.

- [ ] **Step 1: Add `serve_session` to `_IO_ADAPTER_SOURCE`** (above `_enter_worker_exchange`):

```python
def serve_session(in_stream: Any, out_stream: Any) -> int:
    """Session entry: HELLO then frames until EOF/idle (loader calls this).

    The loader has consumed exactly its own bytes from ``in_stream``
    (lockstep contract), so the buffered reader is empty and the loop can
    read fd 0 directly.
    """
    import io as _io

    out_stream.flush()

    def _run_request(raw: bytes, out: Any) -> int:
        # BUNDLE_SHA256 is bound by the stamp line, which runs after this
        # function is defined but before the loader calls serve_session.
        return run_workspace_worker(
            _io.BytesIO(raw), out, _io.BytesIO(), bundle_sha256=globals()["BUNDLE_SHA256"]
        )

    return serve(
        0,
        1,
        run_request=_run_request,
        max_request_bytes=MAX_REQUEST_BYTES,
        max_response_bytes=_SESSION_MAX_RESPONSE_BYTES,
    )
```

Define `_SESSION_MAX_RESPONSE_BYTES` in the adapter as the same literal as the parent's `MAX_RESPONSE_BYTES` (the bundle cannot import the pydantic-side protocol module); add a builder assertion that the two values are equal at build time.

- [ ] **Step 2: Regenerate and check the drift guard**

```bash
.venv/bin/python -m tldw_chatbook.Tools.build_remote_worker_bundle
.venv/bin/python -m tldw_chatbook.Tools.build_remote_worker_bundle --check
.venv/bin/python -m pytest Tests/Tools/test_remote_worker_bundle.py -q
```

Expected: `--check` ok; bundle tests pass (stdlib-only, 3.10 parse, stub-raise allowlist unchanged).

- [ ] **Step 3: Write the loopback integration tests** using the real bundle through the real bootstrap + loader (no ssh):

```python
# Tests/Tools/test_remote_session_loopback.py
import time
from pathlib import Path
import pytest
from tldw_chatbook.Tools.remote_workspace_executor import run_session_loopback

def _ws(tmp_path):
    root = tmp_path / "ws"; root.mkdir()
    (root / "a.txt").write_text("alpha\n"); (root / "big.txt").write_text("a" * 60000 + "b\n")
    return root

def test_ping_then_reads_share_one_session(tmp_path):
    root = _ws(tmp_path)
    out = run_session_loopback(root, [{"op": "ping"}, {"op": "fs_read", "path": "a.txt"}, {"op": "fs_list", "path": "."}])
    assert out[1][-2]["outcome"] == "success" and "alpha" in out[1][-2]["result"]
    assert all(frames[-1]["status"] == (0, None) for frames in out.values())

def test_catastrophic_regex_dies_alone(tmp_path):
    root = _ws(tmp_path)
    out = run_session_loopback(root, [
        {"op": "fs_grep", "pattern": "(a+)+$", "budget": 2},
        {"op": "fs_read", "path": "a.txt"},
    ])
    exit_code, signal_no = out[0][-1]["status"]
    assert exit_code == 75 or signal_no is not None
    assert out[1][-2]["outcome"] == "success"

def test_pin_failure_leaves_session_up(tmp_path):
    root = _ws(tmp_path)
    out = run_session_loopback(root, [{"op": "fs_read", "path": "a.txt", "stale_identity": True}, {"op": "fs_read", "path": "a.txt"}])
    assert out[0][-2]["code"] == "root_pin_failed"
    assert out[1][-2]["outcome"] == "success"

def test_cache_hit_on_second_session(tmp_path):
    root = _ws(tmp_path); cache = tmp_path / "run"; cache.mkdir(mode=0o700)
    run_session_loopback(root, [{"op": "ping"}], cache_dir=cache)
    assert any((cache / "tldw-worker").iterdir())
    run_session_loopback(root, [{"op": "ping"}], cache_dir=cache)  # must not ask for the bundle again (harness asserts READY first)
```

- [ ] **Step 4: Implement `run_session_loopback`** in `remote_workspace_executor.py`, next to `run_bundle_loopback`, reusing its request builder (`_request`-style helpers already used by `Tests/Tools/test_remote_executor_loopback.py` — move the pinned-chain request construction into the harness so tests pass plain op dicts). It spawns `sys.executable -I -c bootstrap_source(len(loader_payload()))` with `XDG_RUNTIME_DIR` set to `cache_dir` (or removed), writes loader + header, answers `NEED` with the compressed bundle, asserts `READY <expected stamp>`, sends `HELLO {"max_children": 4, "idle_s": 30}`, sends all REQUEST frames, and collects LINE/STATUS until every id has a STATUS. When `cache_dir` already holds the bundle, it asserts the first loader line is `READY` (no `NEED`).

- [ ] **Step 5: Wire the 3.10 CI run** — in `.github/workflows/test.yml`, in the job that already compiles the bundle under `uv python install 3.10`, add:

```yaml
      - name: Session loopback under the 3.10 floor
        run: uv run --python 3.10 --with pytest python -m pytest Tests/Tools/test_remote_session_loopback.py -q
```

(Match the job's existing invocation style; the point is that the loader/serve code *executes* under 3.10.)

- [ ] **Step 6: Run locally, commit**

```bash
.venv/bin/python -m pytest Tests/Tools/test_remote_session_loopback.py Tests/Tools/test_remote_worker_bundle.py Tests/Tools/test_remote_executor_loopback.py -q
git add -A tldw_chatbook/Tools Tests/Tools .github/workflows/test.yml
git commit -m "feat: bundle serve_session entry, session loopback harness, 3.10 CI run"
```

---

### Task 6: Laptop `RemoteSessionWorker`

**Files:**
- Create: `tldw_chatbook/Tools/remote_session_worker.py`
- Modify: `tldw_chatbook/Tools/remote_workspace_transport.py` — add public `classify_exchange_failure(...)` delegating to `_classify_failure` (same keyword args) so the session reuses the taxonomy verbatim
- Test: `Tests/Tools/test_remote_session_worker.py`

**Interfaces:**
- Consumes: Task 2 codec; Task 4 `loader_payload()`; `_bundle_payload()` (compressed bundle + stamp); `RemoteWorkspaceTransport.classify_exchange_failure`; `SshMasterManager.client_options/ensure_master/ssh_bin`; `build_ssh_argv`, `_remote_command(python, bootstrap)`.
- Produces:
  - `class SessionStartError(Exception)`: attributes `transport: bool`, `failure: TransportFailure | None`
  - `class RemoteSessionWorker(loc, *, transport: RemoteWorkspaceTransport, python: str, max_children: int, idle_s: float, cache: bool, spawn: Callable[[list[str]], subprocess.Popen[bytes]] | None = None)`
  - methods `start() -> None` (raises `SessionStartError`), `call(request_bytes: bytes, *, budget: float) -> RemoteCallResult`, `close() -> None`; properties `alive: bool`, `idle_since: float | None` (monotonic; `None` while requests are in flight)

- [ ] **Step 1: Failing tests** — use a `spawn` seam that runs the bootstrap locally (`[sys.executable, "-I", "-c", bootstrap]`) instead of ssh, so the real loader/serve/bundle run:

```python
# Tests/Tools/test_remote_session_worker.py (key cases)
def test_many_calls_one_spawn(worker_factory, workspace):
    worker, spawns = worker_factory()
    worker.start()
    for _ in range(20):
        result = worker.call(read_request(workspace, "a.txt"), budget=10)
        assert result.failure is None and result.admitted
    assert len(spawns) == 1

def test_concurrent_callers(worker_factory, workspace):
    worker, _ = worker_factory(); worker.start()
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: worker.call(read_request(workspace, "a.txt"), budget=10), range(32)))
    assert all(r.failure is None for r in results)

def test_watchdog_timeout_maps_to_op_timeout(worker_factory, workspace):
    worker, _ = worker_factory(); worker.start()
    result = worker.call(grep_request(workspace, "(a+)+$"), budget=2)
    assert result.admitted and result.failure.kind is TransportFailureKind.OP_TIMEOUT

def test_stamp_mismatch_is_protocol_start_error(worker_factory, monkeypatch):
    worker, _ = worker_factory(expected_stamp="0" * 64)
    with pytest.raises(SessionStartError) as err:
        worker.start()
    assert err.value.transport is False

def test_unreachable_is_transport_start_error(worker_factory):
    worker, _ = worker_factory(spawn_argv=["sh", "-c", "exit 255"])
    with pytest.raises(SessionStartError) as err:
        worker.start()
    assert err.value.transport is True and err.value.failure.kind is TransportFailureKind.UNREACHABLE

def test_lockstep_sends_nothing_before_need_or_ready(worker_factory):
    worker, spawns = worker_factory(record_stdin=True); worker.start()
    wire = spawns[0].stdin_log
    loader = loader_payload()
    header_end = wire.index(b"\n", len(loader)) + 1
    # after the header the next bytes are either the u32 bundle length (NEED path) or HELLO (READY path)
    assert wire[header_end:header_end + 9] in (HELLO_HEADER_PREFIX_BYTES(), ) or wire[header_end:header_end + 4] == len(_bundle_payload()[1]).to_bytes(4, "big")

def test_hung_parent_kills_session(worker_factory, workspace):
    worker, _ = worker_factory(freeze_parent_after_hello=True); worker.start()
    result = worker.call(read_request(workspace, "a.txt"), budget=1)
    assert result.failure is not None and not worker.alive

def test_session_death_fails_inflight_and_marks_dead(worker_factory, workspace):
    worker, spawns = worker_factory(); worker.start()
    fut = ThreadPoolExecutor(1).submit(worker.call, grep_request(workspace, "(a+)+$"), budget=30)
    time.sleep(0.3); spawns[0].kill()
    result = fut.result(10)
    assert result.failure is not None and not worker.alive
```

`worker_factory` is a fixture in the test file building `RemoteSessionWorker(loc, transport=RemoteWorkspaceTransport(SshMasterManager(enabled=False)), python=sys.executable, max_children=4, idle_s=30, cache=False, spawn=<local seam>)`; `expected_stamp` overrides via monkeypatching `_bundle_payload`; `freeze_parent_after_hello` wraps the spawned process in a shim that stops relaying stdout after HELLO (`kill -STOP` the child after the first frame). `HELLO_HEADER_PREFIX_BYTES()` returns `HEADER.pack(len(hello_body), 0, HELLO)`. Write these helpers explicitly in the test file.

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement** (full module):

```python
# tldw_chatbook/Tools/remote_session_worker.py
"""Laptop side of the per-run SSH session worker (spec 2026-09-27)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from tldw_chatbook.Tools.remote_session_frames import (
    CANCEL, HELLO, LINE, REQUEST, STATUS, FrameError, FrameReader,
    decode_status, encode_frame,
)
from tldw_chatbook.Tools.remote_workspace_transport import (
    RemoteCallResult, RemoteWorkspaceTransport, TransportFailure,
    TransportFailureKind, _frame_is_admitted_marker, _remote_command, build_ssh_argv,
)
from tldw_chatbook.Tools.workspace_tool_protocol import MAX_RESPONSE_BYTES
from tldw_chatbook.Tools.build_remote_worker_bundle import loader_payload
from tldw_chatbook.Tools.remote_worker_bundle import RESPONSE_MAGIC

_NOISE_CAP = 4096
_STUCK_GRACE_S = 5.0


class SessionStartError(Exception):
    """The session could not start; ``transport`` says which failure class."""

    def __init__(self, transport: bool, failure: TransportFailure | None, reason: str) -> None:
        super().__init__(reason)
        self.transport = transport
        self.failure = failure


@dataclass
class _Pending:
    done: threading.Event = field(default_factory=threading.Event)
    lines: list[bytes] = field(default_factory=list)
    admitted_at: float | None = None
    status: tuple[int | None, int | None] | None = None
    dead_exit: int | None = None


class RemoteSessionWorker:
    """One fork-server session for one binding in one Console run."""

    def __init__(self, loc, *, transport: RemoteWorkspaceTransport, python: str,
                 max_children: int, idle_s: float, cache: bool,
                 spawn: Callable[[list[str]], subprocess.Popen[bytes]] | None = None) -> None:
        self._loc, self._transport, self._python = loc, transport, python
        self._max_children, self._idle_s, self._cache = max_children, idle_s, cache
        self._spawn = spawn or (lambda argv: subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True))
        self._proc: subprocess.Popen[bytes] | None = None
        self._write_lock = threading.Lock()
        self._pending: dict[int, _Pending] = {}
        self._pending_lock = threading.Lock()
        self._next_id = 1
        self._alive = False
        self._idle_since: float | None = None

    @property
    def alive(self) -> bool:
        return self._alive

    @property
    def idle_since(self) -> float | None:
        return self._idle_since

    # -- start -------------------------------------------------------------

    def start(self) -> None:
        from tldw_chatbook.Tools.remote_workspace_executor import _bundle_payload, bootstrap_source
        from tldw_chatbook.Tools.build_remote_worker_bundle import expected_bundle_stamp

        artifact, compressed, _ = _bundle_payload()
        loader = loader_payload()
        manager = self._transport.master_manager
        manager.ensure_master(self._loc)
        argv = [manager.ssh_bin, *manager.client_options(self._loc),
                *build_ssh_argv(self._loc, [], _remote_command(self._python, bootstrap_source(len(loader))))]
        try:
            proc = self._spawn(argv)
        except OSError as exc:
            raise SessionStartError(True, TransportFailure(TransportFailureKind.UNREACHABLE, None, f"ssh could not be run: {exc}"), "spawn") from None
        self._proc = proc
        header = json.dumps({"hash": hashlib.sha256(compressed).hexdigest(), "cache": self._cache}).encode() + b"\n"
        self._write(loader + header)
        line = self._read_handshake_line(proc)
        if line.startswith(b"NEED "):
            self._write(len(compressed).to_bytes(4, "big") + compressed)
            line = self._read_handshake_line(proc)
        expected = b"READY " + expected_bundle_stamp(artifact).encode()
        if line != expected:
            self._kill()
            raise SessionStartError(False, None, "handshake: unexpected READY/stamp")
        self._write(encode_frame(HELLO, 0, json.dumps({"max_children": self._max_children, "idle_s": self._idle_s}).encode()))
        self._alive = True
        self._idle_since = time.monotonic()
        threading.Thread(target=self._reader, name="ssh-session-reader", daemon=True).start()

    def _read_handshake_line(self, proc) -> bytes:
        garbage = bytearray()
        buf = bytearray()
        while True:
            chunk = os.read(proc.stdout.fileno(), 1)  # byte-wise: frames must stay unread
            if not chunk:
                code = proc.wait()
                stderr = proc.stderr.read() if proc.stderr else b""
                failure = self._transport.classify_exchange_failure(
                    self._loc, exit_code=code, admitted=False, admitted_at=None, budget=1.0,
                    killed=False, noise_capped=False, stderr=stderr)
                transport_class = failure.kind in {
                    TransportFailureKind.UNREACHABLE, TransportFailureKind.INTERPRETER_MISSING,
                    TransportFailureKind.PYTHON_TOO_OLD, TransportFailureKind.STDOUT_NOISE,
                    TransportFailureKind.DESTINATION_CHANGED}
                raise SessionStartError(transport_class, failure, "session exited during handshake")
            buf += chunk
            if not buf.startswith(RESPONSE_MAGIC[: len(buf)]) and b"\n" not in buf:
                garbage += buf[:1]; del buf[:1]
                if len(garbage) > _NOISE_CAP:
                    self._kill()
                    raise SessionStartError(True, TransportFailure(TransportFailureKind.STDOUT_NOISE, None, "remote shell emits stdout noise"), "noise")
            if buf.endswith(b"\n"):
                if buf.startswith(RESPONSE_MAGIC):
                    return bytes(buf[len(RESPONSE_MAGIC):-1])
                garbage += buf; buf.clear()
```

(Byte-wise reads during the handshake only — two short lines per session; after `READY` the reader thread uses 64 KiB reads.)

```python
    # -- steady state ------------------------------------------------------

    def _write(self, data: bytes) -> None:
        with self._write_lock:
            view = memoryview(data)
            while view:
                view = view[os.write(self._proc.stdin.fileno(), view):]

    def _reader(self) -> None:
        reader = FrameReader(max_body=MAX_RESPONSE_BYTES)
        try:
            while True:
                data = os.read(self._proc.stdout.fileno(), 65536)
                if not data:
                    break
                for kind, request_id, body in reader.feed(data):
                    with self._pending_lock:
                        pending = self._pending.get(request_id)
                    if pending is None:
                        continue  # abandoned request: drop
                    if kind == LINE:
                        if pending.admitted_at is None and _frame_is_admitted_marker(body.rstrip(b"\n")):
                            pending.admitted_at = time.monotonic()
                        else:
                            pending.lines.append(body.rstrip(b"\n"))
                    elif kind == STATUS:
                        pending.status = decode_status(body)
                        pending.done.set()
        except (OSError, FrameError):
            pass
        self._die()

    def _die(self) -> None:
        self._alive = False
        exit_code = self._proc.poll() if self._proc else None
        if exit_code is None:
            self._kill()
            exit_code = self._proc.wait() if self._proc else 255
        with self._pending_lock:
            waiters = list(self._pending.values())
        for pending in waiters:
            pending.dead_exit = exit_code
            pending.done.set()

    def call(self, request_bytes: bytes, *, budget: float) -> RemoteCallResult:
        if not self._alive:
            return RemoteCallResult(False, None, TransportFailure(TransportFailureKind.REMOTE_OP_FAILED, None, "session not running"))
        with self._pending_lock:
            request_id = self._next_id; self._next_id += 1
            pending = self._pending[request_id] = _Pending()
            self._idle_since = None
        started = time.monotonic()
        grace = self._transport.grace_seconds
        try:
            self._write(encode_frame(REQUEST, request_id, request_bytes))
            while not pending.done.is_set():
                anchor = pending.admitted_at or started
                remaining = anchor + budget + grace - time.monotonic()
                if remaining <= 0:
                    self._write(encode_frame(CANCEL, request_id, b""))
                    if not pending.done.wait(_STUCK_GRACE_S):
                        self._kill(); self._die()
                    return self._result(pending, budget, killed=True)
                pending.done.wait(min(remaining, 0.5))
            return self._result(pending, budget, killed=False)
        except OSError:
            self._die()
            return self._result(pending, budget, killed=False)
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)
                if not self._pending:
                    self._idle_since = time.monotonic()

    def _result(self, pending: _Pending, budget: float, *, killed: bool) -> RemoteCallResult:
        admitted = pending.admitted_at is not None
        if pending.lines and not killed and pending.dead_exit is None:
            return RemoteCallResult(admitted, pending.lines[-1], None)
        if pending.status is not None:
            exit_code, signal_no = pending.status
            code = exit_code if exit_code is not None else -(signal_no or 9)
        else:
            code = pending.dead_exit if pending.dead_exit is not None else 255
        failure = self._transport.classify_exchange_failure(
            self._loc, exit_code=code, admitted=admitted, admitted_at=pending.admitted_at,
            budget=budget, killed=killed, noise_capped=False, stderr=b"")
        return RemoteCallResult(admitted, None, failure)

    def _kill(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.kill()
            except OSError:
                pass

    def close(self) -> None:
        self._alive = False
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._kill()
```

Also add to `RemoteWorkspaceTransport`:

```python
    def classify_exchange_failure(self, loc, **kwargs) -> TransportFailure:
        """Public alias of the taxonomy for the session worker (same keywords)."""
        return self._classify_failure(loc, **kwargs)
```

- [ ] **Step 4: Run tests; fix until green.** Also re-run `Tests/Tools/test_remote_transport_call.py` (the public alias must not change behaviour).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Tools/remote_session_worker.py tldw_chatbook/Tools/remote_workspace_transport.py Tests/Tools/test_remote_session_worker.py
git commit -m "feat: laptop RemoteSessionWorker (handshake, framed calls, same RemoteCallResult)"
```

---

### Task 7: Session registry, executor routing, failure policy, config

**Files:**
- Create: `tldw_chatbook/Tools/remote_session_registry.py`
- Modify: `tldw_chatbook/Tools/remote_workspace_executor.py` (`for_ssh(..., session_key: str | None = None)`, `_SshModeConfig.session_key`, `_ssh_call` routing)
- Modify: `tldw_chatbook/config.py` (`ConsoleSshSettings.session_worker/session_idle_s/bundle_cache` + coercion in `get_console_ssh_settings`)
- Test: `Tests/Tools/test_remote_session_registry.py`, extend `Tests/Tools/test_remote_executor_ssh.py`, extend `Tests/test_config_console_ssh_defaults.py`

**Interfaces:**
- Consumes: Task 6 `RemoteSessionWorker`, `SessionStartError`.
- Produces:
  - `class RemoteSessionRegistry` with `acquire(key: tuple[str, str], create: Callable[[], RemoteSessionWorker]) -> RemoteSessionWorker | None` (returns `None` when disabled for this key; raises `SessionStartError` from `create().start()` for transport-class failures, disables the key and returns `None` for protocol-class), `close_key(session_key: str) -> None` (closes every binding under that run key), `close_all() -> None`, `reap_idle(now: float, idle_s: float) -> None`
  - `get_session_registry() -> RemoteSessionRegistry`, `close_remote_sessions(session_key: str) -> None`, `close_all_remote_sessions() -> None`
  - `ConsoleSshSettings.session_worker: bool = True`, `session_idle_s: int = 60`, `bundle_cache: bool = True`

- [ ] **Step 1: Failing tests**

```python
# Tests/Tools/test_remote_session_registry.py
def test_one_session_per_key_and_binding():
    reg = RemoteSessionRegistry(); made = []
    def create():
        w = FakeWorker(); made.append(w); return w
    a = reg.acquire(("run-1", "b1"), create); b = reg.acquire(("run-1", "b1"), create)
    assert a is b and len(made) == 1

def test_protocol_failure_disables_key_for_the_run():
    reg = RemoteSessionRegistry()
    def create():
        return FakeWorker(start_error=SessionStartError(False, None, "stamp"))
    assert reg.acquire(("run-1", "b1"), create) is None
    assert reg.acquire(("run-1", "b1"), lambda: pytest.fail("retried a disabled session")) is None

def test_transport_failure_raises_and_is_retried_next_call():
    reg = RemoteSessionRegistry()
    with pytest.raises(SessionStartError):
        reg.acquire(("run-1", "b1"), lambda: FakeWorker(start_error=SessionStartError(True, None, "255")))
    assert reg.acquire(("run-1", "b1"), FakeWorker) is not None

def test_dead_session_is_restarted_once_then_falls_back():
    reg = RemoteSessionRegistry()
    first = reg.acquire(("run-1", "b1"), FakeWorker)
    first.alive = False                                    # died mid-run
    second = reg.acquire(("run-1", "b1"), FakeWorker)      # one restart allowed
    assert second is not first and second.alive
    second.alive = False                                   # died again
    assert reg.acquire(("run-1", "b1"), lambda: pytest.fail("restarted twice")) is None

def test_slow_start_does_not_block_other_bindings():
    reg = RemoteSessionRegistry(); gate = threading.Event()
    def slow():
        return FakeWorker(start_hook=lambda: gate.wait(5))
    t = threading.Thread(target=reg.acquire, args=(("run-1", "b1"), slow)); t.start()
    time.sleep(0.1)
    started = time.monotonic()
    assert reg.acquire(("run-1", "b2"), FakeWorker) is not None
    assert time.monotonic() - started < 1.0
    gate.set(); t.join(5)

def test_close_key_and_idle_reap():
    reg = RemoteSessionRegistry(); w = reg.acquire(("run-1", "b1"), FakeWorker)
    reg.close_key("run-1"); assert w.closed
    w2 = reg.acquire(("run-2", "b1"), lambda: FakeWorker(idle_since=0.0))
    reg.reap_idle(now=100.0, idle_s=60); assert w2.closed
```

`FakeWorker` lives in the test file:

```python
class FakeWorker:
    def __init__(self, start_error=None, idle_since=None, start_hook=None):
        self.start_error, self.idle_since, self.start_hook = start_error, idle_since, start_hook
        self.alive, self.closed = False, False
    def start(self):
        if self.start_hook: self.start_hook()
        if self.start_error: raise self.start_error
        self.alive = True
    def close(self):
        self.closed, self.alive = True, False
```

Executor tests (extend `test_remote_executor_ssh.py`, using the local-spawn seam from Task 6 via a `session_spawn` keyword on `for_ssh` used only by tests):
- `test_calls_route_through_one_session_per_run_key` — 10 reads with `session_key="run-1"` → one session spawn; one-shot transport never called.
- `test_kill_switch_uses_one_shot` — `session_worker=False` → zero session spawns.
- `test_blocked_binding_probe_uses_one_shot` — cache BLOCKED → `ping()` goes one-shot.
- `test_retargeted_destination_starts_no_session` — mismatched `expected_fingerprint` → `destination_changed`, no session spawn.
- `test_protocol_start_failure_falls_back_for_rest_of_run` — stamp mismatch → one-shot for this and later calls with the same key.

Config tests: defaults `(True, 60, True)`; non-bool `session_worker` → default; `session_idle_s` below 1 → default.

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement the registry**

```python
# tldw_chatbook/Tools/remote_session_registry.py
"""Per-run registry of SSH session workers: one per (run key, binding)."""

from __future__ import annotations

import threading
from typing import Callable

from tldw_chatbook.Tools.remote_session_worker import RemoteSessionWorker, SessionStartError


class RemoteSessionRegistry:
    """Thread-safe owner of live sessions and per-run disabled keys."""

    def __init__(self) -> None:
        self._lock = threading.Lock()  # guards the dicts only; never held across a start
        self._key_locks: dict[tuple[str, str], threading.Lock] = {}
        self._sessions: dict[tuple[str, str], RemoteSessionWorker] = {}
        self._disabled: set[tuple[str, str]] = set()
        self._restarted: set[tuple[str, str]] = set()

    def acquire(self, key: tuple[str, str], create: Callable[[], RemoteSessionWorker]) -> RemoteSessionWorker | None:
        """Return the live session for ``key``, starting one if needed.

        A session start can take seconds (ssh connect, handshake), so it runs
        under a per-key lock: concurrent callers for the same binding wait for
        one start; other bindings and runs are never blocked.

        Raises:
            SessionStartError: transport-class start failure (caller records it).
        """
        with self._lock:
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            with self._lock:
                if key in self._disabled:
                    return None
                worker = self._sessions.get(key)
                if worker is not None and worker.alive:
                    return worker
                if worker is not None:  # died mid-run: one restart, then fall back
                    self._sessions.pop(key, None)
                    if key in self._restarted:
                        self._disabled.add(key)
                        return None
                    self._restarted.add(key)
            worker = create()
            try:
                worker.start()
            except SessionStartError as error:
                if not error.transport:
                    with self._lock:
                        self._disabled.add(key)
                    return None
                raise
            with self._lock:
                self._sessions[key] = worker
            return worker

    def close_key(self, session_key: str) -> None:
        with self._lock:
            keys = [k for k in self._sessions if k[0] == session_key]
            workers = [self._sessions.pop(k) for k in keys]
            self._disabled = {k for k in self._disabled if k[0] != session_key}
            self._restarted = {k for k in self._restarted if k[0] != session_key}
            self._key_locks = {k: v for k, v in self._key_locks.items() if k[0] != session_key}
        for worker in workers:
            worker.close()

    def close_all(self) -> None:
        with self._lock:
            workers = list(self._sessions.values()); self._sessions.clear()
            self._disabled.clear(); self._restarted.clear()
        for worker in workers:
            worker.close()

    def reap_idle(self, now: float, idle_s: float) -> None:
        with self._lock:
            stale = [k for k, w in self._sessions.items()
                     if w.idle_since is not None and now - w.idle_since >= idle_s]
            workers = [self._sessions.pop(k) for k in stale]
        for worker in workers:
            worker.close()


_REGISTRY: RemoteSessionRegistry | None = None
_REGISTRY_LOCK = threading.Lock()


def get_session_registry() -> RemoteSessionRegistry:
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            _REGISTRY = RemoteSessionRegistry()
        return _REGISTRY


def close_remote_sessions(session_key: str) -> None:
    if _REGISTRY is not None:
        _REGISTRY.close_key(session_key)


def close_all_remote_sessions() -> None:
    if _REGISTRY is not None:
        _REGISTRY.close_all()
```

- [ ] **Step 4: Route in the executor** — in `_ssh_call` (before the one-shot path):

```python
        cfg = self._ssh
        from tldw_chatbook.config import get_console_ssh_settings

        settings = get_console_ssh_settings()
        blocked = cfg.cache.status(cfg.binding_id).state is BindingState.BLOCKED
        if cfg.session_key and settings.session_worker and not blocked:
            from tldw_chatbook.Tools.remote_session_registry import get_session_registry
            from tldw_chatbook.Tools.remote_session_worker import RemoteSessionWorker, SessionStartError

            registry = get_session_registry()
            registry.reap_idle(time.monotonic(), settings.session_idle_s)
            key = (cfg.session_key, cfg.binding_id)

            def create() -> RemoteSessionWorker:
                self._verify_destination()          # raises destination_changed, records BLOCKED
                return RemoteSessionWorker(
                    cfg.loc, transport=cfg.transport, python=cfg.python,
                    max_children=cfg.max_concurrent_calls, idle_s=settings.session_idle_s,
                    cache=settings.bundle_cache, spawn=cfg.session_spawn)

            try:
                session = registry.acquire(key, create)
            except SessionStartError as error:
                return RemoteCallResult(False, None, error.failure or TransportFailure(
                    TransportFailureKind.UNREACHABLE, None, str(error)))
            if session is not None:
                with _host_semaphore(cfg.resolved_host_key(), cfg.max_concurrent_calls):
                    return session.call(request_bytes, budget=budget)
        # ...existing one-shot path unchanged...
```

`_verify_destination` currently runs inside `_ssh_ping`; keep that call (one-shot pings still need it) — on the session path it runs once per session start inside `create()`. Add `session_key` and `session_spawn` to `_SshModeConfig` and `for_ssh` keyword args (`session_spawn` documented as a test seam). Add the three config fields with strict coercion mirroring `enable_multiplexing`.

- [ ] **Step 5: Run registry, executor, config, and transport suites; commit**

```bash
.venv/bin/python -m pytest Tests/Tools/test_remote_session_registry.py Tests/Tools/test_remote_executor_ssh.py Tests/test_config_console_ssh_defaults.py Tests/Tools/test_remote_transport_call.py -q
git add -A tldw_chatbook/Tools tldw_chatbook/config.py Tests
git commit -m "feat: session registry, executor routing and failure policy, [console_ssh] session keys"
```

---

### Task 8: Controller wiring — run key, shared session, close at run end and app exit

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`: `_default_remote_run_executor_factory(..., session_key: str | None = None)`, `_default_remote_instruction_executor(..., session_key: str | None = None)`, `capture_run_admitted_workspace_roots(..., remote_session_key: str | None = None)`, `_remote_instruction_io_for_selection(..., session_key: str | None = None)`; in `_run_agent_reply`, pass `assistant_message_id` as the key to all three call sites and wrap `return await self._finalize_agent_reply(...)` in `try/finally: close_remote_sessions(assistant_message_id)`.
- Modify: `tldw_chatbook/app.py` — in the existing shutdown hook that closes SSH control masters, call `close_all_remote_sessions()` first.
- Test: `Tests/Chat/test_console_ssh_composition.py` (extend), `Tests/Tools/test_admitted_root_consumer_census.py` (re-pin if counts move)

**Interfaces:**
- Consumes: Task 7 `close_remote_sessions`, `close_all_remote_sessions`; `for_ssh(session_key=...)`.
- Produces: nothing new for later tasks.

- [ ] **Step 1: Failing tests**

```python
from pathlib import PurePosixPath
from types import SimpleNamespace
from tldw_chatbook.Chat.console_chat_controller import (
    _default_remote_instruction_executor, _default_remote_run_executor_factory,
)
from tldw_chatbook.Tools.remote_binding_status import RemoteBindingStatusCache
from tldw_chatbook.Tools.remote_root_types import RemoteRoot
from tldw_chatbook.Tools.remote_workspace_executor import RemoteWorkspaceToolExecutor

def test_default_factories_share_the_run_key(monkeypatch):
    seen = []
    monkeypatch.setattr(
        RemoteWorkspaceToolExecutor, "for_ssh",
        classmethod(lambda cls, *a, **k: seen.append(k.get("session_key")) or object()),
    )
    row = SimpleNamespace(binding_id="b1", locator="ssh://devbox/srv/www",
                          metadata={"python": "python3", "canonical_fingerprint": "a" * 64})
    selection = SimpleNamespace(
        binding=row,
        root=RemoteRoot(alias="b1", canonical_locator="ssh://devbox/srv/www",
                        root=PurePosixPath("/srv/www"), binding_id="b1"),
    )
    cache = RemoteBindingStatusCache()
    _default_remote_run_executor_factory(row, "b1", status_cache=cache, sensitive_exclusions=lambda: (), session_key="msg-1")
    _default_remote_instruction_executor(selection, "b1", status_cache=cache, sensitive_exclusions=lambda: (), session_key="msg-1")
    assert seen == ["msg-1", "msg-1"]
```

The `_default_remote_run_executor_factory` returns a dispatch adapter wrapping the executor; if the adapter rejects a bare `object()`, return `SimpleNamespace(execute=lambda *a, **k: None)` from the fake instead. For "injected factories keep their signature", extend the existing `test_blocked_remote_and_local_ready_composes_local_only` pattern in `Tests/Chat/test_console_ssh_composition.py`: pass `remote_session_key="msg-1"` alongside its existing stub factory (whose signature has no `session_key`) and assert composition still succeeds.

```python

```

Run-end test: copy the setup of the existing `_run_agent_reply` test at `Tests/Chat/test_console_chat_controller.py:7489` (controller fixture, `resolve_turn_configuration_snapshot`, `_capture_and_resolve_turn_execution_context`, `build_console_request`) into a new test in the same file, then:

```python
    closed = []
    monkeypatch.setattr(
        "tldw_chatbook.Tools.remote_session_registry.close_remote_sessions", closed.append
    )
    await controller._run_agent_reply(
        resolution=resolution,
        provider_messages=[{"role": "user", "content": "hello"}],
        assistant_message_id=assistant.id,
        prepare_retry=False,
        variant_mode=False,
        turn_context=turn_context,
        capture_mode_override=ConsoleTraceCaptureMode.CAPTURE_ON,
        trace_request=trace_request,
    )
    assert closed == [assistant.id]
```

Add a second variant whose provider stub raises, asserting `closed == [assistant.id]` still (the `finally` path). Because `_run_agent_reply` imports `close_remote_sessions` at call time from `tldw_chatbook.Tools.remote_session_registry`, patching the module attribute is sufficient.

App exit: in `tldw_chatbook/app.py` near line 19474, inside the same `try` that runs `get_master_manager().close_all`, add before it:

```python
                from tldw_chatbook.Tools.remote_session_registry import close_all_remote_sessions

                await asyncio.to_thread(close_all_remote_sessions)
```

(Sessions first: closing their stdin lets the remote parents exit before the masters go away.)

- [ ] **Step 2: Implement** — the default-factory partial keeps injected factories untouched:

```python
    factory = remote_executor_factory or functools.partial(
        _default_remote_run_executor_factory, session_key=remote_session_key
    )
```

and in `_run_agent_reply`:

```python
        try:
            return await self._finalize_agent_reply(
                ...unchanged arguments...
            )
        finally:
            from tldw_chatbook.Tools.remote_session_registry import close_remote_sessions

            close_remote_sessions(assistant_message_id)
```

(Deferred import: `remote_session_registry` must not become resident at UI-ready.)

- [ ] **Step 3: Boot census** — `.venv/bin/python -m pytest Tests/Performance/test_ui_ready_module_census.py -q`; it must pass with no new modules. If `remote_session_*` shows up, move the offending import into a function.

- [ ] **Step 4: Run Chat SSH suites + census; commit**

```bash
.venv/bin/python -m pytest Tests/Chat/test_console_ssh_composition.py Tests/Chat/test_console_project_instructions.py Tests/Tools/test_admitted_root_consumer_census.py Tests/Performance/test_ui_ready_module_census.py -q
git add -A tldw_chatbook Tests
git commit -m "feat: one SSH session per binding per run; close at run end and app exit"
```

---

### Task 9: ADR amendment, docs, live UAT, gates

**Files:**
- Modify: `backlog/decisions/181-ssh-remote-workspace-bindings.md` — add "Amendment 2026-09-27: session worker and bundle cache" (the spec's Security posture section, verbatim in substance, including the memory-inheritance and cache threat-model statements)
- Modify: `Docs/User_Guide/console/context-and-rag.md` — SSH section: warm calls reuse a session; `[console_ssh] session_worker/session_idle_s/bundle_cache`
- Modify: `CHANGELOG.md` `[Unreleased]` — "SSH workspace tool calls reuse a per-run session (about one round trip each); the worker is cached in `$XDG_RUNTIME_DIR`."
- Create: `Tests/Tools/test_remote_session_live.py` — opt-in (`@pytest.mark.live_ssh`, skipped unless `TLDW_LIVE_SSH_HOST` is set): the 21 UAT checks through a session, cold miss vs hit, idle exit, no stray processes, latency median/p90 printed
- Backlog task created at execution start (id assigned against `origin/dev` + all refs sweep), Implementation Notes record the spike and live numbers

- [ ] **Step 1: Write the ADR amendment and docs; commit.**
- [ ] **Step 2: Diagnostics** — `python scripts/check_persistent_diagnostic_inventory.py --statements <new/changed files> --since <pin base>`, review every new statement (no paths, no request content), then `--write`.
- [ ] **Step 3: Preflight** — `PYTHON=.venv/bin/python ./scripts/preflight.sh` → all checks pass.
- [ ] **Step 4: Live UAT** — `TLDW_LIVE_SSH_HOST=ml-user@192.168.5.84 .venv/bin/python -m pytest Tests/Tools/test_remote_session_live.py -q -s`; success = warm median ≤ spike echo median + 15 ms, cold-hit ≤ ~3 RTT + 40 ms, all checks pass, no leftovers except the cache file under `/run/user/<uid>/tldw-worker/`.
- [ ] **Step 5: Record numbers in the task notes, mark Done, commit, open the PR.**

---

## Spike results

Measured 2026-09-27, laptop <-> `ml-user@192.168.5.84` (Debian 13, python3
3.13.5) over LAN Wi-Fi, one ControlMaster-backed ssh channel per run, 60 reps
per metric (first 10 discarded as warm-up), 50 ICMP pings (`-i 0.2`) run
immediately after in the same script invocation. Script:
`scratch/echo_spike.py` (throwaway, not committed) — the brief's sample with
two corrections: `shlex.quote(SERVER)` instead of `repr(SERVER)` (ssh rejoins
argv into one string that the remote login shell re-parses, so the server
source must be shell-quoted), and a `struct`-driven frame-accumulating read
loop instead of a fixed byte count. Ping was also run without `-q` and its
per-packet `time=` values parsed, since `-q` only gives min/avg/max/stddev
and not the median/p90 this spike needs.

| Run | echo median / p90 (ms) | fork-probe median / p90 (ms) | ping median / p90 (ms) | echo ≤ ping median + 15ms? |
|---|---|---|---|---|
| 1 (20:02 PDT) | 7.51 / 124.52 | 7.65 / 128.10 | 37.43 / 255.35 | yes (7.51 ≤ 52.43) |
| 2 (20:04 PDT) | 6.50 / 15.35 | 6.09 / 10.02 | 13.44 / 107.67 | yes (6.50 ≤ 28.44) |
| 3 (20:06 PDT) | 8.14 / 35.64 | 8.31 / 93.64 | 28.77 / 113.28 | yes (8.14 ≤ 43.77) |

**Decision: GO.** Echo median beat the `ping median + 15ms` bar in all three
runs at different moments — in fact the framed-echo round trip over the
persistent ControlMaster channel was consistently *at or below* raw ICMP
ping's median (this LAN/Wi-Fi link has a heavy-tailed ping distribution:
p90 up to 9x the median in two of three runs, while the SSH-channel echo's
tail stayed much tighter except in run 1). Echo was never ≥40ms above ping,
so the delayed-ACK stall condition never triggered and `-o IPQoS=lowdelay`
was not tested — no master option to carry into Task 6 beyond what's already
planned there.

**Fork+waitpid overhead** (fork-probe median minus echo median, same run):
run 1 +0.14ms, run 2 -0.41ms (noise), run 3 +0.17ms. A fork+waitpid per
request adds no measurable latency beyond the round trip itself (well under
1ms, within measurement noise) — `fork_pin_op_ms` for Task 9's success check
should be treated as ~0ms extra over the echo floor.

No stray remote processes or leftover control sockets after any run
(verified via `ps -eo pid,args | grep '[p]ython3 -I -c'` and `ls /tmp/tes-*`
after each trial).
