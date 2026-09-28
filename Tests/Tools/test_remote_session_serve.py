import json, os, select, subprocess, sys, textwrap, time
from pathlib import Path

import pytest

from tldw_chatbook.Tools.remote_session_frames import (
    BUSY, CANCEL, HELLO, HOST_SPAWN_FAILED, LINE, REQUEST, STATUS, FrameReader, decode_status,
    encode_frame,
)

# `serve` forks; nightly CI runs `pytest ./Tests/` on Windows too.
pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork server")

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
            # fd hygiene check: the child's own pipe is dup2'd onto fd 1 and
            # every descriptor above 2 must be closed, so any open fd in that
            # band is a leak (the parent's stdin/stdout copies, the epoll/
            # kqueue fd behind the selector, or another live child's pipe).
            # Hardcoding a single fd number is not a real test on its own --
            # if nothing happens to occupy that number the probe reports
            # "clean" whether or not the close actually ran. The caller
            # starts a second long-running child first so there IS a real
            # fd to leak, and this scans a wide band rather than guessing
            # which number it landed on.
            leaked = "clean"
            for fd in range(3, 64):
                try:
                    os.fstat(fd)
                    leaked = "leaked:%%d" %% fd
                    break
                except OSError:
                    pass
            out.write(leaked.encode() + b"\\n")
            return 0
        if cmd in ("twostep", "twostep-slow"):
            out.write(b"one\\n"); out.flush()
            time.sleep(0.002 if cmd == "twostep" else 0.6)
            out.write(b"two\\n"); out.flush()
            return 0
        out.write(b"echo:" + raw + b"\\n"); out.flush()
        return 0
    sys.exit(serve(0, 1, run_request=run_request, max_request_bytes=1 << 20, max_response_bytes=1 << 16))
''' % str(REPO))


def _session(max_children=4, idle_s=30.0, prelude=""):
    script = HANDLER.replace(
        "from tldw_chatbook.Tools.remote_session_serve import serve",
        prelude + "\nfrom tldw_chatbook.Tools.remote_session_serve import serve",
    )
    proc = subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    os.write(proc.stdin.fileno(), encode_frame(HELLO, 0, json.dumps({"max_children": max_children, "idle_s": idle_s}).encode()))
    return proc, FrameReader(max_body=1 << 20)


_FAIL_ONCE = textwrap.dedent('''
    import errno
    _real, _left = os.{name}, [1]
    def _once(*args):
        if _left[0]:
            _left[0] -= 1
            raise OSError(errno.{errno}, "refused")
        return _real(*args)
    os.{name} = _once
''')


@pytest.mark.parametrize("name,errno_name", [("fork", "EAGAIN"), ("pipe", "EMFILE")])
def test_failed_spawn_fails_only_that_request(name, errno_name):
    """TASK-33404: the host at its process/fd limit fails one request, not the session."""
    proc, reader = _session(prelude=_FAIL_ONCE.format(name=name, errno=errno_name))
    try:
        os.write(proc.stdin.fileno(), encode_frame(REQUEST, 1, b"first") + encode_frame(REQUEST, 2, b"second"))
        frames = _collect(proc, reader, 2)
        statuses = {rid: decode_status(body) for kind, rid, body in frames if kind == STATUS}
        assert statuses[1] == (HOST_SPAWN_FAILED, None)
        assert statuses[2] == (0, None)
        assert not any(kind == LINE and rid == 1 for kind, rid, _ in frames)
        os.write(proc.stdin.fileno(), encode_frame(REQUEST, 3, b"third"))
        assert any(kind == STATUS and rid == 3 for kind, rid, _ in _collect(proc, reader, 1))
    finally:
        _terminate(proc)


def _terminate(proc):
    """Guarantee a test never leaks a live/zombie subprocess on assertion failure."""
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    except OSError:
        pass
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(5)
    except subprocess.TimeoutExpired:
        pass


def _collect(proc, reader, want_status, timeout=10.0):
    """Read frames until `want_status` distinct STATUS ids arrive or the deadline passes.

    Uses `select` with the remaining deadline before every read so a child
    that never produces output cannot block this helper forever (a bare
    `os.read` has no timeout of its own).
    """
    done, frames, end = set(), [], time.monotonic() + timeout
    fd = proc.stdout.fileno()
    while len(done) < want_status:
        remaining = end - time.monotonic()
        if remaining <= 0:
            break
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            break
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        for kind, rid, body in reader.feed(chunk):
            frames.append((kind, rid, body))
            if kind == STATUS:
                done.add(rid)
    return frames


def test_concurrent_requests_complete_out_of_order_lines_before_status():
    proc, reader = _session()
    try:
        os.write(proc.stdin.fileno(), encode_frame(REQUEST, 1, b"sleep:0.5") + encode_frame(REQUEST, 2, b"fast"))
        frames = _collect(proc, reader, 2)
        order = [rid for kind, rid, _ in frames if kind == STATUS]
        assert order == [2, 1]
        for rid in (1, 2):
            kinds = [k for k, r, _ in frames if r == rid]
            assert kinds[-1] == STATUS and LINE in kinds and kinds.index(LINE) < kinds.index(STATUS)
        proc.stdin.close(); assert proc.wait(5) == 0
    finally:
        _terminate(proc)


def test_cancel_kills_only_that_child():
    proc, reader = _session()
    try:
        os.write(proc.stdin.fileno(), encode_frame(REQUEST, 1, b"sleep:30") + encode_frame(REQUEST, 2, b"fast"))
        time.sleep(0.3)
        os.write(proc.stdin.fileno(), encode_frame(CANCEL, 1, b""))
        frames = _collect(proc, reader, 2)
        statuses = {rid: decode_status(body) for kind, rid, body in frames if kind == STATUS}
        assert statuses[1][1] == 9 and statuses[2] == (0, None)
    finally:
        _terminate(proc)


def test_cancel_of_queued_request_sends_status_promptly():
    proc, reader = _session(max_children=1)
    try:
        os.write(proc.stdin.fileno(), encode_frame(REQUEST, 1, b"sleep:1") + encode_frame(REQUEST, 2, b"sleep:30"))
        time.sleep(0.1)  # let request 1 start and request 2 queue (BUSY)
        os.write(proc.stdin.fileno(), encode_frame(CANCEL, 2, b""))
        frames = _collect(proc, reader, 2, timeout=5)
        order = [rid for kind, rid, _ in frames if kind == STATUS]
        assert order == [2, 1], "the queued cancel must resolve before request 1 finishes its sleep"
        statuses = {rid: decode_status(body) for kind, rid, body in frames if kind == STATUS}
        assert statuses[2] == (None, 9)
        assert statuses[1] == (0, None)
    finally:
        _terminate(proc)


def test_output_cap_kills_the_flooding_child_only():
    proc, reader = _session()
    try:
        os.write(proc.stdin.fileno(), encode_frame(REQUEST, 1, b"flood") + encode_frame(REQUEST, 2, b"fast"))
        frames = _collect(proc, reader, 2)
        statuses = {rid: decode_status(body) for kind, rid, body in frames if kind == STATUS}
        assert statuses[1][1] == 9 and statuses[2] == (0, None)
        assert sum(len(b) for k, r, b in frames if r == 1 and k == LINE) <= (1 << 16) + 65537
    finally:
        _terminate(proc)


def test_child_cap_queues_and_reports_busy():
    proc, reader = _session(max_children=1)
    try:
        os.write(proc.stdin.fileno(), encode_frame(REQUEST, 1, b"sleep:0.3") + encode_frame(REQUEST, 2, b"fast"))
        frames = _collect(proc, reader, 2)
        assert (BUSY, 2, b"") in frames
        order = [rid for kind, rid, _ in frames if kind == STATUS]
        assert order == [1, 2], "request 2 is queued behind 1 (max_children=1); its STATUS must arrive after 1's"
    finally:
        _terminate(proc)


def test_child_holds_no_channel_fds():
    proc, reader = _session()
    try:
        # A live sibling's pipe must still be open in the parent when the
        # fdprobe child forks, otherwise the probe has nothing real to catch --
        # see the comment on `fdprobe` in HANDLER above.
        os.write(proc.stdin.fileno(), encode_frame(REQUEST, 1, b"sleep:2"))
        time.sleep(0.2)
        os.write(proc.stdin.fileno(), encode_frame(REQUEST, 2, b"fdprobe"))
        frames = _collect(proc, reader, 1, timeout=5)  # only rid=2 (fdprobe) finishes here
        fdprobe_lines = [body for kind, rid, body in frames if kind == LINE and rid == 2]
        assert any(b"clean" in body for body in fdprobe_lines), fdprobe_lines
        os.write(proc.stdin.fileno(), encode_frame(CANCEL, 1, b""))
    finally:
        _terminate(proc)


def test_eof_kills_live_children_and_exits():
    proc, reader = _session()
    try:
        os.write(proc.stdin.fileno(), encode_frame(REQUEST, 1, b"sleep:30"))
        time.sleep(0.3); proc.stdin.close()
        assert proc.wait(5) == 0
    finally:
        _terminate(proc)


def test_idle_exit_ignores_a_long_running_child():
    proc, reader = _session(idle_s=0.5)
    try:
        os.write(proc.stdin.fileno(), encode_frame(REQUEST, 1, b"sleep:1.5"))
        frames = _collect(proc, reader, 1, timeout=5)
        assert any(k == STATUS for k, _, _ in frames), "idle timer fired under a live child"
        assert proc.wait(5) == 0  # then idles out
    finally:
        _terminate(proc)


def test_no_zombies_after_many_requests():
    proc, reader = _session()
    try:
        for i in range(1, 51):
            os.write(proc.stdin.fileno(), encode_frame(REQUEST, i, b"fast"))
        frames = _collect(proc, reader, 50, timeout=15)
        statuses = {rid for kind, rid, _ in frames if kind == STATUS}
        assert statuses == set(range(1, 51)), f"missing STATUS for {set(range(1, 51)) - statuses}"
        # Portable zombie check: `ps --ppid` (Linux-only) and `ps -g` (wrong
        # semantics on macOS) aren't both available, so list every process
        # and filter columns in Python instead -- works the same on both.
        out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,stat="], capture_output=True, text=True).stdout
        zombies = [
            line for line in out.splitlines()
            if len(line.split()) >= 3 and line.split()[1] == str(proc.pid) and "Z" in line.split()[2]
        ]
        assert not zombies, zombies
    finally:
        _terminate(proc)


def test_malformed_hello_exits_with_status_3():
    for bad_body in (b"not json", b'{"max_children": 4}', b'{"max_children": "x", "idle_s": 1.0}'):
        proc = subprocess.Popen([sys.executable, "-c", HANDLER], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            os.write(proc.stdin.fileno(), encode_frame(HELLO, 0, bad_body))
            assert proc.wait(5) == 3, bad_body
        finally:
            _terminate(proc)


_FD_HELPER_SCRIPT = textwrap.dedent('''
    import os, sys, time
    sys.path.insert(0, %r)
    from tldw_chatbook.Tools.remote_session_serve import _close_fds_above_2
    def _fd_open(fd):
        try:
            os.fstat(fd); return True
        except OSError:
            return False
    result_r, result_w = os.pipe()
    extras = [os.pipe() for _ in range(3)]  # fds the helper must close
    start = time.monotonic()
    pid = os.fork()
    if pid == 0:
        os.close(result_r)
        _close_fds_above_2(keep=result_w)
        leaked = any(_fd_open(fd) for pair in extras for fd in pair)
        os.write(result_w, b"1" if leaked else b"0")
        os._exit(0)
    os.close(result_w)
    for r, w in extras:
        os.close(r); os.close(w)
    payload = os.read(result_r, 1)
    os.waitpid(pid, 0)
    elapsed = time.monotonic() - start
    sys.stdout.write(payload.decode() + " " + str(elapsed))
''' % str(REPO))


def test_close_fds_above_2_closes_extras_quickly():
    """Unit test for the `_close_fds_above_2` helper in isolation.

    Runs in a freshly exec'd, single-threaded subprocess (not forked from
    the pytest process itself, which may be multi-threaded under xdist --
    forking that would be exactly the hazard `serve`'s own docstring warns
    about). The helper must close every fd above 2 other than `keep`, and
    do so fast: the old `closerange(3, SC_OPEN_MAX)` approach measured
    ~134 ms per fork on this machine (one close() syscall per fd number, up
    to a million of them); the fix enumerates real open fds via /proc or
    /dev instead.
    """
    out = subprocess.run([sys.executable, "-c", _FD_HELPER_SCRIPT], capture_output=True, text=True, timeout=10)
    assert out.returncode == 0, out.stderr
    leaked_flag, elapsed_s = out.stdout.strip().split()
    assert leaked_flag == "0", "helper left an extra descriptor open in the child"
    elapsed = float(elapsed_s)
    assert elapsed < 0.1, f"fork+close took {elapsed * 1000:.1f} ms -- looks like the SC_OPEN_MAX regression"


@pytest.mark.parametrize(
    "bad_body",
    [
        b'{"max_children": 4, "idle_s": 1e309}',  # json -> inf
        b'{"max_children": 4, "idle_s": Infinity}',
        b'{"max_children": 4, "idle_s": NaN}',
        b'{"max_children": 4, "idle_s": -Infinity}',
        b'{"max_children": true, "idle_s": 1.0}',
        b'{"max_children": 4, "idle_s": false}',
        b'{"max_children": 4.5, "idle_s": 1.0}',
        pytest.param(b'{"max_children": 4, "idle_s": 1' + b"0" * 400 + b"}", id="int-overflows-float"),
        b"[4, 1.0]",
    ],
)
def test_hello_limits_rejected_before_conversion(bad_body):
    proc = subprocess.Popen([sys.executable, "-c", HANDLER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        os.write(proc.stdin.fileno(), encode_frame(HELLO, 0, bad_body))
        assert proc.wait(5) == 3, (bad_body, proc.stderr.read())
    finally:
        _terminate(proc)


def test_huge_finite_idle_is_clamped_not_a_crash():
    proc = subprocess.Popen([sys.executable, "-c", HANDLER], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    reader = FrameReader(max_body=1 << 20)
    try:
        os.write(proc.stdin.fileno(), encode_frame(HELLO, 0, b'{"max_children": 4, "idle_s": 1e300}'))
        time.sleep(0.2)  # the idle wait now runs with the clamped timeout
        os.write(proc.stdin.fileno(), encode_frame(REQUEST, 1, b"fast"))
        frames = _collect(proc, reader, 1, timeout=5)
        assert [decode_status(b) for k, _, b in frames if k == STATUS] == [(0, None)]
    finally:
        _terminate(proc)


def _statuses_for(proc, reader, rid, want, timeout):
    """Collect frames until ``want`` STATUS frames for ``rid`` arrive (duplicates counted)."""
    frames, end, fd = [], time.monotonic() + timeout, proc.stdout.fileno()
    while sum(1 for k, r, _ in frames if k == STATUS and r == rid) < want:
        remaining = end - time.monotonic()
        if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
            break
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        frames.extend(reader.feed(chunk))
    return frames


def test_duplicate_request_id_is_refused_and_first_stays_cancellable():
    proc, reader = _session()
    try:
        os.write(proc.stdin.fileno(), encode_frame(REQUEST, 1, b"sleep:30"))
        time.sleep(0.2)
        os.write(proc.stdin.fileno(), encode_frame(REQUEST, 1, b"fast"))
        frames = _statuses_for(proc, reader, 1, 1, timeout=5)
        assert [decode_status(b) for k, _, b in frames if k == STATUS] == [(None, 9)]
        assert not [b for k, _, b in frames if k == LINE], "the duplicate must never run"
        os.write(proc.stdin.fileno(), encode_frame(CANCEL, 1, b""))
        frames += _statuses_for(proc, reader, 1, 1, timeout=5)
        statuses = [decode_status(b) for k, _, b in frames if k == STATUS]
        assert len(statuses) == 2 and statuses[1][1] == 9, "CANCEL must still reach the first child"
    finally:
        _terminate(proc)


def test_queue_count_bound_refuses_the_excess_at_once():
    proc, reader = _session(max_children=1)
    try:
        # 1 runs, 2..5 fill the queue (4 x max_children), 6 is refused.
        wire = b"".join(encode_frame(REQUEST, i, b"sleep:30" if i == 1 else b"fast") for i in range(1, 7))
        os.write(proc.stdin.fileno(), wire)
        frames = _statuses_for(proc, reader, 6, 1, timeout=5)
        statuses = {r: decode_status(b) for k, r, b in frames if k == STATUS}
        assert statuses == {6: (None, 9)}
        assert {r for k, r, _ in frames if k == BUSY} == {2, 3, 4, 5}
        os.write(proc.stdin.fileno(), encode_frame(CANCEL, 1, b""))
        frames = _collect(proc, reader, 5)
        assert {r for k, r, _ in frames if k == STATUS} == {1, 2, 3, 4, 5}
    finally:
        _terminate(proc)


def test_queue_byte_bound_refuses_the_excess_at_once():
    proc, reader = _session(max_children=1)
    try:
        body = b"x" * (700 * 1024)  # cap is 2 x max_request_bytes (1 MiB) = 2 MiB queued
        wire = encode_frame(REQUEST, 1, b"sleep:30")
        wire += b"".join(encode_frame(REQUEST, i, body) for i in (2, 3, 4))
        os.write(proc.stdin.fileno(), wire)
        frames = _statuses_for(proc, reader, 4, 1, timeout=5)
        assert {r: decode_status(b) for k, r, b in frames if k == STATUS} == {4: (None, 9)}
        assert {r for k, r, _ in frames if k == BUSY} == {2, 3}
    finally:
        _terminate(proc)


def _chunks_until_status(proc, reader, timeout=10.0):
    """Frames grouped by the os.read chunk they arrived in."""
    chunks, end, fd = [], time.monotonic() + timeout, proc.stdout.fileno()
    while time.monotonic() < end:
        ready, _, _ = select.select([fd], [], [], end - time.monotonic())
        if not ready:
            break
        data = os.read(fd, 65536)
        if not data:
            break
        frames = reader.feed(data)
        chunks.append(frames)
        if any(kind == STATUS for kind, _, _ in frames):
            break
    return chunks


# A generous window for the tests (the shipped value is _COALESCE_S): a
# loaded CI runner must not turn "child exited within the window" flaky.
_HOLD_200MS = "import tldw_chatbook.Tools.remote_session_serve as _s; _s._COALESCE_S = 0.2"


def test_fast_op_output_and_status_leave_in_one_write():
    proc, reader = _session(prelude=_HOLD_200MS)
    try:
        os.write(proc.stdin.fileno(), encode_frame(REQUEST, 1, b"twostep"))
        chunks = _chunks_until_status(proc, reader)
        assert len(chunks) == 1, chunks
        assert [kind for kind, _, _ in chunks[0]] == [LINE, LINE, STATUS]
    finally:
        _terminate(proc)


def test_slow_op_first_line_is_not_held_until_the_end():
    proc, reader = _session(prelude=_HOLD_200MS)
    try:
        os.write(proc.stdin.fileno(), encode_frame(REQUEST, 1, b"twostep-slow"))
        chunks = _chunks_until_status(proc, reader)
        assert len(chunks) >= 2
        assert [kind for kind, _, _ in chunks[0]] == [LINE]
    finally:
        _terminate(proc)
