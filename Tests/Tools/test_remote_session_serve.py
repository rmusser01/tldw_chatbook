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
    # Portable zombie check: `ps --ppid` (Linux-only) and `ps -g` (wrong
    # semantics on macOS) aren't both available, so list every process and
    # filter columns in Python instead -- works the same on Linux and macOS.
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,stat="], capture_output=True, text=True).stdout
    zombies = [
        line for line in out.splitlines()
        if len(line.split()) >= 3 and line.split()[1] == str(proc.pid) and "Z" in line.split()[2]
    ]
    assert not zombies, zombies
    proc.stdin.close(); proc.wait(5)
