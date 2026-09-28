"""Opt-in live UAT for the SSH session worker + bundle cache (ADR-181 amendment).

Skipped unless ``TLDW_LIVE_SSH_HOST`` names a reachable host (``user@host``
or an ssh-config alias) with BatchMode key auth and ``python3`` >= 3.10:

    TLDW_LIVE_SSH_HOST=ml-user@192.168.5.84 \\
        python -m pytest Tests/Tools/test_remote_session_live.py -q -s

Drives the REAL stack -- ``RemoteWorkspaceToolExecutor.for_ssh`` with a real
``SshMasterManager`` and a unique per-test session key. On the host it only
creates (and removes) one ``~/tldw-live-<random>`` scratch folder, and clears
this feature's own cache entries under ``$XDG_RUNTIME_DIR/tldw-worker/`` to
force a cold miss. Timing is reported, never hard-asserted beyond a loose
sanity bound (no flaky latency gates).
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import shutil
import statistics
import subprocess
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from tldw_chatbook.Tools.remote_binding_locator import (
    canonical_fingerprint,
    canonicalize_locator,
    parse_remote_locator,
)
from tldw_chatbook.Tools.remote_binding_status import (
    BindingState,
    RemoteBindingStatusCache,
)
from tldw_chatbook.Tools.remote_workspace_executor import (
    RemoteWorkspaceExecutionError,
    RemoteWorkspaceToolExecutor,
    _bundle_payload,
    parse_fs_read_stamps,
)
from tldw_chatbook.Tools.remote_workspace_transport import SshMasterManager

HOST = os.environ.get("TLDW_LIVE_SSH_HOST", "")

pytestmark = [
    pytest.mark.live_ssh,
    pytest.mark.skipif(not HOST, reason="set TLDW_LIVE_SSH_HOST to run the live SSH UAT"),
]

#: Spike reference echo floor (ruling R5: median of the three run medians).
SPIKE_ECHO_FLOOR_MS = 7.51
#: Numbers collected across the module, printed by the last test.
RESULTS: dict[str, Any] = {}


def _host(command: str, *, check: bool = True) -> str:
    """Run one command on the host over a plain BatchMode ssh (not the session)."""
    done = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "--", HOST, command],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if check and done.returncode != 0:
        raise AssertionError(f"host command failed ({done.returncode}): {done.stderr.strip()}")
    return done.stdout


def _host_workers() -> list[str]:
    """This uid's ``python3 -I -c`` processes on the host (session parents/children)."""
    out = _host('ps -u "$(id -u)" -o pid=,args= | grep "[p]ython3 -I -c" || true')
    return [line for line in out.splitlines() if line.strip()]


def _wait_no_workers(timeout: float) -> list[str]:
    deadline = time.monotonic() + timeout
    while True:
        left = _host_workers()
        if not left or time.monotonic() > deadline:
            return left
        time.sleep(0.5)


@pytest.fixture(scope="module")
def live() -> Iterator[SimpleNamespace]:
    from tldw_chatbook import config as config_module
    from tldw_chatbook.Tools import remote_session_worker as worker_module
    from tldw_chatbook.Tools.remote_session_registry import close_remote_sessions

    home = _host('printf %s "$HOME"').strip()
    assert home.startswith("/"), home
    scratch = f"{home}/tldw-live-{uuid.uuid4().hex[:10]}"
    _host(
        f"mkdir -m 700 {shlex.quote(scratch)} && cd {shlex.quote(scratch)} && "
        "printf 'alpha body\\n' > alpha.txt && mkdir sub && printf 'beta\\n' > sub/beta.md "
        "&& mkdir pinroot && printf 'pin\\n' > pinroot/p.txt"
    )
    state_dir = Path(f"/tmp/tl-live-{uuid.uuid4().hex[:6]}")
    state_dir.mkdir(mode=0o700)
    masters = SshMasterManager(state_dir=state_dir)

    settings = SimpleNamespace(
        value=config_module.ConsoleSshSettings(max_concurrent_calls=4)
    )
    starts: list[dict[str, Any]] = []
    original_start = worker_module.RemoteSessionWorker.start
    original_read = worker_module.RemoteSessionWorker._read_handshake_line

    def recording_start(self: Any) -> None:
        record: dict[str, Any] = {"lines": []}
        self._live_record = record
        began = time.perf_counter()
        try:
            original_start(self)
        finally:
            record["seconds"] = time.perf_counter() - began
            starts.append(record)

    def recording_read(self: Any, deadline: float) -> bytes:
        line = original_read(self, deadline)
        getattr(self, "_live_record", {"lines": []})["lines"].append(line)
        return line

    keys: list[str] = []
    mp = pytest.MonkeyPatch()
    mp.setattr(config_module, "get_console_ssh_settings", lambda: settings.value)
    mp.setattr(worker_module.RemoteSessionWorker, "start", recording_start)
    mp.setattr(worker_module.RemoteSessionWorker, "_read_handshake_line", recording_read)

    def make(
        rel: str | None = None,
        *,
        root: str | None = None,
        fingerprint: str | None = None,
        grace: float = 2.0,
    ) -> SimpleNamespace:
        path = root or (scratch if rel is None else f"{scratch}/{rel}")
        loc = parse_remote_locator(f"ssh://{HOST}{path}")
        if fingerprint is None:
            fingerprint = canonical_fingerprint(canonicalize_locator(loc), loc.path)
        key = f"live-{uuid.uuid4().hex}"
        keys.append(key)
        cache = RemoteBindingStatusCache()
        binding_id = f"live-binding-{uuid.uuid4().hex[:8]}"
        executor = RemoteWorkspaceToolExecutor.for_ssh(
            loc,
            binding_id,
            cache=cache,
            masters=masters,
            grace=grace,
            max_concurrent_calls=4,
            recovery_probes=False,
            expected_fingerprint=fingerprint,
            session_key=key,
        )
        return SimpleNamespace(
            executor=executor, cache=cache, binding_id=binding_id, key=key, loc=loc
        )

    loc0 = parse_remote_locator(f"ssh://{HOST}{scratch}")
    masters.ensure_master(loc0)  # master spawn is not part of any measured number
    _artifact, compressed, _bootstrap = _bundle_payload()
    try:
        yield SimpleNamespace(
            home=home,
            scratch=scratch,
            make=make,
            starts=starts,
            settings=settings,
            close=close_remote_sessions,
            bundle_hash=hashlib.sha256(compressed).hexdigest(),
            config=config_module,
            keys=keys,
        )
    finally:
        for key in keys:
            close_remote_sessions(key)
        mp.undo()
        masters.close_all()
        shutil.rmtree(state_dir, ignore_errors=True)
        assert scratch.startswith(f"{home}/tldw-live-")
        _host(f"rm -rf -- {shlex.quote(scratch)}", check=False)


@pytest.fixture(autouse=True)
def _close_test_sessions(live: SimpleNamespace) -> Iterator[None]:
    """Close every session a test opened, even when it fails midway."""
    first = len(live.keys)
    yield
    for key in live.keys[first:]:
        live.close(key)


def _read(executor: RemoteWorkspaceToolExecutor, path: str) -> dict[str, Any]:
    return executor.execute("fs_read", {"path": path}, intent="read")


# -- 1. cold start: miss, then hit ------------------------------------------


def test_cold_miss_then_cache_hit(live: SimpleNamespace) -> None:
    # Force a miss: remove only this feature's own cache entries.
    _host('rm -f -- "$XDG_RUNTIME_DIR"/tldw-worker/* 2>/dev/null; true')

    a = live.make()
    began = time.perf_counter()
    _read(a.executor, "alpha.txt")
    first_call_miss = time.perf_counter() - began
    miss = live.starts[-1]
    assert miss["lines"][0].startswith(b"NEED "), miss["lines"]
    assert miss["lines"][-1].startswith(b"READY "), miss["lines"]
    live.close(a.key)

    # Several cache-hit starts: one sample is at the mercy of the channel
    # open over the master, which alone swings 100-400 ms on a Wi-Fi link.
    hits: list[float] = []
    for _ in range(5):
        b = live.make()
        _read(b.executor, "alpha.txt")
        hit = live.starts[-1]
        assert len(hit["lines"]) == 1 and hit["lines"][0].startswith(b"READY "), hit["lines"]
        hits.append(hit["seconds"] * 1000)
        live.close(b.key)

    RESULTS.update(
        cold_miss_start_ms=miss["seconds"] * 1000,
        first_call_miss_ms=first_call_miss * 1000,
        cold_hit_start_min_ms=min(hits),
        cold_hit_start_median_ms=statistics.median(hits),
        cold_hit_start_max_ms=max(hits),
    )


# -- 2. the file tools through one session -----------------------------------


def test_fs_tools_ride_one_session(live: SimpleNamespace) -> None:
    b = live.make()
    ex = b.executor
    before = len(live.starts)

    listing = ex.execute("fs_list", {"path": "."}, intent="read")
    assert "alpha.txt" in (listing["result"] or "")
    read = _read(ex, "alpha.txt")
    assert "alpha body" in (read["result"] or "")

    wrote = ex.execute(
        "fs_write", {"path": "new.txt", "content": "hello live\n"}, intent="write"
    )
    assert wrote["outcome"] == "success"
    edited = ex.execute(
        "fs_edit",
        {"path": "new.txt", "old_string": "hello", "new_string": "goodbye"},
        intent="write",
    )
    assert edited["outcome"] == "success"
    assert "goodbye live" in (_read(ex, "new.txt")["result"] or "")

    grep = ex.execute("fs_grep", {"pattern": "beta"}, intent="read")
    assert "beta.md" in (grep["result"] or "")
    glob = ex.execute("fs_glob", {"pattern": "**/*.md"}, intent="read")
    assert "beta.md" in (glob["result"] or "")

    # Stale read-before-write stamp: refused, file untouched.
    stamps = parse_fs_read_stamps(_read(ex, "new.txt")["result"] or "")
    assert stamps is not None
    with pytest.raises(RemoteWorkspaceExecutionError):
        ex.execute(
            "fs_write",
            {"path": "new.txt", "content": "clobber\n", "expected_sha256": "0" * 64},
            intent="write",
        )
    assert parse_fs_read_stamps(_read(ex, "new.txt")["result"] or "") == stamps

    assert len(live.starts) == before + 1, "every call rode one session"
    assert b.cache.status(b.binding_id).state is BindingState.READY
    live.close(b.key)


# -- 3. denylist through the session ------------------------------------------


def test_home_rooted_binding_cannot_read_dot_ssh(live: SimpleNamespace) -> None:
    b = live.make(root=live.home)
    ex = b.executor
    listing = ex.execute("fs_list", {"path": "."}, intent="read")
    assert listing["outcome"] == "success"
    for path in (".ssh/authorized_keys", ".ssh/known_hosts"):
        with pytest.raises(RemoteWorkspaceExecutionError):
            _read(ex, path)
    with pytest.raises(RemoteWorkspaceExecutionError):
        ex.execute("fs_list", {"path": ".ssh"}, intent="read")
    assert b.cache.status(b.binding_id).state is BindingState.READY
    live.close(b.key)


# -- 4. retargeted destination --------------------------------------------------


def test_retargeted_destination_starts_no_session(live: SimpleNamespace) -> None:
    b = live.make(fingerprint="0" * 64)
    before = len(live.starts)
    with pytest.raises(RemoteWorkspaceExecutionError) as caught:
        _read(b.executor, "alpha.txt")
    assert caught.value.code == "destination_changed"
    assert len(live.starts) == before, "no session for a retargeted destination"
    assert b.cache.status(b.binding_id).state is BindingState.BLOCKED
    live.close(b.key)


# -- 5. recreated root -------------------------------------------------------


def test_recreated_root_goes_stale_then_recaptures(live: SimpleNamespace) -> None:
    b = live.make("pinroot")
    ex = b.executor
    assert "pin" in (_read(ex, "p.txt")["result"] or "")
    before = len(live.starts)

    # Build the replacement BEFORE removing the original: ext4 hands a freed
    # inode straight back, so rm + mkdir recreates the SAME (dev, ino) and the
    # pin (correctly, by its identity rule) cannot tell the difference.
    root = shlex.quote(f"{live.scratch}/pinroot")
    fresh = shlex.quote(f"{live.scratch}/pinroot.new")
    _host(
        f"mkdir {fresh} && printf 'again\\n' > {fresh}/p.txt && "
        f"rm -rf -- {root} && mv {fresh} {root}"
    )
    with pytest.raises(RemoteWorkspaceExecutionError) as caught:
        _read(ex, "p.txt")
    assert caught.value.code == "root_pin_failed"
    assert b.cache.status(b.binding_id).state is BindingState.STALE_IDENTITY

    ex.ping()  # re-capture inside the live session
    assert b.cache.status(b.binding_id).state is BindingState.READY
    assert "again" in (_read(ex, "p.txt")["result"] or "")
    assert len(live.starts) == before, "pin failure left the session up"
    live.close(b.key)


# -- 6. host idle exit -------------------------------------------------------


def test_host_idle_exit_then_fresh_session(live: SimpleNamespace) -> None:
    assert _wait_no_workers(10) == [], "earlier sessions left host processes"
    live.settings.value = live.config.ConsoleSshSettings(
        max_concurrent_calls=4, session_idle_s=2
    )
    try:
        b = live.make(grace=1.0)  # host idles out at 2 + 1 s
        _read(b.executor, "alpha.txt")
        before = len(live.starts)
        assert _host_workers(), "session parent should be alive on the host"
        assert _wait_no_workers(15) == [], "host did not idle-exit"
        _read(b.executor, "alpha.txt")
        assert len(live.starts) == before + 1, "next call got a fresh session"
        assert b.cache.status(b.binding_id).state is BindingState.READY
        live.close(b.key)
    finally:
        live.settings.value = live.config.ConsoleSshSettings(max_concurrent_calls=4)


# -- 7. warm latency ---------------------------------------------------------


def _ping_samples(host: str, count: int, out: list[float]) -> None:
    target = host.rsplit("@", 1)[-1]
    try:
        done = subprocess.run(
            ["ping", "-n", "-c", str(count), "-i", "0.2", target],
            capture_output=True,
            text=True,
            timeout=count * 0.2 + 30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return
    out.extend(float(ms) for ms in re.findall(r"time=([\d.]+) ?ms", done.stdout))


def _pct(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def test_warm_call_latency(live: SimpleNamespace) -> None:
    b = live.make()
    ex = b.executor
    for _ in range(10):  # warm-up: session start, identity capture, caches
        _read(ex, "alpha.txt")

    pings: list[float] = []
    pinger = threading.Thread(target=_ping_samples, args=(HOST, 50, pings))
    pinger.start()
    # Back-to-back calls while ping runs alongside (same window). No pacing
    # sleep: an idle gap of ~0.1-0.2 s lets Wi-Fi power saving delay the
    # next packet (measured: ~20 ms back-to-back vs ~70 ms at 0.15 s gaps).
    samples: list[float] = []
    while pinger.is_alive() or len(samples) < 50:
        began = time.perf_counter()
        _read(ex, "alpha.txt")
        samples.append((time.perf_counter() - began) * 1000)
    pinger.join()
    live.close(b.key)

    median = statistics.median(samples)
    RESULTS.update(
        warm_n=len(samples),
        warm_median_ms=median,
        warm_p90_ms=_pct(samples, 0.9),
        ping_n=len(pings),
        ping_median_ms=statistics.median(pings) if pings else None,
        ping_p90_ms=_pct(pings, 0.9) if pings else None,
        target_ms=SPIKE_ECHO_FLOOR_MS + 15,
    )
    assert median < 100, f"warm median {median:.1f} ms"


# -- 8. leftovers + report ------------------------------------------------------


def test_no_leftovers_and_cache_only_in_runtime_dir(live: SimpleNamespace) -> None:
    assert _wait_no_workers(10) == [], "stray python3 -I -c processes on the host"
    uid = _host("id -u").strip()
    listing = _host(
        'cd "$XDG_RUNTIME_DIR/tldw-worker" && pwd && stat -c "%a %u %n" -- *'
    ).splitlines()
    assert listing[0] == f"/run/user/{uid}/tldw-worker"
    assert listing[1:] == [f"600 {uid} {live.bundle_hash}"], listing
    assert _host(f"stat -c %a /run/user/{uid}/tldw-worker").strip() == "700"
    stray = _host(
        f"find /tmp /var/tmp \"$HOME\" -maxdepth 4 -xdev -name '{live.bundle_hash}*' 2>/dev/null; true"
    )
    assert stray.strip() == "", stray

    print("\n== SSH session worker live UAT ==")
    for name, value in RESULTS.items():
        shown = f"{value:.2f}" if isinstance(value, float) else value
        print(f"  {name}: {shown}")
