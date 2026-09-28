import hashlib, json, os, select, stat, subprocess, sys, time, zlib
from pathlib import Path
import pytest
from tldw_chatbook.Tools.build_remote_worker_bundle import build_loader_text, loader_payload
from tldw_chatbook.Tools.remote_workspace_executor import bootstrap_source

MAGIC = b"TLDW-REMOTE-0001"
STUB = b'BUNDLE_SHA256 = "stub-stamp"\nimport sys\ndef serve_session(i, o):\n    assert __name__ != "__main__"\n    return 0\n'
STUB_Z = zlib.compress(STUB)
STUB_HASH = hashlib.sha256(STUB_Z).hexdigest()

def _readline(stream, timeout=10.0):
    """Bounded readline: a subprocess hang fails this test in ``timeout``
    seconds instead of stalling the whole run (a FIFO cache path with a
    blocking, non-O_NONBLOCK open hung the entire suite indefinitely
    during this fix round — this converts that into a fast, clear
    failure regardless of what future code changes do)."""
    ready, _, _ = select.select([stream], [], [], timeout)
    if not ready:
        raise AssertionError(f"no output within {timeout}s — subprocess likely hung")
    return stream.readline()

def _run(env_runtime, cache=True, hash_=STUB_HASH, send_bundle=True):
    env = dict(os.environ)
    env.pop("XDG_RUNTIME_DIR", None)
    if env_runtime is not None:
        env["XDG_RUNTIME_DIR"] = str(env_runtime)
    loader = loader_payload()
    proc = subprocess.Popen([sys.executable, "-I", "-c", bootstrap_source(len(loader))],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
    proc.stdin.write(loader + json.dumps({"hash": hash_, "cache": cache}).encode() + b"\n"); proc.stdin.flush()
    first = _readline(proc.stdout)
    if first.startswith(MAGIC + b"NEED") and send_bundle:
        proc.stdin.write(len(STUB_Z).to_bytes(4, "big") + STUB_Z); proc.stdin.flush()
        second = _readline(proc.stdout)
    else:
        second = b""
    proc.stdin.close(); code = proc.wait(10)
    return first, second, code

def _runtime(tmp_path):
    base = tmp_path / "run"; base.mkdir(mode=0o700); return base

def _spawn(env_runtime=None):
    """Spawn the bootstrap+loader and feed it the loader bytes only.

    Low-level counterpart to ``_run`` for tests that need to control the
    header/length-prefix bytes themselves (truncated/malformed input).
    """
    env = dict(os.environ)
    env.pop("XDG_RUNTIME_DIR", None)
    if env_runtime is not None:
        env["XDG_RUNTIME_DIR"] = str(env_runtime)
    loader = loader_payload()
    proc = subprocess.Popen(
        [sys.executable, "-I", "-c", bootstrap_source(len(loader))],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env,
    )
    proc.stdin.write(loader); proc.stdin.flush()
    return proc

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
    assert second == b"" and code == 3

def test_traversal_hash_exits_3(tmp_path):
    # "../" + 61 hex-lookalike chars is 64 bytes but contains non-hex
    # characters, so the loader's charset check refuses it before any
    # path is ever built from it — never emits NEED.
    first, second, code = _run(_runtime(tmp_path), hash_="../" + "a" * 61)
    assert first == b"" and second == b"" and code == 3

def test_header_without_trailing_newline_exits_3(tmp_path):
    proc = _spawn(_runtime(tmp_path))
    proc.stdin.write(b'{"hash": "' + STUB_HASH.encode() + b'", "cache": true}')
    proc.stdin.close()
    assert proc.wait(10) == 3

def test_malformed_json_header_exits_3(tmp_path):
    proc = _spawn(_runtime(tmp_path))
    proc.stdin.write(b"not json at all\n")
    proc.stdin.close()
    assert proc.wait(10) == 3

def test_header_missing_hash_key_exits_3(tmp_path):
    proc = _spawn(_runtime(tmp_path))
    proc.stdin.write(json.dumps({"cache": True}).encode() + b"\n")
    proc.stdin.close()
    assert proc.wait(10) == 3

def test_short_length_prefix_after_need_exits_3(tmp_path):
    proc = _spawn(_runtime(tmp_path))
    proc.stdin.write(json.dumps({"hash": STUB_HASH, "cache": True}).encode() + b"\n")
    proc.stdin.flush()
    assert _readline(proc.stdout) == MAGIC + b"NEED " + STUB_HASH.encode() + b"\n"
    proc.stdin.write(b"\x00\x00")  # only 2 of the 4 length-prefix bytes
    proc.stdin.close()
    assert proc.wait(10) == 3

def test_declared_length_over_cap_exits_3_before_reading(tmp_path):
    proc = _spawn(_runtime(tmp_path))
    proc.stdin.write(json.dumps({"hash": STUB_HASH, "cache": True}).encode() + b"\n")
    proc.stdin.flush()
    assert _readline(proc.stdout) == MAGIC + b"NEED " + STUB_HASH.encode() + b"\n"
    proc.stdin.write(((8 << 20) + 1).to_bytes(4, "big"))
    proc.stdin.close()
    # No bundle bytes were ever sent; a loader that checked the cap AFTER
    # trying to read them would hang here until the 10s wait times out.
    assert proc.wait(10) == 3

def test_cache_flag_must_be_exactly_true(tmp_path):
    base = _runtime(tmp_path)
    _run(base, cache="yes")  # truthy, but not the JSON boolean `true`
    assert not (base / "tldw-worker").exists()

def test_readonly_cache_dir_is_best_effort_no_crash(tmp_path):
    base = _runtime(tmp_path)
    worker_dir = base / "tldw-worker"; worker_dir.mkdir(mode=0o500)
    try:
        first, second, code = _run(base)
        assert first.startswith(MAGIC + b"NEED")
        assert second == MAGIC + b"READY stub-stamp\n"
        assert code == 0
    finally:
        worker_dir.chmod(0o700)

def test_dir_at_cache_path_is_treated_as_a_miss(tmp_path):
    base = _runtime(tmp_path)
    worker_dir = base / "tldw-worker"; worker_dir.mkdir(mode=0o700)
    (worker_dir / STUB_HASH).mkdir()
    first, second, code = _run(base)
    assert first.startswith(MAGIC + b"NEED")
    assert second == MAGIC + b"READY stub-stamp\n"
    assert code == 0

@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX-only")
def test_fifo_at_cache_path_is_treated_as_a_miss_no_hang(tmp_path):
    base = _runtime(tmp_path)
    worker_dir = base / "tldw-worker"; worker_dir.mkdir(mode=0o700)
    os.mkfifo(worker_dir / STUB_HASH)
    first, second, code = _run(base)
    assert first.startswith(MAGIC + b"NEED")
    assert second == MAGIC + b"READY stub-stamp\n"
    assert code == 0

def test_symlinked_worker_dir_not_used_for_cache(tmp_path):
    base = _runtime(tmp_path)
    real = tmp_path / "elsewhere"; real.mkdir(mode=0o700)
    (base / "tldw-worker").symlink_to(real, target_is_directory=True)
    first, second, code = _run(base)
    assert first.startswith(MAGIC + b"NEED")
    assert second == MAGIC + b"READY stub-stamp\n"
    assert code == 0
    assert list(real.iterdir()) == []

def test_old_tmp_leftover_is_swept_fresh_one_is_not(tmp_path):
    base = _runtime(tmp_path)
    worker_dir = base / "tldw-worker"; worker_dir.mkdir(mode=0o700)
    old_tmp = worker_dir / ".tmp-old"; old_tmp.write_bytes(b"x")
    old_time = time.time() - 3700  # > 10 minutes (well over an hour) old
    os.utime(old_tmp, (old_time, old_time))
    fresh_tmp = worker_dir / ".tmp-fresh"; fresh_tmp.write_bytes(b"y")
    _run(base)
    assert not old_tmp.exists()
    assert fresh_tmp.exists()

def test_loader_is_stdlib_only_and_310_compatible():
    import ast
    tree = ast.parse(build_loader_text())
    roots = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert roots <= set(sys.stdlib_module_names)

def test_loader_parses_on_python_310_floor():
    from Tests.Tools.test_remote_worker_bundle import _python_310_interpreter

    interpreter = _python_310_interpreter()
    if interpreter is None:
        pytest.skip("no 3.10 interpreter — CI must install one (uv python install 3.10)")
    result = subprocess.run(
        [interpreter, "-c", "import ast,sys; ast.parse(sys.stdin.read())"],
        input=build_loader_text(),
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"loader does not parse under {interpreter}:\n{result.stderr}"
    )
