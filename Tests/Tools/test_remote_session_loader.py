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
