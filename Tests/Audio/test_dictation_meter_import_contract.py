"""Real Speech preimport and PCM callbacks avoid optional NumPy initialization."""

from __future__ import annotations

import json
import math
from pathlib import Path
import queue
import struct
import subprocess
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[2]

_PROTECTED_SETUP = r"""
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys

repo, private = (Path(value).resolve() for value in sys.argv[1:3])
entry = sys.argv[3]
assert not any(name == prefix or name.startswith(prefix + ".")
               for name in sys.modules
               for prefix in ("tldw_chatbook", "tldw_profile_core"))
core = repo / "packages/tldw_profile_core/src"
sys.path.insert(0, str(repo))
sys.path.insert(1, str(core))
for name, expected in {
    "Tests": repo / "Tests/__init__.py",
    "tldw_chatbook": repo / "tldw_chatbook/__init__.py",
    "tldw_profile_core": core / "tldw_profile_core/__init__.py",
}.items():
    assert Path(importlib.util.find_spec(name).origin).resolve() == expected.resolve()
from Tests import real_profile_guard
real_profile_guard.install()  # Original profile protection BEFORE redirection.
assert real_profile_guard._real_home().resolve() != private / "home"
keep = {"SYSTEMROOT", "WINDIR", "SYSTEMDRIVE", "COMSPEC", "PATH", "PATHEXT",
        "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMW6432", "PROGRAMDATA",
        "PROCESSOR_ARCHITECTURE", "TLDW_TEST_REAL_HOME"}
for name in list(os.environ):
    if name.upper() not in keep:
        os.environ.pop(name)
for name, relative in {
    "HOME": "home", "USERPROFILE": "home", "XDG_CONFIG_HOME": "config",
    "XDG_DATA_HOME": "data", "XDG_CACHE_HOME": "cache", "TEMP": "temp",
    "TMP": "temp", "APPDATA": "appdata", "LOCALAPPDATA": "localappdata",
}.items():
    location = private / relative
    location.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(location)
os.environ.update(TLDW_CONFIG_PATH=str(private / "config/config.toml"),
                  TLDW_TEST_CONFIG_ROOT=str(private), PYTHONUTF8="1",
                  PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
                  HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
# Resolve only the installed source location; do not import urllib3 or bind
# its function before the real Speech-route import reaches it.
assert "urllib3.util.connection" not in sys.modules
urllib3_spec = importlib.util.find_spec("urllib3")
expected_ipv6_source = Path(urllib3_spec.origin).resolve().parent / "util/connection.py"
assert "urllib3.util.connection" not in sys.modules
network = []
refused_ipv6_probes = 0
def audit(event, args):
    global refused_ipv6_probes
    if event in {"socket.connect", "socket.getaddrinfo", "socket.bind", "socket.sendto"}:
        frame = sys._getframe(1)
        partial = sys.modules.get("urllib3.util.connection")
        probe = getattr(partial, "_has_ipv6", None)
        known_ipv6_probe = (
            event == "socket.bind"
            and frame.f_code is getattr(probe, "__code__", None)
            and frame.f_globals is vars(partial)
            and Path(frame.f_code.co_filename).resolve() == expected_ipv6_source
            and Path(partial.__file__).resolve() == expected_ipv6_source
        )
        if known_ipv6_probe:
            refused_ipv6_probes += 1
        else:
            network.append(event)
        raise RuntimeError("test_network_refused")
sys.addaudithook(audit)
from tldw_chatbook.Backup_Recovery import bootstrap
assert bootstrap.startup_permission(bootstrap.effective_config_path(), bootstrap.default_bootstrap_root())[0]
"""

_NUMPY_GUARD = r"""
assert not any(name == "numpy" or name.startswith("numpy.") for name in sys.modules)
bootstrap_imports = sys.modules["importlib._bootstrap"]
original_find_spec = bootstrap_imports._find_spec.__code__
original_load = bootstrap_imports._find_and_load_unlocked.__code__
original_query = importlib.util.find_spec.__code__
attempted = []
class RejectNumPy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            caller = sys._getframe(1)
            if (caller.f_code is original_find_spec
                    and caller.f_back.f_code is original_query):
                return None  # Exact stdlib capability query; no initialization.
            attempted.append(fullname)
            assert caller.f_code is original_find_spec
            assert caller.f_back.f_code is original_load
            raise RuntimeError("NUMPY_INITIALIZATION_ATTEMPT_BLOCKED")
sys.meta_path.insert(0, RejectNumPy())
"""

_COLD_ENTRYPOINT = (
    _PROTECTED_SETUP
    + _NUMPY_GUARD
    + r"""
print("COLD_REAL_ENTRYPOINT_BEGIN:" + entry, flush=True)
try:
    if entry == "module":
        from tldw_chatbook.Audio.dictation_service_lazy import LazyLiveDictationService
        assert LazyLiveDictationService.__module__ == "tldw_chatbook.Audio.dictation_service_lazy"
    elif entry == "speech_route":
        from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_route
        route = resolve_screen_route("stts")
        assert route is not None and route.module_path == "tldw_chatbook.UI.Screens.stts_screen"
        screen_type = route.load_screen_class()
        assert screen_type is not None and screen_type.__name__ == "STTSScreen"
        assert screen_type.__module__ == route.module_path
    else:
        raise AssertionError("Unknown_test_entrypoint")
finally:
    print("NUMPY_ATTEMPTS:" + json.dumps(attempted), flush=True)
assert not attempted
assert not any(name == "numpy" or name.startswith("numpy.") for name in sys.modules)
# The exact original urllib3 probe is classified, never permitted.
# Every other bind, connect, DNS or datagram attempt fails the proof.
assert not network, network
print("REFUSED_IPV6_PROBES:" + str(refused_ipv6_probes), flush=True)
assert not real_profile_guard.take_violations()
(private / "import-observation.json").write_text(json.dumps({
    "entry": entry, "numpy_attempts": attempted,
    "refused_ipv6_probe_count": refused_ipv6_probes,
    "unexpected_network_attempts": network,
}), encoding="utf-8")
print("COLD_REAL_ENTRYPOINT_WITHOUT_NUMPY_OK", flush=True)
"""
)


def _run_child(script, tmp_path, entry):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-X",
            "utf8",
            "-B",
            "-c",
            script,
            str(ROOT),
            str(tmp_path),
            entry,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr[-6000:]
    return result.stdout


@pytest.mark.parametrize("entry", ["module", "speech_route"])
def test_real_fresh_entrypoint_never_attempts_numpy_initialization(tmp_path, entry):
    stdout = _run_child(_COLD_ENTRYPOINT, tmp_path, entry)
    assert "COLD_REAL_ENTRYPOINT_BEGIN:" + entry in stdout
    assert "NUMPY_ATTEMPTS:[]" in stdout
    assert "COLD_REAL_ENTRYPOINT_WITHOUT_NUMPY_OK" in stdout
    observed = json.loads(
        (tmp_path / "import-observation.json").read_text(encoding="utf-8")
    )
    assert observed["entry"] == entry and not observed["numpy_attempts"]
    assert isinstance(observed["refused_ipv6_probe_count"], int)
    assert not isinstance(observed["refused_ipv6_probe_count"], bool)
    assert not observed["unexpected_network_attempts"]


def test_fresh_numpy_guard_allows_spec_query_but_refuses_real_import(tmp_path):
    stdout = _run_child(
        _PROTECTED_SETUP
        + _NUMPY_GUARD
        + r"""
importlib.util.find_spec("numpy")  # Installed and absent NumPy are both valid.
assert not attempted
assert not any(name == "numpy" or name.startswith("numpy.") for name in sys.modules)
try:
    __import__("numpy")
except RuntimeError as error:
    assert error.args == ("NUMPY_INITIALIZATION_ATTEMPT_BLOCKED",)
else:
    raise AssertionError("Real_NumPy_import_not_refused")
assert attempted == ["numpy"]
assert not any(name == "numpy" or name.startswith("numpy.") for name in sys.modules)
assert not network
assert not real_profile_guard.take_violations()
print("NUMPY_QUERY_ALLOWED_REAL_IMPORT_REFUSED", flush=True)
""",
        tmp_path,
        "guard_control",
    )
    assert "NUMPY_QUERY_ALLOWED_REAL_IMPORT_REFUSED" in stdout


def _meter_service():
    """Reuse the existing dictation callback fixture; constructor is unchanged."""
    from tldw_chatbook.Audio.dictation_service_lazy import LazyLiveDictationService

    service = LazyLiveDictationService.__new__(LazyLiveDictationService)
    service.audio_buffer = []
    service.buffer_lock = threading.Lock()
    service.captured_bytes = 0
    service._current_audio_level = 0.75
    service.last_speech_time = 0
    service._capture_saw_first_frame = False
    service.processing_queue = queue.Queue()
    return service


class _FakeRecorder:
    def __init__(self, callback):
        self.callback = callback

    def deliver(self, chunk):
        self.callback(chunk)


@pytest.mark.parametrize(
    "samples",
    [
        pytest.param((0,) * 320, id="silence"),
        pytest.param((1, -1), id="small_signed"),
        pytest.param((-32768,), id="negative_full_scale"),
        pytest.param((32767,), id="positive_full_scale"),
        pytest.param((-32768, 32767, 0, 16384), id="mixed"),
        pytest.param((-32768, 32767) * 16000, id="maximum_chunk"),
    ],
)
def test_real_pcm_callback_preserves_normalized_rms_and_flow(samples):
    service = _meter_service()
    recorder = _FakeRecorder(service._audio_callback)
    chunk = struct.pack(f"{len(samples)}h", *samples)
    expected = min(
        1.0,
        math.sqrt(math.fsum(value * value for value in samples) / len(samples))
        / 32768.0,
    )
    recorder.deliver(chunk)
    assert service.get_audio_level() == pytest.approx(expected, rel=1e-12, abs=1e-15)
    assert service.audio_buffer == [chunk]
    assert service.captured_bytes == len(chunk)
    kind, queued = service.processing_queue.get_nowait()
    assert kind == "audio" and queued is chunk
    assert service.processing_queue.empty()
    assert service._capture_saw_first_frame is True
    assert service.last_speech_time > 0


@pytest.mark.parametrize(
    "chunk", [b"", b"\x01", b"\x01\x02\x03"], ids=["empty", "one_byte", "odd_bytes"]
)
def test_real_callback_empty_or_malformed_meter_resets_zero_without_dropping_chunk(
    chunk,
):
    service = _meter_service()
    recorder = _FakeRecorder(service._audio_callback)
    recorder.deliver(chunk)
    assert service.get_audio_level() == 0.0
    assert service.audio_buffer == [chunk]
    assert service.captured_bytes == len(chunk)
    kind, queued = service.processing_queue.get_nowait()
    assert kind == "audio" and queued is chunk
    assert service.processing_queue.empty()
