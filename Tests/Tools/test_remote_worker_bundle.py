"""Guards for the single-file remote worker bundle (Phase 1d, Task 8).

Four invariants, each with its own gate:

* **Drift**: rebuilding the bundle from the real dependency modules must
  reproduce the committed ``Tools/remote_worker_bundle.py`` byte for byte.
  The artifact is generated, but it is COMMITTED — a silent regeneration
  difference means the artifact and the sources have drifted apart, and
  every consumer (Task 9's loopback harness first) pins against the
  committed bytes.
* **Charset**: the bundle runs on a bare remote interpreter, so EVERY
  import it contains — module-level or nested — must have a standard-
  library root. The builder rewrites non-stdlib lazy imports into loud
  ``ImportError`` raises, so an un-rewritten third-party import here is
  a builder bug, not a cosmetic finding.
* **3.10 floor**: the bundle must PARSE under a real Python 3.10 (the
  oldest interpreter the spec lets a remote host run). ``ast.parse`` on
  the dev interpreter proves nothing about 3.10; the guard shells out to
  a real 3.10 located via PATH or ``uv python find``, and skips LOUDLY
  when neither exists (CI installs one).
* **Runtime probe**: failure paths driven end-to-end in a bare ``-I``
  subprocess with no site-packages — loguru, Metrics and portalocker are
  genuinely absent there, which is the only environment that proves the
  bundle's failure branches never reach for them.
"""

from __future__ import annotations

import ast
import base64
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tldw_chatbook.Tools.build_remote_worker_bundle import build_bundle_text
from tldw_chatbook.Tools.remote_sensitive_paths import REMOTE_SENSITIVE_PATHS

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_BUNDLE_PATH = _REPOSITORY_ROOT / "tldw_chatbook" / "Tools" / "remote_worker_bundle.py"

#: Ruling 1 (corrected): exactly 16 bytes, defined once in the bundle's IO
#: adapter; Task 11 mirrors the literal.
_EXPECTED_RESPONSE_MAGIC = b"TLDW-REMOTE-0001"

#: Ruling 6: the remote-home-relative denylist, verbatim.
_EXPECTED_REMOTE_DENYLIST = (
    ".ssh",
    ".aws",
    ".gnupg",
    ".config/gcloud",
    ".kube",
    ".docker",
    ".netrc",
)


def _load_bundle_module() -> Any:
    """Import the committed artifact standalone, outside the package chain.

    The bundle is defined to never need the ``tldw_chatbook`` package
    machinery; loading it via ``spec_from_file_location`` keeps that
    honest (a package-__init__ requirement would fail here, loudly).
    """
    spec = importlib.util.spec_from_file_location("remote_worker_bundle", _BUNDLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses with slots=True resolve their defining module through
    # sys.modules; an unregistered namespace breaks class creation.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_remote_denylist_is_the_ruling_set() -> None:
    assert REMOTE_SENSITIVE_PATHS == _EXPECTED_REMOTE_DENYLIST
    for entry in REMOTE_SENSITIVE_PATHS:
        assert not entry.startswith("/"), f"{entry} must be home-relative"
        assert "\x00" not in entry


def test_bundle_rebuild_matches_committed_artifact() -> None:
    """Drift guard: regenerate and demand byte equality with the artifact."""
    assert _BUNDLE_PATH.exists(), (
        "tldw_chatbook/Tools/remote_worker_bundle.py is missing; regenerate it "
        "with: python -m tldw_chatbook.Tools.build_remote_worker_bundle"
    )
    assert build_bundle_text() == _BUNDLE_PATH.read_text(encoding="utf-8"), (
        "remote_worker_bundle.py is stale relative to its source modules; "
        "regenerate with: python -m tldw_chatbook.Tools.build_remote_worker_bundle"
    )


def test_bundle_imports_are_stdlib_only() -> None:
    tree = ast.parse(_BUNDLE_PATH.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders.extend(
                alias.name
                for alias in node.names
                if alias.name.split(".")[0] not in sys.stdlib_module_names
            )
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, (
                f"relative import survived the builder at line {node.lineno}: "
                f"{ast.unparse(node)}"
            )
            if (node.module or "").split(".")[0] not in sys.stdlib_module_names:
                offenders.append(node.module or "<relative>")
    assert not offenders, (
        "non-stdlib import roots in the bundle: " + ", ".join(sorted(set(offenders)))
    )


def _python_310_interpreter() -> str | None:
    """Locate a real Python 3.10 via PATH then ``uv python find``."""
    exact = shutil.which("python3.10")
    if exact:
        return exact
    uv = shutil.which("uv")
    if uv is None:
        return None
    found = subprocess.run(
        [uv, "python", "find", "3.10"],
        capture_output=True,
        check=False,
        text=True,
        env={**os.environ, "UV_PYTHON_DOWNLOADS": "never"},
    )
    candidate = found.stdout.strip()
    if found.returncode != 0 or not candidate:
        return None
    return candidate


def test_bundle_parses_on_python_310_floor() -> None:
    interpreter = _python_310_interpreter()
    if interpreter is None:
        pytest.skip("no 3.10 interpreter — CI must install one (uv python install 3.10)")
    result = subprocess.run(
        [
            interpreter,
            "-c",
            "import ast,sys; ast.parse(open(sys.argv[1]).read())",
            str(_BUNDLE_PATH),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"bundle does not parse under {interpreter}:\n{result.stderr}"
    )


def test_bundle_entry_surface() -> None:
    bundle = _load_bundle_module()
    assert bundle.RESPONSE_MAGIC == _EXPECTED_RESPONSE_MAGIC
    # Length is pinned separately so a miscounted literal (the original
    # ruling shipped a 15-byte spelling of a "16-byte" magic) can never
    # drift back in unnoticed.
    assert len(bundle.RESPONSE_MAGIC) == 16
    assert len(_EXPECTED_RESPONSE_MAGIC) == 16
    assert callable(bundle.main)
    assert callable(bundle.split_magic)
    assert callable(bundle.arm_watchdog)
    assert callable(bundle.register_temp)
    assert callable(bundle.unregister_temp)

    had_magic, stripped = bundle.split_magic(
        bundle.RESPONSE_MAGIC + b'{"k": 1}\n'
    )
    assert had_magic is True
    assert stripped == b'{"k": 1}\n'
    had_magic, untouched = bundle.split_magic(b'{"k": 1}\n')
    assert had_magic is False
    assert untouched == b'{"k": 1}\n'

    with pytest.raises(NotImplementedError):
        bundle.arm_watchdog(30, [])

    bundle.register_temp("probe.tmp")
    try:
        assert "probe.tmp" in bundle.TEMP_REGISTRY
    finally:
        bundle.unregister_temp("probe.tmp")
    assert "probe.tmp" not in bundle.TEMP_REGISTRY


def _probe_frame(operation: str, arguments: dict[str, Any], root: str) -> bytes:
    """One wire-legal request frame, per ``workspace_wire_decode``'s contract."""
    identity = {"device": 1, "inode": 2, "mode": 16877, "reparse": False}
    return json.dumps(
        {
            "version": 1,
            "operation_id": "probe-op",
            "operation": operation,
            "intent": "read",
            "root_locator": root,
            "root_identity": identity,
            "ancestor_identities": [identity],
            "arguments": arguments,
            "timeout_seconds": 30,
            "output_max_bytes": 1024,
        },
        separators=(",", ":"),
    ).encode("utf-8")


_PROBE_DRIVER = """
import base64, io, json, sys

bundle_path, payload_b64 = sys.argv[1], sys.argv[2]
import types

module = types.ModuleType("remote_worker_bundle")
sys.modules["remote_worker_bundle"] = module
with open(bundle_path, "r", encoding="utf-8") as handle:
    source = handle.read()
exec(compile(source, "remote_worker_bundle.py", "exec"), module.__dict__)
namespace = module.__dict__

captured = io.BytesIO()
class _Stdout:
    buffer = captured
_stdout, sys.stdout = sys.stdout, _Stdout()
try:
    exit_code = namespace["main"](io.BytesIO(base64.b64decode(payload_b64)))
finally:
    sys.stdout = _stdout
print("PROBE" + json.dumps({"rc": exit_code, "out": captured.getvalue().decode()}))
"""


def _run_bundle_probe(payload: bytes) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            _PROBE_DRIVER,
            str(_BUNDLE_PATH),
            base64.b64encode(payload).decode("ascii"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(_REPOSITORY_ROOT),
    )
    assert completed.returncode == 0, (
        "bare-interpreter probe crashed:\n"
        f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
    )
    marker = next(
        line for line in completed.stdout.splitlines() if line.startswith("PROBE")
    )
    return json.loads(marker[len("PROBE") :])


def _terminal_frame(output: str) -> dict[str, Any]:
    """Decode the LAST magic-prefixed response line emitted by the bundle."""
    lines = [line for line in output.splitlines() if line]
    assert lines, "bundle emitted no response frames"
    raw = lines[-1].encode("utf-8")
    assert raw.startswith(_EXPECTED_RESPONSE_MAGIC), (
        f"response frame lacks RESPONSE_MAGIC prefix: {raw!r}"
    )
    frame = raw[len(_EXPECTED_RESPONSE_MAGIC) :]
    return json.loads(frame)


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        (b"this is not json at all", "invalid_request"),
        (
            _probe_frame(
                "fs_definitely_not_an_op", {"path": ".", "sensitive_exclusions": []},
                "/definitely-not-a-real-tldw-root",
            ),
            "invalid_request",
        ),
        (
            _probe_frame(
                "fs_list",
                {"path": ".", "sensitive_exclusions": []},
                "/definitely-not-a-real-tldw-root",
            ),
            "root_pin_failed",
        ),
    ],
    ids=["garbage-bytes", "unknown-op", "unpinnable-root"],
)
def test_bundle_failure_paths_run_in_a_bare_interpreter(
    payload: bytes, expected_code: str
) -> None:
    """loguru/Metrics/portalocker are absent under ``-I``; the paths still work."""
    assert _BUNDLE_PATH.exists(), "committed bundle artifact is missing"
    result = _run_bundle_probe(payload)
    assert result["rc"] == 2, result
    frame = _terminal_frame(result["out"])
    assert frame["outcome"] == "failure"
    assert frame["code"] == expected_code
    assert frame["operation_id"] in {"probe-op", "unknown"}
