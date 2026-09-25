"""Loopback harness + ping tests (Phase 1e, Task 9).

``run_bundle_loopback`` executes the COMMITTED bundle — not the local
worker module — exactly the way the ssh transport will (Task 11):
``python -I -c '<bootstrap>'`` with the zlib-compressed bundle followed
by the request JSON on stdin. Loopback swaps only the process spawn, so
every test here exercises the real remote code path: bootstrap,
decompress, exec, magic-prefixed frames, and the two-line response
contract ``_parse_worker_output`` established.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import tldw_chatbook.Tools.remote_worker_bundle as bundle_module
from tldw_chatbook.Tools import remote_workspace_executor as executor_module
from tldw_chatbook.Tools.build_remote_worker_bundle import expected_bundle_stamp
from tldw_chatbook.Tools.remote_workspace_executor import (
    RemoteWorkspaceExecutionError,
    RemoteWorkspaceLoopbackError,
    RemoteWorkspaceToolExecutor,
    RESPONSE_MAGIC,
    bootstrap_source,
    parse_fs_read_stamps,
    run_bundle_loopback,
    run_bundle_loopback_frames,
)
from tldw_chatbook.Tools.remote_worker_bundle import BUNDLE_SHA256 as BUNDLE_STAMP
from tldw_chatbook.Tools.workspace_tool_protocol import MAX_RESPONSE_BYTES

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_BUNDLE_PATH = (
    _REPOSITORY_ROOT / "tldw_chatbook" / "Tools" / "remote_worker_bundle.py"
)

#: The spec's bootstrap, verbatim, with N = 12345 embedded (Task 9
#: binding; ``bootstrap_source`` must reproduce this byte for byte).
_EXACT_BOOTSTRAP = (
    'import sys,zlib;exec(compile(zlib.decompress('
    'sys.stdin.buffer.read(12345)),"b","exec"))'
    'if sys.version_info>=(3,10)else(sys.stderr.write('
    '"tldw-worker:python3.10+:found:%d.%d\\n"%sys.version_info[:2]),'
    "sys.exit(76))"
)

#: The spec's shell-safety charset for the bootstrap, plus ``-``: the
#: spec's own fixed string contains ``tldw-worker``, so its prose
#: character list omitted a character its exact string requires. The
#: exact string is authoritative; the class still excludes every
#: shell-active character (no quote/dollar/backtick/bang/globs/controls).
_BOOTSTRAP_CHARSET = re.compile(r"[A-Za-z0-9\-_.\"(),:;<>=+%\\\[\] ]")

#: The artifact's stamp assignment — its final line, and the ONLY byte
#: range outside the stamp's own coverage (review fix: the entry logic
#: moved into the stamped region so a rewritten tail can no longer
#: launch divergent code while echoing a matching stamp).
_STAMP_ASSIGNMENT_MARKER = b"\nBUNDLE_SHA256 = _enter_worker_exchange("

_PING_PROBE_IDENTITY = {"device": 0, "inode": 0, "mode": 0, "reparse": False}


def _workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "alpha.txt").write_text("alpha body\n", encoding="utf-8")
    (root / "beta.bin").write_bytes(b"\x00binary\x00")
    nested = root / "nested"
    nested.mkdir()
    (nested / "gamma.md").write_text("gamma\nbody\n", encoding="utf-8")
    return root


def _request(
    root: Path,
    operation: str,
    arguments: dict[str, Any],
    *,
    chain: dict[str, Any] | None = None,
    intent: str = "read",
) -> dict[str, Any]:
    """One wire-legal request dict, pinning ``root``'s real chain."""
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
        "intent": intent,
        "root_locator": locator,
        "root_identity": identities[0],
        "ancestor_identities": identities,
        "arguments": arguments,
        "timeout_seconds": 30,
        "output_max_bytes": MAX_RESPONSE_BYTES,
    }


def _ping_chain(root: Path) -> dict[str, Any]:
    """Run one loopback ping and return its payload verbatim."""
    result = run_bundle_loopback(root, _request(root, "ping", {}))
    assert result["outcome"] == "success", result
    return json.loads(result["result"])


# ---------------------------------------------------------------------------
# The fixed bootstrap string
# ---------------------------------------------------------------------------


def test_bootstrap_source_is_the_spec_string_verbatim() -> None:
    assert bootstrap_source(12345) == _EXACT_BOOTSTRAP


def test_bootstrap_charset_is_shell_safe() -> None:
    for character in bootstrap_source(40960):
        assert _BOOTSTRAP_CHARSET.fullmatch(character), (
            f"bootstrap contains charset-violating character {character!r}"
        )


def test_response_magic_is_single_sourced_and_pinned() -> None:
    assert RESPONSE_MAGIC == b"TLDW-REMOTE-0001"
    assert len(RESPONSE_MAGIC) == 16
    assert executor_module.RESPONSE_MAGIC is bundle_module.RESPONSE_MAGIC


# ---------------------------------------------------------------------------
# fs ops through the loopback
# ---------------------------------------------------------------------------


def test_fs_list_roundtrip_returns_entries(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    chain = _ping_chain(root)

    result = run_bundle_loopback(
        root, _request(root, "fs_list", {"path": ".", "sensitive_exclusions": []}, chain=chain)
    )

    assert result["outcome"] == "success"
    assert result["code"] == "ok"
    assert result["error"] is None
    listing = result["result"] or ""
    assert "alpha.txt" in listing
    assert "nested/" in listing
    assert "beta.bin" in listing


def test_fs_read_returns_content_sha256_and_size(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    chain = _ping_chain(root)
    body = "gamma\nbody\n"
    expected_digest = hashlib.sha256(body.encode("utf-8")).hexdigest()

    result = run_bundle_loopback(
        root,
        _request(root, "fs_read", {"path": "nested/gamma.md", "sensitive_exclusions": []}, chain=chain),
    )

    assert result["outcome"] == "success"
    stamps = parse_fs_read_stamps(result["result"] or "")
    assert stamps is not None, "fs_read result lacks CAS stamp tail"
    digest, size = stamps
    assert digest == expected_digest
    assert size == len(body.encode("utf-8"))
    # The content lines survive ahead of the stamp tail.
    assert "gamma" in (result["result"] or "")


def test_parse_fs_read_stamps_rejects_stampless_text() -> None:
    assert parse_fs_read_stamps("1\tjust content\n") is None
    assert parse_fs_read_stamps("") is None


@pytest.mark.parametrize(
    "tail",
    [
        # Non-hex digest characters (review minor: pin the shape gate).
        "\nsha256: " + "z" * 64 + "\nsize: 5",
        # Uppercase hex is not the wire form.
        "\nsha256: " + "AB" * 32 + "\nsize: 5",
        # Too short.
        "\nsha256: " + "a" * 63 + "\nsize: 5",
        # Size must be digits, and the tail must be last.
        "\nsha256: " + "ab" * 32 + "\nsize: five",
        "\nsha256: " + "ab" * 32 + "\nsize: 5\ntrailing",
        # Wrong order / missing size line.
        "\nsize: 5\nsha256: " + "ab" * 32,
        "\nsha256: " + "ab" * 32,
    ],
)
def test_parse_fs_read_stamps_rejects_malformed_tails(tail: str) -> None:
    assert parse_fs_read_stamps("1\tbody" + tail) is None


# ---------------------------------------------------------------------------
# ping
# ---------------------------------------------------------------------------


def test_ping_reports_full_identity_chain(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    payload = _ping_chain(root)

    assert set(payload) == {
        "identity_chain",
        "canonical_path",
        "python_version",
        "bundle_sha256",
    }
    canonical = Path(payload["canonical_path"])
    assert canonical == root.resolve()
    chain = payload["identity_chain"]
    assert isinstance(chain, list) and chain
    # Root first, then every ancestor, ending at /.
    assert [Path(entry[0]) for entry in chain] == [
        canonical,
        *canonical.parents,
    ]
    # Each entry is [path, st_dev, st_ino, st_mode] and matches the disk.
    for entry in chain:
        assert len(entry) == 4
        info = os.lstat(entry[0])
        assert entry[1] == info.st_dev
        assert entry[2] == info.st_ino
        assert entry[3] == info.st_mode
    assert re.fullmatch(r"3\.\d+\.\d+", payload["python_version"])


def test_ping_bundle_sha256_matches_the_artifact_stamp(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    payload = _ping_chain(root)

    artifact = _BUNDLE_PATH.read_bytes()
    # Independent derivation: the stamp is the SHA-256 of the artifact's
    # bytes above the (sole) stamp-assignment line.
    boundary = artifact.rindex(_STAMP_ASSIGNMENT_MARKER)
    assert payload["bundle_sha256"] == hashlib.sha256(
        artifact[: boundary + 1]
    ).hexdigest()
    assert payload["bundle_sha256"] == expected_bundle_stamp(artifact)
    assert payload["bundle_sha256"] == BUNDLE_STAMP


def test_ping_on_unsafe_root_fails_without_admitted_marker(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    symlinked = tmp_path / "symlinked-root"
    symlinked.symlink_to(outside, target_is_directory=True)

    frames = run_bundle_loopback_frames(symlinked, _request(symlinked, "ping", {}))

    assert len(frames) == 1
    assert frames[0]["outcome"] == "failure"
    assert frames[0]["code"] == "root_pin_failed"


# ---------------------------------------------------------------------------
# magic stripping under stdout noise
# ---------------------------------------------------------------------------


def test_magic_strip_survives_leading_newline_noise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _workspace(tmp_path)
    real_spawn = executor_module._spawn_loopback_worker

    def noisy_spawn(
        payload: bytes, *, budget_seconds: float
    ) -> subprocess.CompletedProcess[bytes]:
        completed = real_spawn(payload, budget_seconds=budget_seconds)
        return SimpleNamespace(
            stdout=b"noise\nwith newlines\n" + completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )

    monkeypatch.setattr(executor_module, "_spawn_loopback_worker", noisy_spawn)

    result = run_bundle_loopback(
        root, _request(root, "fs_list", {"path": ".", "sensitive_exclusions": []}, chain=_ping_chain(root))
    )

    assert result["outcome"] == "success"
    assert "alpha.txt" in (result["result"] or "")


def test_magicless_stdout_is_a_typed_noise_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _workspace(tmp_path)

    def garbage_spawn(
        payload: bytes, *, budget_seconds: float
    ) -> subprocess.CompletedProcess[bytes]:
        return SimpleNamespace(
            stdout=b"x" * 512, stderr=b"", returncode=0
        )

    monkeypatch.setattr(executor_module, "_spawn_loopback_worker", garbage_spawn)

    with pytest.raises(RemoteWorkspaceLoopbackError) as raised:
        run_bundle_loopback(root, _request(root, "ping", {}))
    assert raised.value.code == "stdout_noise"


# ---------------------------------------------------------------------------
# identity round-trip and staleness
# ---------------------------------------------------------------------------


def test_ping_identity_round_trip_pins_for_fs_ops(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    chain = _ping_chain(root)

    frames = run_bundle_loopback_frames(
        root,
        _request(
            root,
            "fs_read",
            {"path": "alpha.txt", "sensitive_exclusions": []},
            chain=chain,
        ),
    )

    assert len(frames) == 2
    admitted, terminal = frames
    assert admitted["outcome"] == "admitted"
    assert admitted["code"] == "root_pinned"
    assert terminal["outcome"] == "success"
    assert "alpha body" in (terminal["result"] or "")


def test_stale_identity_fails_pinning_with_no_admitted_marker(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path)
    chain = _ping_chain(root)

    # Recreate the root: same path, new inode — the captured chain is stale.
    shutil.rmtree(root)
    root.mkdir()
    (root / "alpha.txt").write_text("replaced body\n", encoding="utf-8")

    frames = run_bundle_loopback_frames(
        root,
        _request(
            root,
            "fs_read",
            {"path": "alpha.txt", "sensitive_exclusions": []},
            chain=chain,
        ),
    )
    assert len(frames) == 1, "pin failure must not surface an admitted marker"
    assert frames[0]["outcome"] == "failure"
    assert frames[0]["code"] == "root_pin_failed"

    executor = RemoteWorkspaceToolExecutor(
        root, root_locator=str(root), identity_chain_source=lambda: chain
    )
    with pytest.raises(RemoteWorkspaceExecutionError) as raised:
        executor.execute(
            "fs_read",
            {"path": "alpha.txt", "sensitive_exclusions": []},
            intent="read",
        )
    assert raised.value.code == "root_pin_failed"
    assert raised.value.admitted is False


# ---------------------------------------------------------------------------
# RemoteWorkspaceToolExecutor (loopback mode)
# ---------------------------------------------------------------------------


def _executor_for(root: Path, chain: dict[str, Any]) -> RemoteWorkspaceToolExecutor:
    return RemoteWorkspaceToolExecutor(
        root,
        root_locator=str(chain["canonical_path"]),
        identity_chain_source=lambda: chain,
    )


def test_executor_execute_returns_the_final_response_dict(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    executor = _executor_for(root, _ping_chain(root))

    result = executor.execute(
        "fs_list", {"path": ".", "sensitive_exclusions": []}, intent="read"
    )

    assert result["outcome"] == "success"
    assert result["code"] == "ok"
    assert set(result) == {
        "version",
        "operation_id",
        "outcome",
        "code",
        "result",
        "error",
        "elapsed_ms",
        "truncated",
        "cleanup_proven",
    }
    assert "alpha.txt" in (result["result"] or "")


def test_executor_ping_returns_verified_payload(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    executor = RemoteWorkspaceToolExecutor(
        root, root_locator=str(root), identity_chain_source=lambda: None
    )

    payload = executor.ping()

    assert payload["canonical_path"] == str(root.resolve())
    assert payload["bundle_sha256"] == expected_bundle_stamp(
        _BUNDLE_PATH.read_bytes()
    )


def test_executor_ping_rejects_a_bundle_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _workspace(tmp_path)
    executor = RemoteWorkspaceToolExecutor(
        root, root_locator=str(root), identity_chain_source=lambda: None
    )
    monkeypatch.setattr(
        executor_module, "expected_bundle_stamp", lambda data: "00" * 32
    )

    with pytest.raises(RemoteWorkspaceExecutionError) as raised:
        executor.ping()
    assert raised.value.code == "bundle_mismatch"
    assert raised.value.admitted is False


def test_executor_rejects_unknown_tool(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    executor = _executor_for(root, _ping_chain(root))

    with pytest.raises(RemoteWorkspaceExecutionError) as raised:
        executor.execute(
            "fs_definitely_not_an_op", {}, intent="read"
        )
    assert raised.value.code == "invalid_request"


def test_executor_post_admission_failure_carries_admitted_flag(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path)
    executor = _executor_for(root, _ping_chain(root))

    with pytest.raises(RemoteWorkspaceExecutionError) as raised:
        executor.execute(
            "fs_read",
            {"path": "missing.txt", "sensitive_exclusions": []},
            intent="read",
        )
    assert raised.value.admitted is True
    assert raised.value.code == "tool_failure"


def test_loopback_requires_a_directory_root(tmp_path: Path) -> None:
    with pytest.raises(RemoteWorkspaceLoopbackError) as raised:
        run_bundle_loopback(tmp_path / "absent", _request(tmp_path, "ping", {}))
    assert raised.value.code == "loopback_root_missing"


# ---------------------------------------------------------------------------
# bundle artifact stamp integrity
# ---------------------------------------------------------------------------


def _dispatch_read(root: Path, relative: str, *, offset: int = 1) -> str:
    """Run one in-process pinned fs_read through the shared dispatch."""
    from tldw_chatbook.Tools.workspace_root_pin import pin_workspace_root
    from tldw_chatbook.Tools.workspace_tool_dispatch import (
        execute_pinned_operation,
    )
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain

    chain = capture_directory_chain(root)
    request = SimpleNamespace(
        operation="fs_read",
        arguments={
            "path": relative,
            "offset": offset,
            "sensitive_exclusions": [],
        },
    )
    with pin_workspace_root(chain.canonical_root, chain) as pinned:
        return execute_pinned_operation(request, pinned)


def test_fs_read_hashes_in_the_same_read_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review minor 2: content and CAS stamps come from ONE read.

    Reopening the target after the content read produced a torn pair
    under concurrent modification; the read path now hashes the very
    bytes it rendered. Pinned by counting the target's materialising
    reads (``Path.read_bytes`` and ``open`` both instrumented, so the
    pin holds whichever mechanism the read uses) during one dispatched
    fs_read: exactly one.
    """
    root = _workspace(tmp_path)
    target = (root / "alpha.txt").resolve()

    reads: list[str] = []
    real_open = builtins.open
    real_read_bytes = Path.read_bytes

    def counting_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            resolved = Path(file).resolve()
        except (TypeError, ValueError, OSError):
            resolved = None
        if resolved == target:
            reads.append("open")
        return real_open(file, *args, **kwargs)

    def counting_read_bytes(self: Path) -> bytes:
        if self.resolve() == target:
            reads.append("read_bytes")
        return real_read_bytes(self)

    monkeypatch.setattr(builtins, "open", counting_open)
    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)
    result = _dispatch_read(root, "alpha.txt")
    monkeypatch.undo()

    assert len(reads) == 1, "fs_read must materialise its target exactly once"
    body = "alpha body\n".encode("utf-8")
    stamps = parse_fs_read_stamps(result)
    assert stamps == (hashlib.sha256(body).hexdigest(), len(body))


def test_fs_read_stamps_cover_empty_and_paged_reads(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    (root / "empty.txt").write_text("", encoding="utf-8")
    (root / "paged.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")

    empty = _dispatch_read(root, "empty.txt")
    assert empty.startswith("(empty file)")
    assert parse_fs_read_stamps(empty) == (hashlib.sha256(b"").hexdigest(), 0)

    data = "one\ntwo\nthree\n".encode("utf-8")
    full = _dispatch_read(root, "paged.txt")
    assert parse_fs_read_stamps(full) == (
        hashlib.sha256(data).hexdigest(),
        len(data),
    )

    past_end = _dispatch_read(root, "paged.txt", offset=99)
    assert "past end of file" in past_end
    assert parse_fs_read_stamps(past_end) == (
        hashlib.sha256(data).hexdigest(),
        len(data),
    )


def test_committed_bundle_stamp_matches_artifact_prefix() -> None:
    artifact = _BUNDLE_PATH.read_bytes()
    assert expected_bundle_stamp(artifact) == BUNDLE_STAMP
    # The stamp assignment occurs exactly once in the artifact.
    assert artifact.count(_STAMP_ASSIGNMENT_MARKER) == 1


def test_stamp_covers_the_executable_entry_guard() -> None:
    """Review fix pin: only the stamp's own assignment line is uncovered.

    The bootstrap entry logic (``_enter_worker_exchange``) lives INSIDE
    the hashed prefix; everything after the stamp assignment is the
    assignment line itself. A divergent bundle can no longer append or
    rewrite executable tail code while echoing a matching stamp — under
    the bootstrap the entry helper's ``SystemExit`` fires from the
    assignment line, so nothing after it ever executes.
    """
    artifact = _BUNDLE_PATH.read_bytes()
    boundary = artifact.rindex(_STAMP_ASSIGNMENT_MARKER)
    prefix = artifact[: boundary + 1]
    suffix = artifact[boundary + 1 :]

    assert b"def _enter_worker_exchange" in prefix
    assert b'raise SystemExit(main(sys.stdin.buffer, bundle_sha256=stamp))' in prefix
    # Nothing but the stamp's own assignment follows: one line plus its
    # terminating newline, closing over the literal digest.
    assert suffix.count(b"\n") == 1
    assert suffix.endswith(b'")\n')
    assert hashlib.sha256(prefix).hexdigest() in suffix.decode()


def test_loopback_spawns_isolated_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loopback spawns exactly [sys.executable, -I, -c, bootstrap]."""
    captured: dict[str, Any] = {}
    real_run = subprocess.run

    def spying_run(argv: list[str], **kwargs: Any) -> Any:
        captured["argv"] = list(argv)
        return real_run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", spying_run)
    root = _workspace(tmp_path)
    run_bundle_loopback(root, _request(root, "ping", {}))

    argv = captured["argv"]
    assert argv[0] == sys.executable
    assert argv[1] == "-I"
    assert argv[2] == "-c"
    assert argv[3].startswith("import sys,zlib;exec(compile(")
