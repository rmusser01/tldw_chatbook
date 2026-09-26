"""Task 16 (Phase 3b): the wrong-file hazard, closed by worker-reported CAS.

The hazard: a ``RemoteRoot`` names a path that ALSO exists on the laptop
(a synced copy, a same-spelled scratch dir). Before Phase 3b the
provider's CAS machinery resolved and hashed targets against the LAPTOP
filesystem -- so a remote run could read, hash, and ledger-stamp the
laptop copy while believing it stamped the remote file.

This suite pins the closed form: for a ``RemoteRoot`` authority, every
stamp the ledger sees comes from WORKER-REPORTED values (the fs_read /
fs_write / fs_edit / fs_patch response strings the remote worker
produced), never from the provider's own disk. The proof is mechanical:

* the loopback executor (Task 9's harness, no ssh) runs the REAL
  committed bundle in a subprocess against a local directory standing in
  for the remote host's filesystem -- its responses are genuine
  worker-reported stamps;
* a laptop directory with the SAME PATH as the descriptor's root holds
  DIFFERENT bytes, so any provider-side hashing would produce a
  DIFFERENT stamp than the worker's;
* tripwires on ``builtins.open`` / ``os.stat`` / ``os.lstat`` /
  ``Path.read_bytes`` / ``Path.stat`` fail the test on any in-process
  access to the watched paths (the worker subprocess is unaffected);
* a spy on ``local_tool_provider._hash_file`` proves the laptop hashing
  entry point is never reached for remote authorities.

Invoke-level remote dispatch is Task 17/18's seam (result redaction and
the executor interface); these tests drive the five migrated CAS sites
directly with real worker responses, which is exactly the surface this
task owns.
"""

from __future__ import annotations

import builtins
import hashlib
import os
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

import tldw_chatbook.Agents.local_tool_provider as local_tool_provider
from tldw_chatbook.Agents.local_tool_provider import (
    _REMOTE_NOW_DISPLAY,
    LocalToolProvider,
    RunAdmittedWorkspaceRoot,
)
from tldw_chatbook.Agents.run_context import use_run_id
from tldw_chatbook.Tools.remote_root_types import RemoteRoot
from tldw_chatbook.Tools.remote_workspace_executor import (
    RemoteWorkspaceExecutionError,
    RemoteWorkspaceToolExecutor,
    parse_fs_read_stamps,
)

#: The "remote host's" file content (lives ONLY under the loopback root).
REMOTE_BODY = "REMOTE server bytes\n"
#: The laptop copy at the SAME path spelling, with DIFFERENT bytes: any
#: provider-side hashing of the laptop file yields a DIFFERENT stamp.
LAPTOP_BODY = "LAPTOP decoy bytes (wrong file)\n"

REMOTE_DIGEST = hashlib.sha256(REMOTE_BODY.encode("utf-8")).hexdigest()
LAPTOP_DIGEST = hashlib.sha256(LAPTOP_BODY.encode("utf-8")).hexdigest()

RUN = "hazard-run"


@pytest.fixture(autouse=True)
def _neutral_config_gates(monkeypatch: pytest.MonkeyPatch):
    """Pin the provider's construction-time config gate reads to defaults.

    ``_default_specs`` reads the web-deep-search / ask-user config gates
    at construction; under this repo's test conftest the config
    bootstrap fails closed with ``RecoveryRequired`` in this environment
    (a pre-existing failure that predates this task and blanks the whole
    provider suite at base). Neutral defaults keep THIS suite about the
    CAS machinery, mirroring the established patch pattern in
    ``Tests/Agents/test_local_tool_provider.py``.
    """
    monkeypatch.setattr(
        local_tool_provider,
        "get_cli_setting",
        lambda section, key=None, default=None: default,
    )


class _DiskTripwire:
    """Fail on any in-process filesystem access under the watched paths.

    Watches the laptop decoy directory tree plus every FILE under the
    loopback stand-in root. The stand-in root directory itself stays
    watchable: the loopback harness's own preflight (``root.is_dir()``)
    stats exactly that path in-process, and that stat belongs to the
    transport harness, not to the provider code under test.
    """

    def __init__(self, laptop_root: Path, remote_root: Path) -> None:
        self._prefixes = (
            str(laptop_root),
            str(remote_root) + os.sep,
        )
        self.hits: list[str] = []

    def _violates(self, target: object) -> bool:
        if isinstance(target, int):
            return False
        try:
            text = os.fspath(target)
        except TypeError:
            return False
        if not isinstance(text, str):
            return False
        return any(text.startswith(prefix) for prefix in self._prefixes)

    def check(self, label: str, target: object) -> None:
        if self._violates(target):
            self.hits.append(f"{label}: {os.fspath(target)}")
            raise AssertionError(
                f"provider touched the laptop/stand-in disk in-process: "
                f"{label} {os.fspath(target)!r}"
            )


def _install_tripwires(
    monkeypatch: pytest.MonkeyPatch, tripwire: _DiskTripwire
) -> None:
    real_open = builtins.open
    real_stat = os.stat
    real_lstat = os.lstat
    real_read_bytes = Path.read_bytes
    real_path_stat = Path.stat

    def guarding_open(file, *args, **kwargs):
        tripwire.check("open", file)
        return real_open(file, *args, **kwargs)

    def guarding_stat(path, *args, **kwargs):
        tripwire.check("os.stat", path)
        return real_stat(path, *args, **kwargs)

    def guarding_lstat(path, *args, **kwargs):
        tripwire.check("os.lstat", path)
        return real_lstat(path, *args, **kwargs)

    def guarding_read_bytes(self):
        tripwire.check("Path.read_bytes", self)
        return real_read_bytes(self)

    def guarding_path_stat(self, *args, **kwargs):
        tripwire.check("Path.stat", self)
        return real_path_stat(self, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarding_open)
    monkeypatch.setattr(os, "stat", guarding_stat)
    monkeypatch.setattr(os, "lstat", guarding_lstat)
    monkeypatch.setattr(Path, "read_bytes", guarding_read_bytes)
    monkeypatch.setattr(Path, "stat", guarding_path_stat)


class _HashSpy:
    """Record every ``_hash_file`` call (the laptop hashing entry point)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[object] = []
        real_hash = local_tool_provider._hash_file

        def spying_hash(path):
            self.calls.append(path)
            return real_hash(path)

        monkeypatch.setattr(local_tool_provider, "_hash_file", spying_hash)


def _loopback_executor(
    standin: Path, *, exclusions=None
) -> RemoteWorkspaceToolExecutor:
    """A loopback executor pinned to the stand-in root's real identity.

    ``exclusions`` (Task 17) attaches the binding's serialized exclusion
    source so the executor's injection seam carries them on every op.
    """
    prober = RemoteWorkspaceToolExecutor(
        standin,
        root_locator=str(standin),
        identity_chain_source=lambda: None,
    )
    payload = prober.ping()
    chain = {
        "identity_chain": payload["identity_chain"],
        "canonical_path": payload["canonical_path"],
    }
    kwargs = {}
    if exclusions is not None:
        kwargs["sensitive_exclusions"] = exclusions
    return RemoteWorkspaceToolExecutor(
        standin,
        root_locator=str(standin),
        identity_chain_source=lambda: chain,
        **kwargs,
    )


def _remote_authority(
    descriptor: RemoteRoot, executor: RemoteWorkspaceToolExecutor
) -> RunAdmittedWorkspaceRoot:
    return RunAdmittedWorkspaceRoot(
        workspace_id="workspace-1",
        binding_id="binding-1",
        alias="w",
        root=descriptor,
        locator_fingerprint="fingerprint-w",
        root_identity=((str(descriptor.root), 1, 2, 0o40755),),
        allow_write=True,
        guard=lambda _write: True,
        workspace_executor=executor,
    )


class _Hazard:
    """The assembled wrong-file scenario with every tripwire armed."""

    def __init__(
        self,
        laptop_root: Path,
        standin: Path,
        descriptor: RemoteRoot,
        executor: RemoteWorkspaceToolExecutor,
        provider: LocalToolProvider,
        tripwire: _DiskTripwire,
        spy: _HashSpy,
    ) -> None:
        self.laptop_root = laptop_root
        self.standin = standin
        self.descriptor = descriptor
        self.executor = executor
        self.provider = provider
        self.tripwire = tripwire
        self.spy = spy

    def key(self, relative: str) -> str:
        return local_tool_provider._remote_ledger_key(
            self.descriptor, relative, intent="write"
        )

    def stamp(self, relative: str):
        return self.provider._read_ledger.stamp_for(RUN, self.key(relative))

    def assert_clean(self) -> None:
        """No laptop hashing, no in-process disk touch, ever."""
        assert self.spy.calls == []
        assert self.tripwire.hits == []


def _hazard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Hazard:
    standin = tmp_path / "remote-fs"
    standin.mkdir()
    (standin / "doc.txt").write_text(REMOTE_BODY, encoding="utf-8")

    laptop_root = tmp_path / "w"
    laptop_root.mkdir()
    (laptop_root / "doc.txt").write_text(LAPTOP_BODY, encoding="utf-8")

    descriptor = RemoteRoot(
        alias="w",
        canonical_locator="ssh://devbox" + str(laptop_root),
        root=PurePosixPath(str(laptop_root)),
        binding_id="binding-1",
    )
    executor = _loopback_executor(standin)
    base = tmp_path / "unrelated-base"
    base.mkdir()
    provider = LocalToolProvider(
        workspace_root=base,
        admitted_roots=(_remote_authority(descriptor, executor),),
    )
    tripwire = _DiskTripwire(laptop_root=laptop_root, remote_root=standin)
    spy = _HashSpy(monkeypatch)
    _install_tripwires(monkeypatch, tripwire)
    return _Hazard(
        laptop_root, standin, descriptor, executor, provider, tripwire, spy
    )


def _read_doc(hazard: _Hazard, relative: str = "doc.txt") -> str:
    """One real worker fs_read; records it into the run's ledger."""
    with use_run_id(RUN):
        response = hazard.executor.execute(
            "fs_read",
            {"path": relative, "sensitive_exclusions": []},
            intent="read",
        )
        hazard.provider._record_fs_read_observation(
            {"path": relative}, hazard.descriptor, worker_result=response["result"]
        )
    return response["result"]


# ---------------------------------------------------------------------------
# Site 1: _record_fs_read_observation
# ---------------------------------------------------------------------------


def test_remote_fs_read_stamps_ledger_from_worker_response(
    tmp_path, monkeypatch
):
    hazard = _hazard(tmp_path, monkeypatch)
    assert REMOTE_DIGEST != LAPTOP_DIGEST  # the decoy would stamp differently

    result = _read_doc(hazard)

    worker_stamps = parse_fs_read_stamps(result)
    assert worker_stamps == (REMOTE_DIGEST, len(REMOTE_BODY.encode("utf-8")))
    with use_run_id(RUN):
        stamp = hazard.stamp("doc.txt")

    assert stamp is not None
    assert (stamp.sha256, stamp.size) == worker_stamps
    # NOT the laptop decoy's digest: a provider that hashed the laptop
    # copy would have stamped LAPTOP_DIGEST here.
    assert stamp.sha256 != LAPTOP_DIGEST
    hazard.assert_clean()


def test_remote_missing_read_records_absent_from_worker_failure(
    tmp_path, monkeypatch
):
    hazard = _hazard(tmp_path, monkeypatch)

    with pytest.raises(RemoteWorkspaceExecutionError) as raised:
        hazard.executor.execute(
            "fs_read",
            {"path": "ghost.txt", "sensitive_exclusions": []},
            intent="read",
        )
    assert "file not found" in str(raised.value)

    with use_run_id(RUN):
        hazard.provider._record_fs_read_observation(
            {"path": "ghost.txt"},
            hazard.descriptor,
            worker_failure=str(raised.value),
        )
        stamp = hazard.stamp("ghost.txt")

    assert stamp is not None and stamp.is_absent
    hazard.assert_clean()


def test_remote_binary_read_failure_records_nothing(tmp_path, monkeypatch):
    """A read that failed for a non-absence reason records no stamp.

    Local semantics: refusal -> nothing, absent -> ABSENT. The remote
    worker reports a binary file as a read FAILURE whose text is not
    "file not found", so nothing is stamped (a deliberately weaker blind
    write, never a false stale refusal).
    """
    hazard = _hazard(tmp_path, monkeypatch)
    (hazard.standin / "blob.bin").write_bytes(b"\x00binary\x00")

    with pytest.raises(RemoteWorkspaceExecutionError) as raised:
        hazard.executor.execute(
            "fs_read",
            {"path": "blob.bin", "sensitive_exclusions": []},
            intent="read",
        )
    assert "binary" in str(raised.value)

    with use_run_id(RUN):
        hazard.provider._record_fs_read_observation(
            {"path": "blob.bin"},
            hazard.descriptor,
            worker_failure=str(raised.value),
        )

    assert hazard.provider._read_ledger._by_run.get(RUN) in (None, {})
    hazard.assert_clean()


def test_remote_stampless_worker_result_records_nothing(tmp_path, monkeypatch):
    hazard = _hazard(tmp_path, monkeypatch)

    with use_run_id(RUN):
        hazard.provider._record_fs_read_observation(
            {"path": "doc.txt"}, hazard.descriptor, worker_result="1\tbody with no tail"
        )

    assert hazard.provider._read_ledger._by_run.get(RUN) in (None, {})


def test_remote_read_of_refused_path_records_nothing(tmp_path, monkeypatch):
    hazard = _hazard(tmp_path, monkeypatch)

    with use_run_id(RUN):
        hazard.provider._record_fs_read_observation(
            {"path": "../escape.txt"}, hazard.descriptor, worker_result="irrelevant"
        )

    assert hazard.provider._read_ledger._by_run.get(RUN) in (None, {})


# ---------------------------------------------------------------------------
# Site 2: _fs_write_guard_injection
# ---------------------------------------------------------------------------


def test_remote_guard_injection_arms_from_worker_stamp(tmp_path, monkeypatch):
    hazard = _hazard(tmp_path, monkeypatch)

    with use_run_id(RUN):
        _read_doc(hazard)
        guard = hazard.provider._fs_write_guard_injection(
            {"path": "doc.txt", "content": "mine\n"}, hazard.descriptor
        )

    assert guard is not None
    injected, stamp, resolved = guard
    assert injected["expected_sha256"] == REMOTE_DIGEST
    assert stamp.sha256 == REMOTE_DIGEST
    # No laptop path exists for a remote target -- the refusal path must
    # never be handed one.
    assert resolved is None
    hazard.assert_clean()


def test_remote_guard_injection_absent_stamp_arms_expected_absent(
    tmp_path, monkeypatch
):
    hazard = _hazard(tmp_path, monkeypatch)
    with use_run_id(RUN):
        hazard.provider._record_fs_read_observation(
            {"path": "fresh.txt"},
            hazard.descriptor,
            worker_failure="file not found: fresh.txt",
        )
        guard = hazard.provider._fs_write_guard_injection(
            {"path": "fresh.txt", "content": "first\n"}, hazard.descriptor
        )

    assert guard is not None
    injected, _stamp, resolved = guard
    assert injected["expected_absent"] is True
    assert resolved is None


def test_remote_guard_injection_without_prior_read_returns_none(
    tmp_path, monkeypatch
):
    hazard = _hazard(tmp_path, monkeypatch)
    with use_run_id(RUN):
        guard = hazard.provider._fs_write_guard_injection(
            {"path": "never-read.txt", "content": "x\n"}, hazard.descriptor
        )
    assert guard is None


# ---------------------------------------------------------------------------
# Site 3: _stale_targets_for
# ---------------------------------------------------------------------------


def test_remote_stale_edit_detected_with_worker_reported_now(
    tmp_path, monkeypatch
):
    hazard = _hazard(tmp_path, monkeypatch)
    peer_body = "PEER overwrote the remote file\n"
    peer_digest = hashlib.sha256(peer_body.encode("utf-8")).hexdigest()

    with use_run_id(RUN):
        _read_doc(hazard)
        # A peer (the worker itself) replaces the remote file.
        hazard.executor.execute(
            "fs_write",
            {
                "path": "doc.txt",
                "content": peer_body,
                "sensitive_exclusions": [],
            },
            intent="write",
        )
        stale = hazard.provider._stale_targets_for(
            "fs_edit",
            {"path": "doc.txt", "old_string": "REMOTE", "new_string": "MINE"},
            hazard.descriptor,
            executor=hazard.executor,
        )

    assert len(stale) == 1
    shown, stamp, now = stale[0]
    assert shown == "doc.txt"
    assert stamp.sha256 == REMOTE_DIGEST
    # The "now" is the WORKER-reported digest of the peer's content --
    # fetched by a worker read, never a laptop hash.
    assert now == (peer_digest, len(peer_body.encode("utf-8")))

    message = hazard.provider._stale_write_refusal(shown, stamp, now)
    assert "Stale write refused" in message
    peer_size = len(peer_body.encode("utf-8"))
    assert f"now {peer_digest[:8]}/{peer_size}" in message
    hazard.assert_clean()


def test_remote_stale_check_skips_when_worker_probe_fails_unclearly(
    tmp_path, monkeypatch
):
    """A target whose worker probe fails for a non-absence reason is not
    stale: the handler surfaces the real error (fail-open parity with
    the local refused-path ``continue``)."""
    hazard = _hazard(tmp_path, monkeypatch)
    (hazard.standin / "blob.bin").write_bytes(b"\x00binary\x00")

    with use_run_id(RUN):
        hazard.provider._read_ledger.record_present(
            RUN, hazard.key("blob.bin"), "0" * 64, 1
        )
        stale = hazard.provider._stale_targets_for(
            "fs_edit",
            {"path": "blob.bin", "old_string": "x", "new_string": "y"},
            hazard.descriptor,
            executor=hazard.executor,
        )

    assert stale == []
    hazard.assert_clean()


def test_remote_stale_check_without_executor_fails_open(tmp_path, monkeypatch):
    hazard = _hazard(tmp_path, monkeypatch)
    with use_run_id(RUN):
        hazard.provider._read_ledger.record_present(
            RUN, hazard.key("doc.txt"), "0" * 64, 1
        )
        stale = hazard.provider._stale_targets_for(
            "fs_edit",
            {"path": "doc.txt", "old_string": "x", "new_string": "y"},
            hazard.descriptor,
            executor=None,
        )
    assert stale == []


# ---------------------------------------------------------------------------
# Site 4: _update_ledger_after_write
# ---------------------------------------------------------------------------


def test_remote_post_write_restamp_uses_write_response(tmp_path, monkeypatch):
    hazard = _hazard(tmp_path, monkeypatch)
    new_body = "REMOTE v2 written by worker\n"

    with use_run_id(RUN):
        response = hazard.executor.execute(
            "fs_write",
            {
                "path": "doc.txt",
                "content": new_body,
                "sensitive_exclusions": [],
            },
            intent="write",
        )
        stamps = parse_fs_read_stamps(response["result"] or "")
        assert stamps == (
            hashlib.sha256(new_body.encode("utf-8")).hexdigest(),
            len(new_body.encode("utf-8")),
        )
        hazard.provider._update_ledger_after_write(
            "fs_write",
            {"path": "doc.txt", "content": new_body},
            hazard.descriptor,
            result_text=response["result"],
        )
        stamp = hazard.stamp("doc.txt")

    assert stamp is not None
    assert (stamp.sha256, stamp.size) == stamps
    hazard.assert_clean()


def test_remote_post_edit_restamp_uses_edit_response(tmp_path, monkeypatch):
    hazard = _hazard(tmp_path, monkeypatch)
    edited_body = REMOTE_BODY.replace("REMOTE", "EDITED")

    with use_run_id(RUN):
        response = hazard.executor.execute(
            "fs_edit",
            {
                "path": "doc.txt",
                "old_string": "REMOTE",
                "new_string": "EDITED",
                "sensitive_exclusions": [],
            },
            intent="write",
        )
        stamps = parse_fs_read_stamps(response["result"] or "")
        assert stamps == (
            hashlib.sha256(edited_body.encode("utf-8")).hexdigest(),
            len(edited_body.encode("utf-8")),
        )
        hazard.provider._update_ledger_after_write(
            "fs_edit",
            {"path": "doc.txt", "old_string": "REMOTE", "new_string": "EDITED"},
            hazard.descriptor,
            result_text=response["result"],
        )
        stamp = hazard.stamp("doc.txt")

    assert stamp is not None
    assert (stamp.sha256, stamp.size) == stamps
    hazard.assert_clean()


def test_remote_patch_restamp_uses_per_target_stamps(tmp_path, monkeypatch):
    hazard = _hazard(tmp_path, monkeypatch)
    (hazard.standin / "second.txt").write_text("second body\n", encoding="utf-8")
    diff = (
        "--- a/doc.txt\n+++ b/doc.txt\n@@ -1 +1 @@\n-REMOTE server bytes\n"
        "+PATCHED server bytes\n"
        "--- a/second.txt\n+++ b/second.txt\n@@ -1 +1 @@\n-second body\n"
        "+second patched\n"
    )
    expected = {
        "doc.txt": "PATCHED server bytes\n".encode("utf-8"),
        "second.txt": "second patched\n".encode("utf-8"),
    }

    with use_run_id(RUN):
        response = hazard.executor.execute(
            "fs_patch",
            {
                "diff": diff,
                # The loopback executor passes arguments verbatim (the
                # production builder computes this field), so the target
                # set rides the request explicitly.
                "targets": ["doc.txt", "second.txt"],
                "sensitive_exclusions": [],
            },
            intent="write",
        )
        hazard.provider._update_ledger_after_write(
            "fs_patch",
            {"diff": diff},
            hazard.descriptor,
            result_text=response["result"],
        )
        for relative, payload in expected.items():
            stamp = hazard.stamp(relative)
            assert stamp is not None, f"no re-stamp recorded for {relative}"
            assert stamp.sha256 == hashlib.sha256(payload).hexdigest()
            assert stamp.size == len(payload)

    hazard.assert_clean()


def test_remote_write_restamp_without_response_falls_back_to_content_arg(
    tmp_path, monkeypatch
):
    """fs_write parity: with no response tail available, the stamp is the
    CONTENT ARG digest (in-memory, zero disk) -- the local path's exact
    source for fs_write."""
    hazard = _hazard(tmp_path, monkeypatch)
    body = "stamped from the argument\n"

    with use_run_id(RUN):
        hazard.provider._update_ledger_after_write(
            "fs_write",
            {"path": "doc.txt", "content": body},
            hazard.descriptor,
            result_text=None,
        )
        stamp = hazard.stamp("doc.txt")

    assert stamp is not None
    assert stamp.sha256 == hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert stamp.size == len(body.encode("utf-8"))
    hazard.assert_clean()


# ---------------------------------------------------------------------------
# Site 5: _stale_write_refusal
# ---------------------------------------------------------------------------


def test_remote_stale_refusal_now_is_the_remote_marker(tmp_path, monkeypatch):
    """Chosen rule (documented on the provider): a remote refusal shows
    "(remote)" for the now-value -- the triggering worker failure carries
    no fresh stamps, and fetching one would be an extra round trip whose
    value is already stale by the time it renders."""
    hazard = _hazard(tmp_path, monkeypatch)

    stamp = SimpleNamespace(
        sha256=REMOTE_DIGEST, size=len(REMOTE_BODY.encode("utf-8"))
    )
    message = hazard.provider._stale_write_refusal(
        "doc.txt", stamp, _REMOTE_NOW_DISPLAY
    )

    assert "Stale write refused" in message
    remote_size = len(REMOTE_BODY.encode("utf-8"))
    assert f"was {REMOTE_DIGEST[:8]}/{remote_size}" in message
    assert f"now {_REMOTE_NOW_DISPLAY}" in message
    # The refusal never embedded the laptop decoy's digest.
    assert LAPTOP_DIGEST[:8] not in message


def test_local_stale_refusal_still_hashes_the_file(tmp_path, monkeypatch):
    """LocalRoot keeps its exact refusal shape: now = the laptop file's
    current digest (byte-identical to the pre-Phase-3b behaviour)."""
    hazard = _hazard(tmp_path, monkeypatch)
    local_root = tmp_path / "local-root"
    local_root.mkdir()
    target = local_root / "f.txt"
    target.write_text("local now\n", encoding="utf-8")
    digest = hashlib.sha256(b"local now\n").hexdigest()

    stamp = SimpleNamespace(sha256="0" * 64, size=1)
    message = hazard.provider._stale_write_refusal(
        str(target), stamp, local_tool_provider._hash_file(target)
    )

    assert f"now {digest[:8]}/10" in message
    assert hazard.spy.calls == [target]


# ---------------------------------------------------------------------------
# _hash_file is LocalRoot-only
# ---------------------------------------------------------------------------


def test_hash_file_refuses_a_remote_root_loudly():
    descriptor = RemoteRoot(
        alias="w",
        canonical_locator="ssh://devbox/w",
        root=PurePosixPath("/w"),
        binding_id="b",
    )
    with pytest.raises(TypeError, match="remote root reached laptop-disk path"):
        local_tool_provider._hash_file(descriptor)


def test_remote_ledger_key_is_lexical_and_confined():
    descriptor = RemoteRoot(
        alias="w",
        canonical_locator="ssh://devbox/w",
        root=PurePosixPath("/srv/work"),
        binding_id="b",
    )
    key = local_tool_provider._remote_ledger_key(
        descriptor, "sub/./file.txt", intent="write"
    )
    assert key == os.path.normcase("/srv/work/sub/file.txt")
    assert (
        local_tool_provider._remote_ledger_key(
            descriptor, "../escape.txt", intent="write"
        )
        is None
    )
    assert (
        local_tool_provider._remote_ledger_key(
            descriptor, ".git/config", intent="write"
        )
        is None
    )


# ---------------------------------------------------------------------------
# Task 17 (Phase 3c): invoke-level remote dispatch -- carried obligations.
# Result redaction no longer walls remote authorities; the CAS chain
# (read-stamp -> write-precondition) works end-to-end through the
# provider's own invoke(); the remote "now" probe carries the binding's
# real serialized exclusions instead of a hardcoded empty list.
# ---------------------------------------------------------------------------


class _ResultTextExecutor:
    """Adapt the transport executor's wire-dict return to the provider's
    ``str`` call surface (the spec handlers feed the returned value
    straight into result bounding).

    Task 18's composition owns the production form of this adapter; the
    invoke-level obligation here builds the minimal equivalent: the
    result frame's ``result`` text, failures raised verbatim.
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    def execute(self, operation: str, arguments: dict, *, intent: str) -> str:
        response = self._inner.execute(operation, arguments, intent=intent)
        return response["result"] or ""

    def ping(self):
        return self._inner.ping()


def _invoke_hazard(tmp_path, monkeypatch, *, exclusions=None):
    """The wrong-file scenario assembled for provider-level invoke().

    Same premise as ``_hazard`` plus two Task 17 pieces: the executor
    carries the binding's serialized exclusions (the ssh transport's
    injection seam), and the provider's resolver allows the call so
    ``invoke()`` reaches dispatch without an approval detour.
    """
    from tldw_chatbook.MCP.permission_store import EffectiveToolState
    from tldw_chatbook.Utils.sensitive_paths import SensitiveExclusion

    standin = tmp_path / "remote-fs"
    standin.mkdir()
    (standin / "doc.txt").write_text(REMOTE_BODY, encoding="utf-8")
    locator_uri = "ssh://devbox" + str(tmp_path / "w")
    (standin / "leaky.txt").write_text(
        f"see {locator_uri}/etc/secret\n", encoding="utf-8"
    )

    laptop_root = tmp_path / "w"
    laptop_root.mkdir()
    (laptop_root / "doc.txt").write_text(LAPTOP_BODY, encoding="utf-8")

    descriptor = RemoteRoot(
        alias="w",
        canonical_locator=locator_uri,
        root=PurePosixPath(str(laptop_root)),
        binding_id="binding-1",
    )
    source = (
        exclusions
        if exclusions is not None
        else (lambda: (SensitiveExclusion("subtree", "standin-secrets"),))
    )
    executor = _loopback_executor(standin, exclusions=source)
    base = tmp_path / "unrelated-base"
    base.mkdir()
    provider = LocalToolProvider(
        workspace_root=base,
        admitted_roots=(
            RunAdmittedWorkspaceRoot(
                workspace_id="workspace-1",
                binding_id="binding-1",
                alias="w",
                root=descriptor,
                locator_fingerprint="fingerprint-w",
                root_identity=((str(descriptor.root), 1, 2, 0o40755),),
                allow_write=True,
                guard=lambda _write: True,
                workspace_executor=_ResultTextExecutor(executor),
            ),
        ),
        resolve_state=lambda hub: EffectiveToolState(
            state="allow", origin="test"
        ),
    )
    return SimpleNamespace(
        standin=standin,
        laptop_root=laptop_root,
        descriptor=descriptor,
        executor=executor,
        provider=provider,
        locator_uri=locator_uri,
    )


def test_invoke_level_remote_fs_read_stamps_ledger_from_worker(tmp_path, monkeypatch):
    """Carried obligation (a): a REAL provider.invoke() fs_read through a
    remote authority + loopback executor -- redaction no longer walls
    remote dispatch, the content is the worker's (never the laptop
    decoy), and the ledger stamp is the worker-reported digest."""
    hazard = _invoke_hazard(tmp_path, monkeypatch)

    with use_run_id(RUN):
        result = hazard.provider.invoke("local:fs_read", {"path": "doc.txt"})

    assert result.ok, result.error
    assert "REMOTE server bytes" in result.content
    assert LAPTOP_BODY not in result.content
    key = local_tool_provider._remote_ledger_key(
        hazard.descriptor, "doc.txt", intent="read"
    )
    with use_run_id(RUN):
        stamp = hazard.provider._read_ledger.stamp_for(RUN, key)
    assert stamp is not None
    assert stamp.sha256 == REMOTE_DIGEST
    assert stamp.sha256 != LAPTOP_DIGEST


def test_invoke_level_remote_write_cas_end_to_end(tmp_path, monkeypatch):
    """Carried obligation (a), second half: invoke-level write CAS. The
    read stamps the ledger from the worker; a CALLER-supplied wrong
    precondition surfaces the worker's typed refusal; an out-of-band
    change trips the provider's own injected guard (armed from the
    worker-stamped ledger) into the stale-refusal; a matching write
    lands on the STAND-IN filesystem only."""
    hazard = _invoke_hazard(tmp_path, monkeypatch)

    with use_run_id(RUN):
        read = hazard.provider.invoke("local:fs_read", {"path": "doc.txt"})
        assert read.ok, read.error
        wrong_precondition = hazard.provider.invoke(
            "local:fs_write",
            {
                "path": "doc.txt",
                "content": "attacker bytes\n",
                "expected_sha256": LAPTOP_DIGEST,
            },
        )
        # Caller-supplied precondition: the worker's typed refusal comes
        # back as the tool error (the provider's relabel applies to its
        # OWN injected guard below).
        assert not wrong_precondition.ok
        assert "precondition failed" in (wrong_precondition.error or "")

    assert (hazard.standin / "doc.txt").read_text(encoding="utf-8") == REMOTE_BODY
    assert (hazard.laptop_root / "doc.txt").read_text(encoding="utf-8") == LAPTOP_BODY

    # Out-of-band remote mutation: the next invoke-level write must trip
    # the provider's stale guard (its ledger stamp is the worker's read
    # digest, its "now" probe reports the changed file).
    (hazard.standin / "doc.txt").write_text("SOMEONE ELSE WROTE\n", encoding="utf-8")
    with use_run_id(RUN):
        stale = hazard.provider.invoke(
            "local:fs_write",
            {"path": "doc.txt", "content": "attacker bytes\n"},
        )
        assert stale.outcome == "blocked", stale
    assert (hazard.standin / "doc.txt").read_text(encoding="utf-8") == (
        "SOMEONE ELSE WROTE\n"
    )

    with use_run_id(RUN):
        reread = hazard.provider.invoke("local:fs_read", {"path": "doc.txt"})
        assert reread.ok, reread.error
        applied = hazard.provider.invoke(
            "local:fs_write", {"path": "doc.txt", "content": "REMOTE v2\n"}
        )
        assert applied.ok, applied.error

    assert (hazard.standin / "doc.txt").read_text(encoding="utf-8") == "REMOTE v2\n"
    assert (hazard.laptop_root / "doc.txt").read_text(encoding="utf-8") == LAPTOP_BODY


def test_invoke_level_remote_result_redacts_locator_lexically(tmp_path, monkeypatch):
    """Task 17 redaction: for a RemoteRoot the provider redacts by LEXICAL
    match against the descriptor's locator forms -- no laptop disk. The
    worker result carries the URI; the model never sees it."""
    hazard = _invoke_hazard(tmp_path, monkeypatch)

    with use_run_id(RUN):
        result = hazard.provider.invoke("local:fs_read", {"path": "leaky.txt"})

    assert result.ok, result.error
    assert hazard.locator_uri not in result.content


def test_remote_now_stamp_carries_the_bindings_real_exclusions(tmp_path, monkeypatch):
    """Carried obligation (b): the CAS probe no longer hardcodes an empty
    exclusion list. An EXCLUDED-but-existing remote target probes as
    refused (absent), and a clean target still reports its real stamps --
    both through the executor's injection seam."""
    hazard = _invoke_hazard(tmp_path, monkeypatch)
    (hazard.standin / "standin-secrets").mkdir()
    (hazard.standin / "standin-secrets" / "kv.txt").write_text(
        "hidden=1\n", encoding="utf-8"
    )

    with use_run_id(RUN):
        excluded_now = hazard.provider._remote_now_stamp(
            hazard.executor, "standin-secrets/kv.txt"
        )
        clean_now = hazard.provider._remote_now_stamp(hazard.executor, "doc.txt")

    # The excluded target is invisible to the probe: the worker's refusal
    # ("file not found" for excluded paths -- ADR-174 full invisibility)
    # classifies as ABSENT, never as the file's real stamps.
    assert excluded_now is None
    assert clean_now is not None
    assert clean_now[0] == REMOTE_DIGEST


def test_redact_root_locator_handles_remote_roots_lexically():
    """Unit pin: ``redact_root_locator`` accepts a RemoteRoot and strips
    every locator spelling (canonical locator, root path, display URI)
    without touching the filesystem."""
    from tldw_chatbook.Agents.tool_catalog import redact_root_locator
    from tldw_chatbook.Tools.remote_root_types import display_uri

    descriptor = RemoteRoot(
        alias="w",
        canonical_locator="ssh://devbox/srv/www",
        root=PurePosixPath("/srv/www"),
        binding_id="b",
    )
    for form in (
        "ssh://devbox/srv/www",  # canonical locator
        "/srv/www",  # remote root path
        display_uri(descriptor),  # ssh:// display form
    ):
        text = f"path {form}/etc/app.conf under root {form}"
        redacted = redact_root_locator(text, descriptor)
        assert form not in redacted, form
    # Container recursion keeps working for remote roots.
    payload = {"note": ["see ssh://devbox/srv/www/x"], "n": 1}
    redacted = redact_root_locator(payload, descriptor)
    assert "ssh://devbox/srv/www" not in str(redacted)
