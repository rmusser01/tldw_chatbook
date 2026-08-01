"""TASK-595 Task 7: provision() fetch phase -- durable staging, sidecar, resume.

Covers ``ArtifactAcquisitionService._fetch_artifact`` (and its private
helpers) directly for the per-file mechanics -- skip/resume/restart/ENOSPC
-- and ``provision()`` end-to-end for the cancellation guarantee, which is
a provision()-level property (session lease release, prior-active-artifact
survival), not something the fetch phase alone can prove.
"""

from __future__ import annotations

import asyncio
import builtins
import errno
import hashlib
import json
import threading
from urllib.parse import urlparse

import pytest

from Tests.Model_Artifacts.fixture_http import FixtureArtifactServer
from Tests.Model_Artifacts.test_acquisition_types import DictCatalog, make_descriptor
from tldw_chatbook.Model_Artifacts import (
    ArtifactDescriptor,
    ArtifactFile,
    ArtifactFormat,
    ArtifactRef,
    ArtifactRole,
    ProvenanceClass,
    closure_fingerprint,
)
from tldw_chatbook.Model_Artifacts import fetch as fetch_module
from tldw_chatbook.Model_Artifacts.acquisition import (
    AcquisitionConsent,
    AcquisitionProgress,
    ArtifactAcquisitionService,
    CatalogError,
    TransferError,
    _ProvisionProgressState,
)
from tldw_chatbook.Model_Artifacts.leases import (
    ArtifactOperationLease,
    LeaseMode,
)
from tldw_chatbook.Model_Artifacts.service import (
    ACQUISITION_SESSION_LEASE_KEY,
    ModelArtifactService,
)
from tldw_chatbook.Utils.atomic_file_ops import atomic_write_json


def _trusted(srv: FixtureArtifactServer) -> frozenset:
    """Trusted-origins set for a fixture server (see test_stream_fetch.py's
    identical helper for why this is the bare hostname, not a URL)."""

    return frozenset({urlparse(srv.url("/")).hostname})


def _make_two_file_descriptor(source_url: str) -> ArtifactDescriptor:
    """A 2-file descriptor -- ``make_descriptor`` only ever builds one file.

    Only the file COUNT matters for the CatalogError coverage below: real
    per-file URLs for a multi-file descriptor don't exist yet (that's
    TASK-596/1301's job), so there is nothing meaningful to make these
    URLs resolve to.
    """

    ref = ArtifactRef("multi-file-model", "r" * 40, "int8")
    files = (
        ArtifactFile("a.bin", 4, hashlib.sha256(b"aaaa").hexdigest()),
        ArtifactFile("b.bin", 4, hashlib.sha256(b"bbbb").hexdigest()),
    )
    return ArtifactDescriptor(
        reference=ref,
        model_id="test/model",
        role=ArtifactRole.ROOT,
        format=ArtifactFormat.ONNX,
        consumer="test",
        model_family="test-family",
        upstream_repository="test/repo",
        upstream_revision="main",
        source_url=source_url,
        precision="int8",
        license_id="test-license",
        license_url="https://example.test/license",
        usage_notice="Test model",
        runtime_name="test-runtime",
        runtime_version_constraint="==1.0.0",
        supported_os=("linux",),
        supported_architectures=("x86-64",),
        provenance=(ProvenanceClass.CHATBOOK_CURATED,),
        files=files,
        expected_installed_bytes=8,
        dependencies=(),
    )


# ---------------------------------------------------------------------------
# Direct _reconcile_durable_bytes unit coverage (Task 3 review carry-over):
# ---------------------------------------------------------------------------


def test_reconcile_truncates_untrusted_excess_beyond_durable_checkpoint(tmp_path):
    """Bytes beyond the sidecar's last durable checkpoint are unverified --
    a crash/disconnect can leave an unfsynced tail on disk that must never
    be trusted for a Range resume."""

    destination = tmp_path / "f.bin"
    destination.write_bytes(b"0" * 4000 + b"?" * 2000)  # 6000 on disk

    result = ArtifactAcquisitionService._reconcile_durable_bytes(destination, 4000)

    assert result == 4000
    assert destination.stat().st_size == 4000
    assert destination.read_bytes() == b"0" * 4000


def test_reconcile_restarts_from_zero_when_sidecar_over_claims(tmp_path):
    """A sidecar claiming MORE durable bytes than the file actually holds
    cannot be trusted at all -- restart that file from zero."""

    destination = tmp_path / "f.bin"
    destination.write_bytes(b"0" * 100)  # far short of the claimed 4000

    result = ArtifactAcquisitionService._reconcile_durable_bytes(destination, 4000)

    assert result == 0
    assert not destination.exists()


def test_reconcile_is_a_noop_when_already_consistent(tmp_path):
    destination = tmp_path / "f.bin"
    destination.write_bytes(b"0" * 4000)

    result = ArtifactAcquisitionService._reconcile_durable_bytes(destination, 4000)

    assert result == 4000
    assert destination.read_bytes() == b"0" * 4000


def test_reconcile_missing_file_with_zero_recorded_is_a_noop(tmp_path):
    destination = tmp_path / "missing.bin"
    assert ArtifactAcquisitionService._reconcile_durable_bytes(destination, 0) == 0


# ---------------------------------------------------------------------------
# _fetch_artifact: fresh download, skip, resume, restart, ENOSPC
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_full_download_writes_sidecar_and_reports_progress(tmp_path):
    body = b"0123456789" * 1000  # 10 KB
    events: list[AcquisitionProgress] = []
    with FixtureArtifactServer() as srv:
        srv.serve("/model.onnx", body, etag='"v1"', support_range=True)
        core = ModelArtifactService(tmp_path / "root")
        svc = ArtifactAcquisitionService(core, trusted_origins=_trusted(srv))
        desc = make_descriptor(files_body=body, source_url=srv.url("/model.onnx"))
        staging_dir = tmp_path / "staging" / "m"

        progress_state = _ProvisionProgressState(callback=events.append, bytes_total=len(body))
        await svc._fetch_artifact(desc, staging_dir, progress_state)

        assert (staging_dir / "model.onnx").read_bytes() == body
        sidecar = json.loads((staging_dir.parent / f"{staging_dir.name}.fetch-state.json").read_text())
        assert sidecar["files"]["model.onnx"] == {
            "etag": '"v1"',
            "last_modified": None,
            "bytes_done": len(body),
            "complete": True,
        }
        assert progress_state.bytes_done == len(body)
        assert events, "on_chunk must emit at least one progress event"
        assert all(event.phase == "fetch" for event in events)
        assert events[-1].bytes_done == len(body)
        assert events[-1].bytes_total == len(body)
        assert events[-1].ref == desc.reference
        assert events[-1].file == "model.onnx"


@pytest.mark.asyncio
async def test_fetch_over_large_checkpoint_restarts_cleanly(tmp_path):
    """A sidecar checkpoint from a PRIOR provision() run can exceed the
    file's CURRENT declared size if the catalog's declared size for this
    artifact/revision/variant shrank between runs (a corrected or re-cut
    upstream entry). ``_reconcile_durable_bytes`` only cross-checks the
    checkpoint against the file's ACTUAL on-disk bytes -- here they agree
    (4000 claimed, 4000 present), so reconciliation leaves it untouched --
    never against the CURRENT ``file.size_bytes``. Left unchecked this
    becomes ``resume_from >= max_bytes`` inside ``stream_fetch``, which
    raises ``FetchTooLargeError`` and gets wrapped as a NON-retryable
    ``TransferError`` ("upstream body exceeds declared size") instead of
    the clean restart-from-zero this really calls for.

    Regression test for the review finding: normalize an over-large
    checkpoint to zero before deriving resume_from.
    """
    body = b"0123456789" * 100  # 1000 bytes -- the file's CURRENT declared size
    stale_bytes = b"x" * 4000  # what a now-stale prior checkpoint claimed
    with FixtureArtifactServer() as srv:
        srv.serve("/model.onnx", body, etag='"v1"', support_range=True)
        core = ModelArtifactService(tmp_path / "root")
        svc = ArtifactAcquisitionService(core, trusted_origins=_trusted(srv))
        desc = make_descriptor(files_body=body, source_url=srv.url("/model.onnx"))
        staging_dir = tmp_path / "staging" / "m"
        staging_dir.mkdir(parents=True)
        (staging_dir / "model.onnx").write_bytes(stale_bytes)
        atomic_write_json(
            staging_dir.parent / f"{staging_dir.name}.fetch-state.json",
            {
                "files": {
                    "model.onnx": {
                        "etag": '"v1"',
                        "last_modified": None,
                        "bytes_done": len(stale_bytes),
                        "complete": False,
                    }
                }
            },
        )

        progress_state = _ProvisionProgressState(callback=None, bytes_total=len(body))
        await svc._fetch_artifact(desc, staging_dir, progress_state)

        assert (staging_dir / "model.onnx").read_bytes() == body
        sidecar = json.loads((staging_dir.parent / f"{staging_dir.name}.fetch-state.json").read_text())
        assert sidecar["files"]["model.onnx"]["complete"] is True
        assert sidecar["files"]["model.onnx"]["bytes_done"] == len(body)
        # A clean restart-from-zero means a FULL GET, never a Range resume.
        assert not any("Range" in r for r in srv.requests["/model.onnx"])


@pytest.mark.asyncio
async def test_fetch_skips_file_already_complete_in_sidecar(tmp_path):
    body = b"x" * 500
    with FixtureArtifactServer() as srv:
        # Deliberately no route registered: any network attempt 404s, which
        # would surface as a TransferError this test never expects.
        core = ModelArtifactService(tmp_path / "root")
        svc = ArtifactAcquisitionService(core, trusted_origins=_trusted(srv))
        desc = make_descriptor(files_body=body, source_url=srv.url("/model.onnx"))
        staging_dir = tmp_path / "staging" / "m"
        staging_dir.mkdir(parents=True)
        (staging_dir / "model.onnx").write_bytes(body)
        atomic_write_json(
            staging_dir.parent / f"{staging_dir.name}.fetch-state.json",
            {
                "files": {
                    "model.onnx": {
                        "etag": '"v1"',
                        "last_modified": None,
                        "bytes_done": len(body),
                        "complete": True,
                    }
                }
            },
        )

        progress_state = _ProvisionProgressState(callback=None, bytes_total=0)
        await svc._fetch_artifact(desc, staging_dir, progress_state)

        assert "/model.onnx" not in srv.requests
        assert progress_state.bytes_done == 0


@pytest.mark.asyncio
async def test_fetch_zero_byte_file_creates_empty_destination_and_skips_network(tmp_path):
    """A zero-byte declared file must never leave the sidecar claiming
    'complete' over a destination that doesn't exist -- Task 8's pre-verify
    would hit FileNotFoundError trying to hash it."""

    with FixtureArtifactServer() as srv:
        # Deliberately no route registered: a zero-byte file has nothing to
        # stream and must never attempt a network request at all.
        core = ModelArtifactService(tmp_path / "root")
        svc = ArtifactAcquisitionService(core, trusted_origins=_trusted(srv))
        desc = make_descriptor(files_body=b"", source_url=srv.url("/model.onnx"))
        staging_dir = tmp_path / "staging" / "m"

        progress_state = _ProvisionProgressState(callback=None, bytes_total=0)
        await svc._fetch_artifact(desc, staging_dir, progress_state)

        destination = staging_dir / "model.onnx"
        assert destination.exists()
        assert destination.read_bytes() == b""
        sidecar = json.loads((staging_dir.parent / f"{staging_dir.name}.fetch-state.json").read_text())
        entry = sidecar["files"]["model.onnx"]
        assert entry["complete"] is True
        assert entry["bytes_done"] == 0
        assert "/model.onnx" not in srv.requests


@pytest.mark.asyncio
async def test_fetch_multi_file_descriptor_raises_catalog_error_without_touching_anything(
    tmp_path,
):
    """Per-file URLs for a multi-file descriptor are undefined until the
    catalog work (TASK-596/1301) specifies them. ``_fetch_artifact`` must
    fail loudly with a typed CatalogError instead of silently guessing a
    joined URL and fetching the wrong bytes -- and must do so before
    touching staging, the sidecar, or the network at all."""

    with FixtureArtifactServer() as srv:
        core = ModelArtifactService(tmp_path / "root")
        svc = ArtifactAcquisitionService(core, trusted_origins=_trusted(srv))
        desc = _make_two_file_descriptor(srv.url("/base/"))
        staging_dir = tmp_path / "staging" / "m"

        progress_state = _ProvisionProgressState(callback=None, bytes_total=8)
        with pytest.raises(CatalogError) as excinfo:
            await svc._fetch_artifact(desc, staging_dir, progress_state)

        assert "multi-file-model" in str(excinfo.value)
        assert not staging_dir.exists()
        assert not srv.requests


@pytest.mark.asyncio
async def test_fetch_resumes_partial_file_with_range_request(tmp_path):
    body = b"0123456789" * 1000  # 10 KB
    with FixtureArtifactServer() as srv:
        srv.serve("/model.onnx", body, etag='"v1"', support_range=True)
        core = ModelArtifactService(tmp_path / "root")
        svc = ArtifactAcquisitionService(core, trusted_origins=_trusted(srv))
        desc = make_descriptor(files_body=body, source_url=srv.url("/model.onnx"))
        staging_dir = tmp_path / "staging" / "m"
        staging_dir.mkdir(parents=True)
        (staging_dir / "model.onnx").write_bytes(body[:4000])
        atomic_write_json(
            staging_dir.parent / f"{staging_dir.name}.fetch-state.json",
            {
                "files": {
                    "model.onnx": {
                        "etag": '"v1"',
                        "last_modified": None,
                        "bytes_done": 4000,
                        "complete": False,
                    }
                }
            },
        )

        progress_state = _ProvisionProgressState(callback=None, bytes_total=len(body) - 4000)
        await svc._fetch_artifact(desc, staging_dir, progress_state)

        assert (staging_dir / "model.onnx").read_bytes() == body
        assert any("Range" in headers for headers in srv.requests["/model.onnx"])
        sidecar = json.loads((staging_dir.parent / f"{staging_dir.name}.fetch-state.json").read_text())
        entry = sidecar["files"]["model.onnx"]
        assert entry["bytes_done"] == len(body)
        assert entry["complete"] is True
        assert entry["etag"] == '"v1"'
        assert progress_state.bytes_done == len(body) - 4000


@pytest.mark.asyncio
async def test_fetch_restarts_from_zero_on_changed_etag(tmp_path):
    old_body = b"0123456789" * 1000
    new_body = b"9876543210" * 1200  # different content and length: new revision
    with FixtureArtifactServer() as srv:
        srv.serve("/model.onnx", new_body, etag='"v2"', support_range=True)
        core = ModelArtifactService(tmp_path / "root")
        svc = ArtifactAcquisitionService(core, trusted_origins=_trusted(srv))
        desc = make_descriptor(files_body=new_body, source_url=srv.url("/model.onnx"))
        staging_dir = tmp_path / "staging" / "m"
        staging_dir.mkdir(parents=True)
        (staging_dir / "model.onnx").write_bytes(old_body[:100])
        atomic_write_json(
            staging_dir.parent / f"{staging_dir.name}.fetch-state.json",
            {
                "files": {
                    "model.onnx": {
                        "etag": '"v1"',
                        "last_modified": None,
                        "bytes_done": 100,
                        "complete": False,
                    }
                }
            },
        )

        progress_state = _ProvisionProgressState(callback=None, bytes_total=len(new_body))
        await svc._fetch_artifact(desc, staging_dir, progress_state)

        assert (staging_dir / "model.onnx").read_bytes() == new_body
        sidecar = json.loads((staging_dir.parent / f"{staging_dir.name}.fetch-state.json").read_text())
        entry = sidecar["files"]["model.onnx"]
        assert entry["etag"] == '"v2"'
        assert entry["bytes_done"] == len(new_body)
        assert entry["complete"] is True


@pytest.mark.asyncio
async def test_fetch_mid_body_disconnect_leaves_durable_sidecar_and_reprovision_resumes(
    tmp_path,
):
    # Large enough that the disconnect lands after one full fetch.py chunk
    # (1 MB) has actually been written to disk -- proving the reconciliation
    # truncation is genuinely exercised, not just tolerated as a no-op. A
    # smaller body would have httpx buffer internally and raise before
    # yielding any bytes at all, which would leave nothing to reconcile.
    body = b"AB" * 1_500_000  # 3,000,000 bytes
    seed = body[:1_000_000]
    with FixtureArtifactServer() as srv:
        core = ModelArtifactService(tmp_path / "root")
        svc = ArtifactAcquisitionService(core, trusted_origins=_trusted(srv))
        desc = make_descriptor(files_body=body, source_url=srv.url("/model.onnx"))
        staging_dir = tmp_path / "staging" / "m"
        staging_dir.mkdir(parents=True)
        (staging_dir / "model.onnx").write_bytes(seed)
        atomic_write_json(
            staging_dir.parent / f"{staging_dir.name}.fetch-state.json",
            {
                "files": {
                    "model.onnx": {
                        "etag": '"v1"',
                        "last_modified": None,
                        "bytes_done": len(seed),
                        "complete": False,
                    }
                }
            },
        )

        # First attempt: server drops the resumed response mid-body, after
        # one full chunk (1 MB) of the resume was already durably written
        # to disk by fetch.py -- but NOT fsynced/recorded to the sidecar,
        # since that only happens after stream_fetch returns successfully.
        srv.serve(
            "/model.onnx", body, etag='"v1"', support_range=True, disconnect_after=1_100_000
        )
        progress_state = _ProvisionProgressState(
            callback=None, bytes_total=len(body) - len(seed)
        )
        with pytest.raises(TransferError) as excinfo:
            await svc._fetch_artifact(desc, staging_dir, progress_state)
        assert excinfo.value.retryable is True

        # Durable state consistent: the sidecar still shows the last
        # checkpoint that was actually fsynced and recorded -- this failed
        # call never got far enough to update it.
        sidecar = json.loads((staging_dir.parent / f"{staging_dir.name}.fetch-state.json").read_text())
        assert sidecar["files"]["model.onnx"]["bytes_done"] == len(seed)
        # The failed attempt genuinely left an untrusted tail beyond the
        # durable checkpoint (the whole point of this scenario).
        assert (staging_dir / "model.onnx").stat().st_size > len(seed)

        # Second attempt ("re-provision"): server now completes normally.
        srv.serve("/model.onnx", body, etag='"v1"', support_range=True)
        progress_state2 = _ProvisionProgressState(
            callback=None, bytes_total=len(body) - len(seed)
        )
        await svc._fetch_artifact(desc, staging_dir, progress_state2)

        assert (staging_dir / "model.onnx").read_bytes() == body
        # The reconciled resume asked for the durable checkpoint, never the
        # untrusted post-disconnect tail -- proves reconciliation actually
        # ran (truncating the leaked bytes), not just that a Range request
        # happened at all.
        assert srv.requests["/model.onnx"][-1].get("Range") == f"bytes={len(seed)}-"
        sidecar_final = json.loads((staging_dir.parent / f"{staging_dir.name}.fetch-state.json").read_text())
        assert sidecar_final["files"]["model.onnx"]["complete"] is True
        assert sidecar_final["files"]["model.onnx"]["bytes_done"] == len(body)


@pytest.mark.asyncio
async def test_fetch_enospc_raises_transfer_error_retryable_and_retains_staging(
    tmp_path, monkeypatch
):
    class _ENOSPCFile:
        """Wraps a real file handle but fails every write with ENOSPC."""

        def __init__(self, real):
            self._real = real

        def write(self, data):
            raise OSError(errno.ENOSPC, "No space left on device")

        def flush(self):
            pass

        def fileno(self):
            return self._real.fileno()

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            self._real.close()
            return False

    def fake_open(path, mode):
        return _ENOSPCFile(builtins.open(path, mode))

    # Shadowing fetch.py's module-global `open` (its own written-as-`open`
    # calls resolve through the module namespace before falling back to the
    # builtin); acquisition.py's own truncate-path `open` calls are
    # untouched since they resolve in a different module's namespace.
    monkeypatch.setattr(fetch_module, "open", fake_open, raising=False)

    body = b"y" * 2000
    with FixtureArtifactServer() as srv:
        srv.serve("/model.onnx", body, etag='"v1"', support_range=True)
        core = ModelArtifactService(tmp_path / "root")
        svc = ArtifactAcquisitionService(core, trusted_origins=_trusted(srv))
        desc = make_descriptor(files_body=body, source_url=srv.url("/model.onnx"))
        staging_dir = tmp_path / "staging" / "m"

        progress_state = _ProvisionProgressState(callback=None, bytes_total=len(body))
        with pytest.raises(TransferError) as excinfo:
            await svc._fetch_artifact(desc, staging_dir, progress_state)

        assert excinfo.value.retryable is True
        # Staging retained -- no cleanup on a failed fetch.
        assert staging_dir.exists()


# ---------------------------------------------------------------------------
# provision() cancellation: session lease released, sidecar durable-only,
# prior active artifact survives untouched.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provision_cancel_mid_fetch_releases_lease_and_preserves_prior_active(
    tmp_path,
):
    core = ModelArtifactService(tmp_path / "root")

    prior_ref = ArtifactRef("prior-model", "r1", "int8")
    prior_body = b"p" * 32
    prior_desc = make_descriptor(ref=prior_ref, files_body=prior_body)
    prior_source = tmp_path / "prior-source"
    prior_source.mkdir()
    (prior_source / prior_desc.files[0].path).write_bytes(prior_body)
    core.install(prior_desc, prior_source)
    core.activate(prior_ref)

    new_ref = ArtifactRef("new-model", "r1", "int8")
    body = b"n" * 3_000_000  # 3 MB
    # fetch.py's aiter_bytes(chunk_size=1 MB) only yields once a full chunk
    # has actually arrived -- pausing at 1.2 MB guarantees one full 1 MB
    # chunk is delivered (firing on_chunk/progress) before the connection
    # genuinely blocks awaiting the rest of the second chunk.
    pause_after = 1_200_000
    gate = threading.Event()

    with FixtureArtifactServer() as srv:
        srv.serve(
            "/model.bin",
            body,
            etag='"v1"',
            support_range=True,
            pause_after=pause_after,
            pause_event=gate,
        )
        svc = ArtifactAcquisitionService(
            core, free_bytes_probe=lambda p: 10**12, trusted_origins=_trusted(srv)
        )
        new_desc = make_descriptor(
            ref=new_ref, files_body=body, source_url=srv.url("/model.bin")
        )
        catalog = DictCatalog({new_ref: new_desc})
        consent = AcquisitionConsent(closure_fingerprint=closure_fingerprint(new_ref, ()))

        progress_seen = asyncio.Event()

        def on_progress(progress: AcquisitionProgress) -> None:
            if progress.phase == "fetch":
                progress_seen.set()

        task = asyncio.create_task(
            svc.provision(new_ref, consent, catalog, progress=on_progress)
        )
        try:
            # Gated by the progress callback firing on the first received
            # chunk, not a sleep-based guess at timing.
            await asyncio.wait_for(progress_seen.wait(), timeout=5.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            gate.set()  # release the paused handler thread promptly

    # The session lease was released in provision()'s finally, not left
    # held by the cancelled task.
    loop = asyncio.get_running_loop()
    probe = ArtifactOperationLease(
        core.locks_path,
        ACQUISITION_SESSION_LEASE_KEY,
        LeaseMode.EXCLUSIVE,
        timeout_seconds=1.0,
    )
    await loop.run_in_executor(None, probe.acquire)
    await loop.run_in_executor(None, probe.release)

    # Sidecar reflects only durable bytes: stream_fetch never returned
    # successfully, so nothing was ever written for the new artifact.
    # Sibling of the payload directory, never a child of it -- see
    # acquisition.py's _fetch_sidecar_path.
    sidecar_path = (
        core.staging_path / "managed" / "new-model" / "r1" / "int8.fetch-state.json"
    )
    assert not sidecar_path.exists()

    # The prior active artifact is untouched and still acquirable.
    with core.acquire(prior_ref) as handle:
        assert handle.handle.root == prior_ref
