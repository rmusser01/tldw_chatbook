"""TASK-34100.14: cancel, queue and mid-file resume for managed downloads.

The first-run wizard runs every managed download as ``asyncio.run(...)``
inside a Textual thread worker, so ``Worker.cancel()`` can never reach the
transfer. These tests drive ``provision()`` the same way -- on a real
thread with its own event loop, against a real localhost fixture source --
and prove three shared-layer guarantees:

* a cancel request made from ANOTHER thread stops the transfer within a
  couple of seconds, removes the download stage and installs nothing
  ([voice-speech-09]);
* a second install started while another holds the global acquisition
  session lease either waits for it and says which model it is waiting
  for, or fails fast naming the model actually downloading -- never the
  caller's own model ([voice-speech-08], task-33079);
* a transfer that dies part-way through one large file leaves a durable
  mid-file checkpoint, preflight credits it, and the retry resumes with a
  ``Range`` request from that checkpoint instead of byte zero
  ([voice-speech-11]).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest

from Tests.Model_Artifacts.acquisition_test_helpers import _trusted, grant_consent
from Tests.Model_Artifacts.fixture_http import FixtureArtifactServer
from Tests.Model_Artifacts.test_acquisition_types import DictCatalog, make_descriptor
from tldw_chatbook.Model_Artifacts import ArtifactRef
from tldw_chatbook.Model_Artifacts import acquisition as acquisition_module
from tldw_chatbook.Model_Artifacts.acquisition import (
    AcquisitionBusyError,
    AcquisitionProgress,
    ArtifactAcquisitionService,
    TransferError,
)
from tldw_chatbook.Model_Artifacts.leases import ArtifactOperationLease, LeaseMode
from tldw_chatbook.Model_Artifacts.service import (
    ACQUISITION_SESSION_LEASE_KEY,
    ModelArtifactService,
)
from tldw_chatbook.UI.Screens.model_browser_state import install_failure_message

pytestmark = pytest.mark.allow_network

# fetch.py streams in 1 MiB chunks and httpx only yields a chunk once it is
# complete, so every paused/dropped body below is sized in whole MiB.
_MIB = 1024 * 1024


def _download_control(label: str, *, wait_if_busy: bool = False):
    from tldw_chatbook.Model_Artifacts.download_control import DownloadControl

    return DownloadControl(label=label, wait_if_busy=wait_if_busy)


class _ThreadedProvision:
    """Run one ``provision()`` the way the wizard does: a thread + asyncio.run."""

    def __init__(self, service, ref, consent, catalog, **kwargs) -> None:
        self.events: list[AcquisitionProgress] = []
        self.result: object = None
        self.error: BaseException | None = None
        self.finished_at: float | None = None
        self._first_fetch = threading.Event()
        self._queued = threading.Event()

        def progress(event: AcquisitionProgress) -> None:
            self.events.append(event)
            if event.phase == "fetch":
                self._first_fetch.set()
            if event.phase == "queued":
                self._queued.set()

        async def run() -> object:
            return await service.provision(
                ref, consent, catalog, progress=progress, **kwargs
            )

        def target() -> None:
            try:
                self.result = asyncio.run(run())
            except BaseException as exc:  # noqa: BLE001 - recorded for asserts
                self.error = exc
            finally:
                self.finished_at = time.monotonic()

        self.thread = threading.Thread(target=target, daemon=True)

    def start(self) -> "_ThreadedProvision":
        self.thread.start()
        return self

    def wait_fetching(self, timeout: float = 5.0) -> None:
        assert self._first_fetch.wait(timeout), "provision never reached fetch"

    def wait_queued(self, timeout: float = 5.0) -> None:
        assert self._queued.wait(timeout), "second provision never queued"

    def join(self, timeout: float = 15.0) -> None:
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "provision thread did not finish"


def _lease_is_free(core: ModelArtifactService) -> bool:
    probe = ArtifactOperationLease(
        core.locks_path,
        ACQUISITION_SESSION_LEASE_KEY,
        LeaseMode.EXCLUSIVE,
        timeout_seconds=1.0,
    )
    probe.acquire()
    probe.release()
    return True


def _installed_refs(core: ModelArtifactService) -> set[ArtifactRef]:
    return {
        item.descriptor.reference
        for item in core.list_installed()
        if item.descriptor is not None
    }


# ---------------------------------------------------------------------------
# [voice-speech-09] Cancel from another thread stops the transfer and cleans up.
# ---------------------------------------------------------------------------


def test_cancel_request_from_the_ui_thread_stops_a_worker_thread_download(
    tmp_path: Path,
) -> None:
    """A cancel lands in seconds, removes the stage, and saves nothing.

    The fixture handler blocks mid-body for up to 5 s before finishing on
    its own, so a cancel that does not actually interrupt the transfer
    shows up as a completed install, not a hang.
    """
    core = ModelArtifactService(tmp_path / "root")
    ref = ArtifactRef("speech-model", "r1", "int8")
    body = b"s" * (3 * _MIB)
    gate = threading.Event()
    with FixtureArtifactServer() as srv:
        srv.serve(
            "/model.onnx",
            body,
            pause_after=int(1.2 * _MIB),
            pause_event=gate,
        )
        descriptor = make_descriptor(
            ref=ref, files_body=body, source_url=srv.url("/model.onnx")
        )
        catalog = DictCatalog({ref: descriptor})
        service = ArtifactAcquisitionService(
            core, free_bytes_probe=lambda _p: 10**12, trusted_origins=_trusted(srv)
        )
        consent = grant_consent(service, ref, catalog)
        control = _download_control("Speech model")
        run = _ThreadedProvision(service, ref, consent, catalog, control=control)
        try:
            run.start().wait_fetching()
            requested_at = time.monotonic()
            control.cancel()
            run.join()
        finally:
            gate.set()

    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionCancelledError

    assert isinstance(run.error, AcquisitionCancelledError), run.error
    assert run.finished_at is not None
    assert run.finished_at - requested_at < 3.0
    assert run.error.installed_refs == ()
    assert run.error.staging_removed is True
    assert ref not in _installed_refs(core)
    # The stage directory (payload + fetch-state sidecar) is gone, not just
    # left for a later reconcile pass.
    assert core._download_stage_for(descriptor, create=False) is None
    assert not any(core.staging_path.glob("download-*"))
    assert _lease_is_free(core)
    assert (
        install_failure_message(run.error, model_label="Speech model")
        == "Download cancelled — nothing was saved."
    )


def test_cancel_is_not_honoured_once_install_has_started(tmp_path: Path) -> None:
    """A request after the bytes are in never reports a lie: it installs."""
    core = ModelArtifactService(tmp_path / "root")
    ref = ArtifactRef("speech-model", "r1", "int8")
    body = b"t" * 4096
    with FixtureArtifactServer() as srv:
        srv.serve("/model.onnx", body)
        descriptor = make_descriptor(
            ref=ref, files_body=body, source_url=srv.url("/model.onnx")
        )
        catalog = DictCatalog({ref: descriptor})
        service = ArtifactAcquisitionService(
            core, free_bytes_probe=lambda _p: 10**12, trusted_origins=_trusted(srv)
        )
        consent = grant_consent(service, ref, catalog)
        control = _download_control("Speech model")
        install = service._install_artifact

        async def cancel_during_install(descriptor, stage):
            control.cancel()
            # Several watcher polls pass while the promotion is "running".
            await asyncio.sleep(0.5)
            return await install(descriptor, stage)

        service._install_artifact = cancel_during_install
        result = asyncio.run(service.provision(ref, consent, catalog, control=control))

    assert result == ref
    assert ref in _installed_refs(core)
    assert core._download_stage_for(descriptor, create=False) is None


def test_cancel_landing_while_the_stage_is_being_created_still_removes_it(
    tmp_path: Path,
) -> None:
    """The stage call cannot be interrupted, so a cancel waits it out -- and
    the stage it just created (or reused, partial and all) is still removed.

    ``_settle_native`` lets the native stage call finish and THEN raises
    ``CancelledError``; the wrapper below reproduces exactly that ordering
    (stage on disk, cancellation delivered before the call returns).
    """
    core = ModelArtifactService(tmp_path / "root")
    ref = ArtifactRef("speech-model", "r1", "int8")
    body = b"c" * 4096
    with FixtureArtifactServer() as srv:
        srv.serve("/model.onnx", body)
        descriptor = make_descriptor(
            ref=ref, files_body=body, source_url=srv.url("/model.onnx")
        )
        catalog = DictCatalog({ref: descriptor})
        service = ArtifactAcquisitionService(
            core, free_bytes_probe=lambda _p: 10**12, trusted_origins=_trusted(srv)
        )
        consent = grant_consent(service, ref, catalog)
        control = _download_control("Speech model")
        run_core_call = service._run_core_call

        async def cancel_as_the_stage_lands(operation, call_ref, func):
            result = await run_core_call(operation, call_ref, func)
            if operation == "stage":
                control.cancel()
                # Several watcher polls: the cancellation arrives here, after
                # the stage exists and before the call has returned.
                await asyncio.sleep(1.0)
            return result

        service._run_core_call = cancel_as_the_stage_lands
        from tldw_chatbook.Model_Artifacts.acquisition import (
            AcquisitionCancelledError,
        )

        with pytest.raises(AcquisitionCancelledError) as excinfo:
            asyncio.run(service.provision(ref, consent, catalog, control=control))

    assert excinfo.value.installed_refs == ()
    assert excinfo.value.staging_removed is True
    assert "/model.onnx" not in srv.requests
    assert core._download_stage_for(descriptor, create=False) is None
    assert not any(core.staging_path.glob("download-*"))
    assert _lease_is_free(core)


# ---------------------------------------------------------------------------
# [voice-speech-08] / task-33079: a second install queues or names the holder.
# ---------------------------------------------------------------------------


def _two_artifacts(tmp_path: Path, srv: FixtureArtifactServer, gate: threading.Event):
    first_ref = ArtifactRef("omnivoice-model", "r1", "int8")
    second_ref = ArtifactRef("parakeet-model", "r1", "int8")
    first_body = b"o" * (3 * _MIB)
    second_body = b"p" * 8192
    srv.serve(
        "/first.onnx", first_body, pause_after=int(1.2 * _MIB), pause_event=gate
    )
    srv.serve("/second.onnx", second_body)
    first = make_descriptor(
        ref=first_ref, files_body=first_body, source_url=srv.url("/first.onnx")
    )
    second = make_descriptor(
        ref=second_ref, files_body=second_body, source_url=srv.url("/second.onnx")
    )
    catalog = DictCatalog({first_ref: first, second_ref: second})
    # Two cores over ONE store root, like the wizard's Voice and Speech steps
    # (each builds its own service for the shared managed store).
    first_core = ModelArtifactService(tmp_path / "root")
    second_core = ModelArtifactService(tmp_path / "root")
    trusted = _trusted(srv)
    first_service = ArtifactAcquisitionService(
        first_core, free_bytes_probe=lambda _p: 10**12, trusted_origins=trusted
    )
    second_service = ArtifactAcquisitionService(
        second_core, free_bytes_probe=lambda _p: 10**12, trusted_origins=trusted
    )
    return (
        (first_ref, first_service, grant_consent(first_service, first_ref, catalog)),
        (second_ref, second_service, grant_consent(second_service, second_ref, catalog)),
        catalog,
        first_core,
    )


def test_second_install_fails_fast_naming_the_model_actually_downloading(
    tmp_path: Path,
) -> None:
    gate = threading.Event()
    with FixtureArtifactServer() as srv:
        (first_ref, first_service, first_consent), (
            second_ref,
            second_service,
            second_consent,
        ), catalog, core = _two_artifacts(tmp_path, srv, gate)
        first = _ThreadedProvision(
            first_service,
            first_ref,
            first_consent,
            catalog,
            control=_download_control("OmniVoice voice model"),
        )
        try:
            first.start().wait_fetching()
            second = _ThreadedProvision(
                second_service,
                second_ref,
                second_consent,
                catalog,
                control=_download_control("Parakeet v2 (English, INT8)"),
            )
            second.start().join()
        finally:
            gate.set()
        first.join()

    assert isinstance(second.error, AcquisitionBusyError), second.error
    assert second.error.holder_label == "OmniVoice voice model"
    message = install_failure_message(
        second.error, model_label="Parakeet v2 (English, INT8)"
    )
    assert "OmniVoice voice model" in message
    assert "Parakeet" not in message
    assert first.error is None
    assert first_ref in _installed_refs(core)


def test_second_install_waits_for_the_first_and_then_both_install(
    tmp_path: Path,
) -> None:
    gate = threading.Event()
    with FixtureArtifactServer() as srv:
        (first_ref, first_service, first_consent), (
            second_ref,
            second_service,
            second_consent,
        ), catalog, core = _two_artifacts(tmp_path, srv, gate)
        first = _ThreadedProvision(
            first_service,
            first_ref,
            first_consent,
            catalog,
            control=_download_control("OmniVoice voice model", wait_if_busy=True),
        )
        second = _ThreadedProvision(
            second_service,
            second_ref,
            second_consent,
            catalog,
            control=_download_control("Parakeet v2 (English, INT8)", wait_if_busy=True),
        )
        try:
            first.start().wait_fetching()
            second.start().wait_queued()
            queued = next(event for event in second.events if event.phase == "queued")
            assert queued.waiting_for == "OmniVoice voice model"
            assert queued.ref == second_ref
            # Still waiting -- it has not touched the network.
            assert "/second.onnx" not in srv.requests
        finally:
            gate.set()
        first.join()
        second.join()

    assert first.error is None, first.error
    assert second.error is None, second.error
    assert {first_ref, second_ref} <= _installed_refs(core)
    phases = [event.phase for event in second.events]
    assert phases.index("queued") < phases.index("fetch")


def test_cancel_while_queued_leaves_the_running_download_alone(tmp_path: Path) -> None:
    gate = threading.Event()
    with FixtureArtifactServer() as srv:
        (first_ref, first_service, first_consent), (
            second_ref,
            second_service,
            second_consent,
        ), catalog, core = _two_artifacts(tmp_path, srv, gate)
        first = _ThreadedProvision(
            first_service,
            first_ref,
            first_consent,
            catalog,
            control=_download_control("OmniVoice voice model"),
        )
        waiting = _download_control("Parakeet v2 (English, INT8)", wait_if_busy=True)
        second = _ThreadedProvision(
            second_service, second_ref, second_consent, catalog, control=waiting
        )
        try:
            first.start().wait_fetching()
            second.start().wait_queued()
            waiting.cancel()
            second.join(timeout=3.0)
            assert first.thread.is_alive(), "cancelling the waiter stopped the holder"
        finally:
            gate.set()
        first.join()

    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionCancelledError

    assert isinstance(second.error, AcquisitionCancelledError), second.error
    assert first.error is None
    assert first_ref in _installed_refs(core)
    assert second_ref not in _installed_refs(core)


# ---------------------------------------------------------------------------
# [voice-speech-11] A large file resumes from a mid-file checkpoint.
# ---------------------------------------------------------------------------


def test_dropped_large_file_resumes_from_a_mid_file_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        acquisition_module, "FETCH_CHECKPOINT_BYTES", _MIB, raising=False
    )
    core = ModelArtifactService(tmp_path / "root")
    ref = ArtifactRef("big-encoder", "r1", "int8")
    body = bytes(range(256)) * (3 * _MIB // 256)
    with FixtureArtifactServer() as srv:
        # The first attempt dies 2.5 MiB in: two whole 1 MiB chunks land.
        srv.serve(
            "/encoder.onnx",
            body,
            etag='"encoder-v1"',
            disconnect_after=int(2.5 * _MIB),
        )
        descriptor = make_descriptor(
            ref=ref, files_body=body, source_url=srv.url("/encoder.onnx")
        )
        catalog = DictCatalog({ref: descriptor})
        service = ArtifactAcquisitionService(
            core, free_bytes_probe=lambda _p: 10**12, trusted_origins=_trusted(srv)
        )
        consent = grant_consent(service, ref, catalog)
        with pytest.raises(TransferError) as excinfo:
            asyncio.run(service.provision(ref, consent, catalog))
        assert excinfo.value.retryable is True

        stage = core._download_stage_for(descriptor, create=False)
        assert stage is not None
        sidecar = json.loads((stage.state / "fetch-state.json").read_text())
        entry = sidecar["files"]["model.onnx"]
        assert entry["complete"] is False
        assert entry["bytes_done"] == 2 * _MIB
        assert entry["etag"] == '"encoder-v1"'

        # The review counts the staged bytes before the retry.
        report = asyncio.run(service.preflight(ref, catalog))
        assert report.already_staged_bytes == 2 * _MIB
        assert report.download_bytes == len(body) - 2 * _MIB

        srv.serve("/encoder.onnx", body, etag='"encoder-v1"')
        asyncio.run(service.provision(ref, report.grant(), catalog))

    assert srv.requests["/encoder.onnx"][-1].get("Range") == f"bytes={2 * _MIB}-"
    assert ref in _installed_refs(core)


def test_weak_validators_never_claim_a_mid_file_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No resumable validator means no credit: the retry restarts honestly."""
    monkeypatch.setattr(
        acquisition_module, "FETCH_CHECKPOINT_BYTES", _MIB, raising=False
    )
    core = ModelArtifactService(tmp_path / "root")
    ref = ArtifactRef("big-encoder", "r1", "int8")
    body = b"w" * (3 * _MIB)
    with FixtureArtifactServer() as srv:
        srv.serve(
            "/encoder.onnx",
            body,
            etag='"weak"',
            weak_etag=True,
            disconnect_after=int(2.5 * _MIB),
        )
        descriptor = make_descriptor(
            ref=ref, files_body=body, source_url=srv.url("/encoder.onnx")
        )
        catalog = DictCatalog({ref: descriptor})
        service = ArtifactAcquisitionService(
            core, free_bytes_probe=lambda _p: 10**12, trusted_origins=_trusted(srv)
        )
        consent = grant_consent(service, ref, catalog)
        with pytest.raises(TransferError):
            asyncio.run(service.provision(ref, consent, catalog))
        report = asyncio.run(service.preflight(ref, catalog))

    assert report.already_staged_bytes == 0
