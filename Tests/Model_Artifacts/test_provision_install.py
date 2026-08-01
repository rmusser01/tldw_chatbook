"""TASK-595 Task 8: pre-verify + install + activate phases.

Covers ``ArtifactAcquisitionService._preverify_artifact`` and
``_install_artifact`` directly for the per-file mechanics (hash success,
zero-byte files, mismatch-then-refetch recovery, install cleanup), and
``provision()`` end-to-end for the three scenarios the task brief calls
out: a full closure (root + dependency) fetched/verified/installed/
activated, a corrupt payload that exhausts its one automatic refetch, and
the crash-after-install completion path that activates a fully-installed
closure without touching the network at all.
"""

from __future__ import annotations

import hashlib
import json
from urllib.parse import urlparse

import pytest

from Tests.Model_Artifacts.fixture_http import FixtureArtifactServer
from Tests.Model_Artifacts.test_acquisition_types import DictCatalog
from tldw_chatbook.Model_Artifacts import (
    ArtifactDescriptor,
    ArtifactFile,
    ArtifactFormat,
    ArtifactRef,
    ArtifactRole,
    ArtifactIntegrityError,
    ArtifactStateError,
    ProvenanceClass,
    closure_fingerprint,
)
from tldw_chatbook.Model_Artifacts.acquisition import (
    AcquisitionConsent,
    AcquisitionProgress,
    ArtifactAcquisitionService,
    MAX_FILE_REFETCHES,
    TransferError,
    _ProvisionProgressState,
)
from tldw_chatbook.Model_Artifacts.service import ModelArtifactService
from tldw_chatbook.Utils.atomic_file_ops import atomic_write_json


def _trusted(srv: FixtureArtifactServer) -> frozenset:
    """Trusted-origins set for a fixture server (see test_stream_fetch.py's
    identical helper for why this is the bare hostname, not a URL)."""

    return frozenset({urlparse(srv.url("/")).hostname})


def _descriptor(
    ref: ArtifactRef,
    *,
    role: ArtifactRole = ArtifactRole.ROOT,
    dependencies: tuple[ArtifactRef, ...] = (),
    files_body: bytes = b"x",
    source_url: str = "https://example.test/model",
) -> ArtifactDescriptor:
    """Build a single-file descriptor with a caller-chosen role.

    ``test_acquisition_types.make_descriptor`` always hardcodes
    ``role=ArtifactRole.ROOT`` -- fine for the closure-walk and preflight
    math it was written for, but ``core.activate``'s closure resolution
    (``_load_installed_descriptor``) rejects an installed dependency whose
    manifest role isn't ``DEPENDENCY``. This task's tests exercise real
    activation over a root+dependency closure, so the role needs to be
    settable per descriptor.
    """

    files = (ArtifactFile("model.bin", len(files_body), hashlib.sha256(files_body).hexdigest()),)
    return ArtifactDescriptor(
        reference=ref,
        model_id="test/model",
        role=role,
        format=ArtifactFormat.ONNX,
        consumer="test",
        model_family="test-family",
        upstream_repository="test/repo",
        upstream_revision="main",
        source_url=source_url,
        precision=ref.variant,
        license_id="test-license",
        license_url="https://example.test/license",
        usage_notice="Test model",
        runtime_name="test-runtime",
        runtime_version_constraint="==1.0.0",
        supported_os=("linux",),
        supported_architectures=("x86-64",),
        provenance=(ProvenanceClass.CHATBOOK_CURATED,),
        files=files,
        expected_installed_bytes=len(files_body),
        dependencies=dependencies,
    )


# ---------------------------------------------------------------------------
# _preverify_artifact: hash success, zero-byte files, mismatch + recovery.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preverify_matches_declared_hash_and_reports_real_progress(tmp_path):
    body = b"0123456789" * 500  # 5 KB
    desc = _descriptor(ArtifactRef("m", "r" * 40, "int8"), files_body=body)
    staging_dir = tmp_path / "staging" / "m"
    staging_dir.mkdir(parents=True)
    (staging_dir / "model.bin").write_bytes(body)

    core = ModelArtifactService(tmp_path / "root")
    svc = ArtifactAcquisitionService(core)
    events: list[AcquisitionProgress] = []
    progress_state = _ProvisionProgressState(callback=events.append, bytes_total=len(body))

    await svc._preverify_artifact(desc, staging_dir, progress_state)

    assert events, "streaming hash must emit at least one progress event"
    assert all(event.phase == "pre-verify" for event in events)
    assert events[-1].bytes_done == len(body)
    assert events[-1].bytes_total == len(body)
    assert events[-1].ref == desc.reference
    assert events[-1].file == "model.bin"
    assert progress_state.preverify_bytes_done == len(body)
    # The fetch counter is untouched by pre-verify -- separately tracked.
    assert progress_state.bytes_done == 0


@pytest.mark.asyncio
async def test_preverify_progress_total_defaults_to_bytes_total_when_unset(tmp_path):
    """Direct construction without ``preverify_bytes_total`` (every other
    test in this file) keeps working: it defaults to ``bytes_total`` via
    ``__post_init__``, matching this class's behavior before Task 4's
    review-finding fix introduced the separate field."""
    body = b"y" * 100
    desc = _descriptor(ArtifactRef("m", "r" * 40, "int8"), files_body=body)
    staging_dir = tmp_path / "staging" / "m"
    staging_dir.mkdir(parents=True)
    (staging_dir / "model.bin").write_bytes(body)

    core = ModelArtifactService(tmp_path / "root")
    svc = ArtifactAcquisitionService(core)
    progress_state = _ProvisionProgressState(callback=None, bytes_total=len(body))

    assert progress_state.preverify_bytes_total == len(body)
    await svc._preverify_artifact(desc, staging_dir, progress_state)
    assert progress_state.preverify_bytes_done == len(body)


@pytest.mark.asyncio
async def test_provision_preverify_total_ignores_fetch_phases_netted_staged_credit(
    tmp_path, monkeypatch
):
    """Pre-verify's progress total is the full closure file size, NOT the
    fetch phase's total (``PreflightReport.download_bytes``), which nets
    out already-staged credit from a resumed run.

    Regression test for the review finding: ``_hash_staged_file`` always
    re-hashes a staged file's ENTIRE on-disk content, but the fetch and
    pre-verify progress events previously shared one ``bytes_total``
    (the fetch phase's netted-down total) -- so ANY resumed run's
    pre-verify progress would overshoot 100%. Here, 500 of 2048 bytes are
    already durably staged and credited before this run starts, netting
    the fetch total (``bytes_total``) down to 1548 while the file's real,
    full size (what pre-verify actually hashes) stays 2048.
    """
    body = b"m" * 2048
    ref = ArtifactRef("root-model", "r" * 40, "int8")
    desc = _descriptor(ref, files_body=body)
    core = ModelArtifactService(tmp_path / "root")

    staged = core.staging_path / "managed" / ref.artifact_id / ref.revision / ref.variant
    staged.mkdir(parents=True)
    atomic_write_json(
        staged / "fetch-state.json",
        {
            "files": {
                "model.bin": {
                    "etag": None,
                    "last_modified": None,
                    "bytes_done": 500,
                    "complete": False,
                }
            }
        },
    )

    svc = ArtifactAcquisitionService(core, free_bytes_probe=lambda p: 10**12)
    catalog = DictCatalog({ref: desc})
    consent = AcquisitionConsent(closure_fingerprint=closure_fingerprint(ref, ()))

    captured: list[_ProvisionProgressState] = []

    async def fake_fetch(descriptor, staging_dir, progress_state):
        captured.append(progress_state)

    async def fake_preverify(descriptor, staging_dir, progress_state):
        captured.append(progress_state)

    async def fake_install(descriptor, staging_dir):
        pass

    monkeypatch.setattr(svc, "_fetch_artifact", fake_fetch)
    monkeypatch.setattr(svc, "_preverify_artifact", fake_preverify)
    monkeypatch.setattr(svc, "_install_artifact", fake_install)
    # Nothing was actually installed (every phase above is faked as a
    # no-op) -- fake activate() too so provision() completes instead of
    # failing on core.activate()'s own "nothing installed" check, which is
    # irrelevant to what this test verifies (the progress_state wiring).
    monkeypatch.setattr(core, "activate", lambda root_reference: root_reference)

    await svc.provision(ref, consent, catalog)

    assert len(captured) == 2
    state = captured[0]
    assert state.bytes_total == 1548, "fetch total nets out the 500 staged bytes"
    assert state.preverify_bytes_total == 2048, (
        "pre-verify total is the full file size, unaffected by staged credit"
    )


@pytest.mark.asyncio
async def test_preverify_zero_byte_file_hashes_without_error(tmp_path):
    desc = _descriptor(ArtifactRef("m", "r" * 40, "int8"), files_body=b"")
    staging_dir = tmp_path / "staging" / "m"
    staging_dir.mkdir(parents=True)
    (staging_dir / "model.bin").touch()

    core = ModelArtifactService(tmp_path / "root")
    svc = ArtifactAcquisitionService(core)
    progress_state = _ProvisionProgressState(callback=None, bytes_total=0)

    await svc._preverify_artifact(desc, staging_dir, progress_state)

    assert progress_state.preverify_bytes_done == 0


@pytest.mark.asyncio
async def test_preverify_mismatch_refetches_once_and_recovers_on_good_content(tmp_path):
    """A corrupt staged file is deleted, its sidecar entry reset, and the
    whole artifact refetched exactly once -- if that refetch's content now
    matches, pre-verify succeeds instead of raising."""

    correct_body = b"correct-content-bytes-" * 50
    wrong_body = b"wr0ng-c0ntent-byte5---" * 50
    assert len(correct_body) == len(wrong_body)

    with FixtureArtifactServer() as srv:
        srv.serve("/model.bin", correct_body, etag='"v2"', support_range=True)
        desc = _descriptor(
            ArtifactRef("m", "r" * 40, "int8"),
            files_body=correct_body,
            source_url=srv.url("/model.bin"),
        )
        staging_dir = tmp_path / "staging" / "m"
        staging_dir.mkdir(parents=True)
        (staging_dir / "model.bin").write_bytes(wrong_body)
        atomic_write_json(
            staging_dir / "fetch-state.json",
            {
                "files": {
                    "model.bin": {
                        "etag": '"v1"',
                        "last_modified": None,
                        "bytes_done": len(wrong_body),
                        "complete": True,
                    }
                }
            },
        )

        core = ModelArtifactService(tmp_path / "root")
        svc = ArtifactAcquisitionService(core, trusted_origins=_trusted(srv))
        progress_state = _ProvisionProgressState(callback=None, bytes_total=len(correct_body))

        await svc._preverify_artifact(desc, staging_dir, progress_state)

        assert (staging_dir / "model.bin").read_bytes() == correct_body
        assert len(srv.requests["/model.bin"]) == 1
        sidecar = json.loads((staging_dir / "fetch-state.json").read_text())
        assert sidecar["files"]["model.bin"]["complete"] is True
        assert sidecar["files"]["model.bin"]["bytes_done"] == len(correct_body)


@pytest.mark.asyncio
async def test_preverify_mismatch_persisting_past_max_refetches_raises_transfer_error(tmp_path):
    """When the refetch's content is ALSO wrong, exactly one refetch is
    attempted (MAX_FILE_REFETCHES == 1) before a typed, retryable failure."""

    correct_body = b"correct-content-bytes-" * 50
    wrong_body = b"wr0ng-c0ntent-byte5---" * 50
    assert len(correct_body) == len(wrong_body)

    with FixtureArtifactServer() as srv:
        srv.serve("/model.bin", wrong_body, etag='"v1"', support_range=True)
        desc = _descriptor(
            ArtifactRef("m", "r" * 40, "int8"),
            files_body=correct_body,
            source_url=srv.url("/model.bin"),
        )
        staging_dir = tmp_path / "staging" / "m"
        staging_dir.mkdir(parents=True)
        (staging_dir / "model.bin").write_bytes(wrong_body)
        atomic_write_json(
            staging_dir / "fetch-state.json",
            {
                "files": {
                    "model.bin": {
                        "etag": '"v1"',
                        "last_modified": None,
                        "bytes_done": len(wrong_body),
                        "complete": True,
                    }
                }
            },
        )

        core = ModelArtifactService(tmp_path / "root")
        svc = ArtifactAcquisitionService(core, trusted_origins=_trusted(srv))
        progress_state = _ProvisionProgressState(callback=None, bytes_total=len(correct_body))

        with pytest.raises(TransferError) as excinfo:
            await svc._preverify_artifact(desc, staging_dir, progress_state)

        assert excinfo.value.retryable is True
        assert len(srv.requests["/model.bin"]) == MAX_FILE_REFETCHES


# ---------------------------------------------------------------------------
# _install_artifact: consume_source install, sidecar + staging cleanup.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_install_promotes_payload_and_removes_staging_dir_and_sidecar(tmp_path):
    body = b"payload-bytes" * 10
    desc = _descriptor(ArtifactRef("m", "r" * 40, "int8"), files_body=body)
    core = ModelArtifactService(tmp_path / "root")
    svc = ArtifactAcquisitionService(core)

    staging_dir = core.staging_path / "managed" / "m" / ("r" * 40) / "int8"
    staging_dir.mkdir(parents=True)
    (staging_dir / "model.bin").write_bytes(body)
    atomic_write_json(
        staging_dir / "fetch-state.json",
        {"files": {"model.bin": {"etag": None, "last_modified": None, "bytes_done": len(body), "complete": True}}},
    )

    await svc._install_artifact(desc, staging_dir)

    installed = core.artifact_path(desc.reference)
    assert installed.exists()
    assert (installed / "model.bin").read_bytes() == body
    assert not staging_dir.exists()


@pytest.mark.asyncio
async def test_install_failure_does_not_install_and_staging_dir_survives(tmp_path):
    """core.install's own integrity check fails (staged content doesn't
    match the declared hash -- as if pre-verify had been skipped, the
    defense-in-depth check this exercises).

    ``consume_source=True`` moves the file out of our staging directory
    (into ``core.install``'s OWN internal staging) before verifying it
    there, so a failure at this point does not leave the corrupt bytes
    behind for inspection -- that is Task 1's sealed core behavior, not
    something this phase controls. What Task 8 owns and this test pins:
    nothing gets installed, and ``_install_artifact`` itself does not
    ``rmtree`` our staging directory on failure (only a successful install
    triggers that cleanup) -- unlike the sidecar, which is removed
    up front regardless, to satisfy install's exact-file-set check.

    ``core.install``'s raw ``ArtifactIntegrityError`` is wrapped by
    ``_run_core_call`` (review finding: core ``ArtifactError`` subclasses
    must never escape ``provision()`` raw) into a non-retryable
    ``TransferError`` with the original preserved as ``__cause__``.
    """

    desc = _descriptor(ArtifactRef("m", "r" * 40, "int8"), files_body=b"expected")
    core = ModelArtifactService(tmp_path / "root")
    svc = ArtifactAcquisitionService(core)

    staging_dir = core.staging_path / "managed" / "m" / ("r" * 40) / "int8"
    staging_dir.mkdir(parents=True)
    (staging_dir / "model.bin").write_bytes(b"corrupt!")  # wrong content, right length
    atomic_write_json(
        staging_dir / "fetch-state.json",
        {"files": {"model.bin": {"etag": None, "last_modified": None, "bytes_done": 8, "complete": True}}},
    )

    with pytest.raises(TransferError) as excinfo:
        await svc._install_artifact(desc, staging_dir)

    assert excinfo.value.retryable is False
    assert isinstance(excinfo.value.__cause__, ArtifactIntegrityError)

    assert staging_dir.exists()
    assert not (staging_dir / "fetch-state.json").exists()
    assert core.list_installed() == ()


# ---------------------------------------------------------------------------
# provision() end-to-end: happy path, corrupt payload, crash-after-install.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provision_end_to_end_installs_and_activates_root_and_dependency(tmp_path):
    dep_ref = ArtifactRef("dep-model", "d" * 40, "int8")
    root_ref = ArtifactRef("root-model", "r" * 40, "int8")
    dep_body = b"dependency-payload-" * 300
    root_body = b"root-payload-bytes-" * 300

    with FixtureArtifactServer() as srv:
        srv.serve("/dep.bin", dep_body, etag='"dep-v1"', support_range=True)
        srv.serve("/root.bin", root_body, etag='"root-v1"', support_range=True)

        dep_desc = _descriptor(
            dep_ref,
            role=ArtifactRole.DEPENDENCY,
            files_body=dep_body,
            source_url=srv.url("/dep.bin"),
        )
        root_desc = _descriptor(
            root_ref,
            role=ArtifactRole.ROOT,
            dependencies=(dep_ref,),
            files_body=root_body,
            source_url=srv.url("/root.bin"),
        )
        catalog = DictCatalog({root_ref: root_desc, dep_ref: dep_desc})

        core = ModelArtifactService(tmp_path / "root")
        svc = ArtifactAcquisitionService(
            core, free_bytes_probe=lambda p: 10**12, trusted_origins=_trusted(srv)
        )
        consent = AcquisitionConsent(
            closure_fingerprint=closure_fingerprint(root_ref, (dep_ref,))
        )

        events: list[AcquisitionProgress] = []
        activated = await svc.provision(root_ref, consent, catalog, progress=events.append)

        assert activated == root_ref

        installed_refs = {
            item.descriptor.reference for item in core.list_installed() if item.descriptor is not None
        }
        assert installed_refs == {root_ref, dep_ref}

        with core.acquire(root_ref) as handle:
            assert handle.handle.root == root_ref
            assert set(handle.handle.closure) == {root_ref, dep_ref}

        # ALL managed staging for both artifacts is gone -- not an empty
        # directory with a stale sidecar, which would look resumable
        # forever to a later GC pass.
        for ref in (root_ref, dep_ref):
            staging_dir = (
                core.staging_path / "managed" / ref.artifact_id / ref.revision / ref.variant
            )
            assert not staging_dir.exists()

        phases_seen = [event.phase for event in events]
        for phase in ("fetch", "pre-verify", "verify-install", "activate"):
            assert phase in phases_seen, f"missing phase {phase!r} in {phases_seen}"
        first = {phase: phases_seen.index(phase) for phase in set(phases_seen)}
        assert first["fetch"] < first["pre-verify"] < first["verify-install"] < first["activate"]
        assert phases_seen.count("activate") == 1
        assert phases_seen[-1] == "activate"

        fetch_events = [event for event in events if event.phase == "fetch"]
        assert fetch_events[-1].bytes_done == len(dep_body) + len(root_body)
        preverify_events = [event for event in events if event.phase == "pre-verify"]
        assert preverify_events[-1].bytes_done == len(dep_body) + len(root_body)


@pytest.mark.asyncio
async def test_provision_corrupt_payload_refetches_exactly_once_then_fails(tmp_path):
    ref = ArtifactRef("bad-model", "b" * 40, "int8")
    correct_body = b"correct-content-bytes-" * 400
    wrong_body = b"wr0ng-c0ntent-byte5---" * 400
    assert len(correct_body) == len(wrong_body)

    with FixtureArtifactServer() as srv:
        # The server never fixes itself -- both the initial fetch and the
        # one automatic refetch see the same wrong bytes.
        srv.serve("/model.bin", wrong_body, etag='"v1"', support_range=True)

        core = ModelArtifactService(tmp_path / "root")
        svc = ArtifactAcquisitionService(
            core, free_bytes_probe=lambda p: 10**12, trusted_origins=_trusted(srv)
        )

        # A prior, unrelated artifact is already installed and active --
        # proves the failed provision leaves other closures untouched.
        prior_ref = ArtifactRef("prior-model", "p" * 40, "int8")
        prior_desc = _descriptor(prior_ref, files_body=b"prior-bytes")
        prior_source = tmp_path / "prior-source"
        prior_source.mkdir()
        (prior_source / "model.bin").write_bytes(b"prior-bytes")
        core.install(prior_desc, prior_source)
        core.activate(prior_ref)

        desc = _descriptor(ref, files_body=correct_body, source_url=srv.url("/model.bin"))
        catalog = DictCatalog({ref: desc})
        consent = AcquisitionConsent(closure_fingerprint=closure_fingerprint(ref, ()))

        with pytest.raises(TransferError) as excinfo:
            await svc.provision(ref, consent, catalog)
        assert excinfo.value.retryable is True

        # Exactly one refetch attempt: the initial fetch plus one refetch,
        # never more.
        assert len(srv.requests["/model.bin"]) == 1 + MAX_FILE_REFETCHES

        installed_refs = {
            item.descriptor.reference for item in core.list_installed() if item.descriptor is not None
        }
        assert ref not in installed_refs

        # Prior state genuinely untouched.
        assert installed_refs == {prior_ref}
        with core.acquire(prior_ref) as handle:
            assert handle.handle.root == prior_ref


@pytest.mark.asyncio
async def test_provision_activates_already_installed_closure_with_zero_fetch_requests(tmp_path):
    """The crash-after-install recovery path: both artifacts are already
    installed (simulating a prior run that crashed before activation) --
    provision() must activate them without any network activity at all."""

    dep_ref = ArtifactRef("dep-model", "d" * 40, "int8")
    root_ref = ArtifactRef("root-model", "r" * 40, "int8")
    dep_body = b"dep-bytes-already-installed"
    root_body = b"root-bytes-already-installed"

    core = ModelArtifactService(tmp_path / "root")

    dep_desc = _descriptor(dep_ref, role=ArtifactRole.DEPENDENCY, files_body=dep_body)
    root_desc = _descriptor(
        root_ref, role=ArtifactRole.ROOT, dependencies=(dep_ref,), files_body=root_body
    )

    dep_source = tmp_path / "dep-source"
    dep_source.mkdir()
    (dep_source / "model.bin").write_bytes(dep_body)
    core.install(dep_desc, dep_source)

    root_source = tmp_path / "root-source"
    root_source.mkdir()
    (root_source / "model.bin").write_bytes(root_body)
    core.install(root_desc, root_source)
    # Deliberately no core.activate() here -- this IS the crash point this
    # test recovers from.

    catalog = DictCatalog({root_ref: root_desc, dep_ref: dep_desc})
    consent = AcquisitionConsent(closure_fingerprint=closure_fingerprint(root_ref, (dep_ref,)))

    with FixtureArtifactServer() as srv:
        # Deliberately no route registered: any network attempt 404s, and
        # this test asserts none happens at all.
        svc = ArtifactAcquisitionService(
            core, free_bytes_probe=lambda p: 10**12, trusted_origins=_trusted(srv)
        )
        events: list[AcquisitionProgress] = []
        activated = await svc.provision(root_ref, consent, catalog, progress=events.append)

        assert activated == root_ref
        assert srv.requests == {}

    with core.acquire(root_ref) as handle:
        assert handle.handle.root == root_ref
        assert set(handle.handle.closure) == {root_ref, dep_ref}

    assert [event.phase for event in events] == ["activate"]


# ---------------------------------------------------------------------------
# _run_core_call: core.install/core.activate ArtifactError never escapes raw.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_install_artifact_wraps_core_integrity_error_as_non_retryable(
    tmp_path, monkeypatch
):
    """``core.install`` raising ``ArtifactIntegrityError`` must surface as a
    non-retryable ``TransferError``, never the raw core error.

    Regression test for the review finding: core ``ArtifactError``
    subclasses escaped ``provision()`` untouched, breaking the spec's
    never-trap rule. Integrity failures are not retryable -- the same
    staged content would fail again.
    """
    desc = _descriptor(ArtifactRef("m", "r" * 40, "int8"), files_body=b"x")
    core = ModelArtifactService(tmp_path / "root")
    svc = ArtifactAcquisitionService(core)

    staging_dir = core.staging_path / "managed" / "m" / ("r" * 40) / "int8"
    staging_dir.mkdir(parents=True)
    (staging_dir / "model.bin").write_bytes(b"x")

    def raise_integrity_error(*args, **kwargs):
        raise ArtifactIntegrityError("simulated integrity failure")

    monkeypatch.setattr(core, "install", raise_integrity_error)

    with pytest.raises(TransferError) as excinfo:
        await svc._install_artifact(desc, staging_dir)

    assert excinfo.value.retryable is False
    assert isinstance(excinfo.value.__cause__, ArtifactIntegrityError)
    assert "m" in str(excinfo.value)
    assert "install" in str(excinfo.value)


@pytest.mark.asyncio
async def test_provision_activate_wraps_core_state_error_as_retryable(
    tmp_path, monkeypatch
):
    """``core.activate`` raising ``ArtifactStateError`` (lease-timeout/
    contention style failure) must surface as a retryable ``TransferError``.

    Both artifacts are pre-installed so ``provision()`` skips straight to
    ``core.activate`` without any network activity, isolating this to the
    activate hop specifically.
    """
    ref = ArtifactRef("root-model", "r" * 40, "int8")
    desc = _descriptor(ref, files_body=b"x")
    core = ModelArtifactService(tmp_path / "root")
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.bin").write_bytes(b"x")
    core.install(desc, source)

    svc = ArtifactAcquisitionService(core, free_bytes_probe=lambda p: 10**12)
    catalog = DictCatalog({ref: desc})
    consent = AcquisitionConsent(closure_fingerprint=closure_fingerprint(ref, ()))

    def raise_state_error(*args, **kwargs):
        raise ArtifactStateError("simulated lease contention")

    monkeypatch.setattr(core, "activate", raise_state_error)

    with pytest.raises(TransferError) as excinfo:
        await svc.provision(ref, consent, catalog)

    assert excinfo.value.retryable is True
    assert isinstance(excinfo.value.__cause__, ArtifactStateError)
    assert "root-model" in str(excinfo.value)
    assert "activate" in str(excinfo.value)
