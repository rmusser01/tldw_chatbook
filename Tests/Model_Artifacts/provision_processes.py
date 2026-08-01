"""Spawn targets for cross-process managed-acquisition crash-recovery tests.

Mirrors ``lease_processes.py``'s style: small, self-contained functions a
``multiprocessing.Process`` (spawn context) can target directly, taking only
picklable primitives as arguments.

Each entrypoint here runs a real ``ArtifactAcquisitionService.provision()``
call against a caller-supplied fixture URL and deterministically freezes
itself -- via a local ``threading.Event`` that nothing ever sets -- the first
time an ``AcquisitionProgress`` event matches a caller-chosen (phase,
artifact_id) pair, after signalling the parent's ``ready`` event. The parent
then sends SIGKILL. Freezing on a never-set local event (rather than timing
a kill against a sleep or a hash/IO duration) makes the crash point exact:
whatever durable state existed the instant the matching progress event fired
is exactly what a real crash landing there would leave behind, with no
timing race against how fast the interpreter can finish the current phase.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Protocol


class EventLike(Protocol):
    """Spawn-safe event operations used by the child targets."""

    def set(self) -> None: ...

    def wait(self, timeout: float | None = None) -> bool: ...


def build_descriptor(
    artifact_id: str,
    revision: str,
    variant: str,
    *,
    role: str,
    source_url: str,
    size_bytes: int,
    sha256: str,
    dependencies: tuple[tuple[str, str, str], ...],
):
    """Construct one single-file ``ArtifactDescriptor`` from primitives.

    ``size_bytes``/``sha256`` are caller-supplied independently of whatever
    the fixture route currently serves -- deliberately, so a caller can
    declare a file larger than what the route serves on a first pass (see
    ``provision_signal_on_phase``'s docstring for why that is the only way
    to get a real, un-seeded, durably-partial sidecar entry out of one
    genuine ``stream_fetch`` call).
    """

    from tldw_chatbook.Model_Artifacts import (
        ArtifactDescriptor,
        ArtifactFile,
        ArtifactFormat,
        ArtifactRef,
        ArtifactRole,
        ProvenanceClass,
    )

    ref = ArtifactRef(artifact_id, revision, variant)
    files = (ArtifactFile("model.bin", size_bytes, sha256),)
    return ArtifactDescriptor(
        reference=ref,
        model_id="test/model",
        role=ArtifactRole(role),
        format=ArtifactFormat.ONNX,
        consumer="test",
        model_family="test-family",
        upstream_repository="test/repo",
        upstream_revision="main",
        source_url=source_url,
        precision=variant,
        license_id="test-license",
        license_url="https://example.test/license",
        usage_notice="Test model",
        runtime_name="test-runtime",
        runtime_version_constraint="==1.0.0",
        supported_os=("linux",),
        supported_architectures=("x86-64",),
        provenance=(ProvenanceClass.CHATBOOK_CURATED,),
        files=files,
        expected_installed_bytes=size_bytes,
        dependencies=tuple(ArtifactRef(*dep) for dep in dependencies),
    )


class DictCatalog:
    """Minimal in-process ``ArtifactCatalog`` keyed by ``ArtifactRef``."""

    def __init__(self, mapping: dict) -> None:
        self._mapping = mapping

    def descriptor(self, ref):
        """Return the descriptor registered for ``ref``."""
        return self._mapping[ref]


def provision_signal_on_phase(
    root_dir: str,
    artifact_specs: tuple[dict, ...],
    root_ref: tuple[str, str, str],
    trusted_origin: str,
    signal_phase: str,
    signal_artifact_id: str | None,
    ready: EventLike,
) -> None:
    """Run ``provision()`` for a closure, freezing at a chosen progress event.

    The child never returns control past the frozen point on its own --
    the parent is expected to send SIGKILL once ``ready`` is observed set.
    If the parent fails to do so, the freeze times out after 30s and the
    process exits normally (a safety net, not the intended path).

    Args:
        root_dir: Filesystem path for the ``ModelArtifactService`` root.
        artifact_specs: One dict per artifact in the closure, each with
            keys ``artifact_id``, ``revision``, ``variant``, ``role``
            (``"root"``/``"dependency"``), ``source_url``, ``size_bytes``,
            ``sha256``, and ``dependencies`` (a tuple of
            ``(artifact_id, revision, variant)`` triples).
        root_ref: The ``(artifact_id, revision, variant)`` triple naming
            which spec is the closure root.
        trusted_origin: Hostname exempted from the private-IP egress block
            (the fixture server's loopback hostname).
        signal_phase: The ``AcquisitionProgress.phase`` value to freeze on
            (``"fetch"``, ``"pre-verify"``, ``"verify-install"``, or
            ``"activate"``).
        signal_artifact_id: Restrict the freeze to progress events about
            this artifact_id, or ``None`` to freeze on the first matching
            phase regardless of which artifact it names.
        ready: Set the instant the matching event fires, immediately before
            freezing -- the parent waits on this before sending SIGKILL.
    """

    import asyncio

    from tldw_chatbook.Model_Artifacts import ArtifactRef, closure_fingerprint
    from tldw_chatbook.Model_Artifacts.acquisition import (
        AcquisitionConsent,
        ArtifactAcquisitionService,
        resolve_catalog_closure,
    )
    from tldw_chatbook.Model_Artifacts.service import ModelArtifactService

    descriptors = {}
    for spec in artifact_specs:
        descriptor = build_descriptor(
            spec["artifact_id"],
            spec["revision"],
            spec["variant"],
            role=spec["role"],
            source_url=spec["source_url"],
            size_bytes=spec["size_bytes"],
            sha256=spec["sha256"],
            dependencies=spec.get("dependencies", ()),
        )
        descriptors[descriptor.reference] = descriptor

    catalog = DictCatalog(descriptors)
    root = ArtifactRef(*root_ref)
    closure = resolve_catalog_closure(root, catalog)
    fingerprint = closure_fingerprint(root, (item.reference for item in closure))
    consent = AcquisitionConsent(closure_fingerprint=fingerprint)

    core = ModelArtifactService(Path(root_dir))
    svc = ArtifactAcquisitionService(
        core,
        free_bytes_probe=lambda _p: 10**12,
        trusted_origins=frozenset({trusted_origin}),
    )

    hold = threading.Event()  # never set -- the parent SIGKILLs us instead

    def on_progress(progress) -> None:
        if signal_phase != progress.phase:
            return
        if signal_artifact_id is not None and progress.ref.artifact_id != signal_artifact_id:
            return
        if ready.is_set():
            return
        ready.set()
        hold.wait(timeout=30.0)

    asyncio.run(svc.provision(root, consent, catalog, progress=on_progress))
