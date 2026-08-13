from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import pytest

from tldw_chatbook.Event_Handlers.LLM_Management_Events.gguf_source_modes import (
    GGUFSourceError,
    GGUFSourceMode,
    GGUFSourceSelection,
    ManagedGGUFChoice,
    acquire_managed_gguf,
    gguf_source_failure_message,
    initial_gguf_selection,
    managed_gguf_choices,
)
from tldw_chatbook.Model_Artifacts import (
    ArtifactDescriptor,
    ArtifactFile,
    ArtifactFormat,
    ArtifactHandle,
    ArtifactIntegrityError,
    ArtifactLeaseTimeoutError,
    ArtifactNotReadyError,
    ArtifactRef,
    ArtifactRole,
    ArtifactStateError,
    InstalledArtifact,
    ProvenanceClass,
)
from tldw_chatbook.Model_Artifacts.gguf_admission import GGUFParseError


REF = ArtifactRef("local-gguf-example", "sha256-abc", "q4-k-m")


def _descriptor(
    reference: ArtifactRef,
    *,
    role: ArtifactRole = ArtifactRole.ROOT,
    artifact_format: ArtifactFormat = ArtifactFormat.GGUF,
    files: tuple[ArtifactFile, ...] | None = None,
    model_id: str = "Example Model",
) -> ArtifactDescriptor:
    declared_files = files or (ArtifactFile("model.gguf", 1024 * 1024, "0" * 64),)
    return ArtifactDescriptor(
        reference=reference,
        model_id=model_id,
        role=role,
        format=artifact_format,
        consumer="local-llm",
        model_family="llama",
        upstream_repository="local-import",
        upstream_revision="local-import",
        source_url="",
        precision=reference.variant,
        expected_installed_bytes=sum(item.size_bytes for item in declared_files),
        license_id="unknown",
        license_url="",
        usage_notice="User supplied local model",
        runtime_name="local-llm",
        runtime_version_constraint="unconstrained",
        supported_os=("linux",),
        supported_architectures=("x86_64",),
        provenance=(ProvenanceClass.LOCAL_INTEGRITY_RECORDED,),
        files=declared_files,
    )


def _installed(
    descriptor: ArtifactDescriptor | None,
    *,
    name: str,
    ready: bool = True,
    error: str | None = None,
) -> InstalledArtifact:
    return InstalledArtifact(
        path=Path("/private/managed-root") / name,
        descriptor=descriptor,
        ready=ready,
        active=False,
        error=error,
    )


@dataclass
class _Lease:
    handle: ArtifactHandle
    closed: bool = False

    def close(self) -> None:
        self.closed = True


class _Service:
    def __init__(
        self,
        installed: tuple[InstalledArtifact, ...],
        lease: _Lease,
        *,
        list_error: BaseException | None = None,
    ) -> None:
        self.installed = installed
        self.lease = lease
        self.list_error = list_error
        self.events: list[tuple[str, object]] = []

    def acquire(self, reference: ArtifactRef) -> _Lease:
        self.events.append(("acquire", reference))
        return self.lease

    def list_installed(self) -> tuple[InstalledArtifact, ...]:
        self.events.append(("list", None))
        if self.list_error is not None:
            raise self.list_error
        return self.installed


class _PostAcquireCancellation(BaseException):
    pass


def _lease(reference: ArtifactRef, root: Path) -> _Lease:
    return _Lease(
        ArtifactHandle(
            root=reference,
            closure=(reference,),
            closure_fingerprint="fingerprint",
            paths=((reference, root),),
        )
    )


def test_legacy_source_values_map_without_importing() -> None:
    llamacpp = initial_gguf_selection("llamacpp", "outside.gguf")
    llamafile = initial_gguf_selection("llamafile", "outside.gguf")

    assert llamacpp.mode is GGUFSourceMode.EXTERNAL
    assert llamacpp.external_path == Path("outside.gguf")
    assert llamafile.mode is GGUFSourceMode.EXTERNAL
    assert llamafile.external_path == Path("outside.gguf")
    assert initial_gguf_selection("llamafile", "").mode is GGUFSourceMode.EMBEDDED


def test_blank_llamacpp_remains_external_for_later_launch_validation() -> None:
    selected = initial_gguf_selection("llamacpp", "")

    assert selected.mode is GGUFSourceMode.EXTERNAL
    assert selected.external_path is None
    assert selected.managed_ref is None


def test_provider_mode_compatibility_is_exact() -> None:
    embedded = GGUFSourceSelection(GGUFSourceMode.EMBEDDED)
    with pytest.raises(ValueError, match="llamacpp"):
        embedded.validate_for("llamacpp")

    for mode in GGUFSourceMode:
        assert GGUFSourceSelection(mode).validate_for("llamafile").mode is mode

    with pytest.raises(ValueError, match="unsupported"):
        GGUFSourceSelection(GGUFSourceMode.EXTERNAL).validate_for("llama-cpp")


def test_source_selection_preserves_inactive_values_without_exposing_path() -> None:
    selected = GGUFSourceSelection(
        mode=GGUFSourceMode.EXTERNAL,
        managed_ref=REF,
        external_path=Path("/private/sentinel.gguf"),
    )

    managed = selected.for_mode(GGUFSourceMode.MANAGED)
    assert managed.managed_ref == REF
    assert managed.external_path == Path("/private/sentinel.gguf")
    assert managed.authority == "Managed GGUF"
    rendered = repr({"selected": (selected, managed)}) + str((selected, managed))
    assert "/private/sentinel.gguf" not in rendered


def test_selection_requires_exact_state_types() -> None:
    with pytest.raises(TypeError, match="mode"):
        GGUFSourceSelection("external")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="managed_ref"):
        GGUFSourceSelection(GGUFSourceMode.MANAGED, managed_ref="latest")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="external_path"):
        GGUFSourceSelection(GGUFSourceMode.EXTERNAL, external_path="model.gguf")  # type: ignore[arg-type]


def test_initial_selection_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        initial_gguf_selection("vllm", "model.gguf")


def test_managed_inventory_includes_only_healthy_ready_root_gguf() -> None:
    selectable = _descriptor(REF)
    dependency_ref = ArtifactRef("dependency", "revision", "q4-k-m")
    onnx_ref = ArtifactRef("onnx-model", "revision", "int8")
    not_ready_ref = ArtifactRef("not-ready", "revision", "q5-k-m")
    broken_ref = ArtifactRef("broken", "revision", "q8-0")
    inventory = (
        _installed(selectable, name="selectable"),
        _installed(
            _descriptor(dependency_ref, role=ArtifactRole.DEPENDENCY),
            name="dependency",
        ),
        _installed(
            _descriptor(
                onnx_ref,
                artifact_format=ArtifactFormat.ONNX,
                files=(ArtifactFile("model.onnx", 12, "1" * 64),),
            ),
            name="onnx",
        ),
        _installed(
            _descriptor(not_ready_ref),
            name="not-ready",
            ready=False,
        ),
        _installed(
            _descriptor(broken_ref),
            name="broken",
            error="corrupt at /private/managed-root/broken",
        ),
        _installed(None, name="unreadable", error="private manifest failure"),
    )

    choices = managed_gguf_choices(inventory)

    assert len(choices) == 1
    assert isinstance(choices[0], ManagedGGUFChoice)
    assert choices[0].reference == REF
    assert "Example Model" in choices[0].label
    assert REF.variant in choices[0].label
    assert "1.0 MiB" in choices[0].label
    assert "Managed · local integrity recorded" in choices[0].label
    assert "/private/managed-root" not in repr({"choices": choices})


def test_acquire_managed_gguf_returns_exact_declared_payload_and_open_lease() -> None:
    root = Path("/private/managed-root") / "artifact"
    descriptor = _descriptor(
        REF,
        files=(
            ArtifactFile("payload/model.gguf", 1024, "2" * 64),
            ArtifactFile("notice.txt", 16, "3" * 64),
        ),
    )
    leased = _lease(REF, root)
    service = _Service((_installed(descriptor, name="artifact"),), leased)

    payload, returned_lease = acquire_managed_gguf(service, REF)

    assert service.events == [("acquire", REF), ("list", None)]
    assert payload == root / "payload/model.gguf"
    assert returned_lease is leased
    assert leased.closed is False


@pytest.mark.parametrize(
    ("installed", "expected_code"),
    [
        ((), "missing"),
        ((_installed(_descriptor(REF), name="not-ready", ready=False),), "not_ready"),
        (
            (
                _installed(
                    _descriptor(REF),
                    name="corrupt",
                    error="corrupt /private/managed-root/corrupt",
                ),
            ),
            "state",
        ),
        (
            (
                _installed(
                    _descriptor(
                        REF,
                        files=(
                            ArtifactFile("first.gguf", 1, "4" * 64),
                            ArtifactFile("second.GGUF", 1, "5" * 64),
                        ),
                    ),
                    name="multiple",
                ),
            ),
            "payload",
        ),
        (
            (
                _installed(
                    _descriptor(REF, role=ArtifactRole.DEPENDENCY),
                    name="dependency",
                ),
            ),
            "payload",
        ),
        (
            (
                _installed(
                    _descriptor(
                        REF,
                        artifact_format=ArtifactFormat.ONNX,
                        files=(ArtifactFile("model.gguf", 12, "6" * 64),),
                    ),
                    name="onnx",
                ),
            ),
            "payload",
        ),
    ],
)
def test_managed_payload_failures_close_lease_and_hide_private_state(
    installed: tuple[InstalledArtifact, ...],
    expected_code: str,
) -> None:
    leased = _lease(REF, Path("/private/managed-root/artifact"))
    service = _Service(installed, leased)

    with pytest.raises(GGUFSourceError) as caught:
        acquire_managed_gguf(service, REF)

    assert caught.value.code == expected_code
    assert leased.closed is True
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    rendered = repr((caught.value, caught.value.args, caught.value.__dict__))
    assert "/private/managed-root" not in rendered
    assert "corrupt" not in rendered


def test_mismatched_leased_root_closes_handle() -> None:
    other_ref = ArtifactRef("other-model", "revision", "q4-k-m")
    root = Path("/private/managed-root/artifact")
    leased = _Lease(
        ArtifactHandle(
            root=other_ref,
            closure=(REF, other_ref),
            closure_fingerprint="fingerprint",
            paths=((REF, root), (other_ref, root.parent / "other")),
        )
    )
    service = _Service((_installed(_descriptor(REF), name="artifact"),), leased)

    with pytest.raises(GGUFSourceError) as caught:
        acquire_managed_gguf(service, REF)

    assert caught.value.code == "payload"
    assert leased.closed is True


def test_post_acquire_base_exception_closes_lease_and_propagates_original() -> None:
    cancellation = _PostAcquireCancellation("cancelled")
    leased = _lease(REF, Path("/private/managed-root/artifact"))
    service = _Service((), leased, list_error=cancellation)

    with pytest.raises(_PostAcquireCancellation) as caught:
        acquire_managed_gguf(service, REF)

    assert caught.value is cancellation
    assert leased.closed is True
    assert caught.value.__context__ is None


def test_managed_inventory_exception_closes_lease_without_chaining_raw_error() -> None:
    leased = _lease(REF, Path("/private/managed-root/artifact"))
    service = _Service(
        (),
        leased,
        list_error=RuntimeError("raw failure at /private/managed-root/artifact"),
    )

    with pytest.raises(GGUFSourceError) as caught:
        acquire_managed_gguf(service, REF)

    assert leased.closed is True
    assert caught.value.__context__ is None
    assert "/private/managed-root" not in repr(caught.value)
    assert "raw failure" not in repr(caught.value)


def test_managed_integrity_exception_closes_lease_without_exposing_raw_error() -> None:
    leased = _lease(REF, Path("/private/managed-root/artifact"))
    service = _Service(
        (),
        leased,
        list_error=ArtifactIntegrityError(
            "digest mismatch at /private/managed-root/artifact"
        ),
    )

    with pytest.raises(GGUFSourceError) as caught:
        acquire_managed_gguf(service, REF)

    assert caught.value.code == "integrity"
    assert leased.closed is True
    assert caught.value.__context__ is None
    assert "/private/managed-root" not in repr(caught.value)
    assert "digest mismatch" not in repr(caught.value)


def test_acquire_managed_gguf_requires_exact_reference_before_service_call() -> None:
    class UnusedService:
        def acquire(self, _reference: ArtifactRef) -> NoReturn:
            pytest.fail("service must not be called for an inexact reference")

    with pytest.raises(TypeError, match="ArtifactRef"):
        acquire_managed_gguf(UnusedService(), "latest")  # type: ignore[arg-type]


def test_acquire_managed_gguf_maps_lease_timeout_to_retryable_busy_code() -> None:
    class BusyService:
        def acquire(self, _reference: ArtifactRef) -> NoReturn:
            raise ArtifactLeaseTimeoutError(
                "raw lock at /private/managed-root/locks/private.lock"
            )

    with pytest.raises(GGUFSourceError) as caught:
        acquire_managed_gguf(BusyService(), REF)  # type: ignore[arg-type]

    assert caught.value.code == "busy"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "/private/" not in repr(caught.value)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            ArtifactNotReadyError("raw /private/sentinel.gguf"),
            "The selected managed GGUF is not ready. Choose another model or import it again.",
        ),
        (
            ArtifactIntegrityError("raw /private/sentinel.gguf"),
            "The selected managed GGUF is corrupt. Delete it and import it again.",
        ),
        (
            ArtifactLeaseTimeoutError("raw /private/sentinel.gguf"),
            "The managed model store is busy. Try again.",
        ),
        (
            ArtifactStateError("raw /private/sentinel.gguf"),
            "The managed model store is unavailable. Try again.",
        ),
        (
            GGUFParseError("raw /private/sentinel.gguf"),
            "The selected file is not a valid GGUF. Choose another file.",
        ),
        (
            RuntimeError("raw /private/sentinel.gguf"),
            "The GGUF source could not be prepared. Try again or choose another source.",
        ),
    ],
)
def test_failure_messages_are_stable_and_path_private(
    error: BaseException,
    expected: str,
) -> None:
    message = gguf_source_failure_message(error)

    assert message == expected
    assert "/private/sentinel.gguf" not in message
