"""DEFERRED TASK-1915 reference code; not exported or active in TASK-597."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .gguf_admission import (
    GGUFError,
    GGUFMetadata,
    _sanitize_display,
    normalize_platform_target,
    require_transcribe_cpp_architecture,
)
from .service import (
    ArtifactDescriptor,
    ArtifactFile,
    ArtifactFormat,
    ArtifactRef,
    ArtifactRole,
    ProvenanceClass,
)


__all__: tuple[str, ...] = ()

TRANSCRIBE_CPP_VERSION = "0.1.3"

_PINNED_RUNTIME_RELEASE = (0, 1, 3)
_RUNTIME_CONSTRAINT_CLAUSE = re.compile(
    r"(==|!=|<=|>=|~=|<|>)([0-9]+(?:\.[0-9]+){0,2})\Z",
    re.ASCII,
)


class GGUFAmbiguousCuratedMatchError(GGUFError):
    """Raised when curated registry entries conflict for identical bytes."""


def _release_tuple(value: str) -> tuple[tuple[int, int, int], int]:
    components = tuple(int(component) for component in value.split("."))
    padded = (*components, 0, 0)
    return (padded[0], padded[1], padded[2]), len(components)


def _compatible_release(
    target: tuple[int, int, int],
    release: tuple[int, int, int],
    component_count: int,
) -> bool:
    upper = list(release)
    upper_index = 0 if component_count <= 2 else component_count - 2
    upper[upper_index] += 1
    upper[upper_index + 1 :] = [0] * (2 - upper_index)
    return release <= target < tuple(upper)


def runtime_constraint_admits_pinned_version(constraint: str) -> bool:
    """Evaluate the bounded release grammar against transcribe.cpp 0.1.3."""
    if not isinstance(constraint, str) or not constraint:
        return False

    clauses = constraint.split(",")
    for raw_clause in clauses:
        clause = raw_clause.strip()
        match = _RUNTIME_CONSTRAINT_CLAUSE.fullmatch(clause)
        if match is None:
            return False
        operator, raw_release = match.groups()
        try:
            release, component_count = _release_tuple(raw_release)
        except ValueError:
            return False
        target = _PINNED_RUNTIME_RELEASE
        if operator == "==" and target != release:
            return False
        if operator == "!=" and target == release:
            return False
        if operator == "<" and not target < release:
            return False
        if operator == "<=" and not target <= release:
            return False
        if operator == ">" and not target > release:
            return False
        if operator == ">=" and not target >= release:
            return False
        if operator == "~=" and not _compatible_release(
            target,
            release,
            component_count,
        ):
            return False
    return True


def _eligible_curated_descriptor(
    descriptor: ArtifactDescriptor,
    *,
    sha256: str,
    size_bytes: int,
    platform_target: tuple[str, str],
) -> bool:
    supported_os, supported_architecture = platform_target
    return (
        descriptor.role is ArtifactRole.ROOT
        and descriptor.format is ArtifactFormat.GGUF
        and descriptor.consumer == "transcribe-cpp"
        and descriptor.runtime_name == "transcribe-cpp"
        and len(descriptor.files) == 1
        and not descriptor.dependencies
        and descriptor.files[0].size_bytes == size_bytes
        and descriptor.files[0].sha256 == sha256
        and supported_os in descriptor.supported_os
        and supported_architecture in descriptor.supported_architectures
        and ProvenanceClass.CHATBOOK_CURATED in descriptor.provenance
        and runtime_constraint_admits_pinned_version(
            descriptor.runtime_version_constraint
        )
    )


def _local_model_label(metadata: GGUFMetadata) -> str:
    if isinstance(metadata.model_name, str):
        label = _sanitize_display(metadata.model_name).strip()
        if label:
            return label
    return f"{metadata.architecture} local GGUF"


def _local_gguf_descriptor(
    metadata: GGUFMetadata,
    *,
    sha256: str,
    size_bytes: int,
    platform_target: tuple[str, str],
) -> ArtifactDescriptor:
    precision = (
        f"filetype-{metadata.file_type}"
        if type(metadata.file_type) is int
        else "unknown"
    )
    reference = ArtifactRef(
        artifact_id=f"local-gguf-{metadata.architecture}-{sha256[:16]}",
        revision=sha256,
        variant=precision,
    )
    supported_os, supported_architecture = platform_target
    return ArtifactDescriptor(
        reference=reference,
        model_id=_local_model_label(metadata),
        role=ArtifactRole.ROOT,
        format=ArtifactFormat.GGUF,
        consumer="transcribe-cpp",
        model_family=metadata.architecture,
        upstream_repository="local-import",
        upstream_revision=sha256,
        source_url="https://local.invalid/gguf-import",
        precision=precision,
        expected_installed_bytes=size_bytes,
        license_id="NOASSERTION",
        license_url="https://local.invalid/noassertion",
        usage_notice="Local GGUF metadata was inspected; runtime use is manual only.",
        runtime_name="transcribe-cpp",
        runtime_version_constraint=f"=={TRANSCRIBE_CPP_VERSION}",
        supported_os=(supported_os,),
        supported_architectures=(supported_architecture,),
        provenance=(ProvenanceClass.LOCAL_INTEGRITY_RECORDED,),
        files=(ArtifactFile("model.gguf", size_bytes, sha256),),
        dependencies=(),
    )


def select_gguf_descriptor(
    metadata: GGUFMetadata,
    *,
    sha256: str,
    size_bytes: int,
    curated_descriptors: Iterable[ArtifactDescriptor] = (),
    system: str,
    machine: str,
) -> ArtifactDescriptor:
    """Reuse one exact curated match or build a deterministic local descriptor."""
    platform_target = normalize_platform_target(system, machine)
    require_transcribe_cpp_architecture(metadata.architecture)
    normalized_sha256 = sha256.lower()
    eligible = tuple(
        descriptor
        for descriptor in curated_descriptors
        if _eligible_curated_descriptor(
            descriptor,
            sha256=normalized_sha256,
            size_bytes=size_bytes,
            platform_target=platform_target,
        )
    )
    if len(eligible) > 1:
        raise GGUFAmbiguousCuratedMatchError(
            "multiple eligible curated descriptors match the GGUF payload"
        )
    if eligible:
        return eligible[0]
    return _local_gguf_descriptor(
        metadata,
        sha256=normalized_sha256,
        size_bytes=size_bytes,
        platform_target=platform_target,
    )
