"""Application-curated model descriptors and their download sources.

The registry structurally satisfies the acquisition catalog protocol without
loading the network-capable acquisition layer into worker processes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from .service import ArtifactDescriptor, ArtifactRef

if TYPE_CHECKING:
    from .acquisition import ArtifactCatalog  # noqa: F401


class CuratedRegistry:
    """Ordered catalog of model descriptors curated by the application."""

    def __init__(self) -> None:
        self._descriptors: dict[ArtifactRef, ArtifactDescriptor] = {}
        self._sources: dict[ArtifactRef, dict[str, str]] = {}

    def register(
        self,
        descriptor: ArtifactDescriptor,
        *,
        sources: Mapping[str, str],
    ) -> None:
        """Register one curated model and its credential-free file sources.

        Args:
            descriptor: Immutable model descriptor.
            sources: Relative file paths mapped to download URLs.
        """
        self._descriptors[descriptor.reference] = descriptor
        self._sources[descriptor.reference] = dict(sources)

    def list(self) -> tuple[ArtifactDescriptor, ...]:
        """Return registered descriptors in registration order.

        Returns:
            The registered descriptors.
        """
        return tuple(self._descriptors.values())

    def descriptor(self, ref: ArtifactRef) -> ArtifactDescriptor:
        """Return the descriptor for an exact reference.

        Args:
            ref: Exact model reference.

        Returns:
            The registered descriptor.

        Raises:
            KeyError: If ``ref`` is not registered.
        """
        return self._descriptors[ref]

    def sources(self, ref: ArtifactRef) -> dict[str, str]:
        """Return a copy of the file-source map for an exact reference.

        Args:
            ref: Exact model reference.

        Returns:
            Relative file paths mapped to download URLs.

        Raises:
            KeyError: If ``ref`` is not registered.
        """
        return dict(self._sources[ref])


_REGISTRY: CuratedRegistry | None = None


def curated_registry() -> CuratedRegistry:
    """Return the shared registry with the built-in models registered once.

    Returns:
        The process-wide curated registry.
    """
    global _REGISTRY
    if _REGISTRY is None:
        registry = CuratedRegistry()
        from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
            PARAKEET_PRECISIONS,
            parakeet_descriptor,
            parakeet_reference,
            parakeet_source_map,
            parakeet_vad_descriptor,
            parakeet_vad_reference,
        )
        from tldw_chatbook.Local_Ingestion.stt_batch_routing import (
            PARAKEET_V2_MODEL,
            PARAKEET_V3_MODEL,
        )

        source_map = parakeet_source_map()
        for model in (PARAKEET_V2_MODEL, PARAKEET_V3_MODEL):
            for precision in PARAKEET_PRECISIONS:
                reference = parakeet_reference(model, precision)
                registry.register(
                    parakeet_descriptor(model, precision),
                    sources=source_map[reference],
                )
        vad_reference = parakeet_vad_reference()
        registry.register(
            parakeet_vad_descriptor(),
            sources=source_map[vad_reference],
        )
        from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
            audio_cpp_curated_entries,
        )

        for descriptor, sources in audio_cpp_curated_entries():
            registry.register(descriptor, sources=sources)
        _REGISTRY = registry
    return _REGISTRY
