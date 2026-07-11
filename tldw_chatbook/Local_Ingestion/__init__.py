# tldw_chatbook/Local_Ingestion/__init__.py
"""
Local file ingestion module for programmatic access.

This module provides functions to ingest various file types (PDFs, documents,
e-books, audio, video, etc.) into the Media database without using the UI.

Lazy re-exports (PEP 562). The names in ``__all__`` below resolve on first
attribute access rather than at package-import time, via module
``__getattr__``. This matters for F3's spawn-Pool parse worker
(``ingest_parse_worker.py``): a spawned pool worker unpickles
``run_parse_job`` by its dotted module path
(``tldw_chatbook.Local_Ingestion.ingest_parse_worker.run_parse_job``), and
resolving *any* dotted submodule path always runs this package's
``__init__.py`` first -- standard Python import semantics, unavoidable even
for a direct submodule import. Before this became lazy, that meant every
spawned worker paid for eagerly importing ``local_file_ingestion``,
``audio_processing``, ``video_processing``, and ``transcription_service``
just to resolve one function -- defeating ``ingest_parse_worker``'s own
"module scope imports stdlib only, real work deferred into the function
body" design (see that module's docstring). Regular callers (e.g. ``from
tldw_chatbook.Local_Ingestion import ingest_local_file``, used by
``app.py``) are unaffected: the first access still returns the exact same
object, just resolved on demand instead of at package-init time, and the
result is cached on the module so later access is a plain attribute
lookup.
"""

from typing import Any

__all__ = [
    'ingest_local_file',
    'batch_ingest_files',
    'ingest_directory',
    'quick_ingest',
    'detect_file_type',
    'get_supported_extensions',
    'FileIngestionError',
    # Audio/Video processing
    'LocalAudioProcessor',
    'LocalVideoProcessor',
    'TranscriptionService',
    'AudioProcessingError',
    'VideoProcessingError',
    'TranscriptionError'
]

# Name -> submodule providing it. Kept as a flat mapping (rather than one
# `from .x import *`-style block per submodule) so `__getattr__` only ever
# imports the one submodule that actually owns the requested name.
_SUBMODULE_BY_NAME = {
    'ingest_local_file': 'local_file_ingestion',
    'batch_ingest_files': 'local_file_ingestion',
    'ingest_directory': 'local_file_ingestion',
    'quick_ingest': 'local_file_ingestion',
    'detect_file_type': 'local_file_ingestion',
    'get_supported_extensions': 'local_file_ingestion',
    'FileIngestionError': 'local_file_ingestion',
    'LocalAudioProcessor': 'audio_processing',
    'AudioProcessingError': 'audio_processing',
    'LocalVideoProcessor': 'video_processing',
    'VideoProcessingError': 'video_processing',
    'TranscriptionService': 'transcription_service',
    'TranscriptionError': 'transcription_service',
}


def __getattr__(name: str) -> Any:
    submodule_name = _SUBMODULE_BY_NAME.get(name)
    if submodule_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    submodule = importlib.import_module(f".{submodule_name}", __name__)
    value = getattr(submodule, name)
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value
