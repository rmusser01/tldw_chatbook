"""The pinned artifact manifest decoder must reject non-JSON constants.

Tier-2 S03/S04 P3: three of the four strict-JSON decoders in `TTS/` pass
`parse_constant=`; `audio_cpp_artifact_catalog` was the one that did not, so
`NaN` / `Infinity` / `-Infinity` -- which are not JSON -- decoded silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
    load_audio_cpp_artifact_source_manifest,
)


@pytest.mark.unit
@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_json_constants_are_refused_by_name(tmp_path: Path, constant: str):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"repository": "audio-cpp/audio.cpp-gguf", "size_bytes": %s}' % constant,
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-JSON constant"):
        load_audio_cpp_artifact_source_manifest(manifest, expected_commit=None)
