"""Ingest reads go through the central path validator (PR #2802 Qodo 4 & 5).

``read_ingest_file_bytes`` and ``reject_image_decompression_bomb`` are the
single choke points for text-shaped and image-shaped ingestion respectively:
plaintext, HTML and the MOBI fallback all reach the filesystem through the
first, ``process_image`` and ``extract_text_from_image`` through the second.
Both used to build a ``Path`` straight from the caller-supplied value and
stat/open it, so nothing central ever saw the path.

Each test feeds a traversal-shaped path that ``Utils/path_validation.py``
refuses. Without the fix these raise ``FileNotFoundError`` (text) or return
normally (image), not ``ValueError`` -- verified red before the fix landed.
"""

from __future__ import annotations

import pytest

# A shape validate_path_simple() refuses outright, independent of what is on
# disk: repeated parent references with no base directory to resolve against.
_TRAVERSAL = "../../etc/passwd"


def test_text_ingest_read_refuses_an_unvalidated_path():
    from tldw_chatbook.Local_Ingestion import local_file_ingestion

    with pytest.raises(ValueError):
        local_file_ingestion.read_ingest_file_bytes(_TRAVERSAL)


def test_image_ingress_refuses_an_unvalidated_path():
    from tldw_chatbook.Local_Ingestion import Image_Processing_Lib

    with pytest.raises(ValueError):
        Image_Processing_Lib.reject_image_decompression_bomb(_TRAVERSAL)


def test_image_guard_hands_back_the_validated_path(tmp_path):
    """Callers must be able to use the RETURNED path, not their own input."""
    from tldw_chatbook.Local_Ingestion import Image_Processing_Lib

    path = tmp_path / "plain.png"
    path.write_bytes(b"not really a png")

    assert Image_Processing_Lib.reject_image_decompression_bomb(str(path)) == path


def test_public_image_ingest_surfaces_the_refusal(tmp_path):
    """``process_image`` reports it as an error result rather than crashing."""
    from tldw_chatbook.Local_Ingestion import Image_Processing_Lib

    result = Image_Processing_Lib.process_image(
        _TRAVERSAL + ".png",
        enable_ocr=False,
        extract_features=False,
        perform_analysis=False,
    )
    assert result["status"] == "Error"
    # Specific on purpose: without the fix the guard falls through and the
    # later stat() produces a generic "Error processing image" instead.
    assert "dangerous pattern" in str(result["error"])
