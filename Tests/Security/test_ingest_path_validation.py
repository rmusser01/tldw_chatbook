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
    """A traversal-shaped path is refused before the text read opens anything.

    Raises:
        ValueError: Raised by ``validate_path_simple`` for the repeated-parent
            shape; without the guard the call reached ``stat()`` and raised
            ``FileNotFoundError`` instead.
    """
    from tldw_chatbook.Local_Ingestion import local_file_ingestion

    with pytest.raises(ValueError):
        local_file_ingestion.read_ingest_file_bytes(_TRAVERSAL)


def test_image_ingress_refuses_an_unvalidated_path():
    """A traversal-shaped path is refused before the image header is read.

    Raises:
        ValueError: Raised by ``validate_path_simple``; without the guard the
            decompression-bomb check returned normally and ``Image.open()``
            received the caller's raw path.
    """
    from tldw_chatbook.Local_Ingestion import Image_Processing_Lib

    with pytest.raises(ValueError):
        Image_Processing_Lib.reject_image_decompression_bomb(_TRAVERSAL)


def test_image_guard_hands_back_the_validated_path(tmp_path):
    """Callers must be able to use the RETURNED path, not their own input.

    Args:
        tmp_path: pytest fixture supplying an isolated directory, so the probe
            file cannot collide with another test's state.
    """
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


def test_legitimate_filenames_with_shell_characters_are_not_refused(tmp_path):
    """Shell metacharacters are legal in filenames and must not block ingestion.

    PR #2823 review (High): routing ingest through ``validate_path_simple``
    also applied its COMMAND-INJECTION blacklist, so ``Q3 P&L; final.txt`` --
    a file the OS creates happily and a user can legitimately select -- was
    refused before it was ever opened.

    Asserted at the validator rather than through ``read_ingest_file_bytes``
    because the full read trips ADR-126's ``RecoveryRequired`` gate in a clean
    worktree; the companion test below pins that ingest actually passes the
    opt-out, so the pair covers the boundary without needing the gate.

    Args:
        tmp_path: pytest fixture supplying an isolated directory to create the
            awkwardly-named probe files in.
    """
    from tldw_chatbook.Utils.path_validation import validate_path_simple

    for name in ("Q3 P&L; final.txt", "notes `draft`.md",
                 "budget ${2024}.csv", "a|b.txt"):
        probe = tmp_path / name
        probe.write_text("hello", encoding="utf-8")
        assert validate_path_simple(
            probe, require_exists=True, reject_shell_metacharacters=False
        ) == probe, name
        with pytest.raises(ValueError):
            # the default is deliberately unchanged for every other caller
            validate_path_simple(probe, require_exists=True)


def test_both_ingest_boundaries_opt_out_of_the_shell_blacklist(monkeypatch):
    """The opt-out must actually be passed, not just available.

    Args:
        monkeypatch: pytest fixture used to capture the kwargs each boundary
            hands the shared validator.
    """
    from tldw_chatbook.Local_Ingestion import local_file_ingestion, Image_Processing_Lib

    for module in (local_file_ingestion, Image_Processing_Lib):
        seen: dict = {}

        def _spy(path, *args, **kwargs):
            seen.update(kwargs)
            raise ValueError("stop here -- kwargs already captured")

        monkeypatch.setattr(module, "validate_path_simple", _spy)
        fn = (module.read_ingest_file_bytes
              if module is local_file_ingestion
              else module.reject_image_decompression_bomb)
        with pytest.raises(ValueError):
            fn("some-file.txt")
        assert seen.get("reject_shell_metacharacters") is False, module.__name__
        monkeypatch.undo()


def test_ingest_still_refuses_traversal_and_nul():
    """Opting out of the shell patterns must not weaken the real guards."""
    from tldw_chatbook.Local_Ingestion import local_file_ingestion

    for bad in (_TRAVERSAL, "~/secret.txt", "pl\x00ain.txt"):
        with pytest.raises(ValueError):
            local_file_ingestion.read_ingest_file_bytes(bad)
