"""Five classes of unbounded read on attacker-influenceable input (TASK-32895).

Each test pins one bound that did not exist. They are grouped in one file
because they are one defect class -- "this path accepts whatever the input
says" -- found across five unrelated packages by the same review sweep.

The decompression-bomb tests feed a SMALL input that decodes LARGE and assert
it is refused. The multi-frame GIF built by ``_animation_bomb`` is ~66 KB on
disk and decodes to 83,886,080 pixels, which is over the real
``MAX_ASSET_DECODED_PIXELS`` with no monkeypatched cap involved.
"""

from __future__ import annotations

import ast
import hashlib
import io
import subprocess
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


# Pillow WARNS above MAX_IMAGE_PIXELS and only RAISES above twice it, so the
# band between 1x and 2x is where an unguarded path silently decodes a bomb.
# Every pixel-bomb test below lands in that band on purpose: a fixture above
# 2x would pass against the unfixed code and prove nothing.
_WARNING_BAND_CAP = 100_000
_WARNING_BAND_SIDE = 400  # 160_000 px == 1.6x the cap


def _flat_png(width: int, height: int) -> bytes:
    """Return a flat PNG: tiny on disk, ``width * height`` pixels decoded."""
    buffer = io.BytesIO()
    Image.new("L", (width, height), 0).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _warning_band_png() -> bytes:
    """Return a ~1 KB PNG that decodes into Pillow's warn-but-decode band."""
    return _flat_png(_WARNING_BAND_SIDE, _WARNING_BAND_SIDE)


def _animation_bomb(frames: int = 5, side: int = 4096) -> bytes:
    """Return a small multi-frame GIF whose FRAMES multiply past the cap.

    A single frame at ``MAX_ASSET_DIMENSION`` is legal; the animation is not.
    Each frame differs by one pixel so Pillow does not collapse them.
    """
    images = []
    for index in range(frames):
        frame = Image.new("P", (side, side), 0)
        frame.putpixel((index, 0), index + 1)
        images.append(frame)
    buffer = io.BytesIO()
    images[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=10,
        disposal=2,
    )
    return buffer.getvalue()


# --------------------------------------------------------------------------
# 1. decoded-pixel caps on three image ingresses
# --------------------------------------------------------------------------


def test_persona_pack_import_refuses_animation_that_decodes_past_the_cap(tmp_path):
    """The pack importer LOADS every frame, so the cap must precede the loop.

    ``assets._decode_selected_frame`` has always enforced
    ``MAX_ASSET_DECODED_PIXELS``; ``importer._inspect_image`` runs first and
    did not, so the import path decoded what the load path refuses.
    """
    from tldw_chatbook.Persona_Visual import importer
    from tldw_chatbook.Persona_Visual.contracts import MAX_ASSET_DECODED_PIXELS

    data = _animation_bomb()
    assert len(data) < 512 * 1024, "the bomb must be small on disk"
    with Image.open(io.BytesIO(data)) as probe:
        decoded = probe.width * probe.height * probe.n_frames
    assert decoded > MAX_ASSET_DECODED_PIXELS, "fixture is not over the real cap"

    path = tmp_path / "bomb.gif"
    path.write_bytes(data)
    record = {
        "mime_type": "image/gif",
        "width": 4096,
        "height": 4096,
        "byte_count": len(data),
        "duration_ms": None,
    }
    with pytest.raises(ValueError):
        importer._inspect_image(path, record)


def test_image_format_conversion_refuses_a_pixel_bomb(monkeypatch):
    """A byte cap says nothing about the decode; the converter had only a byte cap."""
    from tldw_chatbook.Image_Generation.adapters import image_format_utils
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", _WARNING_BAND_CAP)
    content = _warning_band_png()
    assert len(content) < 64 * 1024

    with pytest.raises(ImageGenerationError):
        image_format_utils.maybe_convert_format(content, "image/png", "png", "jpg")


def test_image_ingest_refuses_a_pixel_bomb(monkeypatch, tmp_path):
    """``process_image`` had neither a byte cap nor a pixel cap."""
    from tldw_chatbook.Local_Ingestion import Image_Processing_Lib

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", _WARNING_BAND_CAP)
    path = tmp_path / "bomb.png"
    path.write_bytes(_warning_band_png())

    result = Image_Processing_Lib.process_image(
        path, enable_ocr=False, extract_features=False, perform_analysis=False
    )
    assert result["status"] == "Error"
    assert "pixels" in str(result["error"]).lower()


def test_ocr_ingress_refuses_a_pixel_bomb_without_swallowing_it(monkeypatch, tmp_path):
    """A policy rejection must not degrade into "OCR failed -> None"."""
    from tldw_chatbook.Local_Ingestion import Image_Processing_Lib

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", _WARNING_BAND_CAP)
    path = tmp_path / "bomb.png"
    path.write_bytes(_warning_band_png())

    with pytest.raises(ValueError):
        Image_Processing_Lib.extract_text_from_image(path)


# --------------------------------------------------------------------------
# 2. whole-file read caps on text ingestion
# --------------------------------------------------------------------------


def test_text_ingest_read_is_capped(monkeypatch, tmp_path):
    """Plaintext/HTML ingest read the whole file with no ceiling at all."""
    from tldw_chatbook.Local_Ingestion import local_file_ingestion

    monkeypatch.setattr(local_file_ingestion, "max_text_file_bytes", lambda: 16)
    path = tmp_path / "big.txt"
    path.write_bytes(b"x" * 64)

    with pytest.raises(local_file_ingestion.FileIngestionError) as excinfo:
        local_file_ingestion.read_ingest_file_bytes(path)
    assert "exceeds limit" in str(excinfo.value)

    small = tmp_path / "small.txt"
    small.write_bytes(b"ok")
    assert local_file_ingestion.read_ingest_file_bytes(small) == b"ok"


def test_text_ingest_cap_reads_the_media_processing_setting(monkeypatch):
    """Symmetric with max_audio_file_size_mb / max_video_file_size_mb."""
    from tldw_chatbook import config as config_module
    from tldw_chatbook.Local_Ingestion import local_file_ingestion

    seen: list[tuple] = []

    def fake_setting(key, default=None, *args, **kwargs):
        seen.append((key, default))
        return 7

    monkeypatch.setattr(config_module, "get_cli_setting", fake_setting)
    assert local_file_ingestion.max_text_file_bytes() == 7 * 1024 * 1024
    assert seen == [("media_processing.max_text_file_size_mb", 50)]


def test_mobi_basic_extraction_refuses_an_oversize_file(monkeypatch, tmp_path):
    """The MOBI fallback read the whole file, then walked it one byte at a time."""
    from tldw_chatbook.Local_Ingestion import Book_Ingestion_Lib, local_file_ingestion

    monkeypatch.setattr(local_file_ingestion, "max_text_file_bytes", lambda: 16)
    path = tmp_path / "book.mobi"
    path.write_bytes(b"A" * 4096)

    result = Book_Ingestion_Lib.process_mobi(
        str(path), perform_chunking=False, perform_analysis=False
    )
    assert result["status"] == "Error"
    assert "exceeds limit" in str(result["error"])


# --------------------------------------------------------------------------
# 3. the Kokoro download: a byte ceiling, and a digest that is compared
# --------------------------------------------------------------------------


class _Response:
    def __init__(self, chunks, headers=None):
        self._chunks = chunks
        self.headers = headers or {}

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=8192):
        yield from self._chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def kokoro_get(monkeypatch):
    """Replace ``requests.get`` in the Kokoro backend with a canned response."""
    from tldw_chatbook.TTS.backends import kokoro as kokoro_module

    holder: dict = {}
    monkeypatch.setattr(
        kokoro_module.requests, "get", lambda url, **kw: holder["response"]
    )
    return kokoro_module, holder


def test_kokoro_download_refuses_more_bytes_than_the_ceiling(kokoro_get, tmp_path):
    """An unbounded stream filled the disk; the ceiling is checked as it arrives."""
    kokoro_module, holder = kokoro_get
    holder["response"] = _Response([b"x" * 32, b"x" * 32])

    destination = tmp_path / "model.onnx"
    with pytest.raises(ValueError):
        kokoro_module._kokoro_stream_download(
            "https://example.invalid/model",
            str(destination),
            label="test",
            max_bytes=40,
        )
    assert not destination.exists()
    assert not list(tmp_path.glob("*.part")), "no partial file may be left behind"


def test_kokoro_download_refuses_an_oversize_declared_length(kokoro_get, tmp_path):
    """Fast-fail on content-length, the way the audio downloader does."""
    kokoro_module, holder = kokoro_get
    holder["response"] = _Response([b"x"], headers={"content-length": "999999"})

    with pytest.raises(ValueError):
        kokoro_module._kokoro_stream_download(
            "https://example.invalid/model",
            str(tmp_path / "model.onnx"),
            label="test",
            max_bytes=40,
        )


def test_kokoro_download_compares_the_digest_it_computes(kokoro_get, tmp_path):
    """The digest used to be computed, logged, and then ignored."""
    kokoro_module, holder = kokoro_get
    payload = b"payload-bytes"

    holder["response"] = _Response([payload])
    destination = tmp_path / "model.onnx"
    with pytest.raises(ValueError):
        kokoro_module._kokoro_stream_download(
            "https://example.invalid/model",
            str(destination),
            label="test",
            expected_sha256="0" * 64,
        )
    assert not destination.exists(), "a mismatched artifact must never land"

    holder["response"] = _Response([payload])
    kokoro_module._kokoro_stream_download(
        "https://example.invalid/model",
        str(destination),
        label="test",
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert destination.read_bytes() == payload


# --------------------------------------------------------------------------
# 4. a user-supplied regex compiled without validation (ReDoS)
# --------------------------------------------------------------------------


def test_chat_dictionary_key_rejects_a_catastrophic_pattern():
    """A dictionary key is compiled, then run against message text on the send path."""
    from tldw_chatbook.Character_Chat.Chat_Dictionary_Lib import ChatDictionary

    entry = ChatDictionary(key="/(a+)+$/", content="replacement")
    assert entry.is_regex is False, "nested quantifiers must not reach re.compile"
    assert entry.key == "/(a+)+$/", "the key degrades to a literal, as a bad regex does"


def test_chat_dictionary_key_still_accepts_an_ordinary_pattern():
    """The backstop must not break the feature it guards."""
    import re

    from tldw_chatbook.Character_Chat.Chat_Dictionary_Lib import ChatDictionary

    entry = ChatDictionary(key="/colou?r/i", content="hue")
    assert entry.is_regex is True
    assert isinstance(entry.key, re.Pattern)
    assert entry.matches("COLOR")


# --------------------------------------------------------------------------
# 5. subprocess calls with no timeout
# --------------------------------------------------------------------------


_TIMEOUT_REQUIRED = (
    "tldw_chatbook/Media/local_media_reading_service.py",
    "tldw_chatbook/STT/parakeet_onnx.py",
    "tldw_chatbook/STT/transcribe_cpp.py",
    "tldw_chatbook/TTS/audio_service.py",
)
# system_audio_tap calls an INJECTED runner rather than subprocess.run directly,
# so it is covered behaviourally by test_audiotap_helper_compile_is_bounded_*.

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _subprocess_run_calls(module_path: Path):
    """Yield ``(lineno, kwarg_names)`` for every ``subprocess.run`` in a module."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "run"
            and isinstance(func.value, ast.Name)
            and func.value.id == "subprocess"
        ):
            yield node.lineno, {kw.arg for kw in node.keywords}


@pytest.mark.parametrize("relative_path", _TIMEOUT_REQUIRED)
def test_subprocess_calls_are_bounded(relative_path):
    """A subprocess with no timeout pins the calling worker thread forever."""
    module_path = _REPO_ROOT / relative_path
    calls = list(_subprocess_run_calls(module_path))
    assert calls, f"no subprocess.run found in {relative_path}"
    missing = [lineno for lineno, kwargs in calls if "timeout" not in kwargs]
    assert not missing, f"{relative_path}: subprocess.run without timeout at {missing}"


def test_ffmpeg_m4b_call_does_not_inherit_stdin():
    """ffmpeg reads stdin for interactive commands and competes with the TUI."""
    module_path = _REPO_ROOT / "tldw_chatbook/TTS/audio_service.py"
    calls = list(_subprocess_run_calls(module_path))
    assert calls
    assert all("stdin" in kwargs for _, kwargs in calls)


def test_audiotap_helper_compile_is_bounded_and_never_raises(tmp_path):
    """``ensure_helper`` promises never to raise; a wedged swiftc is a failed build."""
    from tldw_chatbook.Audio import system_audio_tap

    recorded: list[dict] = []

    def fake_run(command, **kwargs):
        recorded.append(kwargs)
        raise subprocess.TimeoutExpired(cmd=command, timeout=kwargs.get("timeout", 0))

    result = system_audio_tap.ensure_helper(
        tmp_path / "cache",
        run=fake_run,
        which=lambda name: "/usr/bin/swiftc",
        executable="/nonexistent/python",
    )
    assert result is None
    if recorded:  # skipped early on a platform with no helper source
        assert recorded[0].get("timeout")
