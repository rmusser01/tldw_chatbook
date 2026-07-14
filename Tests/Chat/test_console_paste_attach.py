from io import BytesIO

from PIL import Image as PILImage

from tldw_chatbook.Chat import console_paste_attach as cpa
from tldw_chatbook.Chat.console_paste_attach import (
    ClipboardGrab,
    DroppedPaste,
    extract_dropped_path,
    grab_clipboard_image,
    looks_attachable,
)


# --- extract_dropped_path matrix ---

def test_extracts_plain_absolute_path():
    result = extract_dropped_path("/Users/me/Pictures/photo.png")
    assert result == DroppedPaste(path="/Users/me/Pictures/photo.png", total_dropped=1)


def test_extracts_path_with_trailing_newline_and_spaces():
    result = extract_dropped_path("/tmp/a.png \n")
    assert result is not None and result.path == "/tmp/a.png"


def test_extracts_single_quoted_path():
    result = extract_dropped_path("'/Users/me/My Files/photo.png'")
    assert result is not None and result.path == "/Users/me/My Files/photo.png"


def test_extracts_double_quoted_path():
    result = extract_dropped_path('"/Users/me/My Files/photo.png"')
    assert result is not None and result.path == "/Users/me/My Files/photo.png"


def test_extracts_backslash_escaped_spaces():
    result = extract_dropped_path("/Users/me/My\\ Files/photo.png")
    assert result is not None and result.path == "/Users/me/My Files/photo.png"


def test_extracts_file_uri_with_percent_encoding():
    result = extract_dropped_path("file:///Users/me/My%20Files/photo.png")
    assert result is not None and result.path == "/Users/me/My Files/photo.png"


def test_multi_drop_returns_first_with_count():
    result = extract_dropped_path("/tmp/a.png\n/tmp/b.png\n/tmp/c.md\n")
    assert result == DroppedPaste(path="/tmp/a.png", total_dropped=3)


def test_prose_containing_a_path_is_not_a_drop():
    assert extract_dropped_path("what does /etc/hosts do?") is None


def test_multiline_prose_is_not_a_drop():
    assert extract_dropped_path("line one\nnot /a/path at all\n") is None


def test_relative_and_tilde_paths_are_not_drops():
    assert extract_dropped_path("notes.md") is None
    assert extract_dropped_path("~/notes.md") is None


def test_empty_and_whitespace_are_not_drops():
    assert extract_dropped_path("") is None
    assert extract_dropped_path("   \n") is None


# --- looks_attachable ---

def test_looks_attachable_true_for_supported_existing_in_root(tmp_path):
    target = tmp_path / "photo.png"
    target.write_bytes(b"x")
    assert looks_attachable(str(target), allowed_root=str(tmp_path)) is True


def test_looks_attachable_false_for_missing_out_of_root_unsupported(tmp_path):
    missing = tmp_path / "nope.png"
    assert looks_attachable(str(missing), allowed_root=str(tmp_path)) is False

    outside = tmp_path / "esc.png"
    outside.write_bytes(b"x")
    assert looks_attachable(str(outside), allowed_root=str(tmp_path / "inner")) is False

    unsupported = tmp_path / "binary.exe"
    unsupported.write_bytes(b"x")
    assert looks_attachable(str(unsupported), allowed_root=str(tmp_path)) is False


# --- grab_clipboard_image kind mapping (ImageGrab monkeypatched) ---

def _png_of(size=(8, 8)):
    return PILImage.new("RGB", size, (5, 5, 200))


def test_grab_maps_image_to_png_bytes(monkeypatch):
    monkeypatch.setattr(cpa, "_grabclipboard", lambda: _png_of())
    grab = grab_clipboard_image()
    assert grab.kind == "image"
    assert grab.png_bytes is not None
    assert PILImage.open(BytesIO(grab.png_bytes)).size == (8, 8)


def test_grab_maps_path_list_to_paths(monkeypatch):
    monkeypatch.setattr(cpa, "_grabclipboard", lambda: ["/tmp/a.png", "/tmp/b.md"])
    grab = grab_clipboard_image()
    assert grab.kind == "paths"
    assert grab.paths == ("/tmp/a.png", "/tmp/b.md")


def test_grab_maps_none_to_empty(monkeypatch):
    monkeypatch.setattr(cpa, "_grabclipboard", lambda: None)
    assert grab_clipboard_image().kind == "empty"


def test_grab_maps_errors_to_unavailable(monkeypatch):
    def _boom():
        raise OSError("no clipboard backend")

    monkeypatch.setattr(cpa, "_grabclipboard", _boom)
    assert grab_clipboard_image().kind == "unavailable"


def test_grab_encodes_cmyk_images_via_rgb_conversion(monkeypatch):
    monkeypatch.setattr(
        cpa, "_grabclipboard", lambda: PILImage.new("CMYK", (8, 8))
    )
    grab = grab_clipboard_image()
    assert grab.kind == "image"
    assert grab.png_bytes is not None
    assert PILImage.open(BytesIO(grab.png_bytes)).mode == "RGB"


def test_grab_maps_unencodable_image_to_unavailable(monkeypatch):
    class _Unencodable:
        def save(self, *a, **k):
            raise OSError("cannot write")

        def convert(self, *a, **k):
            raise OSError("cannot convert")

    monkeypatch.setattr(cpa, "_grabclipboard", lambda: _Unencodable())
    assert grab_clipboard_image().kind == "unavailable"
