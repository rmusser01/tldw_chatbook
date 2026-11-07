"""Exception frame projection is ordered and contains only identifiers."""


def test_exception_frames_preserve_order_without_exception_text_or_paths():
    from tldw_chatbook.app_lifecycle import _exception_frames

    def inner():
        raise ValueError("private-sentinel-content")

    def outer():
        inner()

    try:
        outer()
    except ValueError as error:
        frames = _exception_frames(error)

    assert [name for _, name, _ in frames][-2:] == ["outer", "inner"]
    assert all(isinstance(line, int) for _, _, line in frames)
    assert "private-sentinel-content" not in repr(frames)
    assert all("/" not in module and "\\" not in module for module, _, _ in frames)
