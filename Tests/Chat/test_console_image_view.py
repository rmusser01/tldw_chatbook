from io import BytesIO

from PIL import Image as PILImage

from tldw_chatbook.Chat.console_image_view import (
    IMAGE_CACHE_MAX_ENTRIES,
    IMAGE_DECODE_MAX_DIMENSION,
    ConsoleImageRenderCache,
    ConsoleImageViewState,
    next_view_mode,
    resolve_default_mode,
)


def _png_bytes(size=(64, 64), color=(200, 10, 10)) -> bytes:
    buffer = BytesIO()
    PILImage.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_next_view_mode_cycles_three_states():
    assert next_view_mode("pixels") == "graphics"
    assert next_view_mode("graphics") == "hidden"
    assert next_view_mode("hidden") == "pixels"


def test_resolve_default_mode_explicit_config_wins(monkeypatch):
    import tldw_chatbook.Chat.console_image_view as civ

    monkeypatch.setattr(
        civ, "detect_terminal_capabilities", lambda: {"terminal_type": "kitty"}
    )
    assert resolve_default_mode({"chat": {"images": {"default_render_mode": "pixels"}}}) == "pixels"
    assert resolve_default_mode({"chat": {"images": {"default_render_mode": "regular"}}}) == "graphics"


def test_resolve_default_mode_auto_uses_terminal_override(monkeypatch):
    import tldw_chatbook.Chat.console_image_view as civ

    monkeypatch.setattr(
        civ, "detect_terminal_capabilities", lambda: {"terminal_type": "kitty"}
    )
    config = {
        "chat": {
            "images": {
                "default_render_mode": "auto",
                "terminal_overrides": {"kitty": "regular", "default": "pixels"},
            }
        }
    }
    assert resolve_default_mode(config) == "graphics"


def test_resolve_default_mode_auto_falls_back_to_default_override(monkeypatch):
    import tldw_chatbook.Chat.console_image_view as civ

    monkeypatch.setattr(
        civ, "detect_terminal_capabilities", lambda: {"terminal_type": "xterm"}
    )
    config = {
        "chat": {
            "images": {
                "default_render_mode": "auto",
                "terminal_overrides": {"kitty": "regular", "default": "pixels"},
            }
        }
    }
    assert resolve_default_mode(config) == "pixels"


def test_resolve_default_mode_auto_without_overrides_uses_capability_mode(monkeypatch):
    import tldw_chatbook.Chat.console_image_view as civ

    monkeypatch.setattr(
        civ, "detect_terminal_capabilities", lambda: {"terminal_type": "unknown"}
    )
    monkeypatch.setattr(civ, "get_image_render_mode", lambda mode: "regular")
    assert resolve_default_mode({"chat": {"images": {"default_render_mode": "auto"}}}) == "graphics"


def test_resolve_default_mode_garbage_falls_back_to_pixels(monkeypatch):
    import tldw_chatbook.Chat.console_image_view as civ

    monkeypatch.setattr(
        civ, "detect_terminal_capabilities", lambda: {"terminal_type": "unknown"}
    )
    monkeypatch.setattr(civ, "get_image_render_mode", lambda mode: "regular")
    assert resolve_default_mode({"chat": {"images": {"default_render_mode": "nonsense"}}}) in {
        "pixels",
        "graphics",
    }
    assert resolve_default_mode({}) in {"pixels", "graphics"}


def test_view_state_defaults_overrides_and_prune():
    state = ConsoleImageViewState()
    assert state.mode_for("m-1", default="pixels") == "pixels"

    state.set_mode("m-1", "hidden", default="pixels")
    assert state.mode_for("m-1", default="pixels") == "hidden"
    assert state.serialize() == {"m-1": "hidden"}

    # Setting back to the default drops the entry.
    state.set_mode("m-1", "pixels", default="pixels")
    assert state.serialize() == {}

    state.set_mode("m-1", "graphics", default="pixels")
    state.set_mode("m-2", "hidden", default="pixels")
    state.prune({"m-2"})
    assert state.serialize() == {"m-2": "hidden"}


def test_view_state_restore_ignores_invalid_entries():
    state = ConsoleImageViewState()
    state.restore({"m-1": "graphics", "m-2": "bogus", 3: "hidden"})
    assert state.serialize() == {"m-1": "graphics"}


def test_cache_prepares_downscales_and_serves_both_renderables():
    cache = ConsoleImageRenderCache()
    big = _png_bytes(size=(2048, 512))

    assert cache.prepare("m-1", big) is True
    pil = cache.get_pil("m-1")
    assert pil is not None
    assert max(pil.width, pil.height) <= IMAGE_DECODE_MAX_DIMENSION
    assert cache.get_pixels("m-1") is not None
    assert cache.get_pixels("m-1") is cache.get_pixels("m-1")  # lazy build cached


def test_cache_negative_caches_corrupt_bytes():
    cache = ConsoleImageRenderCache()
    assert cache.prepare("m-bad", b"not an image") is False
    assert cache.is_failed("m-bad") is True
    assert cache.get_pil("m-bad") is None


def test_cache_lru_bound_evicts_oldest():
    cache = ConsoleImageRenderCache()
    payload = _png_bytes(size=(8, 8))
    for index in range(IMAGE_CACHE_MAX_ENTRIES + 1):
        cache.prepare(f"m-{index}", payload)
    assert cache.get_pil("m-0") is None  # evicted
    assert cache.get_pil(f"m-{IMAGE_CACHE_MAX_ENTRIES}") is not None


def test_cache_pending_ids_and_session_eviction():
    class _Message:
        def __init__(self, message_id, image_data):
            self.id = message_id
            self.image_data = image_data

    cache = ConsoleImageRenderCache()
    payload = _png_bytes()
    cache.prepare("m-done", payload)
    cache.prepare("m-bad", b"junk")
    messages = [
        _Message("m-done", payload),
        _Message("m-bad", b"junk"),
        _Message("m-new", payload),
        _Message("m-none", None),
    ]
    pending = cache.pending_ids(messages)
    assert [message_id for message_id, _ in pending] == ["m-new"]

    cache.evict_session({"m-done"})
    assert cache.get_pil("m-done") is None


def test_resolve_default_mode_reads_live_app_config_shape(monkeypatch):
    """The real app nests raw TOML under COMPREHENSIVE_CONFIG_RAW (config.py:1326)."""
    import tldw_chatbook.Chat.console_image_view as civ

    monkeypatch.setattr(
        civ, "detect_terminal_capabilities", lambda: {"terminal_type": "unknown"}
    )
    live_shape = {
        "APP_MODE_STR": "single",
        "COMPREHENSIVE_CONFIG_RAW": {
            "chat": {"images": {"default_render_mode": "regular"}}
        },
    }
    assert resolve_default_mode(live_shape) == "graphics"


def test_resolve_default_mode_live_shape_terminal_override(monkeypatch):
    import tldw_chatbook.Chat.console_image_view as civ

    monkeypatch.setattr(
        civ, "detect_terminal_capabilities", lambda: {"terminal_type": "kitty"}
    )
    live_shape = {
        "COMPREHENSIVE_CONFIG_RAW": {
            "chat": {
                "images": {
                    "default_render_mode": "auto",
                    "terminal_overrides": {"kitty": "regular", "default": "pixels"},
                }
            }
        }
    }
    assert resolve_default_mode(live_shape) == "graphics"
