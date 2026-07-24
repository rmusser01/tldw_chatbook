from io import BytesIO

from PIL import Image as PILImage

from tldw_chatbook.Chat.console_image_view import (
    IMAGE_CACHE_MAX_ENTRIES,
    IMAGE_DECODE_MAX_DIMENSION,
    ConsoleImageRenderCache,
    ConsoleImageViewState,
    fit_image_cell_size,
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
    assert (
        resolve_default_mode({"chat": {"images": {"default_render_mode": "pixels"}}})
        == "pixels"
    )
    assert (
        resolve_default_mode({"chat": {"images": {"default_render_mode": "regular"}}})
        == "graphics"
    )


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
    assert (
        resolve_default_mode({"chat": {"images": {"default_render_mode": "auto"}}})
        == "graphics"
    )


def test_resolve_default_mode_garbage_falls_back_to_pixels(monkeypatch):
    import tldw_chatbook.Chat.console_image_view as civ

    monkeypatch.setattr(
        civ, "detect_terminal_capabilities", lambda: {"terminal_type": "unknown"}
    )
    monkeypatch.setattr(civ, "get_image_render_mode", lambda mode: "regular")
    # An unrecognized value pins to "pixels" immediately instead of falling
    # through to the terminal-auto path.
    assert (
        resolve_default_mode({"chat": {"images": {"default_render_mode": "nonsense"}}})
        == "pixels"
    )
    # Missing/empty behaves as "auto" -- it must still consult the
    # terminal-auto path (proven here by the patched `get_image_render_mode`
    # returning "regular" -> "graphics"), not fall back to "pixels".
    assert resolve_default_mode({}) == "graphics"


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


def test_cache_evict_session_drops_composite_variant_keys():
    """``evict_session`` is key-agnostic: composite ``f"{message_id}:{i}"``
    generation-card cache keys (see ``ConsoleGenerationCardSpec``) pop the
    same way plain message-id keys do -- images, pixels, AND failure marks
    all clear (regression guard for the Keep-leaves-stale-image bug: keep
    swaps store bytes but never invalidated these composite keys, so the
    card kept showing the pre-keep image)."""
    cache = ConsoleImageRenderCache()
    payload = _png_bytes()
    cache.prepare("gen-1:0", payload)
    cache.prepare("gen-1:1", payload)
    assert cache.get_pixels("gen-1:0") is not None  # populate the pixels cache too
    cache.prepare("gen-1:2", b"not an image")  # negative-cached (failed)
    assert cache.is_failed("gen-1:2") is True

    cache.evict_session([f"gen-1:{i}" for i in range(3)])

    assert cache.get_pil("gen-1:0") is None
    assert cache.get_pil("gen-1:1") is None
    assert cache.get_pixels("gen-1:0") is None
    assert cache.is_failed("gen-1:2") is False
    # A sibling message's own composite key is untouched by the eviction.
    cache.prepare("gen-2:0", payload)
    cache.evict_session([f"gen-1:{i}" for i in range(3)])
    assert cache.get_pil("gen-2:0") is not None


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


# --- fit_image_cell_size (P3b avatar guard, generalized for the transcript) ---


def test_fit_image_cell_size_returns_explicit_positive_ints_within_box():
    # The whole point: never "auto" (0-size crash) - always explicit >=1 ints
    # clamped to the box, for any realistic image.
    for w, h in [(512, 512), (1024, 256), (100, 900), (7, 3), (1, 1)]:
        cw, ch = fit_image_cell_size(w, h, 80, 40)
        assert isinstance(cw, int) and isinstance(ch, int)
        assert 1 <= cw <= 80 and 1 <= ch <= 40


def test_fit_image_cell_size_wide_image_fits_width():
    # A very wide image is width-bound: full 80 cols, fewer lines.
    cw, ch = fit_image_cell_size(1600, 200, 80, 40)
    assert cw == 80 and ch < 40


def test_fit_image_cell_size_tall_image_fits_height():
    # A very tall image is height-bound: full 40 lines, fewer cols.
    cw, ch = fit_image_cell_size(100, 1600, 80, 40)
    assert ch == 40 and cw < 80


def test_fit_image_cell_size_preserves_aspect_within_box():
    # Square pixels -> ~2:1 cell aspect (cells are ~2x taller than wide);
    # fitting into 80x40 should be width-bound at 80 with height ~ 80/2 = 40.
    cw, ch = fit_image_cell_size(400, 400, 80, 40)
    assert cw == 80 and ch == 40


def test_fit_image_cell_size_degenerate_returns_full_box():
    assert fit_image_cell_size(0, 100, 80, 40) == (80, 40)
    assert fit_image_cell_size(100, 0, 24, 10) == (24, 10)


def test_fit_image_cell_size_respects_arbitrary_box():
    # The avatar box (24x10) still works through the same helper.
    cw, ch = fit_image_cell_size(1000, 1000, 24, 10)
    assert 1 <= cw <= 24 and 1 <= ch <= 10


# --- resolve_show_character_avatar (P3c) -------------------------------------


def test_resolve_show_character_avatar_defaults_true():
    from tldw_chatbook.Chat.console_image_view import resolve_show_character_avatar

    assert resolve_show_character_avatar({}) is True
    assert resolve_show_character_avatar({"chat": {"images": {}}}) is True


def test_resolve_show_character_avatar_explicit_false():
    from tldw_chatbook.Chat.console_image_view import resolve_show_character_avatar

    assert resolve_show_character_avatar(
        {"chat": {"images": {"show_character_avatar": False}}}
    ) is False


def test_resolve_show_character_avatar_live_shape():
    from tldw_chatbook.Chat.console_image_view import resolve_show_character_avatar

    assert resolve_show_character_avatar(
        {"COMPREHENSIVE_CONFIG_RAW": {"chat": {"images": {"show_character_avatar": False}}}}
    ) is False
