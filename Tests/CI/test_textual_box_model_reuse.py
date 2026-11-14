"""ADR362: share only equal results already owned by the native widget LRU."""

import json
import subprocess
import sys
from fractions import Fraction

import pytest
from textual import widget as widget_module
from textual.cache import LRUCache
from textual.geometry import Size
from textual.widget import Widget

import tldw_chatbook.Widgets  # noqa: F401 - normal compatibility entry
from tldw_chatbook import _install_textual_compatibility_shims


def test_equal_new_native_keys_reuse_one_owner_result_without_skipping_misses():
    """Removing reuse retains40 equal BoxModels in distinct native results."""
    widget = Widget()
    widget.styles.width, widget.styles.height = 10, 3
    results = []
    for index in range(40):
        results.append(
            widget._get_box_model(
                Size(80, 24), Size(80 + index, 24), Fraction(1), Fraction(1)
            )
        )
    assert all(result.width == 10 and result.height == 3 for result in results)
    assert all(result is results[0] for result in results)
    assert widget._box_model_cache.misses == 40
    assert widget._box_model_cache.hits == 0
    assert len(widget._box_model_cache) == widget._box_model_cache.maxsize == 16


def _box(widget, index):
    return widget._get_box_model(
        Size(80, 24), Size(80 + index, 24), Fraction(1), Fraction(1)
    )


def test_reuse_keeps_every_native_computation_hit_promotion_and_eviction(monkeypatch):
    """Skipping a native call or extra cache.get corrupts23 misses/three hits."""
    native = Widget._get_box_model.__wrapped__
    control, subject = Widget(), Widget()
    for widget in (control, subject):
        widget.styles.width, widget.styles.height = 10, 3
    inputs = [*range(20), 4, 5, 20, 4, 6, 1]
    expected = [
        native(control, Size(80, 24), Size(80 + i, 24), Fraction(1), Fraction(1))
        for i in inputs
    ]
    constructor = widget_module.BoxModel
    constructed = []

    def construct(*args):
        model = constructor(*args)
        constructed.append(model)
        return model

    monkeypatch.setattr(widget_module, "BoxModel", construct)
    actual = [_box(subject, i) for i in inputs]
    assert actual == expected
    assert len(constructed) == 23
    assert all(model is actual[0] for model in actual)
    assert subject._extrema == control._extrema
    for cache in (control._box_model_cache, subject._box_model_cache):
        assert (cache.misses, cache.hits, len(cache), cache.maxsize) == (23, 3, 16, 16)

    def order(cache):
        rows, link = [], cache._head
        for _ in range(len(cache)):
            rows.append(link[2][1])
            link = link[1]
        assert link is cache._head
        return rows

    assert order(subject._box_model_cache) == order(control._box_model_cache)


def test_cache_clear_replacement_other_owner_and_new_geometry_end_reuse():
    widget, other = Widget(), Widget()
    for owner in (widget, other):
        owner.styles.width, owner.styles.height = 10, 3
    attributes = set(widget.__dict__)
    first = _box(widget, 0)
    assert _box(widget, 1) is first
    assert _box(other, 0) == first and _box(other, 1) is not first
    widget._box_model_cache.clear()
    cleared = _box(widget, 2)
    assert cleared == first and cleared is not first
    widget._box_model_cache = LRUCache(2)
    replaced = _box(widget, 3)
    assert replaced == cleared and replaced is not cleared
    assert _box(widget, 4) is replaced
    _box(widget, 5)
    assert len(widget._box_model_cache) == widget._box_model_cache.maxsize == 2
    widget.styles.width = 20
    changed = _box(widget, 6)
    assert changed.width == 20 and changed.height == 3 and changed is not replaced
    assert _box(widget, 7) is changed
    assert set(widget.__dict__) == attributes


def test_super_call_overrides_and_custom_lru_are_native_not_adapted():
    class Override(Widget):
        def _get_box_model(self, *args, **kwargs):
            return super()._get_box_model(*args, **kwargs)

    class CustomLRU(LRUCache):
        pass

    for widget in (Override(), Widget()):
        widget.styles.width, widget.styles.height = 10, 3
        if type(widget) is Widget:
            widget._box_model_cache = CustomLRU(16)
        first, second = _box(widget, 0), _box(widget, 1)
        assert first == second and first is not second


def test_existing_cache_and_installation_are_not_reset():
    widget = Widget()
    widget.styles.width, widget.styles.height = 10, 3
    first = _box(widget, 0)
    cache, method = widget._box_model_cache, Widget._get_box_model
    _install_textual_compatibility_shims()
    _install_textual_compatibility_shims()
    assert Widget._get_box_model is method and widget._box_model_cache is cache
    assert _box(widget, 0) is first
    assert (cache.hits, cache.misses) == (1, 1)


@pytest.mark.parametrize(
    ("constrain", "greedy", "width"),
    [(False, True, 7), (True, True, 5), (False, False, 10), (True, False, 5)],
)
def test_fraction_and_constrained_geometry_keep_native_values(constrain, greedy, width):
    class ContentWidth(Widget):
        def get_content_width(self, container, viewport):
            return 10

    widget = ContentWidth()
    widget.styles.width, widget.styles.height = "2fr", "3fr"
    result = widget._get_box_model(
        Size(5, 20), Size(80, 24), Fraction(7, 2), Fraction(5, 2), constrain, greedy
    )
    assert (result.width, result.height, tuple(result.margin)) == (
        width,
        Fraction(15, 2),
        (0, 0, 0, 0),
    )


@pytest.mark.parametrize("change", ["clear", "replace", "error"])
def test_cache_changes_and_native_errors_during_calculation_are_not_hidden(change):
    class ContentWidth(Widget):
        def get_content_width(self, container, viewport):
            if change == "error":
                raise ValueError("native content error")
            if change == "clear":
                self._box_model_cache.clear()
            else:
                self._box_model_cache = LRUCache(16)
            return 10

    widget = ContentWidth()
    widget.styles.width, widget.styles.height = "auto", 3
    if change == "error":
        with pytest.raises(ValueError, match="native content error"):
            _box(widget, 0)
        assert len(widget._box_model_cache) == 0
    else:
        first, second = _box(widget, 0), _box(widget, 1)
        assert first == second and first is not second
        assert len(widget._box_model_cache) == 1


def _child(source):
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.parametrize("mismatch", ["version", "method", "cache_method", "setitem"])
def test_unsupported_native_shape_is_not_modified(mismatch):
    _child(f"""
import textual
from textual.widget import Widget
from textual.cache import LRUCache
if {mismatch!r} == "version":
    textual.__version__ = "8.2.9"
elif {mismatch!r} == "method":
    Widget._get_box_model = lambda *args: "foreign geometry"
elif {mismatch!r} == "cache_method":
    LRUCache.get = lambda *args: None
else:
    LRUCache.__setitem__ = lambda *args: None
native = Widget._get_box_model
import tldw_chatbook.Widgets
assert Widget._get_box_model is native
""")


def test_installer_does_not_retain_native_source_lines_in_global_linecache():
    """Compatibility validation must not add whole native files to the GC heap."""
    _child("""
import linecache
from textual import cache, widget
paths = (cache.__file__, widget.__file__)
before = {path: linecache.cache.get(path) for path in paths}
import tldw_chatbook.Widgets
assert all(linecache.cache.get(path) is before[path] for path in paths)
""")


_PAINT = '''
import asyncio, gc, json
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static
class Paint(App):
    CSS = """
    Screen { padding: 1; }
    Horizontal { width: 100%; height: auto; }
    Vertical { width: 1fr; height: auto; min-width: 5; max-width: 30; }
    Static { width: auto; height: auto; padding: 1; margin: 1; border: solid red; }
    #fraction { width: 1fr; height: 2fr; min-height: 4; max-height: 9; }
    """
    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical():
                yield Static("A界 é [literal]", markup=False)
            yield Static("fraction", id="fraction")
async def run():
    app = Paint()
    rows = []
    before = (gc.get_threshold(), gc.isenabled(), gc.get_freeze_count())
    async with app.run_test(size=(52, 20)) as pilot:
        for size in ((52, 20), (120, 50), (35, 12)):
            await pilot.resize_terminal(*size)
            await pilot.pause()
            rows.append({"regions": [(w.id, tuple(w.region), tuple(w.virtual_size))
                                     for w in app.screen.query("Static")],
                         "paint": [[(s.text, str(s.style), s.control) for s in strip]
                                   for strip in app.screen._compositor.render_strips()]})
    assert before == (gc.get_threshold(), gc.isenabled(), gc.get_freeze_count())
    print(json.dumps(rows))
asyncio.run(run())
'''


def test_native_geometry_unicode_paint_resize_and_gc_policy_are_unchanged():
    native = json.loads(_child(_PAINT))
    adapted = json.loads(_child("import tldw_chatbook.Widgets\n" + _PAINT))
    assert adapted == native
