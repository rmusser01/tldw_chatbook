"""ADR226: remove unused FIFOs, not native rendering or eviction semantics."""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from rich.segment import Segment
from textual.cache import FIFOCache
from textual.strip import Strip

import tldw_chatbook.Widgets  # noqa: F401 - normal compatibility entry point
from tldw_chatbook import _install_textual_compatibility_shims

CACHES = (
    ("_divide_cache", 4),
    ("_crop_cache", 16),
    ("_style_cache", 16),
    ("_filter_cache", 4),
    ("_line_length_cache", 4),
    ("_crop_extend_cache", 4),
    ("_offsets_cache", 4),
)


def _fifo_count(strip: Strip) -> int:
    return sum(type(value) is FIFOCache for value in gc.get_referents(strip))


def _child(source: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_native_strip_allocates_no_unused_fifo_and_only_its_first_used_cache():
    """An eager constructor creates seven tracked objects for an unused line."""
    strip = Strip([Segment("plain")], 5)
    assert _fifo_count(strip) == 0
    cache = strip._crop_cache
    assert type(cache) is FIFOCache
    assert _fifo_count(strip) == 1
    assert strip._crop_cache is cache


@pytest.mark.parametrize(("name", "capacity"), CACHES)
def test_lazy_slot_keeps_native_capacity_fifo_eviction_and_instance_ownership(
    name, capacity
):
    first = Strip([Segment("one")], 3)
    second = Strip([Segment("two")], 3)
    cache = getattr(first, name)
    assert type(cache) is FIFOCache
    assert cache._maxsize == capacity
    assert getattr(first, name) is cache
    assert getattr(second, name) is not cache
    assert _fifo_count(first) == 1
    for key in range(capacity):
        cache[key] = str(key)
    assert cache.get(0) == "0"
    cache[capacity] = "last"
    assert cache.get(0) is None  # FIFO access does not promote the oldest entry.
    assert cache.get(capacity) == "last"
    assert len(cache) == capacity
    assert cache.hits == 2
    assert cache.misses == 1
    assert len(getattr(second, name)) == 0


def test_normal_ui_entry_is_idempotent_and_preserves_explicit_native_cache_values():
    constructor = Strip.__init__
    strip = Strip([Segment("line")], 4)
    cache = strip._divide_cache
    cache[(4,)] = [strip]
    _install_textual_compatibility_shims()
    _install_textual_compatibility_shims()
    assert Strip.__init__ is constructor
    assert strip._divide_cache is cache
    assert strip.divide([4]) == [strip]
    replacement = FIFOCache(3)
    strip._divide_cache = replacement
    assert strip._divide_cache is replacement


def test_strips_created_before_install_keep_their_existing_cache_and_results():
    _child("""
from rich.segment import Segment
from textual.strip import Strip
old = Strip([Segment("line")], 4)
cache = old._divide_cache
result = old.divide([2, 4])
import tldw_chatbook.Widgets
assert old._divide_cache is cache
assert old.divide([2, 4]) is result
assert [part.text for part in result] == ["li", "ne"]
import gc
from textual.cache import FIFOCache
new = Strip([Segment("new")], 3)
assert not any(type(value) is FIFOCache for value in gc.get_referents(new))
""")


@pytest.mark.parametrize("mismatch", ["version", "slot", "data_slot", "constructor"])
def test_unsupported_native_shape_is_not_partially_modified(mismatch):
    _child(f"""
import textual
from textual.strip import Strip
if {mismatch!r} == "version":
    textual.__version__ = "8.2.9"
elif {mismatch!r} == "slot":
    Strip._crop_cache = property(lambda self: None)
elif {mismatch!r} == "data_slot":
    Strip._segments = property(lambda self: None)
else:
    def foreign_constructor(self, *args, **kwargs):
        raise AssertionError("foreign constructor must not be invoked")
    Strip.__init__ = foreign_constructor
before = dict(Strip.__dict__)
import tldw_chatbook.Widgets
assert Strip.__init__ is before["__init__"]
for name in ("_divide_cache", "_crop_cache", "_style_cache", "_filter_cache",
             "_line_length_cache", "_crop_extend_cache", "_offsets_cache"):
    assert Strip.__dict__[name] is before[name]
""")


def test_concurrent_first_use_keeps_one_native_cache_owner(monkeypatch):
    """A first allocator yielding the GIL must not discard a competing cache."""
    strip = Strip([Segment("threaded")], 8)
    original = FIFOCache.__init__
    constructed = []
    ready = threading.Barrier(8)

    def construct(cache, capacity):
        original(cache, capacity)
        constructed.append(cache)
        time.sleep(0.01)

    def first_use():
        ready.wait(timeout=5)
        return strip._crop_cache

    monkeypatch.setattr(FIFOCache, "__init__", construct)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(first_use) for _ in range(8)]
        caches = [future.result(timeout=5) for future in futures]
    assert len(constructed) == 1
    assert all(cache is caches[0] for cache in caches)
    caches[0]["owned"] = "value"
    assert strip._crop_cache.get("owned") == "value"


def test_subclass_owned_cache_slot_still_uses_the_original_constructor():
    """A custom Strip owner must retain its own native constructor assignment."""

    class CustomStrip(Strip):
        @property
        def _crop_cache(self):
            return self._owned_cache

        @_crop_cache.setter
        def _crop_cache(self, cache):
            self._owned_cache = cache

    strip = CustomStrip([Segment("custom")], 6)
    assert type(getattr(strip, "_owned_cache", None)) is FIFOCache
    assert strip.crop(1, 4).text == "ust"


@pytest.mark.parametrize("entry", ["UI", "Widgets"])
def test_compatibility_import_preserves_collector_policy_and_freeze_count(entry):
    _child(f"""
import gc
before = (gc.get_threshold(), gc.isenabled(), gc.get_freeze_count())
import tldw_chatbook.{entry}
after = (gc.get_threshold(), gc.isenabled(), gc.get_freeze_count())
assert before == after, (before, after)
""")


_RENDER = """
from rich.console import Console
from rich.segment import Segment
from rich.style import Style
from textual.color import Color
from textual.filter import Monochrome
from textual.strip import Strip
base = Strip([Segment("A界", Style(color="red", meta={"token": "one"})),
              Segment(" bc", Style(bgcolor="blue"))], 6)
parts = [base, base.crop(1, 4), base.extend_cell_length(8),
         base.adjust_cell_length(3), base.apply_style(Style(bold=True)),
         *base.divide([1, 3, 6]), base.crop_extend(0, 8, Style(bgcolor="green")),
         base.apply_offsets(2, 5), base.apply_filter(Monochrome(), Color.parse("black"))]
console = Console(force_terminal=True, color_system="truecolor", width=20)
import json
print(json.dumps([{
    "text": strip.text, "length": strip.cell_length,
    "segments": [(s.text, str(s.style), s.style.meta if s.style else None, s.control)
                 for s in strip],
    "ansi": strip.render(console),
} for strip in parts], sort_keys=True))
"""


def test_native_unicode_rendering_filtering_and_selection_offsets_match_unmodified_process():
    original = json.loads(_child(_RENDER))
    adapted = json.loads(_child("import tldw_chatbook.Widgets\n" + _RENDER))
    assert adapted == original
    assert original[0]["text"] == "A界 bc"
    assert original[1]["text"] == "界 "
