"""ADR361: native color results must not retain per-widget line caches."""

from __future__ import annotations

import json
import subprocess
import sys
import weakref

import pytest
from rich.segment import Segment
from textual._styles_cache import StylesCache
from textual.color import Color
from textual.strip import Strip

import tldw_chatbook.Widgets  # noqa: F401 - actual compatibility entry
from tldw_chatbook import _install_textual_compatibility_shims


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


def test_discarded_native_color_owners_are_not_retained_without_collection():
    """Putting the unused owner back in the cache key retains every weakref."""
    refs = []
    values = []
    hits = StylesCache.get_inner_outer.cache_info().hits
    for _ in range(32):
        owner = StylesCache()
        owner._cache[0] = Strip([Segment("owned render line")])
        refs.append(weakref.ref(owner))
        values.append(owner.get_inner_outer(Color(17, 22, 31), Color(29, 47, 65, 0.5)))
    del owner
    assert all(ref() is None for ref in refs)
    assert all(value == values[0] for value in values)
    assert StylesCache.get_inner_outer.cache_info().hits - hits >= 31


def test_old_line_caches_and_subclass_overrides_are_not_replaced():
    owner = StylesCache()
    line = Strip([Segment("kept")])
    owner._cache[0] = line
    method = StylesCache.get_inner_outer
    _install_textual_compatibility_shims()
    _install_textual_compatibility_shims()
    assert StylesCache.get_inner_outer is method
    owner.get_inner_outer(Color(1, 2, 3), Color(4, 5, 6))
    assert owner._cache[0] is line

    class Custom(StylesCache):
        def get_inner_outer(self, base_background, background):
            return "custom", self._width

    assert Custom().get_inner_outer(Color(1, 2, 3), Color(4, 5, 6)) == ("custom", 1)


def test_native_owner_reuses_only_its_current_pair_and_still_touches_global_lru():
    owner = StylesCache()
    other = StylesCache()
    inputs = (Color(17, 22, 31), Color(29, 47, 65, 0.5))
    first = owner.get_inner_outer(*inputs)
    hits = StylesCache.get_inner_outer.cache_info().hits
    assert owner.get_inner_outer(*inputs) is first
    assert StylesCache.get_inner_outer.cache_info().hits == hits + 1
    separate = other.get_inner_outer(*inputs)
    assert separate == first
    assert all(left is not right for left, right in zip(first, separate))
    changed = owner.get_inner_outer(Color(1, 2, 3), Color(4, 5, 6))
    assert changed != first
    assert owner.get_inner_outer(*inputs) is not first


def test_owner_pair_invalidates_after_global_clear_or_eviction():
    _child("""
import tldw_chatbook.Widgets
from textual._styles_cache import StylesCache
from textual.color import Color
owner, other = StylesCache(), StylesCache()
inputs = (Color(17, 22, 31), Color(29, 47, 65, 0.5))
first = owner.get_inner_outer(*inputs)
method = StylesCache.get_inner_outer
method.cache_clear()
fresh = owner.get_inner_outer(*inputs)
assert fresh == first and fresh is not first
assert method.cache_info().misses == 1
assert owner.get_inner_outer(*inputs) is fresh
assert method.cache_info().hits == 1
for i in range(1024):
    other.get_inner_outer(Color(i % 256, i // 256, 17), Color(5, 7, 9, 0.25))
assert method.cache_info().currsize == 1024
misses = method.cache_info().misses
after_eviction = owner.get_inner_outer(*inputs)
assert after_eviction == fresh and after_eviction is not fresh
assert method.cache_info().misses == misses + 1
assert method.cache_info().currsize == 1024
""")


def test_subclasses_receive_private_values_without_new_owner_attributes():
    class Strict(StylesCache):
        def __setattr__(self, name, value):
            if name.startswith("_tldw_"):
                raise AssertionError("foreign owner must not receive private memo")
            super().__setattr__(name, value)

    owner = Strict()
    inputs = (Color(17, 22, 31), Color(29, 47, 65, 0.5))
    first = owner.get_inner_outer(*inputs)
    second = owner.get_inner_outer(*inputs)
    assert first == second and first is not second
    assert all(left is not right for left, right in zip(first, second))


def test_color_cache_keeps_1024_entry_lru_promotion_and_eviction():
    _child("""
import tldw_chatbook.Widgets
from textual._styles_cache import StylesCache
from textual.color import Color
owner = StylesCache()
method = StylesCache.get_inner_outer
assert method.cache_parameters() == {"maxsize": 1024, "typed": False}
pairs = [(Color(i % 256, i // 256, 17), Color(5, 7, 9, 0.25)) for i in range(1025)]
owner.get_inner_outer(*pairs[0])
owner.get_inner_outer(*pairs[1])
for pair in pairs[2:1024]:
    owner.get_inner_outer(*pair)
assert method.cache_info().currsize == 1024
misses = method.cache_info().misses
owner.get_inner_outer(*pairs[0])
assert method.cache_info().misses == misses
owner.get_inner_outer(*pairs[1024])
owner.get_inner_outer(*pairs[0])
assert method.cache_info().misses == misses + 1
owner.get_inner_outer(*pairs[1])
assert method.cache_info().misses == misses + 2
assert method.cache_info().currsize == 1024
""")


_RENDER = """
import asyncio, json
from textual.app import App, ComposeResult
from textual.widgets import Static
class Paint(App):
    CSS = "Screen {background: #142536;} Static {color: #e7d6c5; background: #28496a 40%; border: solid #123456; height: auto;}"
    def compose(self) -> ComposeResult:
        yield Static("A界 [bold]color[/bold]", markup=True)
async def run():
    app = Paint()
    async with app.run_test(size=(35, 8)) as pilot:
        await pilot.pause()
        print(json.dumps([[(s.text, str(s.style), s.control) for s in strip]
                          for strip in app.screen._compositor.render_strips()]))
asyncio.run(run())
"""


def test_native_color_and_real_unicode_paint_match_unadapted_process():
    native = json.loads(_child(_RENDER))
    adapted = json.loads(_child("import tldw_chatbook.Widgets\n" + _RENDER))
    assert adapted == native
    assert any("A界" in segment[0] for line in native for segment in line)


def test_cross_color_system_ansi_memos_do_not_leak_between_native_owners():
    source = """
import json
from textual._styles_cache import StylesCache
from textual.color import Color
from textual.strip import Strip
from rich.console import Console
from rich.segment import Segment
output = []
for system in ("truecolor", "256", "standard"):
    owner = StylesCache()
    style = owner.get_inner_outer(Color(13, 42, 57), Color(28, 87, 169, 0.5))[0]
    output.append(Strip([Segment("paint", style.rich_style)]).render(
        Console(force_terminal=True, color_system=system)))
print(json.dumps(output))
"""
    native = json.loads(_child(source))
    adapted = json.loads(_child("import tldw_chatbook.Widgets\n" + source))
    assert adapted == native
    assert len(set(native)) == 3


def test_preexisting_native_lines_colors_and_exceptions_are_preserved():
    _child("""
from rich.segment import Segment
from textual._styles_cache import StylesCache
from textual.color import Color
from textual.strip import Strip
owner = StylesCache()
line = Strip([Segment("before install")])
owner._cache[0] = line
colors = (Color(17, 22, 31), Color(29, 47, 65, 0.5))
before = owner.get_inner_outer(*colors)
try:
    owner.get_inner_outer(None, None)
except Exception as error:
    failure = (type(error), str(error))
else:
    raise AssertionError("invalid colors must fail")
import tldw_chatbook.Widgets
assert owner._cache[0] is line
assert owner.get_inner_outer(*colors) == before
try:
    owner.get_inner_outer(None, None)
except Exception as error:
    assert (type(error), str(error)) == failure
else:
    raise AssertionError("invalid colors must still fail")
""")


@pytest.mark.parametrize(
    "mismatch", ["version", "source", "owner", "nested_owner", "capacity"]
)
def test_unsupported_color_method_is_not_modified(mismatch):
    _child(f"""
import textual
from functools import lru_cache
from textual._styles_cache import StylesCache
native = StylesCache.get_inner_outer
if {mismatch!r} == "version":
    textual.__version__ = "8.2.9"
elif {mismatch!r} == "source":
    @lru_cache(maxsize=1024)
    def foreign(cls, base_background, background):
        return "foreign"
    StylesCache.get_inner_outer = foreign
elif {mismatch!r} in {{"owner", "nested_owner"}}:
    namespace = {{}}
    body = ("    def nested():\\n        return cls._cache\\n    return nested()\\n"
            if {mismatch!r} == "nested_owner" else "    return cls._cache\\n")
    exec(compile("def owned(cls, base_background, background):\\n" + body,
                 native.__wrapped__.__code__.co_filename, "exec"), namespace)
    StylesCache.get_inner_outer = lru_cache(maxsize=1024)(namespace["owned"])
else:
    StylesCache.get_inner_outer = lru_cache(maxsize=32)(native.__wrapped__)
before = StylesCache.__dict__["get_inner_outer"]
import tldw_chatbook.Widgets
assert StylesCache.__dict__["get_inner_outer"] is before
""")
