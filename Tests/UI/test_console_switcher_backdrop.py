"""The live Console backdrop reuses tint arithmetic, never rendered frames."""

from __future__ import annotations

import pytest
from rich.color import Color as RichColor
from rich.color import ColorSystem
from rich.console import Console
from rich.segment import Segment
from rich.style import Style
from textual.app import App, ComposeResult
from textual.color import Color
from textual.renderables.background_screen import BackgroundScreen
from textual.screen import Screen, visible_screen_stack
from textual.widgets import Static

from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
    ConsoleSessionSwitcherModal,
)

pytestmark = pytest.mark.bootstrap_profile


def _backdrop(base=None, color=None):
    modal = ConsoleSessionSwitcherModal()
    modal.styles.background = Color(17, 31, 47, 0.8) if color is None else color
    stack = [base if base is not None else Screen()]
    token = visible_screen_stack.set(stack)
    try:
        result = modal.render()
        assert stack == []
        return result
    finally:
        visible_screen_stack.reset(token)


def test_one_live_render_calculates_each_repeated_tint_only_once(monkeypatch):
    """Repeated native colors must not allocate one tint Style per segment."""
    backdrop = _backdrop()
    original = Style.from_color
    calls = []

    def observed(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(Style, "from_color", observed)
    incoming = [Segment("live", Style(color="red", bgcolor="blue"))] * 40
    output = list(backdrop.process_segments(iter(incoming), backdrop.color))
    assert len(output) == 40
    assert all(segment.text == "live" for segment in output)
    assert len(calls) == 1


def test_repeated_live_renders_do_not_reopen_distribution_metadata(monkeypatch):
    """The version guard must not put filesystem work in every frame."""
    import importlib.metadata

    original = importlib.metadata.version
    reads = []

    def observed(name):
        reads.append(name)
        return original(name)

    monkeypatch.setattr(importlib.metadata, "version", observed)
    for _ in range(3):
        backdrop = _backdrop()
        assert type(backdrop) is not BackgroundScreen
    assert reads == []


def test_replaced_version_getter_disables_native_adapter(monkeypatch):
    import textual

    monkeypatch.setattr(textual, "__getattr__", lambda name: "8.2.8")
    assert type(_backdrop()) is BackgroundScreen


@pytest.mark.parametrize("alpha", [0, 0.2, 0.8, 1])
def test_native_segments_metadata_controls_and_ansi_are_preserved(alpha):
    """Reusing a full styled output would merge attributes or leak click data."""
    color = Color(17, 31, 47, alpha)
    backdrop = _backdrop()
    incoming = [
        Segment(
            "A界",
            Style(
                color="red",
                bgcolor="blue",
                bold=True,
                link="https://example.invalid",
                meta={"click": 1},
            ),
        ),
        Segment("e\u0301", Style(color="red", bgcolor="blue", italic=True)),
        Segment("default", Style(color="default", reverse=True)),
        Segment("plain", None),
        Segment("\a", Style(meta={"control": 1}), ((1,),)),
    ] * 7
    native = list(BackgroundScreen.process_segments(iter(incoming), color))
    actual = list(backdrop.process_segments(iter(incoming), color))
    assert actual == native
    for index, segment in enumerate(actual):
        if segment.control:
            assert segment is incoming[index]
        else:
            assert segment.style.meta == {}
            assert segment.style.link is None
    for system in (ColorSystem.TRUECOLOR, ColorSystem.EIGHT_BIT, ColorSystem.STANDARD):
        assert [
            s.style.render(s.text, color_system=system) if s.style else s.text
            for s in actual
        ] == [
            s.style.render(s.text, color_system=system) if s.style else s.text
            for s in native
        ]


def test_each_native_cleanup_and_style_add_lru_access_order_is_preserved(monkeypatch):
    """Skipping hit-side native LRU touches changes eviction and ANSI ownership."""
    from functools import lru_cache

    clear_native = Style.clear_meta_and_links.__wrapped__
    add_native = Style._add.__wrapped__
    backdrop = _backdrop()
    incoming = [
        Segment(str(i), Style(color=f"#{i:06x}", bgcolor="blue", meta={"click": i}))
        for i in range(1100)
    ]
    incoming += incoming[1050:] + incoming[:20] + incoming[1000:]
    observations = []
    for processor in (BackgroundScreen.process_segments, backdrop.process_segments):
        accesses = []
        clear = lru_cache(128)(clear_native)
        add = lru_cache(1024)(add_native)

        def observed_clear(style, accesses=accesses, clear=clear):
            accesses.append(("clear", style))
            return clear(style)

        def observed_add(left, right, accesses=accesses, add=add):
            accesses.append(("add", left, right))
            return add(left, right)

        with monkeypatch.context() as patch:
            patch.setattr(Style, "clear_meta_and_links", observed_clear)
            patch.setattr(Style, "_add", observed_add)
            output = list(processor(iter(incoming), backdrop.color))
        observations.append((output, accesses, clear.cache_info(), add.cache_info()))
    assert observations[0] == observations[1]
    assert observations[0][2].currsize == 128
    assert observations[0][3].currsize == 1024


def test_pair_memo_is_capped_and_does_not_survive_a_render(monkeypatch):
    """A 129th pair must not grow the memo or displace an admitted pair."""
    backdrop = _backdrop()
    incoming = [Segment(str(i), Style(color=f"#{i:06x}")) for i in range(129)]
    incoming += [incoming[0], incoming[-1], incoming[-1]]
    original = Style.from_color
    calls = []

    def observed(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(Style, "from_color", observed)
    for _ in range(2):
        output = list(backdrop.process_segments(iter(incoming), backdrop.color))
        assert len(output) == 132
    assert len(calls) == 262  # 128 admitted + three overflow calculations per pass.


def test_render_iterator_releases_memo_on_close_and_on_source_error():
    """No frame memo lives on the backdrop, modal or a global owner map."""
    backdrop = _backdrop()
    before = vars(backdrop).copy()
    segment = Segment("first", Style(color="red"))
    iterator = iter(backdrop.process_segments(iter([segment] * 5), backdrop.color))
    next(iterator)
    assert len(iterator.gi_frame.f_locals["memo"]) == 1
    iterator.close()
    assert iterator.gi_frame is None

    def failing_source():
        yield segment
        raise ValueError("source render failed")

    iterator = iter(backdrop.process_segments(failing_source(), backdrop.color))
    next(iterator)
    with pytest.raises(ValueError, match="source render failed"):
        next(iterator)
    assert iterator.gi_frame is None
    assert vars(backdrop) == before


def test_foreign_style_preserves_native_dispatch():
    class ForeignStyle(Style):
        def clear_meta_and_links(self):
            return Style(color="green", bold=True)

    backdrop = _backdrop()
    incoming = [Segment("foreign", ForeignStyle())] * 3
    assert list(backdrop.process_segments(incoming, backdrop.color)) == list(
        BackgroundScreen.process_segments(incoming, backdrop.color)
    )


def test_foreign_color_fallback_does_not_repeat_native_cleanup():
    class ForeignColor(RichColor):
        pass

    backdrop = _backdrop()
    incoming = [
        Segment("foreign color", Style(color=ForeignColor.from_rgb(1, 2, 3)))
    ] * 3
    before = Style.clear_meta_and_links.cache_info()
    output = list(backdrop.process_segments(incoming, backdrop.color))
    after = Style.clear_meta_and_links.cache_info()
    assert after.hits + after.misses - before.hits - before.misses == 3
    assert output == list(BackgroundScreen.process_segments(incoming, backdrop.color))


@pytest.mark.parametrize("failure", ["method", "read"])
def test_native_source_verification_falls_back_without_retaining_source(
    monkeypatch, failure
):
    import linecache

    from tldw_chatbook.Widgets.Console import console_session_switcher_modal as module

    before = set(linecache.cache)
    if failure == "method":
        monkeypatch.setattr(module, "_NATIVE_BACKGROUND_PROCESS", lambda *_args: None)
    else:

        def denied(*_args, **_kwargs):
            raise OSError("native source unavailable")

        monkeypatch.setattr(module, "open", denied, raising=False)
    assert module._native_backdrop_supported() is False
    assert set(linecache.cache) == before


def test_import_with_foreign_native_method_stays_usable(monkeypatch):
    """An unsupported plain function must not crash optional adapter setup."""
    import runpy

    from tldw_chatbook.Widgets.Console import console_session_switcher_modal as module

    monkeypatch.setattr(BackgroundScreen, "process_segments", lambda *_args: ())
    namespace = runpy.run_path(module.__file__)
    assert namespace["_NATIVE_BACKDROP_SUPPORTED"] is False


@pytest.mark.parametrize("change", ["version", "unsupported", "process", "render"])
def test_unknown_native_version_shape_or_method_stays_native(monkeypatch, change):
    import textual

    from tldw_chatbook.Widgets.Console import console_session_switcher_modal as module

    if change == "version":
        monkeypatch.setattr(textual, "__version__", "8.2.9")
        assert module._native_backdrop_supported() is False
    elif change == "unsupported":
        monkeypatch.setattr(module, "_NATIVE_BACKDROP_SUPPORTED", False)
    elif change == "process":
        original = BackgroundScreen.process_segments.__func__
        monkeypatch.setattr(
            BackgroundScreen,
            "process_segments",
            classmethod(lambda cls, segments, color: original(cls, segments, color)),
        )
    else:
        original = BackgroundScreen.__rich_console__
        monkeypatch.setattr(
            BackgroundScreen,
            "__rich_console__",
            lambda self, console, options: original(self, console, options),
        )
    assert type(_backdrop()) is BackgroundScreen


def test_other_modals_and_switcher_subclasses_stay_native():
    class SpecializedSwitcher(ConsoleSessionSwitcherModal):
        pass

    for screen in (Screen(), SpecializedSwitcher()):
        screen.styles.background = Color(17, 31, 47, 0.8)
        token = visible_screen_stack.set([Screen()])
        try:
            assert type(screen.render()) is BackgroundScreen
        finally:
            visible_screen_stack.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(52, 20), (120, 50)])
async def test_backdrop_reads_live_underlying_content_and_resize(size):
    """A frame snapshot would keep OLD text or stale compositor width."""

    class BackdropApp(App):
        def compose(self) -> ComposeResult:
            yield Static("OLD underlying", id="underlying")

    app = BackdropApp()
    console = Console(width=size[0], force_terminal=True, color_system="truecolor")
    async with app.run_test(size=size) as pilot:
        base = app.screen
        await pilot.pause()
        await app.push_screen(ConsoleSessionSwitcherModal())
        await pilot.pause()
        background = _backdrop(base, app.screen.styles.background)
        first = list(console.render(background))
        native = list(console.render(BackgroundScreen(base, background.color)))
        assert first == native
        assert "OLD underlying" in "".join(s.text for s in first)
        base.query_one("#underlying", Static).update("NEW underlying")
        await pilot.pause()
        await pilot.resize_terminal(size[0] + 7, size[1] + 3)
        await pilot.pause()
        console.width = size[0] + 7
        current = list(console.render(background))
        assert current == list(console.render(BackgroundScreen(base, background.color)))
        assert "NEW underlying" in "".join(s.text for s in current)
        assert "OLD underlying" not in "".join(s.text for s in current)
