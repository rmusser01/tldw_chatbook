"""Mounted inspection lifecycle and literal rendering, with controlled readers."""

import asyncio
from threading import Event
from typing import ClassVar

import pytest
from rich.text import Text
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Static

from Tests.UI.consolidated_css import CSS_DIR, ConsolidatedCSSApp
from tldw_chatbook.Personal_Context.settings_provenance import (
    SettingsProfileIdentity,
    SettingsProvenanceField,
    SettingsProvenanceProjection,
    SettingsProvenanceResult,
    SettingsProvenanceSubject,
)
from tldw_chatbook.Widgets.Settings_Widgets.personal_context_provenance import (
    PersonalContextProvenanceDetails,
    literal_metadata,
)

SUBJECT = SettingsProvenanceSubject(
    SettingsProfileIdentity("profile", 0), "record", "record", "v1"
)


class Reader:
    def __init__(self):
        self.calls = 0
        self.gate = None
        self.entered = Event()
        self.state = "available"

    def settings_provenance(self, subject):
        self.calls += 1
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(5)
        if self.state != "available":
            return SettingsProvenanceResult(self.state)
        return SettingsProvenanceResult(
            "available",
            SettingsProvenanceProjection(
                subject,
                (
                    SettingsProvenanceField(
                        "Recorded reason", "SOURCE_MARKER [bold]literal[/bold]"
                    ),
                ),
                "Legacy source reference — quotation not verified",
                tuple(f"source-{i}" for i in range(12)),
                ("0" * 64,),
            ),
        )


class HostScreen(Screen):
    def __init__(self, detail):
        super().__init__()
        self.detail = detail

    def compose(self):
        with VerticalScroll():
            yield self.detail

    def on_screen_suspend(self):
        self.detail.suspend()

    def on_screen_resume(self):
        self.call_after_refresh(self.detail.resume)


class Host(ConsolidatedCSSApp):
    CSS_PATH: ClassVar = [
        CSS_DIR / "tldw_cli_modular.tcss",
        CSS_DIR / "screen_agentic_settings.tcss",
    ]

    def __init__(self, reader, *, detail_class=PersonalContextProvenanceDetails):
        super().__init__()
        self.reader = reader
        self.detail = detail_class(SUBJECT, lambda: self.reader)

    def on_mount(self):
        self.push_screen(HostScreen(self.detail))


async def until(pilot, predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await pilot.pause(0.01)


def rendered(detail):
    return "\n".join(str(widget.renderable) for widget in detail.query(Static))


def test_metadata_is_bounded_literal_unicode():
    assert (
        literal_metadata("[link=https://example.test]x[/link]\x1b\u202ey")
        == "[link=https://example.test]x[/link]  y"
    )
    assert literal_metadata("東京 café") == "東京 café"
    assert len(literal_metadata("x" * 1000)) == 160


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 35), (60, 24)])
async def test_expanded_metadata_is_literal_bounded_and_keyboard_reachable(size):
    reader = Reader()
    app = Host(reader)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        assert reader.calls == 0
        app.detail.query_one("CollapsibleTitle").focus()
        await pilot.press("enter")
        await until(pilot, lambda: "SOURCE_MARKER" in rendered(app.detail))
        text = rendered(app.detail)
        assert "[bold]literal[/bold]" in text
        assert "showing 8 of 12" in text
        assert "source-8" not in text
        content = app.detail.query_one(
            ".personal-context-provenance-content", Static
        ).renderable
        assert isinstance(content, Text) and content.spans == []
        assert app.detail.region.right <= size[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalidation", ["invalidate", "collapse", "suspend", "unmount"]
)
async def test_old_reader_cannot_repopulate_invalidated_details(invalidation):
    reader = Reader()
    reader.gate = Event()
    app = Host(reader)
    async with app.run_test(size=(100, 35)) as pilot:
        app.detail.collapsed = False
        await until(pilot, reader.entered.is_set)
        if invalidation == "collapse":
            app.detail.collapsed = True
        elif invalidation == "suspend":
            await app.push_screen(Screen())
        elif invalidation == "unmount":
            await app.detail.remove()
        else:
            app.detail.invalidate()
        reader.gate.set()
        await until(pilot, lambda: not app.detail._pending)
        assert "SOURCE_MARKER" not in rendered(app.detail)


@pytest.mark.asyncio
async def test_expired_or_stalled_reads_clear_previously_rendered_metadata(monkeypatch):
    from tldw_chatbook.Widgets.Settings_Widgets import (
        personal_context_provenance as module,
    )

    now = [100.0]
    monkeypatch.setattr(module, "monotonic", lambda: now[0])
    reader = Reader()
    app = Host(reader)
    async with app.run_test(size=(100, 35)) as pilot:
        app.detail.collapsed = False
        await until(pilot, lambda: "SOURCE_MARKER" in rendered(app.detail))
        reader.gate = Event()
        reader.entered.clear()
        app.detail._tick()
        await until(pilot, reader.entered.is_set)
        calls = reader.calls
        app.detail._tick()
        assert reader.calls == calls
        now[0] += 3
        app.detail._expire()
        assert "SOURCE_MARKER" not in rendered(app.detail)
        reader.gate.set()
        await until(pilot, lambda: not app.detail._pending)
        assert "SOURCE_MARKER" not in rendered(app.detail)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["changed", "unavailable", "owner"])
async def test_renewal_discards_changed_unavailable_or_replaced_owner(change):
    reader = Reader()
    app = Host(reader)
    async with app.run_test(size=(100, 35)) as pilot:
        app.detail.collapsed = False
        await until(pilot, lambda: "SOURCE_MARKER" in rendered(app.detail))
        if change == "owner":
            app.reader = Reader()
        else:
            reader.state = change
        app.detail._tick()
        await until(pilot, lambda: "SOURCE_MARKER" not in rendered(app.detail))
        assert (
            "unavailable" in rendered(app.detail).lower()
            or "changed" in rendered(app.detail).lower()
        )


@pytest.mark.asyncio
async def test_collapsed_and_covered_sections_do_not_read_and_resume_fresh():
    reader = Reader()
    app = Host(reader)
    async with app.run_test(size=(100, 35)) as pilot:
        app.detail._tick()
        assert reader.calls == 0
        app.detail.collapsed = False
        await until(pilot, lambda: "SOURCE_MARKER" in rendered(app.detail))
        await app.push_screen(Screen())
        await pilot.pause()
        calls = reader.calls
        app.detail._tick()
        assert reader.calls == calls
        assert "SOURCE_MARKER" not in rendered(app.detail)
        await app.pop_screen()
        await until(pilot, lambda: "SOURCE_MARKER" in rendered(app.detail))
        assert reader.calls > calls


class YieldingMount:
    async def on_mount(self):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_yielding_mount_still_starts_initial_read():
    class YieldingDetail(PersonalContextProvenanceDetails, YieldingMount):
        pass

    app = Host(Reader(), detail_class=YieldingDetail)
    app.detail.collapsed = False
    async with app.run_test(size=(100, 35)) as pilot:
        await until(pilot, lambda: "SOURCE_MARKER" in rendered(app.detail))
