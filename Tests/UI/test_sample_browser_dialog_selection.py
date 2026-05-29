from textual.app import App
from textual.widgets import Static

import pytest

from tldw_chatbook.Widgets.Evals.sample_browser_dialog import SampleBrowserDialog


class _ScreenHost(App):
    def __init__(self, screen):
        super().__init__()
        self.screen_under_test = screen

    async def on_mount(self) -> None:
        await self.push_screen(self.screen_under_test)


class _PlainTextArea(Static):
    def __init__(self, text: str = "", *args, id: str | None = None, **kwargs):
        super().__init__(text, id=id)
        self.text = text


def _install_sample_browser_test_doubles(monkeypatch) -> None:
    monkeypatch.setattr(SampleBrowserDialog, "CSS", "", raising=False)
    monkeypatch.setattr(SampleBrowserDialog, "load_samples", lambda self: None)
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.Evals.sample_browser_dialog.TextArea",
        _PlainTextArea,
    )


def _seed_samples(screen: SampleBrowserDialog) -> None:
    screen.samples = [
        {
            "index": 0,
            "id": "sample_0",
            "input_text": "first prompt",
            "expected_output": "first answer",
            "metadata": {},
        },
        {
            "index": 1,
            "id": "sample_1",
            "input_text": "second prompt",
            "expected_output": "second answer",
            "metadata": {},
        },
    ]
    screen.filtered_samples = screen.samples.copy()
    screen.samples_per_page = 20


@pytest.mark.asyncio
async def test_sample_browser_marks_selected_rows_when_rendered(monkeypatch):
    _install_sample_browser_test_doubles(monkeypatch)
    app = _ScreenHost(SampleBrowserDialog(dataset_path="/tmp/empty-dataset.json"))

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen_under_test
        _seed_samples(screen)
        screen.selected_indices = {1}
        screen.update_display()
        await pilot.pause()

        assert not screen.query_one("#sample-0").has_class("selected")
        assert screen.query_one("#sample-1").has_class("selected")


@pytest.mark.asyncio
async def test_sample_browser_click_toggles_visible_row_selection(monkeypatch):
    _install_sample_browser_test_doubles(monkeypatch)
    app = _ScreenHost(SampleBrowserDialog(dataset_path="/tmp/empty-dataset.json"))

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen_under_test
        _seed_samples(screen)
        screen.update_display()
        await pilot.pause()

        await pilot.click("#sample-1")
        await pilot.pause()

        assert screen.selected_indices == {1}
        assert screen.query_one("#sample-1").has_class("selected")

        await pilot.click("#sample-1")
        await pilot.pause()

        assert screen.selected_indices == set()
        assert not screen.query_one("#sample-1").has_class("selected")


@pytest.mark.asyncio
async def test_sample_browser_rebuilds_visible_rows_without_duplicate_ids(monkeypatch):
    _install_sample_browser_test_doubles(monkeypatch)
    app = _ScreenHost(SampleBrowserDialog(dataset_path="/tmp/empty-dataset.json"))

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen_under_test
        _seed_samples(screen)

        screen.update_display()
        await pilot.pause()
        screen.update_display()
        await pilot.pause()

        rows = list(screen.query(".sample-row"))
        assert len(rows) == 2
        assert {row.id for row in rows} == {"sample-0", "sample-1"}


@pytest.mark.asyncio
async def test_sample_browser_bulk_selection_buttons_update_visible_selection(monkeypatch):
    _install_sample_browser_test_doubles(monkeypatch)
    app = _ScreenHost(SampleBrowserDialog(dataset_path="/tmp/empty-dataset.json"))

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen_under_test
        _seed_samples(screen)
        screen.update_display()
        await pilot.pause()

        screen.handle_select_all()
        await pilot.pause()
        assert screen.selected_indices == {0, 1}
        assert screen.query_one("#sample-0").has_class("selected")
        assert screen.query_one("#sample-1").has_class("selected")

        screen.handle_clear_selection()
        await pilot.pause()
        assert screen.selected_indices == set()
        assert not screen.query_one("#sample-0").has_class("selected")
        assert not screen.query_one("#sample-1").has_class("selected")
