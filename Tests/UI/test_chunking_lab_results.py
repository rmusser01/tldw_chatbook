"""Real mounted results navigation; captured text is always literal."""

import json
import os
import threading
from pathlib import Path

import pytest
from textual.app import ComposeResult
from textual.widgets import DataTable, Select, TextArea

from Tests.Chunking.test_lab_comparison import make_result, with_chunks
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp


class ResultsApp(ConsolidatedCSSApp):
    CSS_PATH = str(BUNDLED_STYLESHEET)

    def __init__(self):
        super().__init__()
        self.selections = []
        self.reruns = 0

    def compose(self) -> ComposeResult:
        from tldw_chatbook.UI.Chunking_Lab_Modules.results_region import ResultsRegion

        yield ResultsRegion(id="results")

    def on_results_region_selection_changed(self, event):
        self.selections.append(event)

    def on_results_region_rerun_requested(self, event):
        self.reruns += 1


async def settle(app, pilot):
    await app.workers.wait_for_complete()
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()


@pytest.mark.asyncio
async def test_restored_historical_result_prepares_counts_text_and_raw_config():
    from Tests.Chunking.test_lab_recovery import historical_session
    from tldw_chatbook.Chunking.lab_models import RunResult
    from tldw_chatbook.Chunking.lab_recovery import export_recovery, parse_recovery
    from tldw_chatbook.UI.Chunking_Lab_Modules.results_region import ResultsRegion

    restored = parse_recovery(export_recovery(historical_session()))
    result = RunResult.model_validate(next(iter(restored.results.values())))
    app = ResultsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        region = app.query_one(ResultsRegion)
        region.configure_view(restored.view)
        region.show_results(None, result, stale_ids=frozenset())
        await settle(app, pilot)
        assert region.query_one(TextArea).text == result.report.chunks[0]["text"]
        summary = region._prepared["summaries"]["B"]
        assert summary["chunk_count"] == len(result.report.chunks)
        assert summary["budget"]["unit"] is None
        assert summary["budget"]["limit"] is None
        region.query_one("#detail-kind", Select).value = "effective"
        await settle(app, pilot)
        assert '"legacy_operation"' in region.query_one(TextArea).text


@pytest.mark.parametrize(
    "saved", [[], {"selections": []}, {"active_view": [], "detail": {}}]
)
def test_optional_view_restoration_falls_back_without_crashing(saved):
    from tldw_chatbook.UI.Chunking_Lab_Modules.results_region import ResultsRegion

    region = ResultsRegion()
    region.configure_view({"results": saved})
    assert region._view["active_view"] == "B"
    assert region._view["selections"] == {}


@pytest.mark.asyncio
async def test_unknown_view_extension_survives_result_inspection():
    from tldw_chatbook.UI.Chunking_Lab_Modules.results_region import ResultsRegion

    app = ResultsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        region = app.query_one(ResultsRegion)
        region.configure_view({"results": {"future": {"opaque": [1, 2]}}})
        region.show_results(None, make_result(), stale_ids=frozenset())
        await settle(app, pilot)
        assert await pilot.click("#view-compare")
        await settle(app, pilot)
        assert app.selections[-1].view.get("future") == {"opaque": [1, 2]}


@pytest.mark.asyncio
async def test_exclusive_workers_are_lazy_when_canceled_before_start(monkeypatch):
    import inspect

    from tldw_chatbook.UI.Chunking_Lab_Modules.results_region import ResultsRegion

    app = ResultsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        region = app.query_one(ResultsRegion)
        pending = []

        def defer(work, **kwargs):
            pending.append(work)

        monkeypatch.setattr(region, "run_worker", defer)
        try:
            region.show_results(None, make_result(), stale_ids=frozenset())
            region._inspect()
            assert len(pending) >= 2
            # A canceled-before-start worker never needs to close an eagerly
            # created coroutine; no body exists until Textual actually starts it.
            assert all(inspect.iscoroutinefunction(work) for work in pending)
            await region.remove()
            # A lazy worker may be started after its region was pruned; it must
            # abandon preparation/publication without querying removed children.
            for work in tuple(pending):
                await work()
        finally:
            for work in pending:
                if inspect.iscoroutine(work):
                    work.close()


@pytest.mark.asyncio
async def test_ten_thousand_chunks_last_page_keyboard_selection_and_restore():
    from tldw_chatbook.UI.Chunking_Lab_Modules.results_region import ResultsRegion

    result = with_chunks(
        make_result(), [f"chunk {i} [bold]literal[/bold]" for i in range(10000)]
    )
    app = ResultsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        region = app.query_one(ResultsRegion)
        region.show_results(None, result, stale_ids=frozenset())
        await settle(app, pilot)
        table = region.query_one("#chunks-b", DataTable)
        assert table.row_count == 100
        assert await pilot.click("#last-b")
        await settle(app, pilot)
        table.focus()
        await pilot.press("ctrl+end", "enter")
        await settle(app, pilot)
        assert app.selections[-1].chunk_index == 9999
        assert "chunk 9999 [bold]literal[/bold]" == region.query_one(TextArea).text
        assert sum(t.row_count for t in region.query(DataTable)) == 100
        saved = app.selections[-1].view
        region.configure_view({"results": saved})
        region.show_results(None, result, stale_ids=frozenset({result.request.run_id}))
        await settle(app, pilot)
        assert table.row_count == 100 and table.cursor_row == 99
        assert "Newer draft" in str(region.query_one("#status-b").content)


@pytest.mark.asyncio
async def test_restore_independent_selections_without_spurious_edit_and_previous_badge():
    from tldw_chatbook.UI.Chunking_Lab_Modules.results_region import ResultsRegion

    a, b = (
        with_chunks(make_result(), ["a0", "a1"]),
        with_chunks(make_result(), ["b0", "b1", "b2"]),
    )
    app = ResultsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        region = app.query_one(ResultsRegion)
        region.configure_view(
            {
                "sample_hash": "untouched",
                "results": {
                    "active_view": "A",
                    "selections": {
                        a.request.candidate_id: 1,
                        b.request.candidate_id: 2,
                    },
                    "detail": "chunk",
                },
            },
            previous_ids=frozenset({a.request.run_id}),
        )
        region.show_results(a, b, stale_ids=frozenset())
        await settle(app, pilot)
        assert region.query_one(TextArea).text == "a1"
        assert "Previous" in str(region.query_one("#status-a").content)
        assert app.selections == []
        assert await pilot.click("#view-b")
        await settle(app, pilot)
        assert region.query_one(TextArea).text == "b2"


@pytest.mark.asyncio
async def test_mismatch_rerun_and_missing_member_never_reuses_previous_success():
    from tldw_chatbook.UI.Chunking_Lab_Modules.results_region import ResultsRegion

    app = ResultsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        region = app.query_one(ResultsRegion)
        a, b = make_result(), make_result("different")
        region.show_results(a, b, stale_ids=frozenset())
        await settle(app, pilot)
        assert await pilot.click("#view-compare")
        await settle(app, pilot)
        assert "Sample content differs" in str(
            region.query_one("#comparison-status").content
        )
        assert await pilot.click("#rerun-both")
        assert app.reruns == 1
        region.show_results(a, None, stale_ids=frozenset())
        await settle(app, pilot)
        assert "two successful" in str(region.query_one("#comparison-status").content)
        assert region.query_one("#chunks-b", DataTable).row_count == 0


@pytest.mark.asyncio
async def test_large_literal_inspector_is_paged_without_losing_tail():
    from tldw_chatbook.UI.Chunking_Lab_Modules.results_region import ResultsRegion

    text = "[red]" + "z" * 25000 + "[/red]TAIL"
    result = with_chunks(make_result(), [text])
    app = ResultsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        region = app.query_one(ResultsRegion)
        region.show_results(None, result, stale_ids=frozenset())
        await settle(app, pilot)
        assert region.query_one(TextArea).text.startswith("[red]")
        assert len(region.query_one(TextArea).text) <= 8192
        for _ in range(3):
            assert await pilot.click("#text-next")
            await pilot.pause(
                0.25
            )  # Native Button ignores clicks during its active effect.
        assert region.query_one(TextArea).text.endswith("[/red]TAIL")


@pytest.mark.asyncio
async def test_whole_result_stats_off_loop_and_reused_on_selection(monkeypatch):
    from tldw_chatbook.UI.Chunking_Lab_Modules import results_region

    calls = []
    original = results_region.summarize_result

    def measured(result):
        calls.append(threading.get_ident())
        return original(result)

    monkeypatch.setattr(results_region, "summarize_result", measured)
    result = with_chunks(make_result(), ["a", "b"])
    app = ResultsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        region = app.query_one(results_region.ResultsRegion)
        region.show_results(None, result, stale_ids=frozenset())
        await settle(app, pilot)
        region.query_one("#chunks-b", DataTable).focus()
        await pilot.press("down", "enter")
        await settle(app, pilot)
        region.show_results(None, result, stale_ids=frozenset({result.request.run_id}))
        await settle(app, pilot)
        assert len(calls) == 1
        assert calls[0] != threading.get_ident()


@pytest.mark.asyncio
async def test_verified_source_selection_and_linked_rows():
    from tldw_chatbook.UI.Chunking_Lab_Modules.results_region import ResultsRegion

    a = with_chunks(
        make_result("same same"),
        ["same"],
        [{"start": 5, "end": 9, "coordinate_space": "source"}],
    )
    b = with_chunks(
        make_result("same same"),
        ["same", "same"],
        [
            {"start": 0, "end": 4, "coordinate_space": "source"},
            {"start": 5, "end": 9, "coordinate_space": "source"},
        ],
    )
    app = ResultsApp()
    async with app.run_test(size=(160, 50)) as pilot:
        region = app.query_one(ResultsRegion)
        region.configure_view(
            {
                "results": {
                    "active_view": "Compare",
                    "inspected_candidate": a.request.candidate_id,
                    "detail": "source",
                }
            }
        )
        region.show_results(a, b, stale_ids=frozenset())
        await settle(app, pilot)
        assert region.query_one(TextArea).selected_text == "same"
        assert region.query_one(TextArea).selection.start == (0, 5)
        table = region.query_one("#chunks-b", DataTable)
        assert str(table.get_row_at(0)[0]) == ""
        assert str(table.get_row_at(1)[0]) == "*"


@pytest.mark.asyncio
async def test_authored_diff_shows_both_runtime_assets_and_complete_literal_values():
    from tldw_chatbook.UI.Chunking_Lab_Modules.results_region import ResultsRegion

    a, b = make_result(), make_result()
    recipe = b.request.recipe.model_copy(
        update={
            "authored_json": json.dumps(
                {
                    "chunking": {"method": "words"},
                    "metadata": {"note": "[red]literal[/red]"},
                }
            ),
            "runtime": b.request.recipe.runtime.model_copy(
                update={
                    "assets": (
                        {
                            "kind": "tokenizer",
                            "name": "local-measure-2",
                            "version": "2",
                            "content_digest": "digest",
                        },
                    )
                }
            ),
        }
    )
    b = b.model_copy(
        update={"request": b.request.model_copy(update={"recipe": recipe})}
    )
    app = ResultsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        region = app.query_one(ResultsRegion)
        region.configure_view(
            {"results": {"active_view": "Compare", "detail": "authored"}}
        )
        region.show_results(a, b, stale_ids=frozenset())
        await settle(app, pilot)
        text = region.query_one(TextArea).text
        assert "local-measure-2" in text and "digest" in text
        assert "[red]literal[/red]" in text


@pytest.mark.asyncio
async def test_zero_output_and_failed_output_explain_their_different_states():
    from tldw_chatbook.UI.Chunking_Lab_Modules.results_region import ResultsRegion

    result = with_chunks(make_result(), [])
    app = ResultsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        region = app.query_one(ResultsRegion)
        region.show_results(None, result, stale_ids=frozenset())
        await settle(app, pilot)
        assert "successful recipe emitted zero" in region.query_one(TextArea).text
        region.query_one("#detail-kind", Select).value = "execution"
        await settle(app, pilot)
        assert '"backend": "local"' in region.query_one(TextArea).text
        region.query_one("#detail-kind", Select).value = "chunk"
        await settle(app, pilot)
        failed = result.model_copy(
            update={
                "status": "limited",
                "report": None,
                "error": {"message": "[limit] exceeded"},
            }
        )
        region.show_results(None, failed, stale_ids=frozenset())
        await settle(app, pilot)
        assert "[limit] exceeded" in region.query_one(TextArea).text
        assert "limited" in str(region.query_one("#mapping-status").content)
        region.query_one("#detail-kind", Select).value = "effective"
        await settle(app, pilot)
        assert '"method":"words"' in region.query_one(TextArea).text


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 40), (160, 50)])
async def test_results_viewports_and_real_keyboard_focus(size):
    from tldw_chatbook.UI.Chunking_Lab_Modules.results_region import ResultsRegion

    text = "\n\n".join(
        f"Row {index:02}: Synthetic inspection fixture. [brackets] remain literal."
        for index in range(20)
    )
    a = make_result(
        text,
        {
            "chunking": {
                "method": "fixed_size",
                "config": {"max_size": 64, "overlap": 8},
            }
        },
    )
    b = make_result(
        text,
        {"chunking": {"method": "words", "config": {"max_size": 12, "overlap": 2}}},
    )
    app = ResultsApp()
    async with app.run_test(size=size) as pilot:
        region = app.query_one(ResultsRegion)
        region.configure_view(
            {"results": {"active_view": "Compare"}},
            previous_ids=frozenset({a.request.run_id}),
        )
        region.show_results(a, b, stale_ids=frozenset({b.request.run_id}))
        await settle(app, pilot)
        table = region.query_one("#chunks-b", DataTable)
        table.focus()
        await pilot.press("down")
        await settle(app, pilot)
        assert app.selections[-1].chunk_index == 1
        assert app.focused is table
        assert table.size.height >= 4  # Header plus multiple readable chunks at 80x24.
        area = region.query_one(TextArea)
        assert area.region.bottom <= size[1]
        assert area.region.width >= 70
        for widget in region.query("Button, Select"):
            if widget.visible and widget.display and widget.region.width:
                assert widget.region.right <= size[0]
                assert widget.region.bottom <= size[1]
        if size[0] >= 120:
            assert region.query_one("#column-a").display
            assert (
                region.query_one("#column-a").region.width
                == region.query_one("#column-b").region.width
            )
        else:
            assert not region.query_one("#column-a").display
        if size[0] == 120:
            region.query_one("#detail-kind", Select).value = "effective"
            await settle(app, pilot)
            assert "/chunking/method" in area.text
        elif size[0] == 160:
            b = make_result(
                text,
                {
                    "chunking": {
                        "method": "fixed_size",
                        "config": {"max_size": 96, "overlap": 16},
                    }
                },
            )
            region.configure_view(
                {
                    "results": {
                        "active_view": "Compare",
                        "inspected_candidate": a.request.candidate_id,
                        "detail": "source",
                    }
                }
            )
            region.show_results(a, b, stale_ids=frozenset())
            await settle(app, pilot)
            region.query_one("#chunks-a", DataTable).focus()
            assert area.selected_text == text[:64]
            assert str(region.query_one("#chunks-b", DataTable).get_row_at(0)[0]) == "*"
        artifact_dir = os.environ.get("TLDW_LAB_CAPTURE_DIR")
        if artifact_dir:
            destination = Path(artifact_dir)
            destination.mkdir(parents=True, exist_ok=True)
            app.save_screenshot(
                f"results-{size[0]}x{size[1]}.svg", path=str(destination)
            )
