from pathlib import Path
from types import SimpleNamespace
import asyncio, json
import pytest
from Tests.private_profile import private_profile_test

OUT = Path(__file__).resolve().parent


async def until(predicate, seconds=10):
    async with asyncio.timeout(seconds):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("held", [False, True])
@private_profile_test
async def test_followup_message_queue(request, monkeypatch, tmp_path, held):
    from textual.app import App, ComposeResult
    from textual.widgets import Button, Input
    from tldw_chatbook.UI.Research_Window import ResearchWindow
    from tldw_chatbook.Research_Interop.local_research_service import (
        LocalResearchService,
    )
    from tldw_chatbook.Research_Interop.local_research_engine import LocalResearchEngine
    from tldw_chatbook.Research_Interop.research_scope_service import (
        ResearchScopeService,
    )
    from Tests.Research.test_local_research_engine import _completed_claims_run

    service = LocalResearchService(str(tmp_path / "research.db"))
    run = await asyncio.to_thread(_completed_claims_run, service)
    scope = ResearchScopeService(local_service=service, server_service=None)
    projection = SimpleNamespace(
        local_research_service=service, research_scope_service=scope
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    seen = []

    async def answerer(self, seed, question):
        seen.append(question)
        entered.set()
        if held:
            await release.wait()
        return {"sufficient": True, "answer": "Stored evidence answer"}

    monkeypatch.setattr(LocalResearchEngine, "_default_follow_up_answer_fn", answerer)
    root = Path.cwd()

    class Host(App):
        CSS_PATH = str(root / "tldw_chatbook/css/tldw_cli_modular.tcss")
        ticks = 0

        def compose(self) -> ComposeResult:
            yield ResearchWindow(projection, id="research-window")

        def on_mount(self):
            self.set_interval(0.02, self.tick)

        def tick(self):
            self.ticks += 1

    host = Host()
    record = {"held": held, "run_id": run["id"]}
    try:
        async with host.run_test(size=(180, 65)) as pilot:
            try:
                window = host.query_one(ResearchWindow)
                await window.load_runs("local")
                window.select_run(next(x for x in window.runs if x["id"] == run["id"]))
                window.query_one(
                    "#research-followup-input", Input
                ).value = "Why is that?"
                await pilot.pause()
                ask = window.query_one("#research-followup-ask", Button)
                ask.press()
                await entered.wait()
                if not held:
                    await until(
                        lambda: "Stored evidence answer" in window.followup_answer_text
                    )
                window.event_log_entries = ["Audit event"]
                window._render_event_log()
                before = host.ticks
                clear = window.query_one("#research-clear-events", Button)
                clear.press()
                await asyncio.sleep(0.35)
                record.update(
                    events_while_pending=list(window.event_log_entries),
                    independent_app_ticks=host.ticks - before,
                    answer_while_pending=window.followup_answer_text,
                )
                release.set()
                await until(
                    lambda: "Stored evidence answer" in window.followup_answer_text
                    and not window.event_log_entries
                )
                record.update(
                    events_after_release=list(window.event_log_entries),
                    answer_after_release=window.followup_answer_text,
                    asked=seen,
                )
                assert record["independent_app_ticks"] >= 3
                assert record["events_while_pending"] == (
                    ["Audit event"] if held else []
                )
                assert seen == ["Why is that?"]
            finally:
                release.set()
    finally:
        service.close()
        (OUT / ("followup-held.json" if held else "followup-ordinary.json")).write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
