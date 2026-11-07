"""Read-only audit: Sources list refresh replaces settled create-field editing context."""
import json
from pathlib import Path
import pytest
from textual.widgets import Button, Input
from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import ConsolidatedCSSApp, BUNDLED_STYLESHEET
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import SourcesPane

class Host(ConsolidatedCSSApp):
    CSS_PATH = [BUNDLED_STYLESHEET]
    def compose(self):
        yield SourcesPane(id="sources")

@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["name", "url"])
@pytest.mark.parametrize("trigger", ["busy", "sources"])
@private_profile_test
async def test_settled_create_edit_context(request, field, trigger):
    app=Host()
    async with app.run_test(size=(180,80)) as pilot:
        pane=app.query_one(SourcesPane)
        pane.sources=[dict(id="s1",name="Feed",url="https://example.test",type="rss",active=True)]
        await pilot.pause()
        pane.query_one("#sources-new-button",Button).press()
        await pilot.pause()
        original=pane.query_one("#sources-create-"+field,Input)
        original.focus()
        await pilot.pause()
        await pilot.press(*"AlphaFeed")
        await pilot.pause()
        assert original.value=="AlphaFeed"
        assert original.has_focus
        await pilot.press("home","right","right","shift+right","shift+right")
        await pilot.pause()
        before=(original.selection.start,original.selection.end)
        assert before==(2,4),before
        if trigger=="busy":pane.busy_source_ids=frozenset({"s1"})
        else:pane.sources=[dict(id="s1",name="Feed refreshed",url="https://example.test",type="rss",active=True)]
        await pilot.pause()
        current=pane.query_one("#sources-create-"+field,Input)
        after=(current.selection.start,current.selection.end)
        value_before_key=current.value
        focused=current.has_focus
        await pilot.press("X")
        await pilot.pause()
        receipt=dict(field=field,trigger=trigger,before_selection=before,after_selection=after,value_before_key=value_before_key,after_key=current.value,same_widget=current is original,has_focus=focused,expected_after_key="AlXaFeed")
        Path(__file__).with_name("observation-"+trigger+"-"+field+".json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
        assert value_before_key=="AlphaFeed" and focused
        if trigger=="busy":
            assert current is original and after==before
            assert current.value=="AlXaFeed"
        else:
            assert current is not original and after!=before
            assert current.value!="AlXaFeed"
        assert app._exception is None
