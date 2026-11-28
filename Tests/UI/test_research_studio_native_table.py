from dataclasses import replace
from uuid import UUID

import pytest
from textual.app import App
from textual.widgets import Button, DataTable, Input

from Tests.Research_Workspace.test_studio_models import SERVER, SOURCE
from tldw_chatbook.Research_Workspace.studio_models import (
    ResearchOutputKind,
    WorkspaceOutputRef,
)
from tldw_chatbook.UI.Research_Workspace_Modules.studio_dialogs import (
    ResearchStudioActions,
    ResearchStudioOptions,
)
from tldw_chatbook.UI.Research_Workspace_Modules.studio_history import StudioHistoryRow

pytestmark = pytest.mark.asyncio


async def test_data_table_options_and_native_preview_have_working_controls():
    app = App()
    submissions = []
    async with app.run_test(size=(78, 22)) as pilot:
        options = ResearchStudioOptions(
            ResearchOutputKind.DATA_TABLE,
            "Server Data Tables",
            "ollama · orchid",
            1,
        )
        await app.push_screen(options, submissions.append)
        assert not options.query("#research-studio-retrieval")
        options.query_one("#research-studio-columns", Input).value = "Plant, Plot"
        options.query_one("#research-studio-max-rows", Input).value = "4"
        options.query_one("#research-studio-generate", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        [submitted] = submissions
        assert (submitted.table_columns, submitted.table_max_rows) == (
            ("Plant", "Plot"),
            4,
        )

        output = WorkspaceOutputRef(
            SERVER,
            "data_table",
            "server_data_table",
            str(UUID(int=77)),
            5,
            (replace(SOURCE, ref=SERVER),),
            submitted.fingerprint,
            "complete",
        )
        row = StudioHistoryRow(
            output,
            {
                "title": "Garden",
                "table": {"row_count": 1},
                "columns": [{"name": "Plant", "column_id": "col-1"}],
                "rows": [{"data": {"col-1": "Orchid"}}],
            },
            False,
        )
        actions = ResearchStudioActions(
            row, {"export-csv": "Download CSV"}, "", "Server Data Tables"
        )
        await app.push_screen(actions, submissions.append)
        assert str(actions.query_one(DataTable).get_cell_at((0, 0))) == "Orchid"
        actions.query_one("#research-studio-action-export-csv", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert submissions[-1] == "export-csv"
