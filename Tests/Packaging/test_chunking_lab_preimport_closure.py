"""ADR-097: discovering the Lab route must not load its unvisited actions."""

from Tests.Packaging.test_chat_persistence_import_closure import _run_isolated_python


def test_lab_route_preimport_defers_actions_but_retains_canonical_region_events(
    tmp_path,
):
    result = _run_isolated_python(
        tmp_path,
        """
import sys
import tldw_chatbook.app
import tldw_chatbook.UI.Screens.chat_screen
from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_route

screen_class = resolve_screen_route('chunking_lab').load_screen_class()
from tldw_chatbook.UI.Screens import chunking_lab_screen as screen
assert screen_class is screen.ChunkingLabScreen
deferred = (
    'tldw_chatbook.Chunking.lab_recovery',
    'tldw_chatbook.Chunking.chunking_interop_library',
    'tldw_chatbook.RAG_Admin.chunking_lab_service',
    'tldw_chatbook.UI.Chunking_Lab_Modules.dialogs',
)
resident = [name for name in deferred if name in sys.modules]
assert not resident, 'Unvisited Lab actions resident during preimport: ' + repr(resident)

from tldw_chatbook.UI.Chunking_Lab_Modules.editor_region import EditorRegion
from tldw_chatbook.UI.Chunking_Lab_Modules.results_region import ResultsRegion
from tldw_chatbook.UI.Chunking_Lab_Modules.sample_region import SampleRegion
assert screen.EditorRegion.Edited is EditorRegion.Edited
assert screen.ResultsRegion.SelectionChanged is ResultsRegion.SelectionChanged
assert screen.ResultsRegion.RerunRequested is ResultsRegion.RerunRequested
assert screen.SampleRegion.Changed is SampleRegion.Changed
print('LAB_PREIMPORT_CLOSURE_OK')
""",
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    assert "LAB_PREIMPORT_CLOSURE_OK" in result.stdout
