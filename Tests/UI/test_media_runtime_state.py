import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_MEDIA_RUNTIME_STATE_PATH = (
    Path(__file__).resolve().parents[2] / "tldw_chatbook" / "UI" / "Screens" / "media_runtime_state.py"
)
_MEDIA_RUNTIME_STATE_SPEC = spec_from_file_location("test_media_runtime_state_module", _MEDIA_RUNTIME_STATE_PATH)
assert _MEDIA_RUNTIME_STATE_SPEC is not None and _MEDIA_RUNTIME_STATE_SPEC.loader is not None
_media_runtime_state_module = module_from_spec(_MEDIA_RUNTIME_STATE_SPEC)
sys.modules[_MEDIA_RUNTIME_STATE_SPEC.name] = _media_runtime_state_module
_MEDIA_RUNTIME_STATE_SPEC.loader.exec_module(_media_runtime_state_module)
MediaRuntimeState = _media_runtime_state_module.MediaRuntimeState


def test_media_runtime_state_defaults_to_local_backend():
    state = MediaRuntimeState()

    assert state.runtime_backend == "local"
    assert state.selected_record_id is None
    assert state.search_term == ""


def test_media_runtime_state_backend_reset_clears_selection_and_caches():
    state = MediaRuntimeState(
        runtime_backend="local",
        selected_record_id="local:media:7",
        browse_items=[{"id": "local:media:7"}],
        reading_progress_by_record_id={"local:media:7": {"current_page": 3}},
        ingestion_source_items_by_id={"server:ingestion_source:2": [{"id": 1}]},
    )

    state.reset_for_backend("server")

    assert state.runtime_backend == "server"
    assert state.selected_record_id is None
    assert state.browse_items == []
    assert state.reading_progress_by_record_id == {}
    assert state.ingestion_source_items_by_id == {}


def test_media_runtime_state_backend_reset_restores_safe_saved_view_defaults():
    state = MediaRuntimeState(runtime_backend="server")
    state.active_browse_subview = "read-it-later"
    state.selected_record_id = "server:reading_item:41"

    state.reset_for_backend("local")

    assert state.runtime_backend == "local"
    assert state.active_browse_subview == "all"
    assert state.selected_record_id is None
