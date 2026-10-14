"""The llama.cpp setup 2Hz refresh skips its DOM walk when state is unchanged."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.Event_Handlers.LLM_Management_Events import server_lifecycle
from tldw_chatbook.UI.LLM_Management.llamacpp_setup_view import LlamaCppSetupView

pytestmark = pytest.mark.bootstrap_profile


class _View(LlamaCppSetupView):
    """Bare view shell: refresh collaborators are stubbed per test."""

    is_mounted = True

    def __init__(self) -> None:  # skip the heavy real view init
        pass


def _make_view():
    snapshot = SimpleNamespace(
        code="screen_detached", state="idle", target=None, model_ids=()
    )
    view = _View()
    view.app_instance = SimpleNamespace()
    view._profile_busy = False
    view._profiles_loaded = True
    view._local_claim = None
    view._last_models = ()
    view._last_diagnostics = None
    view._last_refresh_gate = None
    view._current_preview_text = None
    view._active = lambda: True
    view.owner = SimpleNamespace(snapshot=lambda: snapshot)
    renders = []
    view._render_launch_state = lambda: renders.append(None)
    diags = []
    view._render_diagnostics = lambda: diags.append(None)
    widgets = {}

    def query_one(selector, cls=None):
        return widgets.setdefault(
            selector, SimpleNamespace(disabled=False, display=True, update=lambda *a: None)
        )

    view.query_one = query_one
    return view, snapshot, renders, diags


@pytest.fixture(autouse=True)
def no_claims(monkeypatch):
    monkeypatch.setattr(
        server_lifecycle, "current_server_claim", lambda app, provider: None
    )


def test_refresh_skips_dom_walk_when_state_unchanged():
    view, _snapshot, renders, diags = _make_view()

    view.refresh_state()
    view.refresh_state()
    view.refresh_state()
    assert len(renders) == 1, "unchanged inputs must skip the repaint walk"
    assert len(diags) == 3, "diagnostics keeps its own per-tick identity gate"


def test_refresh_repaints_when_the_connection_state_changes():
    view, snapshot, renders, _diags = _make_view()

    view.refresh_state()
    snapshot.state = "checking"
    view.refresh_state()
    assert len(renders) == 2, "a connection-state change must repaint"

    snapshot.code = "ok"
    view.refresh_state()
    assert len(renders) == 3, "a status-code change must repaint"


def test_refresh_repaints_when_a_profile_load_lands():
    view, _snapshot, renders, _diags = _make_view()

    view.refresh_state()
    view._profiles_loaded = False
    view.refresh_state()
    assert len(renders) == 2, "a profile-load change must repaint"
