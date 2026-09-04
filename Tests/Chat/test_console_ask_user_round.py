"""PRD Feature A: the question round (A5-A7, A9-A11, A14) and its marker."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_skill_script_confirm import _FakeApp, _wait_until
from Tests.console_provider_doubles import persisted_console_store
from tldw_chatbook.Agents.ask_user_questions import AskUserBusyRefusal
from tldw_chatbook.Agents.run_context import use_run_id
from tldw_chatbook.Chat.console_agent_bridge import format_question_marker
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController


def _questions():
    return [
        {
            "question": "Which DB?",
            "header": "DB",
            "multiSelect": False,
            "options": [
                {"label": "Postgres", "description": ""},
                {"label": "SQLite", "description": ""},
            ],
        },
        {
            "question": "Regions?",
            "header": "Region",
            "multiSelect": True,
            "options": [
                {"label": "eu", "description": ""},
                {"label": "us", "description": ""},
            ],
        },
    ]


# --- marker (Task 4) --------------------------------------------------------


def test_marker_lists_each_question_with_its_answer():
    result = {
        "answered": True,
        "answers": [
            {"question": "Which DB?", "selected": ["Postgres"], "other_text": None, "unanswered": False},
            {"question": "Regions?", "selected": [], "other_text": "apac only", "unanswered": False},
        ],
    }
    text = format_question_marker("agent", _questions(), result)
    assert text.splitlines() == [
        "? Questions from the agent (2):",
        "  Which DB? → Postgres",
        "  Regions? → other: apac only",
    ]


def test_marker_names_a_sub_agent_and_flattens_control_text():
    result = {
        "answered": True,
        "answers": [
            {"question": "Which DB?", "selected": ["Postgres"], "other_text": None, "unanswered": False},
            {"question": "Regions?", "selected": [], "other_text": None, "unanswered": True},
        ],
    }
    questions = _questions()
    questions[0]["question"] = "Which\nDB?\x07"
    text = format_question_marker("sub-agent", questions, result)
    assert text.splitlines()[0] == "? Questions from a sub-agent (2):"
    assert text.splitlines()[1] == "  Which DB? → Postgres"
    assert text.splitlines()[2] == "  Regions? → (unanswered)"


def test_marker_for_timeout_and_cancel_stamps_every_question():
    text = format_question_marker(
        "agent", _questions(), {"answered": False, "reason": "timeout"}
    )
    assert text.splitlines()[1:] == ["  Which DB? → (timed out)", "  Regions? → (timed out)"]
    text = format_question_marker(
        "agent", _questions(), {"answered": False, "reason": "cancelled"}
    )
    assert text.splitlines()[2] == "  Regions? → (cancelled)"


# --- the round (Task 5) -----------------------------------------------------


@pytest.fixture
def make_controller():
    made = []

    def _make():
        store = persisted_console_store()
        controller = ConsoleChatController(store=store, provider_gateway=object())
        controller.app = _FakeApp()
        controller.pending_question_payloads = []
        controller.set_pending_question = controller.pending_question_payloads.append
        made.append(controller)
        return controller

    yield _make
    for controller in made:
        controller.begin_shutdown()


def _start(controller, questions, *, session_id=None, run_id=""):
    box = {}

    def worker():
        with use_run_id(run_id):
            try:
                box["result"] = controller.request_user_questions(
                    questions, session_id=session_id
                )
            except Exception as exc:  # noqa: BLE001 - the test reads it
                box["error"] = exc

    thread = threading.Thread(target=worker)
    thread.start()
    return thread, box


def test_no_ui_returns_cancelled_immediately(make_controller):
    controller = make_controller()
    controller.set_pending_question = None
    assert controller.request_user_questions(_questions()) == {
        "answered": False,
        "reason": "cancelled",
    }


def test_answer_round_trip_and_marker(make_controller):
    controller = make_controller()
    markers = []
    controller._agent_bridge = SimpleNamespace(
        append_question_marker=lambda sid, text: markers.append((sid, text))
    )
    thread, box = _start(controller, _questions())
    _wait_until(lambda: bool(controller.pending_question_ids()))
    payload = controller.pending_question_payloads[-1]
    assert payload["questions"] == _questions() and payload["asked_by"] == "agent"
    assert payload["timeout_seconds"] == 0.0 and payload["deadline_monotonic"] is None
    answers = [
        {"question": "Which DB?", "selected": ["Postgres"], "other_text": None, "unanswered": False},
        {"question": "Regions?", "selected": ["eu"], "other_text": None, "unanswered": False},
    ]
    controller.resolve_pending_question(answers, request_id=payload["request_id"])
    thread.join(timeout=5)
    assert box["result"] == {"answered": True, "answers": answers}
    assert controller.pending_question_payloads[-1] is None, "teardown clears the card"
    assert markers and "Which DB? → Postgres" in markers[0][1]


def test_resolve_with_a_stale_or_missing_id_is_dropped(make_controller):
    controller = make_controller()
    thread, box = _start(controller, _questions())
    _wait_until(lambda: bool(controller.pending_question_ids()))
    controller.resolve_pending_question([], request_id=None)
    controller.resolve_pending_question([], request_id="not-this-round")
    assert controller.pending_question_ids(), "still armed"
    controller.begin_shutdown()
    thread.join(timeout=5)
    assert box["result"]["answered"] is False


def test_timeout_auto_continues_with_a_deadline_on_the_card(make_controller):
    controller = make_controller()
    controller.ask_user_timeout_seconds = lambda: 1.0
    thread, box = _start(controller, _questions())
    _wait_until(lambda: bool(controller.pending_question_payloads))
    assert controller.pending_question_payloads[0]["timeout_seconds"] == 1.0
    assert controller.pending_question_payloads[0]["deadline_monotonic"] is not None
    thread.join(timeout=5)
    assert box["result"] == {"answered": False, "reason": "timeout"}


def test_timeout_reads_console_config_when_no_seam(make_controller, monkeypatch):
    import tldw_chatbook.Chat.console_chat_controller as module

    monkeypatch.setattr(
        module,
        "get_cli_setting",
        lambda section, key, default=None: 7
        if (section, key) == ("console", "ask_user_timeout_seconds")
        else default,
    )
    controller = make_controller()
    assert controller._resolve_ask_user_timeout_seconds() == 7.0
    monkeypatch.setattr(module, "get_cli_setting", lambda s, k, d=None: "garbage")
    assert controller._resolve_ask_user_timeout_seconds() == 0.0


def test_second_ask_in_the_same_session_is_busy_and_the_third_is_refused(make_controller):
    controller = make_controller()
    session = controller.new_session(title="s")
    thread, box = _start(controller, _questions(), session_id=session.id, run_id="run-1")
    _wait_until(lambda: bool(controller.pending_question_ids()))
    with use_run_id("run-1"):
        first = controller.request_user_questions(_questions(), session_id=session.id)
        assert first["answered"] is False and first["reason"] == "busy"
        with pytest.raises(AskUserBusyRefusal):
            controller.request_user_questions(_questions(), session_id=session.id)
    controller.resolve_pending_question([], request_id=controller.pending_question_ids()[0])
    thread.join(timeout=5)
    assert box["result"]["answered"] is True


def test_a_parked_background_round_mounts_on_switch(make_controller):
    controller = make_controller()
    first = controller.new_session(title="first")
    second = controller.new_session(title="second")
    parked = []
    controller.park_pending_approval = parked.append
    thread, box = _start(controller, _questions(), session_id=first.id)
    _wait_until(lambda: bool(controller.pending_question_ids()))
    assert parked == [first.id]
    assert controller.pending_question_payloads[-1] is None
    controller.switch_session(first.id)
    assert controller.pending_question_payloads[-1]["session_id"] == first.id
    controller.switch_session(second.id)
    assert controller.pending_question_payloads[-1] is None
    controller.resolve_pending_question([], request_id=controller.pending_question_ids()[0])
    thread.join(timeout=5)
    assert box["result"]["answered"] is True


def test_revoking_the_run_returns_cancelled(make_controller):
    controller = make_controller()
    session = controller.new_session(title="s")
    thread, box = _start(controller, _questions(), session_id=session.id, run_id="run-9")
    _wait_until(lambda: bool(controller.pending_question_ids()))
    assert controller.revoke_approval_rounds_for_run("run-9") == 1
    thread.join(timeout=5)
    assert box["result"] == {"answered": False, "reason": "cancelled"}
    assert controller.pending_question_ids() == []


def test_wiring_registers_the_callback_only_with_a_view(make_controller):
    controller = make_controller()
    session = controller.new_session(title="s")
    wiring = controller._ask_user_wiring(session.id)
    assert set(wiring) == {"ask_user"}
    controller.set_pending_question = None
    assert controller._ask_user_wiring(session.id) == {}
    assert controller._ask_user_wiring(None) == {}
