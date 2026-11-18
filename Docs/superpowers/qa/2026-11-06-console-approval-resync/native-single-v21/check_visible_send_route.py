"""Finite actual fixture/stock-owner source controls; no App/product import or GUI."""

import argparse
import ast
import asyncio
import builtins
import copy
import hashlib
import json
import threading
from pathlib import Path
from types import MethodType, SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument("--source-root", required=True, type=Path)
parser.add_argument("--output-root", required=True, type=Path)
args = parser.parse_args()
source = args.source_root.resolve()
output = args.output_root.resolve()
if not output.is_relative_to(source / ".superpowers"):
    raise RuntimeError("Pure outputs must stay beneath selectedsource/.superpowers")
output.mkdir(parents=True, exist_ok=True)
here = Path(__file__).resolve().parent
workload_path = here / "approval_fixture_workload_v7.py"
text = workload_path.read_text(encoding="utf-8")
tree = ast.parse(text)
owner = next(
    node
    for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == "ApprovalFixtureDriver"
)
methods = {
    node.name: node
    for node in owner.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
}
stage = methods["stage"]
marker = next(
    index
    for index, node in enumerate(stage.body)
    if isinstance(node, ast.Expr)
    and isinstance(node.value, ast.Call)
    and isinstance(node.value.func, ast.Attribute)
    and node.value.func.attr == "_record"
    and node.value.args
    and isinstance(node.value.args[0], ast.Constant)
    and node.value.args[0].value == "fixture_prepared"
)
actual_tail = copy.deepcopy(stage)
actual_tail.name = "actual_send_boundary"
actual_tail.body = copy.deepcopy(stage.body[marker + 1 :])
results = {}
details = {}


def compile_actual(node, receive):
    def controlled_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tldw_chatbook.UI.Console_Modules.wiring":
            return SimpleNamespace(receive_console_visible_intent=receive)
        if name.startswith("tldw_chatbook"):
            raise AssertionError("No product import permitted in pure controls")
        return builtins.__import__(name, globals, locals, fromlist, level)

    namespace = {
        "asyncio": asyncio,
        "__builtins__": {**vars(builtins), "__import__": controlled_import},
    }
    module = ast.fix_missing_locations(
        ast.Module(body=[copy.deepcopy(node)], type_ignores=[])
    )
    exec(compile(module, "actual-fixture-source", "exec"), namespace)
    return namespace[node.name]


async def case(mode="valid"):
    calls = []
    event = asyncio.Event()
    actual_result = object()

    async def owned_work():
        await event.wait()
        return actual_result

    owned_task = asyncio.create_task(owned_work())
    store = SimpleNamespace(active_session_id="session")
    controller = SimpleNamespace(store=store)
    selected = SimpleNamespace(
        provider="openai",
        explicit_model="synthetic",
        configured_model=None,
        base_url="http://127.0.0.1:1",
    )
    intent = SimpleNamespace(
        turn_id="turn",
        session_id="session",
        selection=SimpleNamespace(provider_selection=selected),
    )
    record = SimpleNamespace(
        turn_id="turn",
        session_id="session",
        store=store,
        received_intent=intent,
        task=owned_task,
    )
    runtime = SimpleNamespace(
        _chat_store=store, _chat_controller=controller, _turn_custody={"turn": record}
    )
    original_error = RuntimeError("controlled original error")
    stash = object()
    composer = SimpleNamespace(
        load_draft=lambda draft: calls.append(("load", draft)),
        capture_draft_for_send=lambda: stash,
    )
    screen = SimpleNamespace(
        _console_runtime=lambda: runtime,
        _console_composer_or_none=lambda: composer,
        _console_runtime_attachment_generation=7,
        _console_visible_draft_session_id="session",
        _console_transcript_sync_timer=None,
        _console_transcript_poll_asked=False,
    )
    fixture = SimpleNamespace(
        app=SimpleNamespace(screen=screen),
        controller=controller,
        session=SimpleNamespace(id="session"),
        worker=None,
        _visible_send_ownership=None,
        provider=SimpleNamespace(
            provider_key="openai", model="synthetic", base_url="http://127.0.0.1:1"
        ),
        current={"correlation": "a" * 32},
        _lock=threading.Lock(),
        _observation_failed=lambda stage, error: calls.append(
            ("observation_failed", stage, type(error).__name__)
        ),
        snapshot=lambda: actual_result,
        _record=lambda *values, **keywords: calls.append(("record", values, keywords)),
    )

    async def direct_submit(*values, **keywords):
        calls.append(("direct_submit", values, keywords))
        return actual_result

    controller.submit_draft = direct_submit

    def receive(actual_screen, draft, session_id, actual_stash):
        calls.append(
            (
                "receive",
                actual_screen is screen,
                draft,
                session_id,
                actual_stash is stash,
            )
        )
        if mode == "receive_error":
            raise original_error
        if mode in ("none", "empty", "false"):
            return {"none": None, "empty": "", "false": False}[mode]
        return "turn"

    if mode == "load_error":

        def load_error(draft):
            raise original_error

        composer.load_draft = load_error
    if mode == "capture_error":

        def capture_error():
            raise original_error

        composer.capture_draft_for_send = capture_error
    if mode == "missing_record":
        runtime._turn_custody.clear()
    elif mode == "wrong_turn":
        record.turn_id = "other"
    elif mode == "wrong_session":
        record.session_id = "other"
    elif mode == "wrong_store":
        record.store = object()
    elif mode == "wrong_intent_turn":
        intent.turn_id = "other"
    elif mode == "wrong_intent_session":
        intent.session_id = "other"
    elif mode == "missing_task":
        record.task = None
    elif mode == "wrong_controller":
        runtime._chat_controller = object()
    elif mode == "wrong_visible_session":
        screen._console_visible_draft_session_id = "other"
    if mode == "selection_observer_error":
        intent.selection = object()
    if "_record_received_selection" in methods:
        fixture._record_received_selection = MethodType(
            compile_actual(methods["_record_received_selection"], receive), fixture
        )
    if "_submit_visible_draft" in methods:
        fixture._submit_visible_draft = MethodType(
            compile_actual(methods["_submit_visible_draft"], receive), fixture
        )
    boundary = compile_actual(actual_tail, receive)
    caught = None
    answer = None
    try:
        answer = await boundary(fixture, "single", "a" * 32)
    except BaseException as error:
        caught = error
    await asyncio.sleep(0)
    try:
        if mode == "selection_observer_error":
            return (
                caught is None
                and fixture.worker is owned_task
                and any(
                    row[0] == "observation_failed"
                    and row[1] == "received_intent_selection"
                    for row in calls
                )
            )
        if mode == "valid":
            details["original_boundary"] = {
                "receive_calls": sum(row[0] == "receive" for row in calls),
                "direct_submit_calls": sum(row[0] == "direct_submit" for row in calls),
                "captured_exact_custody_task": fixture.worker is owned_task,
            }
            okay = (
                caught is None
                and answer is actual_result
                and fixture.worker is owned_task
                and details["original_boundary"]["receive_calls"] == 1
                and details["original_boundary"]["direct_submit_calls"] == 0
            )
            if okay:
                results["admitted_selection_proof_observed_without_legacy_selector"] = (
                    fixture.current["actual_received_selection_boundary"]
                    == "actual_received_intent_selection"
                    and all(
                        fixture.current["actual_received_selection_equality"].values()
                    )
                )
                event.set()
                result = await fixture.worker
                runtime._turn_custody.pop("turn")
                # Mirror the ORIGINAL runtime release clearing lifetime handles.
                record.task = None
                record.store = None
                record.received_intent = None
                observe_lifecycle = MethodType(
                    compile_actual(methods["_visible_send_lifecycle"], receive), fixture
                )
                try:
                    lifecycle = observe_lifecycle()
                except Exception as error:
                    details["retirement_observer_error"] = type(error).__name__
                    results["real_release_clears_handles_without_observer_failure"] = (
                        False
                    )
                    return False
                results["real_release_clears_handles_without_observer_failure"] = True

                class ResidentTimer:
                    _task = owned_task

                    @property
                    def _active(self):
                        raise AssertionError(
                            "Observer must not initialize cached timer state"
                        )

                timer = ResidentTimer()
                screen._console_transcript_sync_timer = timer
                timer_before = observe_lifecycle()
                timer.__dict__["_active"] = event
                timer_after = observe_lifecycle()
                results["timer_observation_does_not_force_lazy_active_event"] = (
                    timer_before["timer_present"]
                    and timer_before["timer_task_done"]
                    and timer_before["timer_active_resident"] is False
                    and timer_before["timer_active_set"] is None
                    and timer_after["timer_active_resident"] is True
                    and timer_after["timer_active_set"] is True
                )
                fixture.app.screen = object()
                screen._console_runtime_attachment_generation += 1
                changed = observe_lifecycle()
                results["passive_screen_attachment_change_reported_without_reset"] = (
                    changed["same_screen"] is False
                    and changed["attachment_matches"] is False
                )
                results["captured_task_survives_controlled_record_retirement"] = (
                    result is actual_result
                    and lifecycle["custody_released"]
                    and lifecycle["task_done"]
                    and lifecycle["same_screen"]
                    and lifecycle["attachment_matches"]
                )
            return okay
        if mode.endswith("_error"):
            return caught is original_error and not any(
                row[0] == "direct_submit" for row in calls
            )
        return (
            isinstance(caught, RuntimeError)
            and fixture.worker is None
            and not any(row[0] == "direct_submit" for row in calls)
        )
    finally:
        if fixture.worker is not None and fixture.worker is not owned_task:
            fixture.worker.cancel()
            await asyncio.gather(fixture.worker, return_exceptions=True)
        owned_task.cancel()
        await asyncio.gather(owned_task, return_exceptions=True)


async def main():
    results["actual_boundary_calls_receive_once_and_captures_exact_task"] = await case()
    if "_submit_visible_draft" not in methods:
        return
    for mode in (
        "none",
        "empty",
        "false",
        "missing_record",
        "wrong_turn",
        "wrong_session",
        "wrong_store",
        "wrong_intent_turn",
        "wrong_intent_session",
        "missing_task",
        "wrong_controller",
        "wrong_visible_session",
        "receive_error",
        "load_error",
        "capture_error",
        "selection_observer_error",
    ):
        results[mode] = await case(mode)
    route = methods["_submit_visible_draft"]
    results["no_await_or_task_substitution_in_receive_boundary"] = not any(
        isinstance(node, (ast.Await, ast.AsyncFor, ast.AsyncWith))
        for node in ast.walk(route)
    ) and not any(
        isinstance(node, ast.Attribute)
        and node.attr in {"submit_draft", "create_task", "wait_for_turn"}
        for node in ast.walk(route)
    )
    stage_code = ast.unparse(stage)
    results["stock_settings_and_binding_owners_precede_send"] = all(
        term in stage_code
        for term in (
            "._capture_console_draft_switch_snapshot()",
            "._new_session_startup_settings(self.workspace_id)",
            "canonical_settings_baseline=settings",
            "._console_new_chat_default_generation()",
            ".switch_and_bind(self.session.id)",
        )
    ) and stage_code.index(".switch_and_bind(") < stage_code.index(
        "._submit_visible_draft("
    )
    setup_start = next(
        index
        for index, node in enumerate(stage.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "screen"
            for target in node.targets
        )
    )
    setup_end = next(
        index
        for index, node in enumerate(stage.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "switch_and_bind"
    )
    settings = object()
    session = SimpleNamespace(id="new-session")
    setup_calls = []

    def create_session(**keywords):
        setup_calls.append(("create", keywords))
        return session

    session_owner = SimpleNamespace(
        _capture_console_draft_switch_snapshot=lambda: setup_calls.append(("capture",)),
        _new_session_startup_settings=lambda workspace: (
            setup_calls.append(("settings", workspace)) or (settings, {})
        ),
        _console_new_chat_default_generation=lambda: 19,
        switch_and_bind=lambda session_id: setup_calls.append(("bind", session_id)),
    )
    setup_fixture = SimpleNamespace(
        app=SimpleNamespace(screen=SimpleNamespace(_session=session_owner)),
        controller=SimpleNamespace(new_session=create_session),
        workspace_id="private-workspace",
    )
    exec(
        compile(
            ast.Module(
                body=copy.deepcopy(stage.body[setup_start : setup_end + 1]),
                type_ignores=[],
            ),
            "actual-session-setup",
            "exec",
        ),
        {"self": setup_fixture, "correlation": "a" * 32},
    )
    created = setup_calls[2][1]
    results["actual_setup_delegates_settings_capture_create_bind_once"] = (
        [row[0] for row in setup_calls] == ["capture", "settings", "create", "bind"]
        and created["settings"] is settings
        and created["canonical_settings_baseline"] is settings
        and created["new_chat_default_generation"] == 19
        and created["assistant_kind"] is None
        and setup_calls[-1] == ("bind", session.id)
        and setup_fixture.session is session
    )
    wiring = ast.parse(
        (source / "tldw_chatbook/UI/Console_Modules/wiring.py").read_text(
            encoding="utf-8"
        )
    )
    receive = next(
        node
        for node in wiring.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "receive_console_visible_intent"
    )
    receive_code = ast.unparse(receive)
    dispatch_code = (
        source / "tldw_chatbook/Chat/console_received_dispatch.py"
    ).read_text(encoding="utf-8")
    controls_code = (
        source / "tldw_chatbook/UI/Console_Modules/composer_run_controls.py"
    ).read_text(encoding="utf-8")
    results["original_stock_owner_chain_remains_source_bound"] = (
        all(
            term in receive_code
            for term in (
                "._publish_console_authored_draft(",
                ".accept_received_intent(",
                "return turn_id",
            )
        )
        and "record.task = self._create_custody_task(coroutine)" in dispatch_code
        and '"_project_console_received_preparing"' in dispatch_code
        and "self._start_console_transcript_sync_timer()" in controls_code
    )
    results["fixture_never_manually_paints_or_starts_timer"] = not any(
        isinstance(node, ast.Attribute)
        and node.attr
        in {
            "_start_console_transcript_sync_timer",
            "_stop_console_transcript_sync_timer",
            "_sync_native_console_chat_ui",
            "sync_poll_display_ui",
        }
        for node in ast.walk(owner)
    )
    observer_tree = ast.parse(
        (here / "native_product_observer.py").read_text(encoding="utf-8")
    )
    partial = next(
        node
        for node in ast.walk(observer_tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "record"
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "rehearsal_events"
            for target in node.targets
        )
    )
    namespace = {
        "record": {"stage": {"snapshot": {"current": {"correlation": "a"}}}},
        "event_rows": [
            {"correlation": "a", "stage": "app_input_prequeue"},
            {"correlation": "b", "stage": "request_complete"},
        ],
    }
    exec(
        compile(
            ast.Module(body=[partial], type_ignores=[]),
            "actual-final-accounting",
            "exec",
        ),
        namespace,
    )
    results["invalid_round_partial_stages_retained_without_cross_round_rows"] = (
        namespace["record"]["rehearsal_events"] == [namespace["event_rows"][0]]
    )


asyncio.run(main())
receipt = {
    "scope": "Pure actual fixture delegation and stock-owner source contracts only; no App/product import/GUI/dispatch/timing qualification",
    "workload_SHA256": hashlib.sha256(workload_path.read_bytes()).hexdigest(),
    "results": results,
    "details": details,
    "passed": sum(results.values()),
    "failed": sum(not value for value in results.values()),
    "qualified": False,
    "product_samples": 0,
}
(output / "visible-send-route-controls.json").write_text(
    json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(receipt))
raise SystemExit(0 if all(results.values()) else 1)
