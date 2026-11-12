"""One real managed-demand received Send; isolated, unpaid provider boundary."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = r"""import asyncio
import contextvars
import inspect
import json
import os
import sqlite3
import sys
import threading
import time
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures.thread import _WorkItem
from pathlib import Path
from types import BuiltinMethodType, FunctionType

from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner

network_guard.install()
real_profile_guard.install()
from loguru import logger

logger.remove()
assert sys.argv[1:] == ["managed_received", "cold_policy_switch"]


def plain(value):
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def native_closed(connection):
    try:
        sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError:
        return True
    return False


async def exercise():
    from textual.worker import Worker

    from Tests.Performance._stock_private_app_creators import OriginalPrivateAppCreators
    from Tests.Performance.console_storage_unit_observer import (
        OriginalStorageUnitObserver,
    )
    from Tests.UI.test_console_hook_review_send_freeze import _key, _until
    from tldw_chatbook import app as app_source
    from tldw_chatbook import app_service_wiring as wiring
    from tldw_chatbook import app_skill_wiring as skill_wiring
    from tldw_chatbook.Agents.hook_permissions import HookPermissions
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Chat import Chat_Functions
    from tldw_chatbook.Chat import console_configuration_capture as capture
    from tldw_chatbook.Chat import console_configuration_preparation as preparation
    from tldw_chatbook.Chat import console_preparation_reads as read_source
    from tldw_chatbook.Chat import console_received_dispatch as dispatch
    from tldw_chatbook.runtime_policy.bootstrap import (
        derive_configured_server_binding,
        set_authoritative_runtime_source,
    )
    from tldw_chatbook.runtime_policy.enforcement import ServicePolicyEnforcer
    from tldw_chatbook.runtime_policy.engine import PolicyEngine
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
    from tldw_chatbook.UI.Console_Modules.skill import ConsoleSkillController
    from tldw_chatbook.Utils.text_selection_crash_guard import FreshContextExecutor

    root = Path(os.environ["XDG_DATA_HOME"]).parent
    # Public disk creation before App construction: never touch App's lazy factory.
    seed_name = "received-managed-control"
    seed = LocalSkillsService(
        store_dir=wiring.default_local_skills_store_dir(wiring.get_user_data_dir())
    )
    assert seed._trust_service is None and seed._trust_service_factory is None
    await seed.create_skill(
        name=seed_name,
        content="---\nname: received-managed-control\ndescription: Local blocked test skill\n---\nLocal fixture body.\n",
        trust_approved=False,
    )
    assert seed._trust_service is None
    creators = OriginalPrivateAppCreators(app_source, root)
    creators.install()
    try:
        app = app_source.TldwCli()
    except BaseException as primary:
        try:
            creators.close()
        except BaseException as cleanup:
            primary.add_note("constructor_monitor_cleanup:" + type(cleanup).__name__)
        try:
            if not creators.retired:
                creators.retire()
        except BaseException as cleanup:
            primary.add_note("constructor_creator_cleanup:" + type(cleanup).__name__)
        raise
    monitor = None
    try:
        creators.finish_construction(app)
        app._cli_no_splash = True
        runtime, local = app.console_runtime, app.local_skills_service
        context, policy = app.runtime_policy, app.service_policy_enforcer
        factory = vars(local)["_trust_service_factory"]
        creator_connection = getattr(app.chachanotes_db._local, "conn", None)
        assert creator_connection is not None and not native_closed(creator_connection)
        assert (
            local.policy_enforcer is policy
            and app._local_skill_trust_service is local._trust_service is None
        )
        assert derive_configured_server_binding(app.app_config).server_configured
        state = set_authoritative_runtime_source(
            context, "server", app_config=app.app_config
        )
        assert (
            state is context.state
            and state.active_source == "server"
            and state.server_configured
        )
        assert policy.current_state() is state and not network_guard.blocked_attempts()

        loop, main = asyncio.get_running_loop(), threading.current_thread()
        monitor = OriginalStorageUnitObserver({}, lambda: False, lambda _name: None)
        builder = inspect.getattr_static(
            wiring.ServiceWiringMixin, "_build_local_skill_trust_service"
        )
        ensure = inspect.getattr_static(
            wiring.ServiceWiringMixin, "ensure_local_skill_trust_service"
        )
        functions = {
            "builder": builder,
            "ensure": ensure,
            "discovery": ConsoleSkillController._refresh_console_skill_candidates,
            "policy": PolicyEngine.evaluate,
            "hook": HookPermissions.snapshot,
            "catalog": preparation.capture_console_skill_catalog_owned,
            "finish": preparation.finish_console_skill_catalog_owned,
            "visible": LocalSkillsService._visible_records,
            "project": capture._capture_skill_context_from_records,
            "full": capture.capture_console_turn_configuration,
            "fresh": FreshContextExecutor.submit,
            "work": _WorkItem.run,
        }
        codes = {name: monitor._pin(function) for name, function in functions.items()}
        definitions = {function.__code__: function for function in functions.values()}
        monitor.slots.extend(
            (
                (
                    ConsoleSkillController,
                    "_refresh_console_skill_candidates",
                    functions["discovery"],
                ),
                (PolicyEngine, "evaluate", functions["policy"]),
                (HookPermissions, "snapshot", functions["hook"]),
                (
                    preparation,
                    "capture_console_skill_catalog_owned",
                    functions["catalog"],
                ),
                (
                    preparation,
                    "finish_console_skill_catalog_owned",
                    functions["finish"],
                ),
                (LocalSkillsService, "_visible_records", functions["visible"]),
                (capture, "_capture_skill_context_from_records", functions["project"]),
                (capture, "capture_console_turn_configuration", functions["full"]),
                (FreshContextExecutor, "submit", functions["fresh"]),
                (_WorkItem, "run", functions["work"]),
                (
                    dispatch,
                    "_run_received_intent_bound",
                    dispatch._run_received_intent_bound,
                ),
                (read_source, "run_preparation_read", read_source.run_preparation_read),
                (
                    ServicePolicyEnforcer,
                    "require_allowed",
                    ServicePolicyEnforcer.require_allowed,
                ),
            )
        )
        monitor._pin(ServicePolicyEnforcer.require_allowed)
        monitor._pin(dispatch._run_received_intent_bound)
        monitor._pin(read_source.run_preparation_read)
        monitor._pin(ThreadPoolExecutor.submit)
        monitor._pin(factory)
        assert factory.__globals__ is vars(skill_wiring)
        assert factory.__code__ is wiring._CONSOLE_SKILL_TRUST_FACTORY_CODE
        assert (
            factory.__code__.co_freevars == ("self",)
            and factory.__closure__[0].cell_contents is app
        )
        assert factory.__defaults__ is factory.__kwdefaults__ is None
        monitor.slots.extend(
            (
                (wiring, "_skill_wiring", skill_wiring),
                (skill_wiring, "_source_wiring", wiring),
                (
                    wiring.ServiceWiringMixin,
                    "_build_local_skill_trust_service",
                    builder,
                ),
                (wiring.ServiceWiringMixin, "ensure_local_skill_trust_service", ensure),
                (local, "_trust_service_factory", factory),
            )
        )
        snapshot_code = next(
            code
            for code in dispatch._run_received_intent_bound.__code__.co_consts
            if inspect.iscode(code) and code.co_name == "snapshot"
        )
        catalog_capture_code = next(
            code
            for code in functions["catalog"].__code__.co_consts
            if inspect.iscode(code) and code.co_name == "capture"
        )
        invoke_code = next(
            code
            for code in read_source.run_preparation_read.__code__.co_consts
            if inspect.iscode(code) and code.co_name == "invoke"
        )
        entered, release, returned = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        errors, discovery_tasks, provider_calls, submitted = [], [], [], []
        observed = {
            "order": [],
            "visible": 0,
            "builder_count": 0,
            "denied": 0,
            "overflow": False,
        }
        objects = {}
        controller = store = session = screen = None
        sent = False

        async def settle(predicate):
            deadline = time.monotonic() + 10
            while not predicate():
                assert time.monotonic() < deadline, (
                    "managed_received_prerequisite_timeout"
                )
                await asyncio.sleep(0.005)

        def phase(name):
            if len(observed["order"]) >= 64:
                observed["overflow"] = True
                raise AssertionError("managed_received_phase_overflow")
            observed["order"].append(name)

        def failed(label):
            if len(errors) < 16:
                errors.append(label)
            else:
                observed["overflow"] = True

        def receipt(record):
            assert record is runtime._turn_custody[record.turn_id]
            claim, intent = record.received_claim, record.received_intent
            with store._preparation_lock:
                assert (
                    claim is store._preparations_by_session.get(session.id)
                    and claim is not None
                )
            assert intent.session_id == record.session_id == session.id
            assert (
                intent.inputs.draft == draft
                and intent.inputs.draft_revision == revision
            )
            assert intent.selection is objects.setdefault("selection", intent.selection)
            assert record.task is not None and record.task.get_loop() is loop
            assert record.store is store and not provider_calls
            objects["record"] = record
            objects.setdefault("record_task", record.task)
            return record

        def start(code, offset):
            frame = sys._getframe(1)
            try:
                assert (
                    frame.f_code is code
                    and frame.f_globals is definitions[code].__globals__
                )
                if (
                    code is codes["discovery"]
                    and frame.f_locals["self"].app_instance is app
                ):
                    task = asyncio.current_task()
                    if task not in discovery_tasks:
                        assert len(discovery_tasks) < 8
                        workers = tuple(
                            worker
                            for worker in app.workers
                            if type(worker) is Worker
                            and vars(worker).get("_task") is task
                        )
                        assert len(workers) == 1
                        worker = workers[0]
                        assert (
                            vars(worker).get("_node") is runtime.view
                            and worker.group == "console-skill-discovery"
                        )
                        assert vars(worker)["_work"].cr_code is codes["discovery"]
                        discovery_tasks.append(task)
                elif code is codes["ensure"] and frame.f_locals["self"] is app:
                    phase("ensure")
                    record = objects["record"]
                    assert objects["catalog"].maximum is None
                    assert (
                        asyncio.current_task() is record.task and objects["hook_ready"]
                    )
                    receipt(record)
                    assert (
                        app._local_skill_trust_service is local._trust_service is None
                    )
                elif code is codes["project"] and frame.f_locals["local"] is local:
                    assert frame.f_locals["records"] is objects["catalog"].records
                    phase("project")
                elif (
                    code is codes.get("strict")
                    and frame.f_locals["self"] is controller
                    and sent
                ):
                    record = receipt(objects["record"])
                    assert asyncio.current_task() is record.task
                    assert frame.f_locals["_prepared_skills"] is objects["prepared"]
                    assert (
                        frame.f_locals["selection"] is record.received_intent.selection
                    )
                    phase("strict")
                elif (
                    code is codes["full"] and frame.f_locals["store"] is store and sent
                ):
                    assert (
                        frame.f_locals["_skill_context_maximum"]
                        is objects["prepared"].maximum
                    )
                    phase("full")
                elif (
                    code is codes.get("custodied")
                    and frame.f_locals["self"] is runtime
                    and sent
                ):
                    record = receipt(frame.f_locals["record"])
                    assert (
                        asyncio.current_task() is record.task
                        and record.request is not None
                    )
                    objects["configuration"] = record.request.configuration
                    assert plain(
                        record.request.configuration.skill_context_maximum
                    ) == plain(objects["prepared"].maximum)
                    assert (
                        record.request.draft == draft
                        and record.request.session_id == session.id
                    )
                    phase("custodied")
                elif (
                    code is codes.get("submit")
                    and frame.f_locals["self"] is controller
                    and frame.f_locals["function"] is objects["submit_body"]
                    and sent
                ):
                    record = objects["record"]
                    maximum = plain(
                        frame.f_locals["kwargs"]["configuration"].skill_context_maximum
                    )
                    expected = plain(objects["prepared"].maximum)
                    assert maximum == dict(expected, plugin_turn_id=record.turn_id)
                    objects["submitted_configuration"] = frame.f_locals["kwargs"][
                        "configuration"
                    ]
                    phase("submit")
            except BaseException as error:
                failed(type(error).__name__ + ":start:" + code.co_qualname)
            finally:
                del frame

        def finish(code, offset, value):
            frame = sys._getframe(1)
            try:
                assert (
                    frame.f_code is code
                    and frame.f_globals is definitions[code].__globals__
                )
                if code is codes["policy"]:
                    parent = frame.f_back
                    if (
                        parent is not None
                        and parent.f_code
                        is ServicePolicyEnforcer.require_allowed.__code__
                        and parent.f_locals.get("self") is policy
                        and frame.f_locals["action_id"] == "skills.context.list.local"
                        and not value.allowed
                        and value.reason_code == "wrong_source"
                    ):
                        assert not sent and context.state.active_source == "server"
                        observed["denied"] += 1
                elif code is codes["hook"]:
                    parent = frame.f_back
                    if (
                        parent is None
                        or parent.f_code is not snapshot_code
                        or parent.f_locals["runtime"] is not runtime
                    ):
                        return
                    receipt(parent.f_locals["record"])
                    assert value.ready is True and not entered.is_set()
                    objects["hook_ready"] = True
                    phase("hook_ready")
                elif code is codes["visible"] and frame.f_locals["self"] is local:
                    parent = frame.f_back
                    if parent is not None and parent.f_code is catalog_capture_code:
                        assert objects["hook_ready"] and seed_name in value
                        assert value[seed_name].get("source") != "builtin"
                        observed["visible"] += 1
                elif code is codes["catalog"] and sent:
                    assert type(value) is preparation.ConsoleSkillCatalogRead
                    receipt(runtime._turn_custody[value.turn_id])
                    assert not value.unavailable and value.maximum is None
                    assert (
                        seed_name in value.records
                        and value.records[seed_name].get("source") != "builtin"
                    )
                    objects["catalog"] = value
                    phase("catalog")
                elif code is codes["finish"] and sent:
                    assert type(value) is preparation.ConsolePreparedSkillContext
                    assert value.catalog is objects["catalog"]
                    objects["prepared"] = value
                    phase("prepared")
                elif code is codes["fresh"] and type(value) is Future:
                    callback = frame.f_locals.get("fn")
                    args = frame.f_locals.get("args")
                    if (
                        type(callback) is BuiltinMethodType
                        and callback.__name__ == "run"
                        and args
                    ):
                        request = args[0]
                        if (
                            type(request) is FunctionType
                            and request.__code__ is invoke_code
                        ):
                            read = dict(
                                zip(request.__code__.co_freevars, request.__closure__)
                            )["read"].cell_contents
                            if (
                                getattr(read.callback, "__self__", None) is app
                                and getattr(read.callback, "__func__", None) is builder
                            ):
                                assert len(submitted) < 2
                                submitted.append((read, request, value))
                elif code is codes["builder"] and frame.f_locals["self"] is app:
                    observed["builder_count"] += 1
                    assert (
                        sent
                        and objects["hook_ready"]
                        and objects["catalog"].maximum is None
                    )
                    assert (
                        threading.current_thread() is not main and not entered.is_set()
                    )
                    parent = frame.f_back
                    for _ in range(12):
                        if parent is None or parent.f_code is codes["work"]:
                            break
                        parent = parent.f_back
                    assert parent is not None and parent.f_code is codes["work"]
                    item = parent.f_locals["self"]
                    assert (
                        type(item) is _WorkItem
                        and type(item.future) is Future
                        and item.future.running()
                    )
                    assert (
                        type(item.fn) is BuiltinMethodType
                        and type(item.fn.__self__) is contextvars.Context
                        and item.fn.__name__ == "run"
                    )
                    context_run, request = item.args
                    assert not item.kwargs and type(context_run) is BuiltinMethodType
                    assert (
                        type(context_run.__self__) is contextvars.Context
                        and context_run.__name__ == "run"
                    )
                    assert (
                        type(request) is FunctionType
                        and request.__code__ is invoke_code
                        and request.__globals__ is vars(read_source)
                    )
                    read = dict(zip(request.__code__.co_freevars, request.__closure__))[
                        "read"
                    ].cell_contents
                    assert (
                        type(read) is read_source.ConsolePreparationRead
                        and read.creator is app
                    )
                    assert (
                        read.callback.__self__ is app
                        and read.callback.__func__ is builder
                    )
                    assert read.task is objects["record"].task
                    assert (
                        read in controller._preparation_reads
                        and read in runtime._preparation_reads
                    )
                    assert (
                        read._producer is not None
                        and not read._producer.done()
                        and not read.retired.done()
                    )
                    assert (
                        app._local_skill_trust_service is local._trust_service is None
                        and not provider_calls
                    )
                    objects.update(
                        read=read, future=item.future, request=request, winner=value
                    )
                    phase("builder_held")
                    entered.set()
                    try:
                        if not release.wait(10):
                            failed("builder_release_timeout")
                    finally:
                        returned.set()
                elif (
                    code is codes.get("accepted")
                    and frame.f_locals["self"] is runtime
                    and sent
                ):
                    assert frame.f_locals["record"] is objects["record"]
                    assert objects["record"].inputs.durable_accepted is True
                    phase("accepted")
            except BaseException as error:
                failed(type(error).__name__ + ":return:" + code.co_qualname)
            finally:
                del frame

        def enable(name, function):
            codes[name] = monitor._pin(function)
            definitions[codes[name]] = function
            monitor.codes[codes[name]] = name
            assert sys.monitoring.get_local_events(monitor.tool, codes[name]) == 0
            sys.monitoring.set_local_events(
                monitor.tool,
                codes[name],
                sys.monitoring.events.PY_START | sys.monitoring.events.PY_RETURN,
            )

        def reply(**kwargs):
            assert len(provider_calls) == 0 and "accepted" in observed["order"]
            assert objects["submitted_configuration"] is not None
            provider_calls.append(True)
            return "Managed received reply"

    except BaseException as primary:
        try:
            if monitor is not None:
                monitor.close()
        except BaseException as cleanup:
            primary.add_note("setup_monitor_cleanup:" + type(cleanup).__name__)
        try:
            if creators.active:
                creators.close()
        except BaseException as cleanup:
            primary.add_note("setup_creator_monitor_cleanup:" + type(cleanup).__name__)
        try:
            if not creators.retired:
                creators.retire()
        except BaseException as cleanup:
            primary.add_note("setup_creator_cleanup:" + type(cleanup).__name__)
        raise

    primary = None
    source_receipt = {}
    try:
        for tool in range(5, 0, -1):
            if tool == sys.monitoring.DEBUGGER_ID:
                continue
            try:
                sys.monitoring.use_tool_id(tool, "managed-received-control")
            except ValueError:
                continue
            monitor.tool = tool
            break
        assert monitor.tool is not None and sys.monitoring.get_events(monitor.tool) == 0
        monitor.codes = {}
        monitor.active = monitor.installed = True
        for event, callback in (
            (sys.monitoring.events.PY_START, start),
            (sys.monitoring.events.PY_RETURN, finish),
        ):
            assert (
                sys.monitoring.register_callback(monitor.tool, event, callback) is None
            )
            monitor.registered[event] = callback
        for name in (
            "builder",
            "ensure",
            "discovery",
            "policy",
            "hook",
            "catalog",
            "finish",
            "visible",
            "project",
            "full",
            "fresh",
        ):
            enable(name, functions[name])
        import pytest

        provider_patch = pytest.MonkeyPatch()
        provider_patch.setattr(Chat_Functions, "chat_api_call", reply)
        async with app.run_test(size=(140, 42)):
            try:
                await settle(lambda: getattr(app, "_initial_screen_pushed", False))
                initial = app._initial_screen_setup_task
                await asyncio.wait_for(asyncio.shield(initial), timeout=10)
                assert (
                    initial.done()
                    and not initial.cancelled()
                    and initial.exception() is None
                )
                await settle(
                    lambda: (
                        discovery_tasks and all(task.done() for task in discovery_tasks)
                    )
                )
                assert all(
                    not task.cancelled() and task.exception() is None
                    for task in discovery_tasks
                )
                assert observed["denied"] > 0 and observed["builder_count"] == 0
                assert app._local_skill_trust_service is local._trust_service is None
                assert not network_guard.blocked_attempts()
                screen = runtime.view
                assert screen is app.screen and screen.app_instance is app
                assert await app.handle_runtime_backend_changed("local") is True
                assert (
                    app.runtime_policy is context
                    and app.service_policy_enforcer is policy
                )
                assert (
                    context.state.active_source == "local"
                    and policy.current_state() is context.state
                )
                assert vars(local)["_trust_service_factory"] is factory
                assert app._local_skill_trust_service is local._trust_service is None
                assert (
                    not network_guard.blocked_attempts()
                    and observed["builder_count"] == 0
                )
                controller = screen._ensure_console_chat_controller()
                store = controller.store
                await settle(
                    lambda: (
                        screen._console_attach_reconciled
                        and not screen._console_attach_reconcile_running
                    )
                )
                session = store.ensure_session()
                composer = screen._console_composer_or_none()
                draft = "Describe a local managed-skill availability check"
                composer.load_draft(draft)
                composer.screen.set_focus(composer)
                try:
                    assert await _until(lambda: app.focused is composer, 1)
                except AssertionError as focus_primary:
                    try:
                        focused = app.focused
                        current_composers = tuple(
                            screen.query("#console-native-composer")
                        )
                        observed["focus_prerequisite"] = {
                            "focused_id": None if focused is None else focused.id,
                            "focused_class": None
                            if focused is None
                            else type(focused).__name__,
                            "composer_id": composer.id,
                            "composer_class": type(composer).__name__,
                            "composer_mounted": composer.is_mounted,
                            "composer_can_focus": composer.can_focus,
                            "recorded_screen_is_current": screen is app.screen,
                            "composer_owner_is_current": composer.is_mounted
                            and composer.screen is app.screen,
                            "current_composer_matches": len(current_composers) == 1
                            and current_composers[0] is composer,
                            "current_composer_count": len(current_composers),
                        }
                    except BaseException as focus_metadata_error:
                        focus_primary.add_note(
                            "focus_metadata:" + type(focus_metadata_error).__name__
                        )
                    raise
                assert composer.draft_text() == store.session_draft(session.id) == draft
                revision = store.session_input_snapshot(session.id).draft_revision
                assert (
                    app._local_skill_trust_service is local._trust_service is None
                    and observed["builder_count"] == 0
                )
                enable(
                    "strict",
                    inspect.getattr_static(
                        type(controller), "capture_turn_configuration_snapshot"
                    ),
                )
                enable(
                    "custodied",
                    inspect.getattr_static(type(runtime), "_run_custodied_turn"),
                )
                submit_wrapper = inspect.getattr_static(
                    type(controller), "submit_draft"
                )
                submit_body = inspect.getattr_static(submit_wrapper, "__wrapped__")
                objects["submit_body"] = submit_body
                monitor._pin(submit_body)
                enable("submit", submit_wrapper)
                monitor.slots.append((submit_wrapper, "__wrapped__", submit_body))
                enable("accepted", dispatch.publish_received_acceptance)
                monitor.slots.extend(
                    (
                        (
                            type(controller),
                            "capture_turn_configuration_snapshot",
                            definitions[codes["strict"]],
                        ),
                        (
                            type(controller),
                            "submit_draft",
                            definitions[codes["submit"]],
                        ),
                        (
                            type(runtime),
                            "_run_custodied_turn",
                            definitions[codes["custodied"]],
                        ),
                        (
                            dispatch,
                            "publish_received_acceptance",
                            dispatch.publish_received_acceptance,
                        ),
                    )
                )
                sent = True
                _key(app, "enter", "\r")
                await settle(entered.is_set)
                assert (
                    not errors and observed["visible"] == observed["builder_count"] == 1
                )
                read, future, record = (
                    objects["read"],
                    objects["future"],
                    objects["record"],
                )
                assert any(
                    row[0] is read and row[1] is objects["request"] and row[2] is future
                    for row in submitted
                )
                assert (
                    not record.task.done()
                    and not read.retired.done()
                    and future.running()
                )
                assert store.session_draft(session.id) == draft and not native_closed(
                    creator_connection
                )
                release.set()
                try:
                    await asyncio.wait_for(asyncio.shield(read.retired), timeout=10)
                except TimeoutError:
                    observed["failed_bound"] = "held_native_read_retirement_10s"
                    raise
                try:
                    await asyncio.wait_for(
                        asyncio.shield(objects["record_task"]), timeout=10
                    )
                except TimeoutError:
                    observed["failed_bound"] = "received_custody_task_10s"
                    raise
                assert (
                    returned.is_set()
                    and future.done()
                    and not future.cancelled()
                    and future.exception() is None
                )
                assert (
                    read._producer.done()
                    and not read._producer.cancelled()
                    and read._producer.exception() is None
                )
                assert (
                    read not in runtime._preparation_reads
                    and read not in controller._preparation_reads
                )
                assert (
                    app._local_skill_trust_service
                    is objects["winner"]
                    is local.trust_service
                )
                assert len(provider_calls) == 1 and not errors
                assert observed["visible"] == 1 and observed["builder_count"] == 1
                maximum = plain(objects["prepared"].maximum)
                assert any(
                    row["name"] == seed_name and row["trust_blocked"]
                    for row in maximum["blocked_skills"]
                )
                assert not any(
                    row["name"] == seed_name for row in maximum["available_skills"]
                )
                expected = (
                    "hook_ready",
                    "catalog",
                    "ensure",
                    "builder_held",
                    "project",
                    "prepared",
                    "strict",
                    "full",
                    "custodied",
                    "submit",
                    "accepted",
                )
                assert all(observed["order"].count(name) == 1 for name in expected)
                assert [observed["order"].index(name) for name in expected] == sorted(
                    observed["order"].index(name) for name in expected
                )
            except BaseException as error:
                primary = error
                observed["primary_failure_type"] = type(error).__name__
                raise
            finally:
                release.set()
                try:
                    try:
                        if "read" in objects:
                            await asyncio.wait_for(
                                asyncio.shield(objects["read"].retired), timeout=10
                            )
                    finally:
                        if "record_task" in objects:
                            await asyncio.wait_for(
                                asyncio.shield(objects["record_task"]), timeout=10
                            )
                except BaseException as cleanup:
                    if primary is None:
                        raise
                    primary.add_note(
                        "received_read_task_cleanup:" + type(cleanup).__name__
                    )
        assert runtime._disposed and app.console_runtime is None
        assert not runtime._preparation_reads and not controller._preparation_reads
        assert creators.retire() and native_closed(creator_connection)
    except BaseException as error:
        if primary is not None and error is not primary:
            primary.add_note("app_runtime_cleanup:" + type(error).__name__)
            raise primary from error
        primary = error
        raise
    finally:
        try:
            release.set()
            try:
                if not creators.retired:
                    creators.retire()
            finally:
                try:
                    source_receipt = monitor.close()
                finally:
                    pause = storage._begin_local_pause()
                    try:
                        drained = pause.drain(time.monotonic() + 1)
                    finally:
                        pause.resume()
                    facts = dict(
                        observed,
                        observer_errors=errors,
                        drain=drained,
                        source_receipt=source_receipt,
                        creator_receipt=creators.receipt,
                        test_retains_finite_read_catalog_receipt_until_cleanup=True,
                        provider_calls=len(provider_calls),
                        network_attempts=len(network_guard.blocked_attempts()),
                        profile_violations=len(real_profile_guard.take_violations()),
                    )
                    (root / "managed-received-facts.json").write_text(
                        json.dumps(facts, sort_keys=True), encoding="utf-8"
                    )
        except BaseException as cleanup:
            if primary is None:
                raise
            primary.add_note("managed_received_cleanup:" + type(cleanup).__name__)
    assert source_receipt["complete"] and source_receipt["original_source_current"]
    assert (
        source_receipt["hooks_retired_before_inactive"]
        and source_receipt["global_events"] == 0
    )
    assert (
        not source_receipt["invalid"]
        and not errors
        and not observed["overflow"]
        and drained
    )
    assert facts["network_attempts"] == facts["profile_violations"] == 0
    # Failed private children retain the recorder so late custody cannot escape it.
    provider_patch.undo()
    print("retired and reopened")


with user_fixture_default_owner():
    selected = Path(os.environ["TLDW_CONFIG_PATH"]).absolute()
    selected.write_text(
        '[general]\nusers_name="managed-received-control"\n'
        "[first_run]\nsetup_completed=true\n[_first_run]\nsetup_completed=true\n"
        "[splash_screen]\nenabled=false\n"
        '[api_settings.openai]\napi_key="skill-fixture-not-a-real-key"\n'
        '[chat_defaults]\nprovider="openai"\nmodel="gpt-4o-mini"\n'
        '[tldw_api]\nbase_url="http://127.0.0.1:65534"\n',
        encoding="utf-8",
    )
    selected.chmod(0o600)
    asyncio.run(asyncio.wait_for(exercise(), timeout=240))
"""


def test_received_managed_demand_uses_one_catalog_and_real_cold_initializer(tmp_path):
    launch = (
        "from Tests.Performance.test_console_managed_received_acceptance import _SCRIPT; "
        "exec(compile(_SCRIPT, '<managed-received-control>', 'exec'))"
    )
    _run(tmp_path, "managed_received", "cold_policy_switch", script=launch, timeout=240)
