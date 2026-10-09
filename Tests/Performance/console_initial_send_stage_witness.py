"""Opt-in finite original send stages; no payloads, task owners or global hooks."""

import hashlib
import contextlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import CodeType, FunctionType, ModuleType

import pytest


OUTPUT = "TLDW_INITIAL_SEND_STAGE_RECEIPT"
TARGET = (
    "Tests/UI/test_console_turn_resend_ui.py::"
    "test_console_resend_click_re_runs_a_failed_turn_in_place"
)
SPECS = {
    "textual.widgets._button": ("Button.press",),
    "Tests.UI.test_console_native_chat_flow": ("_select_llamacpp_console",),
    "tldw_chatbook.Widgets.Console.console_composer_bar": (
        "ConsoleComposerBar.load_draft",
        "ConsoleComposerBar._sync_current_action_state",
        "ConsoleComposerBar.sync_action_state",
    ),
    "tldw_chatbook.UI.Screens.chat_screen": (
        "ChatScreen.handle_console_send_message",
        "ChatScreen._sync_console_control_bar",
        "ChatScreen._sync_console_control_bar_under_config",
        "ChatScreen._active_console_settings_readiness",
        "ChatScreen._console_setup_blocked_reason",
        "ChatScreen._send_console_message_from_visible_action",
        "ChatScreen._send_console_message_from_visible_action_observed",
        "ChatScreen._dispatch_console_draft_send",
        "ChatScreen._start_console_transcript_sync_timer",
        "ChatScreen._sync_native_console_chat_ui",
        "ChatScreen._sync_native_console_transcript",
        "ChatScreen._sync_console_native_session_tabs",
    ),
    "tldw_chatbook.UI.Console_Modules.prompt_queue": (
        "ConsolePromptQueueUIController.dispatch",
        "ConsolePromptQueueUIController.presentation_for",
        "ConsolePromptQueueUIController._stage_normal_chain",
        "_preparation_refusal_detail",
    ),
    "tldw_chatbook.UI.Console_Modules.hooks": ("ConsoleHooksController.dispatch",),
    "tldw_chatbook.UI.Console_Modules.console_spend_projection": (
        "ConsoleReadinessConfigProjection.warm",
        "ConsoleReadinessConfigProjection.run",
        "ConsoleReadinessConfigProjection._refresh",
        "ConsoleReadinessConfigProjection.for_screen.<locals>.read_current",
        "ConsoleContextReadSnapshot.warm",
        "ConsoleContextReadSnapshot._refresh",
    ),
    "tldw_chatbook.UI.Console_Modules.wiring": (
        "_prepare_console_turn_to_runtime",
        "_capture_console_configuration_async",
        "_admit_console_turn_to_runtime",
        "_commit_captured_console_draft",
    ),
    "tldw_chatbook.Chat.console_chat_controller": (
        "ConsoleChatController.capture_turn_configuration_snapshot",
        "ConsoleChatController.activity_for",
        "ConsoleChatController.submit_draft",
        "ConsoleChatController._submit_draft_lifecycle",
        "ConsoleChatController._submit_draft_body",
    ),
    "tldw_chatbook.UI.Console_Modules.session": (
        "ConsoleSessionController._build_console_turn_execution_context",
    ),
    "tldw_chatbook.Chat.console_runtime": (
        "ConsoleRuntime.accept_turn",
        "ConsoleRuntime._run_custodied_turn",
    ),
    "tldw_chatbook.MCP.console_snapshot": (
        "capture_console_definition_maximum",
        "capture_console_definition_maximum.<locals>.read_sources",
        "_owned_worker",
        "_checked_read",
    ),
}


def _shape(code):
    return (
        code.co_code,
        code.co_exceptiontable,
        code.co_stacksize,
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        tuple(
            _shape(value) if type(value) is CodeType else value
            for value in code.co_consts
        ),
    )


def _nested(code, qualname):
    for part in qualname.split("."):
        if part == "<locals>":
            continue
        matches = [
            value
            for value in code.co_consts
            if type(value) is CodeType and value.co_name == part
        ]
        assert len(matches) == 1, qualname
        code = matches[0]
    return code


def _descriptor(module, qualname):
    owner = module
    for part in qualname.split("."):
        owner = inspect.getattr_static(owner, part)
    return owner


def _function(module, qualname):
    owner = _descriptor(module, qualname)
    if type(owner) is classmethod:
        assert qualname == "ConsoleReadinessConfigProjection.for_screen"
        owner = owner.__func__
    assert type(owner) is FunctionType, qualname
    return owner


def _body(module, selection, installed, compiled):
    """Accept the exact existing maintenance wrapper, never arbitrary unwrap."""
    code = installed.__code__
    if code.co_qualname == selection:
        return installed
    assert selection == "ConsoleChatController.submit_draft"
    assert (
        code.co_qualname == "_maintenance_boundary.<locals>.decorate.<locals>.admitted"
    )
    assert installed.__globals__ is module.__dict__
    assert _shape(code) == _shape(_nested(compiled, code.co_qualname))
    body = installed.__dict__.get("__wrapped__")
    assert type(body) is FunctionType and body.__code__.co_qualname == selection
    bindings = dict(zip(code.co_freevars, installed.__closure__, strict=True))
    assert bindings["function"].cell_contents is body
    assert bindings["kind"].cell_contents == "turn"
    return body


def _runner_lifetime(root):
    """Recognize the original ordinary runner while its owner context is active."""
    path = root / "Tests/windows_private_fixture_runner.py"
    compiled = compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    for name in ("__main__", "Tests.windows_private_fixture_runner"):
        module = sys.modules.get(name)
        filename = getattr(module, "__file__", None)
        if (
            type(module) is not ModuleType
            or type(filename) is not str  # noqa: E721 - exact primitive diagnostic metadata
            or Path(filename).resolve() != path
        ):
            continue
        main = vars(module).get("main")
        factory = vars(module).get("user_fixture_default_owner")
        if type(main) is not FunctionType or type(factory) is not FunctionType:
            continue
        body = factory.__dict__.get("__wrapped__")
        if not (
            main.__globals__ is module.__dict__
            and _shape(main.__code__) == _shape(_nested(compiled, "main"))
            and type(body) is FunctionType
            and body.__globals__ is module.__dict__
            and _shape(body.__code__)
            == _shape(_nested(compiled, "user_fixture_default_owner"))
            and factory.__globals__ is contextlib.__dict__
            and any(
                factory.__code__ is part
                for part in contextlib.contextmanager.__code__.co_consts
            )
            and dict(
                zip(factory.__code__.co_freevars, factory.__closure__, strict=True)
            )["func"].cell_contents
            is body
            and vars(module).get("os") is os
            and vars(module).get("sys") is sys
            and vars(module).get("contextmanager") is contextlib.contextmanager
        ):
            continue
        if _main_frame_active(main, sys._getframe(1)):
            return True
    return False


def _main_frame_active(main, frame):
    for _ in range(64):
        if frame is None:
            break
        if frame.f_code is main.__code__ and frame.f_globals is main.__globals__:
            return True
        frame = frame.f_back
    return False


def _bootstrap_profile_lifetime(request, root, *, environment=None):
    """Check the original collection-selected process profile, without I/O authority."""
    path = root / "Tests/conftest.py"
    matches = [
        module
        for module in request.config.pluginmanager.get_plugins()
        if type(module) is ModuleType
        and type(getattr(module, "__file__", None)) is str  # noqa: E721 - exact primitive diagnostic metadata
        and Path(module.__file__).resolve() == path
    ]
    if len(matches) != 1:
        return False
    module = matches[0]
    selected = vars(module).get("_BOOTSTRAP_CONFIG_ROOT")
    owns = vars(module).get("_OWNS_BOOTSTRAP_CONFIG_ROOT")
    if (
        not isinstance(selected, Path)
        or not selected.is_absolute()
        or type(owns) is not bool  # noqa: E721 - exact primitive diagnostic metadata
    ):
        return False
    values = {
        "TLDW_TEST_CONFIG_ROOT": selected,
        "HOME": selected / "home",
        "USERPROFILE": selected / "home",
        "XDG_DATA_HOME": selected / "data",
        "XDG_CONFIG_HOME": selected / "config",
        "TLDW_CONFIG_PATH": selected / "config/config.toml",
    }
    environment = os.environ if environment is None else environment
    return all(
        environment.get(name) == str(value) for name, value in values.items()
    ) and (not owns or environment.get("TLDW_TEST_CONFIG_ROOT_OWNER") == str(path))


def _send_button_flags(receiver):
    """Transient original control state; no receiver or private value is saved."""
    values = {
        "disabled": receiver.disabled,
        "is_mounted": receiver.is_mounted,
        "display": receiver.display,
    }
    assert all(type(value) is bool for value in values.values())  # noqa: E721 - exact primitive diagnostic metadata
    return values


def _send_parent_chain(
    receiver, screen_class, message_pump_class, parent_descriptor, parent_getter
):
    """Use the original weakref getter, never an arbitrary parent override."""
    node = receiver
    for _depth in range(48):
        if message_pump_class not in type(node).__mro__:
            return False
        if inspect.getattr_static(type(node), "_parent") is not parent_descriptor:
            return False
        node = parent_getter(node)
        if type(node) is screen_class:
            return True
        if node is None:
            return False
    return False


def _decision_call_source(frame, codes):
    """Save only one selected defining stage name, never a frame or arguments."""
    frame = frame.f_back
    for _depth in range(8):
        if frame is None:
            return None
        binding = codes.get(frame.f_code)
        if binding is not None and frame.f_globals is binding[2]:
            return binding[1]
        frame = frame.f_back
    return None


def _send_cached_flags(
    receiver, screen_class, composer_class, parent_getter, projection_class
):
    """Read existing plain metadata only. No draft/config/queue method runs."""
    node, composer, screen = receiver, None, None
    for _depth in range(48):
        node = parent_getter(node)
        if node is None:
            break
        if type(node) is composer_class:
            composer = node
        if type(node) is screen_class:
            screen = node
            break
    flags = {
        "exact_composer_found": composer is not None,
        "exact_screen_found": screen is not None,
    }
    if composer is not None:
        values = vars(composer)
        for field in (
            "_send_blocked",
            "_dispatch_recovery_blocked",
            "_run_active",
            "_stop_available",
            "_wake_turn_active",
            "_raw_cli_prefix_typed",
        ):
            flags["cached" + field] = values.get(field) is True
        for field in (
            "_setup_blocked_reason",
            "_queue_blocked_reason",
            "_send_disabled_reason",
        ):
            value = values.get(field)
            flags["cached" + field + "_present"] = type(value) is str and value != ""  # noqa: E721 - exact primitive metadata
        flags["cached_attachment_present"] = (
            values.get("_pending_attachment_label") is not None
        )
    if screen is not None:
        values = vars(screen)
        for field in (
            "_console_sync_in_progress",
            "_console_sync_requested",
            "_console_sync_maintenance_paused",
        ):
            flags[field.lstrip("_")] = values.get(field) is True
        projection = values.get("_console_readiness_config_projection")
        flags["exact_readiness_projection_present"] = (
            type(projection) is projection_class
        )
        if type(projection) is projection_class:
            cache = vars(projection)
            flags["projection_screen_matches"] = cache.get("screen") is screen
            flags["projection_pending"] = cache.get("pending") is True
            flags["projection_value_present"] = cache.get("value") is not None
            key = cache.get("key")
            flags["projection_owner_key_present"] = (
                type(key) is tuple and len(key) == 11
            )  # noqa: E721 - exact primitive metadata
            if flags["projection_owner_key_present"]:
                store_values = vars(key[4]) if key[4] is not None else {}
                sessions = store_values.get("_sessions")
                active_id = store_values.get("active_session_id")
                owner = None
                if type(sessions) is dict:  # noqa: E721 - exact cached mapping
                    owner = dict.get(sessions, active_id)
                flags["projection_cached_session_owner_matches"] = (
                    active_id == key[5] and owner is key[9]
                )
                flags["projection_cached_settings_revision_matches"] = (
                    owner is not None and vars(owner).get("settings_revision") == key[7]
                )
                settings = vars(owner).get("settings") if owner is not None else None
                settings_values = vars(settings) if settings is not None else {}
                flags["session_selected_llama_cpp"] = (
                    settings_values.get("provider") == "llama_cpp"
                )
                model = settings_values.get("model")
                flags["session_model_present"] = type(model) is str and model != ""  # noqa: E721 - exact primitive metadata
    assert all(type(value) is bool for value in flags.values())  # noqa: E721 - exact primitive metadata
    return flags


def _decision_flags(selection, event, frame, result):
    """Only allow-listed booleans from an actual original call or return."""
    if selection == "ConsoleComposerBar.sync_action_state" and event == "PY_START":
        fields = (
            "has_draft",
            "send_blocked",
            "dispatch_recovery_blocked",
            "run_active",
            "wake_turn_active",
        )
        flags = {field: frame.f_locals.get(field) is True for field in fields}
        for field in ("setup_blocked_reason", "queue_blocked_reason"):
            value = frame.f_locals.get(field)
            flags[field + "_present"] = type(value) is str and value != ""  # noqa: E721 - exact primitive metadata
        return flags
    if event != "PY_RETURN":
        return None
    if selection in (
        "ConsoleReadinessConfigProjection.run",
        "ChatScreen._sync_console_control_bar",
    ):
        return {"returned_true": result is True, "returned_false": result is False}
    if selection == "ChatScreen._console_setup_blocked_reason":
        return {"setup_reason_present": type(result) is str and result != ""}  # noqa: E721 - exact primitive metadata
    if (
        selection == "ChatScreen._active_console_settings_readiness"
        and type(result) is tuple
        and len(result) == 2
    ):  # noqa: E721 - exact primitive metadata
        values = vars(result[1])
        selected = vars(result[0])
        model = selected.get("model")
        return {
            "selected_llama_cpp": selected.get("provider") == "llama_cpp",
            "selected_model_present": type(model) is str and model != "",  # noqa: E721 - exact primitive metadata
            "ready_to_send": values.get("operability") == "ready_to_send",
            "native_send_supported": values.get("native_send_supported") is True,
            "needs_model": values.get("recovery_action") == "select_model",
            "needs_credential": values.get("recovery_action") == "configure_credential",
            "needs_endpoint": values.get("recovery_action")
            in {"configure_endpoint", "save_endpoint"},
            "needs_connection_retry": values.get("recovery_action")
            == "retry_connection",
            "waiting_for_run": values.get("recovery_action") == "wait_for_active_run",
        }
    if selection == "ConsolePromptQueueUIController.presentation_for":
        return {
            "queue_send_enabled": object.__getattribute__(result, "send_enabled")
            is True,
            "queue_empty": object.__getattribute__(result, "count") == 0,
            "queue_paused": object.__getattribute__(result, "paused") is True,
        }
    if selection == "ConsoleChatController.activity_for":
        return {
            field: object.__getattribute__(result, field) is True
            for field in (
                "occupies_slot",
                "preparing_before_acceptance",
                "accepted_live_turn",
                "needs_approval",
                "queue_paused",
            )
        }
    return None


@pytest.fixture(autouse=True)
def _observe_initial_send_stages(request):
    if not os.environ.get(OUTPUT) or request.node.nodeid != TARGET:
        yield
        return
    root = Path(request.config.rootpath).resolve()
    # This unchanged bootstrap-marked node runs directly in the original
    # ordinary runner. It is not decorated with private_profile_test.
    assert request.node.get_closest_marker("bootstrap_profile") is not None
    assert _runner_lifetime(
        root
    ), "diagnostic requires the original private process runner"
    assert _bootstrap_profile_lifetime(
        request, root
    ), "collection profile changed before diagnostic"
    modules, pins, codes = {}, [], {}
    for name, selections in SPECS.items():
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if name == "textual.widgets._button":
            package = sys.modules.get("textual")
            widgets = sys.modules.get("textual.widgets")
            assert type(package) is ModuleType and type(widgets) is ModuleType
            package_path = Path(package.__file__).resolve()
            assert package_path.name == "__init__.py"
            assert path == package_path.parent / "widgets/_button.py"
            assert type(module) is ModuleType
            assert module.__spec__ is not None
            assert Path(module.__spec__.origin).resolve() == path
            assert (
                inspect.getattr_static(module, "Button")
                is widgets.__dict__["_WIDGETS_LAZY_LOADING_CACHE"]["Button"]
            )
            for dependency in (package, widgets):
                dependency_path = Path(dependency.__file__).resolve()
                modules[dependency.__name__] = (
                    dependency,
                    dependency_path,
                    hashlib.sha256(dependency_path.read_bytes()).hexdigest(),
                )
        else:
            assert path == root.joinpath(*name.split(".")).with_suffix(".py")
        source = path.read_bytes()
        compiled = compile(source, str(path), "exec", dont_inherit=True)
        modules[name] = (module, path, hashlib.sha256(source).hexdigest())
        for selection in selections:
            outer = selection.split(".<locals>.")[0]
            installed = _function(module, outer)
            function = _body(module, outer, installed, compiled)
            assert function.__globals__ is module.__dict__
            assert _shape(function.__code__) == _shape(_nested(compiled, outer))
            code = (
                _nested(function.__code__, selection.split(".<locals>.", 1)[1])
                if ".<locals>." in selection
                else function.__code__
            )
            assert _shape(code) == _shape(_nested(compiled, selection))
            pins.append(
                (
                    module,
                    outer,
                    _descriptor(module, outer),
                    installed,
                    function,
                    function.__code__,
                    function.__defaults__,
                    compiled,
                )
            )
            codes[code] = (name, selection, module.__dict__, None)
            if installed is not function:
                # This wrapper code is shared by other controller methods.
                # Observe only the closure that owns this exact original body.
                codes[installed.__code__] = (
                    name,
                    selection + "[maintenance_wrapper]",
                    module.__dict__,
                    function,
                )
    composer_name = "tldw_chatbook.Widgets.Console.console_composer_bar"
    composer_module = sys.modules.get(composer_name)
    assert type(composer_module) is ModuleType
    composer_path = Path(composer_module.__file__).resolve()
    assert composer_path == root.joinpath(*composer_name.split(".")).with_suffix(".py")
    composer_source = composer_path.read_bytes()
    composer_compiled = compile(
        composer_source, str(composer_path), "exec", dont_inherit=True
    )
    modules[composer_name] = (
        composer_module,
        composer_path,
        hashlib.sha256(composer_source).hexdigest(),
    )
    composer_button_class = inspect.getattr_static(
        composer_module, "ComposerControlButton"
    )
    assert composer_button_class.__module__ == composer_name
    assert composer_button_class.__qualname__ == "ComposerControlButton"
    composer_fields = tuple(vars(composer_button_class).items())
    composer_methods = []
    for member, function in composer_fields:
        if type(function) is FunctionType:
            assert function.__globals__ is composer_module.__dict__
            assert _shape(function.__code__) == _shape(
                _nested(composer_compiled, "ComposerControlButton." + member)
            )
            composer_methods.append(
                (
                    function,
                    function.__code__,
                    function.__defaults__,
                    function.__kwdefaults__,
                    function.__closure__,
                    tuple(
                        (cell, cell.cell_contents)
                        for cell in function.__closure__ or ()
                    ),
                )
            )
    factory = _descriptor(composer_module, "ConsoleComposerBar._bounded_button")
    assert type(factory) is staticmethod
    bounded = factory.__func__
    assert (
        type(bounded) is FunctionType
        and bounded.__globals__ is composer_module.__dict__
    )
    assert _shape(bounded.__code__) == _shape(
        _nested(composer_compiled, "ConsoleComposerBar._bounded_button")
    )
    bounded_code = bounded.__code__
    button_module = modules["textual.widgets._button"][0]
    button_class = inspect.getattr_static(button_module, "Button")
    assert composer_button_class.__bases__ == (button_class,)
    composer_mro = (composer_button_class,) + button_class.__mro__
    assert composer_button_class.__mro__ == composer_mro
    assert "press" not in vars(composer_button_class)
    button_press = inspect.getattr_static(button_class, "press")
    assert inspect.getattr_static(composer_button_class, "press") is button_press
    screen_module = modules["tldw_chatbook.UI.Screens.chat_screen"][0]
    screen_class = inspect.getattr_static(screen_module, "ChatScreen")
    composer_class = inspect.getattr_static(composer_module, "ConsoleComposerBar")
    projection_class = inspect.getattr_static(
        modules["tldw_chatbook.UI.Console_Modules.console_spend_projection"][0],
        "ConsoleReadinessConfigProjection",
    )
    pump_name = "textual.message_pump"
    pump_module = sys.modules.get(pump_name)
    assert type(pump_module) is ModuleType
    pump_path = Path(pump_module.__file__).resolve()
    assert pump_path == modules["textual"][1].parent / "message_pump.py"
    assert pump_module.__spec__ is not None
    assert Path(pump_module.__spec__.origin).resolve() == pump_path
    pump_raw = pump_path.read_bytes()
    pump_compiled = compile(pump_raw, str(pump_path), "exec", dont_inherit=True)
    modules[pump_name] = (pump_module, pump_path, hashlib.sha256(pump_raw).hexdigest())
    message_pump_class = inspect.getattr_static(pump_module, "MessagePump")
    parent_descriptor = inspect.getattr_static(message_pump_class, "_parent")
    assert type(parent_descriptor) is property
    parent_getter = parent_descriptor.fget
    assert type(parent_getter) is FunctionType
    assert parent_getter.__globals__ is pump_module.__dict__
    # Getter and setter share the property name in source; select by original
    # first line as well as code shape, rather than blessing an arbitrary getter.
    pump_class_code = _nested(pump_compiled, "MessagePump")
    getter_matches = [
        part
        for part in pump_class_code.co_consts
        if type(part) is CodeType
        and part.co_name == "_parent"
        and part.co_firstlineno == parent_getter.__code__.co_firstlineno
    ]
    assert len(getter_matches) == 1
    assert _shape(parent_getter.__code__) == _shape(getter_matches[0])
    parent_code = parent_getter.__code__
    parent_defaults = parent_getter.__defaults__
    parent_kwdefaults = parent_getter.__kwdefaults__
    parent_closure = parent_getter.__closure__
    parent_cells = tuple((cell, cell.cell_contents) for cell in parent_closure or ())
    assert inspect.getattr_static(composer_button_class, "_parent") is parent_descriptor
    assert inspect.getattr_static(screen_class, "_parent") is parent_descriptor
    button_cache = modules["textual.widgets"][0].__dict__["_WIDGETS_LAZY_LOADING_CACHE"]
    monitor = sys.monitoring
    tool = next(index for index in range(6) if monitor.get_tool(index) is None)
    monitor.use_tool_id(tool, "tldw-original-initial-send-stages")
    assert monitor.get_events(tool) == 0
    rows, pending, actors, incomplete = [], {}, [], []
    counts = {entry[1]: {} for entry in codes.values()}
    started, serial, dropped, active = time.monotonic(), 0, 0, True
    # Unwind/throw require global events on Python 3.12. Keep global zero:
    # missing returns remain explicit incomplete evidence rather than retirement.
    event_names = ("PY_START", "PY_RETURN", "PY_YIELD", "PY_RESUME")
    callbacks = {}

    def callback(event):
        def observe(code, offset, *_ignored):
            nonlocal serial, dropped
            assert active and code in codes
            frame = sys._getframe(1)
            name, selection, globals_owner, wrapper_body = codes[code]
            assert frame.f_code is code and frame.f_globals is globals_owner
            if (
                wrapper_body is not None
                and frame.f_locals.get("function") is not wrapper_body
            ):
                return
            button_flags = None
            if selection == "Button.press":
                receiver = frame.f_locals.get("self")
                if (
                    type(receiver) is not composer_button_class
                    or receiver.id != "console-send-message"
                ):
                    return
                if not _send_parent_chain(
                    receiver,
                    screen_class,
                    message_pump_class,
                    parent_descriptor,
                    parent_getter,
                ):
                    return
                button_flags = _send_button_flags(receiver)
            decision_flags = _decision_flags(
                selection,
                event,
                frame,
                _ignored[-1] if event == "PY_RETURN" and _ignored else None,
            )
            cached_flags = (
                _send_cached_flags(
                    receiver,
                    screen_class,
                    composer_class,
                    parent_getter,
                    projection_class,
                )
                if selection == "Button.press"
                else None
            )
            call_source = _decision_call_source(frame, codes)
            actor = threading.current_thread()
            actor_index = next(
                (i for i, owner in enumerate(actors) if owner is actor), None
            )
            if actor_index is None:
                actor_index = len(actors)
                actors.append(actor)
            key = (code, id(frame), actor_index)
            if event == "PY_START":
                previous = pending.pop(key, None)
                if previous is not None:
                    incomplete.append(
                        {
                            "stage": selection,
                            "span": previous,
                            "actor": actor_index,
                            "reason": "new_start_without_observed_prior_return",
                        }
                    )
                serial += 1
                pending[key] = serial
            span = pending.get(key)
            assert span is not None, (selection, event)
            counts[selection][event] = counts[selection].get(event, 0) + 1
            row = {
                "event": event,
                "stage": selection,
                "span": span,
                "actor": actor_index,
                "pid": os.getpid(),
                "module": name,
                "offset": offset,
                "line": frame.f_lineno,
                "observed_elapsed_seconds": time.monotonic() - started,
            }
            if button_flags is not None:
                row["send_button"] = button_flags
            if cached_flags is not None:
                row["cached_send_state"] = cached_flags
            if decision_flags is not None:
                assert all(type(value) is bool for value in decision_flags.values())  # noqa: E721 - exact primitive metadata
                row["decision_flags"] = decision_flags
            if call_source is not None:
                row["selected_call_source"] = call_source
            if len(rows) < 2048:
                rows.append(row)
            else:
                dropped += 1
            if event == "PY_RETURN":
                del pending[key]

        return observe

    setup_complete = False
    try:
        mask = 0
        for event in event_names:
            event_id = getattr(monitor.events, event)
            callbacks[event_id] = callback(event)
            monitor.register_callback(tool, event_id, callbacks[event_id])
            mask |= event_id
        for code in codes:
            monitor.set_local_events(tool, code, mask)
        setup_complete = True
        yield
    finally:
        # Retire every local event and callback before marking observation inactive.
        try:
            for code in codes:
                monitor.set_local_events(tool, code, 0)
            for event_id in callbacks:
                monitor.register_callback(tool, event_id, None)
            assert monitor.get_events(tool) == 0
        finally:
            monitor.free_tool_id(tool)
            active = False
        current = all(
            sys.modules.get(name) is module
            and hashlib.sha256(path.read_bytes()).hexdigest() == digest
            for name, (module, path, digest) in modules.items()
        ) and all(
            _descriptor(module, name) is descriptor
            and _function(module, name) is installed
            and function.__code__ is code
            and function.__globals__ is module.__dict__
            and function.__defaults__ is defaults
            and (
                function is installed
                or installed.__dict__.get("__wrapped__") is function
            )
            and _body(module, name, installed, compiled) is function
            for module, name, descriptor, installed, function, code, defaults, compiled in pins
        )
        current = current and (
            button_module.__dict__.get("Button") is button_class
            and screen_module.__dict__.get("ChatScreen") is screen_class
            and modules["textual.widgets"][0].__dict__.get(
                "_WIDGETS_LAZY_LOADING_CACHE"
            )
            is button_cache
            and button_cache.get("Button") is button_class
        )
        current = current and (
            composer_module.__dict__.get("ComposerControlButton")
            is composer_button_class
            and composer_button_class.__bases__ == (button_class,)
            and composer_button_class.__mro__ == composer_mro
            and len(vars(composer_button_class)) == len(composer_fields)
            and all(
                vars(composer_button_class).get(name) is value
                for name, value in composer_fields
            )
            and inspect.getattr_static(composer_button_class, "press") is button_press
            and _descriptor(composer_module, "ConsoleComposerBar._bounded_button")
            is factory
            and bounded.__code__ is bounded_code
            and bounded.__globals__ is composer_module.__dict__
            and all(
                function.__code__ is code
                and function.__globals__ is composer_module.__dict__
                and function.__defaults__ is defaults
                and function.__kwdefaults__ is kwdefaults
                and function.__closure__ is closure
                and all(
                    cell is original and cell.cell_contents is value
                    for cell, (original, value) in zip(closure or (), cells)
                )
                for function, code, defaults, kwdefaults, closure, cells in composer_methods
            )
        )
        current = current and (
            composer_module.__dict__.get("ConsoleComposerBar") is composer_class
            and modules["tldw_chatbook.UI.Console_Modules.console_spend_projection"][
                0
            ].__dict__.get("ConsoleReadinessConfigProjection")
            is projection_class
        )
        current = current and (
            pump_module.__dict__.get("MessagePump") is message_pump_class
            and inspect.getattr_static(message_pump_class, "_parent")
            is parent_descriptor
            and parent_descriptor.fget is parent_getter
            and parent_getter.__code__ is parent_code
            and parent_getter.__globals__ is pump_module.__dict__
            and parent_getter.__defaults__ is parent_defaults
            and parent_getter.__kwdefaults__ is parent_kwdefaults
            and parent_getter.__closure__ is parent_closure
            and all(
                cell is original and cell.cell_contents is value
                for cell, (original, value) in zip(parent_closure or (), parent_cells)
            )
            and inspect.getattr_static(composer_button_class, "_parent")
            is parent_descriptor
            and inspect.getattr_static(screen_class, "_parent") is parent_descriptor
        )
        output = Path(os.environ[OUTPUT])
        assert not output.exists()
        output.write_text(
            json.dumps(
                {
                    "diagnostic_only": True,
                    "node": TARGET,
                    "original_runner_and_bootstrap_profile_lifetime": True,
                    "rows": rows,
                    "counts": counts,
                    "dropped_rows": dropped,
                    "setup_complete": setup_complete,
                    "incomplete": incomplete,
                    "complete_pairs_only": not pending
                    and not incomplete
                    and dropped == 0,
                    "source_current": current,
                    "global_events": monitor.get_events(tool),
                    "hooks_retired": monitor.get_tool(tool) is None,
                    "pending": [
                        {"stage": codes[code][1], "span": span, "actor": actor_index}
                        for (code, _frame_id, actor_index), span in pending.items()
                    ],
                    "actors": [
                        {
                            "actor": index,
                            "ident": actor.ident,
                            "alive_at_retirement": actor.is_alive(),
                        }
                        for index, actor in enumerate(actors)
                    ],
                    "sources": {
                        name: digest
                        for name, (_module, _path, digest) in modules.items()
                    },
                    "send_button_observation_is_exact_original_press_and_parent_screen": True,
                    "send_button_flags_only_no_label_action_or_draft_values": True,
                    "send_receiver_is_defining_stock_composer_subclass_with_original_inherited_press": True,
                    "send_parent_chain_uses_original_source_pinned_message_pump_weakref_getter": True,
                    "no_payload_result_exception_or_task_values_recorded": True,
                    "observed_elapsed_is_diagnostic_only_not_acceptance_timing": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        assert current
