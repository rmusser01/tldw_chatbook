"""Opt-in finite original-effect witness; imports no application source.

Construct in the actual private-profile child after its original test imports.
Only exact selected synchronous code START/RETURN events are observed. Guards,
actor context, databases, waits, deadlines and original journey stay unchanged.
"""

import ast
import dis
import hashlib
import inspect
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import CodeType, FunctionType

import pytest


TARGET = "test_session_close_pending_race_and_fleet_journeys"
TESTS = "Tests.UI.test_console_session_tab_close"


def _shape(code):
    return (
        code.co_name,
        code.co_qualname,
        code.co_firstlineno,
        code.co_code,
        code.co_flags,
        code.co_stacksize,
        code.co_exceptiontable,
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        tuple(
            _shape(item) if type(item) is CodeType else item for item in code.co_consts
        ),
    )


def _source_code(module, qualname, *, config=None):
    source = Path(module.__file__)
    if config is None:
        code = compile(source.read_bytes(), str(source), "exec", dont_inherit=True)
    else:
        from _pytest.assertion import rewrite

        loader = module.__spec__.loader
        assert type(loader) is rewrite.AssertionRewritingHook
        assert module.__loader__ is loader and loader.config is config
        assert Path(module.__spec__.origin).resolve() == source.resolve()
        assert loader._rewritten_names[module.__name__].resolve() == source.resolve()
        _, code = rewrite._rewrite_test(source, config)
    for name in (part for part in qualname.split(".") if part != "<locals>"):
        code = next(
            item
            for item in code.co_consts
            if type(item) is CodeType and item.co_name == name
        )
    return code


class OriginalChildCreationAdmission:
    tool_name = "tldw-finite-child-creation-admission"

    def __init__(self, config):
        self.config = config
        self.bindings, self.modules, self.codes = [], {}, {}
        self.owners, self.references, self.rows = {}, [], []
        self.callbacks, self.installed = {}, []
        self.lock = threading.Lock()
        self.overflow = 0
        self.tool, self.active, self.restored = None, False, False
        self.monitor = sys.monitoring
        self.mask = self.monitor.events.PY_START | self.monitor.events.PY_RETURN
        self.host_bound = False
        self.host_module = self.host_type = None
        self.primary_prepare_bound = False
        self.primary_rows, self.primary_inputs_bound = [], False
        self.observation_sequence = 0
        self.cancel_rows, self.cancel_pending = [], {}
        self.cancel_sites, self.cancel_sessions = {}, {}
        self.cancel_after_offsets, self.instruction_codes_active = {}, set()
        self.waiting_bindings, self.waiting_aliases = [], {}
        self.waiting_owner_bound = False
        self.stage_sequence, self.stage_overflow, self.stage_io_errors = 0, 0, 0
        self.stage_output = os.environ.get("TLDW_TEST_FLEET_STAGE_RECEIPT")
        definitions = (
            (TESTS, None, "_prepare_surviving_child", "fixture_prepare"),
            (TESTS, None, "_arm_pending_round", "fixture_arm"),
            (
                "tldw_chatbook.Chat.console_chat_controller",
                "ConsoleChatController",
                "_capture_chat_creation_source",
                "capture_source",
            ),
            (
                "tldw_chatbook.Chat.console_chat_controller",
                "ConsoleChatController",
                "_read_chat_creation_source",
                "read_source",
            ),
            (
                "tldw_chatbook.Chat.console_chat_controller",
                "ConsoleChatController",
                "_chat_creation_source_matches",
                "match_source",
            ),
            (
                "tldw_chatbook.Chat.console_chat_controller",
                "ConsoleChatController",
                "_observe_chat_creation_record",
                "observe_record",
            ),
            (
                "tldw_chatbook.Chat.console_chat_controller",
                "ConsoleChatController",
                "_chat_creation_record_locked",
                "record_locked",
            ),
            (
                "tldw_chatbook.Chat.console_chat_controller",
                "ConsoleChatController",
                "request_chat_create_confirm",
                "controller_request",
            ),
            (
                "tldw_chatbook.Chat.console_chat_controller",
                "ConsoleChatController",
                "update_agent_runtime",
                "runtime_refresh",
            ),
            (
                "tldw_chatbook.UI.Screens.chat_screen",
                "ChatScreen",
                "_sync_console_chat_core_state",
                "screen_core_sync",
            ),
        )
        definitions += (
            (
                TESTS,
                None,
                "_verify_background_pending_close_names_consequences_and_cancels_only_its_owner",
                "fixture_pending_close",
            ),
            (
                "tldw_chatbook.Chat.console_chat_controller",
                "ConsoleChatController",
                "_resume_provider_continuation",
                "cancel_mutator_recovery",
            ),
            (
                "tldw_chatbook.Chat.console_chat_controller",
                "ConsoleChatController",
                "shutdown",
                "cancel_mutator_shutdown",
            ),
            (
                "tldw_chatbook.Chat.console_chat_controller",
                "ConsoleChatController",
                "_stream_assistant_response_inner",
                "cancel_mutator_whole_run",
            ),
            (
                "tldw_chatbook.Chat.console_chat_controller",
                "ConsoleChatController",
                "_run_direct_provider_reply",
                "cancel_mutator_direct_stream",
            ),
        )
        definitions += tuple(
            (TESTS, None, name, "stage_" + name)
            for name in (
                "_verify_background_pending_close_releases_round_without_an_active_turn",
                "_verify_chat_create_enrichment_cannot_arm_after_its_session_closes",
                "_verify_all_close_consequences_keep_named_title_and_actions_painted_at_80x24",
                "_verify_progress_close_failure_reconciles_fleet_before_confirmed_retry",
                "_settle",
            )
        )
        for definition in definitions:
            self._bind_definition(definition)
        self._prepare_primary_branches()
        self._prepare_cancel_mutators()

    def _stamp(self):
        ordinal = self.observation_sequence
        self.observation_sequence += 1
        return {
            "observer_ordinal": ordinal,
            "observer_monotonic_ns": time.monotonic_ns(),
        }

    @staticmethod
    def _after_pop_offsets(code, line):
        instructions = list(dis.get_instructions(code))
        calls = [
            index
            for index, instruction in enumerate(instructions)
            if instruction.opname == "CALL" and instruction.positions.lineno == line
        ]
        assert calls
        assert all(instructions[index + 1].opname == "POP_TOP" for index in calls)
        # CPython emits separate normal/exception copies of one finally body.
        return frozenset(instructions[index + 1].offset for index in calls)

    def _prepare_cancel_mutators(self):
        module = sys.modules["tldw_chatbook.Chat.console_chat_controller"]
        tree = ast.parse(Path(module.__file__).read_bytes())
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ConsoleChatController"
        )
        for code, label in tuple(self.codes.items()):
            if not label.startswith("cancel_mutator_"):
                continue
            body = next(
                node
                for node in cls.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == code.co_name
            )
            calls = [
                node
                for node in ast.walk(body)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "pop"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "_active_cancel_events"
            ]
            assert len(calls) == 1 and isinstance(calls[0].args[0], ast.Name)
            self.cancel_sites[code] = (calls[0].lineno, calls[0].args[0].id)
            self.cancel_after_offsets[code] = self._after_pop_offsets(
                code, calls[0].lineno
            )
        assert len(self.cancel_sites) == 4
        module = sys.modules[TESTS]
        outer = inspect.getattr_static(
            module,
            "_verify_background_pending_close_names_consequences_and_cancels_only_its_owner",
        )
        outer_tree = ast.parse(Path(module.__file__).read_bytes())
        body = next(
            node
            for node in outer_tree.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == outer.__name__
        )
        waiting = next(
            node
            for node in ast.walk(body)
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "waiting_run"
        )
        waiting_code = next(
            code
            for code in outer.__code__.co_consts
            if type(code) is CodeType and code.co_name == "waiting_run"
        )
        assert _shape(waiting_code) == _shape(
            _source_code(module, waiting_code.co_qualname, config=self.config)
        )
        pop = next(
            node
            for node in ast.walk(waiting)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "pop"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_active_cancel_events"
        )
        assert isinstance(pop.args[0], ast.Attribute) and pop.args[0].attr == "id"
        assert (
            isinstance(pop.args[0].value, ast.Name) and pop.args[0].value.id == "doomed"
        )
        created = [
            node
            for node in ast.walk(body)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "run_task"
                for target in node.targets
            )
        ]
        assert len(created) == 1
        self.waiting_outer_code = outer.__code__
        self.waiting_bind_line = created[0].lineno
        self.waiting_code = waiting_code
        self.cancel_sites[waiting_code] = (pop.lineno, "doomed.id")
        self.cancel_after_offsets[waiting_code] = self._after_pop_offsets(
            waiting_code, pop.lineno
        )
        self.codes[waiting_code] = "fixture_waiting_run"

    def _code_mask(self, code):
        mask = self.primary_mask if code is self.match_code else self.mask
        if code in self.cancel_sites or code is self.waiting_outer_code:
            mask |= self.monitor.events.LINE
        if code in self.instruction_codes_active:
            mask |= self.monitor.events.INSTRUCTION
        return mask

    def _waiting_current(self):
        return all(
            frame.f_globals is namespace
            and frame.f_code is self.waiting_outer_code
            and function.__code__ is code
            and code is self.waiting_code
            and function.__globals__ is namespace
            and function.__closure__ is None
            and function.__defaults__ is defaults
            and function.__kwdefaults__ is kwdefaults
            for frame, function, code, namespace, defaults, kwdefaults in self.waiting_bindings
        ) and all(
            frame.f_locals.get("waiting_run") is function
            for frame, function in self.waiting_aliases.values()
        )

    def _bind_waiting_run(self, frame):
        controller = frame.f_locals.get("controller")
        doomed = frame.f_locals.get("doomed")
        assert frame.f_code is self.waiting_outer_code
        function = frame.f_locals["waiting_run"]
        assert type(function) is FunctionType and function.__code__ is self.waiting_code
        assert function.__globals__ is frame.f_globals and function.__closure__ is None
        assert (
            function.__defaults__[0] is controller
            and function.__defaults__[1] is doomed
        )
        self.waiting_aliases[id(frame)] = (frame, function)
        if (
            self.owners.get(id(controller)) is controller
            and doomed is not None
            and doomed.id in self.cancel_sessions.get(id(controller), ())
        ):
            self.waiting_owner_bound = True
        if any(row[1] is function for row in self.waiting_bindings):
            return
        assert len(self.waiting_bindings) < 16
        self.references.extend((frame, function, doomed))
        self.waiting_bindings.append(
            (
                frame,
                function,
                function.__code__,
                function.__globals__,
                function.__defaults__,
                function.__kwdefaults__,
            )
        )

    def _cancel_row(self, phase, frame, controller, session_id, *, exception=None):
        if len(self.cancel_rows) >= 64:
            self.overflow += 1
            return
        event = controller._active_cancel_events.get(session_id)
        task = controller._active_stream_tasks.get(session_id)
        self.references.extend(
            (frame, controller, event, task, threading.current_thread())
        )
        self.cancel_rows.append(
            {
                **self._stamp(),
                "phase": phase,
                "source_kind": self.codes[frame.f_code],
                "line": frame.f_lineno,
                "frame_object": id(frame),
                "controller_object": id(controller),
                "session_id": session_id,
                "thread_object": id(threading.current_thread()),
                "cancel_present": event is not None,
                "cancel_object": None if event is None else id(event),
                "active_task_object": None if task is None else id(task),
                "assistant_message_id": controller._active_assistant_message_ids.get(
                    session_id
                ),
                "exception_type": None
                if exception is None
                else type(exception).__name__,
            }
        )

    def _flush_cancel_pop(self, frame, *, exception=None):
        pending = self.cancel_pending.pop(id(frame), None)
        if pending is None:
            return
        original_frame, controller, session_id = pending
        assert original_frame is frame
        self._cancel_row(
            "after_original_pop", frame, controller, session_id, exception=exception
        )

    def _line(self, code, line):
        if not self.active:
            return
        frame = sys._getframe(1)
        assert frame.f_code is code
        with self.lock:
            if code is self.waiting_outer_code:
                if line == self.waiting_bind_line:
                    self._bind_waiting_run(frame)
                return
            if code not in self.cancel_sites:
                return
            controller = self._controller(self.codes[code], frame)
            if controller is None:
                return
            pop_line, key = self.cancel_sites[code]
            if line != pop_line:
                return
            if key == "doomed.id":
                assert self.waiting_owner_bound and self._waiting_current()
                session_id = frame.f_locals["doomed"].id
            else:
                session_id = frame.f_locals[key]
            if session_id not in self.cancel_sessions.get(id(controller), ()):
                return
            self._cancel_row("before_original_pop", frame, controller, session_id)
            assert id(frame) not in self.cancel_pending
            self.cancel_pending[id(frame)] = (frame, controller, session_id)
            if code not in self.instruction_codes_active:
                assert self.monitor.get_local_events(
                    self.tool, code
                ) == self._code_mask(code)
                self.instruction_codes_active.add(code)
                self.monitor.set_local_events(self.tool, code, self._code_mask(code))

    def _instruction(self, code, offset):
        if not self.active:
            return
        frame = sys._getframe(1)
        assert frame.f_code is code
        with self.lock:
            if id(
                frame
            ) not in self.cancel_pending or offset not in self.cancel_after_offsets.get(
                code, ()
            ):
                return
            self._flush_cancel_pop(frame)
            if not any(row[0].f_code is code for row in self.cancel_pending.values()):
                self.instruction_codes_active.remove(code)
                self.monitor.set_local_events(self.tool, code, self._code_mask(code))

    def _prepare_primary_branches(self):
        module = sys.modules["tldw_chatbook.Chat.console_chat_controller"]
        function = inspect.getattr_static(
            module.ConsoleChatController, "_chat_creation_source_matches"
        )
        self.match_code = function.__code__
        tree = ast.parse(Path(module.__file__).read_bytes())
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ConsoleChatController"
        )
        body = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_chat_creation_source_matches"
        )
        returns = [
            node
            for node in ast.walk(body)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "bool"
            and len(node.value.args) == 1
            and isinstance(node.value.args[0], ast.BoolOp)
            and isinstance(node.value.args[0].op, ast.And)
        ]
        primary = max(returns, key=lambda node: node.lineno)
        assert len(primary.value.args[0].values) == 6
        instructions = list(dis.get_instructions(self.match_code))
        branches = [
            (index, instruction)
            for index, instruction in enumerate(instructions)
            if instruction.opname == "POP_JUMP_IF_FALSE"
            and instruction.positions.lineno is not None
            and primary.lineno <= instruction.positions.lineno <= primary.end_lineno
        ]
        labels = (
            "cancel_present",
            "cancel_not_set",
            "assistant_message_equal",
            "live_primary_run_equal",
            "row_kind_primary",
        )
        assert len(branches) == len(labels)
        self.primary_branches = {
            instruction.offset: (
                label,
                instruction.argval,
                instructions[index + 1].offset,
            )
            for label, (index, instruction) in zip(labels, branches)
        }
        self.primary_mask = self.mask | self.monitor.events.BRANCH

    def _bind_primary_inputs(self, controller):
        if self.primary_inputs_bound:
            return
        module = sys.modules.get("tldw_chatbook.Chat.console_agent_bridge")
        if module is None:
            return  # Never force a stock bridge into a custom fixture route.
        bridge_type = inspect.getattr_static(module, "ConsoleAgentBridge")
        if type(controller._agent_bridge) is not bridge_type:
            return
        self._bind_definition(("threading", "Event", "is_set", "evaluated_cancel"))
        self._bind_definition(
            (
                module.__name__,
                "ConsoleAgentBridge",
                "live_primary_run_id",
                "evaluated_live_primary",
            )
        )
        self.primary_inputs_bound = True

    def _primary_row(self, frame, controller, condition, **values):
        if len(self.primary_rows) >= 128:
            self.overflow += 1
            return
        session = frame.f_locals.get("session")
        self.references.extend((frame, controller, threading.current_thread()))
        self.primary_rows.append(
            {
                **self._stamp(),
                "sequence": len(self.primary_rows),
                "match_frame_object": id(frame),
                "controller_object": id(controller),
                "thread_object": id(threading.current_thread()),
                "session_id": None if session is None else session.id,
                "condition": condition,
                **values,
            }
        )

    def _branch(self, code, offset, destination):
        if (
            not self.active
            or code is not self.match_code
            or offset not in self.primary_branches
        ):
            return
        frame = sys._getframe(1)
        assert frame.f_code is code
        controller = frame.f_locals["self"]
        if self.owners.get(id(controller)) is not controller:
            return
        label, false_target, true_target = self.primary_branches[offset]
        assert destination in (false_target, true_target)
        with self.lock:
            self._primary_row(
                frame,
                controller,
                label,
                kind="actual_evaluated_branch",
                predicate_passed=destination == true_target,
                instruction_offset=offset,
                destination_offset=destination,
            )

    def _evaluated_primary_return(self, label, frame, value):
        parent = frame.f_back
        if parent is None or parent.f_code is not self.match_code:
            return
        controller = parent.f_locals["self"]
        if self.owners.get(id(controller)) is not controller:
            return
        if label == "evaluated_cancel":
            assert frame.f_locals["self"] is parent.f_locals["cancel"]
            assert type(value) is bool  # noqa: E721 - require the exact declared source owner
        else:
            assert frame.f_locals["self"] is parent.f_locals["bridge"]
            assert value is None or type(value) is str  # noqa: E721 - require the exact declared source owner
        self._primary_row(
            parent,
            controller,
            label,
            kind="actual_original_operand_return",
            value=value,
            expected_run_id=parent.f_locals["run_id"],
        )

    def _bind_definition(self, definition):
        module_name, class_name, name, label = definition
        module = sys.modules[module_name]
        owner = (
            module if class_name is None else inspect.getattr_static(module, class_name)
        )
        function = inspect.getattr_static(owner, name)
        assert type(function) is FunctionType
        assert function.__globals__ is module.__dict__
        assert function.__closure__ is None
        rewritten = getattr(module.__spec__.loader, "_rewritten_names", {})
        config_for_source = self.config if module_name in rewritten else None
        assert _shape(function.__code__) == _shape(
            _source_code(module, function.__qualname__, config=config_for_source)
        ), "selected installed original differs from exact defining source"
        path = Path(module.__file__)
        self.modules[module_name] = (
            module,
            path,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        self.bindings.append(
            (
                owner,
                name,
                function,
                function.__code__,
                function.__globals__,
                function.__defaults__,
                function.__kwdefaults__,
            )
        )
        self.codes[function.__code__] = label
        if self.active:
            code = function.__code__
            assert self.monitor.get_local_events(self.tool, code) == 0
            self.monitor.set_local_events(self.tool, code, self._code_mask(code))
            self.installed.append(code)

    def _bind_host(self):
        if self.host_bound:
            return
        # The original fixture has already constructed its controller/App.
        # Never import the lazy host just to make this observer ready.
        name = "tldw_chatbook.Chat.console_interrupt_rounds"
        module = sys.modules.get(name)
        assert (
            module is not None
        ), "required original host not loaded at fixture prepare"
        self._bind_definition(
            (name, "InterruptRoundHost", "request_chat_create_confirm", "host_request")
        )
        self.host_module = module
        self.host_type = inspect.getattr_static(module, "InterruptRoundHost")
        self.host_bound = True

    def _bind_primary_prepare_ancestor(self, frame):
        if self.primary_prepare_bound:
            return
        name = "Tests.Chat.test_console_chat_create_integration"
        module = sys.modules.get(name)
        if module is None:
            return  # The surviving-child route does not use this helper.
        original = inspect.getattr_static(module, "_prepare_close_new_chat")
        caller = frame.f_back
        for _ in range(12):
            if caller is None:
                return
            if caller.f_code is original.__code__:
                self._bind_definition(
                    (name, None, "_prepare_close_new_chat", "fixture_primary_prepare")
                )
                assert caller.f_globals is original.__globals__
                self.primary_prepare_bound = True
                return
            caller = caller.f_back

    def _current(self):
        host_alias_current = (
            not self.host_bound
            or inspect.getattr_static(self.host_module, "InterruptRoundHost")
            is self.host_type
        )
        return (
            host_alias_current
            and self._waiting_current()
            and all(
                inspect.getattr_static(owner, name) is function
                and function.__code__ is code
                and function.__globals__ is namespace
                and function.__defaults__ is defaults
                and function.__kwdefaults__ is kwdefaults
                and function.__closure__ is None
                for owner, name, function, code, namespace, defaults, kwdefaults in self.bindings
            )
            and all(
                sys.modules.get(name) is module
                and hashlib.sha256(path.read_bytes()).hexdigest() == digest
                for name, (module, path, digest) in self.modules.items()
            )
        )

    def _controller(self, label, frame):
        local = frame.f_locals
        if label in {"fixture_prepare", "fixture_arm"}:
            if label == "fixture_arm" and local["kind"] != "chat_create":
                return None
            controller = local["controller"]
            if id(controller) not in self.owners:
                if len(self.owners) >= 16:
                    self.overflow += 1
                    return None
                self.owners[id(controller)] = controller
                self.references.append(controller)
            if label == "fixture_arm":
                self.cancel_sessions.setdefault(id(controller), set()).add(
                    local["session_id"]
                )
            return controller
        if label == "fixture_waiting_run":
            controller = local.get("controller")
            return controller if self.owners.get(id(controller)) is controller else None
        if label == "fixture_primary_prepare":
            controller = local["controller"]
            return controller if self.owners.get(id(controller)) is controller else None
        if label == "host_request":
            host = local.get("self")
            assert self.host_bound and type(host) is self.host_type
            return next(
                (
                    owner
                    for owner in self.owners.values()
                    if owner._interrupt_host is host
                ),
                None,
            )
        if label == "screen_core_sync":
            screen = local.get("self")
            controller = getattr(screen, "_console_chat_controller", None)
        else:
            controller = local.get("self")
        return controller if self.owners.get(id(controller)) is controller else None

    @staticmethod
    def _actor(actor):
        if actor is None:
            return None
        return {key: getattr(actor, key) for key in ("kind", "run_id", "parent_run_id")}

    def _event(self, label, phase, frame, controller, value=None):
        if len(self.rows) >= 512:
            self.overflow += 1
            return
        local = frame.f_locals
        bridge = controller._agent_bridge
        runtime = getattr(controller.app, "console_runtime", None)
        runtime_bridge = getattr(runtime, "_agent_bridge", None)
        payload = local.get("payload")
        if label == "fixture_prepare":
            payload = local.get("prepared")
        observation = local.get("observation")
        if payload is None and observation is not None:
            payload = observation.payload
        session_id = (
            payload.get("session_id")
            if type(payload) is dict  # noqa: E721 - accept only builtin captured payloads
            else local.get("session_id")
        )  # noqa: E721 - require the exact declared source owner
        session = controller.store._sessions.get(session_id)
        self.references.extend(
            (
                frame,
                threading.current_thread(),
                bridge,
                runtime,
                runtime_bridge,
                observation,
                session,
            )
        )
        row = {
            **self._stamp(),
            "sequence": len(self.rows),
            "kind": label,
            "phase": phase,
            "line": frame.f_lineno,
            "thread_object": id(threading.current_thread()),
            "controller_object": id(controller),
            "session_id": session_id,
            "controller_bridge_object": id(bridge) if bridge is not None else None,
            "runtime_bridge_object": id(runtime_bridge)
            if runtime_bridge is not None
            else None,
            "controller_and_runtime_bridge_same": bridge is runtime_bridge,
            "controller_runs_db_object": id(getattr(bridge, "runs_db", None)),
            "session_open_in_store": session is not None,
            "session_close_generation_present": session_id
            in controller._session_close_generations,
            "disposed": controller._disposed,
        }
        if label == "match_source":
            row["match_frame_object"] = id(frame)
        if label == "host_request":
            row["ui_app_is_none"] = controller.app is None
            row["ui_callback_is_none"] = controller.set_pending_chat_create is None
            row["actual_enriched_payload_local_present"] = "enriched_payload" in local
        if "refused" in local:
            row["actual_refused_local"] = local["refused"]
        if "record" in local:
            row["actual_record_local_is_none"] = local["record"] is None
        if "requesting_kind" in local:
            row["actual_requesting_kind"] = local["requesting_kind"]
        if observation is not None:
            row["observation"] = {
                "source_same_current_session": observation.source is session,
                "bridge_same_current": observation.bridge is bridge,
                "runs_db_same_current": observation.runs_db
                is getattr(bridge, "runs_db", None),
                "row_present": observation.row is not None,
                "parent_present": observation.parent is not None,
                "actor": self._actor(observation.actor),
                "row": {
                    key: observation.row.get(key)
                    for key in ("agent_kind", "parent_run_id", "status")
                }
                if type(observation.row) is dict  # noqa: E721 - accept only builtin captured rows
                else None,  # noqa: E721 - require the exact declared source owner
                "row_conversation_matches": bool(
                    observation.row
                    and session
                    and observation.row.get("conversation_id")
                    == session.persisted_conversation_id
                ),
                "record_present": observation.record is not None,
            }
        if "actor" in local:
            row["actual_actor_local"] = self._actor(local["actor"])
        if label == "runtime_refresh":
            row["refresh_argument_bridge_object"] = id(local["bridge"])
        if phase == "return":
            row["return"] = (
                value
                if value is None or type(value) is bool  # noqa: E721 - capture only builtin decision booleans
                else type(value).__name__
            )  # noqa: E721 - require the exact declared source owner
            if label in {"host_request", "controller_request"} and type(value) is dict:  # noqa: E721 - require the exact declared source owner
                row["actual_decision"] = {
                    key: value.get(key) for key in ("allow", "remember")
                }
            if label in {"capture_source", "read_source", "observe_record"}:
                row["return_observation_is_none"] = value is None
                if value is not None:
                    row["returned_row_present"] = value.row is not None
                    row["returned_parent_present"] = value.parent is not None
                    row["returned_actor"] = self._actor(value.actor)
        self.rows.append(row)

    def _checkpoint(self, label, phase, frame, value=None):
        if not self.stage_output:
            return
        if self.stage_sequence >= 256:
            self.stage_overflow += 1
            if self.stage_overflow > 1:
                return
        else:
            self.stage_sequence += 1
        kind = frame.f_locals.get("kind")
        if type(kind) is not str or kind not in {  # noqa: E721 - exact built-in diagnostic scalar only.
            "approval",
            "question",
            "chat_create",
            "worktree_merge",
            "skill_install",
            "skill_script",
        }:
            kind = None
        matches = [
            namespace
            for _, _, function, code, namespace, _, _ in self.bindings
            if code is frame.f_code
        ]
        row = {
            "diagnostic_only": True,
            "terminal_source_native_coverage": False,
            "sequence": self.stage_sequence,
            "monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "stage": label,
            "phase": phase,
            "kind": kind,
            "line": frame.f_lineno,
            "code_object": id(frame.f_code),
            "globals_match_bound": bool(matches)
            and all(frame.f_globals is namespace for namespace in matches),
            "global_events": self.monitor.get_events(self.tool),
            "actual_bool_return": value if type(value) is bool else None,  # noqa: E721 - exact built-in diagnostic scalar only.
            "source_hashes": {
                name: digest for name, (_, _, digest) in self.modules.items()
            },
            "overflow": self.stage_overflow,
            "io_errors": self.stage_io_errors,
        }
        target = Path(self.stage_output)
        temporary = target.with_name(target.name + "." + str(os.getpid()) + ".tmp")
        try:
            temporary.write_text(json.dumps(row), encoding="utf-8")
            temporary.replace(target)
        except OSError:
            self.stage_io_errors += 1

    def _start(self, code, offset):
        if not self.active:
            return
        frame = sys._getframe(1)
        assert frame.f_code is code
        with self.lock:
            label = self.codes[code]
            self._checkpoint(label, "start", frame)
            if label.startswith("stage_"):
                return
            if label == "fixture_prepare" or (
                label == "fixture_arm" and frame.f_locals["kind"] == "chat_create"
            ):
                self._bind_host()
            if label == "capture_source":
                owner = frame.f_locals.get("self")
                if self.owners.get(id(owner)) is owner:
                    self._bind_primary_prepare_ancestor(frame)
            controller = self._controller(label, frame)
            if controller is not None:
                if label == "match_source":
                    self._bind_primary_inputs(controller)
                self._event(label, "start", frame, controller)

    def _return(self, code, offset, value):
        if not self.active:
            return
        frame = sys._getframe(1)
        assert frame.f_code is code
        with self.lock:
            self._flush_cancel_pop(frame)
            label = self.codes[code]
            self._checkpoint(label, "return", frame, value)
            if label.startswith("stage_"):
                return
            if label in {"evaluated_cancel", "evaluated_live_primary"}:
                self._evaluated_primary_return(label, frame, value)
                return
            controller = self._controller(label, frame)
            if controller is not None:
                self._event(label, "return", frame, controller, value)

    def start(self):
        assert self._current()
        self.tool = next(
            slot
            for slot in range(5, 0, -1)
            if slot != self.monitor.DEBUGGER_ID and self.monitor.get_tool(slot) is None
        )
        self.monitor.use_tool_id(self.tool, self.tool_name)
        assert self.monitor.get_events(self.tool) == 0
        try:
            for event, callback in (
                (self.monitor.events.PY_START, self._start),
                (self.monitor.events.PY_RETURN, self._return),
                (self.monitor.events.BRANCH, self._branch),
                (self.monitor.events.LINE, self._line),
                (self.monitor.events.INSTRUCTION, self._instruction),
            ):
                assert (
                    self.monitor.register_callback(self.tool, event, callback) is None
                )
                self.callbacks[event] = callback
            self.active = True
            for code in self.codes:
                assert self.monitor.get_local_events(self.tool, code) == 0
                mask = self._code_mask(code)
                self.monitor.set_local_events(self.tool, code, mask)
                self.installed.append(code)
        except BaseException:
            self.stop()
            raise

    def stop(self):
        assert self.monitor.get_events(self.tool) == 0
        for code in self.installed:
            mask = self._code_mask(code)
            assert self.monitor.get_local_events(self.tool, code) == mask
            self.monitor.set_local_events(self.tool, code, 0)
        for event, callback in self.callbacks.items():
            assert self.monitor.register_callback(self.tool, event, None) is callback
        self.active = False
        self.monitor.free_tool_id(self.tool)
        self.restored = self.monitor.get_tool(self.tool) is None

    def receipt(self):
        return {
            "diagnostic_only": True,
            "original_defining_sources_and_bodies_unchanged": self.host_bound
            and self._current(),
            "required_host_source_coverage_complete": self.host_bound,
            "ordinary_primary_prepare_source_ancestry_bound": self.primary_prepare_bound,
            "original_primary_operand_sources_bound": self.primary_inputs_bound,
            "actual_primary_guard_evaluations": self.primary_rows,
            "actual_original_cancel_pop_stages": self.cancel_rows,
            "required_nested_waiting_run_source_bound": self.waiting_owner_bound,
            "nested_waiting_run_exact_code_defaults_and_globals_current": self._waiting_current(),
            "pending_cancel_pop_intervals": len(self.cancel_pending),
            "remaining_instruction_intervals": len(self.instruction_codes_active),
            "pytest_assertion_rewrite_source_qualified": True,
            "selected_sync_code_count": len(self.codes),
            "global_events": 0,
            "local_hooks_removed_tool_freed": self.restored,
            "bounded_owners": len(self.owners),
            "overflow": self.overflow,
            "partial_stage_writes": self.stage_sequence,
            "partial_stage_overflow": self.stage_overflow,
            "partial_stage_io_errors": self.stage_io_errors,
            "source_hashes": {
                name: digest for name, (_, _, digest) in self.modules.items()
            },
            "events": self.rows,
            "limits": "Original START/RETURN and exact selected cancel-pop LINE with INSTRUCTION enabled only until the actual following POP_TOP, plus the existing exact primary expression BRANCH. Shared observer ordinal and monotonic ns order original effects across arrays. Fixture nested callable binds at its original create_task LINE with exact rewritten source/code/globals/default owner identities. No Event state or authority predicate is called by mutation observation. Operand values are actual original Event.is_set/live_primary_run_id returns; comparisons are actual evaluated bytecode branches. No extra authority/DB reads or predicate reevaluation. Missing operand return is incomplete evidence. "
            "No added storage or actor reads, guards/callables/context/waits/deadlines changed. "
            "Metadata only; diagnostics add timing overhead and do not establish performance.",
        }


@pytest.fixture(autouse=True)
def original_child_creation_admission(request):
    from Tests.private_profile import is_private_profile_child

    if (
        os.environ.get("TLDW_TEST_CHILD_CREATION_ADMISSION") != "1"
        or request.module.__name__ != TESTS
        or request.node.name != TARGET
        or not is_private_profile_child(request)
    ):
        yield
        return
    observer = OriginalChildCreationAdmission(request.config)
    output = Path(
        os.environ.get("TLDW_TEST_CHILD_CREATION_RECEIPT")
        or str(
            Path(os.environ["TLDW_TEST_CONFIG_ROOT"]).parent
            / "child-creation-admission.json"
        )
    )
    try:
        observer.start()
        yield
    finally:
        if observer.active:
            observer.stop()
        output.write_text(json.dumps(observer.receipt(), indent=2), encoding="utf-8")
        assert (
            observer.host_bound
        ), "required original InterruptRoundHost source never bound"
        assert (
            observer.waiting_owner_bound
        ), "required original owned waiting_run callable never bound"
