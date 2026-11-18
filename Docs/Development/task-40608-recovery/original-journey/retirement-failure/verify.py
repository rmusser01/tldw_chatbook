"""Short stdlib-only diagnostic test using compiled ORIGINAL warning methods."""

import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import sqlite3
from types import FunctionType, ModuleType, SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from observer import SITES, checked_site, codes, failure_facts, trace_location  # noqa: E402


class Logger:
    def add(self, sink, **_kwargs):
        self.sink = sink
        return 1

    def remove(self, _token):
        self.sink = None

    def warning(self, *_args):
        frame = sys._getframe(1)
        self.sink(SimpleNamespace(record={"function": frame.f_code.co_name, "line": frame.f_lineno}))


def main():
    root = Path(sys.argv[1]).resolve(strict=True)
    pins = json.loads((HERE / "pins.json").read_bytes())
    logger = Logger()
    module = ModuleType("loguru")
    module.logger = logger
    sys.modules["loguru"] = module
    original = {}
    for name, (relative, owner, _) in SITES.items():
        data, _ = checked_site(root, pins["sources"], name)
        compiled = compile(data, str(root / relative), "exec", dont_inherit=True)
        code = next(code for code in codes(compiled) if code.co_qualname == f"{owner}.{name}")
        cls = type(owner, (), {})
        globals_ = {"__builtins__": __builtins__, owner: cls, "_definitions_current": lambda: False}
        function = FunctionType(code, globals_)
        setattr(cls, name, staticmethod(function) if name == "_failure" else function)
        original[name] = cls, function

    async def exercise():
        cls, _ = original["_settle"]
        owner = cls()
        owner.trees, owner.reads, owner.source = [], [], None
        owner.retired = asyncio.get_running_loop().create_future()
        await owner._settle()
        assert owner.retired.result() is False

    with tempfile.TemporaryDirectory(dir=HERE) as temporary:
        os.environ["TLDW_TEST_LAB_OUTPUT"] = temporary
        with failure_facts(root, {"retirement_failure_diagnostic": True, "sources": pins["sources"], "failure_trace_sources": pins["trace_sources"]}):
            cls, method = original["_failure"]
            try:
                raise RuntimeError("PRIVATE SENTINEL MUST NOT ESCAPE")
            except RuntimeError as error:
                cls._failure("ReceiveTurn", "terminal", error)
            asyncio.run(exercise())
            path = Path(temporary) / f"retirement-failure-{os.getpid()}.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert [row["event"] for row in rows] == ["installed", "backend_failure", "retirement_unconfirmed"], rows
            assert rows[1]["operation"] == "ReceiveTurn" and rows[1]["error_type"] == "RuntimeError"
            assert rows[2]["last_check"] == ["definitions", False]
            assert rows[2]["effect_facts"] is None and rows[2]["reads"] == rows[2]["trees"] == 0
            assert "PRIVATE" not in path.read_text() and str(root) not in path.read_text()
            assert rows[1]["locations"] == ["other"]
            # A real original code frame is identified; forging its filename or
            # an arbitrarily long code name cannot manufacture that provenance.
            try:
                method("ReceiveTurn", "terminal", None)
            except AttributeError as error:
                trace = error.__traceback__.tb_next
                assert trace_location(root, pins["trace_sources"], trace, {}).startswith("_BackendService._failure:")
                wrong_trace_sources = dict(pins["trace_sources"])
                wrong_trace_sources[SITES["_failure"][0]] = "0" * 64
                assert trace_location(root, wrong_trace_sources, trace, {}) == "other"
            def private_identifier():
                raise RuntimeError("PRIVATE")
            forged = FunctionType(private_identifier.__code__.replace(
                co_filename=str(root / SITES["_failure"][0]),
                co_name="PRIVATE_" + "X" * 10000, co_qualname="_BackendService._failure",
            ), globals())
            try:
                forged()
            except RuntimeError as error:
                assert trace_location(root, pins["trace_sources"], error.__traceback__.tb_next, {}) == "other"
                method("ReceiveTurn", "terminal", error)
            assert json.loads(path.read_text().splitlines()[-1])["locations"] == ["other", "other"]
            assert "PRIVATE" not in path.read_text()
            # An authentic-looking original code frame with a changed defining
            # class method is refused; it cannot emit a qualified failure.
            cls._failure = staticmethod(lambda *_args: None)
            method("ReceiveTurn", "terminal", RuntimeError("PRIVATE"))
            assert json.loads(path.read_text().splitlines()[-1])["event"] == "source_refused"
            cls._failure = staticmethod(method)
            for _ in range(40):
                cls._failure("ReceiveTurn", "terminal", RuntimeError("PRIVATE"))
            assert len(path.read_text().splitlines()) == 33  # installation + cap32
            before = path.read_bytes()
            cls._failure("PRIVATE", "terminal", RuntimeError("PRIVATE"))
            assert path.read_bytes() == before
        wrong = dict(pins["sources"])
        wrong[SITES["_failure"][0]] = "0" * 64
        try:
            checked_site(root, wrong, "_failure")
        except RuntimeError:
            pass
        else:
            raise AssertionError("Changed source accepted")
    with tempfile.TemporaryDirectory(dir=HERE) as temporary:
        os.environ["TLDW_TEST_LAB_OUTPUT"] = temporary
        # Exercise the actual original confirmed() code under a small synthetic
        # resource inventory. No application imports or native resource calls.
        names = ["tldw_chatbook", "tldw_chatbook.Backup_Recovery", "tldw_chatbook.MCP",
                 "tldw_chatbook.Backup_Recovery.raw_participants", "tldw_chatbook.Backup_Recovery.storage_admission",
                 "tldw_chatbook.Backup_Recovery.participants", "tldw_chatbook.MCP.local_permission_operation"]
        for name in names:
            module = ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
            if "." in name:
                parent, attr = name.rsplit(".", 1)
                setattr(sys.modules[parent], attr, module)
        storage = sys.modules[names[4]]
        storage._lock = threading.RLock()
        storage._pending_acquisitions, storage._raw_operations, storage._live_leases = set(), set(), set()
        raw = sys.modules[names[3]]
        raw._states = {}
        sys.modules[names[5]].retired_core_connections = set()
        sys.modules[names[6]]._hold_retired_or_shared = lambda *_args: True
        sys.modules[names[6]]._state_resources_retired = lambda *_args: True
        path = root / SITES["observe"][0]
        code_map = {code.co_qualname: code for code in codes(compile(path.read_bytes(), str(path), "exec", dont_inherit=True))}
        cls = type("_ReadResources", (), {})
        turn_cls = type("ConsoleTurnRetirement", (), {})
        global_values = {"__builtins__": __builtins__, "_ReadResources": cls, "sqlite3": sqlite3,
                         "ConsoleTurnRetirement": turn_cls,
                         "_native": threading.local(), "_creator": lambda: (1, 2, 3), "_definitions_current": lambda: True}
        for name in ("__init__", "observe", "require_creator", "confirmed"):
            function = FunctionType(code_map[f"_ReadResources.{name}"], global_values)
            setattr(cls, name, contextmanager(function) if name == "observe" else function)
        turn_cls._settle = FunctionType(code_map["ConsoleTurnRetirement._settle"], {
            "__builtins__": __builtins__, "ConsoleTurnRetirement": turn_cls,
            "_definitions_current": lambda: True,
        })

        async def predicates():
            invocation = turn_cls()
            invocation.reads = []
            future = asyncio.get_running_loop().create_future()
            future.set_result(None)
            read = SimpleNamespace(callback=original["_failure"][1], creator=object(), retired=future,
                                   _producer=None, callback_returned=True)
            observation = cls(invocation, read)
            invocation.reads.append(observation)
            with observation.observe():
                pass
            assert observation.confirmed() is True
            token = object()
            observation.observations["acquisition"].append(token)
            storage._pending_acquisitions.add(token)
            assert observation.confirmed() is False
            storage._pending_acquisitions.clear()
            observation.observations["raw"].append((token, object()))
            raw._states[token] = object()
            assert observation.confirmed() is False
            raw._states.clear()
            connection, lease = object(), SimpleNamespace(_key=None)
            participant = SimpleNamespace(connections={})
            observation.connections.append((SimpleNamespace(_maintenance_participant=participant), connection, participant, lease))
            # Fake connection makes original sqlite descriptor raise TypeError;
            # it must stay unconfirmed, with that original closed=False fact.
            assert observation.confirmed() is False
            observation.connections.clear()
            observation.invalid = True
            assert observation.confirmed() is False
            observation.invalid = False
            read._producer = asyncio.get_running_loop().create_future()
            assert observation.confirmed() is False
            read._producer.set_result(None)
            observation.invalid = True
            invocation.trees, invocation.source = [], None
            invocation.task = future
            invocation.logical_closed, invocation.source_closed = True, True
            invocation.unqualified_read = False
            invocation.retired = asyncio.get_running_loop().create_future()
            await invocation._settle()
            assert invocation.retired.result() is False

        with failure_facts(root, {"retirement_failure_diagnostic": True, "sources": pins["sources"], "failure_trace_sources": pins["trace_sources"]}):
            asyncio.run(predicates())
        path = Path(temporary) / f"retirement-failure-{os.getpid()}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        failures = [row for row in rows if row["event"] == "read_unconfirmed"]
        assert len(failures) == 6, rows
        assert [row["first_false_group"] for row in failures] == ["acquisition", "raw", "connection", None, None, None], failures
        assert failures[2]["last_connection_closed"] is False
        assert failures[3]["read"]["invalid"] is True
        assert failures[4]["read"]["producer"]["done"] is False
        assert all(row["read"]["read_index"] == 0 for row in failures)
        settled = next(row for row in rows if row["event"] == "retirement_unconfirmed")
        assert settled["last_check"] == ["read_0", False]
        assert settled["read"]["read_index"] == 0 and settled["read"]["invalid"] is True
        assert "PRIVATE" not in path.read_text() and str(root) not in path.read_text()
    print("PASS: original warning bodies, immediate records, source replacement/hash refusal, privacy, cap; no product imports/native app")


if __name__ == "__main__":
    main()
