"""Short stdlib-only diagnostic test using compiled ORIGINAL warning methods."""

import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
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
    print("PASS: original warning bodies, immediate records, source replacement/hash refusal, privacy, cap; no product imports/native app")


if __name__ == "__main__":
    main()
