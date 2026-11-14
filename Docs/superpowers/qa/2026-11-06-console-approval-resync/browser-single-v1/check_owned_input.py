"""Actual child AST-owned input delegate controls; no Textual/product imports."""

import ast, sys, types
from pathlib import Path

HERE = Path(__file__).resolve().parent


def control_output():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    if not output.is_relative_to(source / ".superpowers"):
        raise RuntimeError(
            "Control_outputs_must_stay_inside_selectedsource_superpowers"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


CONTROL_OUTPUT = control_output()
module = ast.parse((HERE / "web_product_child.py").read_text(encoding="utf-8"))
node = next(
    n
    for n in module.body
    if isinstance(n, ast.FunctionDef) and n.name == "build_driver"
)
app = object()
sentinel = object()
failure = OSError(22)


class Message:
    pass


class Driver:
    def __init__(self):
        self._app = app
        self.calls = []
        self.error = False

    def process_message(self, message):
        self.calls.append(message)
        if self.error:
            raise failure
        return sentinel


original = Driver.process_message


class Fixture:
    current = {"correlation": "a" * 32}

    def __init__(self):
        self.rows = []
        self.broken = False

    def _record(self, *args, **kw):
        if self.broken:
            raise RuntimeError("unit-observer-failure")
        self.rows.append((args, kw))


class Sink:
    def __init__(self):
        self.failures = 0
        self.broken = False

    def fail(self):
        self.failures += 1
        if self.broken:
            raise RuntimeError("unit-health-failure")


fixture = Fixture()
sink = Sink()
actual = Driver()
seen = []


def build(*args, **kw):
    seen.append((args, kw))
    return actual


scope = {
    "original_build_driver": build,
    "app": app,
    "driver": fixture,
    "sink": sink,
    "WRITE_OBSERVER": None,
    "ORIGIN_OBSERVER": None,
    "WebDriver": Driver,
    "events": types.SimpleNamespace(
        MouseDown=Message, MouseUp=type("Other", (), {}), Key=type("Key", (), {})
    ),
    "time": types.SimpleNamespace(perf_counter_ns=lambda: 1),
}
exec(
    compile(
        ast.Module(body=[node], type_ignores=[]),
        str(HERE / "web_product_child.py"),
        "exec",
    ),
    scope,
)
assert scope["build_driver"](sentinel, code=sentinel) is actual
assert (
    seen == [((sentinel,), {"code": sentinel})] and Driver.process_message is original
)
message = Message()
assert actual.process_message(message) is sentinel and actual.calls == [message]
assert fixture.rows[0][0] == ("app_input_prequeue", "a" * 32)
fixture.broken = True
sink.broken = True
assert (
    actual.process_message(message) is sentinel
    and actual.calls == [message, message]
    and sink.failures == 1
)
actual.error = True
try:
    actual.process_message(message)
except OSError as error:
    assert error is failure
else:
    raise AssertionError("Original_error_not_preserved")
assert (
    actual.calls == [message, message, message] and Driver.process_message is original
)
print(
    "PASS exact_owned_instance_original_once_args_return_error_identity_observer_failure_isolated; App/product imports0"
)
