"""Finite actual-AST deadline/span/failure retention controls; no App or fixture server."""

import ast, asyncio, copy, json, sys, types
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
parent = ast.parse((HERE / "web_smoke.py").read_text(encoding="utf-8"))
main = next(
    n for n in parent.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "main"
)
command = next(
    n for n in main.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "command"
)


class Clock:
    def __init__(self, values=None):
        self.values = iter(values or [])
        self.counter = 0

    def monotonic(self):
        return next(self.values)

    def perf_counter_ns(self):
        self.counter += 1
        return self.counter


class ReviewPath:
    def __init__(self, exists=True):
        self.present = exists
        self.reads = 0

    def exists(self):
        return self.present

    def read_text(self, **kw):
        self.reads += 1
        return '{"observed":true}'


async def run():
    calls = []
    sentinel = object()
    failure = OSError(22)
    broken = False

    async def original(kind, **values):
        calls.append((kind, values))
        if broken:
            raise failure
        return sentinel

    scope = {
        "time": Clock(),
        "active_control_record": {"control_spans_ns": []},
        "result": {},
        "command_original": original,
    }
    exec(
        compile(
            ast.Module(body=[command], type_ignores=[]), "actual-span-wrapper", "exec"
        ),
        scope,
    )
    assert await scope["command"]("snapshot", owned=sentinel) is sentinel
    assert calls == [("snapshot", {"owned": sentinel})]
    for _ in range(32):
        assert await scope["command"]("snapshot") is sentinel
    assert (
        len(scope["active_control_record"]["control_spans_ns"]) == 32
        and scope["active_control_record"]["control_span_overflow"] is True
        and len(calls) == 33
    )
    broken = True
    try:
        await scope["command"]("collect")
    except OSError as error:
        assert error is failure
    else:
        raise AssertionError("Original_control_error_replaced")
    assert len(calls) == 34
    broken = False
    scope["active_control_record"] = None
    for _ in range(17):
        assert await scope["command"]("setup") is sentinel
    assert (
        len(scope["result"]["control_spans_ns"]) == 16
        and scope["result"]["control_span_overflow"] is True
    )

    barriers = []
    for node in ast.walk(main):
        if (
            isinstance(node, ast.If)
            and ast.unparse(node.test) == "SELECTION['method_mode'] == 'validation'"
        ):
            for index, statement in enumerate(node.body):
                if isinstance(
                    statement, ast.While
                ) and "review_path.exists" in ast.unparse(statement.test):
                    body = copy.deepcopy(node.body[index : index + 4])
                    assignment = body[2]
                    assert isinstance(assignment, ast.Assign)
                    name = assignment.targets[0].id
                    fn = ast.AsyncFunctionDef(
                        name="read_actual_barrier",
                        args=ast.arguments(
                            posonlyargs=[],
                            args=[ast.arg(arg="review_path"), ast.arg(arg="deadline")],
                            kwonlyargs=[],
                            kw_defaults=[],
                            defaults=[],
                        ),
                        body=body
                        + [ast.Return(value=ast.Name(id=name, ctx=ast.Load()))],
                        decorator_list=[],
                    )
                    barriers.append(
                        ast.fix_missing_locations(
                            ast.Module(body=[fn], type_ignores=[])
                        )
                    )
    assert len(barriers) == 2
    for source in barriers:
        for values in ([11], [9, 11]):
            path = ReviewPath()
            env = {"time": Clock(values), "asyncio": asyncio, "json": json}
            exec(compile(source, "actual45s-barrier", "exec"), env)
            try:
                await env["read_actual_barrier"](path, 10)
            except TimeoutError:
                pass
            else:
                raise AssertionError("Late_actual_review_released")
        path = ReviewPath()
        env = {"time": Clock([9, 9]), "asyncio": asyncio, "json": json}
        exec(compile(source, "actual45s-barrier", "exec"), env)
        assert (
            await env["read_actual_barrier"](path, 10) == {"observed": True}
            and path.reads == 1
        )

    fences = [
        n
        for n in ast.walk(main)
        if isinstance(n, ast.If)
        and "observed - started >= 300" in ast.unparse(n.test)
        and "observed - progress >= 60" in ast.unparse(n.test)
    ]
    assert len(fences) >= 2
    for fence in fences:
        code = compile(
            ast.Module(body=[fence], type_ignores=[]), "actual-preflight-fence", "exec"
        )
        for values in (
            {"observed": 60, "started": 0, "progress": 0},
            {"observed": 300, "started": 0, "progress": 299},
        ):
            try:
                exec(code, values)
            except TimeoutError:
                pass
            else:
                raise AssertionError("Expired_phase_or_total_released")
        exec(code, {"observed": 59, "started": 0, "progress": 0})

    raw = next(
        n
        for n in ast.walk(main)
        if isinstance(n, ast.If)
        and ast.unparse(n.test).startswith(
            "capture is not None and capture.complete is not None"
        )
    )
    rawfn = ast.AsyncFunctionDef(
        name="retain_failure",
        args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=[copy.deepcopy(raw)],
        decorator_list=[],
    )
    code = compile(
        ast.fix_missing_locations(ast.Module(body=[rawfn], type_ignores=[])),
        "actual-failed-trace-retainer",
        "exec",
    )

    class CDP:
        def __init__(self):
            self.calls = []

        async def send(self, name):
            self.calls.append(name)
            raise OSError(22)

    cdp = CDP()
    capture = types.SimpleNamespace(
        cdp=cdp,
        complete=asyncio.get_running_loop().create_future(),
        events=[],
        mark_brackets=[],
        http_brackets=[],
    )
    output = CONTROL_OUTPUT / "failure-controls"
    output.mkdir(exist_ok=True)
    result = {}
    import hashlib

    env = {
        "capture": capture,
        "result": result,
        "RUN": output,
        "phase": "unit-control",
        "asyncio": asyncio,
        "json": json,
        "hashlib": hashlib,
    }
    exec(code, env)
    try:
        raise failure
    except OSError:
        await env["retain_failure"]()
        try:
            raise
        except OSError as error:
            assert error is failure
    assert (
        cdp.calls == ["Tracing.end"]
        and result["failed_trace_end_error_type"] == "OSError"
    )
    saved = json.loads((output / "failed-partial-trace.json").read_text())
    assert (
        saved["partial_failed_segment"] is True
        and saved["clock_qualified"] is False
        and saved["tracing_complete"] is None
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "actual_command_once_return_error_and32_16_span_caps": True,
                "both_actual45s_barriers_refuse_late_arrival_and_postread": True,
                "actual_preflight_prior60_total300_fences": True,
                "original_error_preserved_failed_raw_partial_uncalibrated_retained": True,
                "App_browser_fixture_server_model_launches": 0,
                "product_samples": 0,
                "qualified": False,
            }
        )
    )


asyncio.run(run())
