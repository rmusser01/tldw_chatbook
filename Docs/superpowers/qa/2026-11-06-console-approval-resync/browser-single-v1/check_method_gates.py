"""Actual AST gates and pure authorization controls; no product/App/venv creation."""

import ast, asyncio, copy, hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from browser_series_method import authorized_rounds


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


def refused(call):
    try:
        call()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("Required_refusal_not_observed")


root = control_output() / "authorization-controls"
root.mkdir(exist_ok=True)
runtime = root / "unit-runtime.bin"
runtime.write_bytes(b"unit logic only, no runtime")
image = root / "NOT-AN-ACTUAL-IMAGE.bin"
image.write_bytes(b"unit reference, no image claim")
review = root / "unit-review.json"
review.write_text(
    json.dumps(
        {
            "observed": True,
            "reviewer": "unit-logic-not-Root",
            "source_png": str(image),
            "png_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        }
    )
)
freeze = root / "unit-freeze.json"
freeze.write_text(
    json.dumps(
        {
            "source_root": str(root),
            "revision": "unit",
            "qualification_authorized": False,
        }
    )
)
selection = {
    "method_mode": "qualification",
    "diagnostic_stack": False,
    "source_root": str(root),
    "revision": "unit",
    "freeze_path": str(freeze),
    "fixture_hashes": {
        "unit-runtime.bin": hashlib.sha256(runtime.read_bytes()).hexdigest()
    },
}
auth = {
    "rounds": 41,
    "qualified_method": True,
    "feedback_root_reviewed": True,
    "terminal_root_reviewed": True,
    "source_root": str(root),
    "revision": "unit",
    "freeze_sha256": hashlib.sha256(freeze.read_bytes()).hexdigest(),
    "runtime_sha256": selection["fixture_hashes"],
    "feedback_review": {
        "path": str(review),
        "sha256": hashlib.sha256(review.read_bytes()).hexdigest(),
    },
    "terminal_review": {
        "path": str(review),
        "sha256": hashlib.sha256(review.read_bytes()).hexdigest(),
    },
}
selection["method_authorization"] = auth
refused(lambda: authorized_rounds(selection, root))
validation = copy.deepcopy(selection)
validation["method_mode"] = "validation"
validation["rounds"] = 41
assert authorized_rounds(validation, root) == 1
bad = copy.deepcopy(selection)
bad["diagnostic_stack"] = True
refused(lambda: authorized_rounds(bad, root))
freeze.write_text(
    json.dumps(
        {"source_root": str(root), "revision": "unit", "qualification_authorized": True}
    )
)
auth["freeze_sha256"] = hashlib.sha256(freeze.read_bytes()).hexdigest()
assert authorized_rounds(selection, root) == 41
runtime.write_bytes(b"changed")
refused(lambda: authorized_rounds(selection, root))
runtime.write_bytes(b"unit logic only, no runtime")
image.write_bytes(b"changed")
refused(lambda: authorized_rounds(selection, root))
image.write_bytes(b"unit reference, no image claim")

child = ast.parse((HERE / "web_product_child.py").read_text(encoding="utf-8"))
command = next(
    n for n in child.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "command"
)
stage = next(
    n
    for n in ast.walk(command)
    if isinstance(n, ast.If) and ast.unparse(n.test) == "kind == 'stage'"
)
fn = ast.AsyncFunctionDef(
    name="stage_gate",
    args=ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg="value")],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
    ),
    body=[ast.Global(names=["STAGE_COUNT", "ACTIVE_CORRELATION"])]
    + copy.deepcopy(stage.body),
    decorator_list=[],
)
module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))


class Driver:
    def __init__(self):
        self.minted = 0
        self.called = []
        self.error = None

    def new_correlation(self):
        self.minted += 1
        return f"{self.minted:032x}"

    async def stage(self, shape, correlation):
        self.called.append((shape, correlation))
        if self.error:
            raise self.error


async def stages():
    scope = {
        "SELECTION": {"mode": "smoke"},
        "PLANNED_ROUNDS": 1,
        "STAGE_COUNT": 0,
        "driver": Driver(),
    }
    exec(compile(module, "actual-child-stage-gate", "exec"), scope)
    await scope["stage_gate"](
        {"shape": "single", "correlation": "externally_supplied_ignored"}
    )
    assert scope["driver"].called == [("single", "0" * 31 + "1")]
    try:
        await scope["stage_gate"]({"shape": "single"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("Validation_second_attempt_released")
    scope["STAGE_COUNT"] = 0
    scope["driver"] = Driver()
    failure = OSError(22)
    scope["driver"].error = failure
    try:
        await scope["stage_gate"]({"shape": "single"})
    except OSError as error:
        assert error is failure
    assert scope["STAGE_COUNT"] == 1
    try:
        await scope["stage_gate"]({"shape": "single"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("Failed_round_replaced")
    for mode, shape in [("negative", "single"), ("smoke", "raw-deny")]:
        scope.update(SELECTION={"mode": mode}, STAGE_COUNT=0, driver=Driver())
        try:
            await scope["stage_gate"]({"shape": shape})
        except RuntimeError:
            pass
        else:
            raise AssertionError("Out_of_scope_stage_released")
        assert not scope["driver"].called
    scope.update(
        SELECTION={"mode": "smoke"}, STAGE_COUNT=0, PLANNED_ROUNDS=41, driver=Driver()
    )
    for _ in range(41):
        await scope["stage_gate"]({"shape": "single"})
    assert scope["STAGE_COUNT"] == 41 and len(scope["driver"].called) == 41
    try:
        await scope["stage_gate"]({"shape": "single"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("Qualification42_released")


asyncio.run(stages())

parent = ast.parse((HERE / "web_smoke.py").read_text(encoding="utf-8"))
parents = {
    child: node for node in ast.walk(parent) for child in ast.iter_child_nodes(node)
}
barriers = []
for node in ast.walk(parent):
    if isinstance(node, ast.While) and "review_path.exists" in ast.unparse(node.test):
        chain = []
        current = node
        while current in parents:
            current = parents[current]
            chain.append(current)
        assert any(
            isinstance(n, ast.If)
            and ast.unparse(n.test) == "SELECTION['method_mode'] == 'validation'"
            for n in chain
        )
        barriers.append(node)
assert len(barriers) == 2
assert not any(
    isinstance(n, ast.Name) and n.id == "faulthandler" for n in ast.walk(parent)
)
assert not any(
    isinstance(n, ast.Name) and n.id == "faulthandler" for n in ast.walk(child)
)
print(
    json.dumps(
        {
            "status": "PASS",
            "actual_parsed_freeze_required": True,
            "runtime_image_reference_hashes_required": True,
            "validation_hard1_qualification41_no_replacement": True,
            "actual_child_stage_AST_once_original_error_identity": True,
            "external_correlation_ignored": True,
            "warm_no_human_barriers_AST": True,
            "QA_Cdiagnostics_absent_AST": True,
            "product_App_venv_model_launches": 0,
            "qualified": False,
            "product_samples": 0,
        }
    )
)
