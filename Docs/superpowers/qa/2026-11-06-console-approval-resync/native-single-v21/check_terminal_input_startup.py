"""Exercise the selected child's startup ordering without a terminal or App."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys


def call_name(node):
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        return ast.unparse(node.value.func)
    return None


def actual_startup(source):
    """Compile just the original startup slice and actual application entry."""
    main = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    start = next(
        index
        for index, node in enumerate(main.body)
        if ast.unparse(node) == "startup_phase('app_import_after')"
    )
    end = next(
        index
        for index, node in enumerate(main.body)
        if ast.unparse(node) == "startup_phase('app_constructor_after')"
    )
    run = next(
        node.body[0]
        for node in main.body
        if isinstance(node, ast.Try) and call_name(node.body[0]) == "app.run"
    )
    return compile(
        ast.Module(body=[*main.body[start : end + 1], run], type_ignores=[]),
        "actual-native-startup",
        "exec",
    ), sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "warm_up_image_protocol"
        for node in ast.walk(main)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    output = args.output_root.resolve()
    if not output.is_relative_to(root / ".superpowers"):
        raise RuntimeError(
            "Pure outputs must stay beneath selected-source/.superpowers"
        )
    sys.path.insert(0, str(args.dependencies.resolve()))
    sys.dont_write_bytecode = True
    from textual._xterm_parser import XTermParser

    source_path = (
        root
        / "Docs/superpowers/qa/2026-11-06-console-approval-resync"
        / "native-single-v21"
        / "native_product_child.py"
    )
    code, warm_calls = actual_startup(source_path.read_text(encoding="utf-8"))

    motion = "\x1b[<35;41;21m"
    click = "\x1b[<0;41;21M\x1b[<0;41;21m"

    def classify(*chunks):
        parser = XTermParser()
        return [
            type(event).__name__ for chunk in chunks for event in parser.feed(chunk)
        ]

    def run_case(image_available):
        order = []
        events = []
        image_cached = False

        def warm_up_image_protocol():
            nonlocal image_cached
            order.append("warm_up_image_protocol")
            image_cached = image_available
            return image_available

        def controlled_import(name, *_args, **_kwargs):
            if name != "tldw_chatbook.Utils.terminal_utils":
                raise AssertionError("Unexpected import in extracted startup")
            from types import SimpleNamespace

            return SimpleNamespace(warm_up_image_protocol=warm_up_image_protocol)

        class SyntheticApp:
            def __init__(self):
                order.append("app_constructor")

            def run(self):
                order.append("app_run")
                # Model only the witnessed lost-prefix symptom of a competing
                # reader. No mock tries to reproduce the Windows input queue.
                if image_available and not image_cached:
                    events.extend(classify(motion[1:], click))
                else:
                    events.extend(classify(motion[:1], motion[1:], click))

        exec(
            code,
            {
                "__builtins__": {"__import__": controlled_import},
                "startup_phase": lambda _name: None,
                "TldwCli": SyntheticApp,
            },
        )
        return {"order": order, "events": events}

    expected_order = ["warm_up_image_protocol", "app_constructor", "app_run"]
    expected_events = ["MouseMove", "MouseDown", "MouseUp"]
    results = {
        "image_dependency_available": run_case(True),
        "image_dependency_absent": run_case(False),
    }
    checks = {
        "stock_helper_called_once": warm_calls == 1,
        "each_start_warms_before_constructor_and_run": all(
            case["order"] == expected_order for case in results.values()
        ),
        "cold_import_cannot_compete_for_the_synthetic_gesture": (
            results["image_dependency_available"]["events"] == expected_events
        ),
        "optional_dependency_absence_preserves_startup": (
            results["image_dependency_absent"]["events"] == expected_events
        ),
        "complete_gesture_decodes": classify(motion, click) == expected_events,
        "lost_prefix_decodes_as_eleven_keys": classify(motion[1:]) == ["Key"] * 11,
    }
    receipt = {
        "scope": "Extracted startup, controlled image-import ordering, actual Textual parser; no App or Win32 input.",
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "qualified": False,
        "product_samples": 0,
        "checks": checks,
        "cases": results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "terminal-input-startup-results.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
    assert all(checks.values()), "Native terminal startup ordering controls failed"


if __name__ == "__main__":
    main()
