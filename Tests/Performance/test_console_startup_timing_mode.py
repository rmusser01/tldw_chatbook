"""No-App controls for diagnostic eligibility and original-route preservation."""

import ast
import importlib.util
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).parent
APP_MODULES_BEFORE = {name for name in sys.modules if name == "tldw_chatbook.app"}
SPEC = importlib.util.spec_from_file_location(
    "startup_timing_mode_driver", HERE / "run_console_startup_liveness.py"
)
DRIVER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DRIVER
SPEC.loader.exec_module(DRIVER)

SHA = "a" * 64
HELPER_PATH = "Tests/Performance/console_startup_timing_witness.py"


def receipt(mode="liveness"):
    result = {
        "mode": mode,
        "launch": "cold",
        "diagnostic_only": mode == "timing",
        "budget_acceptance_eligible": mode != "timing",
        "budgets_pass": None,
        "usable_input": {"seconds_from_parent_spawn": 1.0, "accepted": 10.1},
        "key_posted": 10.0,
        "mounted_heartbeat_window_end": 10.2,
        "heartbeat_intervals": [{"entered": 10.0, "exited": 10.1}],
        "loaded_source_sha256": {HELPER_PATH: SHA},
        "executed_probe_helpers": {
            "timing": {
                "raw_sha256": SHA,
                "start_callable": {
                    "source_relative_path": HELPER_PATH,
                    "raw_sha256": SHA,
                    "actual_defining_code": True,
                },
            }
        },
    }
    result["startup_timing"] = {
        name: True
        for name in (
            "complete",
            "diagnostic_only",
            "source_current",
            "monitoring_global_zero",
            "monitoring_masks_owned",
            "monitoring_callbacks_owned",
            "monitoring_masks_cleared_while_active",
            "monitoring_tool_freed",
        )
    }
    result["startup_timing"].update(budget_acceptance_eligible=False, budgets_pass=None)
    return result


class EligibilityControls(unittest.TestCase):
    def test_original_fast_liveness_still_passes_the_exact_original_bounds(self):
        result = DRIVER.liveness_budget_result(receipt())
        self.assertTrue(result["usable_within_limit"])
        self.assertTrue(result["mounted_heartbeat_within_limit"])
        self.assertEqual(result["usable_limit_seconds"], 15.0)
        self.assertEqual(result["mounted_heartbeat_limit_seconds"], 0.200)
        self.assertEqual(result["key_delivery_limit_seconds"], 0.500)

    def test_instrumented_timing_never_enters_original_budget_evaluation(self):
        with self.assertRaises(AssertionError):
            DRIVER.liveness_budget_result(receipt("timing"))

    def test_diagnostic_marker_alone_refuses_forged_liveness_eligibility(self):
        result = receipt()
        result["diagnostic_only"] = True
        with self.assertRaises(AssertionError):
            DRIVER.liveness_budget_result(result)

    def test_timing_receipt_can_only_return_null_budget_acceptance(self):
        self.assertTrue(callable(getattr(DRIVER, "timing_diagnostic_result", None)))
        result = DRIVER.timing_diagnostic_result(receipt("timing"), SHA)
        self.assertIs(result["budgets_pass"], None)
        self.assertFalse(result["budget_acceptance_eligible"])
        self.assertTrue(result["diagnostic_only"])

    def test_timing_source_retirement_and_eligibility_refusals(self):
        self.assertTrue(callable(getattr(DRIVER, "timing_diagnostic_result", None)))
        for name in (
            "source_current",
            "monitoring_global_zero",
            "monitoring_masks_owned",
            "monitoring_callbacks_owned",
            "monitoring_masks_cleared_while_active",
            "monitoring_tool_freed",
            "complete",
        ):
            result = receipt("timing")
            result["startup_timing"][name] = False
            with self.subTest(name=name), self.assertRaises(AssertionError):
                DRIVER.timing_diagnostic_result(result, SHA)
        for target, key, bad in (
            ("child", "budgets_pass", True),
            ("child", "budget_acceptance_eligible", True),
            ("witness", "budgets_pass", True),
            ("witness", "budget_acceptance_eligible", True),
        ):
            result = receipt("timing")
            (result if target == "child" else result["startup_timing"])[key] = bad
            with self.subTest(target=target, key=key), self.assertRaises(
                AssertionError
            ):
                DRIVER.timing_diagnostic_result(result, SHA)
        result = receipt("timing")
        result["loaded_source_sha256"][HELPER_PATH] = "foreign"
        with self.assertRaises(AssertionError):
            DRIVER.timing_diagnostic_result(result, SHA)
        result = receipt("timing")
        result["executed_probe_helpers"]["timing"]["start_callable"]["raw_sha256"] = (
            "foreign"
        )
        with self.assertRaises(AssertionError):
            DRIVER.timing_diagnostic_result(result, SHA)

    def test_stdlib_driver_import_does_not_import_app(self):
        self.assertEqual(
            APP_MODULES_BEFORE,
            {name for name in sys.modules if name == "tldw_chatbook.app"},
        )


class ModeSourceControls(unittest.TestCase):
    def tree(self, name):
        return ast.parse((HERE / name).read_bytes().decode("utf-8"))

    def test_parser_adds_only_timing_and_keeps_default_liveness(self):
        for name in (
            "console_startup_liveness_child.py",
            "run_console_startup_liveness.py",
        ):
            calls = [
                node
                for node in ast.walk(self.tree(name))
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--mode"
            ]
            self.assertEqual(len(calls), 1)
            options = {
                item.arg: ast.literal_eval(item.value) for item in calls[0].keywords
            }
            self.assertEqual(options["choices"], ("liveness", "timing"))
            if name.startswith("run_"):
                self.assertEqual(options["default"], "liveness")
            else:
                self.assertTrue(options["required"])

    def test_observer_starts_after_real_import_before_original_constructor(self):
        tree = self.tree("console_startup_liveness_child.py")
        observe = next(
            node
            for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "observe"
        )
        original_try = next(node for node in observe.body if isinstance(node, ast.Try))
        imported = next(
            index
            for index, node in enumerate(original_try.body)
            if isinstance(node, ast.ImportFrom) and node.module == "tldw_chatbook.app"
        )
        setup = next(
            index
            for index, node in enumerate(original_try.body)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "args.mode == 'timing'"
        )
        constructor = next(
            index
            for index, node in enumerate(original_try.body)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "TldwCli"
        )
        self.assertLess(imported, setup)
        self.assertLess(setup, constructor)
        call_names = [
            ast.unparse(node.func)
            for node in ast.walk(original_try.body[setup])
            if isinstance(node, ast.Call)
        ]
        self.assertIn("StartupTimingWitness", call_names)
        self.assertIn("timing.start", call_names)
        self.assertEqual(
            sum(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "TldwCli"
                for node in ast.walk(observe)
            ),
            1,
        )

    def test_outer_cleanup_stops_and_keeps_original_error_priority(self):
        observe = next(
            node
            for node in self.tree("console_startup_liveness_child.py").body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "observe"
        )
        original_try = next(node for node in observe.body if isinstance(node, ast.Try))
        cleanup = original_try.finalbody
        stops = [
            node
            for node in ast.walk(ast.Module(body=cleanup, type_ignores=[]))
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "timing.stop"
        ]
        self.assertEqual(len(stops), 1)
        retirement = next(
            node
            for node in cleanup
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "timing is not None"
        )
        error_try = retirement.body[0]
        self.assertIsInstance(error_try, ast.Try)
        self.assertEqual(ast.unparse(error_try.handlers[0].type), "BaseException")
        last = cleanup[-1]
        self.assertIsInstance(last, ast.If)
        self.assertEqual(
            ast.unparse(last.test),
            "timing_stop_failure is not None and (not timing_primary_error_present)",
        )
        self.assertIsInstance(last.body[0], ast.Raise)

    def test_timing_import_has_managed_namespace_and_selected_helper_inventory(self):
        child = self.tree("console_startup_liveness_child.py")
        imports = [
            node
            for node in ast.walk(child)
            if isinstance(node, ast.ImportFrom)
            and node.module == "Tests.Performance.console_startup_timing_witness"
        ]
        self.assertEqual(len(imports), 1)
        driver = self.tree("run_console_startup_liveness.py")
        selected = [
            node
            for node in ast.walk(driver)
            if isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "helper_paths"
        ]
        self.assertEqual(len(selected), 1)
        self.assertIn(
            "console_startup_timing_witness.py", ast.unparse(selected[0].value)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
