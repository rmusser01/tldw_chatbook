"""Pure APIs for the opt-in trace diagnostic, with no app/native launch.

Run directly with Python (stdlib unittest), or collect with pytest. The launcher
command is asserted but replaced only in this pure test by a tiny real Python
process that exits 7. Synthetic defining sources are never app/Tests imports.
"""

import argparse
import ast
import asyncio
import hashlib
import importlib.machinery
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import CodeType, FunctionType, ModuleType, SimpleNamespace


SOURCE_DIR = Path(__file__).parent


def _assert_no_app_imports(before):
    added = set(sys.modules) - before
    assert not any(
        name == "tldw_chatbook"
        or name.startswith("tldw_chatbook.")
        or name == "Tests"
        or name.startswith("Tests.")
        for name in added
    )


def _observer_control(directory, *, sources_loaded=True, missing_source=None):
    EVIDENCE = directory
    PLUGIN = SOURCE_DIR / "console_trace_settlement_witness.py"
    assert missing_source in (None, "store", "coordinator")
    assert not sources_loaded or missing_source is None
    imports_before = set(sys.modules)
    modules = {}
    try:
        tree = ast.parse(PLUGIN.read_text(encoding="utf-8"))
        definitions = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name in {"_shape", "_compiled_code", "TraceSettlementWitness"}
        ]
        assert len(definitions) == 3
        namespace = {
            "__name__": "trace_witness_pure_namespace",
            "ast": ast,
            "hashlib": hashlib,
            "sys": sys,
            "threading": threading,
            "time": time,
            "Path": Path,
            "CodeType": CodeType,
            "FunctionType": FunctionType,
            "ModuleType": ModuleType,
            "STORE_NAME": "trace_pure_store",
            "COORDINATOR_NAME": "trace_pure_coordinator",
            "PROBE_NAME": "trace_pure_probe",
            "TARGET_NAME": "test_native_console_pause_probe",
        }
        exec(
            compile(ast.Module(body=definitions, type_ignores=[]), str(PLUGIN), "exec"),
            namespace,
        )
        Witness = namespace["TraceSettlementWitness"]
        sources = {
            "trace_pure_coordinator": 'from dataclasses import dataclass\nfrom enum import Enum\nimport threading\nclass TraceCallState(Enum):\n    COMPLETE = "complete"\n@dataclass(frozen=True)\nclass TraceCallRecord:\n    state: TraceCallState\n@dataclass(frozen=True)\nclass _PreparedSettlement:\n    call_id: str\n@dataclass(frozen=True)\nclass ConsoleTraceSettlementHandoff:\n    _coordinator: object\n    _database: object\n    _prepared: _PreparedSettlement\n    def settle(self, canonical_message_id):\n        return self._coordinator._settle_claimed(self._database, self._prepared)\nclass ConsoleTraceSettlementCoordinator:\n    def __init__(self):\n        self._queue_lock = threading.Lock()\n        self._pending, self._inflight = {}, {}\n        self._dropped_count = 0\n    def _settle_claimed(self, database, prepared):\n        self._inflight[prepared.call_id] = "pure"\n        try:\n            self._settle_prepared(database, prepared)\n        except Exception:\n            with self._queue_lock:\n                self._inflight.pop(prepared.call_id, None)\n                self._enqueue_locked(database, prepared)\n            return False\n        self._inflight.pop(prepared.call_id, None)\n        return True\n    def _settle_prepared(self, database, prepared):\n        if database:\n            raise RuntimeError("pure API controlled exception")\n        return TraceCallRecord(TraceCallState.COMPLETE)\n    def _enqueue_locked(self, database, prepared):\n        self._pending[prepared.call_id] = (database, prepared)\n',
            "trace_pure_store": 'import threading\nclass ConsoleChatStore:\n    def __init__(self, normal, failure):\n        self.normal, self.failure = normal, failure\n        self._provider_trace_settlement_lock = threading.RLock()\n        self._provider_trace_settlements = {}\n        self._provider_trace_settlement_work = {}\n        self._provider_trace_settlement_failed_work = {}\n        self._provider_trace_settlement_owned_call_ids = set()\n        self._provider_trace_settlement_worker_active = False\n        self._provider_trace_settlement_executor_closed = False\n    @staticmethod\n    def _run_provider_trace_settlement(handoff, canonical_message_id):\n        try:\n            result = handoff.settle(canonical_message_id)\n            return result is not False\n        except Exception:\n            return False\n    def _drain_provider_trace_settlement_work(self):\n        self._run_provider_trace_settlement(self.normal, None)\n        if not self._run_provider_trace_settlement(self.failure, None):\n            self._provider_trace_settlement_failed_work["pure-failure"] = (self.failure, None)\n            self._provider_trace_settlement_owned_call_ids.add("pure-failure")\n',
            "trace_pure_probe": 'async def test_native_console_pause_probe(witness, thread, go, errors, controller):\n    result = {"trace_states": ["complete"] * 3, "response_links": 3, "provider_calls": 3}\n    witness.start()\n    _load_sources()\n    go.set()\n    thread.join(5)\n    assert not thread.is_alive() and not errors\n    assert result["trace_states"] == ["complete"] * 3\n    return witness.stop(result)\n',
        }
        modules = {}
        for name, source in sources.items():
            path = EVIDENCE / (name + ".py")
            path.write_text(source, encoding="utf-8")
            module = ModuleType(name)
            module.__file__ = str(path)
            sys.modules[name] = module
            exec(compile(source, str(path), "exec"), module.__dict__)
            modules[name] = module
        coordinator_module = modules["trace_pure_coordinator"]
        coordinator = coordinator_module.ConsoleTraceSettlementCoordinator()
        normal = coordinator_module.ConsoleTraceSettlementHandoff(
            coordinator, False, coordinator_module._PreparedSettlement("pure-normal")
        )
        failure = coordinator_module.ConsoleTraceSettlementHandoff(
            coordinator, True, coordinator_module._PreparedSettlement("pure-failure")
        )
        store = modules["trace_pure_store"].ConsoleChatStore(normal, failure)
        go, entered = (threading.Event(), threading.Event())
        errors = []

        def worker():
            entered.set()
            go.wait(5)
            try:
                store._drain_provider_trace_settlement_work()
            except BaseException as error:
                errors.append(type(error).__name__)

        thread = threading.Thread(target=worker, name="pure-existing-trace-worker")
        thread.start()
        assert entered.wait(5)
        if not sources_loaded:
            for role in ("store", "coordinator"):
                name = "trace_pure_" + role
                assert sys.modules.pop(name) is modules[name]

        def load_sources():
            if not sources_loaded:
                for role in ("store", "coordinator"):
                    if role != missing_source:
                        name = "trace_pure_" + role
                        sys.modules[name] = modules[name]

        modules["trace_pure_probe"].__dict__["_load_sources"] = load_sources
        body = modules["trace_pure_probe"].test_native_console_pause_probe
        witness = Witness(
            modules["trace_pure_probe"], body, SimpleNamespace(phase="pure_control")
        )
        try:
            control = asyncio.run(
                body(witness, thread, go, errors, SimpleNamespace(store=store))
            )
        finally:
            go.set()
            thread.join(5)
            assert not thread.is_alive()
            if witness.active:
                witness.stop({})
        discovery = control["lazy_source_discovery"]
        assert discovery["initial_store_loaded"] is sources_loaded
        assert discovery["initial_coordinator_loaded"] is sources_loaded
        assert {item["source"] for item in discovery["qualified_bindings"]} == {
            "store",
            "coordinator",
        }
        assert control["original_terminal_source_and_owner_coverage_complete"]
        assert not control["invalid_evidence"] and (
            not control["live_spans_after_teardown"]
        )
        assert any(
            (row.get("terminal_state") == "complete" for row in control["spans"])
        )
        assert any(
            (
                row["label"] == "claimed" and row["result"] is False
                for row in control["spans"]
            )
        )
        assert any(
            (row.get("retired_by_original_parent_return") for row in control["spans"])
        )
        assert control["handled_exceptions"][0]["category"] == "builtins.RuntimeError"
        assert control["handled_exceptions"][0]["selected_exception_origins"]
        assert control["snapshots"][0]["boundary"] == "original_terminal_assertion"
        assert control["snapshots"][0]["stores"][0]["failed"]
        assert control["snapshots"][0]["coordinators"][0]["pending"]
        assert (
            control["global_events"] == 0 and control["hooks_retired_before_inactive"]
        )
        assert witness.monitor.get_tool(witness.tool) is None
        assert thread in witness.threads.values()
        _assert_no_app_imports(imports_before)
        return control
    finally:
        for name, module in modules.items():
            if sys.modules.get(name) is module:
                del sys.modules[name]


def _launcher_control(directory, mode):
    EVIDENCE = directory
    DRAFT = SOURCE_DIR / "run_trace_settlement_diagnostic.py"
    imports_before = set(sys.modules)
    tree = ast.parse(DRAFT.read_text(encoding="utf-8"))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert {node.name for node in definitions} == {
        "digest",
        "snapshot",
        "child_source_valid",
        "main",
    }
    namespace = {
        "argparse": argparse,
        "hashlib": hashlib,
        "json": json,
        "os": os,
        "platform": platform,
        "sys": sys,
        "time": time,
        "Path": Path,
        "PLUGIN": "Tests.Performance.console_trace_settlement_witness",
        "NODE": "Tests/Performance/test_console_native_pause_probe.py::test_native_console_pause_probe",
        "HELPERS": (
            "Tests.private_profile",
            "Tests.real_profile_guard",
            "Tests.network_guard",
            "Tests.windows_private_fixture_runner",
        ),
        "__file__": str(DRAFT),
    }
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), str(DRAFT), "exec"),
        namespace,
    )
    real_popen = subprocess.Popen
    receipts = []
    with tempfile.TemporaryDirectory(
        prefix="trace-launcher-pure-", dir=EVIDENCE
    ) as directory:
        root = Path(directory).resolve()
        core_root = root / "packages/tldw_profile_core/src"
        paths = (
            "tldw_chatbook/__init__.py",
            "tldw_chatbook/body.py",
            "Tests/__init__.py",
            "Tests/Performance/__init__.py",
            "Tests/Performance/console_trace_settlement_witness.py",
            "Tests/private_profile.py",
            "Tests/real_profile_guard.py",
            "Tests/network_guard.py",
            "Tests/windows_private_fixture_runner.py",
            "packages/tldw_profile_core/src/tldw_profile_core/__init__.py",
        )
        for relative in paths:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Synthetic unexecuted pure source\n", encoding="utf-8")
        namespace["importlib"] = SimpleNamespace(
            util=SimpleNamespace(
                find_spec=lambda name: importlib.machinery.PathFinder.find_spec(
                    name, [str(root), str(core_root)]
                )
            ),
            machinery=importlib.machinery,
        )
        native_command_seen = []
        output_base = root / "output" / mode
        original_argv = sys.argv

        def pure_failure_process(command, *, cwd, env, stdout, stderr):
            assert command == [
                sys.executable,
                "-m",
                "Tests.windows_private_fixture_runner",
                namespace["NODE"],
                "-q",
                "--timeout=900",
                "--junitxml=" + str(output_base.with_suffix(".xml")),
            ]
            assert (
                cwd == root
                and env["PYTEST_PLUGINS"].split(",")[-1] == namespace["PLUGIN"]
            )
            native_command_seen.append(command)
            file = root / "Tests/private_profile.py"
            child = {
                "python_executable": sys.executable,
                "python_version": sys.version,
                "modules": {
                    "Tests.private_profile": {
                        "loaded": True,
                        "origin": str(file),
                        "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                    }
                },
            }
            if mode == "foreign_child_origin":
                child["modules"]["Tests.private_profile"]["origin"] = str(
                    root.parent / "foreign-private-profile.py"
                )
            witness = {
                "actual_child_installed_sources_before": child,
                "actual_child_installed_sources_after": child,
                "actual_child_installed_sources_unchanged": True,
            }
            Path(env["TLDW_PAUSE_PROBE_RESULT"]).write_text(
                json.dumps({"diagnostic_trace_settlement": witness}), encoding="utf-8"
            )
            if mode == "source_drift":
                (root / "tldw_chatbook/body.py").write_text(
                    "# Genuine synthetic file drift\n", encoding="utf-8"
                )
            return real_popen(
                [sys.executable, "-c", "raise SystemExit(7)"],
                stdout=stdout,
                stderr=stderr,
            )

        namespace["subprocess"] = SimpleNamespace(
            Popen=pure_failure_process, STDOUT=subprocess.STDOUT
        )
        try:
            sys.argv = [
                str(DRAFT),
                "--repo",
                str(root),
                "--output-base",
                str(output_base),
            ]
            status = namespace["main"]()
        finally:
            sys.argv = original_argv
        receipt = json.loads(
            output_base.with_suffix(".source.json").read_text(encoding="utf-8")
        )
        assert len(native_command_seen) == 1
        assert (
            receipt["test_exit_code"] == 7
            and receipt["original_test_process_positively_reaped"]
        )
        assert (
            receipt["original_pytest_timeout_seconds"] == 900
            and receipt["added_parent_timeout"] is None
        )
        assert len(receipt["source_before"]["managed_source_sha256"]) == len(paths)
        if mode == "original_failure":
            assert (
                status == 7
                and receipt["source_unchanged"]
                and receipt["diagnostic_source_valid"]
            )
        elif mode == "source_drift":
            assert (
                status == 2
                and (not receipt["source_unchanged"])
                and (not receipt["diagnostic_source_valid"])
            )
        else:
            assert (
                status == 2
                and receipt["source_unchanged"]
                and (not receipt["diagnostic_source_valid"])
            )
        receipts.append(
            {
                "control": mode,
                "exit_code": status,
                "original_failure_exit_retained": receipt["test_exit_code"],
                "source_unchanged": receipt["source_unchanged"],
                "diagnostic_source_valid": receipt["diagnostic_source_valid"],
                "actual_pure_child_positively_reaped": True,
            }
        )
    _assert_no_app_imports(imports_before)
    return receipts[0]


class TraceSettlementDiagnosticPureTests(unittest.TestCase):
    def test_code_local_observer_retires_existing_worker_and_exceptional_spans(self):
        for sources_loaded in (True, False):
            with self.subTest(sources_loaded=sources_loaded):
                with tempfile.TemporaryDirectory(
                    prefix="trace-observer-pure-"
                ) as temporary:
                    control = _observer_control(
                        Path(temporary), sources_loaded=sources_loaded
                    )
                self.assertEqual(control["global_events"], 0)
                self.assertTrue(control["hooks_retired_before_inactive"])
                self.assertFalse(control["invalid_evidence"])

    def test_original_terminal_requires_both_lazy_sources_and_observed_owners(self):
        for missing_source in ("store", "coordinator"):
            with self.subTest(missing_source=missing_source):
                with tempfile.TemporaryDirectory(
                    prefix="trace-missing-source-"
                ) as temporary:
                    with self.assertRaisesRegex(
                        AssertionError,
                        "requires qualified stock sources and observed owners",
                    ):
                        _observer_control(
                            Path(temporary),
                            sources_loaded=False,
                            missing_source=missing_source,
                        )
                self.assertTrue(
                    all(sys.monitoring.get_tool(number) is None for number in (3, 4, 5))
                )

    def _launcher(self, mode):
        with tempfile.TemporaryDirectory(prefix="trace-launcher-api-") as temporary:
            result = _launcher_control(Path(temporary), mode)
        self.assertEqual(result["original_failure_exit_retained"], 7)
        self.assertTrue(result["actual_pure_child_positively_reaped"])
        return result

    def test_launcher_preserves_failure_receipt_after_real_child_reap(self):
        result = self._launcher("original_failure")
        self.assertEqual(result["exit_code"], 7)
        self.assertTrue(result["diagnostic_source_valid"])

    def test_launcher_marks_actual_source_drift_invalid_and_retains_child_exit(self):
        result = self._launcher("source_drift")
        self.assertEqual(result["exit_code"], 2)
        self.assertFalse(result["source_unchanged"])

    def test_launcher_rejects_foreign_child_origin_despite_stable_parent_sources(self):
        result = self._launcher("foreign_child_origin")
        self.assertEqual(result["exit_code"], 2)
        self.assertTrue(result["source_unchanged"])
        self.assertFalse(result["diagnostic_source_valid"])


if __name__ == "__main__":
    unittest.main()
