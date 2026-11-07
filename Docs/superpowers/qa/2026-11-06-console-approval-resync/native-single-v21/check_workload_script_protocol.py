"""Pure QA fixture protocol controls. No product, socket, app or GUI launch."""

from pathlib import Path
import ast
import hashlib
import json
import sys
import threading
from types import SimpleNamespace

import argparse

_parser = argparse.ArgumentParser()
_parser.add_argument("--source-root", required=True, type=Path)
_parser.add_argument("--output-root", required=True, type=Path)
_parser.add_argument("--dependencies", type=Path)
_args = _parser.parse_args()
OUTPUT = _args.output_root.resolve()
if not OUTPUT.is_relative_to(_args.source_root.resolve() / ".superpowers"):
    raise RuntimeError(
        "Pure control outputs must stay inside selectedsource/.superpowers"
    )
OUTPUT.mkdir(parents=True, exist_ok=True)
if _args.dependencies is not None:
    sys.path.insert(0, str(_args.dependencies.resolve()))
HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
from approval_fixture_workload_v7 import ScriptedProvider, ScriptPreconditionRefused  # noqa: E402 - Selected-source/profile/dependency guard runs first.


def main():
    source = (HERE / "approval_fixture_workload_v7.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    driver = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ApprovalFixtureDriver"
    )
    stage = next(
        node
        for node in driver.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "stage"
    )
    shape_if = next(
        node
        for node in stage.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "shape == 'raw-deny'"
    )
    correlation = "a" * 32
    fixture = OUTPUT / "static-synthetic-fixture"
    fixture.mkdir(exist_ok=True)
    caps = {}
    negatives = []
    for shape in ("single", "batch", "large", "raw-deny"):
        # Execute ONLY the original private serialization block, excluding raw
        # arming and every controller/setup owner. These are fixture code checks.
        nodes = shape_if.body[2:] if shape == "raw-deny" else shape_if.orelse
        env = {
            "shape": shape,
            "correlation": correlation,
            "self": SimpleNamespace(root=fixture),
            "current": {},
        }
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                str(HERE / "approval_fixture_workload_v7.py"),
                "exec",
            ),
            env,
        )
        tool, catalog_id, targets = env["tool"], env["catalog_id"], env["targets"]
        provider = object.__new__(ScriptedProvider)
        provider._lock = threading.Lock()
        provider._turns = []
        provider.scripted_turns = 0
        provider.plan(
            [
                [("find_tools", {"query": tool})],
                [("load_tools", {"ids": [catalog_id]})],
                targets,
                "Synthetic complete.",
            ],
            correlation,
        )
        request = {
            "messages": [],
            "tools": [{"type": "function", "function": {"name": tool}}],
        }
        provider.consume(request)

        def result(ordinal, content):
            return {
                "role": "tool",
                "tool_call_id": "qa-" + correlation[:12] + "-" + str(ordinal) + "-0",
                "content": content,
            }

        discovery = result(
            0, catalog_id + " — " + tool + ": genuine-response-shape-only"
        )

        def refused(label, value, expected):
            before = (provider.scripted_turns, provider.pending_count())
            try:
                provider.consume(value)
            except ScriptPreconditionRefused as exc:
                assert exc.code == expected, (label, exc.code)
            else:
                raise AssertionError("Expected QA precondition refusal: " + label)
            assert (provider.scripted_turns, provider.pending_count()) == before
            negatives.append(
                {"shape": shape, "control": label, "target_consumed": False}
            )

        refused(
            "actual_V12_ask_user_only_discovery",
            dict(
                request,
                messages=[result(0, "local:ask_user — ask_user: Ask the user.")],
            ),
            "actual_discovery_target_missing",
        )
        refused(
            "wrong_catalog_id",
            dict(request, messages=[result(0, "local:wrong — " + tool + ": wrong")]),
            "actual_discovery_target_missing",
        )
        provider.consume(dict(request, messages=[discovery]))
        loaded = result(1, "loaded: " + tool)
        refused(
            "actual_V12_invalid_id_load",
            dict(
                request,
                messages=[
                    discovery,
                    result(1, "ERROR: invalid tool ids: " + catalog_id),
                ],
            ),
            "actual_load_target_not_accepted",
        )
        refused(
            "budget_omitted_load",
            dict(
                request,
                messages=[
                    discovery,
                    result(1, "not loaded (request budget): " + catalog_id),
                ],
            ),
            "actual_load_target_not_accepted",
        )
        refused(
            "wrong_result_call_id",
            dict(request, messages=[discovery, dict(loaded, tool_call_id="wrong")]),
            "actual_load_target_not_accepted",
        )
        refused(
            "duplicate_result",
            dict(request, messages=[discovery, loaded, loaded]),
            "actual_load_target_not_accepted",
        )
        refused(
            "missing_actual_native_schema",
            dict(request, messages=[discovery, loaded], tools=[]),
            "actual_native_target_schema_missing",
        )
        target_message, _, _ = provider.consume(
            dict(request, messages=[discovery, loaded])
        )
        decoded = json.loads(json.dumps(target_message))["tool_calls"]
        assert len(decoded) == (4 if shape == "batch" else 1)
        assert len({call["id"] for call in decoded}) == len(decoded)
        args = json.loads(decoded[0]["function"]["arguments"])
        if shape == "large":
            caps["large_content_utf8_bytes"] = len(args["content"].encode("utf-8"))
            assert (
                caps["large_content_utf8_bytes"] == 1048576 and args["dry_run"] is True
            )
        elif shape == "raw-deny":
            caps["raw_command_utf8_bytes"] = len(args["command"].encode("utf-8"))
            assert caps["raw_command_utf8_bytes"] == 16384
        elif shape == "batch":
            caps["batch_distinct_native_calls"] = len(decoded)
        provider.consume(request)
        assert provider.pending_count() == 0 and provider.scripted_turns == 4
    assert not any(
        name.startswith(("tldw_chatbook", "tldw_profile_core")) for name in sys.modules
    )
    data = {
        "qualified": False,
        "product_samples": 0,
        "GUI_launched": False,
        "scope": "Pure private fixture validator/serialization only; synthesized protocol controls are NOT product or real discovery evidence",
        "shape_caps": caps,
        "refusal_controls": negatives,
        "product_modules_imported": 0,
        "helper_sha256": hashlib.sha256(
            (HERE / "approval_fixture_workload_v7.py").read_bytes()
        ).hexdigest(),
    }
    (OUTPUT / "script-protocol-selfcheck.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "shape_caps": caps,
                "refusal_controls_passed": len(negatives),
                "product_modules_imported": 0,
                "qualified": False,
                "product_samples": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
