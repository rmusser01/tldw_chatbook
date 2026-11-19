"""Controlled full authorization fixtures; never actual Root/product qualification."""

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path, PureWindowsPath
import shutil
import sys
import tempfile
from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument("--source-root", required=True, type=Path)
parser.add_argument("--output-root", required=True, type=Path)
parser.add_argument("--dependencies", type=Path)
args = parser.parse_args()
output = args.output_root.resolve()
if not output.is_relative_to(args.source_root.resolve() / ".superpowers"):
    raise RuntimeError("Pure outputs must stay within selectedsource/.superpowers")
output.mkdir(parents=True, exist_ok=True)
here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "native_series_method", here / "native_series_method.py"
)
method = importlib.util.module_from_spec(spec)
spec.loader.exec_module(method)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


results = {}
with tempfile.TemporaryDirectory(dir=output) as temporary:
    folder = Path(temporary)
    source = folder / "selected-source"
    source.mkdir()
    run = folder / "run"
    run.mkdir()
    helper = run / "controlled-helper.py"
    helper.write_text(
        "# controlled finite auth fixture; never executed\n", encoding="utf-8"
    )
    png = folder / "reference.png"
    shutil.copyfile(here / "reference-fixtures/accepted-title.png", png)
    review = folder / "controlled-review.json"
    review.write_text(
        json.dumps(
            {
                "observed": True,
                "reviewer": "synthetic control only; not actual Root",
                "source_png": str(png),
                "png_sha256": digest(png),
                "patches": {
                    key: {"RGBA_sha256": "controlled", "pixel_size": [1, 1]}
                    for key in ("header", "send", "provider")
                },
            }
        ),
        encoding="utf-8",
    )
    freeze = folder / "freeze.json"
    selection = {
        "method_mode": "qualification",
        "source_root": str(source),
        "revision": "controlled-revision",
        "freeze_path": str(freeze),
        "helper_sha256": {helper.name: digest(helper)},
    }
    authorization = {
        "qualified_method": True,
        "rounds": 41,
        "feedback_root_reviewed": True,
        "terminal_root_reviewed": True,
        "source_root": str(source),
        "revision": selection["revision"],
        "runtime_sha256": selection["helper_sha256"],
        "feedback_review": {"path": str(review), "sha256": digest(review)},
        "terminal_review": {"path": str(review), "sha256": digest(review)},
    }
    selection["method_authorization"] = authorization
    cases = [
        ("false", False, False),
        ("missing", None, False),
        ("integer1", 1, False),
        ("string_true", "true", False),
        ("explicit_true", True, True),
    ]
    for name, flag, allowed in cases:
        value = {"source_root": str(source), "revision": selection["revision"]}
        if name != "missing":
            value["qualification_authorized"] = flag
        freeze.write_text(json.dumps(value), encoding="utf-8")
        authorization["freeze_sha256"] = digest(freeze)
        reason = None
        try:
            actual = method.authorized_round_limit(selection, run)
        except RuntimeError as error:
            actual = None
            reason = str(error)
        results["qualification_" + name] = {
            "actual_limit": actual,
            "expected_allowed": allowed,
            "reason": reason,
            "pass": (actual == 41) == allowed,
        }
        diagnostic = dict(selection, method_mode="validation")
        assert method.authorized_round_limit(diagnostic, run) == 1
    for label, changed in (
        ("source_mismatch", {"source_root": str(folder / "wrong")}),
        ("revision_mismatch", {"revision": "wrong"}),
    ):
        value = {
            "qualification_authorized": True,
            "source_root": str(source),
            "revision": selection["revision"],
            **changed,
        }
        freeze.write_text(json.dumps(value), encoding="utf-8")
        authorization["freeze_sha256"] = digest(freeze)
        try:
            actual = method.authorized_round_limit(selection, run)
        except RuntimeError:
            actual = None
        results["freeze_" + label] = {"actual_limit": actual, "pass": actual is None}

# Execute the exact real Git-call AST with a Windows path value; no subprocess is started.
windows = PureWindowsPath("Q:/synthetic-selected/repository")
for name in ("prepare_native_run.py", "qa_process_guard.py"):
    tree = ast.parse((here / name).read_text(encoding="utf-8"))
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
        and node.func.attr == "run"
        and node.args
        and isinstance(node.args[0], ast.List)
        and isinstance(node.args[0].elts[0], ast.Constant)
        and node.args[0].elts[0].value == "git"
    )
    commands = []

    def actual_run(argv, **_kwargs):
        commands.append(argv)
        return SimpleNamespace(stdout="controlled-revision")

    namespace = {
        "source": windows,
        "REPO": windows,
        "subprocess": SimpleNamespace(run=actual_run),
    }
    eval(
        compile(ast.fix_missing_locations(ast.Expression(body=call)), name, "eval"),
        namespace,
    )
    results[name + "_windows_safe_directory"] = {
        "safe_directory_argument": commands[0][2],
        "pass": commands[0][2] == "safe.directory=" + windows.as_posix()
        and commands[0][4] == str(windows),
    }
failed = [name for name, value in results.items() if not value["pass"]]
receipt = {
    "status": "RED" if failed else "GREEN",
    "qualified": False,
    "product_samples": 0,
    "source_sha256": {
        name: digest(here / name)
        for name in (
            "native_series_method.py",
            "prepare_native_run.py",
            "qa_process_guard.py",
        )
    },
    "results": results,
    "failures": failed,
    "scope": "All other authorization flags/bytes/reviews match controlled fixtures. No App/product/native import, Git process, GUI, real model, external/service traffic or actual Root qualification.",
}
(output / "qualification-freeze-control-results.json").write_text(
    json.dumps(receipt, indent=2), encoding="utf-8"
)
print(json.dumps(receipt, indent=2))
assert not failed, failed
assert not any(
    name.startswith(("tldw_chatbook", "tldw_profile_core", "textual", "ctypes"))
    for name in sys.modules
)
