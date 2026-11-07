"""Prepare a fresh no-pip selected-source QA venv; NEVER launch a GUI.

Run only after root supplies the final source freeze. Source-first .pth also
protects unchanged python-I physical workers. No shared-environment edits.
"""

from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time
import venv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--shared-dependencies", required=True, type=Path)
    parser.add_argument("--original-home", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--shape", choices=("single", "batch", "raw-deny", "large"))
    mode.add_argument("--negative-controls", action="store_true")
    parser.add_argument("--columns", type=int, default=120)
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument(
        "--method-mode", choices=("validation", "qualification"), default="validation"
    )
    parser.add_argument("--qualification-authorization", type=Path)
    parser.add_argument(
        "--diagnostic-stack", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    assert (args.columns, args.rows) in {(80, 24), (120, 40), (170, 48)}
    here = Path(__file__).resolve().parent
    source = args.source_root.resolve()
    freeze = args.freeze.resolve()
    deps = args.shared_dependencies.resolve()
    assert not any(name.startswith("tldw_chatbook") for name in sys.modules)
    sys.path.insert(0, str(source))
    assert (
        Path(importlib.util.find_spec("Tests").origin).resolve().is_relative_to(source)
    )
    os.environ["TLDW_TEST_REAL_HOME"] = str(args.original_home.resolve())
    from Tests import real_profile_guard

    real_profile_guard.install()  # Before any profile creation/redirect.
    value = json.loads(freeze.read_text(encoding="utf-8-sig"))
    assert (
        Path(value["source_root"]).resolve() == source
        and value["sampling_authorized"] is True
    )
    revision = subprocess.run(
        [
            "git",
            "-c",
            "safe.directory=" + source.as_posix(),
            "-C",
            str(source),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert revision == value["revision"]
    for relative, expected in value["sha256"].items():
        target = (source / relative).resolve()
        assert (
            target.is_relative_to(source)
            and hashlib.sha256(target.read_bytes()).hexdigest() == expected.lower()
        ), relative
    output = args.output_root.resolve()
    if not output.is_relative_to(source / ".superpowers"):
        raise RuntimeError("QA outputs must stay inside selectedsource/.superpowers")
    output.mkdir(parents=True, exist_ok=True)
    run = output / (
        "run-"
        + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-"
        + secrets.token_hex(4)
    )
    run.mkdir()
    files = (
        "native_series_method.py",
        "native_rehearsal_observation.py",
        "exact-pixel-refs.json",
        "common_activity_observer.py",
        "native_bootstrap.py",
        "qa_process_guard.py",
        "native_product_child.py",
        "approval_fixture_workload_v7.py",
        "approval_fixture_driver.py",
        "owned_roi_capture.py",
        "native_window_api.py",
        "native_product_observer.py",
    )
    references = json.loads(
        (here / "exact-pixel-refs.json").read_text(encoding="utf-8")
    )
    files = files + tuple(sorted(references["source_pins"]))
    for name in files:
        original = (here / name).resolve()
        target = (run / name).resolve()
        if not original.is_relative_to(here) or not target.is_relative_to(run):
            raise RuntimeError("QA reference path escaped bundle/run")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
    selection = {
        "run_nonce": secrets.token_hex(16),
        "shape": args.shape,
        "negative_controls": args.negative_controls,
        "columns": args.columns,
        "rows": args.rows,
        "conditions": {
            "canvas_enabled": False,
            "TLDW_CANVAS_ENABLED": "false",
            "ordinary_CPython_cache_validation": True,
            "bytecode_writes": False,
        },
        "source_root": str(source),
        "freeze_path": str(freeze),
        "original_home": str(args.original_home.resolve()),
        "qualified": False,
        "product_samples": 0,
        "helper_sha256": {
            name: hashlib.sha256((run / name).read_bytes()).hexdigest()
            for name in files
        },
    }
    selection["method_mode"] = args.method_mode
    selection["diagnostic_stack"] = args.diagnostic_stack
    selection["conditions"]["diagnostic_stack"] = args.diagnostic_stack
    selection["revision"] = revision
    if args.qualification_authorization is not None:
        selection["method_authorization"] = json.loads(
            args.qualification_authorization.read_text(encoding="utf-8")
        )
    sys.path.insert(0, str(here))
    from native_series_method import authorized_round_limit

    selection["runtime_round_limit"] = authorized_round_limit(selection, run)
    (run / "source-selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    venv.EnvBuilder(with_pip=False).create(run / "qa-venv")
    site = run / "qa-venv/Lib/site-packages"
    paths = [
        run,
        source,
        source / "packages/tldw_profile_core/src",
        deps,
        deps / "win32",
        deps / "win32/lib",
        deps / "pythonwin",
    ]
    for path in paths:
        assert path.is_dir(), str(path)
    entry = "import os; exec(\"try:\\n import qa_process_guard\\nexcept BaseException:\\n os.write(2,b'QA_site_guard_failed_closed'); os._exit(97)\")"
    (site / "qa-source-first.pth").write_text(
        "\n".join(str(path) for path in paths)
        + "\n"
        + entry
        + "\nimport pywin32_bootstrap\n",
        encoding="utf-8",
    )
    interpreter = run / "qa-venv/Scripts/python.exe"
    receipt = {
        "run": str(run),
        "interpreter": str(interpreter),
        "base_executable": sys.executable,
        "with_pip": False,
        "installs": 0,
        "source_root": str(source),
        "revision": revision,
        "freeze_sha256": hashlib.sha256(freeze.read_bytes()).hexdigest(),
        "helper_sha256": selection["helper_sha256"],
        "qualified": False,
        "product_samples": 0,
        "GUI_launched": False,
        "next_child_command": subprocess.list2cmdline(
            [
                str(interpreter),
                "-I",
                str(run / "native_product_observer.py"),
                "--allow-native-observer",
            ]
        ),
    }
    (run / "prepared-environment.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
