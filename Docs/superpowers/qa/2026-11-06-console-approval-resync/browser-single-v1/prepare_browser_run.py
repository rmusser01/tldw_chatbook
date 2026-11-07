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
    mode.add_argument("--shape", choices=("single",))
    mode.add_argument("--negative-controls", action="store_true")
    parser.add_argument(
        "--method-mode", choices=("validation", "qualification"), default="validation"
    )
    parser.add_argument("--qualification-authorization", type=Path)
    args = parser.parse_args()
    if args.method_mode == "qualification" and (
        args.negative_controls or args.qualification_authorization is None
    ):
        parser.error(
            "Qualification needs explicitauthorization and singlepositive scope"
        )
    here = Path(__file__).resolve().parent
    source = args.source_root.resolve()
    freeze = args.freeze.resolve()
    deps = args.shared_dependencies.resolve()
    assert not any(name.startswith("tldw_chatbook") for name in sys.modules)
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in sys.modules
        for prefix in ("tldw_chatbook", "tldw_profile_core")
    )
    sys.path.insert(0, str(source))
    sys.path.insert(1, str(source / "packages/tldw_profile_core/src"))
    assert (
        Path(importlib.util.find_spec("Tests").origin).resolve().is_relative_to(source)
    )
    for name, expected in {
        "tldw_chatbook": source / "tldw_chatbook/__init__.py",
        "tldw_profile_core": source
        / "packages/tldw_profile_core/src/tldw_profile_core/__init__.py",
    }.items():
        assert (
            Path(importlib.util.find_spec(name).origin).resolve() == expected.resolve()
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
        raise RuntimeError("QA_outputs_must_stay_inside_selected_source_superpowers")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((here / "source-manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        target = (here / name).resolve()
        if (
            not target.is_relative_to(here)
            or hashlib.sha256(target.read_bytes()).hexdigest() != expected
        ):
            raise RuntimeError("Reviewed_method_source_manifest_mismatch")
    run = output / (
        "run-"
        + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-"
        + secrets.token_hex(4)
    )
    run.mkdir()
    files = (
        "collect_browser_series.py",
        "browser_series_method.py",
        "native_series_method.py",
        "native_rehearsal_observation.py",
        "passive_attach.py",
        "origin_observer.py",
        "parent_close_observer.py",
        "web_write_observer.py",
        "common_activity_observer.py",
        "qa_process_guard.py",
        "web_product_child.py",
        "approval_fixture_workload_v7.py",
        "approval_fixture_driver.py",
        "owned_roi_capture.py",
        "native_window_api.py",
        "web_smoke.py",
        "browser_viewed_gate.py",
        "browser_presented_frames.py",
        "browser_trace_capture.py",
    )
    if not set(files).issubset(manifest["sha256"]):
        raise RuntimeError("Runtime_source_not_manifest_bound")
    for name in files:
        shutil.copyfile(here / name, run / name)
    selection = {
        "method_scope": "browser_feature_single_120x40_closed_dark",
        "revision": revision,
        "method_mode": args.method_mode,
        "diagnostic_stack": False,
        "method_authorization": json.loads(
            args.qualification_authorization.read_text(encoding="utf-8")
        )
        if args.qualification_authorization
        else None,
        "run_nonce": secrets.token_hex(16),
        "shape": args.shape,
        "negative_controls": args.negative_controls,
        "mode": "negative" if args.negative_controls else "smoke",
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
        "fixture_hashes": {
            name: hashlib.sha256((run / name).read_bytes()).hexdigest()
            for name in files
        },
    }
    sys.path.insert(0, str(here))
    from browser_series_method import authorized_rounds

    planned = authorized_rounds(
        selection, run
    )  # Actualfreeze/runtime/review qualificationgates beforevenv/App.
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
        "helper_sha256": selection["fixture_hashes"],
        "qualified": False,
        "product_samples": 0,
        "GUI_launched": False,
        "method_mode": args.method_mode,
        "planned_rounds": planned,
        "ALL_QA_C_DIAGNOSTICS": False,
        "next_child_command": subprocess.list2cmdline(
            [str(interpreter), "-I", str(run / "web_smoke.py")]
        ),
    }
    (run / "prepared-environment.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
