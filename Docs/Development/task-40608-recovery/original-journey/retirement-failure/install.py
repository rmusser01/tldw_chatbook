"""Install only in a verified disposable process replay; never launch it."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PLUGIN = "task40608_lab_observer.py"
SETTINGS = "task40608-lab-observer.json"
HELPER = "task40608_retirement_failure_observer.py"
INSTALLER = "task40608_install_retirement_failure_observer.py"
ENTRY = '                runpy.run_module("tldw_chatbook.Chat.console_execution_worker", run_name="__main__")'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic(path, data):
    temporary = path.with_name(path.name + ".failure-new")
    with temporary.open("xb") as output:
        output.write(data)
    os.replace(temporary, path)  # Replace a hardlink, never edit its shared inode.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-directory", required=True, type=Path)
    parser.add_argument("--replay-tools", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    sys.path[:0] = [str(HERE), str(args.replay_tools.resolve(strict=True))]
    from observer import SITES, checked_site
    from replay import verify

    root = args.candidate_directory.resolve(strict=True)
    pins = json.loads((HERE / "pins.json").read_bytes())
    helper = (HERE / "observer.py").read_bytes()
    if sha(helper) != pins["observer_sha256"]:
        raise RuntimeError("Reviewed observer changed")
    manifest = verify(root)
    if manifest["lab_observer"]["route"] != "process":
        raise RuntimeError("Original process guard required")
    settings = json.loads((root / SETTINGS).read_bytes())
    if settings.get("backend_failure_diagnostic") or settings.get("retirement_failure_diagnostic"):
        raise RuntimeError("Overlapping failure observer refused")
    for name in SITES:
        checked_site(root, pins["sources"], name)
    for relative, expected in pins["trace_sources"].items():
        if sha((root / relative).read_bytes()) != expected:
            raise RuntimeError("Traceback source pin changed")
    original = (root / PLUGIN).read_bytes()
    text = original.decode("utf-8-sig")
    if text.count(ENTRY) != 1 or (root / HELPER).exists() or (root / INSTALLER).exists():
        raise RuntimeError("Original guarded entry unavailable")
    changed = text.replace(ENTRY,
        '                from task40608_retirement_failure_observer import failure_facts\n'
        '                with failure_facts(_ROOT, settings):\n'
        '                    runpy.run_module("tldw_chatbook.Chat.console_execution_worker", run_name="__main__")', 1).encode()
    ast.parse(changed)
    print(json.dumps({"check_only": args.check_only, "qualifying": False,
                      "observer_sha256": sha(helper), "original_plugin_sha256": sha(original),
                      "sources": pins["sources"]}))
    if args.check_only:
        return
    backup = root / "retirement-failure-preimage"
    backup.mkdir()
    for name in (PLUGIN, SETTINGS, "replay-manifest.json"):
        (backup / name).write_bytes((root / name).read_bytes())
    settings["retirement_failure_diagnostic"] = True
    settings["failure_trace_sources"] = pins["trace_sources"]
    settings["sources"].update(pins["sources"])
    settings["sources"].update({PLUGIN: sha(changed), HELPER: sha(helper)})
    changes = {PLUGIN: changed, HELPER: helper, INSTALLER: Path(__file__).read_bytes(),
               SETTINGS: (json.dumps(settings, indent=2) + "\n").encode()}
    for name, data in changes.items():
        atomic(root / name, data)
        manifest["candidate_sources"][name] = sha(data)
        manifest["diagnostic_sources"][name] = sha(data)
    manifest["retirement_failure_diagnostic"] = {
        "qualifying": False, "product_sources_changed": False,
        "original_manifest_sha256": sha((backup / "replay-manifest.json").read_bytes()),
        "guard_plugin_preimage_sha256": sha(original), "source_pins": pins["sources"],
        "installer_sha256": sha(Path(__file__).read_bytes()),
    }
    atomic(root / "replay-manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    verify(root)
    print(json.dumps({"installed": True, "qualifying": False,
                      "manifest_sha256": sha((root / "replay-manifest.json").read_bytes())}))


if __name__ == "__main__":
    main()
