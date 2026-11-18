"""Package reviewed diagnostic sources; import no product and launch no worker."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from types import CodeType
import zipfile


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def module_path(module):
    return (
        ("" if module.startswith("Tests.") else "tldw_chatbook/")
        + module.replace(".", "/")
        + ".py"
    )


def targets(raw):
    return next(
        ast.literal_eval(node.value)
        for node in ast.parse(raw).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(item, ast.Name) and item.id == "TARGETS" for item in node.targets
        )
    )


def code_at(code, path):
    for name in path.replace("::", ".").split("."):
        [code] = [
            value
            for value in code.co_consts
            if type(value) is CodeType and value.co_name == name
        ]
    return code


def signature(code):
    return (
        code.co_code,
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        code.co_flags,
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        tuple(
            signature(value) if type(value) is CodeType else value
            for value in code.co_consts
        ),
    )


def reviewed_modal_ordering(module, function, reference, candidate):
    """The v3 parent span still times entry/exit when a queued rail save defers a modal."""
    if (module, function) != ("UI.Console_Modules.execution_decisions", "present_selected_decision"):
        return False

    def digest(raw):
        node = next(node for node in ast.parse(raw).body if getattr(node, "name", None) == function)
        return sha(ast.dump(node).encode())

    return (digest(reference), digest(candidate)) == (
        "b4b75aa0bb562adc772da97a8f3a13e9097c1e710a43a4e1ea2c928e0800b024",
        "558845bc5082e691a71e15f8f10d2f110d8fb9db936f45c9663d7ffbaed817ef",
    )


def build(
    foreground,
    approval,
    baseline,
    candidate,
    compatibility,
    output,
    approval_revision=2,
    reviewed_candidate=None,
):
    declared_revision = next(
        (
            ast.literal_eval(node.value)
            for node in ast.parse((approval / "observer.py").read_bytes()).body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(item, ast.Name) and item.id == "PROBE_REVISION"
                for item in node.targets
            )
        ),
        2,
    )
    if declared_revision != approval_revision:
        raise ValueError("Approval revision does not match its original monitor source")
    roots = {
        "baseline": baseline,
        "candidate": candidate,
        "compatibility": compatibility,
    }
    manifests = {
        role: json.loads((root / "replay-manifest.json").read_bytes())
        for role, root in roots.items()
    }
    if (
        manifests["baseline"]["lab_observer"]["route"] != "local"
        or manifests["candidate"]["lab_observer"]["route"] != "process"
    ):
        raise ValueError(
            "Expected the original local baseline and selected process candidate"
        )
    reviewed_candidates = {
        2: "aab319544eb295d5b6ef78daab1bd8674743a839",
        3: "997b3339ffd37fab54dc28054ed4987fb933e478",
    }
    if reviewed_candidate is None:
        reviewed_candidate = reviewed_candidates[approval_revision]
    elif (
        approval_revision != 3
        or not re.fullmatch(r"[0-9a-f]{40}", reviewed_candidate)
        or manifests["compatibility"]["candidate_commit"] != reviewed_candidates[3]
    ):
        raise ValueError("New v3 source pins require the frozen q3 compatibility reference")
    if manifests["candidate"]["candidate_commit"] != reviewed_candidate:
        raise ValueError(
            "This probe revision is reviewed only for its recorded frozen candidate"
        )
    files, code_cache, checked = {}, {}, {}

    def source(role, module):
        key = role, module
        if key not in code_cache:
            name = module_path(module)
            raw = (roots[role] / name).read_bytes()
            if sha(raw) != manifests[role]["candidate_sources"][name]:
                raise ValueError("Frozen source changed: " + name)
            code_cache[key] = raw, compile(raw, name, "exec")
        return code_cache[key]

    def pins(selection, role):
        result = {}
        for module, function, label in selection:
            raw, compiled = source(role, module)
            result[module] = sha(raw)
            code = code_at(compiled, function)
            if role == "candidate":
                reference, old_module = source("compatibility", module)
                old = code_at(old_module, function)
                identical = signature(code) == signature(old)
                if not identical and not (
                    approval_revision == 3 and reviewed_modal_ordering(module, function, reference, raw)
                ):
                    raise ValueError(
                        "Review changed monitored definition before packaging: "
                        + module
                        + ":"
                        + function
                    )
                checked[module, function] = {
                    "module": module,
                    "function": function,
                    "reference_line": old.co_firstlineno,
                    "candidate_line": code.co_firstlineno,
                    "equal_code_constants_names_closures_arguments": identical,
                    "module_source_changed": source("candidate", module)[0]
                    != source("compatibility", module)[0],
                }
            required = {
                "runtime_trace": {"kind", "kw"},
                "tool_timeout_owner": {"fn"},
                "workspace_execute": {"operation"},
                "journey_mark": {"self", "event", "data"},
            }
            if not required.get(label, set()) <= set(code.co_varnames):
                raise ValueError("Original metadata arguments changed: " + label)
            if label == "tool_admitted_call" and "fn" not in code.co_freevars:
                raise ValueError("Original worker callable correlation changed")
            if label == "tool_worker" and "_admitted_call" not in code.co_freevars:
                raise ValueError("Original worker dispatch changed")
        return result

    for name, folder, role in (
        ("foreground", foreground, "candidate"),
        ("foreground/parent", foreground / "parent", "candidate"),
        ("foreground/baseline", foreground / "baseline", "baseline"),
    ):
        raw = (folder / "observer.py").read_bytes()
        files[name + "/observer.py"] = raw
        files[name + "/source-pins.json"] = encoded(pins(targets(raw), role))
    for name in (
        "deferred.py",
        "install_tool_foreground.py",
        "replace_foreground_approval.py",
        "verify_static.py",
        "verify_offline.py",
        "summarize.py",
    ):
        files["foreground/" + name] = (foreground / name).read_bytes()
    for name, folder, role in (
        ("approval/baseline", approval / "baseline", "baseline"),
        ("approval/candidate-ui", approval / "candidate-ui", "candidate"),
        (
            "approval/candidate-ui/backend",
            approval / "candidate-ui/backend",
            "candidate",
        ),
    ):
        plan = json.loads((folder / "targets.json").read_bytes())
        files[name + "/observer.py"] = (folder / "observer.py").read_bytes()
        files[name + "/source-pins.json"] = encoded(pins(plan["targets"], role))
        plan["coordinates"] = {
            label: code_at(source(role, module)[1], function).co_firstlineno
            for module, function, label in plan["targets"]
        }
        files[name + "/targets.json"] = encoded(plan)
    for name in ("observer.py", "prepare.py", "verify_static.py"):
        files["approval/" + name] = (approval / name).read_bytes()
    files["README.md"] = Path(__file__).with_name("DIAGNOSTICS.md").read_bytes()
    for name in (
        "analyze_diagnostics.py",
        "test_analyze_diagnostics.py",
        "build_diagnostic_archive.py",
    ):
        files[name] = Path(__file__).with_name(name).read_bytes()
    files[
        "verify_bundle.py"
    ] = b'''"""Check extracted source hashes and syntax; no product imports."""
import ast
import hashlib
import json
from pathlib import Path
root = Path(__file__).resolve().parent
manifest = json.loads((root / 'bundle-manifest.json').read_bytes())
for name, digest in manifest['files'].items():
    path = root / name
    if not path.resolve().is_relative_to(root):
        raise ValueError('Escaping archive member')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('Changed archive member: ' + name)
    if name.endswith('.py'):
        compile(ast.parse(raw), name, 'exec')
print(json.dumps({'members_verified': len(manifest['files']), 'native_launched': False}))
'''
    files["bundle-manifest.json"] = encoded(
        {
            "format": 1,
            "source_only": True,
            "qualifying": False,
            "baseline_original_commit": manifests["baseline"]["candidate_commit"],
            "candidate_commit": manifests["candidate"]["candidate_commit"],
            "candidate_full_diagnostic_run": False,
            "approval_probe_revision": approval_revision,
            "baseline_diagnostic_run": "foreground-b1-20261117"
            if approval_revision == 2
            else None,
            "compatibility_reference_commit": manifests["compatibility"][
                "candidate_commit"
            ],
            "candidate_monitored_targets_checked": len(checked),
            "candidate_compatibility_checks": [checked[key] for key in sorted(checked)],
            "equal_code_constants_names_closures_arguments": all(
                item["equal_code_constants_names_closures_arguments"] for item in checked.values()
            ),
            "fixed_metadata_contracts_checked": True,
            "files": {name: sha(raw) for name, raw in sorted(files.items())},
        }
    )
    for name, raw in files.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Escaping archive member")
        if re.search(
            r"(?i)(?<![a-z0-9_])[a-z]:[\\/]|/[h]ome/|/[U]sers/|[.]codex[\\/]",
            raw.decode("utf-8-sig"),
        ):
            raise ValueError("Host path in portable source: " + name)
        if name.endswith(".py"):
            compile(ast.parse(raw), name, "exec")
    with zipfile.ZipFile(
        output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, raw in sorted(files.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.create_system, info.external_attr = 3, 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, raw)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC validation failed")
    checksum = sha(output.read_bytes())
    output.with_suffix(".sha256").write_text(
        checksum + "  " + output.name + "\n", encoding="ascii"
    )
    print(
        json.dumps(
            {
                "archive": output.name,
                "bytes": output.stat().st_size,
                "members": len(files),
                "sha256": checksum,
                "candidate_targets_checked": len(checked),
                "native_launched": False,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "foreground-artifact",
        "approval-artifact",
        "baseline-replay",
        "candidate-replay",
        "compatibility-replay",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approval-revision", type=int, choices=(2, 3), default=2)
    parser.add_argument(
        "--reviewed-candidate",
        help="Exact new v3 candidate commit to compare with every monitored q3 definition.",
    )
    args = parser.parse_args()
    build(
        args.foreground_artifact,
        args.approval_artifact,
        args.baseline_replay,
        args.candidate_replay,
        args.compatibility_replay,
        args.output,
        args.approval_revision,
        args.reviewed_candidate,
    )
