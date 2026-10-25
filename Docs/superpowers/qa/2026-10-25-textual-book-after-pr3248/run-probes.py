from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, xml.etree.ElementTree as ET

ROOT = Path.cwd()
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tldw_chatbook.Utils import platform_files

label = sys.argv[1]
if not label.replace("-", "").isalnum():
    raise ValueError("Use a fresh simple label")
BASE = ROOT / ".test-tmp" / ("textual-after-3248-" + label)
if not BASE.parent.exists():
    platform_files.os.mkdir(BASE.parent, 0o700)
platform_files.os.mkdir(BASE, 0o700)
git = ["git", "-c", "safe.directory=" + ROOT.as_posix()]
pins = json.loads((SOURCE / "source-pins.json").read_text())
for path, expected in pins["files"].items():
    current = subprocess.check_output(
        git + ["hash-object", "--path", path, path], text=True
    ).strip()
    if current != expected["blob"]:
        raise RuntimeError("Historical audit probe source changed: " + path)
summary = {}
(BASE / "source-context.json").write_text(
    json.dumps(
        dict(
            current_head=subprocess.check_output(
                git + ["rev-parse", "HEAD"], text=True
            ).strip(),
            audited_dev=pins["audited_dev"],
            source_blobs_match=True,
        ),
        indent=2,
    ),
    encoding="utf-8",
)
domains = sys.argv[2:] or ["study", "watchlists", "personas", "research"]
assert len(domains) == len(set(domains)) and set(domains) <= {
    "study",
    "watchlists",
    "personas",
    "research",
}
for domain in domains:
    run = BASE / domain
    platform_files.os.mkdir(run, 0o700)
    for child in ("tmp", "bootstrap"):
        platform_files.os.mkdir(run / child, 0o700)
    shutil.copyfile(SOURCE / "conftest.py", run / "conftest.py")
    test = run / ("test_" + domain + ".py")
    shutil.copyfile(SOURCE / test.name, test)
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        TMP=str(run / "tmp"),
        TEMP=str(run / "tmp"),
        TLDW_TEST_CONFIG_ROOT=str(run / "bootstrap"),
    )
    command = [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "pytest",
        str(test),
        "-p",
        "pytest_asyncio.plugin",
        "-p",
        "pytest_timeout",
        "-q",
        "--tb=short",
        "--timeout=180",
        "--basetemp=" + str(run / "pytest"),
        "-o",
        "cache_dir=" + str(run / "cache"),
        "--junitxml=" + str(run / "result.xml"),
    ]
    (run / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    process = subprocess.run(
        command, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8"
    )
    (run / "run.log").write_text(
        process.stdout + "\n" + process.stderr, encoding="utf-8"
    )
    reports = []
    for path in run.rglob("*.xml"):
        tree = ET.parse(path)
        suites = list(tree.iter("testsuite"))
        row = dict(
            path=path.relative_to(BASE).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            tests=sum(int(s.get("tests", 0)) for s in suites),
            failures=sum(int(s.get("failures", 0)) for s in suites),
            errors=sum(int(s.get("errors", 0)) for s in suites),
            skipped=sum(int(s.get("skipped", 0)) for s in suites),
            seconds=sum(float(s.get("time", 0)) for s in suites),
        )
        messages = [e.get("message", "") for e in tree.iter("failure")]
        row["intended_study_failure"] = (
            len(messages) == 1
            and "old server Submit discarded the current local attempt" in messages[0]
        )
        reports.append(row)
    parents = [r for r in reports if r["path"] == domain + "/result.xml"]
    assert len(parents) == 1, (
        domain,
        process.returncode,
        "Missing parent XML; inspect run.log",
        reports,
    )
    parent = parents[0]
    children = [r for r in reports if "private-profile-process" in r["path"]]
    count = {"study": 2, "watchlists": 3, "personas": 2, "research": 2}[domain]
    assert (
        parent["tests"] == count
        and len(children) == count
        and all(r["tests"] == 1 for r in children)
    ), (domain, reports)
    assert all(r["errors"] == 0 and r["skipped"] == 0 for r in reports), (
        domain,
        reports,
    )
    if domain == "study":
        assert (
            process.returncode == 1
            and parent["failures"] == 1
            and sum(r["failures"] for r in children) == 1
            and sum(r["intended_study_failure"] for r in children) == 1
        ), reports
    else:
        assert process.returncode == 0 and all(r["failures"] == 0 for r in reports), (
            domain,
            reports,
        )
    summary[domain] = dict(exit_code=process.returncode, reports=reports)
    print(
        json.dumps(
            dict(
                domain=domain,
                tests=count,
                expected_failures=parent["failures"],
                errors=0,
            )
        ),
        flush=True,
    )
(BASE / "verification.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(
    "Qualified selected current-behavior observations; Study expects its named retention failure. These are not repair acceptance tests. Artifacts: "
    + str(BASE)
)
