import json, os, subprocess, sys, tempfile
from pathlib import Path
root = Path(__file__).resolve().parents[3]
evidence = Path(__file__).resolve().parent
name, *args = sys.argv[1:]
if name.endswith("-base"):
    root = Path(json.loads((evidence / "task-1-base-tree.json").read_text())["root"])
profile = Path(tempfile.mkdtemp(prefix=name + "-profile-", dir=evidence))
env = dict(os.environ)
env.update(TLDW_TEST_CONFIG_ROOT=str(profile), PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1")
cmd = [sys.executable, *args]
if args[:2] == ["-m", "pytest"]:
    cmd += ["--basetemp=" + str(profile / "pytest")]
log = evidence / (name + ".log")
with log.open("w") as out:
    result = subprocess.run(cmd, cwd=root, env=env, stdout=out, stderr=subprocess.STDOUT)
record = {"name": name, "command": cmd, "cwd": str(root), "revision": "f843ca811f01da6c39d903b6cd7328d68d50416f" if name.endswith("-base") else subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(), "exit": result.returncode, "log": str(log)}
with (evidence / "task-1-commands.jsonl").open("a") as out:
    out.write(json.dumps(record) + "\n")
print(json.dumps(record))
print(log.read_text()[-3500:])
sys.exit(result.returncode)
