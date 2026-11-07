import hashlib, json, subprocess, sys, time
from pathlib import Path
root = Path.cwd()
folder = root / ".superpowers/sdd/2026-11-01-native-phased-agent-workflows-implementation"
name, *args = sys.argv[1:]
paths = sorted(set(str(p) for base in ("Tests", "tldw_chatbook") for p in Path(base).rglob("*.py")))
before = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
start = time.monotonic()
with (folder / ("final-fix-" + name + ".log")).open("wb") as log:
    proc = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT)
receipt = {"command": args, "exit": proc.returncode, "duration_seconds": time.monotonic()-start, "source_before": before, "source_changed": [p for p,h in before.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]}
(folder / ("final-fix-"+name+"-receipt.json")).write_text(json.dumps(receipt, indent=2)+"\n")
print(json.dumps({k:v for k,v in receipt.items() if k!="source_before"}))
print((folder / ("final-fix-"+name+".log")).read_text()[-5000:])
sys.exit(proc.returncode)
