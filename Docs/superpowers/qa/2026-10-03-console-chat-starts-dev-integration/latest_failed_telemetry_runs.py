from pathlib import Path
import json,subprocess,sys
scratch=Path(__file__).resolve().parent
log=json.loads((scratch/"latest-telemetry-owners.json").read_text())["output"]
nodes=[line.split()[1] for line in log.splitlines() if line.startswith("FAILED ")]
assert len(nodes)==61,len(nodes)
argv=[sys.executable,"-B","-m","pytest","-q","--basetemp=/private/tmp/console-latest-telemetry-controls-01a0fa6c",*nodes]
raise SystemExit(subprocess.run([sys.executable,"-B",str(scratch/"run_check.py"),"latest-telemetry-failure-controls",*argv]).returncode)
