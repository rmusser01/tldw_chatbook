from pathlib import Path
import json,subprocess,sys
root=Path('.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration')
prior=json.loads((root/'latest-telemetry-failure-controls.json').read_text())
nodes=[line.split()[1] for line in prior['output'].splitlines() if line.startswith('FAILED ')]
assert len(nodes)==14
argv=[sys.executable,'-B','-m','pytest','-q','--basetemp=/private/tmp/console-latest-telemetry-controls2-01a0fa6c',*nodes]
raise SystemExit(subprocess.run([sys.executable,'-B',str(root/'run_check.py'),'latest-telemetry-remaining-controls',*argv]).returncode)
