from pathlib import Path
import sys
project = Path(sys.argv[1])
value = (project / 'fixture.txt').read_text()
print('checked fixture.txt: ' + value.strip())
print('validation provenance: trusted check.py', file=sys.stderr)
raise SystemExit(0 if value == 'valid\n' else 7)
