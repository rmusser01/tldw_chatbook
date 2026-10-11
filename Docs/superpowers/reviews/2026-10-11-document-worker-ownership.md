# E4 worker ownership qualification —2026-10-11

Task implementation remains In Progress for independent controller review. Full report,
exact failed/successful scripts, commands, outputs, target timelines and SHA256 index:
`/Users/macbook-dev/.codex/worktrees/structured-document-editing/tldw_chatbook/.superpowers/sdd/2026-09-27-structured-document-editing/task-10-report.md`
and sibling `task-10-evidence/`.

The service/lease primitives own one executing snapshot and latest pending successor,
share check/format, debounce400ms, permanently retire old authority and reject unsafe
source before dispatch. Results are typed/bounded and source-safe. They do not save,
apply history or wire an editor. I2/I3 must invoke retirement on every authority,
editability, conflict and activation transition and apply through I1's captured-source
seam. Shared fixture SHA256 is unchanged:
`fcf516e4c0bb8ff7a19748a4d5f7deb8a30ad0078ea0ce2af0e1e81af50fb0d7`.

Python uses a fixed package-owned `python -I worker_process.py` entry point, an empty
environment, close_fds, neutral cwd and bounded JSON frames. The pure subpackage bootstrap
avoids app/config initializers. Actual mounted multiprocessing qualification had failed
on Textual stderr, and its fd-backed test control inherited a synthetic secret and
replayed app.py. ADR194 records Ruling39; no production stderr or parent environment
workaround was adopted. This separates app authority, not OS-user privileges.

Cancellation/deadline/close complete shielded terminate/kill/reap and pipe closure before
replacement. CPython private transport closure handles inherited pipes; other Python
implementations/OSes remain unqualified. Installed-wheel launcher and actual mounted
Textual positive controls passed on macOS CPython3.12.11 with private ruamel.yaml0.19.1.
1MiB admission0.1295ms, encoding3.2092ms, worker107.44ms; no RSS cap is claimed.

Targeted command: private interpreter `-I -m pytest
Tests/Document_Editing/test_worker_lifecycle.py Tests/Document_Editing/test_scheduler.py
Tests/Document_Editing/test_json_document.py Tests/Document_Editing/test_yaml_document.py
-q`:136passed, including21worker/scheduler tests with actual successful/stalled/crashed/
malformed/cancelled/closed/pending-init/SIGTERM-ignore/inherited-pipe children. Scoped
Ruff, Black, Bandit and mypy pass. No full suite. Wheel SHA256:
`89cb28876c6c26dd16ea38e8dc2546df4da422910bfb5a896e2cf19d8772be32`.

Browser companion limits and corrections are recorded in ADR065 and the full report:
trusted host factory, default Turbopack self-CSP failure, supported webpack qualification,
and native asynchronous termination. Those browser findings do not weaken Python reap.

## Coordinated review fix1

I1 fixed in the existing scheduler: enqueue wakes pending debounce without retaining
the replaced snapshot; immediate Check/Format dispatch and weakref release now pass.
Completed request locals are released before subsequent waits.23targeted scheduler/
worker tests plus scoped Ruff/Black/Bandit/mypy pass. Exact amended-source commands,
outputs and exits are separately retained in task10 evidence/fix1. Earlier silent
Bandit/typecheck exits remain Cannot Verify; this is not historical reconstruction.
ADR194/R39 and all downstream owner/platform gates remain unchanged. Task In Progress
for fresh scoped review.
