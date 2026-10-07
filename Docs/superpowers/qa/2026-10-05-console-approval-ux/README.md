# Task 1 partial measurement contract — no transport timings captured

Characterization source revision: `b0ac6672231e29fb7efa2e9236d2f123c2e0a163`.
The native and browser JSON files are **unqualified status receipts**, not latency
baselines. Zero timing samples were collected. Neither budget has been verified.

The pure recorder accepts defined stages/clocks, internally minted correlations
and integer timestamps only. Its local store is bounded to 4096 markers and 4096
correlations. Summaries reject extra fields and invalid values; separate first
use from warm samples; require at least 40 warm samples; use nearest-rank p95;
retain missing boundaries; reject uncalibrated clocks and compositor/mixed
transport qualification. `qualified` describes supplied measurement completeness,
not passing 100/200 ms targets or independent proof of presented frames.

The recorder is not yet wired to runtime/transport observations. There is no
capture CLI, sampled UI stack attribution, real-app approval journey or measured
last-output-to-card gap in this partial deliverable. Task5 now supplies optional grant-success and backend-start owner observations;
the timing recorder is still not wired to runtime/transport capture. No
performance cause has been measured.

## Reproduce the safe Windows control

From this isolated checkout, using the installed primary Python 3.12 environment:

```powershell
& 'C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe' 'Docs/superpowers/qa/2026-10-05-console-approval-ux/private_control.py'
& 'C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe' 'Docs/superpowers/qa/2026-10-05-console-approval-ux/private_control.py' 'Tests/Benchmarks/test_console_approval_latency_measurement.py'
```

The launcher installs the existing real-profile guard against the original
owner home before redirecting HOME/USERPROFILE. It selects a fresh private
config/home/data/TEMP/TMP tree before product imports, pins module origins,
uses the null keyring, offline models, and existing test network denial.
The delivered-launcher logs record **25 control cases passed** and **16 recorder
cases passed**, with one existing Pydantic warning each. No full suite ran.
The older green logs retain the initial scratch launcher; the red log records
six expected missing-recorder assertions before implementation.

## Failed launch attribution

`selection-diagnostic.txt` preserves content-free relative-selector diagnostics.
Redirecting USERPROFILE before first guard installation on Windows caused
`real_profile_guard._real_home()` to identify the private home as real. The root
conftest bound that config; UI conftest then rejected the supplied root as
containing that home and selected another temporary config. `app.py`'s module
config load raised `RecoveryRequired("raw_source_selection_changed")` at the
unchanged raw participant binding check. No approval test ran in that attempt.
The supported guard-before-redirection ordering passed. The two earlier
`recovery_scope_uncertain` attempts are not attributed by this separate incident.

## External qualification gaps

Native qualification needs an existing owner-approved terminal presentation
observer: real input arrival and visible terminal frame timestamps identifying
this harmless private fixture, in one monotonic clock or a calibrated clock pair
with bounded error. ConPTY bytes or parsed cells alone do not provide this.
The existing terminal qualification scripts test ConPTY I/O, not presented paint;
`winpty` is not installed in the available interpreter. Native computer APIs
are disabled for this browser runtime; no alternate capture framework was built.

The in-app browser offers screenshots (pixels) and read-only page evaluation,
but its documented API provides no presented-frame timestamp or server/browser
clock calibration. It cannot install page event/paint observers through that
read-only surface. Screenshot request/return times, requestAnimationFrame alone,
and server frame emission would not prove actual presented paint. Browser capture
was not run. A supported presented-frame observer and bounded calibration channel
are needed before claiming browser timing qualification.

Task 34564 remains In Progress under ADR-221 and the controller's revised execution
order. Final transport qualification and all size/theme/Inspect/scenario matrices
remain open. No speed improvement or complete spec compliance is claimed.

## Review amendment: refused startup stops collection

The launcher stores and prints startup admission, then exits unsuccessfully
before importing pytest when admission is denied. The focused regression uses
owned malformed recovery evidence and the unchanged admission reader; its
import sentinel proved the former failure and now stays untouched on refusal.
Run it through `private_control.py` with target
`Tests/Benchmarks/test_console_approval_private_control.py`.
`refusal-regression-red.log` records 1 expected failure; its green receipt records
1 passing case. The refreshed delivered-launcher logs retain 16 passing recorder
checks and 25 passing approval controls. Transport qualification is still open.

Task3 durable receipts: [Task3 scoped controls QA](task-3/README.md). Private workflow artifacts remain untracked.

## Final integration status

Task6 records updated guides, captured-view artwork, targeted checks and honest
qualification limits under [final qualification](task-6/README.md). The current
read-only capture preflight found no available in-app browser and native
computer APIs are disabled. Neither transport has qualified samples. The earlier
paragraph describes API limitations, not a successful browser launch.

All controller decisions and their costs are preserved in
[execution decisions](task-6/execution-decisions.md). No full suite, release,
permission-policy migration or guard bypass was performed.
