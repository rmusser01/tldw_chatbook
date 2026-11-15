# Watchlist article audio: controlled-playback qualification

Date: 2026-11-15
Status: Native Windows gate failed on unpadded final-content delivery; investigation in progress
Task: [TASK-41176](../../backlog/tasks/task-41176%20-%20Qualify-ffplay-controlled-WAV-playback.md)
ADR: [366](../../backlog/decisions/366-watchlist-article-listening-and-controlled-playback.md)
Plan: [Task 1](../superpowers/plans/2026-11-15-watchlist-article-listening.md)

## Evidence boundary

This report separates process/telemetry checks, default-render endpoint loopback,
and acoustic/listener confirmation. Process exit, driver submission, and a
Windows run do not establish full playback on other platforms.

Windows WASAPI loopback captures the rendering endpoint's audio-engine stream.
It is not a microphone or independent acoustic measurement. See the
[Microsoft loopback contract](https://learn.microsoft.com/en-us/windows/win32/coreaudio/loopback-recording).
Only synthetic marker output is measured. Captured samples remain bounded in
memory and are discarded; retained receipts contain numerical measurements only,
without endpoint names or raw player output.

## Environment

- Host: Windows; native default output, stereo 48 kHz; device unmuted at preflight.
- ffplay: `7.1-essentials_build-www.gyan.dev`, direct executable rather than Chocolatey shim.
- Executable SHA-256: `f6536c4caf9bac0fee1e8ffaafa5b13b6ad3e386a9a9f0b6613360e8c6d59919`.
- Probe interpreter: Python 3.12.10; existing NumPy 2.2.6, SciPy 1.15.3 and CFFI 2.1.1.
- Optional capture helper: SoundCard 0.4.6, installed only in this plan's ignored workspace; no application/venv dependency change.
- Exact default-speaker identity matched its loopback endpoint; stereo channels verified before recording.

The [SoundCard API](https://soundcard.readthedocs.io/en/latest/) supplies the
native wrapper; capture uses both Windows channels rather than its documented
problematic mono route. Availability of that wrapper alone is not a passing run.

## Preliminary real-player probes

These are telemetry/process preflights, **not seek or audible-completion passes**.
Both used owned direct ffplay processes, `-nodisp -autoexit -stats -sync audio`,
quiet PCM16 sine fixtures, and bounded process cleanup. No loopback was captured
for these two preflights.

| Case | Observed result | Meaning |
| --- | --- | --- |
| 500 ms file, start 0 | Exit 0; 13 numeric records from -0.03 s to 0.39 s | Ignore the negative startup clock; final stats do not establish final output |
| 2 s file, start 1.125 s | Exit 0; first clocks 1.08, 1.12, 1.15, 1.20 s; last 1.84, 1.88, 1.91, 1.95 s | Clock already includes requested media offset; do not add it twice |

The [ffplay documentation](https://ffmpeg.org/ffplay.html) describes seeking and
statistics but promises neither exact seeking nor a stable machine protocol.

## Native gate matrix

| Requirement | Windows | macOS | Linux |
| --- | --- | --- | --- |
| Requested seek versus captured source position, <=250 ms | Pass in seven matched runs: absolute error <=2.42 ms; unmatched cases remain unqualified | Unqualified: no native host run | Unqualified: no native host run |
| Repeated resume to retained audio | Mixed: three matched nonzero seeks pass; one repeated seek unqualified | Unqualified | Unqualified |
| 120 ms / 500 ms clips and near-EOF seek | Short clips match their starts but miss final content; near-EOF unmatched/unqualified | Unqualified | Unqualified |
| Final marker and normal completion | Fail: all six matched natural runs miss 30.4-73.1 ms | Unqualified | Unqualified |
| Telemetry versus media position; invalid/stale rejection | <=49 ms error in matched runs; deterministic parser controls pass | Unqualified | Unqualified |
| Stop during startup/playback and observed silence | Both pass initially; high-resolution playing-Stop follow-up returns process in about 5 ms and observes silence; children reaped | Unqualified | Unqualified |
| Acoustic/listener confirmation | Unqualified: none recorded | Unqualified | Unqualified |

Do not advance dependent feature implementation while the required gate is
unqualified. Fill results from actual run receipts, not parser tests or a dummy
audio driver. TASK-41176 remains In Progress.

## Windows native measurements

The first matrix contains **10 runs**, not ten passes: eight natural-completion
cases and two explicit Stop cases. Of the natural runs, six matched their source
but failed the 20 ms final-content tolerance; one repeated seek and the near-EOF
case could not be matched and remain unqualified. The playing-Stop case also
matched, giving seven measured seek/telemetry runs in total. Startup Stop
prevented a matching marker and is not a seek pass.

| Matched natural case | Requested start | Missing final content |
| --- | --- | --- |
| Start, 2.5 s source | 0 | 30.375 ms |
| Resume 1 | 0.8 s | 50.250 ms |
| Resume 2 | 1.4 s | 73.104 ms |
| Resume 2 repeat | 1.4 s | 60.479 ms |
| 120 ms source | 0 | 40.188 ms |
| 500 ms source | 0 | 50.208 ms |

The fixture has no fade or trailing silence. Independent full-source and truncated
source controls, including a real PCM16 stereo-WAV round trip, recover complete
tails within approximately 2.5 ms and detect a 50 ms truncation. The native
failure cannot be explained by a deliberate fixture fade or basic detector
cutoff alone.

A paired native control changed **only 250 ms of trailing silence after the same
2.5 s content**. Unmodified output again missed 30.375 ms; padded output recovered
the content end to +1.625 ms and passed the tail check. Both had matching source
correlations above 0.99999, successful process exits, and confirmed reaping of
player and capture workers. This supports an exit/output-drain explanation;
it does not prove where the hardware or capture path drops the tail. Padding is
an experimental control, not an adopted application workaround or a passing
cross-platform backend.

The initial Windows Python 3.12 `monotonic` clock uses GetTickCount64 at 15.625 ms
resolution, so recorded zero stop durations mean below that clock's resolution,
not instantaneous shutdown. Numerical source-position matches are independent
of that clock. Final tooling uses QueryPerformanceCounter (100 ns reported
resolution). Its **three-case follow-up** measured an unmodified tail 10.250 ms
short (inside the 20 ms tolerance, correlation 0.791 and alignment drift 16 ms),
padded content end +1.625 ms (correlation above 0.99999), and playing-Stop process
return in 5.016 ms with subsequent captured silence. All three player/capture
workers were reaped. Captured block-completion timestamps are not hardware sample
timestamps, so sub-block silence timing is approximate.

The unmodified end therefore varies by run; the original six failures cannot be
averaged away by the later passing control. Two padded controls are encouraging
but insufficient to adopt a drain-safe production strategy. No workaround was
added to application code.

Ignored numerical receipts: `windows-run-1/qualification.json`,
`windows-tail-control/control.json`, and `windows-qpc-control/control.json` under
this plan's workspace. This is ten matrix runs plus two paired controls and three
high-resolution follow-ups; the two preliminary telemetry probes are separate.
No captured waveforms or raw stderr were saved.

## Deterministic verification

On initial tooling commit `797bd1284b`, the exact unittest command below passed
13 tests in 3.091 s in the controller's check. Review then caught an explicit-path
Chocolatey launcher bypass; `fc311bf150` fixes it and adds device-free tests for
default/custom shim rejection and direct-payload acceptance. The amended targeted
suite passes **15 tests**, with Ruff check and format passing. Existing native
runs used the direct payload and are unaffected by the resolution fix.
The implementer recorded initial RED (six expected assertion
failures across nine tests) before implementing parser/matching behavior.
Additional tests cover PCM16 stereo round-trip, delayed/resampled known offsets,
full versus truncated tails, nonfinite/unrelated/silent input, oversized records,
stale/retired observations, and no-capture refusing to launch even an invalid
supplied player. These checks validate the measuring tool, not native playback.

## Reproduction and focused checks

Tool source and adjacent tests: `tools/evidence/watchlist_article_audio/`.
Use the verified Python environment with the isolated helper directory on
`PYTHONPATH`. Run deterministic tests and static checks first:

```text
python -m unittest discover -s tools/evidence/watchlist_article_audio -p test_qualification.py -v
python -m ruff check tools/evidence/watchlist_article_audio
python -m ruff format --check tools/evidence/watchlist_article_audio
python tools/evidence/watchlist_article_audio/qualification.py --capture-loopback --ffplay <direct-executable> --output-dir <this-plan-ignored-workspace/run-directory>
```

The native invocation plays short quiet markers. Without `--capture-loopback`,
the tool cannot award a native-audio pass. Keep fixture WAVs, isolated packages
and numeric run receipts under `.superpowers/sdd/2026-11-15-watchlist-article-listening/`;
commit only recipes, code/tests and this sanitized report.
