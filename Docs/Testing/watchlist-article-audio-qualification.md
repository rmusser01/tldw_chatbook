# Watchlist article audio: controlled-playback qualification

Date: 2026-11-23
Status: Windows endpoint/listener passed; Linux matrix unqualified; portable gate open
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

## Windows environment

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

## Current packet-based result, 2026-11-23

After the user muted the active background session, **all 21 cases passed
Windows endpoint-loopback qualification**. The same source revision
`b66e13d0ca` ran three rounds with 250 ms WAV padding and a 4,800-frame
capture buffer. All original thresholds remained unchanged: 250 ms seek and
bookmark error, 20 ms tail and alignment drift, correlation at least 0.70,
0.0002 quiet RMS, and the 1.5 s Stop bound.

| Case | Passing rounds |
| --- | --- |
| Fixed nonzero seek, 0.8 s | 3/3 |
| 120 ms clip | 3/3 |
| 500 ms clip | 3/3 |
| Near EOF, 140 ms remaining | 3/3 |
| Near EOF, 60 ms remaining | 3/3 |
| Playing Stop, bookmark alignment and post-stop silence | 3/3 |
| Resume from that Stop's saved frame against the identical WAV | 3/3 |

Maximum absolute seek error was 3.250 ms; natural-content tail error was
2.417 ms. All three Stops interrupted original content before logical EOF.
Maximum bookmark-versus-stopped-source error was 69.542 ms, process Stop
latency 10.003 ms, and last detected signal after Stop 32.633 ms. All 21
telemetry comparisons passed, with maximum absolute error 25.492 ms.
Saved resumes used frames 36,480, 35,520 and 36,000 at 48 kHz and the
identical retained artifact digest.

Every prelaunch and post-stop quiet measurement had RMS 0.0. All source spans
were contiguous, with zero alignment drift, zero later discontinuities and
zero timestamp-error flags. First-packet origin flags remain counted
separately. All 21 players and capture workers were reaped. The
[complete passing 21-case result](evidence/watchlist-article-audio/windows-wasapi-packet-21-session-muted.json)
retains the per-case measurements and retirement evidence.

The fixed 250 ms profile passes for this Windows build, PCM16 stereo 48 kHz,
ffplay executable and default-render route. Native macOS/Linux and other output
routes remain unqualified. The Windows listener check below passed. The portable
feature gate remains open and dependent implementation remains gated. No application
playback behavior or runtime dependency changed.

### Preserved background-audio rejections

The [first packet matrix](evidence/watchlist-article-audio/windows-wasapi-packet-21.json)
launched 18 base players, rejected every prelaunch baseline at RMS
0.004150–0.018304, and skipped three dependent resumes. A
[retry after the initial pause](evidence/watchlist-article-audio/windows-wasapi-packet-21-pause-retry.json)
had the same 18 rejections and three skips at RMS 0.001626–0.009073.
Read-only session enumeration still showed an active render session, with
nonzero peaks approximately 0.054 before the first run and 0.017 after the
retry. An independent code audit confirmed that baseline sampling ends
before player launch and excludes samples outside that interval.

Both rejected runs had zero later discontinuities/timestamp errors and all
started resources retired. They provide no qualified seek, tail or Resume
result. Muting that active session produced the passing matrix above; no
baseline rejection was converted into a pass. Previous wrapper-based
matrices and the reproduced capture mechanisms remain below.

## Windows listener confirmation, 2026-11-23

The user listened to a repeated seven-case sequence on the tested Windows
output: nonzero seek, 120/500 ms clips, 140/60 ms remaining near EOF, playing
Stop, and saved-position Resume. The user confirmed: "Yes—audible, clean Stop,
final clip finished". This is a human audible-playback observation, separate
from the numerical tail/position measurements and from instrumented acoustic
measurement. No microphone or captured acoustic recording was used.

The [observed second sequence](evidence/watchlist-article-audio/windows-listener-round-2.json)
passed all seven Windows endpoint cases and reaped every player/capture.
The [listener receipt](evidence/watchlist-article-audio/windows-listener-confirmation.json)
records the explicit response. The user missed the
[first sequence](evidence/watchlist-article-audio/windows-listener-round-1.json),
which also retained one unqualified baseline observation; it is not counted
as a listener pass. These listener checks do not qualify Linux or macOS.

## Native Linux result, 2026-11-23

**The Linux aggregate remains unqualified.** The same three-round plan
produced 21 rows and 18 player launches. All 18 retained source observations
matched the marker and had an observed quiet end boundary with zero waveform
drift. These are explicitly **unattributed monitor measurements**, not
transport passes. The fifteen natural observations had maximum source-start
delta 2.438 ms and source-end delta 2.417 ms. Minimum edge correlation was
0.9999945. Every prelaunch baseline had zero RMS. All 18 process/reaping
checks passed; three dependent resumes were skipped because their Stop
prerequisites were unqualified. No criterion was relaxed.

| Linux case | Observed marker matches | Qualified transport result |
| --- | --- | --- |
| Fixed 0.8 s seek | 3/3 | Unqualified |
| 120 ms clip | 3/3 | Unqualified |
| 500 ms clip | 3/3 | Unqualified |
| Near EOF, 140 ms remaining | 3/3 | Unqualified |
| Near EOF, 60 ms remaining | 3/3 | Unqualified |
| Playing Stop | 3/3 | Unqualified |
| Saved Resume | Not launched | Three prerequisite skips |

Pulse's asynchronous startup events do not establish a complete initial
player-route history, and its signed latency replies do not bound the age
of the process snapshot. Endpoint/seek/tail grades therefore stay
unqualified, and absolute telemetry/post-Stop timing stays unqualified with
`pulse_snapshot_age_unqualified`. The maximum measured **process** Stop
latency was 7.245 ms; this is independent of sound stopping. The user
reported no connected Linux output, so no Linux listener confirmation is
claimed. No macOS host is available.

The [complete Linux matrix](evidence/watchlist-article-audio/linux-pulse-fragment-21.json)
preserves every unknown and skipped row. The
[final native controls](evidence/watchlist-article-audio/linux-native-controls.json)
cover 500 ms natural playback, playing Stop and a separate 20 ms
launch-relative startup Stop. Natural/playing controls retained matched,
unattributed source observations; startup cancellation had no matched
marker. Every control player and capture worker retired.

### Linux environment and measurement investigation

The supplied host runs Debian 13, Python 3.13.5, ffplay 7.1.5 and PipeWire
1.4.2 with PulseAudio compatibility. Its default output is a real ALSA
digital-stereo hardware route with an exact monitor. No system audio settings
were changed. Qualification-only NumPy/SciPy/CFFI/SoundCard packages were
installed in a private temporary environment; no system package or app
environment changed.

The existing sink-ID/monitor-ID equality check failed on this host with
`IndexError`. Native sink introspection identified the monitor correctly.
A bounded monitor-only probe captured ten contiguous 4,800-frame fragments,
48,000 frames total, with exact read-index/drop advancement, no holes or
corrupt indexes, zero RMS and the verified route throughout. Actual fragment
size was 38,400 bytes and stream capacity 4,194,304 bytes. No player was
launched in this preflight.

Signed latency estimates ranged from 110.691 to 141.349 ms. The longest
completed timing update took 3.135 ms. These estimates do not establish the
age of PipeWire's underlying process snapshot or a bound on acquisition-time
uncertainty. The [numerical preflight](evidence/watchlist-article-audio/linux-native-preflight.json)
therefore records native frame custody separately from unqualified native
playback and wall-clock timing. The matrix used the native fragment capture
path described in ADR366, source revision `ee3cc6ffee`.

The installed client libpulse is 17.0.0. A separate silent-player identity
control matched the actual child PID to the expected hardware sink in 40
observations. The first grader controls incorrectly labeled an unassigned
`PA_INVALID_INDEX` startup as a move. A private native event trace and
parallel sink-input snapshots identified the construction transition; no
linked wrong sink was observed. Those
[initial controls](evidence/watchlist-article-audio/linux-initial-route-controls.json)
remain preserved. The corrected grader classifies uncertain initial history
as unavailable and retains detected source data only as unattributed
diagnostics. A consumer read index does not certify when buffered audio was
produced, so it was not used as a construction-time fence.

All numerical receipts were downloaded and their hashes verified before
removing the private Linux environment. No system audio configuration or
application environment changed. The
[verification receipt](evidence/watchlist-article-audio/listener-linux-verification.json)
records tests, review, the earlier Windows CLI timeouts and cleanup.

## Capture timeline investigation

The previous 500 ms clip and playing Stop had early and late waveform
correlations above 0.99999. Their capture windows contained 5,305 and 5,441
more frames than the aligned source interval, or 110.521 ms and 113.354 ms at
48 kHz. Neither reported a WASAPI discontinuity. A dropped span alone would
shorten the captured interval.

SoundCard 0.4.6 estimates silence from elapsed wall time when an empty read
persists beyond four default device periods. It can insert those frames before
the next real packet, and its wrapper drops the packet's device position and
timestamp. In a device-free control of the installed `_record_chunk`, an empty
poll and a scripted 112 ms wait between two contiguous real-packet fixtures
produced 5,376 extra zero frames. The unchanged matcher rejected the result
with −112 ms drift and both edge correlations effectively 1.0. The same two
packet fixtures matched when concatenated without the inserted frames. This
reproduces the signature of the earlier excess-frame cases. Those earlier
receipts lack packet positions and cannot identify the cause of each rejection.
[SoundCard 0.4.6 capture implementation](https://github.com/bastibe/SoundCard/blob/0.4.6/soundcard/mediafoundation.py#L639-L754).

An instrumented native control traced the legacy reads to their device
positions. It hid a ready packet from the legacy reader for 75 ms once
playback had started. The 500 ms clip then contained exactly 3,871 wrapper
zeros between the matched edges. Capture counted 25,167 frames across those
edges while device position and source both counted 21,296. The drift was
−80.646 ms with both correlations above 0.99999 and zero discontinuity flags.
The undelayed 500 ms control showed the opposite failure. Capture counted
10,560 fewer frames than device and source and WASAPI reported one
discontinuity. An undelayed playing Stop matched without added or missing
frames. The delayed Stop showed both inserted zeros and one discontinuity.
All four players and capture workers were reaped. The
[durable control summary](evidence/watchlist-article-audio/windows-legacy-idle-control.json)
records the frame counts, correlations, flags, and resource retirement.
The full numerical receipt is
`windows-legacy-idle-control/legacy-idle-control.json` in the plan workspace.

The noisy baseline and two discontinuities remain separate observed defects in
the original matrix. The Windows rerun measures real packet frame positions,
QPC timestamps and flags, with no inferred waveform samples for an empty poll.
The existing waveform, seek, bookmark and tail thresholds apply to the same
21 planned cases. [WASAPI packet contract](https://learn.microsoft.com/en-us/windows/win32/api/audioclient/nf-audioclient-iaudiocaptureclient-getbuffer).

The packet path drains the endpoint independently of analysis and uses the
native first-frame QPC timestamp. A first-packet stream-origin discontinuity
is counted separately. Later discontinuities, invalid timestamps, and
discontinuous matched spans remain ineligible. A quiet window requires
continuous drain observations limited by the actual endpoint buffer capacity
and a drain after the window. A monitored empty interval has no sampled RMS.
The 500 ms and playing Stop wiring controls both matched with zero alignment
drift and no later discontinuity or timestamp-error flag.

## Windows 250 ms decision from the 2026-11-15 wrapper-based run

**Keep the workaround provisional.** The final three-round matrix has 21 planned
cases, 20 player launches, and one correctly skipped dependent Resume. Thirteen
natural playbacks and two playing Stops passed their content-position checks.
Three natural recordings and one Stop were unmatched; another natural recording
had a noisy baseline and two capture discontinuities. No rejected/skipped row is
converted into a pass.

| Case | Passes / planned rounds | Remaining result |
| --- | --- | --- |
| Fixed nonzero seek, 0.8 s | 3/3 | None |
| 120 ms clip | 2/3 | Low-correlation capture |
| 500 ms clip | 1/3 | One drift rejection, one low-correlation capture |
| Near EOF, 140 ms remaining | 2/3 | Noisy baseline and two capture discontinuities |
| Near EOF, 60 ms remaining | 3/3 | None |
| Playing Stop and independent bookmark alignment | 2/3 | One drift rejection |
| Same retained WAV resumed from captured bookmark | 2/3 | One skipped after the unqualified Stop |

For the passing measurements, maximum absolute seek error was **2.438 ms**, tail
error **2.417 ms**, stopped-bookmark error **59.5 ms**, and telemetry error
**44.56 ms**. Both successful saved resumes reopened the same padded artifact:
frame 36,480 at 0.76 s and frame 35,520 at 0.74 s, with resumed onset errors of
-2.167 ms and -1.854 ms and original-content end error +2.167 ms. All 20 started
players and their capture workers were reaped.

The recorder buffer was requested and observed at 4,800 frames / 100 ms, while
reads stayed 480 frames / 10 ms. Observed live matching reached 62.50 ms and
receive backlog 118.91 ms. The contaminated near-EOF case exceeded that buffer's
interval and reported two discontinuities. Separate rejected captures showed
approximately 110/113 ms source-to-capture drift or low correlation despite no
reported discontinuities. These are unresolved capture/mix/timing limitations;
the evidence does not establish that padding lost spoken content or that every
gap originates in the player.

All 13 matched natural runs retained the original content tail within the
unchanged 20 ms oracle, including both successful saved resumes. That positive
subset is insufficient for reliable repeated qualification. **Do not adopt the
workaround or start dependent feature work from this result.** Resolve the
capture/timing ambiguity and rerun the unchanged acceptance cases first.
macOS/Linux, physical/listener verification, exact logical-EOF resume and
interruption inside padding remain unqualified. No application implementation
or provider requests were introduced.

Final numerical receipt:
`windows-padding-250ms-buffer4800-final/qualification.json` in the ignored plan
workspace. A [durable allowlisted summary of every case](evidence/watchlist-article-audio/windows-padding-250ms-buffer4800.json)
preserves the decision, metrics and artifact identities in the repository.
Tooling commit: `3f6c1995b6983b8f27cd4c7dfc087b8cd0afad2a`. Earlier
receipts and all censored/rejected attempts below remain preserved.

## Initial unpadded native gate matrix

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

## Windows 250 ms follow-up: preparation-censored attempt

The first repeated padded attempt used the same ffplay hash with fixed 250 ms
of local zero PCM, 120/500 ms clips, 140/60 ms of source remaining near EOF,
fixed seeking and an intended same-WAV Stop/resume chain over three rounds.
Original content and padded file durations were recorded separately.

Its **21 case rows contain only six actual player launches**, not 21 completed
audio tests: twelve capture helpers exceeded their eight-second preparation
budget, two launch-relative Stop timers fired before any matched source audio or
fresh progress was observed, and three dependent resumes were correctly skipped.
The four completed natural cases in round three passed independently: both short
clips and both near-EOF cases. Maximum onset error was 2.438 ms and maximum
absolute original-content tail error was 2.417 ms. All started players and capture
workers were reaped. No successful same-WAV resume was established by this attempt.

This is preparation-censored evidence, not proof that padding failed. The original
receipt retains its conservative failed aggregate; its missing Stop latency was
also treated as infinity by the initial gate. A follow-up corrects that missing
measurement to unqualified, separates capture/player preparation from measured
playback, and anchors playing Stop to independently matched recent source audio.
The 20 ms tail and 250 ms seek tolerances stay unchanged. The stopped bookmark
must also match independently captured stopped source position within 250 ms.

Receipt: `windows-padding-250ms-three-rounds/qualification.json` under this plan's
ignored workspace. Preserve it alongside the subsequent phase-corrected attempt;
never replace it or count its skipped rows as passing playback.

## Windows 250 ms follow-up: phase-corrected attempt

The next attempt used separate 30-second capture/player preparation bounds,
45-second bounded capture, and a 700 ms Stop countdown anchored to recently
matched original source audio. It preserved the same seek/tail oracles and added
an independent saved-bookmark-versus-stopped-source check.

All 18 base players launched and were reaped; the three dependent resumes stayed
unlaunched because their preceding full-capture Stop measurements were unqualified.
Preparation timeouts were eliminated in this attempt.

| Natural case | Individually matched seek/tail passes | Other rows |
| --- | --- | --- |
| Nonzero 0.8 s seek | 1/3 | 2 unmatched |
| 120 ms clip | 2/3 | 1 unmatched |
| 500 ms clip | 3/3 | None |
| Near EOF, 140 ms remaining | 1/3 | 1 unmatched, 1 noisy baseline |
| Near EOF, 60 ms remaining | 3/3 | None |

Every matched natural clip retained its original tail within about 2.5 ms. This
is **10 passes among 15 natural attempts**, not a passing repeated qualification.
All three Stops had a live source match and fresh saved positions (0.77, 0.74 and
0.82 s), but full-capture matching failed, so their bookmark alignment and resumes
could not be certified. Process retirement and captured post-stop silence passed.

Review identified a measurement risk: synchronous live matching in the receiving
loop can backpressure the capture worker while its recorder requests only a
10 ms device buffer. The existing receipt cannot distinguish that mechanism from
other capture/mix contamination. A focused follow-up records numeric capture
quality and matcher rejection metrics before varying only the recorder buffer;
the emitted 480-frame blocks and all source-match tolerances remain fixed.

Receipt: `windows-padding-250ms-phase-corrected/qualification.json`. No native
Windows workaround pass is claimed from this attempt, and no failures or
unmatched rows are removed.

## Focused recorder-buffer control

The diagnostic control kept the same padded WAV, source bytes, 700 ms
observed-playing Stop, emitted 480-frame reads and matching thresholds. It varied
only the requested recorder buffer, then resumed each eligible Stop from its
captured content-frame bookmark against the identical file.

| Measurement | 480 frames requested | 4,800 frames requested |
| --- | --- | --- |
| Actual recorder buffer | 1,056 frames / 22 ms | 4,800 frames / 100 ms |
| Observed discontinuities, Stop / Resume | 1 / 0 | 0 / 0 |
| Largest live-matcher duration in pair | 45.93 ms | 55.60 ms |
| Largest receive backlog in pair | 48.70 ms | 55.27 ms |
| Largest completed-send wait in pair | 38.99 ms | 43.78 ms |
| Stop bookmark versus captured source end | -79.29 ms | -69.50 ms |
| Resumed source onset versus saved bookmark | -2.313 ms | -2.313 ms |
| Resumed original-content end error | +2.167 ms | +2.167 ms |

All four players/capture workers were reaped, both baselines were zero, and the
same artifact digest `3b07a8cade33ae005a9eecc402883e60386b92bfa59a7155da99870342d5ab24`
and frame 36,000 / 48,000 Hz bookmark were retained. The small-buffer pair passed
the audio-position/tail checks but contains a capture discontinuity, so it cannot
be counted as clean qualification evidence. The larger buffer covered the
observed analysis/backpressure interval and had no reported discontinuities in
this control. This supports the measured capture setup without proving that it
caused every earlier unmatched result.

Send metrics cover completed sends through the previous captured block, not an
in-flight final send. Git approval activity may have overlapped the beginning of
this control; no baseline contamination was observed. No further test/approval
activity is scheduled during the final repeated native run.

Receipt: `windows-padding-buffer-control/control.json`. Existing failed/censored
attempts remain preserved. A known capture discontinuity now prevents a native
pass even when source-edge matching succeeds.

## Deterministic verification

The packet correction passes **47 targeted tests** in 21.138 s, Ruff check,
and formatting for four files. Coverage includes native packet counts and
release, silent/null buffers, empty-buffer status, device-frame and QPC gaps,
invalid timestamps/PCM, monitored quiet intervals, content-time Stop
eligibility, and packet quality before saved Resume. Independent code review
found no actionable Important/Critical issue. Native preflight passed the
500 ms natural clip and playing Stop before the full contaminated-baseline
matrices were recorded. The subsequent quiet-endpoint matrix passed all 21
Windows cases; portable qualification remains open. A subsequent Windows
listener check passed as recorded above.

The wrapper-based padding runner passed **38 targeted tests**, Ruff check and format.
Independent review confirmed both capture-diagnostic and partial-evidence/cleanup
findings addressed, with no new Important/Critical finding in the fix diff.
Tests include exact padding geometry, original PCM preservation, 60/140 ms
near-EOF calibration, captured frame bookmarks, identical-artifact resume,
bounded phase timing, buffer forwarding, discontinuity refusal, retained partial
receipts, and aborting later launches after uncertain resource retirement.
This verifies the runner; it does not promote rejected native measurements.

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
python -m unittest discover -s tools/evidence/watchlist_article_audio -v
python -m ruff check tools/evidence/watchlist_article_audio
python -m ruff format --check tools/evidence/watchlist_article_audio
python tools/evidence/watchlist_article_audio/qualification.py --capture-loopback --ffplay <direct-executable> --output-dir <this-plan-ignored-workspace/run-directory> --tail-padding-ms 250 --repeat 3 --capture-buffer-frames 4800 --capture-clock wasapi-packet
```

The native invocation plays short quiet markers. Without `--capture-loopback`,
the tool cannot award a native-audio pass. Keep fixture WAVs, isolated packages
and numeric run receipts under `.superpowers/sdd/2026-11-15-watchlist-article-listening/`;
commit only recipes, code/tests and this sanitized report.
