# Speculative Voice Qualification

Speculative duplex voice remains disabled unless source-bound automated, soak, and
physical evidence produces a valid rollout manifest. Synthetic fixture reports test
the harness only and never authorize a platform.

## Evidence boundary

- Compute one source-tree digest only from a clean checkout with
  `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python Packaging/compute_voice_source_digest.py`.
- Run every automated, soak, and physical trial from that exact checkout and Python
  environment. Changing any file listed in
  `Packaging/speculative_voice_source_paths.txt` invalidates every prior automated,
  soak, and physical report, even when the changed file is documentation, a test, or
  the reference-audio manifest.
- Each platform report qualifies only its recorded CPython ABI. The generated
  authority selects that ABI's repaired wheel and native-extension hash from the
  multi-ABI release artifact; other ABIs fail closed to legacy hands-free.
- Store automated reports in `Artifacts/voice_qualification/automated/` and physical
  reports in `Artifacts/voice_qualification/physical/`.
- Reports contain only aggregate metrics and hashes. Never add captured audio,
  transcripts, response text, credentials, raw device names, or raw device IDs.
- The device identity hash uses a report-local random salt. This prevents correlation
  between reports; it is not intended to make a guessable display name secret. Prefer
  a stable backend identifier over a human-readable device name.
- Physical reports created by the former manual metric-entry workflow are invalid.
  Operator confirmations can start or cancel a prompt and acknowledge a physical route
  change; they cannot supply a safety, latency, route, cleanup, or soak measurement.

## Evidence generation order

Use this exact order whenever qualification source or evidence changes:

1. Freeze and commit every source-listed file.
2. Compute one digest from that clean worktree.
3. Regenerate automated, duplex-soak, and cancellation-soak evidence.
4. Run physical-device evidence from the same commit and digest.
5. Validate every physical report through the strict consumer.
6. Build rollout authority only after the complete platform/device matrix exists.

## Per-platform procedure

For each declared platform/architecture, first produce passing automated,
duplex-soak, and cancellation-soak reports. Name the soak reports
`<platform>-duplex-soak.json` and `<platform>-cancellation-soak.json` so each physical
report binds to evidence from its own platform.

Then qualify built-in, USB, and Bluetooth device classes. Run from the repository root;
the macOS USB invocation uses the project venv explicitly:

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python scripts/qualify_physical_voice.py \
  --source-tree-digest <digest> \
  --platform macos-arm64 \
  --device-class usb \
  --automated-report Artifacts/voice_qualification/automated/macos-arm64-automated.json \
  --output Artifacts/voice_qualification/physical/macos-arm64-usb.json
```

The checked-in CC0 reference sample will be audible. Set the target output to a safe
listening level before starting, and warn anyone nearby. The runner does not
transcribe, save, hash, or report microphone audio or speech. Raw route names and
identifiers also remain in-memory, and reports contain only aggregate measurements
and a report-local salted device hash.

The timed speech protocol is calibration-gated and automatically paced:

1. At the double-talk calibration prompt, press Enter and begin speaking
   continuously. The runner waits up to 10 seconds for actual admitted microphone
   speech before starting playback. If speech is not detected, it aborts before any
   repeated trials.
2. When full-duplex admission is open, the runner next verifies that the same speech
   remains detectable during playback. It then collects all 20 half-second
   double-talk windows from the same continuous speaking block, keeps playback
   continuous, and prints each result immediately. There are no per-window Enter
   prompts. If full-duplex admission is not open after preflight, the runner states
   that it is using the manual-stop fallback instead of claiming overlap calibration.
3. Stop speaking when instructed. The next phase cannot start until the runner has
   measured one continuous second of microphone silence.
4. For full-duplex interruption, acknowledge the phase once, then respond to each
   displayed `speak now` cue. Each cue allows up to 10 seconds for a human response;
   every result is printed immediately. Stop speaking after each sample so the runner
   can verify one second of silence before issuing the next cue. Half-duplex uses an
   explicit manual-stop acknowledgement for each sample.
5. The first failed interruption sample stops the remaining samples and route prompts
   and produces partial, schema-valid failing evidence. It never continues asking the
   operator for trials that can no longer yield a passing qualification.

On hosts that emit status-flagged PortAudio callbacks while a new route is priming, the
transport discards those callbacks during a hard-bounded 500 ms startup pre-roll,
without advancing the app capture sequence, and rebases cadence comparison after each
discard. A status at callback 51 or later and every later discontinuity still fail
closed. Callbacks delivered while an already-fenced native stream is stopping are
silenced and rejected; callbacks arriving after teardown completes remain reportable
stale-callback failures. A manual-stop prompt is latched for its interruption phase and
cannot silently become an acoustic prompt if the safety path recovers mid-prompt.

Follow all prompts. Each of the three route trials is a complete round trip: switch the
default input and output from the target route to a real alternate route, acknowledge
the prompt, then switch both back to the target route and acknowledge the return. The
runner must observe both generation changes and closed admission; pressing Enter alone
is not evidence. Do not begin the unattended soak until the displayed route has
returned to the target input/output pair after every round trip.

Allow at least 30 minutes of active trial time, plus operator-prompt time. After the
timed speaking and three route round trips, the remaining mixed render/silence soak is
unattended. A completed or fail-fast run that misses a safety threshold writes a
schema-valid `FAIL` report and exits non-zero. Permission denial, missing or
unsupported devices,
asset mismatch, unobservable route changes, cancellation, or invalid/incomplete
evidence aborts without a report and must not be represented as a passing or failing
measurement. A healthy AEC full-duplex path must meet the report thresholds. A warming
or degraded AEC health path may remain full duplex only while the native processor is
operational and the shared preprocessor proves sustained acoustic isolation; otherwise
it passes only in explicit half duplex with playback-period speech admission closed and
manual interruption still available. Unsuppressed playback that causes an interruption
always fails.

## Recorded authority

No authoritative physical report has been recorded yet. Task 12C records exact report
hashes here only after the source-digested implementation is committed and the real
macOS, Windows, and Linux device matrix has been run.

## Source-bound local automated evidence

The committed macOS arm64 / CPython 3.12 automated gates were run from revision
`ef02cb23832fbeedd038f276396cf20bb26e3d02` with source-tree digest
`b3dc2024b2c13999516cb127ed5614bf8e94c2f4c5411008fdbc84969ffabf75`:

- automated report SHA-256:
  `6db4c08bb9d51fd266b902e30aba6a28eb2f5246cbd98ddc341064fc23b81451`;
- duplex-soak report SHA-256:
  `457ffce2553f753d7be5e02e66086dcdf58de5866a0f092bdb4f9e29e6b17a8c`;
- cancellation-soak report SHA-256:
  `2033fea8aaa8755d08ae4a3b70b586a1f7e956b783a469af05fec62fa459e8e1`.

The corpus recorded 29.64 dB median ERLE, 28.78 dB p10 ERLE, zero false
barges over 30 rendered minutes, and 0.97 double-talk recall. The duplex soak
completed 160,715 iterations with zero overflows, callbacks, or leaked handles.
The cancellation soak completed 156,501 iterations and 313,002 cancellations,
with a maximum orphan count of two and a final count of zero. Both 30-minute
soaks returned to their baseline task count and reaped the native process.

The automated report and both soak reports passed the deterministic manifest
validators using the exact companion wheel and native-extension identities.

This is only one automated platform/ABI evidence set. It does not authorize
speculative voice until the remaining platform and real physical-device matrix
is complete and the deterministic rollout manifest is regenerated.
