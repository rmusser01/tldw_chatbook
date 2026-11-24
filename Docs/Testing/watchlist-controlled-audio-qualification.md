# Windows controlled audio qualification

On the tested Windows default-render endpoint, seven natural-playback controls independently matched the requested source position and logical tail. The saved Stop bookmark resumed the same retained WAV. Exact EOF launched no player. All three Stop phases returned `stopped`, including a Stop requested after the last logical frame and before process reap. An injected premature zero exit returned `uncertain` with a measured missing tail. The earlier ffplay executable and endpoint qualification supplies the underlying playback profile. [First gate](watchlist-article-audio-qualification.md) · [First-gate machine receipt](evidence/watchlist-article-audio/windows-first-release-gate.json)

A second, complete 12-control run reproduced the terminal and waveform results under the promoted instrument. Every fixture reported native 0600 before admission and after retirement. The run read the physical frame counts from the WAVs, verified the active Stop → Resume inode and digest, and recorded all 17 instrument/oracle/runtime source hashes at capture. Its bookmark was frame 24,480 (510 ms), its resumed onset 507.646 ms and its padding Stop reached physical frame 25,920 before reap. [Canonical 12-control run and manifest](evidence/watchlist-article-audio/controlled-adapter-canonical-controls-6.json)

In 18 ownership transfers, the narrowest of 15 independently measurable old-end to successor-onset gaps was **78.317 ms**. Each exceeds the combined 40 ms edge margin. In three startup transfers, the left channel remained independently quiet through the successor, so a gap has no measurable starting edge. These measurements cover this host, route, executable and synthetic sources. The observer is the digital default-render WASAPI loopback. Acoustic output at a speaker and universal real-time bounds remain unqualified. [Three-round transfer receipts](evidence/watchlist-article-audio/controlled-adapter-transfers-3rounds.json)

## Original playback, bookmark and retirement observations

All markers use 48,000 frames/s, stereo signed PCM16, with 12,000 silent physical frames after logical EOF. The 120, 500 and 2,500 ms sources have 5,760, 24,000 and 120,000 logical frames, respectively. Their physical counts are 17,760, 36,000 and 132,000 by construction. Historical receipts record logical counts and bookmarks. The promoted drivers also read each WAV's physical frame count and record native modes, source counts and digest explicitly.

| Control | Independent result |
| --- | --- |
| Natural 120, 500, 2,500 ms | All three matched at both ends. |
| Resume 1.230 s, remaining 140 ms, remaining 60 ms | Matched from requested frames 59,040, 113,280 and 117,120 through the original tail. |
| Active Stop and saved Resume | Stop saved frame 24,960 (520 ms). Captured content ended at 549.438 ms, a 29.438 ms bookmark difference. The resumed marker started at 518.083 ms and ended at 2,501.625 ms. Digests and requested/saved frames agree. The controller retained the file between attempts. |
| One remaining frame | `natural` at frame 120,000. Only 480 non-silent-span capture frames reached the matcher. Independent waveform fidelity is unqualified below its 30 ms minimum. |
| Exact EOF | `already_at_end` at frame 120,000 with no launch, stop or reap timestamp. |
| Startup / active / padding Stop | `stopped` in all three. Request-to-retired took 252.950 / 253.697 / 256.122 ms. Startup produced no detected old marker. Padding Stop observed logical frame 24,000, physical progress 24,960, then requested Stop 5.555 ms before reap. |
| Early zero exit | The fault targeted a duplicate of this run's exact owned process handle and requested exit code 0. Completion returned `uncertain / terminal_progress_unconfirmed`, retaining frame 7,200. Loopback matched only 199.813 ms of the 2,500 ms source. The duplicate closed and final retirement and capture reap were confirmed. |

Across the seven full independent natural controls, the largest absolute start and tail errors were both 2.417 ms. Minimum early/tail correlation was 0.99999455 and all reported alignment drifts were zero. Each launched control has a monitored post-reap quiet window. [Control and saved-Resume receipts](evidence/watchlist-article-audio/controlled-adapter-native-controls-4.json) · [Padding Stop](evidence/watchlist-article-audio/controlled-adapter-padding-stop-5.json) · [Early zero exit](evidence/watchlist-article-audio/controlled-adapter-early-zero-exit.json)

## Takeover at the endpoint

The original source matcher processes the entire captured left channel for the old marker and the entire right channel for its differently seeded successor. Each channel retains packet flags, hardware frame positions and QPC timestamps. The signed gap is successor onset minus old end. Separately monitored single-source windows check for crosstalk. In the final matrix every measured isolation window had RMS zero. The three startup left-channel checks spanned startup through 100 ms beyond the successor's end. A device-free mixed stereo control measures −204.833 ms for programmed overlap and +195.167 ms for programmed separation. [Channel measurement](../../tools/evidence/watchlist_article_audio/windows_output_transfers.py) · [Artificial mixture](../../tools/evidence/watchlist_article_audio/controlled_instrument.py)

| Old → successor | Phase | Gap in each round, ms |
| --- | --- | --- |
| Controlled WAV → controlled WAV | Active | 338.097 / 348.459 / 348.234 |
| Controlled WAV → controlled WAV | Startup | Old channel quiet in all three |
| Controlled WAV → controlled WAV | Logical EOF / padding | 385.146 / 416.066 / 426.236 |
| Controlled WAV → PCM | Active | 586.411 / 496.544 / 496.514 |
| PCM → controlled WAV | Active | 78.317 / 98.070 / 88.093 |
| PCM → PCM | Active | 236.689 / 246.448 / 236.333 |

Every final-matrix row records positive final retirement of both runs, revocation of the old lease, release of ownership and capture reap. The six PCM successors reported `drained` before final checked retirement. Their later `new_retirement_confirmed` fields provide the physical-custody evidence. Windows ordinary file playback hands off to the OS. Acknowledgement and supersession are covered by mounted UI tests. The acoustic matrix exercises the application's controlled WAV and streaming PCM adapters. Windows audible inherited-descriptor media is unsupported. Linux and macOS controlled playback remain unqualified.

## Decision thresholds

- Require quiet baseline and applicable post-reap or isolation windows. The original monitor accepts RMS ≤0.0002 or a fully polled no-packet window. The isolation checks require observed frames.
- Reconstruct both source and relative capture edges and the alignment drift from the integer early/tail source and capture frames. Check the summary against those edges and verify that the recorded packets can account for the full matched span. Require matched early/tail correlation ≥0.70, absolute seek/bookmark error ≤250 ms, absolute logical tail error and alignment drift ≤20 ms. For two measured source edges, classify separation only at a raw gap ≥40 ms. Keep the signed unrounded gap in the receipt.
- Require zero interior packet discontinuities and timestamp errors, frame-contiguous matched content, packet timestamp inconsistency ≤5 ms, a QPC calibration bracket ≤1 ms, and polling gaps ≤min(150 ms, actual buffer frames / 48,000). The actual 4,800-frame buffer sets a 100 ms bound. A session-origin discontinuity is counted separately. Across the final 18 transfers the largest poll gap was 21.135 ms and the largest matched packet clock error was 0.080 ms.
- Require exact typed terminal/generation, an applicable source bookmark, checked native cleanup and the 1.5 s Stop observer bound. The grader compares absolute packet edges with controlled-process launch and retirement, PCM start/drain, the successor grant and the acquisition request on their common clock. It allows 20 ms uncertainty at a measured edge and applies the 1.5 s Stop limit to the old source's measured end. A clean process exit alone cannot establish that the logical source tail played. [Independent numerical grader](../../tools/evidence/watchlist_article_audio/grade_controlled_audio.py) · [Existing matcher](../../tools/evidence/watchlist_article_audio/qualification.py) · [Packet-clock monitor](../../tools/evidence/watchlist_article_audio/wasapi_packets.py)

## Retained attempts and provenance

| Attempt | Evidence retained |
| --- | --- |
| Quiet capture | Monitored no packets, 61.714 ms maximum poll gap, capture reaped. [Preflight](evidence/watchlist-article-audio/controlled-adapter-quiet-preflight-1.json) |
| First and second control attempts | Both persisted a 500 ms result. The second controller stopped at the next 120 ms preparation refusal before writing that row. [First pilot](evidence/watchlist-article-audio/controlled-adapter-native-control-1.json) · [Second attempt](evidence/watchlist-article-audio/controlled-adapter-native-controls-2.json) |
| Third control attempt | 500 ms matched, 120 ms preparation rejected. Diagnostic showed native mode 0644, one link, owner match, 71,084 bytes, no launch. [Third attempt](evidence/watchlist-article-audio/controlled-adapter-native-controls-3.json) |
| Fourth and fifth control attempts | Eleven accepted rows, then a 144,044-byte 0644 padding fixture refused before launch. The private 0600 padding rerun recorded matching modes at admission and after retirement. The previous accepted rows do not record per-file native modes. [Fourth attempt](evidence/watchlist-article-audio/controlled-adapter-native-controls-4.json) · [Fifth attempt](evidence/watchlist-article-audio/controlled-adapter-padding-stop-5.json) |
| Transfer pilots | Controlled → controlled, 328.236 ms. PCM → controlled, 88.167 ms. These single-row pilots predate the opposite-channel isolation windows and are excluded from the final 18. [Controlled pilot](evidence/watchlist-article-audio/controlled-adapter-transfer-pilot-1.json) · [PCM pilot](evidence/watchlist-article-audio/controlled-adapter-pcm-transfer-pilot-2.json) |
| Preparation and executable path diagnostics | A separate preparation-only refusal reported `private_file_required`. A later 24-case preparation-only repeat retired every prepared case. The controller's version-only probes report error 87 for extended volume GUID and GLOBALROOT forms and exit 0 for the device-GUID form, with and without an explicit application name. The first probe exception ended that controller before explicit pin close. Subsequent probes confirmed handle/pin cleanup. These statements use the retained controller ledger. [Provenance and diagnostics](evidence/watchlist-article-audio/controlled-adapter-provenance.json) |

The historical control receipts identify controlled SDK revision `81ba625ba0`; the transfer receipts identify `a723890926`. The owned output run used by the transfer matrix includes updates through `3bee94eae9`. File comparisons through publication base `7567277869` found the measured adapter, ownership, streaming sink, native file helpers and numerical oracles identical to their transfer-revision versions. The quiet and negative raw JSON have no revision field. The historical dates and revisions are copied only where present. The canonical rerun's manifest records its actual checkout at `75672778696283332ad2634d8bc2659f2a2f9e2b` and byte hashes of the measured files.

The source snapshot is identified by the annotated tag `codex/watchlist-audio-evidence-2026-11-24`. The four exact acquisition-source blobs are preserved in commit `f9e16bd94424c2bffc99c2d489a4e7ccd20f37ff`. All 17 canonical capture digests were verified against that tree. Nine match the Git blob bytes directly. Eight match the original Windows CRLF checkout bytes obtained by expanding the blobs' LF line endings: `audio_output_ownership.py`, `streaming_sink.py`, `windows_files.py`, `private_paths.py`, `platform_files.py`, `qualification.py`, `wasapi_packets.py` and `pulse_fragments.py`. The four new acquisition files all match directly. The canonical command exited 0 and the controller's post-run count of temporary fixture directories under its dedicated prefix was zero.

The current transfer and negative-control drivers include a subsequent device-free cleanup correction. The capture-time manifest continues to identify their checkpoint versions. A failed transfer retirement or mode query now records a bounded diagnostic while the other retirement, both feed waits and capture reap are attempted. A negative run saves its observations and returns a failure exit status if fault injection, custody or independent tail evidence fails. The raw controls, transfers and negative grading commands require their schema-2 source manifest and per-fixture evidence. Historical receipts use the archive command.

The provenance receipt distinguishes the retained controller files hashed at publication from the promoted instrument files. The prototypes evolved across attempts. Those snapshot hashes cannot attest the exact earlier driver revision. SHA-256 of each original raw JSON lets its custodian check the allowlisted export without disclosing capture data or identifiers. No waveform samples were persisted. The earlier first gate records ffplay 7.1-essentials_build-www.gyan.dev, SHA-256 `f6536c4caf9bac0fee1e8ffaafa5b13b6ad3e386a9a9f0b6613360e8c6d59919`.

At publication, a read-only lookup found Windows 10.0.19045 AMD64, Python 3.12.10, NumPy 2.2.6, SciPy 1.15.3, CFFI 2.1.1, SoundCard 0.4.6, sounddevice 0.5.6 and PortAudio 19.7.0-devel (revision unknown). SoundCard came from the ignored measurement-only `deps` directory and optional sounddevice from the separate ignored `sounddevice-deps` directory. These runs add no dependencies to the application. Future runs write a source/dependency manifest before acquisition. [Publication lookup and file hashes](evidence/watchlist-article-audio/controlled-adapter-provenance.json)

## Reproduce

From the repository root, the following checks use only the public numerical receipts and synthetic arrays. The grader itself uses the standard library. The instrument unit tests also need NumPy and SciPy.

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$tool = 'tools/evidence/watchlist_article_audio'
$archive = 'Docs/Testing/evidence/watchlist-article-audio'
py -3.12 "$tool/grade_controlled_audio.py" $archive
py -3.12 -m unittest discover -s $tool
py -3.12 -m ruff check $tool
py -3.12 -m ruff format --check $tool
```

For a new hardware run use the qualified executable/default-render route on Windows, the application's Python environment and isolated measurement dependency directories. Leave SDL_AUDIODRIVER, SDL_AUDIO_DEVICE_NAME and AUDIODEV unset. Start from a quiet endpoint with no concurrent playback, a native-verified private home parent and a fresh output location for each command. These acquisition commands are the promoted CLI. Original pilot shell transcripts were not recorded in the raw JSON.

```powershell
$plan = '.superpowers/sdd/2026-11-15-watchlist-article-listening'
$env:PYTHONPATH = "$(Resolve-Path "$plan/deps");$(Resolve-Path "$plan/sounddevice-deps");$(Resolve-Path '.')"
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$tool = 'tools/evidence/watchlist_article_audio'
python "$tool/windows_controlled_adapter.py" --quiet-preflight --output-dir "$plan/repeat-quiet"
python "$tool/grade_controlled_audio.py" --kind preflight "$plan/repeat-quiet/quiet-preflight.json"
python "$tool/windows_controlled_adapter.py" --play --output-dir "$plan/repeat-controls"
python "$tool/grade_controlled_audio.py" --kind controls "$plan/repeat-controls/adapter-controls.json"
python "$tool/windows_output_transfers.py" --play --rounds 3 --output-dir "$plan/repeat-transfers"
python "$tool/grade_controlled_audio.py" --kind transfers "$plan/repeat-transfers/transfers.json"
python "$tool/windows_early_exit.py" --play --output-dir "$plan/repeat-negative"
python "$tool/grade_controlled_audio.py" --kind negative "$plan/repeat-negative/negative-control.json"
```

The promoted fixture writer sets and verifies native 0600 on each newly created descriptor before writing the marker. The saved-Resume case reuses the same file and records an identity check. Only numerical observations and hashes leave the in-memory loopback recorder. Checked retirement and a verified temporary-directory boundary govern fixture cleanup. The eleven captured JSON exports can be regenerated from the retained ignored input directories with `py -3.12 "$tool/export_controlled_receipts.py" $plan $archive`. [Control driver](../../tools/evidence/watchlist_article_audio/windows_controlled_adapter.py) · [Transfer driver](../../tools/evidence/watchlist_article_audio/windows_output_transfers.py) · [Zero-exit driver](../../tools/evidence/watchlist_article_audio/windows_early_exit.py) · [Mutation and stereo tests](../../tools/evidence/watchlist_article_audio/test_controlled_evidence.py)
