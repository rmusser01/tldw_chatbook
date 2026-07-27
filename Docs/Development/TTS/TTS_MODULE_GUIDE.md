# TTS (Text-to-Speech) Module Guide

## Overview

The TTS module in tldw_chatbook provides a flexible, extensible system for generating speech from text using multiple providers. It supports both cloud-based APIs (OpenAI, ElevenLabs) and local models (Kokoro, Chatterbox, Higgs), with features like streaming audio generation, format conversion, text normalization, advanced voice cloning, and multi-speaker dialog generation.

## Architecture

### TTS adapter service

The application owns one sealed `TTSAdapterRegistry` and one `TTSService`.
Native adapters use canonical provider IDs and `TTSService.synthesize()`.
`audio_cpp` is the first native adapter. It is registered first, by the exact
canonical ID `audio_cpp`, with display label `audio.cpp` and no alias. The
adapter remains unmaterialized until its first operation.

The following six entries remain unchanged behind the temporary compatibility
bridge: `openai`, `elevenlabs`, `kokoro`, `chatterbox`, `higgs`, and `alltalk`.
Each bridge adapter lazily owns one provider-scoped `TTSBackendManager`;
application and UI code must not access that manager or any concrete adapter.
The bridge is removed only after every retained provider has a native adapter
and all legacy internal-model callers have migrated.

New providers are implemented as native adapters. See
[ADR-023](../../../backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md)
and the approved
[audio.cpp adapter design](../../superpowers/specs/2026-07-23-audio-cpp-tts-adapter-registry-design.md).

### Module Structure

```
tldw_chatbook/TTS/
├── __init__.py              # Module exports
├── adapter_types.py         # Provider-neutral adapter contracts
├── adapter_registry.py      # Sealed app-scoped provider registry
├── adapter_bootstrap.py     # Application service construction
├── legacy_bridge.py         # Temporary provider-scoped compatibility adapters
├── audio_cpp_config.py      # Immutable external-server configuration
├── audio_cpp_contract.py    # Pinned JSON and PCM16 WAV validation
├── preferences.py           # Immutable global defaults and config mutations
├── request_admission.py     # Atomic preference/revision/lease admission
├── profile_errors.py        # Value-independent profile/store failures
├── profile_types.py         # Immutable profiles, assignments, and receipts
├── profile_schema.py        # Dedicated SQLite validation and codecs
├── migrations/
│   └── v0_to_v1.py         # Versioned profile-store schema migration
├── profile_store_lock.py    # Cooperative shared/exclusive process locking
├── profile_repository.py    # Serialized CRUD, backup, and restore lifecycle
├── playground_types.py      # Immutable Playground request/artifact contracts
├── adapters/
│   └── audio_cpp.py         # Native external audio.cpp adapter
├── audio_schemas.py         # Pydantic schemas for requests/responses
├── TTS_Generation.py        # Main TTS service orchestration
├── TTS_Backends.py          # Legacy bridge manager and base class
├── audio_service.py         # Audio format conversion service
├── text_processing.py       # Text normalization and chunking
├── backends/                # Backend implementations
│   ├── __init__.py
│   ├── openai.py           # OpenAI TTS API
│   ├── kokoro.py           # Local Kokoro model
│   ├── elevenlabs.py       # ElevenLabs API
│   ├── chatterbox.py       # Chatterbox TTS (voice cloning)
│   ├── higgs.py            # Higgs Audio V2 (advanced voice cloning)
│   └── higgs_voice_manager.py # Voice profile management for Higgs
└── utils/                   # Utility modules
    ├── __init__.py
    ├── download_models.py   # Model download utilities
    ├── voice_utils.py       # Voice mixing utilities
    └── performance.py       # Performance tracking
```

### Core Components

#### 1. TTSService (`TTS_Generation.py`)
The main orchestration layer that:
- Routes canonical provider IDs through the sealed registry
- Exposes provider-neutral synthesis, catalog, voice, and reconfiguration
  operations
- Retains adapter resources until each audio response is closed
- Preserves the legacy byte-stream interface during migration

#### 2. TTSAdapterRegistry (`adapter_registry.py`)
The application-owned registry performs exact provider lookup, lazy adapter
materialization, operation leasing, targeted reconfiguration, and bounded
shutdown. Registration is sealed at construction time.

`TTSBackendBase`, `TTSBackendManager`, and the class-global legacy backend
registry are compatibility-bridge internals. They are not the extension point
for new providers.

### Local generation profile repository (Slice 2A)

Reusable generation profiles now have a dedicated, versioned SQLite ownership
boundary. `TldwCli` constructs one initially closed `TTSProfileRepository` and
opens it lazily for a profile-store consumer such as **Backup All**. The default
file is `tldw_chatbook_tts_profiles.db` in the current Chatbook user data
directory. An installation may instead set a validated path:

```toml
[database]
tts_profiles_db_path = "/absolute/path/to/tts-profiles.db"
```

The store is local-only and separate from character cards, provider
configuration, and conversation storage. Schema version 1 holds complete,
immutable profile snapshots and authority-scoped assignment records. Profile
display names are trimmed and have a unique key derived as
`NFKC(display_name).casefold()`. Creates begin at revision 1; updates require
the exact revision read by the editor and increment it atomically. A stale
revision or normalized-name collision reports a conflict without overwriting
the stored row. Assignment identity is the complete
`(source, authority_id, character_id)` tuple, and a foreign-key restriction
prevents deletion of an assigned profile.

Every repository operation runs through one serialized off-event-loop worker,
which owns at most one long-lived SQLite connection. An open repository keeps a
cooperative shared lock next to the database, so multiple Chatbook processes
may read and write through SQLite while each retains shared ownership. Restore
must first quiesce admitted work and acquire a bounded exclusive lock. A second
process that still holds a shared lock therefore prevents replacement and
causes restore to fail before the live file is changed.

Normal operations and results carry a monotonic lifecycle generation. Restore
advances that generation when admitted, rejects new normal work while
`restoring`, cancels queued older work, and prevents an already-running older
result from being published. The public states are `open`, `restoring`,
`unavailable`, and `closed`; definitive close is terminal.

`TTSProfileRepository.backup_to()` uses SQLite's online-backup API, validates
the completed standalone snapshot, and publishes it atomically at its
destination. **Backup All** reaches the profile database only through this
repository method; it never copies the open profile file. The databases in one
Backup All directory are individually consistent snapshots taken during the
run, not one cross-database atomic snapshot.

Restore is an explicit, bounded repository operation. It validates a private
snapshot of the candidate, stages it through SQLite online backup, performs
schema, full-integrity, foreign-key, and domain-row checks, and creates a
durable pre-restore recovery database before atomic replacement. Quiescence,
candidate validation, exclusive-lock, recovery-backup, or replacement failure
leaves the current store authoritative and rebinds it when safe. If replacement
succeeds but shared-lock reacquisition or authoritative reopen fails, the
repository reports `unavailable`, retains recovery evidence, and does not
create a blank database. Corrupt, partial, unsupported-version, or missing
established stores likewise fail closed instead of being recreated.

The restore timeout is one absolute cooperative budget. SQLite copies run in
bounded page batches; structural, quick-check, foreign-key, integrity, and
count queries use SQLite VM progress interruption; and schema-owned rows and
private candidate-copy chunks check the same deadline. Checkpoint busy waiting
is capped to the remaining budget. Checks also surround staging, sidecar
handling, recovery, replacement, final publication, and durable flush
boundaries so the exclusive lease is released promptly on expiry.
An individual kernel call such as `fsync`, `replace`, `stat`, `read`, or
`write` cannot be interrupted after it starts, so one such in-flight call may
finish just beyond the requested timeout before cleanup releases ownership.

Profiles persist generation selections, not connection or process
configuration. Provider origins, credentials, API keys, custom headers, binary
paths, `server.json` paths, health observations, message text, and raw local
paths are excluded from profile data and safe repository diagnostics.

This slice is storage infrastructure only. It does not yet provide an STTS
profile-management library, character-assignment UI or authority acquisition,
roleplay routing, profile/card portability or synchronization, legacy-provider
profile execution, provider connection details, or managed audio.cpp process
behavior. Those remain separately reviewed Slice 2B, Slice 3, Slice 4, and
managed-process work. See
[ADR-028](../../../backlog/decisions/028-character-tts-generation-profile-ownership.md).

### Global defaults and Console request admission

TASK-710 represents global TTS defaults as one immutable
`TTSPreferencesSnapshot`. audio.cpp supports explicit selection modes in
`[app_tts]`:

```toml
[app_tts]
default_provider = "audio_cpp"
default_model_mode = "first_available" # or "exact"
default_voice_mode = "server_default"  # or "exact"
default_format = "wav"
default_speed = 1.0
```

`exact` mode also requires the corresponding non-empty `default_model` or
`default_voice`. `first_available` resolves the first model from one admitted
catalog snapshot, and `server_default` omits `voice` from the request. Existing
audio.cpp configurations that have no mode keys and contain blank model or
voice values read as `first_available` and `server_default` without a startup
write.

The settings UI translates its local Select sentinels before persistence, so a
sentinel cannot become an empty exact identifier. One atomic configuration
mutation always writes the authoritative mode keys. Exact values are
dual-written to the canonical and legacy exact keys; dynamic modes remove stale
exact values from both locations. Exact-mode configurations therefore remain
readable by older builds. Dynamic-mode downgrade is not transparent: save
explicit model and voice values before downgrading, or restore a trusted
pre-feature configuration backup.

`TTSRequestAdmissionCoordinator` freezes the complete preference selection,
resolves any dynamic model, reads the provider revision, and acquires the
matching registry lease under one writer-preferred admission gate. Settings
publication persists off the Textual event loop in one service-retained task,
then uses the exclusive side of that gate for a bounded handoff. A foreground
save may report **Saved — applying after current speech**; the admitted speech
continues, only the latest pending generation may become active, and the old
audio.cpp adapter closes before a replacement can be created.

Console **Speak** calls `TTSService.synthesize_default()`. An `audio_cpp`
selection uses the native adapter with locked WAV, speed `1.0`, and empty
options. The six retained providers continue through `LegacyTTSAdapter`. The
native complete WAV is still consumed through `TTSAudioResponse`'s asynchronous
iterator and closed through the existing artifact/playback lifecycle.

### Native audio.cpp adapter (external mode)

Slice 2 connects to one existing `audiocpp_server`; it does not launch or
supervise a process. Configuration comes only from `[app_tts.audio_cpp]`:

```toml
[app_tts.audio_cpp]
mode = "external"
base_url = "http://127.0.0.1:8080"
connect_timeout_seconds = 5
synthesis_timeout_seconds = 600
max_input_characters = 10000
max_response_bytes = 134217728
max_metadata_bytes = 1048576
max_catalog_models = 1000
max_voices_per_model = 1000
max_identifier_characters = 256
```

`base_url` must be a canonical absolute HTTP or HTTPS origin. Credentials,
non-root paths, query strings, fragments, and invalid ports are rejected. The
configuration has no environment override, authentication or custom-header
field, binary path, `server.json` path, or other process field. HTTPS keeps
certificate verification enabled. Invalid configuration is rejected during
local projection or adapter materialization with a safe, value-independent
`ValueError`, before any provider operation; the external adapter does not emit
`configuration_invalid`.

`connect_timeout_seconds` configures HTTP connection establishment and also
bounds the complete required health-plus-models discovery sequence, including
an eligible safe-GET retry. The same value independently bounds each optional
voice-discovery operation. `synthesis_timeout_seconds` bounds the speech request
through complete response consumption; the HTTP connect timeout still applies
inside it. There is no read-inactivity timer.

The adapter implements the pinned `audio_cpp_http_v1` structure from
audio.cpp commit
[`d3d748179e5ace353386fbf17bcaedfacf482d75`](https://github.com/0xShug0/audio.cpp/tree/d3d748179e5ace353386fbf17bcaedfacf482d75):

- Required readiness surfaces: `GET /health` and `GET /v1/models`.
- Optional lazy voice metadata:
  `GET /v1/audio/voices?model=<id>`.
- Complete speech response: `POST /v1/audio/speech`.

Readiness retains only bounded TTS model metadata. Voice discovery is lazy,
bounded, per model, and cached by provider configuration and catalog revision.
A missing or invalid optional voices endpoint produces no discovered voices;
it does not make an otherwise compatible provider unavailable. Callers
represent the server-selected voice as `None`: the UI-facing “Server default”
sentinel is not sent in the speech payload.

Requests accept a known model, non-empty bounded text, an optional safe voice,
WAV output, speed exactly `1.0`, and no adapter options. Synthesis sends one
non-retried POST containing only `model`, `input`, `response_format: "wav"`,
and an optional `voice`. Safe GET operations may receive one bounded retry.
All requests disable redirects and request identity encoding.

The adapter bounds metadata and audio reads before parsing. It rejects
compressed, oversized, malformed, or incompatible responses and validates the
entire response as structurally complete, uncompressed PCM16 RIFF/WAV.
Validated bytes are then yielded as one asynchronous chunk. The asynchronous
stream contract is preserved, but Slice 2 does not provide incremental audio
streaming.

`TTSOperationError` exposes only a stable code, safe message, retryability,
local operation ID, and optional recovery action. Connectivity and
required-contract failures make cached health stale; invalid requests, optional
voice failures, busy responses, generation failures, invalid audio, and
cancellation do not. There is no automatic fallback to another model or a
legacy provider.
Successful response metadata contains only safe scalar provenance, sample, and
bounded timing values. Logs exclude submitted text, configured origins and
values, response bodies, and rejected identifiers.

The registry admits only one active audio.cpp adapter. An unchanged normalized
configuration is a no-op. A changed configuration blocks new operations,
drains active leases, closes the old adapter, and only then installs the new
configuration; the replacement remains lazy, so old and new instances never
overlap.

Normal tests use fake HTTP transport and fixtures pinned to the reviewed
upstream commit. They require neither an audio.cpp binary nor model downloads.

The installed Homebrew package `audio-cpp 0.4` was characterized on
2026-07-25 as compatible with the pinned health, model, voice, and speech
endpoints and complete PCM16 `audio/wav` response contract. This is
compatible-build evidence only: it does not move the ADR-023 upstream pin or
grant Chatbook ownership of the external server process.

An isolated clean-config Textual Console UAT subsequently selected
`audio_cpp` at `http://127.0.0.1:8080` with `first_available` model and
`server_default` voice, generated a deterministic Mira response, and exercised
one native adapter. Console produced one owner-only (`0600`) complete WAV of
594,604 bytes: mono PCM16 at 44.1 kHz, 297,280 frames, and 6.741 seconds.
Observed lifecycle counts were complete `1`, playback `1`, progress `4`, and
streaming `0`; `/usr/bin/afplay` exited `0`. The same external listener identity
and healthy response were present before and after the run, and application
shutdown took no action on that user-owned process.

After the implementation was rebased, all 23 patches were range-diff
identical. Fresh focused and broad automated suites passed, but a second live
run was unavailable because the installed `audio-cpp 0.4` binary had no
running process, listener, or healthy endpoint. Chatbook intentionally did not
launch it; external-process ownership remains with the user.

### Catalog-driven STTS Playground (Slice 3)

TASK-569 implements the external audio.cpp Playground vertical. Opening the
Playground reads sealed registry descriptors through `TTSService`; descriptor
discovery does not resolve provider factories or materialize adapters. Only the
selected provider is resolved. Selecting `audio_cpp` for the first time performs
bounded readiness and model discovery against the saved external server.

Catalog and voice discovery use independent Textual worker groups. Their result
tokens include the canonical provider ID and configuration revision, plus the
catalog revision and model ID where applicable. Results from an old selection,
configuration, catalog, or model are discarded. Catalog refresh, generation,
and playback cannot cancel one another, and a second generation cannot replace
the active generation operation.

One catalog-control projection drives provider, model, voice, format, and speed
controls. For audio.cpp, the local **Server default** voice sentinel is initially
selected and becomes `voice=None`; it is never sent as an identifier. Format is
locked to WAV and speed to `1.0`. Switching to one of the six legacy providers
restores that provider's prior model, voice, format, speed, and provider-specific
control state. If refreshed metadata removes a selection, the Playground
announces and selects a valid fallback. A stale catalog remains visible but
disables new generation until readiness recovers.

Generation captures an immutable provider-neutral request. `audio_cpp` is the
native path and calls `TTSService.synthesize(TTSRequest)`; the six existing
providers remain on the temporary `generate_audio_stream()` compatibility path.
The validated complete WAV is stored as an immutable artifact containing its
provider, model, optional voice, source-text snapshot, operation ID, actual
format/content type, and safe response metadata. Playback and export use that
artifact, so later selector changes cannot relabel the result or its filename.

Stable adapter failures map to safe, actionable Playground messages and
recovery actions. Cancellation remains cancellation, existing artifacts remain
playable and exportable after discovery failures, and an audio.cpp generation
never automatically falls back to another model or provider. The UI and logs
do not expose submitted text, configured origins or values, credentials, raw
remote bodies, or unsafe remote identifiers.

Slice 3 connects only to an existing externally managed `audiocpp_server`.
User-provided binary and user-provided `server.json` launch, supervision, and
managed Playground controls remain deferred to Slices 4–5.

#### 3. Audio Service (`audio_service.py`)
Handles audio format conversion with:
- `StreamingAudioWriter`: Real-time encoding for streaming
- Support for MP3, Opus, AAC, FLAC, WAV, PCM
- Async and sync conversion methods

#### 4. Text Processing (`text_processing.py`)
Provides text preparation for TTS:
- `TextNormalizer`: Handles URLs, emails, phone numbers, units
- `TextChunker`: Splits long texts respecting sentence boundaries
- Language detection based on voice selection

## Backend Implementations

### OpenAI Backend

**Features:**
- Supports tts-1 and tts-1-hd models
- Multiple voices: alloy, echo, fable, onyx, nova, shimmer
- Streaming response support
- All OpenAI audio formats

**Configuration:**
```toml
[app_tts]
OPENAI_API_KEY_fallback = "sk-your-api-key"
```

### Kokoro Backend

**Features:**
- Local text-to-speech using Kokoro-82M model
- ONNX runtime support (PyTorch planned)
- Multiple voice packs
- Voice mixing capabilities (planned)
- No internet connection required

**Configuration:**
```toml
[app_tts]
KOKORO_ONNX_MODEL_PATH_DEFAULT = "models/kokoro-v0_19.onnx"
KOKORO_ONNX_VOICES_JSON_DEFAULT = "models/voices.json"
KOKORO_DEVICE_DEFAULT = "cpu"  # or "cuda" for GPU
KOKORO_MAX_TOKENS = 500
```

**Voices:**
- Female: af_bella, af_nicole, af_sarah, af_sky, bf_emma, bf_isabella
- Male: am_adam, am_michael, bm_george, bm_lewis

### Chatterbox Backend

**Features:**
- Zero-shot voice cloning with 7-20 seconds of reference audio
- Emotion exaggeration control
- Ultra-low latency streaming (< 200ms)
- Advanced text preprocessing (dot-letter correction, reference removal)
- Multi-candidate generation with Whisper validation
- Audio normalization and post-processing
- Voice library with metadata tracking
- Fallback strategies for robust generation
- MIT licensed open-source model

**Configuration:**
```toml
[app_tts]
CHATTERBOX_DEVICE = "cuda"  # or "cpu"
CHATTERBOX_EXAGGERATION = 0.5  # Emotion control (0.0-1.0)
CHATTERBOX_CFG_WEIGHT = 0.5    # Pace/style control
CHATTERBOX_TEMPERATURE = 0.5   # Voice variation (0.0-2.0)
CHATTERBOX_NUM_CANDIDATES = 1  # Number of candidates (1-5)
CHATTERBOX_VALIDATE_WHISPER = false  # Enable validation
CHATTERBOX_PREPROCESS_TEXT = true    # Text preprocessing
CHATTERBOX_NORMALIZE_AUDIO = true    # Audio normalization
CHATTERBOX_TARGET_DB = -20.0         # Target volume (dB)
CHATTERBOX_RANDOM_SEED = null        # For reproducibility
CHATTERBOX_MAX_CHUNK_SIZE = 500      # Max text chunk size
CHATTERBOX_VOICE_DIR = "~/.config/tldw_cli/chatterbox_voices"
```

**Advanced Features:**
- **Text Preprocessing**: Automatically converts "J.R.R." to "J R R", removes [1] references, URLs
- **Multi-Candidate Generation**: Generate multiple versions and select the best using Whisper
- **Voice Cloning**: Upload any 7-20 second audio clip for instant voice cloning
- **Metadata Tracking**: Save voices with creation time, duration, sample rate
- **Fallback Strategies**: Three-tier system (high_quality → balanced → safe)

### ElevenLabs Backend

**Features:**
- High-quality voice synthesis
- Advanced voice settings (stability, similarity boost, style)
- Multiple languages and accents
- Speaker boost for enhanced clarity
- Multiple output formats

**Configuration:**
```toml
[app_tts]
ELEVENLABS_API_KEY_fallback = "your-api-key"
ELEVENLABS_DEFAULT_VOICE = "voice-id"
ELEVENLABS_DEFAULT_MODEL = "eleven_multilingual_v2"
ELEVENLABS_OUTPUT_FORMAT = "mp3_44100_192"
ELEVENLABS_VOICE_STABILITY = 0.5
ELEVENLABS_SIMILARITY_BOOST = 0.8
ELEVENLABS_STYLE = 0.0
ELEVENLABS_USE_SPEAKER_BOOST = true
```

### Higgs Audio Backend

**Features:**
- State-of-the-art voice cloning from 15-30 second samples
- Multi-speaker dialog generation
- 15 built-in high-quality voices (professional, energetic, calm, etc.)
- Real-time streaming audio generation
- Voice profile management for custom voices
- Support for mixed cloned and built-in voices in dialogs
- Cross-lingual voice transfer

**Configuration:**
```toml
[app_tts]
HIGGS_MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
HIGGS_DEVICE = "cuda"  # or "cpu", "mps" for Apple Silicon
HIGGS_ENABLE_FLASH_ATTN = true
HIGGS_MAX_NEW_TOKENS = 2048
HIGGS_TEMPERATURE = 0.8
HIGGS_TOP_P = 0.95
HIGGS_REPETITION_PENALTY = 1.05
HIGGS_GUIDANCE_SCALE = 1.0
HIGGS_VOICE_SAMPLES_DIR = "~/.config/tldw_cli/higgs_voices"
```

**Voice Cloning:**
```python
# Create a voice profile
success = await backend.create_voice_profile(
    profile_name="custom_voice",
    reference_audio_path="/path/to/sample.wav",
    display_name="My Custom Voice"
)

# Use the cloned voice
request = OpenAISpeechRequest(
    input="Hello from my cloned voice!",
    voice="custom_voice"
)
```

**Multi-Speaker Dialog:**
```python
# Format text with speaker tags
dialog_text = """[Speaker: professional_female]
Welcome to our presentation.

[Speaker: energetic_male]
We're excited to share our findings!

[Speaker: custom_voice]
Let me add my perspective..."""

request = OpenAISpeechRequest(
    input=dialog_text,
    voice="multi"  # Special voice for multi-speaker
)
```

## Installation

### Basic Installation
The core TTS functionality (OpenAI, ElevenLabs) is included with the base installation:
```bash
pip install tldw_chatbook
```

### Local TTS Support
For local TTS models like Kokoro, install the optional dependencies:
```bash
pip install tldw_chatbook[local_tts]
```

### Chatterbox Support
For Chatterbox voice cloning capabilities:
```bash
pip install tldw_chatbook[chatterbox]
```

This installs:
- chatterbox-tts: Core Chatterbox model
- torchaudio: Audio processing
- torch: PyTorch runtime
- faster-whisper: For validation (optional)

### Higgs Audio Support
For state-of-the-art voice cloning and multi-speaker generation:
```bash
pip install tldw_chatbook[higgs_tts]
```

This installs:
- boson-multimodal: Higgs Audio V2 model
- torch: PyTorch runtime
- torchaudio: Audio processing and voice cloning
- numpy/scipy: Audio manipulation
- librosa: Advanced audio features
- soundfile: Audio I/O
- transformers: Text processing

Local TTS installs:
- kokoro-onnx: ONNX runtime for Kokoro
- scipy: Audio processing
- nltk: Text tokenization
- pyaudio/pydub: Audio playback
- transformers: Advanced tokenization
- torch: PyTorch support (for future backends)
- onnxruntime: ONNX model inference

### Kokoro Model Setup
1. Download the model files:
   - Model: `kokoro-v0_19.onnx` (~300MB)
   - Voices: `voices.json`

2. Place them in your configured paths or use the download utility:
   ```python
   from tldw_chatbook.TTS.utils.download_models import download_kokoro_model
   await download_kokoro_model()
   ```

## Usage

### Basic Usage in the App

1. Click the speak button (🔊) on any chat message
2. The TTS service will:
   - Use the configured default provider
   - Generate audio with the default voice
   - Play the audio automatically

### TTS Playground (S/TT/S Tab)

The S/TT/S tab provides a comprehensive TTS testing environment:

1. **Text Input**: Enter any text to synthesize
2. **Provider Selection**: Choose `audio_cpp` or one of the six legacy
   providers from registry descriptors
3. **Voice Selection**: Discovered audio.cpp voices with Server default, or
   legacy provider-specific voices including custom uploads
4. **Advanced Settings**:
   - **audio.cpp**: Catalog-selected model, complete WAV, and speed `1.0`
   - **Chatterbox**: Exaggeration, CFG weight, temperature, candidates, validation
   - **ElevenLabs**: Stability, similarity boost, style, speaker boost
   - **Kokoro**: Language selection
5. **Audio Controls**: Play, pause, stop, and export generated audio
6. **Generation Log**: Real-time feedback on TTS processing

### Programmatic Usage

```python
from tldw_chatbook.TTS import OpenAISpeechRequest, get_tts_service

# The application binds the service before callers request it.
tts_service = await get_tts_service()

request = OpenAISpeechRequest(
    model="tts-1",
    input="Hello, world!",
    voice="alloy",
    response_format="mp3",
    speed=1.0
)

internal_model_id = "openai_official_tts-1"
async for chunk in tts_service.generate_audio_stream(request, internal_model_id):
    audio_file.write(chunk)
```

`TTSService.synthesize(TTSRequest)` is the native-adapter API. Use it directly
for `audio_cpp`. Its complete validated WAV is exposed as one chunk through the
response's asynchronous iterator, and callers must close the response. The six
legacy registry entries require private bridge metadata that
`generate_audio_stream()` supplies; do not call `synthesize()` directly for
those entries.

### Event System Integration

The TTS module integrates with Textual's event system:

```python
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSRequestEvent

# Request TTS generation
app.post_message(TTSRequestEvent(
    text="Text to speak",
    message_id="msg_123",
    voice="alloy"  # Optional voice override
))

# Handle completion
@on(TTSCompleteEvent)
async def handle_tts_complete(self, event: TTSCompleteEvent):
    if event.error:
        self.notify(f"TTS failed: {event.error}")
    else:
        # Audio file available at event.audio_file
        play_audio_file(event.audio_file)
```

## Configuration Reference

### Global Settings
```toml
[app_tts]
default_provider = "openai"  # openai, kokoro, elevenlabs, chatterbox
default_voice = "alloy"      # Provider-specific voice
default_model = "tts-1"      # Provider-specific model
default_format = "mp3"       # Audio output format
default_speed = 1.0          # Speech speed (0.25-4.0)
```

### Exact legacy route allowlist

The compatibility generator accepts only these internal model IDs:

- `openai_official_tts-1` → `openai`
- `openai_official_tts-1-hd` → `openai`
- `openai_official_tts1` → `openai`
- `openai_official_tts1hd` → `openai`
- `elevenlabs_eleven_monolingual_v1` → `elevenlabs`
- `elevenlabs_eleven_multilingual_v1` → `elevenlabs`
- `elevenlabs_eleven_multilingual_v2` → `elevenlabs`
- `elevenlabs_eleven_turbo_v2` → `elevenlabs`
- `elevenlabs_eleven_turbo_v2_5` → `elevenlabs`
- `elevenlabs_eleven_flash_v2` → `elevenlabs`
- `elevenlabs_eleven_flash_v2_5` → `elevenlabs`
- `elevenlabs_english_v1` → `elevenlabs`
- `elevenlabs_elevenlabs` → `elevenlabs`
- `local_kokoro_default_onnx` → `kokoro`
- `local_kokoro_default_pytorch` → `kokoro`
- `local_chatterbox_default` → `chatterbox`
- `local_higgs_default` → `higgs`
- `local_higgs_v2` → `higgs`
- `alltalk_default` → `alltalk`
- `alltalk_alltalk` → `alltalk`

These IDs are temporary bridge inputs, not native provider/model identities.
New native-adapter code selects a canonical provider and opaque model ID
explicitly with `TTSRequest`.

### Audio Formats
Supported formats vary by provider:
- **All providers**: mp3, wav, pcm
- **OpenAI**: opus, aac, flac
- **ElevenLabs**: Various bitrate/quality options
- **Kokoro**: Best with wav/pcm for quality
- **Chatterbox**: All formats via audio service conversion

## Advanced Features

### Text Normalization
Configure text preprocessing:
```python
normalization_options = NormalizationOptions(
    normalize=True,
    unit_normalization=True,      # 10KB → 10 kilobytes
    url_normalization=True,       # https://example.com → example dot com
    email_normalization=True,     # user@example.com → user at example dot com
    phone_normalization=True,     # 555-1234 → 5 5 5 1 2 3 4
)
```

### Voice Mixing (Kokoro)
Combine multiple voice characteristics:
```python
# Future enhancement
voice = "af_bella:0.6,af_sarah:0.4"  # 60% bella, 40% sarah
```

### Voice Cloning (Chatterbox)
Clone any voice with a reference audio:
```python
# Use custom voice
voice = "custom:/path/to/reference.wav"

# Or save for reuse
await backend.save_reference_voice_with_metadata(
    name="my_voice",
    audio_path="/path/to/reference.wav",
    metadata={"speaker": "John Doe", "emotion": "neutral"}
)
```

### Advanced Text Processing (Chatterbox)
```python
# Automatic preprocessing handles:
# - "Dr. Smith" → "Doctor Smith"
# - "J.R.R. Tolkien" → "J R R Tolkien"
# - "See reference [1]" → "See reference"
# - URLs and email addresses are normalized
```

### Multi-Candidate Generation (Chatterbox)
```python
# Generate multiple candidates and select best
extra_params = {
    "num_candidates": 3,
    "validate_with_whisper": True,
    "temperature": 0.7
}
```

### Legacy Streaming with Chunk Processing

Concrete backend streaming is retained only inside the temporary bridge:

```python
async for chunk in backend.generate_speech_stream(request):
    # Process chunks in real-time
    await websocket.send(chunk)
    
    # Or accumulate for post-processing
    chunks.append(chunk)
```

## Performance Considerations

### Kokoro Performance
- **CPU**: ~3.5s latency for first token
- **GPU**: ~0.3s latency for first token
- **Generation speed**: 35-100x realtime
- **Token rate**: ~140 tokens/second

### Chatterbox Performance
- **CPU**: ~2.0s latency for first generation
- **GPU**: <200ms latency (ultra-low)
- **Generation speed**: Real-time to 50x realtime
- **Multi-candidate overhead**: ~1.5x per additional candidate
- **Whisper validation**: +0.5-1.0s per candidate
- **Voice cloning**: 7-20 second reference audio required
- **Model size**: ~500MB (0.5B parameters)

### Optimization Tips
1. **Use streaming** for better perceived performance
2. **Pre-download models** for Kokoro to avoid first-run delays
3. **Cache frequently used phrases** (future enhancement)
4. **Adjust chunk size** based on network conditions
5. **Use appropriate format**:
   - PCM for lowest latency
   - MP3 for compatibility
   - Opus for best compression

### Memory Usage
- Kokoro model: ~300MB when loaded
- Chatterbox model: ~500MB when loaded
- Audio buffers: Minimal with streaming
- Text processing: Negligible
- Voice library: ~10MB per saved voice

## Troubleshooting

### Common Issues

#### "TTS service not available"
- Check if TTS was initialized successfully
- Verify API keys are configured
- Check logs for initialization errors

#### "No audio output"
- Verify audio playback system is working
- Check file permissions for temp directory
- Ensure audio format is supported by system

#### "Kokoro model not found"
- Download model files to configured paths
- Check file permissions
- Verify ONNX runtime is installed

#### "Chatterbox voice cloning fails"
- Ensure reference audio is 7-20 seconds
- Check audio format (WAV recommended)
- Verify PyTorch/CUDA installation
- Check available GPU memory

#### "API key errors"
- Check key format and validity
- Verify key has required permissions
- Check API quotas/limits

#### "Multi-candidate generation slow"
- Reduce number of candidates
- Disable Whisper validation
- Use GPU acceleration
- Check system resources

### Debug Logging
Enable debug logging for detailed information:
```toml
[logging]
level = "DEBUG"
```

Check logs at: `~/.share/tldw_cli/logs/`

### Performance Issues
1. **Slow generation**:
   - Use streaming for better UX
   - Consider using faster models (tts-1 vs tts-1-hd)
   - Check network latency for API calls
   - For Chatterbox: reduce candidates, disable validation

2. **High memory usage**:
   - Unload models when not in use
   - Use streaming instead of full generation
   - Monitor with Stats tab
   - Clear voice library cache periodically

3. **Voice quality issues**:
   - Adjust exaggeration/CFG parameters
   - Try different reference audio
   - Enable text preprocessing
   - Use multi-candidate generation

## API Reference

### Schemas

#### OpenAISpeechRequest
```python
class OpenAISpeechRequest(BaseModel):
    model: str                    # Model identifier
    input: str                    # Text to synthesize
    voice: str                    # Voice selection
    response_format: str          # Audio format
    speed: float = 1.0           # Speed adjustment (0.25-4.0)
    stream: bool = True          # Enable streaming
    lang_code: Optional[str]     # Language hint
    normalization_options: Optional[NormalizationOptions]
    extra_params: Optional[Dict[str, Any]]  # Provider-specific parameters
```

### Native Adapter Methods

#### ensure_ready()
Initialize or connect to provider resources lazily. The service synthesis path
invokes this as its prerequisite.

#### get_catalog()
Own readiness and return provider health, models, formats, voices, and
supported controls. Callers do not pre-resolve a concrete adapter.

#### get_voices(model_id, refresh=False)
Own readiness and lazily return bounded voices for one model. A refresh bypasses
the adapter's current voice result without exposing the adapter to callers.

#### synthesize()
Return a provider-neutral `TTSAudioResponse` with an asynchronous byte stream.

#### close()
Release provider resources. The registry controls when adapter shutdown occurs.

## Future Enhancements

### Planned Features
1. **SSML Support**: Advanced speech markup
2. **Caching System**: Reduce repeated generations
3. **Batch Processing**: Multiple texts in one request
4. **Real-time Streaming**: WebSocket-based streaming
5. **More Backends**: Edge-TTS, Coqui, Piper
6. **Cross-provider Voice Transfer**: Use one provider's voice with another

### Experimental Features
- **Emotion Control**: Adjust emotional tone
- **Prosody Tuning**: Fine-tune speech characteristics
- **Multi-speaker**: Different voices in one text
- **Audio Effects**: Post-processing effects

## Contributing

### Adding a Native Adapter

1. Implement the asynchronous adapter contract (`ensure_ready`,
   `get_catalog`, `get_voices`, `synthesize`, and `close`) using the
   provider-neutral request, response, catalog, health, and progress types.
   `get_catalog()` and `get_voices()` own their readiness step;
   `ensure_ready()` remains the service synthesis prerequisite.
2. Add one explicit provider specification to application service
   construction.
3. Add configuration validation, contract tests, and provider documentation.

Do not register new providers in `TTS_Backends.py` or subclass
`TTSBackendBase`; those APIs exist only for the six-provider temporary bridge.

### Testing
Run TTS-specific tests:
```bash
pytest Tests/TTS/
```

## Security Considerations

1. **API Keys**: Never log or display API keys
2. **Input Validation**: All text inputs are sanitized
3. **File Paths**: Temporary files use secure generation
4. **Network**: External audio.cpp accepts an explicit HTTP or HTTPS origin;
   synthesis text is sent to that configured origin, HTTPS certificate
   verification remains enabled, and redirects are disabled
5. **Local Models**: Verify model file integrity
6. **Voice Cloning**: Be aware of ethical implications
   - Only clone voices with permission
   - Chatterbox adds watermarks to generated audio
   - Store voice metadata securely
7. **Reference Audio**: Validate file formats and sizes

## License

The TTS module follows the main project's AGPL-3.0+ license. Individual model licenses:
- Kokoro: Apache 2.0
- Chatterbox: MIT License
- API providers: Subject to their respective terms of service
