# External audio.cpp Native TTS Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register `audio_cpp` as Chatbook's first native TTS provider and support bounded discovery, lazy voices, and validated complete-WAV synthesis against one existing `audiocpp_server`.

**Architecture:** Add a small provider-neutral voice-discovery seam to the existing service, then implement the external adapter as three focused units: immutable configuration, pure pinned-contract/WAV validation, and async HTTP lifecycle/orchestration. Register one lazy exclusive provider spec before the six unchanged legacy specs. The adapter uses `httpx` with redirects disabled, identity encoding, bounded reads, one retry for safe GETs, and no retry for speech POSTs.

**Tech Stack:** Python 3.11+, asyncio, httpx, dataclasses, stdlib JSON/struct/urllib, pytest, pytest-asyncio, httpx.MockTransport

---

## Scope boundary

- Implement external connection mode only.
- Do not add Textual/STTS controls, process launching, binary paths, `server.json`
  parsing, port ownership, supervision, restart, or managed-mode behavior.
- Preserve complete WAV output as one asynchronous byte-stream chunk.
- Preserve all six compatibility adapters and the legacy generation API.
- Pin contract fixtures to `0xShug0/audio.cpp` commit
  `d3d748179e5ace353386fbf17bcaedfacf482d75`.

## File map

- Create `tldw_chatbook/TTS/audio_cpp_config.py`: immutable external-mode
  configuration defaults, projection from app config, and local validation.
- Create `tldw_chatbook/TTS/audio_cpp_contract.py`: pure bounded JSON/model/voice
  parsing, safe timing-header parsing, and strict PCM16 WAV validation.
- Create `tldw_chatbook/TTS/adapters/__init__.py`: native-adapter package marker.
- Create `tldw_chatbook/TTS/adapters/audio_cpp.py`: HTTP lifecycle, health/catalog
  state, voice cache, request validation, error mapping, and synthesis.
- Modify `tldw_chatbook/TTS/adapter_types.py`: provider-neutral voice discovery,
  immutable response metadata, and stable safe operation errors.
- Modify `tldw_chatbook/TTS/adapter_registry.py`: leased voice discovery forwarding.
- Modify `tldw_chatbook/TTS/TTS_Generation.py`: service voice-discovery forwarding
  and managed-response metadata preservation.
- Modify `tldw_chatbook/TTS/adapter_bootstrap.py`: prepend the lazy native
  `audio_cpp` spec with exclusive reconfiguration.
- Modify `tldw_chatbook/TTS/legacy_bridge.py`: return static catalog voices through
  the provider-neutral voice operation.
- Modify `tldw_chatbook/TTS/__init__.py`: export the new safe public contracts.
- Create `Tests/TTS/fixtures/audio_cpp_http_v1/*.json`: pinned structural fixtures
  and provenance.
- Create `Tests/TTS/test_audio_cpp_config.py`: configuration and projection tests.
- Create `Tests/TTS/test_audio_cpp_contract.py`: pure parser and WAV tests.
- Create `Tests/TTS/test_audio_cpp_adapter.py`: fake-HTTP discovery/synthesis/error
  tests.
- Modify `Tests/TTS/adapter_fakes.py`, `Tests/TTS/test_adapter_registry.py`,
  `Tests/TTS/test_adapter_types.py`, and
  `Tests/TTS/test_tts_registry_service.py`: provider-neutral seam and bootstrap
  regressions.
- Modify `Docs/Development/TTS/TTS_MODULE_GUIDE.md`,
  `Docs/superpowers/specs/2026-07-23-audio-cpp-tts-adapter-registry-design.md`,
  and `backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md`:
  document the landed external adapter and the explicit provider-neutral lazy
  voice operation.
- Modify
  `backlog/tasks/task-560 - Add-external-audio.cpp-native-TTS-adapter.md`: record
  plan, verification, acceptance completion, and implementation notes.

### Task 1: Extend the provider-neutral response and voice contracts

**Files:**
- Modify: `tldw_chatbook/TTS/adapter_types.py`
- Modify: `tldw_chatbook/TTS/adapter_registry.py`
- Modify: `tldw_chatbook/TTS/TTS_Generation.py`
- Modify: `tldw_chatbook/TTS/legacy_bridge.py`
- Modify: `tldw_chatbook/TTS/__init__.py`
- Modify: `Tests/TTS/adapter_fakes.py`
- Modify: `Tests/TTS/test_adapter_types.py`
- Modify: `Tests/TTS/test_adapter_registry.py`
- Modify: `Tests/TTS/test_tts_registry_service.py`

- [x] **Step 1: Write failing tests for immutable response metadata and safe errors**

Add coverage proving response metadata is copied and immutable and that
`TTSOperationError` exposes only a stable code, safe message, retryability,
operation ID, and optional recovery action.

```python
error = TTSOperationError(
    code="server_busy",
    message="The audio.cpp server is busy",
    retryable=True,
    operation_id="op-test",
    recovery_action="retry",
)
assert error.code == "server_busy"
assert str(error) == "The audio.cpp server is busy"
```

- [x] **Step 2: Write failing tests for leased voice discovery**

Extend the fake adapter with:

```python
async def get_voices(
    self, model_id: str, refresh: bool = False
) -> tuple[str, ...]:
    ...
```

Verify registry/service forwarding retains and releases the provider lease and
leaves legacy voices unchanged. The implemented adapter owns readiness inside
`get_catalog()` and `get_voices()`; only synthesis keeps the explicit service
`ensure_ready()` prerequisite.

- [x] **Step 3: Run the focused tests and confirm RED**

Run:

```bash
pytest \
  Tests/TTS/test_adapter_types.py \
  Tests/TTS/test_adapter_registry.py::test_get_voices_materializes_lazily_and_releases_its_lease \
  Tests/TTS/test_tts_registry_service.py::test_catalog_voice_and_reconfigure_delegate_to_registry -q
```

Expected: failures for missing metadata, error, and voice APIs.

- [x] **Step 4: Implement the minimal provider-neutral seam**

Add:

```python
TTSOperationCode = Literal[
    "configuration_invalid",
    "connection_unavailable",
    "contract_incompatible",
    "not_configured",
    "request_invalid",
    "model_invalid",
    "server_busy",
    "generation_failed",
    "audio_response_invalid",
    "generation_timeout",
]

class TTSOperationError(RuntimeError):
    ...
```

Add immutable `metadata` to `TTSAudioResponse`, preserve it through
`_ManagedAudioResponse`, and forward `get_voices()` through adapter, registry,
and service contracts. The legacy adapter returns the selected static model's
catalog voices without exposing its manager.

- [x] **Step 5: Run focused tests and confirm GREEN**

- [x] **Step 6: Commit**

```bash
git add tldw_chatbook/TTS Tests/TTS
git commit -m "feat(tts): add provider-neutral voice and error contracts"
```

### Task 2: Add external audio.cpp configuration and lazy registration

**Files:**
- Create: `tldw_chatbook/TTS/audio_cpp_config.py`
- Modify: `tldw_chatbook/TTS/adapter_bootstrap.py`
- Create: `Tests/TTS/test_audio_cpp_config.py`
- Modify: `Tests/TTS/test_tts_registry_service.py`
- Modify: `Tests/TTS/test_tts_app_ownership.py`

- [x] **Step 1: Write failing configuration tests**

Cover defaults and raw/normalized nested config projection. Reject unsupported
mode, relative URLs, non-HTTP schemes, credentials, paths other than `/`, query,
fragment, invalid ports, booleans where numeric values are required, non-finite
timeouts, and non-positive limits. Verify diagnostics never echo submitted
values.

- [x] **Step 2: Write the failing bootstrap test**

Expect exact descriptor order:

```python
(
    "audio_cpp",
    "openai",
    "elevenlabs",
    "kokoro",
    "chatterbox",
    "higgs",
    "alltalk",
)
```

Assert `audio_cpp` is native, has display label `audio.cpp`, has no alias, is
exclusive-reconfigure, owns a deep config snapshot, and remains unmaterialized
during app construction.

- [x] **Step 3: Run tests and confirm RED**

- [x] **Step 4: Implement configuration and provider-spec construction**

Use an immutable config dataclass with the approved defaults:

```python
base_url = "http://127.0.0.1:8080"
connect_timeout_seconds = 5.0
synthesis_timeout_seconds = 600.0
max_input_characters = 10_000
max_response_bytes = 128 * 1024 * 1024
max_metadata_bytes = 1024 * 1024
max_catalog_models = 1000
max_voices_per_model = 1000
max_identifier_characters = 256
```

Projection reads `[app_tts.audio_cpp]` from the raw config and falls back to
`APP_TTS_CONFIG.audio_cpp`. Do not add managed fields or environment overrides
in this slice.

- [x] **Step 5: Run tests and confirm GREEN**

- [x] **Step 6: Commit**

```bash
git add tldw_chatbook/TTS/audio_cpp_config.py \
  tldw_chatbook/TTS/adapter_bootstrap.py \
  Tests/TTS/test_audio_cpp_config.py \
  Tests/TTS/test_tts_registry_service.py \
  Tests/TTS/test_tts_app_ownership.py
git commit -m "feat(tts): register lazy external audio cpp provider"
```

### Task 3: Implement pinned contract and WAV validation

**Files:**
- Create: `tldw_chatbook/TTS/audio_cpp_contract.py`
- Create: `Tests/TTS/fixtures/audio_cpp_http_v1/provenance.json`
- Create: `Tests/TTS/fixtures/audio_cpp_http_v1/health.json`
- Create: `Tests/TTS/fixtures/audio_cpp_http_v1/models.json`
- Create: `Tests/TTS/fixtures/audio_cpp_http_v1/voices.json`
- Create: `Tests/TTS/fixtures/audio_cpp_http_v1/server_busy.json`
- Create: `Tests/TTS/test_audio_cpp_contract.py`

- [x] **Step 1: Add pinned fixtures and failing parser tests**

Record repository, commit, review date, endpoint, and source path in
`provenance.json`. Cover exact upstream health shape, model list shape, TTS-only
filtering, task normalization, duplicate IDs, control characters, identifier
lengths, count limits, malformed roots, and optional voice failures.

- [x] **Step 2: Add failing PCM16 WAV tests**

Generate tiny deterministic WAV byte strings in tests and mutate RIFF size,
WAVE signature, chunk lengths, format tag, channels, sample rate, block align,
bits per sample, data alignment, padding, truncation, and trailing bytes.

- [x] **Step 3: Run tests and confirm RED**

- [x] **Step 4: Implement pure parsers**

Keep this module free of HTTP and adapter state. Return immutable normalized
model/voice data and parsed WAV metadata. Require a structurally complete
uncompressed PCM16 RIFF/WAVE with at least one complete frame.

- [x] **Step 5: Run tests and confirm GREEN**

- [x] **Step 6: Commit**

```bash
git add tldw_chatbook/TTS/audio_cpp_contract.py \
  Tests/TTS/fixtures/audio_cpp_http_v1 \
  Tests/TTS/test_audio_cpp_contract.py
git commit -m "feat(tts): validate pinned audio cpp HTTP contract"
```

### Task 4: Implement external discovery and lazy voice caching

**Files:**
- Create: `tldw_chatbook/TTS/adapters/__init__.py`
- Create: `tldw_chatbook/TTS/adapters/audio_cpp.py`
- Create: `Tests/TTS/test_audio_cpp_adapter.py`

- [x] **Step 1: Write failing fake-HTTP discovery tests**

Use `httpx.MockTransport` to prove:

- first readiness calls `/health` and `/v1/models`;
- only TTS models enter the catalog;
- zero TTS models returns `not_configured`;
- cached catalog avoids HTTP until refresh;
- every successful authoritative refresh increments revision and invalidates
  voice caches even when models are unchanged;
- voices load only for the requested model;
- missing/malformed/oversized voices produce an empty tuple without making the
  provider unavailable;
- connection and required-contract failures return safe stale health;
- safe GETs retry at most once;
- every request sends `Accept-Encoding: identity`;
- redirects are not followed.

- [x] **Step 2: Run discovery tests and confirm RED**

- [x] **Step 3: Implement readiness, catalog, and voice state**

Construct `httpx.AsyncClient(follow_redirects=False)` with connect-only network
timeout and explicit operation-level deadlines. Read every body incrementally
through one bounded helper. Never include URLs or remote bodies in errors or
logs.

- [x] **Step 4: Run discovery tests and confirm GREEN**

- [x] **Step 5: Commit**

```bash
git add tldw_chatbook/TTS/adapters Tests/TTS/test_audio_cpp_adapter.py
git commit -m "feat(tts): discover audio cpp models and voices"
```

### Task 5: Implement complete-WAV synthesis and safe error mapping

**Files:**
- Modify: `tldw_chatbook/TTS/adapters/audio_cpp.py`
- Modify: `Tests/TTS/test_audio_cpp_adapter.py`
- Modify: `Tests/TTS/test_tts_registry_service.py`

- [x] **Step 1: Write failing request and response tests**

Cover empty/oversized text, provider mismatch, missing model with one refresh,
unsafe voice, non-WAV format, non-default speed, and unknown options. Assert the
POST payload contains exactly `model`, `input`, `response_format`, and optional
`voice`.

Cover content-length preflight, incremental response bound, non-identity
encoding, accepted WAV/binary MIME types, invalid MIME, malformed WAV, parsed
sample rate, immutable safe timing metadata, one asynchronous chunk, response
cleanup, and service lease lifetime.

- [x] **Step 2: Write failing error tests**

Cover:

- `503` plus structured `server_busy` as retryable;
- speech `404` as contract-incompatible;
- speech `500` as one model refresh without POST retry;
- vanished model after `500` as model-invalid;
- connection failure as unavailable and stale;
- overall synthesis timeout as retryable;
- cancellation propagation without health poisoning or retry;
- malformed error JSON without remote-text disclosure.

- [x] **Step 3: Run synthesis tests and confirm RED**

- [x] **Step 4: Implement synthesis**

Report indeterminate progress before POST and completion only after WAV
validation. Wrap the full POST plus bounded body read in
`asyncio.timeout(synthesis_timeout_seconds)`. Do not set a read-inactivity
deadline and never retry the POST.

- [x] **Step 5: Run synthesis and service tests and confirm GREEN**

- [x] **Step 6: Commit**

```bash
git add tldw_chatbook/TTS/adapters/audio_cpp.py \
  Tests/TTS/test_audio_cpp_adapter.py \
  Tests/TTS/test_tts_registry_service.py
git commit -m "feat(tts): synthesize validated audio cpp WAV responses"
```

### Task 6: Harden lifecycle, privacy, and exclusive reconfiguration

**Files:**
- Modify: `Tests/TTS/test_audio_cpp_adapter.py`
- Modify: `Tests/TTS/test_tts_logging_privacy.py`
- Modify: `Tests/TTS/test_adapter_registry.py` only if a native-specific
  regression reveals a registry gap

- [x] **Step 1: Add failing lifecycle/privacy regressions**

Cover concurrent first use, idempotent adapter close, cancellation during body
read, close during active response, and exclusive configuration handoff with no
old/new adapter overlap. Capture logs and assert absence of synthesis text,
base URL, response bodies, identifiers rejected as invalid, and config values.

- [x] **Step 2: Run tests and confirm RED where behavior is missing**

- [x] **Step 3: Apply only the minimal lifecycle fixes**

Prefer adapter-local fixes. Change the registry only if the new native test
demonstrates a provider-neutral lifecycle defect.

- [x] **Step 4: Run tests and confirm GREEN**

- [x] **Step 5: Commit**

```bash
git add Tests/TTS/test_audio_cpp_adapter.py \
  Tests/TTS/test_tts_logging_privacy.py \
  tldw_chatbook/TTS
git commit -m "test(tts): harden audio cpp lifecycle and privacy"
```

### Task 7: Update governing documentation and task evidence

**Files:**
- Modify: `Docs/Development/TTS/TTS_MODULE_GUIDE.md`
- Modify: `Docs/superpowers/specs/2026-07-23-audio-cpp-tts-adapter-registry-design.md`
- Modify: `backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md`
- Modify:
  `backlog/tasks/task-560 - Add-external-audio.cpp-native-TTS-adapter.md`

- [x] **Step 1: Document the landed external adapter**

Document configuration, privacy boundary, complete-WAV behavior, pinned
contract, safe limits, voice semantics, error behavior, and the absence of
automatic fallback.

- [x] **Step 2: Resolve the lazy-voice contract wording**

Amend the approved design/module guide from four adapter operations to five by
documenting provider-neutral `get_voices(model_id, refresh=False)`. This is the
minimal interface required by the already-approved lazy per-model voice
discovery behavior and avoids concrete-adapter access.

- [x] **Step 3: Preserve later-slice deferrals**

State explicitly that STTS catalog-driven controls are Slice 3 and managed
binary/`server.json` supervision is Slices 4–5.

- [x] **Step 4: Commit documentation**

```bash
git add Docs backlog/decisions backlog/tasks/task-560*
git commit -m "docs(tts): document external audio cpp adapter"
```

### Task 8: Run the full verification gate and finish TASK-560

**Files:**
- Modify:
  `backlog/tasks/task-560 - Add-external-audio.cpp-native-TTS-adapter.md`

- [x] **Step 1: Run focused tests**

```bash
pytest \
  Tests/TTS/test_audio_cpp_config.py \
  Tests/TTS/test_audio_cpp_contract.py \
  Tests/TTS/test_audio_cpp_adapter.py \
  Tests/TTS/test_adapter_types.py \
  Tests/TTS/test_adapter_registry.py \
  Tests/TTS/test_tts_registry_service.py \
  Tests/TTS/test_tts_app_ownership.py \
  Tests/TTS/test_tts_logging_privacy.py -q
```

- [x] **Step 2: Run broad regressions**

```bash
pytest \
  Tests/TTS \
  Tests/UI/test_stts_capability_state.py \
  Tests/UI/test_stts_settings_widget.py \
  Tests/Audio_Services/test_local_audio_services_service.py \
  Tests/Media/test_local_media_reading_service.py -q
```

- [x] **Step 3: Run static and boundary checks**

```bash
ruff check <all changed Python files>
ruff format --check <all changed Python files>
python -m compileall -q tldw_chatbook/TTS
python -m mypy \
  tldw_chatbook/TTS/adapter_types.py \
  tldw_chatbook/TTS/adapter_registry.py \
  tldw_chatbook/TTS/audio_cpp_config.py \
  tldw_chatbook/TTS/audio_cpp_contract.py \
  tldw_chatbook/TTS/adapters/audio_cpp.py \
  tldw_chatbook/TTS/adapter_bootstrap.py \
  tldw_chatbook/TTS/TTS_Generation.py
rg -n "subprocess|create_subprocess|binary_path|server_config_path|server\\.json" \
  tldw_chatbook/TTS/audio_cpp_config.py \
  tldw_chatbook/TTS/audio_cpp_contract.py \
  tldw_chatbook/TTS/adapters/audio_cpp.py
git diff --check
```

Expected: tests and static checks pass; the scope-boundary search prints no
production matches.

- [x] **Step 4: Perform a security and scope self-review**

Verify exact provider identity, lazy materialization, no redirects, identity
encoding, bounded metadata/audio, strict PCM16 WAV, no POST retry, safe
diagnostics, no text/value logging, no fallback, and no managed/UI code.

- [x] **Step 5: Finish Backlog evidence**

Check every acceptance and Definition-of-Done item, add concise implementation
notes with exact verification counts, and move TASK-560 to Done only after all
evidence is current.

- [x] **Step 6: Final commit**

```bash
git add "backlog/tasks/task-560 - Add-external-audio.cpp-native-TTS-adapter.md"
git commit -m "chore(tts): close external audio cpp adapter task"
```

## ADR check

ADR required: yes

ADR path:
`backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md`

Reason: ADR-023 already governs the native provider boundary, external
audio.cpp contract, complete-WAV interface, security limits, exclusive
reconfiguration, and ordered slice delivery. This slice implements and
clarifies that accepted decision; no new ADR is required.
