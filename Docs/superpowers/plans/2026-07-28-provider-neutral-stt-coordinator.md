# Provider-Neutral STT Coordinator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add dependency-free STT contracts, exact provider metadata, deterministic language routing, explicit retry policy, and a thin compatibility facade over the existing transcription backend.

**Architecture:** The new `tldw_chatbook.STT` package owns immutable values, a sealed registry, routing, coordination, and an injected legacy bridge. The current native-heavy implementation remains private and unchanged in behavior; the public `TranscriptionService` becomes an explicit forwarding facade. No native runtime, artifact download, persistence, executor, or default-promotion work belongs in this task.

**Tech Stack:** Python 3.11+, stdlib dataclasses/enums/protocols, pytest, Ruff, mypy.

---

## Scope and ADR

ADR required: yes
ADR path: `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`
Reason: ADR-025 already governs the provider boundary, language routing, fallback policy, and compatibility facade; this task implements that decision without changing it.

Included:

- Typed request, result, segment, progress, cancellation, provenance, provider, capability, and failure values.
- Exact-ID sealed provider registry with declared/runtime capability validation.
- Routing for default and exact providers.
- Coordinator-side capability checks, normalized results, and explicit action policy.
- Injected retained-provider bridge and a thin public compatibility facade.
- Dependency-free tests for every TASK-599 policy row.

Excluded:

- Native Parakeet, faster-whisper, or transcribe.cpp adapters.
- Artifact acquisition, persistence migrations, heavy workers, and UI changes.
- Automatic cross-engine execution.
- Default promotion or legacy-provider removal.

## File map

- Create `tldw_chatbook/STT/contracts.py`: immutable public values and validation.
- Create `tldw_chatbook/STT/registry.py`: metadata, adapter protocol, and sealed registry.
- Create `tldw_chatbook/STT/routing.py`: built-in declarations and deterministic routing.
- Create `tldw_chatbook/STT/coordinator.py`: preflight, execution normalization, and retry-action policy.
- Create `tldw_chatbook/STT/legacy_bridge.py`: injected adapter over the retained backend.
- Create `tldw_chatbook/STT/__init__.py`: deliberate public exports.
- Modify `tldw_chatbook/Local_Ingestion/transcription_service.py`: rename the existing implementation privately and add the explicit public facade.
- Create focused tests under `Tests/STT/`.
- Update `backlog/tasks/task-599 - Introduce-provider-neutral-STT-contracts-and-coordinator.md`.

## Task 1: Land the reviewed contract slice

**Files:**

- Create `tldw_chatbook/STT/contracts.py`
- Create `tldw_chatbook/STT/__init__.py`
- Create `Tests/STT/test_contracts.py`
- Create `Tests/STT/test_boundaries.py`

- [ ] Cherry-pick the three previously reviewed, test-first contract commits:

  ```bash
  git cherry-pick 4688f625926776365207ceb67edf5a5361e35c28
  git cherry-pick b6baa95a7976f82610cb4a0904c9d9a9b632833c
  git cherry-pick 9973f2f6218d14e17190bd50edcd72068b6b56a5
  ```
- [ ] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/STT/test_contracts.py Tests/STT/test_boundaries.py -q
  ```

  Expected: contracts and import-boundary tests pass without importing native STT runtimes.

- [ ] Write parameterized tests for the complete stable failure contract:

  | Code | Same-configuration retryable by default |
  | --- | --- |
  | `model_not_installed` | no |
  | `artifact_corrupt` | no |
  | `artifact_incompatible` | no |
  | `provider_unavailable` | yes |
  | `provider_removed` | no |
  | `unsupported_language` | no |
  | `unsupported_capability` | no |
  | `insufficient_disk_space` | no |
  | `insufficient_memory` | no |
  | `inference_failed` | no |
  | `engine_crashed` | yes |
  | `cancelled` | yes |

  For every row, assert the fixed sanitized message, required identity and
  provenance fields, retryability, representation redaction, and rejection of
  free-form exception text. Contextual action eligibility belongs to Task 4,
  after the coordinator exists.

- [ ] Run and verify RED:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/STT/test_contracts.py -k "failure or progress or retry" -q
  ```

  Expected: import/attribute failures for the missing failure and device-policy values.

- [ ] Add the minimum fixed error mapping, sanitized `TranscriptionFailure`, progress representation, and `DeviceRetryPolicy`.
- [ ] Re-run to GREEN:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/STT/test_contracts.py Tests/STT/test_boundaries.py -q
  ```

- [ ] Commit.

## Task 2: Add the sealed provider registry

**Files:**

- Create `tldw_chatbook/STT/registry.py`
- Create `Tests/STT/conftest.py`
- Create `Tests/STT/test_registry.py`
- Modify `tldw_chatbook/STT/__init__.py`

- [ ] Write failing tests for duplicate provider/model IDs, duplicate adapters, unknown exact IDs, metadata mismatch, unavailable probes, observation identity mismatch, forbidden mutation after sealing, and forbidden runtime capability escalation/loss.
- [ ] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/STT/test_registry.py -q
  ```

  Expected: RED because `registry.py` does not exist.

- [ ] Implement frozen `CapabilitySet`, `ProviderMetadata`, `ModelMetadata`, `RuntimeObservation`, adapter protocol, and `ProviderRegistry.sealed()`.
- [ ] Permit runtime observations to narrow only execution devices and precisions to non-empty subsets. Require exact equality for all semantic capabilities.
- [ ] Re-run registry, contract, and boundary tests to GREEN and commit.

## Task 3: Add built-in metadata and routing

**Files:**

- Create `tldw_chatbook/STT/routing.py`
- Create `Tests/STT/test_routing.py`
- Modify `tldw_chatbook/STT/__init__.py`
- Modify `Tests/STT/test_boundaries.py`

- [ ] Write failing parameterized tests for:

  | Provider | Task/language | Expected |
  | --- | --- | --- |
  | `default` | omitted or empty | Parakeet v2, requested/effective `en` |
  | `default` | explicit `en` | Parakeet v2 |
  | `default` | validated non-English | Parakeet v3, effective `auto`, warning |
  | `default` | `auto` | faster-whisper |
  | `default` | unsupported language | faster-whisper |
  | `default` | translation | faster-whisper |
  | exact provider/model | compatible request | exact selection preserved |
  | exact provider/model | incompatible request | typed failure; no engine switch |

- [ ] Assert the complete built-in metadata matrix:

  - Parakeet v2: English-only, transcription, file/buffer, batch but not true
    streaming, enforced language, INT8 default with F32 optional.
  - Parakeet v3: exactly the injected validated languages, transcription,
    file/buffer, batch but not true streaming, routing-only caller assertion,
    automatic internal selection, no enforced language hint, INT8 default with
    F32 optional.
  - faster-whisper: explicit and automatic language, transcription and
    translation, file/buffer, declared timestamp/VAD behavior, and its exact
    supported device/precision set.

  Assert exact input, timestamp, VAD, diarization, punctuation,
  capitalization, cancellation, execution-device, and precision fields for
  every model; no field may be inferred from a display name.

- [ ] Verify RED because routing is absent.
- [ ] Implement the three built-in declarations and `TranscriptionRouter`.
- [ ] Require Parakeet v3 to declare routing-only language assertion and no enforced hint.
- [ ] Inject the validated v3 language set; do not promote the full upstream list implicitly.
- [ ] Re-run routing, registry, contract, and boundary tests to GREEN and commit.

## Task 4: Add coordinator and explicit retry policy

**Files:**

- Create `tldw_chatbook/STT/coordinator.py`
- Create `Tests/STT/test_coordinator.py`
- Modify `tldw_chatbook/STT/__init__.py`

- [ ] Write failing tests for preflight-before-execution, cancellation, local/privacy constraints, file/buffer capability checks, composed VAD/diarization/timestamp checks, runtime mismatch, normalized language fields, sanitized failures, and progress ordering.
- [ ] Write failing policy tests proving:

  - no failure automatically invokes another provider;
  - for a constraint-compatible request whose failed provider is not
    faster-whisper, the exact action matrix is:

    | Code | Actions before target preflight |
    | --- | --- |
    | `model_not_installed` | install model, choose installed model, retry with faster-whisper |
    | `artifact_corrupt` | install model, choose installed model, retry with faster-whisper |
    | `artifact_incompatible` | choose installed model, retry with faster-whisper |
    | `provider_unavailable` | retry same configuration, retry with faster-whisper |
    | `provider_removed` | retry with faster-whisper |
    | `unsupported_language` | retry with faster-whisper; add change-language-to-auto only for an automatic-only exact model |
    | `unsupported_capability` | retry with faster-whisper only when faster-whisper supports the original constraints |
    | `insufficient_disk_space` | retry with faster-whisper |
    | `insufficient_memory` | retry with faster-whisper |
    | `inference_failed` | retry with faster-whisper |
    | `engine_crashed` | retry same configuration, retry with faster-whisper |
    | `cancelled` | retry same configuration |

    Remove `retry_with_faster_whisper` when the failed provider is already
    faster-whisper or its declaration cannot satisfy the original request.
    These values are choices returned to the caller; Task 4 never executes
    them.
  - only accelerator execution-provider initialization may return a
    same-provider CPU retry plan; the current enum's concrete accelerator
    values are CUDA and Metal, and the implementation must classify them as
    every non-CPU concrete device rather than keying policy to a provider;
  - CPU retry requires worker recycling and is returned as data only.

- [ ] Verify RED because the coordinator is absent.
- [ ] Implement `resolve()`, `preflight()`, and `transcribe()` over injected adapters.
- [ ] Keep the coordinator free of downloads, persistence, UI prompts, native imports, and retry execution.
- [ ] Re-run all `Tests/STT` tests to GREEN and commit.

## Task 5: Add the retained-provider bridge and facade

**Files:**

- Create `tldw_chatbook/STT/legacy_bridge.py`
- Create `Tests/STT/test_legacy_bridge.py`
- Create `Tests/STT/test_transcription_service_facade.py`
- Modify `tldw_chatbook/Local_Ingestion/transcription_service.py`
- Modify only legacy unit-test fixtures that must instantiate the private backend.

- [ ] Freeze the existing zero-argument constructor and public method signatures in golden-master tests before renaming the current class.
- [ ] Write failing bridge tests for exact provider mapping, file/buffer conversion, provider-specific keyword forwarding, dictionary normalization, progress conversion, error sanitization, and import isolation.
- [ ] Write failing facade tests for configured legacy-default preservation,
  exact-provider preservation, result conversion, helper forwarding, and
  cleanup. Prove all three pre-promotion cases:

  - omitted provider still uses the configured current stable provider;
  - omitted language still follows the current facade/backend configuration;
  - an explicit semantic `provider="default"` does not activate the new
    Parakeet routing in the shipped facade before TASK-605 opens the promotion
    gate.

  The coordinator's semantic routing is available and fully tested as an
  injected service boundary, but TASK-599 does not switch production defaults.
- [ ] Rename the current class to `_LegacyTranscriptionBackend` without changing its implementation.
- [ ] Implement `LegacyTranscriptionBridge` with an injected backend factory and exact provider metadata.
- [ ] Implement a small `TranscriptionService` with explicit public methods; do not use broad `__getattr__` forwarding.
- [ ] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/STT Tests/Transcription/test_stt_batch_routing.py Tests/Transcription/test_parakeet_onnx_vertical_slice.py -q
  ```

- [ ] Run focused legacy, Audio, and Diarization caller tests. Fix only facade regressions and commit.

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
    Tests/Transcription/test_faster_whisper_transcription.py \
    Tests/Transcription/test_faster_whisper_edge_cases.py \
    -k "not TestFasterWhisperIntegration" -q

  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
    Tests/Audio/test_audio_integration.py \
    Tests/Audio/test_dictation_service.py \
    Tests/Audio/test_property_based.py \
    Tests/Diarization/test_diarization_integration.py -q
  ```

## Task 6: Verify and close TASK-599

**Files:**

- Modify `backlog/tasks/task-599 - Introduce-provider-neutral-STT-contracts-and-coordinator.md`

- [ ] Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check tldw_chatbook/STT tldw_chatbook/Local_Ingestion/transcription_service.py Tests/STT
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check tldw_chatbook/STT tldw_chatbook/Local_Ingestion/transcription_service.py Tests/STT
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/mypy tldw_chatbook/STT
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/STT Tests/Transcription/test_stt_batch_routing.py Tests/Transcription/test_parakeet_onnx_vertical_slice.py -q
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Transcription/test_faster_whisper_transcription.py Tests/Transcription/test_faster_whisper_edge_cases.py -k "not TestFasterWhisperIntegration" -q
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Audio/test_audio_integration.py Tests/Audio/test_dictation_service.py Tests/Audio/test_property_based.py Tests/Diarization/test_diarization_integration.py -q
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q
  git diff --check
  ```

- [ ] Map each acceptance criterion to named tests.
- [ ] Review for API compatibility, dependency-boundary leaks, security/redaction, and unnecessary abstraction.
- [ ] Check all acceptance criteria, add concise Implementation Notes, and set TASK-599 to Done only if every Definition-of-Done item is satisfied.
- [ ] Commit, push, and open a PR against `dev`.
