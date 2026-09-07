# PR 1642 Qodo Review Remediation Design

## Goal

Correct the valid documentation finding from Qodo's PR 1642 review while
preserving the already-approved artifact lease and maintainer-script contracts.

## Review dispositions

1. Add Google-style `Args:` and `Returns:` sections to the five public Model
   Library helpers (`model_library_focus_locator`,
   `restore_model_library_focus`, `project_audio_cpp_observation`,
   `clear_audio_cpp_observation`, and `audio_cpp_package_projection`) and the
   three public manifest-refresh entry points (`validate_commit`,
   `refresh_manifest_bytes`, and `main`). Add `Raises:` wherever the callable
   exposes a failure contract.
2. Keep `ArtifactOperationLease.release()` retryable after an unlock failure.
   ADR-050's removal and runtime ownership boundaries require the exact lease
   object to retain its handle until unlock succeeds. Existing real-lock tests
   prove that an exclusive contender stays blocked after the first failure and
   that a later release succeeds.
3. Keep the manifest refresh script dependency-free and runnable with
   `python -S`. Its `--manifest` and `--output` arguments are explicit paths
   selected by a trusted maintainer, and `--output` intentionally supports an
   arbitrary destination. Importing the application path validator would add
   Loguru/metrics dependencies and break that contract without adding a useful
   confinement root. Document this trust boundary and retain the direct-run
   regression.

## Testing

- Add a small documentation-contract test that fails on the current short
  docstrings and passes only when all eight named public functions expose the
  required sections.
- Re-run the real unlock-failure retry test and the dependency-free direct
  manifest command test unchanged.
- Add a direct `python -S` command regression with an explicit `--output`
  destination and verify the expected manifest bytes are written there.
- Run Ruff, formatting, and `git diff --check` on the focused change.

## Architecture decision

ADR required: no

ADR path:
[`backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md`](../../../backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md)

Reason: this is review remediation and documentation clarification. It does not
change storage, ownership, runtime, security, dependency, or UI boundaries.
