# Lazy MLX Import Boundary Design

## Goal

Keep unrelated application imports and test collection from initializing the
optional Parakeet MLX or Lightning Whisper MLX runtimes. Installed providers
remain discoverable, but their native Python modules load only when a user
actually invokes that provider.

## Current Problem

`tldw_chatbook.Local_Ingestion.transcription_service` imports both MLX
backends at module scope on macOS. Importing the production application reaches
that module through the local-ingestion chain, so an installed MLX package can
abort Python before non-STT tests collect or the application mounts. Three
ProductionApp tests currently work around the problem by temporarily assigning
`None` to `sys.modules["parakeet_mlx"]`.

Configuration discovery already uses `importlib.util.find_spec` without
importing either runtime. The transcription service should follow the same
boundary until actual provider use.

## Approved Design

At module import:

- Probe `parakeet_mlx` and `lightning_whisper_mlx` with `find_spec`.
- Preserve the existing availability flags and provider-selection behavior.
- Keep the backend class/function references unset.
- Do not import either optional native runtime.

At first provider use:

- Load the selected backend through one explicit loader per backend.
- Cache the imported class/function in the existing module-level reference.
- Reuse the cached reference on later calls.
- If Python raises while importing the optional backend, mark that backend
  unavailable and raise the existing bounded `TranscriptionError`.

Only paths that actually execute Parakeet MLX or Lightning Whisper MLX call
these loaders. Provider discovery, configuration loading, app startup, and
unrelated tests never do.

## Error Behavior

Missing packages continue to appear unavailable through the existing flags.
An installed package that raises a normal Python exception during its lazy
import becomes unavailable for the process and produces a transcription error
instead of leaking the backend exception through the application.

This change does not add a subprocess runtime, retry policy, new dependency
registry, or new provider abstraction. A native library that terminates the
process during an explicitly requested MLX transcription remains the native
backend's responsibility; the approved scope is preventing that initialization
from unrelated imports and test collection.

## Verification

Test-driven verification will:

1. Prove in a subprocess that importing the production app does not import
   either MLX module even when both are discoverable.
2. Prove each backend imports on first actual use and reuses its cached symbol.
3. Prove ordinary lazy-import failures become bounded transcription errors.
4. Remove the three ProductionApp `sys.modules` stubs and run those affected
   modules without replacement stubs.
5. Run the existing config provider-probe test plus touched-file Ruff and
   `git diff --check`.

No full repository test suite is part of this task.

## Scope

In scope:

- The MLX import boundary in `transcription_service.py`.
- Focused import/lazy-loader regression tests.
- Removal of the three obsolete ProductionApp collection stubs.
- TASK-839 documentation and closeout.

Out of scope:

- Citation behavior.
- STT provider routing or defaults.
- Installing, removing, or replacing MLX backends.
- Subprocess-isolated transcription.
- Schema, configuration format, or dependency changes.
- Unrelated test baseline repair.

## ADR Check

ADR required: no

ADR path: N/A

Reason: This defers existing optional imports to their point of use without
changing provider ownership, runtime contracts, dependencies, storage, or
security policy.
