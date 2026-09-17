# Optional install command and STT warning fixes

**Goal:** Make missing-feature install commands recoverable and stop YouTube import warnings from demanding unrelated speech backends.

**Architecture:** Keep Textual's terminal clipboard route, add checked native delivery with the existing pyperclip package, and share that behavior between the two install-command buttons. Keep the current STT routing policy; project captured dependency warnings through the selected import provider before building both warnings and start consent.

**ADR required:** no
**ADR path:** `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md` (STT); N/A (clipboard)
**Reason:** Routine repairs to existing clipboard delivery and supported-provider metadata; no new provider, automatic fallback, dependency, schema, or ownership policy.

## TASK-32754: Clipboard

- [x] Add failing mounted regressions for native delivery, unavailable native clipboard, silent native failure, and literal install text.
- [x] Add `Utils/install_clipboard.py`: check native copy/readback off the event loop; use Textual for remote/browser/failed native delivery and say it is unconfirmed. Keep the command visible.
- [x] Route `Utils/widget_helpers.py` and `Widgets/Library/library_ingest_canvas.py` copy actions through the helper. Quote extras and render commands without Rich markup.
- [x] Run focused tests and native clipboard readback, restoring the original clipboard afterward.

## TASK-32755: Import warnings

- [x] Add failing regressions for a Linux YouTube import with only faster-whisper and for switching among supported backends.
- [x] Remove retired MLX entries from audio/video capability metadata and add transcribe.cpp. Filter the immutable preflight warning snapshot against current options before all state/forecast computations.
- [x] Exercise URL import at the transcription runner boundary, preserving explicit provider choices and the existing Auto=faster-whisper policy.
- [x] Run affected capability, preflight, state, UI and transcription checks. Record baseline failures separately.

## Delivery

- [x] Update user documentation and task notes, self-review the diff, and run formatting and repository preflight checks.
Publish only these changes on `codex/optional-deps-stt-fixes` in a PR targeting `dev`; retain the worktree for review.

Verification: [results and baseline exceptions](../reviews/2026-09-17-optional-install-stt-verification.md).

## Follow-up: Parakeet package versus model setup

The reporter installed only the Python package. Clarify beside the existing
folder field that model files are separate, and point to the current managed
installer. Document selecting Parakeet, confirming the English v2 INT8 install,
and leaving the external-folder override blank afterward. Verify existing
installer/source-selection tests and inspect the rendered hint.

ADR required: no. ADR-025 and ADR-050 already govern explicit model acquisition
and managed/external source selection; this follow-up only clarifies that flow.
