# ADR-054 - Deterministic visual transcript compaction

- **Status:** Accepted
- **Date:** 2026-08-11
- **Task:** TASK-14914
- **Amends:** ADR-052

## Context

ADR-052 makes the stored conversation transcript authoritative and treats generated
memory as a branch-provenanced, derived request input. Text summaries are useful,
but a vision-capable model can sometimes receive an older transcript prefix more
compactly as one or more locally rendered images. That optimization must not blur
the distinction between the conversation budget and the next-response limit, make
vision support a persisted assumption, or create a second durable copy of private
conversation content.

The proposed representation also has material failure modes: provider image token
pricing varies by model, OCR can lose punctuation or code structure, instructions
inside a transcript image remain untrusted conversation content, and a provider or
model may not accept images even when another model in the same session does.

## Decision

### Policy and capability ownership

Add a sparse `compaction_representation` policy field with three user-facing
values: `text_summary`, `visual_transcript`, and `hybrid`. The application default
remains `text_summary`. Global Console Behavior settings own the inherited default;
the active conversation may override it through the Model settings modal.

Vision support, image limits, and image-token accounting remain request-time model
capabilities owned by `model_capabilities` and the prepared-request gateway. They
are never copied into durable conversation policy. A text-only current model keeps
Visual transcript and Hybrid visible but unavailable in the conversation modal,
with the reason exposed to the user. A global visual preference is allowed because
it is intent, not a claim that every future model supports images.

### Canonical content and artifact lifetime

The SQLite transcript remains the only canonical conversation content. Visual
pages are request-scoped derived artifacts generated from a selected, branch-valid
transcript prefix. PNG bytes are held in memory only for preparation and dispatch;
they are not written to SQLite, config, logs, caches, temporary files, or sync
payloads. Text generated memory continues to use ADR-052's immutable memory rows.

Each rendered page carries content-free provenance in the prepared request:
renderer schema/version, page index/count, dimensions, ordered source unit IDs,
the summarized-prefix digest, and a SHA-256 digest of the exact PNG bytes. This
provenance is not sent as user-authored text and is discarded with the prepared
request. The renderer accepts immutable transcript units rather than display
widgets or mutable live buffers.

### Deterministic rendering contract

Rendering is entirely on-device and performs no model, network, OCR, font lookup,
locale lookup, or wall-clock call. The versioned renderer fixes its PNG dimensions, palette,
pixel font, cell metrics, margins, line wrapping, pagination, role headers, code
fence treatment, tool-call/result boundaries, and source ordering. Unsupported
Unicode scalar values are rendered as explicit escaped text rather than relying on
host font fallback. Its identity includes the Pillow version that owns the bundled
bitmap font and PNG encoder. Given the same renderer version and ordered transcript units,
the page count, page bytes, and hashes must be identical across supported hosts.

Transcript text is framed as quoted historical data. Renderer-owned role and unit
boundaries cannot be supplied by conversation content. Recent turns and the active
request remain ordinary text messages.

### Representation semantics

- **Text summary** uses ADR-052's existing auxiliary summary and durable memory.
- **Visual transcript** replaces only the selected older compactable prefix with
  deterministic image pages. It does not create durable generated memory.
- **Hybrid** retains the branch-valid text summary and adds visual pages for its
  selected source prefix when those pages fit the request. Recent turns remain
  text in every mode.

The controller resolves a requested representation to an effective representation
before dispatch. Visual and Hybrid require a vision-capable current model, a wire
adapter that preserves image parts, a page count within the model's image limit,
successful local rendering, and a prepared request that fits exact provider
accounting. If any condition fails, the request falls back to Text summary without
dropping mandatory context. If text compaction also cannot produce a safe request,
ADR-052's configured failure behavior applies.

### Accounting and evaluation gate

The prepared provider request is the accounting authority. It counts the exact
message structure and every emitted image part using model-specific image-token
facts when known and a conservative estimate otherwise. The UI and benchmark
report label estimates as estimates; claimed savings are never inferred from PNG
byte size.

Visual and Hybrid remain opt-in until an offline benchmark report is available for
the intended model. Reports include text-versus-image input token cost, render and
end-to-end latency, OCR fidelity, code/math recovery, instruction recall, and
adversarial-text behavior. Unknown metrics are reported as unknown and cannot be
treated as a pass. Default enablement requires a separate reviewed decision.

## Consequences

- The conversation policy schema gains one nullable sparse override column, but
  the memory schema and sync boundary do not change.
- Visual-only compaction is reproducible but intentionally not a durable memory
  operation; switching to a text-only model is recoverable through text fallback.
- PNG byte size is not a trustworthy proxy for provider tokens, so savings vary by
  model and may be negative.
- A fixed pixel font improves reproducibility but may be less readable than host
  fonts and represents unsupported Unicode as escapes.
- Hybrid can improve recovery of exact details while costing more than either
  representation alone; it remains an explicit user choice.

## Rejected alternatives

- **Persist PNG pages in SQLite or the filesystem.** Rejected because it duplicates
  private transcript content, complicates deletion/sync, and is unnecessary for a
  deterministic renderer.
- **Treat PNG compression ratio as token reduction.** Rejected because vision token
  accounting is provider/model-specific and unrelated to compressed byte size.
- **Silently coerce the saved preference when a model is text-only.** Rejected
  because capability is request-time state and overwriting intent makes model
  switching lossy.
- **Use host fonts or browser screenshots.** Rejected because rendering would vary
  by operating system, installed fonts, DPI, theme, and browser engine.
