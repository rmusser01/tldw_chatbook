# ADR-052: ComfyUI H3 image edits stay inside the Image Generation provider boundary

Status: Accepted
Date: 2026-08-10
Related Task: [backlog/tasks/task-3402 - H3-static-image-edit-through-Image_Generation.md](../tasks/task-3402%20-%20H3-static-image-edit-through-Image_Generation.md)
Supersedes: N/A

## Decision

MiniMax H3 static image editing through ComfyUI is implemented as a dedicated,
strict packaged-workflow adapter inside `Image_Generation`, with independent image
settings and the existing Image Generation validation, attachment persistence, and
generation-metadata contracts as its only application boundary.

## Context

TASK-3402 adds a user-run ComfyUI server as a new image-generation provider. The
existing video ComfyUI adapter speaks a similar API, but its request controls,
output type, storage, metadata, cancellation, and retention contracts are video
specific. Reusing it would make a static edited image travel through the wrong
validation and storage boundary.

The supplied graph also contains comparison/output branches and embedded values
that must not be shipped unchanged. This task needs one repository-owned sanitized
workflow whose node 165 output is treated as the only edited-image result. It does
not need a generic arbitrary-workflow execution engine.

The source image and instruction leave the application for the configured ComfyUI
origin. Both the uploaded input and node 165 `SaveImage` output may remain on that
server because ComfyUI has no standard portable delete API. The trust and retention
boundary must therefore be explicit.

Future image providers such as FAL or the official MiniMax API should register
through the existing Image Generation adapter registry rather than inherit ComfyUI
or video-specific behavior.

## Decision Details

1. `ComfyUIImageAdapter` belongs to `Image_Generation` and returns the existing
   `ImageGenResult` contract.
2. The adapter is bound to one packaged, sanitized H3 workflow. Arbitrary workflow
   paths are not exposed.
3. Image Generation owns separate ComfyUI base URL, polling, and timeout settings.
   Successfully saving the normalized base URL establishes user intent to trust
   that exact host for self-built endpoints; Video Generation settings are never
   read and response data cannot extend trust.
4. `worker.run_generation()` remains the single validation choke point. The
   existing reference-image capability grows a backward-compatible `required`
   flag: current reference providers stay optional, while ComfyUI requires one.
5. Successful output uses the existing image attachment and variant-metadata path.
   No Video Generation adapter, store, or metadata is involved.
6. Local graph and remote object-schema/model validation complete before source
   upload. All exchange and download traffic stays on the configured trusted origin
   with redirects disabled.
7. Only node 165 may produce the result, and the result must be one bounded,
   validated PNG.
8. The Console consumes the exact staged source only after durable result
   persistence. Regenerate requires the user to restage the original source.
9. ComfyUI-side input/output retention is operator-managed and disclosed. The app
   does not use a global server interrupt or claim portable server cleanup.
10. Image/video transport extraction is deferred until a concrete additional
    consumer proves a stable shared contract.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Route the edit through the Video Generation ComfyUI adapter and reinterpret its output | It violates the Image Generation validation, persistence, metadata, and result contracts and couples static images to video behavior. |
| Extract a shared ComfyUI transport before adding the image adapter | The image and video lifecycle/output rules already differ; extracting first would be a speculative cross-package refactor with no proven third consumer. |
| Build a generic custom-workflow ComfyUI image engine | Arbitrary graphs require a substantially wider path, topology, capability, control, and output security contract than TASK-3402 needs. |
| Store the source as a hidden attachment to support Regenerate | Hidden durable media changes ownership and privacy semantics and risks presenting the wrong source; explicit restaging is honest and simpler. |
| Convert node 165 PNG to user-selected JPEG/WebP | It introduces a second lossy transformation and output contract; the canonical workflow already emits PNG. |
| Use one shared Image/Video ComfyUI settings block | It allows one modality's settings reset or workflow selection to alter the other and obstructs independent future provider configuration. |

## Consequences

- Image and video ComfyUI integrations duplicate a small amount of protocol code.
  This is accepted until real reuse evidence justifies extraction.
- The first ComfyUI image backend is edit-only, single-input, single-output, and
  PNG-only. Unsupported request values fail explicitly.
- Users must explicitly configure and trust non-loopback ComfyUI origins.
- Image Generation result metadata grows by one backward-compatible allowlisted
  effective-parameter mapping, and the request contract grows by one optional
  `threading.Event` cancellation seam. Existing adapters keep their current
  behavior when both are absent.
- Staged attachment identity is an in-memory UUID used by an exact-consume store
  operation; it is not persisted as generation provenance.
- H3 prompt preparation resolves the backend before generic style/context logic,
  and success-versus-cancellation is linearized before a shielded durable append.
- H3 operation lifetime is app-owned across Console unmount/remount. Cancellation
  is a typed result that the batch helper must re-raise, not collect as a failed
  variant.
- A byte-free app-owned completion record carries the persisted message ID and
  prevents either the durable result from disappearing or the exact source
  attachment/unchanged command draft from resurrecting through Console's unmount
  stash and fresh-store restoration path.
- The ComfyUI image backend is explicit opt-in, and every server JSON response is
  byte-bounded before parsing.
- Source images and edited outputs may persist on the configured ComfyUI server;
  operator cleanup policy governs them.
- Settings changes reset the Image Generation registry only after successful
  persistence, keeping classification and dispatch on one adapter snapshot.
- Future image providers remain ordinary registry adapters and can define their
  own optional/required reference-image capabilities without importing ComfyUI or
  Video Generation.

## Links

- [TASK-3402](../tasks/task-3402%20-%20H3-static-image-edit-through-Image_Generation.md)
- [ComfyUI H3 image-edit design](../../Docs/superpowers/specs/2026-08-10-comfyui-h3-image-edit-design.md)
- [Image Generation multi-provider foundation](../../Docs/superpowers/specs/2026-07-22-image-generation-multiprovider-foundation-design.md)
- [ADR-044 — Generated video storage, playback, and Video Generation boundary](044-ephemeral-generated-video-storage-playback-and-streaming.md)
