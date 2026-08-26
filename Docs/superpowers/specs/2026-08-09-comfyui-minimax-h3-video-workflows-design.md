# ComfyUI MiniMax H3 Video Workflows — Design Specification

Date: 2026-08-09
Status: Approved
Related task: `TASK-3401.6`
Related ADR: `backlog/decisions/044-ephemeral-generated-video-storage-playback-and-streaming.md`

## 1. Purpose

Make the existing ComfyUI video adapter useful out of the box for a user who has
MiniMax H3 installed in ComfyUI, while preserving the adapter as a generic
workflow-driven provider seam.

The application will ship two API-format MiniMax H3 text-to-video workflows:

- `minimax_h3_t2v.json`, the default workflow;
- `minimax_h3_t2v_spectrum.json`, an opt-in workflow for installations that
  include the `SpectrumApplyMiniMaxH3` node.

Both are packaged copies derived from user-supplied ComfyUI API exports. The
source exports remain untouched.

## 2. Scope

### In scope

- Replace the currently shipped Wan2.2 and SVD workflow assets with the two
  MiniMax H3 workflows above.
- Select a workflow through the existing `comfyui_default_workflow` setting.
- Parameterize prompt, seed, width, height, duration, and the workflow's fixed
  native FPS contract through exact node-title conventions.
- Preserve safe H3 defaults of 864×480, five seconds, 24 FPS, and seed 0 when
  the request omits those values.
- Validate the graph and required node classes before queueing it.
- Submit, poll, cancel, enumerate output, and download through ComfyUI's HTTP
  API.
- Fail closed when a requested parameter cannot be applied.
- Verify the real MiniMax H3 workflow against a running ComfyUI server before
  declaring the task complete.

### Out of scope

- Wan2.2 and SVD workflows or compatibility guarantees.
- A per-generation workflow picker; workflow selection remains a setting.
- MiniMax H3 image editing. That returns a static image and belongs in a
  separate `Image_Generation` task and adapter contract.
- FAL or other new video providers.
- A generalized provider-metadata framework. The current adapter registry is
  sufficient until another provider demonstrates the shared requirements.
- General MIME-driven generated-video filename handling. That is a separate
  cross-provider storage correction required before providers returning
  non-MP4 containers are added.

## 3. Architecture and provider boundary

`VideoGenRequest` and `VideoGenResult` remain the provider-neutral boundary:

```text
VideoGenRequest
  ├── minimax  -> official MiniMax API adapter
  ├── comfyui  -> selected local API-format workflow
  ├── fal      -> future adapter
  └── other    -> future adapter
```

ComfyUI is one adapter, not one adapter per model. Model filenames, custom-node
topology, and sampling details remain inside each workflow asset. The adapter
only understands a small documented title-based control contract and the
standard ComfyUI HTTP surface.

The user workflow directory continues to take precedence over packaged assets.
A user can therefore customize a shipped workflow by placing a file with the
same name under `<user_data_dir>/video_workflows/`. Workflow resolution remains
confined to that directory and rejects traversal and symlink escape.

ADR required: no new ADR.

ADR path: `backlog/decisions/044-ephemeral-generated-video-storage-playback-and-streaming.md`

Reason: ADR-044 already owns the video adapter/provider boundary, ComfyUI
workflow model, trusted-origin treatment, and ephemeral result contract. This
spec directly implements that decision and narrows the shipped workflow assets;
it does not establish a new cross-module boundary.

Before implementation resumes, the description, acceptance criteria, and plan
in `TASK-3401.6` must be updated to remove the stale Wan/SVD scope and record the
H3-only outcomes. No production change may rely on requirements absent from the
task acceptance criteria.

## 4. Packaged workflow assets

### 4.1 Sources and destinations

| Purpose | External source | Packaged destination |
| --- | --- | --- |
| Default H3 T2V | User-supplied base API export, outside the repository | `tldw_chatbook/Video_Generation/workflows/minimax_h3_t2v.json` |
| Spectrum H3 T2V | User-supplied Spectrum API export, outside the repository | `tldw_chatbook/Video_Generation/workflows/minimax_h3_t2v_spectrum.json` |

Only renamed, sanitized packaged copies are repository artifacts. The original
ComfyUI exports are read-only evidence: they must not be modified, staged,
copied into the repository under their original names, included as test
fixtures, or committed anywhere in Git history.

Both JSON destinations are runtime package data. Wheel and source-distribution
builds must each contain exactly these two workflow JSON files and no obsolete
Wan/SVD workflow. A fresh wheel install must be able to resolve and parse both
through the adapter's confined packaged-workflow loader.

### 4.2 Source isolation and commit safety

The import procedure is intentionally one-way:

1. Record SHA-256 hashes of the two external API exports.
2. Copy them into a newly created temporary directory outside the repository.
3. Apply the approved transformations to those temporary copies: replace every
   generation prompt with the neutral filler, rename the files, expose the
   documented controls, and make only the graph changes enumerated in this
   specification.
4. Validate the sanitized temporary copies as API-format JSON before placing
   only those sanitized results at the packaged destinations.
5. Re-hash the external exports and prove their hashes are unchanged.
6. Compare each source and sanitized graph locally with an allowlist of intended
   deltas. Model filenames, class types, and edges not named by this design must
   remain unchanged.
7. Stage the two packaged destinations by exact path. Broad staging commands
   such as `git add -A` or `git add .` are prohibited for this import.
8. Inspect the staged filename list and staged content before committing. It
   must contain no original export filename, no external source path, and no
   source prompt text.

The original files never enter the repository even transiently. No raw-source
snapshot, backup, golden file, patch, or prompt-bearing fixture is created under
the workspace. The committed design also avoids recording the user's absolute
source paths.

The locally checked transformation allowlist is:

- output filename;
- generation prompt value;
- `_meta.title` values needed by the control contract;
- direct width and height values replacing the UI resolution-selector link;
- removal of the now-unreferenced resolution-selector node;
- safe defaults for seed and duration;
- the fixed native-FPS validation title;
- an explicit MP4 `SaveVideo` setting only after real `/object_info` evidence
  confirms the accepted value.

Any other source-to-copy difference stops the import for review.

### 4.3 Preserved topology

Both supplied exports contain the intended MiniMax H3 model, Qwen clip, video
and audio VAE, sampler, frame-grid expression, video construction, and save
nodes. The Spectrum version additionally routes the model through
`SpectrumApplyMiniMaxH3`.

Implementation preserves that model and sampling topology. It may remove
UI-only indirection that prevents safe API parameter injection, specifically the
resolution-selector link, but it must not invent or substitute model nodes.

The packaged default workflow must not require the Spectrum node. The Spectrum
workflow must fail initialization with a message naming
`SpectrumApplyMiniMaxH3` when that class is unavailable.

### 4.4 Neutral filler prompt

Both packaged assets use this exact non-sensitive filler prompt:

> An atmospheric cinematic shot of a red sailboat crossing a calm lake at
> sunrise. Gentle wind ripples the water and nearby reeds while the camera
> slowly tracks from left to right. Natural ambient sound with distant birds
> and soft water. No text, logos, or watermarks.

The filler makes the packaged graph safe and independently runnable, but it is
never a fallback for an application request. Strict injection must replace it
with the request prompt or reject the request before queueing.

## 5. Workflow control contract

Controls are discovered only through exact normalized `_meta.title` values.
Unknown titles do not mutate nodes. Linked values are not overwritten unless the
packaged graph was deliberately simplified to make that input direct.

The H3 packaged workflows expose:

| Node title | Request field(s) | Behavior |
| --- | --- | --- |
| `Prompt Width Height` | `prompt`, `width`, `height` | Inject all three into direct inputs on the H3 generation node. |
| `Seed` | `seed` | Inject the resolved deterministic seed into the random-noise node. |
| `Duration` | `duration_seconds` | Inject seconds into the primitive feeding the frame-count expression. |
| `Native FPS` | `fps` | Validation control fixed at 24; a request at another FPS is rejected rather than mutating the workflow. |

The existing generic conventions for custom workflows remain supported where
they are already valid, but strict injection applies to them as well.

The packaged H3 graph carries direct defaults of width 864, height 480,
duration five seconds, and deterministic seed 0. An omitted value retains that
graph default; a supplied value must be injected successfully. Seed `-1`
preserves the public request contract by resolving once to a valid non-negative
ComfyUI seed, which is returned as `resolved_seed`. The configured default
workflow changes from `wan22_t2v.json` to `minimax_h3_t2v.json`.

Every effective value is validated even when the request omitted it: width and
height must be positive integers aligned to 32 pixels, duration must be finite
and greater than zero, native FPS must remain 24, and seed must be a
non-negative integer. Invalid graph defaults fail before queueing just like
invalid supplied values.

The provider-neutral request can also carry `ratio`. For the H3 presets, a
numeric ratio is treated as a constraint on the effective width and height, not
as a second competing size control. `16:9` is compatible with the aligned
864×480 default. A ratio whose aspect differs from the effective dimensions by
more than three percent, or `adaptive`, fails before submission. The tolerance
exists only for model-alignment rounding and is covered at its boundary by
tests. Explicit width and height remain authoritative when they are present.

The packaged H3 workflows support MP4 output. The request's `format` is part of
the strict contract: non-MP4 requests are rejected for these presets, and live
verification must prove the selected `SaveVideo` configuration actually emits
MP4 rather than trusting `format: auto`.

### 5.1 Fixed native FPS

The supplied H3 frame expression is based on 24 FPS and rounds the generated
frame count to the H3-required `17k + 5` grid. The workflow also constructs
audio/video at 24 FPS. Therefore:

- the packaged H3 workflows run at 24 FPS;
- the duration control changes seconds, and the existing expression derives the
  corresponding valid frame count;
- an omitted FPS resolves to the workflow's 24 FPS contract;
- an explicit FPS other than 24 fails before submission with a message naming
  the requested and supported values;
- the adapter does not silently change only the encoder FPS.

This avoids duration drift and audio/video desynchronization.

### 5.2 Strict injection

For every request value the adapter promises to apply, parameterization records
whether a matching, eligible input was updated or validated. Submission is
blocked if any supplied value lacks a target, targets only a linked input, or
violates a fixed-value control.

For the packaged H3 presets this promise covers prompt, seed, width, height,
duration, FPS, numeric ratio, and output format. H3 has no separate
negative-prompt input;
model, sampler, steps, and guidance remain workflow-owned. These unsupported
fields must not be presented as H3 controls. If a direct programmatic caller
supplies one, the adapter rejects it rather than implying it affected the
result. Console style composition must therefore be regression-tested so its
positive prompt suffix works with H3 without forwarding a style-only negative
prompt that the selected workflow cannot consume.

Style compatibility is classified from the selected graph's capability node
(`class_type == MiniMaxH3ImageToVideo`), not its filename. Classification uses
the existing confined local/package loader in the already-off-thread generation
path and performs no ComfyUI network request. Only a negative prompt known to
come from style composition may be suppressed; an explicit programmatic
negative prompt reaches the adapter and is rejected by the H3 contract.

The failure names the field and the expected title convention. This turns a
custom workflow mismatch into an actionable configuration error and prevents a
packaged filler prompt or stale dimensions from reaching generation unnoticed.

## 6. Runtime flow

1. Resolve the configured bare JSON filename, preferring the confined user
   workflow directory and falling back to packaged assets.
2. Parse an API-format JSON object keyed by node id.
3. Validate reference assets. If the graph is H3 and any otherwise-valid image
   reference is present, reject it before object-info lookup or upload; generic
   input-image workflows retain their upload path.
4. Collect every non-empty `class_type` and compare it with real
   `GET /object_info` data.
5. Upload a supported local input image through
   `POST /upload/image` when the selected custom workflow requires one.
6. Deep-copy and strictly parameterize the graph.
7. Submit `POST /prompt` with the graph and client id.
8. Poll `GET /history/{prompt_id}` until success, terminal failure,
   cancellation, or timeout.
9. Pending unrelated or preview outputs do not terminate polling. Terminal
   failure is reported first; a no-supported-output error is raised only after
   explicit terminal success. Each request and wait receives only the remaining
   shared deadline budget.
10. On cancellation, call `POST /interrupt` best-effort and stop locally.
11. Enumerate the actual output descriptor from history and fetch bytes through
   `GET /view` using its filename, subfolder, and type.
12. Return a `VideoGenResult` containing bytes, observed content type, byte
    length, effective duration/FPS/dimensions/seed, and model metadata only when
    it is actually known from the parameterized graph.

Polling remains the v1 completion mechanism; WebSocket progress is not required
for this task.

## 7. Error handling

The adapter fails closed for:

- missing, traversing, symlinked, malformed, or non-object workflow files;
- a graph with no valid `class_type` entries;
- unreachable or malformed `/object_info` responses;
- missing node classes, with every missing class named;
- unsupported reference-asset kinds or upload failures;
- any promised request field that cannot be injected or validated;
- queue rejection, malformed prompt ids, terminal history failures, or timeout;
- cancellation, after a best-effort interrupt;
- a history response with no supported media descriptor;
- unsupported output suffix or content type;
- egress denial or output download failure.

Errors must not expose API credentials or raw workflow contents. The configured
ComfyUI origin remains a user-selected trusted origin under the existing egress
policy; that trust does not extend to unrelated response URLs.

## 8. Real-server verification gate

Mocked tests cannot establish the custom-node or output-descriptor contract. A
running ComfyUI server was not reachable during design review, so the following
are explicit completion gates rather than assumed facts:

1. Capture real `/object_info` entries for `MiniMaxH3ImageToVideo`,
   `SpectrumApplyMiniMaxH3`, the expression node, and `SaveVideo`.
   Record any declared width/height alignment constraints and enforce them at
   the adapter boundary rather than relying on a queue-time node error.
2. Queue the packaged default H3 workflow with a harmless short prompt and
   conservative dimensions/duration.
3. Confirm the history response's exact output collection key and descriptor
   shape.
4. Confirm the downloaded container, MIME type, duration, dimensions, FPS, and
   presence of expected audio.
5. Verify the stored filename extension agrees with the actual container. If
   `SaveVideo` with `format: auto` does not produce MP4, configure an explicitly
   supported MP4 setting proven by `/object_info`. If this ComfyUI installation
   cannot produce MP4, stop and revise the design rather than returning a
   container that contradicts the request.
6. Repeat initialization against the Spectrum workflow. If the Spectrum node is
   installed, run it; otherwise verify the real missing-class error names it.

Captured payloads and history descriptors used in tests must be copied from the
wire rather than invented to match the adapter.

## 9. Test strategy

### Asset contract tests

- Both shipped files parse as API-format graph objects.
- Built wheel and sdist inventories contain exactly both H3 JSON assets and no
  obsolete Wan/SVD workflow; a fresh installed-wheel probe loads both graphs.
- The default graph does not contain `SpectrumApplyMiniMaxH3`.
- The Spectrum graph contains and routes through it.
- Wan/SVD assets are no longer shipped.
- Both graphs contain `SaveVideo` and the exact neutral filler prompt.
- Every generation-prompt field in each packaged graph equals the exact neutral
  filler before runtime injection.
- No test fixture or assertion embeds either source prompt merely to prove its
  absence.
- Width and height are directly injectable.
- Duration feeds the preserved 24-FPS, `17k + 5` frame-grid expression.
- Native FPS is fixed at 24.

### Adapter tests

- User-workflow precedence and path confinement.
- Exact title matching and deep-copy behavior.
- Successful injection of prompt, dimensions, seed, and duration.
- Omitted dimensions/duration/seed retain 864×480, five seconds, and seed 0;
  seed `-1` resolves once and is reported.
- Numeric ratio is validated against effective dimensions; incompatible and
  adaptive ratios fail.
- Omitted FPS and explicit 24 FPS succeed; any other FPS fails.
- MP4 succeeds only with an observed MP4 result; other requested formats fail
  for the H3 presets.
- Missing, linked, or misspelled controls fail instead of silently no-oping.
- `/object_info` validation names all missing classes.
- Real request signatures and sanitized live-observed payload shapes for upload,
  submit, history, interrupt, and view.
- Pending polling, terminal failure, timeout, and cancellation.
- Output enumeration across the exact real `SaveVideo` history shape.
- Unsupported or absent output fails clearly.
- Trusted-origin propagation remains confined to the configured backend.
- H3 rejects explicitly supplied unsupported controls, while the normal
  Console `@style` path does not accidentally forward its style-only negative
  prompt to H3.

Every new guard must be mutation-checked by breaking its protected behavior and
confirming the corresponding test fails.

### Provenance and staging checks

- External source hashes are identical before and after the import.
- A local canonical graph comparison reports only the transformation-allowlist
  differences from section 4.2.
- `git diff --cached --name-only` contains only intended task files and the two
  renamed sanitized workflow destinations; it contains none of the original
  export filenames.
- A staged-content scan finds the neutral prompt and no unapproved prompt value
  in any generation-prompt field.
- Repository history and tracked paths contain no original H3 export or image-
  edit export. The image-edit source remains external until its separate task.

### Regression reach

Run the focused ComfyUI and Video Generation suites, configuration tests,
Console video-generation tests, architecture/runtime-policy tests that inventory
workflow assets or adapters, full-tree collection, and the repository's required
lint/static checks. Compare any unstable failure set against an identical command
on the baseline rather than comparing raw counts.

For the final-review remediation round, the user explicitly narrows this reach
to touched-file-related packaging, adapter, asset, config, template, and Console
tests plus targeted Ruff, `py_compile`, diff, and provenance checks. Do not
rerun full collection, RuntimePolicy, broad repository suites, or costly live
generation; the existing recorded UAT remains the live evidence.

## 10. Future-provider compatibility

The current lazy adapter registry already permits future `fal` and other
provider implementations to consume the same request/result contract. This task
must not add FAL-specific branches to the ComfyUI adapter.

Adding another provider currently requires changes in several explicit places:
adapter registration, configuration, secret/non-secret ownership, validation,
and settings presentation. When the first new provider is implemented, evaluate
a small `VideoBackendSpec` registry that can own those declarations. Do not add
that abstraction in advance of a second concrete consumer.

Before any provider capable of returning WebM, MOV, or another container ships,
generated-video storage must derive the filename extension from the validated
result MIME/container rather than always writing `.mp4`.

## 11. Separate H3 image-edit follow-up

The supplied API-format image-edit export is not part of this task. It returns a
static image and must enter the existing image-generation validation, attachment
storage, and metadata contract.

The export also contains a UI comparison branch that is not a valid API graph:
node `154` lacks `class_type`, and comparison output node `166` depends on it. The
follow-up must package a cleaned copy that removes nodes `154` and `166` and uses
the independent edited-image `SaveImage` output at node `165` as the canonical
result. The source export remains untouched.

## 12. Acceptance summary

The design is satisfied when a default installation can select the shipped base
H3 workflow, a Spectrum-capable installation can opt into the Spectrum workflow,
all requested values are either applied or rejected explicitly, real ComfyUI
evidence confirms the graph and output contract, and the resulting bytes cross
the existing `VideoGenResult` boundary without introducing model-specific logic
into future provider adapters. Only renamed, prompt-sanitized workflow copies
are committed; the external originals remain unchanged and absent from the
repository and its history.
