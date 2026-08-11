# ComfyUI MiniMax H3 Static Image Edit Design

**Status:** Approved for implementation planning

**Date:** 2026-08-10

**Task:** TASK-3402 — H3 static image edit through Image_Generation

**ADR:** [ADR-052 — ComfyUI H3 image edits stay inside the Image Generation provider boundary](../../../backlog/decisions/052-comfyui-h3-image-edit-provider-boundary.md)

## 1. Purpose

Ship one sanitized MiniMax H3 static image-edit workflow as a first-class
`Image_Generation` backend. A user stages one image, types one edit instruction,
and receives one persistent PNG image variant through the existing Console image
attachment and generation-metadata contract.

This is an API client integration. ComfyUI and its workflow execute on the
configured server; the workflow does not need to exist locally outside the
packaged sanitized resource.

The implementation must not reuse the video adapter or video storage. It must
also leave a clean provider seam for later image providers such as FAL and the
official MiniMax API.

## 2. Goals

- Register a dedicated `comfyui` adapter under `Image_Generation`.
- Package one renamed and sanitized H3 API workflow.
- Remove workflow nodes 154 and 166 and make node 165 the sole canonical output.
- Require exactly one staged source image and one user-authored instruction.
- Produce exactly one PNG while preserving the source image dimensions.
- Apply seed, steps, and sampler controls; reject every unsupported value before
  upload or queue submission.
- Persist the result through the existing image attachment and
  `GenerationVariantMeta` boundary.
- Use independent Image Generation ComfyUI settings and trusted-origin policy.
- Prove the real path with focused automated tests and live UAT against a
  user-configured ComfyUI server.
- Keep source workflow identity, source prompt content, raw export artifacts,
  media bytes, server descriptors, and sensitive server payloads out of repository
  history and verification reports.

## 3. Non-goals

- Generic arbitrary ComfyUI workflow execution.
- Reusing or importing the `Video_Generation` ComfyUI adapter.
- Extracting a shared image/video ComfyUI transport abstraction.
- Text-to-image operation through this H3 workflow.
- Multiple source images, batch editing, comparison images, or multiple outputs.
- JPEG/WebP conversion.
- Hidden retention of the source image for later Regenerate.
- A portable server-side delete mechanism for ComfyUI inputs or outputs.
- Global ComfyUI interruption on a shared server.
- Adding other image providers in this task.

## 4. Architecture

### 4.1 Provider ownership

`ComfyUIImageAdapter` lives under `tldw_chatbook/Image_Generation/adapters/` and
implements the existing synchronous `ImageGenerationAdapter` contract. It accepts
an `ImageGenRequest`, performs strict preparation and ComfyUI API exchange, and
returns an `ImageGenResult` containing one PNG.

The adapter uses Image Generation's existing:

- request/result dataclasses;
- single validation choke point in `worker.run_generation()`;
- adapter registry and lazy construction;
- reference-image resolution contract;
- outbound HTTP and egress controls;
- Console attachment persistence; and
- generation variant metadata.

There is no import from `Video_Generation`, no call to `VideoStore`, and no video
metadata or playback path.

Transport remains adapter-local. Shared ComfyUI transport extraction is deferred
until another concrete consumer proves that the shared behavior and lifecycle are
actually identical.

### 4.2 Packaged-only workflow

The adapter is bound to one stable packaged workflow key. TASK-3402 exposes no
arbitrary path or custom-workflow setting. Loading uses a package-resource-confined
loader and returns a fresh deep copy for every request.

The distribution contains only the renamed sanitized workflow. The raw source
export is never copied into the repository, staged, committed, hashed into a
report, or named in tracked documentation.

### 4.3 ADR

ADR-052 records the new provider/runtime and cross-module boundary. It also records
the rejection of video-adapter reuse, arbitrary workflow execution, and premature
shared transport extraction.

## 5. Sanitized Workflow Contract

### 5.1 Asset hygiene

Sanitization occurs from the externally supplied API graph without placing the raw
export inside the repository. The committed graph:

- has a new repository-owned filename and stable package key;
- removes nodes `154` and `166` completely;
- retains node `165` as the only output node;
- replaces node 114's source-image placeholder with the repository-owned neutral
  value `h3_edit_input.png`;
- uses a newly authored neutral filler instruction, `Apply the requested image edit.`;
- replaces node 165's filename prefix with the repository-owned neutral value
  `h3_edit`;
- contains none of the source workflow's embedded prompt text; and
- contains no original source path, source filename, export hash, or provenance
  label.

Loader and model widget values needed to execute the graph are operational graph
configuration, not provenance. They must not be copied into logs or reports.

The source-to-sanitized transformation allowlist is exactly:

1. delete top-level nodes `154` and `166`;
2. replace `/114/inputs/image` with `h3_edit_input.png`;
3. replace `/133/inputs/prompt` with `Apply the requested image edit.`; and
4. replace `/165/inputs/filename_prefix` with `h3_edit`.

Every other node, class, input name, direct link, and operational literal remains
unchanged. Sanitization verification compares the external source and sanitized
copy only in a private temporary workspace; it emits only a pass/fail allowlist
result and never records either raw graph, source identity, hash, or changed source
literal.

### 5.2 Exact topology

The packaged graph is a versioned contract identified primarily by node ID,
`class_type`, input name, and direct links. Titles may be asserted for diagnostic
quality but never select an injection target.

The normative sanitized node inventory is:

| Node | Required class | Role |
| --- | --- | --- |
| `114` | `LoadImage` | uploaded source image |
| `121` | `VAELoader` | VAE loader |
| `124` | `VAEDecode` | sampled latent decode |
| `125` | `KSamplerSelect` | sampler control |
| `126` | `BasicScheduler` | scheduler and steps control |
| `127` | `SamplerCustomAdvanced` | sampler execution |
| `128` | `BasicGuider` | conditioning guider |
| `129` | `UNETLoader` | H3 model loader |
| `130` | `CLIPLoader` | text encoder loader |
| `131` | `RandomNoise` | seed/noise control |
| `133` | `MiniMaxH3ImageToVideo` | H3 instruction and source conditioning |
| `139` | `PrimitiveInt` | graph-owned frame-length control |
| `140` | `GetImageSize` | scaled processing dimensions |
| `141` | `ImageScaleToTotalPixels` | bounded processing resize |
| `144` | `ImageFromBatch` | one edited frame selection |
| `149` | `ResizeImageMaskNode` | restore source dimensions |
| `150` | `GetImageSize` | original source dimensions |
| `165` | `SaveImage` | sole canonical edited PNG output |

No other node ID is permitted. In particular, `154` and `166` are absent.

The normative direct-link inventory is:

| Destination input | Required source |
| --- | --- |
| `124.samples` | `127[0]` |
| `124.vae` | `121[0]` |
| `126.model` | `129[0]` |
| `127.guider` | `128[0]` |
| `127.latent_image` | `133[1]` |
| `127.noise` | `131[0]` |
| `127.sampler` | `125[0]` |
| `127.sigmas` | `126[0]` |
| `128.conditioning` | `133[0]` |
| `128.model` | `129[0]` |
| `133.clip` | `130[0]` |
| `133.first_frame` | `114[0]` |
| `133.height` | `140[1]` |
| `133.length` | `139[0]` |
| `133.vae` | `121[0]` |
| `133.width` | `140[0]` |
| `140.image` | `141[0]` |
| `141.image` | `114[0]` |
| `144.image` | `124[0]` |
| `149.input` | `144[0]` |
| `149.resize_type.height` | `150[1]` |
| `149.resize_type.width` | `150[0]` |
| `150.image` | `114[0]` |
| `165.images` | `149[0]` |

The controlled literal inputs are exactly `114.image`, `125.sampler_name`,
`126.steps`, `131.noise_seed`, `133.prompt`, and `165.filename_prefix`. Request
preparation changes only the first five request-scoped values; node 165 retains the
neutral packaged prefix. All other literal inputs remain graph-owned and must pass
remote `/object_info` validation before upload.

The node-ID-to-class and direct-link tables above are captured independently in an
asset test. Duplicate expected controls, wrong-class nodes at expected IDs,
unexpected output nodes, missing direct links, or title/class decoys fail closed.

Node 165 must remain connected to the edited-image resize path, not the original
source or a comparison branch. Nodes 154 and 166 must be absent.

### 5.3 Immutability and effective metadata

Preparation deep-copies the graph and never mutates the packaged resource.
Effective values are recorded only after their exact eligible input is updated or
validated. The same prepared graph is queued and its effective values are returned
as result metadata.

`seed=-1` resolves once to a non-negative seed in the supported ComfyUI range. The
resolved value, not `-1`, is injected and persisted.

## 6. Request and User Contract

### 6.1 Command behavior

`/generate-image :comfyui <instruction>` is an edit-only command for this backend.
It requires exactly one staged attachment total, and that attachment must be a
supported image. Zero attachments, multiple attachments, or one non-image
attachment fail with guidance before adapter or network activity.

After normal input length and whitespace validation, the user instruction is sent
as one string. Positive and negative style transformations are both bypassed. No
separate negative-prompt channel is supported; users express exclusions in the
same instruction.

The current command grammar gains no batch syntax. The global image batch default
is ignored for this edit-only backend and the Console dispatches `count=1`.
Programmatic calls to the batch helper for backend `comfyui` require `count == 1`;
every other count is rejected before request construction.

An explicit `@style` token is rejected for this backend, because style expansion
would change the user's instruction or introduce unsupported controls. An empty
instruction is also rejected rather than falling back to conversation-context
prompt synthesis.

| Invocation/input | Result |
| --- | --- |
| `:comfyui <instruction>` with one staged image | Accepted; one raw instruction and one PNG |
| `:comfyui @style <instruction>` | Refused before request construction |
| `:comfyui` without an instruction | Refused; conversation fallback is not used |
| `comfyui` batch helper with `count=1` | Accepted |
| `comfyui` batch helper with any other count | Refused before adapter dispatch |
| Any invocation with zero, multiple, or non-image attachments | Refused before adapter/network activity |

### 6.2 Supported controls

The adapter supports:

- prompt/edit instruction;
- exactly one reference image;
- seed, including one-time `-1` resolution;
- steps;
- sampler; and
- PNG output.

The graph preserves the source dimensions. Explicit width or height overrides are
rejected. Negative prompt, CFG, model override, alternate output format, and every
other nonempty unsupported request value are rejected before source upload or queue
submission. Graph-owned defaults do not masquerade as user-applied controls.

### 6.3 Reference validation

The existing Image Generation reference-image boundary is extended, not bypassed.
Before egress it verifies:

- exactly one resolved content source;
- nonempty bytes within the configured byte limit;
- allowlisted PNG/JPEG/WebP declared MIME;
- MIME and file-signature agreement;
- successful bounded decode;
- a supported image mode;
- positive dimensions and configured width, height, and pixel-count limits; and
- the backend's `requires_reference_image` capability.

`requires_reference_image` is added backward-compatibly to the capability contract
and enforced by `worker.run_generation()` so non-Console callers cannot bypass it.

### 6.4 Success, failure, and Regenerate

`ImageGenResult` gains one backward-compatible field,
`effective_params: Mapping[str, JSONScalar] | None = None`, where `JSONScalar` is
`str | int | float | bool | None`. Existing adapters therefore need no constructor
change. The H3 adapter returns only these allowlisted keys:
`operation`, `workflow_key`, `width`, `height`, `steps`, `sampler`, and `format`.
`resolved_seed` continues to use its existing dedicated result field.

The Console batch mapper copies only those allowlisted scalar keys into
`GenerationVariantMeta.params`; it derives `content_type` from the already
validated `ImageGenResult.content_type` rather than accepting adapter metadata for
it. Unknown keys or nonscalar values are rejected and never persisted. Existing
adapters returning no `effective_params` keep `params={}` and retain current
behavior.

On successful output validation, the normal Console image path persists one PNG
attachment and aligned `GenerationVariantMeta`. Metadata therefore records the
backend, `operation="edit"`, packaged workflow key, user instruction, resolved
seed, steps, sampler, effective source/output dimensions, format, and validated PNG
MIME. It records no source path, source filename, upload name, ComfyUI descriptor,
history payload, or server response body.

Only after durable persistence succeeds may the composer remove the exact staged
attachment that initiated the edit. Removal is gated by an opaque runtime
attachment ID and operation generation, never a persisted filename or hash. If the
attachment was replaced while work ran, the replacement remains staged.

Validation, transport, generation, output, or persistence failure leaves the
source staged for retry.

Regenerate recognizes the durable `operation="edit"` metadata and refuses before
network activity, telling the user to stage the original source again. The app does
not retain a hidden source copy and never substitutes the previous edited output as
the source.

## 7. Independent Configuration and Registry Lifecycle

Image Generation gains its own `[image_generation.comfyui]` settings. It does not
read `[video_generation.comfyui]` or inherit video defaults.

The settings include:

- base URL, defaulting to `http://127.0.0.1:8188`;
- request/connect timeout;
- history poll interval; and
- total generation deadline.

There is no custom workflow path or model selector in this task. Userinfo, query,
and fragment components are forbidden in the base URL.

There is no second allowlist or trust boolean. Successfully saving `base_url` in
the canonical F9 Settings screen is the user's explicit consent to contact that
exact normalized scheme/host/port. For self-built ComfyUI endpoints, the adapter
passes only that normalized hostname to the existing `trusted_origins` egress
parameter, permitting an explicitly configured private-network host while cloud
metadata destinations remain blocked. Response data cannot extend trust: redirects
are disabled, descriptor fields never become URLs, and every endpoint must remain
same-origin with the configured base URL. The live-UAT LAN server is never
committed as a default or named in tracked evidence.

Only the canonical F9 Settings surface exposes the fields. Help text states that
the source image and instruction are transmitted to the configured ComfyUI server
and that ComfyUI retains inputs and saved outputs according to operator policy.

After a successful settings write, Image Generation clears its config cache and
adapter registry. Classification and dispatch resolve the same cached adapter
instance. Existing in-flight requests retain their adapter snapshot; subsequent
requests use the refreshed configuration. Failed settings writes do not reset
runtime state.

## 8. Preparation and Remote Preflight

All locally knowable failures occur before image egress:

1. validate the request and reference image;
2. load and validate the packaged graph topology;
3. validate and inject every supported value into a deep copy;
4. reject every unsupported supplied value;
5. query same-origin `/object_info`; and
6. verify required classes, input schemas, loader choices, sampler choice, and PNG
   output support against the prepared graph.

Remote preflight must confirm that every graph loader selection required by the
packaged workflow is accepted by the server. Missing classes, models, VAE/CLIP
choices, incompatible inputs, or unsupported sampler values fail before upload.
No object-info response body is logged.

## 9. ComfyUI API Exchange

The adapter uses the configured origin for the full exchange:

1. Upload the source with multipart `/upload/image` using a random, opaque,
   request-scoped filename. Never forward the attachment's original filename.
2. Inject the returned safe upload reference into node 114.
3. Submit the prepared graph to `/prompt` and validate the returned prompt ID.
4. Poll `/history/{prompt_id}` with a monotonic deadline, bounded intervals, and
   cancellation checks.
5. Treat unrelated previews or partial output maps as nonterminal.
6. Surface terminal execution failures through stable sanitized errors.
7. Inspect only node 165 and require exactly one descriptor in its image
   collection.
8. Require descriptor `type="output"`, a safe PNG filename, and a safe relative
   subfolder without traversal.
9. Fetch through same-origin `/view` using descriptor fields as query data, never
   as an arbitrary URL.
10. Stream the response with both declared-length and actual-byte bounds.
11. Require normalized `image/png`, PNG signature, successful decode, and the
    expected preserved dimensions.

Redirects are disabled for upload, object info, prompt submission, polling, and
download. Every request passes the existing trusted-origin egress policy.

### 9.1 Cancellation and timeout

`ImageGenRequest` gains an optional `cancel_event: threading.Event`, defaulting to
`None` for existing callers and adapters. `build_request`, `run_generation_batch`,
and the Console H3 dispatch thread the same event object without wrapping or
replacing it. The H3 Console operation owns the event; user cancellation and screen
unmount set it. The adapter checks it before every network phase and waits between
history polls with `cancel_event.wait(interval)` so cancellation is responsive
without a second polling mechanism.

Before prompt submission, an observed cancellation ends the operation without a
queue deletion. After a prompt ID exists, an observed cancellation or monotonic
deadline expiry must attempt exactly one prompt-scoped `POST /queue` with
`{"delete": [prompt_id]}`. A response that says the prompt is already running or
absent, or a sanitized transport failure during this best-effort deletion, does not
mask the original cancellation/timeout. The adapter never calls ComfyUI's global
interrupt endpoint because the server may be shared.

A running prompt can finish server-side after local cancellation or timeout. This
is an accepted and documented server-retention consequence. The Console shields
only validated-result attachment/metadata persistence once adapter success has
been accepted; it does not shield remote polling or later UI synchronization.
Cancellation before adapter success produces no local result. Once a successful
result crosses the local durable-persistence boundary, cancellation cannot erase
the committed attachment or metadata. Subsequent UI synchronization remains
cancellable and runs only for the originating live screen/operation generation.

## 10. Retention and Privacy

Node 165 is a `SaveImage` output, and `/upload/image` stores the source on the
ComfyUI server. ComfyUI has no standard portable API for deleting either artifact.
Server-side source and edited-output retention is therefore operator-managed and
is disclosed in Settings/help copy.

Local temporary sources, streamed download buffers, build artifacts, and UAT
downloads are removed after use. The persistent edited PNG exists only through the
normal Image Generation attachment contract.

Logs and user errors contain a stable component, operation phase, and exception
type only. They never contain user instructions, local paths, original filenames,
opaque upload names, graph payloads, prompt IDs, descriptors, response bodies,
model filenames, image bytes, or tracebacks in user copy.

The source workflow prompt, raw export, source filename/path, and hashes never enter
the repository, Git index, commit messages, reports, or live-UAT documentation.

## 11. Error Model

Failures are classified into stable phases:

- request/reference validation;
- packaged workflow validation;
- remote schema preflight;
- source upload;
- prompt submission;
- history polling/terminal execution;
- output descriptor validation;
- bounded PNG download/decode; and
- attachment/metadata persistence.

Every phase maps to sanitized actionable user guidance. Server status codes may be
recorded when useful; server bodies are never recorded. Persistence failure retains
the staged source and does not append a partial generation card.

## 12. Verification Strategy

Only tests related to touched files are authorized. The implementation plan must
name the exact focused test files and static-analysis targets; it must not run or
claim the full repository suite.

### 12.1 Automated tests

Focused coverage includes:

1. **Asset topology and provenance** — exact node/class/link inventory, 154/166
   absent, 165 sole output, neutral filler, prohibited source material absent.
2. **Distribution** — wheel and sdist contain exactly the sanitized image workflow;
   a fresh wheel install loads it through the confined adapter resource loader.
3. **Validation** — reference required, exactly-one staging, MIME/signature/decode
   and pixel limits, one batch, preserved dimensions, PNG-only, supported controls
   applied, unsupported controls rejected before egress.
4. **Adapter preparation** — graph immutability, exact injection, uniqueness/direct
   links, one-time seed resolution, effective metadata matching the queued graph,
   and no upload on local or remote-preflight failure.
5. **Transport** — opaque upload naming, redirect/origin confinement, safe
   descriptors, pending previews, terminal failures, monotonic timeout,
   prompt-specific pending cancellation, bounded streaming, PNG validation, and
   exact node-165 selection.
6. **Config/registry/Settings** — independent settings, precedence, successful-save
   reset, same-instance classification/dispatch, and no video dependency.
7. **Console/persistence** — staging rules, no style mutation, explicit-batch
   refusal, failure retention, one durable attachment/metadata result,
   identity-gated consumption, cancellation/unmount races, and Regenerate refusal.
8. **Privacy** — sentinel prompt/path/filename/descriptor/server-body values are
   absent from all captured logs, user copy, tracked fixtures, and reports.

Tests use strict RED-to-GREEN development and focused mutations for the load-bearing
guards: pre-upload validation, output-node filtering, same-origin enforcement,
bounded download, identity-gated source consumption, durable persistence, and
sanitized error reporting.

### 12.2 Live UAT

Live UAT is opt-in and runs against the user-configured trusted ComfyUI origin:

1. create a temporary synthetic PNG and a neutral unrecorded instruction;
2. preflight required classes, loader choices, sampler, and PNG output schema;
3. execute one edit through the real `ComfyUIImageAdapter`;
4. require one node-165 PNG with expected dimensions and resolved metadata;
5. persist through a temporary instance of the normal Image Generation
   attachment/variant boundary;
6. prove no Video Generation adapter or video store is invoked;
7. remove all local temporary and downloaded artifacts; and
8. write a prompt-free, host-free, descriptor-free report containing structural
   pass/fail evidence only.

The report explicitly states that server-side input/output cleanup is
operator-managed. It does not claim deletion that ComfyUI cannot prove.

## 13. Acceptance-Criteria Mapping

| TASK-3402 criterion | Design coverage |
| --- | --- |
| #1 Sanitized graph; 154/166 absent; 165 sole output | §§5, 12.1 |
| #2 Exactly one source/result and strict supported-control boundary | §§6, 8, 12.1 |
| #3 Existing Image Generation attachment/metadata lifecycle; no video path | §§4, 6.4, 9.1, 12 |
| #4 Independent settings, strict adapter, trusted-origin and retention boundary | §§4, 7–10, 12.1 |
| #5 No source prompt/raw export/source identity or sensitive evidence | §§5.1, 10, 12 |
| #6 Focused automated verification and live UAT only | §12 |

## 14. Approved Decisions Summary

- Dedicated Image Generation adapter, not video-adapter reuse.
- Separate Image Generation ComfyUI settings.
- Packaged strict workflow only.
- Exactly one staged attachment and one PNG result.
- Source dimensions preserved.
- One unmodified instruction string; no style or negative-prompt composition.
- Seed, steps, and sampler supported; unsupported controls rejected.
- PNG only.
- Source consumed only after durable success and only by exact runtime identity.
- Regenerate requires restaging the source.
- Loopback default; remote server explicitly configured and trusted.
- Live UAT included, with honest operator-managed ComfyUI retention.
