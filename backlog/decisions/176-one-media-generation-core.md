# ADR-176: One Media_Generation core under the two modality packages

Status: Accepted
Date: 2026-09-22
Related Task: [TASK-32856](../tasks/task-32856%20-%20Merge-Image-and-Video-Generation-under-one-core.md)
Supersedes: [ADR-044](044-ephemeral-generated-video-storage-playback-and-streaming.md) decision 2, only insofar as it mandates a mirroring package. ADR-044's own alternatives table recorded that the parallel package "can later merge under a shared umbrella"; this is that merge. ADR-044 decisions 1, 3, 4, 5, 6, and 7 are untouched.

## Decision

The modality-shared machinery of `Image_Generation/` and `Video_Generation/` lives once in a new `tldw_chatbook/Media_Generation/` package, parameterized by modality:

- the adapter-registry skeleton (spec table, lazy resolution, enablement,
  per-backend caching, register/reset lifecycle);
- the TOML/keyring/env secret-precedence config machinery;
- the byte-identical request-validation bound/allowlist helpers.

Measured at merge time, the worker and full request-validation modules
are pattern-shared, not code-shared (0.44-0.45 normalized similarity; no
verbatim block >= 20 lines): extracting them would produce a longer
parameterized pipeline than either file and would obscure video's
decision-1-revision-3 container-agreement sequencing. They remain
modality-local until real duplication emerges; the dataclass builders
and the ComfyUI adapter lifecycle (~50% shared) are the named next
slices if the owner wants deeper consolidation.

`Image_Generation/` and `Video_Generation/` remain the public import
surface — module paths, class names, frozen request/result contracts, and
the single validation choke point per modality are unchanged — reduced to
backend spec tables, modality parameters, and thin adapters. No modality
flag threads through contracts: image requests stay image requests.

## Delta Resolution Rules

The mirror drifted; every behavioral delta resolves to the stricter side
or is an explicit recorded exception:

- **Config snapshot context and runtime lock** (image-only today) apply to
  both modalities: adapter construction always runs inside the modality's
  config snapshot, and registry get/reset always hold the runtime lock.
- **ComfyUI hardening** (`_BodyChunkSupervisor`/`_SendSupervisor`,
  `/object_info` schema validation) is image's and transfers when the
  ComfyUI adapters share lifecycle; video adopts it rather than averaging.
- **Adapter-failure log detail** stays per-modality for now (image logs
  the exception text, video logs only the error type — the privacy-
  stricter form). Unifying it is a follow-up privacy decision, not a
  silent average.
- `Video_Generation/video_store.py` has no image twin and is not in scope
  (ADR-044 decision 1 stands).

## Context

`Video_Generation/` (2026-08) was a deliberate structural clone of the
proven `Image_Generation/` pipeline. The 2026-09-19 cascade review
measured the cost: `adapter_registry.py` differs by 63 normalized lines of
~309; `config.py` duplicates ~250-280 lines of TOML/keyring/secret-
precedence machinery (video's docstring admits the mirroring); worker and
request-validation skeletons are shared; the ComfyUI adapters share ~50%
lifecycle and video already imports image's transport. A third modality
would clone it all again.

## Consequences

- One registry/config-machinery/validation-helper core to harden; fixes
  land for both modalities. The workers and full validators stay
  modality-local (see Decision) until duplication justifies extraction.
- The modality packages shrink to their genuine per-modality content
  (backend tables, adapters, formats/metadata/store).
- Public import paths and request/result contracts are unchanged, so
  callers and the 26 image + 15 video test files are the regression base.
- Boot census: one new module family; ratchets re-pinned in the merging
  PR per ADR-097.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Keep the two mirror packages | Every skeleton fix lands twice; the third modality clones a third copy. |
| Merge everything into one package with a modality flag on contracts | Image contracts (width/height/steps) and video contracts (duration/fps/ratio) are genuinely different; a flag threads through every typed surface and recreates ADR-044's original problem in one file. |
| Wait for a third modality before merging | The duplication cost is paid now, every release; ADR-044's consequences already named the merge as the expected end state. |
| Rewrite both packages onto the core at once | High blast radius; the merge lands in reviewable slices (registry, then config machinery, then worker/validation) with the test files as the pin. |

## Links

- [ADR-044: Ephemeral generated video storage, playback, and streaming](044-ephemeral-generated-video-storage-playback-and-streaming.md)
- Cascade review 2026-09-19 (`qa/cascade-review-2026-09-19/report.md`)
