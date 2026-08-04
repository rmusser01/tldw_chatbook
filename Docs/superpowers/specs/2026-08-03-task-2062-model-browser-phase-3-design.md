# TASK-2062 — Model browser Phase 3: GGUF adoption and legacy-downloader retirement

**Date:** 2026-08-03
**Task:** TASK-2062
**Parent spec:** `Docs/superpowers/specs/2026-08-01-task-596-model-artifact-browser-design.md` (Phases 1–2 delivered across PRs #1175, #1185, #1190, #1210, #1245)
**ADR:** `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`
**Status:** approved section-by-section on 2026-08-03

## Decisions (made with Robert)

1. **Pickers are primary; free text is an explicit escape hatch** (Option A). Each
   local-server section offers a managed-model picker; the raw path field is
   demoted behind an "use an unmanaged path…" disclosure — except where free
   text is a first-class input (see per-server mapping).
2. **Import copies into the store.** Recording digests for a file the store does
   not own would let the file change under the claim. The import dialog says
   plainly: the file is being **copied** into Chatbook's managed store, why
   (integrity can only be guaranteed for bytes the store owns), and that the
   original is untouched and may be deleted afterwards to reclaim the space.

## Import flow

From an unmanaged row's new **Import** action in the Installed view:

```
select unmanaged file
  → import dialog          source path, file size, store destination,
                           free-space check (needs ~1× file size in the store),
                           the copy notice above
  → threaded worker        stream-copy the original into a temp directory
                           INSIDE the store root while hashing (one read of
                           the original, one write); progress + cancel via
                           the existing ModelInstallProgress machinery
  → build descriptor       from the computed digest and size;
                           provenance = LOCAL_INTEGRITY_RECORDED
  → install(descriptor, temp, consume_source=True)
                           the sealed core verifies the STORE-SIDE copy
                           against the declared digest (this read is the
                           guarantee — the store verifies bytes it owns, not
                           bytes that flew past during the copy) and promotes
                           via rename (EXDEV falls back to copy)
  → success state          offers Activate; repeats the you-may-delete-the-
                           original note with the source path
```

**Abort safety requirement:** a cancelled or crashed import must leave nothing
the store treats as live, and stale import temps must be reclaimable by the
existing repair/GC paths. The plan decides the mechanism with the code in front
of it (the marked download-stage seam, or an import-temp naming that
`reconcile()` classifies); the requirement is the contract.

Re-import of a byte-identical file is a no-op (same ref, already installed).
Re-import of a changed file with the same name mints a distinct revision (see
identity) rather than colliding.

## Identity for imported files

| field | value | note |
|---|---|---|
| `artifact_id` | sanitized lowercase filename stem | must satisfy `_validate_canonical_component` |
| `revision` | `sha256-<first 12 hex of the file digest>` | content-addressed: identity follows bytes |
| `variant` = `precision` | parsed quant tag (e.g. `q4-k-m`) or `imported` | validator requires precision == reference.variant |
| `source_url` | percent-encoded `file://` URL of the original path | **requires the sealed-core accommodation below** — `_validate_url` today mandates `scheme in {http, https}` with a hostname, so a `file://` value raises at construction |
| `upstream_repository` | literal `local-import` | |
| `license_id` | `unknown` | an imported file's license is genuinely unknown; never guess one |
| `license_url` | empty, via the accommodation below | `_validate_url` requires a non-empty http(s) URL and no honest one exists |
| `provenance` | `LOCAL_INTEGRITY_RECORDED` only | never a stronger claim for locally computed digests |

### Sealed-core accommodation for local-origin descriptors

An earlier draft of this spec claimed a percent-encoded `file://` URL "satisfies
`_validate_url`". **That was wrong** — the validator's second half explicitly
requires an http(s) scheme and hostname, and the same applies to `license_url`,
which an imported file cannot honestly supply. Rather than smuggle in fake https
URLs, the core gets one deliberate, narrow, additive change (the same class of
change as `ArtifactPreflightEntry.provenance` in Phase 1):

- `source_url` MAY be `file:///` + a percent-encoded absolute path **only when
  every provenance entry is `LOCAL_INTEGRITY_RECORDED`** (cross-field
  validation, like the existing `precision == variant` rule). All other
  descriptors keep the strict http(s) rule.
- `license_url` MAY be empty **only when `license_id == "unknown"`**, same
  gating. The UI renders "License: unknown" with no link.

**Defense in depth, verified against the code:** allowing `file://` in a
descriptor does not create a local-file-read primitive in the download path —
`egress._pre_resolution` rejects any non-http(s) scheme (`reason="scheme"`)
before resolution, so an acquisition flow handed a `file://` source dies at the
egress gate. The accommodation task must still add an explicit test pinning
that refusal, so the two rules can never drift apart silently.

## Per-server launch-path mapping

Five fields exist today (`llamacpp`, `llamafile`, `vllm`, `onnx`, `mlx` — the
parent spec's list of four missed ONNX), read via
`query_one("#<server>-model-path", Input)` in the event handlers.

| server | managed picker lists | free text |
|---|---|---|
| llama.cpp | ready GGUF models | demoted: "use an unmanaged path…" disclosure |
| llamafile | ready GGUF models | demoted, as above |
| ONNX | ready ONNX models | demoted, as above |
| vLLM | ready models of compatible format | **stays first-class** — HF repo ids are a legitimate input, not a legacy path |
| MLX | ready models of compatible format | **stays first-class**, same reason |

**Resolution happens at selection time, and the launch handlers stay
untouched.** The existing launch flow reads each `#<server>-model-path` Input
at button-press time (`handle_start_*_server_button_pressed` in
`Event_Handlers/LLM_Management_Events/llm_management_events.py`), and a
`FileOpen` browse dialog already populates those Inputs via
`_make_path_update_callback`. The picker slots into exactly that seam: choosing
a managed model resolves its payload path from `ModelArtifactService` (for
GGUF, `<artifact_dir>/<files[0].path>`; no re-hashing — `activate()` already
verified the closure) and writes it into the Input. Launch code does not change
at all, which keeps this task's blast radius out of five server-start handlers.

Accepted trade-off, stated plainly: the path is resolved when picked, not when
launched. A model deleted or deactivated between selection and launch yields a
stale path and the server fails with file-not-found — the same failure mode a
hand-typed path has today for a moved file. Launch-time re-resolution would
require rewriting all five handlers and is deliberately out of scope.

The picker is one shared widget (AC #8 discipline: one implementation, five
call sites), and the **import worker follows the TASK-1803/1914 ownership
rule**: the view posts an intent; `LLMScreen` owns the threaded copy-and-hash
worker, so a recompose mid-import cannot orphan it. The import dialog is
bespoke — it is NOT `ModelPlanPanel`, because no `PreflightReport` exists for a
local import and synthesizing a fake one would misuse the consent machinery.

## Retirement

- `Widgets/HuggingFace/` — all five modules (~2,200 lines), including
  `DownloadManager` and the unverified direct-write download path ADR-025
  requires gone.
- The `download-models` rail row in `MODELS_RAIL_SECTIONS` and its
  `LLMManagementWindow` view mount.
- The `model_download_dir` config default's role as a download *destination*.
  The unmanaged **scan** of that directory stays: models a user already has
  remain visible in Installed (now with Import as their path into management).
- ADR-025 rollback note applies: if Phase 3 must be rolled back, new installs
  stay disabled rather than reverting to unverified direct writes.

## Testing

- Import: happy path; cancel mid-copy (nothing live, temp reclaimable); crash
  simulation (stale temp classified by repair, not adopted); digest mismatch
  between copy and verify (store rejects; original untouched); free-space
  refusal; filename→identity edge cases (quant parsing, sanitization,
  Windows-reserved names); marker-absence on any logged paths per TASK-1722's
  conventions.
- Pickers: each server section lists only compatible ready models; the escape
  hatch (where demoted) still launches an arbitrary path; vLLM/MLX free text
  unchanged in behavior.
- Retirement: rail row and views gone; no import of `Widgets.HuggingFace`
  remains; the boundary and no-"artifact" copy rules hold (existing AST and
  subprocess tests extended to any new modules).
- Sabotage-verify per the workstream convention: exact-string edits with count
  assertions; leak/absence tests must be proven red against a broken
  implementation before they count.
