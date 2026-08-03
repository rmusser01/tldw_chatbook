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
| `source_url` | percent-encoded `file://` URL of the original path | real provenance; encoding satisfies `_validate_url` (no `?`/`#`/whitespace) |
| `upstream_repository` | literal `local-import` | |
| `provenance` | `LOCAL_INTEGRITY_RECORDED` only | never a stronger claim for locally computed digests |

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

Selecting a managed model resolves its installed path from
`ModelArtifactService` at launch time **without re-hashing** — `activate()`
already verified the closure; launch is a manifest lookup. The picker is one
shared widget (AC #8 discipline: one implementation, five call sites).

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
