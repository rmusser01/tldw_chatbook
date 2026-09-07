---
id: TASK-21136
title: 'Propositions vendoring + program descope (sub-project #6)'
status: Done
assignee: []
created_date: '2026-08-23'
updated_date: '2026-08-23'
labels:
  - chunking
dependencies: [TASK-21135]
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Sub-project #6 of 6 — the closing act of the chunking parity program (single branch off `origin/dev` @ `c56eab813b`, post-#5-merge; no new ADR — the spec's §8 rulings are the long-form record): one cheap vendoring plus two descopes worth more than their implementations. `strategies/propositions.py` is vendored from the unmoved pin (the 39th engine file, the manifest move) so the already-routed `propositions` method starts working — heuristic engine by default, spacy optional, llm engine through the #1 rolling_summarize callback-contract precedent (payload-dict → positional adapter in `Chunk_Lib.py`; LLM failure falls back to heuristics — upstream design, pinned as parity). Permanently NOT vendored, with recorded rulings: `auto_boundary_assistant.py` (server-stack seams, no consumer — covered by #3 auto-selection + #4 agent tools) and `async_chunker.py` (http_client/exceptions deps; chatbook chunks in-process); telemetry no-op reaffirmed; #1 §0 drift obligation for the two files closed as moot. The manifest's excluded list becomes the descope ledger; zero "deferred to #N" residue survives. Docs, close-out, and this board closure.

Spec: `Docs/superpowers/specs/2026-08-23-propositions-vendoring-design.md` (§4 the descope rulings, §5 vendoring + the LLM contract, §6 testing, §7's 9 ACs, §8's 5 rulings). Plan: `Docs/superpowers/plans/2026-08-23-propositions-vendoring.md` (three tasks; per-task sdd ledger in `.superpowers/sdd/2026-08-23-propositions-vendoring/`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Vendored: `strategies/propositions.py` from the existing pin — the manifest move (37→38 in the list; 39 `.py` files in the tree incl. `__init__.py`), byte-faithful modulo the one prompt_loader import rewrite, zero new shims, zero new dependencies; every `load_prompt` pair in the file covered by the shim's `_KNOWN` (source-scan pinned) (spec §7 #1, #5)
- [x] Method live: `improved_chunking_process(..., {"method": "propositions"})` returns heuristic chunks instead of raising `InvalidChunkingMethodError`; the un-skipped upstream suite (10 tests) green; both formerly-deferred test files carry terminal dispositions; zero "deferred to #6" residue across `Helper_Scripts/`, `tldw_chatbook/`, `Tests/` (spec §7 #2, #3)
- [x] LLM contract + fixtures: the payload-dict→positional adapter per the #1 precedent (callers keep their signature), the fallback-to-heuristics leg pinned, the positional shape pinned verbatim; propositions heuristic cases join the byte-pinned golden corpus 70→77 (spec §7 #4, #7)
- [x] Descope ledger: the manifest `excluded` entries for `auto_boundary_assistant.py`/`async_chunker.py` carry the not-vendored rulings, telemetry no-op reaffirmed, the #1 §0 drift obligation recorded closed — pinned by `Tests/Chunking/test_descope_ledger.py` (spec §7 #6)
- [x] Docs + close-out: CHANGELOG and both live user-guide method rosters gain `propositions` (the dated 2026-08-19 rag.md verification note stays as history); targeted suites green with zero new failures; both Task-2 reviewer follow-ups dispositioned here (TASK-21137 filed; the llm_override_scope switch declined with rationale) (spec §7 #8, #9)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Vendor the file + the descope ledger + residue terminal-wording (plan Task 1)
2. LLM-contract adapter + fallback pin + parity fixtures (plan Task 2)
3. Docs + targeted close-out + residue re-grep + board closure (plan Task 3)
<!-- SECTION:PLAN:END -->

## Implementation Notes

Approach: one branch (`feat/chunking-llm-extras` — the name predates the descope ruling; the PR carries the real one), TDD per task through the sdd ledger, engine tree untouched post-sync (vendored files are never hand-edited; the adapter lives in chatbook-owned `Chunk_Lib.py` only).

- Commits `0a376150f`/`0b894dcb9` (spec + plan) through HEAD: `b16323632` (vendoring + ledger + descope pins), `66a622e5f` (Task-1 review round: honest prompt-defaults rationale ×3 sites + terminal wordings re-synced), `467286eb1` (LLM adapter + fallback pin + fixtures), plus this task's own docs + close-out + board-closure commit.
- **The `_KNOWN` honest-divergence ruling:** the three profile pairs (`proposition_claimify`/`proposition_gemma_aps`/`proposition_generic`) map to `""` — the engine's in-code defaults are chatbook's effective instructions. Upstream DOES ship YAML overrides at the pin (wording deltas on 2 of 3) — a recorded divergence and a candidate for Internal_Prompts catalog entries if true parity is ever wanted; with `""` values, user overrides cannot ride the catalog (the resolver is never consulted — a future override mechanism changes the map values, not the keys). This is load-bearing: `_build_llm_prompt` runs outside the per-window try, so an unmapped pair would raise KeyError straight out of `chunk()` with no heuristic fallback.
- Counts pinned to reality: manifest list 37→38; engine tree 39 files (the spec's "39th file"/"38→39" arithmetic slip — tree vs list — is pinned by `test_engine_tree_complete`).
- **The two Task-2 reviewer follow-ups, dispositioned:** (i) the `llm_override_scope` switch — **declined**: `engine/llm_context.py:15` is the better seam (thread-local, finally-safe contextmanager) and the Task-2 report's "no setter exists" sentence was false (corrected on disk in the sdd report), but the shipped instance-attr set/restore is verified safe in every production flow (fresh `Chunker` per call; restore pinned by test) — the switch is churn on a proven-safe path, and a future touch of the adapter can adopt it; (ii) the multi-chunk propositions golden arm — **filed as TASK-21137** (low priority): the frozen corpus grid packs every small corpus into one chunk for this method, so the drift detector is blind to packing-boundary changes — a real, cheap-to-close coverage gap.
- Review rounds: Task 1 one fix round (approved after); Task 2 approved first pass (the seam correction + follow-up candidates carried here per the ledger).
- Close-out evidence: `Tests/Chunking/` + story + import-weight → 602 passed / 24 skipped / 1 xfailed, zero failures (Chunking alone = the Task-2 baseline 596/22/1 exactly; the story + import-weight files add 6 passed / 2 env-only torch skips); residue grep `grep -rin "deferred to #" Helper_Scripts/ tldw_chatbook/ Tests/` → zero hits; `pyproject.toml` untouched across the branch.
- Depends on TASK-21135 (#5, merged via PR #1984 — this branch's base).
