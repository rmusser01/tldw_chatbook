# Lens: Interactive Logprob Inspector — Design (Phase 1)

Date: 2026-09-22
Status: Draft — awaiting user review
Companion follow-up: Phase 2 (true jacobian-lens via the Transformers lane) gets its own spec later; this document only reserves the seams it will need.

## Why

tldw_chatbook can launch and manage llama.cpp servers, and its Evals/word_bench
subsystem already proves the app can capture and reason about next-token
distributions from a live llama.cpp server. What does not exist is a *tight
iteration loop* for prompt work: a surface where a researcher or prompt
engineer edits a prompt, presses one key, and sees how the model's next-token
distribution shifted — across variants, against probes they care about, with
the data exportable for offline analysis.

The long-term inspiration is Anthropic's jacobian-lens ("j-space"): reading
out what a model's internal states are disposed to say. A stock llama.cpp
server cannot expose per-layer activations or gradients, so true j-lens is
out of scope for Phase 1 (it arrives in Phase 2 via a torch + HuggingFace
subprocess). Phase 1 delivers the **output-space** version of that goal:
monitor what the model is about to say, precisely and immediately, while a
prompt is being engineered.

## Goals

- Single-key capture: prompt in editor → running llama.cpp model → top-K
  next-token distribution rendered in the TUI (logprobs, entropy, top-1 mass,
  near-tie marking).
- Prompt variants kept side by side (a variant rack) with a designated
  baseline; every capture shows Δ against the baseline (per-token logprob
  delta, divergence with shared effective-K, probe rank shifts).
- Named probe tokens whose logprob/rank is tracked on every capture, so a
  user can watch "the feature I'm trying to elicit" move as the prompt
  changes.
- Persistence and export: captures are stored in SQLite and exportable as
  JSON and CSV for offline analysis (researchers keep their data).
- Reuse word_bench's hard-won capture discipline verbatim: pinned neutral
  sampler, fixture-pinned response normalization, bounded responses,
  control-token detection, degenerate-distribution canary.

## Non-goals (Phase 1)

- Streaming generation traces (per-token logprobs/entropy during a
  multi-token generation). Natural Phase 1.5; the capture contract below
  leaves room for it (`kind` field).
- Per-position prompt sweeps (distribution at every prefix of the prompt —
  the closest output-space analog to a j-space slice). O(prompt-length)
  requests per refresh; deferred.
- Per-layer activations / jacobians / logit-lens-at-layer-l. Requires
  in-process torch + HF weights; that is Phase 2, a separate spec.
- Non-llama.cpp providers. Like `Evals/sample_bench.py`, the lens resolves
  targets narrowly: the `llama_cpp` provider's configured/verified endpoint.
  Broadening later is a config-resolution change only.

## UX surface

A fourth Lab mode: the mode strip in `UI/Screens/lab_mode_strip.py` gains a
**Lens** chip routing to `lens`, hosted on the existing Lab frame
(`UI/Screens/lab_frame.py`) and its rail | body | inspector workbench
(`UI/Lab_Modules/lab_workbench.py`). The name "Lens" is deliberate: Phase 2's
jacobian-lens view lands in the same mode later.

Layout (three-region workbench):

- **Rail (left)**: variant rack — list of prompt variants with a marked
  baseline; add/duplicate/delete/rename variants; the active editor target.
- **Body (top-left of body region)**: prompt editor (`TextArea`) for the
  active variant, plus a prompt-mode selector (`raw` prefix vs `chat`
  template, same semantics as word_bench) and the probe-token editor (a
  comma-separated token list, persisted per session).
- **Body (bottom / right of body region)**: the distribution panel — a
  DataTable of top-K tokens: token text, logprob, probability bar (token
  colored cells per the design tokens — no ad-hoc hex), vocabulary rank,
  Δ-vs-baseline columns when a baseline exists. Header row of readouts:
  entropy (nats), top-1 mass, truncated mass, near-tie flag, k
  requested/returned.
- **Inspector (right)**: capture history for the session (one row per
  capture: timestamp, variant, prompt-mode, entropy, top-1) and the probe
  panel — per-probe logprob, rank, and rank-change-since-baseline arrow.
  Export actions live here.

Keybindings follow ADR-031: single-letter htop-style screen actions (e.g.
`r` run capture, `v` new variant, `b` set baseline, `e` export), no
terminal-convention keys, footer hints advertise only implemented actions.

Styling: a new source module `tldw_chatbook/css/features/_lens.tcss`
composing `$ds-*` tokens only, rebuilt via
`python tldw_chatbook/css/build_css.py`; never editing the built
`tldw_cli_modular.tcss` (ADR-150). The governance test
(`Tests/UI/test_design_token_governance.py`) applies.

## Architecture

New package `tldw_chatbook/Lens/` — small, one purpose per module:

- `Lens/capture.py` — `LensCaptureClient`. One method matters:
  `capture(prompt, *, prompt_mode, top_k, probes) -> LensCapture`. It POSTs
  to `{base}/v1/completions` (`max_tokens: 1`, `logprobs: top_k`) or
  `{base}/v1/chat/completions` (`logprobs: true`, `top_logprobs: k`) exactly
  as word_bench's `WordBenchCaptureClient._build_request` does, with the
  pinned **neutral sampler** (temperature 1.0 — not 0 — top_k 0, penalties 0;
  llama.cpp applies samplers before reporting logprobs, and temp-0 collapses
  the distribution). Response parsing and control-token detection delegate
  to `Evals/word_bench/normalizer.py` — the lens does not invent provider
  shapes. Pooled httpx client, bounded response sizes, bounded deadlines.
- `Lens/models.py` — `LensCapture` (variant id, prompt snapshot hash, prompt
  mode, k requested/returned, top-K `TokenProb` tuple reusing
  `Evals/word_bench/models.TokenProb`, entropy, top-1 mass, truncated mass,
  probe readings, `kind: "next_token"` for Phase 1.5/2 forward-compatibility).
  `Variant`, `VariantSet` (ordered, max 8, one baseline), `ProbeReading`
  (reuses `Evals/word_bench/analysis.resolve_probe` semantics:
  observed/bounded/never_observed + logprob).
- `Lens/diffing.py` — variant comparison: per-token logprob delta for shared
  tokens, divergence via `Evals/word_bench/analysis.divergence` (shared
  effective-K), entropy delta, probe rank shifts. No new math; composition
  of word_bench analysis primitives.
- `Lens/targeting.py` — resolves the llama.cpp base URL. Order: the verified
  connection published by `LLM_Management/llamacpp_connection.py` (the
  Models screen's "running and verified" endpoint), falling back to
  `[api_settings.llama_cpp] api_url` (sample_bench's narrow resolution).
  Readiness check reuses the word_bench distribution canary once per target
  per session ("The capital of France is" → " Paris",
  pass/degenerate/never_observed verdicts surfaced in the header).
- `Lens/store.py` — SQLite persistence into the existing Evals database
  (`DB/Evals_DB.py`): new `lens_sessions` and `lens_captures` tables
  (schema version incremented, migration added per repo rules). Captures are
  append-only; sessions group them.
- `Lens/exporters.py` — JSON and CSV export of a session (all captures with
  variant metadata and probe readings), written through
  `Utils/path_validation.py`. JSON round-trips into pandas without post-
  processing.

UI layer:

- `UI/Screens/lens_screen.py` — the Lab-mode screen; owns session state,
  runs captures via `app.run_worker(..., exclusive=True)` (house worker
  pattern), posts nothing to the chat pipeline.
- `UI/Lens/distribution_grid.py` — the top-K DataTable + readout header.
- `UI/Lens/variant_rack.py` — rail widget.
- `UI/Lens/probe_panel.py` — inspector probe readouts + capture history +
  export buttons.

Event flow stays local to the screen (widget → handler → worker → reactive
update), consistent with the Evals screen; no new global events.

## Data flow

1. User edits variant text / probes, presses `r`.
2. Screen validates target readiness (cached canary), reserves the worker,
   calls `LensCaptureClient.capture` for the active variant.
3. Capture is normalized, analyzed (entropy, masses, probes), stored
   append-only, and rendered reactively; if a baseline capture exists for
   the same prompt-mode + probe set, diff columns populate.
4. Capture history row appears in the inspector; export writes JSON/CSV on
   demand.

Concurrency: one capture in flight at a time (`exclusive=True`); no
background sweeps. Server errors surface as bounded, human-readable messages
in the header (the word_bench `CellError` discipline), never tracebacks.

## Error handling

- Target down / wrong port: probe with the existing verified-connection
  machinery (`probe_llamacpp_target` semantics); header shows a stopped/
  unverified state and disables `r`.
- Degenerate server (canary fails): header warning; captures still possible
  but flagged in the export.
- `k_returned < k_requested`: truncated-mass readout shown, near-tie marking
  per `analysis.near_tie`; no silent padding.
- Control-token top-1 (normalizer detection): rendered as a flagged row.
- Oversized prompt: bounded by the server's context; surfaced as a capture
  error with the server's message, bounded length.

## Configuration

`[lens]` section in config.toml: `top_k` (default 20), `max_variants` (8,
matching the store), `history_cap` (captures retained per session, default
500). Settings UI changes, if any, land in the canonical F9 settings screen
only — never the deprecated windows.

## Testing

- `Tests/Lens/test_lens_capture.py` — request shape pinned to fixtures
  (fake server / recorded responses, following word_bench's test patterns);
  neutral-sampler pinning asserted; bounded-response behavior.
- `Tests/Lens/test_lens_models.py` / `test_lens_diffing.py` — property-style
  tests for diffing (symmetry/baseline-zero cases), probe resolution states.
- `Tests/Lens/test_lens_store.py` — real in-memory SQLite; append-only;
  migration from prior schema version.
- `Tests/Lens/test_lens_targeting.py` — verified-connection precedence and
  config fallback.
- `Tests/UI/test_lens_screen.py` — screen wiring, worker exclusivity,
  keybinding/footer conformance (ADR-031), design-token governance via the
  existing suite.
- Targeted runs only per AGENTS.md testing guidance; full sweep only on
  explicit request.

## ADR

ADR required: yes. This introduces a new long-lived Lab surface, a new
capture contract against llama.cpp OpenAI-compat endpoints (extending the
word_bench discipline to an interactive surface), and new Evals-DB tables.
The ADR (`backlog/decisions/NNN-lens-lab-mode-and-capture-contract.md`,
number assigned at creation) records: Lens as a Lab mode destined to host
Phase 2 j-lens; the decision to reuse word_bench's normalizer/analysis
rather than fork it; the neutral-sampler pinning as a contract; and storage
in the Evals DB. It will be created before implementation begins and linked
from the backlog task and implementation plan.

## Phase 2 seams (reserved, not designed here)

- `LensCapture.kind` discriminator (`next_token` today; `jlens_slice` later).
- The Lab "Lens" mode is the host surface for the Phase 2 view; Phase 2 runs
  torch + HuggingFace in a subprocess behind `Utils/optional_deps.py`,
  vendors or pins the jlens library, and applies fitted transport matrices
  to HF-format weights. Its spec will decide fitting UX, model size
  ceilings, and GPU gating.

## Decisions made in the user's absence (review these first)

The design dialogue established: both phases, staged; interactive inspector
as the primary interaction. The following were selected by recommendation
because the dialogue paused — flag anything you want changed:

1. Capture depth: next-token only in Phase 1; streaming generation trace
   deferred to Phase 1.5 (room reserved via `kind`).
2. Surface: a new fourth Lab mode chip ("Lens") rather than folding into
   the Evals screen.
3. Targets: `llama_cpp` provider only (verified endpoint, config fallback).
4. Export: JSON + CSV included in Phase 1 (researchers need data out).
5. Storage: new tables in the existing Evals database rather than a new DB
   file.
