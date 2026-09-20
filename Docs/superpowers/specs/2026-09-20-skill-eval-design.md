# Skill Eval Sub-Harness — Design

Date: 2026-09-20
Status: Proposed (awaiting user review; no implementation until approved)
Influences: [PluginEval](https://github.com/amoustakas/claude-code-plugins/blob/main/docs/plugin-eval.md) (methodology shape), `Evals/character_probe/`, `Evals/word_bench/` (repo patterns)

## 1. Problem

tldw_chatbook can host skills (SKILL.md-based, `Skills_Interop/`), but has no way to
answer "how good is this skill?" — neither how well its definition triggers and
instructs, nor how reliably it behaves in simulated use. The Evals subsystem scores
model outputs; nothing scores a *skill as the subject under test*. This design adds a
PluginEval-style layered evaluation: deterministic static analysis, LLM-as-judge
ratings, and description-only Monte Carlo simulation, blended into one composite
score with provenance.

## 2. Decisions (from brainstorming)

| Question | Decision |
|---|---|
| What does it measure? | Layered blend: definition quality + simulated behavior (PluginEval-style), one composite score |
| Subjects in v1 | Skills only (local store + project-discovered). MCP tools, tool packs, ADR-162 plugins later via subject adapters |
| Surface | Evals (Lab) screen + EvalsDB persistence; no CLI in v1 |
| Simulation mechanics | Description-only sims for standard/deep tiers; real `run_agent_loop` runs deferred to a future "thorough" tier |
| Methodology | Adapted, not cloned: same shape (weighted dimensions, anti-pattern penalties, 3-layer blend, 0–100 composite); dimensions re-cut for this repo's skill model |
| Structure | Self-contained sub-harness package `Evals/skill_eval/`, mirroring character_probe/word_bench; does not touch the classic runner registry or its `task_type` CHECK |

## 3. Goals / Non-goals

**Goals**

- Score any skill visible to `LocalSkillsService` (trusted local store and
  project-discovered) without executing it — untrusted skills are safe to evaluate.
- Three depth tiers with explicit cost: quick (0 LLM calls), standard (~16), deep (~66).
- Every report attributable to an exact skill version: content digest, source path,
  trust tier, and methodology version stored as provenance.
- Reproducible: seeded synthetic-prompt generation; decoy context recorded.
- Compare two skills via existing run-group comparison.

**Non-goals (v1)**

- Real agent-loop execution tier ("thorough") — the runner's depth enum leaves the
  seam; lands as a follow-up task.
- MCP tool / tool pack / managed-plugin subjects — the subject snapshot interface is
  the only accommodation.
- Badges, Elo corpus, certify flow, CLI entrypoint, CI gating.
- Executing skill scripts for any measurement.

## 4. Architecture

```
tldw_chatbook/Evals/skill_eval/
  models.py           # SkillSubject, SkillEvalConfig, layer results,
                      # DimensionScore, SkillEvalReport, per-cell errors
  static_analyzer.py  # Layer 1: pure function over SkillSubject → sub-scores + findings
  prompts.py          # Prompt builders (synthesis / judge rubrics / sim selection)
  judge.py            # Layer 2: rubric ratings, strict-JSON parsing, retry-once
  simulation.py       # Layer 3: seeded prompt set, selection calls, activation /
                      # consistency / failure stats — pure-Python CIs (no numpy)
  runner.py           # SkillEvalRunner.run(subject, config, chat, progress, cancel)
  scoring.py          # blending, renormalization, anti-pattern penalty, composite,
                      # grade, confidence label; methodology constants versioned
  storage.py          # EvalsDB persistence following the character_probe pattern
```

**Execution seam.** The runner takes an injected `ChatCallable` (same shape as
`CharacterProbeRunner`). The UI wires it to `chat_api_call` through the worker-thread
adapter with per-attempt timeout and a semaphore bound, per ADR-031. Tests inject
fakes. No new provider surface.

**Data flow.** UI subject picker → `SkillSubject` snapshot → runner executes layers by
depth → `SkillEvalReport` → storage → results grid / inspector.

## 5. Subject model

`SkillSubject` is an immutable snapshot captured at run start:

- identity: name, description, body text, `allowed_tools`, script declarations
- provenance: source path, content digest (sha256 of the definition), trust tier at
  capture, discovery source (local store vs project)
- derived stats used by static checks: line counts, section headings, code blocks,
  imperative-directive counts, referenced files

Skill packages (body + `references/`/`assets/` directories, scripts) are read for
*structure only*. Nothing is ever executed.

## 6. Layers

### 6.1 Layer 1 — Static analysis (deterministic, <1s, free)

Sub-checks over the snapshot (weights are calibration constants, v1 values in §7):

- `frontmatter_quality` — name validity, description length, trigger phrasing
  ("Use when…"-style intent statement present)
- `body_structure` — headings, progressive disclosure: body length sweet spot,
  `references/`/`assets/` usage, layered ordering (summary → detail)
- `tool_surface_sanity` — every `allowed_tools` entry resolves to a known catalog
  namespace (builtin/local/skill/mcp); flags over-broad grants
- `trust_surface` — consistency between trust tier, declared tools, and script
  presence (e.g. untrusted skill requesting broad local tools)
- `token_efficiency` — body size vs information density, redundancy heuristics
- `structural_completeness` — required frontmatter fields and minimal sections

Anti-pattern findings, each a 5% multiplicative penalty on the composite, floor 50%:

`EMPTY_DESCRIPTION`, `MISSING_TRIGGER`, `OVER_CONSTRAINED` (>15 MUST/ALWAYS/NEVER),
`BLOATED_SKILL` (>800 lines, no `references/`), `ORPHAN_REFERENCE` (referenced file
absent from the package), `UNKNOWN_TOOLS` (unresolvable `allowed_tools` entry),
`NAME_COLLISION` (duplicate name within the evaluated store scope).

### 6.2 Layer 2 — LLM judge (~16 calls at standard)

1. **Prompt synthesis** (1 call, generator model, seeded): produce 10 synthetic
   prompts — 5 that should trigger the skill, 5 plausible neighbors that should not.
2. **Triggering check** (10 calls): for each synthetic prompt, a single-shot selection
   call answers "which skill (or none) would you invoke" given the subject's
   name/description in a decoy context (§6.3); yields precision/recall/F1.
3. **Output quality** (3 calls): judge simulates 3 realistic tasks the skill should
   handle; rates expected output on an anchored 5-point rubric.
4. **Instruction fitness** (1 call): anchored 5-point rubric over the body — clarity,
   when-not-to-use guidance, failure handling, non-obvious-constraint coverage.
5. **Scope calibration** (1 call): anchored 5-point rubric — is this skill-sized, or
   should it be a prompt / tool / nothing.

Rubric anchors live in `prompts.py` as versioned constants; judge responses must
parse against a strict JSON schema.

### 6.3 Layer 3 — Monte Carlo simulation (deep: ~50 calls)

- Generator (seeded) produces M=10 varied prompts spanning the skill's intended range
  plus near-miss neighbors.
- K=5 repeats per prompt (deep default: 50 total sims, concurrency-bounded): each sim
  is one selection call — the subject skill presented alongside a **stable decoy
  set** (up to 8 other installed skills' name/description summaries, seeded
  selection, digests recorded) so activation is measured against realistic
  competition without whole-catalog drift.
- Metrics: activation rate (Wilson CI), output consistency (bootstrap CI, 1000
  resamples), failure rate (Clopper–Pearson; unparseable/refusal counts as
  failures). Pure Python; no new dependencies.

## 7. Scoring model

Dimensions (v1 weights; blends are static/judge/sim; renormalize over available
layers — quick runs use static only, standard drops sim):

| Dimension | Weight | Blend | Sources |
|---|---|---|---|
| triggering_accuracy | 25% | .15/.25/.60 | static trigger phrasing; judge F1; sim activation |
| instruction_fitness | 18% | .20/.70/.10 | static body_structure; judge rubric; sim consistency |
| output_quality | 15% | .00/.40/.60 | judge task simulation; sim consistency |
| scope_calibration | 12% | .30/.55/.15 | static trust/tool surface; judge rubric; sim |
| progressive_disclosure | 10% | .80/.20/.00 | static structure; judge |
| tool_surface_sanity | 8% | 1/0/0 | static only |
| token_efficiency | 5% | .60/.40/.00 | static; judge |
| robustness | 4% | .00/.20/.80 | judge; sim failure rate |
| structural_completeness | 3% | 1/0/0 | static only |

Composite = Σ(weight × blended score) × anti-pattern penalty → 0–100, letter grade
on the PluginEval bands (A+ ≥ 97, A ≥ 93, A− ≥ 90, B+ ≥ 87, B ≥ 83, B− ≥ 80,
C+ ≥ 77, C ≥ 73, C− ≥ 70, D+ ≥ 67, D ≥ 63, D− ≥ 60, F < 60), confidence label by
depth: quick → "Estimated", standard →
"Assessed", deep → "Certified". Reports embed `methodology_version`
(`skill-eval/1`) so historical reports stay interpretable when calibrations change.

## 8. Execution model

- **Depths**: `quick` = Layer 1. `standard` = Layers 1–2. `deep` = all three.
- **Cost transparency**: the UI shows the estimated call count and target models
  before launch; the estimate is derived from config, not guessed.
- **Preflight**: skill readable, models configured, API keys present — before any
  call is spent (word_bench preflight precedent).
- **Cancellation**: `CancelToken` checked between cells; partial results persist on
  cancel with the run marked cancelled (ADR-031 behavior).
- **Error isolation**: per-cell errors captured like character_probe's `CellError`;
  one failed call never kills the run. Judge JSON parse failure → one retry → the
  dimension is marked unavailable and blends renormalize. If a layer becomes empty
  (e.g. all judge calls fail at standard depth), the run completes with confidence
  degraded to the deepest intact layer and a warning finding.

## 9. Storage & provenance

Follow the `character_probe` storage pattern against `EvalsDB`:

- one run row per evaluation (run-grouped), status managed like existing runs
- per-artifact result rows: each judge rating and each sim batch (raw judge JSON kept
  for audit)
- aggregate metrics row: per-dimension scores, composite, grade, confidence,
  anti-pattern findings
- run metadata JSON: subject provenance (digest, path, trust tier, discovery source),
  decoy-set digests, seed, methodology version, depth, model refs

Comparing two skills = comparing two runs; existing run comparison surfaces apply.

## 10. UI (Evals / Lab screen)

- New "Skill eval" section in `EvalsScreen`, support module under `UI/Evals/`
  (e.g. `skill_eval_panel.py`): subject picker listing skills with trust tiers,
  depth selector, generator/judge model pickers resolved from `eval_models`, launch
  button with cost estimate, live progress via Textual workers (`exclusive=True`).
- Results render in the existing results grid + inspector: composite with grade and
  confidence, per-dimension bars, anti-pattern findings with remediation text,
  drill-down to judge artifacts and sim stats.
- Styling uses `$ds-*` design tokens only (ADR-150); keybindings follow the
  htop-style single-letter convention and footer-hint rules (ADR-031 / decision 031).

## 11. Security & trust

- Skills under test **never execute**; scripts are inspected as declarations only.
- Untrusted skill content is data, not instructions: judge/sim prompts delimit the
  skill body explicitly; judge system prompts state that rubric content is inert;
  responses must satisfy a strict JSON schema — anything else is a retry-then-fail
  cell, never interpreted as instructions.
- Body size fed to models is capped; oversize bodies are truncated with a
  `BLOATED_SKILL`-adjacent finding rather than silently dropped.
- Provenance (digest, trust tier) is stored with results; a report about an untrusted
  skill is visibly labeled as such in the UI.

## 12. Testing

- **Unit**: static analyzer over a fixture SKILL.md corpus (good / bad / each
  anti-pattern); scoring math (blending, renormalization, penalty floor, Wilson and
  bootstrap CIs against hand-computed values); judge strict-JSON parsing (valid,
  malformed, injection-payload); prompt builders (snapshots).
- **Integration**: runner with fake `ChatCallable` across all depths; cancel mid-run;
  all-judge-calls-fail degradation; storage round-trip into in-memory EvalsDB with
  provenance assertions.
- **UI**: mount/worker wiring tests following existing Evals screen test patterns.

Per repo policy, targeted runs only unless a full sweep is requested.

## 13. ADR check

```text
ADR required: yes
ADR path: backlog/decisions/<next-available>-skill-eval-sub-harness.md
Reason: new Evals sub-harness subsystem; introduces LLM-as-judge to the repo; new
cross-module interface (Evals ↔ Skills_Interop subject snapshots) and a
provenance/reporting contract. ADR to be created before implementation begins and
linked from the backlog task and implementation plan.
```

## 14. Open items resolved by default (override on review)

- **Decoy context for activation**: fixed seeded decoy set of ≤8 installed skills
  (v1 default) rather than the full catalog, for comparability across runs.
- **Deep sim count**: 50 (PluginEval parity), configurable.
- **Letter grades**: full A+…F bands as in PluginEval.
- **Generator/judge defaults**: separate model roles, both resolved from
  `eval_models`; no bundled provider assumptions.

## 15. Future work (explicitly deferred)

- "Thorough" tier: real `run_agent_loop` runs with the skill bound, sandboxed/
  faked `LoopDeps`, measuring task completion and tool-call correctness.
- MCP tool / tool pack / managed-plugin (ADR-162) subject adapters.
- Badges, Elo corpus, certify flow, CLI entrypoint, CI thresholds.
