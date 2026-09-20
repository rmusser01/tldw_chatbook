# ADR-172: Skill evaluation sub-harness

Status: Proposed (2026-09-20) — design spec reviewed and verified against code;
implementation pending.
Date: 2026-09-20
Related Task: [TASK-32832](../tasks/task-32832%20-%20Add-skill-eval-sub-harness-layered-skill-scoring.md)
Companion spec: [2026-09-20 skill-eval design](../../Docs/superpowers/specs/2026-09-20-skill-eval-design.md)
Related: [ADR-009](009-local-skill-trust-boundary.md) (evaluation reads skills as
data and never executes them)

## Decision

Add a self-contained Evals sub-harness (`Evals/skill_eval/`) that scores a **skill as
the subject under test** through three layers — deterministic static analysis,
LLM-as-judge rubric ratings, and seeded description-only Monte Carlo simulation —
blended into a composite 0–100 score with letter grade and depth-based confidence
label. Methodology adapted from
[PluginEval](https://github.com/amoustakas/claude-code-plugins/blob/main/docs/plugin-eval.md)
for this repo's skill model (bare-name `allowed_tools` narrowing, trust tiers, package
directories).

## Context

The Evals subsystem scores model outputs; nothing scores a skill *definition* or its
simulated behavior. LLM-as-judge does not exist anywhere in the repo yet. The classic
runner registry fits poorly: its `task_type` CHECK constraint and single-sample
runner abstraction were already bypassed by `character_probe` and `word_bench`, which
established the self-contained sub-harness pattern this decision follows. Skills are
untrusted content (ADR-009), so evaluation must be structure-reading only — which
also makes untrusted skills safe to evaluate.

## Contracts

1. **Subject**: an immutable `SkillSubject` snapshot (name, description, body,
   `allowed_tools`, scripts, content digest, trust tier, source: local-store row or
   directory path). Packages are read for structure only; nothing executes.
2. **LLM boundary**: an injected chat callable *wider* than character probe's — it
   carries the resolved target (provider, model_id, API key) so judge/generator roles
   can be any configured provider. Runner-owned semaphore + `asyncio.to_thread`;
   explicit `request_timeout`/`request_retries`; preflight via
   `get_provider_readiness` (which also supplies the resolved key).
3. **Scoring**: nine adapted dimensions with per-layer blend weights renormalized by
   depth (quick → "Estimated", standard → "Assessed", deep → "Certified"); seven
   anti-patterns as multiplicative penalties (floor 50%); every report embeds a
   `methodology_version` so historical scores stay interpretable.
4. **Storage**: existing EvalsDB generic tables only — `eval_tasks` row with
   `task_type="generation"` + `config_data["bench_type"]="skill_eval"` discriminator,
   run-grouped `eval_runs` with the full report snapshot in `config_overrides`,
   per-artifact `eval_results` rows. No schema migration in v1.
5. **Trust**: untrusted skill bodies and decoy skill descriptions are delimited inert
   data in every judge/sim prompt; judge responses must satisfy strict JSON schemas;
   parse failures retry once, then the affected dimension degrades and blends
   renormalize.
6. **UI**: an Evals-screen section plus a dedicated detail-view branch;
   `ResultsGrid` remains word-bench-only (the character-probe placeholder is the
   precedent for per-bench-type detail panes).

## Consequences

- LLM-as-judge enters the repo here, scoped to this package in v1.
- Activation fidelity is description-level only (matching `SkillToolProvider`'s
  verbatim catalog presentation); real agent-loop evaluation is an explicit future
  "thorough" tier, not a v1 claim.
- Comparison of two skills rides existing run-group comparison; badges, Elo corpus,
  certify flows and CI gating are out of scope.

## Alternatives considered

- **Extend the classic runner registry** — rejected: requires a `task_type` CHECK
  migration and forces a layered subject-scored eval into a single-sample runner
  abstraction.
- **Static-only scorer first** — rejected: defers the layered blend that motivated
  the feature and designs storage/UI twice.
- **Faithful PluginEval clone** (badges/Elo/portability scanning) — rejected as
  YAGNI; Claude-ecosystem-specific dimensions do not map to this repo's skills.
