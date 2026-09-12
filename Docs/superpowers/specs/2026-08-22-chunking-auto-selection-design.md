# Chunking Auto-Selection — Design Spec

**Date:** 2026-08-22
**Status:** Draft, maintainer-approved in brainstorming (five decisions in §8),
self-reviewed; awaiting maintainer's review gate.
**Sub-project:** 3 of 6 in the Chunking Parity & Agent Tools program
**Depends on:** sub-project #1 (PR #1852, merged — vendored engine, ADR-073) and
sub-project #2 (PR #1938, open at spec time — template store v7, picker,
resolution chain, re-chunk). **Branching:** stacked on
`feat/chunking-template-parity` until #1938 merges, then rebase to `dev`.
**Author:** brainstormed with the maintainer. Upstream facts verified against
the same pin #1/#2 vendored from (`dev` @ `385afa951922c8a9dc2002c675bb6cad65e4ac23`,
read via the local `tldw_server2` checkout); chatbook-side facts verified
against the #2 branch.

---

## 0. Provenance & upstream pin

Same pin as #1/#2; **not moved**. Two upstream mechanisms are in play:

- **`auto_planner.py`** (~350 lines, `excluded` in `VENDOR_MANIFEST.toml` since #1):
  pure decision logic — `AutoChunkingProfile/Request/Plan/Decision` dataclasses,
  `plan_auto_chunking(perform_chunking, chunking_mode, media_type, goal, profile,
  template_name/status/error, requested_llm, llm_available, semantic_available) ->
  AutoChunkingDecision`. Imports are stdlib-only (`re`, `dataclasses`, `typing`,
  `__future__`) — vendoring expects **zero new shims** (verified against the pin's
  import block; re-verify at execution). LLM availability is a **caller
  contract**, so #3 does not wait for #6's boundary assistant.
- **`TemplateClassifier.score()`** (already vendored in `engine/templates.py`,
  fenced since #2's PR A): static, store-free — scores ONE template dict
  (`media_types` 0.5 weight; `filename_regex`/`title_regex`/`url_regex` hits,
  0.5 weight ÷ 3) and applies its own min_score clamp.

### 0.1 Correction to a brainstorm ruling — the classifier threshold

During brainstorming I described `score()` as having "no threshold of its own";
that was **wrong**. The vendored source clamps internally:

```python
min_score = float(classifier.get('min_score', 0.0))
return score if score >= min_score else 0.0
```

Two consequences, both folded into §4.2:

1. **No classifier block at all → score is 0.0** (empty dict falls through the
   media/regex scoring). With the caller-side `score > 0` guard (§4.2), such
   templates are never auto-selected — the six #2 built-ins included. This
   preserves the *intent* of the approved opt-in ruling.
2. **A present block with absent `min_score` selects at any positive score**
   (clamp defaults 0.0). The approved ruling's literal text ("absent min_score →
   never auto-selected") was based on my mis-description; the corrected behavior
   matches upstream parity, which is the program's premise. Recorded here as a
   correction, the way #2's spec recorded its §0.1 premise corrections.

---

## 0.2 The upstream auto-apply orchestrator — and where chatbook deliberately diverges

Found during the approved-spec review (upstream
`Ingestion_Media_Processing/chunking_options.py:520-660`, the shared
`apply_*_template` helper used by every process-* endpoint):

- **Upstream's selection loop matches §4.2's design**: iterate live candidates,
  `score <= 0 → continue`, best key `(score, priority)` strictly-greater — ties
  keep the first-listed candidate (#2's listing is name-ordered, so
  priority-then-name is the effective equivalent; the coupling is stated here
  so re-ordering the listing doesn't silently change selection).
- **Upstream applies only hierarchical blocks.** After picking a winner (or an
  explicit template by name), upstream extracts **only**
  `chunking.config.hierarchical_template` + `chunking.method` +
  `chunking.config.{max_size, overlap}` into the options dict; a winning
  template with no hierarchical block is **silently ignored**, and
  preprocessing/postprocessing stages never run on either path.
- **Upstream's auto is a separate form flag** (`auto_apply_template: bool`),
  not a sentinel in the template-name slot.

**Divergences chatbook takes deliberately (ruling §8.6):**

1. **A winning template runs in full** — preprocessing, chunking,
   postprocessing — through #2's template engine, exactly as a manually-picked
   template does. Upstream's extraction-only behavior is an artifact of its
   options-dict plumbing; chatbook's #2 exists to run all three stages.
   Auto-selected and manually-selected templates are indistinguishable in
   effect. Filed as `UPSTREAM_DEFECTS.md` #16 (auto/explicit paths silently
   drop template stages).
2. **Auto rides the picker's `chunk_template` slot with the reserved name
   `"auto"`** (§4.3), rather than upstream's extra form flag — chatbook's
   single picker slot is the established surface. The name is **reserved**:
   #2's validator refuses `name == "auto"` at create/rename (ruling §8.7), so
   no user template can be shadowed by the sentinel.
3. **Tier composition is chatbook's** (template tier suppresses the planner —
   §8.2). Upstream runs the planner and the template-apply helper as separate
   mechanisms (the planner takes template name/status purely to emit fallback
   *reasons*). Chatbook's chain is a composition of the two upstream pieces,
   not a mirror of either call site; the composition is the spec's subject.

---

## 1. Why

#2 made templates *real*: stored, validated, pickable, honored on ingest and
re-chunk. But selection is entirely manual — a user must know which template
fits which media. The program's #3 clause: "the `classifier` block plus
`auto_planner` scoring, so a media item gets the right template automatically."

Today's gaps, all verified on the #2 branch:

- The `classifier` block is **stored and validated but inert** — #2's non-goals
  reserved exactly this for #3.
- `TemplateClassifier` is vendored-but-fenced (zero consumers).
- `auto_planner.py` is not vendored at all.
- The picker offers "None (manual settings)" only; the resolution chain
  (picker → config default → plain) has no automatic tier.

## 2. Goals

1. One "Auto" selection path that picks a template when one opts in and fits,
   else derives a method plan, else falls to today's plain defaults — always
   terminating, always explainable.
2. Parity with the upstream planner's decisions (fixture-provable, like #2's
   goldens).
3. The decision recorded per media item, and re-resolved (not replayed) on
   re-chunk.
4. Zero behavior change on upgrade: "None" stays the picker default; Auto is
   opt-in per ingest.

## 3. Non-goals

- **Any LLM boundary work** — `auto_boundary_assistant`, adapter availability
  probing: #6. `llm_available` is passed `False` (§4.2).
- **A goal control or config key** — hardcoded `"balanced"` (ruling §8.4).
- **Authoring classifier blocks** — CRUD ships in #2's service layer; a UI for
  editing classifier blocks is not built here (templates gain classifier blocks
  through the existing service-layer authoring path).
- **Making Auto the default anywhere.**
- **Moving the pin.**

## 4. Design

### 4.1 Vendoring

`auto_planner.py` moves `excluded` → `vendored` (37 files; the manifest-move
pattern from #2's PR A; sync-script list updated in the same commit; the
not-in-both disjointness test already guards it). `TemplateClassifier` stays
where it is — #3 becomes its **first consumer**, and the PR-A fencing test
changes from "no production module constructs the fenced classes" to "no
production module constructs them outside `Chunking/auto_selection.py`"
(`TemplateLearner` stays fully fenced; `TemplateManager` untouched).

### 4.2 The selection engine — `Chunking/auto_selection.py`

One new module, the only place auto-selection is decided:

```python
resolve_auto(db, *, media_type: str | None, title: str | None,
             filename: str | None, url: str | None,
             goal: str = "balanced") -> AutoDecision
```

`AutoDecision` is a dataclass: `{tier: "template"|"plan"|"plain", template:
dict | None, chunk_options: dict | None, rationale: list[str],
fallback_reasons: list[str]}`. The tiers, in order:

1. **Template tier.** Score every live (non-deleted) template with
   `TemplateClassifier.score(template_json, media_type=…, title=…, url=…,
   filename=…)`. **Stored-invalid templates are excluded from candidacy**
   before scoring (their bodies could win and then fail the item at #2's
   apply-refusal; the AC-24a validity flag from `_decorate_template_record`
   is the exclusion signal). The winner is the highest score **strictly
   greater than 0** (§0.1 correction 1 — and upstream's own `score <= 0:
   continue`); ties break by the classifier block's `priority` (higher wins;
   absent → 0), then by listing order — which #2 makes name-ordered, so
   effectively priority-then-name (§0.2 notes the coupling). Each template's
   evaluation is individually guarded — one malformed block is skipped with a
   fallback reason, never poisons the loop (regex safety itself is already
   enforced at write time by #2's validator). On a winner: `tier="template"`,
   the resolved template dict rides, and **auto_planner does not run**
   (ruling §8.2: a selected template's chunk-stage config *is* the plan). A
   winning template is applied **in full** — all three stages through #2's
   engine, indistinguishable from a manual pick (ruling §8.6).
2. **Plan tier.** The vendored `plan_auto_chunking(...)` with
   `chunking_mode="auto"`, `perform_chunking=True`, `goal` (hardcoded
   `"balanced"`), `media_type`, `requested_llm=False`, `llm_available=False`
   (#6 contract), `semantic_available` from the embeddings config's enabled
   state, and the template-status args left None (no template was consulted).
   Its `chunk_options` and the plan's rationale/fallback metadata ride. The
   plan's options follow **#2's precedence rule unchanged**: they beat the
   ingest builder's defaults, and lose only to values the user explicitly
   changed in the form (same treatment as a template's chunk-stage config).
3. **Plain tier.** `chunk_options=None` — the caller keeps today's defaults.
   Auto cannot *fail*; it can only explain why it declined.

### 4.3 Seam wiring

- **Picker:** a new option "Auto" beside "None (manual settings)" (None stays
  the default — ruling §8.3/Option A). It travels the existing
  `chunk_template` slot with the **reserved** name `"auto"` (§0.2 divergence
  2): #2's create/rename validation refuses a template named `"auto"`, so no
  user template can be shadowed by the sentinel.
- **Resolution:** `template_runtime.resolve_ingest_template` detects the
  sentinel at its picker tier and calls `resolve_auto`, returning whichever
  tier won. Everything downstream (#2's precedence machinery, materialization,
  the six seams) is unchanged — auto slots in *above* the config-default tier
  only when the user chose it, and config-default is only consulted when they
  didn't (chain ruling §8.1):
  `user choice (template name | "auto") → [if auto: classifier → planner → plain]
  → config default_template → plain options`.
- **Re-chunk:** `Media.chunking_config` records the *choice* (`mode: "auto"`)
  plus the decision's tier and rationale. Re-chunk re-resolves: a stored
  `mode: "auto"` runs `resolve_auto` again (change a classifier block → the
  tier can change); a stored template name behaves exactly as in #2. The
  `template` key is written **only when the template tier won**, so both of
  #2's readers (`LIKE '%"template": "<name>"%'` and `json_extract($.template)`)
  stay satisfied and template-tier selections remain discoverable by the
  existing statistics queries.

### 4.4 Persistence shape

`chunking_config` gains: `mode` (`"auto"` | `"template"` | absent — #2 rows
untouched), `auto_tier`, `auto_rationale` (short list), and `template` only on
a template-tier win. No schema change — it is all inside the existing JSON
column. `chunking_template`/`chunking_params` columns carry the winning
template's name/params as in #2.

## 5. Errors and edge cases

- Auto always terminates at plain — no unresolvable-template error path exists
  for the Auto choice itself (#2's named failure applies only to explicit
  template names).
- Empty template store → tier 1 is vacuous (zero candidates, one fallback
  reason) → tier 2.
- A classifier block whose regexes were valid at write time but whose *inputs*
  are pathological at score time → guarded per-template (§4.2).
- Server-mode ingest: the picker is hidden in server mode (#2); the config
  `default_template` tier is unaffected and **never** triggers auto — auto is
  exclusively the picker sentinel.
- The `media_type`/`title`/`filename`/`url` inputs come from the ingest job's
  already-known metadata; nothing re-reads file contents at selection time.
  **Vocabulary alignment is an explicit implementation item** (§6.9): the
  planner normalizes `web_document/webpage/article/html → "web"` and switches
  its method choice on the normalized value; chatbook's ingest media-type
  strings must be mapped onto the planner's expectations (or the mapping's
  absence documented as intentional) before the tier-2 fixtures are frozen —
  a mismatch means per-type plans silently never fire.

## 6. Testing

1. **Planner parity fixtures** (the #2-goldens pattern): fixed inputs →
   `plan_auto_chunking` outputs, generated from the vendored module with test
   mode off, byte-pinned, re-run at every sync.
2. **Classifier scoring table:** media-match/regex-match/both/neither ×
   min_score boundary cases, incl. §0.1's two consequences as explicit pins
   (no-block → 0.0/not selected; block-without-min_score → positive-score
   selection).
3. **Tier orchestration:** integration tests per tier outcome; mutation-verified
   pin that the planner never runs when the template tier wins; tie-break
   (priority, then name) pins.
4. **The six built-ins are never auto-selected** (no classifier blocks — the
   opt-in proof, as a standing test).
5. **Chain integration:** picker-Auto → classifier-win → template honored on
   the pdf seam (persisted rows differ from plan-tier); picker-Auto → no
   candidate → plan-tier chunk_options materialize; picker-None byte-identical
   to today (re-assert #2's AC-36 pin still green).
6. **Re-chunk re-resolution:** stored `mode:"auto"` re-resolves; flipping a
   classifier block changes the tier on re-chunk.
7. **UI:** the picker's Auto option (run_test, stubbed store); None still
   default.
8. Suites: `Tests/Chunking/`, `Tests/Local_Ingestion/`, targeted
   `Tests/UI/test_library_ingest_*`, `Tests/RAG_Admin/`, import-weight guard.
9. **Media-type vocabulary check** (§5): a build-time test asserting the
   mapping from chatbook's ingest media-type strings to the planner's
   normalized vocabulary — every string chatbook can send appears in the
   mapping table (explicit `identity` entries count), so a vocabulary drift
   on either side fails loudly instead of silently disabling tier-2 plans.

## 7. Acceptance criteria

- [ ] #1 `auto_planner.py` vendored from the existing pin (manifest move,
      36→37 files, byte-faithful modulo rewrite rules, zero new shims) and the
      sync contract tests updated
- [ ] #2 Exactly one module decides auto-selection
      (`Chunking/auto_selection.py`); `TemplateClassifier`'s fencing test now
      permits construction only there; `TemplateLearner`/`TemplateManager`
      remain fully fenced
- [ ] #3 Tier 1 selects only strictly-positive scores, tie-broken by
      `priority` (absent → 0) then listing order; **stored-invalid templates
      are excluded from candidacy**; a malformed block is skipped with a
      reason, never fatal; the six #2 built-ins are provably never
      auto-selected
- [ ] #4 §0.1's corrected threshold semantics are pinned: no-block → never;
      block-with-absent-min_score → positive score selects
- [ ] #5 Tier 2 uses the vendored planner with `llm_available=False` and
      chatbook-real `semantic_available`; tier 2 never runs when tier 1 won
      (mutation-verified)
- [ ] #6 Planner parity fixtures byte-match the vendored planner, generated
      and verified with test mode disabled
- [ ] #7 The picker offers "Auto"; "None (manual settings)" remains the
      default and its output stays byte-identical to today
- [ ] #8 Auto reaches all six ingest seams via the existing `chunk_template`
      slot with sentinel `"auto"`; no seam-specific branching added
- [ ] #9 `chunking_config` records `mode`/`auto_tier`/rationale; the
      `template` key appears only on template-tier wins; both #2 readers
      round-trip; no schema change
- [ ] #10 Re-chunk re-resolves a stored `mode:"auto"` (classifier change flips
       the tier); stored explicit template names behave exactly as #2
- [ ] #11 Config `default_template` never triggers auto; server mode is
       unaffected
- [ ] #12 Docs: user guide gains the Auto option (with the "opt-in via
       classifier block" explanation) with a re-verified stamp; CHANGELOG entry
- [ ] #13 Targeted suites green (§6.8); import-weight guard green; no new
       core dependencies
- [ ] #14 The name `"auto"` is reserved: create/rename refuses it with a
       named error, and the reservation is pinned by a test; a
       pre-existing template named `"auto"` (created before #3) is flagged by
       the migration-less guard as shadowed and never auto-selected nor
       auto-shadowed (explicitly documented behavior)
- [ ] #15 The media-type mapping table exists, covers every ingest media-type
       string, and is enforced by the §6.9 vocabulary test
- [ ] #16 UPSTREAM_DEFECTS.md gains entry #16 (auto/explicit template paths
       apply only the hierarchical block and silently drop all other stages)

## 8. Decisions taken (brainstorm, 2026-08-22)

1. **One sub-project** covers both mechanisms (`auto_planner` +
   `TemplateClassifier`) — the user-facing surface is one "Auto" choice.
2. **Resolution chain:** user choice → classifier template (strictly-positive
   score) → planner plan → plain; a selected template **is** the plan (no
   double run); the decision is recorded per-media and re-resolved on re-chunk.
3. **"Auto" is an additional picker option; "None" stays default** (Option A) —
   zero upgrade behavior change.
4. **Goal hardcoded `"balanced"`** — no UI, no config key (YAGNI).
5. *(Corrected per §0.1)* — opt-in auto-selection is enforced by the
   caller-side `score > 0` guard (no-block templates return 0.0), not by an
   absent-min_score veto; threshold behavior inside a present block follows
   the upstream clamp for parity.

**Added during the approved-spec review (2026-08-22), findings 1-3:**

6. **A winning template runs in full** (all three stages via #2's engine),
   deliberately diverging from upstream's hierarchical-block-only extraction
   (§0.2); the upstream wart is filed as UPSTREAM_DEFECTS.md #16.
7. **The name `"auto"` is reserved** at create/rename — the picker sentinel
   rides the existing slot without shadowing any user template (upstream
   avoids the collision with a separate flag; chatbook's single-slot surface
   makes the reservation the smaller change).
8. **Stored-invalid templates are not auto-selection candidates** — auto must
   never pick a body that #2's apply path would then refuse.
