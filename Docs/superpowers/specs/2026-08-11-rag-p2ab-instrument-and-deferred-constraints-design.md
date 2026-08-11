# RAG Server-Port Programme — P2ab: Instrument Renewal + P0-Deferred Constraints (design)

Date: 2026-08-11
Status: approved-pending-user-review
Programme: RAG server-port (P0 #1428; P1 #1458; fusion cluster #1469;
weighting #1487 — dev `ced98b9a4`)
Phase: P2, first arc of two (this arc: instrument renewal + the
P0-deferred constraints; the remaining arc: P2c measured feature
admission). Owner directive (2026-08-11): the deferred constraints land in
THIS arc, not a later one.

Two halves, strictly ordered inside one branch (the fusion-cluster
pattern — fixture before fix, one deliberate re-stamp at the end):

- **Half A — instrument renewal.** After the weighting arc, hybrid
  recall/MRR/NDCG sit at 1.000 on every scored query — the corpus can only
  detect regression. Fail-first fixtures (incl. the scoped category)
  restore its power to measure improvement.
- **Half B — the P0-deferred constraints, measured by Half A's fixtures:**
  1. **Scoped searches silently drop to semantic-only** — the engine's
     `metadata_allowlist` never reached the FTS legs (P0 constraint;
     `RAGService.search` RAISES for non-semantic search with an
     allowlist). Scope-aware hybrid retires that whole disclosure family.
  2. **Prompts have no engine seam** — the keyword leg covers
     media/notes/conversations only; prompts are invisible to hybrid
     entirely (and have no semantic index either). Post-weighting, a
     keyword-only prompts sub-leg is genuinely useful for the first time —
     rescue works now.
  3. **The Library surface runs a top_k=5 window** — half the harness's
     k=10 and tighter than everything measured; Console already honors the
     profile's `default_top_k` (P0), the Library canvas does not — a
     recorded asymmetry, resolved here.

## The core idea: fail-first fixture authoring

P1's fixtures were authored to SUCCEED in specific modes (keyword-exact for
FTS, paraphrase for vectors). P2a inverts the criterion: a candidate
fixture is ADMITTED only if today's pipeline FAILS it, measured, not
assumed. Failure today is the fixture's admission ticket, exactly inverse
to features (admitted only on measured improvement). A candidate class
that proves unfailable on this corpus+model is RECORDED as evidence — the
instrument narrowing P2c's feature list is a success outcome, not a
failure (precedent: the vocabulary-mismatch category collapsing killed the
expansion premise before anything was built).

Authoring lessons that bind (from P1/weighting, incident-backed):
- A corpus-unique rare identifier is a BEACON for the embedder (first
  vector-blind attempt came back semantic rank 1). What works is
  corpus-unique but semantically MISLEADING vocabulary, diluted.
- Synonym pairs collapse in MiniLM's space — "hard" must be proven by
  running, never by inspection.
- Overlap/integrity guards must stay canonical (the fixed-point stemmer);
  new categories extend the validator's vocabulary, never bypass it.

## Candidate classes (each measured; kept only where it fails)

1. **Compositional / multi-hop** — the answer doc satisfies a conjunction
   the query states but no single sentence contains ("the vendor that
   supplied the part that failed in the March incident"); distractors
   satisfy each conjunct separately. Expected failure mode: embedding
   similarity rewards partial matches; the true doc ranks below
   single-conjunct distractors.
2. **Negation-sensitive** — the query excludes what most of the corpus
   asserts ("configurations that do NOT use the default port"); the target
   doc is the exception; distractors are the assertions. Embeddings are
   famously negation-blind.
3. **Acronym-without-context** — the query uses a bare acronym whose
   expansion appears only spelled-out in the target (or vice versa), with
   no co-occurrence bridging them; distractors use the acronym letters in
   unrelated senses.
4. **Scoped** (the P2b measurement seed — REQUIRED even if classes 1-3
   partially fail to fail): scoped golden queries whose target is
   keyword-findable but vector-poor INSIDE the scope. Today, an active
   scope forces the semantic path (the P0 constraint P2b removes), so
   these queries fail BY ROUTING — the perfect before-number for P2b's
   scope-aware hybrid. Their failure is structural, so they are admitted
   by construction once the routing is verified live in the harness.
5. **Precision-pressure** (stretch, keep only if authorable): queries with
   MANY near-relevant distractors where today's top-10 admits ≥half junk —
   giving the currently gate-inert precision cells genuine discriminating
   power. Drop the class without ceremony if margin-clean authoring proves
   impractical.

## Components

### 1. Harness scope machinery (new; P1 deliberately cut it)

- Golden schema gains an optional `scope_slugs: list[str]` (fixture slugs
  the scope allows; validator: every slug exists, non-empty when present,
  ONLY the scoped category may carry it).
- The runner, for scoped queries, translates slug scope → the seam's real
  scope object (read `Chat/rag_scope.py`'s `EffectiveScope` and the
  Library seam's `scope=` parameter — build the same shape production
  builds, via the runtime's slug→source-id map). Scoped queries run ONLY
  through modes where scope is meaningful (hybrid/semantic; plain's
  four-seam path also accepts scope — verify and include if cheap).
- Report gains a `scoped` category column set; the runner records, per
  scoped query, WHICH route executed (the existing route-note/backends
  telemetry) so P2b's before/after shows the routing flip itself.

### 2. The authoring loop + admission protocol

- A small env-gated helper (`Tests/RAG_Eval/harness/fixture_probe.py` or a
  sweep-runner mode) that runs ONE candidate query through all three modes
  and prints ranks/scores — the authoring feedback tool. Rides existing
  machinery; no new gating.
- Admission protocol, recorded per candidate in the fixture files
  themselves (a `# admitted: <date> <mode ranks>` comment discipline like
  the vector-blind fixture's): authored → probed → admitted only if the
  target misses top-10 in EVERY vector-bearing mode (hybrid + semantic)
  while remaining findable in principle (keyword rank recorded);
  class-level outcome (admitted N / unfailable) recorded in the README.
- Golden-set integrity tests extend to the new categories (quotas set
  AFTER authoring reveals what is achievable — the spec deliberately does
  not pre-commit quotas for classes that may prove unfailable; the plan
  sets floors per class from the probe results, and the scoped class has a
  hard floor of ≥6).

### 3. Corpus scale-up

- Target ~150 docs total (from 49): the new classes' targets +
  substantially more distractors across all three source types, keeping
  every existing fixture byte-identical (existing baselines' meaning is
  anchored to them; additions only).
- All always-on guards (overlap stemmer, will-free, timeless, composition
  quotas) extend to the new docs; guard runtimes stay trivial.
- Ingestion-time budget: the gated build must stay under ~60s (P1 built 48
  docs in 6.3s; ~150 should be ~20s — verify, and if the model-embed step
  dominates, note the number; do NOT introduce caching complexity for
  this).

### 4. The re-stamp + honest bookkeeping

- Adding fixtures changes `corpus_sha256` → every gated run reads
  `environment_changed` until the ONE deliberate re-stamp that closes the
  sub-arc. The new baselines will show LOWER averages in the failing
  categories — that is the point; the README's at-ceiling warning is
  REPLACED by a per-category headroom table ("category X: hybrid recall
  0.42 — this is P2c's admission target").
- The rescue fixture, the vector-blind guard (AC#5 of 4110), and all
  existing per-category cells must hold their values through the corpus
  addition (new docs must not accidentally become better answers for old
  queries — the probe checks each old query's top-10 is UNCHANGED, or the
  new doc is reworded; this is the P1 "new doc as distractor" precision
  effect, controlled deliberately this time).

## Half B fix designs

### B1 — Scope-aware hybrid (allowlists reach the FTS legs)

- The engine's `metadata_allowlist` guard ("semantic only", raising for
  hybrid/keyword) is REMOVED for hybrid ONLY. (Code-verified at review:
  `build_semantic_allowlists` already returns a LIST of per-source-type
  dicts, each `{"source_type": {type}, "source_id": ids}` — deliberately
  one entry per type because a flat AND-ed dict cannot express the union.
  The B1 translation is therefore nearly direct: each entry feeds its
  matching sub-leg's ID filter.) Each FTS sub-leg adds a parameterized
  `id IN (...)` restriction from its entry's `source_id` set; a sub-leg
  with NO entry (source type absent from the scope) is SKIPPED, not
  unfiltered — fail-closed, matching the semantic side's semantics.
- **Keyword-mode + allowlist keeps raising** (explicit non-goal): the
  engine's `search_type="keyword"` + allowlist ValueError stays —
  unreachable from the Library (plain-profile scoped `rag` uses the
  four-seam Library path, which is already scope-aware). B1's guard
  removal is hybrid-scoped; widening it would be unmeasured scope creep.
- SQLite parameter-cap discipline (the fusion cluster's recorded concern):
  large allowlists chunk or use a json_each/temp-table form — read how the
  ORM's own id-filtered queries handle it (`search_notes`' id_filter uses
  `json_each` — mirror that) and match; never build an unbounded IN list.
- The Library's `_search_hybrid` stops routing scoped queries to semantic;
  the `ROUTE_NOTE_HYBRID_SCOPED` disclosure ("scope active — semantic only
  until scope-aware hybrid lands") RETIRES — this is that landing. The
  route-note vocabulary, the User Guide, and the P0-era comments update
  together (the stale-prose lesson: grep the retired copy's literal text).
- **TASK-14752 folds in here** (coverage copy mis-describing
  keyword-sourced evidence) — this fix touches exactly those disclosure
  seams; closing it separately would collide.
- Measured by: Half A's scoped fixtures flip from routing-failure to
  scored results; the scoped-route telemetry pin flips from
  "semantic-routed" to "hybrid-routed" as a DISCLOSED oracle update.

### B2 — Prompts keyword sub-leg

- A fourth read-only sub-leg over the Prompts DB, exactly the chacha
  pattern (private-sqlite seam owner registration + inventory row +
  ratchet bump; read-only URI; soft-delete filters replicated from the
  ORM's own prompt search). **Go/no-go RESOLVED at spec review: the
  Prompts DB HAS FTS5** — `prompts_fts(name, author, details,
  system_prompt, user_prompt)` with rowid = prompt id (Prompts_DB.py
  ~L270; `prompt_keywords_fts` also exists but is out of scope). The plan
  still reads the ORM's own search for the exact deleted-filter columns
  and rank expression before mirroring.
- Rows stamp `source_type: "prompt"` (singular — extend the fusion-key
  vocabulary pins), title, source_id. The fusion key, canonicalization,
  and Library post-filter maps gain the `prompt` vocabulary WITH
  cross-leg-merge pins (prompts have no vector leg, so every prompt row is
  FTS-only — the rescue path is their ONLY path; that is the point).
- The Library gate's `_FTS_SERVABLE_SOURCE_TYPES` gains prompts; the
  "no keyword leg for the selected sources" note narrows accordingly.
- Corpus/golden support: Half A authors the prompt fixture docs + golden
  queries (the harness's ingest gains a prompts writer via the real API).
  **Their before-state is total absence by construction** — pre-B2, prompt
  docs are invisible to EVERY mode (no vector index, no keyword sub-leg),
  so like the scoped category they are admitted structurally, not
  probe-admitted; the gated before-pin asserts recall 0 in all modes, and
  B2 flips it as a disclosed oracle update.

### B3 — Library window honors the profile

- **The DEFAULT only, never the user's control** (code-verified: the
  canvas carries a real top_k control, coerced via
  `_coerce_positive_int(top_k, LIBRARY_RAG_DEFAULT_TOP_K)` against
  `LIBRARY_RAG_TOP_K_MAX = 50`): an explicit user value keeps winning
  unchanged. What changes is what "unset/invalid" resolves to —
  `LIBRARY_RAG_DEFAULT_TOP_K = 5` is replaced by the profile's
  `default_top_k` resolution, mirroring the Console chip's P0 fix
  (`_console_library_rag_profile_top_k` precedent — one shared resolution
  seam if import-reachable without a cycle, else a twin with a coupling
  test); fallback to the literal 5 only when the profile is unresolvable.
- Measured by: the harness (k=10) and the surface stop disagreeing;
  the README's "Library window is tighter than these numbers" bound
  RETIRES; the User Guide's evidence-list description updates + stamp.
- UI sanity: the default profile's default_top_k is 15 — the Library
  evidence list must remain usable at 15 rows (scroll behavior exists;
  verify in the live check, not by redesign).

## Sequencing (load-bearing)

1. Half A first (scope machinery → fail-first authoring → scoped category
   with the semantic-routing before-pin).
2. B1 → B2 → B3 (B1 unlocks the scoped fixtures' scoring; B2 extends the
   leg B1 just made scope-aware; B3 is independent but its live check
   rides the same session).
3. ONE deliberate re-stamp at the end capturing: the new categories'
   honest (lower) numbers, the scoped category's post-B1 scores, prompts
   coverage, and the per-category headroom table that replaces the
   at-ceiling warning.

## Error handling

- Validator: `scope_slugs` on a non-scoped category is a defect; unknown
  slugs listed; empty scope rejected.
- The scoped runner path inherits every existing skip condition; a scope
  that resolves to zero runtime ids is a fixture defect (fail fast, named),
  never a silent empty search.
- B1: an allowlist naming zero ids for EVERY selected sub-leg degrades the
  keyword leg to [] (hybrid → semantic-with-allowlist, the existing
  disclosed path); mixed cases run the nonempty sub-legs.
- B2: Prompts DB missing/unopenable degrades that sub-leg with one
  warning (the chacha pattern).

## Out of scope (declared)

- P2c feature admission (PRF/expansion/HyDE/etc.) — next arc, measured by
  THIS arc's renewed instrument.
- Answer-layer fixtures (P3). Chunking changes (P4).
- TASK-3501 (fusion twin), 4111 (Library Open), 3500/3502 — independent.
- Semantic indexing of prompts (a P4-adjacent ingestion question; B2
  deliberately ships keyword-only prompt coverage and says so in the
  disclosures).

## Testing

- Always-on: extended integrity/guard tests; scope-schema validation;
  B1's allowlist-pushdown pins (per-sub-leg filtering, fail-closed empty
  sets, json_each/chunking discipline); B2's vocabulary + read-only +
  soft-delete + inventory pins (the chacha suite as the template); B3's
  resolution + fallback pins (the Console chip suite as the template);
  RED-first per the house pattern; mutation checks on the validator rules,
  the allowlist fail-closed direction, and the B3 resolution.
- Gated: the scoped-route telemetry pin (before: semantic-routed; after
  B1: hybrid-routed — a disclosed oracle flip); the full run + the ONE
  re-stamp; the old-queries-unchanged probe.
- Live TUI check (this arc DOES change user-facing behavior): a scoped
  hybrid search returning keyword-found in-scope evidence; a prompts hit
  surfacing in hybrid results; the Library list at the profile depth
  (15) remaining usable. Scratch profile, PR-2 recipe.
- Collection arithmetic; final whole-branch review; the private-sqlite
  inventory suite in EVERY task battery (B2 adds a connection owner — the
  guarded-KIND lesson).

## Plan-phase verification items

1. The seam's `scope=` parameter shape (`EffectiveScope` construction from
   runtime ids — read rag_scope.py + the Library service's scoped tests
   for the canonical test-side construction) AND the allowlist shape
   `build_semantic_allowlists` produces (key vocabulary, id types) — B1's
   translation consumes it.
2. Whether plain mode's four-seam path accepts the same scope object
   (include scoped-plain cells if free).
3. Ingestion wall-time at ~150 docs (measure early; adjust target rather
   than adding complexity).
4. The old-queries-unchanged probe's cheapest honest form.
5. Golden schema versioning (goldenset.py's loader tolerance for unknown
   keys).
6. Prompts_DB.py's FTS reality: table name, rank expression, soft-delete
   columns, whether FTS exists at all (B2's go/no-go), and the Prompts
   DB path resolver in config.py.
7. Where `LIBRARY_RAG_DEFAULT_TOP_K` is consumed (all sites) and whether
   the Console chip's resolution seam is import-reachable from the
   Library screen without a cycle.
8. The engine's allowlist guard sites (the ValueError for non-semantic +
   any Library-side pre-guards) — B1 must remove ALL of them coherently,
   including the cache-key implications (the allowlist is already in the
   cache key; verify hybrid+allowlist keys compose with the fusion-param
   and selection key parts).
