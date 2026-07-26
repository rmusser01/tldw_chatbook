# Evals Console-Style Rebuild Design

Date: 2026-07-25
Status: Draft (pending spec review)
Related: [Master shell design system contract](../../Design/master-shell-design-system-contract.md),
[Chatbook workbench UI system](../../Design/chatbook-workbench-ui-system.md)

A sibling Console-style rebuild exists for Watchlists
(`Docs/superpowers/specs/2026-07-25-watchlists-console-rebuild-design.md`), but it lives on the
unmerged `docs/watchlists-console-rebuild-spec` branch and is deliberately not linked here — the
link would not resolve on `dev`.

## Summary

Rebuild the Evals screen as a Console-styled three-pane workbench on the `$ds-*` token set,
replacing the emoji card hub and its half-implemented navigation. The screen is generic over a
**bench type**; **word bench** is the one type wired end-to-end in this slice.

A word bench measures the next-token distribution a model assigns after each of a set of text
snippets, across a set of targets, and renders the result as a pivotable grid. Its purpose is to
make censorship and directional prompting *visible as a number*: how far a loaded phrasing moves a
model relative to a neutral one, and how far a safety system prompt moves a model relative to its
base.

The evaluation backend (`Evals/`, `Evaluations_Interop/`, `Evals_DB`) does not move. One column is
added to one table.

## Goals

- Rebuild the screen at Console information density using the shared destination-workbench pattern.
- Give a user four working paths: run an existing bench, author or import a snippet set, configure
  a bench, and browse historical runs and their results.
- Ship word bench end-to-end, with capture, analysis, storage, and a results grid.
- Keep every pre-existing `eval_tasks` and `eval_run` reachable and readable. Readable, not runnable
  — see [Classic tasks](#classic-tasks).
- Retire the unreachable Evals UI, which is larger than the reachable one.

## Non-goals

- **Running** classic orchestrator task types (`question_answer`, `generation`, `classification`)
  from the new screen. They list and their results open read-only; launching one is deferred. See
  [Classic tasks](#classic-tasks).
- A/B testing. `ab_tests` and `ab_test_runs` stay in the schema, unused by this screen.
- Server-scope evaluations. `EvaluationScopeService` remains wired in `app.py`; this slice targets
  the local backend and does not add a scope switcher.
- HuggingFace dataset import. `eval_datasets.format` permits it; nothing in this slice uses it.
- Multi-position logits. A word bench measures exactly one token position per snippet.

## Current state

Verified against `origin/dev`. This is what the rebuild replaces.

### The reachable screen

`EvalsScreen` (`UI/Screens/evals_screen.py`) composes a `DestinationHeader`, a `LabModeStrip`, and
`EvalsWindowV3`. That container renders a centered grid of six emoji cards. Three are wired
(`quick_test`, `tasks`, `results`); `comparison`, `batch_eval`, and `models` call
`notify("not yet implemented")`.

`EvalsWindowV3` mounts Textual `Screen` objects **inside a `Container`** and manages them with a
hand-rolled `screen_stack` list (`evals_window_v3.py:70-120`). This is why `EvalsScreen` carries a
custom `action_evals_back` and why the hub binds bare digits `1`-`6`.

Visual language predates the design system: widget-local `DEFAULT_CSS` against `$panel` and
`$primary`, emoji card faces, centered grid. No `.ds-*` classes, no `$ds-*` tokens.
`css/features/_evaluation_unified.tcss` is a separate 288-line legacy sheet.

### The unreachable screens

A complete second-generation Evals UI is imported by nothing. `UI/ResultsDashboardWindow.py`,
`UI/ModelManagementWindow.py`, `UI/DatasetManagementWindow.py`, `UI/Views/evals_views.py`, and
`Event_Handlers/eval_events.py` reference only each other, plus twelve of the fourteen files in
`Widgets/Evals/` that no reachable code imports. The remaining two — `eval_dialogs.py` and
`sample_browser_dialog.py` — are imported by tests only.

### The backend

Real and complete enough to build on. `Evals_DB` (`SCHEMA_VERSION = 3`, `Evals_DB.py:39`) holds
`eval_tasks`, `eval_datasets`, `eval_models`, `eval_runs`, `eval_results`, `eval_run_metrics`,
`ab_tests`, `ab_test_runs`. `EvaluationOrchestrator` runs tasks. `Evaluations_Interop/` normalizes
local and server records and is wired by `_wire_evaluation_services` (`app.py:4130`, called from
`app.py:3407`).

All line references in this document are against `origin/dev` at `8242a5b58`. The `app.py` and
`skills_screen.py` numbers in particular drift quickly; re-resolve by symbol rather than by line if
they do not match.

### Why word bench cannot reuse the existing logprob path

`LogProbRunner._get_text_logprob` (`eval_runner.py:2128-2171`) is a stub. It returns
`float("-inf")` for every provider, including OpenAI, after logging a warning. The runner factory
dispatches `task_type == "logprob"` to it unconditionally (`eval_runner.py:2377`).

Real logprob plumbing exists only on the local OpenAI-compatible path:
`LLM_API_Calls_Local.py:201-208` sends `logprobs` and `top_logprobs`, and `:379` returns the raw
response dict, so the distribution survives. `LLM_API_Calls.py` (commercial providers) has no
logprobs parameter at all.

## Screen IA and layout

`EvalsScreen` keeps its seat, its `DestinationHeader`, and its `LabModeStrip`. **The mode-strip slot
is already occupied** by the Lab strip (Models | Speech | Evals), so Evals-internal navigation must
not be a second strip.

Below the strip, the house three-pane workbench established by `skills_screen.py:764-1064`:

```
Vertical#evals-shell
  Static.ds-destination-header#evals-destination-header
  LabModeStrip#lab-mode-strip
  Horizontal#evals-workbench .ds-panel .destination-workbench
    Vertical#evals-library-pane   .destination-workbench-pane
    Vertical#evals-detail-pane    .destination-workbench-pane
    Vertical#evals-inspector-pane .destination-workbench-pane
```

```
┌ Evals ───────────────────────────────────────────────────── [ Run bench ] ┐
│ Run and review evaluation jobs.          local · 4 benches · 1 run active │
├ Modes:  Models   Speech   Evals ──────────────────────────────────────────┤
├────────────────┬──────────────────────────────────────┬───────────────────┤
│ ▾ BENCHES    4 │  loaded-nouns v1          word bench │ READINESS         │
│   loaded-nouns │  ───────────────────────────────────  │ llama-3-8b  Ready │
│   refusal-open │  Dataset   nouns-12         12 snip. │ qwen-2.5-7b Ready │
│   ─ classic ─  │  Mode      raw continuation          │ llama+prefix Ready│
│   mmlu-subset  │  Top-K     20                        │                   │
│   summarize    │  Probes    " Sure" " I" " Sorry"     │ ESTIMATE          │
│                │                                      │ 36 calls  ~00:14  │
│ ▾ DATASETS   3 │  TARGETS                        3    │ local · no cost   │
│   nouns-12  12 │   llama-3-8b     raw   Ready         │                   │
│   openers-8  8 │   qwen-2.5-7b    raw   Ready         │ [ Run bench     ] │
│   probes-mix 6 │   llama-3-8b +prefix   Ready         │ [ Duplicate     ] │
│                │   [ + Add target ]                   │ [ Delete        ] │
│ ▾ RUNS      14 │                                      │                   │
│   ● 14:31 run  │                                      │                   │
│   ✓ 14:02 run  │                                      │                   │
│   ✗ 13:55 run  │                                      │                   │
└────────────────┴──────────────────────────────────────┴───────────────────┘
```

**Library pane.** Three collapsible sections — `Benches`, `Datasets`, `Runs` — each with a live
count. Selection drives the centre. Classic orchestrator tasks appear in a labelled subgroup under
Benches. The `Runs` section lists **run groups, not runs**: one grid is one entry, never one entry
per target. It groups by bench and caps at a recent window with an explicit "show all".

Each of `Benches` and `Datasets` carries its own creation affordance in the section header — a new
bench and a new snippet set are reachable without first finding an empty state.

**Detail pane.** Swaps on selection kind: word bench → bench editor, classic task → read-only detail
plus run history, dataset → snippet editor, run group → results grid.

**Inspector pane.** Also swaps: bench → per-target readiness, estimate, and the run action; dataset
→ snippet-set statistics; run group → run metadata and export; **focused grid cell → that cell's
full top-K and probe table**, updating as focus moves. No modal.

**The primary action names its object.** The header action reads `Run loaded-nouns v1` when a bench
is selected, and is disabled with a stated reason otherwise. A bare "Run bench" against an ambiguous
selection is how the current screen produced dead-end toasts.

**Escape.** Selection state replaces the hand-rolled screen stack, so the shell's own Escape
handling applies. `EvalsScreen.BINDINGS` loses `escape` and `1`-`6`.

**Width.** The rail and inspector collapse; the grid pages or scrolls its target columns. Six
targets do not fit three panes at 100 columns and the design does not pretend otherwise. Both
`.density-compact` and `.density-comfortable` are supported without separate widget
implementations, per the design contract.

## Data model

Everything maps onto existing tables. One migration, one new column.

| Concept | Storage | Notes |
|---|---|---|
| **Bench** | `eval_tasks` row, `task_type='logprob'` | `config_data` carries `bench_type:"word_bench"`, `prompt_mode`, `top_k`, `probes[]`, `target_ids[]` |
| **Snippet set** | `eval_datasets` row, inline samples | Existing `inline:<name>` convention |
| **Target** | `eval_models` row | A target *is* a model row; system prompt or prefix lives in `config` |
| **Run group** | N `eval_runs` sharing `run_group_id` | One run per target |
| **Cell** | `eval_results` row | `run_id` (target) × `sample_id` (snippet) |

The grid is a pivot of `eval_results` over a run group, not a new structure.

### Why `task_type='logprob'`

`eval_tasks.task_type` has a `CHECK` constraint permitting only
`question_answer | logprob | generation | classification` (`Evals_DB.py:152`). SQLite cannot alter a
`CHECK` without rebuilding the table. Registering word bench as `logprob` with a `bench_type`
discriminator in `config_data` avoids that rebuild, and is semantically honest — a word bench is a
logprob task.

**This collides with the runner factory.** `_create_basic_runner` routes `logprob` to the broken
`LogProbRunner`. Word bench therefore does **not** execute through `EvalRunner` at all; see
[Capture plumbing](#capture-plumbing). Any future generic dispatch must read `config_data.bench_type`
before `task_type`.

### Inline datasets need no schema change

`LocalEvaluationsService.create_dataset` (`local_evaluations_service.py:387-415`) already stores
authored samples in `metadata[RESERVED_LOCAL_DATASET_SAMPLES_KEY]` with `source_path = "inline:<name>"`
and `sample_count`. Snippet authoring reuses this verbatim.

### Targets are model rows

A target is one `eval_models` row. `llama-3-8b` and `llama-3-8b + safety prompt` are two rows —
`UNIQUE(name, provider, model_id)` permits it because the names differ. This makes "same model,
different steering" a first-class grid column with zero schema work, and that comparison is the one
that exposes directional prompting.

Note that `_create_model_config` splats `**model_data["config"]` into the model config dict
(`eval_orchestrator.py:262-269`). The word bench capture client reads the steering field explicitly
rather than relying on that splat.

**Chat mode and raw mode use different steering mechanisms, with different field names.** In chat
mode a target carries `system_prompt`, delivered as a system message. In raw mode a target carries
`prefix`, literal text prepended to the snippet. Raw completions have no system-message slot, and
silently concatenating a "system prompt" onto the snippet would change what is measured while
claiming not to. A target is valid in exactly one mode, and the bench editor enforces it.

### Schema migration: v3 → v4

```sql
ALTER TABLE eval_runs ADD COLUMN run_group_id TEXT;
CREATE INDEX IF NOT EXISTS idx_eval_runs_group ON eval_runs (run_group_id);
```

Existing rows keep `NULL`, which reads as a single-run group. `SCHEMA_VERSION` goes to 4 with a
migration branch alongside the existing v1→v2 and v2→v3 branches.

### Run provenance

`eval_runs` references a mutable `eval_tasks` row. Editing a bench would otherwise silently
reinterpret every historical grid. At launch, the fully-resolved configuration is snapshotted into
`eval_runs.config_overrides`:

- prompt mode, top-K, probe list
- sampler parameters as sent
- per-target provider, model, and steering text
- **snippet IDs and their full text**, plus a text hash per snippet

Results render from the snapshot, never from the live task. Snapshotting the text — not only the
hash — means a grid still renders after its dataset is edited or deleted; the hash then serves its
real purpose, flagging *"this snippet was edited after the run."*

Snippets carry a UUID assigned at authoring time. Positional identifiers would silently remap old
results onto the wrong snippets when a dataset is reordered, and `UNIQUE(run_id, sample_id)` would
not catch it.

`config_data.target_ids[]` is JSON with no foreign key. A deleted `eval_models` row leaves a
dangling reference; the bench editor resolves targets at load and marks unresolvable ones.

## What a word bench measures

For each `(snippet, target)` cell: the model's next-token distribution at exactly one position,
captured as the top-K tokens with their log probabilities.

### Prompt mode

Declared per bench.

| Mode | Request | Measured token | Targets |
|---|---|---|---|
| `raw` | `POST /v1/completions`, snippet as literal prompt | the literal next token | local servers only |
| `chat` | `POST /v1/chat/completions`, snippet as user message | first token of the reply | local + OpenAI |

Both modes send `max_tokens=1`.

Raw mode is the textbook reading of "logits for the next token to each snippet" and is right for
directional-prompting work on mid-sentence fragments. Chat mode measures how a model *opens* a
reply, which is where refusal behaviour lives. The two phenomena named in the brief occur at
different measurement points, so the bench supports both rather than forcing one.

OpenAI retired raw completions except `gpt-3.5-turbo-instruct`, and legacy completions cap
`logprobs` at 5 rather than 20. Preflight reports the K actually returned, so no special case is
needed.

### Sampler neutrality

**Many servers apply sampling parameters before reporting logprobs.** llama.cpp in particular will
report a post-sampler distribution. A bench run against a server configured with `top_k=40` would
produce entropy, divergence, and probe numbers that are artifacts of that setting, with nothing in
the UI to say so.

Every request therefore sends explicit neutral sampling:

```
temperature = 1.0    top_p = 1.0    top_k = 0    min_p = 0.0
presence_penalty = 0    frequency_penalty = 0    repeat_penalty = 1.0
```

`temperature = 1.0`, not `0` — temperature zero collapses the distribution being observed.

This can be requested but not guaranteed: some servers clamp or ignore these. The run snapshot
records the parameters **as sent**; where a server echoes its effective parameters, those are
recorded too and a mismatch is flagged. Where neither is available this is a stated limitation, not
something the design asserts away.

### Top-K and probes

**K must be chosen generously up front. It can never be raised retroactively** — recovering a
rank-25 token from a K=20 run requires re-running the grid. The bench editor states this where K is
set. Default 20 (OpenAI's cap); local targets may go higher.

**Probes cannot be requested.** No API permits asking for a named token's logprob. Probes are read
out of the returned top-K, which makes them a purely **read-side** operation. Consequences:

- Cells store top-K only. Probe readings are computed at render time from the snapshot's probe list.
- **The probe list can be edited after a run and the grid re-reads instantly**, with no re-execution.
- A probe has three states, and conflating them would mislead:

| State | Meaning | Rendered |
|---|---|---|
| observed | present in top-K | `-4.71   0.9%` |
| bounded | absent from this cell's top-K | `< -6.90` |
| never observed | absent from top-K in *every* cell for this target across the whole run | `never observed` |

The third state exists because **probe tokens are not comparable across models**. `" Sure"` may be
one token in Llama's vocabulary and two in Qwen's. A probe that never once appears for a target is
most likely unrepresentable in that target's vocabulary rather than merely unlikely, and rendering
it as a bound would invite a cross-column comparison that means nothing. Probe comparison is
within-column by default; crossing columns carries a visible caveat.

Token identity uses the `bytes` field where the provider returns it. String comparison across
tokenizers with differing escaping is fragile, and byte-fallback tokens (`<0xE2>`) cannot be
rendered or matched correctly without it.

### Cell payload

Stored in the existing `eval_results.logprobs` TEXT column:

```json
{
  "schema": "word_bench/1",
  "prompt_mode": "raw",
  "k_requested": 20,
  "k_returned": 20,
  "top_k": [
    {"token": " a",   "logprob": -0.82, "bytes": [32, 97]},
    {"token": " the", "logprob": -1.51, "bytes": [32, 116, 104, 101]}
  ],
  "entropy": 2.71,
  "top1_mass": 0.44,
  "truncated_mass": 0.031,
  "captured_at": "2026-07-25T14:31:07Z"
}
```

`truncated_mass` is `1 − Σexp(top-K logprobs)`: the probability that was not observed. When it is
large, the top-K is telling you little, and the grid must say so rather than let a confident
conclusion be read off a fraction of the distribution.

A failed cell **writes a row** carrying an error object instead of `top_k`. An absent row means
"not yet run"; the two must be distinguishable in a partially complete grid.

## Capture plumbing

A new package, deliberately not routed through `EvalRunner`:

| Module | Responsibility |
|---|---|
| `Evals/word_bench/models.py` | `BenchConfig`, `Target`, `Snippet`, `CellCapture` dataclasses |
| `Evals/word_bench/capture_client.py` | The one HTTP seam: both endpoints, neutral sampler, shape normalization |
| `Evals/word_bench/runner.py` | Grid execution: order, concurrency, progress, cancel |
| `Evals/word_bench/analysis.py` | Entropy, divergence, probe resolution, spread, group aggregates |
| `Evals/word_bench/storage.py` | Mapping to `eval_tasks` / `eval_datasets` / `eval_runs` / `eval_results` |

Three reasons for a separate runner: it sidesteps the `logprob` → `LogProbRunner` collision, it
avoids coupling the slice to a 96 KB module, and a word bench cell has no expected output and no
metric, so `EvalSampleResult`'s scoring fields would all be dead weight.

### The normalizer

Three response shapes must become one `[(token, logprob, bytes)]` list:

- OpenAI chat: `choices[0].logprobs.content[0].top_logprobs` — list of `{token, logprob, bytes}`
- OpenAI legacy completions: `choices[0].logprobs.top_logprobs` — list of token→logprob dicts
- llama.cpp's native variant

**This is the likeliest thing in the design to be silently wrong.** Each shape is pinned by a
fixture captured from a live server before the parser is written, not asserted from documentation
afterwards. A wrong assumption here produces plausible-looking garbage that no downstream test
would catch.

### Preflight

Before a run, one 1-token call per target resolves it to exactly one state, mapped onto the design
contract's required readable labels:

| Preflight result | Badge | Recovery |
|---|---|---|
| logprobs returned, K recorded | **Ready** | — |
| endpoint unreachable | **Unavailable** | `.ds-recovery-callout`: which endpoint, check the server |
| reachable, no logprobs in response | **Blocked** | callout: this provider cannot report logprobs |
| raw mode unsupported by endpoint | **Blocked** | callout: switch the bench to chat mode or change target |

`k_returned` may be lower than requested; the grid header states the effective K.

Preflight results carry a check timestamp and go stale when a server restarts or a model is
swapped, so preflight always re-runs at launch regardless of a cached result.

Per the design contract, `ds-status-badge` colour must live in app-tier CSS, not widget
`DEFAULT_CSS`, or the bundle outranks it regardless of specificity.

### Security: a bench never carries a base URL

Targets reference `eval_models` rows, which resolve against configured providers. Word bench calls
are provider calls and follow the `LLM_Calls` precedent — direct to the configured endpoint, no
egress policy — which is only safe because the endpoint comes from user configuration and never
from bench content.

This matters because benches are importable, shareable artifacts. `Utils/egress.py` blocks hosts
resolving to private IPs unless explicitly trusted, and local model servers live on `127.0.0.1`; a
bench that could name its own endpoint would be an SSRF vector wearing a benchmark costume. Bench
import re-binds targets to local `eval_models` rows and cannot introduce an endpoint.

### Execution

- **Row-major fill.** Complete, comparable rows appear while the run is still going, which is the
  point of the grid doubling as the progress view. Fail-fast on dead targets is preflight's job, not
  the fill order's.
- **Sequential within a target, sequential across targets by default.** Local servers are frequently
  single-slot; concurrent requests queue or 503. Parallelism is opt-in through a `concurrency` field
  on the bench, so the setting travels with the bench that was tuned for it rather than living in
  global preferences.
- **Cancel operates on the run group**, not per-run — `cancel_evaluation` is per-`run_id`. Cells
  already captured persist and the grid renders as partial; a cancelled run is a real, if
  incomplete, measurement and is never discarded.
- **No cross-run cache.** Reusing a cell captured during an earlier run would place data from a
  different moment — possibly a different server build or swapped weights — inside a grid claiming
  to be one measurement, in direct conflict with the provenance rule above. Duplicate snippets are
  a user error flagged in the editor, not something to silently dedupe.
- Execution runs in a Textual worker with `exit_on_error` handled; an uncaught worker exception
  otherwise takes down the app.

## Results grid

The grid is the run view and the progress view. Cells fill in as the runner completes them.

Rows are snippets, columns are targets, and a **lens** decides what a cell renders:

```
 loaded-nouns v1 · 14:31 · raw continuation · K 20 · 36 cells · 0 failed
 LENS ▸ Top-1 │ Entropy │ Probe " Sure" │ Coverage │ Δ baseline    BASELINE ▸ col 1
────────────────────────────────────────────────────────────────────────────────
                          llama-3-8b     qwen-2.5-7b     llama-3-8b
 snippet                    (base)          (base)       (+prefix)
────────────────────────────────────────────────────────────────────────────────
 the protestors were     " a"     44%    " a"     39%    " a"     41%
 the rioters were        " a"     38%    " ar"    22%    " not"   31%
 the regime said         " it"    29%    " it"    31%    " it"    28%
 the government said     " it"    33%    " that"  27%    " it"    35%
```

| Lens | Cell shows |
|---|---|
| Top-1 | argmax token and its probability |
| Entropy | distribution spread, in nats |
| Probe | a chosen probe's probability, in its observed / bounded / never-observed state |
| Coverage | `truncated_mass` — where the measurement is weak |
| Δ baseline | divergence from the baseline cell |

Switched to Δ and sorted descending, the benchmark answers its own question:

```
 LENS ▸ Δ baseline                          BASELINE ▸ col llama-3-8b (base)
────────────────────────────────────────────────────────────────────────────────
                          llama-3-8b     qwen-2.5-7b     llama-3-8b
 snippet            group   (base)          (base)       (+prefix)      spread
────────────────────────────────────────────────────────────────────────────────
 the rioters were  loaded  baseline       0.31 ▄        ≥0.52 ▆  !       0.52
 the regime said   loaded  baseline       0.09 ▁         0.14 ▂          0.14
 the protestors…  neutral  baseline       0.08 ▁         0.11 ▁          0.11
 the government…  neutral  baseline       0.06 ▁         0.07 ▁          0.07
────────────────────────────────────────────────────────────────────────────────
 group mean        loaded                 0.20           0.33
 group mean       neutral                 0.07           0.09
```

### Baseline

Explicit and switchable between a **column** and a **row**. A column baseline answers "what did the
prefix change"; a row baseline answers "how far did each loaded phrasing move this model from the
neutral one". The header always states which is active — a divergence figure with an unstated
reference point is the easiest way to mislead yourself here.

When the baseline cell itself failed, the whole comparison is unavailable for that row or column and
renders as such, never as zero.

### Divergence

Jensen–Shannon divergence, computed over the top-K tokens **plus one lumped `other` bucket** holding
`truncated_mass`. That support sums to 1, so the divergence is well-defined; scoring absent tokens at
their bound would not form a distribution at all.

Two properties the UI must state rather than hide:

1. **It is a lower bound.** Lumping unobserved mass into one shared symbol assumes both tails
   overlap perfectly when they may be disjoint. The error has a known direction, so the value is
   always a lower bound — but annotating every cell would be noise.
2. **Mixed K biases comparison.** A K=100 cell and a K=20 cell have systematically different
   truncated mass, so their divergence would reflect the K difference rather than model behaviour.
   Both cells are truncated to `min(K)` before comparison.

**One threshold governs both annotations.** When a pair's combined truncated mass exceeds 25%, the
cell renders `≥ 0.52` *and* carries `!`. Below that threshold it renders a bare value. A single
threshold means `≥` and `!` always co-occur, so neither can be misread as saying something the other
doesn't.

The `spread` column is max pairwise divergence across the row: the fastest way to find where targets
disagree most. **Group mean** rows aggregate divergence by the snippet `group` field, which for a
control/treatment set is arguably the headline number. Ungrouped snippets contribute to `spread` but
are excluded from every group mean; a group row appears only for a named group.

**Export** (`e`) writes the grid as CSV for the active lens, or JSON for the whole run group —
snapshot, every cell's top-K, and the resolved probe readings. The JSON form is what makes a run
reproducible outside the app.

### Interaction

Arrow keys move cell focus and the inspector tracks, showing the focused cell's full top-K and probe
table. `l` cycles lens, `b` sets baseline, `s` sorts, `e` exports. All of it registers through
`ShortcutContext` so the shell footer stays truthful and does not retain stale shortcuts after
navigation.

Tokens render through a canonical renderer that makes whitespace visible: `" a"` and `"a"` are
different tokens and must not look identical in a grid about token-level behaviour.

## Dataset authoring and import

Selecting a dataset puts the snippet editor in the detail pane:

```
 nouns-12                                    inline · 12 snippets · 2 groups
 ───────────────────────────────────────────────────────────────────────────
  #   snippet                                      group      chars  flags
  1   The protestors were                          neutral      19
  2   The rioters were                             loaded       16
  3   The regime said                              loaded       15
  4   The government said␣                         neutral      20   trailing ␣
  5   The government said                          neutral      19   exact dup of 4
 ───────────────────────────────────────────────────────────────────────────
 [ + Add ]  [ Import… ]  [ Export… ]                            2 warnings
```

**Whitespace validation is the headline feature of this editor, not a nicety.**
`"The protestors were"` and `"The protestors were "` produce entirely different next-token
distributions: with the trailing space, the leading-space variants (`" a"`, `" the"`) that dominate
the first case become impossible and the column shifts to bare-word tokens. A user comparing two
snippets where one has a stray space would read a large divergence as a finding about the model.

Anomalous whitespace — leading, trailing, or interior runs — renders as a highlighted `␣` and raises
a warning. Normal text carries no marker, so the marker means something wherever it appears.

**Only exact duplicates are flagged**, after whitespace normalization. Near-duplicate detection
would be actively wrong here: minimal pairs differing by one word *are the instrument*, and warning
on every well-constructed word bench would train users to ignore the warning strip where the
whitespace warning also lives.

The character count is a character count. There is no client-side tokenizer, and a token count would
be a guess rendered as a fact in a tool about token-level behaviour.

Each snippet carries a UUID assigned at authoring time, an optional `group` label, and an optional
note. **`group` has exactly one job**: it groups rows in the grid and drives the group-mean
divergence aggregate.

**Import** takes three shapes: plain text one snippet per line (the low-friction path most sets will
use, and which therefore cannot express multi-line snippets), CSV with a `text` column plus optional
`group`, and JSON for round-tripping an exported set. Existing file-backed `eval_datasets` load
through the current `DatasetLoader` rather than a second path.

Datasets soft-delete via the existing `deleted_at`. Historical runs survive because their snapshot
carries snippet text.

## Classic tasks

Pre-existing `eval_tasks` of type `question_answer`, `generation`, and `classification` appear in a
labelled subgroup under Benches. Selecting one gives a **read-only** detail pane and its run history:

```
 mmlu-subset                                          question_answer
 ─────────────────────────────────────────────────────────────────────
 Dataset    mmlu-500                    Metric    exact_match
 Config     read-only

 RUNS  3
  ✓ 2026-07-04   gpt-4o-mini    0.71
  ✓ 2026-06-28   llama-3-8b     0.55
  ✗ 2026-06-28   llama-3-8b     failed

 Running classic tasks is not available in this slice.
```

Their historical runs and results open read-only through `EvaluationOrchestrator.get_run_summary`
and `get_run_results`. Metrics render from `eval_run_metrics`; there is no grid, because a classic
run has scored samples rather than distributions.

**Launching a classic task is deliberately out of scope.** Doing it properly needs a model picker, a
sample cap, a second progress surface, and a second set of failure states — a parallel execution
path through `EvaluationOrchestrator` alongside the word bench runner. That is its own slice. The
capability being deferred is one the current screen barely delivers: `quick_test.py`, the only place
it exists today, is in the deletion list.

The empty-state copy and the detail pane both say so plainly. A stated "not yet" is honest; a run
button that produces a dead-end toast is what this rebuild exists to remove.

## Bench configuration and portability

The bench editor sets name, description, dataset, prompt mode, top-K, probe list, and targets.
Changing prompt mode revalidates targets, since a `system_prompt` target is invalid in raw mode and
a `prefix` target is invalid in chat mode.

Exporting a bench yields config, snippets, and probes as JSON. **Targets export as provider-and-model
hints that must be re-bound to local `eval_models` rows on import**, per the security rule above.

## Empty states and first run

The new screen's most common initial condition is zero benches, zero datasets, zero runs — and
possibly zero configured local providers. The current screen's core failure is looking functional
while not being so; the empty states exist to prevent repeating it.

| Condition | Surface |
|---|---|
| No providers configured | Empty state routes to Settings; no target list, no wall of preflight failures |
| No benches | Offer **one-click sample bench**: the loaded-nouns snippet set, prewired to a configured target |
| No datasets | Offer authoring and import side by side |
| No runs | Point at the selected bench's run action |

The sample bench matters: it gets a new user to a populated grid without authoring anything, which
is the only way the value of the screen is legible before investing in it.

## Cost

The inspector estimate reads `local · no cost` for local targets. A chat-mode OpenAI target does
cost money, and with `max_tokens=1` the cost is almost entirely prompt tokens and cheap to estimate.
The estimate is retained for paid targets rather than disappearing because the widget that used to
implement it was orphaned.

## Deletions

**Landed as a separate PR, first.** Retiring this much code in the same change that introduces a new
screen makes review hard and bisection harder. The orphan-cluster removal is mechanically verifiable
and behaviour-neutral on its own.

| Group | Contents | Lines |
|---|---|---|
| Orphan gen-2 UI | `ResultsDashboardWindow.py`, `ModelManagementWindow.py`, `DatasetManagementWindow.py`, `Views/evals_views.py`, `Event_Handlers/eval_events.py`, and the 12 `Widgets/Evals/` files only they import | ~8,770 |
| Card hub | `UI/Evals/navigation/`, `evals_window_v3.py`, `UI/Evals/screens/`, `widgets/progress_dashboard.py`, `UI/evals_window_v2.py` | ~2,640 |
| Legacy stylesheet | `css/features/_evaluation_unified.tcss` | 288 |
| Dead wiring | `evals_sidebar_collapsed` reactive (`app.py:2923`) and its orphaned watcher `watch_evals_sidebar_collapsed` (`app.py:8125`), `EvalsWindowV3` in the container list (`app.py:1520`), the `"evals-window"` entry in the window-id list (`app.py:2847`), `1`-`6` and `escape` bindings in `evals_screen.py:31-37` | — |

Approximately **11,700 lines retired**. Everything under `Evals/` and `Evaluations_Interop/` stays.

The `toggle-evals-sidebar` handler that used to accompany the `evals_sidebar_collapsed` reactive is
already gone on `dev`; only the reactive and its watcher remain, which is why the watcher is listed
and the handler is not.

**Test collateral to handle, not discover.** Three test files use widgets in the deletion set as
their subjects: `Tests/UI/test_non_obscuring_focus_contract.py`,
`Tests/UI/test_bulk_selection_tooltips.py`, and `Tests/UI/test_file_picker_action_tooltips.py`. The
first verifies the shared flat-button vocabulary and needs a new subject, not deletion.

Each removal is gated on a per-symbol reachability check, not a whole-file assumption.

## Testing

**Unit.** `analysis.py` carries the methodology and takes the bulk of the coverage: entropy;
divergence over the K+1 support; min-K truncation before comparison; the lower-bound property; probe
resolution across observed / bounded / never-observed; spread; group means.

**Fixtures.** One captured payload per response shape, taken from a live server, pinning the
normalizer. Written before the parser.

**Integration.** Round-trip a bench through storage into a grid pivot on real in-memory SQLite, per
project convention. A v3 → v4 migration test. A run-snapshot test proving a grid still renders after
its dataset is edited and after it is deleted.

**UI.** Stable IDs for the primary action, readiness badges, lens selector, baseline control, and
recovery callouts. Readable status text asserted; colours never asserted. Density classes asserted.
A regression test that the footer does not retain stale shortcut context after navigating away.

**Live.** Verification through the `verify` skill against a real llama.cpp instance, covering both
prompt modes, a deliberately unreachable target, and a target that cannot report logprobs.

## Build order

Risk-first. The normalizer is the likeliest thing to be silently wrong, so it is proven against live
fixtures before any UI exists to display its output.

**This is more than one implementation plan.** The work splits into three PRs, each independently
reviewable and independently revertable:

**PR 1 — Deletion.** Orphan clusters, gated per symbol, plus the three affected test files. Behaviour
neutral; lands first so the rebuild starts from a clean baseline.

**PR 2 — Word bench engine.** No UI.

1. `capture_client.py` + normalizer + fixtures; both prompt modes against a live server.
2. `analysis.py` with its unit suite.
3. Schema v4, `storage.py`, run snapshot.
4. `runner.py` — order, cancel, progress.

**PR 3 — Screen rebuild.** Consumes the engine.

5. Screen shell: three panes, library rail, empty states.
6. Bench editor, snippet editor, import, classic-task read-only detail.
7. Results grid, lenses, inspector cell detail.
8. Sample bench, export, cost estimate.

PR 2 is verifiable on its own — an engine that produces a correct grid as JSON is a real deliverable
even before anything renders it, and it is where the methodological risk is concentrated.

## Risks

1. **Normalizer shape assumptions.** Three formats verified from documentation rather than
   observation would produce plausible wrong numbers that no downstream test catches. Mitigated by
   capturing fixtures first.
2. **Sampler neutrality cannot be enforced.** Servers may clamp or ignore the requested parameters.
   Recorded and surfaced; not solvable client-side.
3. **Cross-model probe comparison.** Mitigated by the never-observed state and a within-column
   default, but tokenizer differences remain a limitation of the method, not a bug to fix.
4. **Schema-version collision at merge.** `SCHEMA_VERSION` 3 → 4 while concurrent branches also
   migrate. Re-verify at merge, not only at branch time.
5. **Branch currency.** Work proceeds in a worktree off `origin/dev`.

## Deferred

Cross-run cell caching; server-scope evaluations; A/B testing; HuggingFace import; multi-position
logits; wiring the remaining orchestrator task types into the new authoring surface; a second bench
type.
