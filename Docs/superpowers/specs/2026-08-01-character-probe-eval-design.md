# Character Probe Evals — Design

**Date:** 2026-08-01
**Status:** Approved (brainstorm), ready for implementation planning

## Purpose

Study how models handle particular characters and complex prompts. A run takes a selection of
character cards, asks each of them a set of scripted questions — some single-turn, some multi-turn —
against one or more models, collects the generated replies, and presents them for human review.

There is no objective metric. The human is the instrument; the tool's job is to collect evidence
faithfully, make it readable, and let judgments accumulate.

## What this is NOT

This shares none of the word bench's measurement stack. **No logprobs, no top-K, no normalizer, no
truncated mass, and no canary/degenerate vocabulary** — that entire concept exists to judge token
distributions, which this eval does not look at. Only generated text matters.

Consequently a character-probe target's readiness means one thing: can we reach this model and get
text back. It is a cheap reachability check, not a distribution probe, and the "degenerate canary"
warning must never appear on this bench type.

What it does reuse: targets (`eval_models` rows and their steering), run groups, the Catalog rail,
cancel/progress, and the cost Estimate. What it does not reuse: the capture client, the normalizer,
the results grid, and the lens vocabulary.

## Entities

### Probe and probe set

**A probe is an ordered list of user turns.** A "one-off" question is a probe with one turn; a
sequential probe has several. One structure, no separate types.

**Probe sets reuse the Datasets rail section**, discriminated the way benches already are:
`bench_type` on `eval_tasks` is the precedent for `dataset_type` on `eval_datasets`. A probe set
inherits naming, soft-delete, import, and its rail row; only its content shape differs from a
snippet list.

**v1 authoring is import-only.** Probe sets are nested (probes containing turns) where every editor
in this slice today handles flat text; building a nested editor is the largest single piece of UI
work in this design and is deliberately deferred. v1 defines a plain-text format, imports it through
the existing import path, and displays it read-only.

**Turns are delimited explicitly, never by line breaks.** A turn separated by newline could not
contain a multi-paragraph prompt, and complex prompts are exactly what this eval exists to study. A
line of `---` separates turns within a probe; a line of `===` separates probes. Everything between
delimiters is one turn verbatim, newlines included:

```
What do you think about lying?
---
And if lying protected someone you love?
===
Describe your earliest memory.

Take your time, and include what you could smell.
```

Leading and trailing blank lines around a turn are stripped; interior whitespace is preserved
exactly, since prompt formatting can change behaviour.

**Limitation:** The v1 format does not escape delimiters, so a turn whose content contains a line of
`---` or `===` will not round-trip correctly through parsing — the line will be treated as a
delimiter. This applies to any line whose **stripped** content is a delimiter, not only a bare one:
matching compares `line.strip()`, so `"  ---  "` and `"\t==="` delimit exactly as `---` and `===`
do, and an indented or trailing-space delimiter line cannot appear inside a turn either.

That leniency is a **deliberate ruling**, not an oversight of the "a line of `---`" wording above.
The two failure modes are not symmetric. Under a strict match, an invisible trailing space makes a
delimiter silently *fail* to delimit — merging two turns into one prompt that then runs and returns
plausible-looking results. That is both likely (editors and copy-paste add trailing whitespace
constantly) and near-impossible to spot. Under the lenient match, the cost is that an indented
literal `---` inside a turn is consumed as a delimiter — less likely, and it fails visibly as a
probe split in the wrong place. Both behaviours are pinned by test so neither can drift.

Escaping is out of scope for v1; a rich probe editor (v2, following task-1482's model) will address
this.

A rich probe editor is a follow-up, mirroring how bench authoring (task-1482) followed the sample
bench.

**A built-in starter probe set ships with the feature.** Without one, nothing works until the user
hand-authors a text file — the feature's first impression is a wall. The Evals design spec makes
this argument already about the sample bench: it "gets a new user to a populated grid without
authoring anything, which is the only way the value of the screen is legible before investing in
it." The same reasoning applies with more force here, since a probe set is harder to write than a
snippet list. The starter set contains a handful of probes that exercise the behaviours the built-in
tags name (a one-off question, a multi-turn pressure sequence, an in-character boundary test), so a
first run produces something worth reviewing.

Two empty states matter and are easy to miss: **no probe sets** (offer the starter set), and **no
character cards at all** (route to where cards are created — the eval is unusable without them and
must say so rather than presenting an empty picker).

### Bench

An `eval_tasks` row with `config_data.bench_type = "character_probe"`, whose config holds:

- `probe_set_id` — the dataset supplying the scripts
- `character_ids` — selected cards, plus a name snapshot for display
- `target_ids` — existing `eval_models` rows, so task-1611's steering applies unchanged
- sampler settings — including an **optional seed** and **`samples_per_cell` (default 1)**
- `extra_tags` — bench-local additions to the tag vocabulary

### Characters

Character cards live in `ChaChaNotes_DB`; evals live in `Evals_DB`. There are no foreign keys across
that boundary, so:

- selection stores card **ids plus a name snapshot** for display,
- at run time the card's **actual text is copied into the run snapshot** (`description`,
  `system_prompt`, `personality`, `scenario`, `first_message`, `post_history_instructions`,
  `message_example`).

`description` was missing from that list in the original draft of this spec. That was an **error,
not a decision**: it is a real `character_cards` column, it is the primary V2 persona field, and
Console already sends it. The whole-branch review of task-1691 phase 1 found the engine faithfully
implementing the omission, so every probe ran against a character stripped of its main definition,
with no copy of it kept anywhere in the run. Corrected under the same full-card-fidelity ruling
that governs the rest of this section.

This is the same provenance rule word_bench uses for snippets, and it means editing or deleting a
card later never rewrites history. The run snapshot therefore holds the card TEXT (not just ids),
the composed system prompt per card per target, the resolved target list with its steering, and the
sampler settings — a run that cannot be re-derived from the mutable bench row afterwards.

The Evals view model wraps `EvalsDB` only and will need a second, read-only handle for
`ChaChaNotes_DB`. The card picker (search + multi-select over potentially hundreds of cards) is a
real component that does not exist in this slice today and must be scoped as its own work.

### Cell

**A cell is one conversation**, identified by (card × probe × target × sample index), stored under a
shared `run_group_id`. Within a conversation, model turns are indexed.

**Storage convention** (verified against the live schema, which is shaped for single-answer samples
and must not be guessed at): `eval_results` carries `run_id` + `sample_id` (unique together),
`input_data`/`metadata` JSON columns, and a flat `actual_output` TEXT column. `run_id` already
scopes the target, as it does for word benches. Therefore:

- `sample_id` composes `(card_id, probe_index, sample_index)`,
- `input_data` holds the scripted user turns and the card reference,
- **the ordered turn list lives in the `metadata` JSON**, never in `actual_output` — that column is
  shaped for one answer and cannot represent a conversation.

### Annotations and review state

Two separate things, because they answer different questions:

**Per-turn annotations** — keyed by (run_group, card, probe, target, sample, turn_index), holding
tags, a free note, and timestamps. This is where "it broke character on the third turn" lives.

**Per-conversation review state** — keyed by (run_group, card, probe, target, sample), holding
`reviewed_at` and an optional overall note. This is the only home for "I read this and nothing was
notable", which the queue's progress count depends on: without it, a clean conversation is
indistinguishable from an unopened one. A conversation may be reviewed with zero turn annotations,
and that is a meaningful, common outcome.

Both attach to a **specific run's** answers: a re-run produces new answers to annotate rather than
silently inheriting old judgments. Deleting a run group cascades both, since they describe those
answers and mean nothing without them.

### Tags

Built-in defaults, extendable per bench. Every tag carries a **kind** — `failure`, `notable`, or
`positive` — so the summary cannot imply "fewer tags is better". Tags are stored as canonical slugs
and the UI offers existing tags before creating new ones, to limit the `broke-character` /
`OOC` / `out-of-character` fragmentation that per-bench extension invites.

**Creating a tag requires choosing its kind** — the extension flow asks, with no default. A kind
guessed on the user's behalf would quietly mis-group observations in the summary, and `notable` is
not a safe fallback: it would make genuine failures invisible in exactly the view meant to surface
them.

## Execution

Calls go through the app's normal chat path (`chat_api_call`), not the word-bench capture client:
multi-turn messages in, text out, real sampler, no logprobs.

**`chat_api_call` is synchronous and must never be called from the event loop.** It is a plain
`def` with no awaits; invoking it directly inside the async runner would block the loop and freeze
the whole TUI — a failure this codebase has already been bitten by. Every call dispatches through
`asyncio.to_thread`, the app's established bridge for exactly this (see `console_agent_bridge.py`
and `chat_conversation_scope_service.py`). The bench's own `concurrency` setting is what bounds the
thread fan-out; the default executor's size must not be the de-facto limit.

**Prompt assembly is the engine's own code, not reuse of an existing path.** The original intent
here was to reuse `Character_Chat_Lib`'s card→prompt path rather than write a private copy, but no
such function exists — the actual joiner lives in
`UI/Screens/chat_screen.py::_character_session_prompt_seed`, which the engine cannot import
without violating its no-`UI/`-dependencies boundary. So the engine composes the prompt itself,
deliberately including every field that shapes voice: `system_prompt`, `personality`, `description`,
`scenario`, `message_example`, and `post_history_instructions`. Console's own joiner currently sends
only `system_prompt`, `personality`, `description`, and `scenario` — it omits `message_example` and
`post_history_instructions` — so a probe run is **not** byte-identical to what Console sends
today. Per the human's ruling, the eval's full-card fidelity is correct and Console is the one
that should catch up; TASK-1744 tracks extracting one shared card→prompt function both paths use.
The engine composes those four shared fields **in Console's own order** so that shared function has
one less difference to reconcile, then appends the two Console does not send.

**Card macros are resolved, not passed through.** Cards are authored against SillyTavern-style
macros, and Console resolves `{{char}}`/`{{user}}` (and their aliases) before the text reaches any
provider payload — task-1530 exists because they otherwise leak verbatim. The engine resolves them
the same way, through the same non-UI function (`Character_Chat_Lib.replace_placeholders`), with
`{{user}}` → `"User"` as Console substitutes it: a probe is a script, not a real person, and a
probe that shipped a literal `{{user}}` would be evaluating text no real chat with that card ever
produces. Two things are deliberately **not** macro-resolved: a probe's scripted user turns (the
eval author's text, sent verbatim per the format's own rule) and a target's steering (attached to a
target, not a card — a run spans several cards, so there is no one right name to substitute).

The card's `first_message` seeds the opening assistant turn as in real roleplay, then the probe's
scripted user turns alternate with model replies, accumulating context. **A card with no
`first_message` simply starts with the user's first scripted turn** — no synthetic greeting is
invented, because inventing one would mean evaluating text the character never had.

**Target steering and card system prompts compose, steering first.** Task-1611 lets a target carry
its own `system_prompt`; every card has one too. Steering is a model-level instruction and the card
is the content it operates on, so the steering text is placed ahead of the card's system prompt.
Both are preserved and neither is silently discarded; the run snapshot records the composed result
so what actually ran is never in doubt.

**A conversation runs its turns strictly in sequence** — turn *N* needs turn *N−1*'s reply — while
different conversations run concurrently under the existing concurrency setting. Parallelism scales
with the grid, not within an exchange.

**Cost is shown before running.** Total calls = cards × probes × targets × samples ×
turns-per-probe. For 5 cards × 5 probes × 3 targets × 1 sample averaging 3 turns, that is 225 model
calls; the Estimate must say so before Run is pressed (the lesson task-1710 paid for).

**Sampling is non-deterministic and that is a real limitation.** With realistic sampling each cell is
a single sample of a stochastic process: a model that breaks character 30% of the time looks
identical to one that never does. Two mitigations, both opt-in: an optional **seed** (honoured by
llama.cpp) makes re-runs comparable, and **samples-per-cell > 1** makes variance visible by showing
siblings adjacent in review. Both default off because they multiply review volume. The sampler
settings are stored in the snapshot so every run is self-describing.

**The two settings must compose, not cancel.** A single fixed seed applied to every sample of a cell
would return N identical answers — tripling review volume for zero information. The per-sample seed
is therefore derived as `seed + sample_index`, so a seeded run is reproducible *and* its samples
genuinely differ. Enabling both must be useful, not a trap.

**Failure is per-conversation.** A failed turn ends that conversation, records what was collected
plus the error, and leaves the rest of the grid running. Partial conversations remain reviewable —
completed turns are still evidence and still annotatable. Long cards plus multi-turn scripts will
exceed some models' context limits; that surfaces through this same path and is expected, not
exceptional.

**Cancel stops scheduling; it cannot abort a turn already in flight.** Because turns run through
`asyncio.to_thread`, and this codebase already documents that `to_thread` survives Task
cancellation, a blocking provider call cannot be interrupted — unlike word_bench, whose async client
genuinely cancels mid-request. Cancelling therefore means: start no further turns and no further
conversations; whatever is already in flight runs to completion and is recorded. **The UI must say
this plainly** rather than implying an instant stop, or a user watching calls continue after
pressing Cancel will reasonably conclude it is broken. Partial conversations are preserved, not
discarded.

## Review

**Where it lives.** Selecting a character-probe bench's run group renders the review queue in the
detail pane, in the same slot the results grid occupies for a word bench — the two bench types never
share a detail surface. Both kinds appear in the rail's existing **Benches** section, so a
character-probe bench needs a visible marker distinguishing it from a word bench at a glance;
without one, selecting a bench is a guess about which detail surface will appear.

**Review is a queue, not a grid.** A 3D grid of conversations has no good 2D rendering and the
reader's task is sequential anyway. The run group presents an ordered queue, filterable by card,
probe, target, or "not yet reviewed", so a reviewer can take a deliberate slice — *"just the villain
card across all three models"* — in one sitting.

**The conversation view** renders the full exchange as turns, with the card's opening message and
each scripted user turn in place, and a tag affordance on every model turn.

**Keyboard-first.** Reviewing dozens of conversations by mouse in a terminal app is untenable;
moving between turns and conversations and applying tags must be a few keystrokes.

**"Reviewed" is explicit, and "nothing notable" is a real verdict.** A conversation is done when the
reviewer says it is — not when it happens to carry a tag. Without that, a clean, in-character,
well-handled exchange is indistinguishable from one nobody has opened, and the progress count lies.
This also makes the queue resumable across sessions, which annotation work requires.

**Ordering hints, never verdicts.** Cheap heuristics reorder the queue to put likely-interesting
material first: empty or very short replies, replies containing text from the card's own system
prompt (a leak), refusal-shaped openings, and replies near-identical across targets. These are
rendered as hints and are never tags and never scores — if they become the judgment, the tool has
quietly invented the metric it claims not to have.

**Hints are computed at review time, on demand, never during the run.** The near-identical check
compares replies across cells, so computing it at write time would add a cross-cell pass to every
run for a signal only the reviewer uses. Nothing about a run's cost or duration may depend on
hinting.

## Summary

Tag counts aggregated across the run answer the original question: which model broke character most,
which card was hardest, which probe broke things regardless of model.

**The summary reports per-tag counts and never a composite score.** Ranking models by "fewest bad
tags" would invent the objective metric this eval exists precisely because we lack — and would be
wrong anyway, since `notable` and `positive` tags are not penalties. No view anywhere sums tags into
a number.

## Testing

Real in-memory `EvalsDB` and a fake chat client, as elsewhere in the slice. Specific to this eval:

- multi-turn ordering: turn *N* genuinely sees turn *N−1*'s reply
- prompt assembly includes every field that shapes voice (`description`, `message_example`,
  `post_history_instructions` included), pending TASK-1744 bringing Console up to the same
  fidelity via a shared function
- **targets come from real `eval_models` rows, never hand-built dicts.** A target's steering lives
  inside the row's `config` JSON, not as a top-level column; a fixture that invents the flatter
  shape will agree with code that reads the wrong place, which is exactly how phase 1 shipped seven
  green tasks while every real run dropped its steering. The same rule holds for anything else read
  off a database row.
- annotation persistence and resumption across sessions
- partial-conversation review after a mid-conversation failure
- aggregation math, including that no composite score exists
- snapshot provenance: a card edited or deleted after the run does not change what the run shows

**The review UI's tests must drive real clicks and keypresses and assert the annotation persisted.**
Programmatically setting a widget's value passes while the feature is unusable — exactly what
happened with task-1710's opt-in checkbox, where 867 tests passed on a control no user could toggle.

A live pass against a real llama.cpp instance with real character cards is required before this is
called done.

## Deliberate scope boundaries

- **Rich probe editor** — v1 is import + read-only display; nested authoring is a follow-up.
- **Cross-run comparison** — annotations are per run group; comparing two runs of the same bench is
  out of scope, and non-determinism makes it unsound without seeds anyway.
- **Automated scoring of any kind** — explicitly not a goal; the ordering hints are the only
  machine-produced signal and they never render as judgments.
