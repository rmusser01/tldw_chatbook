# TASK-16588 Task 2 — the dual-index semantic/hybrid route probe

Date: 2026-08-16 · Worktree `.worktrees/rag-16588-route`, branch
`feat/rag-16588-semantic-identity`, parent `923f356c2` (Task 1's payload
addition). Spec `Docs/superpowers/specs/2026-08-16-rag-semantic-identity-design.md`
(`279619d86`), plan `Docs/superpowers/plans/2026-08-16-rag-semantic-identity.md`
(`92fdf243c`).

Mechanical. **No LLM, no network, no spend.** The probe never boots the TUI.

## What was measured, and why this run and not the last one

TASK-16174's Phase E oracle run measured `expand_document` end to end on the
**`plain`** route only. That route structurally cannot produce a chunked row
(no `chunk_start` to carry) and its `source_id` is always a real database id
(no identity fallback ever needed), so both of Task 3b's pre-registered
suspects came back **unrefuted rather than refuted**. This probe runs the two
routes a vector or fused profile actually uses — `semantic` and `hybrid` —
against **two index kinds**, because the corrected spec established that the
identity gap is a property of the INDEX's metadata vocabulary, not of the
route:

| index kind | how it was built | what `_semantic_row` resolves `source_id` to |
|---|---|---|
| **canonical** | 20 items (12 notes, 4 media, 4 conversations) through the app's own `note_document` / `media_document` / `conversation_document` builders + `index_entries` | the real database id — the builders write `source_id`/`source_type` and `store_documents_batch` spreads them into every chunk |
| **non-canonical** | the same 12 notes through a HAND-BUILT `IndexEntry` whose metadata is `{"type", "note_id", "title"}` (TASK-15810's committed QA seeder, `seed_profile.py:64-72`) | the vector store's **point id** (`note_<uuid>_chunk_4`) — no `source_id`, no `document_id` in the metadata |

The vocabularies that actually reached chroma confirm the split (read back
from `vector_store.get_collection_stats()`):

```
canonical     metadata_fields: chunk_end chunk_id chunk_index chunk_size chunk_start
                               doc_id doc_title source_id source_type text_preview
                               title word_count
non-canonical metadata_fields: chunk_end chunk_id chunk_index chunk_size chunk_start
                               doc_id doc_title note_id text_preview title type
                               word_count
```

and so do the top rows the engine returned for the same query:

```
canonical      point_id=note_b53d…_chunk_4  source_id=b53d…   note_id=None   doc_id=note_b53d…
non-canonical  point_id=note_756f…_chunk_4  source_id=None    note_id=756f…  doc_id=note_756f…
```

which the adapter projects as:

```
canonical      {"source_id": "b53d…",              "chunk_id": "note_b53d…_chunk_4", "chunk_start": 9192, "doc_id": "note_b53d…"}
non-canonical  {"source_id": "note_756f…_chunk_4", "chunk_id": "note_756f…_chunk_4", "chunk_start": 9192, "note_id": "756f…", "doc_id": "note_756f…"}
```

The non-canonical row's `source_id` **is the point id**. Before TASK-16588
that was the only candidate the payload gave the tool (`_candidate_ids`
order: `source_id, note_id, media_id, doc_id`), and it names nothing
fetchable.

## Method

* **Corpus.** 7 long documents (4 notes, 2 media, 1 conversation), each
  ~12,300 characters, each carrying a distinctive marker sentence planted
  **9,624–9,736 characters in** — past `expand_document`'s 8,000-character
  default budget, so a document-HEAD window structurally cannot contain it.
  The builder asserts the achieved offset rather than assuming it, and
  rejects any bracket in the corpus (a row's `snippet` is
  `escape_markup`-escaped, so a bracket would make the substring checks
  compare escaped text against raw document text). Filler vocabulary shares
  no token with any marker. Plus 8 short notes, 2 short media and 3 short
  conversations for bulk.
* **Routes.** Ten queries on the canonical index, seven on the non-canonical
  one (its corpus has no media/conversation seams): one per long document,
  targeting that document's marker tokens, plus three marker-free general
  queries. Each query runs at `top_k=10` on **both** routes. The route is
  selected the way Settings selects it — `rag_service.config.search.
  default_search_mode`, which `_resolve_profile_search_mode` reads fresh on
  every `rag`-mode call — never by patching the search path.
* **Surfaces.** The production surface is `LibraryRagToolProvider.invoke`
  (`mode="rag"`, the sealed 32 KiB payload). `run_library_rag_search` is
  called one layer down purely so the probe can read the provenance the
  payload deliberately never carries, and `rag_service.search(search_type=…)`
  is the control that shows what metadata arrived.
* **Expansion arms.** Every row carrying identity (the hint's own
  precondition) is expanded by a DIRECT `ExpandDocumentTool().execute(...)`
  call, three times over the same row:
  * `pre` — `note_id`/`doc_id` stripped by the probe: the payload exactly as
    it was before TASK-16588. This is the before/after arm, done with a
    probe-side flag rather than a checkout dance — no stash, no re-commit,
    same process, same seeds, same rows.
  * `post` — the payload as shipped at `923f356c2`.
  * `head` — `post` minus `chunk_start`: the control proving the marker is
    genuinely out of reach of a document-head window.

Artifact: `probe-artifacts.json` (every row, every arm, every window).
Script: `route_probe.py` (`all` = both kinds + merge; `one` = one isolated
worker).

## Results — per (index kind × route)

Counts are over ALL rows returned across that kind's queries. **hinted** =
rows carrying identity (the hint's precondition, `expandable` either way);
**expandable** = the stricter subset the hint recommends following.
`not_found` is reported on both, because a `not_found` on either is a row
whose identity the payload declared and the tool could not use.

| index × route | rows | hinted | expandable | **not_found PRE** (hinted / expandable) | **not_found POST** (hinted / expandable) | `chunk_start` carried | fallbacks carried | variant rows w/o hint |
|---|---|---|---|---|---|---|---|---|
| canonical × semantic | 100 | 100 | 64 | **0 / 0** | **0 / 0** | 69 | 100 | 0 |
| canonical × hybrid | 100 | 100 | 61 | **0 / 0** | **0 / 0** | 56 | 100 | 0 |
| non-canonical × semantic | 70 | 70 | 49 | **70 / 49** | **0 / 0** | 45 | 70 | 0 |
| non-canonical × hybrid | 70 | 70 | 45 | **66 / 45** | **0 / 0** | 43 | 70 | 0 |

Every payload returned `status: "ready"` with `returned == 10`; 34 payloads
in total, none empty, none blocked.

**The (b) defect, measured.** On the non-canonical index every single row the
adapter declared expandable came back `not_found` before the fix and `ok`
after it — 49/49 on semantic and 45/45 on hybrid, by the strict count. On the
canonical index the reading was **0 both before and after**, exactly as
pre-registered: `source_id` already resolves there, so the fallbacks are
defensive, not load-bearing.

**The four rows hybrid rescued without the fix** (non-canonical, `not_found`
PRE 66 rather than 70) are worth naming: they are the engine's FTS
keyword-leg rows, whose identity comes from the notes database rather than
from the vector metadata, so they carry a real uuid as both `source_id` and
`doc_id`:

```
rank 9  {"source_id": "e643d5c4-…", "doc_id": "e643d5c4-…"}  {"expandable": false, "reason": "text_bearing"}
```

All four are `expandable: false`, which is why the strict count is
unaffected: hybrid's keyword leg rescues nothing the hint actually
recommends following.

## Results — AC#1's residue: does the window contain the matched chunk?

The `head` column is the control. It is the SAME row, expanded without
`chunk_start` — what the payload produced before TASK-16174's fix wave.

| index × route | slug | seam | rank | marker @ | doc chars | `chunk_start` | POST window | POST has marker | HEAD window | HEAD has marker | PRE status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| canonical × semantic | `marrowvane-coupler` | note | 1 | 9732 | 12380 | 9192 | 4380–12380 | **YES** | 0–8000 | no | ok |
| canonical × semantic | `sable-flume-weir` | note | 1 | 9700 | 12303 | 9273 | 4303–12303 | **YES** | 0–8000 | no | ok |
| canonical × semantic | `tindalos-encoder` | note | 1 | 9624 | 12268 | 9224 | 4268–12268 | **YES** | 0–8000 | no | ok |
| canonical × semantic | `hollowmere-ballast` | note | 1 | 9665 | 12288 | 9198 | 4288–12288 | **YES** | 0–8000 | no | ok |
| canonical × semantic | `catafract-lathe` | media | 1 | 9638 | 12245 | 9336 | 4245–12245 | **YES** | 0–8000 | no | ok |
| canonical × semantic | `vermillion-kiln` | media | 1 | 9733 | 12410 | 9370 | 4410–12410 | **YES** | 0–8000 | no | ok |
| canonical × semantic | `starling-gantry` | conversation | 1 | 9736 | 12364 | 9275 | 4364–12364 | **YES** | 0–8000 | no | ok |
| canonical × hybrid | `marrowvane-coupler` | note | 1 | 9732 | 12380 | 9192 | 4380–12380 | **YES** | 0–8000 | no | ok |
| canonical × hybrid | `sable-flume-weir` | note | 1 | 9700 | 12303 | 9273 | 4303–12303 | **YES** | 0–8000 | no | ok |
| canonical × hybrid | `tindalos-encoder` | note | 1 | 9624 | 12268 | 9224 | 4268–12268 | **YES** | 0–8000 | no | ok |
| canonical × hybrid | `hollowmere-ballast` | note | 1 | 9665 | 12288 | 9198 | 4288–12288 | **YES** | 0–8000 | no | ok |
| canonical × hybrid | `catafract-lathe` | media | 1 | 9638 | 12245 | 9336 | 4245–12245 | **YES** | 0–8000 | no | ok |
| canonical × hybrid | `vermillion-kiln` | media | 1 | 9733 | 12410 | 9370 | 4410–12410 | **YES** | 0–8000 | no | ok |
| canonical × hybrid | `starling-gantry` | conversation | 1 | 9736 | 12364 | 9275 | 4364–12364 | **YES** | 0–8000 | no | ok |
| non-canonical × semantic | `marrowvane-coupler` | note | 1 | 9732 | 12380 | 9192 | 4380–12380 | **YES** | 0–8000 | no | **not_found** |
| non-canonical × semantic | `sable-flume-weir` | note | 1 | 9700 | 12303 | 9273 | 4303–12303 | **YES** | 0–8000 | no | **not_found** |
| non-canonical × semantic | `tindalos-encoder` | note | 1 | 9624 | 12268 | 9224 | 4268–12268 | **YES** | 0–8000 | no | **not_found** |
| non-canonical × semantic | `hollowmere-ballast` | note | 1 | 9665 | 12288 | 9198 | 4288–12288 | **YES** | 0–8000 | no | **not_found** |
| non-canonical × hybrid | `marrowvane-coupler` | note | 1 | 9732 | 12380 | 9192 | 4380–12380 | **YES** | 0–8000 | no | **not_found** |
| non-canonical × hybrid | `sable-flume-weir` | note | 1 | 9700 | 12303 | 9273 | 4303–12303 | **YES** | 0–8000 | no | **not_found** |
| non-canonical × hybrid | `tindalos-encoder` | note | 1 | 9624 | 12268 | 9224 | 4268–12268 | **YES** | 0–8000 | no | **not_found** |
| non-canonical × hybrid | `hollowmere-ballast` | note | 1 | 9665 | 12288 | 9198 | 4288–12288 | **YES** | 0–8000 | no | **not_found** |

**22 of 22 marker rows: the `chunk_start`-anchored window contains the
matched chunk's marker. 0 of 22 head windows do.** Long-doc windows missing
their marker: **0** on every (index × route). Every marker-targeted query put
its marker chunk at **rank 1**, on both routes, in all three seams
(note/media/conversation).

This is the first evidence on any route that the window an agent actually
receives contains the chunk it matched. It also shows the two halves are
INDEPENDENT and both required: on the non-canonical index the anchored
window is correct and the expansion still returned `not_found` before the
fallbacks shipped.

### The all-rows window check — the stronger AC#3 reading

The marker check above covers 22 rows. A second, weaker-per-row but far
broader check was recorded for **every** hinted row in **every** arm and is
aggregated here for the first time: `window_has_snippet_head` — does the
returned window contain the first 160 whitespace-normalized characters of
that row's OWN snippet? Recomputed from `probe-artifacts.json`'s raw rows
(not from the artifact's own `counts` block), over all 340 rows:

| arm | window contains the row's own snippet head | fails |
|---|---|---|
| `post` (as shipped) | **340 / 340** | 0 |
| `head` (`chunk_start` stripped) | 186 / 340 | **154** |
| `pre`, canonical (fallbacks stripped) | 200 / 200 | 0 |
| `pre`, non-canonical (fallbacks stripped) | **4 / 140** | 136 (all `not_found`) |

Two things follow, both stronger than the 22-row table. **340/340 on the
`post` arm is proof the expansion resolved the RIGHT document on every
rescued row**, not merely *a* document — a window fetched from the wrong
document could not contain that row's snippet. And the `head` arm's **154**
failures make the anchor control convincing across 340 rows rather than 22:
dropping `chunk_start` breaks the window on 45 % of all rows, not only on
the long-doc ones. The 4 successes in the non-canonical `pre` arm are the
same 4 rows that resolved at all pre-fix (the engine's FTS keyword-leg rows,
`expandable: false`); the other 136 have no window to check because the
expansion returned `not_found`.

## Byte cost, on the real route payloads (AC#4)

Task 1 measured the addition by strip-and-reserialize on a synthetic ten-row
payload: **+15.0 B per carrying row, +75 B over ten rows, payload 4,085 B of
32,768 B.**

*Provenance of that figure, stated because it is not obvious:* it is computed
inside `test_sealed_payload_survives_fallbacks` and rendered into that test's
**assertion message** — a string pytest emits only when the assert FAILS, so
a green run never prints it. It is therefore **not** output of the passing
suite. It has been reproduced independently: TASK-16588's final review
replicated the same strip-and-reserialize in a standalone script and got the
byte-identical string.

The probe repeats that method on the payloads these routes actually produced
(34 of them, ten rows each):

| index × route | payload bytes (as shipped) | without fallbacks | fallback cost | rows carrying | **B / carrying row** | largest payload | ceiling |
|---|---|---|---|---|---|---|---|
| canonical × semantic | 153,434 | 148,840 | 4,594 | 100 | **45.94** | 16,770 | 32,768 |
| canonical × hybrid | 153,062 | 148,444 | 4,618 | 100 | **46.18** | 16,444 | 32,768 |
| non-canonical × semantic | 116,400 | 109,260 | 7,140 | 70 | **102.00** | 17,350 | 32,768 |
| non-canonical × hybrid | 114,527 | 107,603 | 6,924 | 70 | **98.91** | 16,915 | 32,768 |

**Stated plainly, because it is 3–7× Task 1's figure:** Task 1's synthetic
fixture used short ids (`n1`, `note_n1`). Real ids are UUIDs, so a canonical
row's single `doc_id` costs ~46 B and a non-canonical row's `note_id` +
`doc_id` pair costs ~100 B. This does not threaten the ceiling — the largest
payload observed was 17,350 B, **53 % of the 32 KiB limit**, with 15,418 B of
headroom — and the sealing loop dropped nothing: `returned == 10` on all 34
payloads. Task 1's +15.0 B/row remains the correct reading of ITS fixture;
the honest generalization is "one id string per key, whatever the id costs".

On a canonical index the fallback is `doc_id` only, and it is REDUNDANT —
`source_id` already resolves (0 `not_found` PRE). That ~46 B/row is the price
of the defence, paid on every canonical row, and it buys nothing until the
index is legacy or non-canonically built. The spec declared this trade-off
before the measurement; the measurement gives its size.

## Canonicalization-variant rows (TASK-16174 final review, finding 6)

Rows whose raw provenance `source_type` is a variant spelling
(`media_chunk`, `chat`, or a plural) canonicalize as live in
`_SEMANTIC_SOURCE_TYPE_MAP` but are absent from the policy's singular-only
`EXPANDABLE_SOURCE_TYPES`, so they get no hint and therefore no identity at
all.

**Count: 0 on every (index × route).** That zero is only evidence if such a
row would have been counted, so the probe carries a positive control that
runs the same `expand_hint` helper over synthetic rows:

```json
"variant_spellings_get_no_hint": {"notes": true, "media_chunk": true, "conversations": true, "chat": true, "prompts": true},
"singular_spellings_get_a_hint": {"note": true, "media": true, "conversation": true, "prompt": true}
```

The detector fires on every variant and on no singular. The zero is
therefore a real reading about this corpus, and its cause is that **no writer
this corpus can exercise emits a variant spelling into chunk metadata** — the
app's canonical builders write `source_type = note|media|conversation`, and
even the non-canonical 15810-shape entry writes `type = "note"`. The variant
case remains a latent hazard for third-party or legacy metadata, not
something these routes produce. Per the plan's rule (fix here only if the
count is nonzero AND the fix is a one-line allowlist broadening), nothing was
changed and nothing was appended to TASK-16688 — the finding stays there
unmodified.

## Pre-registered expectations: all three held

| expectation (plan, "Pre-registered expectations") | reading | verdict |
|---|---|---|
| Canonical index, PRE-fix **and** POST-fix: `not_found` = 0; nonzero would be a NEW finding | 0 / 0 on semantic, 0 / 0 on hybrid | **HELD** |
| Non-canonical index, PRE-fix: nonzero `not_found` (the (b) evidence); POST-fix: 0 | PRE 70 hinted / 49 expandable (semantic), 66 / 45 (hybrid); POST 0 everywhere | **HELD** |
| Long-doc chunked rows carrying `chunk_start`: window contains the marker; a head window that still contains it means the corpus failed its own design | 22/22 anchored windows contain it; 0/22 head windows do | **HELD** (corpus design validated by the head control) |

No reading fell outside the pre-registered set, so Step 2's STOP-and-report
condition was never triggered and no production code was touched in Task 2.

## Disclosed limits

* **No `label_only` rows appeared on either route** — 0 of 340 rows; every
  hint was `truncated_snippet` (219) or `text_bearing` (121). Label-only rows
  are produced by the Library's own four-seam keyword path (`_media_row` /
  `_conversation_row`, the `plain` route), not by the engine's hybrid FTS
  leg. TASK-16174's oracle run measured that regime; this probe measures the
  two it could not. The two runs are complements, not overlaps.
* **Retrieval quality is not what this measures.** Every marker query put its
  target at rank 1, which makes the window checks clean but says nothing
  about retrieval on a real corpus. The gated eval suite is the instrument
  for that, and it is unchanged (105/105 at +0.000).
* **The `pre` arm is a payload reconstruction, not a checkout of the old
  code.** It strips exactly the two keys Task 1 added, from the real rows the
  real adapter produced, in the same process. `chunk_start` is deliberately
  KEPT in the `pre` arm because it already shipped in TASK-16174's fix wave —
  the before/after being measured is TASK-16588's, not 16174's.
* **Every anchored window in this run is the document TAIL, so 22/22 proves
  "off the head", not "centred on the match".** All 22 POST windows are
  `[total − 8000, total]` (e.g. 4380–12380 of 12380). With ~12.3k-char
  documents and an 8,000-char budget, a `chunk_start` of ~9,200 has exactly
  two reachable outcomes — head or tail — and the check distinguishes those
  two, nothing finer. An extender who wants to show a true MID-document
  slice must vary the two things this instrument holds fixed: make the
  document 3–5× the budget, and plant markers at offsets that are past the
  budget (so the head control still fails) but NOT within one budget of the
  tail — i.e. `budget/2 < chunk_start < total − budget/2`. Nothing here is
  wrong; it is simply a weaker statement than "the window is centred on the
  match".
* **`media_id` was never projected and never needed**: no builder writes it
  (spec verification item 2), and no row in 340 carried one.

## Isolation and teardown

Per `backlog/docs/lessons-live-verification.md` ("a bare interpreter call is
not an isolated test", "isolate the profile before touching runtime state").

* Scratch `HOME` / `XDG_CONFIG_HOME` / `XDG_DATA_HOME` / `XDG_CACHE_HOME` /
  `TLDW_CONFIG_PATH`, one profile per index kind, all set **before any
  `tldw_chatbook` import** — every project import in `route_probe.py` is
  inside a function for exactly this reason.
* The scratch config sets `[paths] data_dir`, because `TLDW_CONFIG_PATH`
  relocates the config FILE only. Verified at runtime, not assumed — the
  worker asserts and prints the resolved directory:

  ```
  RESOLVED data_dir: …/scratchpad/probe16588/run/canonical/data/probe16588_canonical
  RESOLVED data_dir: …/scratchpad/probe16588/run/noncanonical/data/probe16588_noncanonical
  ```

  Both databases and the chroma store live under those paths, and the probe
  seeds and expands through the SAME `get_chachanotes_db_lazy()` /
  `get_media_db_lazy()` handles `expand_document` itself resolves, so it
  cannot seed one database and expand another.
* The embedding model is copied into the scratch **profile-local** cache
  (`<data_dir>/<user>/models/embeddings` — not `$HF_HOME`) with
  `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so a download is
  impossible.
* Real config sha256, before and after the whole run:

  ```
  BEFORE: 42e2f42de95915f59fcc7c36751c2cc041cb0e4ddda9b324668683a270d7ecc0  ~/.config/tldw_cli/config.toml
  AFTER : 42e2f42de95915f59fcc7c36751c2cc041cb0e4ddda9b324668683a270d7ecc0
  UNCHANGED: True
  ```

* Teardown checklist: no TUI boot, so no `css/tldw_cli_modular.tcss`
  regeneration (`git status` confirms it untouched); no tmux session; the
  scratch tree lived under the session scratchpad and was removed after the
  artifact was copied here; the real shared ChaChaNotes/Media databases were
  never opened; no credentials were used or written.

## Reproducing

```bash
.venv/bin/python Docs/superpowers/qa/2026-08-16-rag-semantic-identity/route_probe.py \
    all <scratch-root> <out-dir>
```

Requires this worktree's venv and an on-disk copy of `all-MiniLM-L6-v2` under
the real profile's `models/embeddings` (override with `MODEL_CACHE=`). The
run is deterministic: three consecutive runs produced identical counts and
identical byte totals (the only variation is the freshly generated row
UUIDs). Wall time ~2 minutes for both index kinds.
