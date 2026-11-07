# Phase A — validation of all 284 findings against `origin/dev d0face3ebe`

Re-checked against **current dev**, not the review baseline `3722a85748` (25 commits behind).
Six read-only agents, one per slice group, each required to return `CONFIRMED` / `FIXED` / `WRONG` /
`DEMOTE` / `PROMOTE` with the literal command that proves it.

```
ALL 22 SLICES   confirmed=284  fixed=0  wrong=0  demoted=0  promoted=0
```

## How much weight this carries — and how much it does not

A 100% confirmation rate deserves scepticism, so state the evidence for and against.

**For.** The validators were willing to contradict the review and did so five times (below). Two ran live
reproductions rather than re-reading prose: httpx `MockTransport` for the `ResponseNotRead`/`StreamClosed`
bugs, actual ReDoS timing, an `evaluate_url_policy` SSRF repro, and the S25 O(N^2) fsync count reproduced
**exactly at n=3316**. The lead independently spot-checked three findings at random (the `www.` substring
strip, the `re.compile`-in-loop, the size-ratchet `_BUDGETS` rows) and all three matched exactly.

**Against.** A large share of the 115 P2 and 108 P3 findings are *structural* claims — "this module has no
size-ratchet row", "this regex compiles inside a loop", "these timestamps are naive". Those are close to
tautologically checkable, so confirming them is cheap and means much less than confirming a P0. The
confirmation rate should not be read as "the review was 100% right about severity or about what to do" —
only as "the code these findings describe is, today, in the state they describe."

The 3 P0s are the ones that carry weight, and they were verified three times: the review's own Phase 4 by
reproduction, then by the lead against current dev, then by the slice validators.

## Errors found in the review's own evidence (verdicts unchanged)

The review demanded this standard of the repo; it applies to the review too.

1. **S05 #8** — the worked example says Subscriptions' overshoot is "506 chars". Its `ERROR_CHAR_CAP` is
   **1000**, not 500, so the real output is **1006**. The defect (Subscriptions' copies overshoot their own
   cap by 6; Library's does not) still holds.
2. **S10/S26 #11** — claimed only `list_templates`/`get_diagnostics` have live callers. There is a **third**:
   `library_rechunk_run.py` -> `rechunk_legacy_media`, which runs inline via `_maybe_await`. The P3 verdict
   still holds only because that caller runs inside a `thread=True` worker with its own private
   `asyncio.run()` loop, never the app's main loop — so the conclusion survives for a reason the review
   did not state.
3. **S12 #6** — titled "9-table ad hoc schema"; there are **8** `CREATE TABLE` statements, matching the
   finding's own Evidence block. Off-by-one in the title.
4. **S20 #6** — headline says "eight user actions"; the file has **5** `_spawn_action` call sites, matching
   the finding's own evidence lines. Overcount in the title only.
5. **S19 #1, #10, #11** — the cited line numbers were **wrong at the review's own commit**, not drifted.
   Bad citations. The defects are real at the corrected lines.

None of these changes a verdict, but 4 of the 5 are **a headline or title contradicting the finding's own
evidence block** — which means the summary line, not the analysis, is where this review is least reliable.
Anyone triaging from titles alone should re-read the evidence block first.

## Severity note carried forward, unresolved

The endpoint-probe egress gap is rated **P1, not P0**, and the promotion criterion is explicit: whether any
**non-restore** path can set `api_settings.<p>.base_url`. Chatbook import was checked and cleared;
sync/profile-import paths were not. This is assigned to the security stream to answer.

---

# Addendum — three errors in this report's own "Legacy reachability" table

Found during burn-down (TASK-32899), by an agent instructed to re-derive every count rather than trust the
table. All three would have caused a bad deletion; two of them **contradict this report's own evidence**,
which is worse than being merely wrong.

| row | claimed | actually | how it was caught |
|---|---|---|---|
| 6 `*_Interop` modules | 0 importers, delete | **reachable, keep** | `slices/S24-interop-cluster.md:106-108` already said so: naive walk reports 7 dead packages, *"the lazy-aware walk reports **0 dead packages**. The naive answer is the trap here."* And §Retired/contested in this very report says *"All 31 are reachable from the composition root; residue is 753 lines (1.1%)."* The table said delete anyway. |
| `swarmui_client.py` + `image_generation_service.py` | "0 live", delete | **live, keep** | `chat_screen.py` → `UI/Console_Modules/image.py` → `console_generate_image.py:436` imports `ImageGenerationService`; that module imports `SwarmUIClient` at module scope (`:11`); also re-exported by `Media_Creation/__init__.py:5`. |
| `media_screen.py` | listed in a **delete** row *and* a **keep** row | delete | The keep reason (registry save_state/restore_state tests) is a test-only lifeline whose four tests were already red at baseline. |

All four affected rows are now struck through and annotated in `report.md`.

## What this says about the review

The Phase-A validation pass confirmed 284/284 findings and I cautioned then that the rate should not be
over-read. This is the concrete vindication of that caution — and note **where** the errors are. Every error
found in this review so far, without exception, is in a **summary artefact**: a finding title, a headline
count, or this roll-up table. Not one has been in a finding's evidence block.

Five title/headline errors were found in Phase A ("9-table" that is 8, "eight user actions" that are 5,
"506 chars" that is 1006, "only two callers" that are three, three wrong line citations). Two more surfaced
in burn-down (`test_mlx_lm.py` "30 tests" that are 19; the pixel-guard "eight modules" that is a union of two
families). Now three in this table.

The operational rule, which held every time it was applied: **read the evidence block, never the headline,
and re-derive every count.** The deletion stream was briefed that way explicitly and it is why nothing live
was deleted.


---

# Addendum 2 — the first error that runs the OTHER way

Every error found in this review until now overstated something. The P2 burn-down found one that
**understated**, which is worth separating out because it means the bias is not uniformly optimistic.

**S14 artifact-share index — rated `Confidence: inferred`, actually verified.**
`Web_Server/artifact_share_server.py:355` on `origin/dev`:
```python
f"<a class=\"btn\" href=\"/artifact/{item.key}\" download>Download</a></article>"
```
`item.key` is interpolated raw into an HTML attribute; a crafted key renders a live
`onmouseover="alert(1)"` into the served page. That is a rendering fact, not an inference.

**But be precise about what "verified" upgrades here — the mechanism, not the severity.** Exploitation still
requires an attacker-chosen key, and the normal path cannot produce one: `new_artifact_key()` mints a
128-bit opaque URL-safe key, and the server binds `127.0.0.1` by default (`--host` is settable, so exposure
is possible but opt-in). So it needs a tampered or imported manifest first — the same shape of prerequisite
as the endpoint-probe P1. P2 remains defensible; the fix (escape at render **plus** an opaque-token `pattern`
on `SharedArtifact.key` so `load_manifest` fails closed) makes the severity question moot either way.

## Corrections to counts, from the P2 pass

| claim | measured | consequence |
|---|---|---|
| S06 cites **3** raw `{server_id}` interpolations | **57** raw `/api/v1/mcp/…{var}` sites in `mcp_unified_client.py` | drove the fix to a central `_reject_unsafe_endpoint` guard instead of site-by-site quoting |
| S11 headline "**~9,000 lines**" for the legacy Evals stack | the six named modules are **7,569**; the review reached ~9,000 by adding `eval_templates.py` (1,298) — which a **sibling finding in the same slice** says can never execute | corrected in TASK-32904 |
| S06 `client.py` is **16,661** lines | **16,687** after the batch | the shared error helper's docstrings cost slightly more than the five copies it replaced |
| S14 "21 shipped `test_*` functions, one hitting `searx.be`" | count confirmed; **hazard overstated** — `pyproject.toml` sets `testpaths = ["Tests"]`, so none is ever collected | dead weight, not a live network call (unlike TASK-32907's real one) |
| S11 unpaginated 1000-row reads: one demoted as dead | **all three** call sites unreachable | deader than filed |

Still consistent with the rule: every one of these is in a headline or a cited count. None is in an evidence
block.

---

# Addendum 3 — the "only summary artefacts" claim is FALSE. Retracting it.

Through the first eight streams I stated repeatedly, including in commit messages, that **every error found
in this review was in a summary artefact — a title, a headline count, or a roll-up table — and none in an
evidence block.** The final P2 pass falsified that. Retracting it here rather than leaving it standing.

## Three evidence-level errors

**S05.4 — `library_get_media_structure` "materializes chunk rows used only as three aggregates."**
The evidence block itself asserts the rows are consumed only as counts. They are not:
`Library/local_media_chunk_tool_service.py` uses `chunk_rows` at `:447` (building span-family rows sorted by
`start_char`) and `:480` (`if not chunk_rows`), in addition to `:402/:412/:413`. The rows are load-bearing;
the finding is **not real**. This is the important one: **the Phase-A re-validation pass missed it too.**

**S02.3 — "diagnostics lose context because the sink format has no `{extra}`."**
The stated mechanism is wrong, which means the one-line fix it implies does not exist.
`Logging_Config.py:609` registers `_forward_loguru_to_standard` — a **function** sink. A function sink
receives the message object and decides what to emit; this one takes `record = message.record` (`:475`) and
rebuilds a stdlib record from its fields. The `format=` string only affects `str(message)`, which the
forwarder never uses. Adding `{extra}` there changes nothing. The real fix is inside the forwarder and must
skip keys colliding with `LogRecord` attributes or `logging` raises.

**S02.2 — the recommended correction is actively harmful.**
The review recommends adding `AND deleted = 0` to a keyword collision scan. Applied, it raises
`sqlite3.IntegrityError: UNIQUE constraint failed` against two deliberately-pinned test cases, because
`keywords.keyword` is `UNIQUE NOT NULL COLLATE NOCASE` — a soft-deleted row still owns its name. The
difference the review calls drift is **required**. Demonstrated by applying it and watching it fail, then
reverting; a no-op guard plus an invariant comment was left so the next reader does not re-make the change.

## The methodological lesson, which is worth more than the three findings

Phase A validated 284/284 findings as CONFIRMED by re-reading code against each finding's evidence. I
cautioned at the time that structural P2/P3 claims are cheap to confirm and that the rate should not be
over-read. That caution was right but not strong enough. The sharper statement:

**Re-reading validates that the code matches the finding's description. It cannot validate that the
description is the right description.** All three errors above are of that second kind — the cited lines
exist and say what the finding says they say; what is wrong is the claim about what they *mean*
(rows discarded / a format string that is consulted / a filter that is safe to add).

Only **implementation** found them. Two of the three surfaced specifically when someone tried to apply the
recommended fix. That is an argument for a particular ordering in any future review of this kind: a finding
whose recommendation has never been attempted is less validated than one that has, regardless of how many
people have read it.

## Corrected tally of errors found in this review

| where | count | found by |
|---|---:|---|
| finding titles / headline counts | 8 | re-reading (Phase A) and re-derivation (burn-down) |
| a roll-up table (`report.md` Legacy reachability) | 3 | re-derivation during deletion |
| **evidence blocks / stated mechanism** | **3** | **implementation only** |

The operational rule stands and is unchanged — read the evidence block, re-derive every count — but it is
now explicitly **necessary and not sufficient**.
