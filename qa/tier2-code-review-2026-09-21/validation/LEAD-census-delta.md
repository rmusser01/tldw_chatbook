# Lead verification — duplication census delta, re-measured on origin/dev d0face3ebe

The report's verdict has two legs. `LEAD-guards.md` carries the first (three green guards that prove
nothing). This is the second: **the duplication census does not move.**

Method: the **committed** tier-1 script, unmodified, same roots, same exclusions — comparing like with like.
```
$ .venv/bin/python qa/core-code-review-2026-09-17/candidates/dup_census.py tldw_chatbook /tmp/census_now
dup_by_name.tsv 1944 rows / dup_verbatim.tsv 264 rows / dup_shape.tsv 709 rows / files scanned: 2526
```

| census | review baseline `3722a85748` | now `d0face3ebe` | delta |
|---|---:|---:|---:|
| `dup_by_name` | 1943 | **1944** | **+1** |
| `dup_verbatim` | 264 | 264 | 0 |
| `dup_shape` | 709 | 709 | 0 |

25 commits landed on dev in between. The consolidation surface is **unchanged**, and the one movement
is in the wrong direction.

This extends, rather than merely repeats, the report's finding. The report said the census had not moved
in the 4 days since ten fix-stream PRs merged. It has now also not moved across the review window itself.
Taken with the `_coerce_int` data point already in `report.md` — a *consolidation* PR shipping 4 fresh
byte-identical copies 48 hours after the review catalogued the pattern — the conclusion is that
**consolidation tasks are being closed without the census being consulted**, because nothing makes
anyone consult it.

## Consequence for the burn-down

Every D4 / consolidation task in this program must carry a **census row and a guard**, per the review's
own §2 rule, or it will close the same way. The rule's stated exception still stands: under ~5 copies,
"adopt it and move on" is right and a guard is over-engineering.

A guard here means the repo's own shape: a `scripts/check_*.py` wired into `scripts/preflight.sh`
(which is the required `Derived artifacts` CI job) plus a shrink-only census TSV keyed on
`module<TAB>symbol<TAB>kind<TAB>count`. **And per `LEAD-guards.md`, a guard is only worth adding if its
predicate is checked against the contract it claims to enforce** — two of the three existing ones are
green precisely because nobody did that.
