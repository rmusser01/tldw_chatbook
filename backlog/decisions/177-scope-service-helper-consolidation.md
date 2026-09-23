# ADR-177: The scope-service helpers are a family, not a scaffold — consolidate by helper, not by base class

Status: Proposed
Date: 2026-09-22
Task: TASK-32898 (re-scoped from TASK-32808.6)

## Context

Six private helpers recur across the codebase. Measured on `origin/dev` by AST walk, excluding
`Third_Party/`:

| helper | definitions | total lines |
|---|---:|---|
| `_maybe_await` | 65 | |
| `_enforce_policy` | 51 | |
| `_require_client` | 47 | |
| `_normalize_mode` | 46 | |
| `_dump` | 39 | |
| `_identity` | 31 | |
| **all six** | **279 definitions** | **1,776 lines** |

TASK-32808.6 framed this as "a scaffold across ~45 services" and proposed one base class. Two measurements
say that framing does not survive contact with the tree.

### 1. No service uses the whole scaffold

Across the 47 files named `*scope_service*.py`:

| defines | files |
|---|---:|
| 6 of six | **0** |
| 5 of six | **0** |
| 4 of six | 6 |
| 3 of six | 38 |
| 2 of six | 2 |
| 1 of six | 1 |

**Not one file defines all six**, and the mode is three. So there is no common scaffold to lift — there is a
*family* of overlapping partial duplications. A base class offering all six would give every subclass three
methods it does not use and did not ask for, and inheritance is the most expensive way to deliver an unused
method: it cannot be removed later without checking 47 subclasses.

### 2. The naming convention hides two thirds of the problem

| group | files | Interop | core |
|---|---:|---:|---:|
| named `*scope_service*.py` | 47 | 31 | 16 |
| **everything else** | **102** | 40 | 62 |
| total | **149** | 71 | 78 |

The original scope counted only the first row. **102 further files carry the same helpers** and would be
untouched by a scope-service base class — so the consolidation would leave the majority of the duplication
in place while appearing to have addressed it. That is the failure mode this review already documented
elsewhere: a consolidation task closing without the census moving.

### 3. The mode default cannot be inherited

`_normalize_mode` bodies, by which literal they mention:

| body mentions | count |
|---|---:|
| `SERVER` only | 25 |
| `LOCAL` only | 16 |
| neither literal | 5 |

Services disagree about the default, and four `Interop`/`server_*` twin pairs are **inverted** relative to
each other. A base class would have to pick one, and picking either silently changes the behaviour of the
other group. The dispatch predicate has the same problem in a sharper form: two documented rules currently
disagree about its **fail-direction**, and nine services thread it in five mutually incompatible shapes.

## Decision

**Consolidate per helper, into `tldw_chatbook/runtime_policy/`, in ascending order of risk. Do not
introduce a base class.**

1. **`_maybe_await` (65) and `_identity` (31) first.** These are pure, take no policy, and have no
   fail-direction. They are safe to make one shared function each, and together they are 96 of the 279
   definitions. This is the "adopt it and move on" tier.
2. **`_dump` (39) next**, once its serialisation variants are diffed; expect a small number of deliberate
   exceptions, recorded rather than forced.
3. **`_require_client` (47)** after the three runtime `TLDWAPIClient` importers are annotated `-> Any`.
4. **`_normalize_mode` (46) and `_enforce_policy` (51) LAST, and only after a separate decision** settles
   the mode default and the dispatch fail-direction. Until then they stay where they are. **A wrong
   fail-direction is a security regression, not a style regression** — this is the one place in the family
   where consolidating carelessly is worse than not consolidating at all.

Each converted helper gets a census row and a shrink-only guard in the repo's own shape
(`scripts/check_*.py` wired into `preflight.sh`), because the duplication census for this repo has not moved
across two reviews and ten fix-stream PRs. A consolidation without a guard has already been shown here to
regrow — a consolidation PR shipped four fresh byte-identical copies of `_coerce_int` within 48 hours of the
review that catalogued it.

## Consequences

- The 1,776 lines come out in four tranches, not one PR. Tranche 1 is ~96 definitions and carries almost no
  risk; tranche 4 may never happen, and that is an acceptable outcome.
- The scope **widens** from 47 files to 149. That is the point: the narrower number was an artifact of a
  naming convention, not of the code.
- No base class means no inheritance contract to get wrong, and no subclass inheriting methods it does not
  use.
- `TASK-32808.6` is superseded. Its stated scope is wrong in five measurable ways, already recorded in
  TASK-32903.

## Alternatives rejected

**One base class for the 47 scope services** — the original proposal. Rejected because no service uses all
six helpers, the mode default cannot be inherited without changing behaviour for one of two groups, and it
would leave 102 files untouched.

**A mixin per helper** — rejected for tranche 1: for a pure function with no state, a mixin buys an import
edge and an MRO entry over a plain function call, and the review already rejected the same shape for a
10-copy `relocate` cluster on exactly those grounds.

**Do nothing** — defensible for tranche 4, not for tranche 1. 96 definitions of two pure functions is the
cheapest 1,000-odd lines this codebase will ever delete.
