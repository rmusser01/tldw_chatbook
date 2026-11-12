# Two verification traps hit during this program, both by me

Recorded because both produce **false green**, and one of them is already in this repo's own lessons file
and recurred anyway.

## 1. `cmd | tail` reports the exit code of `tail`

I ran `PYTHON=<venv> ./scripts/preflight.sh 2>&1 | tail -40; echo "PREFLIGHT_EXIT=$?"` and reported
`PREFLIGHT_EXIT=0`. **That 0 was `tail`'s.** `preflight.sh` could have exited non-zero and I would have
printed the same thing. The same shape caught a subagent on `pytest ... | tail -12`, which additionally
truncated the summary line away, so its full-directory claim was unsupported.

This is `lessons-*.md`'s `preflight | tail` trap. It is in writing, and it still happened — which is itself
a data point about how much a written lesson buys you without a check.

**Fix:** capture, then inspect. `out=$(cmd 2>&1); rc=$?` — or read `${PIPESTATUS[0]}` / `${pipestatus[1]}`
in zsh. Re-run without the pipe, all seven fix branches:

```
guards     preflight_rc=0   preflight: all derived-artifact checks passed.
crashes    preflight_rc=0   ...
security   preflight_rc=0   ...
dataloss   preflight_rc=0   ...
bounds     preflight_rc=0   ...
dead       preflight_rc=0   ...
p2b        preflight_rc=0   ...
```
The conclusions held. The evidence for them did not, until this re-run.

## 2. zsh does NOT word-split unquoted parameters

```zsh
RAT="Tests/Architecture/test_a.py Tests/Architecture/test_b.py"
python -m pytest $RAT -q        # zsh passes ONE argument containing a space
```
pytest found no such file, collected nothing, and my `grep '^FAILED'` returned empty — which I read as
**"0 failing node ids"** and very nearly reported as "the ratchet is clean everywhere". In bash this would
have worked; zsh only splits on whitespace with `${=VAR}` or an array.

Note the failure direction: a broken command produced **silence**, and silence read as success. Same family
as the `zsh git rev:path` trap already recorded for this repo.

**Fix:** pass arguments explicitly (`"$A" "$B"`), or use an array. And treat an empty result from a
*negative* grep as suspicious until the command is proven to have run — a count of zero should be
corroborated by a positive signal, here the summary line `5 failed, 58 passed`.

## The corrected ratchet evidence

Comparing **node-id sets**, not counts — which is what I required of every agent and had been accepting
counts for myself:

```
dev baseline: 5 failing node ids
guards / crashes / security / dataloss / bounds / dead / p2b
    -> SET IDENTICAL to dev (5 each)
```

## The generalisable point

Both traps share a shape with the review's own headline finding: **a check that cannot fail reads exactly
like a check that passes.** `check_timestamp_writers.py` printed `0 site(s) ... OK` over 101 live writers;
`grep '^FAILED'` printed nothing over a pytest that never ran; `| tail` printed 0 over an unknown exit code.

The discipline that catches all three is the same one this program required of every fix: **make it fail
once on purpose.** A verification you have never seen produce a red is not yet evidence.
