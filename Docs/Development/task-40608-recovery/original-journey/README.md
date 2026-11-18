# Original Windows journey reproduction

The frozen `test_task40608.py`, `task40608_trace.py`, and `summarize.py` preserve
the original README and approved-edit journeys. Keep their bytes unchanged in
both replays. `guarded-windows-runner.zip` preserves the existing guarded replay
helpers; its historical host paths are examples, not defaults for a new host.
The source-only pair driver here uses local configuration and contains no
private profiles, source inventories, run receipts, or machine paths.
Its current qualification and recovery rules supersede historical orchestration
notes in the archived helper bundle.

For the completed nonqualifying foreground/approval-v2 diagnostic and portable
source recovery, see [DIAGNOSTICS.md](DIAGNOSTICS.md),
[the 84 KiB source archive](windows-foreground-approval-v2.zip) and its
[SHA256](windows-foreground-approval-v2.sha256). The known v2 ancestry-probe defect
is retained and documented; no qualified paint latency is claimed.

ADR required: no new ADR. This is test-only reproduction of the existing
[ADR-126](../../../../backlog/decisions/126-complete-local-backup-and-recovery.md)
and [ADR-230](../../../../backlog/decisions/230-console-send-preparation-and-io-ownership.md)
verification boundaries; it changes no product behavior.

## Prepare and pin the two replays

1. Retain the exact original `before` overlay, including its original controller
   changes. Build the candidate from a committed product delta over that overlay;
   do not substitute the current checkout's tests or erase overlay contributions.
   Keep the original QA and trace byte-identical. Record the exact source hashes,
   delta, committed identity, original private-profile helper, and dependency
   inventory in each replay's existing `replay-manifest.json`.
2. Baseline product remains original local execution. Candidate process selection
   is the documented lab-only App constructor change; the PR default remains
   local. Install the same **guard-only** observer into both. The candidate's
   isolated child must enter the original `only_owned_server(actual_port)` guard
   before the canonical backend worker. Never use count-monitoring timings for
   qualification. Preserve all source, ownership and native retirement checks.
3. Pin the same `packages/tldw_profile_core/src` files and installed dependencies.
   Each replay owns its import environment; both the UI/private-profile subprocess
   and the isolated backend must resolve the intended source roots. Keep private
   profile parents short and create them with the existing native helper.
4. Copy `pair_driver.example.json` into the local lab. Paths are relative to that
   configuration file; use actual reviewed manifest and commit
   SHA256 values. `files` must explicitly pin both manifests and may pin additional
   local evidence. The manifests pin runner helpers; the driver also checks the
   original QA/trace/summary, observer source and equal profile-core inventories.
   A prior diagnostic receipt may be retained as additional evidence; no separate
   passing screening run is required. Each actual baseline/candidate sample must
   pass both original journeys and its guard receipts to count toward a pair.

The original baseline runner is always exhaustive and has no `-Exhaustive`
switch (`exhaustive_switch: false`). The candidate guarded runner requires that
switch (`true`). Do not use `false` to bypass verification: it is compatible only
with the pinned original baseline runner that already verifies all sources before
and after each run. The candidate setting cannot disable exhaustive verification.
The driver passes the configured `private_parent` explicitly, overriding archived
host defaults. The example configuration deliberately cannot launch as supplied.

## Review, run, and recover

From this directory, with Python 3.12+ and the prepared local environment:

```powershell
python -I -B pair_driver.py plan --config /path/to/local/pairs.json
# One actual Windows owner launches this only after reviewing the plan:
python -I -B pair_driver.py run --config /path/to/local/pairs.json --batch q1
python -I -B pair_driver.py report --config /path/to/local/pairs.json --batch q1
# After the existing native runner establishes cleanup following interruption:
python -I -B pair_driver.py run --config /path/to/local/pairs.json --batch q1 --resume
```

The exact sequence is **B1,C1,C2,B2,B3,C3**. A fresh two-character batch prefix
produces short private labels with a fresh attempt suffix, avoiding the observed Windows MAX_PATH
failure from descriptive private labels. Attempts never reuse labels or outputs.
The driver invokes the unchanged original fixture through the pinned runners,
scopes the explicit observer plugin, and preserves their exit codes and receipts.

Each attempt's append-only ledger fsyncs start/finish and retains commands, exact input
and artifact hashes, timing observations, eligibility and failures. It creates a
persistent active marker. Crash, interrupted record or failed native receipt
leaves that marker and evidence intact. After the existing native runner establishes
actual cleanup, `--resume` preserves fully completed pairs and retries each
incomplete pair from its first preset side with new labels and a new attempt
ledger. For example, a crash after C2 but before B2 retries **C2,B2**, then runs
**B3,C3**; the completed B1,C1 pair stands. This does not require six new runs.
Process absence alone is not retirement, and `--resume` does not manufacture a
cleanup receipt. No extra screening or separate approval step is introduced.
Both samples of a retained pair must come from one uninterrupted driver attempt,
so samples across a reboot/retry are never spliced into one pair. All attempts
must retain the same configuration/source pins; a prior positive pair is not
replaced merely to obtain a faster timing. `report` reads incomplete records
without launching a workload. Full source
verification is required before and after **every** qualifying original run.

When selecting a different candidate, create a new configuration with its exact
committed identity, replay manifest and runner/helper pins. Keep previous
configurations/results and use a new batch prefix.
The driver never discovers `HEAD`, updates pins, combines candidate commits, or
relabels an earlier failed attempt. The configuration hash is fixed for the batch.

## Qualification semantics

Use the original completion rule: `reply_completed`, falling back to `reply_seen`.
Application seconds are actual Send-to-completion time minus **only** the original
observed disclosure/approval human-wait windows. No application work or provider
time is removed. Original summary output, raw elapsed seconds, removed waits, and
the common `reply_seen` endpoint are retained separately. Thresholds use unrounded
application seconds, not rounded display values or estimated speedups.

Each journey requires **at least 30% median reduction** across its three pairs.
**Every one of the six candidate application times must be strictly below 5.0
seconds.** Baseline has no absolute speed gate. Per-pair percentage differences
are informational, not additional pass conditions. A complete sequence that
misses either threshold remains a failed qualification with all evidence intact.

The small arithmetic/recovery checks launch no product or native worker:

```powershell
python -I -B test_pair_driver.py
```

They cover the strict five-second boundary, median-only percentage gate,
preserved completed pairs, whole-pair retries, unchanged pins and refusal to mix
samples across attempts. They do not qualify the real journeys.
