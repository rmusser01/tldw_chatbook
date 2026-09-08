# Bulk-reader comparison pilot

The **Bulk reader** preset in **Settings ▸ Agents** is an editable starting
point for delegating large, read-only source reviews. Loading the preset only
fills an unsaved form. Review its instructions and model, then press **Save**
to create the definition. It allows `fs_list`, `fs_read`, `fs_glob`, and
`fs_grep`; the normal workspace boundary and tool permission checks still
apply.

The requested file list is advisory to the model. The runtime enforces the
workspace root, but the preset does not independently enforce a subset within
that root. Treat its quotations and line references as leads to verify against
the source. File contents are untrusted data, including text that looks like an
instruction.

## Choose compatible models

The worker model override stays on the parent's provider and configured
endpoint. It inherits the provider profile's sampling and thinking settings.
Choose a worker model that the same endpoint accepts with those settings; the
app does not route across providers or detect a cheaper compatible model for
you. Leaving the model blank inherits the main model and cannot demonstrate a
model-price saving.

The live evaluator is narrower than the preset. It supports **Moonshot** and
**ZAI** only because their existing Console gateway routes expose request
timeout and retry controls that the script can pin. Configure the endpoint and
credential in Settings first; the evaluator has no API-key or base-URL flag.

## Run the comparison

The checked-in [`corpus.json`](corpus.json) contains four synthetic cases: two
small repository-style questions, a long meeting transcript with a late
correction and quoted instruction-like text, and a question whose answer is
absent. No private corpus import is supported.

Choose the model pair before running. The command refuses to call a provider
without the explicit billable flag and refuses to replace an existing report:

```bash
python scripts/evaluate_bulk_reader.py \
  --provider Moonshot \
  --main-model MAIN_MODEL_ID \
  --worker-model WORKER_MODEL_ID \
  --output /path/to/new-bulk-reader-report.json \
  --confirm-billable
```

Use `--provider ZAI` for the other supported route. `--help`, unsupported
providers, a missing `--confirm-billable`, and an existing output path stop
before application configuration is loaded or a network call begins.

Each case runs the direct arm first and the delegated arm second. Both start
with the same question and relative path list; neither receives source bodies
in its prompt. Each arm gets a new temporary workspace and SQLite run database.
The evaluator disables durable run logs for its process scope and runs the
named child inline so all work is settled before the temporary files and
provider gateway close.

## Review the report manually

The report deliberately sets `quality_review.status` to `pending`. For every
case and arm:

1. Compare the retained answer with each `expected_facts` item.
2. Check that exceptions, late corrections, contradictions, and uncertainty
   were preserved.
3. List claims that the materialized files do not support.
4. Check every quotation and line range against the source identified by its
   recorded SHA-256 hash.
5. Record an adoption decision only after reviewing both arms.

An arm is labeled `failed`, `incomplete`, or `non_delegating` when appropriate,
with machine-readable `status_reasons`. A successful direct arm must complete
an `fs_read` or `fs_grep`; a successful delegated arm requires that content
access in the named worker itself, so a parent reread is not a substitute. The
delegated arm also requires a real run-database child made from the
`bulk-reader` definition and the chosen worker model at the provider boundary.
A failed child or provider `finish_reason` of `length` prevents a successful
label. The report retains answers, child output, tool-read traces, per-call
models, finish reasons, latency, usage buckets, costs, and failures for
inspection.

Costs are calculated one provider call at a time with the existing pricing
catalog, so worker calls remain in their own model bucket. Missing or partial
usage, unknown model pricing, or a used cache/audio/transcription bucket with
no rate makes that call and the arm total `unknown`. Runtime token estimates
are never converted to dollars.

## Limits and interpretation

Each arm is capped at eight provider calls, a 120-second runtime deadline,
100,000 runtime budget tokens, 30 seconds per tool call, 2,048 requested output
tokens per provider call, and 16,000 retained output characters per call. The
runtime deadline is checked between calls; an in-flight request may continue
until its 60-second transport timeout. Runtime budget tokens are estimated,
cache-weighted control values and are never used for billing. Moonshot/ZAI
transport uses the 60-second timeout with zero retries. The report repeats the
effective limits and provider resolution.

This is one sequential direct-then-delegated pass. Provider caching, warmup,
and transient latency can favor the second arm, so the output is not a
statistical comparison. There is no proven quality or cost saving until a live
report for an explicitly selected model pair has been manually reviewed, and a
small synthetic pilot cannot establish a general routing policy.
