# Confirmatory steady-state three-turn Console benchmark

This staging directory contains the five review-bound artifacts for TASK-20010's
separately pre-registered steady-state confirmation. The run mounts the real
Console, types three prompts through its composer, exercises the real local
`load_tools` and confined `fs_write` path, and compares the pinned control with
the same candidate used by the original TASK-20009 benchmark.

The definitive attempt is `attempt-0001`. Its raw SHA-256 is
`2cdda7f369979fb1ac65f4f668bfd8ad4a28b4d2502f081eb767dc766d944a8c`.
The computed verdict is `inconclusive`; review and publication cannot change
that result.

## Prerequisites and immutable identities

- Run from the repository root at clean harness revision
  `1275ffc39f81c38821fdf1c6b3cae42da53287ba`.
- The tracked runner SHA-256 must be
  `6591f6755897c73d03abe1e7481659f6f28a6260e25f63d57e88a895659ca9a2`.
- Serve
  `gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf` through the
  dedicated loopback endpoint `http://127.0.0.1:9099` with one slot.
- The control is
  `5f720a40417eaa78f33619d5cbc82effc470104b`; the candidate and original
  statistics runner revision are
  `eb8225a32f88ea43c337aff99804d360384e7668`.
- The digest-pinned original runner SHA-256 is
  `fbca69703b771f7b7b27fa78ef9bf095fb30712435743877e20fcb01bb6d06ae`.
- Review digest preparation uses harness revision
  `95e9250f2d816305baceee4c999d658500c31981`, whose tracked runner SHA-256 is
  `00eecbba4b748951be4b3591bf689a6a400591e09da9f288c0b19384c4d2015e`.
  This post-acquisition revision changes only artifact-set validation so the
  authenticated empty `samples/`, `control/`, and `candidate/` tombstone
  namespaces are accepted as validated exclusions; they are never hashed.

The original benchmark did not retain a digest of the GGUF model weights.
This confirmation therefore verifies the endpoint-reported alias and complete
retained server/runtime contract, but cannot claim historical byte identity for
the model file.

## Provider preflight

Use a fresh empty output directory:

```bash
confirm_preflight_root="$(mktemp -d /tmp/tldw-console-three-turn-preflight-task-20010.XXXXXX)"
.venv/bin/python Tests/Performance/run_console_three_turn_profile.py \
  --endpoint http://127.0.0.1:9099 \
  --model gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf \
  --candidate-sha eb8225a32f88ea43c337aff99804d360384e7668 \
  --output-root "$confirm_preflight_root" \
  --preflight-only
```

The retained preflight returned `status: ready`, the exact model alias, one
model, and visible completion content.

## Disposable phase-plumbing smoke

The smoke is deliberately non-statistical and cannot be reviewed or promoted:

```bash
confirm_smoke_root="$(mktemp -d /tmp/tldw-console-three-turn-confirmatory-smoke.XXXXXX)"
.venv/bin/python Tests/Performance/run_console_three_turn_profile.py \
  --endpoint http://127.0.0.1:9099 \
  --model gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf \
  --candidate-sha eb8225a32f88ea43c337aff99804d360384e7668 \
  --iterations 1 \
  --burn-in-blocks 1 \
  --campaign-root "$confirm_smoke_root"
```

It runs three warmups, three excluded burn-in conversations, and three
measured plumbing conversations.

## Official acquisition

Create one absent campaign root and run exactly one official attempt:

```bash
confirm_campaign_root="$(mktemp -d /tmp/tldw-console-three-turn-confirmatory-task-20010.XXXXXX)"
.venv/bin/python Tests/Performance/run_console_three_turn_profile.py \
  --endpoint http://127.0.0.1:9099 \
  --model gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf \
  --candidate-sha eb8225a32f88ea43c337aff99804d360384e7668 \
  --iterations 30 \
  --burn-in-blocks 5 \
  --campaign-root "$confirm_campaign_root"
```

The schedule is fixed before provider preflight: three warmups; five complete,
balanced burn-in blocks (15 conversations); and 30 fresh measured blocks (90
conversations). Every conversation makes five provider calls. Burn-in is
contract validation only and is excluded before the digest-verified original
runner validates or summarizes the measured set.

## Verification before review

Set the campaign root created above, then verify the immutable source inputs,
raw hash, row cardinality, and JSON syntax:

```bash
attempt_root="$confirm_campaign_root/attempts/attempt-0001"
shasum -a 256 \
  Tests/Performance/run_console_three_turn_profile.py \
  Docs/superpowers/qa/console-three-turn-real-provider/README.md \
  Docs/superpowers/qa/console-three-turn-real-provider/real-provider-three-turn-summary.md \
  Docs/superpowers/qa/console-three-turn-real-provider/real-provider-three-turn.manifest.json \
  Docs/superpowers/qa/console-three-turn-real-provider/real-provider-three-turn.raw.jsonl \
  Docs/superpowers/qa/console-three-turn-real-provider/real-provider-three-turn.summary.json
shasum -a 256 "$attempt_root/real-provider-three-turn.raw.jsonl"
wc -l "$attempt_root/real-provider-three-turn.raw.jsonl"
jq -e . "$attempt_root/real-provider-three-turn.manifest.json" >/dev/null
jq -e . "$attempt_root/real-provider-three-turn.summary.json" >/dev/null
jq -ce . "$attempt_root/real-provider-three-turn.raw.jsonl" >/dev/null
```

Expected original evidence hashes, in the order above after the runner, are
`724be0f80eff3c9a2eced35b86ae4ce2e6f9a7524d44016cd3f49b61752bd491`,
`fdb4528bd82a33f244b4e6fbcfe3b739bd2374006cfea2df878f2e0d27a7d5c2`,
`f5dec9153845b585d32660ca87f8d4aef7ad31be4dc431bb52e64fdc29187bb6`,
`82150cd55ba701b5a2680f87fce43b15676004fc1609f477f458a7abb2078319`,
and `edec5d347427748e26c93d21da7ecf121cccedb41ea7d304fb6cdad684f3668a`.
The official raw file has 222 lines: 108 sample-start plus 108 terminal sample
rows, and three protocol-preflight start/result pairs. The raw hash must equal
the full value recorded at the top of this README and in the manifest lineage.

Recompute the review-bound five-file digest only after this README and the
human report are final:

```bash
.venv/bin/python Tests/Performance/run_console_three_turn_profile.py \
  --campaign-root "$confirm_campaign_root" \
  --campaign-action digest \
  --attempt-id attempt-0001
```

The command hashes the sorted canonical-JSON mapping from each of the exact
five reviewed filenames to its SHA-256. `reviews/review-001.json` is a
fail-closed template and is intentionally outside that digest. A separate
reviewer must replace its placeholder fields, bind an `approved` or
`changes_required` decision to the emitted digest, and independently confirm
privacy before any registration or promotion.

## Evidence inventory

- `real-provider-three-turn.raw.jsonl`: 111 child-start rows, three protocol
  preflight results, and 108 terminal sample rows with content-free timing,
  usage, tool, ownership, revision, and phase evidence.
- `real-provider-three-turn.manifest.json`: the exact schedule, exclusion rule,
  revisions, harness identities, protocol-equivalence result, sanitized server
  and runtime contracts, listener identity/resources, attempt lineage, and
  original-evidence hashes.
- `real-provider-three-turn.summary.json`: the digest-verified original module's
  measured-only medians, nearest-rank p95 values, paired bounds, claims, and
  computed verdict, plus only the excluded burn-in count and contract status.
- `real-provider-three-turn-summary.md`: conservative human interpretation that
  preserves the computed verdict and makes no burn-in performance claim.
- `README.md`: this reproduction, verification, and provenance guide.

No reviewed artifact contains prompt, response, tool-result, or generated-file
bodies; credentials, headers, environment dumps, absolute workstation paths, or
secret-bearing command lines; or personal workspace content.
