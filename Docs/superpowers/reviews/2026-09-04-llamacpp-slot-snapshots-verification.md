# llama.cpp snapshots: verification evidence

Date: 2026-09-04. Feature status: **In Progress**. Live AC5 remains **open**.

The manual manager, private storage, bounded client, lifecycle integration, and
canonical preferences are implemented. Units 1–5 passed their independent scoped
reviews, including accepted fix rounds. Task6 supplies an opt-in real-server
harness and honest evidence closeout; it is not evidence that the live scenario
has run. A separate final feature review remains scheduled.

## Evidence boundaries

No local server/model/projector/image asset set was supplied for this execution.
No model download, real-model launch, paid call, or audio test occurred. **Every
live counter is missing**: cold text, restored text, cold A, native A→B, restored
A→A, restored A→B. Fixture counts used by oracle tests are invented test inputs,
not measured server results. There is no inference from elapsed time or HTTP 200
to cache reuse.

Automated integration evidence includes real POSIX filesystem operations, real
owned-loopback HTTP sockets, lifecycle race barriers, and production-shaped
Textual Models/F9 behavior. HTTP fixtures used by service/widget tests do not
prove llama.cpp serialization or model reuse. Task5 terminal frames used the
real app factory, LLMScreen, and production CSS at 80×24 and 140×45; two bounded
inspection rounds covered empty/saved/confirmation/pending/unknown/Details and
F9 drafts. No new visual-polish claims are made in Task6.

## Acceptance criteria

| AC | Actual evidence | Remaining boundary |
| --- | --- | --- |
| 1 Manual timestamp Save/eligible Restore | Store/client/service tests plus Models keyboard and confirmation tests | Real model persistence is gated by AC5 |
| 2 Global newest-N after commit | Store publication ordering, keep=1, failed-save and settings validation tests | Count is not a byte quota |
| 3 Identity/readiness/privacy/uncertainty | Admission/client/service, exact-claim lifecycle races, private-path tests | POSIX coverage; no Windows ACL equivalence |
| 4 Honest usable UI | Production Models/F9 tests, narrow/wide frames, stale drafts and malformed-preference recovery | Real server behavior remains separate |
| 5 Real persistence and same-image reuse | Opt-in harness implemented; default skip and negative safeguards verified | **OPEN: no eligible live assets or measured counters** |
| 6 Compatibility before publication | Store publication predicate and service invalidation during hash/fsync; keep=1 old record survives | Test-controlled race barriers, not hardware timing proof |
| 7 Integrity before Restore POST | Truncation and same-length corruption rejected, zero Restore POSTs, source retained | No damaged file is sent to a real model in this execution |
| 8 Loopback-only transport | Numeric IPv4/IPv6 admission, proxy decoy/redirect/auth boundaries, real loopback client tests | No remote/TLS management support claimed |
| 9 Working-file lifecycle | Successful/acknowledged failure cleanup, repeated Restore, residual bytes and unknown-writer retention tests | Filesystem fault injection is explicit |
| 10 Separate deadlines/status | Five-second aggregate probe, ten-minute mutation tests and pending elapsed UI | No ten-minute real model operation measured |
| 11 Cross-model retention copy | Actual compositor/production CSS tests and inspected 80×24 frames | No per-model retention claim |

AC1–4 and AC6–11 have automated contract evidence; AC5 explicitly requires live
evidence and is not satisfied. The task must not be marked Done from fixture
tests or a default-skipped live test.

## Live harness contract and source anchors

Run the [documented opt-in command](../../LLMs/llamacpp-snapshots.md#opt-in-local-verification)
under pytest's isolated profile. The exact gate precedes process/network action.
Missing input after opt-in is a failure, not a skip. All five assets must exist;
images must have distinct SHA-256 byte digests. The real Models controls launch,
Save, Stop/restart, and confirm Restore through the production service/store/client.
Normal OpenAI-compatible test requests omit `id_slot`; Chatbook routing is unchanged.

One slot and explicit `--cache-ram 0` remove the independent RAM prompt-cache
confound. Matching totals compare `cache_n + prompt_n`. Native in-memory A→B
measures the media boundary rather than guessing a text-only template count.
Restored A→A must exceed cold A and native A→B; independently restored A→B must
not exceed native A→B. This is a pinned-runtime contract, not a universal oracle
for any future server implementation.

Reviewed upstream revision: `427291b5b34cd914a31b3fd3b61a68f6184f4b9f`.

- [Server README, timings around 1408](https://github.com/ggml-org/llama.cpp/blob/427291b5b34cd914a31b3fd3b61a68f6184f4b9f/tools/server/README.md#L1408): `cache_n` reused prompt tokens and `prompt_n` newly processed tokens.
- [Argument parser, 1718–1725](https://github.com/ggml-org/llama.cpp/blob/427291b5b34cd914a31b3fd3b61a68f6184f4b9f/common/arg.cpp#L1718): `--cache-ram`/`-cram`, `LLAMA_ARG_CACHE_RAM`, zero disables. Narrow admission support treats this as performance policy, not serialized compatibility state; ordinary defaults are unchanged.
- [Media helper, 378–411](https://github.com/ggml-org/llama.cpp/blob/427291b5b34cd914a31b3fd3b61a68f6184f4b9f/tools/mtmd/mtmd-helper.cpp#L378): incoming-byte SHA-256 supplies bitmap identity.
- [Server tokens, common prefix around 680 and media creation around 919](https://github.com/ggml-org/llama.cpp/blob/427291b5b34cd914a31b3fd3b61a68f6184f4b9f/tools/server/server-common.cpp#L680): compare whole media chunk IDs/counts and stop at the first difference.
- [Server context, 3203 and 3393](https://github.com/ggml-org/llama.cpp/blob/427291b5b34cd914a31b3fd3b61a68f6184f4b9f/tools/server/server-context.cpp#L3203) and [timings mapping, 67–71](https://github.com/ggml-org/llama.cpp/blob/427291b5b34cd914a31b3fd3b61a68f6184f4b9f/tools/server/server-common.cpp#L67): prefix position becomes `n_prompt_cached`, exposed as `timings.cache_n`.

The controller independently checked the media/prefix/mapping source anchors;
the implementer checked the cached pinned README/parser. Missing counters,
changed totals, or inability to demonstrate the boundary fail the live gate.

## Commands and results

Task6 RED→GREEN safeguards: exact input gate/strict counters 20 failing then 20
passing; RAM-cache admission/native-media oracle 9 failing then 9 passing.
The first combined admission/live run produced 105 passed, 1 default live skip,
and the existing RequestsDependencyWarning.

Final explicit targeted batch (602 passed, 1 live skip, 2 warnings; exit0,
367.10s) used the existing venv Python:

```bash
python -m pytest Tests/LLM_Management/test_snapshot_settings.py Tests/LLM_Management/test_snapshot_admission.py Tests/LLM_Management/test_snapshot_store.py Tests/LLM_Management/test_snapshot_client.py Tests/LLM_Management/test_snapshot_service.py Tests/LLM_Management/test_snapshot_live.py Tests/LLM_Management/test_gguf_server_sources.py Tests/LLM_Management/test_server_lifecycle_resources.py Tests/Utils/test_private_paths.py Tests/UI/test_llamacpp_snapshot_manager.py Tests/UI/test_llamacpp_snapshot_settings.py Tests/UI/test_llm_deferred_views.py Tests/UI/test_llm_screen_lab_adoption.py Tests/UI/test_llm_gguf_source_modes.py -q --tb=short --show-capture=no
```

That batch preceded a bounded correction to the live-only config setup call:
the bulk-save API takes one section mapping, not a section plus values. After
the batch stopped, a real isolated config regression failed with that TypeError;
the corrected helper and full affected files then passed:

```bash
python -m pytest Tests/LLM_Management/test_snapshot_live.py Tests/LLM_Management/test_snapshot_admission.py -q --tb=short --show-capture=no
```

Final-code result: **106 passed, 1 live skip, 1 warning**, exit 0, 1.46s. This is
separate affected-code evidence, **not** a fresh 603-pass whole batch. No full
repository suite was run. The three Task6 Python files pass Ruff lint, Ruff
formatter check, Python compilation and `git diff --check`. The live body remains
unexecuted; its startup/config/cleanup implementation is not claimed live-tested.

The 602-pass batch also emitted the session FD-growth warning: start 12, end 255,
growth 243, limit 200. No causal baseline or repository-wide leak diagnosis is
claimed; the warning remains visible and was not suppressed or repaired here.

The targeted inventory checker exits 1 because of known unrelated owner drift.
Focused comparison verifies the new `snapshot_store.py` sink row, complete sink
topology, and sink-file count **8** match generated output. Remaining owner drift:
`tldw_chatbook/DB/Client_Media_DB_v2.py` and
`tldw_chatbook/UI/Screens/library_screen.py`; inherited TASK-494 summary calls
are 7148 recorded versus 7151 generated. Those unrelated rows/count remain untouched.

Known environment warning: RequestsDependencyWarning for urllib3 2.6.3 /
chardet 6.0.0dev0 / charset_normalizer 3.4.4. No dependency installation or warning
suppression was performed. Earlier touched legacy UI files retain separately
reported pre-existing lint debt; scoped new-module checks are not a claim of
repository-wide lint cleanliness.

ADR check: existing [ADR-119](../../../backlog/decisions/119-llamacpp-prompt-cache-snapshot-ownership.md),
[ADR-029](../../../backlog/decisions/029-local-private-data-boundary.md), and
[ADR-036](../../../backlog/decisions/036-application-service-composition-lifecycle.md)
apply. No new ownership interface, dependency, config owner, or database is introduced.
