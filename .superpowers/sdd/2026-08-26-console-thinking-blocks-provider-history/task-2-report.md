# Provider/History Task 2 Report

## Outcome

Added explicit, bounded provider thinking events without changing the existing
string and tool-call stream contract. Local llama.cpp/vLLM-compatible adapters
emit displayable deltas only from the review-clean start-anchored splitter;
Moonshot and Z.ai emit structurally content-free proprietary evidence only after
validated current-turn reasoning. Other adapters remain explicitly ignored.

Provider/model/protocol/source identity is frozen onto every evidence event.
Console resolution now exposes adapter-owned `may_emit_thinking` and round-trip
facts for foundation preflight without model-name inference. Confirmed terminal
capture failures are surfaced content-freely, including the stream-to-nonstream
fallback inspection path.

## Adapter Disposition

| Execution target | Disposition | Actual-turn evidence | Round trip | `may_emit_thinking` |
| --- | --- | --- | --- | --- |
| `llama_cpp`, `local_llamacpp` resolution explicitly marked displayable | Displayable | Start-anchored `<think>`/`<thinking>` capture | v1 | Yes |
| `vllm`, `local_vllm` resolution explicitly marked displayable | Displayable | Start-anchored `<think>`/`<thinking>` capture | v1 | Yes |
| The same local backends with a frozen ignored/non-thinking resolution | Ignored | Tags remain ordinary visible strings; no typed evidence | None | No |
| Moonshot/Kimi | Proprietary | Provider-validated terminal `reasoning_content` for the current turn | v1 | Yes |
| Z.ai/GLM | Proprietary | Provider-validated terminal `reasoning_content` for the current turn | v1 | Yes |
| All other or generic reasoning fields | Ignored | Never parsed as thinking evidence | None | No |

No event contains raw proprietary values, lengths, hashes, tokens, excerpts, or
canaries. No proprietary event is emitted without actual current-turn evidence.

## TDD Evidence

Initial RED slices established the missing event API, adapter policy, direct-local
splitter adoption, stream-boundary behavior, resolver mapping, and capability
invariants. The focused failures progressed through collection failures and then
expected behavioral failures before implementation. Review-driven RED cases also
reproduced four contract gaps: uncollected legacy direct-path tests, duplicated
Moonshot/Z.ai disposition knowledge, dropped non-streaming partial deltas on an
unclosed capture, and a failed fallback capture recorded as complete.

Representative RED commands:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/Chat/test_console_provider_gateway.py -k 'provider_thinking or proprietary or disposition' -q
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/Chat/test_console_provider_gateway.py -k 'direct and thinking' -q
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/Chat/test_console_provider_gateway.py -k 'terminal_capture_failure or fallback_capture_error' -q
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/LLM_Calls/test_hosted_chat.py Tests/LLM_Calls/test_moonshot.py Tests/LLM_Calls/test_zai.py -q
```

The slices failed for the expected missing API or contract behavior, then passed
after each minimal implementation/fix. The final exact provider matrix was:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_kimi_zai_native_tools.py Tests/Chat/test_kimi_zai_provider_contract.py Tests/Chat/test_local_adapter_thinking_dispatch.py Tests/Chat/test_local_thinking_wire_formats.py Tests/LLM_Calls/test_hosted_chat.py Tests/LLM_Calls/test_moonshot.py Tests/LLM_Calls/test_zai.py -q
```

GREEN result: 568 passed, 1 pre-existing Requests dependency warning, in 32.50s.

Splitter/capability regression command:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/Chat/test_llamacpp_think_splitter.py Tests/Chat/test_llamacpp_think_filter.py Tests/Chat/test_thinking_blocks.py Tests/Chat/test_console_thinking_persistence.py -k 'thinking_round_trip or thinking_preflight or splitter or filter' -q
```

GREEN result: 692 passed, 51 deselected, 1 pre-existing Requests dependency
warning, in 24.41s.

Original static commands with reproducible passing results:

```text
../../.venv/bin/python -m ruff check tldw_chatbook/Chat/console_provider_gateway.py tldw_chatbook/LLM_Calls/hosted_chat.py tldw_chatbook/LLM_Calls/moonshot.py tldw_chatbook/LLM_Calls/zai.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_kimi_zai_native_tools.py Tests/Chat/test_kimi_zai_provider_contract.py Tests/Chat/test_local_adapter_thinking_dispatch.py Tests/Chat/test_local_thinking_wire_formats.py Tests/LLM_Calls/test_hosted_chat.py Tests/LLM_Calls/test_moonshot.py Tests/LLM_Calls/test_zai.py
git diff --check 4e5a04c2d2
```

Original static result: Ruff lint and diff checks passed. No whole-file Ruff
format result is attributed to the original closeout. The authoritative
`git show 4e5a04c2d2:<path> | ruff format --check --stdin-filename <path> -`
replay exits 1 for both the production gateway and its large gateway test, proving
that both files were already whole-file non-green at the Task 2 base.

## Changed Files

- `tldw_chatbook/Chat/console_provider_gateway.py`
- `tldw_chatbook/LLM_Calls/hosted_chat.py`
- `tldw_chatbook/LLM_Calls/moonshot.py`
- `tldw_chatbook/LLM_Calls/zai.py`
- `Tests/Chat/test_console_provider_gateway.py`
- `Tests/Chat/test_kimi_zai_provider_contract.py`
- `Tests/LLM_Calls/test_hosted_chat.py`
- `Tests/LLM_Calls/test_moonshot.py`
- `Tests/LLM_Calls/test_zai.py`

The remaining plan-listed adapter suites were exercised unchanged as regressions.

## Self-review

- Events are frozen, slotted, bounded, and redact displayable text from repr; the
  proprietary evidence type has no content-bearing field or instance dictionary.
- Identity is resolved once and attached consistently to each event rather than
  inferred later from model names.
- Moonshot/Z.ai dispositions are consumed from provider-owned finish policies, so
  the gateway does not duplicate provider classification.
- Local captures preserve event ordering and return partial confirmed deltas before
  a content-free terminal error; visible text is never reclassified after capture.
- Compatibility remains intact for string, tool-call, and legacy complete-wrapper
  callers. Inspector capture accepts strings only.
- An independent review found no remaining Critical, Important, or Minor findings
  after the four review-driven fixes above.

## Remaining Concerns

None within Task 2. The Requests dependency warning is pre-existing. Successful
pytest runs also emitted sandbox temp-cleanup warnings after completion; these are
environmental and do not affect provider behavior.

## Commit

Planned commit message: `feat: emit explicit provider thinking evidence`

## Review Fix Round 1/5

Addressed both Important findings against `5bb018c7e3`.

- Direct llama.cpp streaming, non-streaming, fallback, and auxiliary paths now
  receive the frozen resolution's `thinking_stream_disposition`. The low-level
  compatibility methods default to ignored; only an explicit displayable value
  constructs the start-anchored splitter. An ignored/default manual resolution on
  llama.cpp or vLLM preserves `<think>` tags as ordinary visible strings, emits no
  typed evidence, and reports `may_emit_thinking=False`. Displayable resolutions
  retain the reviewed splitter behavior. The backend/execution key alone no longer
  controls output parsing, so thinking and non-thinking models may share one local
  server.
- Both event constructors now strictly reject high and low surrogate code points in
  provider, model, protocol, and source-format identities. Rejected inputs are
  cleared from constructor state before the content-free error is raised, do not
  survive in provider-module traceback locals or exception chaining, and valid
  astral identities remain accepted within the existing character bounds.

Fix RED command:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/Chat/test_console_provider_gateway.py -k 'identity_rejects_surrogates or identity_accepts_valid_astral or ignored_disposition_preserves_tags or default_llamacpp_resolution_preserves_tags' -q --tb=short --show-capture=no
```

Fix RED result: 19 failed and 8 passed. All 16 surrogate cases were accepted,
the low-level ignored-disposition API was missing, and both default-resolution
llama.cpp modes incorrectly emitted typed evidence instead of preserving tags.
The eight valid-astral controls passed.

Fix GREEN slices: 31 identity/bounds cases passed; then 40 direct llama.cpp,
vLLM, and event-contract cases passed.

Final exact provider command remained the Task 2 matrix above. Fix GREEN result:
597 passed with the one pre-existing Requests dependency warning in 34.72s.
Splitter/capability regressions remained 692 passed and 51 deselected with the
same warning in 25.45s.

Fix static result: Ruff lint and `git diff --check` passed for the two changed
files. Both the production gateway and its gateway test were already whole-file
formatter-non-green at `4e5a04c2d2`; no formatting drift is attributed to Task 2.
Review Fix Round 2 mechanically formatted the production gateway, which now passes
its current Ruff format check. The large gateway test retains its pre-existing
whole-file formatter baseline; new additions are locally formatted and lint-clean.

## Review Fix Round 2/5

Addressed the remaining auxiliary-normalization and static-evidence findings
against `33cbd31a5e`.

- Auxiliary completions now normalize local thinking solely from the frozen
  `thinking_stream_disposition`. Explicitly displayable vLLM/local-vLLM results
  use the review-clean start-anchored splitter and return only assistant-visible
  text; the auxiliary consumer intentionally discards the typed thinking event.
  Ignored resolutions on the same backends preserve literal tags. The auxiliary
  contract remains one-shot and non-streaming regardless of the captured session's
  streaming value; no provider-name or execution-key inference was added.
- Direct llama.cpp auxiliary results use the same final normalization boundary.
  A confirmed unclosed capture is surfaced as a content-free `ChatProviderError`
  instead of returning or logging captured text.
- `tldw_chatbook/Chat/console_provider_gateway.py` was mechanically Ruff-formatted.
  A whitespace-insensitive review confirmed that the only semantic production
  changes are the auxiliary normalization described above.

Fix RED command:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/Chat/test_console_provider_gateway.py -q -k 'auxiliary_vllm'
```

Fix RED result: 5 failed and 2 passed. Four displayable vLLM/local-vLLM cases
returned raw start-anchored tags, and the terminal capture-failure case did not
raise; both ignored-disposition controls passed.

Fix GREEN slice:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest Tests/Chat/test_console_provider_gateway.py -q -k 'auxiliary_vllm or auxiliary_direct_llama'
```

Fix GREEN result: 9 passed and 322 deselected.

The final exact provider matrix above passed 604 tests with the one pre-existing
Requests dependency warning in 33.98s. The first sandboxed attempt could not bind
the suites' loopback scripted servers; the exact command was rerun with loopback
permission and passed. Splitter/capability regressions passed 692 tests with 51
deselected and the same warning in 24.51s.

Final static commands and results:

```text
../../.venv/bin/python -m ruff format --check tldw_chatbook/Chat/console_provider_gateway.py
# 1 file already formatted
../../.venv/bin/python -m ruff format --check Tests/Chat/test_console_provider_gateway.py
# Exit 1: would reformat the pre-existing whole-file test baseline
../../.venv/bin/python -m ruff check tldw_chatbook/Chat/console_provider_gateway.py Tests/Chat/test_console_provider_gateway.py
# All checks passed
git diff --check
# Passed
```

The production format check, Ruff lint, and diff check are green. The test
module's known whole-file format baseline remains non-green and was not churned;
the new tests are locally Ruff-styled and lint-clean.
