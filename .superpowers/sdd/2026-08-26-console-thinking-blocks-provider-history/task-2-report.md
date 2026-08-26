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
| `llama_cpp`, `local_llamacpp` | Displayable | Start-anchored `<think>`/`<thinking>` capture | v1 | Yes |
| `vllm`, `local_vllm` | Displayable | Start-anchored `<think>`/`<thinking>` capture | v1 | Yes |
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

Static commands:

```text
../../.venv/bin/python -m ruff format --check tldw_chatbook/Chat/console_provider_gateway.py tldw_chatbook/LLM_Calls/hosted_chat.py tldw_chatbook/LLM_Calls/moonshot.py tldw_chatbook/LLM_Calls/zai.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_kimi_zai_native_tools.py Tests/Chat/test_kimi_zai_provider_contract.py Tests/Chat/test_local_adapter_thinking_dispatch.py Tests/Chat/test_local_thinking_wire_formats.py Tests/LLM_Calls/test_hosted_chat.py Tests/LLM_Calls/test_moonshot.py Tests/LLM_Calls/test_zai.py
../../.venv/bin/python -m ruff check tldw_chatbook/Chat/console_provider_gateway.py tldw_chatbook/LLM_Calls/hosted_chat.py tldw_chatbook/LLM_Calls/moonshot.py tldw_chatbook/LLM_Calls/zai.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_kimi_zai_native_tools.py Tests/Chat/test_kimi_zai_provider_contract.py Tests/Chat/test_local_adapter_thinking_dispatch.py Tests/Chat/test_local_thinking_wire_formats.py Tests/LLM_Calls/test_hosted_chat.py Tests/LLM_Calls/test_moonshot.py Tests/LLM_Calls/test_zai.py
git diff --check 4e5a04c2d2
```

Static result: formatting, Ruff, and diff checks passed.

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
