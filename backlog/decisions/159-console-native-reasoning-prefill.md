# ADR-159: Native reasoning prefill has explicit provider and turn ownership

Status: Accepted — written spec approved by the user
Date: 2026-09-12
Task: [TASK-32521](../tasks/task-32521%20-%20Design-native-reasoning-prefill-for-Console.md)
Spec: [Console native reasoning prefill](../../Docs/superpowers/specs/2026-09-12-console-native-reasoning-prefill-design.md)
Plan: [Implementation tasks and verification](../../Docs/superpowers/plans/2026-09-12-console-native-reasoning-prefill.md)
Extends: ADR-063, ADR-064, ADR-090, ADR-092, ADR-095

## Context

Console already supports answer prefill, displayable thinking, and protected
provider reasoning continuation. A user-authored reasoning prefix is a distinct
input: a capable backend must continue at an unfinished reasoning boundary.
Merely accepting `reasoning_content` or replaying a prior thinking block does not
prove this capability. Existing answer-prefill code can disable thinking and
bypass tools, so adding another text field to that path would violate the feature.

The approved scope is native-only support across qualified targets, both
next-send and pinned lifetimes, and tool-enabled runs when that combination is
supported. Review identified ambiguous consumption timing, stale retries, queued
ownership, seed duplication during tool replay, destructive recovery, and unclear
portability. These require explicit shared contracts.

## Decision

1. Adapters own native-prefill capability and serialization for the exact target
   and execution combination. Results distinguish Supported, Unsupported, and
   Unverified. An enabled seed blocks when native support is unavailable; no
   ordinary-prompt fallback, silent omission, automatic endpoint/mode change, or
   implicit disabling of tools/thinking is allowed.
2. Conversation configuration, accepted submission, and provider continuation
   are separate owners. The conversation pin is durable. The next-send slot is
   live-session state with an opaque revision. An accepted manual/queued turn
   reserves a next-send revision exclusively and captures its seed and target.
3. Completion or a stop after dispatch consumes only the owned revision. Failure
   retains it for retry/release; undispatched cancellation releases it. Unknown
   delivery keeps recovery ownership. A newer armed value survives an older
   run's terminal callbacks. Retries use the failed turn's captured seed, while
   regeneration uses the current pin. Continue injects no fresh seed.
4. The first model call receives the seed once. Tool rounds replay completed
   reasoning under the existing exact assistant/tool owner as required by the
   adapter; they never inject the configured seed again. Parent configuration
   is not inherited by child agents, automatic follow-ups, forks, or new chats.
5. Displayable thinking contains generated continuation only. Exact adapter echo
   semantics separate any echoed input; generic string deduplication is forbidden.
   Required continuation may retain the seed inside its existing protected
   record. Optional replay cannot substitute a suffix-only display record for
   the full reasoning when the target requires the seed.
6. Count the native seed once in the actual input projection. Never report it as
   generated output or truncate it silently. Preserve accepted whitespace;
   reject blank-only, invalid-control, and over-4,000-code-point seeds.
7. Store the pin as a versioned `reasoning_prefill` object in conversation
   metadata containing version, exact text, and enabled state. Merge only this
   key under existing optimistic ownership. Preserve disabled text. The pin
   round-trips through lossless conversation export/import and Sync v2 under
   content protection, and is excluded from model/global defaults and fork
   inheritance. Temporary chats remain non-durable until promotion.
8. Unknown pin versions are preserved but not executed; malformed/unknown stored
   pins require explicit recovery. Persistent backends must support lossless
   round-trip before accepting the setting. Human-readable answer exports,
   search, summaries, speech, notices, and routine diagnostics exclude seed text.
9. Next-send configuration and accepted seed snapshots are memory-only, including
   exclusion from disk screen snapshots. If process loss prevents exact retry,
   recovery offers an explicit new generation or discard rather than rebuilding
   from today's pin. Existing opt-in request capture and mandatory continuation
   retain their established input-storage rules; this is not a no-retention
   promise for data actually sent to a model.
10. Context and `/reasoning-prefill` open one editor with independent Next send
    and Pinned values, enable controls, Save/Clear, origin guards, and dirty-draft
    recovery. The explicit Next Send preview identifies user-authored seed text.
    Existing response-prefill commands keep their meanings.

The spec defines queue reservation, terminal outcomes, exact recovery, and the
verification matrix. Runtime state transitions belong in focused shared code,
not duplicated between the editor, screen, direct gateway, and agent bridge.

## Alternatives considered

| Alternative | Reason rejected |
| --- | --- |
| Fall back to ordinary guidance | The user explicitly requested native prefill only. |
| Send `reasoning_content` to every compatible-looking endpoint | Acceptance does not prove continuation, and unsupported fields can be ignored or rejected. |
| Reuse answer-prefill behavior unchanged | It can disable thinking, bypass tools, and consume staged input before the agreed terminal boundary. |
| Read the current pin on retry or every tool round | Changes accepted intent and can duplicate or corrupt reasoning history. |
| Persist every next-send draft for transparent recovery | Expands an explicitly ephemeral configuration lifetime; honest recovery is preferred. |
| Put seed text into ordinary answer content or generated-thinking display | Misattributes user input and contaminates speech/export/history semantics. |
| Add a standalone historical seed column | Existing protected continuation owns the seed when replay requires it; editor configuration does not justify another historical body store. |
| Require clearing unsupported text | Disable preserves reusable work while making execution intent explicit. |

## Consequences and evidence

The UI can serve all providers without claiming all providers support the native
operation. Qualification remains bounded to tested and documented combinations,
including local template/server constraints, API mode, tools, and answer prefill.
No live qualification has been performed as part of this ADR. At least one native
target must be qualified before shipping enabled execution; tools need their own
multi-round proof.

No SQL migration is specified for configuration: the pin uses existing metadata.
Any serialized envelope extension must increment its version and update validators,
capability advertisements, and round-trip tests. Actual schema changes, if needed
by implementation, follow the repository migration rules before landing.

Related evidence:

- [SillyTavern PR 6001](https://github.com/SillyTavern/SillyTavern/pull/6001)
- [DeepSeek native prefix contract](https://api-docs.deepseek.com/api/create-chat-completion/)
- [vLLM continuation controls](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/protocol/)
- [Provider continuation ownership](063-hosted-provider-wire-and-durable-tool-continuation.md)
- [Thinking display and replay ownership](090-console-thinking-block-ownership-and-replay.md)
- [Fork exclusions](092-console-chat-fork-copy-and-authority-boundary.md)
- [Conversation settings ownership](095-conversation-owned-console-generation-settings.md)
