# Task 3 report — joined lifecycle, refusal, and documentation

## Outcome

The joined Console path now proves that actual adapter events become one durable
assistant-owned thinking envelope, the live disclosure expands and auto-collapses
once, restart hydrates it collapsed, compatible optional history is counted and
dispatched exactly once, and selected JSON, Chatbook V2, and Sync V2 restore the same
thinking owner into a second database. Stop, failure, proprietary, and no-evidence
turns remain honest.

Two real joined REDs required production changes:

1. An unsupported persistent backend was detected only after the turn crossed the
   durable acceptance boundary. `_submit_draft_inner` now applies the established
   persistence compatibility contract immediately after provider resolution and
   before provider preparation/contact or turn acceptance. Refusal deletes only the
   transient optimistic user echo, restores the pre-send title/conversation ID, keeps
   the draft recoverable, and returns the existing content-free upgrade message.
2. Tombstones cleared `thinking_blocks_json` but retained the same generation's
   `provider_continuation_json`. Direct soft delete, subtree delete, and descendant
   tombstoning after an ancestor content edit now clear both owners atomically. The
   pre-delete sync base hash is still captured before the update, so it continues to
   describe the complete prior generation while the committed tombstone carries no
   private sidecar.

No remote persistence adapter, alternate envelope, or generic thinking translator
was introduced.

## RED evidence

- The unsupported displayable/proprietary persistent-backend cases crossed the
  accepted-turn boundary and returned `Accepted turn is retained for recovery.`
  instead of the required pre-acceptance upgrade refusal. This was the genuine
  controller seam; provider contact was already prevented, but the draft/send and
  transcript ownership were not honest.
- The joined ownership test read the raw deleted row and found
  `thinking_blocks_json IS NULL` while `provider_continuation_json` still held its
  complete Moonshot checkpoint.
- The nearest direct/subtree DB regressions then failed **2/2** with the same retained
  continuation. The equivalent descendant-tombstone expectation was updated before
  applying the single three-path SQL fix.
- Early exchange test failures came from invalid test owners: a V1 continuation was
  first constructed for an unsupported provider, then paired with a nonmatching
  visible answer. Replacing it with the repository's established complete Kimi K3
  checkpoint and matching owner content removed those fixture failures without a
  production change.

## Joined evidence

`Tests/integration/test_console_thinking_end_to_end.py` contains a meaningful real
lifecycle spine plus focused joined boundaries:

- real `ConsoleChatController`, adapter event capture, `ChatPersistenceService`,
  `CharactersRAGDB`, conversation-tree hydration, `ConsoleChatStore`, and mounted
  `ConsoleTranscript` prove expanded-live, one terminal auto-collapse, no re-arming,
  and collapsed/lazy restart;
- real history resolution, semantic request building, token-count input, provider
  preparation, and the gateway's pre-provider send path prove Auto/Include/Exclude/
  Required and exact-once compatible llama.cpp/vLLM-style replay;
- a persistent V0 fake plus provider spy proves displayable/proprietary refusal with
  zero provider contacts, no synthetic assistant, content-free copy, a recoverable
  draft, and a successful retry after upgrade; V1, ignored-disposition, and ephemeral
  controls dispatch;
- selected-conversation JSON, Chatbook V2, and encrypted Sync V2 each populate a
  second ChaChaNotes database and hydrate the restored thinking owner/policy;
- edit, direct/subtree delete, descendant tombstoning, and generation replacement
  clear the complete selected-generation owner while unrelated feedback preserves it;
- stopped and failed turns retain only received displayable evidence; proprietary
  evidence renders only `Proprietary thinking obfuscated - not available` without
  persisting that notice; a displayable-capable no-event turn fabricates no envelope
  or activity.

The repository's nearest controls already cover opaque future envelopes and
whole-record conflicts, so no duplicate production seam was added:

- `test_restore_preserves_unknown_opaque_and_blocks_generation_mutations` preserves
  unknown JSON byte-for-byte through unrelated writes, leaves it unrendered and
  unreplayed, and blocks generation replacement;
- `test_conflict_never_merges_thinking_blocks_between_whole_records` proves sync does
  not splice an answer from one side with thinking from the other; incoming malformed
  and future-version payload controls refuse before mutation.

## Documentation

Updated the five approved guides:

- `Docs/User_Guide/console/chat-basics.md`
- `Docs/User_Guide/console/context-and-rag.md`
- `Docs/User_Guide/console/agent-runs-and-tools.md`
- `Docs/User_Guide/settings.md`
- `tldw_chatbook/Chatbooks/CHATBOOKS_GUIDE.md`

They now distinguish actual adapter-reported displayable/proprietary evidence from
capability, document `Thinking · unavailable` and the exact application notice, the
expanded-live/one-time-collapse/manual lifecycle, default-on presentation-only
visibility, conversation Auto/Include/Exclude plus effective Required and **Save as
default for new conversations**, model-specific local replay compatibility, persistent
backend upgrade refusal, sensitive importable exchange versus human-readable
omission, and safe session-only Planning. None promises hidden chain-of-thought.

## GREEN and static evidence

- Joined file: **18 passed, 2 warnings in 8.14s**.
- Tombstone regression plus joined ownership: **4 passed, 31 deselected, 2 warnings
  in 3.45s**.
- Existing opaque/conflict controls: **2 passed, 48 deselected, 1 warning in 0.75s**.
- Required broad targeted gate (using the repository's tracked lowercase integration
  directory): **1,275 passed, 2 skipped, 2 warnings in 132.46s**.
  - Both skips are loopback-listener permission controls in
    `test_console_provider_gateway.py`.
  - Warnings are the existing Requests dependency mismatch and Python 3.12
    `pydub`/`audioop` deprecation.
- CSS bundle and all four generated widget/screen bundles reproduce from source.
- Scoped Ruff format: **4 files already formatted**.
- Scoped Ruff check: **All checks passed**.
- Relevant `py_compile`: passed.
- `git diff --check`: passed.

The broad gate's first terminal stream detached after 16%, so the same exact test list
was rerun once with `PYTEST_DEBUG_TEMPROOT=/private/tmp/task3-final-pytest-root` to
retain auditable final output and avoid unrelated stale pytest-temp cleanup warnings.

## Persistent diagnostic inventory review

Git tracks `scripts/check_persistent_diagnostic_inventory.py`; the plan/brief were
corrected to that authoritative lowercase path. The first check reported:

- `task_492_calls: 1241 -> 1243`
- `task_494_calls: 7340 -> 7341`
- Task 1 added one policy warning in `Character_Chat_Lib.py` and two content-free
  persistence diagnostics in `console_chat_store.py`.
- `console_provider_gateway.py` has the same ten calls reformatted.
- `ChaChaNotes_DB.py` has the same 360 calls; the statement tool identifies 22 removed
  and 22 equivalent formatter rewrites since inventory pin
  `995036264207f4249fce880c6d288c7a369beb0e`.

Root then used the tool's documented `--statements ... --since` workflow to review
every changed statement. The Character Chat warning resolves to the fixed,
content-free unknown-policy warning; the two Console store calls are fixed operation
copy. The gateway and DB rows preserve their prior messages/arguments and only changed
formatting. Task 3 itself adds no logger or sink, and sink topology remains unchanged.
Because all aggregate rows were reviewed, root ran the documented whole-inventory
`--write`. The final guard passes with **537 owners, 1,243 TASK-492 calls, 7,341
TASK-494 calls, and 8 sink files**.

## Path corrections

The authoritative tracked tree uses `Tests/integration/` and `scripts/`. The stale
case variants in the plan/brief were corrected rather than creating case-colliding
entries that would be unsafe on the current filesystem.

## ADR-090 self-review

- Actual evidence remains a versioned assistant-generation sidecar separate from
  answer content and ADR-063 continuation.
- Proprietary evidence is structurally text-free; the exact notice is application-only.
- Selected answer, thinking, usage, and continuation remain one generation owner for
  edit/replacement/delete and whole-record sync.
- Optional replay remains provider-resolved and conversation-owned; Required remains
  an effective continuation overlay and never overwrites the saved preference.
- Importable formats remain the warned sensitive exceptions. Human-readable,
  diagnostic, search, summary, title, usage, speech, copy, log, and error surfaces are
  not broadened.
- Persistent unsupported backends fail before provider contact and without lossy
  acceptance.

ADR required: no new ADR.

ADR path: `backlog/decisions/090-console-thinking-block-ownership-and-replay.md`

Reason: both production corrections enforce ADR-090's already accepted persistence
and generation-ownership contracts; they introduce no new boundary or future choice.

## Residual and root-owned gates

No derived-check residual remains. Root still owns independent specification/code-
quality reviews, isolated live Console verification, and Backlog closeout. This task
does not claim the root-owned live gate and does not mark any Backlog task Done.
