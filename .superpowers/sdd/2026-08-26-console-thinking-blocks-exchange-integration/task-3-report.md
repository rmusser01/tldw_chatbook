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

## Independent review outcome

The final specification review APPROVED the full Task 3 range at `3df136697f`. It
confirmed owner-aware fresh/resumed refusal, recoverable manual and queued paths,
model-specific local replay, atomic tombstone ownership, explicit ADR precedence, and
all prior lifecycle/exchange/privacy/documentation contracts.

The final code-quality review also APPROVED that revision. It found no remaining race,
claim-settlement error, stale identity overwrite, duplicated cleanup, artificial test
evidence, architecture contradiction, private diagnostic leak, or useful
simplification. Root's isolated live verification is the remaining acceptance gate.

## Specification review fix round 1 — model-specific local compatibility

Review found that local thinking capability was frozen only from the execution key,
and optional replay did not consume the target model. One llama.cpp/vLLM endpoint
could therefore make a plain selected model inherit displayable capture/replay and
the persistent V1 preflight requirement from a reasoning model.

Strict TDD produced two initial genuine failures through the real resolver/history
path: the same llama.cpp endpoint resolved both `Qwen3.8-27B` and
`Llama-3.3-8B-Instruct` as displayable, and the plain target accepted the stored
block. A second three-test RED pinned the accepted precedence and exact ownership:

1. explicit `reasoning_effort = "none"` resolves ignored;
2. any explicit non-`none` effort is the existing configured displayable signal;
3. with effort unset, `reasoning_effort_hint_for_model(model)` supplies the existing
   recognized-model signal;
4. an otherwise unknown local model fails closed to ignored; and
5. optional local replay additionally requires the stored and selected model IDs to
   match exactly, alongside the existing local-provider family, protocol, envelope
   version, completion status, and source-encoding checks.

The resolver freezes that decision with the selected model/effort on the immutable
`ConsoleProviderResolution`; a later model switch cannot change an in-flight target.
No new preference, speculative model-name predicate, generic translation, or provider
adapter was added. Explicitly configured custom reasoners remain supported and may
replay only their own exact-model blocks. Required/private continuation remains
provider-owned and is validated/restored independently; the optional-thinking check
does not weaken or translate it.

The joined replay spine now obtains its Qwen target from the actual resolver and
dispatches the prepared request through the direct HTTP adapter. A second joined
control uses the actual resolver/direct adapter with a plain model and persistent V0
backend: it bypasses the thinking round-trip preflight and dispatches exactly once.
The earlier synthetic ignored-disposition control remains only as a narrow enum
control, not the integration evidence.

Two controller tests initially failed after the production change because their
synthetic selected model and stored block model were inconsistent; aligning those
fixtures to one exact model restored their intended direct/agent replay coverage
without another production change.

`chat-basics.md` now states the accepted automatic-collapse boundary: first visible
answer or tool event, terminal fallback if neither occurs, and any manual disclosure
interaction cancels the pending automatic transition.

Fresh review-fix evidence on the formatted tree:

- Focused provider/history/prepared/controller/joined gate: **626 passed, 2 skipped,
  2 warnings in 58.27s**.
- Final-tree exact approved broad targeted matrix: **1,282 collected; 1,280 passed,
  2 skipped, 2 warnings in 134.36s**.
- Post-format resolver/history/controller/joined regression: **12 passed, 2 warnings
  in 2.79s**.
- The two skips remain loopback-listener permission controls. The two warnings remain
  the environment's Requests dependency mismatch and Python 3.12 `pydub`/`audioop`
  deprecation.
- Scoped Ruff formatted 2 files and left 3 unchanged; scoped Ruff check passed.
- CSS and all four derived widget/screen bundles reproduce from source.
- Persistent diagnostic inventory is unchanged and passes at **537 owners, 1,243
  TASK-492 calls, 7,341 TASK-494 calls, and 8 sink files**.
- Relevant `py_compile` and `git diff --check` passed.

This review fix adds no ADR decision; it applies ADR-090's existing compatible,
provider-resolved optional replay boundary and ADR-063's independent mandatory
continuation ownership. Root still owns the isolated live gate and Backlog closeout.

## Quality-review fix round 2 — owner-aware resumed refusal

Review reproduced a separate state-machine defect after a prepared turn had already
created its transient user owner. If persistence changed from V1 to V0 between the
retrieval pause and manual Retry/Bypass, the shared thinking preflight treated the
resumed owner as a fresh optimistic echo: it deleted that owner, restored stale
conversation identity/title snapshots, returned the upgrade refusal, and left the
preparation READY/COMMITTING instead of recoverably paused.

Strict TDD produced four genuine failures before the production fix: joined SQLite
manual Retry and Bypass controls both remained READY rather than PAUSED/PERSISTENCE,
and the persistent queue reclaim equivalents did the same. The refusal made no
provider contact, but the existing owner and current session state were lost.

The minimal correction is one owner-aware branch in the existing early compatibility
preflight. A resumed preparation now calls `_pause_prepared_commit(..., PERSISTENCE)`
and returns the established content-free upgrade copy. That state-machine operation
accepts both READY and COMMITTING. It preserves the exact owner, draft, frozen inline
attachment, preparation continuation sidecar, and intervening conversation title/ID.
Fresh sends retain their existing narrow delete/reset path; the broad matrix includes
the fresh-refusal controls that remove only the just-created echo and preserve prior
conversation state.

The manual tests restore V1 and prove the same prepared owner can retry successfully.
The queued tests prove claim finalization returns the refused owner to the recoverable
queue head, preserves the next waiter, and later dispatches the first owner after V1
restoration without another production change. Refused resumed dispatches make zero
provider contacts and the refusal copy contains no draft or thinking payload.

The older durable-continuation design's tombstone-retention paragraph now carries a
dated supersession note linking ADR-090. ADR-090 carries the reciprocal amendment:
for Console selected-generation edit/delete, visible answer, thinking sidecar, and
ADR-063 continuation are one owner and are cleared together so deleted visible output
cannot retain replayable private state. Ordinary non-deleting Discard and non-deleted
off-branch continuation ownership remain unchanged. This is documentation of the
already implemented ownership rule, not a database-policy rollback.

Fresh final-tree evidence:

- Owner-aware joined/manual plus queue preparation gate: **223 passed, 2 warnings in
  24.20s**.
- Focused controller preflight/recovery selection: **5 passed, 231 deselected,
  1 warning in 1.48s**.
- Exact approved broad targeted matrix: **1,284 collected; 1,282 passed, 2 skipped,
  2 warnings in 143.58s**. The skips and warnings remain the documented loopback
  permission controls and environment warnings.
- CSS and all four derived widget/screen bundles reproduce from source.
- Persistent diagnostic inventory remains unchanged and passes at **537 owners,
  1,243 TASK-492 calls, 7,341 TASK-494 calls, and 8 sink files**.
- Scoped Ruff format/check, relevant `py_compile`, and `git diff --check` pass.

ADR required: no new ADR. ADR-090 already governs the generation-owner contract; this
round explicitly resolves the older reference-only design contradiction. No new
diagnostic, sink, envelope, provider behavior, Backlog status change, or live-gate
claim was introduced. Root retains the isolated live verification and closeout.
