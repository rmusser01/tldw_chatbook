---
id: task-22302
title: Durable turns silently discard citation provenance
status: To Do
labels:
  - console
  - citations
  - data-loss
  - regression
priority: high
---

## Description

Since `a26cdafd8` ("resume Library-gated sends"), a durable Console turn writes
**no citation provenance at all**. Every citation table stays empty for an answer
that carries citations.

Bisected:

| commit | result |
|---|---|
| `a26cdafd8~1` (`52571f925`) | **passes** |
| `a26cdafd8` | **fails** |
| current `dev` | fails |

Reproduced by `Tests/Chat/test_console_terminal_citation_persistence.py::
test_real_atomic_direct_controller_persists_exact_body_and_trace_on_restart`,
which runs a real in-memory ChaChaNotes stack. All six tables report 0 where 1
is expected:

`rag_citation_traces`, `rag_message_trace_owners`, `rag_evidence_runs`,
`rag_evidence_snapshots`, `rag_trace_evidence_refs`,
`rag_answer_attempt_payloads`.

## Mechanism

`ConsoleChatStore.mark_message_complete` pops the terminal citation finalizer at
`console_chat_store.py:8045`:

```python
finalizer = self._terminal_citation_finalizers.pop(message.id, None)
```

Then, at 8052, it takes the in-flight dispatch-recovery branch and **returns at
8063**:

```python
if recovery is not None and recovery.assistant_message_id == message.id and recovery.in_flight:
    ...
    self._settle_owned_dispatch_terminal(message, "complete")
    self._record_message_completed(session_id, message.id)
    return self._snapshot(message)
```

The code that actually uses the finalizer — building `citation_write` and passing
it to `_persist_new_message` — is at 8095, thirty lines below a return that now
fires for every checkpointed turn. The finalizer is consumed and dropped.

Confirming evidence: `_settle_owned_dispatch_terminal` contains zero citation
handling, and `console_dispatch_repository.py` (which writes the checkpoint rows
with raw SQL) contains zero occurrences of "citation". Citations only reach
storage via `create_message(citation_write=...)` -> `CitationTraceRepository`,
and a probe against a real stack confirms **`create_message` is never called** on
the durable path.

## Why it went unnoticed

The one test that would have caught it was failing EARLIER, for an unrelated
reason, so it never reached the citation assertion. Its stack pre-created the
conversation with `create_conversation(...)`, which writes no
`console_conversation_library_policy` row; `commit_durable_turn` then took its
"conversation exists" branch, found `policy_row is None`, and refused the turn
with "Durable Console Library policy no longer matches acceptance." The send
produced a USER row and nothing else.

That upstream refusal was itself silent until TASK-22251 added
`Durable turn commit failed; turn refused (exception_type=...)`, which is what
surfaced it.

## Acceptance Criteria

- [ ] A durable turn with citations writes its trace, owner, evidence and
      attempt rows
- [ ] The real-stack tests in `test_console_terminal_citation_persistence` pass
- [ ] A test fails if the finalizer is dropped again (mutation-proven)
- [ ] The fix is verified on a real ChaChaNotes stack, not a recording double

## Implementation Notes

The fix belongs in the dispatch-recovery branch: it must still honour the
terminal citation finalizer before returning, or the branch must not swallow a
turn that has one. Note the finalizer is popped BEFORE the branch decision, so
simply reordering the pop is not enough — the branch has to carry the write.

Do not fix this by reverting the checkpoint. The checkpoint is what makes a turn
resumable; the defect is that its terminal path forgot one write.

Related: TASK-22301 (the boundary tests' recording doubles observe
`create_message`, which this same finding shows the durable path never calls).


## Update — candidate fix identified, NOT landed

Three sequential blockers were found and each was individually mutation-proven.
Together they make the two real-stack tests pass and the whole
`test_console_terminal_citation_persistence` file green (92/92). They are NOT
committed, because in the full test composition the same change produces a
DUPLICATE `create_message` call (`assert 2 == 1`, the same body written twice)
in seven capability tests. That is reproducible — two identical runs failed the
identical seven — so it is a real interaction, not xdist ordering.

### Blocker 1 — the finalizer never survives the durable hand-off

`_accept_durable_turn` takes `terminal_citation_finalizer` as a parameter
(`console_chat_controller.py:5600`) and uses it exactly once more: it builds a
`_DurablePostcommitContinuation` (5723) that has NO field for it, and
`resume_durable_postcommit` then publishes the live owners with a hard-coded
`terminal_citation_finalizer=None` (5820). Adding the field and forwarding it
fixes this leg.

### Blocker 2 — arming is gated on the wrong condition

`append_message` arms only when its own `persist` flag is true. On this path
`persist=False` is CORRECT — the durable row already exists, written by the
dispatch checkpoint — but "this call does not create the row" is not the same as
"this message has no durable row", and arming needs the latter.
`_hydrate_durable_turn_owner_messages` assigns `persisted_message_id`
immediately after the append, so arming there works.

### Blocker 3 — the in-flight dispatch shortcut discards the finalizer

`mark_message_complete` pops the finalizer (`console_chat_store.py:8045`) and
then returns early via the dispatch-recovery branch (8052-8063), thirty lines
above the code that would use it. Skipping that branch when a finalizer is owed
reaches the citation write.

### Blocker 4 — the body then persists as ''

`create_message`'s existing-row branch is an idempotent-RETRY path: it verifies
and writes the citation but deliberately does not update content. The checkpoint
wrote `content=''`, so the answer persisted empty. Flushing the body first (via
`_persist_existing_message`) makes the row match, after which the citation write
keys to it — no persistence API change required.

### Why it is not landed

Narrowing both store-side changes to the citation path (skip the shortcut only
when a `finalizer` is owed; flush content only when a `citation_write` will
follow) fixed 7 regressions the first attempt caused, and the file went 92/92.
But the full composition still shows the duplicate `create_message` in those
seven capability tests, and it could not be reproduced in any smaller set
(terminal file alone: 92/92; terminal + boundary: clean; terminal + 10 union
files: clean). Each full-composition iteration costs ~25 minutes.

A duplicate durable write is a worse defect than the one being fixed, so this
needs the trigger identified before it can land. The likely suspect is the
interaction between arming `_terminal_persistence_deferred_ids` and
`mark_message_complete`'s `terminal_persistence` predicate, which then routes a
turn to `_persist_new_message` whose row already exists.
