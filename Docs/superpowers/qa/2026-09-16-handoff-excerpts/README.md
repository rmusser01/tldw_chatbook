# Source excerpts in Console evidence — 2026-09-16

TASK-2376 repairs media and conversation source handoffs that previously supplied
a generic staging label to the model-context capture. Console now prefers their
source body. Library Media puts its existing 500-character stored-text excerpt
before metadata. Conversations supplies up to 3,000 characters of speaker-labeled
text from the matching, complete loaded reader, with speaker names capped at 80
characters. Truncation is recorded; empty sources are described explicitly.
Notes and other handoff kinds retain their previous summary/body preference.
No stylesheet, authority, source-versus-Resume, or send-gating behavior changed.

ADR required: no new ADR. This repairs existing
[ADR-005](../../../../backlog/decisions/005-console-workspace-server-readiness.md)
and [ADR-147](../../../../backlog/decisions/147-conversation-archive-and-exact-resume.md)
source handoffs without changing the shared payload contract or persistence.

## Tests and review

The new content assertions initially produced [22 failures and 3 passes](red-summary.txt).
They cover actual text through source builders and send-time capture, loaded
identity/version/generation fences, incomplete/unavailable readers, empty sources,
exact excerpt bounds, omitted later messages, sanitization, and long metadata.
Notes body capture and both RAG summary/body cases remain covered.

[Independent review](review.json) found that an unbounded speaker name could
consume the entire transcript budget. Its regression [failed before repair](speaker-red-summary.txt);
the separate label cap now retains both the first message and subsequent replies.
The follow-up review found no further production issue.

The final [targeted run](targeted-tests.txt) passed **207 tests in 243.27s**.
The shared staging path's [legacy media checks](legacy-media-tests.txt) also passed
**7 tests in 1.32s**. The focused post-review content run passed 26 tests.
[Verification](verification.json) records commands and production file hashes.
No full suite was run. Tests and QA helpers pass Ruff and formatting; changed
production ranges are formatted. The [production lint comparison](static-baseline.json)
adds no diagnostics to the existing 205/212/17/1 counts across the four touched
modules. Shifted source-line numbers in duplicate-definition messages are normalized.

## Native journey

The [runner](native_check.py) executes real TldwCli with LinuxDriver and an
exclusive private-profile lock. It validates configured paths before imports,
asserts the imported checkout, primes terminal capability probing, and records
its own SHA-256. The baseline is `f09a7786d4`; final production hashes are in the
verification receipt.

[Final run-007](result.json) passes dark/light themes at 170×48 and 80×24.
Four seeded conversations contain distinct user and assistant messages; a fifth
belongs to another workspace. Four long media sources are already linked to
Default. Each matrix cell verifies:

1. Actual Library row navigation and Use as source, then exact CHAT revision
   settlement in the retained Console. Conversation linking, repeated use,
   Un-stage, and retained-receipt Undo preserve the unrelated foreign link.
2. Both conversation messages occur in the staged evidence and the result of
   the real `capture_console_staged_evidence_for_chat`, labeled CHAT HISTORY.
3. Actual Media row navigation and Use in Console. Its first 500 stored-text
   characters occur in evidence and capture context; the full longer source and
   the generic staging label do not. Media evidence is then unstaged.
4. The active Console session, complete session list, populated draft, and
   empty message list remain unchanged throughout.

The runner opens collapsed Nav/Items disclosures when crossing destinations at
compact width. It uses programmatic focus followed by actual Enter activation;
this does not qualify complete Tab traversal or mouse operation. The eight
`capture-*.json` files retain exact synthetic-source evidence and returned context.
These assertions qualify staging and the send-time capture helper, **not a
completed send, provider generation, or durable citation commit**. No Send action
is used, and there is no independent network/provider-call counter.

## Persistence, lifecycle, and visual review

Complete in-process source snapshots match. The independent [read-only checker](verify_profile.py)
then verifies every original field in five conversations, ten messages, and four
media records. It normalizes only equivalent ISO datetime encodings, including
Media ingestion timestamps. The [receipt](persistence.json) confirms ten healthy
SQLite databases, no additional source records/messages, only original media and
foreign-conversation links, unchanged default-profile config/UI/runtime hashes,
and zero ERROR/CRITICAL lines, traceback headers, or faulthandler bytes.
[Lifecycle](lifecycle.json) records normal app return, shell exit 0, and independent
process-absence verification. Only the owned terminal was closed.

All sixteen SVGs were rendered and inspected together; compact dark conversation
and light media staging received individual full-size inspection. The source
action, link receipt, source strip, Un-stage, and three-line populated draft stay
readable. Wide Inspector secondary text retains its existing truncation. Actual
excerpt delivery is established by capture assertions, not inferred from images.
[Inspection](inspection.json) and [capture hashes](capture-hashes.json) record the
scope and distinguish raw SVG bytes from stored trailing-whitespace normalization.

| Theme/size | Source action | Conversation staged | Link receipt | Media staged |
| --- | --- | --- | --- | --- |
| Dark 170×48 | [Reader](textual-dark-170-reader.svg) | [Console](textual-dark-170-staged.svg) | [Undo](textual-dark-170-linked.svg) | [Media](textual-dark-170-media-staged.svg) |
| Dark 80×24 | [Reader](textual-dark-80-reader.svg) | [Console](textual-dark-80-staged.svg) | [Undo](textual-dark-80-linked.svg) | [Media](textual-dark-80-media-staged.svg) |
| Light 170×48 | [Reader](textual-light-170-reader.svg) | [Console](textual-light-170-staged.svg) | [Undo](textual-light-170-linked.svg) | [Media](textual-light-170-media-staged.svg) |
| Light 80×24 | [Reader](textual-light-80-reader.svg) | [Console](textual-light-80-staged.svg) | [Undo](textual-light-80-linked.svg) | [Media](textual-light-80-media-staged.svg) |

## Reproduction and limits

Use the project interpreter to run `prepare_profile.py PROFILE` with a fresh
absolute private path. It disables first-run dialogs/catalog networking and
configures a no-secret local provider placeholder. In an owned tmux terminal at
170×48, with PYTHONPATH pointing to this checkout, run:

```text
.venv/bin/python Docs/superpowers/qa/2026-09-16-handoff-excerpts/native_check.py PROFILE SOCKET SESSION
```

Record the shell exit in `PROFILE/exit-status.txt`, verify the recorded PID is
absent, then run `verify_profile.py PROFILE` from this evidence directory using
the same interpreter. Raw final evidence remains at
`/private/tmp/tldw-2376-run-007` on this host.

[Attempt history](native-history.json) retains the six setup/navigation retries:
wrong setup interpreter, omitted onboarding/catalog/provider configuration,
wrong media-ID expectation, and missing compact Nav disclosure in each direction.
They are not counted as successful qualification. The first stopped before app
creation; all six later app processes are independently confirmed absent.

Native empty/remote/archived source flows, sending to a provider, and unprimed
terminal-probe startup remain outside this slice. Empty/large source bodies and
reader fences have targeted automated coverage. The earlier source-staging report
remains historical; this repair closes its missing-excerpt limitation. Next:
Library Search/RAG journeys and the remaining feature/component review.
