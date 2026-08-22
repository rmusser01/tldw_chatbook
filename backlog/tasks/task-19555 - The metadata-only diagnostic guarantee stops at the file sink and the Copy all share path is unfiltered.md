---
id: TASK-19555
title: >-
  The metadata-only diagnostic guarantee stops at the file sink; the in-app log
  buffer and "Copy all" share path are unfiltered
status: Done
assignee: []
created_date: '2026-08-21 20:05'
labels:
  - security
  - privacy
  - logging
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 2 (security & privacy) — its **Tier 2
#5**, and one of the two choke-point repairs the lane says covers the most
ground. CONFIRMED; re-verified at this branch base.

**Calibration first, because it matters for scoping:** ADR-029's metadata-only
guarantee is enforced by `PersistentDiagnosticFilter` attached at **exactly two
places, both the rotating FILE handler** (`Logging_Config.py:284,312`). The
file sink is airtight. **Nothing else is.**

The in-app path has no filter anywhere on it:

- `app.py:7836` — `self._log_buffer = deque()  # No maxlen - keep all logs`
- `app.py:7844-7854` — `PersistentLogHandler.emit` appends the formatted record
  straight into the buffer and into `_log_records`. The handler is level
  `NOTSET`, carries **no filter**, and is installed unconditionally.
- `UI/Logs_Window.py:271` — no sanitization on display
- `UI/Logs_Window.py:219 / 515` — a **"Copy all"** control whose
  `_on_copy_all` joins the entire session buffer to the **system clipboard**
  (OSC 52)
- `UI/Logs_Window.py:197-199` — the empty state **tells users to copy and
  share their logs**

So the app collects unfiltered diagnostics without bound, renders them, and
actively invites the user to put them on the clipboard.

The privacy test suite proves the gap by omission:
`test_remaining_diagnostic_sentinel_matrix.py:90-129` asserts only against the
**file** handler; the unfiltered collector is never asserted at all.

Scale, from the lane's AST scan: **872 leaky interpolations**; at INFO and
above, **59 user-content sites and 243 path sites**. Named live examples:
attachment paths, note **titles** on creation, keywords, the message body
prefix (`app.py:8999`, `event.text[:50]`), image prompts, **tool argument
values** (5 sites in `file_operation_tools.py` — a TASK-492 owner class), full
search queries and terms, provider response bodies, note filenames, and 75
ingestion path sites.

**Escalation:** three shipped documents tell users to set `log_level=DEBUG` for
troubleshooting, which admits roughly **800 more** sites including
`sql_logging.preview_params` — 80 characters of *every* SQL string parameter.

TASK-19552 (the Google CSE key at INFO) rides this exact path, which is why
that key does not merely land in a file.

Also worth recording while in this area: `Utils/log_sanitizer.py` has **exactly
one call site in the whole package** — the primitive exists and is essentially
unused. Its own normalization defect is filed separately (TASK-19558).

## Acceptance Criteria

**AC #1 was rewritten before implementing** (see Implementation Notes, "The
one AC that changed"). The filed wording asked for the existing
`PersistentDiagnosticFilter` to be attached at `PersistentLogHandler.emit`.
That filter is an all-or-nothing *admission* filter, not a redactor: attaching
it there drops every descriptive record on the floor and leaves the Logs
screen showing only `event=… status=…` lines. That is not a privacy fix with a
usability cost, it is the deletion of the screen. The original text is kept
below, struck, so the deviation is visible rather than quietly absorbed.

- ~~[ ] The metadata-only guarantee holds on the in-app path, not just the
  file sink — attaching the existing filter at `PersistentLogHandler.emit` is
  the preferred repair~~
- [x] `PersistentLogHandler.emit` is the single choke point where the in-app
      path is made safe, covering every consumer of the buffer at once (~800
      call sites, no per-call-site edits), splitting the two surfaces by what
      each can honestly carry:
      - the **live view** keeps descriptive text but never credentials and
        never the operating-system account name;
      - the **"Copy all" artifact** is metadata-only in exactly the ADR-029
        sense, decided by an instance of the same `PersistentDiagnosticFilter`
        class the rotating file handler uses — not the same object, but the
        class is stateless (no `__init__`, `filter` fully overridden), so the
        two sinks cannot drift on what counts as metadata-only.
- [x] "Copy all" cannot place unredacted user content or credentials on the
      system clipboard
- [x] `_log_buffer` is bounded — an unbounded session buffer is both a
      memory and a disclosure surface
- [x] The privacy suite asserts against the **collector**, not only the file
      handler, so the omission that hid this cannot recur
- [x] The assertion is mutation-checked: removing the filter makes the test
      red
- [x] The empty-state copy that invites users to share logs is only shown once
      the shared artifact is actually safe to share
- [x] The `log_level=DEBUG` troubleshooting advice in the three shipped docs is
      revisited in light of what DEBUG admits, and either made safe or made
      honest about what it exposes

## Implementation Plan

1. Reproduce the leak against the real collector (born-red), not a stand-in.
2. Decide the bar per surface; rewrite AC #1 with the argument before coding.
3. Add the sink-side redactor to `Utils/log_sanitizer.py`, repairing the
   normalization defect it depends on.
4. Redact at `PersistentLogHandler.emit`; bound `_log_buffer`.
5. Make the Logs screen's labels, notifications and empty state describe what
   each action actually produces.
6. Mutation-check every new assertion; revisit the three `DEBUG` docs.

## Implementation Notes

### What was wrong

ADR-029's metadata-only guarantee was enforced by `PersistentDiagnosticFilter`
at exactly two places, both the rotating FILE handler. `TldwCli._setup_
buffered_logging` installs a second collector on the *same* root logger —
`PersistentLogHandler`, level `NOTSET`, no filter — feeding an **unbounded**
`deque`, which `LogsWindow._on_copy_all` joined onto the system clipboard,
under an empty state that told the user to reproduce the problem and share
their logs. The existing privacy suite attached a filtered file handler beside
an unfiltered collector and asserted only against the file, so the collector
the app actually ships was never asserted against at all.

### The one AC that changed, and why

The filed repair — attach `PersistentDiagnosticFilter` at the collector —
fails on contact. That filter admits only schema-validated
`log_persistent_metadata` records and drops everything else, so the Logs
screen would render `event=app_started component=app` and almost nothing
besides. Its filter bar, level chips, regex search and next-error jump would
all operate on an empty set, and "reproduce the problem, then share the logs"
would hand a helper nothing. The metadata-only artifact already exists on
disk; the in-app screen's entire value is that it is the rich one.

So the bar is set per surface, by a single rule: **remove what is never
wanted; disclose what is sometimes wanted.**

- *Never wanted*: API keys, bearer tokens, `Authorization`-class headers,
  passwords, and the operating-system account name. No maintainer has ever
  needed to read a key, and `/Users/janedoe/…` is a real-name identifier on
  most desktops. These are removed from **everything** the in-app path holds
  — buffer, record store, live view — so a screenshot is covered too.
- *Sometimes wanted*: note titles, search queries, prompts, tool arguments,
  file names. These are the content a user is deliberately disclosing when
  they ask for help, and no sink-side rule can tell them apart from the
  wording around them. They stay in the live view and in **Copy visible
  logs**, which copies a set the user chose with a filter and can read.
- *Never consented to*: the whole session, unread. **"Copy all"** exports
  thousands of lines nobody looked at, so it now carries the metadata-only
  form and is labelled "Copy all (redacted)".

Alternatives rejected: (a) the filed all-or-nothing filter at emit, above;
(b) redacting only in the copy handlers — leaves the buffer and the on-screen
view holding live credentials for the whole session, and protects nothing
against a screenshot or a future surface wired to the same buffer; (c) a
separate unredacted field for the viewer — unnecessary, because `_log_buffer`
and `_log_records` were *already* two stores fed by the one emit and merely
duplicated each other; this change gives each of them one job instead of
adding a third.

### What a user can still leak

Everything in the *sometimes wanted* class, via **Copy visible logs** or a
screenshot: file names and paths below `~`, note titles, keywords, search
queries, prompts, tool argument values, provider response text, and the
message-body prefix. That is a real residual and it is not closable at a
sink — it is per-call-site work (TASK-492's class), deliberately out of scope
here. It is now *disclosed* rather than silent: the empty state, both copy
notifications and `Docs/User_Guide/logs.md` say so in as many words.

The narrower residual is credential coverage. Redaction is a **denylist**, so
a secret survives when it is in a shape `_STANDALONE_CREDENTIALS` does not
know *and* carries no label `_is_sensitive_log_key` recognises. The review
round widened the shape set (GitHub `ghp_`/`github_pat_`, Hugging Face `hf_`,
OpenRouter `sk-or-v1-`, AWS `AKIA…`, Slack `xox…`, bare JWTs — the first four
of which the original set genuinely missed) and every user-facing claim now
says **"recognised API-key formats"** rather than "always removed", which was
false as written.

One label gap is deliberately left open: bare `key=` is **not** treated as
sensitive, though bare `token`/`secret`/`password` are. A census of every
`key=`/`key:` label inside a logger call in this package found **5 sites, all
of them non-secrets** — a dict key name, a cache key, a provider settings key
name, and two config key *names* (`config_encryption.py:221`,
`config.py:4886`). Adding `key` would redact 5 known debugging values, catch
0 known secrets, and blind a security-adjacent diagnostic that reports which
config key failed to decrypt. The shape patterns are the right investment for
that risk, and they are label-independent.

### The buffer bound

Bounded to `Logs_Window.MAX_LOG_RECORDS` (10,000), the same window
`_log_records`, the `RichLog` widget and the screen's own status line
("buffer keeps last 10,000") already use. Beyond memory, the unbounded buffer
was a straight honesty defect: "Copy all" could export an entire multi-hour
session while the screen told the user it kept the last 10,000 lines. The two
stores are now pinned equal by test.

### Also repaired here

`log_sanitizer._is_sensitive_log_key` computed a hyphen normalization and
then passed the **raw** key to `is_sensitive_config_key`, whose
`_key`/`_token`/`_secret`/`_password` rules are underscore-suffix matches. So
`x-auth-token`, `x-session-key` and `x-client-secret` — the exact names
provider request logging produces — were classified harmless and their values
written out verbatim. Fixed here rather than deferred, because the sink-side
redactor depends on that predicate. Normalizing cannot create a false
positive: `max-tokens` → `max_tokens`, which still does not end in `_token`.

**TASK-19558 must NOT be closed against this.** It carries five items (FTS5
dead-store quoting, defusedxml OPML, this predicate, `redact_paths=True`,
local-tool risk tags); closing it wholesale would silently drop four security
items. This branch satisfies **part of item 3 only** — the hyphenated-suffix
header shapes — and even there bare `key` remains open by the deliberate
decision recorded above.

The same review round also repaired the Windows half of `redact_user_paths`:
`Users` was a literal in a pattern whose comment claimed case-insensitivity,
so `c:\users\<name>\…` and every UNC `\\SERVER\Users\<name>\…` path kept the
account name. Windows paths are case-insensitive end to end, so the Windows
branch is now its own `re.IGNORECASE` pattern with a UNC alternative, while
the POSIX branch stays case-sensitive (a case-insensitive `/users/` would
rewrite REST URLs).

`app._display_buffered_logs` (dead code, zero callers) was pointed at
`_log_records` so that reviving it shows real diagnostics rather than a
screen of redaction markers.

### Measured cost, and the length cap

`redact_log_line` costs 22–33 µs on a typical ~140-character line against
~1.4 µs for the `Formatter` call it follows. The first draft of this note
called that "small beside the downstream render work" — **that framing was
wrong** and the review round corrected it: it is only true while the Logs
screen is OPEN. With no screen mounted there is no downstream work at all,
and redaction is ~93% of what the handler costs.

Worse, cost is linear in line length, so an uncapped kv-dense line scaled
badly: 3.4 KB → 755 µs, 100 KB → **22 ms**, paid on whichever thread emitted
the record, the UI thread included.

No fast path was added — a fast path on a redactor is a bypass hole. A
**length cap** was, which is the opposite: it keeps strictly *less* data than
the uncapped line, so it cannot be a bypass. Lines are truncated to
`MAX_REDACTED_LINE_CHARS` (2,000) before redaction, with the original length
disclosed in the marker. Measured effect: typical unchanged (33.0 vs 33.6 µs),
3.4 KB 755 → 501 µs, 100 KB **22.2 ms → 0.50 ms**, a 45× bound on the worst
case. 2,000 characters is far past terminal readability and still keeps a
20-parameter `sql_logging.preview_params` line (80 chars each) whole.

This also closes the second residual the first draft recorded honestly and
could not then fix: the buffer bounded line *count* but not line *size*, so a
dumped provider response body was retained whole. It no longer is.

### Verification

Born red at base: 5 of 6 new tests in `Tests/test_logs_share_path_privacy.py`
failed with the sentinel present, including a full traceback carrying the
user's content into the clipboard buffer, and `_log_buffer.maxlen` reading
`None`. Mutation-checked three ways after the fix: removing the share
admission (11 red), removing the emit redaction (2 red), reverting the
hyphen normalization (4 red). A sandboxed-`HOME` runtime probe through the
real handler confirmed a realistic leak set — an attachment path, a note
title, a search query, an `x-auth-token`, an `sk-` key and an exception
carrying a note title — lands in the view but not on the clipboard, with the
key and the account name in neither.

**The review round found a fourth mutation that SURVIVED**, and it is the
important one. `emit` fills two stores and then hands the line to whichever
on-screen surface is mounted; the first draft pinned the two STORES and left
the live FEED unpinned. Passing the unredacted `formatted` to
`logs_window.append_record` — which puts live credentials into
`LogsWindow._records`, precisely what `_on_copy_visible` copies to the
clipboard — produced **111 passed, 0 failed**. Only
`Tests/UI/test_logs_ux_fixes.py` referenced `append_record` at all, and it
passed under the mutation.

That is this task's own lesson ("count the sinks") failing one seam further
in: I counted the *stores* and stopped. Both feeds are now pinned
(`test_live_logs_window_feed_receives_the_redacted_line`,
`test_legacy_rich_log_feed_receives_the_redacted_line`) and each mutation
reds — the reviewer's `append_record` mutation now fails with both the key
and the account name present in the captured feed, and the equivalent
mutation on the legacy `_current_log_widget.write` branch reds too. The
lessons entry has been updated with this second incident.

### Modified or added files

- `tldw_chatbook/app.py` — bounded buffer, sink-side redaction, share-artifact
  construction, `_display_buffered_logs` store fix
- `tldw_chatbook/Utils/log_sanitizer.py` — `redact_log_line` (with the length
  cap), `redact_user_paths` (POSIX + case-insensitive Windows/UNC),
  `_is_sensitive_log_key` normalization repair, six added standalone
  credential shapes
- `tldw_chatbook/UI/Logs_Window.py` — button label, empty state, both copy
  notifications
- `Tests/test_logs_share_path_privacy.py` (new)
- `Tests/test_remaining_diagnostic_sentinel_matrix.py` — asserts against the
  real collector, closing the omission
- `Tests/Utils/test_log_sanitizer.py`, `Tests/UI/test_logs_ux_fixes.py`
- `Docs/User_Guide/logs.md`, `Docs/Features/ChatDictionaries-Documented.md`,
  `Docs/Features/World-Lore-Books-Documented.md`,
  `Docs/Development/RAG/RAG-OCR.md`

### Inventory note for the merge controller

`Docs/security/production-diagnostic-inventory.json` is **not** restamped
here. It is already stale on `origin/dev` — 9 owner rows drifted before this
branch existed — and restamping would absorb that unreviewed drift into a
privacy commit, which is the exact failure mode the checker's own docstring
warns about.

**Two suites are red at base for that reason, and both stay red here:**

1. `Tests/LLM_Calls/test_summarization_diagnostic_privacy.py` — 3 failed /
   254 passed, identical count at base and on this branch. Whole-inventory
   digest mismatch; this branch does not change it further.
2. `Tests/Architecture/test_persistent_diagnostic_inventory.py::
   test_production_diagnostic_inventory_and_sink_topology_are_unchanged` —
   1 failed / 64 passed. Verified red at base by running it in a detached
   worktree at `origin/dev` (`fdc6ad663`), same 1/64. **Unlike (1), this
   branch contributes the 10th drifted row**, so the row count in its diff
   will be one higher than a clean `origin/dev` run.

This branch's own delta, recomputed against its base `5f720a404`, is exactly
one row: `tldw_chatbook/app.py`, `call_count` unchanged at 325,
`diagnostic_digest` `bb5156c2be2e60286533` → `14fba1fb194bef30c90b`, from
rewording one `logger.debug` in `_display_buffered_logs`. No owners added or
removed; sink topology byte-identical.
