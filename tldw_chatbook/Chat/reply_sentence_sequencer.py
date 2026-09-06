"""Headless sentence-boundary sequencer for spoken replies.

Deliberately free of Textual, TTS, and wall-clock imports: `speak` and
`stop_speech` are injected callables, and every transition is driven purely
by `feed()`/`reply_completed()`/`utterance_finished()`/`flush()`/
`begin_reply()` calls from the caller. That split is what makes this module
unit-testable without a running app or a real speech path (see the
hands-free-loop design doc,
`Docs/superpowers/specs/2026-08-02-hands-free-loop-design.md`).

## Lifetime: one instance per LOOP, `begin_reply()` per REPLY

A `SentenceSequencer` is constructed once for a whole hands-free loop and
reused across every reply in it (the loop never knows in advance how many
replies there will be). The constructor already leaves the instance ready
for a FIRST reply, so calling `begin_reply()` before the first `feed()` is
optional; it becomes REQUIRED before feeding a SECOND (or later) reply's
deltas on the same instance -- it resets the per-reply buffer, fence state,
suppression latch, and the `_completed`/`drained_fired` flags so `on_drained`
can fire again. `_utterance_token` (see F2 below) deliberately survives
`begin_reply()` and stays monotonic for the instance's whole life: a stale
token from an abandoned reply must never collide with a fresh reply's token.

## Pipeline

1. **Line buffering / fence tracking** (`_consume_pending_lines`): raw
   deltas accumulate in `_pending_line` until a complete line (newline) is
   available. A fenced-code delimiter line (``` optionally followed by a
   language tag) toggles `_in_fence` and is never spoken; while `_in_fence`
   is true, whole lines are discarded. A delimiter line can arrive split
   across `feed()` calls (e.g. one delta ends in "``", the next continues
   with "`python\\n") -- the ambiguity window is exactly "the pending line
   is 1-2 backtick characters and nothing else", and we hold it back from
   being treated as regular content until either a third backtick
   disambiguates it as a fence line, a non-backtick character disproves it,
   or the reply ends (`reply_completed()` forces resolution). Regular
   (non-ambiguous) line content is committed immediately -- it does NOT
   wait for a trailing newline, since ordinary sentence text streams with
   no line breaks at all (see `test_sentences_emit_one_at_a_time...`).

   **Line-position tracking (`_at_line_start`):** a fence delimiter only
   toggles `_in_fence` when it is at the TRUE start of a markdown line, not
   merely at the start of whatever happens to be currently buffered. When
   regular content is committed early (mid-line, no newline yet -- the
   common case for ordinary prose), `_at_line_start` flips to False, and
   everything committed after that point is known to be a line
   *continuation* until the next real newline is crossed. Without this, a
   ``` arriving as its OWN delta mid-sentence (e.g. "Wrap it in " / "```" /
   " markers.") would be mistaken for a fence line START purely because it
   happened to be the first character of the currently-buffered fragment,
   silently discarding the rest of the reply.

2. **Boundary scan** (`_scan_content`): fence-filtered raw markdown text
   (newlines preserved literally, NOT collapsed to spaces yet -- see
   "Deviations" below for why this matters) accumulates in `_content`.
   `.`, `!`, `?` are candidate terminators, subject to, in order:
   - **Markdown-link guard:** a terminator inside unclosed or just-closed
     `[link text]` (tracked via a simple bracket-depth counter as the scan
     proceeds) is never a boundary -- "Read [Hello. World](url) now."
     must not split mid-link and leak the raw URL as its own utterance.
     If the buffer ends with an unmatched `[`, the scan waits for more
     data (the closing `]` may not have streamed in yet).
   - **Ellipsis rule (pinned):** a run of 2+ consecutive `.` characters
     ("..." or even "..") is never a boundary, matching the intent that
     trailing off mid-thought must not fragment speech.
   - **Decimal guard:** a `.` between two digits ("3.14") is never a
     boundary.
   - **Ordered-list guard:** a `.` immediately after a digit run that
     starts at a true line start (`_content` position 0 or right after a
     `\\n`) is a list marker ("1. First item"), not a sentence end --
     needs the newline-preserving buffer above to know what "line start"
     means by the time the scanner runs.
   - **Abbreviation guard:** a `.` immediately preceded by a known title/
     abbreviation token (case-insensitive lookbehind, see
     `_ABBREVIATIONS`) is never a boundary ("Dr. Smith").
   - A terminator only confirms as a boundary once the buffer contains at
     least one more character after it and that character is whitespace;
     if the terminator is the very last character currently buffered, the
     scanner waits for more `feed()` data rather than guessing (this is
     what lets a sentence split cleanly across delta chunks, e.g. `"Half a
     sen"` + `"tence. "`, or a decimal split exactly at the `.`, e.g.
     `"Value is 3."` + `"14 done. "`).
   Newline is treated as plain whitespace for boundary purposes (NOT an
   independent sentence boundary) -- see "Deviations" below; it is
   collapsed to a single space only at normalization time, after the
   ordered-list guard above has already used its literal position.
3. **Normalization** (`_normalize`, applied per extracted candidate, in
   order): markdown links `[text](url)` -> `text`, heading markers (`#` ..
   `######` at line start) stripped, `~~strike~~` -> `strike`, `**bold**`/
   `*em*`/`__bold__`/`_em_` (paired markers only -- a lone, unmatched `*`
   or `_`, e.g. in `2 * 3` or `variable_name`, is left alone) stripped,
   then whitespace collapsed to single spaces and trimmed. A candidate
   that normalizes to the empty string is dropped silently (this is how a
   reply that is pure code -- `test_zero_speakable_reply_drains_immediately`
   -- produces zero utterances). A candidate that normalizes to pure
   punctuation with no letters/digits at all AND is shorter than
   `MIN_SENTENCE_LENGTH` is also dropped as noise (never a real sentence
   like "A." -- see "Deviations" below).
4. **Dispatch** (`_maybe_dispatch`): exactly one utterance is ever handed
   to `speak()` at a time; the next only follows a matching
   `utterance_finished()`. An internal `_utterance_token` counter is
   bumped immediately BEFORE `speak()` is called, so a caller can read
   `seq.current_utterance_token` synchronously from inside its own
   `speak()` callback and thread it back through to
   `utterance_finished(ok, token=...)` later (see F2 below).
5. **Max-length force split** (`_force_split`): a terminator-free run
   longer than `MAX_UTTERANCE_LENGTH` is cut at the last whitespace (any
   kind -- space, tab, newline) at or before the cap, never mid-word; if
   no whitespace exists in range at all, it hard-cuts at the cap as a last
   resort.
6. **Barge-in latch** (`flush()`): clears the queue, stops the in-flight
   utterance (`stop_speech()`, exactly once iff something was in flight),
   AND sets a sticky `_suppressed` flag. While suppressed, NOTHING may
   speak again for this reply -- not the reply's own later sentences fed
   after the barge-in (reply generation is never cancelled, so deltas keep
   arriving), and not `reply_completed()`'s final-partial flush. Only
   `begin_reply()` clears the latch. `flush()` also re-checks the drained
   condition: if it lands after `reply_completed()` was already called
   (the barge-in caught the final utterance mid-playback), the reply is
   now genuinely done and `on_drained` must still fire -- otherwise a
   caller waiting on it to reopen the mic would hang forever.

## Deviations from the brief's step-3 hints (documented, evidence-based)

- **Newline is not an independent sentence boundary**, despite the design
  doc's parenthetical "(`.` `!` `?` and newline)". Treating a bare newline
  as a boundary breaks the load-bearing `test_code_fences_are_skipped_
  entirely`: "Here you go:\\n```...```\\nDone now. " would split "Here you
  go:" off as its OWN utterance, dispatch it immediately (one-utterance-at-
  a-time), and leave "Done now." queued forever since that test never
  calls `utterance_finished()`. Newlines are kept literally in `_content`
  (not immediately collapsed to a space) so the ordered-list guard above
  can use their position, then collapsed to a single space like any other
  whitespace run at normalization time.
- **The "min sentence length 4 chars" guard does not merge or drop real
  terminator-confirmed sentences.** A blanket "candidates shorter than 4
  chars merge into the next sentence" rule directly contradicts the
  load-bearing `test_failed_utterance_skips_and_continues`, which requires
  the two-character fragments "A." and "B." to be spoken as two SEPARATE
  utterances. `MIN_SENTENCE_LENGTH` is instead applied narrowly: a
  confirmed-boundary candidate that normalizes to a string with **no
  alphanumeric characters at all** (pure punctuation/symbol noise, e.g. a
  stray "?!" left over from mid-buffer punctuation) and is shorter than
  `MIN_SENTENCE_LENGTH` is dropped rather than spoken. "A." is exempt (it
  contains a letter) and is spoken normally, per the test.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

#: A terminator-confirmed candidate must normalize to at least this many
#: characters to be spoken UNLESS it contains at least one letter/digit (see
#: the "Deviations" note in the module docstring -- this never merges or
#: drops a real short sentence like "A.").
MIN_SENTENCE_LENGTH = 4

#: Hard cap on a single utterance's raw (pre-normalization) length before a
#: terminator-free run is force-split at the nearest whitespace boundary.
MAX_UTTERANCE_LENGTH = 200

#: Case-insensitive lookbehind list of common title/abbreviation tokens that
#: must not be treated as sentence-ending when followed by a period. Not
#: exhaustive -- matches the scope of typical sentence-boundary detectors
#: (only single-token abbreviations are representable here; multi-period
#: abbreviations like "e.g." are out of scope, see decimal/ellipsis guards
#: for the punctuation-heavy cases that are handled). Deliberately excludes
#: common standalone English words that also happen to be plausible
#: abbreviations ("no", "st", "co", "ft", "est", "gen", "mt") -- those swallow
#: genuine sentence boundaries far more often than they save a real
#: abbreviation (task-2 review F7: "The answer is no. Moving on now." must
#: not merge into one utterance).
_ABBREVIATIONS = frozenset(
    {
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "sr",
        "jr",
        "vs",
        "etc",
        "inc",
        "ltd",
        "fig",
        "approx",
        "capt",
        "col",
        "gov",
        "sgt",
        "ave",
        "blvd",
        "dept",
        "univ",
        "assn",
    }
)

_FENCE_RE = re.compile(r"^\s*```")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+")
_STRIKE_RE = re.compile(r"~~(.+?)~~")
_BOLD_STAR_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_STAR_RE = re.compile(r"\*(.+?)\*")
_BOLD_UNDERSCORE_RE = re.compile(r"__(.+?)__")
_ITALIC_UNDERSCORE_RE = re.compile(r"_(.+?)_")
_WS_RE = re.compile(r"\s+")


def _word_before(buf: str, idx: int) -> str:
    """Return the contiguous run of alphabetic characters immediately
    preceding `buf[idx]`, used as the abbreviation-guard lookbehind."""
    j = idx
    while j > 0 and buf[j - 1].isalpha():
        j -= 1
    return buf[j:idx]


def _find_confirmed_boundary(buf: str) -> Optional[int]:
    """Return the index of the earliest confirmed sentence-terminating
    character in `buf`, or None if no boundary can yet be confirmed (either
    none exists, or the buffer's tail is too short to disambiguate an
    ellipsis run / decimal / unclosed link / trailing terminator -- caller
    should wait for more `feed()` data in that case)."""
    n = len(buf)
    i = 0
    link_depth = 0
    while i < n:
        ch = buf[i]
        if ch == "[":
            link_depth += 1
            i += 1
            continue
        if ch == "]":
            if link_depth > 0:
                link_depth -= 1
            i += 1
            continue
        if link_depth > 0:
            i += 1
            continue  # inside markdown link text; no boundary can land here
        if ch in ".!?":
            if ch == ".":
                # Ellipsis rule: any run of 2+ '.' is never a boundary.
                if (i + 1 < n and buf[i + 1] == ".") or (i > 0 and buf[i - 1] == "."):
                    j = i
                    while j < n and buf[j] == ".":
                        j += 1
                    if j == n:
                        return None  # run extends to buffer end; wait for more
                    i = j
                    continue
                if i > 0 and buf[i - 1].isdigit():
                    # Decimal guard: '.' between two digits (e.g. "3.14").
                    if i + 1 >= n:
                        return None  # could still turn into a decimal; wait
                    if buf[i + 1].isdigit():
                        i += 1
                        continue
                    # Ordered-list guard: '<n>.' whose digit run starts at a
                    # true line start ("\n" or buffer start) is a list
                    # marker ("1. First item"), not a sentence end.
                    j = i
                    while j > 0 and buf[j - 1].isdigit():
                        j -= 1
                    if j == 0 or buf[j - 1] == "\n":
                        i += 1
                        continue
                elif _word_before(buf, i).lower() in _ABBREVIATIONS:
                    i += 1
                    continue
            if i + 1 >= n:
                return None  # terminator is the last buffered char; wait
            if buf[i + 1].isspace():
                return i
            i += 1
            continue
        i += 1
    return None


def _normalize(raw: str) -> str:
    """Markdown-strip + whitespace-collapse a raw (fence-filtered) text
    span into speakable text. See the module docstring's numbered pipeline
    for the exact order and rationale. Only PAIRED emphasis/strikethrough
    markers are stripped -- a lone, unmatched `*` or `_` (e.g. "2 * 3",
    "variable_name") is left untouched."""
    text = _LINK_RE.sub(r"\1", raw)
    text = _HEADING_RE.sub("", text)
    text = _STRIKE_RE.sub(r"\1", text)
    text = _BOLD_STAR_RE.sub(r"\1", text)
    text = _ITALIC_STAR_RE.sub(r"\1", text)
    text = _BOLD_UNDERSCORE_RE.sub(r"\1", text)
    text = _ITALIC_UNDERSCORE_RE.sub(r"\1", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


class SentenceSequencer:
    """Turns streamed reply text into gated, sequential speakable-sentence
    utterances. Pure module: no Textual, no TTS imports, no wall-clock.

    Args:
        speak: Called with exactly one utterance (str) at a time. The next
            call only follows a matching `utterance_finished()`.
        stop_speech: Called by `flush()` iff an utterance is currently in
            flight (barge-in abandon signal).
    """

    def __init__(
        self, speak: Callable[[str], None], stop_speech: Callable[[], None]
    ) -> None:
        self._speak = speak
        self._stop_speech = stop_speech

        #: Fired when `drained` becomes True after `reply_completed()`.
        #: Re-armed by `begin_reply()` -- fires (at most) once per reply.
        self.on_drained: Optional[Callable[[], None]] = None

        self._pending_line: str = ""
        self._at_line_start: bool = True
        self._in_fence: bool = False
        self._content: str = ""
        self._queue: List[str] = []
        self._inflight: bool = False
        self._completed: bool = False
        self._drained_fired: bool = False
        self._suppressed: bool = False

        #: Monotonic across the instance's WHOLE lifetime (never reset by
        #: `begin_reply()`), so a stale token from an abandoned reply can
        #: never collide with a fresh reply's token.
        self._utterance_token: int = 0

    # -- public state -----------------------------------------------------

    @property
    def drained(self) -> bool:
        """True when the queue is empty AND nothing is in flight. Does not
        by itself imply the reply is complete -- see `on_drained`, which
        additionally gates on `reply_completed()` having been called."""
        return not self._queue and not self._inflight

    @property
    def current_utterance_token(self) -> Optional[int]:
        """The token identifying the currently in-flight utterance, or None
        if nothing is in flight. Read this synchronously from inside your
        own `speak()` callback (the token is bumped before `speak()` is
        called) and pass it back via `utterance_finished(ok, token=...)`."""
        return self._utterance_token if self._inflight else None

    # -- public API ---------------------------------------------------------

    def begin_reply(self) -> None:
        """Reset per-reply state so this SAME instance can be reused for
        the next reply in the loop (one `SentenceSequencer` per loop,
        `begin_reply()` per reply). Clears the line/fence buffer, the
        barge-in suppression latch, and the completed/drained flags so
        `on_drained` can fire again. Optional before the first reply (the
        constructor already leaves the instance in this state); required
        before feeding any later reply's deltas, whether the previous
        reply drained naturally or was cut short by `flush()`."""
        self._pending_line = ""
        self._at_line_start = True
        self._in_fence = False
        self._content = ""
        self._queue.clear()
        self._completed = False
        self._drained_fired = False
        self._suppressed = False

    def feed(self, delta: str) -> None:
        """Consume a streamed reply text delta.

        Args:
            delta: The next incremental chunk of streamed reply text.
        """
        if self._drained_fired:
            # A reused instance whose caller forgot `begin_reply()`: the
            # symptom is otherwise SILENT (speech still works, but
            # `on_drained` never fires again, so the loop never reopens the
            # microphone -- Task-2 re-review N1). Say so once per lapse.
            logger.warning(
                "SentenceSequencer.feed() after drain without begin_reply(); "
                "on_drained will not fire for this reply"
            )
        self._pending_line += delta
        self._consume_pending_lines(final=False)
        self._scan_content()
        self._maybe_dispatch()

    def reply_completed(self) -> None:
        """Flush the final partial sentence (if any) and mark the reply
        complete. Idempotent: calling it again after the queue/in-flight
        state is already drained does not re-fire `on_drained`. If the
        reply was barge-in-suppressed (`flush()` already latched it),
        the final partial is discarded, not spoken."""
        self._consume_pending_lines(final=True)
        remainder = self._content
        self._content = ""
        self._emit(_normalize(remainder))
        self._completed = True
        self._maybe_dispatch()
        self._check_drained()

    def utterance_finished(self, ok: bool, token: Optional[int] = None) -> None:
        """Completion signal from the speech path for the currently
        in-flight utterance. A failed utterance (`ok=False`) is skipped,
        not retried -- the queue simply advances, matching every other
        completion.

        Args:
            ok: Whether the utterance played successfully.
            token: The utterance identity from `current_utterance_token` at
                the time `speak()` was called for it. Production callers
                (the hands-free speech entry) MUST thread this token --
                `None` degrades to "whatever is currently in flight", which
                cannot distinguish a LATE completion for a superseded
                utterance from the current one and reopens the double-voice
                defect (Task-2 review F2). `None` exists only for unit tests
                that never overlap utterances. A real token that no longer
                matches the in-flight utterance is ignored rather than
                misattributed.
        """
        if not self._inflight:
            return  # stale/late signal with nothing in flight (see flush())
        if token is not None and token != self._utterance_token:
            return  # late completion for a superseded utterance; ignore
        self._inflight = False
        if not ok:
            logger.debug("SentenceSequencer: utterance failed; skipping to next")
        self._maybe_dispatch()
        self._check_drained()

    def flush(self) -> None:
        """Barge-in: clear the queue, abandon the in-flight utterance
        (calling `stop_speech()` exactly once iff something was in
        flight), and latch suppression -- nothing speaks again for this
        reply until `begin_reply()`. A LATE `utterance_finished()` for the
        abandoned utterance that arrives afterward is a safe no-op
        (nothing is in flight by then). Re-checks the drained condition in
        case this lands after `reply_completed()` already ran (see the
        module docstring's barge-in-latch section)."""
        self._suppressed = True
        self._queue.clear()
        if self._inflight:
            self._inflight = False
            self._stop_speech()
        self._check_drained()

    # -- line buffering / fence tracking ------------------------------------

    def _consume_pending_lines(self, final: bool) -> None:
        while True:
            nl_idx = self._pending_line.find("\n")
            if nl_idx != -1:
                line = self._pending_line[:nl_idx]
                self._pending_line = self._pending_line[nl_idx + 1 :]
                self._resolve_line(
                    line, had_newline=True, at_line_start=self._at_line_start
                )
                self._at_line_start = True  # whatever follows starts a fresh line
                continue

            tail = self._pending_line
            if final:
                self._pending_line = ""
                if tail:
                    self._resolve_line(
                        tail, had_newline=False, at_line_start=self._at_line_start
                    )
                    self._at_line_start = False
                return

            if not self._at_line_start:
                # Mid-line continuation: can never be a fence delimiter (it
                # is not at a true line start), so there is no ambiguity to
                # wait out -- commit immediately.
                if tail:
                    self._pending_line = ""
                    self._resolve_line(tail, had_newline=False, at_line_start=False)
                return

            if self._is_ambiguous_fence_prefix(tail):
                return  # a future delta may still complete/disprove this fence line
            if _FENCE_RE.match(tail):
                return  # confirmed fence line; still need its newline to toggle+consume
            # Definitely regular content, at a true line start, not
            # fence-shaped: commit now, without waiting for a newline that
            # may never come this call.
            self._pending_line = ""
            self._resolve_line(tail, had_newline=False, at_line_start=True)
            self._at_line_start = False  # anything arriving next is mid-line
            return

    @staticmethod
    def _is_ambiguous_fence_prefix(tail: str) -> bool:
        stripped = tail.lstrip()
        return len(stripped) <= 2 and all(c == "`" for c in stripped)

    def _resolve_line(self, line: str, had_newline: bool, at_line_start: bool) -> None:
        if at_line_start and _FENCE_RE.match(line):
            self._in_fence = not self._in_fence
            return  # delimiter line itself is never spoken
        if self._in_fence:
            return  # fence body, discarded
        self._content += line
        if had_newline:
            # Kept literal (not collapsed to a space) so the ordered-list
            # guard in _find_confirmed_boundary can see true line starts;
            # collapsed like any other whitespace at normalization time.
            self._content += "\n"

    # -- boundary scan / dispatch --------------------------------------------

    def _scan_content(self) -> None:
        while True:
            idx = _find_confirmed_boundary(self._content)
            if idx is not None:
                raw = self._content[: idx + 1]
                self._content = self._content[idx + 1 :]
                self._emit(_normalize(raw))
                continue
            if len(self._content) > MAX_UTTERANCE_LENGTH:
                self._force_split()
                continue
            return

    def _force_split(self) -> None:
        cap = MAX_UTTERANCE_LENGTH
        limit = min(cap, len(self._content) - 1)
        split_at = -1
        for idx in range(limit, -1, -1):
            if self._content[idx].isspace():
                split_at = idx
                break
        if split_at <= 0:
            split_at = cap  # no whitespace in range; hard cut as a fallback
        raw = self._content[:split_at]
        self._emit(_normalize(raw))
        self._content = self._content[split_at:].lstrip()

    def _emit(self, normalized: str) -> None:
        if self._suppressed:
            return  # barge-in latch: nothing queues for this reply again
        if not normalized:
            return
        if len(normalized) < MIN_SENTENCE_LENGTH and not any(
            c.isalnum() for c in normalized
        ):
            return  # pure punctuation noise below the floor; see module docstring
        self._queue.append(normalized)

    def _maybe_dispatch(self) -> None:
        if self._suppressed or self._inflight or not self._queue:
            return
        text = self._queue.pop(0)
        self._utterance_token += 1
        self._inflight = True
        self._speak(text)

    def _check_drained(self) -> None:
        if (
            self._completed
            and not self._queue
            and not self._inflight
            and not self._drained_fired
        ):
            self._drained_fired = True
            if self.on_drained:
                self.on_drained()
