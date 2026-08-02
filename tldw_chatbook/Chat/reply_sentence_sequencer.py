"""Headless sentence-boundary sequencer for spoken replies.

Deliberately free of Textual, TTS, and wall-clock imports: `speak` and
`stop_speech` are injected callables, and every transition is driven purely
by `feed()`/`reply_completed()`/`utterance_finished()`/`flush()` calls from
the caller. That split is what makes this module unit-testable without a
running app or a real speech path (see the hands-free-loop design doc,
`Docs/superpowers/specs/2026-08-02-hands-free-loop-design.md`).

## Pipeline

1. **Line buffering / fence tracking** (`_consume_pending_lines`): raw
   deltas accumulate in `_pending_line` until a complete line (newline) is
   available. A fenced-code delimiter line (``` optionally followed by a
   language tag) toggles `_in_fence` and is never spoken; while
   `_in_fence` is true, whole lines are discarded. A delimiter line can
   arrive split across `feed()` calls (e.g. one delta ends in "``", the
   next continues with "`python\\n") -- the ambiguity window is exactly
   "the pending line is 1-2 backtick characters and nothing else", and we
   hold it back from being treated as regular content until either a third
   backtick disambiguates it as a fence line, a non-backtick character
   disproves it, or the reply ends (`reply_completed()` forces resolution).
   Regular (non-ambiguous) line content is committed immediately -- it does
   NOT wait for a trailing newline, since ordinary sentence text streams
   with no line breaks at all (see `test_sentences_emit_one_at_a_time...`).
2. **Boundary scan** (`_scan_content`): fence-filtered raw markdown text
   accumulates in `_content`. `.`, `!`, `?` are candidate terminators,
   subject to:
   - **Ellipsis rule (pinned):** a run of 2+ consecutive `.` characters
     ("..." or even "..") is never a boundary, matching the intent that
     trailing off mid-thought must not fragment speech. This is checked
     before the decimal/abbreviation guards below.
   - **Decimal guard:** a `.` between two digits ("3.14") is never a
     boundary.
   - **Abbreviation guard:** a `.` immediately preceded by a known title/
     abbreviation token (case-insensitive lookbehind, see
     `_ABBREVIATIONS`) is never a boundary ("Dr. Smith").
   - A terminator only confirms as a boundary once the buffer contains at
     least one more character after it and that character is whitespace;
     if the terminator is the very last character currently buffered, the
     scanner waits for more `feed()` data rather than guessing (this is
     what lets a sentence split cleanly across delta chunks, e.g. `"Half a
     sen"` + `"tence. "`).
   Newline is treated as plain whitespace here, NOT as an independent
   sentence boundary -- see "Deviation" below.
3. **Normalization** (`_normalize`, applied per extracted candidate, in
   order): markdown links `[text](url)` -> `text`, heading markers (`#`
   .. `######` at line start) stripped, `*`/`**` emphasis markers stripped,
   then whitespace collapsed to single spaces and trimmed. A candidate that
   normalizes to the empty string is dropped silently (this is how a
   reply that is pure code -- `test_zero_speakable_reply_drains_immediately`
   -- produces zero utterances).
4. **Dispatch** (`_maybe_dispatch`): exactly one utterance is ever handed
   to `speak()` at a time; the next only follows `utterance_finished()`.
5. **Max-length force split** (`_force_split`): a terminator-free run
   longer than `MAX_UTTERANCE_LENGTH` is cut at the last whitespace at or
   before the cap (never mid-word) so the buffer cannot grow without bound
   while waiting for punctuation that may never arrive.

## Deviations from the brief's step-3 hints (documented, evidence-based)

- **Newline is not an independent sentence boundary**, despite the design
  doc's parenthetical "(`.` `!` `?` and newline)". Treating a bare newline
  as a boundary breaks the load-bearing `test_code_fences_are_skipped_
  entirely`: "Here you go:\\n```...```\\nDone now. " would split "Here you
  go:" off as its OWN utterance, dispatch it immediately (one-utterance-at-
  a-time), and leave "Done now." queued forever since that test never
  calls `utterance_finished()`. Newlines are folded into `_content` as
  plain whitespace (collapsed like any other run of whitespace at
  normalization time); they still matter structurally for fence-line
  detection.
- **The "min sentence length 4 chars" guard does not merge or drop real
  terminator-confirmed sentences.** A blanket "candidates shorter than 4
  chars merge into the next sentence" rule directly contradicts the
  load-bearing `test_failed_utterance_skips_and_continues`, which requires
  the two-character fragments "A." and "B." to be spoken as two SEPARATE
  utterances. `MIN_SENTENCE_LENGTH` is instead applied narrowly: a
  confirmed-boundary candidate that normalizes to a string with **no
  alphanumeric characters at all** (pure punctuation/symbol noise, e.g. a
  stray "!!" left over after markdown stripping) and is shorter than
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
#: for the punctuation-heavy cases that are handled).
_ABBREVIATIONS = frozenset(
    {
        "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st",
        "vs", "etc", "inc", "ltd", "co", "fig", "no", "approx",
        "gen", "rev", "capt", "lt", "col", "gov", "sgt", "mt",
        "ft", "ave", "blvd", "dept", "univ", "assn", "est",
    }
)

_FENCE_RE = re.compile(r"^\s*```")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+")
_EMPHASIS_RE = re.compile(r"\*\*|\*")
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
    ellipsis run / decimal / trailing terminator -- caller should wait for
    more `feed()` data in that case)."""
    n = len(buf)
    i = 0
    while i < n:
        ch = buf[i]
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
                # Decimal guard: '.' between two digits.
                if i > 0 and buf[i - 1].isdigit():
                    if i + 1 >= n:
                        return None  # need to see what follows the digit
                    if buf[i + 1].isdigit():
                        i += 1
                        continue
                # Abbreviation guard.
                if _word_before(buf, i).lower() in _ABBREVIATIONS:
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
    for the exact order and rationale."""
    text = _LINK_RE.sub(r"\1", raw)
    text = _HEADING_RE.sub("", text)
    text = _EMPHASIS_RE.sub("", text)
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

    def __init__(self, speak: Callable[[str], None], stop_speech: Callable[[], None]) -> None:
        self._speak = speak
        self._stop_speech = stop_speech

        #: Fired when `drained` becomes True after `reply_completed()`.
        #: Exactly once per instance (a fresh `SentenceSequencer` is
        #: expected per reply -- there is no reset API by design).
        self.on_drained: Optional[Callable[[], None]] = None

        self._pending_line: str = ""
        self._in_fence: bool = False
        self._content: str = ""
        self._queue: List[str] = []
        self._inflight: bool = False
        self._completed: bool = False
        self._drained_fired: bool = False

    # -- public state -----------------------------------------------------

    @property
    def drained(self) -> bool:
        """True when the queue is empty AND nothing is in flight. Does not
        by itself imply the reply is complete -- see `on_drained`, which
        additionally gates on `reply_completed()` having been called."""
        return not self._queue and not self._inflight

    # -- public API ---------------------------------------------------------

    def feed(self, delta: str) -> None:
        """Consume a streamed reply text delta."""
        self._pending_line += delta
        self._consume_pending_lines(final=False)
        self._scan_content()
        self._maybe_dispatch()

    def reply_completed(self) -> None:
        """Flush the final partial sentence (if any) and mark the reply
        complete. Idempotent: calling it again after the queue/in-flight
        state is already drained does not re-fire `on_drained`."""
        self._consume_pending_lines(final=True)
        remainder = self._content
        self._content = ""
        normalized = _normalize(remainder)
        if normalized:
            self._queue.append(normalized)
        self._completed = True
        self._maybe_dispatch()
        self._check_drained()

    def utterance_finished(self, ok: bool) -> None:
        """Completion signal from the speech path for the currently
        in-flight utterance. A failed utterance (`ok=False`) is skipped,
        not retried -- the queue simply advances, matching every other
        completion."""
        if not self._inflight:
            return  # stale/late signal with nothing in flight (see flush())
        self._inflight = False
        if not ok:
            logger.debug("SentenceSequencer: utterance failed; skipping to next")
        self._maybe_dispatch()
        self._check_drained()

    def flush(self) -> None:
        """Barge-in: clear the queue and abandon the in-flight utterance
        (calling `stop_speech()` exactly once iff something was in
        flight). A LATE `utterance_finished()` for the abandoned utterance
        that arrives afterward is a safe no-op (nothing is in flight by
        then, so it returns immediately without dispatching anything)."""
        self._queue.clear()
        if self._inflight:
            self._inflight = False
            self._stop_speech()

    # -- line buffering / fence tracking ------------------------------------

    def _consume_pending_lines(self, final: bool) -> None:
        while True:
            nl_idx = self._pending_line.find("\n")
            if nl_idx != -1:
                line = self._pending_line[:nl_idx]
                self._pending_line = self._pending_line[nl_idx + 1:]
                self._resolve_line(line, had_newline=True)
                continue

            tail = self._pending_line
            if final:
                self._pending_line = ""
                if tail:
                    self._resolve_line(tail, had_newline=False)
                return

            if self._is_ambiguous_fence_prefix(tail):
                return  # a future delta may still complete/disprove this fence line
            if _FENCE_RE.match(tail):
                return  # confirmed fence line; still need its newline to toggle+consume
            # Definitely regular content (not fence-line-shaped): commit now,
            # without waiting for a newline that may never come this call.
            self._pending_line = ""
            self._resolve_line(tail, had_newline=False)
            return

    @staticmethod
    def _is_ambiguous_fence_prefix(tail: str) -> bool:
        stripped = tail.lstrip()
        return len(stripped) <= 2 and all(c == "`" for c in stripped)

    def _resolve_line(self, line: str, had_newline: bool) -> None:
        if _FENCE_RE.match(line):
            self._in_fence = not self._in_fence
            return  # delimiter line itself is never spoken
        if self._in_fence:
            return  # fence body, discarded
        self._content += line
        if had_newline:
            self._content += " "  # structural separator only, not a boundary

    # -- boundary scan / dispatch --------------------------------------------

    def _scan_content(self) -> None:
        while True:
            idx = _find_confirmed_boundary(self._content)
            if idx is not None:
                raw = self._content[: idx + 1]
                self._content = self._content[idx + 1:]
                self._emit(_normalize(raw))
                continue
            if len(self._content) > MAX_UTTERANCE_LENGTH:
                self._force_split()
                continue
            return

    def _force_split(self) -> None:
        cap = MAX_UTTERANCE_LENGTH
        split_at = self._content.rfind(" ", 0, cap + 1)
        if split_at <= 0:
            split_at = cap  # no whitespace in range; hard cut as a fallback
        raw = self._content[:split_at]
        self._emit(_normalize(raw))
        self._content = self._content[split_at:].lstrip(" ")

    def _emit(self, normalized: str) -> None:
        if not normalized:
            return
        if len(normalized) < MIN_SENTENCE_LENGTH and not any(c.isalnum() for c in normalized):
            return  # pure punctuation noise below the floor; see module docstring
        self._queue.append(normalized)

    def _maybe_dispatch(self) -> None:
        if self._inflight or not self._queue:
            return
        text = self._queue.pop(0)
        self._inflight = True
        self._speak(text)

    def _check_drained(self) -> None:
        if self._completed and not self._queue and not self._inflight and not self._drained_fired:
            self._drained_fired = True
            if self.on_drained:
                self.on_drained()
