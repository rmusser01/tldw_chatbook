"""Server-compatible character emote directive parsing."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

from tldw_chatbook.Character_Chat.visual_identity import (
    CUSTOM_EXPRESSION_PREFIX,
    normalize_expression_key,
)

EMOTE_EVENT_LIMIT = 5
EMOTE_PROMPT_STATE_LIMIT = 25
STREAM_PREFIX_BUFFER_LIMIT = 64
CHARACTER_EMOTE_STATE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")

_DIRECTIVE_PATTERN = re.compile(r"^emote:(.*)$", re.IGNORECASE)
_NO_DIRECTIVE = object()


@dataclass(frozen=True, slots=True)
class CharacterEmoteEvent:
    """One accepted expression state at a sanitized UTF-16 offset."""

    state: str
    at_char: int


@dataclass(frozen=True, slots=True)
class CharacterEmoteParseResult:
    """Sanitized visible text and accepted expression events."""

    clean_text: str
    events: tuple[CharacterEmoteEvent, ...]


@dataclass(frozen=True, slots=True)
class CharacterEmoteStreamResult:
    """Newly visible text and expression events from one parser operation."""

    visible_text: str
    events: tuple[CharacterEmoteEvent, ...]


@dataclass(frozen=True, slots=True)
class CharacterEmoteAssetReference:
    """Profile-local immutable asset identity for one projected state."""

    state: str
    expression_key: str
    asset_id: int
    expression_id: int | None = None


@dataclass(frozen=True, slots=True)
class CharacterEmoteRunSnapshot:
    """One immutable authority shared by prompt, capture, and resolution."""

    actor_id: int | None = None
    pack_id: int | None = None
    pack_version_id: int | None = None
    states: tuple[str, ...] = ()
    assets: tuple[CharacterEmoteAssetReference, ...] = ()
    fallback_reason: str = ""

    def asset_for_state(self, state: str) -> CharacterEmoteAssetReference | None:
        """Return the exact captured asset identity for a projected slug."""

        return next((asset for asset in self.assets if asset.state == state), None)


def normalize_character_emote_state(value: object) -> str | None:
    """Return a normalized safe state slug when valid."""

    if not isinstance(value, str):
        return None
    normalized = re.sub(r"\s+", "-", value.strip().lower())
    if CHARACTER_EMOTE_STATE_PATTERN.fullmatch(normalized) is None:
        return None
    return normalized


def utf16_length(value: str) -> int:
    """Return JavaScript-compatible string length in UTF-16 code units."""

    return len(value.encode("utf-16-le", errors="surrogatepass")) // 2


def _iter_lines(text: str) -> Iterator[tuple[str, str]]:
    index = 0
    while index < len(text):
        newline_index = text.find("\n", index)
        if newline_index == -1:
            yield text[index:], ""
            break
        yield text[index:newline_index], "\n"
        index = newline_index + 1


def _is_fence_line(line: str) -> bool:
    return line.strip().startswith("```")


def _parse_directive_state(line: str) -> str | None | object:
    match = _DIRECTIVE_PATTERN.fullmatch(line.strip())
    if match is None:
        return _NO_DIRECTIVE
    return normalize_character_emote_state(match.group(1))


def parse_character_emote_directives(text: str) -> CharacterEmoteParseResult:
    """Return a sanitized one-shot character completion."""

    clean_parts: list[str] = []
    events: list[CharacterEmoteEvent] = []
    clean_length = 0
    in_fence = False
    last_state: str | None = None

    for line, separator in _iter_lines(text):
        if _is_fence_line(line):
            visible = line + separator
            clean_parts.append(visible)
            clean_length += utf16_length(visible)
            in_fence = not in_fence
            continue

        if not in_fence:
            state = _parse_directive_state(line)
            if state is not _NO_DIRECTIVE:
                if (
                    isinstance(state, str)
                    and state != last_state
                    and len(events) < EMOTE_EVENT_LIMIT
                ):
                    events.append(
                        CharacterEmoteEvent(state=state, at_char=clean_length)
                    )
                    last_state = state
                continue

        visible = line + separator
        clean_parts.append(visible)
        clean_length += utf16_length(visible)

    return CharacterEmoteParseResult(
        clean_text="".join(clean_parts),
        events=tuple(events),
    )


def project_character_emote_states(
    assets: Iterable[Mapping[str, object] | object],
) -> tuple[str, ...]:
    """Project ordered safe emote slugs from canonical expression keys."""

    candidates: list[tuple[str, str]] = []
    keys_by_slug: dict[str, set[str]] = {}
    for asset in assets:
        raw_key = (
            asset.get("expression_key")
            if isinstance(asset, Mapping)
            else getattr(asset, "expression_key", None)
        )
        if not isinstance(raw_key, str):
            continue
        slug = (
            raw_key[len(CUSTOM_EXPRESSION_PREFIX) :]
            if raw_key.startswith(CUSTOM_EXPRESSION_PREFIX)
            else raw_key
        )
        if normalize_character_emote_state(slug) != slug:
            continue
        if normalize_expression_key(slug) != raw_key:
            continue
        candidates.append((slug, raw_key))
        keys_by_slug.setdefault(slug, set()).add(raw_key)

    states: list[str] = []
    seen: set[str] = set()
    for slug, _raw_key in candidates:
        if slug in seen or len(keys_by_slug[slug]) != 1:
            continue
        states.append(slug)
        seen.add(slug)
    return tuple(states)


def append_character_emote_prompt_instruction(
    system_prompt: str,
    states: Iterable[str],
) -> str:
    """Append the pinned emote instruction with a bounded safe inventory."""

    safe_states: list[str] = []
    seen: set[str] = set()
    for state in states:
        normalized = normalize_character_emote_state(state)
        if normalized != state or state in seen:
            continue
        safe_states.append(state)
        seen.add(state)

    visible_states = safe_states[:EMOTE_PROMPT_STATE_LIMIT]
    hidden_count = len(safe_states) - len(visible_states)
    suffix = f" (+{hidden_count} more)" if hidden_count else ""
    prefer = (
        f" Prefer these available states: {', '.join(visible_states)}{suffix}."
        if visible_states
        else ""
    )
    instruction = (
        "When the character expression should change, emit a standalone line exactly "
        "like `Emote: <state>`."
        f"{prefer} Do not emit an emote after every sentence."
    )
    base = system_prompt.strip()
    return f"{base}\n\n{instruction}" if base else instruction


class CharacterEmoteStreamParser:
    """Incrementally strip character emote directives from visible text."""

    def __init__(self) -> None:
        self._mode = "prefix"
        self._prefix = ""
        self._in_fence = False
        self._clean_length = 0
        self._event_count = 0
        self._last_state: str | None = None
        self._directive_chars: list[str] = []
        self._directive_pending_space = False
        self._directive_invalid = False
        self._finished = False

    @property
    def pending_char_count(self) -> int:
        """Return the number of buffered, not-yet-published characters."""

        return len(self._prefix) + len(self._directive_chars)

    def push(self, chunk: str) -> CharacterEmoteStreamResult:
        """Consume a completion chunk without publishing control syntax."""

        if self._finished or not chunk:
            return CharacterEmoteStreamResult("", ())

        working = self._clone()
        visible_parts: list[str] = []
        events: list[CharacterEmoteEvent] = []
        for character in chunk:
            working._consume(character, visible_parts, events)
        self._adopt(working)
        return CharacterEmoteStreamResult("".join(visible_parts), tuple(events))

    def safe_copy(self) -> CharacterEmoteStreamParser:
        """Return a base-parser checkpoint suitable for fail-closed recovery."""

        clone = CharacterEmoteStreamParser()
        clone._adopt(self)
        clone._directive_chars = self._directive_chars.copy()
        return clone

    def flush(self) -> CharacterEmoteStreamResult:
        """Finalize a successful stream and publish any ordinary suffix."""

        if self._finished:
            return CharacterEmoteStreamResult("", ())

        working = self._clone()
        visible_parts: list[str] = []
        events: list[CharacterEmoteEvent] = []
        if working._mode == "directive":
            working._finish_directive(events)
        elif working._mode == "prefix":
            working._publish(working._prefix, visible_parts)
            working._prefix = ""
        working._finished = True
        self._adopt(working)
        return CharacterEmoteStreamResult("".join(visible_parts), tuple(events))

    def cancel(self) -> CharacterEmoteStreamResult:
        """Discard an incomplete control candidate after cancellation."""

        if self._finished:
            return CharacterEmoteStreamResult("", ())
        self._prefix = ""
        self._reset_directive()
        self._finished = True
        return CharacterEmoteStreamResult("", ())

    def _clone(self) -> CharacterEmoteStreamParser:
        clone = object.__new__(type(self))
        clone._mode = self._mode
        clone._prefix = self._prefix
        clone._in_fence = self._in_fence
        clone._clean_length = self._clean_length
        clone._event_count = self._event_count
        clone._last_state = self._last_state
        clone._directive_chars = self._directive_chars.copy()
        clone._directive_pending_space = self._directive_pending_space
        clone._directive_invalid = self._directive_invalid
        clone._finished = self._finished
        return clone

    def _adopt(self, other: CharacterEmoteStreamParser) -> None:
        self._mode = other._mode
        self._prefix = other._prefix
        self._in_fence = other._in_fence
        self._clean_length = other._clean_length
        self._event_count = other._event_count
        self._last_state = other._last_state
        self._directive_chars = other._directive_chars
        self._directive_pending_space = other._directive_pending_space
        self._directive_invalid = other._directive_invalid
        self._finished = other._finished

    def _consume(
        self,
        character: str,
        visible_parts: list[str],
        events: list[CharacterEmoteEvent],
    ) -> None:
        if self._mode == "ordinary":
            self._publish(character, visible_parts)
            if character == "\n":
                self._mode = "prefix"
            return

        if self._mode == "fence":
            self._publish(character, visible_parts)
            if character == "\n":
                self._in_fence = not self._in_fence
                self._mode = "prefix"
            return

        if self._mode == "directive":
            if character == "\n":
                self._finish_directive(events)
                self._mode = "prefix"
            else:
                self._consume_directive_character(character)
            return

        if character == "\n":
            self._publish(self._prefix + character, visible_parts)
            self._prefix = ""
            return

        self._prefix += character
        stripped = self._prefix.lstrip()
        lowered = stripped.lower()

        if not self._in_fence and lowered.startswith("emote:"):
            remainder = stripped[len("emote:") :]
            self._prefix = ""
            self._mode = "directive"
            for item in remainder:
                self._consume_directive_character(item)
            return

        if stripped.startswith("```"):
            buffered = self._prefix
            self._prefix = ""
            self._mode = "fence"
            self._publish(buffered, visible_parts)
            return

        possible_directive = (
            not self._in_fence and "emote:".startswith(lowered)
        )
        possible_fence = "```".startswith(stripped)
        if possible_directive or possible_fence:
            if len(self._prefix) <= STREAM_PREFIX_BUFFER_LIMIT:
                return

        buffered = self._prefix
        self._prefix = ""
        self._mode = "ordinary"
        self._publish(buffered, visible_parts)

    def _consume_directive_character(self, character: str) -> None:
        if self._directive_invalid:
            return
        if character.isspace():
            if self._directive_chars:
                self._directive_pending_space = True
            return

        if self._directive_pending_space:
            self._append_directive_character("-")
            self._directive_pending_space = False
        self._append_directive_character(character.lower())

    def _append_directive_character(self, character: str) -> None:
        if self._directive_invalid:
            return
        if not (character.isascii() and (character.isalnum() or character in "_-")):
            self._directive_invalid = True
            self._directive_chars.clear()
            return
        if len(self._directive_chars) >= 40:
            self._directive_invalid = True
            self._directive_chars.clear()
            return
        self._directive_chars.append(character)

    def _finish_directive(self, events: list[CharacterEmoteEvent]) -> None:
        state = "".join(self._directive_chars)
        if (
            not self._directive_invalid
            and CHARACTER_EMOTE_STATE_PATTERN.fullmatch(state) is not None
            and state != self._last_state
            and self._event_count < EMOTE_EVENT_LIMIT
        ):
            event = CharacterEmoteEvent(state, self._clean_length)
            events.append(event)
            self._last_state = state
            self._event_count += 1
        self._reset_directive()

    def _reset_directive(self) -> None:
        self._directive_chars = []
        self._directive_pending_space = False
        self._directive_invalid = False

    def _publish(self, text: str, visible_parts: list[str]) -> None:
        if not text:
            return
        visible_parts.append(text)
        self._clean_length += utf16_length(text)
