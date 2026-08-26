"""Pinned server-WebUI character mood fallback classifier."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

CHARACTER_MOOD_LABELS = (
    "neutral",
    "happy",
    "excited",
    "sad",
    "angry",
    "thinking",
    "confused",
    "surprised",
)

_TOPIC_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "also",
        "because",
        "before",
        "between",
        "could",
        "great",
        "hello",
        "please",
        "should",
        "thanks",
        "there",
        "their",
        "these",
        "those",
        "through",
        "would",
        "while",
        "which",
        "where",
        "when",
        "what",
        "your",
        "yours",
        "have",
        "with",
        "this",
        "that",
        "from",
        "they",
        "them",
        "been",
        "into",
        "then",
        "than",
        "just",
        "dont",
        "cant",
        "wont",
        "lets",
    }
)

_MOOD_PATTERNS = {
    "happy": (
        re.compile(
            r"\b(happy|glad|joy|cheerful|delighted|nice|great|awesome|lovely)\b"
        ),
        re.compile(r"\b(thank you|thanks|appreciate it)\b"),
    ),
    "excited": (
        re.compile(
            r"\b(excited|amazing|incredible|fantastic|let'?s go|hyped|thrilled)\b"
        ),
        re.compile(r"!{1,}"),
    ),
    "sad": (
        re.compile(r"\b(sad|sorry|apolog(?:y|ize)|unfortunately|regret|upset)\b"),
        re.compile(r"\b(i'?m sorry|i am sorry)\b"),
    ),
    "angry": (
        re.compile(r"\b(angry|mad|furious|annoyed|frustrated|rage|outrage)\b"),
        re.compile(r"\b(hate|ridiculous|unacceptable)\b"),
    ),
    "thinking": (
        re.compile(
            r"\b(think|consider|analy(?:ze|sis)|reason|step by step|let'?s break)\b"
        ),
        re.compile(r"\b(maybe|perhaps|possibly)\b"),
    ),
    "confused": (
        re.compile(
            r"\b(confused|unclear|unsure|uncertain|puzzled|don'?t understand)\b"
        ),
        re.compile(r"\b(not sure|hard to tell)\b"),
    ),
    "surprised": (
        re.compile(
            r"\b(surprised|unexpected|whoa|wow|didn'?t expect|astonished|shocked)\b"
        ),
        re.compile(r"\?{2,}"),
    ),
}


@dataclass(frozen=True, slots=True)
class CharacterMoodDetection:
    """One bounded mood fallback result."""

    label: str
    confidence: float
    topic: str | None


def _normalize_text(value: object) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _clamp_confidence(value: float) -> float:
    return max(0.35, min(0.98, value))


def _javascript_two_decimal_number(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _extract_topic(assistant_text: str, user_text: str | None) -> str | None:
    combined = f"{user_text or ''} {assistant_text}".lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", combined)
    words = [
        token
        for token in re.split(r"\s+", cleaned)
        if len(token) >= 4 and token not in _TOPIC_STOPWORDS
    ]
    if not words:
        return None

    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1

    winner = ""
    winner_count = 0
    for word, count in counts.items():
        if count > winner_count:
            winner = word
            winner_count = count
    return winner[:40] if winner else None


def detect_character_mood(
    *,
    assistant_text: str,
    user_text: str | None = None,
) -> CharacterMoodDetection:
    """Classify sanitized visible text using the pinned server-WebUI heuristic.

    Cost (TASK-22227): 14 compiled-pattern scans plus the two topic passes,
    all linear in ``len(assistant_text) + len(user_text)``. Measured at
    ~2.3 ms for a 16k-char turn (~9 ms at a degenerate 64k), and the store
    calls it at most ONCE per completed character turn at the terminal seam
    -- bounded, so it deliberately stays on the event loop rather than
    paying an off-thread hop. The heuristic itself is pinned to the server
    corpus; do not trim or truncate its input to save time.

    Args:
        assistant_text: Sanitized assistant-visible text to classify.
        user_text: Optional user-visible text that provides topic context.

    Returns:
        The bounded mood label, confidence, and optional topic.
    """

    normalized_assistant = _normalize_text(assistant_text)
    normalized_user = _normalize_text(user_text)
    combined = f"{normalized_assistant} {normalized_user}".strip()
    if not combined:
        return CharacterMoodDetection("neutral", 0.4, None)

    scores = {label: 0.0 for label in CHARACTER_MOOD_LABELS}
    for label, patterns in _MOOD_PATTERNS.items():
        for pattern in patterns:
            scores[label] += sum(1 for _match in pattern.finditer(combined))

    question_marks = assistant_text.count("?")
    if question_marks > 0:
        scores["thinking"] += 0.35 * question_marks

    exclamation_marks = assistant_text.count("!")
    if exclamation_marks > 1:
        scores["excited"] += 0.2 * exclamation_marks

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_label, top_score = ranked[0]
    second_score = ranked[1][1]
    confidence = _clamp_confidence(
        0.42 + top_score * 0.12 + (top_score - second_score) * 0.05
    )
    label = top_label
    if top_score < 0.85:
        label = "neutral"
        confidence = _clamp_confidence(0.5 - max(0.0, top_score) * 0.06)
    if label == "neutral":
        confidence = min(confidence, 0.72)

    return CharacterMoodDetection(
        label=label,
        confidence=_javascript_two_decimal_number(confidence),
        topic=_extract_topic(assistant_text, user_text),
    )
