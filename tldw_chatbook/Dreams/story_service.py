"""[dreams] Story generation: one candidate in, one honest row out.

``generate_story`` wraps one chat call per candidate with the metadata
hallucination guard (spec §event metadata): ``kind``/``event_date`` are
extracted ONLY from explicit text in the candidate's title + snippet, and
``location`` only when the configured region actually appears in that source
text — absent evidence stores ``None``. Every outcome is a row-shaped
``StoryResult`` (``complete``/``empty``/``failed``); nothing raises.

``resolve_dreams_chat`` is the provider seam the cycle injects: it copies
``briefing_service.resolve_persisted_briefing_defaults``'s resolution chain
(``[dreams] provider``/``model`` when set, else the remembered chat defaults)
and returns a closure with ``api_endpoint``/``api_key``/``model`` pre-bound —
the kwargs ``query_synthesis._invoke`` deliberately does not carry.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from loguru import logger

from tldw_chatbook.Chat.Chat_Functions import (
    chat_api_call,
    extract_response_content,
)
from tldw_chatbook.Chat.provider_readiness import get_provider_readiness
from tldw_chatbook.Chat.provider_setup_persistence import (
    canonical_provider_key,
    resolve_remembered_provider_model,
)
from tldw_chatbook.config import load_cli_config_and_ensure_existence
from tldw_chatbook.Dreams.discovery import Candidate
from tldw_chatbook.Dreams.settings import dreams_setting
from tldw_chatbook.Library.ingest_analysis import chat_dispatch_name

#: Fixed sampling for the one story call: 120-200 words fits comfortably, and
#: a plain narrative (not a list) wants a middling temperature.
_STORY_MAX_TOKENS = 768
_STORY_TEMPERATURE = 0.7

_STORY_SYSTEM_PROMPT = (
    "You write Dreams stories: one discovered item turned into a short, "
    "personal story. Write 120-200 words, in second person (\"you\"), saying "
    "what this is and why it fits the user right now given their interests. "
    "Use only the source material provided; never invent dates, places, "
    "prices, or other facts it does not state. Return only the story body."
)

#: Month-name + day + optional year, e.g. "Nov 3 2026", "November 3", "Sep. 22".
#: The ``(?!\d)`` day-guard stops the day group from swallowing the first
#: two digits of a 4-digit year: "Sept 2026" must yield NO date (there is
#: no day in that text), not an invented "Sept 20".
_DATE_RE = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2})(?!\d)"
    r"(?:,?\s+(\d{4}))?",
    re.IGNORECASE,
)
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
#: ~11 months in days: the horizon past which a bare month-day parse (year
#: defaulted to the current one) is read as next year's occurrence.
_ELEVEN_MONTHS_DAYS = 335

#: Price/ticket vocabulary in the SNIPPET marks a deal (spec §event metadata).
_DEAL_RE = re.compile(
    r"\$\s?\d|\btickets?\b|\bfares?\b|\bprices?\b|\bdiscount\w*\b|\bdeal\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class StoryResult:
    """Row-shaped outcome of generating one story; never an exception.

    Attributes:
        title: Story headline (the candidate's title).
        body: Generated second-person story body (``""`` when not produced).
        kind: Event metadata kind (``content``/``event``/``deal``).
        event_date: Explicitly parsed source date (``YYYY-MM-DD``), or None.
        location: Region when it appears in the source text, else None.
        status: Outcome (``complete``/``empty``/``failed``).
        error: Failure reason for ``empty``/``failed`` rows, else None.
    """
    title: str
    body: str
    kind: str
    event_date: str | None
    location: str | None
    status: str
    error: str | None


def _extract_metadata(
    title: str, snippet: str, *, now: datetime
) -> tuple[str | None, str]:
    """Pull ``(event_date, kind)`` out of explicit title+snippet text only.

    A month-name + day (+ optional year) match becomes an ISO date; a bare
    month-day defaults its year to ``now``'s and bumps to next year when that
    lands it more than ~11 months in the past (an event mentioned without a
    year is the next occurrence, not last year's). Kind: explicit date wins
    (``event``), else deal vocabulary in the snippet (``deal``), else
    ``content``. No match anywhere stores ``None`` — never a guess.

    Args:
        title: Candidate title text.
        snippet: Candidate snippet text.
        now: Injection clock the year default anchors to.

    Returns:
        ``(event_date, kind)`` with ``event_date`` None when no explicit
        date parses.
    """
    text = f"{title} {snippet}"
    event_date: str | None = None
    for match in _DATE_RE.finditer(text):
        month = _MONTHS[match.group(1).lower()]
        try:
            parsed = date(now.year, month, int(match.group(2)))
        except ValueError:  # e.g. "Nov 45" — not a date, keep looking
            continue
        explicit_year = match.group(3)
        if explicit_year is None:
            if (now.date() - parsed).days > _ELEVEN_MONTHS_DAYS:
                parsed = parsed.replace(year=parsed.year + 1)
        else:
            try:
                parsed = date(int(explicit_year), month, int(match.group(2)))
            except ValueError:
                continue
        event_date = parsed.isoformat()
        break
    if event_date is not None:
        return event_date, "event"
    if _DEAL_RE.search(snippet):
        return None, "deal"
    return None, "content"


def generate_story(
    chat: Callable[..., Any],
    *,
    candidate: Candidate,
    snapshot: dict,
) -> StoryResult:
    """Generate one story for one candidate; sync — the caller thread-offloads.

    The prompt embeds ONLY the candidate's title/snippet/url, the distilled
    topic names, and the region (spec §privacy boundaries); metadata comes
    from the source text, never the model's prose. An empty model body is
    ``status="empty"``; any exception is ``status="failed"`` with ``error``
    set — this function never raises.

    Args:
        chat: ``chat_api_call``-shaped callable (endpoint/key/model already
            bound by :func:`resolve_dreams_chat`).
        candidate: The discovery candidate to write about.
        snapshot: ``interest_profile.snapshot`` result.

    Returns:
        The row-shaped outcome for the story.
    """
    title = str(candidate.title or "")
    try:
        now = datetime.now().astimezone()
        event_date, kind = _extract_metadata(
            title, str(candidate.snippet or ""), now=now)
        region = str(snapshot.get("region") or "").strip()
        source_text = f"{title} {candidate.snippet}".lower()
        location = region if region and region.lower() in source_text else None
        # Bounded, JSON-encoded egress: title/snippet/url + topic names +
        # region, nothing else (spec §privacy boundaries).
        user = json.dumps(
            {
                "title": title,
                "snippet": str(candidate.snippet or ""),
                "url": candidate.url,
                "topics": [str(t.get("text", "")) for t in
                           snapshot.get("topics", [])],
                "region": region,
            },
            ensure_ascii=False,
        )
        response = chat(
            messages_payload=[{"role": "user", "content": user}],
            system_message=_STORY_SYSTEM_PROMPT,
            streaming=False,
            max_tokens=_STORY_MAX_TOKENS,
            temp=_STORY_TEMPERATURE,
        )
        body = extract_response_content(response).strip()
        if not body:
            return StoryResult(
                title=title, body="", kind=kind, event_date=event_date,
                location=location, status="empty",
                error="model returned an empty story body",
            )
        return StoryResult(
            title=title, body=body, kind=kind, event_date=event_date,
            location=location, status="complete", error=None,
        )
    except Exception as exc:  # noqa: BLE001 - every failure is a row, not a raise
        logger.warning("Dreams story generation failed for {}: {}",
                       candidate.url, type(exc).__name__)
        return StoryResult(
            title=title, body="", kind="content", event_date=None,
            location=None, status="failed", error=str(exc),
        )


def resolve_dreams_chat() -> Callable[..., Any]:
    """Resolve and pre-bind the Dreams chat seam; zero-arg closure.

    Resolution chain (copies ``briefing_service``'s persisted-defaults
    chain): ``[dreams] provider`` + ``[dreams] model`` when set, else the
    persisted ``chat_defaults`` provider plus its remembered model via
    ``resolve_remembered_provider_model`` (provider-scoped, no cross-provider
    borrowing). The endpoint maps through ``chat_dispatch_name`` — the one
    provider→``chat_api_call``-handler table the chat surfaces share — and
    the credential through ``get_provider_readiness`` (explicit config key,
    env, or the keyless-local case). The returned closure pre-binds
    ``api_endpoint``/``api_key``/``model`` so callers (query synthesis, story
    generation) pass only payload and sampling kwargs.

    Returns:
        ``chat(**kwargs) -> Any`` with the wiring pre-bound; carries
        ``provider``/``model`` attributes for collection-row stamping.

    Raises:
        RuntimeError: ``"Dreams provider/model unavailable"`` when neither
            the Dreams settings nor the persisted chat defaults resolve a
            usable provider/model, or the provider has no chat handler.
    """
    persisted = load_cli_config_and_ensure_existence(force_reload=True)
    provider = dreams_setting("provider")
    model = dreams_setting("model")
    if not provider:
        defaults = persisted.get("chat_defaults")
        provider = (defaults.get("provider")
                    if isinstance(defaults, Mapping) else None)
    try:
        provider_key = canonical_provider_key(provider)
        if not model:
            model = resolve_remembered_provider_model(persisted, provider_key)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Dreams provider/model unavailable") from exc
    endpoint = chat_dispatch_name(provider_key)
    if not provider_key or not model or not endpoint:
        raise RuntimeError("Dreams provider/model unavailable")
    api_key = get_provider_readiness(provider_key, persisted).api_key

    def dreams_chat(**kwargs: Any) -> Any:
        return chat_api_call(
            api_endpoint=endpoint, api_key=api_key, model=model, **kwargs
        )

    # Stamped on the collection row by the cycle (spec §scheduling): plain
    # function attributes, read via getattr, so test fakes without them stay
    # None rather than breaking the cycle.
    dreams_chat.provider = provider_key  # type: ignore[attr-defined]
    dreams_chat.model = model  # type: ignore[attr-defined]
    return dreams_chat
