"""Guardian check pipeline: the dispatch-seam pre-send checker (ADR-204).

Pipeline order per check (spec §Rules engine + pipeline):

1. **Gate** -- ``[guardian] enabled`` first, as a single cached config
   read; a disabled Guardian never touches the store (contract 1).
2. **Command skip** -- command-kind drafts (``/...``) are app actions,
   not conversational intent; slash dispatch diverts upstream of the seam
   anyway, this is the belt-and-suspenders pass.
3. **Match** -- compiled rules (version-keyed, TTL-bounded cache).
4. **Record hop** -- ONE ``asyncio.to_thread`` hop doing all store work
   for the matched rules: alert insert (digest only) + escalation bump +
   dedup counts. Recording precedes the act decision ON PURPOSE: a hit
   counts even when the act refuses the send (contract 8 -- Guardian
   measures typed intent), and a slow/errored store degrades to
   notice-without-record via fail-open, never a lost send.
5. **Act** -- ``notify`` surfaces an inline system row; ``redact``
   rewrites the draft in memory; ``block`` returns the notice for the
   seam's existing refusal plumbing (the checker never appends block
   notices itself -- the refusal path owns that row).

Fail-open, loud (contract 5): any checker exception returns ``allow`` and
surfaces ONE ``guardian_checker_error`` notification per error signature
per visit.

Crisis content (design-review P1 ruling, completed in Task 2): the crisis
resource block and disclaimer live in ``Guardian/crisis_resources.py``;
this module renders them into surfaced notices via
:func:`~tldw_chatbook.Guardian.crisis_resources.render_crisis_block`.

Escalated-block cooldown arming (ADR-204 contract 6): when a check's
outcome reaches ``block`` through the escalation LADDER (not a base-action
``block``), the record hop arms the rule's ``cooldown_minutes`` via
``set_cooldown`` in the same store transaction neighborhood -- the
escalation event is what the cooldown throttles, a configured block is the
user's own intent.

Visit end (Task 2): :meth:`GuardianChecker.finalize_visit` computes the
post-visit summary (per-topic counts, escalated rules, a fresh trend
analyzer run, and the retention sweep) and stores it ONLY when the visit
produced at least one non-silent alert or trend notice -- empty visits
mint nothing (spec §Post-visit summary).
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import time
import uuid
from collections.abc import Awaitable
from datetime import datetime, timedelta
from typing import Any, Callable

from loguru import logger

from . import settings as guardian_settings
from .crisis_resources import render_crisis_block
from .rules_engine import DEFAULT_CACHE_TTL_SECONDS, GuardianMatch, RulesCache

_EPOCH_ISO = "1970-01-01T00:00:00+00:00"
_ACTION_RANK = {"notify": 0, "redact": 1, "block": 2}
_GateCallable = Callable[[str, str], None]
_RowCallable = Callable[[str], Any]


class GuardianChecker:
    """Pre-send pattern checker injected at the prompt-queue dispatch seam.

    Visit identity is DERIVED (spec §Visit identity): the Console runtime
    has a visit LIFECYCLE (``attach_view`` opens one, ``on_unmount`` ->
    ``leave_console_runtime`` ends one) but no visit IDENTIFIER, so the
    checker mints a uuid at its first check after construction
    (:attr:`visit_id`) and holds it for that visit's alerts.
    :meth:`begin_visit` is the explicit reset -- the mount-side call
    lands with Task 2's ChatScreen visit wiring (``chat_screen.py`` is
    Task 2's file), which resets the visit id and the per-visit error
    dedup for every fresh Console mount (``attach_view`` bumps the
    runtime's monotonic attachment generation; the checker is rebuilt per
    ChatScreen construction, so a fresh screen implies a fresh visit).

    Args:
        db_getter: Returns the :class:`~tldw_chatbook.DB.Guardian_DB.GuardianDB`
            (or ``None`` when unavailable -- treated as no rules).
        session_id_getter: Returns the active Console session id.
        notify: ``notify(text, severity)`` surface for fail-open error
            notifications.
        append_system_row: ``append_system_row(text)`` inline transcript
            surface; an awaitable result is awaited.
        now: Returns the current UTC ISO timestamp (injectable clock).
    """

    #: The enabled gate is a single cached read refreshed on this TTL.
    _GATE_TTL_SECONDS: float = 60.0
    #: How often the compiled-rule snapshot is revalidated against the
    #: store (rule edits apply within this window -- the server's own
    #: 30-60s compilation-cache semantics).
    _CACHE_TTL_SECONDS: float = DEFAULT_CACHE_TTL_SECONDS

    def __init__(
        self,
        *,
        db_getter: Callable[[], Any],
        session_id_getter: Callable[[], str],
        notify: _GateCallable,
        append_system_row: _RowCallable,
        now: Callable[[], str],
    ) -> None:
        self._db_getter = db_getter
        self._session_id_getter = session_id_getter
        self._notify = notify
        self._append_system_row = append_system_row
        self._now = now
        self._visit_id: str | None = None
        self._error_signatures: set[str] = set()
        self._gate_value: bool | None = None
        self._gate_read_at: float = 0.0
        self._rules_cache: RulesCache | None = None
        self._rules_cache_db: Any = None
        self._compiled: Any = None
        self._compiled_at: float = 0.0

    # ------------------------------------------------------------------
    # Visit lifecycle
    # ------------------------------------------------------------------

    @property
    def visit_id(self) -> str | None:
        """This visit's uuid, or None until the first check mints one."""
        return self._visit_id

    def begin_visit(self) -> None:
        """Start a fresh visit: drop the held visit id and visit-scoped state.

        The next check mints a new uuid. Called on Console mount by the
        visit lifecycle wiring (Task 2); unit-pinned here.
        """
        self._visit_id = None
        self._error_signatures.clear()

    # ------------------------------------------------------------------
    # Check
    # ------------------------------------------------------------------

    async def check(self, draft: str) -> dict:
        """Run the pre-send pipeline over one draft; never raises.

        Args:
            draft: The user's typed draft exactly as it reached dispatch.

        Returns:
            ``{"action": "allow"|"notify"|"redact"|"block",
               "redacted_draft": str | None,
               "notice": dict | None,
               "recorded": bool}`` -- ``notice`` is ``{"text", "rule_name",
            "topic", "severity", "is_crisis", "span_text"}``. The seam
            routes ``block`` through the existing refusal plumbing with
            ``notice["text"]`` as the visible reason, and replaces the
            draft with ``redacted_draft`` on ``redact``.
        """
        try:
            return await self._check(draft)
        except Exception as exc:  # noqa: BLE001 - fail-open, loud (contract 5)
            self._fail_open(exc, draft)
            return {
                "action": "allow",
                "redacted_draft": None,
                "notice": None,
                "recorded": False,
            }

    async def _check(self, draft: str) -> dict:
        # 1. Gate first: a disabled Guardian touches nothing (contract 1).
        if not self._gate_enabled():
            return self._allow()
        # 2. Command-kind drafts are app actions, not conversational intent.
        if draft.lstrip().startswith("/"):
            return self._allow()

        db = self._db_getter()
        if db is None:
            # Unavailable store (disabled mid-flight, failed open): no
            # rules to check against; the send proceeds.
            return self._allow()

        if self._visit_id is None:
            self._visit_id = str(uuid.uuid4())
        session_id = str(self._session_id_getter() or "")
        visit_id = self._visit_id

        # 3. Match against the compiled snapshot; the store is consulted
        # only when the snapshot TTL expired (one thread hop then).
        moment = time.monotonic()
        if (
            self._compiled is None
            or (moment - self._compiled_at) >= self._CACHE_TTL_SECONDS
        ):
            if self._rules_cache is None or self._rules_cache_db is not db:
                self._rules_cache = RulesCache(db)
                self._rules_cache_db = db
            self._compiled = await asyncio.to_thread(self._rules_cache.get)
            self._compiled_at = moment
        matches = self._compiled.match(draft)
        if not matches:
            return self._allow()

        # 4. Record hop: alerts + escalation + dedup counts off the event
        # thread, in ONE hop for every matched rule.
        now_iso = str(self._now())
        digest = hashlib.sha256("".join(draft.split()).encode()).hexdigest()
        outcomes = await asyncio.to_thread(
            self._record_matches, db, matches, session_id, visit_id, digest, now_iso
        )
        return await self._act(draft, matches, outcomes)

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _gate_enabled(self) -> bool:
        moment = time.monotonic()
        if (
            self._gate_value is None
            or (moment - self._gate_read_at) >= self._GATE_TTL_SECONDS
        ):
            self._gate_value = bool(
                guardian_settings.guardian_setting("enabled", False)
            )
            self._gate_read_at = moment
        return self._gate_value

    def _record_matches(
        self,
        db: Any,
        matches: list[GuardianMatch],
        session_id: str,
        visit_id: str,
        digest: str,
        now_iso: str,
    ) -> list[dict]:
        """Record every match and return per-rule outcomes (thread hop body)."""
        outcomes: list[dict] = []
        for match in matches:
            rule = match.rule_row
            rule_id = int(rule["id"])
            db.insert_alert(
                rule_id=rule_id,
                session_id=session_id,
                visit_id=visit_id,
                topic=str(rule.get("topic") or ""),
                message_digest=digest,
                ts=now_iso,
            )
            state = db.bump_escalation(rule_id, now=now_iso)
            self._arm_cooldown_if_escalated_block(db, rule, rule_id, state, now_iso)
            outcomes.append(
                {
                    "rule_row": rule,
                    "span_text": match.span_text,
                    "current_action": str(state["current_action"]),
                    "notice_due": self._notice_due(
                        db, rule, rule_id, session_id, visit_id, now_iso
                    ),
                }
            )
        return outcomes

    @staticmethod
    def _arm_cooldown_if_escalated_block(
        db: Any, rule: dict, rule_id: int, state: dict, now_iso: str
    ) -> None:
        """Arm the rule's cooldown when THIS outcome is an escalated block.

        ADR-204 contract 6 (carried P4 ruling): ``current_action`` reached
        ``block`` through the ladder (``bump_escalation``'s return), not a
        base-action ``block`` -- a configured block is the user's own
        intent, not an escalation event, and arms nothing. Crisis rules can
        never reach here (the ladder caps them at ``notify``).
        """
        if str(state.get("current_action") or "") != "block":
            return
        if str(rule.get("action") or "notify") == "block":
            return
        cooldown_minutes = rule.get("cooldown_minutes")
        if cooldown_minutes is None:
            return
        until_iso = _shift_iso(now_iso, hours=float(cooldown_minutes) / 60.0)
        if until_iso:
            db.set_cooldown(rule_id, until_iso)

    @staticmethod
    def _notice_due(
        db: Any,
        rule: dict,
        rule_id: int,
        session_id: str,
        visit_id: str,
        now_iso: str,
    ) -> bool:
        """Dedup gates NOTICE SURFACING only (contract 8), never recording.

        Server naming mapped onto chatbook scopes: ``once_per_conversation``
        scopes to the persisted chat session id; ``once_per_session``
        scopes to the visit; ``once_per_day`` scopes to a 24h window.
        """
        frequency = str(rule.get("notification_frequency") or "every_message")
        if frequency == "every_message":
            return True
        if frequency == "once_per_conversation":
            return (
                db.count_recent_alerts(
                    rule_id, session_id=session_id, since_iso=_EPOCH_ISO
                )
                == 1
            )
        if frequency == "once_per_session":
            return (
                db.count_recent_alerts(
                    rule_id, visit_id=visit_id, since_iso=_EPOCH_ISO
                )
                == 1
            )
        if frequency == "once_per_day":
            cutoff = _shift_iso(now_iso, hours=-24) or _EPOCH_ISO
            return db.count_recent_alerts(rule_id, since_iso=cutoff) == 1
        return True

    async def _act(
        self, draft: str, matches: list[GuardianMatch], outcomes: list[dict]
    ) -> dict:
        """Apply the outcomes: surface notices, rewrite redactions, refuse."""
        top_action = "allow"
        block_notice: dict | None = None
        first_notice: dict | None = None
        redacted = draft
        surfaced: list[dict] = []

        for outcome in outcomes:
            rule = outcome["rule_row"]
            action = outcome["current_action"]
            name = str(rule.get("name") or rule.get("topic") or "rule")
            if _ACTION_RANK.get(action, 0) > _ACTION_RANK.get(top_action, -1):
                top_action = action

            if action == "redact":
                redacted = _substitute_spans(
                    str(rule.get("pattern") or ""), f"[redacted: {name}]", redacted
                )

            notice = self._build_notice(rule, outcome["span_text"], action=action)
            if action == "block":
                # A refusal reason is never deduped or silenced: the seam
                # appends it through the existing refusal plumbing.
                if block_notice is None:
                    block_notice = notice
                continue
            if outcome["notice_due"] and str(
                rule.get("display_mode") or "inline_banner"
            ) == "inline_banner":
                if first_notice is None:
                    first_notice = notice
                surfaced.append(notice)

        for notice in surfaced:
            result = self._append_system_row(notice["text"])
            if isinstance(result, Awaitable):
                await result

        if top_action == "block":
            return {
                "action": "block",
                "redacted_draft": None,
                "notice": block_notice,
                "recorded": True,
            }
        if top_action == "redact":
            return {
                "action": "redact",
                "redacted_draft": redacted,
                "notice": first_notice,
                "recorded": True,
            }
        if top_action == "notify":
            return {
                "action": "notify",
                "redacted_draft": None,
                "notice": first_notice,
                "recorded": True,
            }
        return self._allow()

    @staticmethod
    def _build_notice(rule: dict, span_text: str, *, action: str) -> dict:
        """Render one plain-text notice (markup-free; transient span only)."""
        name = str(rule.get("name") or rule.get("topic") or "rule")
        topic = str(rule.get("topic") or "")
        severity = str(rule.get("severity") or "info")
        is_crisis = bool(rule.get("is_crisis"))
        excerpt = f'"{span_text}"'
        if action == "block":
            text = (
                f"Guardian held this message · {name}: the rule '{name}'"
                + (f" (topic: {topic})" if topic else "")
                + " stopped this send. Edit the draft and try again, or"
                " adjust the rule in Settings."
            )
        elif action == "redact":
            text = (
                f"Guardian notice · {name} [{severity}]: part of your"
                f" message was redacted before sending (matched {excerpt})."
            )
        else:
            text = (
                f"Guardian notice · {name} [{severity}]: noticed"
                f" {excerpt} in your message."
            )
        if is_crisis:
            # Crisis-flagged rules surface resources + disclaimer wherever
            # they appear (contract 7); the block itself lives in
            # crisis_resources (P1 ruling, extracted in Task 2).
            text = f"{text}\n\n{render_crisis_block()}"
        return {
            "text": text,
            "rule_name": name,
            "topic": topic,
            "severity": severity,
            "is_crisis": is_crisis,
            "span_text": span_text,
        }

    # ------------------------------------------------------------------
    # Visit end (post-visit summary, spec §Post-visit summary)
    # ------------------------------------------------------------------

    def finalize_visit(self) -> dict | None:
        """Compute, store, and return this visit's summary; never raises.

        The visit-end half of the trend cadence (spec §Trend analyzer):
        runs the analyzer BEFORE the summary is written so its notices ride
        the payload, then runs the retention sweep (best-effort, logged
        count), and stores the summary ONLY when the visit produced at
        least one non-silent alert or trend notice -- empty visits (a
        30-second nav bounce) mint nothing. Degrade-never-raise: any
        failure is logged and returns ``None`` (analyzer failures degrade
        to summary-without-trends per spec §Failure modes).

        Runs on a worker thread (the Console's unmount path offloads it);
        everything here is blocking store work with no event-loop calls.

        Returns:
            ``{"per_topic_counts": {topic: hits}, "escalated_rules":
            [rule names], "trend_notices": [notice dicts]}`` when a
            summary was stored, else ``None``.
        """
        try:
            return self._finalize_visit()
        except Exception:  # noqa: BLE001 - a visit summary never raises
            logger.opt(exception=True).warning(
                "Guardian visit summary failed; the visit ends without one"
            )
            return None

    def _finalize_visit(self) -> dict | None:
        # Contract 1 first: a disabled Guardian never touches the store.
        if not self._gate_enabled():
            return None
        visit_id = self._visit_id
        if visit_id is None:
            # No check ever minted a visit (gate was off, or nothing was
            # typed): an empty nav bounce mints nothing.
            return None
        db = self._db_getter()
        if db is None:
            return None
        now_iso = str(self._now())
        session_id = str(self._session_id_getter() or "")

        # Trend cadence (a): synchronously at visit end, BEFORE the summary
        # is written. A failure degrades to summary-without-trends.
        try:
            from .trend_analyzer import analyze

            trend_notices = analyze(
                db, now=now_iso, visit_id=visit_id, session_id=session_id
            )
        except Exception:  # noqa: BLE001 - R9 degrade, never block the summary
            logger.opt(exception=True).warning(
                "Guardian trend analysis degraded to summary-without-trends"
            )
            trend_notices = []

        rows = db.visit_alert_rows(visit_id)
        per_topic_counts: dict[str, int] = {}
        escalated_rules: list[str] = []
        for row in rows:
            topic = str(row.get("topic") or "")
            hits = int(row.get("hits") or 0)
            # Silent-log topics are invisible by design: the summary is a
            # surfaced artifact, so it counts non-silent rules only.
            if str(row.get("display_mode") or "") != "silent_log":
                per_topic_counts[topic] = per_topic_counts.get(topic, 0) + hits
            current = str(row.get("current_action") or "")
            base = str(row.get("rule_action") or "notify")
            if (
                current
                and _ACTION_RANK.get(current, 0) > _ACTION_RANK.get(base, 0)
                and str(row.get("name") or "") not in escalated_rules
            ):
                escalated_rules.append(str(row.get("name") or ""))

        # Retention (contract 9): best-effort at each visit end, logged.
        try:
            retention_days = float(
                guardian_settings.guardian_setting("alert_retention_days", 180)
                or 180
            )
            cutoff = (
                _shift_iso(now_iso, hours=-retention_days * 24.0)
                or _EPOCH_ISO
            )
            pruned = db.prune_alerts(cutoff)
            if pruned:
                logger.info(
                    "Guardian retention pruned {} alert(s) at visit end",
                    pruned,
                )
        except Exception:  # noqa: BLE001 - retention never blocks the summary
            logger.opt(exception=True).debug("Guardian retention sweep failed")

        # Storage gate (carried P5 ruling, Task 2 review): store ONLY when
        # the visit has at least one surfaced topic count or trend notice.
        # ``escalated_rules`` alone is NOT a storing condition -- a
        # silent_log rule that escalates leaves its bookkeeping in the
        # escalation state table, and the visit itself stays invisible by
        # design (empty visits mint nothing, spec §Post-visit summary).
        if not per_topic_counts and not trend_notices:
            return None

        payload = {
            "per_topic_counts": per_topic_counts,
            "escalated_rules": escalated_rules,
            "trend_notices": [
                {
                    "kind": notice.kind,
                    "topic": notice.topic,
                    "label": notice.label,
                    "hits": notice.hits,
                    "message": notice.message,
                }
                for notice in trend_notices
            ],
        }
        db.insert_visit_summary(
            visit_id=visit_id, session_id=session_id, payload=payload
        )
        return payload

    # ------------------------------------------------------------------
    # Fail-open
    # ------------------------------------------------------------------

    def _fail_open(self, exc: Exception, draft: str) -> None:
        """Never eat a send; be loud once per signature per visit."""
        signature = f"{type(exc).__name__}: {exc}"[:200]
        logger.opt(exception=True).warning(
            "Guardian checker failed open (signature={}, draft_len={}): "
            "the send proceeds unchecked",
            signature,
            len(draft),
        )
        if signature in self._error_signatures:
            return
        self._error_signatures.add(signature)
        try:
            self._notify(
                "Guardian checker error — sends continue unchecked "
                f"({signature}).",
                "error",
            )
        except Exception:  # noqa: BLE001 - a broken surface must not raise
            logger.warning("Guardian error notification surface failed")

    @staticmethod
    def _allow() -> dict:
        return {
            "action": "allow",
            "redacted_draft": None,
            "notice": None,
            "recorded": False,
        }


def _substitute_spans(pattern: str, replacement: str, text: str) -> str:
    """Replace pattern spans with ``replacement`` (literal, no backrefs)."""
    if not pattern:
        return text
    try:
        return re.sub(pattern, lambda _m: replacement, text)
    except re.error:
        return text


def _shift_iso(iso: str, *, hours: float) -> str | None:
    """Return ``iso + hours`` preserving the input's suffix style.

    Timestamps compare lexicographically in SQL (``ts >= ?``), so a
    shifted cutoff must keep the source's ``Z``/``+00:00`` shape rather
    than ``isoformat()``'s ``+00:00`` default -- a mixed suffix would
    compare wrongly at exact boundaries.
    """
    try:
        moment = datetime.fromisoformat(iso)
    except ValueError:
        return None
    shifted = (moment + timedelta(hours=hours)).isoformat()
    if iso.endswith("Z") and shifted.endswith("+00:00"):
        return shifted[:-6] + "Z"
    return shifted
