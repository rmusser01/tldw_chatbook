"""Guardian rules engine: compiled-rule cache and match/except evaluation.

Compilation (regex + except patterns) is keyed on the store's
``rules_version`` counter -- bumped by every ``upsert_rule``/``delete_rule``
-- with a 60s TTL that bounds the life of a compiled set even without
writes (the server's per-run 30-60s cache semantics, ported). The warm
send path hits the cache and never recompiles.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from loguru import logger

#: Matched spans are transient (live notices only, never persisted) and
#: capped to keep a notice a notice (spec §Notices).
MAX_SPAN_CHARS: int = 80

#: Compiled-set TTL in seconds (spec §Rules engine: 60s).
DEFAULT_CACHE_TTL_SECONDS: float = 60.0


@dataclass(frozen=True, slots=True)
class GuardianMatch:
    """One matched rule against one draft.

    ``span_text`` is the transient matched excerpt (capped at
    :data:`MAX_SPAN_CHARS`) shown in the live notice only -- never stored.
    """

    rule_row: dict
    span_text: str
    is_crisis: bool


class CompiledRules:
    """An immutable compiled rule set; ``match`` is pure and thread-safe."""

    __slots__ = ("_compiled",)

    def __init__(
        self,
        compiled: tuple[
            tuple[dict, re.Pattern[str], tuple[re.Pattern[str], ...]], ...
        ],
    ) -> None:
        self._compiled = compiled

    def match(self, draft: str) -> list[GuardianMatch]:
        """Return every rule that matches ``draft`` (one match per rule).

        Except patterns are CONTEXT vocabulary (research/treatment/
        clinical...): when any of a rule's except patterns matches
        anywhere in the draft, the rule does not fire at all -- the
        crisis-awareness seed must stay quiet on clinical discussion, not
        merely on spans that happen to contain the vocabulary inline.
        """
        matches: list[GuardianMatch] = []
        for rule_row, pattern, excepts in self._compiled:
            if any(exc.search(draft) for exc in excepts):
                continue
            found = pattern.search(draft)
            if found is None:
                continue
            span = found.group(0)
            matches.append(
                GuardianMatch(
                    rule_row=rule_row,
                    span_text=span[:MAX_SPAN_CHARS],
                    is_crisis=bool(rule_row.get("is_crisis")),
                )
            )
        return matches


def _parse_excepts(value: Any) -> tuple[re.Pattern[str], ...]:
    """Compile a rule's except patterns from a JSON string or list."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            value = []
    if not value:
        return ()
    compiled: list[re.Pattern[str]] = []
    for item in value:
        try:
            compiled.append(re.compile(str(item)))
        except re.error:
            logger.warning("Guardian rule except pattern skipped: {!r}", item)
    return tuple(compiled)


def compile_rules(rows: Iterable[dict]) -> CompiledRules:
    """Compile rule rows (regex pattern + except patterns) for matching.

    Args:
        rows: Rule dicts as returned by ``GuardianDB.list_rules``; rows
            with ``enabled`` falsy are excluded.

    Returns:
        A :class:`CompiledRules` ready for ``match``.
    """
    compiled: list[tuple[dict, re.Pattern[str], tuple[re.Pattern[str], ...]]] = []
    for row in rows:
        if not row.get("enabled", 1):
            continue
        try:
            pattern = re.compile(str(row["pattern"]))
        except re.error:
            logger.warning(
                "Guardian rule {!r} has an invalid pattern; disabled",
                row.get("name"),
            )
            continue
        compiled.append((row, pattern, _parse_excepts(row.get("except_patterns"))))
    return CompiledRules(tuple(compiled))


class RulesCache:
    """Version-keyed cache of one :class:`CompiledRules` with a TTL-bounded
    version revalidation.

    The ``rules_version`` counter is the cache key: any rule write bumps it
    and the very next ``get`` recompiles (immediate invalidation). The TTL
    bounds how long a version read may be skipped by the CALLER (the
    checker refreshes its snapshot at this cadence); expiry re-reads the
    version and, when it is unchanged, reuses the compiled set without
    recompiling. ``compile_count`` exists for deterministic warm-path
    assertions -- the latency budget is proven by spies, not timing.
    """

    def __init__(
        self,
        db: Any,
        *,
        ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        compile_fn: Callable[[Iterable[dict]], CompiledRules] = compile_rules,
    ) -> None:
        self._db = db
        self._ttl_seconds = ttl_seconds
        self._compile_fn = compile_fn
        self._version: int | None = None
        self._compiled: CompiledRules | None = None
        self._compiled_at: float = 0.0
        #: Number of compilations performed (spy hook for tests).
        self.compile_count: int = 0

    def get(self, *, now: float | None = None) -> CompiledRules:
        """Return the compiled rules, recompiling only on version change.

        The version is read on every call (one indexed SELECT); TTL expiry
        alone never recompiles -- it only means the version just got
        revalidated, and an unchanged version proves the compiled set is
        still current. Recompilation happens exactly when a rule write
        bumped the version.

        Args:
            now: Monotonic clock override (tests); defaults to
                ``time.monotonic()``.

        Returns:
            The current :class:`CompiledRules`.
        """
        moment = time.monotonic() if now is None else now
        version = self._db.rules_version()
        if self._compiled is not None and version == self._version:
            if (moment - self._compiled_at) < self._ttl_seconds:
                return self._compiled
            self._compiled_at = moment
            return self._compiled
        rows = self._db.list_rules(enabled_only=True)
        self._compiled = self._compile_fn(rows)
        self._version = version
        self._compiled_at = moment
        self.compile_count += 1
        return self._compiled
