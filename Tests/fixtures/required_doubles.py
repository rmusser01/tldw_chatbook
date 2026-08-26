"""Doubles a test must actually call, and a detector for the ones it didn't.

task-15270 / task-15210. `test_send_proceeds_when_auto_retrieve_fails` was
green for two months while never once calling the exploding retrieval backend
it existed to exercise: the toggle that would have fired auto-retrieval never
reached the code under test, so the test silently degraded into "an ordinary
send works". The assertion that would have caught it is one line --
``assert exploding_search.await_count == 1`` -- and nothing made its absence
visible.

This module makes that shape checkable rather than conventional:

* `must_be_called` / `exploding_double` register a double whose whole purpose
  is to be called. The root conftest's autouse `_required_doubles_are_called`
  fixture fails the test if one never was, so the guard cannot be forgotten
  once the double is built through here.
* `install_uncalled_double_audit` is an off-by-default DETECTOR for doubles
  nobody registered: it records every `monkeypatch.setattr` install of a mock
  that raises, and reports the ones no test ever called.

Why detection and enforcement are separate -- i.e. why the audit does not
simply fail the run: the same object shape carries two opposite intents. A
double with `side_effect=RuntimeError(...)` is sometimes the failure the test
claims to survive (must be called) and sometimes a tripwire installed so an
escape fails loudly -- this suite is full of the latter (the autouse network,
audio-device and local-server-probe guards all work that way), and a tripwire
that is never called is the PASSING outcome. Nothing on the object
distinguishes the two, so the audit reports candidates for a human to triage
and `must_be_called` records the decision.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable
from unittest.mock import AsyncMock, Mock

#: ``(double, reason)`` for every double registered during the current test.
_REQUIRED: list[tuple[Any, str]] = []

#: ``(target, uncalled_double)`` recorded by the audit, when it is installed.
_AUDITED: list[tuple[str, Any]] = []

#: Set this env var to a file path to turn the audit on; every uncalled
#: exploding double is appended there as ``<nodeid>\t<target>``.
AUDIT_ENV_VAR = "TLDW_AUDIT_UNCALLED_DOUBLES"


def must_be_called(double: Any, reason: str) -> Any:
    """Register ``double`` as one this test is required to call.

    Args:
        double: Any mock (sync or async). Returned unchanged, so this wraps an
            existing construction expression in place.
        reason: What the call proves, phrased for the failure message -- e.g.
            "the send must survive a retrieval failure".

    Returns:
        ``double``, so callers can register inline.
    """
    _REQUIRED.append((double, reason))
    return double


def exploding_double(
    error: BaseException | type[BaseException],
    *,
    reason: str,
    awaitable: bool = True,
) -> Any:
    """Build the "Y fails" double for an "X still works when Y fails" test.

    The double is registered by construction, so a test whose subject never
    reaches the failure it claims to survive fails instead of passing
    vacuously.

    Args:
        error: Exception instance or class the double raises when called.
        reason: What calling it proves (see `must_be_called`).
        awaitable: True (default) builds an `AsyncMock` for an awaited seam;
            False builds a plain `Mock`.

    Returns:
        The registered mock.
    """
    double = AsyncMock(side_effect=error) if awaitable else Mock(side_effect=error)
    return must_be_called(double, reason)


def reset_required_doubles() -> None:
    """Drop registrations left over from collection or a previous test."""
    _REQUIRED.clear()
    _AUDITED.clear()


def uncalled_required_doubles() -> list[str]:
    """Return one message per registered double that was never called."""
    messages = []
    for double, reason in _REQUIRED:
        if _call_count(double):
            continue
        messages.append(
            f"required double was never called: {reason} "
            f"(registered via Tests.fixtures.required_doubles; if the double "
            f"is genuinely optional do not register it)"
        )
    return messages


def _call_count(double: Any) -> int:
    """Total calls seen by ``double``, awaited or not.

    Read defensively: a plain `Mock` auto-creates `await_count` as a child
    mock rather than raising, so an unguarded `int()` on it explodes.
    """
    return _as_count(getattr(double, "call_count", 0)) or _as_count(
        getattr(double, "await_count", 0)
    )


def _as_count(value: Any) -> int:
    """Return ``value`` as a count, or 0 when it is not really one."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def is_exploding_double(value: Any) -> bool:
    """True when ``value`` is a mock whose only behaviour is to raise."""
    if not isinstance(value, Mock):
        return False
    side_effect = getattr(value, "side_effect", None)
    if isinstance(side_effect, BaseException):
        return True
    return isinstance(side_effect, type) and issubclass(side_effect, BaseException)


def install_uncalled_double_audit() -> bool:
    """Record every exploding double installed through ``monkeypatch.setattr``.

    Off unless `AUDIT_ENV_VAR` names a report file. Wraps the `MonkeyPatch`
    class once per session; the per-test bookkeeping is drained by
    `audited_uncalled_doubles`.

    Returns:
        True when the audit was installed.
    """
    if not os.environ.get(AUDIT_ENV_VAR):
        return False
    from _pytest.monkeypatch import MonkeyPatch

    if getattr(MonkeyPatch.setattr, "_tldw_audit_wrapped", False):
        return True
    original = MonkeyPatch.setattr

    def audited_setattr(self, target, name=..., value=..., raising=True):
        candidate = name if value is ... else value
        if is_exploding_double(candidate):
            label = getattr(target, "__name__", str(target))
            _AUDITED.append((f"{label}.{name}" if value is not ... else label, candidate))
        if value is ...:
            return original(self, target, name, raising=raising)
        return original(self, target, name, value, raising=raising)

    audited_setattr._tldw_audit_wrapped = True
    MonkeyPatch.setattr = audited_setattr
    return True


def audited_uncalled_doubles() -> list[str]:
    """Return the audit's uncalled installs (targets), for reporting only."""
    return [target for target, double in _AUDITED if not _call_count(double)]


def write_audit_report(nodeid: str, targets: Iterable[str]) -> None:
    """Append one ``<nodeid>\\t<target>`` line per uncalled audited double."""
    path = os.environ.get(AUDIT_ENV_VAR)
    if not path:
        return
    lines = "".join(f"{nodeid}\t{target}\n" for target in targets)
    if not lines:
        return
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(lines)
