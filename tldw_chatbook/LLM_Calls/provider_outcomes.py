"""Trusted adapter outcomes from a local gate BEFORE dispatch.

Only adapter code that has not sent this request may raise these errors. An
HTTP status, timeout, response text, or user recovery acknowledgment cannot
supply this proof. No existing remote rate response is classified this way.
The proof covers this one call only, never helpers or the accepted iteration.
"""


class PreEffectRateLimitError(RuntimeError):
    """A trusted adapter locally refused this request before dispatch."""


class PreEffectPermanentError(RuntimeError):
    """A trusted adapter locally rejected this request permanently before dispatch."""


PRE_EFFECT_ERRORS = (PreEffectRateLimitError, PreEffectPermanentError)
