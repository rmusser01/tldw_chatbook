"""
Shared regex safety helpers for boundary and classifier patterns.

Provides lightweight checks to reject obviously dangerous constructs
and normalize flags consistently across modules.
"""

from __future__ import annotations

import re

try:  # pragma: no cover
    from loguru import logger as _logger
except ImportError:  # pragma: no cover
    class _N:
        def warning(self, *a, **k):
            pass
    _logger = _N()
import time

_REGEX_SAFETY_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    re.error,
)

# Pre-compiled heuristics to catch catastrophic backtracking risks
_DANGEROUS_CHECKS = [
    re.compile(r"\([^)]*[+*]\)[+*?]"),        # (a+)+, (a*)*, (a+)?
    re.compile(r"\([^)]*[+*]\){"),            # (a+){n,m}
    re.compile(r"\(\w+\+\)\+"),             # (word+)+
    re.compile(r"\(\w+\*\)\*"),             # (word*)*
    re.compile(r"\(\w+\?\)\?"),             # (word?)?
]


def check_pattern(pattern: str, *, max_len: int = 256) -> str | None:
    """Return an error message if pattern appears dangerous/invalid, else None.

    - Enforces a maximum length.
    - Applies nested-quantifier heuristics to reduce ReDoS risk.
    - Ensures the pattern compiles under Python's re.
    """
    try:
        pat = str(pattern or "")
    except (TypeError, ValueError):
        return "Pattern must be a string"
    if not pat:
        return None
    if len(pat) > max_len:
        return f"Pattern too long (max {max_len})"
    try:
        for chk in _DANGEROUS_CHECKS:
            if chk.search(pat):
                return "Pattern contains potentially dangerous regex constructs (nested quantifiers/alternations)"
    except re.error:
        # Non-fatal, continue to compile test
        pass
    try:
        re.compile(pat)
    except re.error as e:
        return f"Invalid regex: {e}"
    return None


def compile_flags(flags_str: str, *, allowed: set[str] | None = None, max_len: int = 10) -> tuple[int, str | None]:
    """Map a flags string (e.g., "im") to re flags, enforcing an allowlist.

    Returns (flags_value, error_message). error_message is None if ok.
    """
    allowed = allowed or {"i", "m"}
    try:
        s = str(flags_str or "").lower()
    except (TypeError, ValueError):
        return (0, "Flags must be a string")
    if len(s) > max_len:
        return (0, f"Flags too long (max {max_len})")
    flags = 0
    for f in s:
        if f not in allowed:
            return (0, "Only 'i' and 'm' flags are allowed")
        if f == 'i':
            flags |= re.IGNORECASE
        elif f == 'm':
            flags |= re.MULTILINE
    return (flags, None)


def warn_ambiguity(pattern: str) -> str | None:
    """Return a warning string for patterns that are valid but ambiguous.

    Heuristics:
    - Pattern not anchored with ^ and contains a wide wildcard like ".*" or ".+".
    """
    try:
        pat = str(pattern or "")
    except (TypeError, ValueError):
        return None
    if not pat:
        return None
    # crude detection of ".*" or ".+" outside of character classes
    if "^" not in pat and (".*" in pat or ".+" in pat):
        return "Unanchored pattern with wide wildcard may overmatch"
    return None


# Optional RE2 support for safer regex execution
_re2 = None
try:  # pragma: no cover - optional dependency
    import re2 as _re2  # type: ignore
except ImportError:  # pragma: no cover - absence is fine
    _re2 = None


def safe_search(compiled_pat: re.Pattern, text: str, *, timeout_env: str = "CHUNKING_REGEX_TIMEOUT") -> bool:
    """Perform a safe regex search with optional timeout and RE2 fallback.

    - If python-re is used, we cannot enforce hard timeouts; this function measures elapsed time
      and bails early across many calls, but a single pathological call can still block.
    - If re2 is available and the pattern can be compiled there, we prefer it.
    - timeout is read from env var (float seconds). Values <= 0 disable the guard.
    """
    # Try RE2 when available by recompiling pattern string
    t0 = time.perf_counter()
    timeout_s = 0.0
    # Read timeout from config.txt [Chunking] regex_timeout_seconds; do not rely on env
    try:
        from tldw_chatbook.Chunking._shims.config import load_comprehensive_config
        _cp = load_comprehensive_config()
        if hasattr(_cp, 'has_section') and _cp.has_section('Chunking'):
            _val = _cp.get('Chunking', 'regex_timeout_seconds', fallback='0')
            try:
                timeout_s = float(str(_val) or 0.0)
            except (TypeError, ValueError):
                timeout_s = 0.0
    except _REGEX_SAFETY_NONCRITICAL_EXCEPTIONS:
        timeout_s = 0.0
    # Fast path: optional RE2 search (only when flags are zero to preserve semantics)
    if _re2 is not None and getattr(compiled_pat, 'flags', 0) == 0:
        try:
            rp = _re2.compile(compiled_pat.pattern)
            return rp.search(text) is not None
        except _REGEX_SAFETY_NONCRITICAL_EXCEPTIONS:
            # fall back to python-re
            pass
    # Python re fallback
    # Best-effort to detect overlong execution: measure post-call
    try:
        found = compiled_pat.search(text) is not None
        if timeout_s > 0 and (time.perf_counter() - t0) > timeout_s:
            # Consider this a timeout condition from caller's perspective
            _logger.warning("Regex search exceeded configured timeout; treating as no match")
            return False
        return found
    except _REGEX_SAFETY_NONCRITICAL_EXCEPTIONS:
        return False
