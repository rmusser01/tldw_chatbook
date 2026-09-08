"""Content-free correlation for Console settings failure boundaries."""

from uuid import UUID

from loguru import logger

from ...Utils.persistent_diagnostics import safe_metadata_token


def _opaque_uuid(value: object) -> str:
    if type(value) is not str or len(value) not in (32, 36):
        return "invalid"
    try:
        parsed = UUID(value)
    except ValueError:
        return "invalid"
    if parsed.version != 4 or value not in (str(parsed), parsed.hex):
        return "invalid"
    return value


def log_settings_failure(
    phase: str,
    exc: Exception,
    *,
    session_id: object,
    submission_id: object = None,
    generation: object = None,
) -> None:
    """Log fixed phases and validated correlation values, never exception text.

    Args:
        phase: Fixed settings operation phase supplied by the call site.
        exc: Failure whose type, but not message or traceback, is recorded.
        session_id: Canonical UUIDv4 or its compact hexadecimal form.
        submission_id: Optional UUIDv4 identifying the settings submission.
        generation: Optional nonnegative integer default-intent generation.
    """
    logger.bind(module="ChatScreen").opt(exception=None).error(
        "operation=console_settings phase={} failure={} session_id={} "
        "submission_id={} generation={}",
        safe_metadata_token(phase),
        safe_metadata_token(type(exc).__name__),
        _opaque_uuid(session_id),
        "none" if submission_id is None else _opaque_uuid(submission_id),
        "none"
        if generation is None
        else (generation if type(generation) is int and generation >= 0 else "invalid"),
    )
