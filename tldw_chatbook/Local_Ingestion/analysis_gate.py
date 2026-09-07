"""Credential gate shared by the local processors' analysis passes.

(task-3301 xhigh review round, F8) task-3301 originally relaxed the
pdf/ebook analysis gates from ``api_name and api_key`` to ``api_name``
alone so keyless local providers (ollama, llama.cpp, ...) could analyze.
That silently broke a long-standing contract for DIRECT processor callers
(``batch_ingest_files``, ``local_media_reading_service`` jobs, the server
ingest path): passing ``api_name`` WITHOUT a key used to be a guaranteed
silent skip, and after the relaxation it triggered real LLM calls with
whatever credential the provider function loaded from config -- spend the
caller never asked for.

This predicate restores the invariant without losing the keyless-provider
feature: a call is allowed when the caller supplied a credential OR when
the caller explicitly opted in to keyless dispatch. Only the Library
ingest seam sets that opt-in (``analysis_keyless_ok`` in the job options,
attached by ``app._ingest_job_options`` strictly after
``resolve_ingest_analysis_provider`` said the provider is keyless-READY).
"""

from __future__ import annotations

from typing import Optional


def analysis_credentials_ok(
    api_key: Optional[str], keyless_ok: bool = False
) -> bool:
    """Return whether an analysis LLM call may be made.

    Args:
        api_key: The caller-supplied credential, if any.
        keyless_ok: Explicit opt-in for keyless dispatch. Set only by the
            Library ingest seam after provider readiness confirmed the
            provider works without a credential. Direct callers that never
            pass it keep the historical contract: no key, no call.

    Returns:
        True when a credential is present or keyless dispatch was
        explicitly sanctioned; False otherwise (silent skip).
    """
    return bool(api_key) or bool(keyless_ok)
