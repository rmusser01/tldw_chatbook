"""Launch-time helpers: provider preflight + chat factory. UI-side glue.

Everything here exists so ``evals_screen.py``'s skill-eval worker can stay
thin: resolving one API key per distinct provider (``make_skill_eval_chat``),
enumerating builtin tool names for the static analyzer's reserved-name
checks (``builtin_tool_names``), and reading the local skills store for the
simulation layer's decoy pool (``store_skill_names``). No Textual imports --
the one UI module this touches (``evals_screen``'s reply-text extractor) is
imported lazily, at call time, because ``evals_screen`` imports THIS module
at load time and a module-level import back would be circular.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Tuple

from ...Chat.Chat_Functions import chat_api_call
from ...Chat.provider_readiness import get_provider_readiness
from ...config import get_user_data_dir
from ...Evals.skill_eval.models import EvalTarget

_CHAT_REQUEST_TIMEOUT = 120.0
_CHAT_RETRIES = 2
_CHAT_RETRY_DELAY = 2.0


def _reply_text(response: Any) -> str:
    """The generated text out of one ``chat_api_call(streaming=False)`` reply.

    Delegates to ``evals_screen._extract_chat_reply_text`` -- the one place
    this app already defines the "OpenAI-shaped reply, contentless reply is
    ``''`` not an error, anything else raises" discipline -- rather than
    copying its semantics into a second drift-prone twin. Imported at CALL
    time, not module level: ``evals_screen`` imports this module at load
    time, so a top-level import back would be circular (and would fail --
    the extractor is defined after that import block).
    """
    from ..Screens.evals_screen import _extract_chat_reply_text

    return _extract_chat_reply_text(response)


def make_skill_eval_chat(
    app_config: Mapping, generator: EvalTarget, judge: EvalTarget
) -> Tuple[Callable, List[str]]:
    """Preflight both targets' providers and build the run's chat callable.

    One readiness check per DISTINCT provider (a run whose generator and
    judge share a provider asks for that key once), each failed check
    contributing its ``user_message`` to the returned problem list -- an
    empty list means the callable is safe to hand to ``SkillEvalRunner``,
    a non-empty one means the caller must stop before spending a single
    provider call (the worker marks the already-created run failed).

    The returned callable is keyword-only and SYNCHRONOUS (a plain ``def``,
    never a coroutine): ``SkillEvalRunner``/``judge``/``simulation`` dispatch
    every call through ``asyncio.to_thread`` themselves, exactly like the
    character-probe ``ChatCallable`` seam this mirrors.

    Args:
        app_config: The app's loaded settings, as ``get_provider_readiness``
            reads it.
        generator: The run's generator target.
        judge: The run's judge target.

    Returns:
        ``(chat, problems)`` where ``chat(*, messages, target, temperature,
        max_tokens, seed) -> str`` wraps one ``chat_api_call`` per
        invocation with the resolved per-provider key, explicit
        ``request_timeout=120.0``/``retries=2``/``delay=2.0`` (a skill-eval
        judge prompt carries a whole SKILL.md; the app-wide chat defaults
        are tuned for interactive turns, not batch evaluation), and
        OpenAI-shaped reply extraction.
    """
    keys: Dict[str, str] = {}
    problems: List[str] = []
    for provider in dict.fromkeys([generator.provider, judge.provider]):
        readiness = get_provider_readiness(provider, app_config)
        if not readiness.ready:
            problems.append(readiness.user_message)
            continue
        keys[provider] = readiness.api_key or ""

    def chat(
        *,
        messages,
        target: EvalTarget,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> str:
        response = chat_api_call(
            api_endpoint=target.provider,
            messages_payload=messages,
            api_key=keys.get(target.provider, ""),
            temp=temperature,
            model=target.model_id,
            streaming=False,
            max_tokens=max_tokens,
            seed=seed,
            request_timeout=_CHAT_REQUEST_TIMEOUT,
            request_retries=_CHAT_RETRIES,
            request_retry_delay=_CHAT_RETRY_DELAY,
        )
        return _reply_text(response)

    return chat, problems


def builtin_tool_names() -> FrozenSet[str]:
    """The builtin tool catalog's bare names, for the static analyzer's
    ``UNKNOWN_TOOLS``/reserved-name checks. Degrades to an empty set if the
    catalog cannot be built -- a broken catalog must not take down the
    static layer, only widen what it flags as unknown."""
    from ...Agents.tool_catalog import BuiltinToolProvider

    try:
        return frozenset(e.name for e in BuiltinToolProvider().list_catalog())
    except Exception:
        return frozenset()


async def store_skill_names(app_config: Mapping) -> Tuple[List[dict], FrozenSet[str]]:
    """``(skill summaries, name set)`` from the local skills store.

    The store directory is resolved exactly the way ``app.py``'s own local
    skills stack resolves it (``app.py:8635`` and ``_build_local_skills_
    stack``): ``default_local_skills_store_dir(get_user_data_dir())`` --
    the one in-repo precedent for this resolution, copied rather than
    reinvented. ``app_config`` is kept in the signature for interface parity
    with ``make_skill_eval_chat`` (both are launch-time helpers the worker
    calls back to back); the directory itself deliberately does NOT come
    from it, because no config key names it -- the secured user data dir
    does.

    Async, not sync: ``LocalSkillsService.list_skills`` is a coroutine (the
    shared Skills service protocol), and this helper's only caller is the
    async worker -- ``asyncio.run`` from inside a running loop is not an
    option and hand-cracking the coroutine is not either.

    The summaries feed the simulation layer's decoy pool (``select_decoys``
    filters the subject itself out); the name set feeds the static layer's
    reserved names (builtin ∪ store skill names), so a skill whose name
    shadows a real tool or sibling skill is caught as ``NAME_COLLISION``.

    Args:
        app_config: Unused today (see above); kept for interface parity.

    Returns:
        ``([skill summary dicts], frozenset(names))`` -- empty list and
        empty set when the store holds nothing, never an exception.
    """
    from ...Skills_Interop.local_skills_service import (
        LocalSkillsService,
        default_local_skills_store_dir,
    )

    service = LocalSkillsService(
        store_dir=default_local_skills_store_dir(get_user_data_dir())
    )
    listing = await service.list_skills(limit=200)
    skills = list(listing.get("skills", []))
    return skills, frozenset(
        str(s.get("name", "")) for s in skills if s.get("name")
    )
