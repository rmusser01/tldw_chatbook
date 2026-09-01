"""Post-commit launch helpers for Personal Context interviews."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger
from tldw_profile_core import InterviewAudience, InterviewPack, InterviewQuestion

from .interview_coordinator import ProfileInterviewCoordinator
from .interview_draft_repository import InterviewDraftRepository
from .interview_provider import (
    ConfiguredModelQuestionProvider,
    FixedQuestionProvider,
)
from .key_protector import KeyringProfileKeyProtector
from .paths import get_personal_context_db_path


@dataclass(frozen=True, slots=True)
class ProfileInterviewLaunchRequest:
    """A fully resolved interview target created after its owner commits."""

    kind: Literal["personal", "workspace"]
    scope_id: str
    local_workspace_id: str | None = None
    mode: Literal["fixed", "adaptive"] = "fixed"
    source: Literal["setup", "workspace", "settings"] | None = None


_QUESTIONS = {
    "personal": (
        ("identity.preferred_name", "What name should assistants use for you?"),
        ("preference.communication_style", "What response style do you prefer?"),
        ("preference.detail_level", "How much detail is usually useful to you?"),
        ("preference.formatting", "What answer format do you find easiest to use?"),
        ("preference.interests", "What subjects do you especially enjoy?"),
        ("dislike.response_habits", "What response habits should assistants avoid?"),
        (
            "preference.decision_support",
            "How should assistants support your decisions?",
        ),
        ("preference.learning_style", "What helps you learn unfamiliar material?"),
        (
            "preference.accessibility",
            "What accessibility preferences should assistants follow?",
        ),
        (
            "constraint.boundaries",
            "What personal boundaries should assistants remember?",
        ),
    ),
    "workspace": (
        ("goal.main_outcome", "What is the main outcome for this workspace?"),
        ("working_context.audience", "Who is this project intended to help?"),
        (
            "convention.collaboration",
            "What project convention should collaborators follow?",
        ),
        ("goal.success", "What result would make this project successful?"),
        (
            "working_context.constraints",
            "What constraint most affects this project?",
        ),
        ("convention.tools", "What tool choice should remain consistent here?"),
        ("working_context.current_state", "What is the project's current state?"),
        ("working_context.risks", "What risk deserves continued attention?"),
        (
            "goal.non_goals",
            "What outcome is explicitly outside this project's goals?",
        ),
    ),
}


def _pack(kind: Literal["personal", "workspace"]) -> InterviewPack:
    questions = _QUESTIONS[kind]
    return InterviewPack(
        pack_id=f"chatbook-{kind}-v1",
        pack_version=1,
        audience=InterviewAudience(kind),
        coverage_version=1,
        coverage_topics=tuple(dict.fromkeys(topic for topic, _text in questions)),
        questions=tuple(
            InterviewQuestion(
                question_id=f"{kind}-{index}",
                topic=topic,
                text=text,
            )
            for index, (topic, text) in enumerate(questions, start=1)
        ),
    )


def _draft_repository() -> InterviewDraftRepository:
    """Use encrypted resumable drafts when secure custody is available."""

    try:
        protector = KeyringProfileKeyProtector()
        probe_ref = "personal-context-interview-draft-probe-v1"
        protector.load_or_create(probe_ref)
        protector.delete(probe_ref)
        path = get_personal_context_db_path().with_name(
            "tldw_chatbook_personal_context_interviews.db"
        )
        return InterviewDraftRepository(
            path,
            key_protector=protector,
        )
    except Exception:  # noqa: BLE001 - memory-only is the specified safe fallback.
        return InterviewDraftRepository.memory_only()


def build_profile_interview_screen(
    app: Any,
    request: ProfileInterviewLaunchRequest,
):
    """Build the Task 2 screen against the app's current service/config."""

    from tldw_chatbook.Chat.Chat_Functions import chat_api_call
    from tldw_chatbook.Chat.console_session_settings import (
        build_default_console_session_settings,
    )
    from tldw_chatbook.UI.Screens.profile_interview_screen import (
        ProfileInterviewScreen,
    )
    from tldw_chatbook.config import get_runtime_config_snapshot
    from tldw_chatbook.Chat.provider_readiness import provider_config_key

    service = app.get_personal_context_service(retry_locked=True)
    fixed = FixedQuestionProvider(_pack(request.kind))
    adaptive = None
    if request.mode == "adaptive":
        defaults = build_default_console_session_settings(
            get_runtime_config_snapshot().values
        )
        provider_id = provider_config_key(defaults.provider)
        model_id = str(defaults.model or "").strip()
        if not provider_id or not model_id:
            raise RuntimeError(
                "Adaptive profile interviews require a configured provider and model."
            )
        adaptive = ConfiguredModelQuestionProvider(
            provider_id=provider_id,
            provider_label=str(defaults.provider or provider_id),
            model_id=model_id,
            call=chat_api_call,
        )
    coordinator = ProfileInterviewCoordinator(
        service=service,
        drafts=_draft_repository(),
        fixed_provider=fixed,
        adaptive_provider=adaptive,
    )
    return ProfileInterviewScreen(
        coordinator,
        kind=request.kind,
        scope_id=request.scope_id,
        mode=request.mode,
    )


def launch_profile_interview_after_commit(
    app: Any,
    request: ProfileInterviewLaunchRequest,
    continuation: Callable[[], None],
) -> None:
    """Launch after commit and settle the caller continuation exactly once."""

    continued = False

    def continue_once(_result: Any = None) -> None:
        nonlocal continued
        if continued:
            return
        continued = True
        continuation()

    source = request.source
    if source is None:
        source = "workspace" if request.kind == "workspace" else "settings"
    failure_copy = {
        "setup": "Setup was saved, but the optional profile interview is unavailable.",
        "workspace": "Workspace created, but project context setup is unavailable.",
        "settings": "Profile interview is unavailable. Your existing profile was not changed.",
    }[source]

    try:
        builder = getattr(app, "build_personal_context_interview_screen")
        screen = builder(request)
        app.push_screen(screen, continue_once)
    except Exception:  # noqa: BLE001 - optional interview cannot block its owner.
        logger.opt(exception=True).warning("Personal Context interview launch failed")
        notify = getattr(app, "notify", None)
        if callable(notify):
            try:
                notify(failure_copy, severity="warning")
            except Exception:  # noqa: BLE001 - notification is best-effort too.
                logger.opt(exception=True).warning(
                    "Personal Context interview failure notification failed"
                )
        continue_once()


def launch_workspace_profile_interview_after_commit(
    app: Any,
    *,
    workspace_id: str,
    workspace_label: str,
    continuation: Callable[[], None],
) -> None:
    """Resolve the committed workspace's profile scope, then launch."""

    try:
        request = app.prepare_personal_context_interview_request(
            kind="workspace",
            mode="fixed",
            local_workspace_id=workspace_id,
            workspace_label=workspace_label,
            source="workspace",
        )
    except Exception:  # noqa: BLE001 - workspace already exists and must survive.
        logger.opt(exception=True).warning("Workspace profile scope preparation failed")
        notify = getattr(app, "notify", None)
        if callable(notify):
            try:
                notify(
                    "Workspace created, but project context setup is unavailable.",
                    severity="warning",
                )
            except Exception:  # noqa: BLE001 - notification is best-effort too.
                logger.opt(exception=True).warning(
                    "Workspace profile failure notification failed"
                )
        continuation()
        return
    launch_profile_interview_after_commit(app, request, continuation)


__all__ = [
    "ProfileInterviewLaunchRequest",
    "build_profile_interview_screen",
    "launch_profile_interview_after_commit",
    "launch_workspace_profile_interview_after_commit",
]
