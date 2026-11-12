"""Explicit component requests and atomic dependent-group publication."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from ...Character_Chat.expression_generation import compose_performance_component_prompt
from ...Character_Chat.portrait_compositing import (
    COMPONENT_ROLES,
    ComponentRole,
    assemble_components,
    compose_component,
)
from ...Character_Chat.portrait_performance import PerformanceConflict
from ...Character_Chat.portrait_performance_authoring import (
    PortraitCompositionCandidate,
)
from ...Chat.character_expression_playback import expression_motion_enabled
from ...Image_Generation.adapter_registry import get_registry
from ...Image_Generation.adapters.base import ImageGenResult
from ...Image_Generation.capabilities import resolve_backend_reference_image_capability
from ...Image_Generation.config import get_image_generation_config
from ...Image_Generation.worker import build_request, run_generation
from ...Utils.input_validation import escape_markup
from ...Widgets.confirmation_dialog import ConfirmationDialog
from ...Widgets.Console.console_auto_speak_consent import (
    sanitize_auto_speak_destination,
)

if TYPE_CHECKING:
    from ...Character_Chat.portrait_performance_authoring import (
        PortraitPerformanceDraft,
    )
    from .portrait_performance import PortraitPerformanceController, _Authority

AFFECTED = {
    "eyelids": {"blink_rest", "blink_small", "blink_open"},
    "small": {"small", "blink_small"},
    "open": {"open", "blink_open"},
}


def recompose(
    previous: PortraitCompositionCandidate,
    base: bytes,
    component: ComponentRole,
    raw: bytes,
) -> PortraitCompositionCandidate:
    """Feather only the raw replacement, retaining unaffected encoded roles exactly."""
    updated = dict(previous.components)
    updated[component] = compose_component(base, previous.recipe, component, raw)
    old = {frame.role: frame for frame in previous.frames}
    frames = tuple(
        old.get(frame.role, frame) if frame.role not in AFFECTED[component] else frame
        for frame in assemble_components(base, previous.recipe, updated)
    )
    return PortraitCompositionCandidate(
        previous.recipe,
        previous.regions_revision,
        tuple((role, updated[role]) for role in COMPONENT_ROLES if role in updated),
        frames,
    )


async def generate_components(
    controller: PortraitPerformanceController,
    authority: _Authority,
    draft: PortraitPerformanceDraft,
    component: ComponentRole | None,
    epoch: int,
) -> bool:
    """Admit three new requests or one explicit retry against one immutable recipe."""
    screen, view = controller._screen(), controller._view()
    settings, reference = controller._regions, controller._reference
    if reference is None:
        view.status(
            "Generation reference unavailable within image limits. Import ready-made frames instead."
        )
        return False
    if settings is None:
        view.status(
            "Edit and Save regions on an open-eyes, closed-mouth base before generating. Ready-made imports remain available."
        )
        return False
    if component is not None and component not in COMPONENT_ROLES:
        raise ValueError("Choose an eyelids, small-mouth or open-mouth component.")
    if component is not None and (
        draft.composition is None
        or draft.composition.recipe != settings.recipe
        or draft.composition.regions_revision != settings.revision
    ):
        view.status(
            "Generate a component batch first. Manual frames have no component lineage."
        )
        return False
    cfg = get_image_generation_config()
    backend = cfg.default_backend
    if (
        not backend
        or backend not in cfg.enabled_backends
        or not resolve_backend_reference_image_capability(backend, config=cfg).supported
    ):
        view.status(
            "Choose an enabled reference-image backend in Settings. Imports remain available."
        )
        return False
    warning = screen._image_generation_backend_warning()
    if warning:
        view.status(warning + " Imports remain available.")
        return False
    registry = get_registry()
    if registry.config is not cfg:
        view.status(
            "Image settings changed. Reload the configured image engine before generating."
        )
        return False
    model = getattr(cfg, f"{backend}_image_default_model", None)
    destination = sanitize_auto_speak_destination(
        getattr(cfg, f"{backend}_image_base_url", None)
    )
    roles = (component,) if component is not None else COMPONENT_ROLES
    event = threading.Event()
    # Validate every request before consent/reset; construction performs no transport.
    requests = tuple(
        build_request(
            backend=backend,
            model=model,
            prompt=compose_performance_component_prompt(
                component=role, expression_key=draft.source.expression_key
            ),
            width=reference.width,
            height=reference.height,
            image_format="png",
            reference_image=reference,
            cancel_event=event,
        )
        for role in roles
    )

    async def current() -> bool:
        if not await controller._source_current(authority, draft, epoch):
            if controller.draft is draft and not controller._current(authority):
                controller.discard()
            return False
        if (
            controller.draft is not draft
            or controller._regions is not settings
            or controller._reference is not reference
        ):
            return False
        if get_image_generation_config() is not cfg or get_registry() is not registry:
            view.status(
                "Image settings changed. Remaining requests cancelled; current draft retained."
            )
            return False

        def read_settings():
            saved = authority.db.get_character_portrait_regions(
                character_id=draft.source.character_id,
                expression_key=draft.source.expression_key,
            )
            controller._verify_source(
                authority,
                draft,
                controller._source_request,
                controller._local_authority,
            )
            return saved

        outcome = await controller._native(authority, read_settings)
        if outcome.error:
            if isinstance(outcome.error, ValueError):
                raise PerformanceConflict(
                    "portrait_regions_revision_changed"
                ) from outcome.error
            raise outcome.error
        if outcome.cancellation:
            raise outcome.cancellation
        if (
            not outcome.completed
            or epoch != controller._epoch
            or not controller._current(authority)
        ):
            return False
        if outcome.value != settings:
            raise PerformanceConflict("portrait_regions_revision_changed")
        return get_image_generation_config() is cfg and get_registry() is registry

    if not await current():
        return False
    confirmed = await screen.app.push_screen_wait(
        ConfirmationDialog(
            title="Generate portrait components",
            message=escape_markup(
                f"Send {len(roles)} image-edit requests to {backend} / {model or 'configured workflow/default'} ({destination}). Use saved character {authority.character_id}, expression {draft.source.expression_key}, with one identical prepared reference. {'A fresh batch clears all disposable frames and components before the first request. ' if component is None else 'Replace only the selected component and dependent eye/mouth pairs. '}Charges may apply. Uncertain requests are never retried automatically; review before Apply."
            ),
            confirm_label=f"Generate {len(roles)} {'components' if component is None else 'component'}",
            cancel_label="Cancel",
        )
    )
    if confirmed is not True or not await current():
        return False
    controller._generation_event = event
    failed = []
    try:
        if component is None:

            def fresh():
                return PortraitCompositionCandidate(
                    settings.recipe,
                    settings.revision,
                    (),
                    assemble_components(reference.content, settings.recipe, {}),
                )

            outcome = await controller._drain(
                fresh, task_name="portrait-performance-fresh"
            )
            if outcome.error:
                raise outcome.error
            if outcome.cancellation:
                raise outcome.cancellation
            if not outcome.completed or event.is_set() or not await current():
                return False
            draft.replace_composition(outcome.value)
            controller._retire_pair_review()
            view.set_busy(True)
        for role, request in zip(roles, requests, strict=True):
            if event.is_set() or not await current():
                return False
            previous = draft.composition
            view.status(
                f"Generating {role} via {backend}; {len(roles)} explicit requests, no automatic retry."
            )
            try:
                outcome = await controller._drain(
                    run_generation, request, task_name="portrait-performance-generate"
                )
                if outcome.cancellation:
                    raise outcome.cancellation
                if event.is_set() or not await current():
                    return False
                if outcome.error:
                    raise outcome.error
                result = outcome.value
                if (
                    not outcome.completed
                    or type(result) is not ImageGenResult
                    or type(result.content) is not bytes
                    or result.bytes_len != len(result.content)
                    or result.content_type
                    not in {"image/png", "image/jpeg", "image/webp"}
                ):
                    raise ValueError("Invalid generated still.")
                outcome = await controller._drain(
                    recompose,
                    previous,
                    reference.content,
                    role,
                    result.content,
                    task_name="portrait-performance-compose",
                )
                if outcome.error:
                    raise outcome.error
                if outcome.cancellation:
                    raise outcome.cancellation
                if (
                    event.is_set()
                    or not outcome.completed
                    or not await current()
                    or draft.composition is not previous
                ):
                    return False
                draft.replace_composition(outcome.value)
                controller._retire_pair_review()
            except PerformanceConflict:
                raise
            except Exception:  # noqa: BLE001 -- uncertain paid responses need explicit component retry
                failed.append(role)
        try:
            await view.show_draft(
                draft,
                lambda: epoch == controller._epoch and controller._current(authority),
                motion=expression_motion_enabled(
                    getattr(screen.app_instance, "app_config", {}) or {},
                    react=True,
                    manual=True,
                ),
            )
        except Exception:  # noqa: BLE001 -- preview failure must not imply another paid request
            view.status(
                "Component results retained; preview unavailable. Review before further requests."
            )
            return True
        view.status(
            f"Review partial results. Failed or uncertain: {', '.join(failed)}. Retry only the selected component explicitly; charges may already apply."
            if failed
            else "Paired poses ready for review. Nothing activates until Apply."
        )
        return True
    finally:
        event.set()
        if controller._generation_event is event:
            controller._generation_event = None
