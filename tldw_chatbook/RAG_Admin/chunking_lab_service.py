"""Headless, conflict-safe saves for Chunking Lab authored recipes."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from ..Chunking.chunking_interop_library import (
    ChunkingInteropService,
    TemplateNotFoundError,
    TemplateUpdateConflictError,
)
from ..Chunking.lab_preflight import current_local_runtime, prepare_recipe


@dataclass(frozen=True, slots=True)
class ExpectedTemplate:
    """Stable identity captured when a catalog record becomes editable."""

    id: int
    uuid: str
    version: int


class TemplateSaveConflict(Exception):
    """The template changed or disappeared after the caller loaded it."""


def save_lab_template(
    service: ChunkingInteropService,
    *,
    body: dict,
    name: str,
    description: str,
    tags: list[str],
    expected: ExpectedTemplate | None = None,
) -> dict:
    """Validate and save one complete Lab recipe through the canonical catalog.

    The caller's body remains its editable draft. A detached copy is normalized
    to the catalog's documented tags column before capability admission. The
    prepared recipe proves faithful local executability but is deliberately not
    used as the save payload: authored semantics, including advanced metadata,
    remain the source of truth.

    Args:
        service: Canonical Media DB chunking-template service.
        body: Complete authored flat recipe body.
        name: Final record name.
        description: Final record description.
        tags: Final canonical record tags.
        expected: Existing identity/version for an atomic update, or ``None``
            to create a detached custom record.

    Returns:
        The refreshed canonical catalog record.

    Raises:
        TemplateSaveConflict: If an expected record changed or disappeared.
        PreviewUnsupportedError: If the body cannot execute faithfully in Lab.
        ChunkingTemplateError: For canonical catalog validation/write errors.
    """
    detached_body = service._parse_template_body(copy.deepcopy(body))
    service._pop_body_tags(detached_body)
    # Admission's normalized result is execution-only; persist the authored copy.
    prepare_recipe(detached_body, runtime=current_local_runtime())

    if expected is None:
        template_id = service.create_template(
            name=name,
            description=description,
            template_json=detached_body,
            tags=list(tags),
        )
    else:
        try:
            service.update_template(
                expected.id,
                name=name,
                description=description,
                template_json=detached_body,
                tags=list(tags),
                expected_uuid=expected.uuid,
                expected_version=expected.version,
            )
        except (TemplateNotFoundError, TemplateUpdateConflictError) as exc:
            raise TemplateSaveConflict(
                "The template changed or was deleted; reload it or save as new."
            ) from exc
        template_id = expected.id

    return service.get_template_by_id(template_id)
