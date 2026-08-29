from typing import Annotated, Literal, Union

from pydantic import AfterValidator, BaseModel, ConfigDict, Field


def reject_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("value must not be blank")
    return value


BoundedText = Annotated[
    str, Field(min_length=1, max_length=16_384), AfterValidator(reject_blank)
]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IdentityPayload(FrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["identity"] = "identity"
    subject: BoundedText
    value: BoundedText


class PreferencePayload(FrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["preference"] = "preference"
    subject: BoundedText
    polarity: Literal["like", "dislike"]
    value: BoundedText


class RelationshipPayload(FrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["relationship"] = "relationship"
    subject: BoundedText
    value: BoundedText


class CorrectionPayload(FrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["correction"] = "correction"
    subject: BoundedText
    value: BoundedText


class ConstraintPayload(FrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["constraint"] = "constraint"
    subject: BoundedText
    value: BoundedText


class GoalPayload(FrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["goal"] = "goal"
    subject: BoundedText
    outcome: BoundedText


class ConventionPayload(FrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["convention"] = "convention"
    subject: BoundedText
    value: BoundedText


class WorkingContextPayload(FrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["working_context"] = "working_context"
    subject: BoundedText
    value: BoundedText


class LegacyUnclassifiedPayload(FrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["legacy_unclassified"] = "legacy_unclassified"
    text: BoundedText


ProfilePayload = Annotated[
    Union[
        IdentityPayload,
        PreferencePayload,
        RelationshipPayload,
        CorrectionPayload,
        ConstraintPayload,
        GoalPayload,
        ConventionPayload,
        WorkingContextPayload,
        LegacyUnclassifiedPayload,
    ],
    Field(discriminator="kind"),
]
