from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PreferencePayload(FrozenModel):
    kind: Literal["preference"] = "preference"
    subject: str
    polarity: Literal["like", "dislike"]
    value: str


class FactPayload(FrozenModel):
    kind: Literal["fact"] = "fact"
    subject: str
    value: str


class GoalPayload(FrozenModel):
    kind: Literal["goal"] = "goal"
    subject: str
    outcome: str


class WorkingContextPayload(FrozenModel):
    kind: Literal["working_context"] = "working_context"
    subject: str
    value: str


class LegacyUnclassifiedPayload(FrozenModel):
    kind: Literal["legacy_unclassified"] = "legacy_unclassified"
    text: str


ProfilePayload = Annotated[
    Union[PreferencePayload, FactPayload, GoalPayload, WorkingContextPayload, LegacyUnclassifiedPayload],
    Field(discriminator="kind"),
]
