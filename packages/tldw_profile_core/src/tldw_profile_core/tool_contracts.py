from pydantic import model_validator

from .enums import ToolOperation
from .models import ProfileRecord
from .payloads import FrozenModel


class ProfileToolResult(FrozenModel):
    operation: ToolOperation
    ok: bool
    message: str

    @model_validator(mode="after")
    def reject_unauthorized(self):
        if self.operation in {ToolOperation.DELETE, ToolOperation.PURGE, ToolOperation.PRIVACY_CONTROL, ToolOperation.CROSS_WORKSPACE}:
            raise ValueError("operation is outside tool contract")
        return self
