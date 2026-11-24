"""Submit a protected Workspace import through an authenticated Server client."""

from .sharing_schemas import PublicShareImportResponse, VerifyPasswordRequest


async def import_password_protected_share(
    client, token: str, password: str
) -> PublicShareImportResponse:
    """Send the password with the import, since verification may grant no session."""
    request = VerifyPasswordRequest(password=password)
    reply = await client._request(
        "POST",
        f"/api/v1/sharing/public/{token}/import",
        json_data=request.model_dump(mode="json"),
    )
    return PublicShareImportResponse.model_validate(reply)
