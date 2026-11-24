"""Version-qualified exports from the canonical Server workspace owner."""

from urllib.parse import quote


async def export_accepted_workspace_markdown(
    client, workspace_id: str, artifact_id: str, version_id: str
) -> bytes:
    response = await client._request(
        "POST",
        f"/api/v1/workspaces/{quote(workspace_id, safe='')}/artifacts/{quote(artifact_id, safe='')}/exports",
        json_data={"format": "md", "artifact_version_id": version_id},
    )
    identity = {
        "workspace_id": workspace_id,
        "artifact_id": artifact_id,
        "artifact_version_id": version_id,
        "review_state": "accepted",
        "format": "md",
        "content_type": "text/markdown",
    }
    if not isinstance(response, dict) or any(
        response.get(key) != value for key, value in identity.items()
    ):
        raise ValueError("The Server returned a different artifact export")
    content = response.get("content")
    if not isinstance(content, str):
        raise ValueError("The Server did not return Markdown content")
    payload = content.encode("utf-8")
    if (
        type(response.get("bytes")) is not int
        or response["bytes"] != len(payload)
        or not 0 < len(payload) <= 512 * 1024
    ):
        raise ValueError("The Server returned an invalid export size")
    return payload
