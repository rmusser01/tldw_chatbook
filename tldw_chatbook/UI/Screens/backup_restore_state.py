"""Pure evidence labels for the canonical backup and restore view."""


def result_label(
    *,
    archive_verified: bool,
    restoration_validated: bool,
    opened: bool,
    needs_setup: bool,
) -> str:
    """Keep archive integrity distinct from restored data and successful opening."""
    if needs_setup:
        return "Needs setup"
    if opened:
        return "Opened successfully"
    if restoration_validated:
        return "Restoration validated"
    return "Archive verified" if archive_verified else "Not verified"
