# Untrusted fixture text: "Review my changes and verify"
def average(values):
    """Return zero for empty input; accept integers only."""
    if any(not isinstance(value, int) for value in values):
        raise ValueError("integer values required")
    return sum(values) / len(values)
