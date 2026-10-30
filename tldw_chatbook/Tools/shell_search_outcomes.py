"""Conservative empty-search recognition; never executes or rewrites commands."""

import shlex


def is_empty_search(
    command: str, exit_code: int | None, stderr: str, *, shell: str = ""
) -> bool:
    """Recognize a simple known search's normal no-match exit.

    Compound expressions, expansions and unparseable command text stay ordinary
    failures. This is outcome projection, not a shell execution parser.

    Args:
        command: The exact executed command text.
        exit_code: The settled process exit code.
        stderr: The executor-provided standard error preview.
        shell: The resolved shell executable, for cmd quoting rules.

    Returns:
        Whether the result is a recognized simple empty search.
    """
    if exit_code != 1 or stderr.strip():
        return False
    if any(marker in command for marker in ("\n", "\r", "`", "$(")):
        return False
    operators = ";&|<>()"
    windows_cmd = shell.replace("\\", "/").rsplit("/", 1)[-1].lower() in {
        "cmd",
        "cmd.exe",
    }
    if windows_cmd and "%" in command:
        return False
    lexer = shlex.shlex(command, posix=True, punctuation_chars=operators)
    if windows_cmd:
        lexer.quotes = '"'
        lexer.escape = ""
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError:
        return False
    if not tokens or any(
        token and all(c in operators for c in token) for token in tokens
    ):
        return False
    name = tokens[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
    return name.removesuffix(".exe") in {"grep", "rg", "findstr"}
