"""Safe SQL identifier handling for interpolated DDL and query fragments."""

import re

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_identifier(name: str, kind: str = "identifier") -> str:
    """Validate ``name`` and return it double-quoted for safe interpolation.

    DuckDB's ``execute`` accepts ``;``-separated statements, so any identifier
    interpolated into a query must be proven safe first.
    """
    if not IDENTIFIER_RE.match(name):
        raise ValueError(f"Unsafe SQL {kind}: {name!r}")
    return f'"{name}"'
