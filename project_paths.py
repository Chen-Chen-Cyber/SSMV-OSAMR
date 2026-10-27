"""Resolve data paths without embedding an author's local directory names."""
import os
from pathlib import Path


def configured_path(variable, default):
    """Resolve an environment override relative to RADIO_SSL_WORKDIR or cwd."""
    candidate = Path(os.environ.get(variable, default)).expanduser()
    if not candidate.is_absolute():
        candidate = Path(os.environ.get("RADIO_SSL_WORKDIR", os.getcwd())) / candidate
    return str(candidate.resolve())
