"""STEP-safe names.

STEP consumers are picky about names: spaces, slashes and punctuation survive
the file but confuse feature trees and selection in CAD.  We sanitise once,
keep the mapping, and write it into ``report.json`` so nothing is lost.
"""

from __future__ import annotations

import re

__all__ = ["NameMap", "sanitize"]

_BAD = re.compile(r"[^0-9A-Za-z_]+")


def sanitize(name: str) -> str:
    """A conservative STEP-safe identifier."""
    clean = _BAD.sub("_", name).strip("_")
    if not clean:
        clean = "unnamed"
    if clean[0].isdigit():
        clean = "n" + clean
    return clean


class NameMap:
    """Sanitises names and guarantees uniqueness, remembering the original."""

    def __init__(self) -> None:
        self._taken: set[str] = set()
        self.mapping: dict[str, str] = {}

    def add(self, original: str) -> str:
        if original in self.mapping:
            return self.mapping[original]
        base = sanitize(original)
        candidate = base
        n = 2
        while candidate in self._taken:
            candidate = f"{base}_{n}"
            n += 1
        self._taken.add(candidate)
        self.mapping[original] = candidate
        return candidate

    def __getitem__(self, original: str) -> str:
        return self.mapping[original]
