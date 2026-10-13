"""Versioned prompts as files: `prompts/<name>.v<N>.md`.

Prompts are code: reviewed in PRs, pinned by version, and identified in traces by a
content hash so you can tell exactly which text produced a given answer.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from string import Template

_FILE_RE = re.compile(r"^(?P<name>[a-z_]+)\.v(?P<version>\d+)\.md$")


@dataclass(frozen=True, slots=True)
class Prompt:
    name: str
    version: int
    text: str
    sha: str

    @property
    def ref(self) -> str:
        return f"{self.name}@v{self.version}"

    def render(self, **variables: str) -> str:
        """Substitute `$var` placeholders. Missing variables raise (fail loudly)."""
        return Template(self.text).substitute(**variables)


class PromptNotFoundError(LookupError):
    pass


class PromptRegistry:
    def __init__(self, directory: Path) -> None:
        self._prompts: dict[tuple[str, int], Prompt] = {}
        for path in sorted(directory.glob("*.md")):
            match = _FILE_RE.match(path.name)
            if match is None:
                continue
            text = path.read_text(encoding="utf-8").strip()
            name, version = match["name"], int(match["version"])
            sha = hashlib.sha256(text.encode()).hexdigest()[:12]
            self._prompts[(name, version)] = Prompt(name, version, text, sha)

    def get(self, name: str, version: int | None = None) -> Prompt:
        if version is not None:
            try:
                return self._prompts[(name, version)]
            except KeyError:
                raise PromptNotFoundError(f"{name}@v{version}") from None
        versions = [v for (n, v) in self._prompts if n == name]
        if not versions:
            raise PromptNotFoundError(name)
        return self._prompts[(name, max(versions))]
