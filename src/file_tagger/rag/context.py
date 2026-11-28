from collections.abc import Sequence
import re
import unicodedata

from pydantic import Field, field_validator

from ..domain.contracts import DomainRecord


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_HTML_COMMENT = re.compile(r"<!--.*?-->", flags=re.DOTALL)
_LIST_PREFIX = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_INLINE_CODE = re.compile(r"`([^`]+)`")
_WIKILINK = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_LEVEL = re.compile(r"\blevel\s*[:=\-]?\s*(\d+)\b", flags=re.IGNORECASE)
_PARTY_SIZE = re.compile(
    r"\b(?:party size|players?|characters?)\s*[:=\-]?\s*(\d+)\b",
    flags=re.IGNORECASE,
)


class MarkdownSection(DomainRecord):
    name: str = Field(min_length=1)
    text: str


class NormalizedMarkdownContext(DomainRecord):
    normalized_text: str
    sections: tuple[MarkdownSection, ...] = ()
    entities: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()

    @field_validator("entities", "constraints", mode="before")
    @classmethod
    def normalize_terms(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        return _unique_terms(value)

    @property
    def section_names(self) -> tuple[str, ...]:
        return tuple(section.name for section in self.sections)


def normalize_markdown_context(markdown_context: str) -> NormalizedMarkdownContext:
    if not isinstance(markdown_context, str):
        raise ValueError("Markdown context must be a string.")

    normalized = unicodedata.normalize("NFKC", markdown_context)
    normalized = _HTML_COMMENT.sub(_comment_newlines, normalized)
    sections: list[MarkdownSection] = []
    current_name: str | None = None
    current_lines: list[str] = []
    normalized_lines: list[str] = []
    entities: list[str] = []
    for line in normalized.splitlines():
        heading = _HEADING.match(line)
        if heading is not None:
            if current_name is not None:
                sections.append(
                    MarkdownSection(
                        name=current_name,
                        text="\n".join(current_lines).strip(),
                    )
                )
            current_name = _normalize_text(heading.group(2))
            current_lines = []
            entities.append(current_name)
            if current_name:
                normalized_lines.append(current_name)
            continue

        stripped = _normalize_text(_LIST_PREFIX.sub("", line))
        if not stripped:
            if normalized_lines and normalized_lines[-1] != "":
                normalized_lines.append("")
            continue
        normalized_lines.append(stripped)
        if current_name is not None:
            current_lines.append(stripped)
        entities.extend(_extract_explicit_entities(line))

    if current_name is not None:
        sections.append(
            MarkdownSection(
                name=current_name,
                text="\n".join(current_lines).strip(),
            )
        )

    normalized_text = "\n".join(normalized_lines).strip()
    constraints = _extract_constraints(normalized_text)
    entities.extend(_display_constraint(constraint) for constraint in constraints)
    return NormalizedMarkdownContext(
        normalized_text=normalized_text,
        sections=tuple(sections),
        entities=tuple(entities),
        constraints=constraints,
    )


def _normalize_text(value: str) -> str:
    normalized = re.sub(
        r"!\[([^\]]*)\]\([^)]*\)",
        lambda match: match.group(1),
        value,
    )
    normalized = re.sub(
        r"\[([^\]]+)\]\([^)]*\)",
        lambda match: match.group(1),
        normalized,
    )
    normalized = _WIKILINK.sub(
        lambda match: match.group(2) or match.group(1),
        normalized,
    )
    normalized = _INLINE_CODE.sub(
        lambda match: match.group(1),
        normalized,
    )
    normalized = _BOLD.sub(
        lambda match: match.group(1) or match.group(2),
        normalized,
    )
    normalized = re.sub(r"(?<!\w)[*_~](.+?)[*_~](?!\w)", r"\1", normalized)
    normalized = re.sub(r"<[^>]+>", "", normalized)
    return " ".join(normalized.split())


def _comment_newlines(match: re.Match[str]) -> str:
    return "\n" * match.group(0).count("\n")


def _extract_explicit_entities(line: str) -> tuple[str, ...]:
    entities: list[str] = []
    for match in _WIKILINK.finditer(line):
        entities.append(match.group(2) or match.group(1))
    for match in _INLINE_CODE.finditer(line):
        entities.append(match.group(1))
    for match in _BOLD.finditer(line):
        value = match.group(1) or match.group(2)
        if ":" not in value:
            entities.append(value)
    return tuple(_unique_terms(entities))


def _extract_constraints(text: str) -> tuple[str, ...]:
    constraints = [
        *(f"level {match.group(1)}" for match in _LEVEL.finditer(text)),
        *(f"party size {match.group(1)}" for match in _PARTY_SIZE.finditer(text)),
    ]
    return _unique_terms(constraints)


def _display_constraint(constraint: str) -> str:
    prefix, separator, suffix = constraint.partition(" ")
    if not separator:
        return constraint.capitalize()
    return f"{prefix.capitalize()} {suffix}"


def _unique_terms(values: Sequence[object]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = " ".join(unicodedata.normalize("NFKC", value).split())
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return tuple(result)
