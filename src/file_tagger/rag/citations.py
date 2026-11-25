from collections.abc import Mapping, Sequence
import json
import re
from typing import Any, TypeAlias

from pydantic import Field, ValidationError

from ..domain.contracts import DomainRecord
from ..domain.errors import UncitedAnswerError
from ..search.results import MatchedSection, SearchCitation, SearchResult

NOT_FOUND_ANSWER = "not found in indexed files"
MAX_ANSWER_RESPONSE_BYTES = 64 * 1024


class AnswerClaim(DomainRecord):
    text: str = Field(min_length=1)
    chunk_ids: tuple[str, ...] = Field(min_length=1)
    citations: tuple[SearchCitation, ...] = Field(min_length=1)

    @property
    def citation_ids(self) -> tuple[str, ...]:
        return tuple(citation.citation_id for citation in self.citations)


class CitedAnswer(DomainRecord):
    question: str = Field(min_length=1)
    text: str = Field(min_length=1)
    claims: tuple[AnswerClaim, ...] = ()
    citations: tuple[SearchCitation, ...] = ()
    evidence: tuple[SearchResult, ...] = ()

    @property
    def answer(self) -> str:
        return self.text

    def __str__(self) -> str:
        return self.text

    @property
    def sections(self) -> tuple[MatchedSection, ...]:
        return tuple(
            section
            for result in self.evidence
            for section in result.matched_sections
        )


RagAnswer: TypeAlias = CitedAnswer


class StructuredClaim(DomainRecord):
    text: str = Field(min_length=1)
    chunk_ids: tuple[str, ...] = Field(min_length=1)


class StructuredAnswer(DomainRecord):
    answer: str = Field(min_length=1)
    claims: tuple[StructuredClaim, ...]


def answer_schema() -> dict[str, Any]:
    return StructuredAnswer.model_json_schema()


def parse_cited_answer(
    question: str,
    response: str,
    evidence: Sequence[SearchResult],
) -> CitedAnswer:
    if not isinstance(response, str):
        raise UncitedAnswerError(
            "Structured answer generation returned non-text data."
        )
    if len(response.encode("utf-8")) > MAX_ANSWER_RESPONSE_BYTES:
        raise UncitedAnswerError(
            "Structured answer generation returned oversized data."
        )
    try:
        raw = json.loads(response)
    except (TypeError, json.JSONDecodeError) as exc:
        raise UncitedAnswerError(
            "The model did not return a citation-validated structured answer."
        ) from exc
    if not isinstance(raw, Mapping):
        raise UncitedAnswerError(
            "The model returned an invalid structured answer."
        )

    normalized = _normalize_answer_payload(raw)
    try:
        structured = StructuredAnswer.model_validate(normalized)
    except (TypeError, ValueError, ValidationError) as exc:
        raise UncitedAnswerError(
            "The model returned a malformed citation claim."
        ) from exc

    if structured.answer.strip() == NOT_FOUND_ANSWER and not structured.claims:
        return CitedAnswer(
            question=question,
            text=NOT_FOUND_ANSWER,
            evidence=tuple(evidence),
        )
    if not structured.claims:
        raise UncitedAnswerError(
            "The model answer contains no citations to retrieved evidence."
        )

    by_chunk, by_citation = _citation_indexes(evidence)
    known_markers = {*by_chunk, *by_citation}
    marker_text = "\n".join(
        (structured.answer, *(claim.text for claim in structured.claims))
    )
    unknown_markers = [
        marker
        for marker in re.findall(
            r"\[source:\s*([^\]]+?)\s*\]",
            marker_text,
            flags=re.IGNORECASE,
        )
        if marker not in known_markers
    ]
    if unknown_markers:
        raise UncitedAnswerError(
            "The model answer references unknown evidence citation "
            f"{unknown_markers[0]!r}."
        )
    claims: list[AnswerClaim] = []
    seen_citations: dict[str, SearchCitation] = {}
    for raw_claim in structured.claims:
        resolved: list[SearchCitation] = []
        chunk_ids: list[str] = []
        for reference in raw_claim.chunk_ids:
            match = by_chunk.get(reference)
            if match is None:
                match = by_citation.get(reference)
            if match is None:
                raise UncitedAnswerError(
                    f"The model answer references unknown evidence ID {reference!r}."
                )
            if match.chunk_id not in chunk_ids:
                chunk_ids.append(match.chunk_id)
                resolved.append(match)
                seen_citations[match.citation_id] = match
        if not resolved:
            raise UncitedAnswerError(
                "The model answer contains a claim without retrieved evidence."
            )
        claims.append(
            AnswerClaim(
                text=raw_claim.text,
                chunk_ids=tuple(chunk_ids),
                citations=tuple(resolved),
            )
        )

    return CitedAnswer(
        question=question,
        text=structured.answer,
        claims=tuple(claims),
        citations=tuple(seen_citations.values()),
        evidence=tuple(evidence),
    )


def _normalize_answer_payload(raw: Mapping[Any, Any]) -> dict[str, Any]:
    answer = raw.get("answer", raw.get("text"))
    raw_claims = raw.get("claims")
    if not isinstance(raw_claims, list):
        if raw_claims is None:
            raw_claims = []
        else:
            raise UncitedAnswerError("Structured answer claims must be a list.")

    claims: list[dict[str, Any]] = []
    for item in raw_claims:
        if not isinstance(item, Mapping):
            raise UncitedAnswerError("Each structured answer claim must be an object.")
        text = item.get("text", item.get("claim"))
        references = item.get("chunk_ids")
        if references is None:
            references = item.get("citation_ids", item.get("citations"))
        if isinstance(references, str):
            references = [references]
        claims.append({"text": text, "chunk_ids": references})
    return {"answer": answer, "claims": claims}


def _citation_indexes(
    evidence: Sequence[SearchResult],
) -> tuple[dict[str, SearchCitation], dict[str, SearchCitation]]:
    by_chunk: dict[str, SearchCitation] = {}
    by_citation: dict[str, SearchCitation] = {}
    for result in evidence:
        for citation in result.citations:
            by_chunk[citation.chunk_id] = citation
            by_citation[citation.citation_id] = citation
    return by_chunk, by_citation


__all__ = (
    "AnswerClaim",
    "CitedAnswer",
    "NOT_FOUND_ANSWER",
    "RagAnswer",
    "StructuredAnswer",
    "StructuredClaim",
    "UncitedAnswerError",
    "answer_schema",
    "parse_cited_answer",
)
