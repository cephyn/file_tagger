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
SOURCE_MARKER_PATTERN = re.compile(
    r"\[source:\s*([^\]]+?)\s*\]",
    flags=re.IGNORECASE,
)


class AnswerClaim(DomainRecord):
    text: str = Field(min_length=1)
    evidence_quote: str = Field(min_length=1)
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
    evidence_quote: str = Field(min_length=1)
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
    _resolved_markers(
        "\n".join(
            (structured.answer, *(claim.text for claim in structured.claims))
        ),
        by_chunk,
        by_citation,
    )
    claims: list[AnswerClaim] = []
    seen_citations: dict[str, SearchCitation] = {}
    resolved_claims: list[tuple[StructuredClaim, tuple[SearchCitation, ...]]] = []
    for raw_claim in structured.claims:
        resolved: list[SearchCitation] = []
        chunk_ids: list[str] = []
        seen_references: set[str] = set()
        for reference in raw_claim.chunk_ids:
            match = by_chunk.get(reference)
            if match is None:
                match = by_citation.get(reference)
            if match is None:
                raise UncitedAnswerError(
                    f"The model answer references unknown evidence ID {reference!r}."
                )
            if match.citation_id in seen_references:
                raise UncitedAnswerError(
                    "The model answer contains duplicate evidence chunk IDs "
                    f"for {match.chunk_id!r}."
                )
            seen_references.add(match.citation_id)
            if match.chunk_id not in chunk_ids:
                chunk_ids.append(match.chunk_id)
                resolved.append(match)
                seen_citations[match.citation_id] = match
        if not resolved:
            raise UncitedAnswerError(
                "The model answer contains a claim without retrieved evidence."
            )
        if not any(
            raw_claim.evidence_quote in citation.excerpt
            for citation in resolved
        ):
            raise UncitedAnswerError(
                "The model answer evidence quote is not an exact quoted span "
                "from its cited chunk."
            )
        if not _claim_is_supported(raw_claim.text, raw_claim.evidence_quote):
            raise UncitedAnswerError(
                "The model answer contains a claim not supported by its evidence quote."
            )
        _validate_claim_markers(raw_claim.text, tuple(resolved), by_chunk, by_citation)
        resolved_claims.append((raw_claim, tuple(resolved)))
        claims.append(
            AnswerClaim(
                text=raw_claim.text,
                evidence_quote=raw_claim.evidence_quote,
                chunk_ids=tuple(chunk_ids),
                citations=tuple(resolved),
            )
        )

    _validate_answer_markers(
        structured.answer,
        resolved_claims,
        by_chunk,
        by_citation,
    )
    if not _answer_is_supported(structured.answer, structured.claims):
        raise UncitedAnswerError(
            "The model answer contains text unsupported by its validated claims."
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
        evidence_quote = item.get("evidence_quote", item.get("quote"))
        if not isinstance(evidence_quote, str) or not evidence_quote:
            raise UncitedAnswerError(
                "Each structured answer claim requires a verifiable evidence quote."
            )
        claims.append(
            {
                "text": text,
                "evidence_quote": evidence_quote,
                "chunk_ids": references,
            }
        )
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


def _normalized_words(value: str) -> tuple[str, ...]:
    without_markers = SOURCE_MARKER_PATTERN.sub("", value)
    return tuple(
        re.findall(
            r"[^\W_]+(?:[-'][^\W_]+)?",
            without_markers.casefold(),
        )
    )


def _claim_is_supported(claim: str, quote: str) -> bool:
    raw_claim_words = _normalized_words(claim)
    raw_quote_words = _normalized_words(quote)
    relation_markers = {
        "according",
        "allege",
        "allegation",
        "alleged",
        "allegedly",
        "appears",
        "apparently",
        "assumed",
        "assumes",
        "believed",
        "believes",
        "believe",
        "barely",
        "couldn't",
        "no",
        "not",
        "never",
        "without",
        "cannot",
        "can't",
        "claimed",
        "claims",
        "can",
        "could",
        "doubted",
        "doubts",
        "evidently",
        "estimated",
        "estimates",
        "expected",
        "expects",
        "indicated",
        "indicates",
        "likely",
        "may",
        "maybe",
        "might",
        "mistakenly",
        "mustn't",
        "must",
        "neither",
        "none",
        "nor",
        "nothing",
        "nobody",
        "perhaps",
        "purported",
        "possibly",
        "possible",
        "impossible",
        "improbable",
        "presumably",
        "probable",
        "probably",
        "purportedly",
        "reputedly",
        "report",
        "reported",
        "reportedly",
        "reports",
        "say",
        "said",
        "says",
        "seem",
        "seemed",
        "seemingly",
        "seems",
        "shall",
        "should",
        "source",
        "sources",
        "speculated",
        "speculates",
        "supposedly",
        "suggested",
        "suggests",
        "suspected",
        "suspects",
        "thought",
        "unlikely",
        "unconfirmed",
        "uncertain",
        "unknown",
        "will",
        "wouldn't",
        "won't",
        "would",
    }
    claim_markers = tuple(word for word in raw_claim_words if word in relation_markers)
    quote_markers = tuple(word for word in raw_quote_words if word in relation_markers)
    if claim_markers != quote_markers:
        return False
    stop_words = {
        "a",
        "an",
        "and",
        "as",
        "at",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
    }
    claim_words = tuple(
        word for word in raw_claim_words if word not in stop_words
    )
    quote_words = tuple(
        word for word in raw_quote_words if word not in stop_words
    )
    if not claim_words or not quote_words:
        return False
    if len(quote_words) - len(claim_words) > 1:
        return False
    quote_index = 0
    for word in claim_words:
        while quote_index < len(quote_words) and quote_words[quote_index] != word:
            quote_index += 1
        if quote_index == len(quote_words):
            return False
        quote_index += 1
    return True


def _answer_is_supported(
    answer: str,
    claims: Sequence[StructuredClaim],
) -> bool:
    if not claims:
        return False
    answer_without_markers = re.sub(
        r"\[source:\s*[^\]]+?\s*\]",
        "",
        answer,
        flags=re.IGNORECASE,
    )
    answer_parts = [
        part
        for part in re.split(
            r"(?<=[.!?])\s+|\n+",
            answer_without_markers.strip(),
        )
        if part.strip()
    ]
    return bool(answer_parts) and all(
        any(_claim_is_supported(part, claim.text) for claim in claims)
        for part in answer_parts
    )


def _validate_claim_markers(
    text: str,
    resolved: Sequence[SearchCitation],
    by_chunk: Mapping[str, SearchCitation],
    by_citation: Mapping[str, SearchCitation],
) -> None:
    markers = _resolved_markers(text, by_chunk, by_citation)
    allowed_ids = {
        citation.citation_id
        for citation in resolved
    }
    if any(marker.citation_id not in allowed_ids for marker in markers):
        raise UncitedAnswerError(
            "A visible source marker does not match the claim chunk_ids."
        )


def _validate_answer_markers(
    answer: str,
    resolved_claims: Sequence[
        tuple[StructuredClaim, tuple[SearchCitation, ...]]
    ],
    by_chunk: Mapping[str, SearchCitation],
    by_citation: Mapping[str, SearchCitation],
) -> None:
    for part in _answer_parts_preserving_markers(answer):
        if not part.strip():
            continue
        markers = _resolved_markers(part, by_chunk, by_citation)
        if not markers:
            continue
        supported_claims = [
            citations
            for claim, citations in resolved_claims
            if _claim_is_supported(part, claim.text)
        ]
        if not supported_claims:
            raise UncitedAnswerError(
                "A visible source marker is not bound to a structured claim."
            )
        allowed_ids = {
            citation.citation_id
            for citations in supported_claims
            for citation in citations
        }
        if any(marker.citation_id not in allowed_ids for marker in markers):
            raise UncitedAnswerError(
                "A visible source marker does not match its structured claim."
            )


def _answer_parts_preserving_markers(answer: str) -> tuple[str, ...]:
    return tuple(
        part
        for part in re.split(
            r"(?<=[.!?])\s+(?!(?i:\[source:))|(?<=\])\s+(?=\S)|\n+",
            answer.strip(),
        )
        if part.strip()
    )


def _resolved_markers(
    text: str,
    by_chunk: Mapping[str, SearchCitation],
    by_citation: Mapping[str, SearchCitation],
) -> tuple[SearchCitation, ...]:
    markers = tuple(
        match.group(1).strip()
        for match in SOURCE_MARKER_PATTERN.finditer(text)
    )
    resolved: list[SearchCitation] = []
    seen: set[str] = set()
    for marker in markers:
        citation = by_chunk.get(marker) or by_citation.get(marker)
        if citation is None:
            raise UncitedAnswerError(
                "The model answer references unknown evidence citation "
                f"{marker!r}."
            )
        if citation.citation_id in seen:
            raise UncitedAnswerError(
                f"The model answer contains duplicate evidence marker {marker!r}."
            )
        seen.add(citation.citation_id)
        resolved.append(citation)
    return tuple(resolved)


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
