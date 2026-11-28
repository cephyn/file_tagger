from collections.abc import Mapping, Sequence
import json
import re
from typing import Any, TypeAlias
import unicodedata

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
    text: str = Field(
        min_length=1,
        description=(
            "The factual claim. Its text must equal evidence_quote after only "
            "Unicode NFKC, case, and whitespace normalization."
        ),
    )
    evidence_quote: str = Field(
        min_length=1,
        description=(
            "The complete full text of every chunk listed in chunk_ids; do not "
            "quote individual sentences or other fragments."
        ),
    )
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
        if not all(
            _quote_matches_full_evidence_chunk(
                raw_claim.evidence_quote,
                citation.text,
            )
            for citation in resolved
        ):
            raise UncitedAnswerError(
                "The model answer evidence quote is not the complete full cited chunk."
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


def _normalize_text(value: str, *, strip_markers: bool = True) -> str:
    if strip_markers:
        value = SOURCE_MARKER_PATTERN.sub("", value)
    normalized = unicodedata.normalize("NFKC", value.casefold())
    return " ".join(normalized.split())


def _claim_is_supported(claim: str, quote: str) -> bool:
    normalized_claim = _normalize_text(claim)
    normalized_quote = _normalize_text(quote)
    return bool(normalized_claim) and normalized_claim == normalized_quote


def _quote_matches_full_evidence_chunk(quote: str, evidence_text: str) -> bool:
    normalized_quote = _normalize_text(quote, strip_markers=False)
    normalized_evidence = _normalize_text(evidence_text, strip_markers=False)
    return bool(normalized_quote) and normalized_quote == normalized_evidence


def _answer_is_supported(
    answer: str,
    claims: Sequence[StructuredClaim],
) -> bool:
    if not claims:
        return False
    normalized_claims = tuple(
        _normalize_text(claim.text)
        for claim in claims
    )
    return (
        all(normalized_claims)
        and _normalize_text(answer) == " ".join(normalized_claims)
    )


def _validate_claim_markers(
    text: str,
    resolved: Sequence[SearchCitation],
    by_chunk: Mapping[str, SearchCitation],
    by_citation: Mapping[str, SearchCitation],
) -> None:
    _validate_marker_layout(text)
    _validate_marker_bindings(
        text,
        ((text, resolved),),
        by_chunk,
        by_citation,
        mismatch_message=(
            "A visible source marker does not match the claim chunk_ids."
        ),
    )


def _validate_answer_markers(
    answer: str,
    resolved_claims: Sequence[
        tuple[StructuredClaim, tuple[SearchCitation, ...]]
    ],
    by_chunk: Mapping[str, SearchCitation],
    by_citation: Mapping[str, SearchCitation],
) -> None:
    _validate_marker_layout(answer)
    _validate_marker_bindings(
        answer,
        tuple((claim.text, citations) for claim, citations in resolved_claims),
        by_chunk,
        by_citation,
        mismatch_message=(
            "A visible source marker does not match its structured claim."
        ),
    )


def _validate_marker_layout(text: str) -> None:
    for match in SOURCE_MARKER_PATTERN.finditer(text):
        whitespace_start = match.start()
        while whitespace_start > 0 and text[whitespace_start - 1] in " \t\r\n":
            whitespace_start -= 1
        if any(
            character in "\r\n"
            for character in text[whitespace_start : match.start()]
        ):
            raise UncitedAnswerError(
                "A visible source marker is not bound to a structured claim."
            )
        suffix = text[match.end() :]
        if suffix and not suffix[0].isspace():
            raise UncitedAnswerError(
                "A visible source marker is not bound to a structured claim."
            )
        normalized_suffix = unicodedata.normalize("NFKC", suffix.lstrip())
        if normalized_suffix.startswith(":"):
            raise UncitedAnswerError(
                "A visible source marker is not bound to a structured claim."
            )


def _validate_marker_bindings(
    text: str,
    resolved_claims: Sequence[
        tuple[str, Sequence[SearchCitation]]
    ],
    by_chunk: Mapping[str, SearchCitation],
    by_citation: Mapping[str, SearchCitation],
    *,
    mismatch_message: str,
) -> None:
    matches = tuple(SOURCE_MARKER_PATTERN.finditer(text))
    if not matches:
        return
    markers = _resolved_markers(text, by_chunk, by_citation)
    claim_prefixes: list[str] = []
    normalized_claims: list[str] = []
    for claim, _citations in resolved_claims:
        normalized_claims.append(_normalize_text(claim))
        claim_prefixes.append(" ".join(normalized_claims))
    previous_marker_end = 0
    previous_supported_claims: list[Sequence[SearchCitation]] = []
    for match, marker in zip(matches, markers, strict=True):
        prefix = _normalize_text(text[: match.start()])
        segment = _normalize_text(text[previous_marker_end : match.start()])
        supported_claims = (
            previous_supported_claims
            if not segment and previous_marker_end
            else [
                resolved_claims[index][1]
                for index, claim_prefix in enumerate(claim_prefixes)
                if (
                    prefix == claim_prefix
                    and segment == normalized_claims[index]
                )
            ]
        )
        if not supported_claims:
            raise UncitedAnswerError(
                "A visible source marker is not bound to a structured claim."
            )
        allowed_ids = {
            citation.citation_id
            for citations in supported_claims
            for citation in citations
        }
        if marker.citation_id not in allowed_ids:
            raise UncitedAnswerError(mismatch_message)
        previous_marker_end = match.end()
        previous_supported_claims = supported_claims


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
