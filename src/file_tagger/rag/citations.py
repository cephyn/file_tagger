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
_SENTENCE_TERMINATORS = frozenset(".!?。！？．")
_SENTENCE_CLOSERS = frozenset("\"'”’»›）〕］】」』)]}")
_COMMON_ABBREVIATIONS = frozenset(
    {
        "approx",
        "ave",
        "capt",
        "col",
        "corp",
        "dept",
        "dr",
        "e.g",
        "etc",
        "fig",
        "gen",
        "gov",
        "i.e",
        "inc",
        "jr",
        "ltd",
        "mr",
        "mrs",
        "ms",
        "no",
        "prof",
        "rep",
        "sen",
        "sgt",
        "sr",
        "st",
        "vs",
    }
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
            _quote_matches_complete_evidence_span(
                raw_claim.evidence_quote,
                citation.excerpt,
            )
            for citation in resolved
        ):
            raise UncitedAnswerError(
                "The model answer evidence quote is not a complete normalized "
                "evidence sentence or full-excerpt quoted span from its cited chunk."
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


def _normalized_words(value: str, *, strip_markers: bool = True) -> tuple[str, ...]:
    if strip_markers:
        value = SOURCE_MARKER_PATTERN.sub("", value)
    normalized = unicodedata.normalize("NFKC", value.casefold())
    words: list[str] = []
    current: list[str] = []
    for character in normalized:
        category = unicodedata.category(character)[0]
        if category in {"L", "M", "N", "S"}:
            current.append(character)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return tuple(words)


def _claim_is_supported(claim: str, quote: str) -> bool:
    claim_words = _normalized_words(claim)
    quote_words = _normalized_words(quote)
    return bool(claim_words) and claim_words == quote_words


def _quote_matches_complete_evidence_span(quote: str, excerpt: str) -> bool:
    normalized_quote = _normalized_words(quote, strip_markers=False)
    if not normalized_quote:
        return False
    return any(
        normalized_quote
        == _normalized_words(span, strip_markers=False)
        for span in _evidence_sentence_spans(excerpt)
    )


def _evidence_sentence_spans(excerpt: str) -> tuple[str, ...]:
    text = excerpt.strip()
    if not text:
        return ()

    spans: list[str] = []
    start = 0
    index = 0
    while index < len(text):
        terminator = text[index]
        if terminator not in _SENTENCE_TERMINATORS:
            index += 1
            continue

        terminator_end = index + 1
        while (
            terminator_end < len(text)
            and text[terminator_end] in _SENTENCE_TERMINATORS
        ):
            terminator_end += 1
        span_end = terminator_end
        while span_end < len(text) and text[span_end] in _SENTENCE_CLOSERS:
            span_end += 1

        if span_end == len(text):
            break

        next_start = span_end
        while next_start < len(text) and text[next_start].isspace():
            next_start += 1
        if next_start == span_end:
            if terminator == "." and _period_is_internal(text, index):
                index = terminator_end
                continue
            return (text,)
        if terminator == "." and _period_is_abbreviation(text, index):
            index = terminator_end
            continue
        if (
            terminator in {".", "。", "．"}
            and len(text[index:terminator_end]) >= 3
            and len(set(text[index:terminator_end])) == 1
        ):
            return (text,)
        if not _looks_like_sentence_start(text, next_start):
            return (text,)

        spans.append(text[start:span_end].strip())
        start = next_start
        index = next_start

    final_span = text[start:].strip()
    if final_span:
        if start and not _has_sentence_ending(final_span):
            return (text,)
        spans.append(final_span)
    return tuple(span for span in spans if span)


def _period_is_internal(text: str, index: int) -> bool:
    return (
        index > 0
        and index + 1 < len(text)
        and (text[index - 1].isalnum() or text[index - 1] == "_")
        and (text[index + 1].isalnum() or text[index + 1] == "_")
    )


def _period_is_abbreviation(text: str, index: int) -> bool:
    prefix = text[:index]
    match = re.search(r"([^\W\d_]+(?:\.[^\W\d_]+)*)$", prefix, flags=re.UNICODE)
    if match is None:
        return False
    abbreviation = match.group(1).casefold()
    return abbreviation in _COMMON_ABBREVIATIONS or (
        "." in abbreviation or len(abbreviation) == 1
    )


def _has_sentence_ending(value: str) -> bool:
    candidate = value.rstrip()
    while candidate and candidate[-1] in _SENTENCE_CLOSERS:
        candidate = candidate[:-1]
    if not candidate or candidate[-1] not in _SENTENCE_TERMINATORS:
        return False
    if candidate[-1] == "." and _period_is_abbreviation(
        candidate,
        len(candidate) - 1,
    ):
        return False
    return True


def _looks_like_sentence_start(text: str, index: int) -> bool:
    while index < len(text) and text[index] in "\"'“‘«‹（〔［【「『([{":
        index += 1
    return index < len(text) and (
        text[index].isupper() or text[index].isdigit()
    )


def _answer_is_supported(
    answer: str,
    claims: Sequence[StructuredClaim],
) -> bool:
    if not claims:
        return False
    if any(
        _claim_is_supported(answer, claim.text)
        for claim in claims
    ):
        return True
    answer_parts = _answer_parts_preserving_markers(answer)
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
    if any(
        _claim_is_supported(answer, claim.text)
        for claim, _citations in resolved_claims
    ):
        _validate_marker_bindings(
            answer,
            tuple((claim.text, citations) for claim, citations in resolved_claims),
            by_chunk,
            by_citation,
            mismatch_message=(
                "A visible source marker does not match its structured claim."
            ),
        )
        return
    for part in _answer_parts_preserving_markers(answer):
        _validate_marker_bindings(
            part,
            tuple((claim.text, citations) for claim, citations in resolved_claims),
            by_chunk,
            by_citation,
            mismatch_message=(
                "A visible source marker does not match its structured claim."
            ),
        )


def _answer_parts_preserving_markers(answer: str) -> tuple[str, ...]:
    text = answer.strip()
    parts: list[str] = []
    start = 0
    index = 0
    while index < len(text):
        if text[index] in "\r\n":
            part = text[start:index].strip()
            if part:
                parts.append(part)
            while index < len(text) and text[index] in "\r\n":
                index += 1
            start = index
            continue
        if text[index] not in ".!?":
            index += 1
            continue

        sentence_end = index + 1
        cursor = sentence_end
        while cursor < len(text) and text[cursor] in " \t":
            cursor += 1
        marker_end = cursor
        marker_found = False
        while match := SOURCE_MARKER_PATTERN.match(text, marker_end):
            marker_found = True
            marker_end = match.end()
            while marker_end < len(text) and text[marker_end] in " \t":
                marker_end += 1
        if marker_found:
            part = text[start:marker_end].strip()
            if part:
                parts.append(part)
            start = marker_end
            index = marker_end
            continue
        if (
            cursor == len(text)
            or cursor > sentence_end
            or text[cursor] in "\r\n"
        ):
            part = text[start:sentence_end].strip()
            if part:
                parts.append(part)
            start = cursor
            index = cursor
            continue
        index += 1

    final_part = text[start:].strip()
    if final_part:
        parts.append(final_part)
    return tuple(parts)


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
    for match, marker in zip(matches, markers, strict=True):
        prefix = text[: match.start()]
        supported_claims = [
            citations
            for claim, citations in resolved_claims
            if (
                _claim_is_supported(prefix, claim)
                and _claim_is_supported(text, claim)
            )
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
        if marker.citation_id not in allowed_ids:
            raise UncitedAnswerError(mismatch_message)


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
