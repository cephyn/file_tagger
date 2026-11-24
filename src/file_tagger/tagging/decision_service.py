import asyncio
from collections.abc import Coroutine, Mapping, Sequence
import json
import math
from typing import Any, Literal, Protocol, TypeVar

from ..adapters.ollama.gateway import OllamaGateway
from ..domain.contracts import DomainRecord, TagDecision


DEFAULT_DECISION_MODEL = "nimble:9b"
MAX_SYSTEM_ONE_QUESTIONS = 64
MAX_SYSTEM_ONE_TEXT_AND_SCHEMA_BYTES = 64 * 1024

T = TypeVar("T")
TagOutcome = Literal["accept", "reject", "review"]


class DecisionGateway(Protocol):
    async def system_one(
        self,
        model: str,
        *,
        state: Any,
        questions: Mapping[str, Any],
        keep_alive: float | str | None = "5m",
    ) -> Any: ...


class TagDecisionResult(DomainRecord):
    decisions: tuple[TagDecision, ...]

    @property
    def accepted(self) -> tuple[TagDecision, ...]:
        return tuple(item for item in self.decisions if item.decision == "accept")

    @property
    def rejected(self) -> tuple[TagDecision, ...]:
        return tuple(item for item in self.decisions if item.decision == "reject")

    @property
    def review(self) -> tuple[TagDecision, ...]:
        return tuple(item for item in self.decisions if item.decision == "review")


TagDecisionBatch = TagDecisionResult


def decide_tags(
    document_state: str,
    candidates: Sequence[str],
    model: str = DEFAULT_DECISION_MODEL,
    *,
    gateway: DecisionGateway | None = None,
) -> TagDecisionResult:
    return _run_sync(
        decide_tags_async(
            document_state,
            candidates,
            model,
            gateway=gateway,
        )
    )


async def decide_tags_async(
    document_state: str,
    candidates: Sequence[str],
    model: str = DEFAULT_DECISION_MODEL,
    *,
    gateway: DecisionGateway | None = None,
) -> TagDecisionResult:
    normalized = _validate_request(document_state, candidates, model)
    questions, question_candidates = _build_questions(normalized)

    owns_gateway = gateway is None
    selected_gateway: DecisionGateway = gateway or OllamaGateway()
    try:
        response = await selected_gateway.system_one(
            model.strip(),
            state=document_state,
            questions=questions,
        )
    finally:
        if owns_gateway:
            await _close_gateway(selected_gateway)

    return TagDecisionResult(
        decisions=_parse_decisions(response, normalized, question_candidates)
    )


def _validate_request(
    document_state: str,
    candidates: Sequence[str],
    model: str,
) -> tuple[str, ...]:
    if not isinstance(document_state, str) or not document_state.strip():
        raise ValueError("Document state must be a nonempty string.")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("A local Ollama model name is required.")
    if isinstance(candidates, str):
        raise ValueError("Candidates must be a sequence of tag names.")

    normalized = tuple(candidate.strip() for candidate in candidates)
    if not normalized:
        raise ValueError("At least one candidate tag is required.")
    if any(not candidate for candidate in normalized):
        raise ValueError("Candidate tag names must not be blank.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Candidate tag names must be unique.")
    if len(normalized) > MAX_SYSTEM_ONE_QUESTIONS:
        raise ValueError(
            "System One supports at most "
            f"{MAX_SYSTEM_ONE_QUESTIONS} questions per request."
        )

    questions, _ = _build_questions(normalized)
    request_size = len(
        json.dumps(
            {"state": document_state, "questions": questions},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    if request_size > MAX_SYSTEM_ONE_TEXT_AND_SCHEMA_BYTES:
        raise ValueError(
            "The System One document state and question schema are too large; "
            f"the limit is {MAX_SYSTEM_ONE_TEXT_AND_SCHEMA_BYTES} bytes."
        )
    return normalized


def _build_questions(
    candidates: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    questions: dict[str, dict[str, Any]] = {}
    question_candidates: dict[str, str] = {}
    for index, candidate in enumerate(candidates):
        question_name = f"candidate_{index:03d}"
        question_candidates[question_name] = candidate
        questions[question_name] = {
            "type": "choice",
            "instructions": (
                f"Decide whether the candidate tag {candidate!r} should be applied "
                "to the document state."
            ),
            "criteria": {
                "accept": "The tag is clearly relevant to the document state.",
                "review": "The tag may be relevant but needs a human decision.",
                "reject": "The tag is not supported by the document state.",
            },
        }
    return questions, question_candidates


def _parse_decisions(
    response: Any,
    candidates: Sequence[str],
    question_candidates: Mapping[str, str],
) -> tuple[TagDecision, ...]:
    answers = _required_field(response, "answers", "System One response")
    if not isinstance(answers, Mapping):
        raise ValueError("System One answers must be a mapping.")

    by_candidate: dict[str, Any] = {}
    candidate_set = set(candidates)
    unknown: list[str] = []
    for answer_name, answer in answers.items():
        if answer_name in question_candidates:
            candidate = question_candidates[answer_name]
        elif answer_name in candidate_set:
            candidate = answer_name
        else:
            unknown.append(str(answer_name))
            continue
        if candidate in by_candidate:
            raise ValueError(
                f"System One returned duplicate answers for candidate {candidate!r}."
            )
        by_candidate[candidate] = answer

    if unknown:
        raise ValueError(
            f"System One returned unknown candidates: {', '.join(sorted(unknown))}."
        )
    missing = candidate_set - set(by_candidate)
    if missing:
        raise ValueError(f"Missing System One answers: {', '.join(sorted(missing))}.")

    return tuple(
        _parse_answer(
            candidate,
            by_candidate[candidate],
            candidates,
        )
        for candidate in candidates
    )


def _parse_answer(
    candidate: str,
    answer: Any,
    candidates: Sequence[str],
) -> TagDecision:
    answer_type = _required_field(answer, "type", f"answer for {candidate}")
    if answer_type == "choice":
        decision, confidence = _parse_choice(answer, candidate, candidates)
    elif answer_type == "noul":
        decision, confidence = _parse_noul(answer)
    elif answer_type == "score":
        decision, confidence = _parse_score(answer)
    else:
        raise ValueError(f"System One returned unsupported answer type {answer_type!r}.")

    return TagDecision(
        tag=candidate,
        decision=decision,
        confidence=confidence,
    )


def _parse_choice(
    answer: Any,
    candidate: str,
    candidates: Sequence[str],
) -> tuple[TagOutcome, float]:
    selected = _required_field(answer, "choice", f"choice answer for {candidate}")
    if not isinstance(selected, str):
        raise ValueError("System One choice answers must select a string choice.")

    probabilities = _required_field(
        answer, "probabilities", f"choice answer for {candidate}"
    )
    if not isinstance(probabilities, Mapping):
        raise ValueError("System One choice probabilities must be a mapping.")
    checked_probabilities = {
        str(key): _probability(value, f"probability for {key}")
        for key, value in probabilities.items()
    }

    raw_confidence = _required_field(
        answer, "confidence", f"choice answer for {candidate}"
    )
    confidence = _probability(raw_confidence, "choice confidence")

    decisions = {"accept", "reject", "review"}
    if selected in decisions:
        if selected not in checked_probabilities:
            raise ValueError(f"Missing probability for selected choice {selected!r}.")
        if set(checked_probabilities) - decisions:
            raise ValueError("System One returned an unknown choice decision.")
        if selected == "accept":
            return "accept", confidence
        if selected == "reject":
            return "reject", confidence
        return "review", confidence

    if selected in candidates:
        unknown = set(checked_probabilities) - set(candidates)
        if unknown:
            raise ValueError(
                f"System One returned unknown candidate choices: "
                f"{', '.join(sorted(unknown))}."
            )
        if selected != candidate:
            raise ValueError(
                f"System One choice {selected!r} does not match candidate {candidate!r}."
            )
        if selected not in checked_probabilities:
            raise ValueError(f"Missing probability for selected candidate {selected!r}.")
        return "accept", confidence

    raise ValueError(f"System One returned unknown candidate choice {selected!r}.")


def _parse_noul(answer: Any) -> tuple[TagOutcome, float]:
    probability = _probability(
        _required_field(answer, "noul", "noul answer"),
        "noul probability",
    )
    if probability >= 0.5:
        return "accept", probability
    return "reject", 1.0 - probability


def _parse_score(answer: Any) -> tuple[TagOutcome, float]:
    legend = _required_field(answer, "legend", "score answer")
    probabilities = _required_field(answer, "probabilities", "score answer")
    if not isinstance(legend, Mapping) or len(legend) < 2:
        raise ValueError("System One score answers require at least two legend levels.")
    if not isinstance(probabilities, Mapping):
        raise ValueError("System One score probabilities must be a mapping.")
    for key, value in probabilities.items():
        _probability(value, f"score probability for {key}")

    raw_score = _required_field(answer, "score", "score answer")
    score = _finite_number(raw_score, "score")
    maximum = float(len(legend) - 1)
    if score < 0 or score > maximum:
        raise ValueError(f"System One score must be between 0 and {maximum:g}.")
    normalized = score / maximum
    decision: TagOutcome
    if normalized < 1.0 / 3.0:
        decision = "reject"
    elif normalized < 2.0 / 3.0:
        decision = "review"
    else:
        decision = "accept"
    confidence = _probability(
        _required_field(answer, "confidence", "score answer"),
        "score confidence",
    )
    return decision, confidence


def _required_field(value: Any, field: str, context: str) -> Any:
    if isinstance(value, Mapping):
        if field in value:
            return value[field]
    elif hasattr(value, field):
        return getattr(value, field)
    raise ValueError(f"Missing {field} in {context}.")


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    return number


def _probability(value: Any, label: str) -> float:
    probability = _finite_number(value, label)
    if probability < 0.0 or probability > 1.0:
        raise ValueError(f"{label} must be a probability between 0 and 1.")
    return probability


async def _close_gateway(gateway: DecisionGateway) -> None:
    close = getattr(gateway, "close", None)
    if close is None:
        return
    result = close()
    if hasattr(result, "__await__"):
        await result


def _run_sync(coroutine: Coroutine[Any, Any, T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()
