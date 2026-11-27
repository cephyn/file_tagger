import asyncio
from collections.abc import Coroutine, Mapping, Sequence
from dataclasses import dataclass
import json
import math
from typing import Any, Literal, Protocol, TypeVar

from ..adapters.gemini.gateway import GEMINI_DECISION_MODEL
from ..adapters.ollama.gateway import OllamaGateway
from ..domain.contracts import DomainRecord, TagDecision
from ..domain.errors import GeminiUnavailableError


DEFAULT_DECISION_MODEL = "nimble:9b"
MAX_SYSTEM_ONE_QUESTIONS = 64
MAX_SYSTEM_ONE_TEXT_AND_SCHEMA_BYTES = 64 * 1024
# System One probability values are tolerated to six decimal places for rounding.
DISTRIBUTION_TOLERANCE = 1e-6

T = TypeVar("T")
TagOutcome = Literal["accept", "reject", "review"]
CHOICE_OUTCOMES = frozenset({"accept", "reject", "review"})


@dataclass(frozen=True)
class _Candidate:
    original: str
    prompt: str


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
    checked_candidates = _validate_request(document_state, candidates, model)
    questions, question_candidates = _build_questions(checked_candidates)

    if (
        model.strip() == GEMINI_DECISION_MODEL
        and (gateway is None or isinstance(gateway, OllamaGateway))
    ):
        raise GeminiUnavailableError(
            "The Gemini decision gateway is unavailable."
        )
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
        decisions=_parse_decisions(response, question_candidates)
    )


def _validate_request(
    document_state: str,
    candidates: Sequence[str],
    model: str,
) -> tuple[_Candidate, ...]:
    if not isinstance(document_state, str) or not document_state.strip():
        raise ValueError("Document state must be a nonempty string.")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("A decision model name is required.")
    if isinstance(candidates, str):
        raise ValueError("Candidates must be a sequence of tag names.")

    if not all(isinstance(candidate, str) for candidate in candidates):
        raise ValueError("Candidate tag names must be strings.")

    checked = tuple(
        _Candidate(original=candidate, prompt=candidate.strip())
        for candidate in candidates
    )
    if not checked:
        raise ValueError("At least one candidate tag is required.")
    if any(not candidate.prompt for candidate in checked):
        raise ValueError("Candidate tag names must not be blank.")
    if len({candidate.prompt for candidate in checked}) != len(checked):
        raise ValueError("Candidate tag names must be unique.")
    if len(checked) > MAX_SYSTEM_ONE_QUESTIONS:
        raise ValueError(
            "System One supports at most "
            f"{MAX_SYSTEM_ONE_QUESTIONS} questions per request."
        )

    questions, _ = _build_questions(checked)
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
    return checked


def _build_questions(
    candidates: Sequence[_Candidate],
) -> tuple[dict[str, dict[str, Any]], dict[str, _Candidate]]:
    questions: dict[str, dict[str, Any]] = {}
    question_candidates: dict[str, _Candidate] = {}
    for index, candidate in enumerate(candidates):
        question_name = f"candidate_{index:03d}"
        question_candidates[question_name] = candidate
        questions[question_name] = {
            "type": "choice",
            "instructions": (
                f"Decide whether the candidate tag {candidate.prompt!r} should be applied "
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
    question_candidates: Mapping[str, _Candidate],
) -> tuple[TagDecision, ...]:
    answers = _required_field(response, "answers", "System One response")
    if not isinstance(answers, Mapping):
        raise ValueError("System One answers must be a mapping.")

    by_question: dict[str, Any] = {}
    unknown: list[str] = []
    for answer_name, answer in answers.items():
        if not isinstance(answer_name, str) or answer_name not in question_candidates:
            unknown.append(str(answer_name))
            continue
        if answer_name in by_question:
            raise ValueError(
                f"System One returned duplicate answers for question {answer_name!r}."
            )
        by_question[answer_name] = answer

    if unknown:
        raise ValueError(
            f"System One returned unknown question IDs: {', '.join(sorted(unknown))}."
        )
    missing = set(question_candidates) - set(by_question)
    if missing:
        raise ValueError(
            f"Missing System One answers for questions: {', '.join(sorted(missing))}."
        )

    return tuple(
        _parse_answer(
            candidate.original,
            by_question[question_name],
        )
        for question_name, candidate in question_candidates.items()
    )


def _parse_answer(
    candidate: str,
    answer: Any,
) -> TagDecision:
    answer_type = _required_field(answer, "type", f"answer for {candidate}")
    if answer_type == "choice":
        decision, confidence = _parse_choice(answer, candidate)
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
) -> tuple[TagOutcome, float]:
    selected = _required_field(answer, "choice", f"choice answer for {candidate}")
    if not isinstance(selected, str):
        raise ValueError("System One choice answers must select a string choice.")

    probabilities = _required_field(
        answer, "probabilities", f"choice answer for {candidate}"
    )
    if not isinstance(probabilities, Mapping):
        raise ValueError("System One choice probabilities must be a mapping.")
    checked_probabilities = _validated_distribution(
        probabilities,
        CHOICE_OUTCOMES,
        "System One choice",
    )

    raw_confidence = _required_field(
        answer, "confidence", f"choice answer for {candidate}"
    )
    confidence = _probability(raw_confidence, "choice confidence")

    if selected not in CHOICE_OUTCOMES:
        raise ValueError(f"System One returned unknown choice {selected!r}.")

    selected_probability = checked_probabilities[selected]
    highest_probability = max(checked_probabilities.values())
    if selected_probability < highest_probability:
        raise ValueError(
            f"System One selected choice {selected!r} is not the highest probability."
        )
    if not math.isclose(
        confidence,
        selected_probability,
        rel_tol=0.0,
        abs_tol=DISTRIBUTION_TOLERANCE,
    ):
        raise ValueError(
            "System One choice confidence does not match the selected probability "
            f"within tolerance {DISTRIBUTION_TOLERANCE:g}."
        )
    if selected == "accept":
        return "accept", confidence
    if selected == "reject":
        return "reject", confidence
    return "review", confidence


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
    legend_keys = set(legend)
    if not all(isinstance(key, str) for key in legend_keys):
        raise ValueError("System One score legend keys must be strings.")
    expected_keys = {str(index) for index in range(len(legend))}
    if legend_keys != expected_keys:
        raise ValueError(
            "System One score legend keys must be a contiguous zero-based sequence."
        )
    if not isinstance(probabilities, Mapping):
        raise ValueError("System One score probabilities must be a mapping.")
    checked_probabilities = _validated_distribution(
        probabilities,
        expected_keys,
        "System One score",
    )

    raw_score = _required_field(answer, "score", "score answer")
    score = _finite_number(raw_score, "score")
    maximum = float(len(legend) - 1)
    if score < 0 or score > maximum:
        raise ValueError(f"System One score must be between 0 and {maximum:g}.")
    expected_score = math.fsum(
        int(level) * checked_probabilities[level] for level in expected_keys
    )
    if not math.isclose(
        score,
        expected_score,
        rel_tol=0.0,
        abs_tol=DISTRIBUTION_TOLERANCE,
    ):
        raise ValueError(
            "System One score does not match the weighted probability score "
            f"within tolerance {DISTRIBUTION_TOLERANCE:g}."
        )
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


def _validated_distribution(
    value: Mapping[Any, Any],
    expected_keys: set[str] | frozenset[str],
    label: str,
) -> dict[str, float]:
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} probability keys must be strings.")
    checked = {
        key: _probability(item, f"{label} probability for {key}")
        for key, item in value.items()
    }
    if set(checked) != set(expected_keys):
        raise ValueError(
            f"{label} probability keys are unknown or missing; expected exactly "
            f"{', '.join(sorted(expected_keys))}."
        )
    if not math.isclose(
        math.fsum(checked.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=DISTRIBUTION_TOLERANCE,
    ):
        raise ValueError(
            f"{label} probabilities must sum to 1 within tolerance "
            f"{DISTRIBUTION_TOLERANCE:g}."
        )
    return checked


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
