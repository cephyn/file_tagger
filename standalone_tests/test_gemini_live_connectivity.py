import os
import asyncio
from typing import Any

import pytest
from google import genai
from google.genai import types

from file_tagger.adapters.gemini.gateway import (
    GEMINI_DECISION_MODEL,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_GENERATION_MODEL,
    GeminiGateway,
)


def _live_key() -> str:
    key = os.getenv("GEMINI_LIVE_TEST_API_KEY")
    if not isinstance(key, str) or not key.strip():
        pytest.skip("Set GEMINI_LIVE_TEST_API_KEY to run live Gemini connectivity tests.")
    return key.strip()


def _client() -> genai.Client:
    return genai.Client(
        api_key=_live_key(),
        http_options=types.HttpOptions(api_version="v1beta"),
    )


def test_quickstart_gemini_flash_latest() -> None:
    client = _client()
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents="Reply with OK.",
    )
    assert isinstance(response.text, str)
    assert response.text.strip()


def test_generation_model() -> None:
    client = _client()
    response = client.models.generate_content(
        model=GEMINI_GENERATION_MODEL,
        contents="Reply with OK.",
    )
    assert isinstance(response.text, str)
    assert response.text.strip()


def test_decision_model() -> None:
    client = _client()
    questions: Any = {
        "candidate_000": {
            "type": "choice",
            "instructions": "Should the adventure tag be applied?",
            "criteria": {
                "accept": "Clearly relevant.",
                "reject": "Not relevant.",
                "review": "Needs human review.",
            },
        }
    }
    result = client.decisions.create(
        model=GEMINI_DECISION_MODEL,
        input="A flooded ruin named Cursed Scroll 1.",
        questions=questions,
    )
    answers = result.answers or {}
    answer = answers["candidate_000"]
    assert hasattr(answer, "choice")
    assert hasattr(answer, "confidence")
    assert answer.type == "choice"
    assert answer.choice in {"accept", "reject", "review"}
    assert 0.0 <= answer.confidence <= 1.0


def test_embedding_model() -> None:
    client = _client()
    response = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents="A flooded ruin named Cursed Scroll 1.",
        config=types.EmbedContentConfig(
            output_dimensionality=768,
        ),
    )
    embeddings = response.embeddings or []
    assert len(embeddings) == 1
    values = embeddings[0].values
    assert isinstance(values, (list, tuple))
    assert len(values) == 768


def test_app_gateway_readiness_all_models() -> None:
    gateway = GeminiGateway(_live_key())

    async def check() -> None:
        try:
            report = await gateway.readiness(
                {
                    "generation": GEMINI_GENERATION_MODEL,
                    "decision": GEMINI_DECISION_MODEL,
                    "embedding": GEMINI_EMBEDDING_MODEL,
                }
            )
            assert report.available is True
            assert report.errors == ()
            assert report.incompatible_models == ()
        finally:
            await gateway.close()

    asyncio.run(check())
