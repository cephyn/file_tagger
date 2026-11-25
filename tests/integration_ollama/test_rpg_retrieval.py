from __future__ import annotations

import asyncio
import math
from typing import Iterable

import pytest

from file_tagger.domain.contracts import SearchCandidate
from file_tagger.rag.citations import answer_schema, parse_cited_answer
from file_tagger.rag.planner import build_search_plan_async
from file_tagger.search.results import make_result
from file_tagger.search.rrf import reciprocal_rank_fusion

pytestmark = pytest.mark.ollama


RPG_CORPUS = (
    (
        "rpg-encounters",
        "encounters",
        "A level 3 party in a flooded temple faces drowned sentinels and "
        "an octopus swarm near the moonlit altar.",
    ),
    (
        "rpg-treasure",
        "treasure",
        "The submerged reliquary contains 300 silver pieces, a pearl necklace, "
        "and a moon-shaped silver key.",
    ),
    (
        "rpg-tables",
        "random tables",
        "Roll d6 for temple hazards: 1 rising water, 2 falling masonry, "
        "3 hostile spirits, 4 current shift, 5 treasure glow, 6 strange echo.",
    ),
    (
        "rpg-npc",
        "npc",
        "Anira is an airship cartographer trapped beside the Sunken Vault.",
    ),
)


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / denominator


def _lexical_rank(query: str, chunk_ids: Iterable[str]) -> tuple[str, ...]:
    terms = frozenset(query.casefold().split())
    content = {chunk_id: text.casefold() for chunk_id, _, text in RPG_CORPUS}
    ranked = sorted(
        chunk_ids,
        key=lambda chunk_id: (
            -sum(term in content[chunk_id] for term in terms),
            chunk_id,
        ),
    )
    return tuple(ranked)


def test_rpg_rrf_recall_is_not_regressive(real_ollama, integration_profile) -> None:
    chunks = tuple(chunk_id for chunk_id, _, _ in RPG_CORPUS)
    query = "level 3 flooded temple encounters treasure"
    vectors = {
        chunk_id: real_ollama.embed(
            integration_profile.embedding,
            text,
        ).vector
        for chunk_id, _, text in RPG_CORPUS
    }
    query_vector = real_ollama.embed(
        integration_profile.embedding,
        query,
    ).vector
    vector_rank = tuple(
        sorted(
            chunks,
            key=lambda chunk_id: (
                -_cosine(query_vector, vectors[chunk_id]),
                chunk_id,
            ),
        )
    )
    lexical_rank = _lexical_rank(query, chunks)
    fused = reciprocal_rank_fusion(
        {"lexical": lexical_rank, "vector": vector_rank},
        k=60,
    )

    def recall_at_k(ranking: Iterable[str], relevant: frozenset[str], k: int) -> float:
        hits = sum(1 for chunk_id in tuple(ranking)[:k] if chunk_id in relevant)
        return hits / len(relevant)

    relevant = frozenset({"rpg-encounters", "rpg-treasure"})
    lexical_recall = recall_at_k(lexical_rank, relevant, 4)
    fused_recall = recall_at_k(
        tuple(hit.chunk_id for hit in fused),
        relevant,
        4,
    )
    real_ollama.metrics["recall_at_k"] = {
        "lexical": lexical_recall,
        "rrf": fused_recall,
    }

    assert fused_recall >= lexical_recall
    assert fused_recall >= 0.5


def test_rpg_plan_covers_evidence_categories_and_citations(
    real_ollama,
    integration_profile,
) -> None:
    corpus_context = "\n\n".join(f"## {category}\n{text}" for _, category, text in RPG_CORPUS)
    question = "Find level 3 encounters, treasure, and random tables in the flooded temple."
    plan = asyncio.run(
        build_search_plan_async(
            question,
            corpus_context,
            model=integration_profile.generation,
            fallback_model=None,
            gateway=real_ollama.gateway,
        )
    )

    category_names = frozenset(category.casefold() for _, category, _ in RPG_CORPUS)
    assert category_names.intersection(plan.evidence_categories)

    evidence = tuple(
        make_result(
            {
                "stable_id": chunk_id,
                "file_path": f"rpg/{chunk_id}.md",
                "page_number": 1,
                "heading": category,
                "text": text,
            },
            source="hybrid",
            rank=index,
            score=1.0,
        )
        for index, (chunk_id, category, text) in enumerate(RPG_CORPUS, start=1)
    )
    candidates = tuple(
        SearchCandidate(
            chunk_id=result.chunk_id,
            file_path=result.file_path,
            page_number=result.page_number,
            heading=result.heading,
            text=result.text,
            source="hybrid",
            rank=result.rank,
            score=result.score,
        )
        for result in evidence
    )

    # Verify the citation contract against synthetic RPG evidence.
    first = candidates[0]
    citation = first.citation("[source:rpg-encounters]")
    assert citation.chunk_id == "rpg-encounters"
    assert citation.excerpt == first.text

    response = real_ollama.gateway
    answer_text = asyncio.run(
        response.chat(
            integration_profile.generation,
            messages=[
                {
                    "role": "system",
                    "content": "Use only the supplied evidence and return a cited JSON answer.",
                },
                {
                    "role": "user",
                    "content": (f"{question}\nEvidence: {first.text}\nChunk ID: rpg-encounters"),
                },
            ],
            schema=answer_schema(),
        )
    )
    parsed = parse_cited_answer(
        question,
        answer_text,
        evidence,
    )
    assert parsed.claims
    assert parsed.citations
