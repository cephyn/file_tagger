from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class FusedHit:
    chunk_id: str
    rank: int
    score: float
    source_ranks: Mapping[str, int]

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(self.source_ranks)


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int = 60,
) -> tuple[FusedHit, ...]:
    if k <= 0:
        raise ValueError("The reciprocal-rank constant must be positive.")

    scores: dict[str, float] = {}
    source_ranks: dict[str, dict[str, int]] = {}
    first_seen: dict[str, int] = {}
    seen_at = 0
    for source, chunk_ids in rankings.items():
        unique_in_source: set[str] = set()
        for rank, chunk_id in enumerate(chunk_ids, start=1):
            if chunk_id in unique_in_source:
                continue
            unique_in_source.add(chunk_id)
            if chunk_id not in first_seen:
                first_seen[chunk_id] = seen_at
                seen_at += 1
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            source_ranks.setdefault(chunk_id, {})[source] = rank

    ordered = sorted(
        scores,
        key=lambda chunk_id: (
            -scores[chunk_id],
            min(source_ranks[chunk_id].values()),
            first_seen[chunk_id],
            chunk_id,
        ),
    )
    return tuple(
        FusedHit(
            chunk_id=chunk_id,
            rank=rank,
            score=scores[chunk_id],
            source_ranks=source_ranks[chunk_id],
        )
        for rank, chunk_id in enumerate(ordered, start=1)
    )
