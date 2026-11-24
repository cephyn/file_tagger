from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import fsum


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

    source_ranks: dict[str, dict[str, int]] = {}
    for source in sorted(rankings):
        chunk_ids = rankings[source]
        unique_in_source: set[str] = set()
        for rank, chunk_id in enumerate(chunk_ids, start=1):
            if chunk_id in unique_in_source:
                continue
            unique_in_source.add(chunk_id)
            source_ranks.setdefault(chunk_id, {})[source] = rank

    scores = {
        chunk_id: fsum(
            1.0 / (k + source_ranks[chunk_id][source])
            for source in sorted(source_ranks[chunk_id])
        )
        for chunk_id in source_ranks
    }
    ordered = sorted(
        scores,
        key=lambda chunk_id: (
            -scores[chunk_id],
            min(source_ranks[chunk_id].values()),
            chunk_id,
            tuple(sorted(source_ranks[chunk_id].items())),
        ),
    )
    return tuple(
        FusedHit(
            chunk_id=chunk_id,
            rank=rank,
            score=scores[chunk_id],
            source_ranks=dict(sorted(source_ranks[chunk_id].items())),
        )
        for rank, chunk_id in enumerate(ordered, start=1)
    )
