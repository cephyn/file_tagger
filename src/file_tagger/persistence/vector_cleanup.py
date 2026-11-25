from collections.abc import Iterable

from sqlalchemy import bindparam, select, text
from sqlalchemy.orm import Session

from .models import Chunk, Document


VECTOR_DIMENSIONS = (256, 512, 768)
_DELETE_BATCH_SIZE = 900


def delete_chunk_embeddings(
    session: Session,
    chunk_ids: Iterable[int],
) -> None:
    """Delete vec0 rows for chunks that are about to leave the relational schema."""
    unique_ids = sorted({int(chunk_id) for chunk_id in chunk_ids})
    if not unique_ids:
        return

    statements = tuple(
        text(
            f"DELETE FROM chunk_embeddings_{dimensions} "
            "WHERE chunk_id IN :chunk_ids"
        ).bindparams(bindparam("chunk_ids", expanding=True))
        for dimensions in VECTOR_DIMENSIONS
    )
    for offset in range(0, len(unique_ids), _DELETE_BATCH_SIZE):
        parameters = {
            "chunk_ids": unique_ids[offset : offset + _DELETE_BATCH_SIZE]
        }
        for statement in statements:
            session.execute(statement, parameters)


def delete_file_embeddings(
    session: Session,
    file_ids: Iterable[int],
) -> None:
    unique_file_ids = sorted({int(file_id) for file_id in file_ids})
    if not unique_file_ids:
        return
    chunk_ids = session.scalars(
        select(Chunk.id)
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.file_id.in_(unique_file_ids))
    )
    delete_chunk_embeddings(session, chunk_ids)


def delete_document_embeddings(
    session: Session,
    document_ids: Iterable[int],
) -> None:
    unique_document_ids = sorted({int(document_id) for document_id in document_ids})
    if not unique_document_ids:
        return
    chunk_ids = session.scalars(
        select(Chunk.id).where(Chunk.document_id.in_(unique_document_ids))
    )
    delete_chunk_embeddings(session, chunk_ids)


def purge_orphaned_embeddings(session: Session) -> None:
    """Delete vec0 rows whose relational chunk no longer exists."""
    available_tables = set(
        session.scalars(
            text(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name IN "
                "('chunks', 'chunk_embeddings_256', "
                "'chunk_embeddings_512', 'chunk_embeddings_768')"
            )
        )
    )
    if "chunks" not in available_tables:
        return

    for dimensions in VECTOR_DIMENSIONS:
        if f"chunk_embeddings_{dimensions}" not in available_tables:
            continue
        orphaned_ids = list(
            session.scalars(
                text(
                    f"SELECT embeddings.chunk_id "
                    f"FROM chunk_embeddings_{dimensions} AS embeddings "
                    "LEFT JOIN chunks ON chunks.id = embeddings.chunk_id "
                    "WHERE chunks.id IS NULL"
                )
            )
        )
        for offset in range(0, len(orphaned_ids), _DELETE_BATCH_SIZE):
            session.execute(
                text(
                    f"DELETE FROM chunk_embeddings_{dimensions} "
                    "WHERE chunk_id IN :chunk_ids"
                ).bindparams(bindparam("chunk_ids", expanding=True)),
                {
                    "chunk_ids": orphaned_ids[
                        offset : offset + _DELETE_BATCH_SIZE
                    ]
                },
            )
