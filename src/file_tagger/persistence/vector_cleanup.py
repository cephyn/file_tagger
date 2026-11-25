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
