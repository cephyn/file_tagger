import pytest
from pydantic import ValidationError

from file_tagger.domain.contracts import (
    ChunkRecord,
    Citation,
    ExtractionRecord,
    HardwareReport,
    ModelProfile,
    OllamaReadiness,
    PageRecord,
    RagPlan,
    SearchCandidate,
    TagDecision,
)
from file_tagger.domain.errors import (
    EmbeddingContextRejectedError,
    FileTaggerError,
    GeminiAuthenticationError,
    GeminiStructuredOutputError,
    GeminiTimeoutError,
    GeminiUnavailableError,
    IndexTransactionError,
    ModelIncompatibleError,
    ModelMissingError,
    NotIndexedError,
    OcrError,
    OllamaUnavailableError,
    PasswordEncryptionError,
)


def test_model_profile_accepts_provider_neutral_model_names() -> None:
    profile = ModelProfile(
        role="generation",
        model="gemma4:cloud",
        digest="sha256:test",
        context_length=256_000,
    )

    assert profile.model == "gemma4:cloud"


def test_local_ai_contracts_preserve_provenance_and_bounds() -> None:
    extraction = ExtractionRecord(
        file_path=r"C:\docs\guide.pdf",
        file_hash="abc123",
        parser="docling",
        page_count=1,
        status="complete",
    )
    page = PageRecord(
        document_id=7,
        page_number=1,
        ordinal=0,
        text="Guide text",
        markdown="## Guide\n\nGuide text",
    )
    chunk = ChunkRecord(
        chunk_id="document-7-page-1-0",
        document_id=7,
        page_id=3,
        ordinal=0,
        text="Guide text",
        content_hash="def456",
        model_digest="embeddinggemma-2:270m@sha256:test",
        index_version="v1",
    )
    candidate = SearchCandidate(
        chunk_id=chunk.chunk_id,
        file_path=extraction.file_path,
        page_number=page.page_number,
        heading="Guide",
        text=chunk.text,
        source="vector",
        rank=1,
        score=0.82,
    )
    citation = Citation(
        citation_id="source-1",
        file_path=candidate.file_path,
        page_number=1,
        chunk_id=chunk.chunk_id,
        excerpt="Guide text",
    )
    decision = TagDecision(tag="adventure", decision="accept", confidence=0.91)
    plan = RagPlan(
        question="Where does the guide begin?",
        intents=["locate"],
        exact_entities=["Guide"],
        exclusions=[],
        subqueries=["Guide beginning"],
    )

    assert extraction.file_hash == "abc123"
    assert candidate.citation(citation_id="source-1") == citation
    assert decision.confidence == 0.91
    assert plan.subqueries == ("Guide beginning",)


def test_search_candidate_rejects_out_of_range_score() -> None:
    with pytest.raises(ValidationError):
        SearchCandidate(
            chunk_id="chunk-1",
            file_path=r"C:\docs\guide.pdf",
            page_number=1,
            text="Guide text",
            source="vector",
            rank=1,
            score=1.1,
        )


def test_readiness_and_hardware_profiles_are_typed() -> None:
    hardware = HardwareReport(
        cpu_architecture="AMD64",
        logical_processors=16,
        total_ram_bytes=32 * 1024**3,
        available_ram_bytes=20 * 1024**3,
        available_disk_bytes=100 * 1024**3,
        gpu_name="NVIDIA RTX",
        vram_bytes=12 * 1024**3,
    )
    readiness = OllamaReadiness(
        available=True,
        version="0.35.0",
        base_url="http://127.0.0.1:11434",
        missing_models=[],
        incompatible_models=[],
        errors=[],
    )

    assert hardware.logical_processors == 16
    assert readiness.ready


@pytest.mark.parametrize(
    "error_type",
    [
        NotIndexedError,
        OllamaUnavailableError,
        GeminiUnavailableError,
        GeminiTimeoutError,
        GeminiAuthenticationError,
        GeminiStructuredOutputError,
        ModelMissingError,
        ModelIncompatibleError,
        EmbeddingContextRejectedError,
        OcrError,
        PasswordEncryptionError,
        IndexTransactionError,
    ],
)
def test_external_failures_share_application_base(error_type: type[FileTaggerError]) -> None:
    assert issubclass(error_type, FileTaggerError)
