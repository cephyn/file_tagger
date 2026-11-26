from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DomainRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HardwareReport(DomainRecord):
    cpu_architecture: str = Field(min_length=1)
    logical_processors: int = Field(gt=0)
    total_ram_bytes: int = Field(gt=0)
    available_ram_bytes: int = Field(ge=0)
    available_disk_bytes: int = Field(ge=0)
    gpu_name: str | None = None
    vram_bytes: int | None = Field(default=None, ge=0)


class OllamaReadiness(DomainRecord):
    available: bool
    version: str | None = None
    base_url: str
    missing_models: tuple[str, ...] = ()
    incompatible_models: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return (
            self.available
            and self.version is not None
            and not self.missing_models
            and not self.incompatible_models
            and not self.errors
        )


class ModelProfile(DomainRecord):
    role: Literal["embedding", "decision", "generation"]
    model: str = Field(min_length=1)
    digest: str = Field(min_length=1)
    context_length: int = Field(gt=0)
    dimensions: int | None = Field(default=None, gt=0)
    capabilities: tuple[str, ...] = ()
    measured_latency_ms: float | None = Field(default=None, ge=0)
    measured_memory_bytes: int | None = Field(default=None, ge=0)

    @property
    def default_index_version(self) -> str | None:
        if self.dimensions is None:
            return None
        return f"{self.model}@{self.digest}:{self.dimensions}"


class ExtractionRecord(DomainRecord):
    file_path: str = Field(min_length=1)
    file_hash: str = Field(min_length=1)
    parser: str = Field(min_length=1)
    page_count: int = Field(ge=0)
    status: Literal["complete", "warnings", "failed"]
    diagnostics: tuple[str, ...] = ()
    parser_version: str | None = None


class PageRecord(DomainRecord):
    document_id: int = Field(gt=0)
    page_number: int = Field(gt=0)
    ordinal: int = Field(ge=0)
    text: str
    markdown: str
    headings: tuple[str, ...] = ()
    quality: Literal["good", "degraded", "ocr"] = "good"


class ChunkRecord(DomainRecord):
    chunk_id: str = Field(min_length=1)
    document_id: int = Field(gt=0)
    page_id: int = Field(gt=0)
    ordinal: int = Field(ge=0)
    text: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)
    model_digest: str = Field(min_length=1)
    index_version: str = Field(min_length=1)
    heading: str | None = None
    section: str | None = None


class SearchCandidate(DomainRecord):
    chunk_id: str = Field(min_length=1)
    file_path: str = Field(min_length=1)
    page_number: int = Field(gt=0)
    heading: str | None = None
    text: str = Field(min_length=1)
    source: Literal["fts", "vector", "hybrid"]
    rank: int = Field(gt=0)
    score: float = Field(ge=0, le=1)

    def citation(self, citation_id: str) -> "Citation":
        return Citation(
            citation_id=citation_id,
            file_path=self.file_path,
            page_number=self.page_number,
            chunk_id=self.chunk_id,
            excerpt=self.text,
        )


class Citation(DomainRecord):
    citation_id: str = Field(min_length=1)
    file_path: str = Field(min_length=1)
    page_number: int = Field(gt=0)
    chunk_id: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)


class TagDecision(DomainRecord):
    tag: str = Field(min_length=1)
    decision: Literal["accept", "reject", "review"]
    confidence: float = Field(ge=0, le=1)
    rationale: str | None = None
    is_new: bool = False


class RagPlan(DomainRecord):
    question: str = Field(min_length=1)
    intents: tuple[str, ...] = ()
    exact_entities: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()
    subqueries: tuple[str, ...] = Field(min_length=1)

