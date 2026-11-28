from pathlib import Path
import tomllib


FORBIDDEN = {
    "openai",
    "anthropic",
    "google-generativeai",
    "chromadb",
    "sentence-transformers",
    "transformers",
    "llama-cpp-python",
    "ctransformers",
}


def test_runtime_dependencies_reject_competing_stacks_and_allow_gemini() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    names = {
        dependency.partition(">=")[0].partition("==")[0].strip().lower()
        for dependency in project["dependencies"]
    }
    assert names.isdisjoint(FORBIDDEN)
    assert "google-genai" in names


def test_readme_documents_ollama_default_and_gemini_provider() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8").lower()

    assert "ollama" in readme
    assert "gemini" in readme
    assert "gemini-4-hydrogen" in readme
    assert "gemini-decision-1" in readme
    assert "gemini-embedding-2" in readme
    assert "encrypted" in readme
    assert "reindex" in readme


def test_lockfile_has_no_forbidden_runtime_stacks() -> None:
    root = Path(__file__).resolve().parents[1]
    lock_text = (root / "uv.lock").read_text(encoding="utf-8").lower()
    residuals = sorted(name for name in FORBIDDEN if name in lock_text)
    assert residuals == []


def test_pdf_extraction_uses_docling_format_pdf_extra() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    dependencies = project["dependencies"]

    assert "docling-slim[format-pdf]==2.152.0" in dependencies


def test_runtime_source_has_no_cloud_or_competing_model_stacks() -> None:
    root = Path(__file__).resolve().parents[1]
    source_root = root / "src" / "file_tagger"
    residuals: set[str] = set()

    for source in source_root.rglob("*.py"):
        if "migration" in source.relative_to(source_root).parts:
            continue
        text = source.read_text(encoding="utf-8").lower()
        residuals.update(name for name in FORBIDDEN if name in text)

    assert residuals == set()


def test_readme_does_not_advertise_removed_runtime_stacks() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8").lower()
    residuals = sorted(name for name in FORBIDDEN if name in readme)
    assert residuals == []

