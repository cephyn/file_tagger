from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src" / "file_tagger"

FORBIDDEN_SOURCE_NAMES = (
    "openai",
    "anthropic",
    "google-generativeai",
    "chromadb",
    "sentence-transformers",
    "transformers",
    "llama-cpp-python",
    "ctransformers",
)


def test_legacy_runtime_modules_and_debug_scripts_are_removed() -> None:
    removed_paths = (
        SOURCE_ROOT / "ai_service.py",
        SOURCE_ROOT / "vector_search" / "vector_search.py",
        SOURCE_ROOT / "vector_search" / "content_extractor.py",
        SOURCE_ROOT / "vector_search" / "document_chunker.py",
        SOURCE_ROOT / "vector_search" / "search_utils.py",
        SOURCE_ROOT / "check_status.py",
        SOURCE_ROOT / "check_vector_search.py",
        SOURCE_ROOT / "debug_vector_search.py",
    )
    assert not [path.relative_to(ROOT) for path in removed_paths if path.exists()]


def test_runtime_source_has_no_competing_stacks_outside_migration_allowlist() -> None:
    migration_root = SOURCE_ROOT / "migration"
    residuals: list[str] = []
    pattern = re.compile("|".join(re.escape(name) for name in FORBIDDEN_SOURCE_NAMES), re.I)

    for path in SOURCE_ROOT.rglob("*"):
        if not path.is_file() or path.suffix != ".py":
            continue
        if migration_root in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            residuals.append(str(path.relative_to(ROOT)))

    assert residuals == []


def test_project_metadata_has_no_forbidden_runtime_names() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    residuals = sorted(name for name in FORBIDDEN_SOURCE_NAMES if name in text)
    assert residuals == []
