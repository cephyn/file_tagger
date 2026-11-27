from pathlib import Path
import ast
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

MIGRATION_LEGACY_ALLOWLIST = {
    "src/file_tagger/migration/config_migration.py": {
        "api_keys",
        "selected_provider",
        "local_models",
    },
}

GEMINI_ADAPTER_ALLOWLIST = {
    "src/file_tagger/adapters/gemini/gateway.py",
}


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
    residuals: list[str] = []
    pattern = re.compile("|".join(re.escape(name) for name in FORBIDDEN_SOURCE_NAMES), re.I)

    for path in SOURCE_ROOT.rglob("*"):
        if not path.is_file() or path.suffix != ".py":
            continue
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            relative = path.relative_to(ROOT).as_posix()
            allowlisted = MIGRATION_LEGACY_ALLOWLIST.get(relative, frozenset())
            lines = [
                line
                for line in text.splitlines()
                if any(
                    match.group(0).lower() not in allowlisted
                    for match in pattern.finditer(line)
                )
            ]
            if lines:
                residuals.append(relative)

    assert residuals == []


def test_migration_code_has_no_forbidden_runtime_imports_or_calls() -> None:
    migration_root = SOURCE_ROOT / "migration"
    forbidden_roots = {
        "openai",
        "anthropic",
        "google",
        "google_generativeai",
        "chromadb",
        "sentence_transformers",
        "transformers",
        "llama_cpp",
        "llama_cpp_python",
        "ctransformers",
    }

    for path in migration_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = {alias.name.split(".", 1)[0].lower() for alias in node.names}
                assert forbidden_roots.isdisjoint(modules), str(path.relative_to(ROOT))
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".", 1)[0].lower()
                assert root not in forbidden_roots, str(path.relative_to(ROOT))
            elif isinstance(node, ast.Call):
                called = node.func
                root = (
                    called.value.id
                    if isinstance(called, ast.Attribute)
                    and isinstance(called.value, ast.Name)
                    else called.id
                    if isinstance(called, ast.Name)
                    else ""
                ).lower()
                assert root not in forbidden_roots, str(path.relative_to(ROOT))


def test_project_metadata_has_no_forbidden_runtime_names() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    residuals = sorted(name for name in FORBIDDEN_SOURCE_NAMES if name in text)
    assert residuals == []
    assert "google-genai==3.6.1" in text


def test_gemini_sdk_imports_are_confined_to_the_gemini_adapter() -> None:
    residuals: list[str] = []

    for path in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: set[str] = set()
            if isinstance(node, ast.Import):
                modules.update(alias.name.lower() for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module.lower())

            if any(module == "google" or module.startswith("google.") for module in modules):
                relative = path.relative_to(ROOT).as_posix()
                if relative not in GEMINI_ADAPTER_ALLOWLIST:
                    residuals.append(relative)

    assert residuals == []
