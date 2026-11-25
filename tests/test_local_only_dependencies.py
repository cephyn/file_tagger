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


def test_runtime_dependencies_are_local_ollama_only() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    names = {
        dependency.partition(">=")[0].partition("==")[0].strip().lower()
        for dependency in project["dependencies"]
    }
    assert names.isdisjoint(FORBIDDEN)


def test_lockfile_has_no_forbidden_runtime_stacks() -> None:
    root = Path(__file__).resolve().parents[1]
    lock_text = (root / "uv.lock").read_text(encoding="utf-8").lower()
    residuals = sorted(name for name in FORBIDDEN if name in lock_text)
    assert residuals == []

