import ast
from pathlib import Path


CHAT_MODULE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "file_tagger"
    / "search"
    / "chat.py"
)


def test_chat_module_does_not_import_ai_service() -> None:
    tree = ast.parse(CHAT_MODULE.read_text(encoding="utf-8"))

    imported_modules = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_names = {
        alias.name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert "file_tagger.ai_service" not in imported_modules
    assert "ai_service" not in imported_modules
    assert "AIService" not in imported_names


def test_chat_dialog_has_no_legacy_provider_dispatch() -> None:
    tree = ast.parse(CHAT_MODULE.read_text(encoding="utf-8"))
    dialog = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ChatWithResultsDialog"
    )
    provider_names = {"openai", "anthropic", "gemini", "local"}
    provider_branches = [
        node
        for node in ast.walk(dialog)
        if isinstance(node, ast.If)
        and any(
            isinstance(value, ast.Constant)
            and isinstance(value.value, str)
            and value.value.lower() in provider_names
            for value in ast.walk(node.test)
        )
    ]
    provider_methods = {
        node.name
        for node in dialog.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and "provider" in node.name.lower()
    }

    assert provider_branches == []
    assert provider_methods == set()


def test_chat_module_does_not_import_cloud_provider_sdks() -> None:
    tree = ast.parse(CHAT_MODULE.read_text(encoding="utf-8"))
    imported_modules = {
        node.module.split(".", 1)[0].lower()
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_names = {
        alias.name.split(".", 1)[0].lower()
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert imported_modules.isdisjoint({"openai", "anthropic", "google", "google_generativeai"})
    assert imported_names.isdisjoint({"openai", "anthropic", "google", "google_generativeai"})
