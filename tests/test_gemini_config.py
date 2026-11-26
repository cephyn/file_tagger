import pytest

from file_tagger.config import Config


def test_gemini_config_defaults_to_ollama_without_a_key(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path / "default"))

    config = Config("test-password")

    assert config.get_ai_provider() == "ollama"
    assert config.get_gemini_api_key() is None


def test_gemini_key_round_trips_in_encrypted_config(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path / "round-trip"))

    config = Config("test-password")
    assert config.set_ai_provider("gemini") is True
    config.set_gemini_api_key("test-secret")

    reloaded = Config("test-password")

    assert reloaded.get_ai_provider() == "gemini"
    assert reloaded.get_gemini_api_key() == "test-secret"
    assert "test-secret" not in config.paths.config_file.read_bytes().decode(
        errors="ignore"
    )


def test_gemini_provider_and_api_key_validation(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path / "validation"))
    config = Config("test-password")

    assert config.set_ai_provider("openai") is False
    assert config.get_ai_provider() == "ollama"
    assert config.set_ai_provider("gemini") is True
    assert config.get_ai_provider() == "gemini"

    with pytest.raises(ValueError, match="Gemini API key"):
        config.set_gemini_api_key("   ")


def test_clear_gemini_api_key_round_trips(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path / "clear"))
    config = Config("test-password")
    config.set_gemini_api_key("stored-secret")

    config.clear_gemini_api_key()

    assert Config("test-password").get_gemini_api_key() is None
