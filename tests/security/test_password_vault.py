from file_tagger.infrastructure.app_paths import AppPaths
from file_tagger.security.password_vault import CredentialStatus, PasswordVault


def test_password_vault_discovers_first_run_and_envelope_paths(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path))
    paths = AppPaths.discover()

    assert paths.password_key_file == tmp_path / "config" / ".password-key"
    assert PasswordVault.status(paths) is CredentialStatus.UNINITIALIZED


def test_password_vault_identifies_legacy_credentials(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path))
    paths = AppPaths.discover()
    paths.ensure_directories()
    paths.salt_file.write_bytes(b"salt")
    paths.password_hash_file.write_text("{}")

    assert PasswordVault.status(paths) is CredentialStatus.LEGACY
