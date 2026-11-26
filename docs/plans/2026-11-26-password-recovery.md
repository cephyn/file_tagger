# Password Recovery Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add reliable recovery-key password reset from the login screen, preserve future encrypted settings, and provide a quarantined reset for legacy or keyless lockouts.

**Architecture:** Replace password-direct config encryption with a random data key protected by password and recovery envelopes. Migrate legacy credentials after successful authentication, expose recover/reset through a login dialog, and preserve the tagging database during configuration quarantine.

**Tech Stack:** Python 3.13, PySide6, cryptography/Fernet, PBKDF2-HMAC-SHA256, pytest, pytest-qt, Ruff, mypy

---

## Working Rules

- Run every new test before implementation and record the expected failure.
- Do not put passwords, recovery keys, wrapped keys, or decrypted configuration
  in logs.
- Do not catch recovery or decryption failures and return default configuration.
- Use existing `FILE_TAGGER_DATA_DIR` test isolation.
- Commit each task independently with the
  `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>` trailer.

### Task 1: Add credential format discovery and envelope paths

**Files:**
- Modify: `src/file_tagger/infrastructure/app_paths.py`
- Modify: `src/file_tagger/domain/errors.py`
- Create: `tests/security/test_password_vault.py`

**Step 1: Write the failing test**

```python
from file_tagger.infrastructure.app_paths import AppPaths
from file_tagger.security.password_vault import CredentialStatus, PasswordVault


def test_password_vault_discovers_first_run_and_envelope_paths(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path))
    paths = AppPaths.discover()

    assert paths.password_key_file == tmp_path / "config" / ".password-key"
    assert PasswordVault.status(paths) is CredentialStatus.UNINITIALIZED
```

Add a legacy-detection case:

```python
def test_password_vault_identifies_legacy_credentials(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path))
    paths = AppPaths.discover()
    paths.ensure_directories()
    paths.salt_file.write_bytes(b"salt")
    paths.password_hash_file.write_text("{}")

    assert PasswordVault.status(paths) is CredentialStatus.LEGACY
```

**Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/security/test_password_vault.py -v`

Expected: FAIL because the security package, vault class, and
`password_key_file` do not exist.

**Step 3: Write the minimal implementation**

Add `password_key_file: Path` to `AppPaths` and populate it with
`config_dir / ".password-key"`.

Create `src/file_tagger/security/__init__.py` and
`src/file_tagger/security/password_vault.py` with:

```python
from enum import Enum, auto
from pathlib import Path


class CredentialStatus(Enum):
    UNINITIALIZED = auto()
    ENVELOPE = auto()
    LEGACY = auto()
    CORRUPT = auto()


class PasswordVault:
    @classmethod
    def status(cls, paths) -> CredentialStatus:
        envelope_files = (
            paths.password_key_file,
            paths.password_hash_file,
            paths.recovery_file,
        )
        if all(path.is_file() for path in envelope_files):
            return CredentialStatus.ENVELOPE
        if any(path.exists() for path in envelope_files) or paths.salt_file.exists():
            return CredentialStatus.LEGACY
        return CredentialStatus.UNINITIALIZED
```

Expand the real implementation in subsequent tasks; the initial version only
establishes the public format discovery API.

Add distinct exception types in `domain/errors.py`:

```python
class CredentialFormatError(PasswordEncryptionError):
    """Raised when credential metadata is incomplete or unsupported."""


class CredentialRecoveryError(PasswordEncryptionError):
    """Raised when recovery or reset cannot be completed safely."""
```

**Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/security/test_password_vault.py -v`

Expected: PASS for first-run and legacy discovery.

**Step 5: Commit**

```bash
git add src/file_tagger/infrastructure/app_paths.py src/file_tagger/domain/errors.py src/file_tagger/security tests/security/test_password_vault.py
git commit -m "feat: discover password envelope state" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 2: Implement password-envelope authentication

**Files:**
- Modify: `src/file_tagger/security/password_vault.py`
- Modify: `tests/security/test_password_vault.py`

**Step 1: Write the failing test**

```python
def test_password_vault_encrypts_config_and_changes_password(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path))
    paths = AppPaths.discover()
    paths.ensure_directories()

    vault, recovery_key = PasswordVault.create(paths, "old-password")
    ciphertext = vault.encrypt(b'{"home_directory":"C:\\\\Users\\\\owner"}')
    paths.config_file.write_bytes(ciphertext)

    assert recovery_key
    assert vault.decrypt(ciphertext).startswith(b'{"home_directory"')
    vault.change_password("old-password", "new-password")
    assert PasswordVault.open(paths, "new-password").decrypt(
        paths.config_file.read_bytes()
    ) == vault.decrypt(ciphertext)

    try:
        PasswordVault.open(paths, "old-password")
    except CredentialRecoveryError:
        pass
    else:
        raise AssertionError("the old password must stop working")
```

Also add:

```python
def test_password_vault_rejects_corrupt_envelope(tmp_path, monkeypatch) -> None:
    ...
    paths.password_key_file.write_text("{invalid")

    assert PasswordVault.status(paths) is CredentialStatus.CORRUPT
    with pytest.raises(CredentialFormatError):
        PasswordVault.open(paths, "old-password")
```

**Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/security/test_password_vault.py -v`

Expected: FAIL because `create`, `open`, `encrypt`, `decrypt`, and
`change_password` are missing.

**Step 3: Write the minimal implementation**

Implement the vault with one JSON format version:

```python
PASSWORD_ENVELOPE_VERSION = 1
PBKDF2_ITERATIONS = 310_000


def _derive_fernet(secret: str, salt: bytes) -> Fernet:
    material = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    ).derive(secret.encode())
    return Fernet(base64.urlsafe_b64encode(material))
```

`PasswordVault.create` must:

1. Generate a raw data key with `Fernet.generate_key()`.
2. Create the password envelope from an independent random salt.
3. Store the PBKDF2 password verifier in `.passhash`.
4. Create a recovery envelope from a fresh high-entropy recovery key.
5. Encrypt the three metadata files through an all-or-nothing staged writer.
6. Return a `PasswordVault(paths, data_key)` and the one-time recovery key.

`PasswordVault.open` must validate the JSON version and types, verify the
password with `hmac.compare_digest`, decrypt the password envelope, and reject
invalid material with `CredentialRecoveryError` or `CredentialFormatError`.

`encrypt` and `decrypt` must use the raw data key. `change_password` must verify
the old password, create a new password verifier/envelope transactionally, and
replace `self.data_key` only if both writes succeed.

Use cryptographically secure tokens from `secrets.token_urlsafe(24)` for
recovery keys. Normalize whitespace only at the UI boundary, not in cryptography
helpers.

**Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/security/test_password_vault.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/file_tagger/security/password_vault.py tests/security/test_password_vault.py
git commit -m "feat: authenticate password key envelopes" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 3: Implement recovery and recovery-key rotation

**Files:**
- Modify: `src/file_tagger/security/password_vault.py`
- Modify: `tests/security/test_password_vault.py`

**Step 1: Write the failing test**

```python
def test_recovery_key_resets_password_without_reencrypting_config(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path))
    paths = AppPaths.discover()
    paths.ensure_directories()
    vault, recovery_key = PasswordVault.create(paths, "forgotten")
    ciphertext = vault.encrypt(b"config")
    paths.config_file.write_bytes(ciphertext)

    recovered = PasswordVault.recover(paths, recovery_key, "replacement")

    assert recovered.decrypt(paths.config_file.read_bytes()) == b"config"
    assert paths.config_file.read_bytes() == ciphertext
    with pytest.raises(CredentialRecoveryError):
        PasswordVault.open(paths, "forgotten")


def test_rotating_recovery_key_invalidates_only_old_recovery_key(
    tmp_path, monkeypatch
) -> None:
    ...
    new_key = vault.rotate_recovery_key("current-password")

    with pytest.raises(CredentialRecoveryError):
        PasswordVault.recover(paths, old_key, "invalid")
    PasswordVault.recover(paths, new_key, "replacement")
```

**Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/security/test_password_vault.py -v`

Expected: FAIL because `recover` and `rotate_recovery_key` are missing.

**Step 3: Write the minimal implementation**

`recover` reads and authenticates the recovery envelope, unwraps the same data
key, and transactionally writes a new password verifier and password envelope.
It must not rewrite `config.encrypted` or the recovery envelope.

`rotate_recovery_key` verifies the current password, generates a new key, and
atomically replaces the recovery envelope. Return the new plaintext key exactly
once.

All decryption exceptions must be translated at the service boundary into a
generic `CredentialRecoveryError`; invalid envelope JSON remains a
`CredentialFormatError`.

**Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/security/test_password_vault.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/file_tagger/security/password_vault.py tests/security/test_password_vault.py
git commit -m "feat: recover and rotate password credentials" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 4: Migrate legacy credentials and configuration

**Files:**
- Modify: `src/file_tagger/security/password_vault.py`
- Modify: `src/file_tagger/config.py`
- Modify: `tests/security/test_password_vault.py`
- Modify: `tests/test_startup_modern_integration.py`
- Modify: `tests/migration/test_config_migration.py`

**Step 1: Write the failing test**

Add a helper that creates current legacy `.salt`, `.passhash`, `.recovery`, and
password-encrypted `config.encrypted`, then assert:

```python
def test_legacy_config_migrates_after_authenticated_login(
    tmp_path, monkeypatch
) -> None:
    ...
    config = Config(password)

    assert PasswordVault.status(paths) is CredentialStatus.ENVELOPE
    assert config.get_home_directory() == r"C:\Users\owner"
    assert config.pending_recovery_key
    assert Config(password).get_system_message() == "Keep this setting"
    with pytest.raises(CredentialRecoveryError):
        PasswordVault.open(paths, old_wrong_password)
```

Add an assertion to the existing config-migration test that copied target files
remain legacy until `ReloadableConfig.reload_after_migration()` runs, then become
an envelope without exposing secret values.

**Step 2: Run the test to verify it fails**

Run:

`uv run pytest tests/security/test_password_vault.py tests/migration/test_config_migration.py tests/test_startup_modern_integration.py::test_pending_legacy_config_does_not_bootstrap_a_modern_password tests/test_startup_modern_integration.py::test_bad_pending_legacy_password_fails_instead_of_poisoning_startup -v`

Expected: FAIL because legacy migration into envelopes is missing.

**Step 3: Write the minimal implementation**

Add `PasswordVault.migrate_legacy(paths, password)`. It must:

1. Verify the legacy password hash.
2. Decrypt existing config with the legacy password/salt.
3. Build fresh envelope data and the final config ciphertext.
4. Stage the config, `.passhash`, `.password-key`, and `.recovery`.
5. Back up all pre-migration credentials.
6. Replace files with rollback to the backup if any replacement fails.
7. Return the migrated vault and one new recovery key.

Refactor `Config` to own a `PasswordVault`:

```python
class Config:
    def __init__(self, password: str):
        self.paths = _get_app_paths()
        status = PasswordVault.status(self.paths)
        if status is CredentialStatus.ENVELOPE:
            self._vault = PasswordVault.open(self.paths, password)
            self._pending_recovery_key = None
        elif status is CredentialStatus.LEGACY:
            self._vault, self._pending_recovery_key = (
                PasswordVault.migrate_legacy(self.paths, password)
            )
        elif self._legacy_config_pending:
            self._vault = None
            self._pending_recovery_key = None
        else:
            self._vault, self._pending_recovery_key = (
                PasswordVault.create(self.paths, password)
            )
        self.password = password
        self.config_data = self._load_config()
```

Replace `_load_config`'s broad fallback with explicit `PasswordEncryptionError`.
Keep only the documented "file absent" default behavior. `Config.save` must
encrypt through `self._vault.encrypt`.

`reload_after_migration` must open or migrate the copied target credentials and
refresh `config_data`. Remove module-level `get_encryption_key` use from normal
runtime after migration; legacy helpers may remain private to the vault.

Keep `migrate_legacy_config` responsible only for copying CWD credentials into
the per-user location. The envelope migration happens in `Config` after that
copy, so existing startup ordering remains valid.

**Step 4: Run the test to verify it passes**

Run the Task 3 command again.

Expected: PASS with the pending-legacy startup tests unchanged.

**Step 5: Commit**

```bash
git add src/file_tagger/security/password_vault.py src/file_tagger/config.py tests/security/test_password_vault.py tests/migration/test_config_migration.py tests/test_startup_modern_integration.py
git commit -m "feat: migrate legacy password encryption" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 5: Add quarantined configuration reset

**Files:**
- Create: `src/file_tagger/security/password_reset.py`
- Create: `tests/security/test_password_reset.py`
- Modify: `src/file_tagger/security/__init__.py`

**Step 1: Write the failing test**

```python
def test_reset_quarantines_encrypted_files_but_preserves_database(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("FILE_TAGGER_DATA_DIR", str(tmp_path))
    paths = AppPaths.discover()
    paths.ensure_directories()
    PasswordVault.create(paths, "forgotten")
    paths.config_file.write_bytes(b"encrypted settings")
    paths.database.write_bytes(b"tag database")

    result = reset_encrypted_configuration(paths, "replacement")

    assert result.backup_dir.is_dir()
    assert (result.backup_dir / "config.encrypted").read_bytes() == b"encrypted settings"
    assert paths.database.read_bytes() == b"tag database"
    PasswordVault.open(paths, "replacement")
    assert result.recovery_key
```

Add rollback and external-legacy tests:

```python
def test_reset_restores_files_if_new_credentials_cannot_be_created(...):
    ...

def test_reset_quarantines_pending_cwd_legacy_credentials(...):
    ...
```

**Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/security/test_password_reset.py -v`

Expected: FAIL because the reset service is missing.

**Step 3: Write the minimal implementation**

Create:

```python
@dataclass(frozen=True, slots=True)
class PasswordResetResult:
    backup_dir: Path
    recovery_key: str


def reset_encrypted_configuration(
    paths: AppPaths,
    new_password: str,
) -> PasswordResetResult:
    ...
```

The function determines the active credential set:

- target per-user files if `paths.config_file` exists;
- otherwise legacy CWD files if `paths.legacy_database.parent /
  "config.encrypted"` exists;
- otherwise target first-run state.

It must never move `paths.database`, cache, index data, `file_tags.db`, or
`.chroma`. It creates a timestamped directory below
`paths.data_dir / "password-reset-backups"`, verifies copied bytes, then moves
only config/credential files. After `PasswordVault.create` succeeds, initialize
and save default encrypted configuration. On initialization failure, restore
moved files before re-raising `CredentialRecoveryError`.

**Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/security/test_password_reset.py -v`

Expected: PASS, including rollback and legacy CWD cases.

**Step 5: Commit**

```bash
git add src/file_tagger/security/password_reset.py src/file_tagger/security/__init__.py tests/security/test_password_reset.py
git commit -m "feat: add quarantined password reset" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 6: Add recovery and reset UI to the login screen

**Files:**
- Modify: `src/file_tagger/login.py`
- Modify: `src/file_tagger/password_management.py`
- Create: `tests/ui/test_password_recovery.py`
- Modify: `tests/test_startup_modern_integration.py`

**Step 1: Write the failing test**

```python
def test_login_exposes_forgot_password_dialog(qtbot, monkeypatch) -> None:
    from file_tagger.login import LoginScreen
    from file_tagger.password_management import PasswordRecoveryDialog

    opened = []
    monkeypatch.setattr(
        PasswordRecoveryDialog,
        "exec",
        lambda self: opened.append(self) or QDialog.Rejected,
    )
    screen = LoginScreen()
    qtbot.addWidget(screen)
    qtbot.mouseClick(screen.forgot_password_button, Qt.MouseButton.LeftButton)

    assert len(opened) == 1
```

Add dialog tests for these public behaviors:

```python
def test_recovery_dialog_rejects_mismatched_passwords(...):
    ...

def test_recovery_dialog_disables_recovery_for_legacy_and_explains_reset(...):
    ...

def test_recovery_dialog_quarantines_only_after_typed_confirmation(...):
    ...

def test_accepted_recovery_retries_login_with_new_password(...):
    ...
```

**Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/ui/test_password_recovery.py -v`

Expected: FAIL because the forgot-password button and recovery dialog do not
exist.

**Step 3: Write the minimal implementation**

In `password_management.py`, add `PasswordRecoveryDialog`:

- A `Recover with key` section with recovery key, new password, and confirm
  fields.
- A `Reset configuration` section with new password, confirm, and an input that
  must equal `RESET`.
- Legacy/envelope status text from `PasswordVault.status`.
- Generic invalid-key messaging without credential detail.
- `new_password` and `new_recovery_key` result properties.
- No secret logging.

Its recover action calls `PasswordVault.recover`. Its reset action calls
`reset_encrypted_configuration`. Disable recovery for
`CredentialStatus.LEGACY` and require the reset controls in that state.

In `LoginScreen._setup_password_section`, add a visible
`self.forgot_password_button`. Connect it to `show_password_recovery`.
After an accepted dialog, show a newly generated reset key if present, set the
password field from `dialog.new_password`, and call `submit_password`.

Remove the inaccessible recover tab and `Show My Recovery Key` action from the
authenticated `PasswordManagementDialog`; recovery keys are one-time secrets.
The existing password-management dialog must keep password change and add a
`Rotate Recovery Key` action.

**Step 4: Run the test to verify it passes**

Run:

`uv run pytest tests/ui/test_password_recovery.py tests/test_startup_modern_integration.py::test_login_accepts_modern_runtime_without_legacy_search -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/file_tagger/login.py src/file_tagger/password_management.py tests/ui/test_password_recovery.py tests/test_startup_modern_integration.py
git commit -m "feat: add login password recovery" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 7: Add one-time key acknowledgement and password-management wiring

**Files:**
- Modify: `src/file_tagger/config.py`
- Modify: `src/file_tagger/login.py`
- Modify: `src/file_tagger/password_management.py`
- Modify: `src/file_tagger/file_tag_manager.py`
- Modify: `tests/ui/test_password_recovery.py`

**Step 1: Write the failing test**

```python
def test_first_setup_requires_recovery_key_acknowledgement(
    qtbot, monkeypatch
) -> None:
    ...
    config = SimpleNamespace(
        pending_recovery_key="one-time-key",
        acknowledge_recovery_key=acknowledge,
    )
    screen.on_initialization_finished((session, config, runtime))

    assert shown == ["one-time-key"]
    assert acknowledged == 1
    assert screen.password_input.text() == ""
```

Also test:

```python
def test_manager_opens_password_management_with_correct_arguments(qtbot, monkeypatch):
    ...
    dialog = PasswordManagementDialog(config=config, parent=manager)
```

**Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/ui/test_password_recovery.py -v`

Expected: FAIL because `pending_recovery_key`,
acknowledgement, and correct dialog construction do not exist.

**Step 3: Write the minimal implementation**

Add:

```python
@property
def pending_recovery_key(self) -> str | None:
    return self._pending_recovery_key


def acknowledge_recovery_key(self) -> None:
    self._pending_recovery_key = None
```

In login initialization completion, if `pending_recovery_key` exists, display a
modal recovery-key message with explicit save advice and acknowledgement. Clear
the key only after the user acknowledges. Ensure the application does not open
behind the acknowledgement dialog.

In `FileTagManager.show_password_management`, construct
`PasswordManagementDialog(self.config, self)`. In the management dialog,
`Rotate Recovery Key` calls `Config.rotate_recovery_key`, shows the returned key
once, and never offers decryption of the existing key.

`Config.change_password` delegates to the vault and keeps the in-memory vault
current. `Config.rotate_recovery_key` returns the key from the vault.

**Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/ui/test_password_recovery.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/file_tagger/config.py src/file_tagger/login.py src/file_tagger/password_management.py src/file_tagger/file_tag_manager.py tests/ui/test_password_recovery.py
git commit -m "feat: acknowledge and rotate recovery keys" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 8: Verify migration safety and full behavior

**Files:**
- Modify: `README.md`
- Modify: `tests/security/test_password_reset.py`
- Modify: `tests/security/test_password_vault.py`
- Modify: `tests/ui/test_password_recovery.py`

**Step 1: Add the final failing documentation/security assertions**

Add tests that inspect log capture and ensure it contains no password or
recovery key:

```python
def test_failed_recovery_does_not_log_secret(caplog, tmp_path, monkeypatch):
    ...
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(CredentialRecoveryError):
            PasswordVault.recover(paths, "wrong-secret-key", "new-password")
    assert "wrong-secret-key" not in caplog.text
    assert "new-password" not in caplog.text
```

Add reset assertions that the original config backup remains byte-identical,
database paths survive, and invalid reset confirmation never invokes the reset
service.

**Step 2: Run the focused suite to establish remaining failures**

Run:

`uv run pytest tests/security tests/migration/test_config_migration.py tests/ui/test_password_recovery.py tests/test_startup_modern_integration.py::test_pending_legacy_config_does_not_bootstrap_a_modern_password tests/test_startup_modern_integration.py::test_bad_pending_legacy_password_fails_instead_of_poisoning_startup tests/test_startup_modern_integration.py::test_login_accepts_modern_runtime_without_legacy_search -v`

Expected: any remaining failures identify integration gaps; no unrelated full
suite is run yet.

**Step 3: Close integration gaps and document usage**

Fix only failures caused by the password work. Update `README.md` with:

- password recovery is available from **Forgot password**;
- recovery keys are shown only once;
- old pre-envelope recovery keys cannot recover an already locked legacy
  install;
- reset quarantines encrypted settings and preserves SQLite/tag/index data;
- backup locations and manual-restoration warning.

Do not document raw key storage or recovery bypasses.

**Step 4: Run verification**

Run in sequence:

```bash
uv run pytest tests/security tests/migration/test_config_migration.py tests/ui/test_password_recovery.py -q
uv run pytest tests/test_startup_modern_integration.py -q
uv run ruff check src/file_tagger tests/security tests/ui/test_password_recovery.py
uv run mypy src/file_tagger
```

Expected: all commands exit `0`, except pre-existing mypy diagnostics must be
compared to the baseline and no new password-recovery diagnostics may be
introduced.

**Step 5: Commit**

```bash
git add README.md tests/security tests/ui/test_password_recovery.py
git commit -m "docs: explain password recovery" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

## Completion Checklist

- [ ] Existing successful legacy login migrates to envelopes and reveals a new key.
- [ ] Existing forgotten-password legacy installs explain the limitation and allow reset.
- [ ] Recovery key resets the password without rewriting encrypted config.
- [ ] Reset preserves the tagging database and quarantines active credential files.
- [ ] Corruption and failed backup do not silently reset or erase data.
- [ ] Password/recovery secrets never appear in logs.
- [ ] Focused tests, startup integration tests, Ruff, and type diagnostics pass.
