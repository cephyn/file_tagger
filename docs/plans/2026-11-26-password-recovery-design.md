# Password Recovery Design

## Goal

Let a user reset a forgotten File Tagger password from the login screen without
losing encrypted settings when a valid recovery key is available. Provide a
clearly destructive, database-preserving fallback when the key is also
unavailable.

## Constraints

- A recovery key must not appear in logs, status text, or exception messages.
- Configuration remains encrypted at rest.
- Normal password changes must not require decrypting and re-encrypting the
  entire configuration.
- Existing credentials must remain usable.
- A failed recovery, migration, or reset must not overwrite or discard the
  existing configuration.
- Tag databases and search data are not configuration credentials and must
  survive a reset.

## Chosen Architecture

Use a random data encryption key for the configuration. Store the data key in
two independently encrypted envelopes:

1. A password envelope derived from the login password.
2. A recovery envelope derived from the high-entropy recovery key.

The password verifier, key-derivation salt, and password envelope are stored
separately from the recovery envelope. Both envelopes contain only encrypted
material. The configuration file contains data encrypted by the random data
key.

Changing the password decrypts the data key with the old password and replaces
only the password envelope. Recovering the password decrypts the data key with
the recovery key and replaces only the password envelope. Rotating the recovery
key replaces only the recovery envelope. No supported operation stores the
plaintext recovery key.

New recovery keys use a cryptographically secure generator and are displayed
only when created or deliberately rotated. The user must acknowledge storing the
key. Generation and acknowledgement happen at first setup and after migration
from the old format.

## Legacy Compatibility

The current `.recovery` file encrypts the recovery key with the forgotten
password. It is therefore impossible to validate or use that recovery key
without the old password. This cannot be repaired for an installation that is
already locked out.

After a user successfully authenticates to a legacy installation, File Tagger
will migrate the existing encrypted configuration to the envelope format as one
atomic, backed-up operation. The newly displayed recovery key becomes the valid
non-destructive recovery credential for future lockouts.

If legacy metadata is detected during a failed login/recovery attempt, the
recovery dialog explicitly explains this limitation and offers the reset path.
It must never claim that the old recovery key can restore the legacy
configuration.

## Login and Recovery Flow

The login screen includes a **Forgot password** button that opens a recovery
dialog with two paths.

### Recover

1. The user enters the recovery key and a new password twice.
2. The recovery envelope is decrypted and authenticated.
3. The existing data key is rewrapped in a new password envelope.
4. The recovery envelope and encrypted configuration remain unchanged.
5. File Tagger performs the normal initialization flow with the new password.

An invalid key or incompatible legacy metadata returns a generic recovery
failure with only the safe guidance needed for the next action.

### Reset Configuration

1. The dialog warns that encrypted settings and credentials will no longer be
   active.
2. The user types an explicit confirmation and supplies a new password twice.
3. File Tagger moves encrypted configuration and credential files to a
   timestamped backup directory.
4. SQLite tagging/search data is untouched.
5. A new password envelope, recovery envelope, and default encrypted
   configuration are initialized with the new password.
6. The user sees and acknowledges the new recovery key.

The reset aborts if quarantine cannot be completed. Moving files to a backup is
preferred over immediate deletion so that someone who later remembers the old
password can attempt a manual restoration.

## Error Handling

- Wrong password and invalid recovery key errors are explicit to the user but
  never expose cryptographic details.
- Missing, corrupt, unsupported, or internally inconsistent envelope metadata
  prevents recovery and preserves all files.
- Configuration decryption failure is reported; it must not silently become
  default configuration.
- Migration and reset use atomic file replacement and do not destroy originals
  before backups are verified.
- A UI error returns the login/recovery controls to a usable state.
- Logs record operation stage and exception category, but never passwords,
  recovery keys, wrapped material, or decrypted configuration.

## Testing

Focused automated tests will cover:

- First-time setup and recovery-key acknowledgement.
- Successful and failed password authentication.
- Password changes preserving decryptable configuration.
- Recovery-key recovery with and without password retention.
- Invalid, missing, corrupt, and unsupported recovery metadata.
- Recovery-key rotation.
- Legacy successful-login migration.
- Legacy lockout messaging and the inability to use the old recovery format.
- Reset quarantine, abort-on-backup-failure, and preservation of SQLite/search
  data.
- Login-screen and recovery-dialog wiring.

The implementation is complete only after the relevant password/migration/UI
pytest suites pass, and lint/type diagnostics report no errors caused by the
change.
