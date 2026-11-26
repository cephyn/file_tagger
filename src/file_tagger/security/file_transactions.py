from __future__ import annotations

import os
from pathlib import Path
import secrets


def replace_files_transactionally(files: dict[Path, bytes]) -> None:
    token = secrets.token_hex(8)
    staged: dict[Path, Path] = {}
    originals: dict[Path, bytes | None] = {}
    replaced: list[Path] = []
    try:
        for destination, content in files.items():
            temporary = destination.with_name(
                f".{destination.name}.{token}.migration"
            )
            with temporary.open("wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            staged[destination] = temporary
            originals[destination] = (
                destination.read_bytes() if destination.is_file() else None
            )
        for destination, temporary in staged.items():
            temporary.replace(destination)
            replaced.append(destination)
    except OSError:
        rollback_error: OSError | None = None
        for destination in reversed(replaced):
            try:
                original = originals[destination]
                if original is None:
                    destination.unlink(missing_ok=True)
                else:
                    rollback = destination.with_name(
                        f".{destination.name}.{token}.rollback"
                    )
                    rollback.write_bytes(original)
                    rollback.replace(destination)
            except OSError as error:
                rollback_error = error
        if rollback_error is not None:
            raise OSError(
                "File transaction failed and rollback was incomplete."
            ) from rollback_error
        raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
