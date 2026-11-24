class FileTaggerError(Exception):
    """Base class for externally visible application failures."""


class NotIndexedError(FileTaggerError):
    """Raised when an operation requires an indexed file."""


class OllamaUnavailableError(FileTaggerError):
    """Raised when the required local Ollama server is unavailable."""


class OllamaTimeoutError(OllamaUnavailableError):
    """Raised when a local Ollama operation exceeds its timeout."""


class RemoteOllamaHostError(FileTaggerError):
    """Raised when Ollama is configured with a non-loopback origin."""


class ModelMissingError(FileTaggerError):
    """Raised when a required Ollama model is not installed."""


class ModelIncompatibleError(FileTaggerError):
    """Raised when an installed model lacks a required capability."""


class EmbeddingContextRejectedError(FileTaggerError):
    """Raised when content cannot fit the embedding model without truncation."""


class OcrError(FileTaggerError):
    """Raised when a required OCR stage cannot extract usable content."""


class PasswordEncryptionError(FileTaggerError):
    """Raised when configuration encryption or password handling fails."""


class IndexTransactionError(FileTaggerError):
    """Raised when a file revision cannot be indexed transactionally."""


class DatabaseExtensionError(FileTaggerError):
    """Raised when required SQLite search extensions cannot be loaded."""
