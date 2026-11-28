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


class GeminiUnavailableError(FileTaggerError):
    """Raised when Gemini provider operations are unavailable."""


class GeminiTimeoutError(GeminiUnavailableError):
    """Raised when a Gemini operation exceeds its timeout."""


class GeminiAuthenticationError(FileTaggerError):
    """Raised when Gemini authentication fails."""


class ModelMissingError(FileTaggerError):
    """Raised when a required Ollama model is not installed."""


class ModelIncompatibleError(FileTaggerError):
    """Raised when an installed model lacks a required capability."""


class GenerationContextRejectedError(ModelIncompatibleError):
    """Raised when generation input cannot fit the model context."""


class StructuredOutputError(ModelIncompatibleError, ValueError):
    """Raised when a model cannot produce the requested structured output."""


class GeminiStructuredOutputError(StructuredOutputError):
    """Raised when Gemini cannot produce the requested structured output."""


class UncitedAnswerError(StructuredOutputError):
    """Raised when an answer cannot be tied to retrieved indexed evidence."""


class ModelRecommendationError(FileTaggerError):
    """Raised when local hardware cannot support a required model role."""


class EmbeddingContextRejectedError(FileTaggerError):
    """Raised when content cannot fit the embedding model without truncation."""


class OcrError(FileTaggerError):
    """Raised when a required OCR stage cannot extract usable content."""


class PasswordEncryptionError(FileTaggerError):
    """Raised when configuration encryption or password handling fails."""


class CredentialFormatError(PasswordEncryptionError):
    """Raised when credential metadata is incomplete or unsupported."""


class CredentialRecoveryError(PasswordEncryptionError):
    """Raised when recovery or reset cannot be completed safely."""


class IndexTransactionError(FileTaggerError):
    """Raised when a file revision cannot be indexed transactionally."""


class DatabaseExtensionError(FileTaggerError):
    """Raised when required SQLite search extensions cannot be loaded."""
