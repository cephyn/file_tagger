from pydantic import BaseModel, ConfigDict, Field


class ExtractionDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    severity: str = "warning"


def classify_failure(errors: tuple[str, ...]) -> str | None:
    combined = " ".join(errors).lower()
    if "encrypt" in combined or "password" in combined:
        return "encrypted"
    if "ocr" in combined and (
        "failed" in combined or "unable" in combined or "no text" in combined
    ):
        return "ocr_failed"
    if (
        "no recognizable format" in combined
        or "unsupported" in combined
        or "not recognized" in combined
    ):
        return "unsupported"
    return "extraction_failed" if errors else None


def diagnostics_from_errors(
    errors: tuple[str, ...],
    *,
    severity: str = "warning",
) -> tuple[ExtractionDiagnostic, ...]:
    return tuple(
        ExtractionDiagnostic(
            code="conversion_warning" if index else "conversion_error",
            message=message,
            severity=severity,
        )
        for index, message in enumerate(errors)
    )


def useful_text_length(markdown: str) -> int:
    visible = markdown
    for placeholder in ("<!-- image -->", "<!-- page break -->"):
        visible = visible.replace(placeholder, "")
    return sum(character.isalnum() for character in visible)


def extract_headings(markdown: str) -> tuple[str, ...]:
    headings: list[str] = []
    for line in markdown.splitlines():
        stripped = line.lstrip()
        if not stripped.startswith("#"):
            continue
        marker, separator, title = stripped.partition(" ")
        if separator and marker and set(marker) == {"#"}:
            cleaned = title.strip()
            if cleaned:
                headings.append(cleaned)
    return tuple(headings)
