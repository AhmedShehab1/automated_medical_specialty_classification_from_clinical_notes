import re


class ClinicalTextPipeline:
    """Base pipeline for lightweight text cleaning and de-identification."""

    _DATE_PATTERNS = (
        re.compile(r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:\d{2}|\d{4})\b"),
        re.compile(r"\b\d{4}-(?:0?[1-9]|1[0-2])-(?:0?[1-9]|[12]\d|3[01])\b"),
    )
    _PHONE_PATTERN = re.compile(
        r"(?<!\w)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)"
    )
    _SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

    def __init__(self) -> None:
        """Initialize the base class without loading heavy NLP assets."""
        self.is_ready = True

    def de_identify(self, text: str) -> str:
        """Replace common PII patterns with standard placeholders."""
        masked_text = text

        for date_pattern in self._DATE_PATTERNS:
            masked_text = date_pattern.sub("[DATE]", masked_text)

        masked_text = self._PHONE_PATTERN.sub("[PHONE]", masked_text)
        masked_text = self._SSN_PATTERN.sub("[SSN]", masked_text)
        return masked_text

    def clean_text(self, text: str) -> str:
        """Normalize whitespace and remove line breaks."""
        cleaned_text = re.sub(r"[\r\n\t]+", " ", text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        return cleaned_text.strip()


def main() -> None:
    pipeline = ClinicalTextPipeline()
    sample_text = "Patient John Doe seen on 04/12/2026.\nCall him at 555-123-4567."
    cleaned = pipeline.clean_text(sample_text)
    de_identified = pipeline.de_identify(cleaned)
    print(de_identified)


if __name__ == "__main__":
    main()
